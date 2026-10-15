"""Exercise 3 — Escalation branch with reviewer Q&A.

When confidence < 60%, the agent doesn't ask approve/reject — it asks specific
clarifying questions and then synthesizes a refined review from the answers.
"""

from __future__ import annotations

import argparse
import uuid

from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from rich.console import Console
from rich.panel import Panel

from common.github import fetch_pr, post_review_comment
from common.llm import get_llm
from common.schemas import (
    AUTO_APPROVE_THRESHOLD,
    ESCALATE_THRESHOLD,
    PRAnalysis,
    ReviewState,
)


console = Console()


def node_fetch_pr(state):
    console.print("[cyan]→ fetch_pr[/cyan]")
    with console.status("[dim]Fetching PR from GitHub...[/dim]"):
        pr = fetch_pr(state["pr_url"])
    console.print(f"  [green]✓[/green] {len(pr.files_changed)} files, head {pr.head_sha[:7]}")
    return {"pr_title": pr.title, "pr_diff": pr.diff, "pr_files": pr.files_changed, "pr_head_sha": pr.head_sha}


def node_analyze(state):
    console.print("[cyan]→ analyze[/cyan]")
    # Dùng temperature 0 để confidence và các câu hỏi ít dao động
    # giữa các lần chạy cùng một PR.
    llm = get_llm(temperature=0.0).with_structured_output(PRAnalysis)
    with console.status("[dim]LLM reviewing the diff...[/dim]"):
        analysis = llm.invoke([
            {"role": "system", "content": (
                "You are a conservative senior PR reviewer. Confidence means the "
                "review is complete enough to act without human context. "
                f"Use >={AUTO_APPROVE_THRESHOLD:.2f} only for small, clear, low-risk, "
                f"well-tested changes; use [{ESCALATE_THRESHOLD:.2f}, "
                f"{AUTO_APPROVE_THRESHOLD:.2f}) when migration, compatibility, intent, "
                f"or test questions remain; use <{ESCALATE_THRESHOLD:.2f} for auth, "
                "crypto, secrets, SQL injection, data-loss risks, or multiple serious "
                "issues. Finding a serious bug confidently does not justify high confidence. "
                # TODO: add an instruction: if confidence < 60%, populate escalation_questions
                # with 2–4 specific, context-rich questions (reference which file/section in the diff).
                f"When confidence is below {ESCALATE_THRESHOLD:.2f}, populate "
                "escalation_questions with 2-4 concrete questions. Each question must "
                "reference the relevant file, function, or diff section and ask for "
                "missing intent, constraints, or security context."
            )},
            {"role": "user", "content": f"Title: {state['pr_title']}\nDiff:\n{state['pr_diff']}"},
        ])
    console.print(f"  [green]✓[/green] confidence={analysis.confidence:.0%}, {len(analysis.escalation_questions)} question(s)")
    return {"analysis": analysis}


def node_route(state):
    console.print("[cyan]→ route[/cyan]")
    c = state["analysis"].confidence
    if c >= AUTO_APPROVE_THRESHOLD: decision = "auto_approve"
    elif c < ESCALATE_THRESHOLD:    decision = "escalate"
    else:                           decision = "human_approval"
    console.print(f"  [green]✓[/green] decision=[bold]{decision}[/bold] (confidence={c:.0%})")
    return {"decision": decision}


def node_escalate(state: ReviewState) -> dict:
    """Ask the reviewer specific questions; return their answers in state."""
    a = state["analysis"]
    questions = a.escalation_questions
    if not questions:
        # fallback when the LLM didn't generate any questions
        questions = ["What is the intent of this PR?", "Any migration concerns?"]

    # TODO: call interrupt(payload) where payload kind="escalation" contains:
    #       pr_url, confidence, confidence_reasoning, summary, risk_factors, questions.
    # answers = interrupt({...})
    # return {"escalation_answers": answers}
    # Lần gọi đầu sẽ dừng graph và chuyển payload này cho
    # handle_interrupt(). Sau Command(resume=...), interrupt() trả lại dict answers.
    answers = interrupt({
        "kind": "escalation",
        "pr_url": state["pr_url"],
        "confidence": a.confidence,
        "confidence_reasoning": a.confidence_reasoning,
        "summary": a.summary,
        "risk_factors": a.risk_factors,
        "questions": questions,
    })
    return {"escalation_answers": answers}


def node_synthesize(state: ReviewState) -> dict:
    """Re-prompt LLM with the reviewer's answers and produce a refined review."""
    # TODO:
    #   - read state["escalation_answers"] (dict[question, answer])
    #   - call get_llm().with_structured_output(PRAnalysis).invoke(...) with a prompt
    #     containing the original diff + initial analysis + Q&A.
    #   - return {"analysis": refined}
    # `node_commit` will then post the refined review to the PR.
    answers = state.get("escalation_answers") or {}
    qa = "\n\n".join(f"Q: {question}\nA: {answer}" for question, answer in answers.items())
    initial_analysis = state["analysis"].model_dump_json(indent=2)

    # Lần gọi thứ hai nhận cả kết quả ban đầu lẫn ngữ cảnh do
    # reviewer cung cấp, sau đó thay thế state["analysis"] bằng bản mới.
    llm = get_llm(temperature=0.0).with_structured_output(PRAnalysis)
    with console.status("[dim]LLM refining review with reviewer answers...[/dim]"):
        refined = llm.invoke([
            {
                "role": "system",
                "content": (
                    "You are a senior PR reviewer. Produce a complete, refined "
                    "PRAnalysis using the reviewer's answers. Treat answers as context, "
                    "not proof that code defects are fixed. Reassess risks, comments, "
                    "and confidence, and explain the new confidence concretely."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Original diff:\n{state['pr_diff']}\n\n"
                    f"Initial analysis:\n{initial_analysis}\n\n"
                    f"Reviewer Q&A:\n{qa}"
                ),
            },
        ])

    console.print(
        f"  [green]✓[/green] refined confidence={refined.confidence:.0%}, "
        f"{len(refined.comments)} comment(s)"
    )
    return {"analysis": refined}


def node_human_approval(state):
    a = state["analysis"]
    response = interrupt({
        "kind": "approval_request", "pr_url": state["pr_url"],
        "confidence": a.confidence, "confidence_reasoning": a.confidence_reasoning,
        "summary": a.summary,
        "comments": [c.model_dump() for c in a.comments],
        "diff_preview": state["pr_diff"][:2000],
    })
    return {"human_choice": response.get("choice"), "human_feedback": response.get("feedback")}


def _render_comment_body(state) -> str:
    a = state["analysis"]
    lines = [f"### Automated review (confidence {a.confidence:.0%})", "", a.summary, ""]
    for c in a.comments:
        lines.append(f"- **[{c.severity}]** `{c.file}:{c.line or '?'}` — {c.body}")
    if state.get("human_feedback"):
        lines.append(f"\n_Reviewer note: {state['human_feedback']}_")
    if state.get("escalation_answers"):
        lines.append("\n_Reviewer answered escalation questions:_")
        for q, ans in state["escalation_answers"].items():
            lines.append(f"> **{q}** {ans}")
    return "\n".join(lines)


def _post(state, label: str) -> str:
    try:
        post_review_comment(state["pr_url"], _render_comment_body(state))
        console.print(f"  [green]✓[/green] posted comment to {state['pr_url']}")
        return label
    except Exception as e:
        console.print(f"  [red]✗[/red] post failed: {e}")
        return "commit_failed"


def node_commit(state):
    console.print("[cyan]→ commit[/cyan]")
    # Two paths converge here:
    #   1. human_approval → commit (only post if approved)
    #   2. escalate → synthesize → commit (always post the refined review)
    if state.get("escalation_answers"):
        return {"final_action": _post(state, "committed_after_escalation")}
    if state.get("human_choice") == "approve":
        return {"final_action": _post(state, "committed")}
    console.print(f"  [yellow]·[/yellow] skipping comment (choice={state.get('human_choice')})")
    return {"final_action": "rejected"}


def node_auto_approve(state):
    console.print("[cyan]→ auto_approve[/cyan]  [dim]high confidence — posting directly[/dim]")
    return {"final_action": _post(state, "auto_approved")}


def build_graph():
    g = StateGraph(ReviewState)
    for name, fn in [
        ("fetch_pr", node_fetch_pr), ("analyze", node_analyze), ("route", node_route),
        ("auto_approve", node_auto_approve), ("human_approval", node_human_approval),
        ("commit", node_commit), ("escalate", node_escalate), ("synthesize", node_synthesize),
    ]:
        g.add_node(name, fn)
    g.add_edge(START, "fetch_pr")
    g.add_edge("fetch_pr", "analyze")
    g.add_edge("analyze", "route")
    g.add_conditional_edges(
        "route", lambda s: s["decision"],
        {"auto_approve": "auto_approve", "human_approval": "human_approval", "escalate": "escalate"},
    )
    g.add_edge("auto_approve", END)
    g.add_edge("human_approval", "commit")
    g.add_edge("commit", END)
    # TODO: wire escalate → synthesize → commit  (commit already → END)
    # Chỉ sau khi có answers, graph mới phân tích lại và chuyển sang node
    # có side-effect đăng comment. Không đăng trong node chứa interrupt.
    g.add_edge("escalate", "synthesize")
    g.add_edge("synthesize", "commit")
    return g.compile(checkpointer=MemorySaver())


def handle_interrupt(payload):
    kind = payload["kind"]
    if kind == "approval_request":
        console.print(Panel.fit(
            payload["summary"],
            title=f"Approve? conf={payload['confidence']:.0%}",
            border_style="green",
        ))
        choice = console.input("approve/reject/edit? ").strip().lower()
        return {"choice": choice, "feedback": console.input("Feedback: ").strip()}
    if kind == "escalation":
        console.print(Panel.fit(
            payload["summary"],
            title=f"Escalation conf={payload['confidence']:.0%}",
            border_style="yellow",
        ))
        return {q: console.input(f"Q: {q}\nA: ").strip() for q in payload["questions"]}
    raise ValueError(kind)


def main():
    load_dotenv()
    p = argparse.ArgumentParser(); p.add_argument("--pr", required=True)
    args = p.parse_args()

    console.rule("[bold]Exercise 3 — escalation with reviewer Q&A[/bold]")
    console.print(f"[dim]PR: {args.pr}[/dim]\n")

    app = build_graph()
    thread_id = str(uuid.uuid4())
    cfg = {"configurable": {"thread_id": thread_id}}
    console.print(f"[dim]thread_id = {thread_id}[/dim]\n")

    result = app.invoke({"pr_url": args.pr, "thread_id": thread_id}, cfg)
    while "__interrupt__" in result:
        result = app.invoke(Command(resume=handle_interrupt(result["__interrupt__"][0].value)), cfg)

    console.rule("Final")
    console.print(f"final_action = {result.get('final_action')}")
    if "analysis" in result:
        console.print(f"final confidence = {result['analysis'].confidence:.0%}")


if __name__ == "__main__":
    main()
