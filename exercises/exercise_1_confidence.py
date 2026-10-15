"""Exercise 1 — Confidence scoring + routing.

Build a small LangGraph that fetches a PR, analyzes it, then routes to one of
three terminal nodes by confidence. Goal: see the three branches print
different messages on different PRs.

"""

from __future__ import annotations

import argparse

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from rich.console import Console

from common.github import fetch_pr
from common.llm import get_llm
from common.schemas import (
    AUTO_APPROVE_THRESHOLD,
    ESCALATE_THRESHOLD,
    PRAnalysis,
    ReviewState,
)


console = Console()


def node_fetch_pr(state: ReviewState) -> dict:
    console.print("[cyan]→ fetch_pr[/cyan]")
    with console.status("[dim]Fetching PR from GitHub...[/dim]"):
        pr = fetch_pr(state["pr_url"])
    console.print(f"  [green]✓[/green] {len(pr.files_changed)} files, head {pr.head_sha[:7]}")
    return {
        "pr_title": pr.title, "pr_diff": pr.diff,
        "pr_files": pr.files_changed, "pr_head_sha": pr.head_sha,
    }


def node_analyze(state: ReviewState) -> dict:
    console.print("[cyan]→ analyze[/cyan]")
    # TODO: call the LLM with structured output PRAnalysis.
    # Hint:  llm = get_llm().with_structured_output(PRAnalysis)
    #        analysis = llm.invoke([...])
    #        return {"analysis": analysis}
    # When implemented, wrap the call in:
    #        with console.status("[dim]LLM thinking...[/dim]"):
    #            analysis = llm.invoke([...])
    # `with_structured_output` buộc phản hồi của model tuân theo
    # PRAnalysis. Nhờ đó các node sau có thể đọc confidence và
    # comments trực tiếp, không phải parse một chuỗi văn bản tự do.
    # Temperature 0 giúp confidence ít dao động hơn giữa các lần chạy.
    llm = get_llm(temperature=0.0).with_structured_output(PRAnalysis)
    with console.status("[dim]LLM thinking...[/dim]"):
        analysis = llm.invoke([
            {
                "role": "system",
                "content": (
                    "You are a conservative senior PR reviewer. Confidence means the "
                    "review is complete enough to act without human context. Use "
                    ">=0.73 only for small, clear, low-risk, well-tested changes; "
                    "use 0.58-0.72 when migration, compatibility, intent, or test "
                    "questions remain; use <0.58 for auth, crypto, secrets, SQL "
                    "injection, data-loss risks, or multiple serious issues. Finding "
                    "a serious bug confidently does not justify high confidence."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Pull request title: {state['pr_title']}\n\n"
                    f"Unified diff:\n{state['pr_diff']}"
                ),
            },
        ])

    console.print(
        f"  [green]✓[/green] confidence={analysis.confidence:.0%}, "
        f"{len(analysis.comments)} comment(s)"
    )
    # Mỗi node chỉ trả về phần state mà nó tạo hoặc cập nhật.
    return {"analysis": analysis}


def node_route(state: ReviewState) -> dict:
    console.print("[cyan]→ route[/cyan]")
    # TODO: read state["analysis"].confidence and return
    #       {"decision": "auto_approve" | "human_approval" | "escalate"}
    # Thresholds provided: AUTO_APPROVE_THRESHOLD (0.85) and ESCALATE_THRESHOLD (0.60).
    # Lưu ý: dòng comment phía trên ghi ngưỡng cũ. Luôn dùng các
    # hằng đã import từ common.schemas để có giá trị hiện hành.
    confidence = state["analysis"].confidence
    if confidence >= AUTO_APPROVE_THRESHOLD:
        decision = "auto_approve"
    elif confidence < ESCALATE_THRESHOLD:
        decision = "escalate"
    else:
        decision = "human_approval"

    console.print(
        f"  [green]✓[/green] decision=[bold]{decision}[/bold] "
        f"(confidence={confidence:.0%})"
    )
    # Node route ghi quyết định vào state; conditional edge bên dưới
    # sẽ đọc lại giá trị này để chọn node tiếp theo.
    return {"decision": decision}


def node_auto_approve(state: ReviewState) -> dict:
    console.print("[green]✓ AUTO APPROVE[/green] — high confidence, no human needed")
    return {"final_action": "auto_approved"}


def node_human_approval(state: ReviewState) -> dict:
    console.print("[yellow]✓ HUMAN APPROVAL[/yellow] — placeholder, exercise 2 will pause here")
    return {"final_action": "pending_human_approval"}


def node_escalate(state: ReviewState) -> dict:
    console.print("[red]✓ ESCALATE[/red] — placeholder, exercise 3 will ask the reviewer questions")
    return {"final_action": "pending_escalation"}


def build_graph():
    g = StateGraph(ReviewState)
    # TODO: add_node for the 6 nodes above (fetch_pr, analyze, route, auto_approve, human_approval, escalate)
    # Tên node là khóa dùng khi nối edge; hàm tương ứng là phần
    # xử lý sẽ nhận ReviewState và trả về state update.
    g.add_node("fetch_pr", node_fetch_pr)
    g.add_node("analyze", node_analyze)
    g.add_node("route", node_route)
    g.add_node("auto_approve", node_auto_approve)
    g.add_node("human_approval", node_human_approval)
    g.add_node("escalate", node_escalate)

    # TODO: add_edge from START → fetch_pr → analyze → route
    # Ba edge này tạo phần tuyến tính chung cho mọi PR.
    g.add_edge(START, "fetch_pr")
    g.add_edge("fetch_pr", "analyze")
    g.add_edge("analyze", "route")

    # TODO: add_conditional_edges on "route" with mapping
    #       {"auto_approve": "auto_approve", "human_approval": "human_approval", "escalate": "escalate"}
    # Hàm điều kiện không tính lại confidence; nó chỉ lấy decision
    # mà node_route vừa ghi vào state, rồi tra mapping để chọn nhánh.
    g.add_conditional_edges(
        "route",
        lambda state: state["decision"],
        {
            "auto_approve": "auto_approve",
            "human_approval": "human_approval",
            "escalate": "escalate",
        },
    )

    # TODO: add_edge from each terminal node → END
    # Exercise 1 chưa có HITL, vì vậy cả ba nhánh chỉ in kết quả
    # và kết thúc graph ngay lập tức.
    g.add_edge("auto_approve", END)
    g.add_edge("human_approval", END)
    g.add_edge("escalate", END)

    return g.compile()


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--pr", required=True)
    args = parser.parse_args()

    console.rule("[bold]Exercise 1 — confidence routing[/bold]")
    console.print(f"[dim]PR: {args.pr}[/dim]\n")

    app = build_graph()
    final = app.invoke({"pr_url": args.pr})

    console.rule("Final")
    console.print(f"confidence = {final['analysis'].confidence:.0%}")
    console.print(f"action     = {final.get('final_action')}")


if __name__ == "__main__":
    main()
