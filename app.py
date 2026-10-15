"""Exercise 5 — Streamlit approval UI for the HITL PR review agent.

Run with:
    uv run streamlit run app.py

Goal: wrap the LangGraph built in exercises 1–4 in a web UI that adapts to
the confidence bucket of each PR.

Routing thresholds (common/schemas.py):
    > 72%        auto_approve     UI shows a success card; reviewer does nothing
    58 – 72%     human_approval   UI shows Approve / Reject / Edit buttons
    <  58%       escalate         UI shows a question form for the reviewer
"""

from __future__ import annotations

import asyncio
import uuid

import streamlit as st
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from common.db import db_conn, db_path
# TODO: import the graph builder + helpers from your exercise 4 solution.
# Suggestion: rename `exercises/exercise_4_audit.py` functions you need
# (build_graph, handle_interrupt logic) and import them here, OR copy the
# graph wiring inline.
# from exercises.exercise_4_audit import build_graph
from exercises.exercise_4_audit import build_graph


load_dotenv()

# ─── Session state ─────────────────────────────────────────────────────────
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "pr_url" not in st.session_state:
    st.session_state.pr_url = ""
if "interrupt_payload" not in st.session_state:
    st.session_state.interrupt_payload = None
if "final" not in st.session_state:
    st.session_state.final = None


# ─── Page setup ────────────────────────────────────────────────────────────
st.set_page_config(page_title="HITL PR Review", layout="wide")
st.title("HITL PR Review Agent")


async def load_recent_sessions() -> list[dict]:
    """Return recent audit sessions for the Streamlit sidebar.

    `audit.replay.list_threads()` prints a Rich table instead of returning
    rows, so the web UI executes the equivalent SQL query directly.
    """
    async with db_conn() as conn:
        async with conn.execute(
            """
            SELECT thread_id,
                   pr_url,
                   MAX(timestamp) AS last_event,
                   CASE MAX(
                       CASE risk_level
                           WHEN 'high' THEN 3
                           WHEN 'med'  THEN 2
                           ELSE 1
                       END
                   )
                       WHEN 3 THEN 'high'
                       WHEN 2 THEN 'med'
                       ELSE 'low'
                   END AS worst_risk
              FROM audit_events
             GROUP BY thread_id, pr_url
             ORDER BY MAX(timestamp) DESC
             LIMIT 25
            """
        ) as cur:
            rows = await cur.fetchall()
    return [dict(row) for row in rows]


# ─── Sidebar — recent sessions ─────────────────────────────────────────────
with st.sidebar:
    st.header("Recent sessions")
    # TODO: call `audit.replay.list_threads`-style query against audit_events
    # and render thread_id + pr_url + worst_risk + last_event as a small table.
    # On row click, set st.session_state.thread_id and rerun.
    try:
        recent_sessions = asyncio.run(load_recent_sessions())
    except Exception as exc:
        # Lỗi sidebar không được làm hỏng form review chính.
        recent_sessions = []
        st.warning(f"Could not load audit sessions: {exc}")

    if recent_sessions:
        selected = st.dataframe(
            recent_sessions,
            column_order=("thread_id", "pr_url", "worst_risk", "last_event"),
            column_config={
                "thread_id": st.column_config.TextColumn("Thread"),
                "pr_url": st.column_config.LinkColumn("PR"),
                "worst_risk": st.column_config.TextColumn("Risk"),
                "last_event": st.column_config.DatetimeColumn("Last event"),
            },
            hide_index=True,
            width="stretch",
            on_select="rerun",
            selection_mode="single-row",
            key="recent_sessions_table",
        )
        selected_rows = selected.selection.rows
        if selected_rows:
            session = recent_sessions[selected_rows[0]]
            if st.session_state.thread_id != session["thread_id"]:
                # Chọn phiên cũ chỉ khôi phục context để xem/replay;
                # nó không tạo một graph run mới.
                st.session_state.thread_id = session["thread_id"]
                st.session_state.pr_url = session["pr_url"]
                st.session_state.interrupt_payload = None
                st.session_state.final = None
                st.rerun()
    else:
        st.caption("No audited sessions yet.")

    if st.session_state.thread_id:
        st.caption(f"Selected thread: `{st.session_state.thread_id}`")


# ─── Top form — start a new review ─────────────────────────────────────────
with st.form("start"):
    pr_url = st.text_input(
        "PR URL", value=st.session_state.pr_url,
        placeholder="https://github.com/VinUni-AI20k/PR-Demo/pull/1",
    )
    submitted = st.form_submit_button("Run review")


# ─── Renderers per interrupt kind ──────────────────────────────────────────
def render_approval_card(payload: dict) -> dict | None:
    """58–72% bucket: show the LLM review + 3 buttons. Return resume dict or None."""
    conf = payload["confidence"]
    st.subheader(f"Approval requested — confidence {conf:.0%}")
    st.caption(payload["confidence_reasoning"])
    st.markdown(payload["summary"])

    for c in payload.get("comments", []):
        st.markdown(f"- **[{c['severity']}]** `{c['file']}:{c.get('line') or '?'}` — {c['body']}")

    with st.expander("Diff"):
        st.code(payload.get("diff_preview", ""), language="diff")

    # Widget key gắn với thread_id để phiên mới không kế thừa
    # feedback hoặc trạng thái nút từ phiên trước.
    thread_id = st.session_state.thread_id or "current"
    feedback = st.text_input(
        "Feedback (optional)", key=f"approval_feedback_{thread_id}",
    )
    col1, col2, col3 = st.columns(3)
    # TODO: hook up the three buttons. Each click should return one of:
    #   {"choice": "approve", "feedback": feedback}
    #   {"choice": "reject",  "feedback": feedback}
    #   {"choice": "edit",    "feedback": feedback}
    if col1.button("Approve", type="primary", key=f"approve_{thread_id}"):
        return {"choice": "approve", "feedback": feedback}
    if col2.button("Reject", key=f"reject_{thread_id}"):
        return {"choice": "reject", "feedback": feedback}
    if col3.button("Edit", key=f"edit_{thread_id}"):
        return {"choice": "edit", "feedback": feedback}
    return None


def render_escalation_card(payload: dict) -> dict | None:
    """< 58% bucket: show risk factors + question form. Return {question: answer} or None."""
    conf = payload["confidence"]
    st.subheader(f"Strong escalation — confidence {conf:.0%}")
    st.caption(payload["confidence_reasoning"])
    if payload.get("risk_factors"):
        st.error("Risks: " + ", ".join(payload["risk_factors"]))
    st.markdown(payload["summary"])

    with st.form("escalation"):
        # TODO: render one text_input per question in payload["questions"]
        #       collect answers into a dict {question: answer_str}
        #       on submit, return the dict.
        answers: dict[str, str] = {}
        questions = payload.get("questions", [])
        thread_id = st.session_state.thread_id or "current"
        for index, question in enumerate(questions):
            st.markdown(f"**Q{index + 1}. {question}**")
            answers[question] = st.text_input(
                "Answer",
                key=f"escalation_{thread_id}_{index}",
                placeholder="Provide the missing context...",
                label_visibility="collapsed",
            )

        submitted_answers = st.form_submit_button("Submit answers", type="primary")
        if submitted_answers:
            # Không resume bằng câu trả lời rỗng: node_synthesize có
            # thể coi nó là context thật và tăng confidence sai.
            unanswered = [question for question, answer in answers.items() if not answer.strip()]
            if unanswered:
                st.warning("Please answer every escalation question before continuing.")
            else:
                return answers
    return None


# ─── Drive the graph ───────────────────────────────────────────────────────
async def run_graph(pr_url: str, thread_id: str, resume_value=None):
    """Invoke the graph once. Returns the final result or {'__interrupt__': ...}."""
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        # TODO: build the graph with `cp` as the checkpointer (use the function
        # you imported/copied at the top of this file).
        # app = build_graph(cp)
        app = build_graph(cp)
        cfg = {"configurable": {"thread_id": thread_id}}

        # TODO:
        # - If resume_value is None: result = await app.ainvoke(
        #       {"pr_url": pr_url, "thread_id": thread_id}, cfg)
        # - Else:                    result = await app.ainvoke(
        #       Command(resume=resume_value), cfg)
        # - Return result.
        # Checkpointer tra state bằng thread_id, vì vậy invoke đầu và
        # tất cả lần resume phải dùng lại chính cfg này.
        if resume_value is None:
            result = await app.ainvoke(
                {"pr_url": pr_url, "thread_id": thread_id}, cfg,
            )
        else:
            result = await app.ainvoke(Command(resume=resume_value), cfg)
        return result


# ─── Main flow ─────────────────────────────────────────────────────────────
if submitted and pr_url:
    st.session_state.pr_url = pr_url.strip()
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.interrupt_payload = None
    st.session_state.final = None

    with st.spinner("Fetching PR + asking the LLM..."):
        result = asyncio.run(run_graph(st.session_state.pr_url, st.session_state.thread_id))

    if "__interrupt__" in result:
        st.session_state.interrupt_payload = result["__interrupt__"][0].value
    else:
        st.session_state.final = result

# Render the current interrupt card, if any
payload = st.session_state.interrupt_payload
if payload is not None:
    kind = payload["kind"]
    answer = render_approval_card(payload) if kind == "approval_request" else render_escalation_card(payload)
    if answer is not None:
        with st.spinner("Resuming..."):
            result = asyncio.run(run_graph(
                st.session_state.pr_url, st.session_state.thread_id, resume_value=answer,
            ))
        if "__interrupt__" in result:
            st.session_state.interrupt_payload = result["__interrupt__"][0].value
        else:
            st.session_state.interrupt_payload = None
            st.session_state.final = result
        st.rerun()

# Render final state, if reached
if st.session_state.final is not None:
    final = st.session_state.final
    action = final.get("final_action", "?")
    if action.startswith("auto") or action.startswith("committed"):
        st.success(f"✓ {action} — comment posted to {st.session_state.pr_url}")
        analysis = final.get("analysis")
        if analysis is not None:
            # Checkpoint có thể khôi phục Pydantic model hoặc dict tuỳ
            # serializer/version, nên UI hỗ trợ cả hai dạng.
            if isinstance(analysis, dict):
                confidence = analysis.get("confidence")
                summary = analysis.get("summary", "")
            else:
                confidence = getattr(analysis, "confidence", None)
                summary = getattr(analysis, "summary", "")
            if confidence is not None:
                st.metric("Final confidence", f"{confidence:.0%}")
            if summary:
                st.markdown(summary)
        st.link_button("View comment on GitHub", st.session_state.pr_url)
    elif action == "rejected":
        st.warning("Rejected — no comment posted")
    elif action == "commit_failed":
        st.error("The review completed, but posting the GitHub comment failed.")
    else:
        st.info(f"final_action = {action}")
    st.caption(f"thread_id = {st.session_state.thread_id}  ·  replay: "
               f"`uv run python -m audit.replay --thread {st.session_state.thread_id}`")
