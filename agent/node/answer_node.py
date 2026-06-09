import os
import re
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore

from .types import AgentState, existence_grounder_enabled


def _looks_like_binary_query(text: str) -> bool:
    query = str(text or "").strip().lower()
    if not query:
        return False
    return bool(
        re.match(r"^(is|are|was|were|do|does|did|can|could|has|have|had)\b", query)
        or " is there " in f" {query} "
        or query.endswith("?")
    )


def _select_final_rows(state: AgentState) -> list[dict[str, Any]]:
    if "rerank_result" in state:
        return list(state.get("rerank_result") or [])
    if "hybrid_result" in state:
        return list(state.get("hybrid_result") or [])
    return list(state.get("sql_result") or [])


def _format_existence_answer(verifier: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    decision = str(verifier.get("decision") or "").strip().lower()
    video_id = str(verifier.get("video_id") or "").strip()
    start_time = verifier.get("start_time")
    end_time = verifier.get("end_time")
    primary_summary = str(verifier.get("primary_summary") or "").strip()
    reason = str(verifier.get("reason") or "").strip()

    # ── v2.4: Multi-camera answer formatting ──
    if verifier.get("multi_camera") and verifier.get("cameras"):
        cameras = verifier["cameras"]
        if not isinstance(cameras, list):
            cameras = []
        if decision == "mismatch":
            return (
                "No matching clip is expected."
                + (f" Reason: {reason}." if reason else "")
            )
        label = "Yes" if decision == "exact" else "Likely yes"
        camera_list = ", ".join(c.get("camera_id", "") for c in cameras)
        parts = [f"{label}. The queried entity appears across multiple cameras."]
        parts.append(f"Cameras: {camera_list}")
        times_parts: list[str] = []
        for c in cameras:
            cam_id = c.get("camera_id", "")
            st = c.get("start_time", "")
            et = c.get("end_time", "")
            times_parts.append(f"{cam_id} {st}-{et}")
        parts.append(f"Times: {'; '.join(times_parts)}")
        if primary_summary or reason:
            parts.append(f"Summary: {primary_summary or reason}")
        return " | ".join(parts) + "."

    # ── Single-camera answer formatting ──
    if decision == "mismatch" or not rows:
        return (
            "No matching clip is expected."
            + (f" Reason: {reason}." if reason else "")
        )
    # ── Appearance conflict: person found but color differs from query ──
    appearance_conflict = verifier.get("appearance_conflict")
    if appearance_conflict:
        asked_str = ", ".join(appearance_conflict.get("asked") or [])
        found_str = ", ".join(appearance_conflict.get("found") or [])
        loc = ""
        if video_id:
            loc += f" on {video_id}"
        if start_time is not None and end_time is not None:
            loc += f" at {start_time}–{end_time}"
        return (
            f"Yes — a person was detected{loc}, "
            f"but their appearance shows {found_str} clothing, not {asked_str} as queried. "
            f"Please verify visually."
        )

    label = "Yes" if decision == "exact" else "Likely yes"
    parts = [f"{label}."]
    if video_id:
        parts.append(f"Video={video_id}")
    if start_time is not None:
        parts.append(f"start={start_time}")
    if end_time is not None:
        parts.append(f"end={end_time}")
    summary_head = ". " + ", ".join(parts[1:]) if len(parts) > 1 else ""
    body = f"{parts[0]}{summary_head}."
    if primary_summary:
        body += f" Summary: {primary_summary}"
    return body


def final_answer_node(state: AgentState, config: RunnableConfig, store: BaseStore) -> dict[str, Any]:
    del config, store
    rows = _select_final_rows(state)
    answer_type = str(state.get("answer_type") or "").strip().lower()
    query = str(state.get("original_user_query") or state.get("user_query") or "").strip()
    verifier_result = state.get("verifier_result") or {}
    if not isinstance(verifier_result, dict):
        verifier_result = {}

    agent_summary = ""
    if state.get("sql_debug") and isinstance(state.get("sql_debug"), dict):
        agent_summary = state["sql_debug"].get("agent_summary", "")
    elif state.get("search_explain"):
        agent_summary = state.get("search_explain", "")

    if (
        existence_grounder_enabled()
        and (answer_type in {"existence", "unknown"} or _looks_like_binary_query(query))
        and verifier_result.get("decision") in {"exact", "partial", "mismatch"}
    ):
        final_answer = _format_existence_answer(verifier_result, rows)
        return {
            "raw_final_answer": final_answer,
            "final_answer": final_answer,
            "current_node": "final_answer_node",
            "messages": [AIMessage(content=final_answer)],
        }

    if not rows:
        final_answer = agent_summary if agent_summary else "No matching results found. You can add more specific descriptions like colors or actions."
        return {
            "raw_final_answer": final_answer,
            "final_answer": final_answer,
            "current_node": "final_answer_node",
            "messages": [AIMessage(content=final_answer)],
        }

    parts: list[str] = []
    if agent_summary:
        parts.append(agent_summary + "\n\nDetailed results:")
    else:
        parts.append("Retrieval complete. Most relevant results:")

    for idx, row in enumerate(rows[:5], start=1):
        is_parent = str(row.get("_record_level") or "").lower() == "parent"
        if is_parent:
            parts.append(
                f"[{idx}] video={row.get('video_id')} | child_hits={row.get('_parent_hit_count', 0)} | "
                f"time={row.get('start_time')}-{row.get('end_time')} | summary={row.get('event_summary_en', row.get('event_text_en', 'N/A'))}"
            )
        else:
            parts.append(
                f"[{idx}] event_id={row.get('event_id')} | video={row.get('video_id')} | "
                f"distance={row.get('_distance', 'N/A')} | summary={row.get('event_summary_en', row.get('event_text_cn', 'N/A'))}"
            )
    final_answer = "\n".join(parts)
    return {
        "raw_final_answer": final_answer,
        "final_answer": final_answer,
        "current_node": "final_answer_node",
        "messages": [AIMessage(content=final_answer)],
    }


def final_error_node(state: AgentState, config: RunnableConfig, store: BaseStore) -> dict[str, Any]:
    del config, store
    final_answer = f"Retrieval pipeline failed: {state.get('tool_error')}"
    return {"final_answer": final_answer, "messages": [AIMessage(content=final_answer)]}


if __name__ == "__main__":
    rows = [
        {
            "event_id": 1,
            "video_id": "demo.mp4",
            "start_time": 10,
            "end_time": 20,
            "_distance": 0.12,
            "event_summary_cn": "红色目标进入画面",
        }
    ]
    print(final_answer_node({"rerank_result": rows}, config={}, store=None)["final_answer"])
    print(final_error_node({"tool_error": "SQL检索失败"}, config={}, store=None)["final_answer"])
