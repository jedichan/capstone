import os
from typing import Any
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore

from memory.short_term import format_history_for_prompt

from .types import AgentState, existence_grounder_enabled


def _bail_out_strict_enabled() -> bool:
    # P1-Next-A: tighten the 'No matching clip is expected.' bail-out.
    # When enabled (default), ``rows>0`` cases are forbidden from emitting
    # the bail-out string unless the existence-grounder explicitly classified
    # the query as a verified ``mismatch``. Set ``=0`` to recover the legacy
    # behaviour where any code path may yield 'No matching clip is expected.'
    raw = os.getenv("AGENT_SUMMARY_BAIL_OUT_STRICT", "1").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _select_rows(state: AgentState) -> list[dict[str, Any]]:
    if "rerank_result" in state:
        return list(state.get("rerank_result") or [])
    if "hybrid_result" in state:
        return list(state.get("hybrid_result") or [])
    return list(state.get("sql_result") or [])


def _infer_source_type(row: dict[str, Any]) -> str:
    trace = row.get("_fusion_trace") if isinstance(row.get("_fusion_trace"), dict) else {}
    if trace.get("sql_rank") and trace.get("hybrid_rank"):
        return "mixed"
    if trace.get("sql_rank"):
        return "sql"
    if trace.get("hybrid_rank"):
        return "hybrid"
    if row.get("_distance") not in (None, "N/A"):
        return "hybrid"
    return "sql"


def _build_citations(rows: list[dict[str, Any]], limit: int = 3) -> list[dict[str, Any]]:
    citations: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        source_type = _infer_source_type(row)
        video_id = str(row.get("video_id") or "unknown_video")
        is_parent = str(row.get("_record_level") or "").lower() == "parent"
        event_id = None if is_parent else row.get("event_id")
        start_time = row.get("start_time")
        end_time = row.get("end_time")
        key = f"{source_type}|{video_id}|{event_id if event_id is not None else 'parent'}|{start_time}|{end_time}"
        if key in seen:
            continue
        seen.add(key)
        citations.append(
            {
                "source_type": source_type,
                "video_id": video_id,
                "event_id": event_id,
                "start_time": start_time,
                "end_time": end_time,
                "record_level": "parent" if is_parent else "child",
            }
        )
        if len(citations) >= limit:
            break
    return citations


def _render_citations(citations: list[dict[str, Any]]) -> str:
    if not citations:
        return ""
    rendered = []
    for item in citations:
        if item.get("record_level") == "parent":
            rendered.append(
                f"[{item['source_type']}] {item['video_id']} | parent | {item.get('start_time')}-{item.get('end_time')}"
            )
        else:
            rendered.append(
                f"[{item['source_type']}] {item['video_id']} | event_id={item.get('event_id')} | "
                f"{item.get('start_time')}-{item.get('end_time')}"
            )
    return "Sources: " + "; ".join(rendered)


def _format_clip_time(value: Any) -> str:
    try:
        seconds = max(0, int(round(float(value))))
    except Exception:
        return "unknown"
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours}:{minutes:02d}:{secs:02d}"


def _pick_primary_row(row: dict[str, Any]) -> dict[str, Any]:
    child_rows = row.get("_child_rows") if isinstance(row.get("_child_rows"), list) else []
    if not child_rows:
        return row

    def _sort_key(item: dict[str, Any]) -> tuple[float, float]:
        distance = item.get("_distance")
        hybrid = item.get("_hybrid_score")
        safe_distance = float(distance) if isinstance(distance, (int, float)) else 999999.0
        safe_hybrid = -float(hybrid) if isinstance(hybrid, (int, float)) else 0.0
        return (safe_distance, safe_hybrid)

    ranked_children = sorted(
        [item for item in child_rows if isinstance(item, dict)],
        key=_sort_key,
    )
    return ranked_children[0] if ranked_children else row


def _grounder_mismatch_forbids_positive_clip_summary(verifier_result: dict[str, Any] | None) -> bool:
    """Verifier mismatch must not be turned into a Yes-style clip answer when
    the existence grounder is on.

    This covers both LLM/heuristic re-selection mismatches and the stricter
    attribute guard where the time window matches but the asked appearance or
    exit direction contradicts the retrieved evidence.
    """
    if not existence_grounder_enabled():
        return False
    if not isinstance(verifier_result, dict):
        return False
    return str(verifier_result.get("decision") or "").strip().lower() == "mismatch"


def _looks_like_binary_query(text: str) -> bool:
    query = str(text or "").strip().lower()
    if not query:
        return False
    return bool(
        re.match(r"^(is|are|was|were|do|does|did|can|could|has|have|had)\b", query)
        or " is there " in f" {query} "
        or query.endswith("?")
    )


def _build_factual_summary(
    rows: list[dict[str, Any]],
    query: str,
    *,
    verifier_result: dict[str, Any] | None = None,
) -> str:
    """Build the canonical Yes / The-most-relevant template.

    P1-7 v2.3: when the verifier re-selected a different chunk inside
    ``rerank_result`` (``verifier_result.span_source == "rerank_reselected"``),
    prefer the verifier's ``video_id / start_time / end_time``. Otherwise fall
    back to the rerank top-1 row. ``rows`` empty still maps to the bail-out
    string so the early-return path in ``summary_node`` keeps working.

    P1-7 follow-up: if the existence grounder is on and the verifier still says
    ``mismatch`` for that re-selected span, return ``No matching clip is
    expected.`` instead of a false-positive Yes (``rerank_reselected`` can point
    at the wrong video). When the grounder is off, behaviour is unchanged.
    """
    if not rows:
        return "No matching clip is expected."

    if _grounder_mismatch_forbids_positive_clip_summary(verifier_result):
        return "No matching clip is expected."

    # v2.5: When the verifier was skipped (non-existence answer type), build
    # a factual summary from the top row instead of returning a bail-out.
    verifier_decision = str(verifier_result.get("decision") or "").strip().lower() if isinstance(verifier_result, dict) else ""
    if verifier_decision == "skipped":
        top_row = rows[0]
        primary = _pick_primary_row(top_row)
        video_id = str(primary.get("video_id") or top_row.get("video_id") or "unknown_video").strip()
        return f"The most relevant clip is in {video_id}."

    use_verifier_span = (
        isinstance(verifier_result, dict)
        and str(verifier_result.get("span_source") or "").strip().lower() == "rerank_reselected"
        and verifier_result.get("video_id")
    )
    if use_verifier_span:
        video_id = str(verifier_result.get("video_id") or "unknown_video").strip()
        start_time = verifier_result.get("start_time")
        end_time = verifier_result.get("end_time")
    else:
        top_row = rows[0]
        primary = _pick_primary_row(top_row)
        video_id = str(primary.get("video_id") or top_row.get("video_id") or "unknown_video").strip()
        start_time = primary.get("start_time", top_row.get("start_time"))
        end_time = primary.get("end_time", top_row.get("end_time"))

    start_text = _format_clip_time(start_time)
    end_text = _format_clip_time(end_time)
    if _looks_like_binary_query(query):
        return f"Yes. The relevant clip is in {video_id}, around {start_text} - {end_text}."
    return f"The most relevant clip is in {video_id}, around {start_text} - {end_text}."


def _normalize_summary_output(
    text: str,
    fallback: str,
    *,
    allow_no_match: bool = True,
) -> str:
    cleaned = " ".join(str(text or "").strip().split())
    if not cleaned:
        return fallback
    if "No matching clip is expected." in cleaned:
        # P1-Next-A: only honour the bail-out string when the caller explicitly
        # allows it (rows empty, or grounder verdict said mismatch). Otherwise
        # demote to the factual fallback so retrieval-correct cases stop
        # emitting a vacuously empty answer.
        return "No matching clip is expected." if allow_no_match else fallback
    if len(cleaned) > 140:
        return fallback
    if cleaned.count("Abuse") > 1:
        return fallback
    return cleaned


def _allow_no_match_decision(
    *,
    rows: list[dict[str, Any]],
    answer_type: str,
    verifier_decision: str,
    grounder_enabled: bool,
    bail_out_strict: bool,
) -> bool:
    # Truth table (P1-Next-A):
    #   rows == [] -> always allow (early return path also covers this).
    #   bail_out_strict == False -> always allow (legacy behaviour, escape hatch).
    #   grounder ON + answer_type == existence + verifier mismatch -> allow.
    #   otherwise -> forbid.
    if not rows:
        return True
    if not bail_out_strict:
        return True
    if (
        grounder_enabled
        and answer_type in {"existence", "unknown", "semantic", "structured"}
        and verifier_decision == "mismatch"
    ):
        return True
    return False


def _canonicalize_summary(
    text: str,
    *,
    fallback: str,
    rows: list[dict[str, Any]],
    query: str,
    answer_type: str = "",
    verifier_decision: str = "",
    grounder_enabled: bool = False,
    bail_out_strict: bool = True,
    verifier_result: dict[str, Any] | None = None,
) -> str:
    allow_no_match = _allow_no_match_decision(
        rows=rows,
        answer_type=answer_type,
        verifier_decision=verifier_decision,
        grounder_enabled=grounder_enabled,
        bail_out_strict=bail_out_strict,
    )
    normalized = _normalize_summary_output(text, fallback, allow_no_match=allow_no_match)
    if normalized == "No matching clip is expected.":
        return normalized
    if normalized.startswith("Yes. The relevant clip is in ") or normalized.startswith(
        "The most relevant clip is in "
    ):
        if _grounder_mismatch_forbids_positive_clip_summary(verifier_result):
            return _build_factual_summary(rows, query, verifier_result=verifier_result)
        return normalized
    return _build_factual_summary(rows, query, verifier_result=verifier_result)


def create_summary_node(llm: Any = None):
    def summary_node(state: AgentState, config: RunnableConfig, store: BaseStore) -> dict[str, Any]:
        del store
        raw_answer = str(state.get("raw_final_answer") or state.get("final_answer") or "").strip()
        rows = _select_rows(state)
        citations = _build_citations(rows)
        rendered_citations = _render_citations(citations)
        original_query = str(state.get("original_user_query") or state.get("user_query") or "").strip()
        rewritten_query = str(state.get("rewritten_query") or state.get("user_query") or "").strip()
        row_digest = [
            {
                "video_id": row.get("video_id"),
                "event_id": row.get("event_id"),
                "start_time": row.get("start_time"),
                "end_time": row.get("end_time"),
                "summary": row.get("event_summary_en") or row.get("event_text_en") or row.get("event_text_cn"),
            }
            for row in rows[:5]
        ]

        verifier_payload = state.get("verifier_result")
        if not isinstance(verifier_payload, dict):
            verifier_payload = None

        # ── v2.4: Multi-camera early return ──
        # When the verifier confirmed a multi-camera match, preserve the
        # multi-camera formatted answer from final_answer_node instead of
        # running it through the conservative single-video summary prompt.
        if isinstance(verifier_payload, dict) and verifier_payload.get("multi_camera"):
            final_text = raw_answer
            if rendered_citations:
                final_text += "\n" + rendered_citations
            return {
                "final_answer": final_text,
                "summary_result": {
                    "summary": raw_answer,
                    "style": "multi_camera_preserved",
                    "confidence": 0.9,
                    "citations": citations,
                },
                "current_node": "summary_node",
                "messages": [AIMessage(content=final_text)],
            }

        factual_fallback = _build_factual_summary(
            rows, original_query or rewritten_query, verifier_result=verifier_payload
        )
        fallback_summary = factual_fallback or raw_answer or "No matching clip is expected."
        if not rows:
            return {
                "final_answer": fallback_summary,
                "summary_result": {
                    "summary": fallback_summary,
                    "style": "empty_result_fallback",
                    "confidence": 0.95,
                    "citations": [],
                },
                "current_node": "summary_node",
                "messages": [AIMessage(content=fallback_summary)],
            }
        if llm is None:
            final_text = fallback_summary if not rendered_citations else f"{fallback_summary}\n{rendered_citations}"
            return {
                "final_answer": final_text,
                "summary_result": {
                    "summary": fallback_summary,
                    "style": "fallback",
                    "confidence": 0.3,
                    "citations": citations,
                },
                "current_node": "summary_node",
                "messages": [AIMessage(content=final_text)],
            }

        # P1-Next-A: when ``rows`` is non-empty we drop the bail-out clause
        # entirely. Only the empty-rows branch (handled earlier with an early
        # return) ever reaches the legacy 'No matching' wording.
        prompt_lines: list[str] = [
            "You are the final response summarizer for a retrieval assistant.",
            "Return a short factual answer that matches the reference-answer style used in evaluation.",
            "Use only the single strongest result. Do not merge multiple videos.",
            "Even if the evidence is partial, summarize using the strongest result.",
            "",
            "CONSERVATIVE EXISTENCE REASONING: If the retrieved evidence does NOT "
            "clearly and directly describe the specific action or scene the user "
            "asked about, answer 'No matching clip is expected.' A false positive "
            "(claiming a clip exists when it doesn't) is worse than a false negative. "
            "Do NOT infer or assume actions not explicitly stated in the evidence.",
            "When the evidence SUPPORTS the query, use EXACTLY: "
            "'Yes. The relevant clip is in <video_id>, around <h:mm:ss> - <h:mm:ss>.'",
            "For non-binary descriptive questions, use: "
            "'The most relevant clip is in <video_id>, around <h:mm:ss> - <h:mm:ss>.'",
            "Do NOT add extra scene details, explanations, or a sources section.",
        ]
        # Inject conversation history if short-term memory is enabled
        thread_id = ""
        user_id = ""
        if isinstance(config, dict) and "configurable" in config:
            cfg = config["configurable"]
            thread_id = str(cfg.get("thread_id", ""))
            user_id = str(cfg.get("user_id", ""))
        if thread_id and user_id:
            history_block = format_history_for_prompt(thread_id, user_id)
            if history_block:
                prompt_lines.extend(["", history_block])
        # Appearance conflict note (injected when verifier detected color/clothing mismatch)
        appearance_conflict = (
            verifier_payload.get("appearance_conflict")
            if isinstance(verifier_payload, dict)
            else None
        )
        if appearance_conflict:
            asked_str = ", ".join(appearance_conflict.get("asked") or [])
            found_str = ", ".join(appearance_conflict.get("found") or [])
            prompt_lines.extend([
                "",
                "APPEARANCE DISCREPANCY DETECTED:",
                f"  The user asked for: {asked_str}",
                f"  The retrieved evidence shows: {found_str}",
                "  Rule: A person WAS detected, but their appearance differs from what was queried.",
                "  You MUST acknowledge this in your answer, e.g.:",
                f"  'Yes, a person was detected at <camera> around <time>, but their appearance shows {found_str}, not {asked_str} as queried.'",
                "  Do NOT say the person was wearing the queried color — only report what the evidence shows.",
            ])

        prompt_lines.extend(
            [
                "",
                "CRITICAL: Never copy appearance attributes (colors, clothing) from the user query into",
                "your answer. Only report what the retrieved evidence explicitly states.",
                "",
                f"Original user query: {original_query}",
                f"Rewritten retrieval query: {rewritten_query}",
                f"Retrieved result count: {len(rows)}",
                f"Top results: {row_digest[:2]}",
                f"Preferred fallback answer: {factual_fallback}",
                f"Draft answer: {raw_answer}",
            ]
        )
        prompt = "\n".join(prompt_lines)
        verifier_decision = ""
        if isinstance(verifier_payload, dict):
            verifier_decision = str(verifier_payload.get("decision") or "").strip().lower()
        answer_type = str(state.get("answer_type") or "").strip().lower()
        grounder_enabled = existence_grounder_enabled()
        bail_out_strict = _bail_out_strict_enabled()
        allow_no_match = _allow_no_match_decision(
            rows=rows,
            answer_type=answer_type,
            verifier_decision=verifier_decision,
            grounder_enabled=grounder_enabled,
            bail_out_strict=bail_out_strict,
        )

        try:
            model = llm.bind(max_tokens=120) if hasattr(llm, "bind") else llm
            raw = model.invoke(
                [SystemMessage(content="Return only the final English summary text."), HumanMessage(content=prompt)],
                config=config,
            )
            summary_text = raw.content if hasattr(raw, "content") else str(raw)
            payload = {
                "summary": _normalize_summary_output(
                    str(summary_text), fallback_summary, allow_no_match=allow_no_match
                ),
                "style": "llm_summary",
                "confidence": 0.8,
            }
        except Exception:
            payload = {"summary": fallback_summary, "style": "fallback", "confidence": 0.3}

        final_summary = _canonicalize_summary(
            str(payload.get("summary", fallback_summary)).strip(),
            fallback=fallback_summary,
            rows=rows,
            query=original_query or rewritten_query,
            answer_type=answer_type,
            verifier_decision=verifier_decision,
            grounder_enabled=grounder_enabled,
            bail_out_strict=bail_out_strict,
            verifier_result=verifier_payload,
        )
        final_text = final_summary if not rendered_citations else f"{final_summary}\n{rendered_citations}"
        payload["citations"] = citations
        payload["summary"] = final_summary
        return {
            "final_answer": final_text,
            "summary_result": payload,
            "current_node": "summary_node",
            "messages": [AIMessage(content=final_text)],
        }

    return summary_node
