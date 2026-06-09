"""Existence-grounder node (P1-7 v2.3).

Only runs when ``state.answer_type == "existence"``. For other question
shapes it is a pass-through so retrieval scores are untouched.

In v2.3 (2026-05-02), the node became a *span re-selector*: instead of
inspecting only the rerank top-1 row, it presents the LLM with multiple
candidates from ``rerank_result`` (same-video chunks plus a few
cross-video top-1s for diversity) and asks it to pick the best matching
chunk together with a verdict. The chosen chunk's ``(video_id,
start_time, end_time, primary_summary)`` is written to
``verifier_result`` and downstream nodes (``summary_node``,
``final_answer_node``) prefer it whenever
``verifier_result.span_source == "rerank_reselected"``.

Why this matters: 56% of 50-case e2e runs had top-K filled by chunks
from a single video, and in 4 of 5 verifier=mismatch existence cases
the genuinely matching chunk was already inside ``rerank_result`` -
just not in the top-1 slot. Re-selecting inside ``rerank_result`` fixes
those without any new chroma fetch.

Compatibility:
- ``AGENT_VERIFIER_RESELECT_SPAN=0``: legacy single-row LLM verdict.
- ``AGENT_VERIFIER_USE_LLM=0`` (env): heuristic only (no LLM call).
- ``answer_type != "existence"``: pass-through, identical to v1 behaviour.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore

from .types import AgentState


_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "around",
    "after",
    "any",
    "before",
    "being",
    "by",
    "can",
    "clip",
    "conducting",
    "did",
    "does",
    "for",
    "from",
    "full",
    "have",
    "how",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "one",
    "or",
    "other",
    "over",
    "person",
    "there",
    "their",
    "then",
    "to",
    "two",
    "up",
    "video",
    "was",
    "were",
    "while",
    "with",
}

_SCENE_TERMS = {"road", "room", "sofa", "wall", "wheelchair", "conveyor", "parking", "court", "bleachers"}

_FAST_DECISIONS = {"exact", "partial", "mismatch"}

_DIRECTION_ALIASES = {
    "left": {"left", "left side"},
    "right": {"right", "right side"},
    "up": {"up", "top", "upper", "top side"},
    "down": {"down", "bottom", "lower", "bottom side"},
}

_COLOR_TERMS = {
    "black",
    "blue",
    "brown",
    "dark",
    "gray",
    "grey",
    "green",
    "khaki",
    "light",
    "olive",
    "purple",
    "red",
    "tan",
    "white",
    "yellow",
}

_COLOR_COMPATIBLE = {
    "black": {"black", "dark"},
    "dark": {"dark", "black", "brown", "blue", "gray", "grey"},
    "gray": {"gray", "grey", "light", "white"},
    "grey": {"grey", "gray", "light", "white"},
    "light": {"light", "white", "gray", "grey", "khaki", "tan"},
    "white": {"white", "light", "gray", "grey"},
    "khaki": {"khaki", "tan", "brown", "light"},
    "tan": {"tan", "khaki", "brown", "light"},
    "brown": {"brown", "tan", "khaki", "dark"},
}

_COLOR_CONTRAST = {
    "red": {"black", "dark", "blue", "green", "gray", "grey", "white", "yellow"},
    "yellow": {"black", "dark", "blue", "red", "gray", "grey"},
    "blue": {"red", "yellow", "green", "brown"},
    "green": {"red", "blue", "yellow"},
    "white": {"black", "dark", "red", "blue", "green", "brown"},
    "black": {"red", "yellow", "white", "light"},
    "dark": {"red", "yellow", "white", "light"},
}

_CLOTHING_TERMS = {
    "backpack",
    "bag",
    "beanie",
    "coat",
    "hoodie",
    "jacket",
    "pants",
    "shirt",
    "top",
    "trousers",
}


def _parse_clock_to_sec(text: str) -> float | None:
    parts = str(text or "").strip().split(":")
    if len(parts) == 2:
        try:
            return float(parts[0]) * 60.0 + float(parts[1])
        except Exception:
            return None
    if len(parts) == 3:
        try:
            return float(parts[0]) * 3600.0 + float(parts[1]) * 60.0 + float(parts[2])
        except Exception:
            return None
    return None


def _query_time_window(query: str) -> tuple[float, float] | None:
    q = str(query or "").lower()
    match = re.search(r"\baround\s+(\d{1,2}:\d{1,2}(?::\d{1,2})?)\s*-\s*(\d{1,2}:\d{1,2}(?::\d{1,2})?)", q)
    if not match:
        return None
    start = _parse_clock_to_sec(match.group(1))
    end = _parse_clock_to_sec(match.group(2))
    if start is None or end is None:
        return None
    if end < start:
        start, end = end, start
    return start, end


def _row_time_overlap(row: dict[str, Any], window: tuple[float, float] | None) -> float | None:
    if window is None:
        return None
    primary = _pick_primary_row(row)
    try:
        start = float(primary.get("start_time", row.get("start_time")))
        end = float(primary.get("end_time", row.get("end_time")))
    except Exception:
        return None
    if end < start:
        start, end = end, start
    return max(0.0, min(end, window[1]) - max(start, window[0]))


def _row_matches_query_time(row: dict[str, Any], query: str) -> bool | None:
    window = _query_time_window(query)
    if window is None:
        return None
    overlap = _row_time_overlap(row, window)
    return bool(overlap is not None and overlap > 0.0)


# ---------------------------------------------------------------------------
# v2.3 env knobs
# ---------------------------------------------------------------------------


def _reselect_span_enabled() -> bool:
    raw = os.getenv("AGENT_VERIFIER_RESELECT_SPAN", "1").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _candidate_limit() -> int:
    raw = os.getenv("AGENT_VERIFIER_CANDIDATE_LIMIT", "8").strip()
    try:
        value = int(raw)
    except Exception:
        return 8
    return max(1, value)


def _cross_video_top_n() -> int:
    raw = os.getenv("AGENT_VERIFIER_CROSS_VIDEO_TOP_N", "2").strip()
    try:
        value = int(raw)
    except Exception:
        return 2
    return max(0, value)


def _llm_disabled() -> bool:
    raw = os.getenv("AGENT_MATCH_VERIFIER_USE_LLM", "1").strip().lower()
    return raw in {"0", "false", "no", "off"}


# ---------------------------------------------------------------------------
# Row helpers (shared with v1)
# ---------------------------------------------------------------------------


def _select_rows(state: AgentState) -> list[dict[str, Any]]:
    if "rerank_result" in state:
        return list(state.get("rerank_result") or [])
    if "hybrid_result" in state:
        return list(state.get("hybrid_result") or [])
    return list(state.get("sql_result") or [])


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


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(text or "").lower())


def _mentioned_cameras(text: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for cam in re.findall(r"\bG\d+\b", str(text or ""), flags=re.IGNORECASE):
        cam = cam.upper()
        if cam not in seen:
            out.append(cam)
            seen.add(cam)
    return out


def _looks_like_cross_camera_query(query: str) -> bool:
    q = str(query or "").lower()
    return len(_mentioned_cameras(query)) >= 2 and any(
        phrase in q
        for phrase in (
            "also appear",
            "appear again",
            "then appear",
            "same person",
            "cross-camera",
            "cross camera",
        )
    )


def _query_terms(query: str) -> set[str]:
    terms: set[str] = set()
    for token in _tokenize(query):
        if len(token) < 3 or token in _STOPWORDS:
            continue
        terms.add(token)
    return terms


def _evidence_text(row: dict[str, Any], primary: dict[str, Any] | None = None) -> str:
    primary = primary or row
    parts = [
        row.get("event_summary_en"),
        row.get("event_text_en"),
        row.get("event_text"),
        row.get("appearance_notes"),
        row.get("appearance_notes_en"),
        row.get("keywords"),
        primary.get("event_summary_en"),
        primary.get("event_text_en"),
        primary.get("event_text"),
        primary.get("appearance_notes"),
        primary.get("appearance_notes_en"),
        primary.get("keywords"),
        row.get("object_type"),
        row.get("object_color_en"),
        row.get("object_color"),
        row.get("scene_zone_en"),
    ]
    rendered: list[str] = []
    for item in parts:
        if isinstance(item, list):
            rendered.append(" ".join(str(x) for x in item))
        elif item:
            rendered.append(str(item))
    return " ".join(rendered).lower()


def _row_supports_mentioned_cameras(row: dict[str, Any], query: str) -> bool:
    cams = _mentioned_cameras(query)
    if len(cams) < 2:
        return True
    primary = _pick_primary_row(row)
    text = _evidence_text(row, primary).upper()
    row_video = str(primary.get("video_id") or row.get("video_id") or "").upper()
    # A row can support a cross-camera query either through explicit trajectory
    # text or through its own video id plus trajectory text from child rows.
    haystack = f"{text} {row_video}"
    return all(cam in haystack for cam in cams)


def _candidates_cover_cameras(
    query: str,
    candidates: list[dict[str, Any]],
) -> bool:
    """Return True if some entity_hint appears in candidates for ALL mentioned cameras.

    This allows Q09-style cross-camera queries (G329→G421) to pass when the
    candidates collectively include both G329 and G421 records for the same
    tracked entity, even though no single row mentions both camera IDs.
    """
    cams = _mentioned_cameras(query)
    if len(cams) < 2:
        return True
    # Group candidate video_ids by entity identity
    entity_cams: dict[str, set[str]] = {}
    for c in candidates:
        hint = str(c.get("entity_hint") or c.get("global_entity_id") or "").strip()
        if not hint:
            # Chroma track records embed "Track person_global_N" in event_text;
            # event_id format is "{video_id}_person_global_N" for global-entity tracks.
            for field in ("event_text", "event_id", "event_summary_en", "event_text_en"):
                val = str(c.get(field) or "")
                m = re.search(r"\bperson_global_\d+\b", val, re.IGNORECASE)
                if m:
                    hint = m.group(0).lower()
                    break
        if not hint:
            continue
        vid = str(c.get("video_id") or "").upper()
        for cam in cams:
            if cam in vid:
                entity_cams.setdefault(hint, set()).add(cam)
    return any(set(cams).issubset(covered) for covered in entity_cams.values())


def _cross_camera_verdict(
    query: str,
    row: dict[str, Any],
    candidates: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    if not _looks_like_cross_camera_query(query):
        return None
    cams = _mentioned_cameras(query)
    if _row_supports_mentioned_cameras(row, query):
        return None
    # Collective check: if the candidate pool contains the same entity in all
    # queried cameras, trust the LLM verdict rather than forcing mismatch.
    if candidates and _candidates_cover_cameras(query, candidates):
        return None
    return {
        "decision": "mismatch",
        "confidence": 0.97,
        "reason": f"cross_camera_camera_pair_missing: required={cams}",
        "mode": "strict_rule",
    }


def _prefer_cross_camera_candidate(query: str, candidates: list[dict[str, Any]], current_idx: int) -> int:
    """For cross-camera queries, prefer a candidate that mentions every requested camera.

    This prevents a visually similar trajectory such as G421->G424 from answering
    a query that explicitly asks about G421->G508.
    """
    if not _looks_like_cross_camera_query(query) or not candidates:
        return current_idx
    if 0 <= current_idx < len(candidates) and _row_supports_mentioned_cameras(candidates[current_idx], query):
        return current_idx
    for idx, row in enumerate(candidates):
        if _row_supports_mentioned_cameras(row, query):
            return idx
    return current_idx


def _looks_like_binary_query(text: str) -> bool:
    query = str(text or "").strip().lower()
    if not query:
        return False
    return bool(
        re.match(r"^(is|are|was|were|do|does|did|can|could|has|have|had)\b", query)
        or " is there " in f" {query} "
        or query.endswith("?")
    )


def _query_direction(query: str) -> str:
    q = str(query or "").lower()
    patterns = [
        r"\bexit(?:ed|s)?\s+from\s+the\s+([a-z ]+?)\s+side\b",
        r"\bleav(?:e|es|ing)\s+(?:from\s+)?the\s+([a-z ]+?)\s+side\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, q)
        if not match:
            continue
        raw = match.group(1).strip()
        if raw in {"top", "upper"}:
            return "up"
        if raw in {"bottom", "lower"}:
            return "down"
        if raw in _DIRECTION_ALIASES:
            return raw
    return ""


def _evidence_direction(evidence: str) -> str:
    text = f" {str(evidence or '').lower()} "
    patterns = [
        r"\bexit(?:ed|s|ing)?\s+from\s+the\s+(left|right|up|down|top|bottom)\s+side\b",
        r"\bleav(?:e|es|ing|t)?\s+(?:frame\s+)?(?:from\s+)?(?:the\s+)?(left|right|up|down|top|bottom)\b",
        r"\bperson\s+leaving\s+frame\s+(left|right|up|down|top|bottom)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        raw = match.group(1)
        if raw == "top":
            return "up"
        if raw == "bottom":
            return "down"
        return raw
    return ""


def _query_appearance_terms(query: str) -> set[str]:
    q = str(query or "").lower()
    phrase = ""
    patterns = [
        r"person\s+with\s+(.+?)\s+visible",
        r"person\s+wearing\s+(.+?)\s+in\s+camera",
        r"person\s+with\s+(.+?)\s+in\s+camera",
        r"wearing\s+(.+?)(?:\?|$)",
        # "person in white", "person in a dark jacket", etc.
        r"person\s+in\s+(?:a\s+|an\s+)?(.+?)(?:\s+appear|\s+in\s+(?:the\s+)?camera|\s*\?|$)",
        # "any person in white", "someone in black coat"
        r"(?:any\s+)?(?:person|someone|man|woman)\s+in\s+(?:a\s+|an\s+)?(.+?)(?:\s+appear|\s+in\s+(?:the\s+)?camera|\s*\?|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, q)
        if match:
            phrase = match.group(1)
            break
    # Fallback: if no phrase matched, scan the full query for any color adjacent to
    # clothing/appearance terms (e.g. "white shirt", "dark coat", "in white")
    if not phrase:
        tokens = _tokenize(q)
        terms: set[str] = set()
        for i, tok in enumerate(tokens):
            if tok in _COLOR_TERMS:
                # Color preceded or followed by clothing term counts
                neighbors = tokens[max(0, i - 1):i] + tokens[i + 1:i + 2]
                if any(n in _CLOTHING_TERMS for n in neighbors):
                    terms.add(tok)
                # Color immediately after "in" / "wearing" also counts
                if i > 0 and tokens[i - 1] in {"in", "wearing"}:
                    terms.add(tok)
        return terms
    return {
        token
        for token in _tokenize(phrase)
        if token in _COLOR_TERMS or token in _CLOTHING_TERMS
    }


def _strict_attribute_verdict(query: str, row: dict[str, Any]) -> dict[str, Any] | None:
    """Conservative yes/no guard for attribute-negative questions.

    Retrieval may correctly hit the requested time window while the retrieved
    evidence contradicts the asked attribute (e.g. query asks "red jacket" but
    evidence says "black coat"). In that case the final answer should be no,
    not "yes because someone exists in the window".
    """
    if not _looks_like_binary_query(query):
        return None

    primary = _pick_primary_row(row)
    evidence = _evidence_text(row, primary)

    time_match = _row_matches_query_time(row, query)
    if time_match is False:
        return {
            "decision": "mismatch",
            "confidence": 0.96,
            "reason": "time_conflict: evidence does not overlap requested query window",
            "mode": "strict_rule",
        }

    asked_direction = _query_direction(query)
    if asked_direction:
        actual_direction = _evidence_direction(evidence)
        if actual_direction and actual_direction != asked_direction:
            return {
                "decision": "mismatch",
                "confidence": 0.96,
                "reason": f"direction_conflict: asked={asked_direction}; evidence={actual_direction}",
                "mode": "strict_rule",
            }

    appearance_terms = _query_appearance_terms(query)
    if appearance_terms:
        asked_colors = appearance_terms & _COLOR_TERMS
        evidence_colors = {token for token in _tokenize(evidence) if token in _COLOR_TERMS}
        # Also read object_color / object_color_en fields directly from the row
        # (event text may omit color even when the DB field is populated)
        for _color_field in ("object_color", "object_color_en"):
            _raw = str(primary.get(_color_field) or row.get(_color_field) or "").strip().lower().replace("_", " ")
            for tok in _tokenize(_raw):
                if tok in _COLOR_TERMS:
                    evidence_colors.add(tok)
        # Only reject on explicit contradiction. Generic evidence like
        # "dark upper-body clothing" should still support "black coat" weakly,
        # but it should reject "red jacket" when the time-matched evidence
        # explicitly says dark/black.
        compatible_evidence = set(evidence_colors)
        for color in evidence_colors:
            compatible_evidence.update(_COLOR_COMPATIBLE.get(color, set()))
        explicit_conflicts = [
            asked
            for asked in asked_colors
            if any(evidence_color in _COLOR_CONTRAST.get(asked, set()) for evidence_color in evidence_colors)
        ]
        if asked_colors and evidence_colors and explicit_conflicts and not asked_colors.intersection(compatible_evidence):
            return {
                "decision": "partial",
                "confidence": 0.72,
                "reason": f"appearance_conflict: asked_colors={sorted(asked_colors)}; evidence_colors={sorted(evidence_colors)}",
                "appearance_conflict": {
                    "asked": sorted(asked_colors),
                    "found": sorted(evidence_colors),
                },
                "mode": "strict_rule",
            }

    if time_match is True:
        return {
            "decision": "exact",
            "confidence": 0.74,
            "reason": "time_window_match_without_explicit_attribute_conflict",
            "mode": "strict_rule",
        }
    return None


def _prefer_time_matching_candidate(query: str, candidates: list[dict[str, Any]], current_idx: int) -> int:
    window = _query_time_window(query)
    if window is None or not candidates:
        return current_idx
    current_overlap = _row_time_overlap(candidates[current_idx], window)
    if current_overlap and current_overlap > 0:
        return current_idx
    best_idx = current_idx
    best_overlap = 0.0
    for idx, row in enumerate(candidates):
        overlap = _row_time_overlap(row, window) or 0.0
        if overlap > best_overlap:
            best_overlap = overlap
            best_idx = idx
    return best_idx if best_overlap > 0 else current_idx


def _row_summary(row: dict[str, Any], max_chars: int = 220) -> str:
    """Compact text used inside the multi-candidate prompt."""
    primary = _pick_primary_row(row)
    text = (
        primary.get("event_summary_en")
        or primary.get("event_text_en")
        or row.get("event_summary_en")
        or row.get("event_text_en")
        or ""
    )
    cleaned = " ".join(str(text).strip().split())
    return cleaned[: max_chars - 1] + "..." if len(cleaned) > max_chars else cleaned


def _row_descriptor(row: dict[str, Any]) -> dict[str, Any]:
    """Extract candidate fields downstream needs (used both in prompt and in verifier_result)."""
    primary = _pick_primary_row(row)
    return {
        "video_id": str(primary.get("video_id") or row.get("video_id") or "").strip(),
        "start_time": primary.get("start_time", row.get("start_time")),
        "end_time": primary.get("end_time", row.get("end_time")),
        "primary_summary": _row_summary(row, max_chars=400),
    }


# ---------------------------------------------------------------------------
# v1 single-row paths (kept as the legacy fallback)
# ---------------------------------------------------------------------------


def _heuristic_verdict_single(query: str, row: dict[str, Any], primary: dict[str, Any]) -> dict[str, Any]:
    evidence = _evidence_text(row, primary)
    query_terms = _query_terms(query)
    if not query_terms:
        return {"decision": "partial", "confidence": 0.5, "reason": "empty_query_terms", "mode": "heuristic"}

    matched = sorted(term for term in query_terms if term in evidence)
    scene_terms = [term for term in query_terms if term in _SCENE_TERMS]
    scene_conflict = bool(scene_terms) and any(term not in evidence for term in scene_terms)
    coverage = len(matched) / max(len(query_terms), 1)

    if scene_conflict:
        return {"decision": "mismatch", "confidence": 0.85, "reason": "scene_conflict", "mode": "heuristic"}
    if coverage >= 0.58:
        return {
            "decision": "exact",
            "confidence": min(0.95, 0.55 + coverage / 2),
            "reason": f"coverage={coverage:.2f}; matched={matched[:6]}",
            "mode": "heuristic",
        }
    if coverage >= 0.34:
        return {
            "decision": "partial",
            "confidence": min(0.8, 0.35 + coverage / 2),
            "reason": f"coverage={coverage:.2f}; matched={matched[:6]}",
            "mode": "heuristic",
        }
    return {
        "decision": "mismatch",
        "confidence": max(0.55, 1 - coverage),
        "reason": f"coverage={coverage:.2f}; matched={matched[:6]}",
        "mode": "heuristic",
    }


def _parse_json_object(text: str) -> dict[str, Any] | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        match = re.search(r"\{[\s\S]*\}", raw)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except Exception:
            return None


def _llm_verdict_single(
    llm: Any,
    query: str,
    row: dict[str, Any],
    primary: dict[str, Any],
    config: RunnableConfig,
) -> dict[str, Any] | None:
    if llm is None or _llm_disabled():
        return None
    prompt = (
        "Decide whether the retrieved evidence truly matches the user query.\n"
        "Be strict and conservative.\n"
        "Return JSON only with keys: decision, confidence, reason.\n"
        "decision must be one of: exact, partial, mismatch.\n"
        "- exact: the evidence directly describes the QUERIED action with matching "
        "subject, object, and scene. Synonyms count as match.\n"
        "- partial: core incident matches but details or time boundaries are coarse, "
        "or the subject/object role is slightly different.\n"
        "- mismatch: evidence is only LOOSELY related (same general scene but "
        "different action), mixes different incidents, or requires assumptions "
        "to connect to the query. Thematic similarity without the specific "
        "queried action is NOT a match.\n"
        f"\nUser query: {query}"
        f"\nTop row video: {row.get('video_id')}"
        f"\nTop row time: {row.get('start_time')} - {row.get('end_time')}"
        f"\nTop row summary: {row.get('event_summary_en') or row.get('event_text_en') or ''}"
        f"\nPrimary evidence time: {primary.get('start_time')} - {primary.get('end_time')}"
        f"\nPrimary evidence summary: {primary.get('event_summary_en') or primary.get('event_text_en') or ''}"
    )
    try:
        model = llm.bind(max_tokens=120) if hasattr(llm, "bind") else llm
        raw = model.invoke(
            [
                SystemMessage(content="You are a strict retrieval match verifier. Return JSON only."),
                HumanMessage(content=prompt),
            ],
            config=config,
        )
        content = raw.content if hasattr(raw, "content") else str(raw)
        data = _parse_json_object(str(content))
        if not isinstance(data, dict):
            return None
        decision = str(data.get("decision") or "").strip().lower()
        if decision not in _FAST_DECISIONS:
            return None
        try:
            confidence = float(data.get("confidence", 0.0))
        except Exception:
            confidence = 0.0
        return {
            "decision": decision,
            "confidence": max(0.0, min(1.0, confidence)),
            "reason": str(data.get("reason") or "").strip()[:200],
            "mode": "llm",
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# v2.3 multi-candidate path
# ---------------------------------------------------------------------------


def _collect_candidates(
    rerank_rows: list[dict[str, Any]],
    *,
    candidate_limit: int,
    cross_video_top_n: int,
) -> list[dict[str, Any]]:
    """Pick the candidate set the verifier LLM will inspect.

    Strategy:
      1. Take ``rerank_rows[:candidate_limit]`` as the working pool (already
         distance-/RRF-sorted upstream).
      2. Identify the top video_id (rerank top-1's video_id).
      3. ``same_video`` = all chunks from that video (preserves rerank order).
      4. ``other_video`` = top-N chunks from *other* videos (so the LLM still
         has a way out if retrieval picked the wrong video; usually small).
      5. Final order: same_video first, then other_video. ``index 0`` is the
         current rerank top-1 (used as the reference baseline).
    """
    pool = list(rerank_rows[:candidate_limit])
    if not pool:
        return []
    top_video_id = str(pool[0].get("video_id") or "").strip()
    same_video: list[dict[str, Any]] = []
    other_video: list[dict[str, Any]] = []
    for row in pool:
        if str(row.get("video_id") or "").strip() == top_video_id:
            same_video.append(row)
        else:
            other_video.append(row)
    candidates = same_video + other_video[:cross_video_top_n]
    # always keep at least the rerank top-1 as candidate index 0
    if not candidates:
        candidates = [pool[0]]
    return candidates


def _heuristic_select_best_chunk(
    query: str, candidates: list[dict[str, Any]]
) -> tuple[int, dict[str, Any]]:
    """Token-overlap heuristic: pick the candidate whose evidence covers the
    most query terms. Ties resolved by preferring the earlier (higher-rerank)
    candidate.

    Returns (best_index, verdict_dict).
    """
    query_terms = _query_terms(query)
    if not query_terms or not candidates:
        return 0, {
            "decision": "partial" if candidates else "mismatch",
            "confidence": 0.5 if candidates else 0.95,
            "reason": "empty_query_terms" if not query_terms else "no_candidates",
            "mode": "heuristic",
        }

    best_idx = 0
    best_coverage = -1.0
    best_matched: list[str] = []
    for idx, row in enumerate(candidates):
        primary = _pick_primary_row(row)
        evidence = _evidence_text(row, primary)
        matched = sorted(term for term in query_terms if term in evidence)
        coverage = len(matched) / max(len(query_terms), 1)
        if coverage > best_coverage:
            best_coverage = coverage
            best_idx = idx
            best_matched = matched

    # Decision band uses the *winning* candidate's coverage.
    if best_coverage >= 0.58:
        decision = "exact"
        confidence = min(0.95, 0.55 + best_coverage / 2)
    elif best_coverage >= 0.34:
        decision = "partial"
        confidence = min(0.8, 0.35 + best_coverage / 2)
    else:
        decision = "mismatch"
        confidence = max(0.55, 1.0 - best_coverage)

    return best_idx, {
        "decision": decision,
        "confidence": confidence,
        "reason": f"heuristic best_idx={best_idx} coverage={best_coverage:.2f} matched={best_matched[:6]}",
        "mode": "heuristic",
    }


def _has_attribute_signals(query: str) -> bool:
    """Detect whether the query contains fine-grained attribute signals
    (color, clothing, direction, absence) that event summaries may omit."""
    low = (query or "").strip().lower()
    color_words = {
        "black", "white", "red", "blue", "green", "grey", "gray",
        "brown", "beige", "khaki", "olive", "yellow", "orange", "pink",
        "dark", "light", "bright",
    }
    clothing_words = {
        "jacket", "coat", "hoodie", "shirt", "top", "trousers", "pants",
        "bag", "hat", "hood", "scarf", "jeans", "sweater", "shoes",
        "carrying", "wearing",
    }
    direction_words = {
        "exit from", "enter from", "left side", "right side",
        "up side", "down side", "top side", "bottom side",
        "bottom-left", "bottom-right", "top-left", "top-right",
    }
    absence_words = {"no bag", "not carrying", "without", "no visible"}
    tokens = set(re.findall(r"\b[a-z]+\b", low))
    has_color = bool(tokens & color_words)
    has_clothing = bool(tokens & clothing_words)
    has_direction = any(d in low for d in direction_words)
    has_absence = any(a in low for a in absence_words)
    return has_color or has_clothing or has_direction or has_absence


def _llm_select_best_chunk(
    llm: Any,
    query: str,
    candidates: list[dict[str, Any]],
    config: RunnableConfig,
) -> tuple[int, dict[str, Any]] | None:
    """Single-shot LLM call: pick the best candidate AND a verdict in one go.

    Returns (best_index, verdict_dict) or None on any LLM/parse failure
    (caller falls back to heuristic).
    """
    if llm is None or _llm_disabled() or not candidates:
        return None

    candidate_lines: list[str] = []
    for idx, row in enumerate(candidates):
        primary = _pick_primary_row(row)
        v = str(primary.get("video_id") or row.get("video_id") or "").strip()
        t1 = primary.get("start_time", row.get("start_time"))
        t2 = primary.get("end_time", row.get("end_time"))
        s = _row_summary(row, max_chars=220)
        candidate_lines.append(f"[{idx}] video={v} time={t1}-{t2} summary={s}")

    prompt_lines = [
        "You are an existence-query verifier for a video event database.",
        "",
        f"User query: {query}",
        "",
        "Below are candidate event chunks retrieved for this query (already "
        "ranked by retrieval score; index 0 is rerank top-1).",
        "Pick the SINGLE BEST chunk that most directly matches the query, "
        "and decide whether the picked chunk supports the query.",
        "",
        "Decision guide:",
        "- exact: picked chunk clearly describes the queried action with matching "
        "subject, object, and scene. Synonyms count as match.",
        "- partial: picked chunk is in the right context but details or subject "
        "roles are coarser or slightly different.",
        "- mismatch: NO chunk in the list supports the query. The evidence may "
        "be from the same general scene but describes a different action, or "
        "involves different actors/objects. Thematic similarity without the "
        "specific queried action does NOT count as a match.",
    ]

    # P1/P2: Attribute-heavy query (color/clothing/direction/absence) → relaxed prompt
    if _has_attribute_signals(query):
        prompt_lines.extend([
            "",
            "ATTRIBUTE-AWARE GUIDANCE: The query asks about specific attributes "
            "(color, clothing, spatial direction, or absence of an item). "
            "Event summaries often omit these fine-grained attributes even when "
            "the video contains them — the summary captures the general action, "
            "not every visual detail.",
            "",
            "- If a candidate chunk matches the CORRECT CAMERA and describes the "
            "same TYPE of entity/action (e.g., 'a person walks'), give at least "
            "'partial' even if the exact color/clothing/direction is not mentioned.",
            "- Camera match + entity type + action type is SUFFICIENT for partial.",
            "- If the query mentions ABSENCE ('no bag', 'without', 'not carrying'), "
            "the absence of that attribute in the summary does NOT disqualify "
            "the match — event summaries rarely describe what is NOT present.",
            "- Only choose 'mismatch' when the entity type clearly differs "
            "(e.g., car vs person) or the action contradicts the query.",
            "",
            "Be practical, not perfectionist. Camera + entity match alone is "
            "valuable evidence.",
        ])
    else:
        prompt_lines.extend([
            "",
            "CRITICAL: Be conservative. If the evidence only loosely resembles the "
            "query or requires assumptions to connect, choose mismatch. A false "
            "positive (saying a clip exists when it doesn't) is worse than a false "
            "negative. Verify that the specific action described in the query "
            "appears in the chosen chunk.",
        ])

    prompt_lines.extend([
        "",
        "CANDIDATES:",
    ])
    prompt_lines.extend(candidate_lines)
    prompt_lines.extend(
        [
            "",
            "Return JSON only:",
            "{",
            '  "best_chunk_index": <int, 0-based, in [0..n-1]>,',
            '  "decision": "exact" | "partial" | "mismatch",',
            '  "confidence": <float 0.0-1.0>,',
            '  "reason": "<one-line evidence-based>"',
            "}",
        ]
    )
    prompt = "\n".join(prompt_lines)

    try:
        model = llm.bind(max_tokens=200) if hasattr(llm, "bind") else llm
        raw = model.invoke(
            [
                SystemMessage(content="You are a video event verifier. Return JSON only."),
                HumanMessage(content=prompt),
            ],
            config=config,
        )
        content = raw.content if hasattr(raw, "content") else str(raw)
        data = _parse_json_object(str(content))
        if not isinstance(data, dict):
            return None
        try:
            best_idx = int(data.get("best_chunk_index", 0))
        except Exception:
            return None
        if best_idx < 0 or best_idx >= len(candidates):
            return None
        decision = str(data.get("decision") or "").strip().lower()
        if decision not in _FAST_DECISIONS:
            return None
        try:
            confidence = float(data.get("confidence", 0.0))
        except Exception:
            confidence = 0.0
        return best_idx, {
            "decision": decision,
            "confidence": max(0.0, min(1.0, confidence)),
            "reason": str(data.get("reason") or "").strip()[:200],
            "mode": "llm",
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Multi-camera verdict (v2.4)
# ---------------------------------------------------------------------------


def _detect_multi_camera_context(state: AgentState) -> bool:
    """Check whether the current query is a multi-camera scenario."""
    cr = state.get("classification_result")
    if isinstance(cr, dict):
        if cr.get("multi_camera"):
            return True
        signals = cr.get("signals")
        if isinstance(signals, dict) and signals.get("multi_camera_cues"):
            return True
    return False


def _collect_ge_rows(state: AgentState) -> list[dict[str, Any]]:
    """Gather GE rows from fusion result or global_entity_result."""
    # GE rows may be in rerank_result (post-fusion) or global_entity_result (pre-fusion).
    ge_result = state.get("global_entity_result")
    if isinstance(ge_result, list) and ge_result:
        return ge_result
    # Fallback: check rerank_result for rows with _source_type == "global_entity"
    rerank = state.get("rerank_result")
    if isinstance(rerank, list):
        return [r for r in rerank if isinstance(r, dict) and r.get("_source_type") == "global_entity"]
    return []


def _group_by_global_entity(ge_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group GE rows by global_entity_id, keeping per-camera info."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in ge_rows:
        ge_id = str(row.get("global_entity_id") or "").strip()
        if not ge_id:
            continue
        groups.setdefault(ge_id, []).append(row)
    return groups


def _fetch_camera_appearances_from_sqlite(
    camera_ids: list[str],
    max_per_camera: int = 6,
) -> dict[str, list[str]]:
    """Query SQLite directly for per-camera appearance descriptions.

    Bypasses vector retrieval ranking so appearance evidence is always
    available regardless of which GE rows ranked highest.
    """
    import sqlite3

    db_path = os.getenv("AGENT_SQLITE_DB_PATH", "").strip()
    if not db_path or not os.path.exists(db_path):
        return {}
    result: dict[str, list[str]] = {}
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        for cam in camera_ids:
            rows = conn.execute(
                "SELECT DISTINCT appearance_notes_en, object_color_en, event_text_en "
                "FROM episodic_events "
                "WHERE camera_id = ? AND object_type = 'person' "
                "AND (appearance_notes_en != '' OR object_color_en != '') "
                "LIMIT ?",
                (cam, max_per_camera),
            ).fetchall()
            snippets: list[str] = []
            for r in rows:
                notes = str(r["appearance_notes_en"] or "").strip()
                text = str(r["event_text_en"] or "").strip()
                snippet = notes or text[:100]
                if snippet and snippet not in snippets:
                    snippets.append(snippet)
            if snippets:
                result[cam] = snippets
        conn.close()
    except Exception:
        pass
    return result


_CAMERA_ID_RE = re.compile(r"\b([A-Z]\d{3})\b")
_MAX_CROSS_CAMERA_TRANSIT_SEC = 240.0  # sequential gap threshold


def _extract_queried_cameras(query: str) -> set[str]:
    """Extract camera IDs (e.g. G421, G506) mentioned in the query."""
    return set(_CAMERA_ID_RE.findall(query))


def _per_camera_time_ranges(
    events: list[dict[str, Any]],
) -> dict[str, tuple[float, float]]:
    """Compute (min_start, max_end) per camera for an entity's events."""
    ranges: dict[str, list[float]] = {}
    for e in events:
        cam = str(e.get("camera_id") or e.get("video_id") or "").strip()
        if not cam:
            continue
        try:
            st = float(e["start_time"])
            et = float(e["end_time"])
        except (KeyError, TypeError, ValueError):
            continue
        bucket = ranges.setdefault(cam, [float("inf"), float("-inf")])
        bucket[0] = min(bucket[0], st)
        bucket[1] = max(bucket[1], et)
    return {c: (v[0], v[1]) for c, v in ranges.items() if v[0] != float("inf")}


def _camera_coverage_prefilter(
    query: str,
    ge_groups: dict[str, list[dict[str, Any]]],
) -> dict[str, Any] | None:
    """Return a mismatch verdict immediately if no GE entity covers all queried cameras.

    Two hard checks (run before calling the LLM):
    1. Camera coverage: at least one entity must appear in ALL cameras the query mentions.
    2. Temporal gap: if the entity's appearances in the queried cameras are sequential
       (non-overlapping) and the gap exceeds _MAX_CROSS_CAMERA_TRANSIT_SEC, the
       appearances are too far apart to be the same person in a single slot.
    """
    queried_cameras = _extract_queried_cameras(query)
    if not queried_cameras:
        return None  # can't determine cameras → skip filter

    # Check 1: camera coverage
    covering: list[str] = []
    for ge_id, events in ge_groups.items():
        entity_cams = {
            str(e.get("camera_id") or e.get("video_id") or "").strip()
            for e in events
        } - {""}
        if queried_cameras.issubset(entity_cams):
            covering.append(ge_id)

    if not covering:
        return {
            "decision": "mismatch",
            "confidence": 0.95,
            "reason": (
                f"no global entity covers all queried cameras "
                f"{sorted(queried_cameras)} — cannot confirm cross-camera identity"
            ),
            "mode": "pre_filter_camera_coverage",
            "best_global_entity_id": "",
        }

    # Check 2: temporal gap — only fires when appearances are sequential (non-overlapping)
    all_gap_failures: list[str] = []
    for ge_id in covering:
        cam_ranges = _per_camera_time_ranges(ge_groups[ge_id])
        relevant = {c: r for c, r in cam_ranges.items() if c in queried_cameras}
        if len(relevant) < 2:
            continue
        sorted_ranges = sorted(relevant.values(), key=lambda r: r[0])
        large_gap = False
        for i in range(len(sorted_ranges) - 1):
            a_end = sorted_ranges[i][1]
            b_start = sorted_ranges[i + 1][0]
            if b_start > a_end:  # sequential: gap between end of one and start of next
                gap = b_start - a_end
                if gap > _MAX_CROSS_CAMERA_TRANSIT_SEC:
                    large_gap = True
                    break
        if large_gap:
            all_gap_failures.append(ge_id)

    if len(all_gap_failures) == len(covering):
        # Every covering entity has a suspiciously large time gap
        return {
            "decision": "mismatch",
            "confidence": 0.85,
            "reason": (
                f"entity appearances in queried cameras are separated by more than "
                f"{_MAX_CROSS_CAMERA_TRANSIT_SEC:.0f}s — too large for same-slot transit"
            ),
            "mode": "pre_filter_time_gap",
            "best_global_entity_id": all_gap_failures[0],
        }

    return None  # both checks passed → proceed to LLM


def _multi_camera_llm_verdict(
    llm: Any,
    query: str,
    ge_groups: dict[str, list[dict[str, Any]]],
    config: RunnableConfig,
) -> dict[str, Any] | None:
    """LLM-based multi-camera verdict: does the cross-camera evidence support the query?"""
    if llm is None or _llm_disabled() or not ge_groups:
        return None

    # Hard pre-filters before calling LLM
    prefilter = _camera_coverage_prefilter(query, ge_groups)
    if prefilter is not None:
        return prefilter

    # Build a compact summary of the multi-camera evidence
    entity_summaries: list[str] = []
    for ge_id, events in ge_groups.items():
        camera_set = sorted(set(str(e.get("camera_id") or e.get("video_id") or "").strip() for e in events))
        if len(camera_set) < 2:
            continue  # single-camera GE is not "multi-camera"
        time_ranges: list[str] = []
        for e in events[:10]:  # limit per GE
            cam = str(e.get("camera_id") or e.get("video_id") or "").strip()
            st = e.get("start_time")
            et = e.get("end_time")
            summary = (e.get("event_summary_en") or e.get("event_text_en") or "")[:120]
            color = str(e.get("object_color") or e.get("object_color_en") or "").strip()
            appearance = str(e.get("appearance_notes") or "").strip()[:80]
            appearance_str = ""
            if color:
                appearance_str += f" color={color}"
            if appearance:
                appearance_str += f" appearance={appearance}"
            time_ranges.append(f"  {cam}: {st}-{et}{appearance_str}  {summary}")
        entity_summaries.append(
            f"Entity {ge_id} appears in cameras: {', '.join(camera_set)}\n"
            + "\n".join(time_ranges)
        )

    if not entity_summaries:
        return None  # no multi-camera groups with >=2 cameras

    prompt_lines = [
        "You are a multi-camera event verifier. The user is asking about whether a person/object "
        "appeared across multiple surveillance cameras.",
        "",
        f"User query: {query}",
        "",
        "Below is evidence from a cross-camera Global Entity retrieval. Each entity group shows "
        "which cameras captured the same tracked person/object, with time ranges.",
        "",
        "Your task: decide whether the multi-camera evidence SUPPORTS the user's query.",
        "",
        "Decision guide for multi-camera:",
        "- exact: Evidence from TWO OR MORE cameras clearly shows the queried person/object "
        "with matching characteristics (color, type, action). The cameras and times are "
        "consistent with the query intent. Synonym matching is acceptable. If the global "
        "entity appears in the right set of cameras (matching the queried camera IDs), that "
        "alone strongly supports 'exact'.",
        "- partial: Entity appears in at least two of the queried cameras but some details "
        "(color, exact action, time precision) are missing from the event summary. This is "
        "still a positive verdict — the entity IS there across cameras.",
        "- mismatch: The evidence does NOT contain any entity that appears in the queried "
        "cameras, or only one camera appears (not truly cross-camera).",
        "",
        "CRITICAL RULES:",
        "1. If a global entity appears in the same set of cameras the user asks about AND "
        "the query does NOT specify appearance attributes, the answer MUST be 'exact' or "
        "'partial'. Missing appearance descriptions do not invalidate the camera match.",
        "2. APPEARANCE CHECK: If the query specifies a SPECIFIC appearance (e.g., 'dark "
        "hoodie', 'white shirt', 'beige jacket', 'black coat with fur'), you MUST verify "
        "that the retrieved entity's appearance data (color, clothing, summary) actually "
        "matches. If the entity's appearance clearly differs from the query (e.g., entity "
        "has 'grey hoodie' but query asks for 'dark hoodie'), return 'mismatch' even when "
        "the camera IDs match. Camera match is necessary but NOT sufficient when a "
        "specific appearance is part of the query.",
        "3. Cross-camera tracking proves an entity moved between cameras, but the entity "
        "must still match the described appearance. Use both camera coverage AND appearance "
        "consistency to decide.",
        "4. Only return 'mismatch' when NO entity appears in the queried cameras, or "
        "the evidence (camera or appearance) clearly contradicts the query.",
        "5. TEMPORAL GAP: For cross-camera movement, check the entity's per-camera time "
        "ranges. If the entity's appearances in camera A and camera B are sequential "
        "(non-overlapping) AND the gap between them exceeds 240 seconds, this is too "
        "large for a single 5-minute recording slot — return 'mismatch'. Overlapping "
        "time ranges (same person visible in both cameras simultaneously) are acceptable.",
        "",
        "EVIDENCE:",
    ]
    prompt_lines.extend(entity_summaries)
    # Fetch per-camera appearance from SQLite (bypasses retrieval ranking)
    all_ge_cameras = {
        str(e.get("camera_id") or e.get("video_id") or "").strip()
        for events in ge_groups.values()
        for e in events
    } - {""}
    cam_appearance = _fetch_camera_appearances_from_sqlite(list(all_ge_cameras))
    if cam_appearance:
        prompt_lines.append("")
        prompt_lines.append(
            "DETECTED APPEARANCES IN CAMERAS (from event/track records — use this "
            "to verify whether the queried appearance actually exists in each camera):"
        )
        for cam, snippets in sorted(cam_appearance.items()):
            prompt_lines.append(f"  {cam}: " + " ; ".join(snippets[:4]))

    prompt_lines.extend(
        [
            "",
            "Return JSON only:",
            "{",
            '  "decision": "exact" | "partial" | "mismatch",',
            '  "confidence": <float 0.0-1.0>,',
            '  "reason": "<one-line evidence-based, mention which cameras and why>",',
            '  "best_global_entity_id": "<the entity id you picked>"',
            "}",
        ]
    )
    prompt = "\n".join(prompt_lines)

    try:
        model = llm.bind(max_tokens=250) if hasattr(llm, "bind") else llm
        raw = model.invoke(
            [
                SystemMessage(content="You are a multi-camera video event verifier. Return JSON only."),
                HumanMessage(content=prompt),
            ],
            config=config,
        )
        content = raw.content if hasattr(raw, "content") else str(raw)
        data = _parse_json_object(str(content))
        if not isinstance(data, dict):
            return None
        decision = str(data.get("decision") or "").strip().lower()
        if decision not in _FAST_DECISIONS:
            return None
        try:
            confidence = float(data.get("confidence", 0.0))
        except Exception:
            confidence = 0.0
        return {
            "decision": decision,
            "confidence": max(0.0, min(1.0, confidence)),
            "reason": str(data.get("reason") or "").strip()[:200],
            "mode": "llm_multi_camera",
            "best_global_entity_id": str(data.get("best_global_entity_id") or "").strip(),
        }
    except Exception:
        return None


def _multi_camera_heuristic_verdict(
    query: str,
    ge_groups: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Heuristic multi-camera verdict based on camera count and token coverage."""
    query_terms = _query_terms(query)
    best_ge_id = ""
    best_camera_count = 0
    best_coverage = 0.0
    best_matched: list[str] = []

    for ge_id, events in ge_groups.items():
        camera_set = set(str(e.get("camera_id") or e.get("video_id") or "").strip() for e in events)
        n_cameras = len(camera_set)
        evidence = " ".join(
            (e.get("event_summary_en") or e.get("event_text_en") or "") for e in events
        ).lower()
        matched = sorted(term for term in query_terms if term in evidence)
        coverage = len(matched) / max(len(query_terms), 1)
        # Prefer more cameras, then better coverage
        if n_cameras > best_camera_count or (n_cameras == best_camera_count and coverage > best_coverage):
            best_ge_id = ge_id
            best_camera_count = n_cameras
            best_coverage = coverage
            best_matched = matched

    if best_camera_count >= 2 and best_coverage >= 0.34:
        decision = "exact" if best_coverage >= 0.58 else "partial"
        confidence = min(0.95, 0.55 + best_coverage / 2)
    elif best_camera_count >= 2:
        # At least 2 cameras matched — give partial (not mismatch)
        decision = "partial"
        confidence = 0.55
    else:
        decision = "mismatch"
        confidence = max(0.55, 1.0 - best_coverage)

    return {
        "decision": decision,
        "confidence": confidence,
        "reason": f"multi_camera heuristic: cameras={best_camera_count} coverage={best_coverage:.2f} matched={best_matched[:6]}",
        "mode": "heuristic_multi_camera",
        "best_global_entity_id": best_ge_id,
    }


def _build_multi_camera_descriptor(
    ge_groups: dict[str, list[dict[str, Any]]],
    best_ge_id: str,
) -> dict[str, Any]:
    """Build the multi-camera descriptor (cameras, times, summaries) for downstream nodes."""
    events = ge_groups.get(best_ge_id, [])
    cameras: list[dict[str, Any]] = []
    camera_ids: list[str] = []
    seen_cameras: set[str] = set()
    for e in events:
        cam = str(e.get("camera_id") or e.get("video_id") or "").strip()
        if cam in seen_cameras:
            continue
        seen_cameras.add(cam)
        camera_ids.append(cam)
        cameras.append({
            "camera_id": cam,
            "start_time": e.get("start_time"),
            "end_time": e.get("end_time"),
            "summary": (e.get("event_summary_en") or e.get("event_text_en") or "")[:200],
        })
    return {
        "multi_camera": True,
        "global_entity_id": best_ge_id,
        "cameras": cameras,
        "camera_ids": camera_ids,
        "camera_count": len(camera_ids),
        "span_source": "multi_camera_ge",
    }


# ---------------------------------------------------------------------------
# Verdict shapes
# ---------------------------------------------------------------------------


def _skipped_verdict(reason: str) -> dict[str, Any]:
    return {
        "decision": "skipped",
        "confidence": 0.0,
        "reason": reason,
        "mode": "skipped",
    }


def _no_rows_verdict() -> dict[str, Any]:
    return {
        "decision": "mismatch",
        "confidence": 0.98,
        "reason": "no_rows",
        "mode": "rule",
    }


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


def create_match_verifier_node(llm: Any = None):
    def match_verifier_node(state: AgentState, config: RunnableConfig, store: BaseStore) -> dict[str, Any]:
        del store
        answer_type = str(state.get("answer_type") or "").strip().lower()
        # P2-3: Run verifier for "existence" AND "unknown" (previously skipped
        # "unknown", causing 8/30 false-positive "yes" answers for queries
        # whose answer should have been "no"). When answer_type is unknown the
        # classifier could not determine the question shape, so it is safer to
        # verify existence than to skip. Only skip for non-existence types
        # where verification would be meaningless (list, count, description).
        _SKIP_ANSWER_TYPES = {"list", "count", "description"}
        query = str(state.get("original_user_query") or state.get("user_query") or "").strip()
        if answer_type in _SKIP_ANSWER_TYPES:
            return {
                "verifier_result": _skipped_verdict(f"answer_type={answer_type}"),
                "current_node": "match_verifier_node",
            }

        # ── v2.4: Multi-camera path ──
        is_multi_camera = _detect_multi_camera_context(state)
        if is_multi_camera:
            ge_rows = _collect_ge_rows(state)
            ge_groups = _group_by_global_entity(ge_rows)
            if ge_groups:
                # Check if any group has >=2 cameras (true multi-camera)
                has_multi = any(
                    len(set(
                        str(e.get("camera_id") or e.get("video_id") or "").strip()
                        for e in events
                    )) >= 2
                    for events in ge_groups.values()
                )
                if has_multi:
                    mc_llm = _multi_camera_llm_verdict(llm, query, ge_groups, config)
                    if mc_llm is not None:
                        verdict = mc_llm
                    else:
                        verdict = _multi_camera_heuristic_verdict(query, ge_groups)
                    best_ge_id = verdict.get("best_global_entity_id", "")
                    mc_descriptor = _build_multi_camera_descriptor(ge_groups, best_ge_id)
                    return {
                        "verifier_result": {
                            **verdict,
                            **mc_descriptor,
                            "best_chunk_index": 0,
                            "candidate_count": sum(len(v) for v in ge_groups.values()),
                        },
                        "current_node": "match_verifier_node",
                    }

        # ── Existing single-camera paths ──
        rows = _select_rows(state)
        if not rows:
            return {
                "verifier_result": _no_rows_verdict(),
                "current_node": "match_verifier_node",
            }

        # Legacy single-row path (escape hatch)
        if not _reselect_span_enabled():
            top_row = rows[0]
            primary = _pick_primary_row(top_row)
            verdict = _llm_verdict_single(llm, query, top_row, primary, config) or _heuristic_verdict_single(
                query, top_row, primary
            )
            verdict = _cross_camera_verdict(query, top_row) or _strict_attribute_verdict(query, top_row) or verdict
            descriptor = _row_descriptor(top_row)
            return {
                "verifier_result": {
                    **verdict,
                    **descriptor,
                    "span_source": "candidate_top_row",
                    "best_chunk_index": 0,
                    "candidate_count": 1,
                },
                "current_node": "match_verifier_node",
            }

        # v2.3 multi-candidate path
        candidates = _collect_candidates(
            rows,
            candidate_limit=_candidate_limit(),
            cross_video_top_n=_cross_video_top_n(),
        )
        llm_pick = _llm_select_best_chunk(llm, query, candidates, config)
        if llm_pick is not None:
            best_idx, verdict = llm_pick
        else:
            best_idx, verdict = _heuristic_select_best_chunk(query, candidates)

        # Guard against pathological out-of-range; should never trigger because
        # both helpers clamp inside [0, len(candidates)-1].
        if best_idx < 0 or best_idx >= len(candidates):
            best_idx = 0

        best_idx = _prefer_time_matching_candidate(query, candidates, best_idx)
        best_idx = _prefer_cross_camera_candidate(query, candidates, best_idx)
        chosen_row = candidates[best_idx]
        verdict = _cross_camera_verdict(query, chosen_row, candidates) or _strict_attribute_verdict(query, chosen_row) or verdict

        # Cross-camera collective upgrade: if the LLM/heuristic said "mismatch" but
        # the candidate pool has the same tracked entity in ALL queried cameras, the
        # cross-camera appearance IS confirmed — upgrade to "partial".
        if verdict.get("decision") == "mismatch" and _candidates_cover_cameras(query, candidates):
            verdict = {
                "decision": "partial",
                "confidence": 0.70,
                "reason": "cross_camera_entity_covers_all_queried_cameras_collectively",
                "mode": "cross_camera_collective_coverage",
            }
        descriptor = _row_descriptor(chosen_row)
        # span_source is "rerank_reselected" only when the LLM/heuristic chose
        # something other than the rerank top-1; otherwise the rerank top-1
        # already won, so downstream behaviour is unchanged.
        span_source = "candidate_top_row" if best_idx == 0 else "rerank_reselected"

        return {
            "verifier_result": {
                **verdict,
                **descriptor,
                "span_source": span_source,
                "best_chunk_index": best_idx,
                "candidate_count": len(candidates),
            },
            "current_node": "match_verifier_node",
        }

    return match_verifier_node
