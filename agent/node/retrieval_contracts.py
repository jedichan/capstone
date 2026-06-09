import re
import os
from typing import Any

from db.config import get_graph_chroma_parent_collection, get_graph_chroma_path
from tools.db_access import ChromaGateway


DEFAULT_SEARCH_CONFIG = {
    "candidate_limit": 80,
    "top_k_per_event": 20,
    "rerank_top_k": 5,
    "rerank_candidate_limit": 20,
    "distance_threshold": None,
    "hybrid_alpha": 0.7,
    "hybrid_fallback_alpha": 0.9,
    "hybrid_limit": 50,
    "sql_limit": 80,
}


def parent_projection_enabled() -> bool:
    # Default is OFF: child rows flow straight to answer/summary so the temporal
    # range reported to the user stays per-track instead of being collapsed to
    # the full video span.
    # Re-enable with AGENT_ENABLE_PARENT_PROJECTION=1.
    # Legacy AGENT_DISABLE_PARENT_PROJECTION is kept for backward compatibility
    # but only takes effect when it explicitly disables the feature.
    disable_raw = os.getenv("AGENT_DISABLE_PARENT_PROJECTION", "").strip().lower()
    if disable_raw in {"1", "true", "yes", "on"}:
        return False
    enable_raw = os.getenv("AGENT_ENABLE_PARENT_PROJECTION", "").strip().lower()
    return enable_raw in {"1", "true", "yes", "on"}


def build_search_config(existing: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = dict(DEFAULT_SEARCH_CONFIG)
    env_overrides = {
        "candidate_limit": os.getenv("AGENT_CANDIDATE_LIMIT"),
        "top_k_per_event": os.getenv("AGENT_TOP_K_PER_EVENT"),
        "rerank_top_k": os.getenv("AGENT_RERANK_TOP_K"),
        "rerank_candidate_limit": os.getenv("AGENT_RERANK_CANDIDATE_LIMIT"),
        "hybrid_limit": os.getenv("AGENT_HYBRID_LIMIT"),
        "sql_limit": os.getenv("AGENT_SQL_LIMIT"),
    }
    for key, raw in env_overrides.items():
        if raw is None or not str(raw).strip():
            continue
        try:
            cfg[key] = int(raw)
        except Exception:
            pass
    if isinstance(existing, dict):
        cfg.update({k: v for k, v in existing.items() if v is not None})
    return cfg


def _word_match(query: str, token: str) -> bool:
    """Match *token* as a whole word within *query*, avoiding substring false-positives
    (e.g. ``"scar"`` matching ``"car"``, ``"personal"`` matching ``"person"``)."""
    return bool(re.search(rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])", query))


def extract_structured_filters(user_query: str) -> dict[str, str]:
    """Extract structured (enum) filters from the user query.

    Only sets a filter when *exactly one* entity/color/zone is detected.
    Multi-entity queries (e.g. ``"black car and white truck"``) produce no
    hard filter — the downstream text search handles them.

    Uses word-boundary matching to avoid substring false-positives
    (``"scar"`` matching ``"car"``, ``"barking"`` matching ``"parking"``).
    """
    q = (user_query or "").lower()
    out: dict[str, str] = {}

    object_matches = [t for t in ["person", "car", "truck", "bus", "bike", "motorcycle"]
                      if _word_match(q, t)]
    if len(object_matches) == 1:
        out["object_type"] = object_matches[0]

    color_matches = [t for t in ["dark", "black", "white", "red", "blue", "unknown"]
                     if _word_match(q, t)]
    if len(color_matches) == 1:
        out["object_color_en"] = color_matches[0]

    zone_aliases = [
        ("left bleachers", "left bleachers"),
        ("bleachers", "bleachers"),
        ("parking area", "parking"),
        ("parking", "parking"),
        ("sidewalk", "sidewalk"),
        ("right side", "road_right"),
        ("right-center", "road_right"),
        ("sideline", "road_right"),
        ("road right", "road_right"),
        ("court center-right", "center-right"),
        ("center-right", "center-right"),
        ("center court", "center"),
        ("court", "court"),
        ("baseline", "baseline"),
        ("center", "center"),
    ]
    for phrase, normalized in zone_aliases:
        if _word_match(q, phrase):
            out["scene_zone_en"] = normalized
            break
    return out


def infer_sql_plan(user_query: str, search_config: dict[str, Any]) -> dict[str, Any]:
    filters = extract_structured_filters(user_query)
    where = [{"field": k, "op": "contains" if k == "scene_zone_en" else "=", "value": v} for k, v in filters.items()]
    return {
        "table": "episodic_events",
        "fields": [
            "event_id",
            "video_id",
            "track_id",
            "start_time",
            "end_time",
            "object_type",
            "object_color_en",
            "scene_zone_en",
            "appearance_notes_en",
            "event_summary_en",
        ],
        "where": where,
        "order_by": "start_time ASC",
        "limit": int(search_config.get("sql_limit", 80)),
    }


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _normalize_event_id(value: Any) -> Any:
    """P0-1: Coerce numeric event_ids to int for cross-branch RRF matching."""
    if value is None:
        return value
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return value


def normalize_sql_rows(rows: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows or []:
        normalized.append(
            {
                **row,
                "event_id": _normalize_event_id(row.get("event_id")),
                "video_id": row.get("video_id"),
                "track_id": row.get("track_id"),
                "start_time": row.get("start_time"),
                "end_time": row.get("end_time"),
                "object_type": row.get("object_type"),
                "object_color_en": row.get("object_color_en"),
                "scene_zone_en": row.get("scene_zone_en"),
                "event_summary_en": row.get("event_summary_en") or row.get("event_text_en") or row.get("event_text_cn"),
                "event_text_en": row.get("event_text_en") or row.get("event_summary_en"),
                "_distance": row.get("_distance", 0.0),
                "_source_type": row.get("_source_type", "sql"),
            }
        )
    return normalized


def normalize_hybrid_rows(rows: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows or []:
        normalized.append(
            {
                **row,
                "event_id": _normalize_event_id(row.get("event_id")),
                "video_id": row.get("video_id"),
                "track_id": row.get("track_id"),
                "start_time": row.get("start_time"),
                "end_time": row.get("end_time"),
                "object_type": row.get("object_type"),
                "object_color_en": row.get("object_color_en"),
                "scene_zone_en": row.get("scene_zone_en"),
                "event_summary_en": row.get("event_summary_en") or row.get("event_text") or row.get("event_text_en"),
                "event_text_en": row.get("event_text_en") or row.get("event_text") or row.get("event_summary_en"),
                "_distance": _safe_float(row.get("_distance", row.get("distance"))),
                "_hybrid_score": _safe_float(row.get("_hybrid_score", row.get("hybrid_score"))),
                "_bm25": _safe_float(row.get("_bm25")),
                "_source_type": row.get("_source_type", "hybrid"),
            }
        )
    return normalized


def _dedupe_texts(values: list[Any]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _safe_min_time(rows: list[dict[str, Any]]) -> float | None:
    nums = [float(row["start_time"]) for row in rows if isinstance(row.get("start_time"), (int, float))]
    return min(nums) if nums else None


def _safe_max_time(rows: list[dict[str, Any]]) -> float | None:
    nums = [float(row["end_time"]) for row in rows if isinstance(row.get("end_time"), (int, float))]
    return max(nums) if nums else None


def _aggregate_parent_fallback(video_id: str, child_rows: list[dict[str, Any]]) -> dict[str, Any]:
    start_time = _safe_min_time(child_rows)
    end_time = _safe_max_time(child_rows)
    track_ids = _dedupe_texts([row.get("track_id") for row in child_rows])
    child_event_ids = _dedupe_texts([row.get("event_id") for row in child_rows])
    summaries = _dedupe_texts([row.get("event_summary_en") or row.get("event_text_en") for row in child_rows])
    object_types = _dedupe_texts([row.get("object_type") for row in child_rows])
    object_colors = _dedupe_texts([row.get("object_color_en") for row in child_rows])
    scene_zones = _dedupe_texts([row.get("scene_zone_en") for row in child_rows])
    best_distance = min([float(row["_distance"]) for row in child_rows if isinstance(row.get("_distance"), (int, float))], default=None)
    best_hybrid = max([float(row["_hybrid_score"]) for row in child_rows if isinstance(row.get("_hybrid_score"), (int, float))], default=None)
    summary_parts = [
        f"Video {video_id}.",
        f"This parent view summarizes {len(child_rows)} child results.",
    ]
    if object_types:
        summary_parts.append("Object types: " + ", ".join(object_types) + ".")
    if object_colors:
        summary_parts.append("Object colors: " + ", ".join(object_colors) + ".")
    if scene_zones:
        summary_parts.append("Scene zones: " + ", ".join(scene_zones) + ".")
    if summaries:
        summary_parts.append("Child summaries: " + " ".join(summaries[:5]))
    return {
        "event_id": video_id,
        "record_id": video_id,
        "video_id": video_id,
        "track_id": ",".join(track_ids[:8]) if track_ids else None,
        "start_time": start_time,
        "end_time": end_time,
        "event_summary_en": " ".join(summary_parts),
        "event_text_en": " ".join(summary_parts),
        "_distance": best_distance,
        "_hybrid_score": best_hybrid,
        "_source_type": "parent_projection",
        "_record_level": "parent",
        "_parent_hit_count": len(child_rows),
        "_child_event_ids": child_event_ids,
        "_child_rows": child_rows,
    }


def summarize_parent_context(rows: list[dict[str, Any]] | None, limit: int = 5) -> list[dict[str, Any]]:
    """Lightweight parent summary that never hits the Chroma parent collection.

    Used when parent projection is disabled so downstream debug / UI still has a
    video-level overview without paying the network round trip to fetch parent
    records. Output is intentionally NOT written into ``rerank_result``; it is a
    debug artifact only.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows or []:
        video_id = str(row.get("video_id") or "").strip()
        if not video_id:
            continue
        grouped.setdefault(video_id, []).append(row)
    if not grouped:
        return []

    ordered_videos = sorted(
        grouped.keys(),
        key=lambda video_id: min(
            [
                float(item.get("_distance"))
                for item in grouped[video_id]
                if isinstance(item.get("_distance"), (int, float))
            ]
            or [999999.0]
        ),
    )

    out: list[dict[str, Any]] = []
    for rank, video_id in enumerate(ordered_videos[:limit], start=1):
        child_rows = grouped[video_id]
        out.append(
            {
                "video_id": video_id,
                "child_hit_count": len(child_rows),
                "start_time": _safe_min_time(child_rows),
                "end_time": _safe_max_time(child_rows),
                "best_distance": min(
                    [
                        float(row["_distance"])
                        for row in child_rows
                        if isinstance(row.get("_distance"), (int, float))
                    ],
                    default=None,
                ),
                "best_hybrid_score": max(
                    [
                        float(row["_hybrid_score"])
                        for row in child_rows
                        if isinstance(row.get("_hybrid_score"), (int, float))
                    ],
                    default=None,
                ),
                "track_ids": _dedupe_texts([row.get("track_id") for row in child_rows])[:8],
                "object_types": _dedupe_texts([row.get("object_type") for row in child_rows]),
                "object_colors": _dedupe_texts([row.get("object_color_en") for row in child_rows]),
                "scene_zones": _dedupe_texts([row.get("scene_zone_en") for row in child_rows]),
                "parent_rank": rank,
            }
        )
    return out


def project_rows_to_parent_context(rows: list[dict[str, Any]] | None, limit: int = 5) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows or []:
        video_id = str(row.get("video_id") or "").strip()
        if not video_id:
            continue
        grouped.setdefault(video_id, []).append(row)
    if not grouped:
        return []

    ordered_videos = sorted(
        grouped.keys(),
        key=lambda video_id: min(
            [
                float(item.get("_distance"))
                for item in grouped[video_id]
                if isinstance(item.get("_distance"), (int, float))
            ]
            or [999999.0]
        ),
    )

    parent_lookup: dict[str, dict[str, Any]] = {}
    try:
        gateway = ChromaGateway(
            db_path=get_graph_chroma_path(),
            collection_name=get_graph_chroma_parent_collection(),
        )
        parent_records = gateway.get_records_by_ids(ordered_videos)
        for item in parent_records:
            parent_lookup[str(item.get("record_id"))] = item
    except Exception:
        parent_lookup = {}

    out: list[dict[str, Any]] = []
    for rank, video_id in enumerate(ordered_videos[:limit], start=1):
        child_rows = grouped[video_id]
        parent_record = parent_lookup.get(video_id)
        if parent_record:
            metadata = parent_record.get("metadata") if isinstance(parent_record.get("metadata"), dict) else {}
            row = _aggregate_parent_fallback(video_id, child_rows)
            parent_doc = str(parent_record.get("document") or "").strip()
            if parent_doc:
                row["event_summary_en"] = parent_doc
                row["event_text_en"] = parent_doc
            row["start_time"] = metadata.get("start_time", row.get("start_time"))
            row["end_time"] = metadata.get("end_time", row.get("end_time"))
            row["_parent_collection_hit"] = True
            row["_parent_child_count"] = metadata.get("child_count")
            row["_parent_scene_zones"] = metadata.get("scene_zones")
            row["_parent_object_types"] = metadata.get("object_types")
            row["_parent_object_colors"] = metadata.get("object_colors")
        else:
            row = _aggregate_parent_fallback(video_id, child_rows)
            row["_parent_collection_hit"] = False
        row["_parent_rank"] = rank
        out.append(row)
    return out


def build_routing_metrics(
    *,
    execution_mode: str,
    label: str,
    query: str,
    sql_rows_count: int,
    hybrid_rows_count: int,
    sql_error: str | None = None,
    hybrid_error: str | None = None,
) -> dict[str, Any]:
    return {
        "execution_mode": execution_mode,
        "label": label,
        "query": query,
        "sql_rows_count": sql_rows_count,
        "hybrid_rows_count": hybrid_rows_count,
        "sql_error": sql_error,
        "hybrid_error": hybrid_error,
    }


# P0-4: Stopwords scoped to true function words and query-chrome only.
# Content words like ``person``, ``car``, ``moving``, ``area`` were previously
# blacklisted here, which dropped the strongest signal tokens and also
# duplicated the work already done by ``filter_terms`` dedup below. They are
# NOT included here anymore.
#
# Exported as ``SQL_TOKEN_STOPWORDS`` so ``sql_debug_utils`` can reuse the same
# set, avoiding two copies drifting apart.
SQL_TOKEN_STOPWORDS: frozenset[str] = frozenset(
    {
        # determiners / articles
        "the", "this", "that", "these", "those",
        "some", "any", "all", "every", "each", "both", "either", "neither",
        # pronouns
        "you", "him", "her", "them", "they", "their", "its", "our", "your",
        "who", "whom", "what", "which",
        "where", "when", "why", "how",
        # auxiliary / modal verbs (len >= 3 only; shorter ones fall out via length guard)
        "are", "was", "were", "been", "being",
        "does", "did", "done",
        "has", "had", "having",
        "can", "could", "may", "might", "must", "shall", "will",
        "should", "would",
        # conjunctions / prepositions (len >= 3)
        "and", "but", "nor", "not", "yet",
        "for", "into", "onto", "out", "off", "over", "under", "above", "below",
        "about", "between", "among", "across", "through", "around", "against",
        "before", "after",
        "with", "without", "from", "than", "then",
        # query-chrome / instruction verbs
        "show", "tell", "find", "look", "see", "get", "give", "please",
        "want", "need",
        # generic domain nouns that never help as text filters
        "database", "video", "videos", "clip", "clips", "footage",
        "scene", "scenes", "moment", "moments", "record", "records",
        "there", "here",
        # hedge / meta
        "maybe", "possibly", "just", "really",
        # additional stopwords used by sql_debug_utils (merged from _expanded_query_terms)
        "a", "an", "is", "do", "me", "of", "to", "on", "in", "at",
        "while", "someone", "repeatedly", "your", "did",
    }
)

# Small, explicit plural -> singular map. We intentionally avoid heuristic
# stemming because it causes false merges (news -> new, lens -> len, etc.).
_PLURAL_TO_SINGULAR: dict[str, str] = {
    "persons": "person",
    "people": "person",
    "men": "man",
    "women": "woman",
    "children": "child",
    "kids": "kid",
    "boys": "boy",
    "girls": "girl",
    "babies": "baby",
    "officers": "officer",
    "civilians": "civilian",
    "police": "police",
    "guards": "guard",
    "caregivers": "caregiver",
    "animals": "animal",
    "dogs": "dog",
    "puppies": "puppy",
    "cats": "cat",
    "cars": "car",
    "trucks": "truck",
    "buses": "bus",
    "vehicles": "vehicle",
    "bikes": "bike",
    "bicycles": "bicycle",
    "motorcycles": "motorcycle",
    "events": "event",
    "tracks": "track",
    "objects": "object",
    "items": "item",
    "shoes": "shoe",
    "zones": "zone",
    "rooms": "room",
    "stairs": "stair",
    "supermarkets": "supermarket",
    "stores": "store",
    "scenes": "scene",
}


def _singularize_token(token: str) -> str:
    return _PLURAL_TO_SINGULAR.get(token, token)


_APPEARANCE_QUERY_EXPANSIONS: list[tuple[re.Pattern[str], list[str], set[str]]] = [
    (
        re.compile(r"\blight\s+gr(?:e|a)y\s+hoodie\b"),
        ["light grey hoodie", "light_grey_hoodie", "hoodie"],
        {"light", "grey", "gray", "hoodie"},
    ),
    (
        re.compile(r"\bbeige\s+(?:jacket|coat)\b"),
        ["beige jacket", "beige_jacket", "jacket", "coat"],
        {"beige", "jacket", "coat"},
    ),
    (
        re.compile(r"\bfur(?:\s*|-)?trimmed\s+hood\b"),
        ["fur trimmed hood", "fur_trimmed_hood", "hood"],
        {"fur", "trimmed", "hood"},
    ),
    (
        re.compile(r"\bhood\s+up\b"),
        ["hood up", "hood_up", "hoodie"],
        {"hood", "up"},
    ),
    (
        re.compile(r"\bhooded\s+jacket\b"),
        ["hooded jacket", "hood_up", "hoodie", "jacket"],
        {"hooded", "jacket"},
    ),
    (
        re.compile(r"\bdark\s+hoodie\b"),
        ["dark hoodie", "hood_up", "hoodie"],
        {"dark", "hoodie"},
    ),
    (
        re.compile(r"\bdark\s+long\s+coat\b"),
        ["dark long coat", "dark coat", "long coat", "dark_coat", "coat"],
        {"dark", "long", "coat"},
    ),
    (
        re.compile(r"\bdark\s+coat\b"),
        ["dark coat", "long coat", "dark_coat", "coat"],
        {"dark", "coat"},
    ),
    (
        re.compile(r"\blong\s+coat\b"),
        ["long coat", "dark coat", "dark_coat", "coat"],
        {"long", "coat"},
    ),
]


def _expand_appearance_query_terms(user_query: str) -> tuple[list[str], set[str]]:
    q = str(user_query or "").lower()
    expansions: list[str] = []
    suppressed: set[str] = set()
    seen: set[str] = set()
    for pattern, extra_terms, raw_terms in _APPEARANCE_QUERY_EXPANSIONS:
        if not pattern.search(q):
            continue
        suppressed.update(raw_terms)
        for term in extra_terms:
            clean = str(term).strip().lower()
            if clean and clean not in seen:
                expansions.append(clean)
                seen.add(clean)
    return expansions, suppressed


def extract_text_tokens_for_sql(user_query: str, filters: dict[str, str]) -> list[str]:
    """Extract free-text tokens from the query for FTS5 / LIKE search.

    Tokens already captured as structured filters are excluded (they are
    applied via exact WHERE clauses instead).

    The maximum number of tokens returned is controlled by
    ``AGENT_SQL_TOKEN_LIMIT`` (default: 10, raised from the previous
    hard-coded 6 to preserve more signal for longer queries).
    Set ``AGENT_SQL_TOKEN_LIMIT=0`` to return all tokens (unbounded).
    """
    limit = int(os.getenv("AGENT_SQL_TOKEN_LIMIT", "10") or "10")
    filter_terms: set[str] = set()
    for value in filters.values():
        filter_terms.update(t for t in re.findall(r"[a-z0-9_]+", str(value).lower()) if t)

    phrase_expansions, suppressed_terms = _expand_appearance_query_terms(user_query)
    out: list[str] = []
    seen: set[str] = set()
    for term in phrase_expansions:
        if term in seen:
            continue
        seen.add(term)
        out.append(term)
    for raw in re.findall(r"[a-z0-9_]+", user_query.lower()):
        if len(raw) <= 2:
            continue
        token = _singularize_token(raw)
        if token in SQL_TOKEN_STOPWORDS:
            continue
        if token in filter_terms:
            continue
        if token in suppressed_terms:
            continue
        if token in seen:
            continue
        seen.add(token)
        out.append(token)
    if limit > 0:
        return out[:limit]
    return out
