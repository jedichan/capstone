import json
import os
import re
import sqlite3
from pathlib import Path
from typing import Any

from node.retrieval_contracts import SQL_TOKEN_STOPWORDS


SQL_KEYWORDS = {
    "select",
    "from",
    "where",
    "and",
    "or",
    "order",
    "by",
    "group",
    "having",
    "limit",
    "asc",
    "desc",
    "like",
    "in",
    "is",
    "null",
    "not",
    "as",
    "case",
    "when",
    "then",
    "else",
    "end",
    "on",
    "join",
    "left",
    "right",
    "inner",
    "outer",
    "distinct",
    "count",
    "sum",
    "avg",
    "min",
    "max",
    "coalesce",
    "lower",
}

ENUM_COLUMNS = ("object_type", "object_color_en", "scene_zone_en")
OBJECT_ALIASES = {
    "vehicle": "car",
    "sedan": "car",
    "automobile": "car",
    "puppy": "dog",
    "animal": "dog",
    "elderly": "person",
    "caregiver": "person",
    "woman": "person",
    "man": "person",
    "staff": "person",
    "officer": "person",
    "police": "person",
}
COLOR_ALIASES = {
    "blonde": "white",
    "white-haired": "white",
    "gray-haired": "gray",
}
ZONE_ALIASES = {
    "street": "road",
    "yard": "outdoor",
}


def log_sql_debug(stage: str, **payload: Any) -> None:
    raw = os.getenv("AGENT_SQL_DEBUG", "0").strip().lower()
    if raw not in {"1", "true", "yes", "on", "summary"}:
        return
    if raw == "summary":
        payload = {k: v for k, v in payload.items() if k in {"user_query", "row_count", "guided_row_count", "fallback_row_count", "legacy_row_count", "error", "table_name", "db_path"}}
    safe_payload = {"stage": stage, **payload}
    try:
        print("[SQL_DEBUG] " + json.dumps(safe_payload, ensure_ascii=False, sort_keys=True))
    except Exception:
        print(f"[SQL_DEBUG] stage={stage} payload={payload}")


def get_sqlite_table_columns(db_path: Path, table_name: str = "episodic_events") -> list[str]:
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return [str(row[1]) for row in rows]


def get_distinct_column_values(db_path: Path, column_name: str, table_name: str = "episodic_events", limit: int = 30) -> list[str]:
    sql = (
        f"SELECT DISTINCT {column_name} FROM {table_name} "
        f"WHERE {column_name} IS NOT NULL AND trim(CAST({column_name} AS TEXT)) != '' "
        f"ORDER BY {column_name} LIMIT ?"
    )
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(sql, (int(limit),)).fetchall()
    return [str(row[0]).strip() for row in rows if row and str(row[0]).strip()]


def extract_where_clause(sql_query: str) -> str:
    query = str(sql_query or "").strip()
    match = re.search(
        r"\bWHERE\b\s+(.*?)(?=\bORDER\s+BY\b|\bGROUP\s+BY\b|\bLIMIT\b|$)",
        query,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return ""
    return " ".join(match.group(1).split()).strip()


def find_unknown_sql_columns(
    sql_query: str,
    schema_columns: list[str],
    table_name: str = "episodic_events",
) -> list[str]:
    known = {item.lower() for item in schema_columns}
    known.update({table_name.lower(), "sqlite_master"})
    found: set[str] = set()

    select_match = re.search(r"\bSELECT\b(.*?)\bFROM\b", sql_query, flags=re.IGNORECASE | re.DOTALL)
    if select_match:
        select_body = select_match.group(1)
        for part in select_body.split(","):
            fragment = part.strip()
            if not fragment or fragment == "*":
                continue
            fragment = re.sub(r"\bAS\b\s+[A-Za-z_][A-Za-z0-9_]*", "", fragment, flags=re.IGNORECASE)
            base_match = re.search(r"([A-Za-z_][A-Za-z0-9_]*)$", fragment)
            if base_match:
                found.add(base_match.group(1).lower())

    for pattern in [
        r"([A-Za-z_][A-Za-z0-9_]*)\s*(?:=|!=|<>|>=|<=|>|<|\bLIKE\b|\bIN\b|\bIS\b)",
        r"\bORDER\s+BY\s+([A-Za-z_][A-Za-z0-9_]*)",
        r"\bGROUP\s+BY\s+([A-Za-z_][A-Za-z0-9_]*)",
    ]:
        for match in re.finditer(pattern, sql_query, flags=re.IGNORECASE):
            found.add(match.group(1).lower())

    unknown = sorted(
        item
        for item in found
        if item not in known and item not in SQL_KEYWORDS and not item.isdigit()
    )
    return unknown


def _simple_stems(token: str) -> list[str]:
    variants = [token]
    if token.endswith("ing") and len(token) > 5:
        stem = token[:-3]
        variants.append(stem)
        if len(stem) > 2 and stem[-1] == stem[-2]:
            variants.append(stem[:-1])
    if token.endswith("ed") and len(token) > 4:
        stem = token[:-2]
        variants.append(stem)
        if len(stem) > 2 and stem[-1] == stem[-2]:
            variants.append(stem[:-1])
    if token.endswith("ies") and len(token) > 4:
        variants.append(token[:-3] + "y")
    elif token.endswith("s") and len(token) > 4:
        variants.append(token[:-1])
    return [item for item in variants if len(item) >= 3]


def _expanded_query_terms(user_query: str) -> list[str]:
    normalized = re.sub(r"[^a-z0-9]+", " ", str(user_query or "").lower()).strip()
    tokens = [item for item in normalized.split() if item]
    synonym_map = {
        "car": ["vehicle", "vehicles", "automobile", "sedan"],
        "vehicle": ["car", "cars"],
        "dog": ["animal", "animals", "puppy", "puppies"],
        "animal": ["dog", "dogs", "puppy"],
        "person": ["people", "adult", "adults", "pedestrian", "pedestrians", "man", "woman", "elderly", "caregiver"],
        "persons": ["person", "people", "adult", "adults"],
        "people": ["person", "persons", "adult", "adults"],
        "dark": ["black", "gray", "unknown"],
        "black": ["dark"],
        "gray": ["grey", "dark"],
        "white": ["white-haired", "blonde"],
        "hitting": ["hit", "hits", "slap", "slaps"],
        "beating": ["beat", "hit", "slap"],
        "running": ["run", "runs", "drove", "driving"],
        "sit": ["sitting", "seated"],
        "sitting": ["sit", "seated"],
        "sofa": ["couch"],
        "head": ["face"],
        "elderly": ["person", "adult", "woman", "man", "white", "haired"],
        "road": ["street"],
    }
    terms: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if len(token) < 3 or token in SQL_TOKEN_STOPWORDS:
            continue
        candidates = [token, *_simple_stems(token), *synonym_map.get(token, [])]
        for candidate in candidates:
            clean = candidate.strip().lower()
            if not clean or clean in SQL_TOKEN_STOPWORDS or clean in seen or len(clean) < 3:
                continue
            seen.add(clean)
            terms.append(clean)
    return terms[:16]


def _query_phrases(user_query: str) -> list[str]:
    normalized = re.sub(r"[^a-z0-9]+", " ", str(user_query or "").lower()).strip()
    tokens = [item for item in normalized.split() if len(item) >= 3]
    filtered = [item for item in tokens if item not in SQL_TOKEN_STOPWORDS]
    phrases: list[str] = []
    seen: set[str] = set()
    for size in (2, 3):
        for idx in range(0, max(len(filtered) - size + 1, 0)):
            chunk = filtered[idx : idx + size]
            phrase = " ".join(chunk).strip()
            if len(phrase) < 7 or phrase in seen:
                continue
            if len([item for item in chunk if item not in SQL_TOKEN_STOPWORDS]) < size:
                continue
            seen.add(phrase)
            phrases.append(phrase)
    return phrases[:8]


def _normalized_query(user_query: str) -> str:
    return re.sub(r"[^a-z0-9_\-]+", " ", str(user_query or "").lower()).strip()


def _extract_video_ids(user_query: str) -> list[str]:
    pattern = r"\b[A-Za-z]+[0-9]{3}_x264\b"
    seen: list[str] = []
    for match in re.finditer(pattern, str(user_query or ""), flags=re.IGNORECASE):
        value = match.group(0)
        if value not in seen:
            seen.append(value)
    return seen


def _match_enum_values(user_query: str, values: list[str], aliases: dict[str, str] | None = None) -> list[str]:
    query = _normalized_query(user_query)
    matched: list[str] = []
    aliases = aliases or {}
    for value in values:
        canonical = str(value).strip().lower()
        if not canonical:
            continue
        candidates = {canonical}
        for alias, mapped in aliases.items():
            if mapped == canonical:
                candidates.add(alias.lower())
        if any(re.search(rf"(?<![a-z0-9]){re.escape(candidate)}(?![a-z0-9])", query) for candidate in candidates):
            matched.append(canonical)
    return matched


def _query_contains_phrase(user_query: str, phrase: str) -> bool:
    query = _normalized_query(user_query)
    return bool(re.search(rf"(?<![a-z0-9]){re.escape(phrase.lower())}(?![a-z0-9])", query))


def build_text2sql_plan(
    *,
    user_query: str,
    db_path: Path,
    table_name: str = "episodic_events",
) -> dict[str, Any]:
    schema_columns = get_sqlite_table_columns(db_path, table_name=table_name)
    enum_values = {
        column: get_distinct_column_values(db_path, column, table_name=table_name)
        for column in ENUM_COLUMNS
        if column in schema_columns
    }
    object_matches = _match_enum_values(user_query, enum_values.get("object_type", []), OBJECT_ALIASES)
    color_matches = _match_enum_values(user_query, enum_values.get("object_color_en", []), COLOR_ALIASES)
    zone_matches = _match_enum_values(user_query, enum_values.get("scene_zone_en", []), ZONE_ALIASES)
    video_ids = _extract_video_ids(user_query)

    hard_filters: list[dict[str, Any]] = []
    reasoning: list[str] = []

    if video_ids and "video_id" in schema_columns:
        hard_filters.append({"field": "video_id", "op": "=", "value": video_ids[0]})
        reasoning.append("Detected explicit video_id mention and promoted it to a hard filter.")

    unique_object_matches = sorted(set(object_matches))
    if len(unique_object_matches) == 1:
        hard_filters.append({"field": "object_type", "op": "=", "value": unique_object_matches[0]})
        reasoning.append("Exactly one object entity matched the schema, so object_type is safe as a hard filter.")
    elif len(unique_object_matches) > 1:
        reasoning.append("Multiple object entities were mentioned, so object_type stays soft to avoid over-filtering.")

    unique_zone_matches = sorted(set(zone_matches))
    if len(unique_zone_matches) == 1:
        hard_filters.append({"field": "scene_zone_en", "op": "contains", "value": unique_zone_matches[0]})
        reasoning.append("Exactly one scene zone matched the schema, so scene_zone_en is used as a hard filter.")

    unique_color_matches = sorted(set(color_matches))
    if len(unique_color_matches) == 1 and len(unique_object_matches) == 1:
        color_value = unique_color_matches[0]
        object_value = unique_object_matches[0]
        if _query_contains_phrase(user_query, f"{color_value} {object_value}"):
            hard_filters.append({"field": "object_color_en", "op": "=", "value": color_value})
            reasoning.append("Color is directly attached to the only detected entity, so object_color_en is safe as a hard filter.")
        else:
            reasoning.append("Color mention is not safely attached to the primary entity, so it remains soft evidence.")
    elif unique_color_matches:
        reasoning.append("Color mention remains soft because the query references multiple entities or ambiguous appearance details.")

    soft_terms = _expanded_query_terms(user_query)
    soft_phrases = _query_phrases(user_query)
    return {
        "schema_columns": schema_columns,
        "enum_values": enum_values,
        "hard_filters": hard_filters,
        "soft_terms": soft_terms,
        "soft_phrases": soft_phrases,
        "reasoning": reasoning,
        "object_matches": unique_object_matches,
        "color_matches": unique_color_matches,
        "zone_matches": unique_zone_matches,
        "video_ids": video_ids,
    }


def _build_hard_filter_clause(filters: list[dict[str, Any]]) -> tuple[str, list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    for item in filters:
        field = str(item.get("field") or "").strip()
        op = str(item.get("op") or "=").strip().lower()
        value = item.get("value")
        if not field or value is None:
            continue
        if op in {"=", "=="}:
            clauses.append(f"lower({field}) = ?")
            params.append(str(value).strip().lower())
        elif op == "contains":
            clauses.append(f"lower({field}) LIKE ?")
            params.append(f"%{str(value).strip().lower()}%")
    return " AND ".join(clauses), params


def run_guided_sql_candidate(
    *,
    user_query: str,
    plan: dict[str, Any],
    limit: int,
    db_path: Path,
    table_name: str = "episodic_events",
) -> dict[str, Any]:
    text_blob = (
        "lower("
        "coalesce(object_type,'') || ' ' || "
        "coalesce(object_color_en,'') || ' ' || "
        "coalesce(scene_zone_en,'') || ' ' || "
        "coalesce(event_summary_en,'') || ' ' || "
        "coalesce(event_text_en,'') || ' ' || "
        "coalesce(appearance_notes_en,'') || ' ' || "
        "coalesce(video_id,'')"
        ")"
    )
    phrases = list(plan.get("soft_phrases") or [])
    terms = list(plan.get("soft_terms") or [])
    object_matches = list(plan.get("object_matches") or [])
    color_matches = list(plan.get("color_matches") or [])
    zone_matches = list(plan.get("zone_matches") or [])
    hard_filter_clause, hard_filter_params = _build_hard_filter_clause(list(plan.get("hard_filters") or []))

    score_parts: list[str] = []
    select_params: list[Any] = []
    text_where_parts: list[str] = []
    text_where_params: list[Any] = []

    for value in object_matches:
        score_parts.append("CASE WHEN lower(coalesce(object_type,'')) = ? THEN 4 ELSE 0 END")
        select_params.append(str(value).strip().lower())
    for value in color_matches:
        score_parts.append("CASE WHEN lower(coalesce(object_color_en,'')) = ? THEN 3 ELSE 0 END")
        select_params.append(str(value).strip().lower())
    for value in zone_matches:
        score_parts.append("CASE WHEN lower(coalesce(scene_zone_en,'')) LIKE ? THEN 2 ELSE 0 END")
        select_params.append(f"%{str(value).strip().lower()}%")

    for phrase in phrases:
        score_parts.append(f"CASE WHEN {text_blob} LIKE ? THEN 3 ELSE 0 END")
        select_params.append(f"%{phrase}%")
        text_where_parts.append(f"{text_blob} LIKE ?")
        text_where_params.append(f"%{phrase}%")
    for term in terms:
        score_parts.append(f"CASE WHEN {text_blob} LIKE ? THEN 1 ELSE 0 END")
        select_params.append(f"%{term}%")
        text_where_parts.append(f"{text_blob} LIKE ?")
        text_where_params.append(f"%{term}%")

    score_expr = " + ".join(score_parts) if score_parts else "0"
    where_parts: list[str] = []
    params: list[Any] = []
    if hard_filter_clause:
        where_parts.append(f"({hard_filter_clause})")
        params.extend(hard_filter_params)
    if text_where_parts:
        where_parts.append("(" + " OR ".join(text_where_parts) + ")")
        params.extend(text_where_params)
    where_expr = " AND ".join(where_parts) if where_parts else "1=1"
    sql = (
        "SELECT event_id, video_id, track_id, start_time, end_time, object_type, "
        "object_color_en, scene_zone_en, event_summary_en, "
        f"({score_expr}) AS _lexical_score "
        f"FROM {table_name} "
        f"WHERE {where_expr} "
        "ORDER BY _lexical_score DESC, start_time ASC "
        "LIMIT ?"
    )
    params = [*select_params, *params, int(limit)]
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = [dict(row) for row in conn.execute(sql, params).fetchall()]

    if score_parts:
        rows = [row for row in rows if float(row.get("_lexical_score") or 0.0) > 0.0]

    for row in rows:
        row.setdefault("_source_type", "sql_guided_plan")
        row.setdefault("_distance", 0.0)
    return {
        "sql": sql,
        "params": params[:-1],
        "rows": rows,
        "hard_filters": plan.get("hard_filters") or [],
        "soft_terms": terms,
        "soft_phrases": phrases,
    }


def run_relaxed_sql_fallback(
    *,
    user_query: str,
    limit: int,
    db_path: Path,
    table_name: str = "episodic_events",
) -> dict[str, Any]:
    text_blob = (
        "lower("
        "coalesce(object_type,'') || ' ' || "
        "coalesce(object_color_en,'') || ' ' || "
        "coalesce(scene_zone_en,'') || ' ' || "
        "coalesce(event_summary_en,'') || ' ' || "
        "coalesce(event_text_en,'') || ' ' || "
        "coalesce(appearance_notes_en,'') || ' ' || "
        "coalesce(video_id,'')"
        ")"
    )
    terms = _expanded_query_terms(user_query)
    phrases = _query_phrases(user_query)
    score_parts: list[str] = []
    select_params: list[Any] = []
    where_parts: list[str] = []
    where_params: list[Any] = []

    for phrase in phrases:
        score_parts.append(f"CASE WHEN {text_blob} LIKE ? THEN 3 ELSE 0 END")
        select_params.append(f"%{phrase}%")
        where_parts.append(f"{text_blob} LIKE ?")
        where_params.append(f"%{phrase}%")

    for term in terms:
        score_parts.append(f"CASE WHEN {text_blob} LIKE ? THEN 1 ELSE 0 END")
        select_params.append(f"%{term}%")
        where_parts.append(f"{text_blob} LIKE ?")
        where_params.append(f"%{term}%")

    score_expr = " + ".join(score_parts) if score_parts else "0"
    where_expr = " OR ".join(where_parts) if where_parts else "1=1"
    sql = (
        "SELECT event_id, video_id, track_id, start_time, end_time, object_type, "
        "object_color_en, scene_zone_en, event_summary_en, "
        f"({score_expr}) AS _lexical_score "
        f"FROM {table_name} "
        f"WHERE ({where_expr}) "
        "ORDER BY _lexical_score DESC, start_time ASC "
        "LIMIT ?"
    )
    params = [*select_params, *where_params, int(limit)]

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = [dict(row) for row in conn.execute(sql, params).fetchall()]

    if score_parts:
        rows = [row for row in rows if float(row.get("_lexical_score") or 0.0) > 0.0]

    for row in rows:
        row.setdefault("_source_type", "sql_relaxed_fallback")
        row.setdefault("_distance", 0.0)

    return {
        "sql": sql,
        "params": params[:-1],
        "terms": terms,
        "phrases": phrases,
        "rows": rows,
        "relaxation_notes": [
            "Converted hard WHERE equality filters into lexical recall over structured and textual evidence.",
            "Ranked by phrase hits, field matches, and matched query term count instead of requiring every structured condition.",
        ],
    }
