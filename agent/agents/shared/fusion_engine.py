import os
from typing import Any, Dict, List, Optional, Tuple


def _safe_float(v: Any, default: float) -> float:
    try:
        return float(v)
    except Exception:
        return default


def load_fusion_weights() -> Dict[str, Dict[str, float]]:
    return {
        "structured": {
            "sql": _safe_float(os.getenv("AGENT_FUSION_STRUCTURED_SQL_WEIGHT", "0.8"), 0.8),
            "hybrid": _safe_float(os.getenv("AGENT_FUSION_STRUCTURED_HYBRID_WEIGHT", "0.2"), 0.2),
        },
        "semantic": {
            "sql": _safe_float(os.getenv("AGENT_FUSION_SEMANTIC_SQL_WEIGHT", "0.2"), 0.2),
            "hybrid": _safe_float(os.getenv("AGENT_FUSION_SEMANTIC_HYBRID_WEIGHT", "0.8"), 0.8),
        },
        "mixed": {
            "sql": _safe_float(os.getenv("AGENT_FUSION_MIXED_SQL_WEIGHT", "0.5"), 0.5),
            "hybrid": _safe_float(os.getenv("AGENT_FUSION_MIXED_HYBRID_WEIGHT", "0.5"), 0.5),
        },
        "multi_hop": {
            "sql": _safe_float(os.getenv("AGENT_FUSION_MULTIHOP_SQL_WEIGHT", "0.3"), 0.3),
            "hybrid": _safe_float(os.getenv("AGENT_FUSION_MULTIHOP_HYBRID_WEIGHT", "0.7"), 0.7),
        },
        "rrf_k": {
            "k": _safe_float(os.getenv("AGENT_FUSION_RRF_K", "60"), 60.0),
        },
    }


# P1-6: structured signals can soft-bias the {sql, hybrid} weights. Evidence
# is additive but bounded so the classifier label remains the primary driver.
_SIGNAL_BIAS_PER_HIT = _safe_float(os.getenv("AGENT_FUSION_SIGNAL_BIAS_PER_HIT", "0.05"), 0.05)
_SIGNAL_BIAS_CAP = _safe_float(os.getenv("AGENT_FUSION_SIGNAL_BIAS_CAP", "0.2"), 0.2)


def _apply_signal_bias(
    base_sql: float,
    base_hybrid: float,
    signals: Optional[Dict[str, Any]],
) -> Tuple[float, float, Dict[str, Any]]:
    if not isinstance(signals, dict):
        return base_sql, base_hybrid, {"applied": False, "reason": "no_signals"}

    def _hits(key: str) -> int:
        value = signals.get(key)
        if isinstance(value, (list, tuple, set)):
            return len(value)
        if isinstance(value, int):
            return max(0, value)
        return 0

    metadata_n = _hits("metadata_hits")
    relation_n = _hits("relation_cues")
    multi_step_n = _hits("multi_step_cues")

    sql_bias = min(_SIGNAL_BIAS_CAP, metadata_n * _SIGNAL_BIAS_PER_HIT)
    hybrid_bias = min(_SIGNAL_BIAS_CAP, (relation_n + multi_step_n) * _SIGNAL_BIAS_PER_HIT)

    sql_adj = max(0.0, base_sql + sql_bias - hybrid_bias)
    hybrid_adj = max(0.0, base_hybrid + hybrid_bias - sql_bias)
    total = sql_adj + hybrid_adj
    if total <= 0.0:
        return base_sql, base_hybrid, {"applied": False, "reason": "degenerate_after_bias"}
    sql_norm = sql_adj / total
    hybrid_norm = hybrid_adj / total
    return sql_norm, hybrid_norm, {
        "applied": True,
        "metadata_hits": metadata_n,
        "relation_cues": relation_n,
        "multi_step_cues": multi_step_n,
        "sql_bias": sql_bias,
        "hybrid_bias": hybrid_bias,
        "base_sql": base_sql,
        "base_hybrid": base_hybrid,
    }


def _normalize_event_id(event_id: Any) -> str | None:
    """Normalize event_id to a canonical string for cross-branch matching.

    SQL branch returns ``int`` (e.g. 42), Chroma branch may return ``str``
    (e.g. "42") or ``None``.  This function coerces numeric event_ids to
    ``str(int(...))`` so that ``_row_key`` generates the same key regardless
    of source branch, fixing the RRF overlap bug (P0-1).
    """
    if event_id is None:
        return None
    try:
        # Try int conversion: 42 → "42", "42" → "42", 42.0 → "42"
        return str(int(float(event_id)))
    except (ValueError, TypeError):
        # Non-numeric event_id (e.g. Chroma record ID): use as-is
        return str(event_id).strip()


def _row_key(row: Dict[str, Any]) -> str:
    event_id = _normalize_event_id(row.get("event_id"))
    if event_id is not None:
        return f"event_id:{event_id}"
    return "|".join(
        [
            str(row.get("video_id", "")),
            str(row.get("track_id", "")),
            str(row.get("start_time", "")),
            str(row.get("end_time", "")),
            str(row.get("event_summary_en", row.get("event_text", ""))),
        ]
    )


def weighted_rrf_fuse(
    sql_rows: List[Dict[str, Any]],
    hybrid_rows: List[Dict[str, Any]],
    label: str,
    limit: int = 50,
    *,
    signals: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    cfg = load_fusion_weights()
    weights = cfg.get(label, cfg["mixed"])
    base_sql = float(weights["sql"])
    base_hybrid = float(weights["hybrid"])
    w_sql, w_hybrid, bias_meta = _apply_signal_bias(base_sql, base_hybrid, signals)
    rrf_k = float(cfg["rrf_k"]["k"])

    score_map: Dict[str, float] = {}
    row_map: Dict[str, Dict[str, Any]] = {}
    trace: Dict[str, Dict[str, Any]] = {}

    def _merge_rows(existing: Dict[str, Any] | None, incoming: Dict[str, Any], *, prefer_incoming: bool) -> Dict[str, Any]:
        if existing is None:
            return dict(incoming)
        merged = dict(existing)
        items = incoming.items() if prefer_incoming else existing.items()
        for key, value in incoming.items():
            if key not in merged or merged.get(key) in (None, "", [], {}):
                merged[key] = value
        if prefer_incoming:
            for key, value in incoming.items():
                if value not in (None, "", [], {}):
                    merged[key] = value
        return merged

    for rank, row in enumerate(sql_rows, start=1):
        key = _row_key(row)
        score = w_sql * (1.0 / (rrf_k + rank))
        score_map[key] = score_map.get(key, 0.0) + score
        row_map[key] = _merge_rows(row_map.get(key), row, prefer_incoming=False)
        trace.setdefault(key, {}).update({"sql_rank": rank, "sql_rrf": score, "seen_in_sql": True})

    for rank, row in enumerate(hybrid_rows, start=1):
        key = _row_key(row)
        score = w_hybrid * (1.0 / (rrf_k + rank))
        score_map[key] = score_map.get(key, 0.0) + score
        row_map[key] = _merge_rows(row_map.get(key), row, prefer_incoming=True)
        trace.setdefault(key, {}).update({"hybrid_rank": rank, "hybrid_rrf": score, "seen_in_hybrid": True})

    ranked = sorted(score_map.items(), key=lambda x: x[1], reverse=True)[:limit]
    out: List[Dict[str, Any]] = []
    overlap_count = 0
    for key, score in ranked:
        row = dict(row_map[key])
        row_trace = dict(trace.get(key, {}))
        if row_trace.get("seen_in_sql") and row_trace.get("seen_in_hybrid"):
            overlap_count += 1
            row["_source_type"] = "fused"
        row_trace["total_rrf"] = float(score)
        row["_fusion_score"] = float(score)
        row["_fusion_label"] = label
        row["_fusion_trace"] = row_trace
        out.append(row)

    return out, {
        "label": label,
        "weights": {"sql": w_sql, "hybrid": w_hybrid},
        "rrf_k": rrf_k,
        "sql_count": len(sql_rows),
        "hybrid_count": len(hybrid_rows),
        "fused_count": len(out),
        "overlap_count": overlap_count,
        "method": "weighted_rrf",
        "signal_bias": bias_meta,
    }

