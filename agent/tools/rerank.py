import os
import threading
from typing import Any


_MODEL_LOCK = threading.Lock()
_CROSS_ENCODER = None
_LOAD_ERROR: str | None = None


def rerank_enabled() -> bool:
    raw = os.getenv("AGENT_ENABLE_RERANK", "1").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def get_rerank_model_name() -> str:
    return os.getenv("AGENT_RERANK_MODEL", "jinaai/jina-reranker-v2-base-multilingual")


def _rerank_metadata_in_query() -> bool:
    """When enabled, video/type/color/zone metadata is appended to the *query* side
    of the cross-encoder pair instead of the doc side.  This lets the reranker
    learn to down-weight rows whose metadata does not align with the query intent
    rather than blindly boosting every row that mentions metadata-like tokens.

    .. warning::
        Defaults to OFF — injecting metadata into the query side causes
        cross-video noise for broad queries (e.g. "is there any footage of
        a collision").  The preferred approach is doc-side prefix at index
        time (see ``agent/todo.md`` R4 Step 1)."""
    raw = os.getenv("AGENT_RERANK_METADATA_IN_QUERY", "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _strip_keywords(text: str) -> str:
    """Remove ``Keywords: ...`` tails that may leak annotation metadata into
    the cross-encoder doc side and confuse the reranker."""
    import re

    return re.sub(r"\s*Keywords:\s*\[[^\]]*\].*$", "", text).strip()


def _build_pair_text(row: dict[str, Any]) -> str:
    summary = row.get("event_summary_en") or row.get("event_text_en") or row.get("event_text") or ""
    summary = _strip_keywords(str(summary))
    parts: list[str] = []
    if not _rerank_metadata_in_query():
        if row.get("video_id"):
            parts.append(f"video {row.get('video_id')}")
        if row.get("object_type"):
            parts.append(f"type {row.get('object_type')}")
        if row.get("object_color_en"):
            parts.append(f"color {row.get('object_color_en')}")
        if row.get("scene_zone_en"):
            parts.append(f"zone {row.get('scene_zone_en')}")
    if summary:
        parts.append(summary)
    return ". ".join(str(part).strip() for part in parts if str(part).strip())


def _enrich_query_with_metadata(query: str, row: dict[str, Any]) -> str:
    """When AGENT_RERANK_METADATA_IN_QUERY=1, inject row metadata into the
    query side so that the cross-encoder scores semantic relevance *given*
    the metadata rather than simply boosting rows that mention those tokens."""
    if not _rerank_metadata_in_query():
        return query
    tags: list[str] = []
    if row.get("video_id"):
        tags.append(f"in video {row.get('video_id')}")
    if row.get("object_type"):
        tags.append(f"object type {row.get('object_type')}")
    if row.get("object_color_en"):
        tags.append(f"color {row.get('object_color_en')}")
    if row.get("scene_zone_en"):
        tags.append(f"zone {row.get('scene_zone_en')}")
    if not tags:
        return query
    return f"{query} [{'; '.join(tags)}]"


def _get_cross_encoder():
    global _CROSS_ENCODER, _LOAD_ERROR
    if _CROSS_ENCODER is not None:
        return _CROSS_ENCODER
    if _LOAD_ERROR is not None:
        raise RuntimeError(_LOAD_ERROR)
    with _MODEL_LOCK:
        if _CROSS_ENCODER is not None:
            return _CROSS_ENCODER
        if _LOAD_ERROR is not None:
            raise RuntimeError(_LOAD_ERROR)
        try:
            from sentence_transformers import CrossEncoder
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
            _CROSS_ENCODER = CrossEncoder(get_rerank_model_name(), device=device, trust_remote_code=True)
            return _CROSS_ENCODER
        except Exception as exc:
            _LOAD_ERROR = f"rerank model load failed: {exc}"
            raise RuntimeError(_LOAD_ERROR) from exc


def rerank_rows(
    query: str,
    rows: list[dict[str, Any]] | None,
    *,
    top_k: int,
    candidate_limit: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_rows = list(rows or [])
    if not source_rows:
        return [], {"enabled": False, "reason": "empty_input"}
    if not rerank_enabled():
        return source_rows[:top_k], {"enabled": False, "reason": "disabled"}

    candidates = source_rows[: max(1, int(candidate_limit))]
    pairs = [(_enrich_query_with_metadata(query, row), _build_pair_text(row)) for row in candidates]
    try:
        model = _get_cross_encoder()
        scores = model.predict(pairs, batch_size=min(16, len(pairs)), show_progress_bar=False)
    except Exception as exc:
        return source_rows[:top_k], {"enabled": False, "reason": "load_or_predict_failed", "error": str(exc)}

    rescored: list[dict[str, Any]] = []
    for row, score in zip(candidates, scores):
        updated = dict(row)
        updated["_rerank_score"] = float(score)
        rescored.append(updated)
    rescored.sort(
        key=lambda item: (
            float(item.get("_rerank_score", 0.0)),
            float(item.get("_hybrid_score", 0.0) or 0.0),
            -float(item.get("_distance", 1.0) or 1.0),
        ),
        reverse=True,
    )
    tail = source_rows[len(candidates) :]
    return rescored[:top_k] + tail, {
        "enabled": True,
        "model": get_rerank_model_name(),
        "input_count": len(source_rows),
        "candidate_count": len(candidates),
        "output_count": min(len(source_rows), top_k + len(tail)),
    }
