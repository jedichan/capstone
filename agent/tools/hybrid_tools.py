import json
import logging
import os
import sqlite3
from typing import Any

from langchain_core.tools import tool

from node.types import default_chroma_collection, default_chroma_path, default_sqlite_db_path
from tools.bm25_index import BM25Index, reciprocal_rank_fuse
from tools.db_access import ChromaGateway
from tools.llamaindex_adapter import run_llamaindex_vector_query, use_llamaindex_vector


_LOG = logging.getLogger(__name__)


def _hybrid_bm25_fused_enabled() -> bool:
    """``AGENT_HYBRID_BM25_FUSED`` controls the new full-corpus BM25 fusion.

    Defaults to enabled. Set ``AGENT_HYBRID_BM25_FUSED=0`` to fall back to the
    pure-vector path while diagnosing regressions (the old subset-BM25 was
    removed in P1-2 and is intentionally not recoverable).
    """

    raw = (os.getenv("AGENT_HYBRID_BM25_FUSED", "1") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _bm25_top_k(query: str, filters: dict[str, Any] | None, *, limit: int) -> list[dict[str, Any]]:
    if not _hybrid_bm25_fused_enabled() or limit <= 0:
        return []
    try:
        oversample = max(int(os.getenv("AGENT_HYBRID_BM25_OVERSAMPLE", "3")), 1)
        index = BM25Index(default_sqlite_db_path())
        return index.search(query, top_k=limit * oversample, filters=filters or None)
    except Exception as exc:
        _LOG.warning("BM25Index search failed: %s; falling back to vector-only.", exc)
        return []


def _format_hybrid_payload(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in rows:
        fused_score = r.get("_fused_score")
        out.append(
            {
                "event_id": r.get("event_id"),
                "video_id": r.get("video_id"),
                "track_id": r.get("track_id"),
                "event_text": r.get("event_text") or r.get("event_summary_en") or r.get("event_text_en"),
                "start_time": r.get("start_time"),
                "end_time": r.get("end_time"),
                "object_type": r.get("object_type"),
                "object_color_en": r.get("object_color_en"),
                "scene_zone_en": r.get("scene_zone_en"),
                "distance": r.get("_distance"),
                # Legacy field names preserved so ``normalize_hybrid_rows`` and
                # downstream consumers (rerank/answer/summary) keep working
                # without per-call adapters.
                "_distance": r.get("_distance"),
                "_hybrid_score": float(fused_score) if fused_score is not None else r.get("_vector_score"),
                "_bm25": r.get("_bm25"),
                "_vector_score": r.get("_vector_score"),
                "_fused_rank": r.get("_fused_rank"),
                "_source_ranks": r.get("_source_ranks"),
                "_source_type": r.get("_source_type") or "hybrid",
            }
        )
    return out

@tool
def get_temporal_anchor(event_description: str) -> str:
    """
    Find the timestamp (start_time, end_time) of a specific semantic event to serve as a temporal anchor.
    Example query: 'When did the red car leave?'
    """
    # 简易实现：使用 SQL 的 LIKE 模糊匹配在 sqlite 中查找。也可以让 LLM 动态写 SQL，但工具化封装更安全。
    db_path = default_sqlite_db_path()
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # 简单提取关键字（为了演示），实际中可依赖更复杂的提取，这里直接做简单的模糊搜索
        # 如果 event_description 包含颜色或动作，简单分词
        keywords = event_description.split()
        if not keywords:
            keywords = [event_description]
            
        where_clauses = " OR ".join(["COALESCE(event_text_en, event_summary_en, '') LIKE ?"] * len(keywords))
        params = [f"%{k}%" for k in keywords]
        
        sql = (
            f"SELECT video_id, start_time, end_time, "
            f"COALESCE(event_text_en, event_summary_en) AS event_text "
            f"FROM episodic_events WHERE {where_clauses} LIMIT 3"
        )
        cursor.execute(sql, params)
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return "未找到符合描述的时间锚点。"
            
        res = [dict(r) for r in rows]
        return f"Found temporal anchors:\n{json.dumps(res, ensure_ascii=False, indent=2)}\n\nYou can extract start_time from this."
    except Exception as e:
        return f"Temporal anchor failed. Please note: Time-dimension queries are generally unsupported, but if finding an event time, it failed here: {str(e)}"

@tool
def dynamic_weighted_vector_search(
    query: str,
    filters: dict,
    alpha: float = 0.5,
    limit: int = 5,
    *,
    video_filter: list[str] | None = None,
) -> str:
    """Hybrid retrieval: pure-vector (Chroma) fused with corpus-wide BM25 via RRF.

    Parameters:
    - query: Semantic query string from the user.
    - filters: Metadata filter dictionary (e.g., {"object_color_en": "red"}).
    - alpha: Deprecated legacy parameter. The fusion is now rank-based (RRF)
      so ``alpha`` is no longer used for ranking; it is only consulted as a
      hint to the metadata filter pass-through (``alpha >= 0.8`` keeps the
      semantic branch unfiltered, mirroring the previous behaviour).
    - limit: Number of fused results to return.
    - video_filter: If provided, restrict vector search to these video_ids
      (Chroma ``$in`` filter).  Used by Tier 1 two-stage retrieval.

    Set ``AGENT_HYBRID_BM25_FUSED=0`` to disable the BM25 channel and fall back
    to vector-only retrieval (used as a clean rollback knob during P1-2).
    """

    db_path = default_chroma_path()
    collection = default_chroma_collection()
    fused_enabled = _hybrid_bm25_fused_enabled()
    over = max(int(os.getenv("AGENT_HYBRID_VECTOR_OVERSAMPLE", "3")), 1)
    vector_top_k = max(int(limit) * over, int(limit))
    pass_filters = dict(filters) if (alpha is None or alpha < 0.8) else {}
    if video_filter:
        pass_filters["video_id"] = {"$in": video_filter}
    try:
        if use_llamaindex_vector():
            _, vector_rows = run_llamaindex_vector_query(
                query,
                filters=pass_filters,
                limit=vector_top_k,
                db_path=db_path,
                collection_name=collection,
            )
        else:
            gateway = ChromaGateway(db_path=db_path, collection_name=collection)
            meta_list = [{"key": k, "value": v} for k, v in (pass_filters or {}).items()]
            vector_rows = gateway.search(
                query=query,
                metadata_filters=meta_list,
                limit=vector_top_k,
            )
    except Exception as e:
        return f"Hybrid search failed on Chroma: {str(e)}"

    bm25_rows = _bm25_top_k(query, pass_filters, limit=int(limit)) if fused_enabled else []

    if not vector_rows and not bm25_rows:
        # Keep the "on Chroma" / "on LlamaIndex" suffix so the existing retry
        # parser in ``parallel_retrieval_fusion_node._run_hybrid_branch`` still
        # recognises this as an empty-result message.
        backend_label = "LlamaIndex" if use_llamaindex_vector() else "Chroma"
        suffix = f"vector=0, bm25={'disabled' if not fused_enabled else 0}"
        return (
            f"Hybrid search returned no results on {backend_label} "
            f"(collection={collection}, {suffix}). Try relaxing filters."
        )

    if bm25_rows:
        fused_rows = reciprocal_rank_fuse(
            [vector_rows, bm25_rows],
            top_k=int(limit),
        )
        backend = "llamaindex+bm25" if use_llamaindex_vector() else "chroma+bm25"
    else:
        fused_rows = vector_rows[: int(limit)]
        for rank, row in enumerate(fused_rows, start=1):
            row.setdefault("_fused_rank", rank)
            row.setdefault("_source_ranks", [(0, rank)])
        backend = "llamaindex" if use_llamaindex_vector() else "chroma"

    payload = _format_hybrid_payload(fused_rows)
    header = (
        f"Hybrid search successful on {backend} "
        f"(vector={len(vector_rows)}, bm25={len(bm25_rows)}, fused={len(payload)})"
    )
    return f"{header}:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
