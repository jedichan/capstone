import json
import os
import re
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Any, Dict, List, Tuple

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore

from agents.shared import weighted_rrf_fuse
from tools.hybrid_tools import dynamic_weighted_vector_search
from tools.llamaindex_adapter import (
    run_llamaindex_sql_query,
    run_llamaindex_vector_query,
    use_llamaindex_sql,
    use_llamaindex_vector,
)
from tools.rerank import rerank_rows
from tools.sql_debug_utils import _build_hard_filter_clause, build_text2sql_plan, run_relaxed_sql_fallback
from .retrieval_contracts import (
    build_routing_metrics,
    build_search_config,
    extract_structured_filters,
    extract_text_tokens_for_sql,
    infer_sql_plan,
    normalize_hybrid_rows,
    parent_projection_enabled,
    normalize_sql_rows,
    project_rows_to_parent_context,
    summarize_parent_context,
)
from .types import AgentState, InputValidator
from .types import default_sqlite_db_path


def _safe_sub_agent_call(func, *args, **kwargs) -> Tuple[str, List[Dict[str, Any]], str | None]:
    try:
        summary, rows = func(*args, **kwargs)
        return summary or "", rows or [], None
    except Exception as exc:
        return "", [], str(exc)


def _get_video_ids_for_cameras(camera_ids: list[str]) -> list[str]:
    """Return all distinct video_ids in episodic_events for the given camera IDs."""
    if not camera_ids:
        return []
    db_path = str(default_sqlite_db_path())
    try:
        placeholders = ",".join("?" * len(camera_ids))
        upper = [c.strip().upper() for c in camera_ids]
        with sqlite3.connect(db_path, timeout=5) as conn:
            rows = conn.execute(
                f"SELECT DISTINCT video_id FROM episodic_events "
                f"WHERE UPPER(camera_id) IN ({placeholders})",
                upper,
            ).fetchall()
        return [r[0] for r in rows if r[0]]
    except Exception as exc:
        print(f"[camera_filter] video_id lookup failed: {exc}")
        return []


def _sql_use_fts5_enabled() -> bool:
    """``AGENT_SQL_USE_FTS5`` toggles the FTS5 lexical path (default on).

    Set ``AGENT_SQL_USE_FTS5=0`` to fall back to the legacy LIKE-OR scan when
    diagnosing FTS5 regressions. Independent of the BM25 fusion knob in
    ``hybrid_tools``.
    """

    raw = (os.getenv("AGENT_SQL_USE_FTS5", "1") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _fts5_table_present(conn: sqlite3.Connection, table_name: str = "episodic_events_fts") -> bool:
    try:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type IN ('table','view') AND name = ?",
            (table_name,),
        ).fetchone()
        return row is not None
    except sqlite3.OperationalError:
        return False


# FTS5 MATCH syntax: inside ``"..."`` the ONLY special character is ``"`` itself
# (escaped as ``""``). Operators like ``*():"+-,`` are literal inside quotes.
# We wrap each token in double quotes so user content is always a phrase search.
_FTS5_TOKEN_QUOTE = re.compile(r'"')


def _build_fts5_match_expr(tokens: List[str]) -> str:
    if not tokens:
        return ""
    quoted: List[str] = []
    for tok in tokens:
        clean = _FTS5_TOKEN_QUOTE.sub('""', tok)
        quoted.append(f'"{clean}"')
    return " OR ".join(quoted)


def _run_sql_branch(
    user_query: str,
    search_config: Dict[str, Any],
    *,
    camera_filter: list[str] | None = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    if use_llamaindex_sql():
        return run_llamaindex_sql_query(
            user_query,
            limit=int(search_config.get("sql_limit", 80)),
        )
    db_path = default_sqlite_db_path()
    sql_limit = int(search_config.get("sql_limit", 80))

    # Phase 1 — fast FTS5 / LIKE path powered by build_text2sql_plan
    # The plan provides alias-expanded hard filters and soft terms for text search.
    plan = build_text2sql_plan(user_query=user_query, db_path=db_path)
    hard_clause, hard_params = _build_hard_filter_clause(list(plan.get("hard_filters") or []))

    # Camera hard-filter (eval harness): prepend so it takes highest priority
    cam_clause = ""
    cam_params: List[Any] = []
    if camera_filter:
        upper_cams = [c.strip().upper() for c in camera_filter]
        _ph = ",".join("?" * len(upper_cams))
        cam_clause = f"UPPER(camera_id) IN ({_ph})"
        cam_params = list(upper_cams)

    clauses: List[str] = []
    params: List[Any] = cam_params + list(hard_params)
    if cam_clause:
        clauses.append(cam_clause)
    if hard_clause:
        clauses.append(f"({hard_clause})")

    # Soft terms from the plan are richer than extract_text_tokens_for_sql:
    # they include alias expansion, stemming, and phrase extraction.
    soft_terms = list(plan.get("soft_terms") or [])
    soft_phrases = list(plan.get("soft_phrases") or [])

    # Deduplicate and prioritize: phrases first (they carry more signal), then terms.
    text_tokens: List[str] = []
    seen_tokens: set[str] = set()
    for p in soft_phrases:
        for t in re.findall(r"[a-z0-9_]+", p):
            if t not in seen_tokens:
                seen_tokens.add(t)
                text_tokens.append(t)
    for t in soft_terms:
        if t not in seen_tokens:
            seen_tokens.add(t)
            text_tokens.append(t)

    text_strategy = "none"
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        # P3-10: WAL mode + cache for better concurrent read performance
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=-8000")

        use_fts5 = _sql_use_fts5_enabled() and text_tokens and _fts5_table_present(conn)
        if use_fts5:
            match_expr = _build_fts5_match_expr(text_tokens)
            clauses.append(
                "event_id IN (SELECT rowid FROM episodic_events_fts WHERE episodic_events_fts MATCH ?)"
            )
            params.append(match_expr)
            text_strategy = "fts5"
        elif text_tokens:
            text_clauses = []
            for t in text_tokens:
                text_clauses.append(
                    "(lower(coalesce(event_text_en,'')) LIKE ? OR lower(coalesce(event_summary_en,'')) LIKE ? OR lower(coalesce(appearance_notes_en,'')) LIKE ?)"
                )
                params.extend([f"%{t}%", f"%{t}%", f"%{t}%"])
            clauses.append("(" + " OR ".join(text_clauses) + ")")
            text_strategy = "like"

        where_sql = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = (
            "SELECT event_id, video_id, track_id, start_time, end_time, object_type, "
            "object_color_en, scene_zone_en, event_summary_en "
            "FROM episodic_events"
            f"{where_sql} ORDER BY start_time ASC LIMIT {sql_limit}"
        )
        rows = normalize_sql_rows([dict(r) for r in conn.execute(sql, params).fetchall()])

    # Phase 2 — relaxed fallback when the structured plan yields zero rows.
    # Skip when a camera_filter is active: the relaxed path has no camera
    # awareness and would re-introduce cross-camera pollution.
    if not rows and (hard_clause or text_tokens) and not camera_filter:
        try:
            fallback = run_relaxed_sql_fallback(
                user_query=user_query, limit=sql_limit, db_path=db_path,
            )
            fb_rows = normalize_sql_rows(fallback.get("rows") or [])
            if fb_rows:
                return f"SQL relaxed-fallback rows={len(fb_rows)} text_strategy={text_strategy}", fb_rows
        except Exception:
            pass

    return f"SQL direct retrieval rows={len(rows)} text_strategy={text_strategy}", rows


def _run_hybrid_branch(
    user_query: str,
    search_config: Dict[str, Any],
    *,
    video_filter: list[str] | None = None,
    camera_filter: list[str] | None = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    if not (user_query or "").strip():
        return "Hybrid retrieval skipped: empty query", []
    # Camera filter from eval harness: resolve to video_ids for those cameras,
    # then intersect with the coarse temporal filter to preserve time-based
    # focusing while still enforcing the camera constraint.
    # Intersection strategy:
    #   - coarse_filter non-empty AND overlaps camera set → use intersection
    #     (camera-correct + temporally relevant, best of both worlds)
    #   - coarse_filter non-empty but no overlap → wrong-camera coarse result
    #     (classic G423-black-hole case) → use all camera video_ids
    #   - coarse_filter empty → use all camera video_ids
    if camera_filter:
        cam_video_ids = _get_video_ids_for_cameras(camera_filter)
        if cam_video_ids:
            cam_set = set(cam_video_ids)
            coarse_ids = _coarse_video_filter(user_query) or []
            coarse_set = set(coarse_ids)
            intersect = cam_set & coarse_set
            if intersect:
                video_filter = list(intersect)
                print(f"[camera_filter] hybrid restricted to {len(video_filter)} "
                      f"video_ids (cam∩coarse) for cameras {camera_filter}")
            else:
                # Coarse filter returned wrong-camera candidates (e.g. G423 black hole)
                # or returned nothing — fall back to all camera video_ids.
                video_filter = cam_video_ids
                print(f"[camera_filter] hybrid restricted to {len(cam_video_ids)} "
                      f"video_ids (camera only, coarse gave no overlap) for cameras {camera_filter}")
        else:
            print(f"[camera_filter] no video_ids found for cameras {camera_filter}; "
                  "using coarse filter fallback")
            video_filter = _coarse_video_filter(user_query)
    # Tier 1: if no video_filter provided, run coarse filter — its embedding
    # will be cached (P1-3 LRU) and reused by the hybrid search below.
    elif video_filter is None:
        video_filter = _coarse_video_filter(user_query)
    filters = extract_structured_filters(user_query)
    if use_llamaindex_vector():
        summary, rows = run_llamaindex_vector_query(
            user_query,
            filters=filters,
            limit=int(search_config.get("hybrid_limit", 50)),
        )
        return summary, normalize_hybrid_rows(rows)
    invoke_kwargs: dict[str, Any] = {
        "query": user_query,
        "filters": filters,
        "alpha": float(search_config.get("hybrid_alpha", 0.7)),
        "limit": int(search_config.get("hybrid_limit", 50)),
    }
    if video_filter:
        invoke_kwargs["video_filter"] = video_filter
    msg = dynamic_weighted_vector_search.invoke(invoke_kwargs)
    lowered = (msg or "").lower()
    if "failed on chroma" in lowered or "error code" in lowered:
        raise RuntimeError(msg)

    # Retry with looser semantic emphasis when metadata filters are too strict.
    if "returned no results on chroma" in lowered and filters:
        fallback_msg = dynamic_weighted_vector_search.invoke(
            {
                "query": user_query,
                "filters": {},
                "alpha": float(search_config.get("hybrid_fallback_alpha", 0.9)),
                "limit": int(search_config.get("hybrid_limit", 50)),
            }
        )
        fallback_lowered = (fallback_msg or "").lower()
        if "failed on chroma" in fallback_lowered or "error code" in fallback_lowered:
            raise RuntimeError(fallback_msg)
        if ":\n" in fallback_msg:
            msg = fallback_msg
        else:
            return f"{msg} | fallback={fallback_msg}", []

    if ":\n" not in msg:
        return msg, []
    payload = msg.split(":\n", 1)[1]
    rows = json.loads(payload)
    normalized = normalize_hybrid_rows(rows)
    return "Hybrid direct retrieval complete", normalized


def _apply_scene_boost(
    rows: list[dict[str, Any]], scene_constraints: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Tier 2: boost rows whose video matches query scene constraints."""
    from pathlib import Path
    lam = 0.1  # Scene boost lambda — conservative default
    try:
        lam = float(os.environ.get("AGENT_SCENE_BOOST_LAMBDA", "0.1"))
    except Exception:
        pass
    try:
        from tools.scene_attrs import query_video_attrs
        from db.config import get_graph_sqlite_db_path
        db_path = get_graph_sqlite_db_path()
    except Exception:
        return rows

    boosted: list[dict[str, Any]] = []
    total_boost = 0.0
    for row in rows:
        vid = str(row.get("video_id", "")).strip()
        boost = 0.0
        if vid:
            try:
                attrs = query_video_attrs(Path(db_path), vid)
            except Exception:
                attrs = {}
            for sc in scene_constraints:
                attr_name = str(sc.get("attr_name", "")).strip()
                if attr_name in attrs:
                    boost += lam * float(sc.get("weight", 0.5)) * attrs[attr_name]
        updated = dict(row)
        updated["_scene_boost"] = round(boost, 4)
        total_boost += boost
        boosted.append(updated)

    if total_boost > 0:
        # Sort by boost descending (existing order preserved via stable sort)
        boosted.sort(key=lambda r: r.get("_scene_boost", 0.0), reverse=True)
        print(f"[scene_boost] applied to {len(boosted)} rows, total_boost={total_boost:.4f}, "
              f"constraints={[c['attr_name'] for c in scene_constraints]}")
    return boosted


def _coarse_video_filter(user_query: str) -> list[str] | None:
    """Tier 1 coarse stage: query the video collection for top-3 candidate videos.
    
    Called inside _run_hybrid_branch. The embedding is computed here and
    cached by P1-3 LRU; the subsequent hybrid search will hit the cache.
    """
    import time
    t0 = time.perf_counter()
    try:
        from db.config import get_graph_chroma_path, get_graph_chroma_video_collection
        from tools.db_access import ChromaGateway
        from tools.llm import get_qwen_embedding

        gateway = ChromaGateway(
            db_path=get_graph_chroma_path(),
            collection_name=get_graph_chroma_video_collection(),
        )
        query_vec = get_qwen_embedding(user_query)
        res = gateway._collection.query(
            query_embeddings=[query_vec], n_results=3, include=["metadatas"]
        )
        ids = res.get("ids", [[]])[0]
        elapsed = (time.perf_counter() - t0) * 1000
        if ids:
            print(f"[coarse_video] {len(ids)} candidates in {elapsed:.0f}ms: {ids}")
            return [str(i) for i in ids]
        print(f"[coarse_video] no candidates in {elapsed:.0f}ms")
    except Exception as exc:
        elapsed = (time.perf_counter() - t0) * 1000
        print(f"[coarse_video] failed in {elapsed:.0f}ms: {exc}")
    return None


def create_parallel_retrieval_fusion_node(llm=None, **kwargs):
    del kwargs
    del llm
    branch_timeout = float(os.getenv("AGENT_PARALLEL_BRANCH_TIMEOUT_SEC", "30"))
    fused_limit = int(os.getenv("AGENT_FUSION_TOP_K", "50"))

    def parallel_retrieval_fusion_node(state: AgentState, config: RunnableConfig, store: BaseStore) -> dict[str, Any]:
        del config, store
        user_query = InputValidator.resolve_active_query(state)
        search_config = build_search_config(state.get("search_config", {}))
        classification_result = state.get("classification_result", {}) or {}
        if not isinstance(classification_result, dict):
            classification_result = {}
        label = classification_result.get("label") or "mixed"
        classification_signals = classification_result.get("signals") if isinstance(classification_result.get("signals"), dict) else {}
        sql_plan = infer_sql_plan(user_query, search_config)

        start = time.perf_counter()
        sql_summary = ""
        hybrid_summary = ""
        sql_rows: List[Dict[str, Any]] = []
        hybrid_rows: List[Dict[str, Any]] = []
        sql_error = None
        hybrid_error = None

        # Camera filter injected by the eval harness (via AgentState.camera_filter).
        # Restricts both SQL (WHERE camera_id IN (...)) and ChromaDB (video_id $in)
        # retrieval to the specified camera(s), fixing the G423 black-hole issue.
        camera_filter: list[str] | None = state.get("camera_filter") or None
        if camera_filter:
            # Normalise to uppercase strings
            camera_filter = [c.strip().upper() for c in camera_filter if c.strip()]

        disable_sql = os.getenv("AGENT_DISABLE_SQL_BRANCH", "0").strip().lower() in {"1", "true", "yes", "on"}

        with ThreadPoolExecutor(max_workers=2) as ex:
            if disable_sql:
                sql_summary, sql_rows, sql_error = "", [], None
            else:
                f_sql = ex.submit(
                    _safe_sub_agent_call, _run_sql_branch, user_query, search_config,
                    camera_filter=camera_filter,
                )
            f_hybrid = ex.submit(
                _safe_sub_agent_call, _run_hybrid_branch, user_query, search_config,
                camera_filter=camera_filter,
            )

            if not disable_sql:
                try:
                    sql_summary, sql_rows, sql_error = f_sql.result(timeout=branch_timeout)
                except TimeoutError:
                    sql_error = f"pure_sql timeout ({branch_timeout}s)"
            try:
                hybrid_summary, hybrid_rows, hybrid_error = f_hybrid.result(timeout=branch_timeout)
            except TimeoutError:
                hybrid_error = f"hybrid timeout ({branch_timeout}s)"

        both_failed = (sql_error is not None and hybrid_error is not None)
        if both_failed:
            duration = time.perf_counter() - start
            return {
                "sql_result": [],
                "hybrid_result": [],
                "rerank_result": [],
                "tool_error": f"Both branches failed: sql={sql_error}; hybrid={hybrid_error}",
                "current_node": "parallel_retrieval_fusion_node",
                "metrics": {
                    "duration": duration,
                    "degraded": True,
                    "degraded_reason": "both_failed",
                },
                "messages": [AIMessage(content="Parallel retrieval failed on both branches.")],
            }

        if sql_error and not hybrid_error:
            fused = hybrid_rows[:fused_limit]
            fusion_meta = {
                "label": label,
                "degraded": True,
                "degraded_reason": "sql_failed",
                "method": "fallback_hybrid_only",
            }
        elif hybrid_error and not sql_error:
            fused = sql_rows[:fused_limit]
            fusion_meta = {
                "label": label,
                "degraded": True,
                "degraded_reason": "hybrid_failed",
                "method": "fallback_sql_only",
            }
        else:
            fused, fusion_meta = weighted_rrf_fuse(
                sql_rows,
                hybrid_rows,
                label=label,
                limit=fused_limit,
                signals=classification_signals or None,
            )
            fusion_meta["degraded"] = False
            if classification_signals:
                fusion_meta["signals"] = classification_signals

        # P0-3: Soft degradation replaces the former hard ``structured_zero_guardrail``.
        # Previously, when the classifier said "structured", SQL returned zero rows
        # and at least one structured filter was present, the node wiped ``fused``
        # to empty, even when the hybrid branch had useful candidates. That killed
        # recoverable cases such as UCFCrime prompts where the filter token hit
        # the synonym dictionary but the SQL row did not.
        # New behavior:
        #   * sql=0 + hybrid>0 -> keep the RRF-fused hybrid tail (rank-preserved),
        #     mark ``degraded_reason=sql_zero_hybrid_fallback`` for observability.
        #   * sql=0 + hybrid=0 -> return empty with ``all_branches_zero``.
        #   * sql>0 -> untouched, normal weighted RRF wins.
        structured_filters = extract_structured_filters(user_query)
        if label == "structured" and not sql_rows and structured_filters:
            if hybrid_rows:
                fusion_meta = {
                    **fusion_meta,
                    "degraded": True,
                    "degraded_reason": "sql_zero_hybrid_fallback",
                    "method": fusion_meta.get("method", "weighted_rrf"),
                    "structured_filters": structured_filters,
                }
            else:
                fused = []
                fusion_meta = {
                    "label": label,
                    "degraded": True,
                    "degraded_reason": "all_branches_zero",
                    "method": "no_candidates",
                    "structured_filters": structured_filters,
                }

        if hybrid_error:
            fusion_meta["degraded"] = True
        if sql_error:
            fusion_meta["degraded"] = True

        rerank_top_k = int(search_config.get("rerank_top_k", 5))
        rerank_candidate_limit = int(search_config.get("rerank_candidate_limit", 20))

        # Tier 2: apply scene attribute boost before reranking
        scene_constraints = (state.get("self_query_result") or {}).get("scene_constraints") or []
        if scene_constraints:
            fused = _apply_scene_boost(fused, scene_constraints)

        reranked_rows, rerank_meta = rerank_rows(
            user_query,
            fused,
            top_k=fused_limit,
            candidate_limit=rerank_candidate_limit,
        )
        parent_context: List[Dict[str, Any]] = []
        if parent_projection_enabled():
            final_rows = project_rows_to_parent_context(reranked_rows, limit=rerank_top_k)
            result_mode = "parent_projection"
        else:
            final_rows = reranked_rows[:rerank_top_k]
            result_mode = "child_only"
            parent_context = summarize_parent_context(reranked_rows, limit=rerank_top_k)
        duration = time.perf_counter() - start
        routing_metrics = build_routing_metrics(
            execution_mode="parallel_fusion",
            label=label,
            query=user_query,
            sql_rows_count=len(sql_rows),
            hybrid_rows_count=len(hybrid_rows),
            sql_error=sql_error,
            hybrid_error=hybrid_error,
        )
        return {
            "sql_result": sql_rows,
            "hybrid_result": hybrid_rows,
            "rerank_result": final_rows,
            "tool_error": None,
            "current_node": "parallel_retrieval_fusion_node",
            "search_explain": (
                "Parallel retrieval completed. "
                f"label={label}, sql_rows={len(sql_rows)}, hybrid_rows={len(hybrid_rows)}, fused_rows={len(fused)}, final_rows={len(final_rows)}, result_mode={result_mode}"
            ),
            "routing_metrics": routing_metrics,
            "search_config": search_config,
            "sql_plan": sql_plan,
            "sql_debug": {
                "duration": duration,
                "sql_summary": sql_summary,
                "hybrid_summary": hybrid_summary,
                "sql_error": sql_error,
                "hybrid_error": hybrid_error,
                "fusion_meta": fusion_meta,
                "rerank_meta": rerank_meta,
                "parent_rows_count": len(final_rows),
                "result_mode": result_mode,
                "parent_context": parent_context,
            },
            "messages": [
                AIMessage(
                    content=(
                        f"Parallel retrieval + fusion complete (label={label}, "
                        f"sql={len(sql_rows)}, hybrid={len(hybrid_rows)}, fused={len(reranked_rows)})."
                    )
                )
            ],
        }

    return parallel_retrieval_fusion_node
