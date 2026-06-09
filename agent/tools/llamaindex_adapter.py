import os
import re
import sqlite3
from pathlib import Path
from typing import Any

from langchain_openai import ChatOpenAI

from node.types import default_chroma_collection, default_chroma_path, default_sqlite_db_path
from tools.bm25_index import BM25Index, reciprocal_rank_fuse
from tools.db_access import ChromaGateway
from tools.llm import get_qwen_embedding


def _hybrid_bm25_fused_enabled() -> bool:
    """``AGENT_HYBRID_BM25_FUSED`` rollback knob, defaults on.

    Mirrors the helper in ``hybrid_tools`` so this module does not import the
    LangChain tool wrapper (would create a circular import).
    """

    raw = (os.getenv("AGENT_HYBRID_BM25_FUSED", "1") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}
from tools.sql_debug_utils import (
    build_text2sql_plan,
    extract_where_clause,
    find_unknown_sql_columns,
    get_sqlite_table_columns,
    log_sql_debug,
    run_guided_sql_candidate,
    run_relaxed_sql_fallback,
)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, "1" if default else "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def get_sql_retrieval_strategy() -> str:
    return os.getenv("AGENT_SQL_RETRIEVAL_STRATEGY", "deterministic").strip().lower() or "deterministic"


def use_llamaindex_sql() -> bool:
    return _env_flag("AGENT_USE_LLAMAINDEX_SQL", default=False)


def use_llamaindex_vector() -> bool:
    return _env_flag("AGENT_USE_LLAMAINDEX_VECTOR", default=False)


def llamaindex_enabled() -> bool:
    return use_llamaindex_sql() or use_llamaindex_vector()


def get_llamaindex_install_hint() -> str:
    return (
        "缺少 LlamaIndex 依赖。请安装: "
        "pip install llama-index llama-index-llms-openai "
        "llama-index-vector-stores-chroma"
    )


def _require_llamaindex_base() -> dict[str, Any]:
    try:
        from llama_index.core import Settings, SQLDatabase, VectorStoreIndex
        from llama_index.core.base.llms.types import CompletionResponse, CompletionResponseGen, LLMMetadata
        from llama_index.core.bridge.pydantic import PrivateAttr
        from llama_index.core.embeddings import BaseEmbedding
        from llama_index.core.llms import CustomLLM
        from llama_index.core.llms.callbacks import llm_completion_callback
        from llama_index.core.query_engine import NLSQLTableQueryEngine
        from llama_index.core.vector_stores import FilterOperator, MetadataFilter, MetadataFilters
        from llama_index.vector_stores.chroma import ChromaVectorStore
    except Exception as exc:
        raise RuntimeError(f"{get_llamaindex_install_hint()} ({exc})") from exc

    return {
        "Settings": Settings,
        "SQLDatabase": SQLDatabase,
        "VectorStoreIndex": VectorStoreIndex,
        "CompletionResponse": CompletionResponse,
        "CompletionResponseGen": CompletionResponseGen,
        "LLMMetadata": LLMMetadata,
        "PrivateAttr": PrivateAttr,
        "BaseEmbedding": BaseEmbedding,
        "CustomLLM": CustomLLM,
        "llm_completion_callback": llm_completion_callback,
        "NLSQLTableQueryEngine": NLSQLTableQueryEngine,
        "MetadataFilter": MetadataFilter,
        "MetadataFilters": MetadataFilters,
        "FilterOperator": FilterOperator,
        "ChromaVectorStore": ChromaVectorStore,
    }


def _build_llamaindex_llm() -> Any:
    modules = _require_llamaindex_base()
    CustomLLM = modules["CustomLLM"]
    CompletionResponse = modules["CompletionResponse"]
    LLMMetadata = modules["LLMMetadata"]
    llm_completion_callback = modules["llm_completion_callback"]
    # ``AGENT_LLAMAINDEX_LLM_PROVIDER`` (default OpenAI when an OpenAI key is
    # set) controls which OpenAI-compatible endpoint we use for NL2SQL. The
    # legacy behaviour preferred DashScope unconditionally which 401'd entire
    # eval runs whenever the DashScope key was absent or expired.
    provider = (os.getenv("AGENT_LLAMAINDEX_LLM_PROVIDER", "") or "").strip().lower()
    if provider == "dashscope" or (not provider and os.getenv("DASHSCOPE_API_KEY") and not os.getenv("OPENAI_API_KEY")):
        api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("DASHSCOPE_URL") or os.getenv("OPENAI_BASE_URL")
        default_model = "qwen3.6-plus"
    else:
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL") or ""
        default_model = os.getenv("AGENT_CHAT_MODEL", os.getenv("OPENAI_CHAT_MODEL", "gpt-5.4"))
    model_name = os.getenv("AGENT_LLAMAINDEX_SQL_MODEL") or os.getenv("AGENT_LLAMAINDEX_LLM_MODEL") or default_model
    common_kwargs: dict[str, Any] = {
        "model_name": model_name,
        "temperature": 0.0,
        "api_key": api_key,
    }
    if base_url:
        common_kwargs["base_url"] = base_url
    lc_llm = ChatOpenAI(**common_kwargs)

    class LangChainChatOpenAILLM(CustomLLM):
        _llm: ChatOpenAI = modules["PrivateAttr"]()
        _model_name: str = modules["PrivateAttr"]()

        def __init__(self, llm: ChatOpenAI, model_name: str, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self._llm = llm
            self._model_name = model_name

        @classmethod
        def class_name(cls) -> str:
            return "langchain_chat_openai_llm"

        @property
        def metadata(self) -> Any:
            return LLMMetadata(num_output=-1, is_chat_model=True, model_name=self._model_name)

        @llm_completion_callback()
        def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> Any:
            del formatted, kwargs
            response = self._llm.invoke(prompt)
            content = getattr(response, "content", "")
            if isinstance(content, list):
                text = "\n".join(str(item) for item in content)
            else:
                text = str(content or "")
            return CompletionResponse(text=text)

        @llm_completion_callback()
        def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> Any:
            del formatted, kwargs

            def _gen() -> Any:
                response = self.complete(prompt)
                yield CompletionResponse(text=response.text, delta=response.text)

            return _gen()

    return LangChainChatOpenAILLM(lc_llm, model_name=model_name)


def _build_llamaindex_embedding_model() -> Any:
    modules = _require_llamaindex_base()
    BaseEmbedding = modules["BaseEmbedding"]
    PrivateAttr = modules["PrivateAttr"]

    class DashScopeEmbedding(BaseEmbedding):
        _batch_size: int = PrivateAttr(default=10)

        def __init__(self, batch_size: int = 10, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self._batch_size = batch_size

        @classmethod
        def class_name(cls) -> str:
            return "dashscope_qwen_embedding"

        def _get_query_embedding(self, query: str) -> list[float]:
            return list(get_qwen_embedding(query))

        async def _aget_query_embedding(self, query: str) -> list[float]:
            return self._get_query_embedding(query)

        def _get_text_embedding(self, text: str) -> list[float]:
            return list(get_qwen_embedding(text))

        async def _aget_text_embedding(self, text: str) -> list[float]:
            return self._get_text_embedding(text)

        def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
            return [list(item) for item in get_qwen_embedding(texts)]

    return DashScopeEmbedding(batch_size=10)


def _coerce_response_metadata(response: Any) -> dict[str, Any]:
    for attr in ("metadata", "response_metadata", "extra_info"):
        value = getattr(response, attr, None)
        if isinstance(value, dict):
            return value
    return {}


def _extract_sql_query_from_response(response: Any) -> str:
    metadata = _coerce_response_metadata(response)
    for key in ("sql_query", "sql", "query_code"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    response_text = str(getattr(response, "response", response) or "").strip()
    match = re.search(r"SELECT[\s\S]+", response_text, flags=re.IGNORECASE)
    if match:
        return match.group(0).strip()
    return ""


def _ensure_select_limit(sql_query: str, limit: int) -> str:
    query = sql_query.strip().rstrip(";")
    if not query.upper().startswith("SELECT"):
        raise RuntimeError(f"LlamaIndex 生成了非 SELECT SQL: {query}")
    if re.search(r"\bLIMIT\s+\d+\b", query, flags=re.IGNORECASE):
        return query
    return f"{query} LIMIT {int(limit)}"


def _execute_select_rows(db_path: Path, sql_query: str) -> list[dict[str, Any]]:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql_query).fetchall()
    return [dict(row) for row in rows]


def _format_text2sql_plan_prompt(plan: dict[str, Any]) -> str:
    return (
        "Schema-aware planning context:\n"
        f"- Validated hard filters: {plan.get('hard_filters') or []}\n"
        f"- Soft evidence phrases: {plan.get('soft_phrases') or []}\n"
        f"- Soft evidence terms: {plan.get('soft_terms') or []}\n"
        f"- Matched object candidates: {plan.get('object_matches') or []}\n"
        f"- Matched color candidates: {plan.get('color_matches') or []}\n"
        f"- Matched zone candidates: {plan.get('zone_matches') or []}\n"
        f"- Schema enum values: {plan.get('enum_values') or {}}\n"
        f"- Planning notes: {plan.get('reasoning') or []}\n"
        "Rules:\n"
        "1. Only convert validated hard filters into strict WHERE predicates.\n"
        "2. Keep soft evidence in broad text predicates or ranking logic instead of hard equality filters.\n"
        "3. If the query mentions multiple entities, avoid forcing all entities and colors into equality filters.\n"
        "4. Use only real schema column names from episodic_events.\n"
    )


def _build_li_metadata_filters(filters: dict[str, Any] | None) -> Any:
    if not filters:
        return None
    modules = _require_llamaindex_base()
    MetadataFilter = modules["MetadataFilter"]
    MetadataFilters = modules["MetadataFilters"]
    FilterOperator = modules["FilterOperator"]
    li_filters = []
    for key, value in (filters or {}).items():
        if value is None:
            continue
        li_filters.append(MetadataFilter(key=str(key), operator=FilterOperator.EQ, value=value))
    if not li_filters:
        return None
    return MetadataFilters(filters=li_filters)


def _node_text(node: Any) -> str:
    if hasattr(node, "get_content"):
        try:
            return str(node.get_content(metadata_mode="none") or "").strip()
        except TypeError:
            return str(node.get_content() or "").strip()
    for attr in ("text", "content"):
        value = getattr(node, attr, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _score_to_distance(score: Any) -> float | None:
    try:
        score_f = float(score)
    except Exception:
        return None
    if 0.0 <= score_f <= 1.0:
        return max(0.0, 1.0 - score_f)
    return 0.0


def _legacy_sql_fallback_rows(user_query: str, limit: int, db_path: Path) -> list[dict[str, Any]]:
    from node.retrieval_contracts import extract_structured_filters, extract_text_tokens_for_sql

    filters = extract_structured_filters(user_query)
    params: list[Any] = []
    clauses: list[str] = []
    if "object_type" in filters:
        clauses.append("lower(object_type) = ?")
        params.append(filters["object_type"])
    if "object_color_en" in filters:
        clauses.append("lower(object_color_en) = ?")
        params.append(filters["object_color_en"])
    if "scene_zone_en" in filters:
        clauses.append("lower(scene_zone_en) LIKE ?")
        params.append(f"%{filters['scene_zone_en']}%")

    tokens = extract_text_tokens_for_sql(user_query, filters)
    text_clauses = []
    for t in tokens:
        text_clauses.append(
            "(lower(coalesce(event_text_en,'')) LIKE ? OR lower(coalesce(event_summary_en,'')) LIKE ? OR lower(coalesce(appearance_notes_en,'')) LIKE ?)"
        )
        params.extend([f"%{t}%", f"%{t}%", f"%{t}%"])
    if text_clauses:
        clauses.append("(" + " OR ".join(text_clauses) + ")")
    where_sql = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    sql = (
        "SELECT event_id, video_id, track_id, start_time, end_time, object_type, "
        "object_color_en, scene_zone_en, event_summary_en "
        "FROM episodic_events"
        f"{where_sql} ORDER BY start_time ASC LIMIT {int(limit)}"
    )
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute(sql, params).fetchall()]


def _run_deterministic_sql_query(
    user_query: str,
    *,
    limit: int = 80,
    table_name: str = "episodic_events",
    db_path: Path | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    target_db = Path(db_path or default_sqlite_db_path())
    plan = build_text2sql_plan(user_query=user_query, db_path=target_db, table_name=table_name)
    log_sql_debug(
        "deterministic_sql_start",
        user_query=user_query,
        db_path=str(target_db),
        table_name=table_name,
        text2sql_plan=plan,
    )

    guided = run_guided_sql_candidate(
        user_query=user_query,
        plan=plan,
        limit=limit,
        db_path=target_db,
        table_name=table_name,
    )
    log_sql_debug(
        "deterministic_sql_guided",
        guided_sql=guided["sql"],
        guided_hard_filters=guided["hard_filters"],
        guided_soft_terms=guided["soft_terms"],
        guided_soft_phrases=guided["soft_phrases"],
        guided_row_count=len(guided["rows"]),
    )
    if guided["rows"]:
        return (
            "Deterministic SQL planner retrieval complete "
            f"rows={len(guided['rows'])} stage=guided",
            guided["rows"],
        )

    strict_fallback_rows = _legacy_sql_fallback_rows(user_query, limit=limit, db_path=target_db)
    log_sql_debug(
        "deterministic_sql_legacy_fallback",
        legacy_row_count=len(strict_fallback_rows),
    )
    if strict_fallback_rows:
        return (
            "Deterministic SQL planner fallback recovered "
            f"rows={len(strict_fallback_rows)} stage=legacy",
            strict_fallback_rows,
        )

    relaxed = run_relaxed_sql_fallback(
        user_query=user_query,
        limit=limit,
        db_path=target_db,
        table_name=table_name,
    )
    log_sql_debug(
        "deterministic_sql_relaxed",
        fallback_sql=relaxed["sql"],
        fallback_terms=relaxed["terms"],
        fallback_phrases=relaxed.get("phrases", []),
        fallback_row_count=len(relaxed["rows"]),
        relaxation_notes=relaxed["relaxation_notes"],
    )
    if relaxed["rows"]:
        return (
            "Deterministic SQL planner retrieval complete "
            f"rows={len(relaxed['rows'])} stage=relaxed",
            relaxed["rows"],
        )

    return ("Deterministic SQL planner retrieval returned 0 rows", [])


def run_llamaindex_sql_query(
    user_query: str,
    *,
    limit: int = 80,
    table_name: str = "episodic_events",
    db_path: Path | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    strategy = get_sql_retrieval_strategy()
    if strategy not in {"llm_nl2sql", "nlsql", "llamaindex_nlsql"}:
        return _run_deterministic_sql_query(
            user_query,
            limit=limit,
            table_name=table_name,
            db_path=db_path,
        )

    modules = _require_llamaindex_base()
    SQLDatabase = modules["SQLDatabase"]
    NLSQLTableQueryEngine = modules["NLSQLTableQueryEngine"]
    target_db = Path(db_path or default_sqlite_db_path())
    try:
        from sqlalchemy import create_engine
    except Exception as exc:
        raise RuntimeError(f"缺少 SQLAlchemy，无法启用 LlamaIndex SQL: {exc}") from exc

    llm = _build_llamaindex_llm()
    engine = create_engine(f"sqlite:///{target_db}")
    sql_database = SQLDatabase(engine, include_tables=[table_name])
    schema_columns = get_sqlite_table_columns(target_db, table_name=table_name)
    plan = build_text2sql_plan(user_query=user_query, db_path=target_db, table_name=table_name)
    query_engine = NLSQLTableQueryEngine(
        sql_database=sql_database,
        tables=[table_name],
        llm=llm,
        synthesize_response=False,
    )
    guarded_query = (
        f"{user_query}\n\n"
        "Use SQLite only. Generate a SELECT query over episodic_events. "
        "Always include event_id, video_id, track_id, start_time, end_time, "
        "object_type, object_color_en, scene_zone_en, event_summary_en in the SELECT "
        f"when available. Limit the result size to at most {int(limit)} rows.\n\n"
        f"{_format_text2sql_plan_prompt(plan)}"
    )
    log_sql_debug(
        "llamaindex_sql_start",
        user_query=user_query,
        db_path=str(target_db),
        table_name=table_name,
        schema_columns=schema_columns,
        text2sql_plan=plan,
    )
    response = query_engine.query(guarded_query)
    sql_query = _extract_sql_query_from_response(response)
    if not sql_query:
        raise RuntimeError("LlamaIndex SQL 未返回可执行的 SQL 语句")
    final_sql = _ensure_select_limit(sql_query, limit=limit)
    where_clause = extract_where_clause(final_sql)
    unknown_columns = find_unknown_sql_columns(final_sql, schema_columns, table_name=table_name)
    log_sql_debug(
        "llamaindex_sql_generated",
        final_sql=final_sql,
        where_clause=where_clause,
        unknown_columns=unknown_columns,
    )
    try:
        rows = _execute_select_rows(target_db, final_sql)
    except Exception as exc:
        log_sql_debug(
            "llamaindex_sql_execute_error",
            final_sql=final_sql,
            where_clause=where_clause,
            unknown_columns=unknown_columns,
            error=str(exc),
        )
        return _run_deterministic_sql_query(
            user_query,
            limit=limit,
            table_name=table_name,
            db_path=target_db,
        )

    log_sql_debug(
        "llamaindex_sql_execute_complete",
        final_sql=final_sql,
        where_clause=where_clause,
        row_count=len(rows),
    )
    if not rows:
        return _run_deterministic_sql_query(
            user_query,
            limit=limit,
            table_name=table_name,
            db_path=target_db,
        )
    summary = f"LlamaIndex SQL retrieval complete rows={len(rows)} sql={final_sql}"
    return summary, rows


def run_llamaindex_vector_query(
    query: str,
    *,
    filters: dict[str, Any] | None = None,
    limit: int = 5,
    db_path: Path | None = None,
    collection_name: str | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    modules = _require_llamaindex_base()
    Settings = modules["Settings"]
    VectorStoreIndex = modules["VectorStoreIndex"]
    ChromaVectorStore = modules["ChromaVectorStore"]

    import chromadb

    target_db = Path(db_path or default_chroma_path())
    target_collection = collection_name or default_chroma_collection()
    embed_model = _build_llamaindex_embedding_model()
    Settings.embed_model = embed_model

    client = chromadb.PersistentClient(path=str(target_db))
    collection = client.get_or_create_collection(
        name=target_collection,
        metadata={"hnsw:space": "cosine"},
    )
    vector_store = ChromaVectorStore(chroma_collection=collection)
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embed_model)
    retriever_kwargs: dict[str, Any] = {"similarity_top_k": max(int(limit) * 6, int(limit))}
    retrieved = []
    li_filters = _build_li_metadata_filters(filters)
    if li_filters is not None:
        retriever = index.as_retriever(**{**retriever_kwargs, "filters": li_filters})
        retrieved = retriever.retrieve(query)
    if not retrieved:
        retriever = index.as_retriever(**retriever_kwargs)
        retrieved = retriever.retrieve(query)

    candidates: list[dict[str, Any]] = []
    texts: list[str] = []
    vector_scores: list[float] = []
    for item in retrieved:
        node = getattr(item, "node", item)
        metadata = getattr(node, "metadata", {}) or {}
        score = getattr(item, "score", None)
        text = _node_text(node)
        candidates.append(
            {
                "event_id": metadata.get("event_id") or getattr(node, "node_id", None),
                "video_id": metadata.get("video_id"),
                "track_id": metadata.get("entity_hint") or metadata.get("track_id"),
                "start_time": metadata.get("start_time"),
                "end_time": metadata.get("end_time"),
                "object_type": metadata.get("object_type"),
                "object_color_en": metadata.get("object_color") or metadata.get("object_color_en"),
                "scene_zone_en": metadata.get("scene_zone") or metadata.get("scene_zone_en"),
                "event_summary_en": text,
                "event_text": text,
                "_distance": _score_to_distance(score),
                "_hybrid_score": float(score) if score is not None else None,
                "_bm25": None,
                "_source_type": "hybrid",
            }
        )
        texts.append(text)
        vector_scores.append(float(score) if score is not None else 0.0)

    if not candidates:
        gateway = ChromaGateway(db_path=target_db, collection_name=target_collection)
        meta_list = [{"key": k, "value": v} for k, v in (filters or {}).items()]
        fallback_rows = gateway.search(
            query=query,
            metadata_filters=meta_list,
            limit=limit,
        )
        if not fallback_rows and meta_list:
            fallback_rows = gateway.search(
                query=query,
                metadata_filters=[],
                limit=limit,
            )
        return (
            f"LlamaIndex vector returned 0 rows; fallback pure-vector rows={len(fallback_rows)} collection={target_collection}",
            fallback_rows,
        )

    # Vector list is already ranked by LlamaIndex (descending by score / cosine
    # similarity). Reciprocal-rank fuse with the corpus-wide BM25 channel so the
    # lexical signal is real (not the legacy subset BM25 over the candidates we
    # already vector-retrieved).
    del texts, vector_scores  # candidates already preserve rank order

    fused_enabled = _hybrid_bm25_fused_enabled()
    bm25_rows: list[dict[str, Any]] = []
    if fused_enabled:
        try:
            oversample = max(int(os.getenv("AGENT_HYBRID_BM25_OVERSAMPLE", "3")), 1)
            bm25_index = BM25Index(default_sqlite_db_path())
            bm25_rows = bm25_index.search(query, top_k=int(limit) * oversample, filters=filters or None)
        except Exception as exc:  # pragma: no cover - defensive guard
            bm25_rows = []
            summary_suffix = f" (bm25 disabled: {exc})"
        else:
            summary_suffix = ""
    else:
        summary_suffix = " (bm25 disabled by AGENT_HYBRID_BM25_FUSED=0)"

    if bm25_rows:
        fused = reciprocal_rank_fuse([candidates, bm25_rows], top_k=int(limit))
    else:
        fused = candidates[: int(limit)]
        for rank, row in enumerate(fused, start=1):
            row.setdefault("_fused_rank", rank)
            row.setdefault("_source_ranks", [(0, rank)])

    for row in fused:
        # Legacy fields kept so ``normalize_hybrid_rows`` continues to surface
        # ``_hybrid_score`` / ``_bm25`` on the downstream rerank rows.
        if row.get("_hybrid_score") is None:
            row["_hybrid_score"] = row.get("_fused_score")
        if row.get("_distance") is None and row.get("_vector_score") is not None:
            row["_distance"] = max(0.0, 1.0 - float(row["_vector_score"]))
        row.setdefault("_source_type", "hybrid")

    summary = (
        f"LlamaIndex vector retrieval complete rows={len(fused)} "
        f"vector={len(candidates)} bm25={len(bm25_rows)} "
        f"collection={target_collection}{summary_suffix}"
    )
    return summary, fused
