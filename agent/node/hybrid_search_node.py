from typing import Any
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore

from agents.hybrid_search.sub_agent import run_hybrid_sub_agent
from tools.rerank import rerank_rows
from .retrieval_contracts import (
    build_routing_metrics,
    build_search_config,
    normalize_hybrid_rows,
    parent_projection_enabled,
    project_rows_to_parent_context,
    summarize_parent_context,
)
from .types import AgentState, InputValidator
import time

def create_hybrid_search_node(llm=None, **kwargs):
    def hybrid_search_node(state: AgentState, config: RunnableConfig, store: BaseStore) -> dict[str, Any]:
        del config, store
        
        user_query = InputValidator.resolve_active_query(state)
        
        # If it's a reflection retry, use optimized query
        reflection_result = state.get("reflection_result", {})
        if reflection_result.get("needs_retry", False) and state.get("retry_count", 0) > 0:
            user_query = state.get("optimized_query", user_query)
            
        current_retry = int(state.get("retry_count", 0) or 0)
        
        start = time.perf_counter()
        
        try:
            summary, raw_rows = run_hybrid_sub_agent(user_query, llm)
            duration = time.perf_counter() - start
            search_config = build_search_config(state.get("search_config", {}))
            normalized_rows = normalize_hybrid_rows(raw_rows)
            reranked_rows, rerank_meta = rerank_rows(
                user_query,
                normalized_rows,
                top_k=max(len(normalized_rows), int(search_config.get("rerank_top_k", 5))),
                candidate_limit=int(search_config.get("rerank_candidate_limit", 20)),
            )
            rerank_top_k = int(search_config.get("rerank_top_k", 5))
            parent_context: list[dict[str, Any]] = []
            if parent_projection_enabled():
                final_rows = project_rows_to_parent_context(
                    reranked_rows,
                    limit=rerank_top_k,
                )
                result_mode = "parent_projection"
            else:
                final_rows = reranked_rows[:rerank_top_k]
                result_mode = "child_only"
                parent_context = summarize_parent_context(reranked_rows, limit=rerank_top_k)
            
            return {
                "hybrid_result": normalized_rows,
                "rerank_result": final_rows,
                "reflection_result": {"feedback": "Retrieval successful", "quality_score": 1.0, "needs_retry": False},
                "tool_error": None,
                "retry_count": current_retry,
                "current_node": "hybrid_search_node",
                "search_explain": f"{summary}\nRerank={rerank_meta}\nResult mode={result_mode}, rows={len(final_rows)}",
                "routing_metrics": build_routing_metrics(
                    execution_mode="parallel_fusion",
                    label="semantic",
                    query=user_query,
                    sql_rows_count=0,
                    hybrid_rows_count=len(normalized_rows),
                ),
                "search_config": search_config,
                "sql_debug": {
                    "result_mode": result_mode,
                    "parent_context": parent_context,
                    "rerank_meta": rerank_meta,
                },
                "messages": [AIMessage(content=f"Hybrid Search Sub-Agent complete. Summary:\n{summary}")],
            }
        except Exception as exc:
            return {
                "hybrid_result": [],
                "rerank_result": [],
                "reflection_result": {"feedback": f"Sub-Agent execution failed: {exc}", "quality_score": 0.0, "needs_retry": True},
                "tool_error": f"Hybrid Search Sub-Agent failed: {exc}",
                "retry_count": current_retry,
                "current_node": "hybrid_search_node",
                "messages": [AIMessage(content=f"Hybrid Search Sub-Agent failed: {exc}")],
            }

    return hybrid_search_node
