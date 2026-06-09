from typing import Any
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore

from agents.pure_sql.sub_agent import run_pure_sql_sub_agent
from .retrieval_contracts import build_routing_metrics, build_search_config, infer_sql_plan, normalize_sql_rows
from .types import AgentState, InputValidator, default_sqlite_db_path
import time

def create_pure_sql_node(llm=None, **kwargs):
    def pure_sql_node(state: AgentState, config: RunnableConfig, store: BaseStore) -> dict[str, Any]:
        del config, store
        
        user_query = InputValidator.resolve_active_query(state)
        
        # If it's a reflection retry, use optimized query
        reflection_result = state.get("reflection_result", {})
        if reflection_result.get("needs_retry", False) and state.get("retry_count", 0) > 0:
            user_query = state.get("optimized_query", user_query)
            
        current_retry = int(state.get("retry_count", 0) or 0)
        
        start = time.perf_counter()
        print(f"[DEBUG] pure_sql_node user_query: {user_query}")
        print(f"[DEBUG] pure_sql_node sqlite_db_path: {default_sqlite_db_path()}")
        try:
            summary, raw_rows = run_pure_sql_sub_agent(user_query, llm)
            duration = time.perf_counter() - start
            normalized_rows = normalize_sql_rows(raw_rows)
            search_config = build_search_config(state.get("search_config", {}))
            sql_plan = infer_sql_plan(user_query, search_config)
            
            return {
                "sql_result": normalized_rows,
                "rerank_result": normalized_rows,
                "reflection_result": {"feedback": "Retrieval successful", "quality_score": 1.0, "needs_retry": False},
                "tool_error": None,
                "retry_count": current_retry,
                "current_node": "pure_sql_node",
                "routing_metrics": build_routing_metrics(
                    execution_mode="parallel_fusion",
                    label="structured",
                    query=user_query,
                    sql_rows_count=len(normalized_rows),
                    hybrid_rows_count=0,
                ),
                "search_config": search_config,
                "sql_plan": sql_plan,
                "sql_debug": {
                    "duration": duration,
                    "agent_summary": summary,
                    "db_path": str(default_sqlite_db_path()),
                },
                "messages": [AIMessage(content=f"SQL Sub-Agent retrieval complete. Summary:\n{summary}")],
            }
        except Exception as exc:
            return {
                "sql_result": [],
                "rerank_result": [],
                "reflection_result": {"feedback": f"Sub-Agent execution failed: {exc}", "quality_score": 0.0, "needs_retry": True},
                "tool_error": f"Pure SQL Sub-Agent retrieval failed: {exc}",
                "retry_count": current_retry,
                "current_node": "pure_sql_node",
                "sql_debug": {"last_error": str(exc)},
                "messages": [AIMessage(content=f"Pure SQL Sub-Agent retrieval failed: {exc}")],
            }

    return pure_sql_node
