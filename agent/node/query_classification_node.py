from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore

from agents.shared import classify_mode_from_label, classify_query
from .types import AgentState, InputValidator


def create_query_classification_node(**kwargs):
    llm = kwargs.get("llm")

    def query_classification_node(state: AgentState, config: RunnableConfig, store: BaseStore) -> dict[str, Any]:
        del store
        user_query = InputValidator.resolve_active_query(state)
        result = classify_query(user_query, llm=llm, config=config)
        label = result["label"]
        answer_type = result.get("answer_type", "unknown")
        compat_mode = classify_mode_from_label(label)
        return {
            "classification_result": result,
            "answer_type": answer_type,
            "tool_choice": {
                "mode": compat_mode,
                "sql_needed": compat_mode == "pure_sql",
                "hybrid_needed": compat_mode == "hybrid_search",
                "sub_queries": {"sql": {}} if compat_mode == "pure_sql" else {"hybrid": {}},
            },
            "current_node": "query_classification_node",
            "messages": [
                AIMessage(
                    content=(
                        f"Classification complete: label={label}, answer_type={answer_type}, "
                        f"compat_mode={compat_mode}, execution=parallel_sql_hybrid"
                    )
                )
            ],
        }

    return query_classification_node
