import json

from langchain_core.messages import ToolMessage

from agents.shared import run_react_sub_agent
from tools.llamaindex_adapter import run_llamaindex_vector_query, use_llamaindex_vector
from tools.hybrid_tools import dynamic_weighted_vector_search, get_temporal_anchor

HYBRID_SEARCH_AGENT_PROMPT = """
You are an advanced hybrid retrieval assistant responsible for searching complex temporal and semantic events from a video event library.

【AVAILABLE TOOLS】
1. `get_temporal_anchor`: Find the timestamp when a specific event occurred. Use this when the query has temporal dependencies — e.g., "after the red car leaves" → first search for "red car leaves" to get the anchor time.
2. `dynamic_weighted_vector_search`: Hybrid retrieval combining metadata filters with semantic vector search. Adjust alpha (0.0-1.0) to balance:
   - low alpha (0.1-0.3): emphasize exact metadata filters (object type, color, zone)
   - high alpha (0.7-0.9): emphasize semantic meaning (actions, relationships, context)

【WORKFLOW】
1. Analyze the user query. Identify temporal dependencies ("after/before", "while", "then").
2. If temporal dependency exists, call `get_temporal_anchor` first to establish the time anchor.
3. Choose alpha based on query characteristics:
   - Mostly metadata/attributes → alpha 0.2-0.4
   - Mostly semantic/relational → alpha 0.7-0.9
   - Mixed → alpha 0.5-0.6
4. Call `dynamic_weighted_vector_search` with the chosen alpha and any time constraints.
5. Summarize the results in natural English. Do NOT output raw JSON.
"""


def _extract_hybrid_result(response: dict) -> tuple[str, list[dict]]:
    summary = response["messages"][-1].content
    raw_rows: list[dict] = []
    for msg in reversed(response["messages"]):
        if isinstance(msg, ToolMessage) and msg.name == "dynamic_weighted_vector_search":
            content = msg.content
            if "Hybrid search successful on Chroma" in content or "混合检索成功" in content:
                try:
                    json_str = content.split(":\n", 1)[1]
                    raw_rows = json.loads(json_str)
                except Exception:
                    pass
            break
    return summary, raw_rows


def run_hybrid_sub_agent(user_query: str, llm):
    if use_llamaindex_vector():
        return run_llamaindex_vector_query(user_query, limit=5)
    return run_react_sub_agent(
        user_query=user_query,
        llm=llm,
        tools=[get_temporal_anchor, dynamic_weighted_vector_search],
        system_prompt=HYBRID_SEARCH_AGENT_PROMPT,
        result_extractor=_extract_hybrid_result,
    )
