import os

from langgraph.graph import END, START, StateGraph

from node.answer_node import final_answer_node
from node.match_verifier_node import create_match_verifier_node
from node.parallel_retrieval_fusion_node import create_parallel_retrieval_fusion_node
from node.query_classification_node import create_query_classification_node
from node.self_query_node import create_self_query_node
from node.summary_node import create_summary_node
from node.types import AgentState


def _verifier_node_disabled() -> bool:
    # P1-Next-A sanity check: when set to ``1``, the parallel-fusion graph
    # skips ``match_verifier_node`` entirely (parallel_retrieval_fusion ->
    # final_answer directly). Useful for measuring whether the advisory
    # verifier introduces any silent side-effects on the final answer/summary.
    raw = os.getenv("AGENT_DISABLE_VERIFIER_NODE", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _build_parallel_fusion_graph(llm):
    self_query_node = create_self_query_node(llm=llm)
    classify_node = create_query_classification_node(llm=llm)
    parallel_node = create_parallel_retrieval_fusion_node(llm=llm)
    summary_node = create_summary_node(llm=llm)
    verifier_disabled = _verifier_node_disabled()

    builder = StateGraph(AgentState)
    builder.add_node("self_query_node", self_query_node)
    builder.add_node("query_classification_node", classify_node)
    builder.add_node("parallel_retrieval_fusion_node", parallel_node)
    if not verifier_disabled:
        match_verifier = create_match_verifier_node(llm=llm)
        builder.add_node("match_verifier_node", match_verifier)
    builder.add_node("final_answer_node", final_answer_node)
    builder.add_node("summary_node", summary_node)

    builder.add_edge(START, "self_query_node")
    builder.add_edge("self_query_node", "query_classification_node")
    builder.add_edge("query_classification_node", "parallel_retrieval_fusion_node")
    if verifier_disabled:
        builder.add_edge("parallel_retrieval_fusion_node", "final_answer_node")
    else:
        builder.add_edge("parallel_retrieval_fusion_node", "match_verifier_node")
        builder.add_edge("match_verifier_node", "final_answer_node")
    builder.add_edge("final_answer_node", "summary_node")
    builder.add_edge("summary_node", END)
    return builder.compile()


def build_graph(llm, init_prompt_text: str = ""):
    # P1-5/P3-3 (2026-05-02): the legacy_router execution mode and its
    # supporting nodes (tool_router_node / reflection_node / cot_engine /
    # router_prompts) were removed. ``init_prompt_text`` is kept on the
    # signature for backwards-compatible callers but no longer consumed.
    del init_prompt_text
    return _build_parallel_fusion_graph(llm)
