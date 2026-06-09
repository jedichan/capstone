from agents.shared import NodeComponent
from node.hybrid_search_node import create_hybrid_search_node


HYBRID_SEARCH_COMPONENT = NodeComponent(
    name="hybrid_search_node",
    factory=create_hybrid_search_node,
)


def build_hybrid_search_node(llm):
    return HYBRID_SEARCH_COMPONENT.build(llm)

