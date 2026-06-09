from agents.shared import NodeComponent
from node.pure_sql_node import create_pure_sql_node


PURE_SQL_COMPONENT = NodeComponent(
    name="pure_sql_node",
    factory=create_pure_sql_node,
)


def build_pure_sql_node(llm):
    return PURE_SQL_COMPONENT.build(llm)

