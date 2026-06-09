from collections.abc import Callable
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent


def run_react_sub_agent(
    *,
    user_query: str,
    llm: Any,
    tools: list[Any],
    system_prompt: str,
    result_extractor: Callable[[dict[str, Any]], tuple[str, list[dict[str, Any]]]],
    recursion_limit: int | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    agent = create_react_agent(llm, tools, prompt=SystemMessage(content=system_prompt))
    invoke_config: dict[str, Any] = {}
    if recursion_limit is not None:
        invoke_config["recursion_limit"] = recursion_limit
    response = agent.invoke({"messages": [HumanMessage(content=user_query)]}, invoke_config)
    return result_extractor(response)

