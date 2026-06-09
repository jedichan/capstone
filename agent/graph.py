from langchain_core.messages import HumanMessage

from core.runtime import build_default_llm, load_env, load_init_prompt
from graph_builder import build_graph


def create_graph():
    load_env()
    llm = build_default_llm()
    init_prompt_text = load_init_prompt()
    return build_graph(llm, init_prompt_text=init_prompt_text)


graph = create_graph()


if __name__ == "__main__":
    local_graph = create_graph()
    config = {"configurable": {"thread_id": "1", "user_id": "Lance"}}
    for chunk in local_graph.stream({"messages": [HumanMessage(content="车进入镜头")]}, config, stream_mode="values"):
        if chunk.get("messages"):
            chunk["messages"][-1].pretty_print()
    final_state = local_graph.get_state(config)
    print("final_answer:", final_state.values.get("final_answer"))
