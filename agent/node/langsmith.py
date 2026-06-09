import os


def enable_langsmith(project_name: str = "capstone-agent") -> bool:
    api_key = os.getenv("LANGSMITH_API_KEY", "").strip()
    if not api_key:
        os.environ["LANGSMITH_TRACING"] = "false"
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        return False
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = project_name
    if not os.getenv("LANGCHAIN_ENDPOINT"):
        os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    return True


if __name__ == "__main__":
    enabled = enable_langsmith("capstone-agent")
    print("langsmith_enabled:", enabled)
