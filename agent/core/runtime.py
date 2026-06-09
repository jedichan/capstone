import os
from pathlib import Path

from langchain_openai import ChatOpenAI


def load_env(project_root: Path | None = None) -> None:
    root = project_root or Path(__file__).resolve().parents[2]
    env_file = root / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    # DashScope is exposed via an OpenAI-compatible endpoint, so we may alias
    # it onto the ``OPENAI_*`` env vars for code paths that hard-code the
    # OpenAI SDK -- but ONLY when both halves (key + base_url) are missing.
    # Aliasing just the URL while a real ``OPENAI_API_KEY`` is set sends real
    # OpenAI-keyed requests to the DashScope endpoint and returns 401 (the
    # exact symptom that masked the hybrid retrieval branch in P1-2 evals).
    openai_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    openai_base = (os.getenv("OPENAI_BASE_URL") or "").strip()
    dashscope_key = (os.getenv("DASHSCOPE_API_KEY") or "").strip()
    dashscope_url = (os.getenv("DASHSCOPE_URL") or "").strip()
    if not openai_key and not openai_base and dashscope_key and dashscope_url:
        os.environ["OPENAI_API_KEY"] = dashscope_key
        os.environ["OPENAI_BASE_URL"] = dashscope_url
    os.environ.setdefault("AGENT_USE_LLAMAINDEX_SQL", "1")
    os.environ.setdefault("AGENT_USE_LLAMAINDEX_VECTOR", "1")
    os.environ.setdefault("AGENT_ENABLE_RERANK", "1")
    os.environ.setdefault("AGENT_RERANK_MODEL", "jinaai/jina-reranker-v2-base-multilingual")
    os.environ.setdefault("AGENT_EMBEDDING_PROVIDER", "openai")
    os.environ.setdefault("AGENT_EMBEDDING_MODEL", "text-embedding-3-large")
    os.environ.setdefault("AGENT_CHAT_MODEL", os.getenv("OPENAI_CHAT_MODEL", "gpt-5.4"))
    # The LlamaIndex NL2SQL LLM previously preferred DashScope unconditionally;
    # default it to OpenAI when an OpenAI key is available so eval runs do not
    # need a working DashScope key. Override with AGENT_LLAMAINDEX_LLM_PROVIDER.
    os.environ.setdefault(
        "AGENT_LLAMAINDEX_LLM_PROVIDER",
        "openai" if os.getenv("OPENAI_API_KEY") else "dashscope",
    )
    os.environ.setdefault("AGENT_LLAMAINDEX_LLM_MODEL", os.getenv("AGENT_CHAT_MODEL", "gpt-5.4"))
    os.environ.setdefault("AGENT_ENABLE_SHORT_TERM_MEMORY", "0")
    os.environ.setdefault("AGENT_MEMORY_MAX_TURNS", "5")


def build_default_llm() -> ChatOpenAI:
    kwargs = {
        "model_name": os.getenv("AGENT_CHAT_MODEL", os.getenv("OPENAI_CHAT_MODEL", "gpt-5.4")),
        "temperature": 0.0,
        "api_key": os.getenv("OPENAI_API_KEY"),
    }
    base_url = (os.getenv("OPENAI_BASE_URL") or "").strip()
    if base_url:
        kwargs["base_url"] = base_url
    return ChatOpenAI(**kwargs)


def load_init_prompt(project_root: Path | None = None) -> str:
    root = project_root or Path(__file__).resolve().parents[2]
    prompt_path = root / "agent" / "init" / "agent_init_prompt.md"
    if not prompt_path.exists():
        return ""
    return prompt_path.read_text(encoding="utf-8").strip()
