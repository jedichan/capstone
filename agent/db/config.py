import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SQLITE_PATH = PROJECT_ROOT / "data" / "SQLite" / "episodic_events.sqlite"
DEFAULT_LANCEDB_PATH = PROJECT_ROOT / "data" / "lancedb"
DEFAULT_CHROMA_PATH = PROJECT_ROOT / "data" / "chroma" / "basketball_tracks"
DEFAULT_UCFCRIME_SQLITE_PATH = PROJECT_ROOT / "agent" / "test" / "generated" / "datasets" / "ucfcrime_eval.sqlite"

# Dataset-level namespace shared by all three Chroma collections.
# Override at runtime via AGENT_CHROMA_NAMESPACE (for example: ucfcrime, mevid, synthetic).
# Historical default stays "basketball" so existing deployments keep working.
DEFAULT_CHROMA_NAMESPACE = "basketball"
CHROMA_CHILD_SUFFIX = "tracks"
CHROMA_PARENT_SUFFIX = "tracks_parent"
CHROMA_EVENT_SUFFIX = "events"
CHROMA_GLOBAL_ENTITY_SUFFIX = "global_entities"

# Canonical collection names under the default "basketball" namespace.
# Kept as module-level constants so downstream code that imports them directly
# keeps its historical behavior. The runtime getters below honour the namespace
# env variable and should be preferred for any new code path.
DEFAULT_CHROMA_CHILD_COLLECTION = f"{DEFAULT_CHROMA_NAMESPACE}_{CHROMA_CHILD_SUFFIX}"
DEFAULT_CHROMA_PARENT_COLLECTION = f"{DEFAULT_CHROMA_NAMESPACE}_{CHROMA_PARENT_SUFFIX}"
DEFAULT_CHROMA_EVENT_COLLECTION = f"{DEFAULT_CHROMA_NAMESPACE}_{CHROMA_EVENT_SUFFIX}"
DEFAULT_CHROMA_GLOBAL_ENTITY_COLLECTION = f"{DEFAULT_CHROMA_NAMESPACE}_{CHROMA_GLOBAL_ENTITY_SUFFIX}"
DEFAULT_CHROMA_COLLECTION = DEFAULT_CHROMA_CHILD_COLLECTION
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"

# Supported values: "child" (historical default), "event" (new event-level collection).
# Controls which Chroma collection `get_graph_chroma_collection()` returns when no
# explicit `AGENT_CHROMA_COLLECTION` override is set.
CHROMA_RETRIEVAL_LEVELS = {"child", "event"}


def _first_existing_path(candidates: list[Path]) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def get_graph_sqlite_db_path() -> Path:
    raw = os.getenv("AGENT_SQLITE_DB_PATH", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return _first_existing_path(
        [
            DEFAULT_UCFCRIME_SQLITE_PATH,
            DEFAULT_SQLITE_PATH,
        ]
    )


def get_graph_lancedb_path() -> Path:
    raw = os.getenv("AGENT_LANCEDB_PATH", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return DEFAULT_LANCEDB_PATH


def get_graph_chroma_path() -> Path:
    raw = os.getenv("AGENT_CHROMA_PATH", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return DEFAULT_CHROMA_PATH


def get_graph_chroma_namespace() -> str:
    """Return the dataset-level Chroma namespace.

    Resolution order:
    1. AGENT_CHROMA_NAMESPACE environment variable (trimmed, non-empty).
    2. DEFAULT_CHROMA_NAMESPACE ("basketball") so historical deployments keep working.
    """
    raw = os.getenv("AGENT_CHROMA_NAMESPACE", "").strip()
    return raw or DEFAULT_CHROMA_NAMESPACE


def get_graph_chroma_retrieval_level() -> str:
    raw = os.getenv("AGENT_CHROMA_RETRIEVAL_LEVEL", "").strip().lower()
    if raw in CHROMA_RETRIEVAL_LEVELS:
        return raw
    return "child"


def get_graph_chroma_collection() -> str:
    # Priority: explicit full collection name > retrieval-level derived > namespace-derived child.
    raw = os.getenv("AGENT_CHROMA_COLLECTION", "").strip()
    if raw:
        return raw
    level = get_graph_chroma_retrieval_level()
    if level == "event":
        return get_graph_chroma_event_collection()
    return get_graph_chroma_child_collection()


def get_graph_chroma_child_collection() -> str:
    # Priority: explicit full collection name > namespace-derived.
    raw = os.getenv("AGENT_CHROMA_CHILD_COLLECTION", "").strip()
    if raw:
        return raw
    return f"{get_graph_chroma_namespace()}_{CHROMA_CHILD_SUFFIX}"


def get_graph_chroma_parent_collection() -> str:
    raw = os.getenv("AGENT_CHROMA_PARENT_COLLECTION", "").strip()
    if raw:
        return raw
    return f"{get_graph_chroma_namespace()}_{CHROMA_PARENT_SUFFIX}"


def get_graph_chroma_event_collection() -> str:
    raw = os.getenv("AGENT_CHROMA_EVENT_COLLECTION", "").strip()
    if raw:
        return raw
    return f"{get_graph_chroma_namespace()}_{CHROMA_EVENT_SUFFIX}"


CHROMA_VIDEO_SUFFIX = "video"
CHROMA_GLOBAL_ENTITY_SUFFIX = "global_entity"


def get_graph_chroma_global_entity_collection() -> str:
    raw = os.getenv("AGENT_CHROMA_GLOBAL_ENTITY_COLLECTION", "").strip()
    if raw:
        return raw
    return f"{get_graph_chroma_namespace()}_{CHROMA_GLOBAL_ENTITY_SUFFIX}"


def get_graph_chroma_video_collection() -> str:
    raw = os.getenv("AGENT_CHROMA_VIDEO_COLLECTION", "").strip()
    if raw:
        return raw
    return f"{get_graph_chroma_namespace()}_{CHROMA_VIDEO_SUFFIX}"


def get_graph_chroma_global_entity_collection() -> str:
    """Return the cross-camera global entity collection name.

    Resolution order:
    1. AGENT_CHROMA_GLOBAL_ENTITY_COLLECTION env var (explicit override).
    2. Namespace-derived default: ``{namespace}_global_entities``.
    """
    raw = os.getenv("AGENT_CHROMA_GLOBAL_ENTITY_COLLECTION", "").strip()
    if raw:
        return raw
    return f"{get_graph_chroma_namespace()}_{CHROMA_GLOBAL_ENTITY_SUFFIX}"


def persist_env_value(key: str, value: str, env_file: Path | None = None) -> Path:
    target = env_file or DEFAULT_ENV_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.write_text("", encoding="utf-8")

    lines = target.read_text(encoding="utf-8").splitlines()
    replaced = False
    new_lines: list[str] = []
    for line in lines:
        if line.strip().startswith(f"{key}="):
            new_lines.append(f"{key}={value}")
            replaced = True
        else:
            new_lines.append(line)

    if not replaced:
        new_lines.append(f"{key}={value}")

    target.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    return target
