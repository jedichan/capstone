import argparse
import json
import logging
from pathlib import Path

from .config import (
    CHROMA_RETRIEVAL_LEVELS,
    DEFAULT_ENV_FILE,
    get_graph_chroma_child_collection,
    get_graph_chroma_collection,
    get_graph_chroma_event_collection,
    get_graph_chroma_global_entity_collection,
    get_graph_chroma_namespace,
    get_graph_chroma_parent_collection,
    get_graph_chroma_path,
    get_graph_chroma_retrieval_level,
    get_graph_lancedb_path,
    get_graph_sqlite_db_path,
    persist_env_value,
)
from .chroma_builder import ChromaBuildConfig, ChromaIndexBuilder
from .sqlite_builder import SQLiteBuildConfig, SQLiteDatabaseBuilder


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("manage_graph_db")


def cmd_build(args: argparse.Namespace) -> None:
    db_path = Path(args.db_path).expanduser().resolve() if args.db_path else get_graph_sqlite_db_path()
    seed_files = [Path(x).expanduser().resolve() for x in args.seed_json]
    init_prompt_md_path = (
        Path(args.init_prompt_md).expanduser().resolve() if args.init_prompt_md else None
    )
    init_prompt_json_path = (
        Path(args.init_prompt_json).expanduser().resolve() if args.init_prompt_json else None
    )
    builder = SQLiteDatabaseBuilder(
        SQLiteBuildConfig(
            db_path=db_path,
            reset_existing=bool(args.reset),
            generate_init_prompt=not bool(args.no_init_prompt),
            init_prompt_md_path=init_prompt_md_path or SQLiteBuildConfig.init_prompt_md_path,
            init_prompt_json_path=init_prompt_json_path or SQLiteBuildConfig.init_prompt_json_path,
        )
    )
    result = builder.build(seed_files=seed_files)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def cmd_build_chroma(args: argparse.Namespace) -> None:
    chroma_path = Path(args.chroma_path).expanduser().resolve() if args.chroma_path else get_graph_chroma_path()
    seed_files = [Path(x).expanduser().resolve() for x in args.seed_json]
    child_collection = args.child_collection or get_graph_chroma_child_collection()
    parent_collection = args.parent_collection or get_graph_chroma_parent_collection()
    event_collection = args.event_collection or get_graph_chroma_event_collection()
    global_entity_collection = args.global_entity_collection or get_graph_chroma_global_entity_collection()
    builder = ChromaIndexBuilder(
        ChromaBuildConfig(
            chroma_path=chroma_path,
            child_collection=child_collection,
            parent_collection=parent_collection,
            event_collection=event_collection,
            global_entity_collection=global_entity_collection,
            reset_existing=bool(args.reset),
        )
    )
    result = builder.build(seed_files=seed_files)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def cmd_switch(args: argparse.Namespace) -> None:
    env_path = Path(args.env_file).expanduser().resolve() if args.env_file else DEFAULT_ENV_FILE
    if args.sqlite_path:
        sqlite_path = str(Path(args.sqlite_path).expanduser().resolve())
        persist_env_value("AGENT_SQLITE_DB_PATH", sqlite_path, env_file=env_path)
        logger.info("Updated AGENT_SQLITE_DB_PATH=%s", sqlite_path)
    if args.lancedb_path:
        lancedb_path = str(Path(args.lancedb_path).expanduser().resolve())
        persist_env_value("AGENT_LANCEDB_PATH", lancedb_path, env_file=env_path)
        logger.info("Updated AGENT_LANCEDB_PATH=%s", lancedb_path)
    if args.chroma_path:
        chroma_path = str(Path(args.chroma_path).expanduser().resolve())
        persist_env_value("AGENT_CHROMA_PATH", chroma_path, env_file=env_path)
        logger.info("Updated AGENT_CHROMA_PATH=%s", chroma_path)
    if args.chroma_namespace:
        namespace = args.chroma_namespace.strip()
        if not namespace:
            raise SystemExit("--chroma-namespace cannot be blank")
        persist_env_value("AGENT_CHROMA_NAMESPACE", namespace, env_file=env_path)
        logger.info("Updated AGENT_CHROMA_NAMESPACE=%s", namespace)
    if args.chroma_collection:
        persist_env_value("AGENT_CHROMA_COLLECTION", args.chroma_collection, env_file=env_path)
        logger.info("Updated AGENT_CHROMA_COLLECTION=%s", args.chroma_collection)
    if args.chroma_child_collection:
        persist_env_value("AGENT_CHROMA_CHILD_COLLECTION", args.chroma_child_collection, env_file=env_path)
        logger.info("Updated AGENT_CHROMA_CHILD_COLLECTION=%s", args.chroma_child_collection)
    if args.chroma_parent_collection:
        persist_env_value("AGENT_CHROMA_PARENT_COLLECTION", args.chroma_parent_collection, env_file=env_path)
        logger.info("Updated AGENT_CHROMA_PARENT_COLLECTION=%s", args.chroma_parent_collection)
    if args.chroma_event_collection:
        persist_env_value("AGENT_CHROMA_EVENT_COLLECTION", args.chroma_event_collection, env_file=env_path)
        logger.info("Updated AGENT_CHROMA_EVENT_COLLECTION=%s", args.chroma_event_collection)
    if args.chroma_global_entity_collection:
        persist_env_value("AGENT_CHROMA_GLOBAL_ENTITY_COLLECTION", args.chroma_global_entity_collection, env_file=env_path)
        logger.info("Updated AGENT_CHROMA_GLOBAL_ENTITY_COLLECTION=%s", args.chroma_global_entity_collection)
    if args.chroma_retrieval_level:
        level = args.chroma_retrieval_level.strip().lower()
        if level not in CHROMA_RETRIEVAL_LEVELS:
            raise SystemExit(
                f"Invalid --chroma-retrieval-level={level!r}; expected one of {sorted(CHROMA_RETRIEVAL_LEVELS)}"
            )
        persist_env_value("AGENT_CHROMA_RETRIEVAL_LEVEL", level, env_file=env_path)
        logger.info("Updated AGENT_CHROMA_RETRIEVAL_LEVEL=%s", level)

    print(
        json.dumps(
            {
                "env_file": str(env_path),
                "effective_sqlite_path": str(get_graph_sqlite_db_path()),
                "effective_lancedb_path": str(get_graph_lancedb_path()),
                "effective_chroma_path": str(get_graph_chroma_path()),
                "effective_chroma_namespace": get_graph_chroma_namespace(),
                "effective_chroma_collection": get_graph_chroma_collection(),
                "effective_chroma_child_collection": get_graph_chroma_child_collection(),
                "effective_chroma_parent_collection": get_graph_chroma_parent_collection(),
                "effective_chroma_event_collection": get_graph_chroma_event_collection(),
                "effective_chroma_global_entity_collection": get_graph_chroma_global_entity_collection(),
                "effective_chroma_retrieval_level": get_graph_chroma_retrieval_level(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build/switch databases used by agent graph")
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="Create sqlite schema, indexes, and seed rows")
    p_build.add_argument("--db-path", type=str, default="", help="Target sqlite db path")
    p_build.add_argument("--seed-json", nargs="*", default=[], help="Seed json file(s)")
    p_build.add_argument("--reset", action="store_true", help="Delete existing db before build")
    p_build.add_argument("--no-init-prompt", action="store_true", help="Disable init prompt generation")
    p_build.add_argument("--init-prompt-md", type=str, default="", help="Output markdown path for init prompt")
    p_build.add_argument("--init-prompt-json", type=str, default="", help="Output json path for init prompt profile")
    p_build.set_defaults(func=cmd_build)

    p_build_chroma = sub.add_parser("build-chroma", help="Create Chroma parent/child/event indexes from seed json")
    p_build_chroma.add_argument("--chroma-path", type=str, default="", help="Target Chroma persistent path")
    p_build_chroma.add_argument("--seed-json", nargs="*", default=[], help="Seed json file(s)")
    p_build_chroma.add_argument("--child-collection", type=str, default="", help="Child collection name")
    p_build_chroma.add_argument("--parent-collection", type=str, default="", help="Parent collection name")
    p_build_chroma.add_argument("--event-collection", type=str, default="", help="Event-level collection name")
    p_build_chroma.add_argument(
        "--global-entity-collection", type=str, default="",
        help="Cross-camera global-entity collection name (default: {namespace}_global_entities)",
    )
    p_build_chroma.add_argument("--reset", action="store_true", help="Delete existing Chroma collections before build")
    p_build_chroma.set_defaults(func=cmd_build_chroma)

    p_switch = sub.add_parser("switch", help="One-click switch db path for graph via .env")
    p_switch.add_argument("--sqlite-path", type=str, default="", help="New sqlite db path")
    p_switch.add_argument("--lancedb-path", type=str, default="", help="New LanceDB path")
    p_switch.add_argument("--chroma-path", type=str, default="", help="New Chroma persistent path")
    p_switch.add_argument(
        "--chroma-namespace",
        type=str,
        default="",
        help=(
            "New Chroma dataset namespace (default 'basketball'). "
            "Automatically derives {namespace}_tracks, {namespace}_tracks_parent and {namespace}_events."
        ),
    )
    p_switch.add_argument("--chroma-collection", type=str, default="", help="New Chroma collection name")
    p_switch.add_argument("--chroma-child-collection", type=str, default="", help="New Chroma child collection name")
    p_switch.add_argument("--chroma-parent-collection", type=str, default="", help="New Chroma parent collection name")
    p_switch.add_argument("--chroma-event-collection", type=str, default="", help="New Chroma event-level collection name")
    p_switch.add_argument(
        "--chroma-global-entity-collection", type=str, default="",
        help="New Chroma cross-camera global-entity collection name",
    )
    p_switch.add_argument(
        "--chroma-retrieval-level",
        type=str,
        default="",
        help="Default retrieval collection level: child | event",
    )
    p_switch.add_argument("--env-file", type=str, default="", help="Custom .env file path")
    p_switch.set_defaults(func=cmd_switch)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
