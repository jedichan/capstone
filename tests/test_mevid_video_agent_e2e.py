"""
MEVID video -> agent end-to-end smoke/evaluation runner.

This script stitches together the existing project pieces:

1. MEVID multi-camera pipeline cache
2. vector-flat seed generation for agent indexing
3. temporary SQLite + Chroma build
4. LangGraph agent query run
5. compact JSON/Markdown report

Default mode is cache-first and safe:
    python tests/test_mevid_video_agent_e2e.py --slot 13-50 --limit 8

To generate missing pipeline cache from videos, opt in explicitly:
    python tests/test_mevid_video_agent_e2e.py --slot 13-50 --run-pipeline --reid-device cuda
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
AGENT_DIR = ROOT / "agent"
AGENT_TEST_DIR = AGENT_DIR / "test"

for p in (ROOT, AGENT_DIR, AGENT_TEST_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from scripts.generate_mevid_vector_flat import SLOT_CAMERAS  # noqa: E402


DEFAULT_XLSX = ROOT / "agent" / "test" / "data" / "agent_test_mevid.xlsx"
DEFAULT_SEED_DIR = ROOT / "agent" / "test" / "data" / "events_vector_flat"
DEFAULT_VIDEO_DIR = ROOT / "_data" / "mevid_slots"
DEFAULT_OUTPUT_ROOT = ROOT / "results" / "mevid_agent_e2e"


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _check_environment(run_pipeline: bool) -> dict[str, Any]:
    try:
        from agent.core.runtime import load_env

        load_env(ROOT)
    except Exception as exc:
        # Keep the check readable even when the active Python environment is wrong.
        env_file = ROOT / ".env"
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
        runtime_import_error = str(exc)
    else:
        runtime_import_error = ""
    imports: dict[str, str] = {}
    for mod in ["openpyxl", "chromadb", "langgraph", "langchain_openai"]:
        try:
            importlib.import_module(mod)
            imports[mod] = "ok"
        except Exception as exc:
            imports[mod] = f"missing/error: {exc}"

    if run_pipeline:
        for mod in ["cv2", "torch", "ultralytics", "torchreid"]:
            try:
                importlib.import_module(mod)
                imports[mod] = "ok"
            except Exception as exc:
                imports[mod] = f"missing/error: {exc}"

    env = {
        "DASHSCOPE_API_KEY": bool(os.getenv("DASHSCOPE_API_KEY")),
        "DASHSCOPE_URL": bool(os.getenv("DASHSCOPE_URL")),
        "OPENAI_API_KEY": bool(os.getenv("OPENAI_API_KEY")),
        "OPENAI_BASE_URL": bool(os.getenv("OPENAI_BASE_URL")),
        "AGENT_EMBEDDING_PROVIDER": os.getenv("AGENT_EMBEDDING_PROVIDER", ""),
        "AGENT_LLAMAINDEX_LLM_PROVIDER": os.getenv("AGENT_LLAMAINDEX_LLM_PROVIDER", ""),
    }
    return {"imports": imports, "env": env, "runtime_import_error": runtime_import_error}


def _assert_environment_ready(env_report: dict[str, Any], *, run_pipeline: bool, skip_agent: bool) -> None:
    required = ["openpyxl", "chromadb"]
    if not skip_agent:
        required.extend(["langgraph", "langchain_openai"])
    if run_pipeline:
        required.extend(["cv2", "torch", "ultralytics", "torchreid"])

    imports = env_report.get("imports", {})
    missing = [name for name in required if str(imports.get(name, "")).lower() != "ok"]
    env = env_report.get("env", {})
    key_missing: list[str] = []
    if not env.get("DASHSCOPE_API_KEY") and not env.get("OPENAI_API_KEY"):
        key_missing.append("DASHSCOPE_API_KEY or OPENAI_API_KEY")
    if not skip_agent and (not env.get("DASHSCOPE_API_KEY") or not env.get("DASHSCOPE_URL")):
        key_missing.append("DASHSCOPE_API_KEY and DASHSCOPE_URL for the agent LLM")

    if missing or key_missing:
        details = {
            "missing_imports": missing,
            "missing_env": key_missing,
            "imports": imports,
            "env": env,
            "runtime_import_error": env_report.get("runtime_import_error", ""),
        }
        raise RuntimeError(
            "Environment is not ready for MEVID video+agent e2e. "
            "Use the capstone conda environment and install missing packages if needed.\n"
            + json.dumps(details, ensure_ascii=False, indent=2)
        )


def _slot_video_ids(slot: str) -> set[str]:
    if slot not in SLOT_CAMERAS:
        raise ValueError(f"Unknown slot {slot!r}. Available: {', '.join(SLOT_CAMERAS)}")
    return set(SLOT_CAMERAS[slot].values())


def _infer_category(question: str, expected_label: str = "") -> str:
    q = question.lower()
    label = expected_label.lower()
    if label == "no":
        return "negative"
    if any(token in q for token in ["then appear", "also appear", "same person", "again in camera", "another camera"]):
        return "cross_camera"
    if any(token in q for token in ["exit", "enter", "walk", "move", "appear in camera", "leave"]):
        return "event"
    if any(token in q for token in [
        "wearing", "visible", "with ", "hoodie", "jacket", "coat", "shirt",
        "hat", "bag", "backpack", "color", "colour", "appearance",
    ]):
        return "appearance"
    return "unknown"


def _parse_yes_no(answer: Any) -> str:
    text = str(answer or "").strip().lower()
    text = re.sub(r"^[\s`*_>-]+", "", text)
    if text.startswith("yes") or text.startswith("是") or text.startswith("有"):
        return "yes"
    if text.startswith("no") or text.startswith("否") or text.startswith("没有") or text.startswith("无"):
        return "no"
    return "unknown"


def _stratified_sample(cases: list[dict[str, Any]], limit: int, seed: int) -> list[dict[str, Any]]:
    if limit <= 0 or len(cases) <= limit:
        return cases
    rng = random.Random(seed)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for case in cases:
        key = (str(case.get("_category") or "unknown"), str(case.get("expected_answer_label") or "unknown"))
        groups.setdefault(key, []).append(case)

    selected: list[dict[str, Any]] = []
    total = len(cases)
    for _key, items in sorted(groups.items()):
        k = max(1, round(limit * len(items) / total))
        selected.extend(rng.sample(items, min(k, len(items))))

    if len(selected) > limit:
        rng.shuffle(selected)
        selected = selected[:limit]
    elif len(selected) < limit:
        used = {id(c) for c in selected}
        remainder = [c for c in cases if id(c) not in used]
        rng.shuffle(remainder)
        selected.extend(remainder[: limit - len(selected)])

    selected.sort(key=lambda c: str(c.get("case_id") or ""))
    return selected


def _balanced_category_sample(cases: list[dict[str, Any]], limit: int, seed: int) -> list[dict[str, Any]]:
    if limit <= 0 or len(cases) <= limit:
        return cases
    rng = random.Random(seed)
    preferred = ["appearance", "cross_camera", "event", "negative", "unknown"]
    by_category: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        by_category.setdefault(str(case.get("_category") or "unknown"), []).append(case)
    categories = [cat for cat in preferred if by_category.get(cat)]
    categories.extend(sorted(cat for cat in by_category if cat not in categories))
    if not categories:
        return []

    base = max(1, limit // len(categories))
    selected: list[dict[str, Any]] = []
    for cat in categories:
        items = by_category[cat]
        selected.extend(rng.sample(items, min(base, len(items))))

    if len(selected) < limit:
        used = {id(c) for c in selected}
        remainder = [c for c in cases if id(c) not in used]
        rng.shuffle(remainder)
        selected.extend(remainder[: limit - len(selected)])
    elif len(selected) > limit:
        rng.shuffle(selected)
        selected = selected[:limit]

    selected.sort(key=lambda c: str(c.get("case_id") or ""))
    return selected


def _import_cases(xlsx_path: Path, output_dir: Path, include_sheets: list[str]) -> list[dict[str, Any]]:
    from agent_test_importer import AgentTestDatasetImporter, AgentTestImportConfig

    dataset_dir = output_dir / "dataset"
    report = AgentTestDatasetImporter(
        AgentTestImportConfig(
            xlsx_path=xlsx_path,
            output_dir=dataset_dir,
            sqlite_path=dataset_dir / "agent_test_eval.sqlite",
            normalized_json_path=dataset_dir / "agent_test_normalized.json",
            report_json_path=dataset_dir / "agent_test_import_report.json",
            retrieval_view_path=dataset_dir / "agent_test_retrieval_eval.json",
            e2e_view_path=dataset_dir / "agent_test_e2e_eval.json",
            generation_view_path=dataset_dir / "agent_test_generation_eval.json",
            reset_existing=True,
            include_sheets=include_sheets,
        )
    ).build()
    cases_path = dataset_dir / "agent_test_normalized.json"
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    print(f"[mevid-agent] Imported {len(cases)} cases from {xlsx_path.name}: {report}")
    return cases


def _select_cases(
    cases: list[dict[str, Any]],
    slot: str,
    categories: set[str],
    limit: int,
    sample_mode: str,
    seed: int,
) -> list[dict[str, Any]]:
    slot_ids = _slot_video_ids(slot)
    eligible: list[dict[str, Any]] = []
    for case in cases:
        video_id = str(case.get("video_id") or "").strip()
        if video_id not in slot_ids:
            continue
        metadata = case.get("metadata") if isinstance(case.get("metadata"), dict) else {}
        raw_row = metadata.get("raw_row") if isinstance(metadata.get("raw_row"), list) else []
        category = str(metadata.get("category") or (raw_row[7] if len(raw_row) > 7 else "") or "").strip()
        if not category:
            category = _infer_category(
                str(case.get("question") or ""),
                str(case.get("expected_answer_label") or ""),
            )
        case["_category"] = category or "unknown"
        if categories and case["_category"] not in categories:
            continue
        eligible.append(case)
    if sample_mode == "stratified":
        return _stratified_sample(eligible, limit=limit, seed=seed)
    if sample_mode == "balanced":
        return _balanced_category_sample(eligible, limit=limit, seed=seed)
    if limit > 0:
        return eligible[:limit]
    return eligible


def _ensure_vector_seeds(
    *,
    slot: str,
    seed_dir: Path,
    video_dir: Path,
    run_pipeline: bool,
    reid_device: str,
    force_seed: bool,
    force_pipeline_cache: bool,
    precision_matching: bool,
    appearance_refine: bool,
    force_appearance_refine: bool,
) -> None:
    missing = [
        stem
        for stem in sorted(_slot_video_ids(slot))
        if not (seed_dir / f"{stem}_events_vector_flat.json").exists()
    ]
    if not missing and not force_seed:
        print(f"[mevid-agent] Vector seeds already exist for slot {slot}: {seed_dir}")
        return

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "generate_mevid_vector_flat.py"),
        "--slot",
        slot,
        "--out-dir",
        str(seed_dir),
        "--video-dir",
        str(video_dir.relative_to(ROOT) if video_dir.is_relative_to(ROOT) else video_dir),
    ]
    if run_pipeline:
        cmd.extend(["--run-pipeline", "--reid-device", reid_device])
    if force_pipeline_cache:
        cmd.append("--force-pipeline-cache")
    if precision_matching:
        cmd.append("--precision-matching")
    else:
        cmd.append("--legacy-matching")
    if appearance_refine:
        cmd.append("--appearance-refine")
    if force_appearance_refine:
        cmd.append("--force-appearance-refine")
    if force_seed:
        cmd.append("--force")

    print("[mevid-agent] Generating vector-flat seeds ...")
    print("[mevid-agent] " + " ".join(cmd))
    child_env = os.environ.copy()
    child_env.setdefault("PYTHONIOENCODING", "utf-8")
    subprocess.run(cmd, cwd=str(ROOT), check=True, env=child_env)


def _resolve_seed_files(seed_dir: Path, cases: list[dict[str, Any]]) -> list[Path]:
    video_ids = sorted({str(c.get("video_id") or "").strip() for c in cases if c.get("video_id")})
    missing: list[str] = []
    seed_files: list[Path] = []
    for video_id in video_ids:
        seed_file = seed_dir / f"{video_id}_events_vector_flat.json"
        if seed_file.exists():
            seed_files.append(seed_file)
        else:
            missing.append(video_id)
    if missing:
        raise FileNotFoundError(
            "Missing vector-flat seed files. Run with --run-pipeline if pipeline cache is missing. "
            f"Missing examples: {missing[:8]}"
        )
    return seed_files


def _prepare_databases(
    *,
    output_dir: Path,
    seed_files: list[Path],
    namespace: str,
) -> dict[str, Any]:
    from agent.core.runtime import load_env
    from agent.db.chroma_builder import ChromaBuildConfig, ChromaIndexBuilder
    from agent.db.sqlite_builder import SQLiteBuildConfig, SQLiteDatabaseBuilder
    from langchain_openai import ChatOpenAI

    load_env(ROOT)
    discriminator_llm = ChatOpenAI(
        model_name=os.getenv("MEVID_VIDEO_DISCRIMINATOR_MODEL", "qwen3.6-plus"),
        temperature=0.0,
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url=os.getenv("DASHSCOPE_URL"),
    )
    runtime_dir = output_dir / "runtime"
    sqlite_path = runtime_dir / "mevid_agent_e2e.sqlite"
    chroma_path = runtime_dir / "mevid_agent_e2e_chroma"
    child_collection = f"{namespace}_tracks"
    parent_collection = f"{namespace}_tracks_parent"
    event_collection = f"{namespace}_events"
    video_collection = f"{namespace}_video"

    sqlite_result = SQLiteDatabaseBuilder(
        SQLiteBuildConfig(
            db_path=sqlite_path,
            reset_existing=True,
            generate_init_prompt=False,
        )
    ).build(seed_files=seed_files)

    chroma_result = ChromaIndexBuilder(
        ChromaBuildConfig(
            chroma_path=chroma_path,
            child_collection=child_collection,
            parent_collection=parent_collection,
            event_collection=event_collection,
            video_collection=video_collection,
            reset_existing=True,
        )
    ).build(seed_files=seed_files, llm=discriminator_llm)

    return {
        "sqlite": sqlite_result,
        "chroma": chroma_result,
        "sqlite_path": str(sqlite_path),
        "chroma_path": str(chroma_path),
        "child_collection": child_collection,
        "parent_collection": parent_collection,
        "event_collection": event_collection,
        "video_collection": video_collection,
    }


def _load_agent_graph(db_info: dict[str, Any]):
    from agent.core.runtime import load_env

    load_env(ROOT)
    os.environ["AGENT_SQLITE_DB_PATH"] = str(db_info["sqlite_path"])
    os.environ["AGENT_CHROMA_PATH"] = str(db_info["chroma_path"])
    os.environ["AGENT_CHROMA_COLLECTION"] = str(db_info["child_collection"])
    os.environ["AGENT_CHROMA_CHILD_COLLECTION"] = str(db_info["child_collection"])
    os.environ["AGENT_CHROMA_PARENT_COLLECTION"] = str(db_info["parent_collection"])
    os.environ["AGENT_CHROMA_EVENT_COLLECTION"] = str(db_info["event_collection"])
    os.environ["AGENT_CHROMA_VIDEO_COLLECTION"] = str(db_info["video_collection"])
    os.environ.setdefault("AGENT_CHROMA_RETRIEVAL_LEVEL", "child")

    if "graph" in sys.modules:
        graph_module = importlib.reload(sys.modules["graph"])
    else:
        graph_module = importlib.import_module("graph")
    return graph_module.create_graph()


def _final_rows(state: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("rerank_result", "merged_result", "hybrid_result", "sql_result"):
        rows = state.get(key)
        if isinstance(rows, list):
            return [r for r in rows if isinstance(r, dict)]
    return []


_CAMERA_RE = re.compile(r"\bG\d{3}\b", re.IGNORECASE)
_GLOBAL_PERSON_RE = re.compile(r"\bperson_global_\d+\b", re.IGNORECASE)
_CAMERA_SEQUENCE_RE = re.compile(r"Camera sequence:\s*([^.;\n]+)", re.IGNORECASE)


def _ordered_unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        normalized = item.upper()
        if normalized not in seen:
            seen.add(normalized)
            out.append(normalized)
    return out


def _mentioned_cameras(text: str) -> list[str]:
    return _ordered_unique(_CAMERA_RE.findall(text or ""))


def _row_text_blob(row: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in (
        "video_id",
        "camera_id",
        "event_text",
        "event_text_en",
        "event_summary_en",
        "appearance_notes",
        "object_color",
        "scene_zone",
        "entity_hint",
        "global_entity_id",
    ):
        value = row.get(key)
        if value is not None:
            parts.append(str(value))
    keywords = row.get("keywords")
    if isinstance(keywords, list):
        parts.extend(str(k) for k in keywords)
    elif keywords:
        parts.append(str(keywords))
    return " ".join(parts)


def _row_cameras(row: dict[str, Any]) -> set[str]:
    cameras = set(_mentioned_cameras(_row_text_blob(row)))
    camera_id = row.get("camera_id")
    if camera_id:
        cameras.add(str(camera_id).upper())
    video_id = str(row.get("video_id") or "")
    cameras.update(_mentioned_cameras(video_id))
    return cameras


def _row_globals(row: dict[str, Any]) -> set[str]:
    values = set(g.lower() for g in _GLOBAL_PERSON_RE.findall(_row_text_blob(row)))
    global_id = row.get("global_entity_id")
    if global_id:
        values.update(g.lower() for g in _GLOBAL_PERSON_RE.findall(str(global_id)))
    entity_hint = row.get("entity_hint")
    if entity_hint:
        values.update(g.lower() for g in _GLOBAL_PERSON_RE.findall(str(entity_hint)))
    return values


def _row_time_range(row: dict[str, Any]) -> tuple[float, float] | None:
    try:
        start = float(row.get("start_time"))
        end = float(row.get("end_time"))
    except (TypeError, ValueError):
        return None
    return min(start, end), max(start, end)


def _identity_camera_support(rows: list[dict[str, Any]], cameras: list[str], top_k: int) -> dict[str, Any]:
    required = set(cameras[:2])
    by_global: dict[str, set[str]] = {}
    explicit_pair = False
    for row in rows[:top_k]:
        row_cameras = _row_cameras(row)
        row_globals = _row_globals(row)
        if required <= row_cameras and row_globals:
            explicit_pair = True
        for global_id in row_globals:
            by_global.setdefault(global_id, set()).update(row_cameras)
    matched_globals = sorted(global_id for global_id, cams in by_global.items() if required <= cams)
    return {
        "required_cameras": sorted(required),
        "explicit_pair": explicit_pair,
        "matched_globals": matched_globals,
        "supported": bool(explicit_pair or matched_globals),
    }


def _identity_time_support(
    rows: list[dict[str, Any]],
    cameras: list[str],
    top_k: int,
    *,
    max_gap_sec: float = 240.0,
    overlap_tolerance_sec: float | None = None,
) -> dict[str, Any]:
    if len(cameras) < 2:
        return {"status": "not_applicable"}
    first_camera, second_camera = cameras[0], cameras[1]
    by_global_camera: dict[str, dict[str, list[tuple[float, float]]]] = {}
    for row in rows[:top_k]:
        time_range = _row_time_range(row)
        if time_range is None:
            continue
        row_cameras = _row_cameras(row)
        row_globals = _row_globals(row)
        for global_id in row_globals:
            for camera in row_cameras:
                by_global_camera.setdefault(global_id, {}).setdefault(camera, []).append(time_range)

    checked: list[dict[str, Any]] = []
    for global_id, camera_ranges in by_global_camera.items():
        first_ranges = camera_ranges.get(first_camera, [])
        second_ranges = camera_ranges.get(second_camera, [])
        for first_start, first_end in first_ranges:
            for second_start, second_end in second_ranges:
                overlap = min(first_end, second_end) - max(first_start, second_start)
                if overlap_tolerance_sec is not None and overlap > overlap_tolerance_sec:
                    checked.append(
                        {
                            "global_id": global_id,
                            "status": "simultaneous_overlap",
                            "first_camera": first_camera,
                            "second_camera": second_camera,
                            "overlap_sec": round(overlap, 3),
                        }
                    )
                    continue
                if second_start + max_gap_sec < first_start:
                    checked.append(
                        {
                            "global_id": global_id,
                            "status": "reverse_order",
                            "first_camera": first_camera,
                            "second_camera": second_camera,
                            "gap_sec": round(first_start - second_end, 3),
                        }
                    )
                    continue
                gap = max(0.0, second_start - first_end)
                if gap <= max_gap_sec:
                    return {
                        "status": "supported",
                        "global_id": global_id,
                        "first_camera": first_camera,
                        "second_camera": second_camera,
                        "gap_sec": round(gap, 3),
                    }
                checked.append(
                    {
                        "global_id": global_id,
                        "status": "gap_too_large",
                        "first_camera": first_camera,
                        "second_camera": second_camera,
                        "gap_sec": round(gap, 3),
                    }
                )
    return {"status": "unsupported", "checked": checked[:5]}


_GENERIC_DARK_RE = re.compile(
    r"\b(?:dark|black)\s+(?:clothing|clothes|top|shirt|jacket|coat|hoodie|hood|pants|trousers)\b"
    r"|\b(?:dark|black)\b.*\b(?:clothing|clothes|top|shirt|jacket|coat|hoodie|hood|pants|trousers)\b",
    re.IGNORECASE,
)


def _looks_generic_dark_query(question: str) -> bool:
    q = (question or "").lower()
    if any(marker in q for marker in ("fur", "trim", "yellow", "white", "beige", "brown", "grey", "gray", "logo", "stripe")):
        return False
    return bool(_GENERIC_DARK_RE.search(q))


def _precision_identity_evidence_check(
    question: str,
    rows: list[dict[str, Any]],
    cameras: list[str],
    top_k: int,
) -> dict[str, Any]:
    if len(cameras) < 2 or not _looks_source_target_query(question):
        return {"status": "not_applicable"}

    camera_support = _identity_camera_support(rows, cameras, top_k)
    if not camera_support.get("supported"):
        return {
            "status": "refute",
            "reason": "missing_same_identity_camera_pair",
            "camera_support": camera_support,
        }

    time_support = _identity_time_support(
        rows,
        cameras,
        top_k,
        max_gap_sec=180.0,
        overlap_tolerance_sec=2.0,
    )
    if time_support.get("status") in {"unsupported", "reverse_order", "gap_too_large"}:
        return {
            "status": "refute",
            "reason": "failed_temporal_filter",
            "camera_support": camera_support,
            "time_support": time_support,
        }
    if any(
        item.get("status") == "simultaneous_overlap"
        for item in (time_support.get("checked") or [])
        if isinstance(item, dict)
    ):
        return {
            "status": "refute",
            "reason": "simultaneous_camera_overlap",
            "camera_support": camera_support,
            "time_support": time_support,
        }

    matched_globals = set(camera_support.get("matched_globals") or [])
    if matched_globals:
        global_cameras: dict[str, set[str]] = {global_id: set() for global_id in matched_globals}
        for row in rows[:top_k]:
            row_globals = _row_globals(row)
            for global_id in matched_globals & row_globals:
                global_cameras[global_id].update(_row_cameras(row))
        over_broad = {
            global_id: sorted(cams)
            for global_id, cams in global_cameras.items()
            if len(cams) > 4
        }
        if over_broad:
            return {
                "status": "refute",
                "reason": "over_broad_global_entity",
                "camera_support": camera_support,
                "time_support": time_support,
                "over_broad_globals": over_broad,
            }

    if _looks_generic_dark_query(question) and not camera_support.get("explicit_pair"):
        return {
            "status": "refute",
            "reason": "generic_dark_appearance_without_explicit_pair",
            "camera_support": camera_support,
            "time_support": time_support,
        }

    return {
        "status": "support",
        "reason": "supported_by_precision_identity_checks",
        "camera_support": camera_support,
        "time_support": time_support,
    }


def _camera_sequence_support(rows: list[dict[str, Any]], cameras: list[str], top_k: int) -> dict[str, Any]:
    if len(cameras) < 2:
        return {"status": "not_applicable"}
    first_camera, second_camera = cameras[0], cameras[1]
    checked: list[dict[str, Any]] = []
    for row in rows[:top_k]:
        text = _row_text_blob(row)
        for match in _CAMERA_SEQUENCE_RE.finditer(text):
            sequence = _mentioned_cameras(match.group(1))
            if first_camera not in sequence or second_camera not in sequence:
                continue
            first_idx = sequence.index(first_camera)
            second_idx = sequence.index(second_camera)
            status = "supported" if first_idx <= second_idx else "reverse_order"
            item = {
                "status": status,
                "sequence": sequence,
                "first_camera": first_camera,
                "second_camera": second_camera,
            }
            if status == "supported":
                return item
            checked.append(item)
    return {"status": "unsupported", "checked": checked[:5]}


def _camera_row_counts(rows: list[dict[str, Any]], cameras: list[str], top_k: int) -> dict[str, int]:
    counts = {camera: 0 for camera in cameras[:2]}
    for row in rows[:top_k]:
        row_cameras = _row_cameras(row)
        for camera in counts:
            if camera in row_cameras:
                counts[camera] += 1
    return counts


def _looks_source_target_query(question: str) -> bool:
    q = question.lower()
    return " from camera " in q and " also appear in camera " in q


def _videorag_style_absence_check(
    question: str,
    rows: list[dict[str, Any]],
    cameras: list[str],
    top_k: int,
) -> dict[str, Any]:
    """Conservative evidence check inspired by the VideoRAG baseline.

    It uses only the query/evidence, not the expected label. For explicit
    "from camera A also appear in camera B" claims, separate similar-looking
    rows are not enough: the retrieved evidence should connect the two cameras
    through the same identity or trajectory.
    """
    if len(cameras) < 2 or not _looks_source_target_query(question):
        return {"status": "not_applicable"}

    camera_support = _identity_camera_support(rows, cameras, top_k)
    counts = _camera_row_counts(rows, cameras, top_k)
    if camera_support.get("supported"):
        return {
            "status": "supported",
            "camera_support": camera_support,
            "camera_counts": counts,
        }

    missing = [camera for camera, count in counts.items() if count == 0]
    if missing:
        return {
            "status": "refute",
            "reason": "queried_camera_missing_from_top_evidence",
            "missing_cameras": missing,
            "camera_support": camera_support,
            "camera_counts": counts,
        }

    globals_by_camera: dict[str, set[str]] = {camera: set() for camera in cameras[:2]}
    for row in rows[:top_k]:
        row_globals = _row_globals(row)
        if not row_globals:
            continue
        for camera in _row_cameras(row):
            if camera in globals_by_camera:
                globals_by_camera[camera].update(row_globals)
    if all(globals_by_camera.values()):
        shared = set.intersection(*(globals_by_camera[camera] for camera in cameras[:2]))
        if not shared:
            return {
                "status": "refute",
                "reason": "different_identity_sets_across_queried_cameras",
                "camera_support": camera_support,
                "camera_counts": counts,
                "globals_by_camera": {k: sorted(v) for k, v in globals_by_camera.items()},
            }

    return {
        "status": "insufficient",
        "reason": "no_same_identity_pair_but_not_enough_for_rule_refute",
        "camera_support": camera_support,
        "camera_counts": counts,
    }


def _llm_identity_verdict(
    question: str,
    rows: list[dict[str, Any]],
    top_k: int,
    *,
    strict_pair: bool = False,
) -> dict[str, Any]:
    from openai import OpenAI

    api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {"status": "insufficient", "reason": "missing_api_key"}

    evidence_lines: list[str] = []
    for idx, row in enumerate(rows[:top_k], start=1):
        text = _row_text_blob(row)
        text = re.sub(r"\s+", " ", text).strip()
        evidence_lines.append(f"{idx}. {text[:1200]}")

    strict_rules = ""
    if strict_pair:
        strict_rules = (
            "- For questions phrased as 'from camera A also appear in camera B', support requires an explicit same-person link across A and B. "
            "Separate rows showing similar-looking people in A and B are insufficient; return refute with camera_pair_absent unless the evidence links them by the same global identity or trajectory.\n"
        )

    prompt = (
        "You are an evidence verifier for multi-camera surveillance QA.\n"
        "Decide whether the retrieved evidence supports the exact claim in the question.\n"
        "Use only the evidence below. Do not guess from general similarity.\n\n"
        "Rules:\n"
        "- The claim is supported only if the evidence links the queried cameras to the same person/global identity.\n"
        f"{strict_rules}"
        "- Treat appearance descriptions as soft evidence: broad phrases like dark clothing, coat, jacket, top, or shirt can be compatible unless there is a very clear contradiction.\n"
        "- Treat camera order as soft unless the evidence explicitly says the queried camera pair is impossible.\n"
        "- Return refute only for strong contradictions: the same identity is clearly not seen in the queried camera, the queried camera pair is absent, or the evidence clearly links a different identity.\n"
        "- If evidence is ambiguous, appearance is missing, or order is uncertain, return insufficient rather than refute.\n"
        "- Return compact JSON only: {\"verdict\":\"support|refute|insufficient\",\"refute_type\":\"none|camera_pair_absent|different_identity|appearance_contradiction|reverse_order|other\",\"reason\":\"...\"}.\n\n"
        f"Question: {question}\n\n"
        "Evidence:\n" + "\n".join(evidence_lines)
    )
    try:
        client = OpenAI(
            api_key=api_key,
            base_url=os.getenv("DASHSCOPE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        )
        resp = client.chat.completions.create(
            model=os.getenv("DASHSCOPE_CHAT_MODEL", "qwen3.6-plus"),
            messages=[
                {"role": "system", "content": "Return JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        raw = resp.choices[0].message.content or ""
        start = raw.find("{")
        end = raw.rfind("}")
        payload = json.loads(raw[start : end + 1]) if start >= 0 and end > start else {}
        verdict = str(payload.get("verdict") or "").strip().lower()
        if verdict not in {"support", "refute", "insufficient"}:
            verdict = "insufficient"
        return {
            "status": verdict,
            "refute_type": str(payload.get("refute_type") or "none").strip().lower(),
            "reason": str(payload.get("reason") or "").strip()[:500],
            "raw": raw[:1000],
        }
    except Exception as exc:
        return {"status": "insufficient", "reason": f"verifier_error: {exc}"}


def _is_strong_llm_refute(verdict: dict[str, Any], cameras: list[str]) -> bool:
    if verdict.get("status") != "refute":
        return False
    refute_type = str(verdict.get("refute_type") or "").strip().lower()
    if refute_type in {"camera_pair_absent", "different_identity"}:
        return True
    if refute_type in {"appearance_contradiction", "reverse_order"}:
        return False

    reason = str(verdict.get("reason") or "").lower()
    for camera in cameras[:2]:
        cam = camera.lower()
        strong_patterns = [
            f"not in camera {cam}",
            f"not seen in camera {cam}",
            f"not appear in camera {cam}",
            f"does not appear in camera {cam}",
            f"does not include camera {cam}",
            f"but not in camera {cam}",
            f"skipping {cam}",
        ]
        if any(pattern in reason for pattern in strong_patterns):
            return True
    if "different identity" in reason or "different person" in reason:
        return True
    if "appearance" in reason or "sequence" in reason or "reverse" in reason or "order" in reason:
        return False
    return False


def _apply_negative_guard(
    *,
    case: dict[str, Any],
    question: str,
    predicted_label: str,
    rows: list[dict[str, Any]],
    top_k: int,
    mode: str = "conservative",
) -> tuple[str, dict[str, Any]]:
    looks_cross_camera = len(_mentioned_cameras(question)) >= 2 and (
        "also appear" in question.lower()
        or "again" in question.lower()
        or "same person" in question.lower()
        or "then appear" in question.lower()
    )
    if predicted_label != "yes" or not looks_cross_camera:
        return predicted_label, {"applied": False, "reason": "not_applicable"}

    cameras = _mentioned_cameras(question)
    if mode == "precision":
        precision_check = _precision_identity_evidence_check(question, rows, cameras, top_k)
        if precision_check.get("status") == "refute":
            return "no", {
                "applied": True,
                "mode": mode,
                "reason": precision_check.get("reason"),
                "cameras": cameras[:2],
                "precision_check": precision_check,
            }
        return predicted_label, {
            "applied": False,
            "mode": mode,
            "reason": "precision_check_support_or_not_applicable",
            "cameras": cameras[:2],
            "precision_check": precision_check,
        }

    if mode in {"llm", "llm_videorag"}:
        structured_check = _videorag_style_absence_check(question, rows, cameras, top_k)
        if mode == "llm_videorag" and structured_check.get("status") == "refute":
            return "no", {
                "applied": True,
                "mode": mode,
                "reason": "videorag_style_absence_refute",
                "cameras": cameras[:2],
                "structured_check": structured_check,
            }

        verdict = _llm_identity_verdict(
            question,
            rows,
            top_k,
            strict_pair=(mode == "llm_videorag"),
        )
        if verdict.get("status") == "support":
            return predicted_label, {
                "applied": False,
                "mode": mode,
                "reason": "llm_verifier_support",
                "cameras": cameras[:2],
                "structured_check": structured_check,
                "verdict": verdict,
            }
        if not _is_strong_llm_refute(verdict, cameras):
            return predicted_label, {
                "applied": False,
                "mode": mode,
                "reason": "llm_verifier_not_strong_refute",
                "cameras": cameras[:2],
                "structured_check": structured_check,
                "verdict": verdict,
            }
        return "no", {
            "applied": True,
            "mode": mode,
            "reason": "llm_verifier_strong_refute",
            "cameras": cameras[:2],
            "structured_check": structured_check,
            "verdict": verdict,
        }

    if mode == "strict":
        sequence_support = _camera_sequence_support(rows, cameras, top_k)
        if sequence_support.get("status") == "supported":
            return predicted_label, {
                "applied": False,
                "mode": mode,
                "reason": "supported_by_camera_sequence",
                "cameras": cameras[:2],
                "sequence_support": sequence_support,
            }
        return "no", {
            "applied": True,
            "mode": mode,
            "reason": "failed_camera_sequence_filter",
            "cameras": cameras[:2],
            "sequence_support": sequence_support,
        }

    camera_support = _identity_camera_support(rows, cameras, top_k)
    if not camera_support.get("supported"):
        return "no", {
            "applied": True,
            "mode": mode,
            "reason": "missing_same_identity_camera_pair",
            "cameras": cameras[:2],
            "camera_support": camera_support,
        }

    time_support = _identity_time_support(rows, cameras, top_k)
    if time_support.get("status") in {"unsupported", "reverse_order", "gap_too_large"}:
        return "no", {
            "applied": True,
            "mode": mode,
            "reason": "failed_time_filter",
            "cameras": cameras[:2],
            "camera_support": camera_support,
            "time_support": time_support,
        }

    return predicted_label, {
        "applied": False,
        "mode": mode,
        "reason": "supported_by_camera_time_evidence",
        "cameras": cameras[:2],
        "camera_support": camera_support,
        "time_support": time_support,
    }


def _run_agent_cases(
    graph: Any,
    cases: list[dict[str, Any]],
    top_k: int,
    *,
    negative_guard: bool = False,
    negative_guard_mode: str = "conservative",
) -> list[dict[str, Any]]:
    from langchain_core.messages import HumanMessage

    results: list[dict[str, Any]] = []
    for idx, case in enumerate(cases, start=1):
        question = str(case.get("question") or "").strip()
        print(f"[mevid-agent] Case {idx}/{len(cases)} {case.get('case_id')}: {question}")
        config = {"configurable": {"thread_id": f"mevid-agent-e2e-{idx}", "user_id": "mevid-agent-e2e"}}
        last_state: dict[str, Any] = {}
        node_trace: list[str] = []
        error = None
        t0 = time.perf_counter()
        try:
            for chunk in graph.stream({"messages": [HumanMessage(content=question)]}, config, stream_mode="values"):
                last_state = chunk
                current_node = chunk.get("current_node")
                if current_node and (not node_trace or node_trace[-1] != current_node):
                    node_trace.append(str(current_node))
        except Exception as exc:
            error = str(exc)

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        rows = _final_rows(last_state)
        top_rows = rows[:top_k]
        top_video_ids = [str(r.get("video_id") or "").strip() for r in top_rows if r.get("video_id")]
        expected_video_id = str(case.get("video_id") or "").strip()
        expected_label = str(case.get("expected_answer_label") or "").strip().lower()
        raw_predicted_label = _parse_yes_no(last_state.get("final_answer"))
        predicted_label = raw_predicted_label
        guard_result: dict[str, Any] = {"enabled": negative_guard, "applied": False}
        if negative_guard:
            predicted_label, guard_result = _apply_negative_guard(
                case=case,
                question=question,
                predicted_label=raw_predicted_label,
                rows=top_rows,
                top_k=top_k,
                mode=negative_guard_mode,
            )
            guard_result["enabled"] = True
        answer_correct = expected_label in {"yes", "no"} and predicted_label == expected_label
        raw_answer_correct = expected_label in {"yes", "no"} and raw_predicted_label == expected_label
        result = {
            "case_id": case.get("case_id"),
            "category": case.get("_category"),
            "video_id": expected_video_id,
            "question": question,
            "expected_answer_label": expected_label,
            "raw_predicted_answer_label": raw_predicted_label,
            "raw_answer_correct": raw_answer_correct,
            "predicted_answer_label": predicted_label,
            "answer_correct": answer_correct,
            "negative_guard": guard_result,
            "expected_time_raw": case.get("expected_time_raw"),
            "elapsed_ms": elapsed_ms,
            "error": error,
            "node_trace": node_trace,
            "route_mode": ((last_state.get("tool_choice") or {}).get("mode") if isinstance(last_state.get("tool_choice"), dict) else None),
            "answer_type": last_state.get("answer_type"),
            "final_answer": last_state.get("final_answer"),
            "top_video_ids": top_video_ids,
            "top_hit": expected_video_id in top_video_ids,
            "top_rows": [
                {
                    "video_id": r.get("video_id"),
                    "start_time": r.get("start_time"),
                    "end_time": r.get("end_time"),
                    "event_text": r.get("event_text") or r.get("event_text_en") or r.get("event_summary_en"),
                    "object_type": r.get("object_type"),
                    "object_color": r.get("object_color"),
                    "scene_zone": r.get("scene_zone"),
                    "entity_hint": r.get("entity_hint"),
                }
                for r in top_rows
            ],
        }
        results.append(result)
    return results


def _build_summary(
    *,
    args: argparse.Namespace,
    env_report: dict[str, Any],
    db_info: dict[str, Any] | None,
    selected_cases: list[dict[str, Any]],
    case_results: list[dict[str, Any]],
) -> dict[str, Any]:
    success = [r for r in case_results if not r.get("error")]
    answer_scored = [r for r in case_results if str(r.get("expected_answer_label") or "") in {"yes", "no"}]
    answer_correct = [r for r in answer_scored if r.get("answer_correct")]
    raw_answer_correct = [r for r in answer_scored if r.get("raw_answer_correct")]
    guard_applied = [r for r in case_results if (r.get("negative_guard") or {}).get("applied")]
    by_category: dict[str, dict[str, Any]] = {}
    for category in sorted({str(r.get("category") or "unknown") for r in case_results}):
        items = [r for r in case_results if str(r.get("category") or "unknown") == category]
        scored_items = [r for r in items if str(r.get("expected_answer_label") or "") in {"yes", "no"}]
        by_category[category] = {
            "count": len(items),
            "answer_accuracy": round(
                sum(1 for r in scored_items if r.get("answer_correct")) / max(len(scored_items), 1),
                4,
            ),
            "raw_answer_accuracy": round(
                sum(1 for r in scored_items if r.get("raw_answer_correct")) / max(len(scored_items), 1),
                4,
            ),
            "top_hit_rate": round(
                sum(1 for r in items if r.get("top_hit")) / max(len(items), 1),
                4,
            ),
            "negative_guard_applied_count": sum(
                1 for r in items if (r.get("negative_guard") or {}).get("applied")
            ),
        }
    return {
        "timestamp": _now_stamp(),
        "slot": args.slot,
        "case_count": len(selected_cases),
        "run_count": len(case_results),
        "success_count": len(success),
        "error_count": len(case_results) - len(success),
        "top_hit_rate": round(
            sum(1 for r in case_results if r.get("top_hit")) / max(len(case_results), 1),
            4,
        ),
        "answer_accuracy": round(len(answer_correct) / max(len(answer_scored), 1), 4),
        "answer_correct_count": len(answer_correct),
        "answer_scored_count": len(answer_scored),
        "raw_answer_accuracy": round(len(raw_answer_correct) / max(len(answer_scored), 1), 4),
        "raw_answer_correct_count": len(raw_answer_correct),
        "negative_guard_enabled": bool(getattr(args, "negative_guard", False)),
        "negative_guard_applied_count": len(guard_applied),
        "category_breakdown": by_category,
        "environment": env_report,
        "database": db_info,
        "args": vars(args),
    }


def _write_markdown(path: Path, summary: dict[str, Any], cases: list[dict[str, Any]]) -> None:
    lines = [
        "# MEVID Video + Agent E2E Report",
        "",
        f"- Slot: `{summary['slot']}`",
        f"- Cases: `{summary['run_count']}`",
        f"- Success: `{summary['success_count']}`",
        f"- Errors: `{summary['error_count']}`",
        f"- Top-hit rate: `{summary['top_hit_rate']}`",
        f"- Answer accuracy: `{summary.get('answer_accuracy')}` "
        f"({summary.get('answer_correct_count')}/{summary.get('answer_scored_count')})",
        f"- Raw answer accuracy: `{summary.get('raw_answer_accuracy')}` "
        f"({summary.get('raw_answer_correct_count')}/{summary.get('answer_scored_count')})",
        f"- Negative guard: enabled=`{summary.get('negative_guard_enabled')}`, "
        f"applied=`{summary.get('negative_guard_applied_count')}`",
        "",
        "## Category Breakdown",
        "",
    ]
    for category, stats in (summary.get("category_breakdown") or {}).items():
        lines.append(
            f"- `{category}`: count={stats.get('count')}, "
            f"answer_accuracy={stats.get('answer_accuracy')}, "
            f"raw_answer_accuracy={stats.get('raw_answer_accuracy')}, "
            f"top_hit_rate={stats.get('top_hit_rate')}"
            f", guard_applied={stats.get('negative_guard_applied_count')}"
        )
    lines.extend([
        "",
        "## Cases",
    ])
    for case in cases:
        lines.extend(
            [
                f"### {case.get('case_id')}",
                f"- Category: `{case.get('category')}`",
                f"- Video: `{case.get('video_id')}`",
                f"- Question: {case.get('question')}",
                f"- Error: `{case.get('error')}`",
                f"- Route: `{case.get('route_mode')}`",
                f"- Top hit: `{case.get('top_hit')}`",
                f"- Expected/Predicted: `{case.get('expected_answer_label')}` / "
                f"`{case.get('predicted_answer_label')}` "
                f"(correct={case.get('answer_correct')})",
                f"- Top videos: `{case.get('top_video_ids')}`",
                f"- Answer: {case.get('final_answer')}",
                "",
            ]
        )
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MEVID video-cache + agent end-to-end test")
    parser.add_argument("--slot", default="13-50", choices=sorted(SLOT_CAMERAS), help="MEVID slot to test")
    parser.add_argument("--limit", type=int, default=8, help="Max cases to run; 0 = all selected")
    parser.add_argument("--sample-mode", choices=["balanced", "stratified", "first"], default="balanced",
                        help="Case selection mode. balanced gives each inferred category similar weight")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for stratified sampling")
    parser.add_argument("--top-k", type=int, default=5, help="Top retrieved rows to inspect")
    parser.add_argument("--xlsx-path", default=str(DEFAULT_XLSX), help="MEVID agent test xlsx")
    parser.add_argument("--seed-dir", default=str(DEFAULT_SEED_DIR), help="Directory for *_events_vector_flat.json")
    parser.add_argument("--video-dir", default=str(DEFAULT_VIDEO_DIR), help="MEVID video directory")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Report/output root")
    parser.add_argument("--include-sheets", nargs="*", default=["Part1", "Part4"], help="XLSX sheets to import")
    parser.add_argument("--categories", nargs="*", default=[], help="Optional categories to include")
    parser.add_argument("--run-pipeline", action="store_true", help="Run YOLO+ReID if pipeline cache is missing")
    parser.add_argument("--reid-device", default="cpu", help="cpu | cuda when --run-pipeline is used")
    parser.add_argument("--force-seed", action="store_true", help="Regenerate vector-flat seed files")
    parser.add_argument("--force-pipeline-cache", action="store_true", help="Re-run YOLO+ReID and replace the slot pipeline cache")
    parser.add_argument("--precision-matching", action="store_true", help="Use precision-focused cross-camera matching when generating seeds (default)")
    parser.add_argument("--legacy-matching", action="store_true", help="Use older looser cross-camera matching when generating seeds")
    parser.add_argument("--appearance-refine", action="store_true", help="Run/load crop-based appearance refinement while generating seeds")
    parser.add_argument("--force-appearance-refine", action="store_true", help="Rebuild crop-based appearance refinement cache")
    parser.add_argument("--skip-agent", action="store_true", help="Only check env, import cases, generate seeds, build DB")
    parser.add_argument(
        "--identity-verifier",
        dest="negative_guard",
        action="store_true",
        default=True,
        help="Apply cross-camera identity consistency verification to yes predictions (default)",
    )
    parser.add_argument(
        "--negative-guard",
        dest="negative_guard",
        action="store_true",
        help="Compatibility alias for --identity-verifier",
    )
    parser.add_argument(
        "--disable-identity-verifier",
        dest="negative_guard",
        action="store_false",
        help="Disable cross-camera identity consistency verification",
    )
    parser.add_argument(
        "--negative-guard-mode",
        choices=["conservative", "strict", "llm", "llm_videorag", "precision"],
        default="precision",
        help="conservative uses camera/time rules; strict uses camera-sequence rules; llm uses an evidence verifier; llm_videorag adds stricter same-identity evidence; precision adds temporal/generic-appearance overmerge checks",
    )
    parser.add_argument("--namespace", default="", help="Chroma namespace; default mevid_e2e_<timestamp>")
    parser.add_argument(
        "--embedding-provider",
        default="dashscope",
        choices=["dashscope", "openai"],
        help="Embedding provider for Chroma build; MEVID default is dashscope",
    )
    parser.add_argument(
        "--embedding-model",
        default="text-embedding-v3",
        help="Embedding model for Chroma build",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.legacy_matching:
        args.precision_matching = False
    else:
        args.precision_matching = True
    if args.embedding_provider:
        os.environ["AGENT_EMBEDDING_PROVIDER"] = args.embedding_provider
    if args.embedding_model:
        os.environ["AGENT_EMBEDDING_MODEL"] = args.embedding_model
    stamp = _now_stamp()
    output_dir = Path(args.output_root).expanduser().resolve() / f"{args.slot}_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    xlsx_path = Path(args.xlsx_path).expanduser().resolve()
    seed_dir = Path(args.seed_dir).expanduser().resolve()
    video_dir = Path(args.video_dir).expanduser().resolve()
    namespace = args.namespace.strip() or f"mevid_e2e_{stamp}"

    env_report = _check_environment(run_pipeline=args.run_pipeline)
    _assert_environment_ready(env_report, run_pipeline=args.run_pipeline, skip_agent=args.skip_agent)
    cases = _import_cases(xlsx_path, output_dir, include_sheets=args.include_sheets)
    selected = _select_cases(
        cases,
        slot=args.slot,
        categories=set(args.categories),
        limit=args.limit,
        sample_mode=args.sample_mode,
        seed=args.seed,
    )
    if not selected:
        raise RuntimeError(f"No cases selected for slot={args.slot}, categories={args.categories}")
    print(f"[mevid-agent] Selected {len(selected)} cases for slot {args.slot}")

    _ensure_vector_seeds(
        slot=args.slot,
        seed_dir=seed_dir,
        video_dir=video_dir,
        run_pipeline=args.run_pipeline,
        reid_device=args.reid_device,
        force_seed=args.force_seed,
        force_pipeline_cache=args.force_pipeline_cache,
        precision_matching=args.precision_matching,
        appearance_refine=args.appearance_refine,
        force_appearance_refine=args.force_appearance_refine,
    )
    seed_files = _resolve_seed_files(seed_dir, selected)
    db_info = _prepare_databases(output_dir=output_dir, seed_files=seed_files, namespace=namespace)

    case_results: list[dict[str, Any]] = []
    if not args.skip_agent:
        graph = _load_agent_graph(db_info)
        case_results = _run_agent_cases(
            graph,
            selected,
            top_k=args.top_k,
            negative_guard=args.negative_guard,
            negative_guard_mode=args.negative_guard_mode,
        )

    summary = _build_summary(
        args=args,
        env_report=env_report,
        db_info=db_info,
        selected_cases=selected,
        case_results=case_results,
    )
    _write_json(output_dir / "selected_cases.json", selected)
    _write_json(output_dir / "case_results.json", case_results)
    _write_json(output_dir / "summary.json", summary)
    _write_markdown(output_dir / "summary.md", summary, case_results)
    print(f"[mevid-agent] Done. Report: {output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
