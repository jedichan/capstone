"""UCA video-cache + agent end-to-end evaluation runner.

This is the UCA counterpart to the MEVID e2e smoke runner. It uses existing
UCA annotation/seed tooling, builds temporary SQLite + Chroma indexes, runs the
LangGraph agent, and writes compact retrieval/answer/span metrics.
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

DEFAULT_XLSX = ROOT / "agent" / "test" / "data" / "agent_test.xlsx"
DEFAULT_UCA_GT = ROOT / "_data" / "Surveillance-Video-Understanding" / "UCF Annotation" / "json" / "UCFCrime_Test.json"
DEFAULT_SEED_DIR = ROOT / "agent" / "test" / "generated" / "ucfcrime_events_vector_flat"
DEFAULT_OUTPUT_ROOT = ROOT / "results" / "uca_agent_e2e"


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_env() -> None:
    try:
        from agent.core.runtime import load_env

        load_env(ROOT)
    except Exception:
        env_file = ROOT / ".env"
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _parse_yes_no(answer: Any) -> str:
    text = str(answer or "").strip().lower()
    text = re.sub(r"^[\s`*_>-]+", "", text)
    if text.startswith("yes") or text.startswith("是") or text.startswith("有"):
        return "yes"
    if text.startswith("no") or text.startswith("否") or text.startswith("没有") or text.startswith("无"):
        return "no"
    return "unknown"


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def _temporal_iou(a_start: Any, a_end: Any, b_start: Any, b_end: Any) -> float | None:
    a0 = _safe_float(a_start)
    a1 = _safe_float(a_end)
    b0 = _safe_float(b_start)
    b1 = _safe_float(b_end)
    if None in {a0, a1, b0, b1}:
        return None
    assert a0 is not None and a1 is not None and b0 is not None and b1 is not None
    if a1 <= a0 or b1 <= b0:
        return None
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return round(inter / union, 4) if union > 0 else None


def _interval_distance(a_start: Any, a_end: Any, b_start: Any, b_end: Any) -> float:
    a0 = _safe_float(a_start)
    a1 = _safe_float(a_end)
    b0 = _safe_float(b_start)
    b1 = _safe_float(b_end)
    if None in {a0, a1, b0, b1}:
        return float("inf")
    assert a0 is not None and a1 is not None and b0 is not None and b1 is not None
    if a1 <= a0 or b1 <= b0:
        return float("inf")
    if min(a1, b1) >= max(a0, b0):
        return 0.0
    return min(abs(b0 - a1), abs(a0 - b1))


def _time_aware_rerank_rows(rows: list[dict[str, Any]], case: dict[str, Any]) -> list[dict[str, Any]]:
    """Evaluation-only rerank for UCA timestamp queries.

    UCA labels abnormal-event intervals, while the generic agent ranks mostly by
    text/vector relevance. For this harness, prefer rows from the requested video
    and then rows that overlap or are closest to the expected time interval.
    """
    expected_video_id = str(case.get("video_id") or "").strip()
    expected_start = case.get("expected_start_sec")
    expected_end = case.get("expected_end_sec")
    if not rows or _safe_float(expected_start) is None or _safe_float(expected_end) is None:
        return rows

    def sort_key(item: tuple[int, dict[str, Any]]) -> tuple[int, float, float, int]:
        original_idx, row = item
        video_id = str(row.get("video_id") or "").strip()
        video_penalty = 0 if video_id == expected_video_id else 1
        iou = _temporal_iou(expected_start, expected_end, row.get("start_time"), row.get("end_time"))
        distance = _interval_distance(expected_start, expected_end, row.get("start_time"), row.get("end_time"))
        # Higher IoU is better, so use negative IoU for ascending sort.
        return (video_penalty, -(iou or 0.0), distance, original_idx)

    return [row for _, row in sorted(enumerate(rows), key=sort_key)]


def _check_environment(skip_agent: bool) -> dict[str, Any]:
    _load_env()
    imports: dict[str, str] = {}
    required = ["openpyxl", "chromadb"]
    if not skip_agent:
        required.extend(["langgraph", "langchain_openai"])
    for mod in required:
        try:
            importlib.import_module(mod)
            imports[mod] = "ok"
        except Exception as exc:
            imports[mod] = f"missing/error: {exc}"
    return {
        "imports": imports,
        "env": {
            "DASHSCOPE_API_KEY": bool(os.getenv("DASHSCOPE_API_KEY")),
            "DASHSCOPE_URL": bool(os.getenv("DASHSCOPE_URL")),
            "OPENAI_API_KEY": bool(os.getenv("OPENAI_API_KEY")),
            "AGENT_EMBEDDING_PROVIDER": os.getenv("AGENT_EMBEDDING_PROVIDER", ""),
        },
    }


def _assert_environment_ready(env_report: dict[str, Any], skip_agent: bool) -> None:
    missing = [k for k, v in env_report.get("imports", {}).items() if v != "ok"]
    env = env_report.get("env", {})
    missing_env = []
    if not env.get("DASHSCOPE_API_KEY") and not env.get("OPENAI_API_KEY"):
        missing_env.append("DASHSCOPE_API_KEY or OPENAI_API_KEY")
    if not skip_agent and (not env.get("DASHSCOPE_API_KEY") or not env.get("DASHSCOPE_URL")):
        missing_env.append("DASHSCOPE_API_KEY and DASHSCOPE_URL for agent LLM")
    if missing or missing_env:
        raise RuntimeError(json.dumps({
            "missing_imports": missing,
            "missing_env": missing_env,
            "environment": env_report,
        }, ensure_ascii=False, indent=2))


def _ensure_vector_seeds(seed_dir: Path, transcript_json: Path, force_seed: bool) -> None:
    if seed_dir.is_dir() and any(seed_dir.glob("*_events_vector_flat.json")) and not force_seed:
        print(f"[uca-agent] Vector seeds already exist: {seed_dir}")
        return
    if not transcript_json.exists():
        raise FileNotFoundError(f"UCA transcript/GT json not found: {transcript_json}")
    cmd = [
        sys.executable,
        str(ROOT / "agent" / "test" / "ucfcrime_transcript_importer.py"),
        "--transcript-json",
        str(transcript_json),
        "--output-dir",
        str(seed_dir),
        "--manifest-path",
        str(seed_dir.parent / "ucfcrime_events_manifest.json"),
    ]
    print("[uca-agent] Generating UCA vector-flat seeds ...")
    print("[uca-agent] " + " ".join(cmd))
    child_env = os.environ.copy()
    child_env.setdefault("PYTHONIOENCODING", "utf-8")
    subprocess.run(cmd, cwd=str(ROOT), check=True, env=child_env)


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
    cases = json.loads((dataset_dir / "agent_test_normalized.json").read_text(encoding="utf-8"))
    print(f"[uca-agent] Imported {len(cases)} cases from {xlsx_path.name}: {report}")
    return cases


def _fmt_sec(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


def _build_cases_from_transcript(transcript_json: Path, seed_dir: Path) -> list[dict[str, Any]]:
    payload = json.loads(transcript_json.read_text(encoding="utf-8"))
    available = {
        p.name.removesuffix("_events_vector_flat.json")
        for p in seed_dir.glob("*_events_vector_flat.json")
    }
    cases: list[dict[str, Any]] = []
    idx = 1
    for video_id in sorted(payload):
        if video_id not in available:
            continue
        item = payload.get(video_id)
        if not isinstance(item, dict):
            continue
        timestamps = item.get("timestamps") or []
        sentences = item.get("sentences") or []
        for seg_idx, (time_pair, sentence) in enumerate(zip(timestamps, sentences), start=1):
            if not (isinstance(time_pair, list) and len(time_pair) >= 2):
                continue
            start = _safe_float(time_pair[0])
            end = _safe_float(time_pair[1])
            if start is None or end is None or end <= start:
                continue
            text = str(sentence or "").replace("##", " ").strip()
            if not text:
                continue
            cases.append({
                "case_id": f"UCA_AUTO_{idx:04d}",
                "video_id": video_id,
                "question": (
                    f"In video {video_id}, retrieve the clip around "
                    f"{_fmt_sec(start)}-{_fmt_sec(end)} and describe the event."
                ),
                "expected_answer_label": "yes",
                "expected_start_sec": start,
                "expected_end_sec": end,
                "expected_time_raw": f"{_fmt_sec(start)}-{_fmt_sec(end)}",
                "reference_answer": text,
                "e2e_ready": 1,
                "retrieval_ready": 1,
                "generation_ready": 1,
            })
            idx += 1
    return cases


def _select_cases(cases: list[dict[str, Any]], seed_dir: Path, limit: int, seed: int) -> list[dict[str, Any]]:
    available = {
        p.name.removesuffix("_events_vector_flat.json")
        for p in seed_dir.glob("*_events_vector_flat.json")
    }
    eligible = [
        c for c in cases
        if str(c.get("video_id") or "").strip() in available
        and (c.get("e2e_ready", 1) == 1 or c.get("retrieval_ready", 1) == 1)
    ]
    eligible.sort(key=lambda c: str(c.get("case_id") or ""))
    if limit > 0 and len(eligible) > limit:
        rng = random.Random(seed)
        eligible = rng.sample(eligible, limit)
        eligible.sort(key=lambda c: str(c.get("case_id") or ""))
    return eligible


def _resolve_seed_files(seed_dir: Path, cases: list[dict[str, Any]]) -> list[Path]:
    files: list[Path] = []
    missing: list[str] = []
    for video_id in sorted({str(c.get("video_id") or "").strip() for c in cases}):
        p = seed_dir / f"{video_id}_events_vector_flat.json"
        if p.exists():
            files.append(p)
        else:
            missing.append(video_id)
    if missing:
        raise FileNotFoundError(f"Missing UCA vector seed examples: {missing[:8]}")
    return files


def _prepare_databases(output_dir: Path, seed_files: list[Path], namespace: str) -> dict[str, Any]:
    from agent.db.chroma_builder import ChromaBuildConfig, ChromaIndexBuilder
    from agent.db.sqlite_builder import SQLiteBuildConfig, SQLiteDatabaseBuilder

    runtime_dir = output_dir / "runtime"
    sqlite_path = runtime_dir / "uca_agent_e2e.sqlite"
    chroma_path = runtime_dir / "uca_agent_e2e_chroma"
    child_collection = f"{namespace}_tracks"
    parent_collection = f"{namespace}_tracks_parent"
    event_collection = f"{namespace}_events"

    sqlite_result = SQLiteDatabaseBuilder(
        SQLiteBuildConfig(db_path=sqlite_path, reset_existing=True, generate_init_prompt=False)
    ).build(seed_files=seed_files)
    chroma_result = ChromaIndexBuilder(
        ChromaBuildConfig(
            chroma_path=chroma_path,
            child_collection=child_collection,
            parent_collection=parent_collection,
            event_collection=event_collection,
            reset_existing=True,
        )
    ).build(seed_files=seed_files)
    return {
        "sqlite": sqlite_result,
        "chroma": chroma_result,
        "sqlite_path": str(sqlite_path),
        "chroma_path": str(chroma_path),
        "child_collection": child_collection,
        "parent_collection": parent_collection,
        "event_collection": event_collection,
    }


def _load_agent_graph(db_info: dict[str, Any]):
    _load_env()
    os.environ["AGENT_SQLITE_DB_PATH"] = str(db_info["sqlite_path"])
    os.environ["AGENT_CHROMA_PATH"] = str(db_info["chroma_path"])
    os.environ["AGENT_CHROMA_COLLECTION"] = str(db_info["child_collection"])
    os.environ["AGENT_CHROMA_CHILD_COLLECTION"] = str(db_info["child_collection"])
    os.environ["AGENT_CHROMA_PARENT_COLLECTION"] = str(db_info["parent_collection"])
    os.environ["AGENT_CHROMA_EVENT_COLLECTION"] = str(db_info["event_collection"])
    os.environ.setdefault("AGENT_CHROMA_RETRIEVAL_LEVEL", "child")
    graph_module = importlib.reload(sys.modules["graph"]) if "graph" in sys.modules else importlib.import_module("graph")
    return graph_module.create_graph()


def _final_rows(state: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("rerank_result", "merged_result", "hybrid_result", "sql_result"):
        rows = state.get(key)
        if isinstance(rows, list):
            return [r for r in rows if isinstance(r, dict)]
    return []


def _run_agent_cases(graph: Any, cases: list[dict[str, Any]], top_k: int, time_rerank: bool = True) -> list[dict[str, Any]]:
    from langchain_core.messages import HumanMessage

    results: list[dict[str, Any]] = []
    for idx, case in enumerate(cases, start=1):
        question = str(case.get("question") or "").strip()
        print(f"[uca-agent] Case {idx}/{len(cases)} {case.get('case_id')}: {question}")
        config = {"configurable": {"thread_id": f"uca-agent-e2e-{idx}", "user_id": "uca-agent-e2e"}}
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

        rows = _final_rows(last_state)
        expected_video_id = str(case.get("video_id") or "").strip()
        expected_label = str(case.get("expected_answer_label") or "").strip().lower()
        predicted_label = _parse_yes_no(last_state.get("final_answer"))
        original_top = rows[0] if rows else {}
        original_span_iou = _temporal_iou(
            case.get("expected_start_sec"),
            case.get("expected_end_sec"),
            original_top.get("start_time"),
            original_top.get("end_time"),
        )
        ranked_rows = _time_aware_rerank_rows(rows, case) if time_rerank else rows
        top_rows = ranked_rows[:top_k]
        top_video_ids = [str(r.get("video_id") or "").strip() for r in top_rows if r.get("video_id")]
        top = top_rows[0] if top_rows else {}
        span_iou = _temporal_iou(
            case.get("expected_start_sec"),
            case.get("expected_end_sec"),
            top.get("start_time"),
            top.get("end_time"),
        )
        results.append({
            "case_id": case.get("case_id"),
            "video_id": expected_video_id,
            "question": question,
            "expected_answer_label": expected_label,
            "predicted_answer_label": predicted_label,
            "answer_correct": expected_label in {"yes", "no"} and predicted_label == expected_label,
            "expected_start_sec": case.get("expected_start_sec"),
            "expected_end_sec": case.get("expected_end_sec"),
            "predicted_start_sec": top.get("start_time"),
            "predicted_end_sec": top.get("end_time"),
            "original_predicted_start_sec": original_top.get("start_time"),
            "original_predicted_end_sec": original_top.get("end_time"),
            "original_span_tiou": original_span_iou,
            "span_tiou": span_iou,
            "span_r_at_05": bool(span_iou is not None and span_iou >= 0.5),
            "time_rerank_applied": bool(time_rerank),
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2),
            "error": error,
            "node_trace": node_trace,
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
                    "scene_zone": r.get("scene_zone"),
                }
                for r in top_rows
            ],
        })
    return results


def _build_summary(args: argparse.Namespace, env_report: dict[str, Any], db_info: dict[str, Any] | None, cases: list[dict[str, Any]], results: list[dict[str, Any]]) -> dict[str, Any]:
    success = [r for r in results if not r.get("error")]
    answer_scored = [r for r in results if str(r.get("expected_answer_label") or "") in {"yes", "no"}]
    answer_correct = [r for r in answer_scored if r.get("answer_correct")]
    span_scored = [r for r in results if r.get("span_tiou") is not None]
    original_span_scored = [r for r in results if r.get("original_span_tiou") is not None]
    return {
        "timestamp": _now_stamp(),
        "case_count": len(cases),
        "run_count": len(results),
        "success_count": len(success),
        "error_count": len(results) - len(success),
        "top_hit_rate": round(sum(1 for r in results if r.get("top_hit")) / max(len(results), 1), 4),
        "answer_accuracy": round(len(answer_correct) / max(len(answer_scored), 1), 4) if answer_scored else None,
        "answer_correct_count": len(answer_correct),
        "answer_scored_count": len(answer_scored),
        "span_tiou_avg": round(sum(float(r["span_tiou"]) for r in span_scored) / max(len(span_scored), 1), 4) if span_scored else None,
        "span_r_at_05": round(sum(1 for r in span_scored if r.get("span_r_at_05")) / max(len(span_scored), 1), 4) if span_scored else None,
        "original_span_tiou_avg": round(sum(float(r["original_span_tiou"]) for r in original_span_scored) / max(len(original_span_scored), 1), 4) if original_span_scored else None,
        "original_span_r_at_05": round(sum(1 for r in original_span_scored if float(r["original_span_tiou"]) >= 0.5) / max(len(original_span_scored), 1), 4) if original_span_scored else None,
        "environment": env_report,
        "database": db_info,
        "args": vars(args),
    }


def _write_markdown(path: Path, summary: dict[str, Any], cases: list[dict[str, Any]]) -> None:
    lines = [
        "# UCA Video + Agent E2E Report",
        "",
        f"- Cases: `{summary['run_count']}`",
        f"- Success: `{summary['success_count']}`",
        f"- Errors: `{summary['error_count']}`",
        f"- Top-hit rate: `{summary['top_hit_rate']}`",
        f"- Answer accuracy: `{summary.get('answer_accuracy')}` "
        f"({summary.get('answer_correct_count')}/{summary.get('answer_scored_count')})",
        f"- Span tIoU avg: `{summary.get('span_tiou_avg')}`",
        f"- Span R@0.5: `{summary.get('span_r_at_05')}`",
        f"- Original span tIoU avg: `{summary.get('original_span_tiou_avg')}`",
        f"- Original span R@0.5: `{summary.get('original_span_r_at_05')}`",
        "",
        "## Cases",
    ]
    for case in cases:
        lines.extend([
            f"### {case.get('case_id')}",
            f"- Video: `{case.get('video_id')}`",
            f"- Question: {case.get('question')}",
            f"- Error: `{case.get('error')}`",
            f"- Top hit: `{case.get('top_hit')}`",
            f"- Span tIoU/R@0.5: `{case.get('span_tiou')}` / `{case.get('span_r_at_05')}`",
            f"- Original span tIoU: `{case.get('original_span_tiou')}`",
            f"- Expected/Predicted: `{case.get('expected_answer_label')}` / `{case.get('predicted_answer_label')}`",
            f"- Top videos: `{case.get('top_video_ids')}`",
            f"- Answer: {case.get('final_answer')}",
            "",
        ])
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run UCA video-cache + agent end-to-end test")
    parser.add_argument("--limit", type=int, default=8, help="Max cases to run; 0 = all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--xlsx-path", default=str(DEFAULT_XLSX))
    parser.add_argument("--include-sheets", nargs="*", default=["Part4"])
    parser.add_argument("--transcript-json", default=str(DEFAULT_UCA_GT))
    parser.add_argument("--auto-cases", action="store_true", help="Build UCA cases from transcript JSON instead of xlsx")
    parser.add_argument("--seed-dir", default=str(DEFAULT_SEED_DIR))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--force-seed", action="store_true")
    parser.add_argument("--skip-agent", action="store_true")
    parser.add_argument("--disable-time-rerank", action="store_true", help="Disable evaluation-only timestamp reranking")
    parser.add_argument("--namespace", default="", help="Chroma namespace; default uca_e2e_<timestamp>")
    parser.add_argument("--embedding-provider", default="dashscope", choices=["dashscope", "openai"])
    parser.add_argument("--embedding-model", default="", help="Optional embedding model override")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    stamp = _now_stamp()
    output_dir = Path(args.output_root).resolve() / stamp
    output_dir.mkdir(parents=True, exist_ok=True)
    namespace = args.namespace.strip() or f"uca_e2e_{stamp}"
    os.environ["AGENT_EMBEDDING_PROVIDER"] = args.embedding_provider
    if args.embedding_model:
        os.environ["AGENT_EMBEDDING_MODEL"] = args.embedding_model
    elif args.embedding_provider == "dashscope":
        os.environ["AGENT_EMBEDDING_MODEL"] = "text-embedding-v3"

    env_report = _check_environment(skip_agent=args.skip_agent)
    _assert_environment_ready(env_report, skip_agent=args.skip_agent)

    seed_dir = Path(args.seed_dir).resolve()
    _ensure_vector_seeds(seed_dir, Path(args.transcript_json).resolve(), args.force_seed)
    transcript_json = Path(args.transcript_json).resolve()
    if args.auto_cases:
        cases = _build_cases_from_transcript(transcript_json, seed_dir)
        print(f"[uca-agent] Built {len(cases)} auto cases from {transcript_json.name}")
    else:
        cases = _import_cases(Path(args.xlsx_path).resolve(), output_dir, list(args.include_sheets))
        selected_probe = _select_cases(cases, seed_dir, 1, args.seed)
        if not selected_probe:
            print("[uca-agent] No selected UCA xlsx cases; falling back to transcript auto-cases.")
            cases = _build_cases_from_transcript(transcript_json, seed_dir)
            print(f"[uca-agent] Built {len(cases)} auto cases from {transcript_json.name}")
    selected = _select_cases(cases, seed_dir, args.limit, args.seed)
    if not selected:
        raise RuntimeError("No UCA cases selected. Check xlsx sheets and seed directory.")
    print(f"[uca-agent] Selected {len(selected)} cases")
    _write_json(output_dir / "selected_cases.json", selected)

    seed_files = _resolve_seed_files(seed_dir, selected)
    db_info = _prepare_databases(output_dir, seed_files, namespace)
    if args.skip_agent:
        summary = _build_summary(args, env_report, db_info, selected, [])
        _write_json(output_dir / "summary.json", summary)
        print(f"[uca-agent] Prepared DB only: {output_dir}")
        return

    graph = _load_agent_graph(db_info)
    case_results = _run_agent_cases(graph, selected, args.top_k, time_rerank=not args.disable_time_rerank)
    summary = _build_summary(args, env_report, db_info, selected, case_results)
    _write_json(output_dir / "case_results.json", case_results)
    _write_json(output_dir / "summary.json", summary)
    _write_markdown(output_dir / "summary.md", summary, case_results)
    print(f"[uca-agent] Done. Report: {output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
