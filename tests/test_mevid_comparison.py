"""Run report-ready MEVID comparisons against adapted surveillance baselines.

This script does two things:

1. Runs small, fair MEVID evaluations using the existing full video pipeline
   harness. By default it runs one slot and a small number of cases so the test
   stays affordable.
2. Writes an external-reference table with reported numbers from related
   surveillance papers. Those numbers are useful for report context, but are not
   treated as direct wins/losses because the datasets and tasks differ.

Default run:
    python tests/test_mevid_comparison.py --slot 13-50 --limit 20

Optional agent run:
    python tests/test_mevid_comparison.py --slot 13-50 --limit 20 --run-agent
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "results" / "mevid_comparison"
DEFAULT_VIDEO_DIR = ROOT / "_data" / "mevid_slots"
DEFAULT_XLSX = ROOT / "agent" / "test" / "data" / "agent_test_mevid.xlsx"


EXTERNAL_REFERENCES: list[dict[str, str]] = [
    {
        "work": "Evaluation of Vision-LLMs in Surveillance Video",
        "dataset": "UCF-Crime",
        "task": "zero-shot anomaly/action classification",
        "metric": "Top-1 / AUC / FP / wrong-label",
        "reported_result": "best guided Top-1 about 34.69%; best few-shot Top-1 about 45.05%",
        "comparison_role": "reference only; motivates VLM-only MEVID baseline",
        "directly_comparable": "no",
    },
    {
        "work": "Evaluation of Vision-LLMs in Surveillance Video",
        "dataset": "RWF-2000",
        "task": "normal vs fighting classification",
        "metric": "Top-1 / false positive",
        "reported_result": "Top-1 around 82.25%-86.25% depending on VLM",
        "comparison_role": "reference only; binary anomaly classification is easier than MEVID multi-camera QA",
        "directly_comparable": "no",
    },
    {
        "work": "Surveillance VALU",
        "dataset": "UCA / UCF-Crime Annotation",
        "task": "temporal grounding, captioning, dense captioning, anomaly detection, MLLM captioning",
        "metric": "R@K@IoU, BLEU, METEOR, ROUGE-L, CIDEr, AUC",
        "reported_result": "benchmark tables across TSGV, VC, DVC, MAD, and MLLM captioning",
        "comparison_role": "metric inspiration; use R@K@IoU/top-k retrieval for MEVID later",
        "directly_comparable": "no",
    },
    {
        "work": "SAFER-VLM",
        "dataset": "warehouse CCTV",
        "task": "worker detection, tracking, activity log, LLM reporting",
        "metric": "AP@0.5, ID consistency, FPS, VLM activity accuracy, alert precision",
        "reported_result": "AP@0.5 80-85%; ID consistency 85-95%; 36-38 FPS; activity accuracy 84%; alert precision 92%",
        "comparison_role": "closest system-level reference; compare architecture and efficiency, not raw accuracy",
        "directly_comparable": "partial",
    },
    {
        "work": "SV-FDT",
        "dataset": "traffic intersection surveillance",
        "task": "surveillance-assisted federated digital twin for traffic control",
        "metric": "mirroring delay, recognition accuracy, traffic flow, resource robustness, subjective QoE",
        "reported_result": "about 0.24s mirroring delay above 80% confidence; recognition accuracy about 0.90-0.93",
        "comparison_role": "efficiency/scalability reference for structured semantic video representations",
        "directly_comparable": "no",
    },
]


VIDEO_MODES: dict[str, list[str]] = {
    # Full video system: YOLO + OSNet + topology + matching + clip refine +
    # crop-based person/entity appearance refine + VLM QA.
    "video_full_refine_crop": ["--appearance-refine"],
    # Same complete pipeline/refine, but no crop-based appearance refinement.
    # This is the safest low-cost ablation to show why person-centric evidence matters.
    "video_refine_no_crop": [],
    # Useful for report ablation, but disabled by default because the user asked
    # to emphasize refine-enabled full pipeline comparisons.
    "video_no_refine": ["--no-refine"],
}


@dataclass(frozen=True)
class RunResult:
    method: str
    result_path: str
    accuracy: Any
    n_evaluated: int
    n_correct: int
    n_skipped: int
    cost_cny: Any
    total_tokens: int
    runtime_sec: float
    per_category: dict[str, Any]
    pipeline_summary: dict[str, Any]
    notes: str = ""


def _now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _run_command(cmd: list[str], cwd: Path, log_path: Path) -> tuple[int, float]:
    start = time.time()
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    return proc.returncode, round(time.time() - start, 2)


def _latest_json(directory: Path, pattern: str, before: set[Path]) -> Path:
    candidates = [
        p for p in directory.glob(pattern)
        if p.is_file() and p not in before
    ]
    if not candidates:
        candidates = [p for p in directory.glob(pattern) if p.is_file()]
    if not candidates:
        raise FileNotFoundError(f"No result file matching {pattern} in {directory}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _video_result(method: str, result_path: Path, runtime_sec: float) -> RunResult:
    data = _read_json(result_path)
    token_usage = data.get("token_usage") or {}
    return RunResult(
        method=method,
        result_path=str(result_path),
        accuracy=data.get("accuracy"),
        n_evaluated=int(data.get("n_evaluated") or 0),
        n_correct=int(data.get("n_correct") or 0),
        n_skipped=int(data.get("n_skipped") or 0),
        cost_cny=data.get("cost_cny"),
        total_tokens=int(token_usage.get("total_tokens") or 0),
        runtime_sec=runtime_sec,
        per_category=data.get("per_category") or {},
        pipeline_summary=data.get("pipeline_summary") or {},
    )


def _agent_result(result_path: Path, runtime_sec: float) -> RunResult:
    data = _read_json(result_path)
    return RunResult(
        method="agent_full",
        result_path=str(result_path),
        accuracy=data.get("answer_accuracy"),
        n_evaluated=int(data.get("answer_scored_count") or data.get("run_count") or 0),
        n_correct=int(data.get("answer_correct_count") or 0),
        n_skipped=int(data.get("error_count") or 0),
        cost_cny="not_reported",
        total_tokens=0,
        runtime_sec=runtime_sec,
        per_category=data.get("category_breakdown") or {},
        pipeline_summary={},
        notes=f"top_hit_rate={data.get('top_hit_rate')}",
    )


def run_video_mode(
    *,
    mode: str,
    slot: str,
    limit: int,
    frames: int,
    seed: int,
    video_dir: Path,
    xlsx: Path,
    output_dir: Path,
    reid_device: str,
    force_pipeline: bool,
    appearance_refine_max_entities: int,
    exclude_cameras: list[str],
) -> RunResult:
    mode_args = VIDEO_MODES[mode]
    run_dir = output_dir / "runs" / mode
    run_dir.mkdir(parents=True, exist_ok=True)
    before = set(run_dir.glob("mevid_full_*.json"))
    cmd = [
        sys.executable,
        "tests/test_mevid_full.py",
        "--slot",
        slot,
        "--limit",
        str(limit),
        "--frames",
        str(frames),
        "--seed",
        str(seed),
        "--video-dir",
        str(video_dir),
        "--xlsx",
        str(xlsx),
        "--out-dir",
        str(run_dir),
        "--reid-device",
        reid_device,
    ]
    if exclude_cameras:
        cmd.append("--exclude-cameras")
        cmd.extend(exclude_cameras)
    if force_pipeline:
        cmd.append("--force-pipeline")
    if appearance_refine_max_entities:
        cmd.extend(["--appearance-refine-max-entities", str(appearance_refine_max_entities)])
    cmd.extend(mode_args)
    code, runtime_sec = _run_command(cmd, ROOT, run_dir / "run.log")
    if code != 0:
        raise RuntimeError(f"{mode} failed with exit code {code}; see {run_dir / 'run.log'}")
    result_path = _latest_json(run_dir, "mevid_full_*.json", before)
    return _video_result(mode, result_path, runtime_sec)


def run_agent_mode(
    *,
    slot: str,
    limit: int,
    seed: int,
    video_dir: Path,
    xlsx: Path,
    output_dir: Path,
    reid_device: str,
    run_pipeline: bool,
    force_seed: bool,
    force_pipeline_cache: bool,
    precision_matching: bool,
    appearance_refine: bool,
    force_appearance_refine: bool,
    negative_guard: bool,
    negative_guard_mode: str,
) -> RunResult:
    agent_root = output_dir / "runs" / "agent_full"
    agent_root.mkdir(parents=True, exist_ok=True)
    before = set(agent_root.glob(f"{slot}_*/summary.json"))
    cmd = [
        sys.executable,
        "tests/test_mevid_video_agent_e2e.py",
        "--slot",
        slot,
        "--limit",
        str(limit),
        "--seed",
        str(seed),
        "--sample-mode",
        "balanced",
        "--xlsx-path",
        str(xlsx),
        "--video-dir",
        str(video_dir),
        "--output-root",
        str(agent_root),
        "--reid-device",
        reid_device,
        "--embedding-provider",
        "dashscope",
        "--embedding-model",
        "text-embedding-v3",
    ]
    if run_pipeline:
        cmd.append("--run-pipeline")
    if force_seed:
        cmd.append("--force-seed")
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
    if negative_guard:
        cmd.append("--negative-guard")
        cmd.extend(["--negative-guard-mode", negative_guard_mode])
    code, runtime_sec = _run_command(cmd, ROOT, agent_root / "run.log")
    if code != 0:
        raise RuntimeError(f"agent_full failed with exit code {code}; see {agent_root / 'run.log'}")
    result_path = _latest_json(agent_root, f"{slot}_*/summary.json", before)
    return _agent_result(result_path, runtime_sec)


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _flatten_method_rows(results: list[RunResult]) -> list[dict[str, Any]]:
    rows = []
    for r in results:
        row = {
            "method": r.method,
            "accuracy": r.accuracy,
            "correct": r.n_correct,
            "evaluated": r.n_evaluated,
            "skipped_or_errors": r.n_skipped,
            "cost_cny": r.cost_cny,
            "total_tokens": r.total_tokens,
            "runtime_sec": r.runtime_sec,
            "result_path": r.result_path,
            "notes": r.notes,
        }
        for category in ("event", "appearance", "cross_camera", "negative"):
            st = r.per_category.get(category) or {}
            row[f"{category}_accuracy"] = st.get("accuracy", st.get("answer_accuracy", ""))
            row[f"{category}_total"] = st.get("total", st.get("count", ""))
        rows.append(row)
    return rows


def _build_missing_data(results: list[RunResult], modes: list[str], run_agent: bool) -> list[str]:
    missing = [
        "Direct paper-to-MEVID accuracy comparison is not valid because datasets/tasks/protocols differ.",
        "Published external numbers are included only as related-work references.",
        "Detector AP@0.5 and tracking ID consistency require MEVID detection/tracking ground truth; current QA run does not provide them.",
        "Temporal R@K@IoU requires ground-truth event time spans for every MEVID question; current Excel mostly provides yes/no labels.",
        "VLM-only baseline is not included by default; add a dedicated mode that sends sampled frames/crops without structured pipeline context.",
    ]
    if "video_no_refine" not in modes:
        missing.append("No-refine ablation was not run in this comparison; use --include-no-refine to add it.")
    if not run_agent:
        missing.append("Full agent E2E was not run; use --run-agent to add answer accuracy and top-hit rate.")
    if not results:
        missing.append("No MEVID runs completed.")
    return missing


def write_report(
    *,
    output_dir: Path,
    args: argparse.Namespace,
    results: list[RunResult],
    missing_data: list[str],
) -> None:
    method_rows = _flatten_method_rows(results)
    _write_csv(
        output_dir / "mevid_method_comparison.csv",
        method_rows,
        [
            "method",
            "accuracy",
            "correct",
            "evaluated",
            "skipped_or_errors",
            "event_accuracy",
            "event_total",
            "appearance_accuracy",
            "appearance_total",
            "cross_camera_accuracy",
            "cross_camera_total",
            "negative_accuracy",
            "negative_total",
            "cost_cny",
            "total_tokens",
            "runtime_sec",
            "notes",
            "result_path",
        ],
    )
    _write_csv(
        output_dir / "external_reported_results.csv",
        EXTERNAL_REFERENCES,
        [
            "work",
            "dataset",
            "task",
            "metric",
            "reported_result",
            "comparison_role",
            "directly_comparable",
        ],
    )

    summary = {
        "timestamp": _now_stamp(),
        "slot": args.slot,
        "limit": args.limit,
        "frames": args.frames,
        "modes": args.modes,
        "run_agent": args.run_agent,
        "external_references": EXTERNAL_REFERENCES,
        "mevid_results": [r.__dict__ for r in results],
        "missing_data": missing_data,
        "report_guidance": (
            "Use external reported numbers as related-work context only. "
            "Use MEVID method rows for fair comparison because they share the same slot, cases, and QA protocol."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# MEVID Comparison Report",
        "",
        f"- Slot: `{args.slot}`",
        f"- Case limit per method: `{args.limit}`",
        f"- Frames per QA call: `{args.frames}`",
        f"- Output folder: `{output_dir}`",
        "",
        "## Key Interpretation",
        "",
        (
            "External papers are evaluated on different datasets and tasks, so their "
            "published scores are references rather than direct competitors. The fair "
            "comparison is the MEVID adapted-baseline table below."
        ),
        "",
        "## MEVID Adapted Baselines",
        "",
        "| Method | Overall | Event | Appearance | Cross-camera | Negative | Cost CNY | Runtime s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in method_rows:
        lines.append(
            "| {method} | {accuracy} | {event_accuracy} | {appearance_accuracy} | "
            "{cross_camera_accuracy} | {negative_accuracy} | {cost_cny} | {runtime_sec} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## External Reported Results",
            "",
            "| Work | Dataset | Task | Metric | Reported Result | Comparable? |",
            "|---|---|---|---|---|---|",
        ]
    )
    for ref in EXTERNAL_REFERENCES:
        lines.append(
            f"| {ref['work']} | {ref['dataset']} | {ref['task']} | "
            f"{ref['metric']} | {ref['reported_result']} | {ref['directly_comparable']} |"
        )
    lines.extend(["", "## Missing Data / Caveats", ""])
    lines.extend(f"- {item}" for item in missing_data)
    lines.extend(
        [
            "",
            "## Report-ready Claim",
            "",
            (
                "Unlike prior surveillance VLM systems that mainly evaluate single-camera "
                "classification, captioning, traffic digital twins, or warehouse activity "
                "logging, this MEVID comparison evaluates multi-camera person-centric QA. "
                "Our full pipeline explicitly combines detection, ReID, topology-aware "
                "cross-camera matching, LLM refinement, person-crop evidence, and optional "
                "agent retrieval under one unified MEVID protocol."
            ),
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run report-ready MEVID comparison baselines")
    parser.add_argument("--slot", default="13-50", help="MEVID slot to evaluate")
    parser.add_argument("--limit", type=int, default=20, help="QA cases per method")
    parser.add_argument("--frames", type=int, default=10, help="Sampled frames per QA case")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--video-dir", default=str(DEFAULT_VIDEO_DIR))
    parser.add_argument("--xlsx", default=str(DEFAULT_XLSX))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--reid-device", default="cpu")
    parser.add_argument("--force-pipeline", action="store_true", help="Re-run YOLO+OSNet instead of using cache")
    parser.add_argument(
        "--exclude-cameras",
        nargs="*",
        default=[],
        help="Camera IDs to remove from video-pipeline eval, e.g. --exclude-cameras G339",
    )
    parser.add_argument(
        "--appearance-refine-max-entities",
        type=int,
        default=0,
        help="Debug cap passed to appearance refinement; 0 means all entities",
    )
    parser.add_argument(
        "--modes",
        nargs="*",
        default=["video_full_refine_crop", "video_refine_no_crop"],
        choices=sorted(VIDEO_MODES),
        help="Video comparison modes to run",
    )
    parser.add_argument("--include-no-refine", action="store_true", help="Also run video_no_refine ablation")
    parser.add_argument("--run-agent", action="store_true", help="Also run full video+agent E2E comparison")
    parser.add_argument("--agent-force-seed", action="store_true", help="Regenerate agent vector seeds")
    parser.add_argument("--agent-run-pipeline", action="store_true", help="Allow agent script to run pipeline if cache is missing")
    parser.add_argument("--agent-force-pipeline-cache", action="store_true", help="Force agent seed generation to rebuild the slot pipeline cache")
    parser.add_argument("--agent-precision-matching", action="store_true", help="Use precision-focused cross-camera matching for agent seed generation (default)")
    parser.add_argument("--agent-legacy-matching", action="store_true", help="Use older looser cross-camera matching for agent seed generation")
    parser.add_argument("--agent-appearance-refine", action="store_true", help="Run/load crop-based appearance refinement for agent seed generation")
    parser.add_argument("--agent-force-appearance-refine", action="store_true", help="Rebuild crop-based appearance refinement for agent seed generation")
    parser.add_argument(
        "--agent-identity-verifier",
        dest="agent_negative_guard",
        action="store_true",
        default=True,
        help="Enable cross-camera identity consistency verifier in the full agent E2E run (default)",
    )
    parser.add_argument(
        "--agent-negative-guard",
        dest="agent_negative_guard",
        action="store_true",
        help="Compatibility alias for --agent-identity-verifier",
    )
    parser.add_argument(
        "--agent-disable-identity-verifier",
        dest="agent_negative_guard",
        action="store_false",
        help="Disable cross-camera identity consistency verifier in the full agent E2E run",
    )
    parser.add_argument(
        "--agent-negative-guard-mode",
        choices=["conservative", "strict", "llm", "llm_videorag", "precision"],
        default="precision",
        help="Negative guard mode for the full agent E2E run",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.agent_legacy_matching:
        args.agent_precision_matching = False
    else:
        args.agent_precision_matching = True
    modes = list(dict.fromkeys(args.modes))
    if args.include_no_refine and "video_no_refine" not in modes:
        modes.append("video_no_refine")
    args.modes = modes

    stamp = _now_stamp()
    output_dir = Path(args.output_root).expanduser().resolve() / f"{args.slot}_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    video_dir = Path(args.video_dir).expanduser().resolve()
    xlsx = Path(args.xlsx).expanduser().resolve()

    results: list[RunResult] = []
    errors: list[str] = []
    for mode in modes:
        print(f"[mevid-comparison] Running {mode} ...")
        try:
            results.append(
                run_video_mode(
                    mode=mode,
                    slot=args.slot,
                    limit=args.limit,
                    frames=args.frames,
                    seed=args.seed,
                    video_dir=video_dir,
                    xlsx=xlsx,
                    output_dir=output_dir,
                    reid_device=args.reid_device,
                    force_pipeline=args.force_pipeline,
                    appearance_refine_max_entities=args.appearance_refine_max_entities,
                    exclude_cameras=args.exclude_cameras,
                )
            )
        except Exception as exc:
            errors.append(f"{mode}: {exc}")
            print(f"[mevid-comparison] {mode} failed: {exc}")

    if args.run_agent:
        print("[mevid-comparison] Running agent_full ...")
        try:
            results.append(
                run_agent_mode(
                    slot=args.slot,
                    limit=args.limit,
                    seed=args.seed,
                    video_dir=video_dir,
                    xlsx=xlsx,
                    output_dir=output_dir,
                    reid_device=args.reid_device,
                    run_pipeline=args.agent_run_pipeline,
                    force_seed=args.agent_force_seed,
                    force_pipeline_cache=args.agent_force_pipeline_cache,
                    precision_matching=args.agent_precision_matching,
                    appearance_refine=args.agent_appearance_refine,
                    force_appearance_refine=args.agent_force_appearance_refine,
                    negative_guard=args.agent_negative_guard,
                    negative_guard_mode=args.agent_negative_guard_mode,
                )
            )
        except Exception as exc:
            errors.append(f"agent_full: {exc}")
            print(f"[mevid-comparison] agent_full failed: {exc}")

    missing_data = _build_missing_data(results, modes, args.run_agent)
    missing_data.extend(f"Run failed: {err}" for err in errors)
    write_report(output_dir=output_dir, args=args, results=results, missing_data=missing_data)

    print(f"[mevid-comparison] Wrote {output_dir / 'summary.md'}")
    print(f"[mevid-comparison] Wrote {output_dir / 'mevid_method_comparison.csv'}")
    print(f"[mevid-comparison] Wrote {output_dir / 'external_reported_results.csv'}")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
