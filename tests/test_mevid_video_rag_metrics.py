"""Comprehensive MEVID video/RAG metrics harness.

This script reuses the existing MEVID video->agent pipeline and adds a
single report with the metrics needed for single-camera and multi-camera
experiments:

Video-level:
  accuracy, positive/negative accuracy, mean tIoU, R@1/R@5@tIoU=0.5,
  ReID Rank-1/Rank-5/mAP, cross-camera accuracy, temporal order accuracy.

RAG-level:
  Recall@1/5/10, mAP, Precision@5, evidence hit rate, answer accuracy,
  grounded correct rate, hallucination rate, query latency.

Default mode is cache-first. It expects vector-flat MEVID seeds to already
exist, or it can generate them from the pipeline cache.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = ROOT / "tests"
for p in (ROOT, TESTS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from test_mevid_video_agent_e2e import (  # noqa: E402
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SEED_DIR,
    DEFAULT_VIDEO_DIR,
    DEFAULT_XLSX,
    SLOT_CAMERAS,
    _assert_environment_ready,
    _balanced_category_sample,
    _check_environment,
    _ensure_vector_seeds,
    _import_cases,
    _load_agent_graph,
    _prepare_databases,
    _run_agent_cases,
    _select_cases,
    _stratified_sample,
)

PIPELINE_CACHE_DIR = ROOT / "_cache" / "mevid_pipeline"


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _camera_from_video_id(video_id: str) -> str:
    match = re.search(r"\.(G\d+)\.r\d+$", str(video_id or ""))
    return match.group(1) if match else ""


def _mentioned_cameras(text: str) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for cam in re.findall(r"\bG\d+\b", text or "", flags=re.IGNORECASE):
        cam = cam.upper()
        if cam not in seen:
            out.append(cam)
            seen.add(cam)
    return out


def _tiou(a_start: Any, a_end: Any, b_start: Any, b_end: Any) -> float | None:
    try:
        a0 = float(a_start)
        a1 = float(a_end)
        b0 = float(b_start)
        b1 = float(b_end)
    except Exception:
        return None
    if a1 < a0:
        a0, a1 = a1, a0
    if b1 < b0:
        b0, b1 = b1, b0
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    if union <= 0:
        return 0.0
    return inter / union


def _time_pair(start: Any, end: Any) -> tuple[float, float] | None:
    try:
        s = float(start)
        e = float(end)
    except Exception:
        return None
    if e < s:
        s, e = e, s
    return s, e


def _duration_sec(start: Any, end: Any) -> float | None:
    pair = _time_pair(start, end)
    if pair is None:
        return None
    return max(0.0, pair[1] - pair[0])


def _overlap_sec(a_start: Any, a_end: Any, b_start: Any, b_end: Any) -> float | None:
    a = _time_pair(a_start, a_end)
    b = _time_pair(b_start, b_end)
    if a is None or b is None:
        return None
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def _case_time(case: dict[str, Any]) -> tuple[float | None, float | None]:
    start = case.get("expected_start_sec")
    end = case.get("expected_end_sec")
    if start is None or end is None:
        return None, None
    try:
        s = float(start)
        e = float(end)
    except Exception:
        return None, None
    if e < s:
        s, e = e, s
    if e == s:
        e = s + 3.0
    return s, e


def _case_category(case: dict[str, Any]) -> str:
    return str(case.get("_category") or case.get("category") or "unknown")


def _is_temporal_localization_category(case: dict[str, Any]) -> bool:
    return _case_category(case) in {"event", "cross_camera"}


def _video_or_camera_matches(row: dict[str, Any], case: dict[str, Any]) -> bool:
    row_video = str(row.get("video_id") or "")
    case_video = str(case.get("video_id") or "")
    row_cam = _camera_from_video_id(row_video)
    case_cam = _camera_from_video_id(case_video)
    mentioned = _mentioned_cameras(str(case.get("question") or ""))
    if _case_category(case) == "cross_camera":
        return bool(row_cam and row_cam in set(mentioned or [case_cam]))
    return row_video == case_video


def _passes_min_duration(row: dict[str, Any], min_evidence_sec: float) -> bool:
    if min_evidence_sec <= 0:
        return True
    dur = _duration_sec(row.get("start_time"), row.get("end_time"))
    # Missing times are allowed for non-temporal evidence; zero-length rows are not.
    return dur is None or dur >= min_evidence_sec


def _row_relevance(
    row: dict[str, Any],
    case: dict[str, Any],
    *,
    temporal_threshold: float = 0.5,
    min_evidence_sec: float = 1.0,
    loose_without_time: bool = True,
) -> tuple[bool, float | None]:
    """Return (is_relevant, tiou_or_none) for one retrieved evidence row.

    Event/cross-camera questions need temporal localization, so they use
    tIoU>=threshold. Appearance/existence questions only need evidence inside
    the annotated visibility window, so any overlap is enough.
    """
    if not _video_or_camera_matches(row, case):
        return False, None
    if not _passes_min_duration(row, min_evidence_sec):
        return False, None

    exp_start, exp_end = _case_time(case)
    if exp_start is None or exp_end is None:
        return (loose_without_time, None)

    iou = _tiou(row.get("start_time"), row.get("end_time"), exp_start, exp_end)
    if _is_temporal_localization_category(case):
        return (bool(iou is not None and iou >= temporal_threshold), iou)

    overlap = _overlap_sec(row.get("start_time"), row.get("end_time"), exp_start, exp_end)
    return (bool(overlap is not None and overlap > 0), iou)


def _best_tiou(
    rows: list[dict[str, Any]],
    case: dict[str, Any],
    *,
    top_k: int,
    min_evidence_sec: float,
) -> float | None:
    exp_start, exp_end = _case_time(case)
    if exp_start is None or exp_end is None:
        return None
    vals: list[float] = []
    for row in rows[:top_k]:
        if not _video_or_camera_matches(row, case):
            continue
        if not _passes_min_duration(row, min_evidence_sec):
            continue
        iou = _tiou(row.get("start_time"), row.get("end_time"), exp_start, exp_end)
        if iou is not None:
            vals.append(iou)
    return max(vals) if vals else 0.0


def _average_precision(relevance: list[bool]) -> float:
    hits = 0
    score = 0.0
    for idx, rel in enumerate(relevance, start=1):
        if rel:
            hits += 1
            score += hits / idx
    return score / hits if hits else 0.0


def _temporal_order_ok(rows: list[dict[str, Any]], case: dict[str, Any]) -> bool | None:
    question = str(case.get("question") or "")
    if " then " not in question.lower() and "again in camera" not in question.lower():
        return None
    cams = _mentioned_cameras(question)
    if len(cams) < 2:
        return None
    first_cam, second_cam = cams[0], cams[1]

    first_times = [
        float(r.get("start_time"))
        for r in rows
        if _camera_from_video_id(str(r.get("video_id") or "")) == first_cam
        and r.get("start_time") is not None
    ]
    second_times = [
        float(r.get("start_time"))
        for r in rows
        if _camera_from_video_id(str(r.get("video_id") or "")) == second_cam
        and r.get("start_time") is not None
    ]
    if not first_times or not second_times:
        return False
    return min(first_times) <= min(second_times)


def _row_text(row: dict[str, Any]) -> str:
    parts = [
        row.get("video_id"),
        row.get("event_text"),
        row.get("event_text_en"),
        row.get("event_summary_en"),
        row.get("appearance_notes"),
        row.get("appearance_notes_en"),
        row.get("keywords"),
        row.get("entity_hint"),
    ]
    rendered: list[str] = []
    for item in parts:
        if isinstance(item, list):
            rendered.append(" ".join(str(x) for x in item))
        elif item:
            rendered.append(str(item))
    return " ".join(rendered)


def _row_camera_set(row: dict[str, Any]) -> set[str]:
    cams = set(_mentioned_cameras(_row_text(row)))
    cam = _camera_from_video_id(str(row.get("video_id") or ""))
    if cam:
        cams.add(cam)
    return cams


def _row_person_globals(row: dict[str, Any]) -> set[str]:
    return {
        item.lower()
        for item in re.findall(r"\bperson_global_\d+\b", _row_text(row), flags=re.IGNORECASE)
    }


def _case_camera_pair(case: dict[str, Any]) -> tuple[str, str] | None:
    cams = _mentioned_cameras(str(case.get("question") or ""))
    if len(cams) < 2:
        return None
    return cams[0], cams[1]


def _camera_pair_hit(rows: list[dict[str, Any]], case: dict[str, Any], *, top_k: int) -> bool | None:
    pair = _case_camera_pair(case)
    if pair is None:
        return None
    required = set(pair)
    scoped_rows = rows[:top_k]

    # Best case: one row explicitly carries the full cross-camera trajectory.
    if any(required.issubset(_row_camera_set(row)) for row in scoped_rows):
        return True

    # Fallback: multiple rows share the same person_global id and jointly cover
    # the requested camera pair.
    by_global: dict[str, set[str]] = defaultdict(set)
    for row in scoped_rows:
        globals_ = _row_person_globals(row)
        if not globals_:
            continue
        cams = _row_camera_set(row)
        for global_id in globals_:
            by_global[global_id].update(cams)
    return any(required.issubset(cams) for cams in by_global.values())


def _person_global_hit(rows: list[dict[str, Any]], case: dict[str, Any], *, top_k: int) -> bool | None:
    pair = _case_camera_pair(case)
    if pair is None:
        return None
    required = set(pair)
    by_global: dict[str, set[str]] = defaultdict(set)
    for row in rows[:top_k]:
        globals_ = _row_person_globals(row)
        cams = _row_camera_set(row)
        for global_id in globals_:
            by_global[global_id].update(cams)
    return any(required.issubset(cams) for cams in by_global.values())


def _trajectory_order_ok(rows: list[dict[str, Any]], case: dict[str, Any], *, top_k: int) -> bool | None:
    pair = _case_camera_pair(case)
    if pair is None:
        return None
    first_cam, second_cam = pair
    for row in rows[:top_k]:
        text = _row_text(row).upper()
        first_pos = text.find(first_cam)
        second_pos = text.find(second_cam)
        if first_pos >= 0 and second_pos >= 0:
            return first_pos <= second_pos
    return _temporal_order_ok(rows[:top_k], case)


def _summarize_multi_camera_level(
    rows: list[dict[str, Any]],
    *,
    top_k: int,
) -> dict[str, Any] | None:
    multi_rows = [
        r for r in rows
        if _case_camera_pair(r) is not None
        and (
            str(r.get("_category") or r.get("category") or "") in {"cross_camera", "negative"}
            or "also appear" in str(r.get("question") or "").lower()
            or "appear again" in str(r.get("question") or "").lower()
        )
    ]
    if not multi_rows:
        return None

    scored = [r for r in multi_rows if str(r.get("expected_answer_label") or "") in {"yes", "no"}]
    positives = [r for r in scored if r.get("expected_answer_label") == "yes"]
    negatives = [r for r in scored if r.get("expected_answer_label") == "no"]

    def ratio(num: int, den: int) -> float | None:
        return round(num / den, 4) if den else None

    def rows_for(r: dict[str, Any]) -> list[dict[str, Any]]:
        return list(r.get("top_rows") or [])

    pair_hit_1 = sum(1 for r in positives if _camera_pair_hit(rows_for(r), r, top_k=1))
    pair_hit_5 = sum(1 for r in positives if _camera_pair_hit(rows_for(r), r, top_k=5))
    global_hit_1 = sum(1 for r in positives if _person_global_hit(rows_for(r), r, top_k=1))
    global_hit_5 = sum(1 for r in positives if _person_global_hit(rows_for(r), r, top_k=5))
    trajectory_hit = sum(1 for r in positives if _camera_pair_hit(rows_for(r), r, top_k=top_k))

    order_scored = 0
    order_correct = 0
    for r in positives:
        order = _trajectory_order_ok(rows_for(r), r, top_k=top_k)
        if order is None:
            continue
        order_scored += 1
        order_correct += 1 if order else 0

    false_positives = sum(1 for r in negatives if str(r.get("predicted_answer_label") or "") == "yes")
    negative_rejections = sum(1 for r in negatives if r.get("answer_correct"))
    answer_correct = sum(1 for r in scored if r.get("answer_correct"))
    latencies = [float(r.get("elapsed_ms")) for r in multi_rows if r.get("elapsed_ms") is not None]

    return {
        "Cross-camera Answer Accuracy": ratio(answer_correct, len(scored)),
        "Positive Accuracy": ratio(sum(1 for r in positives if r.get("answer_correct")), len(positives)),
        "Negative Rejection Accuracy": ratio(negative_rejections, len(negatives)),
        "False Positive Rate": ratio(false_positives, len(negatives)),
        "Camera Pair Hit@1": ratio(pair_hit_1, len(positives)),
        "Camera Pair Hit@5": ratio(pair_hit_5, len(positives)),
        "Person-global Hit@1": ratio(global_hit_1, len(positives)),
        "Person-global Hit@5": ratio(global_hit_5, len(positives)),
        "Trajectory Evidence Hit Rate": ratio(trajectory_hit, len(positives)),
        "Temporal Order Accuracy": ratio(order_correct, order_scored),
        "Query Latency ms avg": round(sum(latencies) / len(latencies), 2) if latencies else None,
        "Evaluated Positives": len(positives),
        "Evaluated Negatives": len(negatives),
    }


def _merge_cases_with_results(cases: list[dict[str, Any]], results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {str(c.get("case_id")): c for c in cases}
    merged: list[dict[str, Any]] = []
    for result in results:
        case = dict(by_id.get(str(result.get("case_id")), {}))
        case.update(result)
        if "_category" not in case and "category" in result:
            case["_category"] = result.get("category")
        merged.append(case)
    return merged


def _load_reid_metrics(path: Path | None) -> dict[str, Any]:
    if not path or not path.exists():
        return {
            "rank_1": None,
            "rank_5": None,
            "mAP": None,
            "source": str(path) if path else "",
            "status": "missing",
            "note": "Run tests/test_mevid_evaluation.py with MEVID annotation+bbox crops to fill this.",
        }
    data = json.loads(path.read_text(encoding="utf-8"))
    metrics = data.get("reid_topology_augmented") or data.get("reid_baseline") or {}
    return {
        "rank_1": metrics.get("rank_1"),
        "rank_5": metrics.get("rank_5"),
        "mAP": metrics.get("mAP"),
        "source": str(path),
        "status": "loaded",
        "baseline": data.get("reid_baseline"),
        "topology_augmented": data.get("reid_topology_augmented"),
    }


def _summarize_split(
    name: str,
    rows: list[dict[str, Any]],
    reid: dict[str, Any],
    *,
    top_k: int,
    min_evidence_sec: float,
) -> dict[str, Any]:
    if not rows:
        return {"name": name, "count": 0}

    scored = [r for r in rows if str(r.get("expected_answer_label") or "") in {"yes", "no"}]
    positives = [r for r in scored if r.get("expected_answer_label") == "yes"]
    negatives = [r for r in scored if r.get("expected_answer_label") == "no"]
    answer_correct = [r for r in scored if r.get("answer_correct")]

    tiou_vals: list[float] = []
    r1_hits = 0
    r5_hits = 0
    recall_hits = {1: 0, 5: 0, 10: 0}
    precision5_vals: list[float] = []
    ap_vals: list[float] = []
    evidence_hits = 0
    grounded_correct = 0
    hallucinated = 0
    order_scored = 0
    order_correct = 0
    appearance_evidence_hits = 0
    appearance_evidence_scored = 0
    event_temporal_hits = 0
    event_temporal_scored = 0
    short_evidence_rows = 0

    temporal_scored = 0
    retrieval_scored = 0
    for r in rows:
        top_rows = list(r.get("top_rows") or [])
        short_evidence_rows += sum(
            1 for row in top_rows[:top_k]
            if (_duration_sec(row.get("start_time"), row.get("end_time")) or 0.0) < min_evidence_sec
        ) if min_evidence_sec > 0 else 0

        best_iou = _best_tiou(top_rows, r, top_k=top_k, min_evidence_sec=min_evidence_sec)
        if best_iou is not None:
            temporal_scored += 1
            tiou_vals.append(best_iou)
            if top_rows[:1] and _row_relevance(top_rows[0], r, min_evidence_sec=min_evidence_sec)[0]:
                r1_hits += 1
            if any(_row_relevance(row, r, min_evidence_sec=min_evidence_sec)[0] for row in top_rows[:5]):
                r5_hits += 1

        category = _case_category(r)
        if category == "appearance":
            appearance_evidence_scored += 1
            if any(_row_relevance(row, r, min_evidence_sec=min_evidence_sec)[0] for row in top_rows[:top_k]):
                appearance_evidence_hits += 1
        if _is_temporal_localization_category(r):
            event_temporal_scored += 1
            if any(_row_relevance(row, r, min_evidence_sec=min_evidence_sec)[0] for row in top_rows[:top_k]):
                event_temporal_hits += 1

        relevance = [
            _row_relevance(row, r, min_evidence_sec=min_evidence_sec)[0]
            for row in top_rows[:top_k]
        ]
        if relevance:
            retrieval_scored += 1
            for k in (1, 5, 10):
                if any(relevance[: min(k, len(relevance))]):
                    recall_hits[k] += 1
            precision5_vals.append(sum(1 for x in relevance[:5] if x) / 5.0)
            ap_vals.append(_average_precision(relevance[:10]))
            hit = any(relevance[:10])
            if hit:
                evidence_hits += 1
            if hit and r.get("answer_correct"):
                grounded_correct += 1
            if str(r.get("predicted_answer_label") or "") == "yes" and not hit:
                hallucinated += 1

        order = _temporal_order_ok(top_rows, r)
        if order is not None:
            order_scored += 1
            order_correct += 1 if order else 0

    cross = [r for r in rows if str(r.get("_category") or r.get("category") or "") == "cross_camera"]
    latencies = [float(r.get("elapsed_ms")) for r in rows if r.get("elapsed_ms") is not None]

    def ratio(num: int, den: int) -> float | None:
        return round(num / den, 4) if den else None

    return {
        "name": name,
        "count": len(rows),
        "video_level": {
            "accuracy": ratio(len(answer_correct), len(scored)),
            "positive_accuracy": ratio(sum(1 for r in positives if r.get("answer_correct")), len(positives)),
            "negative_accuracy": ratio(sum(1 for r in negatives if r.get("answer_correct")), len(negatives)),
            "mean_tIoU": round(sum(tiou_vals) / len(tiou_vals), 4) if tiou_vals else None,
            "R@1@tIoU=0.5": ratio(r1_hits, temporal_scored),
            "R@5@tIoU=0.5": ratio(r5_hits, temporal_scored),
            "reid_rank_1": reid.get("rank_1"),
            "reid_rank_5": reid.get("rank_5"),
            "reid_mAP": reid.get("mAP"),
            "cross_camera_accuracy": ratio(sum(1 for r in cross if r.get("answer_correct")), len(cross)),
            "temporal_order_accuracy": ratio(order_correct, order_scored),
            "appearance_evidence_hit": ratio(appearance_evidence_hits, appearance_evidence_scored),
            "event_temporal_hit@0.5": ratio(event_temporal_hits, event_temporal_scored),
            "short_evidence_rows_filtered": short_evidence_rows,
        },
        "rag_level": {
            "Recall@1": ratio(recall_hits[1], retrieval_scored),
            "Recall@5": ratio(recall_hits[5], retrieval_scored),
            "Recall@10": ratio(recall_hits[10], retrieval_scored),
            "mAP": round(sum(ap_vals) / len(ap_vals), 4) if ap_vals else None,
            "Precision@5": round(sum(precision5_vals) / len(precision5_vals), 4) if precision5_vals else None,
            "Evidence Hit Rate": ratio(evidence_hits, retrieval_scored),
            "Answer Accuracy": ratio(len(answer_correct), len(scored)),
            "Grounded Correct Rate": ratio(grounded_correct, len(scored)),
            "Hallucination Rate": ratio(hallucinated, len(scored)),
            "Query Latency ms avg": round(sum(latencies) / len(latencies), 2) if latencies else None,
        },
        "multi_camera_level": _summarize_multi_camera_level(rows, top_k=top_k),
    }


def _build_summary(
    *,
    args: argparse.Namespace,
    cases: list[dict[str, Any]],
    results: list[dict[str, Any]],
    reid_metrics: dict[str, Any],
    env_report: dict[str, Any],
    db_info: dict[str, Any],
) -> dict[str, Any]:
    merged = _merge_cases_with_results(cases, results)
    single = [
        r for r in merged
        if str(r.get("_category") or r.get("category") or "") != "cross_camera"
    ]
    multi = [
        r for r in merged
        if str(r.get("_category") or r.get("category") or "") == "cross_camera"
        or len(_mentioned_cameras(str(r.get("question") or ""))) >= 2
    ]
    return {
        "timestamp": _now_stamp(),
        "slot": args.slot,
        "camera": args.camera,
        "task": args.task,
        "case_count": len(cases),
        "run_count": len(results),
        "metrics": {
            "all": _summarize_split(
                "all",
                merged,
                reid_metrics,
                top_k=args.top_k,
                min_evidence_sec=args.min_evidence_duration_sec,
            ),
            "single_camera": _summarize_split(
                "single_camera",
                single,
                reid_metrics,
                top_k=args.top_k,
                min_evidence_sec=args.min_evidence_duration_sec,
            ),
            "multi_camera": _summarize_split(
                "multi_camera",
                multi,
                reid_metrics,
                top_k=args.top_k,
                min_evidence_sec=args.min_evidence_duration_sec,
            ),
        },
        "reid_metrics": reid_metrics,
        "environment": env_report,
        "database": db_info,
        "args": vars(args),
        "metric_notes": {
            "temporal_relevance": "Event/cross-camera rows are relevant when video/camera matches and tIoU >= 0.5. Appearance rows are relevant when video matches and evidence overlaps the annotated visibility window.",
            "multi_camera_primary": "For multi-camera reports, prefer the Multi-camera-level section over tIoU. Cross-camera questions are evaluated by camera-pair/person-global trajectory evidence and negative rejection.",
            "min_evidence_duration_sec": args.min_evidence_duration_sec,
            "grounded_correct": "Answer is correct and at least one relevant evidence row appears in top-10.",
            "hallucination": "Predicted yes but no relevant evidence row appears in top-10.",
            "reid": "Loaded from MEVID standard ReID result JSON when provided.",
        },
    }


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# MEVID Comprehensive Video/RAG Metrics",
        "",
        f"- Slot: `{summary['slot']}`",
        f"- Camera: `{summary.get('camera') or 'all'}`",
        f"- Task: `{summary['task']}`",
        f"- Cases: `{summary['run_count']}`",
        f"- ReID status: `{summary['reid_metrics'].get('status')}`",
        "",
    ]
    for split_name, split in summary["metrics"].items():
        lines.extend([f"## {split_name}", "", "### Video-level", ""])
        for key, value in split.get("video_level", {}).items():
            lines.append(f"- {key}: `{value}`")
        lines.extend(["", "### RAG-level", ""])
        for key, value in split.get("rag_level", {}).items():
            lines.append(f"- {key}: `{value}`")
        multi_level = split.get("multi_camera_level")
        if multi_level:
            lines.extend(["", "### Multi-camera-level", ""])
            for key, value in multi_level.items():
                lines.append(f"- {key}: `{value}`")
        lines.append("")
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def _filter_task(cases: list[dict[str, Any]], task: str) -> list[dict[str, Any]]:
    if task == "both":
        return cases
    out = []
    for case in cases:
        category = str(case.get("_category") or "")
        n_cams = len(_mentioned_cameras(str(case.get("question") or "")))
        is_multi = category == "cross_camera" or n_cams >= 2
        if task == "multi_camera" and is_multi:
            out.append(case)
        elif task == "single_camera" and not is_multi:
            out.append(case)
    return out


def _expand_cameras(camera_arg: str) -> set[str]:
    raw = str(camera_arg or "").strip()
    if not raw:
        return set()
    return {item.strip().upper() for item in re.split(r"[,;]", raw) if item.strip()}


def _expand_slots(slot_arg: str) -> list[str]:
    raw = str(slot_arg or "").strip()
    if not raw or raw.lower() == "all":
        return sorted(SLOT_CAMERAS)
    slots = [item.strip() for item in re.split(r"[,;]", raw) if item.strip()]
    unknown = [slot for slot in slots if slot not in SLOT_CAMERAS]
    if unknown:
        raise ValueError(f"Unknown MEVID slot(s): {unknown}. Available: {sorted(SLOT_CAMERAS)} or all")
    return slots


def _slots_for_cameras(slots: list[str], cameras: set[str]) -> list[str]:
    if not cameras:
        return slots
    return [
        slot for slot in slots
        if cameras.intersection({cam.upper() for cam in SLOT_CAMERAS.get(slot, {})})
    ]


def _slot_has_local_inputs(slot: str, cameras: set[str], seed_dir: Path) -> bool:
    """True when the slot can be evaluated without running the video pipeline."""
    if (PIPELINE_CACHE_DIR / f"{slot}_pipeline.json").exists():
        return True
    cam_map = SLOT_CAMERAS.get(slot, {})
    target_items = [
        (cam, stem) for cam, stem in cam_map.items()
        if not cameras or cam.upper() in cameras
    ]
    return any((seed_dir / f"{stem}_events_vector_flat.json").exists() for _cam, stem in target_items)


def _filter_locally_available_slots(
    slots: list[str],
    cameras: set[str],
    seed_dir: Path,
    *,
    run_pipeline: bool,
) -> tuple[list[str], list[str]]:
    if run_pipeline:
        return slots, []
    available = [slot for slot in slots if _slot_has_local_inputs(slot, cameras, seed_dir)]
    skipped = [slot for slot in slots if slot not in available]
    return available, skipped


def _missing_seed_video_ids(seed_dir: Path, cases: list[dict[str, Any]]) -> list[str]:
    video_ids = sorted({str(c.get("video_id") or "").strip() for c in cases if c.get("video_id")})
    return [
        video_id for video_id in video_ids
        if not (seed_dir / f"{video_id}_events_vector_flat.json").exists()
    ]


def _filter_camera(cases: list[dict[str, Any]], cameras: set[str]) -> list[dict[str, Any]]:
    if not cameras:
        return cases
    return [
        case for case in cases
        if _camera_from_video_id(str(case.get("video_id") or "")).upper() in cameras
    ]


def _select_cases_for_slots(
    imported: list[dict[str, Any]],
    *,
    slots: list[str],
    cameras: set[str],
    task: str,
    limit: int,
    sample_mode: str,
    seed: int,
) -> list[dict[str, Any]]:
    combined: list[dict[str, Any]] = []
    for slot in slots:
        slot_cases = _select_cases(
            imported,
            slot=slot,
            categories=set(),
            limit=0,
            sample_mode="first",
            seed=seed,
        )
        for case in slot_cases:
            case["_slot"] = slot
        combined.extend(slot_cases)

    combined = _filter_camera(combined, cameras)
    combined = _filter_task(combined, task)
    if limit <= 0 or len(combined) <= limit:
        return combined
    if sample_mode == "balanced":
        return _balanced_category_sample(combined, limit=limit, seed=seed)
    if sample_mode == "stratified":
        return _stratified_sample(combined, limit=limit, seed=seed)
    return combined[:limit]


def _required_seed_video_ids(
    cases: list[dict[str, Any]],
    *,
    slots: list[str],
    cameras: set[str],
    task: str,
) -> list[str]:
    """Resolve all seed video_ids required to build the runtime DB.

    For multi-camera evaluation we must index every selected camera for the slot,
    not only the target ``case.video_id`` rows. Cross-camera questions often ask
    about a source camera that differs from the answer video, so limiting the DB
    to target videos can silently omit required evidence (e.g. G508).
    """
    video_ids = {
        str(case.get("video_id") or "").strip()
        for case in cases
        if str(case.get("video_id") or "").strip()
    }
    if task in {"multi_camera", "both"}:
        for slot in slots:
            for cam, stem in SLOT_CAMERAS.get(slot, {}).items():
                if cameras and cam.upper() not in cameras:
                    continue
                video_ids.add(str(stem).strip())
    return sorted(video_id for video_id in video_ids if video_id)


def _missing_required_seed_video_ids(
    seed_dir: Path,
    *,
    cases: list[dict[str, Any]],
    slots: list[str],
    cameras: set[str],
    task: str,
) -> list[str]:
    return [
        video_id
        for video_id in _required_seed_video_ids(cases, slots=slots, cameras=cameras, task=task)
        if not (seed_dir / f"{video_id}_events_vector_flat.json").exists()
    ]


def _resolve_metrics_seed_files(
    seed_dir: Path,
    *,
    cases: list[dict[str, Any]],
    slots: list[str],
    cameras: set[str],
    task: str,
) -> list[Path]:
    seed_files: list[Path] = []
    missing: list[str] = []
    for video_id in _required_seed_video_ids(cases, slots=slots, cameras=cameras, task=task):
        seed_file = seed_dir / f"{video_id}_events_vector_flat.json"
        if seed_file.exists():
            seed_files.append(seed_file)
        else:
            missing.append(video_id)
    if missing:
        raise FileNotFoundError(
            "Missing vector-flat seed files required by metrics runtime DB. "
            f"Missing examples: {missing[:8]}"
        )
    return seed_files


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run comprehensive MEVID video/RAG metrics")
    parser.add_argument("--slot", default="13-50", help="MEVID slot, comma list, or all. e.g. 13-50 or 13-50,14-20")
    parser.add_argument("--camera", default="", help="Optional camera filter, e.g. G328 or G328,G339. Useful for single-camera multi-time evaluation")
    parser.add_argument("--task", choices=["single_camera", "multi_camera", "both"], default="both")
    parser.add_argument("--limit", type=int, default=40, help="Max cases after task filtering; 0 = all")
    parser.add_argument("--sample-mode", choices=["balanced", "stratified", "first"], default="balanced")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=10, help="Rows retained for RAG metrics")
    parser.add_argument(
        "--min-evidence-duration-sec",
        type=float,
        default=1.0,
        help="Ignore retrieved evidence rows shorter than this duration; use 0 to disable",
    )
    parser.add_argument("--xlsx-path", default=str(DEFAULT_XLSX))
    parser.add_argument("--seed-dir", default=str(DEFAULT_SEED_DIR))
    parser.add_argument("--video-dir", default=str(DEFAULT_VIDEO_DIR))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT.parent / "mevid_comprehensive_eval"))
    parser.add_argument("--include-sheets", nargs="*", default=["Part1", "Part4"])
    parser.add_argument("--run-pipeline", action="store_true")
    parser.add_argument("--reid-device", default="cpu")
    parser.add_argument("--force-seed", action="store_true")
    parser.add_argument("--skip-agent", action="store_true", help="Prepare DB and selected cases only")
    parser.add_argument("--namespace", default="")
    parser.add_argument("--embedding-provider", choices=["dashscope", "openai"], default="dashscope")
    parser.add_argument("--embedding-model", default="text-embedding-v3")
    parser.add_argument("--reid-json", default=str(ROOT / "results" / "mevid_eval.json"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    os.environ["AGENT_EMBEDDING_PROVIDER"] = args.embedding_provider
    os.environ["AGENT_EMBEDDING_MODEL"] = args.embedding_model

    stamp = _now_stamp()
    slots = _expand_slots(args.slot)
    cameras = _expand_cameras(args.camera)
    effective_slots = _slots_for_cameras(slots, cameras)
    seed_dir = Path(args.seed_dir).expanduser().resolve()
    effective_slots, skipped_input_slots = _filter_locally_available_slots(
        effective_slots,
        cameras,
        seed_dir,
        run_pipeline=args.run_pipeline,
    )
    if not effective_slots:
        available = sorted({cam for slot in slots for cam in SLOT_CAMERAS.get(slot, {})})
        raise RuntimeError(
            "No selected slot has local vector seeds or pipeline cache. "
            f"camera={sorted(cameras)}, available_cameras={available}. "
            "Run with --run-pipeline or generate seeds first."
        )
    slot_label = "all" if len(effective_slots) == len(SLOT_CAMERAS) else "+".join(effective_slots)
    cam_label = "+".join(sorted(cameras)) if cameras else "allcams"
    output_dir = Path(args.output_root).expanduser().resolve() / f"{slot_label}_{cam_label}_{args.task}_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    namespace = args.namespace.strip() or f"mevid_metrics_{stamp}"

    env_report = _check_environment(run_pipeline=args.run_pipeline)
    _assert_environment_ready(env_report, run_pipeline=args.run_pipeline, skip_agent=args.skip_agent)
    if skipped_input_slots:
        print(
            "[mevid-metrics] Skipping slots without local seed/cache "
            f"(use --run-pipeline to build them): {skipped_input_slots}"
        )

    imported = _import_cases(Path(args.xlsx_path).resolve(), output_dir, include_sheets=args.include_sheets)
    selected = _select_cases_for_slots(
        imported,
        slots=effective_slots,
        cameras=cameras,
        task=args.task,
        limit=args.limit,
        sample_mode=args.sample_mode,
        seed=args.seed,
    )
    if not selected:
        raise RuntimeError(f"No MEVID cases selected for slot={args.slot}, camera={args.camera or 'all'}, task={args.task}")
    print(
        f"[mevid-metrics] Selected {len(selected)} cases for "
        f"slots={effective_slots}, camera={sorted(cameras) if cameras else 'all'}, task={args.task}"
    )

    required_seed_ids = _required_seed_video_ids(
        selected,
        slots=effective_slots,
        cameras=cameras,
        task=args.task,
    )
    missing_seed_ids = _missing_required_seed_video_ids(
        seed_dir,
        cases=selected,
        slots=effective_slots,
        cameras=cameras,
        task=args.task,
    )
    if missing_seed_ids or args.force_seed:
        video_dir = Path(args.video_dir).expanduser().resolve()
        for slot in effective_slots:
            _ensure_vector_seeds(
                slot=slot,
                seed_dir=seed_dir,
                video_dir=video_dir,
                run_pipeline=args.run_pipeline,
                reid_device=args.reid_device,
                force_seed=args.force_seed,
            )
    else:
        print("[mevid-metrics] Required vector seeds already exist for selected cases")
    seed_files = _resolve_metrics_seed_files(
        seed_dir,
        cases=selected,
        slots=effective_slots,
        cameras=cameras,
        task=args.task,
    )
    print(
        f"[mevid-metrics] Building runtime DB from {len(seed_files)} seed files "
        f"(required_video_ids={len(required_seed_ids)})"
    )
    db_info = _prepare_databases(output_dir=output_dir, seed_files=seed_files, namespace=namespace)

    results: list[dict[str, Any]] = []
    if not args.skip_agent:
        graph = _load_agent_graph(db_info)
        results = _run_agent_cases(graph, selected, top_k=args.top_k)

    reid_metrics = _load_reid_metrics(Path(args.reid_json).expanduser().resolve() if args.reid_json else None)
    summary = _build_summary(
        args=args,
        cases=selected,
        results=results,
        reid_metrics=reid_metrics,
        env_report=env_report,
        db_info=db_info,
    )

    _write_json(output_dir / "selected_cases.json", selected)
    _write_json(output_dir / "case_results.json", results)
    _write_json(output_dir / "summary.json", summary)
    _write_markdown(output_dir / "summary.md", summary)

    all_metrics = summary["metrics"]["all"]
    print("\n[mevid-metrics] Done")
    print(f"  Report: {output_dir / 'summary.md'}")
    print(f"  Video accuracy: {all_metrics.get('video_level', {}).get('accuracy')}")
    print(f"  RAG Recall@5: {all_metrics.get('rag_level', {}).get('Recall@5')}")
    print(f"  RAG Answer Accuracy: {all_metrics.get('rag_level', {}).get('Answer Accuracy')}")
    print(f"  ReID Rank-1/Rank-5/mAP: {reid_metrics.get('rank_1')} / {reid_metrics.get('rank_5')} / {reid_metrics.get('mAP')}")


if __name__ == "__main__":
    main()
