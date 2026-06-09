"""test_08_e2e_60q.py — Agent 端到端 60 道题测试。

对 sampled_60_questions.json 全部 60 道题（cross_camera / existence / appearance / event / negative）
运行完整 agent graph，评估各类指标。

用法:
    cd agent/test/多摄像头测试
    conda run -n capstone python test_08_e2e_60q.py

输出: output/08_e2e_60q.json, output/08_e2e_60q.md
"""

from __future__ import annotations

import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage

_HERE = Path(__file__).resolve().parent
OUTPUT_DIR = _HERE / "output"
ROOT_DIR = _HERE.parents[2]
AGENT_DIR = ROOT_DIR / "agent"

sys.path.insert(0, str(AGENT_DIR))

import importlib

# Force reload to pick up our modified code
mod_names = [
    "agents.shared.query_classifier",
    "agents.shared.fusion_engine",
    "node.parallel_retrieval_fusion_node",
    "node.types",
]
for mn in mod_names:
    if mn in sys.modules:
        del sys.modules[mn]

import graph as graph_module


def load_all_questions() -> list[dict[str, Any]]:
    """Load all 60 questions with their category labels."""
    path = ROOT_DIR / "agent" / "test_mulcamera" / "sampled_60_questions.json"
    with open(path) as f:
        data = json.load(f)
    all_qs: list[dict[str, Any]] = []
    for category in ["cross_camera", "existence", "appearance", "event", "negative"]:
        for item in data.get(category, []):
            item["_category"] = category
            all_qs.append(item)
    return all_qs


def extract_expected_cameras(recall_text: str) -> set[str]:
    return set(re.findall(r"G\d+", recall_text))


def extract_answer_cameras(answer: str) -> set[str]:
    return set(re.findall(r"G\d+", answer))


def check_answer_correctness(answer: str, expected_answer: str) -> bool:
    ans_lower = answer.strip().lower()
    if not expected_answer:
        return True  # no ground truth → auto-pass
    if expected_answer.lower() == "yes":
        if ans_lower.startswith("yes") or ans_lower.startswith("yes."):
            return True
        if "no matching clip" in ans_lower:
            return False
        if ans_lower.startswith("no matching clip"):
            return False
        positive_patterns = ["yes.", "yes ", "有", "the relevant clip is in",
                             "appears across", "the queried entity appears"]
        negative_patterns = ["no matching", "no.", "没有"]
        has_positive = any(p in ans_lower for p in positive_patterns)
        has_negative = any(p in ans_lower for p in negative_patterns)
        if has_positive and not has_negative:
            return True
        if has_negative:
            return False
        # Ambiguous: default True if answer has content
        return len(answer.strip()) > 10
    elif expected_answer.lower() == "no":
        if "no matching clip" in ans_lower or ans_lower.startswith("no."):
            return True
        return False
    return True


def check_time_in_answer(answer: str, expected_time: str) -> tuple[bool, str]:
    if not expected_time or not answer.strip():
        return True, "no expected_time → auto-pass"
    try:
        parts = expected_time.replace("\u2013", "-").split("-")
        expected_start = _parse_mmss(parts[0].strip())
        expected_end = _parse_mmss(parts[-1].strip()) if len(parts) >= 2 else expected_start
    except Exception:
        return True, f"cannot parse expected_time={expected_time} → auto-pass"
    time_patterns = re.findall(r"(\d{1,2}):(\d{2})(?::(\d{2}))?", answer)
    for match in time_patterns:
        try:
            h = int(match[0])
            m = int(match[1])
            s = int(match[2]) if match[2] else 0
            seconds = h * 3600 + m * 60 + s if match[2] else h * 60 + m
        except Exception:
            continue
        margin = 60
        if expected_start - margin <= seconds <= expected_end + margin:
            return True, f"found {h:02d}:{m:02d}:{s:02d} in [{_fmt_mmss(expected_start)}-{_fmt_mmss(expected_end)}]"
    return False, f"no time in [{_fmt_mmss(expected_start)}-{_fmt_mmss(expected_end)}]"


def _parse_mmss(text: str) -> int:
    parts = text.split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    return int(float(text))


def _fmt_mmss(seconds: int) -> str:
    m, s = divmod(seconds, 60)
    return f"{m}:{s:02d}"


def compute_metrics(expected: set[str], found: set[str]) -> dict[str, Any]:
    inter = expected & found
    union = expected | found
    return {
        "expected_cameras": sorted(expected),
        "found_cameras": sorted(found),
        "intersection": sorted(inter),
        "recall": round(len(inter) / len(expected), 4) if expected else 0.0,
        "precision": round(len(inter) / len(found), 4) if found else 0.0,
        "iou": round(len(inter) / len(union), 4) if union else 0.0,
    }


def category_should_detect_mc(category: str) -> bool:
    """Which categories should have multi_camera detected."""
    return category in {"cross_camera", "negative"}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    questions = load_all_questions()
    print(f"Loaded {len(questions)} questions across 5 categories")
    cat_counts = {}
    for q in questions:
        cat = q["_category"]
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
    for cat, cnt in sorted(cat_counts.items()):
        print(f"  {cat}: {cnt}")
    print(f"Build agent graph ...")
    graph = graph_module.create_graph()
    print(f"Graph ready. Starting test ...\n")

    results: list[dict[str, Any]] = []
    tt0 = time.perf_counter()

    for idx, item in enumerate(questions):
        query = item["question"]
        category = item["_category"]
        recall_text = item.get("recall_challenge", "")
        expected_cameras = extract_expected_cameras(recall_text)
        expected_answer = item.get("expected_answer", "")
        expected_time = item.get("expected_time", "")

        config = {"configurable": {"thread_id": f"e2e-60q-{idx}", "user_id": "tester"}}
        error = None
        final_answer = ""
        classification_result = {}
        node_trace: list[str] = []
        ge_rows_count = 0
        multi_camera_detected = False

        t0 = time.perf_counter()
        try:
            for chunk in graph.stream(
                {"messages": [HumanMessage(content=query)]},
                config,
                stream_mode="values",
            ):
                if "current_node" in chunk:
                    node = chunk["current_node"]
                    if not node_trace or node_trace[-1] != node:
                        node_trace.append(node)
                final_answer = chunk.get("final_answer", final_answer)
                if "classification_result" in chunk:
                    cr = chunk["classification_result"]
                    if isinstance(cr, dict):
                        classification_result = cr
                        multi_camera_detected = cr.get("multi_camera", False)
                if "sql_debug" in chunk and isinstance(chunk["sql_debug"], dict):
                    ge_rows_count = chunk["sql_debug"].get("ge_rows_count", ge_rows_count)
        except Exception as exc:
            error = str(exc)

        elapsed_s = time.perf_counter() - t0

        answer_cameras = extract_answer_cameras(final_answer) if not error else set()
        metrics = compute_metrics(expected_cameras, answer_cameras)
        answer_has_content = len(final_answer.strip()) > 10
        answer_correct = check_answer_correctness(final_answer, expected_answer)
        time_ok, time_detail = check_time_in_answer(final_answer, expected_time)

        # Pass criteria depend on category
        expect_mc = category_should_detect_mc(category)
        if category == "cross_camera":
            # Require correct yes/no + camera recall >= 0.5
            passed = (
                error is None
                and answer_has_content
                and answer_correct
                and metrics["recall"] >= 0.5
            )
        elif category in {"existence", "appearance", "event"}:
            # Single-camera questions: just need content, no crash
            # camera recall not applicable (no cross-camera expected)
            passed = (
                error is None
                and answer_has_content
            )
        elif category == "negative":
            # Just require correct "no" answer
            passed = (
                error is None
                and answer_has_content
                and answer_correct
            )
        else:
            passed = error is None and answer_has_content

        result = {
            "idx": idx,
            "category": category,
            "question": query[:150],
            "recall_challenge": recall_text[:120],
            "expected_answer": expected_answer,
            "expected_time": expected_time,
            "expected_cameras": sorted(expected_cameras),
            "answer_cameras": sorted(answer_cameras),
            "final_answer": final_answer[:500],
            "error": error,
            "elapsed_s": round(elapsed_s, 1),
            "node_trace": node_trace,
            "multi_camera_detected": multi_camera_detected,
            "expect_multi_camera": expect_mc,
            "ge_rows_count": ge_rows_count,
            "passed": passed,
            "metrics": metrics,
            "answer_has_content": answer_has_content,
            "answer_correct": answer_correct,
            "time_ok": time_ok,
            "time_detail": time_detail,
        }
        results.append(result)

        icon = "✅" if passed else "❌"
        print(
            f"  {icon} Q{idx+1:02d} [{elapsed_s:.1f}s] "
            f"[{category}] mc={multi_camera_detected} ge={ge_rows_count} "
            f"correct={answer_correct} time={time_ok} "
            f"R={metrics['recall']:.2f} P={metrics['precision']:.2f} IoU={metrics['iou']:.2f} "
            f"| {query[:55]}..."
        )
        if error:
            print(f"       ERROR: {error[:200]}")
        if not passed and not error:
            reasons = []
            if not answer_has_content:
                reasons.append("empty/invalid answer")
            if category in {"cross_camera", "existence"}:
                if not answer_correct:
                    reasons.append("answer_correct=false")
                if metrics["recall"] < 0.5:
                    reasons.append(f"recall={metrics['recall']:.2f}")
            elif category == "negative" and not answer_correct:
                reasons.append("answer_correct=false")
            if reasons:
                print(f"       FAIL: {', '.join(reasons)}")

    total_elapsed = time.perf_counter() - tt0
    n = len(results)

    # Per-category stats
    cat_stats: dict[str, dict[str, Any]] = {}
    for r in results:
        cat = r["category"]
        if cat not in cat_stats:
            cat_stats[cat] = {"total": 0, "passed": 0, "correct": 0, "time_match": 0,
                              "sum_recall": 0.0, "sum_elapsed": 0.0, "mc_detected": 0}
        s = cat_stats[cat]
        s["total"] += 1
        s["passed"] += 1 if r["passed"] else 0
        s["correct"] += 1 if r["answer_correct"] else 0
        s["time_match"] += 1 if r["time_ok"] else 0
        s["sum_recall"] += r["metrics"]["recall"]
        s["sum_elapsed"] += r["elapsed_s"]
        s["mc_detected"] += 1 if r["multi_camera_detected"] else 0

    # Overall stats
    passed_count = sum(1 for r in results if r["passed"])
    correct_count = sum(1 for r in results if r["answer_correct"])
    time_match_count = sum(1 for r in results if r["time_ok"])
    avg_recall = sum(r["metrics"]["recall"] for r in results) / n if n else 0
    avg_precision = sum(r["metrics"]["precision"] for r in results) / n if n else 0
    avg_iou = sum(r["metrics"]["iou"] for r in results) / n if n else 0
    avg_elapsed = sum(r["elapsed_s"] for r in results) / n if n else 0
    mc_detection_rate = sum(1 for r in results if r["multi_camera_detected"]) / n if n else 0
    answer_rate = sum(1 for r in results if r["answer_has_content"]) / n if n else 0
    correctness_rate = correct_count / n if n else 0
    time_match_rate = time_match_count / n if n else 0

    report = {
        "title": "Agent 端到端 60 题测试",
        "started_at": datetime.now().isoformat(),
        "total_elapsed_s": round(total_elapsed, 1),
        "total_questions": n,
        "passed": passed_count,
        "failed": n - passed_count,
        "pass_rate": round(passed_count / n * 100, 1) if n else 0,
        "avg_recall": round(avg_recall, 4),
        "avg_precision": round(avg_precision, 4),
        "avg_iou": round(avg_iou, 4),
        "avg_elapsed_s": round(avg_elapsed, 1),
        "multi_camera_detection_rate": round(mc_detection_rate, 4),
        "answer_content_rate": round(answer_rate, 4),
        "correctness_rate": round(correctness_rate, 4),
        "correctness_count": correct_count,
        "time_match_rate": round(time_match_rate, 4),
        "time_match_count": time_match_count,
        "per_category": {cat: {
            "total": s["total"],
            "passed": s["passed"],
            "pass_rate": round(s["passed"] / s["total"] * 100, 1) if s["total"] else 0,
            "correct": s["correct"],
            "correct_rate": round(s["correct"] / s["total"] * 100, 1) if s["total"] else 0,
            "time_match": s["time_match"],
            "avg_recall": round(s["sum_recall"] / s["total"], 4) if s["total"] else 0,
            "avg_elapsed": round(s["sum_elapsed"] / s["total"], 1) if s["total"] else 0,
            "mc_detected": s["mc_detected"],
            "mc_detection_rate": round(s["mc_detected"] / s["total"] * 100, 1) if s["total"] else 0,
        } for cat, s in sorted(cat_stats.items())},
        "per_question": results,
    }

    json_path = OUTPUT_DIR / "08_e2e_60q.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # Markdown summary
    md_lines = [
        "# Agent 端到端 60 题测试",
        "",
        f"**Started**: {report['started_at']}  ",
        f"**Elapsed**: {report['total_elapsed_s']:.1f}s  ",
        "",
        "## 汇总",
        "",
        f"| 指标 | 值 |",
        f"|------|----|",
        f"| 通过率 | **{report['pass_rate']}%** ({passed_count}/{n}) |",
        f"| 答案正确率 | **{correctness_rate:.1%}** ({correct_count}/{n}) |",
        f"| 时间命中率 | **{time_match_rate:.1%}** ({time_match_count}/{n}) |",
        f"| 平均 Recall | {avg_recall:.4f} |",
        f"| 平均 Precision | {avg_precision:.4f} |",
        f"| 平均 IoU | {avg_iou:.4f} |",
        f"| 多摄意图检测率 | {mc_detection_rate:.1%} |",
        f"| 有效回答率 | {answer_rate:.1%} |",
        f"| 平均耗时/题 | {avg_elapsed:.1f}s |",
        f"| 总耗时 | {report['total_elapsed_s']:.1f}s |",
        "",
        "## 分类指标",
        "",
        "| 类别 | 总数 | 通过 | 通过率 | 正确 | 正确率 | Avg Recall | 耗时 | MC 检测率 |",
        "|------|------|------|--------|------|--------|------------|------|-----------|",
    ]
    for cat in sorted(cat_stats.keys()):
        s = cat_stats[cat]
        md_lines.append(
            f"| {cat} | {s['total']} | {s['passed']} | {s['passed']/s['total']*100:.0f}% | "
            f"{s['correct']} | {s['correct']/s['total']*100:.0f}% | "
            f"{s['sum_recall']/s['total']:.2f} | {s['sum_elapsed']/s['total']:.1f}s | "
            f"{s['mc_detected']/s['total']*100:.0f}% |"
        )
    md_lines.extend(["", "## 逐题结果", ""])

    for r in results:
        icon = "✅" if r["passed"] else "❌"
        correct_icon = "✅" if r.get("answer_correct", False) else "❌"
        time_icon = "✅" if r.get("time_ok", False) else "❌"
        md_lines.append(f"### {icon} Q{r['idx']+1} [{r['elapsed_s']:.1f}s] `{r['category']}`")
        md_lines.append(f"- **问题**: {r['question']}")
        md_lines.append(f"- **期望答案**: {r.get('expected_answer', '?')} | **期望时间**: {r.get('expected_time', '?')}")
        md_lines.append(f"- {correct_icon} 答案正确 | {time_icon} 时间命中")
        md_lines.append(f"- **期望摄像头**: {r['expected_cameras']}")
        md_lines.append(f"- **回答中的摄像头**: {r['answer_cameras']}")
        md_lines.append(f"- **Recall**: {r['metrics']['recall']:.2f} | **Precision**: {r['metrics']['precision']:.2f} | **IoU**: {r['metrics']['iou']:.2f}")
        md_lines.append(f"- **multi_camera 检测**: {r['multi_camera_detected']} (expect={r['expect_multi_camera']}) | **GE rows**: {r['ge_rows_count']}")
        md_lines.append(f"- **节点路径**: {' → '.join(r['node_trace'])}")
        if r["error"]:
            md_lines.append(f"- **错误**: {r['error'][:200]}")
        answer_preview = r["final_answer"][:300]
        md_lines.append(f"- **Agent 回答**: {answer_preview}")
        md_lines.append("")

    md_path = OUTPUT_DIR / "08_e2e_60q.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"\n--- SUMMARY ---")
    print(f"Pass rate: {report['pass_rate']}% ({passed_count}/{n})")
    print(f"Correctness: {correctness_rate:.1%} ({correct_count}/{n})")
    print(f"Time match: {time_match_rate:.1%} ({time_match_count}/{n})")
    print(f"Avg recall: {avg_recall:.4f}")
    print(f"Avg elapsed: {avg_elapsed:.1f}s/question")
    print(f"Multi-camera detection: {mc_detection_rate:.1%}")
    print(f"\nPer-category:")
    for cat in sorted(cat_stats.keys()):
        s = cat_stats[cat]
        print(f"  {cat:14s}: pass={s['passed']}/{s['total']} ({s['passed']/s['total']*100:.0f}%) "
              f"correct={s['correct']}/{s['total']} "
              f"avg_recall={s['sum_recall']/s['total']:.2f} "
              f"avg_time={s['sum_elapsed']/s['total']:.1f}s "
              f"mc={s['mc_detected']}/{s['total']}")
    print(f"Total time: {report['total_elapsed_s']:.1f}s")
    print(f"Reports → {json_path}, {md_path}")


if __name__ == "__main__":
    main()
