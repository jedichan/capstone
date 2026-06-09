"""Generate ``REPORT_TABLES.md`` from a RAGAS e2e output directory.

Reads ``e2e_report.json``, ``summary_report.json``, and optionally a sibling ``.log``
for RAGAS wall-clock timings.  Embeds **custom_correctness** (P1-Next-C) and
task-native localization fields alongside RAGAS LLM metrics.

See ``agent/challenge.md`` §5.4 for ``custom_correctness`` definitions.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


def _resolve_log_path(output_dir: Path, log_path: Path | None) -> Path | None:
    if log_path is not None:
        p = log_path.expanduser().resolve()
        return p if p.is_file() else None
    candidate = output_dir.parent / f"{output_dir.name}.log"
    if candidate.is_file():
        return candidate
    return None


def _parse_ragas_durations(log_text: str, total_cases: int) -> dict[str, float]:
    """Map case_id -> seconds from ``[ragas] done … in X.Xs`` lines."""
    pat = re.compile(rf"\[ragas\] done\s+\d+/{total_cases}\s+(PART\d+_\d+)\s+in\s+([0-9.]+)s")
    out: dict[str, float] = {}
    for m in pat.finditer(log_text):
        out[m.group(1)] = float(m.group(2))
    return out


def _fmt(x: Any, nd: int = 4) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        s = f"{x:.{nd}f}".rstrip("0").rstrip(".")
        return s or "0"
    return str(x)


def write_eval_report_tables(
    output_dir: Path,
    *,
    log_path: Path | None = None,
) -> Path:
    """Write ``REPORT_TABLES.md`` under ``output_dir``. Returns the file path."""
    output_dir = output_dir.expanduser().resolve()
    e2e_path = output_dir / "e2e_report.json"
    summary_path = output_dir / "summary_report.json"
    if not e2e_path.is_file():
        raise FileNotFoundError(f"Missing {e2e_path}")
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing {summary_path}")

    e2e = json.loads(e2e_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    cases: list[dict[str, Any]] = e2e.get("cases") or []
    n = len(cases)
    if n == 0:
        raise ValueError("e2e_report.json has no cases")

    log_file = _resolve_log_path(output_dir, log_path)
    log_text = log_file.read_text(encoding="utf-8", errors="replace") if log_file else ""
    ragas_sec = _parse_ragas_durations(log_text, n) if log_text else {}

    lines: list[str] = []

    lines.append("# RAGAS E2E 评测报告（表格版）\n\n")
    lines.append("> 由 `eval_report_tables.py` / `regen_report_tables.py` 生成。  \n")
    if log_file:
        lines.append(f"> 终端日志：`{log_file}`  \n")
    else:
        lines.append("> 终端日志：未找到（RAGAS 耗时列将显示为 —；可传 `--log`）  \n")
    lines.append(f"> 数据目录：`{output_dir}/`  \n")
    lines.append(f"> 输出文件：`{output_dir}/REPORT_TABLES.md`\n\n")

    # --- 1. Overview ---
    lines.append("## 1. 总览\n\n")
    lines.append("| 指标 | 值 |\n| --- | --- |\n")
    lines.append(f"| Case 数 | {summary.get('case_count', n)} |\n")
    lines.append(f"| 成功 | {summary.get('success_count', '—')} |\n")
    lines.append(f"| Top hit rate | {summary.get('top_hit_rate', '—')} |\n")
    lines.append(f"| 图推理平均延迟 | {summary.get('avg_latency_ms', '—')} ms |\n")
    timing = (summary.get("runtime_profile") or {}).get("timing") or {}
    if timing:
        lines.append(f"| Wall 总耗时 | {float(timing.get('wall_total_ms', 0)) / 1000:.1f} s |\n")
        lines.append(f"| RAGAS 总耗时 | {float(timing.get('ragas_total_ms', 0)) / 1000:.1f} s |\n")
    err = summary.get("error_summary") or {}
    lines.append(f"| RAGAS metric 错误 case | {err.get('ragas_metric_error_cases', '—')} |\n")
    lines.append(f"| Graph 错误 case | {err.get('graph_error_cases', '—')} |\n\n")

    # --- 2. Summaries ---
    lines.append("## 2. 汇总指标\n\n")

    lines.append("### 2.1 检索（RAGAS LLM）\n\n")
    rs = summary.get("retrieval_summary") or {}
    lines.append("| 指标 | 均值 |\n| --- | --- |\n")
    lines.append(f"| context_precision_avg | {rs.get('context_precision_avg', '—')} |\n")
    lines.append(f"| context_recall_avg | {rs.get('context_recall_avg', '—')} |\n")
    lines.append(f"| reference_used_rich_count | {rs.get('reference_used_rich_count', '—')} / {n} |\n\n")

    lines.append("### 2.2 生成（RAGAS LLM：faithfulness / factual）\n\n")
    gs = summary.get("generation_summary") or {}
    lines.append("| 指标 | 均值 | 说明 |\n| --- | --- | --- |\n")
    lines.append(f"| faithfulness_avg | {gs.get('faithfulness_avg', '—')} | 答句与检索上下文一致性 |\n")
    lines.append(
        f"| factual_correctness_avg | {gs.get('factual_correctness_avg', '—')} | LLM 裁判，方差大，**仅作对照** |\n\n"
    )

    lines.append("### 2.3 自定义指标（P1-Next-C，规则型、0 LLM）\n\n")
    lines.append(
        "端到端合成 `ragas_e2e_score` 使用 **custom_correctness**，不使用 `factual_correctness`。"
        " 公式与分支含义见 `agent/challenge.md` §5.4。\n\n"
    )
    lines.append("| 指标 | 均值 | 说明 |\n| --- | --- | --- |\n")
    lines.append(
        f"| **custom_correctness_avg** | **{gs.get('custom_correctness_avg', '—')}** | yes/no + video + 时间 IoU（按期望标签分支加权） |\n"
    )
    lines.append(
        f"| factual_correctness_avg（对照） | {gs.get('factual_correctness_avg', '—')} | 同 2.2，与上对比可看 LLM 抖动 |\n\n"
    )

    # Branch distribution
    branches: list[str] = []
    for c in cases:
        gen = (c.get("ragas") or {}).get("generation") or {}
        det = gen.get("custom_correctness_detail") or {}
        b = det.get("branch")
        if b:
            branches.append(str(b))
    bc = Counter(branches)
    if bc:
        lines.append("**custom_correctness 分支分布**（`custom_correctness_detail.branch`）\n\n")
        lines.append("| 分支 | 条数 |\n| --- | ---: |\n")
        for name, cnt in sorted(bc.items(), key=lambda x: (-x[1], x[0])):
            lines.append(f"| {name} | {cnt} |\n")
        lines.append("\n")

    lines.append("### 2.4 任务原生时间 / 视频对齐（非 RAGAS）\n\n")
    ts = summary.get("temporal_summary") or {}
    ls = summary.get("localization_summary") or {}
    lines.append("| 指标 | 值 |\n| --- | --- |\n")
    lines.append(f"| time_range_overlap_iou_avg | {ts.get('time_range_overlap_iou_avg', '—')} |\n")
    lines.append(f"| IoU 有效 case 数 | {ts.get('time_range_overlap_iou_case_count', '—')} |\n")
    lines.append(f"| IoU hit@0.3 | {ts.get('time_range_overlap_iou_hit_rate_at_0_3', '—')} |\n")
    lines.append(f"| IoU hit@0.5 | {ts.get('time_range_overlap_iou_hit_rate_at_0_5', '—')} |\n")
    lines.append(
        f"| video_match_score_avg（top-1 视频） | {ls.get('video_match_score_avg', '—')} "
        f"(n={ls.get('video_match_case_count', '—')}) |\n"
    )
    lines.append(
        f"| localization_score_avg | {ls.get('localization_score_avg', '—')} "
        f"(n={ls.get('localization_case_count', '—')}) |\n\n"
    )

    lines.append("### 2.5 端到端（RAGAS 合成）\n\n")
    lines.append("| 指标 | 值 | 说明 |\n| --- | --- | --- |\n")
    e2s = summary.get("end_to_end_summary") or {}
    lines.append(
        f"| ragas_e2e_score_avg | {e2s.get('ragas_e2e_score_avg', '—')} | "
        "mean(ctx_p, ctx_r, **custom_correctness**, faithfulness) |\n\n"
    )

    # --- 3. Environment ---
    lines.append("## 3. 运行环境与子库\n\n")
    rp = (summary.get("runtime_profile") or {}).get("ragas_runtime") or {}
    ec = (summary.get("runtime_profile") or {}).get("eval_config") or {}
    lines.append("| 项 | 值 |\n| --- | --- |\n")
    lines.append(f"| RAGAS 模型 | {rp.get('ragas_model', '—')} |\n")
    emb = rp.get("agent_embedding") or {}
    lines.append(f"| Embedding | {emb.get('model', '—')} |\n")
    lines.append(f"| execution_mode | {rp.get('agent_execution_mode', '—')} |\n")
    lines.append(f"| AGENT_USE_LLAMAINDEX_SQL | {rp.get('agent_use_llamaindex_sql', '—')} |\n")
    lines.append(f"| AGENT_USE_LLAMAINDEX_VECTOR | {rp.get('agent_use_llamaindex_vector', '—')} |\n")
    lines.append(f"| SQLite | `{rp.get('agent_sqlite_db_path', '—')}` |\n")
    lines.append(f"| Chroma | `{rp.get('agent_chroma_path', '—')}` |\n")
    lines.append(f"| top_k | {ec.get('top_k', '—')} |\n")
    lines.append(f"| ragas_max_contexts | {ec.get('ragas_max_contexts', '—')} |\n")
    lines.append(f"| ragas_max_total_context_chars | {ec.get('ragas_max_total_context_chars', '—')} |\n\n")

    boot = summary.get("bootstrap") or {}
    if boot:
        lines.append("### 子库构建（bootstrap）\n\n")
        sqlite = boot.get("sqlite") or {}
        chroma = boot.get("chroma") or {}
        lines.append("| 项 | 值 |\n| --- | --- |\n")
        lines.append(f"| SQLite rows | {sqlite.get('inserted_rows', '—')} |\n")
        lines.append(
            f"| Chroma child / parent / event | "
            f"{chroma.get('child_record_count', '—')} / "
            f"{chroma.get('parent_record_count', '—')} / "
            f"{chroma.get('event_record_count', '—')} |\n"
        )
        seeds = sqlite.get("seed_files") or []
        lines.append(f"| Seed 文件数 | {len(seeds)} |\n\n")

    # --- 4. Graph phase ---
    lines.append("## 4. 图执行阶段（按出题顺序）\n\n")
    lines.append("| # | Case ID | Classifier 标签 | Answer type | 题干摘要 |\n| --- | --- | --- | --- | --- |\n")
    for i, c in enumerate(cases, 1):
        cid = c.get("case_id", "—")
        cr = c.get("classification_result") or {}
        label = cr.get("label") or "—"
        at = c.get("answer_type") or "—"
        q = str(c.get("question") or "")
        if len(q) > 90:
            q = q[:88] + "…"
        lines.append(f"| {i} | `{cid}` | {label} | {at} | {q} |\n")

    # --- 5. RAGAS wall times ---
    lines.append("\n## 5. RAGAS 评分阶段（日志 wall 耗时）\n\n")
    lines.append("| Case ID | RAGAS 耗时 (s) |\n| --- | --- |\n")
    for c in cases:
        cid = c.get("case_id", "")
        sec = ragas_sec.get(cid)
        lines.append(f"| `{cid}` | {_fmt(sec, 1) if sec is not None else '—'} |\n")

    # --- 6. Custom detail per case ---
    lines.append("\n## 6. 自定义指标明细（逐 case）\n\n")
    lines.append(
        "| Case | 期望标签 | 预测标签 | yes_no | video | time_IoU | time_term | custom | 分支 |\n"
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |\n"
    )
    for c in cases:
        cid = c.get("case_id", "")
        gen = (c.get("ragas") or {}).get("generation") or {}
        det = gen.get("custom_correctness_detail") or {}
        exp_l = det.get("expected_label")
        pred_l = det.get("predicted_label")
        lines.append(
            f"| `{cid}` | {exp_l or '—'} | {pred_l or '—'} | "
            f"{_fmt(det.get('yes_no_score'))} | {_fmt(det.get('video_id_score'))} | "
            f"{_fmt(det.get('time_iou_score'))} | {_fmt(det.get('time_term'))} | "
            f"{_fmt(gen.get('custom_correctness'))} | {det.get('branch') or '—'} |\n"
        )

    # --- 7. Full metrics table ---
    lines.append("\n## 7. RAGAS + 自定义 + 任务原生（宽表）\n\n")
    lines.append(
        "| Case | GT video | 期望 | Route | Graph ms | RAGAS s | Top-1 | ctx_p | ctx_r | faith | factual | **custom** | e2e | IoU | v_match | loc |\n"
        "| --- | --- | --- | --- | ---: | ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
    )
    for c in cases:
        cid = c.get("case_id", "")
        vid = c.get("video_id") or "—"
        exp = c.get("expected_answer_label") or "—"
        route = c.get("route_mode") or "—"
        gms = c.get("elapsed_ms")
        rs = ragas_sec.get(str(cid))
        th = "✓" if c.get("top_hit") else "×"
        r = c.get("ragas") or {}
        ret = r.get("retrieval") or {}
        gen = r.get("generation") or {}
        e2 = r.get("end_to_end") or {}
        t = c.get("temporal") or {}
        iou = t.get("time_range_overlap_iou")
        vm = t.get("video_match_score")
        loc = t.get("localization_score")
        lines.append(
            f"| `{cid}` | `{vid}` | {exp} | {route} | {_fmt(gms, 2)} | "
            f"{_fmt(rs, 1) if rs is not None else '—'} | {th} | "
            f"{_fmt(ret.get('context_precision'))} | {_fmt(ret.get('context_recall'))} | "
            f"{_fmt(gen.get('faithfulness'))} | {_fmt(gen.get('factual_correctness'))} | "
            f"**{_fmt(gen.get('custom_correctness'))}** | {_fmt(e2.get('ragas_e2e_score'))} | "
            f"{_fmt(iou) if iou is not None else '—'} | {_fmt(vm) if vm is not None else '—'} | "
            f"{_fmt(loc) if loc is not None else '—'} |\n"
        )

    lines.append("\n## 8. 模型最终回答（摘要）\n\n")
    lines.append("| Case | 最终回答（不含 Sources） |\n| --- | --- |\n")
    for c in cases:
        cid = c.get("case_id", "")
        resp = str(c.get("response") or "").replace("\n", " ").strip()
        if len(resp) > 200:
            resp = resp[:198] + "…"
        lines.append(f"| `{cid}` | {resp} |\n")

    lines.append("\n## 9. Verifier 摘要\n\n")
    lines.append("| Case | decision | span_source | video_id (verifier) |\n| --- | --- | --- | --- |\n")
    for c in cases:
        cid = c.get("case_id", "")
        vr = c.get("verifier_result") or {}
        if not vr:
            lines.append(f"| `{cid}` | — | — | — |\n")
            continue
        lines.append(
            f"| `{cid}` | {vr.get('decision', '—')} | {vr.get('span_source', '—')} | `{vr.get('video_id', '—')}` |\n"
        )

    out = output_dir / "REPORT_TABLES.md"
    out.write_text("".join(lines), encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate REPORT_TABLES.md for a ragas_eval_runner output directory."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Run output directory containing e2e_report.json and summary_report.json",
    )
    parser.add_argument(
        "--log",
        type=str,
        default="",
        help="Optional path to tee log (default: <parent>/<dirname>.log)",
    )
    args = parser.parse_args()
    log_p = Path(args.log).expanduser().resolve() if str(args.log).strip() else None
    path = write_eval_report_tables(Path(args.output_dir), log_path=log_p)
    print(f"[eval_report_tables] Wrote {path}")


if __name__ == "__main__":
    main()
