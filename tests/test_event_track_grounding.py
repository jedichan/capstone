"""隔离评估 event_track 层的时间精度, 以及 LLM 时间戳是否"扎根"于 YOLO.

三组对比 (针对 UCA Test 上的同 46 个视频):
  A. LLM raw      : result_uca_46_v2.json 里 LLM 直接产出的 timestamps vs GT
  B. YOLO clips   : _pipeline_output/uca_eval/*_clips.json 的 clip_segments vs GT
  C. YOLO events  : _pipeline_output/uca_eval/*_events.json 的 per-track events vs GT
  D. LLM snapped  : 把 LLM 每个 [s,e] 吸附到最近的 YOLO event 边界 vs GT

每组都算: mean tIoU / Recall@{0.3,0.5,0.7,0.9} / boundary-F1
对 LLM raw 额外算: 每个 LLM 时间戳到最近 YOLO event 边界的 |dt|, 以判断 LLM 是否在编时间

结论用途:
  - B/C 是你 event_track 层的**纯时间精度** (LLM 完全不参与)
  - A 是 LLM-rewritten 时间精度
  - D 是 "按你项目设计理念 (LLM 扎根于 tracker) 应该的成绩"
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

GT_PATH = PROJECT_ROOT / "_data" / "Surveillance-Video-Understanding" / "UCF Annotation" / "json" / "UCFCrime_Test.json"
YOLO_DIR = PROJECT_ROOT / "_pipeline_output" / "uca_eval"
LLM_V2 = PROJECT_ROOT / "result_uca_46_v2.json"


def tiou(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def greedy_max_iou(pred_ts, gt_ts):
    """对每个 GT 贪心选 tIoU 最大的 pred; 每个 pred 只能用一次."""
    used = set()
    out = []
    for g in gt_ts:
        best, bi = 0.0, -1
        for i, p in enumerate(pred_ts):
            if i in used:
                continue
            v = tiou(p, g)
            if v > best:
                best, bi = v, i
        if bi >= 0:
            used.add(bi)
        out.append(best)
    return out


def metrics_from_ious(ious_per_gt, n_pred):
    """返回 (mean_tIoU, R@0.3/0.5/0.7/0.9, boundary-F1 at 0.5)."""
    n_gt = len(ious_per_gt)
    if n_gt == 0:
        return {"mean_tIoU": 0.0, "R@0.3": 0, "R@0.5": 0, "R@0.7": 0, "R@0.9": 0, "F1@0.5": 0}
    m = {"mean_tIoU": round(mean(ious_per_gt), 4)}
    for tau in (0.3, 0.5, 0.7, 0.9):
        m[f"R@{tau}"] = round(sum(1 for i in ious_per_gt if i >= tau) / n_gt, 4)
    # boundary F1 at τ=0.5: TP = matched at ≥0.5, P = TP / n_pred, R = TP / n_gt
    tp = sum(1 for i in ious_per_gt if i >= 0.5)
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gt
    m["F1@0.5"] = round(2 * p * r / (p + r), 4) if (p + r) else 0.0
    m["n_pred"] = n_pred
    m["n_gt"] = n_gt
    return m


def yolo_events_for(video: str) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """返回 (per-track events timestamps, clip_segments timestamps)."""
    ev = json.loads((YOLO_DIR / f"{video}_events.json").read_text(encoding="utf-8"))
    cl = json.loads((YOLO_DIR / f"{video}_clips.json").read_text(encoding="utf-8"))
    ev_ts = [(float(e["start_time"]), float(e["end_time"]))
             for e in ev["events"] if e["end_time"] > e["start_time"]]
    cl_ts = [(float(c["start_sec"]), float(c["end_sec"]))
             for c in cl["clip_segments"] if c["end_sec"] > c["start_sec"]]
    return ev_ts, cl_ts


def snap_to_boundaries(ts: tuple[float, float], yolo_events: list[tuple[float, float]]) -> tuple[float, float]:
    """把一个 LLM timestamp [s,e] 吸附到最近的 YOLO event 边界."""
    if not yolo_events:
        return ts
    all_bounds = sorted(set([e[0] for e in yolo_events] + [e[1] for e in yolo_events]))
    def closest(x):
        return min(all_bounds, key=lambda b: abs(b - x))
    s2, e2 = closest(ts[0]), closest(ts[1])
    if s2 >= e2:  # 保险: 非法就退回原值
        return ts
    return (s2, e2)


def llm_boundary_distance(llm_ts: list[tuple[float, float]],
                          yolo_events: list[tuple[float, float]]) -> float:
    """LLM 每个时间戳的边界到最近 YOLO event 边界的平均距离 (秒)."""
    if not yolo_events or not llm_ts:
        return float("nan")
    bounds = sorted(set([e[0] for e in yolo_events] + [e[1] for e in yolo_events]))
    ds = []
    for s, e in llm_ts:
        ds.append(min(abs(b - s) for b in bounds))
        ds.append(min(abs(b - e) for b in bounds))
    return sum(ds) / len(ds)


# --------------------------------------------------------------------------- #
def main() -> int:
    gt_all = json.loads(GT_PATH.read_text(encoding="utf-8"))
    llm = json.loads(LLM_V2.read_text(encoding="utf-8"))

    videos = [r for r in llm["per_video"] if "error" not in r and (YOLO_DIR / f"{r['video']}_events.json").exists()]
    print(f"[INFO] evaluating {len(videos)} videos")
    print()

    # 聚合
    rows_A, rows_B, rows_C, rows_D = [], [], [], []
    total_pred = {"A": 0, "B": 0, "C": 0, "D": 0}
    total_gt = 0
    snap_distances_avg = []

    per_video_rows = []

    for r in videos:
        name = r["video"]
        gt = gt_all[name]
        gt_ts = [(float(a), float(b)) for a, b in gt["timestamps"]]
        total_gt += len(gt_ts)

        # A: LLM raw
        llm_ts = [(float(a), float(b)) for a, b in (r["pred"].get("timestamps") or [])]
        iou_A = greedy_max_iou(llm_ts, gt_ts)
        rows_A.extend(iou_A); total_pred["A"] += len(llm_ts)

        # B + C: YOLO
        ev_ts, cl_ts = yolo_events_for(name)
        iou_B = greedy_max_iou(cl_ts, gt_ts)
        rows_B.extend(iou_B); total_pred["B"] += len(cl_ts)
        iou_C = greedy_max_iou(ev_ts, gt_ts)
        rows_C.extend(iou_C); total_pred["C"] += len(ev_ts)

        # D: LLM snapped to YOLO events
        snapped = [snap_to_boundaries(t, ev_ts) for t in llm_ts] if ev_ts else llm_ts
        iou_D = greedy_max_iou(snapped, gt_ts)
        rows_D.extend(iou_D); total_pred["D"] += len(snapped)

        # 距离
        d = llm_boundary_distance(llm_ts, ev_ts)
        if d == d:  # not nan
            snap_distances_avg.append(d)

        per_video_rows.append({
            "video": name, "n_gt": len(gt_ts),
            "A_llm_tIoU": round(mean(iou_A) if iou_A else 0, 3),
            "B_clips_tIoU": round(mean(iou_B) if iou_B else 0, 3),
            "C_events_tIoU": round(mean(iou_C) if iou_C else 0, 3),
            "D_snap_tIoU": round(mean(iou_D) if iou_D else 0, 3),
            "avg_bound_dist_s": round(d, 2) if d == d else None,
        })

    # 汇总
    mA = metrics_from_ious(rows_A, total_pred["A"])
    mB = metrics_from_ious(rows_B, total_pred["B"])
    mC = metrics_from_ious(rows_C, total_pred["C"])
    mD = metrics_from_ious(rows_D, total_pred["D"])

    print("=" * 84)
    print(f"{'source':<30}{'mean_tIoU':>11}{'R@0.3':>8}{'R@0.5':>8}{'R@0.7':>8}{'R@0.9':>8}{'F1@0.5':>9}{'n_pred':>9}")
    print("-" * 84)
    def prow(lbl, m):
        print(f"  {lbl:<28}{m['mean_tIoU']:>11.4f}{m['R@0.3']:>8.3f}{m['R@0.5']:>8.3f}"
              f"{m['R@0.7']:>8.3f}{m['R@0.9']:>8.3f}{m['F1@0.5']:>9.4f}{m['n_pred']:>9d}")
    prow("A. LLM raw (as-is)", mA)
    prow("B. YOLO clip_segments", mB)
    prow("C. YOLO per-track events", mC)
    prow("D. LLM snapped->YOLO", mD)
    print("=" * 84)
    print()
    print(f"n_gt total = {total_gt}")
    print(f"LLM timestamp avg distance to nearest YOLO event boundary = {mean(snap_distances_avg):.2f} s "
          f"(over {len(snap_distances_avg)} videos)")
    print()
    print("Interpretation:")
    print("  - B / C = pure event_track temporal precision (no LLM).")
    print("  - A vs D: if D significantly beats A, LLM was inventing times and should be")
    print("             constrained to snap to tracker boundaries.")
    print("  - If A ≈ D: LLM is already respecting tracker timeline.")

    out = {
        "A_llm_raw": mA, "B_yolo_clips": mB, "C_yolo_events": mC, "D_llm_snapped": mD,
        "avg_boundary_distance_sec": round(mean(snap_distances_avg), 3),
        "n_videos": len(videos), "n_gt_events": total_gt,
        "per_video": per_video_rows,
    }
    (PROJECT_ROOT / "result_event_track_grounding.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[saved] result_event_track_grounding.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
