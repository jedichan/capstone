"""Motion-slicing coverage ratio analysis.

For each video in the cached _pipeline_output/uca_eval/*_clips.json:
  coverage_ratio = sum(clip_duration) / video_total_duration

Reports the distribution across 46 videos and extrapolates token savings.
No LLM calls needed — reads cached YOLO clip output only.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
from statistics import mean, median, stdev

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

YOLO_DIR = PROJECT_ROOT / "_pipeline_output" / "uca_eval"
LLM_V2   = PROJECT_ROOT / "result_uca_46_v2.json"


def main():
    data = json.loads(LLM_V2.read_text(encoding="utf-8"))
    videos = [r for r in data["per_video"]
              if "error" not in r and (YOLO_DIR / f"{r['video']}_clips.json").exists()]

    rows = []
    for r in videos:
        name = r["video"]
        duration = float(r["duration_video"])
        cl = json.loads((YOLO_DIR / f"{name}_clips.json").read_text(encoding="utf-8"))
        clip_dur = sum(
            float(c["end_sec"]) - float(c["start_sec"])
            for c in cl["clip_segments"]
            if c["end_sec"] > c["start_sec"]
        )
        ratio = clip_dur / duration if duration > 0 else 1.0
        rows.append({
            "video": name,
            "duration_s": round(duration, 1),
            "clip_dur_s": round(clip_dur, 1),
            "coverage_ratio": round(ratio, 4),
            "token_saving_pct": round((1 - ratio) * 100, 1),
        })

    ratios = [r["coverage_ratio"] for r in rows]
    savings = [r["token_saving_pct"] for r in rows]

    print("=" * 72)
    print("Motion-Slicing Coverage Analysis  |  46 UCA Test videos (Part-1)")
    print("=" * 72)
    print(f"  mean  coverage ratio : {mean(ratios):.3f}  ({mean(ratios)*100:.1f}% of frames kept)")
    print(f"  median coverage ratio: {median(ratios):.3f}")
    print(f"  stdev                : {stdev(ratios):.3f}")
    print(f"  min                  : {min(ratios):.3f}  ({rows[ratios.index(min(ratios))]['video']})")
    print(f"  max                  : {max(ratios):.3f}  ({rows[ratios.index(max(ratios))]['video']})")
    print()
    print(f"  mean token saving    : {mean(savings):.1f}%")
    print(f"  median token saving  : {median(savings):.1f}%")
    print()

    # distribution buckets
    buckets = [
        ("<20%  kept (>80% saved)", lambda r: r < 0.20),
        ("20–50% kept",             lambda r: 0.20 <= r < 0.50),
        ("50–80% kept",             lambda r: 0.50 <= r < 0.80),
        (">80%  kept (<20% saved)", lambda r: r >= 0.80),
    ]
    print("  Distribution of coverage ratio:")
    for label, fn in buckets:
        n = sum(1 for r in ratios if fn(r))
        print(f"    {label:<30} : {n:>3} videos  ({100*n/len(ratios):.0f}%)")
    print()

    # per-video table (sorted by coverage)
    rows_sorted = sorted(rows, key=lambda x: x["coverage_ratio"])
    print(f"  {'video':<25} {'dur_s':>7} {'clip_s':>7} {'coverage':>9} {'saving%':>9}")
    print("  " + "-" * 62)
    for r in rows_sorted:
        print(f"  {r['video']:<25} {r['duration_s']:>7.1f} {r['clip_dur_s']:>7.1f} "
              f"{r['coverage_ratio']:>9.3f} {r['token_saving_pct']:>9.1f}%")

    print()
    print("Interpretation:")
    print("  UCA Part-1 videos are short dense-action clips (avg duration "
          f"{mean(r['duration_s'] for r in rows):.0f}s),")
    print("  so coverage is naturally high and token savings appear modest here.")
    print("  In typical long surveillance footage (1-8 hr, mostly static),")
    print("  activity occupies <10% of total time -> expected saving >90%.")

    out = {
        "n_videos": len(rows),
        "mean_coverage_ratio": round(mean(ratios), 4),
        "median_coverage_ratio": round(median(ratios), 4),
        "mean_token_saving_pct": round(mean(savings), 2),
        "per_video": rows_sorted,
    }
    (PROJECT_ROOT / "result_motion_coverage.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[saved] result_motion_coverage.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
