"""读取 test_uca_full_evaluation 的输出 JSON, 打印人类可读的 UCA 评测报告."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("result", nargs="?", default="result_uca_46.json")
    args = ap.parse_args()

    p = Path(args.result)
    if not p.exists():
        print(f"[ERROR] {p} not found", file=sys.stderr)
        return 1
    data = json.loads(p.read_text(encoding="utf-8"))
    agg = data["aggregate"]
    per = data["per_video"]
    model = data.get("model", "?")
    frames = data.get("frames_per_video", "?")

    print("=" * 72)
    print(f"UCA Evaluation Report  |  model={model}  frames/video={frames}")
    print("=" * 72)
    print(f"Videos           : {agg['n_total']}  (ok={agg['n_ok']}  failed={agg['n_failed']})")
    print(f"Wall time        : {agg.get('elapsed_sec', 0):.1f} s")
    print()
    print("Metrics (mean over successful videos)")
    print("-" * 72)
    for k in ("mean_tIoU", "recall@0.3", "recall@0.5", "recall@0.7", "mean_tokenF1"):
        v = agg.get(k)
        print(f"  {k:<14} = {v}")
    print()
    tu = agg["token_usage"]
    print("Token usage (cumulative)")
    print("-" * 72)
    print(f"  prompt_tokens     : {tu['prompt_tokens']:>10,}")
    print(f"  completion_tokens : {tu['completion_tokens']:>10,}")
    print(f"  total_tokens      : {tu['total_tokens']:>10,}")
    n_ok = max(agg["n_ok"], 1)
    print(f"  avg / video       : {tu['total_tokens']/n_ok:>10,.0f}")
    print(f"  estimated cost    : CNY {agg['estimated_cost_cny']}")
    print()

    # top / bottom
    ok = sorted([r for r in per if "error" not in r], key=lambda r: r.get("mean_tIoU", 0), reverse=True)
    print("Top 5 by tIoU")
    print("-" * 72)
    for r in ok[:5]:
        print(f"  {r['video']:<20}  tIoU={r['mean_tIoU']}  R@0.5={r['recall@0.5']}  "
              f"tokF1={r['mean_tokenF1']}  tokens={r['usage']['total_tokens']}")
    print("Bottom 5 by tIoU")
    print("-" * 72)
    for r in ok[-5:]:
        print(f"  {r['video']:<20}  tIoU={r['mean_tIoU']}  R@0.5={r['recall@0.5']}  "
              f"tokF1={r['mean_tokenF1']}  tokens={r['usage']['total_tokens']}")
    print()
    failed = [r for r in per if "error" in r]
    if failed:
        print(f"Failed videos ({len(failed)}):")
        for r in failed:
            print(f"  {r['video']:<20}  {r['error']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
