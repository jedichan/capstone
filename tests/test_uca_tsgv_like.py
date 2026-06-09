"""TSGV-like evaluation derived from DVC output, to allow diff vs UCA paper Table 4.

For each GT (video, sentence_query):
  - rank all LLM predictions of that video by METEOR(pred_sentence, sentence_query)
  - take top-n (n=1, 5)
  - R@n-IoU=tau = 1 if max IoU(pred_interval, GT_interval) over top-n >= tau else 0

NOTE: This is a DVC->TSGV adaptation. The paper's Table 4 trains TSGV-specific models
      (one query -> one moment). Our system solves DVC (video -> set of (moment, sentence)).
      Numbers are not strictly apples-to-apples; we report the mapping honestly.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import nltk
for _p in ("wordnet", "omw-1.4", "punkt", "punkt_tab"):
    try: nltk.download(_p, quiet=True)
    except Exception: pass
from nltk.translate.meteor_score import meteor_score as nltk_meteor
from nltk.tokenize import word_tokenize


def _tok(s: str):
    try:
        return [w.lower() for w in word_tokenize(s) if any(c.isalnum() for c in w)]
    except Exception:
        return [w.lower() for w in s.split() if w]

def meteor(p, g):
    return float(nltk_meteor([_tok(g)], _tok(p))) if p and g else 0.0

def tiou(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def main():
    data = json.loads((PROJECT_ROOT / "result_uca_46_v2.json").read_text(encoding="utf-8"))
    gt_all = json.loads((PROJECT_ROOT / "_data/Surveillance-Video-Understanding/UCF Annotation/json/UCFCrime_Test.json").read_text(encoding="utf-8"))

    taus = [0.3, 0.5, 0.7]
    ns = [1, 5]
    hit = {(n, t): 0 for n in ns for t in taus}
    n_queries = 0

    per_video = []
    for r in data["per_video"]:
        if "error" in r or "pred" not in r: continue
        name = r["video"]
        g = gt_all[name]
        g_ts = [tuple(map(float, x)) for x in g["timestamps"]]
        g_sent = list(g["sentences"])
        p_ts = [tuple(map(float, x)) for x in (r["pred"].get("timestamps") or [])]
        p_sent = list(r["pred"].get("sentences") or [])
        m = min(len(p_ts), len(p_sent))
        p_ts, p_sent = p_ts[:m], p_sent[:m]
        if not p_ts:
            n_queries += len(g_ts)
            continue
        for qi, (gq_ts, gq_s) in enumerate(zip(g_ts, g_sent)):
            n_queries += 1
            sims = [meteor(ps, gq_s) for ps in p_sent]
            order = sorted(range(len(p_ts)), key=lambda i: sims[i], reverse=True)
            ious_sorted = [tiou(p_ts[i], gq_ts) for i in order]
            for n in ns:
                topn_max = max(ious_sorted[:n]) if ious_sorted else 0.0
                for t in taus:
                    if topn_max >= t:
                        hit[(n, t)] += 1

    print(f"n_queries = {n_queries}")
    print(f"{'':<8}{'IoU=0.3':>20}{'IoU=0.5':>20}{'IoU=0.7':>20}")
    print(f"{'n':<8}{'R@1':>10}{'R@5':>10}{'R@1':>10}{'R@5':>10}{'R@1':>10}{'R@5':>10}")
    row = ["Ours"]
    for t in taus:
        for n in ns:
            row.append(f"{100*hit[(n,t)]/n_queries:.2f}")
    print("  ".join(row))

    out = {
        "n_queries": n_queries,
        "metrics_pct": {
            f"IoU={t}_R@{n}": round(100 * hit[(n, t)] / n_queries, 2)
            for t in taus for n in ns
        },
    }
    (PROJECT_ROOT / "result_uca_46_v2_tsgv_like.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[saved] result_uca_46_v2_tsgv_like.json")

if __name__ == "__main__":
    raise SystemExit(main())
