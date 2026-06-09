"""UCA 论文口径的评测: BLEU-4 / METEOR / CIDEr / ROUGE-L + DVC P/R/F1 + SODA_c.

复用 result_uca_46.json 里已缓存的 LLM 预测（prompt/model 没变, 重跑 LLM 不会变分数,
只会多花 token）. 运行:

    python tests/test_uca_coco_metrics.py --result result_uca_46.json

指标
----
1. Paragraph-level (把每条视频所有事件拼成一段再打分):
     BLEU-1..4, METEOR, ROUGE-L, CIDEr
2. Dense Video Captioning (DVC) paper-style:
     对 tIoU 阈值 τ in {0.3, 0.5, 0.7, 0.9}:
       - 贪心二分匹配: 选 tIoU >= τ 的最优一对一配对
       - 在配对上算 BLEU-4 / METEOR / CIDEr / ROUGE-L
       - Precision = 命中 / |pred| , Recall = 命中 / |gt| , F1
3. SODA_c (Fujita et al. 2020 简化版):
     sim[i][j] = tIoU(pred_i, gt_j) * METEOR(pred_i, gt_j)
     LCS 式 DP 求最大单调匹配和 -> Precision/Recall/F1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---- scorers
from pycocoevalcap.bleu.bleu import Bleu  # noqa: E402
from pycocoevalcap.cider.cider import Cider  # noqa: E402
from pycocoevalcap.rouge.rouge import Rouge  # noqa: E402
from nltk.translate.meteor_score import meteor_score as nltk_meteor  # noqa: E402
from nltk.tokenize import word_tokenize  # noqa: E402
import nltk  # noqa: E402
for _pkg in ("wordnet", "omw-1.4", "punkt", "punkt_tab"):
    try:
        nltk.download(_pkg, quiet=True)
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# util                                                                         #
# --------------------------------------------------------------------------- #
def _tok(s: str) -> list[str]:
    try:
        return [w.lower() for w in word_tokenize(s) if any(c.isalnum() for c in w)]
    except Exception:
        return [w.lower() for w in s.split() if w]


def meteor(pred: str, gt: str) -> float:
    return float(nltk_meteor([_tok(gt)], _tok(pred)))


def tiou(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def _coco_score(pred_by_id: dict[str, list[str]], gt_by_id: dict[str, list[str]]) -> dict[str, float]:
    """pycocoevalcap 的 BLEU/CIDEr/ROUGE; METEOR 用 nltk 逐对平均."""
    if not pred_by_id:
        return {"BLEU-1": 0.0, "BLEU-2": 0.0, "BLEU-3": 0.0, "BLEU-4": 0.0,
                "METEOR": 0.0, "ROUGE-L": 0.0, "CIDEr": 0.0}
    bleu = Bleu(4).compute_score(gt_by_id, pred_by_id)[0]
    cider = Cider().compute_score(gt_by_id, pred_by_id)[0]
    rouge = Rouge().compute_score(gt_by_id, pred_by_id)[0]
    met_vals = []
    for k in pred_by_id:
        met_vals.append(meteor(pred_by_id[k][0], gt_by_id[k][0]))
    met = sum(met_vals) / len(met_vals)
    return {
        "BLEU-1": float(bleu[0]), "BLEU-2": float(bleu[1]),
        "BLEU-3": float(bleu[2]), "BLEU-4": float(bleu[3]),
        "METEOR": float(met), "ROUGE-L": float(rouge), "CIDEr": float(cider),
    }


# --------------------------------------------------------------------------- #
# DVC paired metrics                                                           #
# --------------------------------------------------------------------------- #
def greedy_bipartite(
    pred_ts: list[tuple[float, float]], gt_ts: list[tuple[float, float]], threshold: float
) -> list[tuple[int, int, float]]:
    """从大到小贪心选 tIoU >= threshold 的 (pred_i, gt_j) 对, 每个 i/j 只能用一次."""
    candidates: list[tuple[float, int, int]] = []
    for i, p in enumerate(pred_ts):
        for j, g in enumerate(gt_ts):
            v = tiou(p, g)
            if v >= threshold:
                candidates.append((v, i, j))
    candidates.sort(reverse=True)
    used_p: set[int] = set()
    used_g: set[int] = set()
    out: list[tuple[int, int, float]] = []
    for v, i, j in candidates:
        if i in used_p or j in used_g:
            continue
        used_p.add(i); used_g.add(j)
        out.append((i, j, v))
    return out


def dvc_at_threshold(
    per_video: list[dict[str, Any]], tau: float
) -> dict[str, float]:
    """对所有视频聚合: 取 tIoU>=tau 的配对, 再算 BLEU/METEOR/CIDEr/ROUGE 以及 P/R/F1."""
    pred_map: dict[str, list[str]] = {}
    gt_map: dict[str, list[str]] = {}
    n_matched = 0
    n_pred = 0
    n_gt = 0
    uid = 0
    for rec in per_video:
        if "pred" not in rec:
            continue
        pred = rec["pred"]
        p_ts = [tuple(map(float, x)) for x in pred.get("timestamps") or []]
        p_sent = list(pred.get("sentences") or [])
        g_ts = [tuple(map(float, x)) for x in rec["gt_timestamps"]]
        g_sent = list(rec["gt_sentences"])
        n_pred += len(p_ts)
        n_gt += len(g_ts)
        if not p_ts or not g_ts:
            continue
        m = min(len(p_sent), len(p_ts))
        p_ts = p_ts[:m]; p_sent = p_sent[:m]
        for i, j, _ in greedy_bipartite(p_ts, g_ts, tau):
            key = f"v{rec['video']}_{uid}"; uid += 1
            pred_map[key] = [p_sent[i]]
            gt_map[key] = [g_sent[j]]
            n_matched += 1
    cap = _coco_score(pred_map, gt_map)
    precision = n_matched / n_pred if n_pred else 0.0
    recall = n_matched / n_gt if n_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {**cap, "precision": precision, "recall": recall, "f1": f1,
            "n_matched": n_matched, "n_pred": n_pred, "n_gt": n_gt}


# --------------------------------------------------------------------------- #
# SODA_c                                                                       #
# --------------------------------------------------------------------------- #
def _lcs_max_sum(M: list[list[float]]) -> float:
    """LCS 式单调匹配最大化 sum M[i][j]; i,j 严格递增."""
    if not M:
        return 0.0
    N = len(M); K = len(M[0])
    dp = [[0.0] * (K + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(1, K + 1):
            dp[i][j] = max(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1] + M[i - 1][j - 1])
    return dp[N][K]


def soda_c_one(pred_ts, pred_sent, gt_ts, gt_sent) -> float:
    if not pred_ts or not gt_ts:
        return 0.0
    # 以时间中心排序保证单调匹配意义
    p_idx = sorted(range(len(pred_ts)), key=lambda i: (pred_ts[i][0] + pred_ts[i][1]) / 2)
    g_idx = sorted(range(len(gt_ts)), key=lambda j: (gt_ts[j][0] + gt_ts[j][1]) / 2)
    M = []
    for i in p_idx:
        row = []
        for j in g_idx:
            row.append(tiou(pred_ts[i], gt_ts[j]) * meteor(pred_sent[i], gt_sent[j]))
        M.append(row)
    s = _lcs_max_sum(M)
    P = s / len(pred_ts); R = s / len(gt_ts)
    return 2 * P * R / (P + R) if (P + R) > 0 else 0.0


# --------------------------------------------------------------------------- #
# paragraph-level                                                              #
# --------------------------------------------------------------------------- #
def paragraph_metrics(per_video: list[dict[str, Any]]) -> dict[str, float]:
    pred_map: dict[str, list[str]] = {}
    gt_map: dict[str, list[str]] = {}
    for rec in per_video:
        if "pred" not in rec:
            continue
        p_sent = " ".join(rec["pred"].get("sentences") or [])
        g_sent = " ".join(rec["gt_sentences"])
        if p_sent and g_sent:
            pred_map[rec["video"]] = [p_sent]
            gt_map[rec["video"]] = [g_sent]
    return _coco_score(pred_map, gt_map)


# --------------------------------------------------------------------------- #
# main                                                                         #
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", default="result_uca_46.json")
    ap.add_argument("--gt",
                    default="_data/Surveillance-Video-Understanding/UCF Annotation/json/UCFCrime_Test.json")
    ap.add_argument("--out", default="result_uca_46_cocometrics.json")
    args = ap.parse_args()

    data = json.loads((PROJECT_ROOT / args.result).read_text(encoding="utf-8"))
    gt_all = json.loads((PROJECT_ROOT / args.gt).read_text(encoding="utf-8"))

    # 把 gt 挂到每个视频记录上
    per: list[dict[str, Any]] = []
    for r in data["per_video"]:
        if "error" in r:
            continue
        g = gt_all[r["video"]]
        rr = dict(r)
        rr["gt_timestamps"] = g["timestamps"]
        rr["gt_sentences"] = g["sentences"]
        per.append(rr)

    print(f"[INFO] scoring {len(per)} videos from {args.result}")
    print(f"[INFO] model = {data.get('model')}  frames/video = {data.get('frames_per_video')}")
    print()

    # 1) paragraph-level
    print("=" * 72)
    print("Paragraph-level captioning  (per-video concat)")
    print("-" * 72)
    para = paragraph_metrics(per)
    for k, v in para.items():
        print(f"  {k:<10} = {v:.4f}")
    print()

    # 2) DVC at τ
    print("=" * 72)
    print("Dense Video Captioning (paper-style)  |  tIoU threshold sweep")
    print("-" * 72)
    dvc_rows = {}
    header = f"  {'τ':<6}{'BLEU-4':>9}{'METEOR':>9}{'ROUGE-L':>9}{'CIDEr':>9}" \
             f"{'P':>8}{'R':>8}{'F1':>8}{'n_match':>9}"
    print(header)
    for tau in (0.3, 0.5, 0.7, 0.9):
        m = dvc_at_threshold(per, tau)
        dvc_rows[str(tau)] = m
        print(f"  {tau:<6}{m['BLEU-4']:>9.4f}{m['METEOR']:>9.4f}{m['ROUGE-L']:>9.4f}"
              f"{m['CIDEr']:>9.4f}{m['precision']:>8.3f}{m['recall']:>8.3f}"
              f"{m['f1']:>8.3f}{m['n_matched']:>9d}")
    avg = {k: sum(dvc_rows[t][k] for t in dvc_rows) / len(dvc_rows)
           for k in ("BLEU-4", "METEOR", "ROUGE-L", "CIDEr", "precision", "recall", "f1")}
    print(f"  {'avg':<6}{avg['BLEU-4']:>9.4f}{avg['METEOR']:>9.4f}{avg['ROUGE-L']:>9.4f}"
          f"{avg['CIDEr']:>9.4f}{avg['precision']:>8.3f}{avg['recall']:>8.3f}{avg['f1']:>8.3f}")
    print()

    # 3) SODA_c
    print("=" * 72)
    print("SODA_c (monotonic alignment, sim = tIoU × METEOR)")
    print("-" * 72)
    soda_scores = []
    for r in per:
        p_ts = [tuple(map(float, x)) for x in r["pred"].get("timestamps") or []]
        p_sent = list(r["pred"].get("sentences") or [])
        g_ts = [tuple(map(float, x)) for x in r["gt_timestamps"]]
        g_sent = list(r["gt_sentences"])
        m = min(len(p_ts), len(p_sent))
        p_ts = p_ts[:m]; p_sent = p_sent[:m]
        soda_scores.append(soda_c_one(p_ts, p_sent, g_ts, g_sent))
    soda_mean = sum(soda_scores) / max(len(soda_scores), 1)
    print(f"  SODA_c (mean over {len(soda_scores)} videos) = {soda_mean:.4f}")
    print()

    # 4) token usage (from source file)
    print("=" * 72)
    print("Token usage  (cached from LLM run, unchanged)")
    print("-" * 72)
    agg = data["aggregate"]
    tu = agg["token_usage"]
    print(f"  prompt_tokens     : {tu['prompt_tokens']:>10,}")
    print(f"  completion_tokens : {tu['completion_tokens']:>10,}")
    print(f"  total_tokens      : {tu['total_tokens']:>10,}")
    print(f"  estimated cost    : CNY {agg['estimated_cost_cny']}")
    print("=" * 72)

    out_json = {
        "model": data.get("model"),
        "n_videos": len(per),
        "paragraph": para,
        "dvc_per_threshold": dvc_rows,
        "dvc_avg": avg,
        "soda_c": soda_mean,
        "soda_c_per_video": dict(zip([r["video"] for r in per], soda_scores)),
        "token_usage": tu,
        "estimated_cost_cny": agg["estimated_cost_cny"],
    }
    (PROJECT_ROOT / args.out).write_text(
        json.dumps(out_json, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[saved] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
