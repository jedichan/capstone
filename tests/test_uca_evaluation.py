"""UCA (UCF-Crime Annotation) 格式 LLM 输出评测脚本.

用途
----
在 UCA Test split 上评测 LLM 以 UCA 格式输出的稠密事件描述质量:
    - 时间定位:  tIoU (逐对贪心匹配) + Recall@tIoU{0.3, 0.5, 0.7}
    - 文本质量:  token F1 (简易 BLEU-ish, 仅依赖 stdlib)
    - 格式合规:  字段齐全 & timestamps/sentences 长度一致 & start<end<=duration

直接运行
--------
    # 1) mock 模式 (不调用 LLM, 用 ground-truth 作为预测来自检评测代码):
    python tests/test_uca_evaluation.py --mock --num 5

    # 2) 真实 LLM 模式 (需要 DASHSCOPE_API_KEY, 用 qwen3.6-plus):
    python tests/test_uca_evaluation.py --num 3

    # 3) 指定单个视频:
    python tests/test_uca_evaluation.py --video Abuse037_x264 --mock

视频文件本身不是跑这个脚本所必需的 —— 评测基于 UCA 标注 JSON 与 LLM
(以文本事件证据为输入) 的文本输出. 若要接入真实视频, 将
`collect_frame_evidence()` 替换为你的 pipeline 输出即可.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable

# 让脚本可以 `python tests/test_uca_evaluation.py` 直接跑
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.node.uca_prompts import (  # noqa: E402
    UCA_OUTPUT_SCHEMA,
    UCA_SYSTEM_PROMPT,
    build_uca_dense_caption_prompt,
)

UCA_TEST_JSON = (
    PROJECT_ROOT
    / "_data"
    / "Surveillance-Video-Understanding"
    / "UCF Annotation"
    / "json"
    / "UCFCrime_Test.json"
)


# --------------------------------------------------------------------------- #
# 1.  加载 UCA ground truth                                                    #
# --------------------------------------------------------------------------- #
def load_uca_ground_truth(path: Path = UCA_TEST_JSON) -> dict[str, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"UCA 标注未找到: {path}\n请确认 _data/ 已放置 Surveillance-Video-Understanding 数据。"
        )
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# 2.  评测指标                                                                 #
# --------------------------------------------------------------------------- #
def temporal_iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(s: str) -> list[str]:
    return _TOKEN_RE.findall(s.lower())


def token_f1(pred: str, gt: str) -> float:
    p, g = _tokens(pred), _tokens(gt)
    if not p or not g:
        return 0.0
    common: dict[str, int] = {}
    for tok in p:
        if tok in g and common.get(tok, 0) < g.count(tok):
            common[tok] = common.get(tok, 0) + 1
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(p)
    recall = overlap / len(g)
    return 2 * precision * recall / (precision + recall)


def greedy_match(
    pred_ts: list[list[float]],
    pred_sent: list[str],
    gt_ts: list[list[float]],
    gt_sent: list[str],
) -> list[tuple[int, int, float, float]]:
    """对每条 gt 贪心找最高 tIoU 的 pred, 返回 (gt_idx, pred_idx, tIoU, tokenF1)。"""
    used: set[int] = set()
    out: list[tuple[int, int, float, float]] = []
    for gi, gts in enumerate(gt_ts):
        best_i, best_iou = -1, 0.0
        for pi, pts in enumerate(pred_ts):
            if pi in used:
                continue
            iou = temporal_iou(tuple(pts), tuple(gts))
            if iou > best_iou:
                best_iou, best_i = iou, pi
        if best_i >= 0:
            used.add(best_i)
            f1 = token_f1(pred_sent[best_i], gt_sent[gi])
            out.append((gi, best_i, best_iou, f1))
        else:
            out.append((gi, -1, 0.0, 0.0))
    return out


def validate_uca(payload: dict[str, Any], duration: float) -> list[str]:
    errs: list[str] = []
    for key in ("video_name", "duration", "timestamps", "sentences"):
        if key not in payload:
            errs.append(f"missing field: {key}")
    ts = payload.get("timestamps") or []
    sents = payload.get("sentences") or []
    if len(ts) != len(sents):
        errs.append(f"len(timestamps)={len(ts)} != len(sentences)={len(sents)}")
    for i, t in enumerate(ts):
        if not (isinstance(t, (list, tuple)) and len(t) == 2):
            errs.append(f"timestamps[{i}] shape invalid: {t}")
            continue
        s, e = float(t[0]), float(t[1])
        if not (0 <= s < e <= duration + 0.5):
            errs.append(f"timestamps[{i}]=({s},{e}) out of [0,{duration}]")
    return errs


# --------------------------------------------------------------------------- #
# 3.  预测生成: mock 或 真实 LLM                                               #
# --------------------------------------------------------------------------- #
def collect_frame_evidence(video_name: str, gt: dict[str, Any]) -> list[dict[str, Any]]:
    """占位: 真实项目中此处替换为 pipeline 的帧级检测/caption 输出.

    这里用 gt 的时间戳中点作为"证据伪事件", 让 LLM 有最小可用上下文. 在真实评测时,
    应改为从 YOLO + BLIP/Qwen-VL 的 pipeline 读取每秒/每片段的检测与短 caption.
    """
    events = []
    for (s, e), sent in zip(gt["timestamps"], gt["sentences"]):
        events.append(
            {"t": round((float(s) + float(e)) / 2, 1), "caption": sent[:40] + "..."}
        )
    return events


def predict_mock(video_name: str, gt: dict[str, Any]) -> dict[str, Any]:
    """mock 预测: 直接返回 gt, 用于自检评测代码是否给出满分."""
    return {
        "video_name": video_name,
        "duration": gt["duration"],
        "timestamps": [list(x) for x in gt["timestamps"]],
        "sentences": list(gt["sentences"]),
    }


def predict_llm(video_name: str, gt: dict[str, Any]) -> dict[str, Any]:
    """真实 LLM 预测: qwen3.6-plus via DashScope OpenAI-compatible API."""
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv()
    except Exception:
        pass
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY 未配置, 无法调用真实 LLM. 可改用 --mock.")

    from openai import OpenAI  # type: ignore

    client = OpenAI(
        api_key=api_key,
        base_url=os.environ.get(
            "DASHSCOPE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        ),
    )

    evidence = collect_frame_evidence(video_name, gt)
    user_prompt = build_uca_dense_caption_prompt(
        video_name=video_name, duration=gt["duration"], frame_events=evidence
    )
    resp = client.chat.completions.create(
        model=os.environ.get("DASHSCOPE_CHAT_MODEL", "qwen3.6-plus"),
        temperature=0.0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": UCA_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    text = resp.choices[0].message.content or ""
    text = text.replace("```json", "").replace("```", "").strip()
    return json.loads(text)


# --------------------------------------------------------------------------- #
# 4.  主流程                                                                   #
# --------------------------------------------------------------------------- #
def evaluate_one(video_name: str, gt: dict[str, Any], use_mock: bool) -> dict[str, Any]:
    predict_fn = predict_mock if use_mock else predict_llm
    try:
        pred = predict_fn(video_name, gt)
    except Exception as e:
        return {"video": video_name, "error": f"predict_failed: {e}"}

    errs = validate_uca(pred, gt["duration"])
    pairs = greedy_match(pred["timestamps"], pred["sentences"], gt["timestamps"], gt["sentences"])
    ious = [p[2] for p in pairs]
    f1s = [p[3] for p in pairs]
    n = max(len(gt["timestamps"]), 1)

    return {
        "video": video_name,
        "schema_errors": errs,
        "num_gt": len(gt["timestamps"]),
        "num_pred": len(pred.get("timestamps", [])),
        "mean_tIoU": round(sum(ious) / n, 4),
        "recall@0.3": round(sum(1 for i in ious if i >= 0.3) / n, 4),
        "recall@0.5": round(sum(1 for i in ious if i >= 0.5) / n, 4),
        "recall@0.7": round(sum(1 for i in ious if i >= 0.7) / n, 4),
        "mean_tokenF1": round(sum(f1s) / n, 4),
    }


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in results if "error" not in r]
    if not ok:
        return {"n": 0}
    keys = ["mean_tIoU", "recall@0.3", "recall@0.5", "recall@0.7", "mean_tokenF1"]
    agg = {k: round(sum(r[k] for r in ok) / len(ok), 4) for k in keys}
    agg["n_videos"] = len(ok)
    agg["n_failed"] = len(results) - len(ok)
    return agg


def main() -> int:
    parser = argparse.ArgumentParser(description="UCA-format LLM evaluation")
    parser.add_argument("--mock", action="store_true", help="不调用 LLM, 用 gt 自测评测代码")
    parser.add_argument("--num", type=int, default=3, help="抽样视频数 (默认 3)")
    parser.add_argument("--video", type=str, default=None, help="仅评测指定视频名")
    parser.add_argument(
        "--gt", type=str, default=str(UCA_TEST_JSON), help="UCA Test JSON 路径"
    )
    parser.add_argument("--out", type=str, default=None, help="把逐视频结果写到 JSON")
    args = parser.parse_args()

    all_gt = load_uca_ground_truth(Path(args.gt))
    if args.video:
        if args.video not in all_gt:
            print(f"[ERROR] 视频 {args.video} 不在 UCA Test 中", file=sys.stderr)
            return 2
        selected = [args.video]
    else:
        selected = list(all_gt.keys())[: args.num]

    print(f"[INFO] schema: {json.dumps(UCA_OUTPUT_SCHEMA, ensure_ascii=False)}")
    print(f"[INFO] 评测 {len(selected)} 个视频  (mock={args.mock})")
    print("-" * 70)

    results: list[dict[str, Any]] = []
    for name in selected:
        r = evaluate_one(name, all_gt[name], use_mock=args.mock)
        results.append(r)
        print(json.dumps(r, ensure_ascii=False))

    print("-" * 70)
    agg = aggregate(results)
    print("[AGGREGATE]", json.dumps(agg, ensure_ascii=False))

    if args.out:
        Path(args.out).write_text(
            json.dumps({"per_video": results, "aggregate": agg}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[INFO] 写出: {args.out}")

    # mock 模式下应该拿到满分, 可作为 CI 断言
    if args.mock and agg.get("n_videos", 0) > 0:
        assert agg["mean_tIoU"] == 1.0, f"mock 模式 tIoU 应为 1.0, 实际 {agg['mean_tIoU']}"
        assert agg["mean_tokenF1"] == 1.0, f"mock 模式 tokenF1 应为 1.0"
        print("[OK] mock 自检通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
