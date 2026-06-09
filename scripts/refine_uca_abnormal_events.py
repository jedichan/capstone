"""Refine UCA dense-caption predictions with abnormal-event wording.

This is a lightweight UCA-specific post-process. It takes the output of
``tests/test_uca_unified.py`` and asks a text LLM to rewrite each predicted
dense-caption sentence toward abnormal/crime-relevant wording while preserving
the original timestamps and event count. The result keeps the same JSON shape
so it can be passed directly to
``scripts/run_uca_agent_eval.py --video-result-json``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_PIPELINE_DIR = ROOT / "_pipeline_output" / "uca_eval"


SYSTEM_PROMPT = (
    "You refine surveillance dense captions for UCA/UCF-Crime evaluation. "
    "Your task is to preserve the original time coverage while making each "
    "caption more useful for abnormal-event retrieval. Return only valid JSON."
)


def _load_env() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except Exception:
        env_file = ROOT / ".env"
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _safe_json(text: str) -> dict[str, Any]:
    raw = (text or "").replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(raw)
    except Exception:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        return json.loads(raw[start:end + 1])
    raise ValueError(f"No JSON object found in response: {text[:200]}")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _clip_interval(start: Any, end: Any, duration: float) -> list[float] | None:
    s = max(0.0, min(duration, _safe_float(start)))
    e = max(0.0, min(duration, _safe_float(end)))
    if e <= s:
        return None
    return [round(s, 1), round(e, 1)]


def _normalize_pred(payload: dict[str, Any], video_name: str, duration: float) -> dict[str, Any]:
    timestamps = payload.get("timestamps") or []
    sentences = payload.get("sentences") or []
    out_ts: list[list[float]] = []
    out_sent: list[str] = []

    for time_pair, sentence in zip(timestamps, sentences):
        if not (isinstance(time_pair, (list, tuple)) and len(time_pair) >= 2):
            continue
        clipped = _clip_interval(time_pair[0], time_pair[1], duration)
        text = str(sentence or "").replace("##", " ").strip()
        text = re.sub(r"\s+", " ", text)
        if not clipped or not text:
            continue
        if text[-1] not in ".!?":
            text += "."
        out_ts.append(clipped)
        out_sent.append(text)

    return {
        "video_name": str(payload.get("video_name") or video_name),
        "duration": duration,
        "timestamps": out_ts,
        "sentences": out_sent,
    }


def _normalize_enhanced_pred(
    original_pred: dict[str, Any],
    payload: dict[str, Any],
    video_name: str,
    duration: float,
) -> dict[str, Any]:
    original_ts = original_pred.get("timestamps") or []
    original_sent = original_pred.get("sentences") or []
    enhanced_sent = payload.get("sentences") or []
    abnormal_scores = payload.get("abnormal_scores") or []
    abnormal_keywords = payload.get("abnormal_keywords") or []

    out_ts: list[list[float]] = []
    out_sent: list[str] = []
    for idx, (time_pair, old_sentence) in enumerate(zip(original_ts, original_sent)):
        if not (isinstance(time_pair, (list, tuple)) and len(time_pair) >= 2):
            continue
        clipped = _clip_interval(time_pair[0], time_pair[1], duration)
        if not clipped:
            continue

        new_sentence = enhanced_sent[idx] if idx < len(enhanced_sent) else old_sentence
        text = str(new_sentence or old_sentence or "").replace("##", " ").strip()
        text = re.sub(r"\s+", " ", text)
        if not text:
            continue
        if text[-1] not in ".!?":
            text += "."

        score = abnormal_scores[idx] if idx < len(abnormal_scores) else None
        keywords = abnormal_keywords[idx] if idx < len(abnormal_keywords) else []
        keyword_text = ""
        if isinstance(keywords, list):
            compact = [str(k).strip().lower().replace(" ", "_") for k in keywords if str(k).strip()]
            if compact:
                keyword_text = " Abnormal retrieval keywords: " + ", ".join(compact[:8]) + "."
        score_text = ""
        if isinstance(score, (int, float)):
            score_text = f" Abnormal relevance score: {float(score):.2f}."

        out_ts.append(clipped)
        out_sent.append((text + score_text + keyword_text).strip())

    return {
        "video_name": str(payload.get("video_name") or video_name),
        "duration": duration,
        "timestamps": out_ts,
        "sentences": out_sent,
    }


def _format_segments(pred: dict[str, Any], max_segments: int) -> str:
    lines: list[str] = []
    timestamps = pred.get("timestamps") or []
    sentences = pred.get("sentences") or []
    for idx, (time_pair, sentence) in enumerate(zip(timestamps, sentences), start=1):
        if idx > max_segments:
            lines.append(f"... (+{len(timestamps) - max_segments} more)")
            break
        if isinstance(time_pair, (list, tuple)) and len(time_pair) >= 2:
            lines.append(f"{idx}. {float(time_pair[0]):.1f}-{float(time_pair[1]):.1f}s: {sentence}")
    return "\n".join(lines)


def _summarize_yolo(video_id: str, pipeline_dir: Path, max_lines: int = 30) -> str:
    path = pipeline_dir / f"{video_id}_events.json"
    if not path.exists():
        return ""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    lines: list[str] = []
    for idx, event in enumerate(payload.get("events", [])[:max_lines], start=1):
        start = _safe_float(event.get("start_time"))
        end = _safe_float(event.get("end_time"))
        obj = event.get("class_name") or event.get("object_type") or "object"
        track = event.get("track_id", "?")
        desc = event.get("description_for_llm") or event.get("motion_description") or ""
        lines.append(f"{idx}. {start:.1f}-{end:.1f}s: {obj} track {track}; {desc}")
    return "\n".join(lines)


def _call_refine_llm(
    *,
    client: Any,
    model: str,
    video_name: str,
    duration: float,
    category_hint: str,
    dense_segments: str,
    yolo_evidence: str,
    preserve_segments: bool,
) -> tuple[dict[str, Any], dict[str, int]]:
    yolo_block = f"\nYOLO motion evidence:\n{yolo_evidence}\n" if yolo_evidence else ""
    if preserve_segments:
        user_prompt = (
            f"Video name: {video_name}\n"
            f"Category hint from filename: {category_hint}\n"
            f"Duration: {duration:.2f} seconds\n\n"
            "Dense caption predictions, numbered in order:\n"
            f"{dense_segments}\n"
            f"{yolo_block}\n"
            "Rewrite every dense-caption segment for abnormal/crime-event retrieval.\n"
            "Rules:\n"
            "1. Preserve the exact number and order of input segments.\n"
            "2. Do not merge, split, remove, or alter timestamps.\n"
            "3. If a segment is normal/context only, keep it but mark low abnormal relevance.\n"
            "4. If a segment contains the abnormal action, make the action explicit.\n"
            "5. Do not invent objects or actions that are not supported by the captions/evidence.\n\n"
            "Return strict JSON with fields:\n"
            "{\n"
            '  "video_name": "...",\n'
            '  "duration": 0.0,\n'
            '  "sentences": ["same length as input, rewritten sentence"],\n'
            '  "abnormal_scores": [0.0],\n'
            '  "abnormal_keywords": [["retrieval", "keywords"]]\n'
            "}"
        )
    else:
        user_prompt = (
            f"Video name: {video_name}\n"
            f"Category hint from filename: {category_hint}\n"
            f"Duration: {duration:.2f} seconds\n\n"
            "Dense caption predictions:\n"
            f"{dense_segments}\n"
            f"{yolo_block}\n"
            "Refine these into abnormal/crime-relevant event intervals.\n"
            "Rules:\n"
            "1. Focus on when the abnormal action happens, not the whole time a person/object is visible.\n"
            "2. Merge adjacent segments only if they describe the same abnormal episode.\n"
            "3. Keep timestamps in seconds, 0.1 precision, within duration.\n"
            "4. Prefer concise intervals and factual descriptions.\n"
            "5. If the evidence is too weak, keep the most relevant dense-caption intervals instead of inventing.\n\n"
            "Return strict JSON with fields:\n"
            "{\n"
            '  "video_name": "...",\n'
            '  "duration": 0.0,\n'
            '  "timestamps": [[start_sec, end_sec]],\n'
            '  "sentences": ["one abnormal-event sentence"]\n'
            "}"
        )
    resp = client.chat.completions.create(
        model=model,
        temperature=0.0,
        max_tokens=1200,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    usage_obj = getattr(resp, "usage", None)
    usage = {
        "prompt_tokens": int(getattr(usage_obj, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage_obj, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage_obj, "total_tokens", 0) or 0),
    }
    return _safe_json(resp.choices[0].message.content or "{}"), usage


def _category_hint(video_name: str, path: str = "") -> str:
    text = f"{video_name} {path}".lower()
    for category in ["abuse", "arrest", "arson", "assault", "burglary", "explosion", "fighting", "robbery", "shooting", "roadaccidents"]:
        if category in text:
            return category
    return "unknown"


def refine_file(
    *,
    input_path: Path,
    output_path: Path,
    model: str,
    limit: int,
    force: bool,
    pipeline_dir: Path,
    max_input_segments: int,
    max_output_events: int,
    mode: str,
) -> dict[str, Any]:
    _load_env()
    api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY or OPENAI_API_KEY is required")

    from openai import OpenAI

    client = OpenAI(
        api_key=api_key,
        base_url=os.getenv("DASHSCOPE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    )

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    rows = payload.get("per_video") or []
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    processed = skipped = failed = 0

    for idx, row in enumerate(rows, start=1):
        if limit > 0 and processed >= limit:
            break
        if not isinstance(row, dict) or row.get("error"):
            skipped += 1
            continue
        if row.get("abnormal_refinement") and not force:
            skipped += 1
            continue

        video_name = str(row.get("video") or (row.get("pred") or {}).get("video_name") or "").strip()
        pred = row.get("pred") or {}
        duration = _safe_float(row.get("duration_video") or pred.get("duration"))
        dense_segments = _format_segments(pred, max_segments=max_input_segments)
        if not video_name or not dense_segments:
            skipped += 1
            continue

        print(f"[uca-abnormal] {idx}/{len(rows)} {video_name}")
        t0 = time.perf_counter()
        try:
            refined_raw, usage = _call_refine_llm(
                client=client,
                model=model,
                video_name=video_name,
                duration=duration,
                category_hint=_category_hint(video_name, str(row.get("path") or "")),
                dense_segments=dense_segments,
                yolo_evidence=_summarize_yolo(video_name, pipeline_dir),
                preserve_segments=(mode == "enhance"),
            )
            if mode == "enhance":
                refined_pred = _normalize_enhanced_pred(pred, refined_raw, video_name, duration)
            else:
                refined_pred = _normalize_pred(refined_raw, video_name, duration)
            if not refined_pred["timestamps"]:
                raise ValueError("Refined prediction produced no valid events")
            row["pred_before_abnormal_refine"] = pred
            row["pred"] = refined_pred
            row["num_pred"] = len(refined_pred["timestamps"])
            row["abnormal_refinement"] = {
                "enabled": True,
                "mode": mode,
                "model": model,
                "elapsed_sec": round(time.perf_counter() - t0, 2),
                "usage": usage,
            }
            for key in total_usage:
                total_usage[key] += usage.get(key, 0)
            processed += 1
        except Exception as exc:
            row["abnormal_refinement"] = {
                "enabled": True,
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_sec": round(time.perf_counter() - t0, 2),
            }
            failed += 1
            print(f"  [WARN] failed: {row['abnormal_refinement']['error']}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    payload["abnormal_refinement_summary"] = {
        "source": str(input_path),
        "model": model,
        "mode": mode,
        "processed": processed,
        "skipped": skipped,
        "failed": failed,
        "usage": total_usage,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload["abnormal_refinement_summary"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Refine UCA predictions into abnormal-event intervals")
    parser.add_argument("--input", required=True, help="Input result JSON from tests/test_uca_unified.py")
    parser.add_argument("--out", default="", help="Output refined JSON path")
    parser.add_argument("--model", default=os.getenv("DASHSCOPE_CHAT_MODEL", "qwen3.6-plus"))
    parser.add_argument("--limit", type=int, default=0, help="Max videos to refine; 0 = all")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--pipeline-dir", default=str(DEFAULT_PIPELINE_DIR))
    parser.add_argument("--max-input-segments", type=int, default=80)
    parser.add_argument("--max-output-events", type=int, default=12)
    parser.add_argument(
        "--mode",
        choices=["enhance", "compress"],
        default="enhance",
        help="enhance preserves all timestamps; compress may merge abnormal episodes.",
    )
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    output_path = Path(args.out).resolve() if args.out else input_path.with_name(input_path.stem + "_abnormal_refined.json")
    summary = refine_file(
        input_path=input_path,
        output_path=output_path,
        model=args.model,
        limit=args.limit,
        force=args.force,
        pipeline_dir=Path(args.pipeline_dir).resolve(),
        max_input_segments=args.max_input_segments,
        max_output_events=args.max_output_events,
        mode=args.mode,
    )
    print(f"[uca-abnormal] Done: {summary}")
    print(f"[uca-abnormal] Output: {output_path}")


if __name__ == "__main__":
    main()
