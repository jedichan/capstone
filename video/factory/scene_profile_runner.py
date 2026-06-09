"""Camera-level scene profiling for retrieval-friendly event text.

This module is intentionally a light optional layer. It samples a few full
frames per camera/video, asks the VLM for stable scene zones, then downstream
seed generation can use those zones without changing detection/tracking logic.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI

logger = logging.getLogger(__name__)


@dataclass
class SceneProfileConfig:
    api_key: str = ""
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model: str = "qwen3-vl-flash"
    temperature: float = 0.0
    max_tokens: int = 420
    sample_ratios: tuple[float, ...] = (0.08, 0.50, 0.92)
    max_image_width: int = 960
    cache_path: Path | None = None
    force: bool = False

    @classmethod
    def from_env(cls, **overrides: Any) -> "SceneProfileConfig":
        values = {
            "api_key": os.getenv("DASHSCOPE_API_KEY", ""),
            "base_url": os.getenv("DASHSCOPE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            "model": os.getenv("DASHSCOPE_CHAT_MODEL", "qwen3-vl-flash"),
        }
        values.update(overrides)
        return cls(**values)


def _json_object_from_text(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)
    except Exception:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        return json.loads(raw[start:end + 1])
    raise ValueError(f"No JSON object found in response: {text[:200]}")


def _normalize_zone_map(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        value = {}
    defaults = {
        "left": "left side of the frame",
        "right": "right side of the frame",
        "top": "far upper part of the frame",
        "bottom": "near lower foreground",
        "center": "central area of the frame",
    }
    out = {}
    for key, fallback in defaults.items():
        text = str(value.get(key) or fallback).strip()
        out[key] = text[:140]
    return out


def _sample_frame_images(video_path: Path, config: SceneProfileConfig) -> tuple[list[str], dict[str, Any]]:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video for scene profile: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if frame_count <= 0:
        cap.release()
        raise RuntimeError(f"Video has no readable frames: {video_path}")

    b64_images: list[str] = []
    seen_indices: set[int] = set()
    for ratio in config.sample_ratios:
        idx = int(max(0, min(frame_count - 1, round(frame_count * float(ratio)))))
        if idx in seen_indices:
            continue
        seen_indices.add(idx)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        if config.max_image_width > 0 and frame.shape[1] > config.max_image_width:
            scale = config.max_image_width / float(frame.shape[1])
            frame = cv2.resize(frame, (config.max_image_width, max(1, int(frame.shape[0] * scale))))
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        if ok:
            b64_images.append(base64.b64encode(encoded.tobytes()).decode("ascii"))
    cap.release()
    meta = {
        "frame_count": frame_count,
        "fps": fps,
        "frame_width": width,
        "frame_height": height,
        "sampled_frame_indices": sorted(seen_indices),
    }
    return b64_images, meta


def _call_scene_vlm(client: OpenAI, images_b64: list[str], camera_id: str, config: SceneProfileConfig) -> dict[str, Any]:
    user_content: list[dict[str, Any]] = [{
        "type": "text",
        "text": (
            f"Camera: {camera_id}\n"
            "These full-frame images are sampled from one fixed surveillance camera. "
            "Describe stable scene layout for retrieval. Return strict JSON:\n"
            "{\n"
            '  "scene_summary": "short stable scene description",\n'
            '  "zones": {"left": "...", "right": "...", "top": "...", "bottom": "...", "center": "..."},\n'
            '  "exit_meaning": {"left": "...", "right": "...", "up": "...", "down": "..."},\n'
            '  "keywords": ["scene", "zone", "tokens"]\n'
            "}\n"
            "Use stable visible regions only. Do not describe transient people or vehicles."
        ),
    }]
    for b64 in images_b64:
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })

    resp = client.chat.completions.create(
        model=config.model,
        messages=[
            {"role": "system", "content": "Return only valid JSON for camera scene profiling."},
            {"role": "user", "content": user_content},
        ],
        temperature=float(config.temperature),
        max_tokens=int(config.max_tokens),
    )
    parsed = _json_object_from_text(resp.choices[0].message.content or "{}")
    keywords = parsed.get("keywords") or []
    if not isinstance(keywords, list):
        keywords = []
    return {
        "scene_summary": str(parsed.get("scene_summary") or "surveillance camera scene").strip()[:200],
        "zones": _normalize_zone_map(parsed.get("zones")),
        "exit_meaning": _normalize_zone_map(parsed.get("exit_meaning")),
        "keywords": [str(k).strip().lower().replace(" ", "_") for k in keywords if str(k).strip()][:16],
    }


def run_scene_profiles_for_pipeline(
    *,
    slot: str,
    camera_to_video: dict[str, str | Path],
    camera_video_stems: dict[str, str] | None = None,
    config: SceneProfileConfig | None = None,
) -> dict[str, Any]:
    cfg = config or SceneProfileConfig.from_env()
    if cfg.cache_path and cfg.cache_path.exists() and not cfg.force:
        logger.info("[%s] Loading cached scene profiles from %s", slot, cfg.cache_path)
        return json.loads(cfg.cache_path.read_text(encoding="utf-8"))
    if not cfg.api_key:
        raise RuntimeError("DASHSCOPE_API_KEY is required for scene profiling")

    timeout_sec = float(os.getenv("VIDEO_VLM_TIMEOUT_SEC", "600"))
    client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url, timeout=timeout_sec, max_retries=2)
    per_camera: dict[str, dict[str, Any]] = {}
    t0 = time.time()
    for cam_id, video in sorted(camera_to_video.items()):
        video_path = Path(video)
        if not video_path.exists():
            continue
        try:
            images, meta = _sample_frame_images(video_path, cfg)
            if not images:
                continue
            profile = _call_scene_vlm(client, images, cam_id, cfg)
            profile.update(meta)
            profile["video_id"] = (camera_video_stems or {}).get(cam_id, video_path.stem)
            per_camera[cam_id] = profile
        except Exception as exc:
            logger.warning("[%s] scene profile failed for %s: %s", slot, cam_id, exc)

    result = {
        "slot": slot,
        "elapsed_sec": round(time.time() - t0, 1),
        "per_camera": per_camera,
    }
    if cfg.cache_path:
        cfg.cache_path.parent.mkdir(parents=True, exist_ok=True)
        cfg.cache_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
