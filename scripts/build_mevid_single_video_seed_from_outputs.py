"""
Build one MEVID agent vector seed from existing single-video pipeline outputs.

Use this after:
  python -m video.factory.coordinator video ...
  python -m video.factory.coordinator refine ...
  python -m video.factory.coordinator appearance ...

The coordinator refine output is not itself the final agent seed. This script
normalizes video_id, merges optional clip-level and crop-level refinements, and
writes agent/test/data/events_vector_flat/<video_id>_events_vector_flat.json.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except Exception:
    pass

from scripts.generate_mevid_vector_flat import (  # noqa: E402
    OUT_DIR,
    _get_video_duration,
    pipeline_events_to_vector_flat,
)


def _clean_stem(value: str) -> str:
    path = Path(str(value or ""))
    name = path.name or str(value or "")
    for suffix in ("_events.json", "_clips.json", "_events_vector_flat.json", "_events_appearance_refined.json"):
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
            break
    for suffix in (".avi", ".mp4", ".mov", ".mkv"):
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name


def _camera_from_stem(stem: str) -> str:
    match = re.search(r"\.(G\d+)\.r\d+$", stem, flags=re.IGNORECASE)
    return match.group(1).upper() if match else ""


def _read_json(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    return json.loads(p.read_text(encoding="utf-8"))


def _events_from_refinement(path: str | Path | None) -> list[dict[str, Any]]:
    payload = _read_json(path)
    events = payload.get("events", [])
    return list(events) if isinstance(events, list) else []


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert single-video MEVID outputs into one agent seed")
    parser.add_argument("--events", required=True, help="Path to *_events.json from coordinator video")
    parser.add_argument("--refined", default="", help="Optional *_events_vector_flat.json from coordinator refine")
    parser.add_argument("--appearance", default="", help="Optional *_events_appearance_refined.json")
    parser.add_argument("--video-id", default="", help="Override seed video_id stem, without .avi")
    parser.add_argument("--camera", default="", help="Override camera id, e.g. G423")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument(
        "--min-event-duration-sec",
        type=float,
        default=1.0,
        help="Drop shorter events from the agent seed only; raw output files are unchanged",
    )
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    events_path = Path(args.events).expanduser().resolve()
    payload = _read_json(events_path)
    meta = payload.get("meta", {}) if isinstance(payload.get("meta"), dict) else {}
    raw_events = list(payload.get("events", []))

    video_path_raw = str(meta.get("video_path") or "")
    stem = args.video_id.strip() or _clean_stem(video_path_raw or events_path.name)
    camera_id = args.camera.strip().upper() or str(meta.get("camera_id") or "").upper() or _camera_from_stem(stem)
    if not camera_id:
        raise ValueError("Cannot infer camera id; pass --camera Gxxx")

    refined_events: list[dict[str, Any]] = []
    refined_events.extend(_events_from_refinement(args.refined))
    refined_events.extend(_events_from_refinement(args.appearance))
    for event in refined_events:
        event["video_id"] = stem
        event["camera_id"] = camera_id

    seed_events = raw_events
    if args.min_event_duration_sec > 0:
        before = len(seed_events)
        seed_events = [
            event for event in seed_events
            if float(event.get("end_time", event.get("start_time", 0.0)))
            - float(event.get("start_time", 0.0)) >= args.min_event_duration_sec
        ]
        print(
            "[single-video-seed] Duration filter: "
            f"kept={len(seed_events)} dropped={before - len(seed_events)} "
            f"threshold={args.min_event_duration_sec:.2f}s"
        )

    video_path = Path(video_path_raw) if video_path_raw else None
    duration = _get_video_duration(video_path) if video_path and video_path.exists() else 300.0
    flat = pipeline_events_to_vector_flat(
        video_id=stem,
        camera_id=camera_id,
        events=seed_events,
        global_entities=[],
        duration=duration,
        refined_events=refined_events or None,
    )

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stem}_events_vector_flat.json"
    if out_path.exists() and not args.force:
        print(f"[single-video-seed] Seed already exists: {out_path}")
        print("                    Use --force to overwrite.")
        return
    out_path.write_text(json.dumps(flat, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[single-video-seed] Wrote {len(flat.get('events', []))} events -> {out_path}")


if __name__ == "__main__":
    main()
