"""
Helpers for DB / API: return dicts matching on-disk JSON shape, or JSON strings.

Does not write files; use coordinator.run_video_to_events(save=True) or run_refine_events_from_files.

Refinement APIs import refinement_runner lazily so video_events_as_* callers avoid pulling LangChain/OpenAI.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from video.factory.refinement_runner import RefineEventsConfig


def video_events_as_json_dicts(
    video_path: str,
    *,
    _run_pipeline: Callable[..., tuple[list[dict[str, Any]], list[dict[str, float]], dict[str, Any]]]
    | None = None,
    **run_kwargs: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Run YOLO+track+slicing; return two dicts matching *_events.json and *_clips.json.

    Returns: (events_document, clips_document), ready for DB insert or json.dumps.

    _run_pipeline: inject for tests; defaults to event_track_pipeline.run_pipeline.
    """
    if _run_pipeline is None:
        from video.factory.processors.event_track_pipeline import run_pipeline as _run_pipeline

    events, clip_segments, meta = _run_pipeline(video_path, **run_kwargs)
    events_document = {"meta": meta, "events": events}
    clips_document = {"meta": meta, "clip_segments": clip_segments}
    return events_document, clips_document


def video_events_as_json_strings(
    video_path: str,
    *,
    indent: int | None = 2,
    _run_pipeline: Callable[..., tuple[list[dict[str, Any]], list[dict[str, float]], dict[str, Any]]]
    | None = None,
    **run_kwargs: Any,
) -> tuple[str, str]:
    """Same as video_events_as_json_dicts but returns UTF-8 JSON strings (compact if indent is None)."""
    ev, cl = video_events_as_json_dicts(video_path, _run_pipeline=_run_pipeline, **run_kwargs)
    dump_kw: dict[str, Any] = {"ensure_ascii": False}
    if indent is not None:
        dump_kw["indent"] = indent
    return json.dumps(ev, **dump_kw), json.dumps(cl, **dump_kw)


def refined_events_as_dict(
    events_document: dict[str, Any],
    clips_document: dict[str, Any],
    config: RefineEventsConfig | None = None,
) -> dict[str, Any]:
    """
    LLM-refine in-memory pipeline documents; return dict matching written JSON.

    events_document / clips_document match video_events_as_json_dicts output shape.
    """
    from video.factory.refinement_runner import run_refine_events_to_dict

    return run_refine_events_to_dict(events_document, clips_document, config)


def refined_events_as_json_str(
    events_document: dict[str, Any],
    clips_document: dict[str, Any],
    config: RefineEventsConfig | None = None,
    *,
    indent: int | None = 2,
) -> str:
    """Single JSON string for refined output (vector mode: flat events; full mode: rich schema)."""
    payload = refined_events_as_dict(events_document, clips_document, config)
    dump_kw: dict[str, Any] = {"ensure_ascii": False}
    if indent is not None:
        dump_kw["indent"] = indent
    return json.dumps(payload, **dump_kw)


def refined_events_from_json_files_as_dict(
    events_path: str | Path,
    clips_path: str | Path,
    config: RefineEventsConfig | None = None,
) -> dict[str, Any]:
    """Read pipeline JSON from disk, refine, return dict (does not write a file)."""
    events_path = Path(events_path)
    clips_path = Path(clips_path)
    events_data = json.loads(events_path.read_text(encoding="utf-8"))
    clips_data = json.loads(clips_path.read_text(encoding="utf-8"))
    return refined_events_as_dict(events_data, clips_data, config)
