"""Load pipeline JSON + sample frames and call LLM refinement; supports in-memory dict or file output."""

from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from video.common.frames import (
    enrich_events_with_normalized_location,
    get_video_size,
    sample_frames_uniform,
)
from video.core.models.event_refinement_llm import (
    build_entities_with_hard_constraints,
    refine_events_with_llm,
    refine_vector_events_with_llm,
)
from video.core.schema.refined_event_llm import RefinedAllClipsPayload, RefinedEventsPayload, VectorEventsPayload

logger = logging.getLogger(__name__)


def _refine_concurrency() -> int:
    """Max concurrent VLM clip refinements (I/O-bound remote calls).

    Controlled by ``VIDEO_REFINE_CONCURRENCY`` (default 4). Values < 1 fall back
    to sequential execution.
    """
    try:
        return max(1, int(os.getenv("VIDEO_REFINE_CONCURRENCY", "4")))
    except (TypeError, ValueError):
        return 4


@dataclass
class RefineEventsConfig:
    mode: Literal["full", "vector"] = "vector"
    clip_index: int | None = None
    # Adaptive sampling: prefer frames_per_sec from clip duration
    # If num_frames > 0, use fixed count (backward compatible)
    num_frames: int = 0            # 0 = adaptive
    frames_per_sec: float = 0.1   # adaptive: frames per second (default ~1 frame / 10s)
    min_frames: int = 6
    max_frames: int = 48           # cap LLM token cost
    model: str = "qwen3-vl-flash"
    temperature: float = 0.1
    max_time_adjust_sec: float = 0.5
    merge_location_iou_threshold: float = 0.9
    merge_center_dist_px: float = 30.0
    merge_location_norm_diff: float = 0.10
    min_event_duration_sec: float = 1.0
    # Static vehicle events longer than this threshold are excluded from LLM
    # refinement: they span the whole recording, appear in every clip, and the
    # VLM cannot reliably determine color from full surveillance frames.
    static_vehicle_max_refine_sec: float = 30.0
    # Entity merge hard constraints (on by default)
    entity_merge_max_gap_sec: float = 300.0
    entity_merge_min_llm_confidence: float = 0.75

    def compute_num_frames(self, clip_duration_sec: float) -> int:
        """Compute sample count from clip duration."""
        if self.num_frames > 0:
            # fixed count (backward compatible)
            return int(self.num_frames)
        adaptive = round(clip_duration_sec * self.frames_per_sec)
        return max(self.min_frames, min(self.max_frames, adaptive))


def run_refine_events_to_dict(
    events_data: dict[str, Any],
    clips_data: dict[str, Any],
    config: RefineEventsConfig | None = None,
) -> dict[str, Any]:
    """
    Run refinement in memory; returns the same dict shape as written JSON (ready for json.dumps).

    events_data / clips_data match *_events.json and *_clips.json.
    """
    cfg = config or RefineEventsConfig()
    video_path = events_data["meta"]["video_path"]
    vw, vh = get_video_size(video_path)
    raw_events = enrich_events_with_normalized_location(events_data["events"], vw, vh)

    clip_segments = clips_data.get("clip_segments", [])
    if not isinstance(clip_segments, list) or not clip_segments:
        raise RuntimeError("clips.json has no clip_segments")

    if cfg.clip_index is None:
        indices = list(range(len(clip_segments)))
    else:
        if cfg.clip_index < 0 or cfg.clip_index >= len(clip_segments):
            raise ValueError(f"clip_index out of range: {cfg.clip_index} (valid 0..{len(clip_segments)-1})")
        indices = [int(cfg.clip_index)]

    refined_list_full: list[RefinedEventsPayload] = []
    refined_list_vector: list[VectorEventsPayload] = []

    def _process_clip(idx: int) -> tuple[str, Any] | None:
        """Refine one clip; returns (mode, payload) or None when skipped."""
        clip = clip_segments[idx]
        clip_start = float(clip["start_sec"])
        clip_end = float(clip["end_sec"])
        clip_events = []
        for e in raw_events:
            s = float(e.get("start_time", 0.0))
            t = float(e.get("end_time", 0.0))
            if t < clip_start or s > clip_end:
                continue
            if (t - s) < float(cfg.min_event_duration_sec):
                continue
            clip_events.append(e)
        if not clip_events:
            return None

        clip_duration = clip_end - clip_start
        n_frames = cfg.compute_num_frames(clip_duration)
        frames = sample_frames_uniform(
            video_path=video_path,
            start_sec=clip_start,
            end_sec=clip_end,
            num_frames=n_frames,
        )
        if not frames:
            raise RuntimeError(f"clip_index={idx}: no frames sampled; check clip range or video path")

        t_clip = time.time()
        if cfg.mode == "full":
            pre_entities = build_entities_with_hard_constraints(
                video_path=video_path,
                raw_events=clip_events,
                model=cfg.model,
                max_gap_sec=float(cfg.entity_merge_max_gap_sec),
                min_llm_confidence=float(cfg.entity_merge_min_llm_confidence),
            )
            payload = refine_events_with_llm(
                video_path=video_path,
                clip={"start_sec": clip_start, "end_sec": clip_end},
                raw_events=clip_events,
                frames=frames,
                model=cfg.model,
                temperature=float(cfg.temperature),
                max_time_adjust_sec=float(cfg.max_time_adjust_sec),
                merge_location_iou_threshold=float(cfg.merge_location_iou_threshold),
                merge_center_dist_px=float(cfg.merge_center_dist_px),
                merge_location_norm_diff=float(cfg.merge_location_norm_diff),
                pre_entities=pre_entities,
            )
            n_ev = len(getattr(payload, "refined_events", []) or [])
            print(
                f"[clip-refine] clip #{idx} [{clip_start:.1f}-{clip_end:.1f}s]: "
                f"{n_ev} events, {len(frames)} frames in {time.time() - t_clip:.1f}s",
                flush=True,
            )
            return ("full", payload)
        payload = refine_vector_events_with_llm(
            video_id=Path(video_path).name,
            clip={"start_sec": clip_start, "end_sec": clip_end},
            raw_events=clip_events,
            frames=frames,
            model=cfg.model,
            temperature=float(cfg.temperature),
        )
        n_ev = len(getattr(payload, "events", []) or [])
        print(
            f"[clip-refine] clip #{idx} [{clip_start:.1f}-{clip_end:.1f}s]: "
            f"{n_ev} events, {len(frames)} frames in {time.time() - t_clip:.1f}s",
            flush=True,
        )
        return ("vector", payload)

    # Clips are independent and dominated by remote VLM latency → run concurrently.
    # A single failed clip is logged and skipped so the rest still produce output.
    results_by_idx: dict[int, tuple[str, Any]] = {}
    max_workers = min(_refine_concurrency(), max(1, len(indices)))
    t_all = time.time()
    print(
        f"[clip-refine] start: {len(indices)} clips, mode={cfg.mode}, "
        f"model={cfg.model}, concurrency={max_workers}",
        flush=True,
    )
    if max_workers <= 1 or len(indices) <= 1:
        for idx in indices:
            try:
                outcome = _process_clip(idx)
            except Exception as exc:  # noqa: BLE001
                logger.warning("clip_index=%s refinement failed; skipping. Error: %s", idx, exc)
                continue
            if outcome is not None:
                results_by_idx[idx] = outcome
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {executor.submit(_process_clip, idx): idx for idx in indices}
            for future, idx in future_to_idx.items():
                try:
                    outcome = future.result()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("clip_index=%s refinement failed; skipping. Error: %s", idx, exc)
                    continue
                if outcome is not None:
                    results_by_idx[idx] = outcome

    # Re-assemble in original clip order to keep downstream output deterministic.
    for idx in indices:
        outcome = results_by_idx.get(idx)
        if outcome is None:
            continue
        kind, payload = outcome
        if kind == "full":
            refined_list_full.append(payload)
        else:
            refined_list_vector.append(payload)
    print(
        f"[clip-refine] done: {len(results_by_idx)}/{len(indices)} clips refined "
        f"in {time.time() - t_all:.1f}s",
        flush=True,
    )

    if cfg.mode == "full":
        if not refined_list_full:
            return {"video_path": str(video_path), "mode": "full"}
        if len(refined_list_full) == 1:
            return refined_list_full[0].model_dump()
        return RefinedAllClipsPayload(video_path=str(video_path), clips=refined_list_full).model_dump()

    flat_events: list[dict[str, Any]] = []
    for ve in refined_list_vector:
        for ev in ve.events:
            flat_events.append(ev.model_dump())

    return {
        "video_id": Path(video_path).name,
        "events": flat_events,
    }


def _build_cross_camera_context(output: Any) -> str | None:
    """
    Build a compact cross-camera trajectory string for the VLM prompt.

    Example output line:
        entity_003 (person): cam1 12.3s–18.7s → cam2 22.1s–30.4s (transit ≈3.4s)

    Returns None when there are no multi-camera entities (single-camera case).
    """
    from video.core.schema.multi_camera import GlobalEntity

    entities: list[GlobalEntity] = getattr(output, "global_entities", [])
    if not entities:
        return None

    lines: list[str] = []
    for ent in entities:
        apps = sorted(ent.appearances, key=lambda a: a.start_time)
        if not apps:
            continue

        # Determine object class from the first appearance's camera events
        # (best-effort; falls back to "entity")
        obj_class = "entity"
        per_camera: list[Any] = getattr(output, "per_camera", [])
        cam_map = {cr.camera_id: cr for cr in per_camera}
        first_app = apps[0]
        cam_result = cam_map.get(first_app.camera_id)
        if cam_result:
            for t in cam_result.tracks:
                if t.get("track_id") == first_app.track_id:
                    obj_class = str(t.get("class_name", "entity"))
                    break

        # Build per-camera segments
        segments: list[str] = []
        for app in apps:
            segments.append(
                f"{app.camera_id} {app.start_time:.1f}s–{app.end_time:.1f}s"
            )

        # Annotate inter-camera transit gaps
        parts: list[str] = [segments[0]]
        for i in range(1, len(apps)):
            gap = apps[i].start_time - apps[i - 1].end_time
            arrow = f" → (transit ≈{gap:.1f}s) " if gap > 0 else " → "
            parts.append(arrow + segments[i])

        lines.append(f"{ent.global_entity_id} ({obj_class}): {''.join(parts)}")

    return "\n".join(lines) if lines else None


def refine_multi_camera_output(
    output: Any,
    config: RefineEventsConfig | None = None,
) -> dict[str, Any]:
    """
    Run VLM refinement on a :class:`~video.core.schema.multi_camera.MultiCameraOutput`.

    Granularity: **camera × clip** (same as the single-camera path).
    A cross-camera context string derived from ``output.global_entities`` is injected
    into every VLM call so the model can reference cross-camera entity trajectories.

    Args:
        output: A :class:`MultiCameraOutput` produced by
            :func:`~video.factory.multi_camera_coordinator.run_multi_camera_pipeline`.
        config: Refinement hyper-parameters; :class:`RefineEventsConfig` defaults if *None*.

    Returns:
        A dict keyed by ``camera_id``; each value has the same shape as
        :func:`run_refine_events_to_dict` in ``"vector"`` mode::

            {
                "cam1": {"video_id": "cam1.mp4", "events": [...]},
                "cam2": {"video_id": "cam2.mp4", "events": [...]},
            }
    """
    cfg = config or RefineEventsConfig()
    cross_ctx = _build_cross_camera_context(output)

    if cross_ctx:
        logger.debug(
            "Cross-camera context (%d entities):\n%s",
            len(getattr(output, "global_entities", [])),
            cross_ctx,
        )

    # ── Build deterministic track_id → global_entity_id map ─────────────────
    # Key: (camera_id, track_id)  Value: global_entity_id string
    track_to_global: dict[tuple[str, int], str] = {}
    for ent in getattr(output, "global_entities", []):
        for app in ent.appearances:
            track_to_global[(app.camera_id, app.track_id)] = ent.global_entity_id

    # Per-camera appearance intervals for overlap-based fallback fill
    # Key: camera_id  Value: list of (start, end, global_entity_id)
    cam_appearance_intervals: dict[str, list[tuple[float, float, str]]] = {}
    for ent in getattr(output, "global_entities", []):
        for app in ent.appearances:
            cam_appearance_intervals.setdefault(app.camera_id, []).append(
                (app.start_time, app.end_time, ent.global_entity_id)
            )

    results: dict[str, Any] = {}

    for camera_result in output.per_camera:
        cam_id = camera_result.camera_id
        video_path = camera_result.video_path
        vw, vh = get_video_size(video_path)

        # All merged_events that belong to this camera (already have camera_id set)
        cam_events_raw = [
            ev for ev in output.merged_events
            if ev.get("camera_id") == cam_id
        ]
        raw_events = enrich_events_with_normalized_location(cam_events_raw, vw, vh)

        # Inject entity_hint into raw_events so LLM can copy it verbatim.
        # We build new dicts to avoid mutating the shared merged_events list.
        def _with_hint(e: dict[str, Any]) -> dict[str, Any]:
            tid = e.get("track_id")
            if tid is not None:
                hint = track_to_global.get((cam_id, int(tid)), f"track_{tid}")
            else:
                hint = None
            return {**e, "entity_hint": hint} if hint else e

        raw_events = [_with_hint(e) for e in raw_events]

        clip_segments: list[dict[str, float]] = camera_result.clips
        if not clip_segments:
            logger.warning("Camera %s has no clip_segments; skipping VLM refinement", cam_id)
            results[cam_id] = {"video_id": Path(video_path).name, "events": []}
            continue

        def _process_cam_clip(clip: dict[str, float]) -> VectorEventsPayload | None:
            clip_start = float(clip["start_sec"])
            clip_end = float(clip["end_sec"])

            clip_events = [
                e for e in raw_events
                if not (float(e.get("end_time", 0.0)) < clip_start
                        or float(e.get("start_time", 0.0)) > clip_end)
                and (float(e.get("end_time", 0.0)) - float(e.get("start_time", 0.0)))
                    >= cfg.min_event_duration_sec
                # Skip long-duration static vehicle events: they span the whole
                # video, inflate every clip's event list, and the LLM cannot
                # determine color/appearance from surveillance full-frames anyway.
                and not (
                    str(e.get("class_name", "")).lower()
                        in {"car", "truck", "bus", "van", "motorcycle", "bicycle"}
                    and str(e.get("event_type", "")) == "presence_static"
                    and (float(e.get("end_time", 0.0)) - float(e.get("start_time", 0.0)))
                        > cfg.static_vehicle_max_refine_sec
                )
            ]
            if not clip_events:
                return None

            clip_duration = clip_end - clip_start
            n_frames = cfg.compute_num_frames(clip_duration)
            frames = sample_frames_uniform(
                video_path=video_path,
                start_sec=clip_start,
                end_sec=clip_end,
                num_frames=n_frames,
            )
            if not frames:
                logger.warning(
                    "Camera %s clip %.1f-%.1f: no frames sampled; skipping",
                    cam_id, clip_start, clip_end,
                )
                return None

            t_clip = time.time()
            payload = refine_vector_events_with_llm(
                video_id=Path(video_path).name,
                clip={"start_sec": clip_start, "end_sec": clip_end},
                raw_events=clip_events,
                frames=frames,
                model=cfg.model,
                temperature=float(cfg.temperature),
                cross_camera_context=cross_ctx,
            )
            print(
                f"[clip-refine] {cam_id} clip [{clip_start:.1f}-{clip_end:.1f}s]: "
                f"{len(payload.events)} events, {len(frames)} frames in {time.time() - t_clip:.1f}s",
                flush=True,
            )
            return payload

        # Clips within a camera are independent → refine concurrently, preserving
        # clip order and isolating per-clip failures.
        clip_results: list[VectorEventsPayload | None] = [None] * len(clip_segments)
        cam_workers = min(_refine_concurrency(), max(1, len(clip_segments)))
        if cam_workers <= 1 or len(clip_segments) <= 1:
            for i, clip in enumerate(clip_segments):
                try:
                    clip_results[i] = _process_cam_clip(clip)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Camera %s clip #%d: VLM refinement failed; keeping other clips. Error: %s",
                        cam_id, i, exc,
                    )
        else:
            with ThreadPoolExecutor(max_workers=cam_workers) as executor:
                future_to_i = {
                    executor.submit(_process_cam_clip, clip): i
                    for i, clip in enumerate(clip_segments)
                }
                for future, i in future_to_i.items():
                    try:
                        clip_results[i] = future.result()
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "Camera %s clip #%d: VLM refinement failed; keeping other clips. Error: %s",
                            cam_id, i, exc,
                        )
        refined_list_vector: list[VectorEventsPayload] = [r for r in clip_results if r is not None]

        flat_events: list[dict[str, Any]] = []
        for ve in refined_list_vector:
            for ev in ve.events:
                d = ev.model_dump()
                d["video_id"] = d.get("video_id") or ve.video_id
                d["clip_start_sec"] = (
                    d.get("clip_start_sec")
                    if d.get("clip_start_sec") is not None
                    else ve.clip_start_sec
                )
                d["clip_end_sec"] = (
                    d.get("clip_end_sec")
                    if d.get("clip_end_sec") is not None
                    else ve.clip_end_sec
                )
                d["camera_id"] = cam_id
                flat_events.append(d)

        # Fallback: fill entity_hint for any events the LLM left null,
        # using time-overlap against global entity appearances on this camera.
        intervals = cam_appearance_intervals.get(cam_id, [])
        if intervals:
            for d in flat_events:
                if d.get("entity_hint"):
                    continue  # already filled (by LLM copying the hint)
                ev_s = float(d.get("start_time", 0.0))
                ev_e = float(d.get("end_time", 0.0))
                best_id: str | None = None
                best_overlap = 0.0
                for a_s, a_e, eid in intervals:
                    ov = max(0.0, min(ev_e, a_e) - max(ev_s, a_s))
                    if ov > best_overlap:
                        best_overlap = ov
                        best_id = eid
                if best_id:
                    d["entity_hint"] = best_id
                    logger.debug(
                        "Camera %s: entity_hint fallback filled %s for event %.1f-%.1f",
                        cam_id, best_id, ev_s, ev_e,
                    )

        results[cam_id] = {
            "video_id": Path(video_path).name,
            "events": flat_events,
        }

    return results


def run_refine_events_from_files(
    events_path: str | Path,
    clips_path: str | Path,
    config: RefineEventsConfig | None = None,
) -> Path:
    """Refine *_events.json + *_clips.json; write output and return path."""
    cfg = config or RefineEventsConfig()
    events_path = Path(events_path)
    clips_path = Path(clips_path)

    events_data = json.loads(events_path.read_text(encoding="utf-8"))
    clips_data = json.loads(clips_path.read_text(encoding="utf-8"))
    out_dict = run_refine_events_to_dict(events_data, clips_data, cfg)
    text = json.dumps(out_dict, ensure_ascii=False, indent=2)

    if cfg.mode == "vector":
        out_path = events_path.with_name(events_path.stem + "_vector_flat.json")
    elif out_dict.get("mode") == "full" and "refined_events" not in out_dict and "clips" not in out_dict:
        out_path = events_path.with_name(events_path.stem + "_refined_empty.json")
    elif isinstance(out_dict.get("clips"), list) and len(out_dict["clips"]) > 1:
        out_path = events_path.with_name(events_path.stem + "_refined_all.json")
    else:
        out_path = events_path.with_name(events_path.stem + "_refined.json")

    out_path.write_text(text, encoding="utf-8")
    return out_path
