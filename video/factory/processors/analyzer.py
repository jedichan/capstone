"""
Track aggregation and event slicing (no YOLO; consumes per-frame detection lists).
See README for factory/processors/analyzer.py responsibilities.
"""

from __future__ import annotations

from typing import Any


def _bbox_center(xyxy: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = xyxy
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def aggregate_tracks(
    fps: float,
    frame_detections: list[list[tuple[int | None, str, float, list[float]]]],
) -> list[dict[str, Any]]:
    """
    Aggregate per-frame detections by track_id into tracks.
    Each track has track_id, class_name, start/end_time, motion_score, etc.
    """
    tracks_raw: dict[int, list[tuple[int, float, float, float, str, float, list[float]]]] = {}
    for frame_idx, dets in enumerate(frame_detections):
        t_sec = frame_idx / fps if fps > 0 else 0.0
        for tid, cls_name, conf, xyxy in dets:
            if tid is None:
                continue
            cx, cy = _bbox_center(xyxy)
            if tid not in tracks_raw:
                tracks_raw[tid] = []
            tracks_raw[tid].append((frame_idx, t_sec, cx, cy, cls_name, conf, xyxy))

    tracks: list[dict[str, Any]] = []
    for tid, points in tracks_raw.items():
        if not points:
            continue
        frame_indices = [p[0] for p in points]
        times = [p[1] for p in points]
        positions = [(p[2], p[3]) for p in points]
        class_name = points[0][4]

        motion_sum = 0.0
        motion_count = 0
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i - 1][0]
            dy = positions[i][1] - positions[i - 1][1]
            motion_sum += (dx * dx + dy * dy) ** 0.5
            motion_count += 1
        motion_score = motion_sum / motion_count if motion_count else 0.0

        time_xyxy: list[tuple[float, list[float]]] = [(p[1], list(p[6])) for p in points]

        tracks.append({
            "track_id": tid,
            "class_name": class_name,
            "start_frame": min(frame_indices),
            "end_frame": max(frame_indices),
            "start_time": min(times),
            "end_time": max(times),
            "positions": positions,
            "frame_indices": frame_indices,
            "times": times,
            "time_xyxy": time_xyxy,
            "motion_score": round(motion_score, 2),
        })
    return tracks


def _bbox_at_time(time_xyxy: list[tuple[float, list[float]]], t_sec: float) -> list[float]:
    if not time_xyxy:
        return [0.0, 0.0, 0.0, 0.0]
    best = min(time_xyxy, key=lambda x: abs(x[0] - t_sec))
    return list(best[1])


def _build_motion_edges(tr: dict[str, Any]) -> list[tuple[float, float, float]]:
    edges: list[tuple[float, float, float]] = []
    fi, pos, times = tr["frame_indices"], tr["positions"], tr["times"]
    for i in range(1, len(pos)):
        gap = fi[i] - fi[i - 1]
        if gap <= 0:
            continue
        dx = pos[i][0] - pos[i - 1][0]
        dy = pos[i][1] - pos[i - 1][1]
        dist = (dx * dx + dy * dy) ** 0.5
        if gap > 15 or dist > 400:
            continue
        edges.append((times[i - 1], times[i], dist))
    return edges


def _find_motion_time_segments(
    edges: list[tuple[float, float, float]],
    t_start: float,
    t_end: float,
    window_sec: float,
    sum_threshold: float,
    sample_step: float = 0.15,
    jitter_floor_px: float = 3.0,
) -> list[tuple[float, float]]:
    if not edges or t_end <= t_start:
        return []
    moving_windows: list[tuple[float, float]] = []
    t = t_start
    while t <= t_end - 0.05:
        tw = t + window_sec
        s = 0.0
        for a, b, d in edges:
            if b <= t or a >= tw:
                continue
            overlap = min(b, tw) - max(a, t)
            if overlap > 0 and (b - a) > 1e-6:
                adj = max(0.0, d - jitter_floor_px)
                s += adj * (overlap / (b - a))
        if s >= sum_threshold:
            moving_windows.append((t, tw))
        t += sample_step

    if not moving_windows:
        return []

    moving_windows.sort(key=lambda x: x[0])
    merged: list[list[float]] = [[moving_windows[0][0], moving_windows[0][1]]]
    for a, b in moving_windows[1:]:
        if a <= merged[-1][1] + sample_step:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(float(x[0]), float(x[1])) for x in merged]


def _merge_overlapping_segments(
    segments: list[dict[str, float]],
    min_gap: float = 0.5,
) -> list[dict[str, float]]:
    if not segments:
        return []
    sorted_segs = sorted(segments, key=lambda s: s["start_sec"])
    out: list[dict[str, float]] = [dict(sorted_segs[0])]
    for s in sorted_segs[1:]:
        if s["start_sec"] <= out[-1]["end_sec"] + min_gap:
            out[-1]["end_sec"] = max(out[-1]["end_sec"], s["end_sec"])
        else:
            out.append(dict(s))
    return out


def _min_track_frames(class_name: str) -> int:
    """Suppress one-frame tracker fragments before they become events."""
    cls = str(class_name or "").lower()
    if cls == "car":
        return 5
    return 3


def _bbox_iou(a: list[float] | None, b: list[float] | None) -> float:
    if not a or not b or len(a) != 4 or len(b) != 4:
        return 0.0
    ax1, ay1, ax2, ay2 = [float(x) for x in a]
    bx1, by1, bx2, by2 = [float(x) for x in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _bbox_center_distance(a: list[float] | None, b: list[float] | None) -> float:
    if not a or not b or len(a) != 4 or len(b) != 4:
        return float("inf")
    ac = _bbox_center([float(x) for x in a])
    bc = _bbox_center([float(x) for x in b])
    dx = ac[0] - bc[0]
    dy = ac[1] - bc[1]
    return (dx * dx + dy * dy) ** 0.5


def _event_bbox_at_start(ev: dict[str, Any]) -> list[float] | None:
    bbox = ev.get("start_bbox_xyxy")
    if isinstance(bbox, list) and len(bbox) == 4:
        return [float(x) for x in bbox]
    return None


def _event_bbox_at_end(ev: dict[str, Any]) -> list[float] | None:
    bbox = ev.get("end_bbox_xyxy")
    if isinstance(bbox, list) and len(bbox) == 4:
        return [float(x) for x in bbox]
    return None


def _is_person_event(ev: dict[str, Any]) -> bool:
    return str(ev.get("class_name") or "").lower() in {"person", "people", "pedestrian"}


def _is_vehicle_event(ev: dict[str, Any]) -> bool:
    return str(ev.get("class_name") or "").lower() in {"car", "vehicle", "bus", "truck", "motorcycle", "bicycle"}


def _can_merge_person_events(
    prev: dict[str, Any],
    cur: dict[str, Any],
    *,
    max_gap_sec: float,
    max_center_dist_px: float,
    min_iou: float,
) -> bool:
    if not (_is_person_event(prev) and _is_person_event(cur)):
        return False
    prev_end = float(prev.get("end_time", 0.0))
    cur_start = float(cur.get("start_time", 0.0))
    gap = cur_start - prev_end
    if gap < 0 or gap > max_gap_sec:
        return False

    prev_box = _event_bbox_at_end(prev)
    cur_box = _event_bbox_at_start(cur)
    if prev_box is None or cur_box is None:
        return False
    if _bbox_iou(prev_box, cur_box) >= min_iou:
        return True
    return _bbox_center_distance(prev_box, cur_box) <= max_center_dist_px


def _can_merge_vehicle_events(
    prev: dict[str, Any],
    cur: dict[str, Any],
    *,
    max_gap_sec: float,
    max_center_dist_px: float,
    min_iou: float,
) -> bool:
    if not (_is_vehicle_event(prev) and _is_vehicle_event(cur)):
        return False
    prev_end = float(prev.get("end_time", 0.0))
    cur_start = float(cur.get("start_time", 0.0))
    gap = cur_start - prev_end
    if gap < 0 or gap > max_gap_sec:
        return False

    prev_box = _event_bbox_at_end(prev)
    cur_box = _event_bbox_at_start(cur)
    if prev_box is None or cur_box is None:
        return False
    if _bbox_iou(prev_box, cur_box) >= min_iou:
        return True
    return _bbox_center_distance(prev_box, cur_box) <= max_center_dist_px


def _merge_person_event_fragments(
    events: list[dict[str, Any]],
    *,
    max_gap_sec: float = 2.5,
    max_center_dist_px: float = 90.0,
    min_iou: float = 0.05,
) -> list[dict[str, Any]]:
    """Conservatively join short adjacent person fragments caused by tracker ID switches."""
    merged: list[dict[str, Any]] = []
    for ev in sorted(events, key=lambda e: (float(e.get("start_time", 0.0)), float(e.get("end_time", 0.0)))):
        if not merged or not _can_merge_person_events(
            merged[-1],
            ev,
            max_gap_sec=max_gap_sec,
            max_center_dist_px=max_center_dist_px,
            min_iou=min_iou,
        ):
            merged.append(dict(ev))
            continue

        dst = merged[-1]
        dst["end_time"] = max(float(dst.get("end_time", 0.0)), float(ev.get("end_time", 0.0)))
        dst["end_bbox_xyxy"] = ev.get("end_bbox_xyxy") or dst.get("end_bbox_xyxy")
        dst["motion_level"] = "high" if "high" in {dst.get("motion_level"), ev.get("motion_level")} else dst.get("motion_level", "low")
        dst["event_type"] = "person_track_fragment_merged"
        dst["description_for_llm"] = "person appears as adjacent tracker fragments and is treated as one continuous event."

        track_ids = list(dst.get("merged_track_ids") or [dst.get("track_id")])
        if ev.get("track_id") not in track_ids:
            track_ids.append(ev.get("track_id"))
        dst["merged_track_ids"] = [tid for tid in track_ids if tid is not None]

    return sorted(merged, key=lambda e: float(e.get("start_time", 0.0)))


def _merge_vehicle_event_fragments(
    events: list[dict[str, Any]],
    *,
    max_gap_sec: float = 3.0,
    max_center_dist_px: float = 80.0,
    min_iou: float = 0.35,
) -> list[dict[str, Any]]:
    """Conservatively join short adjacent vehicle fragments caused by tracker splits."""
    merged: list[dict[str, Any]] = []
    for ev in sorted(events, key=lambda e: (float(e.get("start_time", 0.0)), float(e.get("end_time", 0.0)))):
        if not merged or not _can_merge_vehicle_events(
            merged[-1],
            ev,
            max_gap_sec=max_gap_sec,
            max_center_dist_px=max_center_dist_px,
            min_iou=min_iou,
        ):
            merged.append(dict(ev))
            continue

        dst = merged[-1]
        dst["end_time"] = max(float(dst.get("end_time", 0.0)), float(ev.get("end_time", 0.0)))
        dst["end_bbox_xyxy"] = ev.get("end_bbox_xyxy") or dst.get("end_bbox_xyxy")
        dst["motion_level"] = "high" if "high" in {dst.get("motion_level"), ev.get("motion_level")} else dst.get("motion_level", "low")
        dst["event_type"] = "vehicle_track_fragment_merged"
        dst["description_for_llm"] = "vehicle appears as adjacent tracker fragments and is treated as one continuous event."

        track_ids = list(dst.get("merged_track_ids") or [dst.get("track_id")])
        if ev.get("track_id") not in track_ids:
            track_ids.append(ev.get("track_id"))
        dst["merged_track_ids"] = [tid for tid in track_ids if tid is not None]

    return sorted(merged, key=lambda e: float(e.get("start_time", 0.0)))


def _merge_same_track_semantic_events(
    events: list[dict[str, Any]],
    *,
    max_gap_sec: float = 2.0,
    max_center_dist_px: float = 85.0,
    min_iou: float = 0.08,
) -> list[dict[str, Any]]:
    """Merge adjacent events from the same camera+track to reduce over-segmentation.

    This is a conservative post-pass:
    - only merges when `camera_id` and `track_id` are the same,
    - only within a short time gap,
    - only if boxes are spatially continuous (IoU/center distance),
    - keeps people and vehicles separate through class_name matching.
    """
    if not events:
        return []
    out: list[dict[str, Any]] = []
    ordered = sorted(events, key=lambda e: (str(e.get("camera_id", "")), str(e.get("track_id", "")), float(e.get("start_time", 0.0))))
    for ev in ordered:
        cur = dict(ev)
        if not out:
            out.append(cur)
            continue

        prev = out[-1]
        same_track = (
            str(prev.get("camera_id", "")) == str(cur.get("camera_id", ""))
            and str(prev.get("track_id", "")) == str(cur.get("track_id", ""))
            and str(prev.get("class_name", "")).lower() == str(cur.get("class_name", "")).lower()
        )
        if not same_track:
            out.append(cur)
            continue

        prev_end = float(prev.get("end_time", 0.0))
        cur_start = float(cur.get("start_time", 0.0))
        gap = cur_start - prev_end
        if gap < 0 or gap > max_gap_sec:
            out.append(cur)
            continue

        prev_box = _event_bbox_at_end(prev)
        cur_box = _event_bbox_at_start(cur)
        if prev_box is None or cur_box is None:
            out.append(cur)
            continue

        if not (_bbox_iou(prev_box, cur_box) >= min_iou or _bbox_center_distance(prev_box, cur_box) <= max_center_dist_px):
            out.append(cur)
            continue

        prev["end_time"] = max(float(prev.get("end_time", 0.0)), float(cur.get("end_time", 0.0)))
        prev["end_bbox_xyxy"] = cur.get("end_bbox_xyxy") or prev.get("end_bbox_xyxy")
        prev["motion_level"] = "high" if "high" in {prev.get("motion_level"), cur.get("motion_level")} else prev.get("motion_level", "low")
        prev["event_type"] = str(prev.get("event_type") or "track_event_merged")
        prev["description_for_llm"] = (
            f"{prev.get('class_name', 'object')} (id={prev.get('track_id')}) "
            "continuous motion/presence merged from adjacent segments."
        )
    return sorted(out, key=lambda e: float(e.get("start_time", 0.0)))


def slice_events(
    tracks: list[dict[str, Any]],
    fps: float,
    frame_detections: list[list[tuple[int | None, str, float, list[float]]]],
    motion_threshold: float = 3.0,
    min_clip_duration: float = 1.0,
    max_static_duration: float = 30.0,
    motion_window_sec: float = 1.5,
    motion_window_sum_threshold: float = 24.0,
    motion_segment_pad_sec: float = 1.0,
):
    """Build events and clip segments from tracks."""
    events: list[dict[str, Any]] = []
    clip_segments: list[dict[str, float]] = []

    for tr in tracks:
        if len(tr.get("frame_indices", [])) < _min_track_frames(str(tr.get("class_name", ""))):
            continue

        t_start = tr["start_time"]
        t_end = tr["end_time"]
        duration = t_end - t_start
        motion_mean = tr["motion_score"]
        edges = _build_motion_edges(tr)
        motion_segs = _find_motion_time_segments(
            edges,
            t_start,
            t_end,
            motion_window_sec,
            motion_window_sum_threshold,
        )

        if not motion_segs and duration < 25 and edges:
            max_edge = max(e[2] for e in edges)
            total_adj = sum(max(0.0, e[2] - 3.0) for e in edges)
            if max_edge >= 12 or total_adj >= motion_window_sum_threshold * 0.8:
                motion_segs = [(t_start, t_end)]

        if motion_segs:
            for ms, me in motion_segs:
                cs = max(t_start, ms - motion_segment_pad_sec)
                ce = min(t_end, me + motion_segment_pad_sec)
                if ce - cs < min_clip_duration:
                    ce = min(t_end, cs + min_clip_duration)
                events.append({
                    "event_type": "motion_segment",
                    "track_id": tr["track_id"],
                    "class_name": tr["class_name"],
                    "start_time": round(cs, 3),
                    "end_time": round(ce, 3),
                    "start_bbox_xyxy": _bbox_at_time(tr["time_xyxy"], cs),
                    "end_bbox_xyxy": _bbox_at_time(tr["time_xyxy"], ce),
                    "motion_level": "high",
                    "motion_window_sec": motion_window_sec,
                    "description_for_llm": (
                        f"{tr['class_name']} (id={tr['track_id']}) shows significant movement "
                        f"in this interval (e.g. driving in, walking)."
                    ),
                })
                if ce - cs >= min_clip_duration * 0.5:
                    clip_segments.append({"start_sec": cs, "end_sec": ce})

            last_m_end = motion_segs[-1][1]
            if t_end - last_m_end >= 5.0:
                events.append({
                    "event_type": "presence_after_motion",
                    "track_id": tr["track_id"],
                    "class_name": tr["class_name"],
                    "start_time": round(last_m_end, 3),
                    "end_time": round(t_end, 3),
                    "start_bbox_xyxy": _bbox_at_time(tr["time_xyxy"], last_m_end),
                    "end_bbox_xyxy": _bbox_at_time(tr["time_xyxy"], t_end),
                    "motion_level": "low",
                    "description_for_llm": (
                        f"{tr['class_name']} (id={tr['track_id']}) mostly stationary after moving."
                    ),
                })

        elif motion_mean >= motion_threshold:
            t_app_end = t_start + min(min_clip_duration, duration)
            t_dis_start = max(t_end - min_clip_duration, t_start)
            events.append({
                "event_type": "appearance",
                "track_id": tr["track_id"],
                "class_name": tr["class_name"],
                "start_time": t_start,
                "end_time": t_app_end,
                "start_bbox_xyxy": _bbox_at_time(tr["time_xyxy"], t_start),
                "end_bbox_xyxy": _bbox_at_time(tr["time_xyxy"], t_app_end),
                "motion_level": "high",
                "description_for_llm": f"{tr['class_name']} (id={tr['track_id']}) appears and moves.",
            })
            events.append({
                "event_type": "disappearance",
                "track_id": tr["track_id"],
                "class_name": tr["class_name"],
                "start_time": t_dis_start,
                "end_time": t_end,
                "start_bbox_xyxy": _bbox_at_time(tr["time_xyxy"], t_dis_start),
                "end_bbox_xyxy": _bbox_at_time(tr["time_xyxy"], t_end),
                "motion_level": "high",
                "description_for_llm": f"{tr['class_name']} (id={tr['track_id']}) leaves.",
            })
            t = t_start
            while t < t_end:
                seg_end = min(t + max_static_duration, t_end)
                if seg_end - t >= min_clip_duration:
                    clip_segments.append({"start_sec": t, "end_sec": seg_end})
                t = seg_end
        else:
            clip_dur = min(duration, max_static_duration)
            if duration >= min_clip_duration:
                events.append({
                    "event_type": "presence_static",
                    "track_id": tr["track_id"],
                    "class_name": tr["class_name"],
                    "start_time": t_start,
                    "end_time": t_end,
                    "start_bbox_xyxy": _bbox_at_time(tr["time_xyxy"], t_start),
                    "end_bbox_xyxy": _bbox_at_time(tr["time_xyxy"], t_end),
                    "motion_level": "low",
                    "description_for_llm": f"{tr['class_name']} (id={tr['track_id']}) present with little motion.",
                })
                clip_segments.append({"start_sec": t_start, "end_sec": t_start + clip_dur})

    events = _merge_person_event_fragments(events)
    events = _merge_vehicle_event_fragments(events)
    events = _merge_same_track_semantic_events(events)
    merged = _merge_overlapping_segments(clip_segments, min_gap=0.8)
    return events, merged
