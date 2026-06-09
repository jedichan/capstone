"""LangChain + OpenAI multimodal refinement for pipeline events (no frame sampling or CLI here)."""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI

from video.common.frames import FrameSample, PersonCrop, coarse_color_label_from_bgr, crop_bgr_at_time_xyxy
from video.core.schema.multi_camera import MatchVerification
from video.core.schema.refined_event_llm import RefinedEntity, RefinedEventsPayload, VectorEventsPayload

logger = logging.getLogger(__name__)


def _make_llm(model: str, temperature: float = 0.1) -> ChatOpenAI:
    """Build ChatOpenAI, routing Qwen models through DashScope compatible endpoint."""
    timeout_sec = float(os.environ.get("VIDEO_VLM_TIMEOUT_SEC", "180"))
    max_retries = int(os.environ.get("VIDEO_VLM_MAX_RETRIES", "1"))
    if model.lower().startswith("qwen"):
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=os.environ.get("DASHSCOPE_API_KEY", ""),
            base_url=os.environ.get(
                "DASHSCOPE_URL",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            ),
            timeout=timeout_sec,
            max_retries=max_retries,
        )
    return ChatOpenAI(model=model, temperature=temperature, timeout=timeout_sec, max_retries=max_retries)


# ─────────────────────────────────────────────────────────────────────────────
# Slim event serialisation  (saves ~80% tokens vs indent=2 JSON)
# ─────────────────────────────────────────────────────────────────────────────

_EV_TYPE_SHORT: dict[str, str] = {
    "motion_segment":        "mv",   # object actively moving
    "presence_after_motion": "pam",  # object still after moving
    "appearance":            "app",  # object enters scene
    "disappearance":         "dis",  # object leaves scene
    "presence_static":       "sta",  # object present, barely moving
}

# One-line key for pasting into prompts so the LLM knows what the abbreviations mean
SLIM_EVENT_LEGEND = (
    "# raw_events compact format — "
    "ev: mv=moving, pam=still_after_move, app=appear, dis=disappear, sta=static | "
    "id=track_id | cls=class | s/e=start/end_sec | b0/b1=start/end_bbox_xyxy(px) | "
    "hint=entity_hint(copy verbatim to output)\n"
)


def _slim_event(e: dict[str, Any]) -> dict[str, Any]:
    """Convert a full pipeline event dict to a compact LLM-input form.

    Full event  ≈ 130 tokens (indent=2 JSON)
    Slim event  ≈  25 tokens (compact JSON)
    Savings     ≈  80%

    The original dict is NOT mutated; it is only used for reading.
    """
    raw_b0 = e.get("start_bbox_xyxy") or [0, 0, 0, 0]
    raw_b1 = e.get("end_bbox_xyxy")   or [0, 0, 0, 0]
    out: dict[str, Any] = {
        "ev":  _EV_TYPE_SHORT.get(str(e.get("event_type", "")), e.get("event_type", "?")),
        "id":  e.get("track_id"),
        "cls": e.get("class_name"),
        "s":   round(float(e.get("start_time", 0)), 1),
        "e":   round(float(e.get("end_time",   0)), 1),
        "b0":  [int(v) for v in raw_b0],
        "b1":  [int(v) for v in raw_b1],
    }
    # Carry entity_hint if pre-injected by caller (multi-camera pipeline)
    hint = e.get("entity_hint")
    if hint:
        out["hint"] = hint
    return out


def _compact_events_str(raw_events: list[dict[str, Any]]) -> str:
    """Serialise events as a compact one-liner with legend header."""
    slim = [_slim_event(e) for e in raw_events]
    return SLIM_EVENT_LEGEND + json.dumps(slim, separators=(",", ":"), ensure_ascii=False)


@dataclass(frozen=True)
class _TrackSummary:
    class_name: str
    start_time: float
    end_time: float
    rep_t_sec: float
    rep_bbox_xyxy: list[float]
    coarse_color: str


def _time_overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _summarize_tracks_for_merge(video_path: str, raw_events: list[dict[str, Any]]) -> dict[int, _TrackSummary]:
    """
    Aggregate time range per track_id from raw_events and estimate coarse color from a representative bbox crop.
    """
    by_tid: dict[int, list[dict[str, Any]]] = {}
    for e in raw_events:
        tid = e.get("track_id")
        if tid is None:
            continue
        by_tid.setdefault(int(tid), []).append(e)

    out: dict[int, _TrackSummary] = {}
    for tid, evs in by_tid.items():
        cls = str(evs[0].get("class_name", "unknown"))
        s = float(min(float(e.get("start_time", 0.0)) for e in evs))
        t = float(max(float(e.get("end_time", 0.0)) for e in evs))
        rep = evs[0]
        rep_t = float(rep.get("start_time", s))
        rep_bbox = rep.get("start_bbox_xyxy") or rep.get("end_bbox_xyxy") or [0, 0, 0, 0]
        rep_bbox = [float(x) for x in rep_bbox]
        crop = crop_bgr_at_time_xyxy(video_path, rep_t, rep_bbox)
        coarse_color = coarse_color_label_from_bgr(crop) if crop is not None else "unknown"
        out[tid] = _TrackSummary(
            class_name=cls,
            start_time=s,
            end_time=t,
            rep_t_sec=rep_t,
            rep_bbox_xyxy=rep_bbox,
            coarse_color=coarse_color,
        )
    return out


class _MergeDecision(RefinedEventsPayload.__class__):  # type: ignore[misc]
    pass


def _verify_merge_yesno_with_llm(
    *,
    video_path: str,
    a_tid: int,
    b_tid: int,
    a: _TrackSummary,
    b: _TrackSummary,
    model: str,
    temperature: float = 0.0,
) -> tuple[bool, float]:
    """
    LLM may only output YES/NO + confidence. Returns (is_yes, confidence).
    """
    from pydantic import BaseModel, Field

    class MergeYesNo(BaseModel):
        answer: Literal["YES", "NO"] = Field(description="Must be YES or NO only")
        confidence: float = Field(ge=0.0, le=1.0, description="Confidence between 0 and 1")

    parser = PydanticOutputParser(pydantic_object=MergeYesNo)

    crop_a = crop_bgr_at_time_xyxy(video_path, a.rep_t_sec, a.rep_bbox_xyxy)
    crop_b = crop_bgr_at_time_xyxy(video_path, b.rep_t_sec, b.rep_bbox_xyxy)
    images: list[dict[str, Any]] = []
    if crop_a is not None:
        # reuse PersonCrop encoder path
        pc_a = PersonCrop(t_sec=a.rep_t_sec, camera_id="single", track_id=a_tid, image_array=crop_a, jpg_base64="")
        pc_a.jpg_base64 = ""  # placeholder
        images.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{FrameSample(0,'').jpg_base64}"}})
    # We cannot easily reuse encoder without importing; so we pass no images if crop fails.
    # Note: model can still decide from metadata conservatively.

    system = (
        "You are a surveillance-video entity merge judge. "
        "Decide whether two track_ids belong to the same real-world target. "
        "You must output strict JSON with only YES/NO and a 0~1 confidence."
    )
    user_text = (
        f"track_a: id={a_tid}, class={a.class_name}, time=[{a.start_time:.3f},{a.end_time:.3f}], color_guess={a.coarse_color}\n"
        f"track_b: id={b_tid}, class={b.class_name}, time=[{b.start_time:.3f},{b.end_time:.3f}], color_guess={b.coarse_color}\n"
        "Same target? Answer YES only when very confident; otherwise NO.\n"
        f"{parser.get_format_instructions()}"
    )
    llm = _make_llm(model, temperature)
    resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=[{"type": "text", "text": user_text}])])
    text = resp.content if isinstance(resp.content, str) else str(resp.content)
    parsed = parser.parse(text)
    return parsed.answer == "YES", float(parsed.confidence)


def build_entities_with_hard_constraints(
    *,
    video_path: str,
    raw_events: list[dict[str, Any]],
    model: str,
    max_gap_sec: float = 300.0,
    min_llm_confidence: float = 0.75,
) -> list[RefinedEntity]:
    """
    Filter candidate pairs with hard rules first, then LLM YES/NO + confidence for merging.

    Hard rules (skip LLM if any fails):
    - Temporal overlap -> cannot be same target
    - Clearly different coarse color -> skip
    - Time gap > 5min -> skip
    """
    tracks = _summarize_tracks_for_merge(video_path, raw_events)
    tids = sorted(tracks.keys())

    parent: dict[int, int] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(len(tids)):
        for j in range(i + 1, len(tids)):
            a_tid, b_tid = tids[i], tids[j]
            a, b = tracks[a_tid], tracks[b_tid]
            if a.class_name != b.class_name:
                continue
            if _time_overlap(a.start_time, a.end_time, b.start_time, b.end_time) > 0:
                continue
            gap = min(abs(b.start_time - a.end_time), abs(a.start_time - b.end_time))
            if gap > max_gap_sec:
                continue
            if a.coarse_color != "unknown" and b.coarse_color != "unknown" and a.coarse_color != b.coarse_color:
                continue
            yes, conf = _verify_merge_yesno_with_llm(
                video_path=video_path,
                a_tid=a_tid,
                b_tid=b_tid,
                a=a,
                b=b,
                model=model,
                temperature=0.0,
            )
            if yes and conf > min_llm_confidence:
                union(a_tid, b_tid)

    groups: dict[int, list[int]] = {}
    for tid in tids:
        groups.setdefault(find(tid), []).append(tid)

    entities: list[RefinedEntity] = []
    for idx, (_, members) in enumerate(sorted(groups.items(), key=lambda x: min(x[1])), start=1):
        cls = tracks[members[0]].class_name
        entities.append(
            RefinedEntity(
                entity_id=f"{cls}_{idx}",
                class_name=cls,
                local_track_ids=sorted(members),
                appearance={"color": tracks[members[0]].coarse_color, "color_confidence": 0.5},
                location={},
                notes="Entities merged via hard-rule filtering + LLM YES/NO.",
            )
        )
    return entities


def refine_events_with_llm(
    *,
    video_path: str,
    clip: dict[str, float],
    raw_events: list[dict[str, Any]],
    frames: list[FrameSample],
    model: str = "qwen3-vl-flash",
    temperature: float = 0.1,
    max_time_adjust_sec: float = 0.5,
    merge_location_iou_threshold: float = 0.9,
    merge_center_dist_px: float = 30.0,
    merge_location_norm_diff: float = 0.10,
    pre_entities: list[RefinedEntity] | None = None,
) -> RefinedEventsPayload:
    """Send sampled frames + raw events to the LLM; return corrected refined_events (strict JSON)."""
    parser = PydanticOutputParser(pydantic_object=RefinedEventsPayload)

    images_content: list[dict[str, Any]] = []
    for f in frames:
        images_content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{f.jpg_base64}"},
            }
        )

    system = (
        "You are a surveillance-video event labeling and correction assistant. "
        "You work from timestamped key frames and a draft event list (from YOLO+tracking). "
        "Correct semantics and add action detail. "
        "First understand the scene (parking lot, road, entrance/exit, lane sides, sidewalk, etc.). "
        "For each event explanation, describe both how the subject moves and which scene sub-region it is in "
        "(e.g. passing mid-right on the road, entering the parking entrance, walking on the sidewalk near the exit). "
        "Most important: preserve pipeline temporal information; do not change times without strong evidence. "
        "Use English for all free-text string values. Output must strictly match the given JSON schema."
    )

    user_text = (
        f"Video path: {video_path}\n"
        f"Clip under analysis: start_sec={clip['start_sec']}, end_sec={clip['end_sec']}\n\n"
        "Raw pipeline events for this clip (may contain false positives, bad timing, or split duplicates):\n"
        f"{_compact_events_str(raw_events)}\n\n"
        "Uniformly sampled key frames for this clip (timestamps are listed below; state in evidence which frame times you rely on):\n"
        + "\n".join([f"- frame_time_sec={f.t_sec:.3f}" for f in frames])
        + "\n\n"
        "Tasks (prioritize preserving timing; always include scene + motion in English):\n"
        "0) Emit scene_context: summarize in English where the road vs parking vs entrances/exits/main paths appear "
        "(left/center/right, far/near).\n"
        "1) entities will be provided by the system when applicable — use them verbatim; do not add or merge entities.\n"
        f"     Location-merge guidance (tunable):\n"
        f"     - Merge only when you are very sure it is the same physical target.\n"
        f"     - For cars, do not merge different track_ids unless all hold:\n"
        f"       (a) time is contiguous/adjacent (<=1s)\n"
        f"       (b) bboxes almost coincide: IoU >= {merge_location_iou_threshold}\n"
        f"       (c) color/appearance matches (e.g. both white/silver, no contradictory cues)\n"
        f"       Center distance <= {merge_center_dist_px} px is supporting evidence only, not sufficient alone.\n"
        f"     - Hard proportional constraint:\n"
        f"       raw_events may include start_center_norm/end_center_norm (0~1 normalized centers).\n"
        f"       If normalized center delta on x or y exceeds {merge_location_norm_diff} (~10% of frame width/height), forbid merge.\n"
        f"       Exception only if frames show near-identical appearance (color + salient features) and continuous time; explain in notes.\n"
        "     - Stationary cars: merge broken track_ids only if same fixed slot, bbox stable over time, same color.\n"
        "     - Persons: merge cautiously only when clothing, body shape, and position continuity all agree.\n"
        "   - Each entity location must include: scene_zone, region_text, movement_in_scene (one English sentence).\n"
        "   - Each entity must include appearance (at least color + color_confidence) in English.\n"
        "2) refined_events:\n"
        "   - Bind events to entity_id (optionally keep one representative track_id).\n"
        "   - details must include movement_scene_narrative: one English sentence for motion + sub-region (entrance/exit/mid-road right/slot, etc.).\n"
        "   - location must include: scene_zone, start_scene, end_scene, movement_scene (English strings); "
        "keep start_bbox_xyxy/end_bbox_xyxy from raw_events when present and relate them to the scene.\n"
        "3) Temporal correction (hard rules):\n"
        f"   - Treat raw_events start_time/end_time as a strong prior.\n"
        f"   - Adjust boundaries only with clear frame evidence; one-sided shift <= {max_time_adjust_sec} s.\n"
        "   - If timing seems wrong but evidence is weak: do not change times; write \"uncertain\" in evidence.\n"
        "4) False positives:\n"
        "   - Remove obvious false events or lower confidence with a short English reason.\n\n"
        f"{parser.get_format_instructions()}"
    )

    if pre_entities is not None:
        user_text += "\n\nPre-defined entities (keep exactly; do not add/merge/delete; only reference entity_id in refined_events):\n"
        user_text += json.dumps([e.model_dump() for e in pre_entities], ensure_ascii=False, indent=2)

    user_content: list[dict[str, Any]] = [
        {"type": "text", "text": user_text},
        *images_content,
    ]
    llm = _make_llm(model, temperature)
    resp = llm.invoke(
        [
            SystemMessage(content=system),
            HumanMessage(content=user_content),
        ]
    )
    text = resp.content if isinstance(resp.content, str) else str(resp.content)
    return parser.parse(text)


def refine_vector_events_with_llm(
    *,
    video_id: str,
    clip: dict[str, float],
    raw_events: list[dict[str, Any]],
    frames: list[FrameSample],
    model: str = "qwen3-vl-flash",
    temperature: float = 0.0,
    cross_camera_context: str | None = None,
) -> VectorEventsPayload:
    """Production vector-store mode: minimal fields for retrieval; never change start/end times.

    cross_camera_context: optional pre-formatted string describing how global entities
    move across cameras (injected into prompt when provided by multi-camera pipeline).
    """
    parser = PydanticOutputParser(pydantic_object=VectorEventsPayload)

    images_content: list[dict[str, Any]] = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{f.jpg_base64}"}}
        for f in frames
    ]

    system = (
        "You are a surveillance-video event extractor. "
        "From raw events (with start_time/end_time and bbox) and key frames, output minimal records for vector retrieval. "
        "Hard rule: never modify start_time/end_time — copy them exactly. "
        "No extra fields, no Markdown, no prose outside JSON."
    )

    cross_cam_block = (
        f"\n[cross-camera entity trajectories]\n{cross_camera_context}\n"
        "Use these trajectories to enrich event_text with cross-camera movement "
        "(e.g. 'entity_003 entered from cam1 3s ago'). "
        "Only reference entities whose track_id appears in the raw_events above.\n"
        if cross_camera_context else ""
    )

    user_text = (
        f"video_id: {video_id}\n"
        f"clip: start_sec={clip['start_sec']}, end_sec={clip['end_sec']}\n\n"
        "raw_events (copy s/e times exactly into start_time/end_time of your output):\n"
        f"{_compact_events_str(raw_events)}\n\n"
        "Key frame timestamps:\n"
        + "\n".join([f"- t={f.t_sec:.3f}" for f in frames])
        + cross_cam_block
        + "\nOutput rules:\n"
        "- JSON only; must match the schema.\n"
        "- Top-level video_id/clip_start_sec/clip_end_sec are required. "
        "For each event, include them too when possible; if omitted, the caller will copy them from the top level.\n"
        "- track_id: copy the raw event 'id' value exactly into track_id. If id is missing, use null.\n"
        "- event_text: One English sentence: '<time range> — <subject with coarse color> <action> <scene region>'. "
        "  If cross-camera context is provided and the track matches a global entity, mention it "
        "  (e.g. 'entity_global_3, last seen on G329, walked into frame from the left').\n"
        "- object_color: coarse bucket — white/black/silver_gray/red/blue/green/yellow/dark/light/unknown (English).\n"
        "- appearance_notes: visible clothing & physical description in English. "
        "  For persons include: (1) upper-body clothing color+type (e.g. 'dark blue hoodie'), "
        "  (2) lower-body clothing color+type (e.g. 'grey pants' or 'unknown'), "
        "  (3) accessories if visible (backpack/bag/hat/cap/umbrella/etc.), "
        "  (4) rough build/height if estimable (tall/medium/short, slim/heavy). "
        "  For vehicles include: color, vehicle type (sedan/SUV/truck/bus/etc.), plate if readable. "
        "  Use 'unknown' for anything not visible. Keep it concise (1–2 short phrases).\n"
        "- scene_zone: one token from sidewalk/road/parking/entrance/exit/crosswalk/indoor/unknown.\n"
        "- entity_hint: if the raw event has a 'hint' field, copy its value verbatim here; "
        "  otherwise write null.\n"
        "- keywords: 3–6 short English retrieval tokens, e.g. walking/running/standing/dark_hoodie/backpack/entrance.\n"
        f"{parser.get_format_instructions()}"
    )

    llm = _make_llm(model, temperature)
    resp = llm.invoke(
        [
            SystemMessage(content=system),
            HumanMessage(content=[{"type": "text", "text": user_text}, *images_content]),
        ]
    )
    text = resp.content if isinstance(resp.content, str) else str(resp.content)

    # Pre-parse cleanup: LLMs occasionally append a bare wrapper-level dict
    # (e.g. {"video_id": "..."}) as an extra event, which fails required-field
    # validation.  Strip any event entry that is missing required VectorEvent fields
    # before handing off to Pydantic so the valid events are not thrown away.
    _REQUIRED_EVENT_FIELDS = {
        "start_time", "end_time", "object_type", "object_color",
        "appearance_notes", "scene_zone", "event_text", "keywords",
    }
    try:
        import json as _json
        raw_dict = _json.loads(text)
        if isinstance(raw_dict.get("events"), list):
            orig_len = len(raw_dict["events"])
            raw_dict["events"] = [
                ev for ev in raw_dict["events"]
                if _REQUIRED_EVENT_FIELDS.issubset(ev.keys())
            ]
            dropped = orig_len - len(raw_dict["events"])
            if dropped:
                logger.warning(
                    "refine_vector_events_with_llm: dropped %d malformed event(s) "
                    "(missing required fields) before Pydantic validation",
                    dropped,
                )
            text = _json.dumps(raw_dict)
    except Exception:
        pass  # let parser.parse surface the real error

    return parser.parse(text)


# ─────────────────────────────────────────────────────────────────────────────
# Batched refinement  —  handles arbitrarily large event lists
# ─────────────────────────────────────────────────────────────────────────────

def _frames_for_window(
    frames: list[FrameSample],
    t_start: float,
    t_end: float,
) -> list[FrameSample]:
    """Return frames whose timestamp falls within [t_start, t_end].
    Always includes at least the single closest frame so no batch is imageless."""
    window = [f for f in frames if t_start <= f.t_sec <= t_end]
    if not window and frames:
        window = [min(frames, key=lambda f: abs(f.t_sec - (t_start + t_end) / 2))]
    return window


def refine_events_batched(
    *,
    video_path: str,
    clip: dict[str, float],
    raw_events: list[dict[str, Any]],
    frames: list[FrameSample],
    model: str = "qwen3-vl-flash",
    temperature: float = 0.1,
    max_events_per_batch: int = 20,
    max_time_adjust_sec: float = 0.5,
    merge_location_iou_threshold: float = 0.9,
    merge_center_dist_px: float = 30.0,
    merge_location_norm_diff: float = 0.10,
) -> RefinedEventsPayload:
    """Split raw_events into batches of ≤ max_events_per_batch, call the LLM once
    per batch, then merge all partial RefinedEventsPayload results into one.

    When event count ≤ max_events_per_batch the function degrades gracefully to a
    single call (identical behaviour to refine_events_with_llm).

    Merge strategy
    ──────────────
    • scene_context  : taken from the first batch (widest temporal view).
    • entities       : concatenated; track_id de-duplication prevents double-counting
                       across batches (same id in two batches → kept once, later
                       batch wins for field values).
    • refined_events : concatenated in chronological order.
    • temporal_policy / location_policy : taken from the first batch.
    """
    if not raw_events:
        return refine_events_with_llm(
            video_path=video_path, clip=clip, raw_events=[], frames=frames,
            model=model, temperature=temperature,
        )

    # ── Sort events by start_time so batches are time-contiguous ──────────────
    sorted_events = sorted(raw_events, key=lambda e: float(e.get("start_time", 0)))

    n_batches = math.ceil(len(sorted_events) / max_events_per_batch)
    logger.info(
        "refine_events_batched: %d events → %d batch(es) of ≤%d",
        len(sorted_events), n_batches, max_events_per_batch,
    )

    partial_payloads: list[RefinedEventsPayload] = []

    for batch_idx in range(n_batches):
        batch_events = sorted_events[
            batch_idx * max_events_per_batch : (batch_idx + 1) * max_events_per_batch
        ]

        # Time window for this batch
        t_start = float(batch_events[0].get("start_time",  clip["start_sec"]))
        t_end   = float(batch_events[-1].get("end_time",   clip["end_sec"]))
        batch_clip = {"start_sec": t_start, "end_sec": t_end}

        # Only pass frames that fall in this time window
        batch_frames = _frames_for_window(frames, t_start, t_end)

        logger.info(
            "  Batch %d/%d: events=%d  t=[%.1f, %.1f]  frames=%d",
            batch_idx + 1, n_batches, len(batch_events), t_start, t_end, len(batch_frames),
        )

        partial = refine_events_with_llm(
            video_path=video_path,
            clip=batch_clip,
            raw_events=batch_events,
            frames=batch_frames,
            model=model,
            temperature=temperature,
            max_time_adjust_sec=max_time_adjust_sec,
            merge_location_iou_threshold=merge_location_iou_threshold,
            merge_center_dist_px=merge_center_dist_px,
            merge_location_norm_diff=merge_location_norm_diff,
        )
        partial_payloads.append(partial)

    # ── Merge partial payloads ────────────────────────────────────────────────
    first = partial_payloads[0]

    # De-duplicate entities: track_id is the identity key; later batch wins
    entity_by_tid: dict[str, RefinedEntity] = {}
    for payload in partial_payloads:
        for ent in payload.entities:
            key = f"{ent.class_name}_{ent.local_track_ids}"
            entity_by_tid[key] = ent   # last-write wins → preserves temporal order
    merged_entities = list(entity_by_tid.values())

    # Concatenate refined_events sorted by start_time
    all_refined = [
        ev
        for payload in partial_payloads
        for ev in payload.refined_events
    ]
    all_refined.sort(key=lambda ev: ev.start_time)

    return RefinedEventsPayload(
        video_path=video_path,
        analyzed_clip={"start_sec": clip["start_sec"], "end_sec": clip["end_sec"]},
        scene_context=first.scene_context,         # widest temporal view
        entities=merged_entities,
        refined_events=all_refined,
        temporal_policy=first.temporal_policy,
        location_policy=first.location_policy,
    )


# ------------------------------------------------------------------
# Cross-camera person match verification
# ------------------------------------------------------------------

def verify_person_match_with_llm(
    crop_a: PersonCrop,
    crop_b: PersonCrop,
    model: str = "qwen3-vl-flash",
    temperature: float = 0.0,
) -> MatchVerification:
    """Send two person crops to the VLM to decide if they are the same person."""
    system = (
        "You are a surveillance person-reidentification assistant. "
        "You receive two person crops from different cameras. "
        "Decide whether they depict the same individual. "
        "Use clothing color, build, accessories, hair, and other visible traits. "
        'Output strict JSON: {"is_match": bool, "confidence": 0.0~1.0, "reasoning": "..."}'
    )

    user_content: list[dict[str, Any]] = [
        {"type": "text", "text": (
            f"Image A: camera {crop_a.camera_id} at t={crop_a.t_sec:.1f}s\n"
            f"Image B: camera {crop_b.camera_id} at t={crop_b.t_sec:.1f}s\n"
            "Are these the same person?"
        )},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{crop_a.jpg_base64}"},
        },
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{crop_b.jpg_base64}"},
        },
    ]

    llm = _make_llm(model, temperature)
    resp = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=user_content),
    ])
    raw = resp.content if isinstance(resp.content, str) else str(resp.content)

    try:
        data = json.loads(raw)
        return MatchVerification(
            is_match=bool(data.get("is_match", False)),
            confidence=float(data.get("confidence", 0.0)),
            reasoning=str(data.get("reasoning", "")),
        )
    except (json.JSONDecodeError, KeyError, TypeError):
        logger.warning("LLM returned invalid JSON; treating as no match: %s", raw[:200])
        return MatchVerification(is_match=False, confidence=0.0, reasoning=f"parse error: {raw[:100]}")
