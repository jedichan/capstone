"""Pydantic output schema for the LangChain refinement stage (decoupled from the YOLO pipeline)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator


class RefinedEntity(BaseModel):
    """
    Entity layer: the LLM merges fragmented track_ids into closer-to-real entities (e.g. five cars).
    entity_id is a new global entity id (usable within a single camera view).
    """

    entity_id: str = Field(description="Global entity id, e.g. car_1 / person_2")
    class_name: str = Field(description="Class label, e.g. car / person")
    local_track_ids: list[int] = Field(default_factory=list, description="Local track_ids merged into this entity")
    appearance: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Appearance cues to avoid bad merges. Suggested keys:\n"
            "- color: coarse color in English (e.g. white/black/silver_gray/red/blue/dark/unknown).\n"
            "- color_confidence: 0~1\n"
            "- vehicle_type: sedan/SUV/truck/van/unknown (cars only); English values.\n"
            "- distinctive_marks: salient traits (roof rack, stickers, tint, clothing color, etc.); English.\n"
        ),
    )
    location: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Location + scene semantics; include sub-region and motion relative to scene, e.g.:\n"
            "- scene_zone: entrance|exit|road|road_right|road_left|road_center|parking|parking_slot|sidewalk|far_background\n"
            "- region_text: short English phrase, e.g. inner side of parking entrance, right lane mid-frame\n"
            "- movement_in_scene: one English sentence for how it moves and where\n"
            "- representative_bbox_xyxy / representative_time_sec\n"
        ),
    )
    notes: str = Field(default="", description="Merge rationale, appearance, location, timing, etc. (English).")


class RefinedEvent(BaseModel):
    event_id: str = Field(description="Unique event id (may be track_id + time range)")
    event_type: str = Field(description="Event type, e.g. driving_in / parking / walking / interaction")
    class_name: str = Field(description="Subject class, e.g. car / person")
    track_id: int | None = Field(default=None, description="Local track_id from tracker if known")
    entity_id: str | None = Field(default=None, description="Corrected global entity id (for merging fragments)")
    start_time: float = Field(description="Start time in seconds")
    end_time: float = Field(description="End time in seconds")
    confidence: float = Field(ge=0.0, le=1.0, description="Model confidence the event is real (0~1)")
    details: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Action details; must include movement_scene_narrative: one English sentence for motion + "
            "where in the scene (entrance/exit/mid-road right/parking slot, etc.); aligns with location."
        ),
    )
    evidence: dict[str, Any] = Field(
        default_factory=dict,
        description="Evidence: which frame timestamps, visibility, why temporal bounds were adjusted, etc.",
    )
    location: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Per-event location + scene (aligned with bbox). Suggested keys:\n"
            "- scene_zone: same enum as above\n"
            "- start_scene / end_scene: English sub-region at start/end\n"
            "- movement_scene: one English sentence summarizing motion + scene in this event\n"
            "- start_bbox_xyxy/end_bbox_xyxy (reuse from raw if present)\n"
        ),
    )


class RefinedEventsPayload(BaseModel):
    video_path: str
    analyzed_clip: dict[str, float]  # {"start_sec":..., "end_sec":...}
    scene_context: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Whole-clip scene understanding (global first, then events). Suggested keys:\n"
            "- overview: 1~3 English sentences (where lot/road/building/entrances roughly are)\n"
            "- layout: English: left/center/right, far/mid/near — what is road/slots/sidewalk, etc.\n"
            "- entrance_exit_guess: English guess where vehicle/pedestrian entrances, exits, main paths are\n"
            "- landmarks: optional list, e.g. street light, crosswalk, gate, trees\n"
        ),
    )
    entities: list[RefinedEntity] = Field(default_factory=list, description="Global entity list (deduped vehicles/people)")
    refined_events: list[RefinedEvent]
    temporal_policy: dict[str, Any] = Field(
        default_factory=dict,
        description="Temporal policy: how strictly raw timing is kept, max adjustment allowed, etc.",
    )
    location_policy: dict[str, Any] = Field(
        default_factory=dict,
        description="Location policy: rules used to merge visually similar targets (IoU, center distance, same region)",
    )


class RefinedAllClipsPayload(BaseModel):
    """Per-segment results for every clip in clips.json."""

    video_path: str
    clips: list[RefinedEventsPayload]


class VectorEvent(BaseModel):
    video_id: str | None = None
    clip_start_sec: float | None = None
    clip_end_sec: float | None = None
    track_id: int | None = None
    start_time: float
    end_time: float
    object_type: str
    object_color: str
    appearance_notes: str
    scene_zone: str
    event_text: str
    keywords: list[str]
    start_bbox_xyxy: list[float] | None = None
    end_bbox_xyxy: list[float] | None = None
    entity_hint: str | None = None


class VectorEventsPayload(BaseModel):
    video_id: str
    clip_start_sec: float
    clip_end_sec: float
    events: list[VectorEvent]

    @model_validator(mode="before")
    @classmethod
    def drop_incomplete_events(cls, data: Any) -> Any:
        """Drop malformed trailing event stubs emitted by some LLM completions.

        The model sometimes appends an object like {"video_id": "..."} inside the
        events array. Keeping valid events is more useful than failing the whole
        clip, while required semantic fields still protect real event records.
        """
        if not isinstance(data, dict):
            return data
        events = data.get("events")
        if not isinstance(events, list):
            return data
        required = {
            "start_time",
            "end_time",
            "object_type",
            "object_color",
            "appearance_notes",
            "scene_zone",
            "event_text",
            "keywords",
        }
        clean_events = [
            ev for ev in events
            if isinstance(ev, dict) and required.issubset(ev.keys())
        ]
        return {**data, "events": clean_events}


class VectorAllClipsPayload(BaseModel):
    video_id: str
    clips: list[VectorEventsPayload]
