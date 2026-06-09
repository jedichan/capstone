"""Data structures for multi-camera cross-view person tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class PersonCrop:
    """Single person crop image."""

    t_sec: float
    camera_id: str
    track_id: int
    image_array: np.ndarray
    jpg_base64: str


@dataclass
class CameraAppearance:
    """One appearance of a global entity on a camera."""

    camera_id: str
    track_id: int
    start_time: float
    end_time: float
    confidence: float


@dataclass
class GlobalEntity:
    """Merged global entity across cameras (same person)."""

    global_entity_id: str
    appearances: list[CameraAppearance] = field(default_factory=list)


@dataclass
class CameraResult:
    """Full Stage-1 result for one camera."""

    camera_id: str
    video_path: str
    tracks: list[dict[str, Any]]
    events: list[dict[str, Any]]
    clips: list[dict[str, float]]
    meta: dict[str, Any] = field(default_factory=dict)
    person_crops: dict[int, list[PersonCrop]] = field(default_factory=dict)
    person_embeddings: dict[int, np.ndarray] = field(default_factory=dict)


@dataclass
class CrossCameraConfig:
    """Hyperparameters for cross-camera matching."""

    max_transition_sec: float = 30.0
    min_overlap_sec: float = 0.0
    embedding_threshold: float = 0.65
    llm_verify_top_k: int = 3
    person_only: bool = True
    # Same-camera fragment stitching
    same_camera_max_gap_sec: float = 3.0
    same_camera_reid_threshold: float = 0.80
    # Cross-camera combined score threshold
    cross_camera_min_score: float = 0.65
    # Trigger VLM only for borderline cosine
    llm_verify_cosine_min: float = 0.65
    llm_verify_cosine_max: float = 0.80
    # Cross-camera simultaneous-presence tolerance.
    # Pairs whose time windows overlap by MORE than this many seconds are
    # hard-rejected as different individuals (a person cannot be in two cameras
    # at once).  A small positive value (default 5 s) accommodates tracker
    # boundary jitter when a person lingers at the edge of two camera FOVs.
    cross_camera_overlap_tolerance_sec: float = 5.0
    # Camera topology prior weights.
    # When a CameraTopologyPrior is injected the combined score becomes:
    #   topology_weight_reid  * cosine
    # + topology_weight_topo  * topology_score(cam_a, cam_b, delta_t)
    # (must sum to ≤ 1.0; remainder is ignored)
    topology_weight_reid: float = 0.55
    topology_weight_topo: float = 0.45
    # Optional precision caps. 0 disables the cap. These are useful for
    # evaluation runs where false cross-person merges are more damaging than
    # missed long-range matches.
    max_global_entity_cameras: int = 0
    max_global_entity_tracks: int = 0


@dataclass
class MatchVerification:
    """Optional LLM verification result."""

    is_match: bool
    confidence: float
    reasoning: str


@dataclass
class MultiCameraOutput:
    """Final multi-camera pipeline output."""

    cameras: dict[str, str]
    config: CrossCameraConfig
    global_entities: list[GlobalEntity]
    per_camera: list[CameraResult]
    merged_events: list[dict[str, Any]] = field(default_factory=list)
