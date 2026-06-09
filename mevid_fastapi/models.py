"""Pydantic models for mevid_fastapi."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ── Ingest ────────────────────────────────────────────────────────────────────

class IngestFromSeedsRequest(BaseModel):
    seeds_dir: str = Field(..., description="Absolute path to a directory of pre-generated events_vector_flat files")
    namespace: str = Field("mevid", description="Chroma namespace prefix")
    reset_existing: bool = Field(True, description="Whether to rebuild an existing database")


class IngestFromSingleVideoRequest(BaseModel):
    video_path: str = Field(..., description="Absolute path to the video file")
    namespace: str = Field("mevid", description="Chroma namespace prefix")
    camera_id: str | None = Field(None, description="Camera ID; inferred from filename if omitted")
    conf: float = Field(0.35, description="YOLO detection confidence")
    iou: float = Field(0.35, description="YOLO tracking IoU")
    target_classes: list[str] | None = Field(
        None,
        description="YOLO target classes, e.g. ['person', 'car']; None uses pipeline defaults",
    )
    clip_refine: bool = Field(False, description="Whether to run LLM clip semantic refinement")
    clip_refine_model: str = Field("qwen3-vl-flash", description="Model used for clip refinement")
    appearance_refine: bool = Field(False, description="Whether to run VLM appearance refinement")
    scene_profile: bool = Field(False, description="Whether to generate a scene profile")
    reset_existing: bool = Field(True)


class IngestFromVideosRequest(BaseModel):
    camera_videos: dict[str, str] = Field(
        ...,
        description='Camera ID -> absolute video path, e.g. {"G329": "/data/G329.avi", ...}',
    )
    namespace: str = Field("mevid", description="Chroma namespace prefix")
    conf: float = Field(0.40, description="YOLO detection confidence")
    iou: float = Field(0.40, description="YOLO tracking IoU")
    target_classes: list[str] | None = Field(
        None,
        description="YOLO target classes, e.g. ['person', 'car']; None uses pipeline defaults",
    )
    reid_device: str = Field("cpu", description="Re-ID device (cpu/cuda)")
    refine: bool = Field(False, description="Whether to run LLM text refinement (refine_multi_camera_output)")
    refine_model: str = Field("qwen3-vl-flash", description="LLM refinement model name")
    appearance_refine: bool = Field(False, description="Whether to run VLM crop appearance refinement (object_color / appearance_notes)")
    campus_topology: bool = Field(
        False,
        description=(
            "Enable the MEVID campus warm-start topology prior (school G299/G328/G330/G419/G420 <-> bus G506/G508). "
            "Uses a pre-fitted GMM that rewards plausible transitions and penalizes too-fast/too-slow ones. "
            "Only applicable to MEVID campus videos."
        ),
    )
    reset_existing: bool = Field(True)


class IngestResult(BaseModel):
    status: str
    namespace: str
    sqlite_path: str
    chroma_path: str
    seed_file_count: int
    sqlite_rows: int
    chroma_child_records: int
    chroma_ge_records: int
    elapsed_ms: float


# ── Database ──────────────────────────────────────────────────────────────────

class DatabaseInfo(BaseModel):
    namespace: str
    sqlite_path: str
    chroma_path: str
    selected: bool = False


class SelectDatabaseRequest(BaseModel):
    namespace: str
    sqlite_path: str
    chroma_path: str


# ── Agent ─────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    thread_id: str | None = None
    user_id: str = "mevid_user"
    include_rows: bool = False
    include_node_trace: bool = True
    top_k_rows: int = 8
    response_style: str = "concise"  # "concise" (eval-compatible) | "natural" (conversational)


class QueryResponse(BaseModel):
    query: str
    answer: str
    answer_type: str = ""
    node_trace: list[str] = []
    elapsed_ms: float = 0.0
    verifier_result: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []


# ── Eval ──────────────────────────────────────────────────────────────────────

class EvalRunRequest(BaseModel):
    questions_file: str | None = Field(
        None,
        description="Path to a sampled JSON file; falls back to the built-in sampled_10.json when None",
    )
    video_dir: str | None = Field(
        None,
        description="Video directory (used for timestamp parsing in IoU computation); may be empty",
    )


class EvalCaseResult(BaseModel):
    case_id: str
    category: str
    question: str
    expected: str
    predicted: str
    correct: bool
    top_hit: bool
    iou: float
    elapsed_ms: float


class EvalSummary(BaseModel):
    total: int
    correct: int
    accuracy: float
    top_hit_total: int
    top_hit_rate: float
    mean_iou: float
    iou_threshold: float = 0.15
    n_iou_pass: int = 0
    iou_pass_rate: float = 0.0
    individual_iou: list[float] = []
    by_category: dict[str, dict[str, Any]]
    cases: list[EvalCaseResult]
    elapsed_total_ms: float


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    database_selected: bool
    sqlite_path: str
    chroma_path: str
    namespace: str
