"""MEVID FastAPI — video pipeline + agent pipeline unified service."""
from __future__ import annotations

import shutil
import sqlite3
import sys
import time
import uuid
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_AGENT_ROOT = _PROJECT_ROOT / "agent"
for _p in (str(_PROJECT_ROOT), str(_AGENT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models import (  # noqa: E402
    EvalRunRequest,
    EvalSummary,
    EvalCaseResult,
    HealthResponse,
    IngestFromSeedsRequest,
    IngestFromSingleVideoRequest,
    IngestFromVideosRequest,
    IngestResult,
    QueryRequest,
    QueryResponse,
    SelectDatabaseRequest,
)
from service import agent_service  # noqa: E402
import pipeline_service  # noqa: E402
import eval_service  # noqa: E402
import eval_mcq_service  # noqa: E402

_DATA_DIR = Path(__file__).resolve().parent / "data"
_UPLOAD_DIR = _DATA_DIR / "uploads"
_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

import os as _os
_CLIENT_FIXED_NAMESPACE: str | None = _os.environ.get("CLIENT_FIXED_NAMESPACE") or None

# In-memory pipeline task status  {task_id: {"status": ..., "result": ...}}
_pipeline_tasks: dict[str, dict] = {}

app = FastAPI(
    title="Surveillance Video Query API",
    version="1.0.0",
    description="Multi-camera video pipeline + LangGraph agent query service.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    db = agent_service.db_info
    return HealthResponse(
        status="ok",
        database_selected=agent_service.database_selected,
        sqlite_path=db.get("sqlite_path", ""),
        chroma_path=db.get("chroma_path", ""),
        namespace=db.get("namespace", ""),
    )


# ── Ingest ────────────────────────────────────────────────────────────────────

@app.post("/api/v1/ingest/seeds", response_model=IngestResult)
def ingest_seeds(payload: IngestFromSeedsRequest) -> IngestResult:
    """Build SQLite + Chroma from pre-generated events_vector_flat seed files."""
    output_base = str(_DATA_DIR / "runtime" / payload.namespace)
    try:
        result = pipeline_service.ingest_from_seeds(
            seeds_dir=payload.seeds_dir,
            namespace=payload.namespace,
            output_base=output_base,
            reset_existing=payload.reset_existing,
        )
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # Auto-select the newly built database
    agent_service.select_database(
        sqlite_path=result["sqlite_path"],
        chroma_path=result["chroma_path"],
        namespace=result["namespace"],
    )
    return IngestResult(**result)


@app.post("/api/v1/ingest/single")
def ingest_single(payload: IngestFromSingleVideoRequest) -> dict:
    """Run single-camera pipeline (YOLO+track, no cross-camera Re-ID) and build databases."""
    output_base = str(_DATA_DIR / "runtime" / payload.namespace)
    try:
        result = pipeline_service.ingest_from_single_video(
            video_path=payload.video_path,
            namespace=payload.namespace,
            output_base=output_base,
            camera_id=payload.camera_id,
            conf=payload.conf,
            iou=payload.iou,
            target_classes=payload.target_classes,
            clip_refine=payload.clip_refine,
            appearance_refine=payload.appearance_refine,
            scene_profile=payload.scene_profile,
            reset_existing=payload.reset_existing,
        )
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    agent_service.select_database(
        sqlite_path=result["sqlite_path"],
        chroma_path=result["chroma_path"],
        namespace=result["namespace"],
    )
    return result


@app.post("/api/v1/ingest/videos", response_model=IngestResult)
def ingest_videos(payload: IngestFromVideosRequest) -> IngestResult:
    """Run full multi-camera pipeline (YOLO+Re-ID+matching) and build databases."""
    output_base = str(_DATA_DIR / "runtime" / payload.namespace)
    try:
        result = pipeline_service.ingest_from_videos(
            camera_videos=payload.camera_videos,
            namespace=payload.namespace,
            output_base=output_base,
            conf=payload.conf,
            iou=payload.iou,
            target_classes=payload.target_classes,
            reid_device=payload.reid_device,
            refine=payload.refine,
            refine_model=payload.refine_model,
            appearance_refine=payload.appearance_refine,
            campus_topology=payload.campus_topology,
            reset_existing=payload.reset_existing,
        )
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    agent_service.select_database(
        sqlite_path=result["sqlite_path"],
        chroma_path=result["chroma_path"],
        namespace=result["namespace"],
    )
    return IngestResult(**result)


# ── Database ──────────────────────────────────────────────────────────────────

@app.get("/api/v1/databases")
def list_databases() -> dict:
    """List all available namespaces that have a built SQLite database."""
    runtime_dir = _DATA_DIR / "runtime"
    databases: list[dict] = []
    if runtime_dir.exists():
        for ns_dir in sorted(runtime_dir.iterdir()):
            if not ns_dir.is_dir():
                continue
            sqlite_p = ns_dir / "runtime" / "episodic_events.sqlite"
            chroma_p = ns_dir / "runtime" / "chroma"
            if not sqlite_p.exists():
                continue
            try:
                conn = sqlite3.connect(str(sqlite_p), timeout=3)
                n = conn.execute("SELECT COUNT(*) FROM episodic_events").fetchone()[0]
                cams = [r[0] for r in conn.execute(
                    "SELECT DISTINCT camera_id FROM episodic_events "
                    "WHERE camera_id IS NOT NULL LIMIT 6"
                ).fetchall()]
                conn.close()
                databases.append({
                    "namespace": ns_dir.name,
                    "sqlite_path": str(sqlite_p),
                    "chroma_path": str(chroma_p),
                    "row_count": n,
                    "size_kb": sqlite_p.stat().st_size // 1024,
                    "cameras": cams,
                    "has_chroma": chroma_p.exists(),
                })
            except Exception:
                pass
    current_ns = agent_service.db_info.get("namespace", "") if agent_service.database_selected else ""
    return {"databases": databases, "selected": current_ns}


@app.post("/api/v1/database/select")
def select_database(payload: SelectDatabaseRequest) -> dict:
    """Manually point the agent at an existing database."""
    if _CLIENT_FIXED_NAMESPACE and payload.namespace != _CLIENT_FIXED_NAMESPACE:
        raise HTTPException(
            status_code=400,
            detail=f"Only namespace '{_CLIENT_FIXED_NAMESPACE}' is enabled in client mode.",
        )
    try:
        agent_service.select_database(
            sqlite_path=payload.sqlite_path,
            chroma_path=payload.chroma_path,
            namespace=payload.namespace,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "ok", "namespace": payload.namespace}


@app.post("/api/v1/database/reset")
def reset_database(payload: dict) -> dict:
    """Drop SQLite file and Chroma collections for a namespace.

    Body: {"namespace": "mevid"} — paths are inferred automatically from the
    default runtime layout (data/runtime/<namespace>/...).
    """
    from agent.db.config import CHROMA_CHILD_SUFFIX, CHROMA_EVENT_SUFFIX, CHROMA_PARENT_SUFFIX

    namespace = str(payload.get("namespace") or "").strip()
    if not namespace:
        raise HTTPException(status_code=400, detail="namespace is required")

    # Infer paths from the standard runtime layout
    # Actual structure: data/runtime/<namespace>/runtime/episodic_events.sqlite
    runtime_dir = _DATA_DIR / "runtime" / namespace / "runtime"
    sqlite_p = runtime_dir / "episodic_events.sqlite"
    chroma_p = runtime_dir / "chroma"

    errors: list[str] = []

    # SQLite — delete file
    if sqlite_p.exists():
        try:
            sqlite_p.unlink()
        except Exception as exc:
            errors.append(f"sqlite: {exc}")

    # Chroma — delete collections via client
    if chroma_p.exists():
        try:
            import chromadb
            client = chromadb.PersistentClient(path=str(chroma_p))
            for suffix in (CHROMA_CHILD_SUFFIX, CHROMA_PARENT_SUFFIX, CHROMA_EVENT_SUFFIX):
                col_name = f"{namespace}_{suffix}"
                try:
                    client.delete_collection(col_name)
                except Exception:
                    pass
        except Exception as exc:
            errors.append(f"chroma: {exc}")

    # Deselect if it was the active DB
    if agent_service.database_selected and agent_service.db_info.get("namespace") == namespace:
        agent_service._db = {}

    if errors:
        raise HTTPException(status_code=500, detail="; ".join(errors))
    return {"status": "ok", "namespace": namespace, "message": f"Database '{namespace}' cleared"}


# ── Agent ─────────────────────────────────────────────────────────────────────

@app.post("/api/v1/agent/query", response_model=QueryResponse)
def agent_query(payload: QueryRequest) -> QueryResponse:
    """Run a single natural-language query through the LangGraph agent."""
    try:
        result = agent_service.run_query(
            query=payload.query,
            thread_id=payload.thread_id,
            user_id=payload.user_id,
            include_rows=payload.include_rows,
            include_node_trace=payload.include_node_trace,
            top_k_rows=payload.top_k_rows,
            response_style=payload.response_style,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return QueryResponse(**result)


# ── File Upload & Video Streaming ────────────────────────────────────────────

@app.post("/api/v1/upload/videos")
async def upload_videos(files: list[UploadFile] = File(...)) -> dict:
    """Upload one or more video files; returns their saved paths."""
    saved: list[dict] = []
    for f in files:
        dest = _UPLOAD_DIR / f.filename
        with dest.open("wb") as out:
            shutil.copyfileobj(f.file, out)
        saved.append({"filename": f.filename, "path": str(dest)})
    return {"status": "ok", "files": saved}


@app.get("/api/v1/video/{file_path:path}")
def stream_video(file_path: str) -> FileResponse:
    """Stream / serve a video file by absolute or relative path."""
    p = Path(file_path)
    if not p.is_absolute():
        p = _DATA_DIR / file_path
    if not p.exists() and not Path(file_path).is_absolute():
        upload_candidate = _UPLOAD_DIR / file_path
        if upload_candidate.exists():
            p = upload_candidate
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"Video not found: {file_path}")
    return FileResponse(str(p), media_type="video/mp4")


# ── Async Pipeline Task ───────────────────────────────────────────────────────

@app.post("/api/v1/pipeline/start")
def pipeline_start(payload: IngestFromVideosRequest, background_tasks: BackgroundTasks) -> dict:
    """Start the full video pipeline in the background; returns a task_id to poll."""
    task_id = str(uuid.uuid4())
    _pipeline_tasks[task_id] = {
        "status": "running", "stage": "detection",
        "result": None, "error": None, "started_at": time.time(),
    }

    def _run() -> None:
        output_base = str(_DATA_DIR / "runtime" / payload.namespace)
        try:
            _pipeline_tasks[task_id]["stage"] = "detection"
            result = pipeline_service.ingest_from_videos(
                camera_videos=payload.camera_videos,
                namespace=payload.namespace,
                output_base=output_base,
                conf=payload.conf,
                iou=payload.iou,
                target_classes=payload.target_classes,
                reid_device=payload.reid_device,
                refine=payload.refine,
                refine_model=payload.refine_model,
                appearance_refine=payload.appearance_refine,
                campus_topology=payload.campus_topology,
                reset_existing=payload.reset_existing,
            )
            _pipeline_tasks[task_id]["stage"] = "indexing"
            agent_service.select_database(
                sqlite_path=result["sqlite_path"],
                chroma_path=result["chroma_path"],
                namespace=result["namespace"],
            )
            _pipeline_tasks[task_id]["status"] = "done"
            _pipeline_tasks[task_id]["stage"] = "done"
            _pipeline_tasks[task_id]["result"] = result
        except Exception as exc:  # noqa: BLE001
            _pipeline_tasks[task_id]["status"] = "error"
            _pipeline_tasks[task_id]["stage"] = "error"
            _pipeline_tasks[task_id]["error"] = str(exc)

    background_tasks.add_task(_run)
    return {"task_id": task_id, "status": "running"}


@app.post("/api/v1/pipeline/start/single")
def pipeline_start_single(payload: IngestFromSingleVideoRequest, background_tasks: BackgroundTasks) -> dict:
    """Start single-camera pipeline in the background; returns task_id to poll."""
    task_id = str(uuid.uuid4())
    _pipeline_tasks[task_id] = {
        "status": "running", "stage": "detection",
        "result": None, "error": None, "started_at": time.time(),
    }

    def _run() -> None:
        output_base = str(_DATA_DIR / "runtime" / payload.namespace)
        try:
            _pipeline_tasks[task_id]["stage"] = "detection"
            result = pipeline_service.ingest_from_single_video(
                video_path=payload.video_path,
                namespace=payload.namespace,
                output_base=output_base,
                camera_id=payload.camera_id,
                conf=payload.conf,
                iou=payload.iou,
                target_classes=payload.target_classes,
                clip_refine=payload.clip_refine,
                clip_refine_model=payload.clip_refine_model,
                appearance_refine=payload.appearance_refine,
                scene_profile=payload.scene_profile,
                reset_existing=payload.reset_existing,
            )
            _pipeline_tasks[task_id]["stage"] = "indexing"
            agent_service.select_database(
                sqlite_path=result["sqlite_path"],
                chroma_path=result["chroma_path"],
                namespace=result["namespace"],
            )
            _pipeline_tasks[task_id]["status"] = "done"
            _pipeline_tasks[task_id]["stage"] = "done"
            _pipeline_tasks[task_id]["result"] = result
        except Exception as exc:  # noqa: BLE001
            _pipeline_tasks[task_id]["status"] = "error"
            _pipeline_tasks[task_id]["stage"] = "error"
            _pipeline_tasks[task_id]["error"] = str(exc)

    background_tasks.add_task(_run)
    return {"task_id": task_id, "status": "running"}


@app.get("/api/v1/pipeline/status/{task_id}")
def pipeline_status(task_id: str) -> dict:
    """Poll the status of an async pipeline task."""
    task = _pipeline_tasks.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return {"task_id": task_id, **task}


# ── Eval ──────────────────────────────────────────────────────────────────────

@app.post("/api/v1/eval/run", response_model=EvalSummary)
def run_eval(payload: EvalRunRequest) -> EvalSummary:
    """Run evaluation suite through the agent and return accuracy / IoU metrics."""
    if not agent_service.database_selected:
        raise HTTPException(
            status_code=400,
            detail="Select a database first via /api/v1/ingest/seeds or /api/v1/database/select",
        )

    def _query_fn(question: str) -> dict:
        return agent_service.run_query(
            query=question,
            thread_id=None,
            user_id="eval_runner",
            include_rows=True,
            include_node_trace=False,
            top_k_rows=5,
        )

    try:
        summary = eval_service.run_eval(
            questions_file=payload.questions_file,
            agent_query_fn=_query_fn,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    cases = [EvalCaseResult(**{k: v for k, v in c.items() if k != "answer_raw"}) for c in summary["cases"]]
    return EvalSummary(**{**summary, "cases": cases})


@app.post("/api/v1/eval/mcq")
def run_mcq_eval(payload: EvalRunRequest) -> dict:
    """Run ForeSea-style multiple-choice QA evaluation on MEVID.

    Questions cover 6 subtasks: Search, Temporal, Event, Activity, Counting, Anomaly.
    Metrics: Accuracy (A/B/C/D) + temporal IoU from retrieved rows.
    Body: {"questions_file": null}  — null uses the built-in mcq_questions.json.
    """
    if not agent_service.database_selected:
        raise HTTPException(
            status_code=400,
            detail="Select a database first via /api/v1/ingest/seeds or /api/v1/database/select",
        )

    def _query_fn(question: str) -> dict:
        return agent_service.run_query(
            query=question,
            thread_id=None,
            user_id="mcq_eval_runner",
            include_rows=True,
            include_node_trace=False,
            top_k_rows=5,
        )

    try:
        summary = eval_mcq_service.run_mcq_eval(
            questions_file=payload.questions_file,
            agent_query_fn=_query_fn,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return summary
