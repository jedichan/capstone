# Surveillance Video Query System - Client Release

This package contains the client-ready source code for the surveillance video query system. It provides:

- A video preprocessing pipeline for single-camera and multi-camera surveillance videos.
- Structured event generation with camera IDs, timestamps, track IDs, appearance notes, and optional cross-camera global entity IDs.
- A FastAPI backend for video ingestion, database selection, and natural-language querying.
- A React frontend for uploading videos, running the pipeline, selecting databases, and asking questions.
- A LangGraph-based retrieval agent using SQL, hybrid vector/BM25 search, camera-aware filtering, and evidence verification.

This release intentionally does **not** include private API keys, full video datasets, large model weights, cached pipeline outputs, experiment results, or local annotation spreadsheets.

For convenience, this package **does** bundle two ready-to-use demo runtime databases (`mevid` and `mevid_obs_1420`) under `mevid_fastapi/data/runtime/`, so you can query immediately after starting the backend without rebuilding an index. All other runtime databases and uploaded videos are excluded.

## 1. Package Contents

Important directories:

```text
agent/             LangGraph agent, retrieval tools, SQLite/Chroma builders
video/             Video pipeline, detection/tracking, refinement, Re-ID logic
mevid_fastapi/     FastAPI backend service
frontend/          React/Vite frontend
config/            Tracker and model configuration
scripts/           Utility scripts that are safe for source release
tests/             Lightweight tests and reusable evaluation harnesses
```

Important files:

```text
.env.example       Environment variable template with placeholder values
README_CLIENT.md   This deployment and usage guide
README.md          Developer-oriented project notes
deploy.md          Additional local deployment notes
```

Excluded from the release package:

```text
.env
_data/
_cache/
results/
outputs/
logs/
mevid_fastapi/data/uploads/                      (raw uploaded videos)
mevid_fastapi/data/runtime/ except mevid + mevid_obs_1420
agent/test/generated/
node_modules/
*.pt, *.pth, *.onnx
local annotation spreadsheets
```

## 2. System Requirements

Recommended environment:

- Windows 10/11 or Linux.
- Python 3.10 or 3.11.
- Conda environment recommended.
- Node.js 18+ for the frontend.
- CUDA-capable GPU recommended for video processing, but CPU mode can run smaller demos.
- Valid DashScope API key for Qwen models.

The original development environment used a Conda environment named `capstone`.

## 3. Environment Setup

Create or activate your Python environment:

```powershell
conda activate capstone
```

If starting from a fresh environment, install the required project dependencies according to your deployment constraints. The project uses the following major libraries:

- `fastapi`
- `uvicorn`
- `pydantic`
- `openai`
- `langchain`
- `langchain-openai`
- `chromadb`
- `numpy`
- `opencv-python`
- `ultralytics`
- `torch`
- `torchreid`

Frontend setup:

```powershell
cd frontend
npm install
```

## 4. Configure API Keys

Copy the environment template:

```powershell
copy .env.example .env
```

Then edit `.env` and fill in real values:

```env
DASHSCOPE_API_KEY=your_real_dashscope_key
AGENT_CHAT_MODEL=qwen3.6-plus
AGENT_MCQ_JUDGE_MODEL=qwen3.6-plus
VIDEO_VLM_MODEL=qwen3.6-plus
VIDEO_VLM_TIMEOUT_SEC=600
```

Do not commit `.env`.

## 5. Start the Backend

From the repository root:

```powershell
python -m uvicorn main:app --app-dir mevid_fastapi --host 127.0.0.1 --port 8765
```

Health check:

```powershell
curl http://127.0.0.1:8765/healthz
```

Expected response contains:

```json
{
  "status": "ok"
}
```

API documentation is available at:

```text
http://127.0.0.1:8765/docs
```

## 6. Start the Frontend

Open a second terminal:

```powershell
cd frontend
npm run dev -- --host 127.0.0.1
```

Open:

```text
http://127.0.0.1:5173/
```

The frontend proxies API requests to the backend on port `8765`.

## 7. Build a Database from Existing Seed Files

If you already have generated `*_events_vector_flat.json` seed files, use the seed ingestion endpoint.

Example request body:

```json
{
  "seeds_dir": "D:/path/to/events_vector_flat",
  "namespace": "client_demo",
  "reset_existing": true
}
```

Endpoint:

```text
POST /api/v1/ingest/seeds
```

After ingestion, the backend automatically selects the new database for querying.

Runtime databases are built under:

```text
mevid_fastapi/data/runtime/<namespace>/runtime/
```

This runtime directory is generated locally and is not included in the source release.

## 8. Run Single-Camera Video Ingestion

Use this when processing one uploaded or local video without cross-camera identity matching.

Endpoint:

```text
POST /api/v1/ingest/single
```

Example body:

```json
{
  "video_path": "D:/path/to/video.mp4",
  "namespace": "single_camera_demo",
  "camera_id": "G001",
  "conf": 0.35,
  "iou": 0.35,
  "target_classes": ["person", "car"],
  "clip_refine": true,
  "appearance_refine": true,
  "scene_profile": true,
  "reset_existing": true
}
```

For asynchronous frontend usage, the backend also provides:

```text
POST /api/v1/pipeline/start/single
GET  /api/v1/pipeline/status/{task_id}
```

## 9. Run Multi-Camera Video Ingestion

Use this when processing multiple synchronized or related camera streams and building cross-camera global entities.

Endpoint:

```text
POST /api/v1/ingest/videos
```

Example body:

```json
{
  "camera_videos": {
    "G328": "D:/path/to/G328.mp4",
    "G329": "D:/path/to/G329.mp4",
    "G508": "D:/path/to/G508.mp4"
  },
  "namespace": "multi_camera_demo",
  "conf": 0.35,
  "iou": 0.40,
  "target_classes": ["person", "car"],
  "reid_device": "cuda",
  "refine": true,
  "refine_model": "qwen3.6-plus",
  "reset_existing": true
}
```

For asynchronous frontend usage:

```text
POST /api/v1/pipeline/start
GET  /api/v1/pipeline/status/{task_id}
```

## 10. Query the Agent

Once a database is selected, call:

```text
POST /api/v1/agent/query
```

Example body:

```json
{
  "query": "Did a person wearing a dark coat appear in camera G508?",
  "thread_id": null,
  "user_id": "client_demo",
  "include_rows": true,
  "include_node_trace": false,
  "top_k_rows": 5,
  "response_style": "natural"
}
```

Good smoke-test questions:

```text
How many unique tracked objects are detected in camera G508?
Did a person exit from the left side of camera G508?
Is there a person wearing a dark coat in camera G328?
Did the same person appear in camera G328 and then camera G508?
What happened around 00:02:10 in camera G421?
```

Query behavior:

- Camera IDs such as `G508` are stored in the seed files and SQLite database.
- The release includes camera-aware retrieval support for evaluation and internal agent calls.
- Time and camera constraints are the strongest structured evidence for surveillance queries.
- Cross-camera answers depend on global entity construction quality.

## 11. Database Selection

This package ships two bundled demo databases you can select immediately:

```text
mevid             single-/multi-camera MEVID demo
mevid_obs_1420    observation-enriched MEVID demo
```

List available local runtime databases:

```text
GET /api/v1/databases
```

Select an existing database:

```text
POST /api/v1/database/select
```

Example body:

```json
{
  "sqlite_path": "D:/repo/mevid_fastapi/data/runtime/client_demo/runtime/episodic_events.sqlite",
  "chroma_path": "D:/repo/mevid_fastapi/data/runtime/client_demo/runtime/chroma",
  "namespace": "client_demo"
}
```

## 12. Recommended Operating Modes

Fast demo:

```text
Use existing seed files -> /api/v1/ingest/seeds -> query.
```

Single-camera upload demo:

```text
Upload one video -> /api/v1/pipeline/start/single -> poll status -> query.
```

Full multi-camera demo:

```text
Provide camera_id -> video_path mapping -> /api/v1/pipeline/start -> poll status -> query.
```

## 13. Security Notes

- Do not commit `.env`.
- Do not share API keys inside screenshots, logs, or result files.
- Large source videos and runtime databases should be transferred separately from the code package.
- If deploying publicly, place the FastAPI backend behind an authenticated reverse proxy.
- The default development CORS setting allows all origins. Restrict this before public deployment.

## 14. Troubleshooting

Backend starts but query fails:

- Confirm `.env` contains a valid `DASHSCOPE_API_KEY`.
- Confirm a database has been ingested or selected.
- Check `GET /healthz`.
- Check backend logs for model, embedding, SQLite, or Chroma errors.

Video pipeline is slow:

- VLM refinement is the slowest stage.
- Disable `clip_refine`, `appearance_refine`, or `scene_profile` for faster smoke tests.
- Use GPU for detection/Re-ID when available.
- Increase `VIDEO_VLM_TIMEOUT_SEC` for long VLM calls.

Query retrieves the wrong camera:

- Make sure the question explicitly includes the camera ID, for example `G508`.
- Make sure the seed records contain `camera_id`.
- For cross-camera questions, provide both source and target cameras in the wording.

Frontend keeps loading:

- Check `GET /api/v1/pipeline/status/{task_id}` directly.
- Confirm the backend task ended with `status: done` or `status: error`.
- Restart the frontend dev server after backend URL or proxy changes.

Chroma or SQLite database missing:

- Re-run `/api/v1/ingest/seeds` or the relevant pipeline endpoint.
- Confirm runtime files exist under `mevid_fastapi/data/runtime/<namespace>/runtime/`.

## 15. Release Branch

This client release is based on:

```text
release/client-ready-20260529
```

Use this branch as the clean deployment baseline.
