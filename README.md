# Capstone: Surveillance Video Query System

Capstone is a surveillance-video understanding and natural-language query system.
It combines a video processing pipeline, a retrieval database builder, a
LangGraph-based query agent, a FastAPI service, and a React web UI.

The public repository contains source code, configuration, lightweight fixtures,
and developer documentation. It intentionally excludes reports, submitted
coursework documents, demo videos, uploaded videos, runtime databases, model
weights, API keys, and generated experiment outputs.

## Features

- Single-camera and multi-camera video ingestion.
- YOLO-based detection and tracking with configurable tracker settings.
- Optional appearance, scene, and cross-camera refinement with LLM/VLM models.
- SQLite, Chroma, BM25, and hybrid retrieval paths for event search.
- LangGraph orchestration for query classification, retrieval, verification,
  and answer generation.
- FastAPI backend for uploads, database selection, ingestion, evaluation, and
  streaming queries.
- React/Vite frontend for video upload and interactive querying.

## Repository Layout

```text
agent/             LangGraph agent, retrieval tools, database builders
video/             Video ingestion, tracking, refinement, indexing modules
mevid_fastapi/     FastAPI service and API models
frontend/          React/Vite web application
config/            Tracker and multi-camera configuration
scripts/           Utility scripts for seeds, evaluation, and processing
tests/             Smoke tests, evaluation harnesses, and fixtures
skill/             Local project skill metadata
```

Additional guides:

- `README_CLIENT.md`: client-facing setup and usage notes.
- `deploy.md`: deployment-oriented notes.
- `mevid_fastapi/API_REFERENCE.md`: API endpoint reference.
- `pipeline_flowchart.md`: pipeline architecture notes.

## Requirements

- Python 3.10 or 3.11.
- Node.js 18+.
- A valid API key for the hosted model provider you use, commonly DashScope or
  an OpenAI-compatible endpoint.
- GPU with CUDA is recommended for video processing. Smaller smoke tests can run
  without a GPU, but full video processing will be slow on CPU.
- Local datasets, uploaded videos, runtime databases, and model weights are not
  committed to this repository.

## Quick Start

Clone the repository and enter it:

```bash
git clone https://github.com/jedichan/capstone.git
cd capstone
```

Create a Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install the frontend dependencies:

```bash
npm --prefix frontend install
```

Create local environment configuration:

```bash
cp .env.example .env
```

Edit `.env` with real API keys. Do not commit `.env`.

Start the backend:

```bash
python -m uvicorn main:app --app-dir mevid_fastapi --host 127.0.0.1 --port 8765
```

Start the frontend in a second terminal:

```bash
npm --prefix frontend run dev -- --host 127.0.0.1
```

Open:

```text
http://127.0.0.1:5173/
```

You can also use:

```bash
./quick_start.sh
```

The script starts the API and UI together and stops both when interrupted.

## Make Targets

Common commands are collected in `Makefile`:

```bash
make install-backend
make install-frontend
make dev-api
make dev-ui
make dev
make test-smoke
```

`make dev` starts both backend and frontend in one terminal. For daily work, two
separate terminals with `make dev-api` and `make dev-ui` are often easier to
debug.

## Environment

Use `.env.example` as the source of truth for configurable runtime values. Common
variables include:

```text
DASHSCOPE_API_KEY
DASHSCOPE_URL
OPENAI_API_KEY
OPENAI_BASE_URL
AGENT_CHAT_MODEL
AGENT_EMBEDDING_PROVIDER
FASTAPI_MAX_UPLOAD_MB
HOST
PORT
```

The backend loads `.env` from the repository root. The frontend proxies `/api`
and `/healthz` to `http://localhost:8765` during development.

## Runtime Data

Runtime data is generated locally and ignored by Git:

```text
mevid_fastapi/data/
_data/
_model/
_models/
outputs/
results/
logs/
```

Use the ingestion endpoints or scripts to build local runtime databases from
videos or seed files. Because the repository is public, real videos, reports,
runtime databases, and model artifacts should stay outside Git.

## API

Health check:

```bash
curl http://127.0.0.1:8765/healthz
```

API documentation:

```text
http://127.0.0.1:8765/docs
```

The service includes endpoints for:

- `POST /api/v1/ingest/seeds`
- `POST /api/v1/ingest/single`
- `POST /api/v1/ingest/videos`
- `GET /api/v1/databases`
- query and evaluation routes documented in `mevid_fastapi/API_REFERENCE.md`

## Development Checks

Run a lightweight Python syntax check:

```bash
python -m compileall -q agent video mevid_fastapi scripts tests
```

Run the focused smoke target:

```bash
make test-smoke
```

Run the frontend build after installing Node dependencies:

```bash
npm --prefix frontend run build
```

Many evaluation scripts require model API keys, local videos, local runtime
databases, or GPU-capable dependencies. They are not expected to pass in a fresh
clone without those assets.

## License

This project is released under the MIT License. See `LICENSE`.
