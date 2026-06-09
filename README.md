# Capstone Project Structure

```text
Capstone/
├── README.md
├── data/                         # External databases
│   ├── raw/
│   └── annotations/
├── config/
│   ├── trackers/
│   ├── models/
│   └── retrieval/
├── video/
│   ├── core/                     # Low-level model wrappers and unified data protocol
│   │   ├── models/               # YOLO / embedding / LLM / VLM wrappers
│   │   └── schema/               # Event / Track / Clip / Query / Evidence
│   │
│   ├── ingestion/                # Video / JSON input adapters
│   │   ├── video_loader.py
│   │   └── json_loader.py
│   │
│   ├── factory/                  # Offline video-understanding pipeline coordination
│   │   ├── processors/
│   │   │   ├── vision.py         # Detection, tracking, basic visual features
│   │   │   ├── captioner.py      # Optional: clip captioning / multimodal description
│   │   │   └── analyzer.py       # Event extraction, action-logic decisions
│   │   └── coordinator.py        # Video -> Event/Clip/Metadata orchestration
│   │
│   ├── indexing/                 # Knowledge building and index persistence
│   │   ├── document_builder.py   # Event doc / Summary doc construction
│   │   ├── embedder.py           # Text / Image embedding
│   │   ├── graph_builder.py      # Temporal / entity / relation graph building
│   │   └── store_manager.py      # Consistent vector / graph / metadata writes
│   │
│   └── common/                   # Shared config, logging, paths, utilities
│
├── agent/                        # LangGraph orchestration layer (must keep)
│   ├── state.py                  # Global state / context / intermediate results
│   ├── graph.py                  # Main decision graph
│   ├── nodes/                    # parse / route / retrieve / answer
│   ├── tools/                    # Search, verify, playback, summarize tool wrappers
│   │
│   ├── retrieval/                # Pure retrieval layer
│   │   ├── event_retriever.py    # Event-level vector retrieval
│   │   ├── summary_retriever.py  # Time-window / global summary retrieval
│   │   ├── graph_retriever.py    # Graph-relation retrieval
│   │   ├── metadata_filter.py    # Structured filtering
│   │   ├── multi_modal.py        # Cross-modal matching
│   │   ├── reranker.py           # Recall result reranking
│   │   └── fusion.py             # Multi-path evidence fusion
│   │
│   └── common/                   # Shared config, logging, paths, utilities
│
├── outputs/
│   ├── video_understanding/      # events.json / clips.json / tracked.mp4
│   ├── indexing/                 # docs / embeddings / intermediate index artifacts
│   ├── vector_store/
│   ├── graph_store/
│   └── cache/
├── scripts/
├── docs/
└── tests/
```

## Current Chroma Chunk Strategy

- The vector index now uses a parent-child layout.
- Child index keeps the current track-level chunking:
  - record id: `{video_id}_{entity_hint}`
  - granularity: one child record per track
  - purpose: semantic retrieval over track-level behavior
- Parent index is grouped by `video_id`:
  - record id: `{video_id}`
  - granularity: one parent record per video
  - purpose: coarse video-level recall and parent-child routing
- Child metadata stores `parent_id=video_id` so parent and child records stay linked.
- Current online retrieval still reads the child collection by default, while the parent collection is built for hierarchical retrieval expansion.

## MEVID Evaluation Flow

The MEVID multi-camera test path is split into reusable video modules plus two stable evaluation wrappers.

### Added Video Modules

- `video.factory.person_crop_sampler`
  - Samples person crops from tracked bbox events.
  - Used by appearance refinement and QA evidence sampling.
- `video.factory.appearance_refinement_runner`
  - Runs crop-based person/global-entity appearance refinement for multi-camera output.
  - Also supports single-camera track-level refinement with `entity_hint=track_<id>`.
  - Writes into existing fields only: `object_color`, `appearance_notes`, `keywords`, `event_text`, `entity_hint`.
  - Does not require a schema migration.
- `video.indexing.search_enrichment`
  - Normalizes clothing/color tokens for RAG.
  - Adds retrieval-friendly keywords such as `light_grey_hoodie`, `hood_up`, `cross_camera`, and `same_person`.

### Video-Only QA

Use this to test YOLO + OSNet + topology + matching + optional appearance refinement before involving the agent:

```powershell
python scripts\run_mevid_video_eval.py --slot 13-50 --limit 40
```

Equivalent direct command:

```powershell
python tests\test_mevid_full.py --slot 13-50 --limit 40 --no-refine --appearance-refine
```

Useful options:

```powershell
--force-appearance-refine
--with-clip-refine
--video-dir _data/mevid_slots
```

### Single-Camera Appearance Refinement

For a normal single-camera pipeline output, run detection/tracking first:

```powershell
python -m video.factory.coordinator video path\to\video.mp4 --out-dir pipeline_output
```

Then run track-level crop appearance refinement on the generated `*_events.json`:

```powershell
python -m video.factory.coordinator appearance --events pipeline_output\your_video_events.json
```

This creates:

```text
*_events_appearance_refined.json
```

The result is shaped like vector refinement events and can be merged/exported for RAG without changing the event schema.

### Video + Agent E2E

Before running the agent, regenerate vector seeds if the pipeline/refinement cache changed:

```powershell
python scripts\generate_mevid_vector_flat.py --slot 13-50 --force
```

Then run the balanced agent evaluation:

```powershell
python scripts\run_mevid_agent_eval.py --slot 13-50 --limit 40
```

Equivalent direct command:

```powershell
python tests\test_mevid_video_agent_e2e.py --slot 13-50 --limit 40 --sample-mode balanced
```

Results are written under:

```text
results/mevid_full_*.json
results/mevid_agent_e2e/<slot>_<timestamp>/
```

The agent report includes answer accuracy, top-hit rate, category breakdown, and per-case predictions. Negative/cross-camera cases should be inspected separately because a model that answers `yes` for every case can still look strong on positive-heavy samples.

### Module Smoke Tests

Run the fast tests for the new MEVID video helpers:

```powershell
python -m pytest tests\test_mevid_video_modules.py
```

## UCA Video + Agent Evaluation

For UCA dense-caption/video grounding metrics only:

```powershell
python tests\test_uca_unified.py --config full --parts 1 --out results\uca_unified_part1.json
```

For the whole UCA video-to-agent path, first generate or resume video predictions, then feed those video-derived events into the agent index:

```powershell
python tests\test_uca_unified.py --config full --parts 1 --out results\uca_unified_part1.json --resume
python scripts\run_uca_agent_eval.py --limit 40 --video-result-json results\uca_unified_part1.json --force-video-seed
```

Without `--video-result-json`, the agent harness uses the UCA transcript/GT seeds directly. With `--video-result-json`, the chain is `video predictions -> vector-flat seeds -> SQLite/Chroma -> agent graph`.

Outputs are written under:

```text
results/uca_agent_e2e/<timestamp>/
```

The report includes top-hit rate, answer accuracy when labels are available, and span `tIoU` / `R@0.5` when expected timestamps are present in the agent test cases.

## FastAPI Service

- The current MEVID API service lives under `mevid_fastapi/`.
- The React Web UI lives under `frontend/` and proxies `/api` requests to `http://localhost:8765`.
- Local startup on Windows:

```powershell
python -m uvicorn main:app --app-dir mevid_fastapi --host 127.0.0.1 --port 8765
```

```powershell
cd frontend
npm run dev -- --host 127.0.0.1
```

- Default local entry:
  - Web UI: `http://127.0.0.1:5173/`
  - API docs: `http://127.0.0.1:8765/docs`
- Deployment guide:
  - See `deploy.md`

