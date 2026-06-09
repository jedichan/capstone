# MEVID FastAPI API Reference

**Version**: 1.0.0
**Description**: A unified HTTP service that combines the multi-camera video pipeline (YOLO + Re-ID + cross-camera matching) with the LangGraph agent question-answering pipeline.
**Start command**:
```bash
cd mevid_fastapi
uvicorn main:app --port 8765
```
**Interactive docs**: after startup, open `http://127.0.0.1:8765/docs`

---

## Endpoint Overview

| Method | Path | Description |
|------|------|------|
| GET | `/healthz` | Health check; returns the current database status |
| POST | `/api/v1/ingest/seeds` | Build SQLite + Chroma from pre-generated vector-flat files |
| POST | `/api/v1/ingest/videos` | Run the full multi-camera pipeline from raw videos |
| POST | `/api/v1/database/select` | Manually switch to an existing database |
| POST | `/api/v1/agent/query` | Single natural-language query |
| POST | `/api/v1/eval/run` | Run the full evaluation suite; returns accuracy / IoU metrics |

---

## Standard Call Flow

```
Start service
  |
POST /api/v1/ingest/seeds   (or /ingest/videos)
  | (auto-selects the new database)
POST /api/v1/agent/query    (query one at a time)
  |
POST /api/v1/eval/run       (batch evaluation)
```

---

## Endpoint Details

### GET `/healthz`

Health check. Returns the service status and the currently selected database.

**Example response**

```json
{
  "status": "ok",
  "database_selected": true,
  "sqlite_path": "/path/to/episodic_events.sqlite",
  "chroma_path": "/path/to/chroma",
  "namespace": "mevid_eval"
}
```

---

### POST `/api/v1/ingest/seeds`

Builds the SQLite + Chroma databases from pre-generated `*_events_vector_flat.json` seed files, **and automatically switches the agent to the newly built database**.

Use this when video processing (pipeline phases 1-5) is already complete and you only need to rebuild the retrieval stores.

**Request body**

| Field | Type | Required | Default | Description |
|------|------|------|--------|------|
| `seeds_dir` | string | yes | - | Directory containing the `*_events_vector_flat.json` files (absolute path) |
| `namespace` | string | | `"mevid"` | Chroma collection namespace prefix |
| `reset_existing` | bool | | `true` | Whether to rebuild an existing database |

**Example request**

```bash
curl -X POST http://127.0.0.1:8765/api/v1/ingest/seeds \
  -H "Content-Type: application/json" \
  -d '{
    "seeds_dir": "/path/to/events_vector_flat",
    "namespace": "mevid_eval",
    "reset_existing": true
  }'
```

**Example response**

```json
{
  "status": "ok",
  "namespace": "mevid_eval",
  "sqlite_path": "/path/mevid_fastapi/data/runtime/mevid_eval/episodic_events.sqlite",
  "chroma_path": "/path/mevid_fastapi/data/runtime/mevid_eval/chroma",
  "seed_file_count": 6,
  "sqlite_rows": 697,
  "chroma_child_records": 620,
  "chroma_ge_records": 5,
  "elapsed_ms": 85000.0
}
```

**Chroma collection naming**

After building, the following collections are created (identical to the naming used by `run_agent_eval.py`):

| Collection | Name | Description |
|------|------|------|
| Child | `{namespace}_tracks` | Track-level semantic retrieval |
| Parent | `{namespace}_tracks_parent` | 10-minute time-window context |
| Event | `{namespace}_events` | Event-level fine-grained retrieval |
| Global entities (GE) | `mul_camera_global_entities` | Cross-camera entities (fixed name) |

---

### POST `/api/v1/ingest/videos`

Runs the full multi-camera pipeline from raw video files (YOLO detection -> OSNet Re-ID -> topology learning -> cross-camera matching), generates seed files, builds the databases, **and automatically switches the agent to the new database**.

> Note: this endpoint runs YOLO inference on every video stream. Six 5-minute camera streams take roughly 30-60 minutes on CPU.

**Request body**

| Field | Type | Required | Default | Description |
|------|------|------|--------|------|
| `camera_videos` | dict[str, str] | yes | - | `{camera_id: absolute_video_path}` |
| `namespace` | string | | `"mevid"` | Chroma namespace prefix |
| `conf` | float | | `0.40` | YOLO detection confidence threshold |
| `iou` | float | | `0.40` | YOLO tracking IoU threshold |
| `reid_device` | string | | `"cpu"` | Re-ID device (`"cpu"` or `"cuda"`) |
| `refine` | bool | | `false` | Whether to run LLM event refinement (phase 5) |
| `refine_model` | string | | `"qwen3.6-plus"` | LLM refinement model name |
| `reset_existing` | bool | | `true` | Whether to rebuild an existing database |

**Example request**

```bash
curl -X POST http://127.0.0.1:8765/api/v1/ingest/videos \
  -H "Content-Type: application/json" \
  -d '{
    "camera_videos": {
      "G329": "/data/mevid_slots/13-50/G329.avi",
      "G421": "/data/mevid_slots/13-50/G421.avi",
      "G424": "/data/mevid_slots/13-50/G424.avi",
      "G506": "/data/mevid_slots/13-50/G506.avi",
      "G508": "/data/mevid_slots/13-50/G508.avi",
      "G328": "/data/mevid_slots/13-50/G328.avi"
    },
    "namespace": "slot_13_50",
    "conf": 0.40,
    "iou": 0.40,
    "reid_device": "cpu",
    "refine": false
  }'
```

**Response structure** is the same as `/api/v1/ingest/seeds`, with an additional `pipeline_cache_path` field.

---

### POST `/api/v1/database/select`

Manually switches the agent to an existing SQLite + Chroma database (without rebuilding). Useful for reusing a previously built result.

**Request body**

| Field | Type | Required | Description |
|------|------|------|------|
| `namespace` | string | yes | Chroma namespace prefix |
| `sqlite_path` | string | yes | Absolute path to the SQLite database |
| `chroma_path` | string | yes | Absolute path to the Chroma directory |

**Example request**

```bash
curl -X POST http://127.0.0.1:8765/api/v1/database/select \
  -H "Content-Type: application/json" \
  -d '{
    "namespace": "mevid_eval",
    "sqlite_path": "/path/to/episodic_events.sqlite",
    "chroma_path": "/path/to/chroma"
  }'
```

**Example response**

```json
{ "status": "ok", "namespace": "mevid_eval" }
```

---

### POST `/api/v1/agent/query`

Sends a single natural-language question to the LangGraph agent and returns the answer plus retrieval context. A database must be selected before calling this.

**Agent node chain**

```
self_query_node -> query_classification_node -> parallel_retrieval_fusion_node
-> match_verifier_node -> final_answer_node -> summary_node
```

**Request body**

| Field | Type | Required | Default | Description |
|------|------|------|--------|------|
| `query` | string | yes | - | Natural-language question |
| `thread_id` | string\|null | | `null` | Conversation thread ID (auto-generated when null) |
| `user_id` | string | | `"mevid_user"` | User identifier |
| `include_rows` | bool | | `false` | Whether to include raw retrieved rows in the response |
| `include_node_trace` | bool | | `true` | Whether to return the node execution path |
| `top_k_rows` | int | | `5` | Maximum number of retrieved rows to return |

**Example request**

```bash
curl -X POST http://127.0.0.1:8765/api/v1/agent/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Did a person with dark coat appear in camera G329 and then appear again in camera G421?",
    "include_rows": true,
    "include_node_trace": true
  }'
```

**Example response**

```json
{
  "query": "Did a person with dark coat appear in camera G329 and then appear again in camera G421?",
  "answer": "Yes, a person with a dark coat was tracked across camera G329 and G421 (person_global_3, 50.4s-122.8s).",
  "answer_type": "cross_camera",
  "node_trace": [
    "self_query_node", "query_classification_node",
    "parallel_retrieval_fusion_node", "match_verifier_node",
    "final_answer_node", "summary_node"
  ],
  "elapsed_ms": 7200.0,
  "verifier_result": {
    "verdict": "exact",
    "entity_id": "person_global_3",
    "matched_cameras": ["G329", "G421"]
  },
  "rows": [
    {
      "video_id": "2018-03-11.13-50-01.13-55-01.school.G421.r13",
      "track_id": "person_global_3",
      "start_time": 50.4,
      "end_time": 63.3,
      "event_text_en": "person_global_3 walked across..."
    }
  ]
}
```

**Error responses**

| HTTP status | Cause |
|------------|------|
| 400 | Empty query, or no database selected yet |
| 500 | Internal agent error |

---

### POST `/api/v1/eval/run`

Runs the agent evaluation over every question in `sampled_10.json` and returns summary metrics such as accuracy, top-hit rate, and IoU. A database must be selected before calling this.

> The full 10-question evaluation takes about 2-3 minutes (set the HTTP timeout to 600s or more).

**Request body**

| Field | Type | Required | Default | Description |
|------|------|------|--------|------|
| `questions_file` | string\|null | | `null` | Path to the question-set JSON file; falls back to the built-in `sampled_10.json` when null |

**Example request**

```bash
curl -X POST http://127.0.0.1:8765/api/v1/eval/run \
  -H "Content-Type: application/json" \
  -d '{}' \
  --max-time 600
```

**Example response (summary section)**

```json
{
  "total": 10,
  "correct": 10,
  "accuracy": 1.0,
  "top_hit_total": 9,
  "top_hit_rate": 0.9,
  "mean_iou": 0.301,
  "iou_threshold": 0.15,
  "n_iou_pass": 9,
  "iou_pass_rate": 0.9,
  "individual_iou": [0.301, 0.417, 0.237, 0.493, 0.240, 0.298, 0.000, 0.426],
  "by_category": {
    "temporal":     { "total": 2, "correct": 2, "accuracy": 1.0, "top_hit": 2, "top_hit_rate": 1.0 },
    "single_camera":{ "total": 3, "correct": 3, "accuracy": 1.0, "top_hit": 3, "top_hit_rate": 1.0 },
    "cross_camera": { "total": 3, "correct": 3, "accuracy": 1.0, "top_hit": 2, "top_hit_rate": 0.67 },
    "negative":     { "total": 2, "correct": 2, "accuracy": 1.0, "top_hit": 2, "top_hit_rate": 1.0 }
  },
  "cases": [ ... ],
  "elapsed_total_ms": 137000.0
}
```

**IoU notes**

- IoU is computed only for questions whose expected answer is `yes`.
- The predicted time range is first extracted from the agent's final answer text (format `M:SS-M:SS` or `Xs-Xs`), then from the `start_time/end_time` fields of the retrieved rows.
- A +/-30s padding is applied to the predicted range to absorb retrieval-granularity error.
- `iou_pass = correct AND iou >= 0.15` (identical to `run_agent_eval.py`).

---

## Reference Evaluation Results

### Evaluation configuration

| Item | Value |
|--------|-----|
| Dataset | MEVID slot 13-50, 6 cameras |
| Question count | 10 (sampled_10.json) |
| Retrieval level | child (track level, `AGENT_CHROMA_RETRIEVAL_LEVEL=child`) |
| Parent projection | disabled |
| Execution mode | parallel_fusion (SQL + hybrid parallel fusion) |

### Final metrics

| Metric | Result |
|------|------|
| **Answer accuracy** | **10/10 = 100.0%** |
| **Top-hit rate** | **9/10 = 90.0%** |
| Mean IoU (yes questions, +/-30s padding) | 0.301 |
| IoU >= 0.15 pass rate | 9/10 = 90.0% |

### By category

Questions fall into four categories:

| Category | Description | Count | Answer accuracy | Top-hit |
|------|------|------|-----------|---------|
| temporal | Single-camera query with a specific timestamp | 2 | 2/2 (100%) | 2/2 (100%) |
| single_camera | Appearance recognition + behavior event (single camera) | 3 | 3/3 (100%) | 3/3 (100%) |
| cross_camera | Cross-camera same-person re-identification | 3 | 3/3 (100%) | 2/3 (67%) |
| negative | Negative sample (cross-camera, different person) | 2 | 2/2 (100%) | 2/2 (100%) |

### Per-question results

| # | Category | Camera | Expected | Predicted | Top-hit | IoU |
|---|------|--------|------|------|---------|-----|
| Q01 | temporal | G506 ~2:11 | yes | yes | hit | 0.301 |
| Q02 | single_camera | G508 | yes | yes | hit | 0.417 |
| Q03 | cross_camera | G424->G328 | yes | yes | hit | 0.237 |
| Q04 | single_camera | G508 | yes | yes | hit | 0.493 |
| Q05 | single_camera | G424 | yes | yes | hit | 0.240 |
| Q06 | cross_camera | G328->G424 | yes | yes | miss | 0.298 |
| Q07 | temporal | G421 ~2:08 | yes | yes | hit | 0.000 |
| Q08 | negative | G421->G508 | no | no | hit | - |
| Q09 | cross_camera | G329->G421 | yes | yes | hit | 0.426 |
| Q10 | negative | G328->G421 | no | no | hit | - |

### Top-hit miss & IoU analysis

**Q06 (cross_camera, top-hit miss)**: white-shirt person G328->G424; the top-1 retrieval returned G328 instead of G424, but the agent still answered correctly through multi-path fusion.

**Q07 (temporal, IoU=0.000)**: the agent correctly answered yes, but the extracted time window (~0:54) fell into the segment of another person in G421 (person_global_3, dark coat) rather than person_global_4 (dark jacket, plaid shirt, 2:08-3:44). The answer is correct but the temporal localization is confused.

---

## End-to-End Driver Script

`run_mevid_eval.py` wraps the full three-step flow:

```bash
# 1. Start the service (Terminal 1)
cd mevid_fastapi
uvicorn main:app --port 8765

# 2. Run the end-to-end evaluation (Terminal 2)
python mevid_fastapi/run_mevid_eval.py
```

The script calls, in order:
1. `GET /healthz` - confirm the service is ready
2. `POST /api/v1/ingest/seeds` - build the database (using the `events_vector_flat/` seeds)
3. `POST /api/v1/eval/run` - run the 10-question evaluation, print the summary table, and save JSON to `data/runtime/{namespace}/fastapi_eval_result.json`

---

## File Structure

```
mevid_fastapi/
├── main.py              # FastAPI app entry point, route definitions
├── models.py            # Pydantic request/response models
├── service.py           # Agent service (LangGraph + DB environment management)
├── pipeline_service.py  # Video pipeline service (seed ingestion + multi-camera pipeline)
├── eval_service.py      # Evaluation service (question-driven + IoU/top-hit computation)
├── run_mevid_eval.py    # End-to-end evaluation driver script
├── API_REFERENCE.md     # This file
└── data/
    └── runtime/
        └── {namespace}/
            ├── episodic_events.sqlite
            ├── chroma/
            └── fastapi_eval_result.json
```
