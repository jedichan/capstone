# CS32-1 Surveillance Video Query System — Pipeline Flowchart

> **Legend:** 🟢 Fully implemented  |  🟡 Partially implemented  |  🔴 Stub / Not yet implemented

---

## 1. Overall System Architecture

```mermaid
flowchart TD
    INPUT([🎥 Input: Video File\nMP4 / AVI / MOV])

    subgraph S1["⚙️ STAGE 1 — Visual Event Extraction 🟢"]
        direction TB
        DET["YOLO11m Detection\nconf=0.25, iou=0.45\nvision.py"]
        TRACK["BoT-SORT + ReID Tracking\nwith_reid=True\nbotsort_reid.yaml"]
        AGG["aggregate_tracks()\nPer-frame → Per-track records\nanalyzer.py"]
        SLICE["slice_events()\n5 event types classification\nanalyzer.py"]
        OUT1[("📄 events.json\n📄 clips.json")]
    end

    subgraph MULTICAM["🔀 OPTIONAL — Multi-Camera Extension 🟢"]
        direction TB
        REID["ReIDEmbedder\nOSNet x0_5 (512-dim)\n+ MobileNetV2 fallback\nreid_embedder.py"]
        STITCH["_stitch_same_camera_fragments()\ngap ≤ 3s AND cosine ≥ 0.80\nmulti_camera_coordinator.py"]
        MATCH["match_across_cameras()\n0.7×cosine + 0.3×time\ncross_camera_matcher.py"]
        VLM["verify_person_match_with_llm()\nGPT-5 vision for cosine ∈ [0.65, 0.80]\nevent_refinement_llm.py"]
        ENTITY["_build_global_entities()\nUnion-Find merge\ncross_camera_matcher.py"]
        OUT3[("📄 multi_cam.json\nGlobalEntity list")]
    end

    subgraph S2["🧠 STAGE 2 — LLM Refinement 🟢"]
        direction TB
        FRAMES["sample_frames_uniform()\nfps-based adaptive sampling\nframes.py"]
        HARD["build_entities_with_hard_constraints()\nHard rules + LLM YES/NO merge\nevent_refinement_llm.py"]
        LLM["refine_events_with_llm() [mode=full]\nOR\nrefine_vector_events_with_llm() [mode=vector]\nGPT-5 via LangChain + PydanticOutputParser\nevent_refinement_llm.py"]
        OUT2[("📄 refined_events.json [full]\nOR\n📄 vector_flat.json [vector]")]
    end

    subgraph S3["📦 STAGE 3 — Indexing 🟡"]
        direction TB
        DOCBUILD["document_builder.py\nBuild text docs from events\n🔴 stub"]
        EMBED["get_qwen_embedding()\nDashScope text-embedding-v3\n1024-dim float32\nembedder.py 🟢"]
        STORE["store_manager.py\nInsert to SQLite + sqlite-vec\n🔴 stub"]
        DB[("🗄️ episodic_memory.db\nepisodic_events table\nepisodic_events_vec table")]
    end

    subgraph S4["🤖 STAGE 4 — LangGraph Query Agent 🔴"]
        direction TB
        PARSE["nodes/parse.py\nExtract intent + entities\n🔴 stub"]
        ROUTE["nodes/route.py\nRoute: STRUCTURED / VECTOR / HYBRID\n🔴 stub"]
        RETRIEVE["nodes/retrieve.py\n→ EventRetriever\n🟡 partial"]
        ANSWER["nodes/answer.py\nGPT-5 answer synthesis\n🔴 stub"]
        OUT4(["💬 Natural Language Answer\n+ evidence_events + timestamps"])
    end

    subgraph RETRIEVAL["🔍 Retrieval Layer 🟡"]
        SR["structured_search()\nPure SQL filters\nevent_retriever.py 🟢"]
        HR["hybrid_event_search()\nQwen KNN + SQL re-rank\nevent_retriever.py 🟢"]
    end

    INPUT --> S1
    DET --> TRACK --> AGG --> SLICE --> OUT1
    OUT1 --> S2
    OUT1 --> MULTICAM
    REID --> STITCH --> MATCH
    MATCH --> VLM --> ENTITY --> OUT3
    OUT3 --> S2
    FRAMES --> HARD --> LLM --> OUT2
    OUT2 --> S3
    DOCBUILD --> EMBED --> STORE --> DB
    DB --> RETRIEVAL
    RETRIEVAL --> SR & HR --> RETRIEVE
    PARSE --> ROUTE --> RETRIEVE --> ANSWER --> OUT4

    USER([👤 User Query]) --> PARSE
```



---

## 2. Stage 1 — Detailed Detection & Tracking Flow

```mermaid
flowchart TD
    V([🎥 video_path])

    subgraph EP["event_track_pipeline.py — run_pipeline()"]
        RM["resolve_model(model_path)\n'11m' / 'yolo11m' / 'yolov11m'\n→ _model/yolo11m.pt"]
        RT["resolve_tracker(tracker)\n'botsort_reid' → botsort_reid.yaml\n'bytetrack' → bytetrack.yaml"]
        RD["resolve_device(device)\n'mps' (Apple Silicon)\n'cuda' / 'cpu'"]
    end

    subgraph VIS["vision.py — run_yolo_track_on_video()"]
        YOLO["YOLO11m.predict()\nclasses: person, car, bus,\ntruck, motorbike, bicycle\nconf=0.25, iou=0.45"]
        BOTS["BoT-SORT+ReID tracking\ntrack_high_thresh: 0.25\ntrack_buffer: 100\nmatch_thresh: 0.6\ngmc_method: sparseOptFlow\nappearance_thresh: 0.25"]
        FD["frame_detections\nlist[list[tuple[\n  track_id: int,\n  class_name: str,\n  confidence: float,\n  bbox_xyxy: list[float]\n]]]"]
    end

    subgraph ANA["analyzer.py"]
        AGG["aggregate_tracks(fps, frame_detections)\n→ list[dict] per track:\n  track_id, class_name\n  start/end_time (sec)\n  positions, motion_score\n  camera_id (optional)"]

        MOTION["Motion Detection:\nSliding window 1.5s\nFrame-diff edge sum\nthreshold: 20.0px\njitter floor: 3.0px"]

        SLICE["slice_events(tracks, fps, ...)\nState machine per track_id"]

        E1["motion_segment\nhigh motion throughout"]
        E2["presence_after_motion\nstationary after motion ≥5s"]
        E3["appearance\nnew track enters scene"]
        E4["disappearance\ntrack leaves scene"]
        E5["presence_static\nlow motion throughout"]
    end

    SAVE["save_pipeline_output()\n→ {base}_events.json\n→ {base}_clips.json"]

    V --> EP
    RM & RT & RD --> YOLO
    YOLO --> BOTS --> FD --> AGG
    AGG --> MOTION --> SLICE
    SLICE --> E1 & E2 & E3 & E4 & E5 --> SAVE

    style E1 fill:#FEF3C7
    style E2 fill:#FEF3C7
    style E3 fill:#D1FAE5
    style E4 fill:#FEE2E2
    style E5 fill:#EDE9FE
```



---

## 3. Stage 2 — LLM Refinement Flow

```mermaid
flowchart TD
    IN1([📄 events.json])
    IN2([📄 clips.json])

    subgraph RUNNER["refinement_runner.py — run_refine_events_from_files()"]
        CFG["RefineEventsConfig\nmode: 'full' | 'vector'\nmodel: 'gpt-5.4-mini'\ntemperature: 0.1\nnum_frames: adaptive\nmax_time_adjust_sec: 0.5\nentity_merge_min_llm_confidence: 0.75"]
        NORM["enrich_events_with_normalized_location()\nAdd start/end_center_norm [0,1]"]
        SAMPLE["sample_frames_uniform()\nvideo_path, start_sec, end_sec\n→ list[FrameSample(t_sec, jpg_base64)]"]
    end

    subgraph FULL["mode = 'full'  (rich structured output)"]
        HARD["build_entities_with_hard_constraints()\n  Hard rules before LLM:\n  ✗ temporal overlap not allowed\n  ✗ max gap > 5min → skip\n  ✓ colour compatibility check\n  → _verify_merge_yesno_with_llm()\n    confidence ≥ 0.75 to merge\n  → list[RefinedEntity]"]

        LLM_F["refine_events_with_llm()\nLangChain ChatOpenAI\nPydanticOutputParser\ntemperature=0.1\nInputs: raw_events + frames + entities\nOutputs: RefinedEventsPayload\n  scene_context\n  entities (merged tracks)\n  refined_events\n  temporal_policy\n  location_policy"]

        OUT_F[("📄 refined_events.json\nFull structured output\nper clip")]
    end

    subgraph VECTOR["mode = 'vector'  (flat for indexing)"]
        LLM_V["refine_vector_events_with_llm()\ntemperature=0.0\nOutputs: VectorEventsPayload\n  per event:\n    object_type\n    object_color_cn (English)\n    scene_zone_cn (English)\n    event_text_cn (English)\n    keywords\n    context_en"]

        OUT_V[("📄 vector_flat.json\nFlat list of VectorEvent\nReady for embedding")]
    end

    IN1 & IN2 --> CFG --> NORM --> SAMPLE
    SAMPLE --> HARD --> LLM_F --> OUT_F
    SAMPLE --> LLM_V --> OUT_V

    style FULL fill:#EFF6FF
    style VECTOR fill:#F0FDF4
```



---

## 4. Multi-Camera Extension Flow

```mermaid
flowchart TD
    VIDS([🎥 camera_videos\ndict cam_id → video_path])

    subgraph COORD["multi_camera_coordinator.py — run_multi_camera_pipeline()"]

        subgraph PER_CAM["For each camera (parallel)"]
            S1P["_process_single_camera(cam_id, video_path)\n  1. run_yolo_track_on_video()\n  2. aggregate_tracks()\n  3. slice_events()\n  4. extract_person_crops() × num_crops\n  5. embedder.embed_crops() → L2-norm vectors\n  → CameraResult"]
        end

        STITCH["_stitch_same_camera_fragments()\nFor each track pair in same camera:\n  gap ≤ same_camera_max_gap_sec (3s)\n  AND cosine ≥ same_camera_reid_threshold (0.80)\n  → merge: avg embeddings, extend time range"]
    end

    subgraph MATCHER["cross_camera_matcher.py — match_across_cameras()"]
        CAND["build_candidate_pairs(per_camera, config)\n  Filter: person_only=True\n  Filter: time gap ≤ max_transition_sec (30s)\n  → candidate (CameraResult_A, track_A, CameraResult_B, track_B)"]

        SCORE["score_candidate_pairs(pairs, embedder)\n  score = 0.7 × cosine_similarity\n          + 0.3 × time_window_score\n  → sorted by score desc"]

        VLM_CHECK{{"cosine in\n[0.65, 0.80]?"}}

        VLM["verify_person_match_with_llm(crop_a, crop_b)\n  GPT-5 vision API\n  base64 PNG crops\n  → MatchVerification:\n    is_match: bool\n    confidence: float\n    reasoning: str"]

        GREEDY["_greedy_assign(scored, threshold=0.65)\n  Greedy match by score desc\n  One-to-one constraint per camera pair"]

        UF["_build_global_entities(assignments)\n  Union-Find merge\n  → list[GlobalEntity:\n      global_entity_id,\n      appearances[CameraAppearance]]"]
    end

    subgraph MERGE["Merge Output"]
        LOOKUP["_build_entity_lookup()\n  (cam_id, track_id) → global_entity_id"]
        MERGE_EV["_merge_events(per_camera, lookup)\n  Inject global_entity_id into events\n  where matched"]
        OUT[("📄 multi_cam.json\n  meta: cameras + config\n  global_entities: [...]\n  events: all events\n    + global_entity_id")]
    end

    VIDS --> PER_CAM --> STITCH --> CAND --> SCORE
    SCORE --> VLM_CHECK
    VLM_CHECK -- Yes, borderline → |top_k=3 pairs| VLM --> GREEDY
    VLM_CHECK -- No, clear case → GREEDY
    GREEDY --> UF --> LOOKUP --> MERGE_EV --> OUT

    style VLM fill:#FEF3C7
    style UF fill:#D1FAE5
```



---

## 5. ReID Embedder Architecture

```mermaid
flowchart TD
    INIT["ReIDEmbedder.__init__()\n  config_file, weights, device\n  input_size: (256, 128)"]

    BACK{{"backend?"}}

    OSNET["torchreid OSNet x0_5\n512-dim features\nL2 normalised\nPrimary backend"]

    MOB["torchvision MobileNetV2\n1280-dim features\nGlobal avg pool + flatten\nFallback backend"]

    CROPS["embed_crops(crops, batch_size=64)\n  list[np.ndarray BGR]\n  → resize to 256×128\n  → normalise (ImageNet stats)\n  → batch inference\n  → L2 normalise\n  → (N, dim) np.ndarray"]

    SIM["cosine_similarity(feats_a, feats_b)\n  (Na, Nb) similarity matrix\n  via np.dot (L2-normalised vectors)"]

    USE1["Same-camera stitching\ncosine ≥ 0.80"]
    USE2["Cross-camera candidate scoring\n0.7 × cosine"]
    USE3["BoT-SORT appearance matching\nappearance_thresh: 0.25"]

    INIT --> BACK
    BACK -- torchreid available --> OSNET --> CROPS
    BACK -- torchreid missing --> MOB --> CROPS
    CROPS --> SIM
    SIM --> USE1 & USE2 & USE3

    style OSNET fill:#D1FAE5
    style MOB fill:#FEF3C7
```



---

## 6. Stage 3+4 — Indexing & Agent (Current Status)

```mermaid
flowchart TD
    VEC([📄 vector_flat.json])

    subgraph IDX["video/indexing/ — Indexing Pipeline"]
        DOCB["document_builder.py\nbuild_documents_from_events()\n🔴 STUB — empty file"]
        EMB["embedder.py — get_qwen_embedding(text)\n✅ IMPLEMENTED\nDashScope text-embedding-v3\n1024-dim float32\nRequires: DASHSCOPE_API_KEY"]
        STORE["store_manager.py\ninsert_events_to_sqlite()\ncreate_vec_index()\n🔴 STUB — empty file"]
        DB[("🗄️ episodic_memory.db\nepisodic_events table\nepisodic_events_vec\n(sqlite-vec KNN)")]
    end

    subgraph RET["agent/retrieval/ — Retrieval Layer"]
        SR["structured_search()\nvideo_id, object_type\nscene_zone, start_time_after\nPure SQL WHERE\n✅ IMPLEMENTED"]
        HR["hybrid_event_search()\nquery_text, top_k, filters\nQwen embed → KNN\n+ SQL re-rank\n✅ IMPLEMENTED"]
        GR["graph_retriever.py\n🔴 STUB"]
        FUS["fusion.py\n🔴 STUB"]
        RR["reranker.py\n🔴 STUB"]
    end

    subgraph AGENT["agent/ — LangGraph Query Agent"]
        ST["state.py — AgentState\n🔴 STUB"]
        PARSE["nodes/parse.py\nExtract: intent, object_type\ntime_range, camera_id\n🔴 STUB"]
        ROUTE["nodes/route.py\nSTRUCTURED / VECTOR\nHYBRID / CROSS_CAM / TIME\n🔴 STUB"]
        RETR["nodes/retrieve.py\nDispatch to EventRetriever\n🔴 STUB"]
        ANS["nodes/answer.py\nGPT-5 synthesis\nevidence_ids, timestamps\n🔴 STUB"]
        GRAPH["graph.py\nLangGraph StateGraph\n🔴 STUB"]
    end

    TOOLS["agent/tools/\nsearch.py  playback.py\nsummarize.py  verify.py\n🔴 ALL STUBS"]

    USER([👤 User: 'When did the\nwhite car leave?'])
    ANS_OUT(["💬 Answer + timestamps\n+ evidence events"])

    VEC --> DOCB --> EMB --> STORE --> DB
    DB --> SR & HR
    SR & HR --> RET
    GR & FUS & RR --> RET
    USER --> PARSE --> ROUTE --> RETR
    RET --> RETR --> ANS --> ANS_OUT
    TOOLS --> ANS
    ST --> GRAPH
    PARSE & ROUTE & RETR & ANS --> GRAPH

    style DOCB fill:#FEE2E2
    style STORE fill:#FEE2E2
    style GR fill:#FEE2E2
    style FUS fill:#FEE2E2
    style RR fill:#FEE2E2
    style ST fill:#FEE2E2
    style PARSE fill:#FEE2E2
    style ROUTE fill:#FEE2E2
    style RETR fill:#FEE2E2
    style ANS fill:#FEE2E2
    style GRAPH fill:#FEE2E2
    style TOOLS fill:#FEE2E2
    style EMB fill:#D1FAE5
    style SR fill:#D1FAE5
    style HR fill:#D1FAE5
```



---

## 7. Data Schema Flow

```mermaid
flowchart LR
    V([🎥 Video])

    E["events.json\n─────────────\nevent_type\ntrack_id\nclass_name\nstart/end_time\nbbox_xyxy\nmotion_level\ncamera_id\ndescription_for_llm"]

    C["clips.json\n─────────────\nclip_segments\n  start_sec\n  end_sec"]

    VF["vector_flat.json\n─────────────\nper VectorEvent:\n  object_type\n  object_color\n  scene_zone\n  event_text (EN)\n  keywords\n  context_en\n  start/end_time\n  bbox_xyxy"]

    RF["refined_events.json\n─────────────\nscene_context\n  overview, layout\n  landmarks\nentities[]\n  entity_id\n  merged track_ids\n  appearance, location\nrefined_events[]\n  event_id, entity_id\n  confidence\n  details, evidence\n  location narrative"]

    MC["multi_cam.json\n─────────────\nglobal_entities[]\n  global_entity_id\n  appearances[]\n    camera_id\n    track_id\n    confidence\nevents[]\n  + global_entity_id"]

    DB["episodic_memory.db\n─────────────\nepisodic_events\n  video_id\n  object_type\n  scene_zone\n  event_summary\n  start/end_time\nepisodic_events_vec\n  embedding vec(1024)"]

    V -->|"run_pipeline()"| E & C
    E & C -->|"refine_vector_events_with_llm()"| VF
    E & C -->|"refine_events_with_llm()"| RF
    E -->|"run_multi_camera_pipeline()"| MC
    VF -->|"get_qwen_embedding()\n+ store_manager 🔴"| DB
```



---

## 8. Key Configuration Parameters


| Parameter           | Value                   | Location                    | Effect                      |
| ------------------- | ----------------------- | --------------------------- | --------------------------- |
| Detection model     | `yolo11m.pt`            | `vision.py resolve_model()` | Small object recall         |
| Detection conf      | `0.25`                  | `run_pipeline()` default    | Miss rate vs false positive |
| Detection iou       | `0.45`                  | `run_pipeline()` default    | NMS overlap threshold       |
| Tracker             | `botsort_reid`          | `run_pipeline()` default    | Appearance-based ReID       |
| `with_reid`         | `True`                  | `botsort_reid.yaml`         | Enable OSNet embeddings     |
| `track_buffer`      | `100`                   | `botsort_reid.yaml`         | Frames to keep lost tracks  |
| `appearance_thresh` | `0.25`                  | `botsort_reid.yaml`         | ReID match threshold        |
| `gmc_method`        | `sparseOptFlow`         | `botsort_reid.yaml`         | Camera motion compensation  |
| ReID primary        | OSNet x0_5 (512-dim)    | `reid_embedder.py`          | Person appearance features  |
| ReID fallback       | MobileNetV2 (1280-dim)  | `reid_embedder.py`          | Cross-platform safety       |
| Fragment stitch gap | `3.0s`                  | `default.yaml`              | Same-cam track merge        |
| Fragment stitch cos | `0.80`                  | `default.yaml`              | Same-cam cosine threshold   |
| Cross-cam max gap   | `30.0s`                 | `default.yaml`              | Time window for matching    |
| Cross-cam threshold | `0.65`                  | `default.yaml`              | Min match score             |
| Cross-cam scoring   | `0.7×cosine + 0.3×time` | `cross_camera_matcher.py`   | Combined score formula      |
| LLM verify range    | `[0.65, 0.80]`          | `default.yaml`              | Borderline VLM cases        |
| LLM verify top-k    | `3`                     | `default.yaml`              | Max VLM calls per run       |
| LLM model (refine)  | `gpt-5.4-mini`          | `refinement_runner.py`      | Event description           |
| LLM model (verify)  | `gpt-4o-mini` (default) | `event_refinement_llm.py`   | Person identity check       |
| Embedding model     | Qwen text-embedding-v3  | `embedder.py`               | 1024-dim vector index       |
| Motion window       | `1.5s`                  | `run_pipeline()`            | Sliding window for motion   |
| Motion threshold    | `20.0px`                | `run_pipeline()`            | Frame-diff sum threshold    |
| Entity merge gap    | `300s` (5 min)          | `RefineEventsConfig`        | Max gap for LLM merge       |
| Entity merge conf   | `0.75`                  | `RefineEventsConfig`        | Min LLM confidence          |
| Max time adjust     | `0.5s`                  | `RefineEventsConfig`        | LLM timestamp correction    |


