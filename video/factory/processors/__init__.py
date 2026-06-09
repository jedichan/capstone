"""
Processors (split per README):

- vision.py — YOLO + tracking, per-frame detections
- analyzer.py — track aggregation, event slicing
- event_track_pipeline.py — chain the two and write JSON
- event_refine_langchain.py — LLM refine CLI (load_dotenv)
- captioner.py — optional multimodal captioning

Orchestration: video.factory.coordinator (run_video_to_events / run_refine_events).
"""
