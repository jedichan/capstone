"""
Quick smoke test for the multi-camera VLM refinement pipeline.
Uses two short UCF-Crime videos as fake "different cameras".
Run from project root with the capstone conda env.
"""
import json
import logging
import os
import sys
from pathlib import Path

# ── Load .env and point ChatOpenAI → DashScope compatible endpoint ────────────
from dotenv import load_dotenv
load_dotenv()

# LangChain's ChatOpenAI reads OPENAI_API_KEY / OPENAI_BASE_URL from env
dashscope_key = os.getenv("DASHSCOPE_API_KEY", "")
dashscope_url = os.getenv("DASHSCOPE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
dashscope_model = os.getenv("DASHSCOPE_CHAT_MODEL", "qwen3.6-plus")

os.environ.setdefault("OPENAI_API_KEY", dashscope_key)
os.environ.setdefault("OPENAI_BASE_URL", dashscope_url)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("smoke_test")

# ── pick two short videos ────────────────────────────────────────────────────
DATA = Path("_data/Anomaly-Videos-Part-1")
candidates = sorted(DATA.glob("**/*.mp4"))[:40]

import cv2

def video_duration(p: Path) -> float:
    cap = cv2.VideoCapture(str(p))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fc  = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return fc / fps if fps > 0 else 0.0

# pick two ≤ 60-second videos from different categories
short = [(p, video_duration(p)) for p in candidates if video_duration(p) <= 60]
short.sort(key=lambda x: x[1])

if len(short) < 2:
    logger.error("Could not find 2 videos ≤ 60s in %s", DATA)
    sys.exit(1)

cam1_path, d1 = short[0]
cam2_path, d2 = short[1]
logger.info("cam1: %s  (%.1fs)", cam1_path, d1)
logger.info("cam2: %s  (%.1fs)", cam2_path, d2)

# ── Stage 1–3: multi-camera coordinator ─────────────────────────────────────
from video.factory.multi_camera_coordinator import run_multi_camera_pipeline

camera_videos = {
    "cam1": str(cam1_path),
    "cam2": str(cam2_path),
}

logger.info("=== Stage 1-3: YOLO + Track + ReID + CrossCam ===")
mc_output = run_multi_camera_pipeline(
    camera_videos=camera_videos,
    model_path="n",          # yolo11n - fastest
    conf=0.25,
    iou=0.45,
    reid_device="cpu",
    num_crops=3,
    use_llm_verify=False,
)

logger.info("global_entities found: %d", len(mc_output.global_entities))
logger.info("merged_events total:   %d", len(mc_output.merged_events))
for cr in mc_output.per_camera:
    logger.info("  %s: %d tracks, %d events, %d clips",
                cr.camera_id, len(cr.tracks), len(cr.events), len(cr.clips))

# ── Stage 4: VLM refinement via refine_multi_camera_output() ────────────────
from video.factory.refinement_runner import RefineEventsConfig, refine_multi_camera_output

cfg = RefineEventsConfig(
    mode="vector",
    frames_per_sec=0.5,   # denser sampling for short clips
    min_frames=4,
    max_frames=12,
    model=dashscope_model,  # from DASHSCOPE_CHAT_MODEL env var
    temperature=0.1,
)

logger.info("=== Stage 4: VLM refinement (refine_multi_camera_output) ===")
refined = refine_multi_camera_output(mc_output, config=cfg)

# ── Print results ────────────────────────────────────────────────────────────
out_path = Path("results/smoke_multi_camera.json")
out_path.parent.mkdir(exist_ok=True)
out_path.write_text(json.dumps(refined, ensure_ascii=False, indent=2), encoding="utf-8")
logger.info("Results written to %s", out_path)

print("\n" + "="*60)
print("PIPELINE OUTPUT SUMMARY")
print("="*60)
for cam_id, cam_result in refined.items():
    events = cam_result.get("events", [])
    print(f"\n[{cam_id}]  video_id={cam_result.get('video_id')}  events={len(events)}")
    for i, ev in enumerate(events[:5]):
        print(f"  [{i}] {ev.get('start_time',0):.1f}s-{ev.get('end_time',0):.1f}s"
              f"  {ev.get('object_color','?')} {ev.get('object_type') or ev.get('class_name','?')}"
              f"  zone={ev.get('scene_zone','?')}")
        print(f"       {ev.get('event_text','(no text)')}")
    if len(events) > 5:
        print(f"  ... and {len(events)-5} more events")
print("\n" + "="*60)
print("GLOBAL ENTITIES (cross-camera matches)")
print("="*60)
for ent in mc_output.global_entities:
    apps = ent.appearances
    print(f"  {ent.global_entity_id}: {len(apps)} camera(s)")
    for a in apps:
        print(f"    {a.camera_id}  track={a.track_id}  {a.start_time:.1f}s-{a.end_time:.1f}s  conf={a.confidence:.2f}")
print()
