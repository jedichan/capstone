"""
Quick MOTA evaluation on UCF-Crime videos.
Since MOT16 download is blocked, we use a two-model pseudo-GT approach:
  - GT:   YOLO11m @ conf=0.55 (stronger model, higher precision → GT)
  - Pred: YOLO11n + BotSORT  @ conf=0.35 (actual tracker being evaluated)

This is a standard "detector oracle" approach for estimating tracking MOTA
when human-annotated GT is unavailable.

Run: python scripts/eval_mota_mot16.py
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

# ── Fix motmetrics / NumPy 2.0 incompatibility ───────────────────────────────
np.asfarray = lambda a, dtype=float: np.asarray(a, dtype=dtype)
import motmetrics as mm

# ── Config ────────────────────────────────────────────────────────────────────
MAX_FRAMES  = 200
GT_CONF     = 0.55   # higher conf → cleaner GT
PRED_CONF   = 0.35
IOU_THRESH  = 0.5    # match threshold

# Pick a video with people visible
VIDEO_PATH = "_data/Anomaly-Videos-Part-1/Abuse/Abuse001_x264.mp4"
if not Path(VIDEO_PATH).exists():
    VIDEO_PATH = str(sorted(Path("_data").glob("**/*.mp4"))[0])

print("=" * 60)
print("MOTA Quick Evaluation — YOLO11n + BotSORT")
print(f"Video:  {Path(VIDEO_PATH).name}")
print(f"Frames: {MAX_FRAMES}")
print(f"GT model:   YOLO11m  conf≥{GT_CONF}")
print(f"Pred model: YOLO11n + BotSORT  conf≥{PRED_CONF}")
print("=" * 60)

import cv2
from ultralytics import YOLO

# ── Step 1: Build pseudo-GT with YOLO11m (high-conf detections) ──────────────
print("\n[1/3] Building pseudo-GT with YOLO11m...")
gt_model = YOLO("yolo11m.pt")

cap = cv2.VideoCapture(VIDEO_PATH)
gt_rows = []
frame_id = 1

# Simple IoU-based ID assignment across frames for GT
gt_active: dict[int, tuple] = {}   # id → last bbox
next_id = 1

def boxes_iou(b1, b2):
    """IoU between two [x,y,w,h] boxes."""
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[0]+b1[2], b2[0]+b2[2])
    y2 = min(b1[1]+b1[3], b2[1]+b2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    area1 = b1[2]*b1[3]; area2 = b2[2]*b2[3]
    return inter / (area1 + area2 - inter + 1e-9)

while frame_id <= MAX_FRAMES:
    ret, frame = cap.read()
    if not ret:
        break

    results = gt_model(frame, classes=[0], conf=GT_CONF, verbose=False)
    dets = []
    for r in results:
        for box in r.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = box
            dets.append([float(x1), float(y1), float(x2-x1), float(y2-y1)])

    # Greedy IoU matching for GT track IDs
    new_active = {}
    used = set()
    for det in dets:
        best_id, best_iou = -1, 0.3
        for gid, prev in gt_active.items():
            if gid in used:
                continue
            iou = boxes_iou(det, prev)
            if iou > best_iou:
                best_iou, best_id = iou, gid
        if best_id == -1:
            best_id = next_id
            next_id += 1
        new_active[best_id] = det
        used.add(best_id)
        gt_rows.append([frame_id, best_id] + det + [1, 1, 1])

    gt_active = new_active
    frame_id += 1

cap.release()
gt_df = pd.DataFrame(gt_rows, columns=['frame','id','x','y','w','h','conf','cls','vis'])
print(f"  GT: {len(gt_df)} detections, {gt_df['id'].nunique()} unique IDs")

# ── Step 2: Run YOLO11n + BotSORT (actual tracker) ───────────────────────────
print("\n[2/3] Running YOLO11n + BotSORT...")
pred_model = YOLO("yolo11n.pt")

cap = cv2.VideoCapture(VIDEO_PATH)
pred_rows = []
frame_id = 1

while frame_id <= MAX_FRAMES:
    ret, frame = cap.read()
    if not ret:
        break
    results = pred_model.track(
        frame, classes=[0], conf=PRED_CONF, iou=0.45,
        tracker="botsort.yaml", persist=True, verbose=False,
    )
    for r in results:
        if r.boxes.id is None:
            continue
        for box, tid in zip(r.boxes.xyxy.cpu().numpy(),
                            r.boxes.id.cpu().numpy().astype(int)):
            x1, y1, x2, y2 = box
            pred_rows.append([frame_id, int(tid),
                               float(x1), float(y1),
                               float(x2-x1), float(y2-y1)])
    frame_id += 1

cap.release()
pred_df = pd.DataFrame(pred_rows, columns=['frame','id','x','y','w','h']) if pred_rows \
          else pd.DataFrame(columns=['frame','id','x','y','w','h'])
print(f"  Pred: {len(pred_df)} detections, {pred_df['id'].nunique() if len(pred_df) else 0} unique IDs")

# ── Step 3: Compute MOTA with motmetrics ──────────────────────────────────────
print("\n[3/3] Computing MOTA...")

acc = mm.MOTAccumulator(auto_id=True)
total_frames = max(gt_df['frame'].max() if len(gt_df) else 0,
                   pred_df['frame'].max() if len(pred_df) else 0)

for fid in range(1, int(total_frames) + 1):
    gt_f   = gt_df[gt_df['frame'] == fid]
    pred_f = pred_df[pred_df['frame'] == fid] if len(pred_df) > 0 else pd.DataFrame()

    gt_ids   = gt_f['id'].tolist()
    pred_ids = pred_f['id'].tolist() if len(pred_f) > 0 else []

    if gt_ids and pred_ids:
        gt_boxes   = gt_f[['x','y','w','h']].values.astype(float)
        pred_boxes = pred_f[['x','y','w','h']].values.astype(float)
        dist = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=IOU_THRESH)
    else:
        dist = np.empty((len(gt_ids), len(pred_ids)))

    acc.update(gt_ids, pred_ids, dist)

mh = mm.metrics.create()
summary = mh.compute(acc, metrics=[
    'mota', 'motp', 'idf1',
    'num_matches', 'num_false_positives', 'num_misses', 'num_switches',
    'num_objects', 'num_predictions',
], name='YOLO11n+BotSORT')

r = summary.iloc[0]
mota_pct = float(r['mota']) * 100
motp_pct = (1 - float(r['motp'])) * 100 if r['motp'] == r['motp'] else 0.0
idf1_pct = float(r['idf1']) * 100

# ── Print results ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("MOTA EVALUATION RESULTS")
print("=" * 60)
print(f"  Video:       {Path(VIDEO_PATH).name} (first {int(total_frames)} frames)")
print(f"  Tracker:     YOLO11n + BotSORT")
print(f"  GT source:   YOLO11m @ conf≥{GT_CONF}")
print()
print(f"  MOTA:        {mota_pct:6.1f}%  {'✓ ≥60%' if mota_pct >= 60 else '✗ <60%'}")
print(f"  MOTP:        {motp_pct:6.1f}%  (localisation precision)")
print(f"  IDF1:        {idf1_pct:6.1f}%  (identity preservation)")
print()
print(f"  GT objects:     {int(r['num_objects'])}")
print(f"  Predictions:    {int(r['num_predictions'])}")
print(f"  Matches:        {int(r['num_matches'])}")
print(f"  Misses  (FN):   {int(r['num_misses'])}")
print(f"  FP:             {int(r['num_false_positives'])}")
print(f"  ID Switches:    {int(r['num_switches'])}")
print("=" * 60)
print()
print("Note: GT = YOLO11m high-conf detections (pseudo-GT).")
print("For formal MOTA, run against VIRAT/MOT16 human annotations.")

# Save result
result = {
    "video": Path(VIDEO_PATH).name,
    "frames_evaluated": int(total_frames),
    "tracker": "YOLO11n + BotSORT",
    "gt_source": f"YOLO11m @ conf>={GT_CONF}",
    "MOTA_%": round(mota_pct, 1),
    "MOTP_%": round(motp_pct, 1),
    "IDF1_%": round(idf1_pct, 1),
    "num_objects": int(r['num_objects']),
    "num_matches": int(r['num_matches']),
    "FN": int(r['num_misses']),
    "FP": int(r['num_false_positives']),
    "ID_switches": int(r['num_switches']),
    "note": "Pseudo-GT via YOLO11m; for reference only",
}
out_path = Path("results/mota_tracking.json")
out_path.parent.mkdir(exist_ok=True)
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nSaved to {out_path}")
