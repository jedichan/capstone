"""
Official MOTA evaluation on MOT16 train set.
Uses human-annotated GT from MOT16 and runs YOLO11n + BotSORT as the tracker.

Dataset layout expected:
  _data/OpenDataLab___MOT16/extracted/train/MOT16-XX/
      gt/gt.txt       ← official annotations
      img1/*.jpg      ← frames

Run: python scripts/eval_mota_mot16_official.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# ── Fix motmetrics / NumPy 2.0 incompatibility ────────────────────────────────
np.asfarray = lambda a, dtype=float: np.asarray(a, dtype=dtype)
import motmetrics as mm

# ── Config ────────────────────────────────────────────────────────────────────
MOT16_ROOT  = Path("_data/OpenDataLab___MOT16/extracted/train")
MAX_FRAMES  = 300          # frames per sequence (set to 0 for full sequence)
PRED_CONF   = 0.10
IOU_THRESH  = 0.5
SEQUENCES   = ["MOT16-02", "MOT16-04", "MOT16-05",
               "MOT16-09", "MOT16-10", "MOT16-11", "MOT16-13"]

print("=" * 60)
print("MOTA Official Evaluation — YOLO11m + BotSORT on MOT16")
print(f"Sequences: {SEQUENCES}")
print(f"Max frames per seq: {MAX_FRAMES if MAX_FRAMES else 'full'}")
print(f"Tracker conf: {PRED_CONF}  IoU threshold: {IOU_THRESH}")
print("=" * 60)

import cv2
from ultralytics import YOLO

# ── Accumulate across all sequences ──────────────────────────────────────────
acc_all = mm.MOTAccumulator(auto_id=True)
seq_results = []

for seq in SEQUENCES:
    seq_dir = MOT16_ROOT / seq
    gt_file = seq_dir / "gt" / "gt.txt"
    img_dir = seq_dir / "img1"

    if not gt_file.exists():
        print(f"  [SKIP] {seq}: gt.txt not found")
        continue
    if not img_dir.exists():
        print(f"  [SKIP] {seq}: img1/ not found (frames not extracted yet?)")
        continue

    # ── Load official GT (class=1 pedestrian, conf=1 → not ignore) ──────────
    gt_raw = pd.read_csv(gt_file, header=None,
                         names=['frame','id','x','y','w','h','conf','cls','vis'])
    # Standard MOT16 filter: conf=1 (not crowd/ignore), cls=1 (pedestrian)
    gt_df = gt_raw[(gt_raw['conf'] == 1) & (gt_raw['cls'] == 1)].copy()

    frames_available = sorted(img_dir.glob("*.jpg"))
    if MAX_FRAMES and MAX_FRAMES > 0:
        frames_available = frames_available[:MAX_FRAMES]
    n_frames = len(frames_available)

    print(f"\n[{seq}]  GT persons: {gt_df['id'].nunique()}  "
          f"frames: {n_frames}  GT dets: {len(gt_df[gt_df['frame'] <= n_frames])}")

    # ── Run YOLO11n + BotSORT on frames ──────────────────────────────────────
    pred_rows = []
    # Fully reset tracker by creating a fresh model instance per sequence
    seq_model = YOLO("yolo11m.pt")

    for img_path in frames_available:
        frame_id = int(img_path.stem)   # MOT16 frames: 000001.jpg → 1
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        results = seq_model.track(
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
                                   float(x2 - x1), float(y2 - y1)])

    pred_df = pd.DataFrame(pred_rows, columns=['frame','id','x','y','w','h']) \
              if pred_rows else pd.DataFrame(columns=['frame','id','x','y','w','h'])

    print(f"  Pred: {len(pred_df)} dets, {pred_df['id'].nunique() if len(pred_df) else 0} IDs")

    # ── Per-sequence MOTA ─────────────────────────────────────────────────────
    acc_seq = mm.MOTAccumulator(auto_id=True)
    frame_ids = sorted(gt_df['frame'].unique())
    if MAX_FRAMES:
        frame_ids = [f for f in frame_ids if f <= n_frames]

    for fid in frame_ids:
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

        acc_seq.update(gt_ids, pred_ids, dist)
        acc_all.update(gt_ids, pred_ids, dist)   # also add to global acc

    mh = mm.metrics.create()
    s = mh.compute(acc_seq, metrics=[
        'mota', 'motp', 'idf1', 'num_matches',
        'num_false_positives', 'num_misses', 'num_switches', 'num_objects',
    ], name=seq).iloc[0]

    mota = float(s['mota']) * 100
    motp = (1 - float(s['motp'])) * 100 if s['motp'] == s['motp'] else 0.0
    idf1 = float(s['idf1']) * 100
    print(f"  → MOTA {mota:5.1f}%  MOTP {motp:5.1f}%  IDF1 {idf1:5.1f}%  "
          f"FN={int(s['num_misses'])} FP={int(s['num_false_positives'])} "
          f"IDSW={int(s['num_switches'])}")

    seq_results.append({
        "seq": seq, "MOTA_%": round(mota, 1), "MOTP_%": round(motp, 1),
        "IDF1_%": round(idf1, 1), "GT": int(s['num_objects']),
        "FN": int(s['num_misses']), "FP": int(s['num_false_positives']),
        "IDSW": int(s['num_switches']),
    })

# ── Overall results ───────────────────────────────────────────────────────────
mh = mm.metrics.create()
overall = mh.compute(acc_all, metrics=[
    'mota', 'motp', 'idf1', 'num_matches',
    'num_false_positives', 'num_misses', 'num_switches', 'num_objects', 'num_predictions',
], name='OVERALL').iloc[0]

mota_all = float(overall['mota']) * 100
motp_all = (1 - float(overall['motp'])) * 100 if overall['motp'] == overall['motp'] else 0.0
idf1_all = float(overall['idf1']) * 100

print("\n" + "=" * 60)
print("OVERALL RESULTS  —  YOLO11m + BotSORT on MOT16 train")
print("=" * 60)
print(f"  MOTA:       {mota_all:6.1f}%  {'✓ ≥60%' if mota_all >= 60 else '✗ <60%'}")
print(f"  MOTP:       {motp_all:6.1f}%")
print(f"  IDF1:       {idf1_all:6.1f}%")
print(f"  GT objects: {int(overall['num_objects'])}")
print(f"  Matches:    {int(overall['num_matches'])}")
print(f"  FN:         {int(overall['num_misses'])}")
print(f"  FP:         {int(overall['num_false_positives'])}")
print(f"  ID Switches:{int(overall['num_switches'])}")
print("=" * 60)
print("\nPer-sequence breakdown:")
print(f"  {'Seq':<12} {'MOTA':>7} {'MOTP':>7} {'IDF1':>7} {'GT':>6} {'FN':>6} {'FP':>6} {'IDSW':>6}")
for r in seq_results:
    print(f"  {r['seq']:<12} {r['MOTA_%']:>6.1f}% {r['MOTP_%']:>6.1f}% "
          f"{r['IDF1_%']:>6.1f}% {r['GT']:>6} {r['FN']:>6} {r['FP']:>6} {r['IDSW']:>6}")

# ── Save ──────────────────────────────────────────────────────────────────────
import json
result = {
    "dataset": "MOT16 train (official human-annotated GT)",
    "tracker": "YOLO11m + BotSORT",
    "conf": PRED_CONF,
    "max_frames_per_seq": MAX_FRAMES,
    "MOTA_%": round(mota_all, 1),
    "MOTP_%": round(motp_all, 1),
    "IDF1_%": round(idf1_all, 1),
    "num_objects": int(overall['num_objects']),
    "num_matches": int(overall['num_matches']),
    "FN": int(overall['num_misses']),
    "FP": int(overall['num_false_positives']),
    "ID_switches": int(overall['num_switches']),
    "per_sequence": seq_results,
}
out_path = Path("results/mota_mot16_official.json")
out_path.parent.mkdir(exist_ok=True)
out_path.write_text(json.dumps(result, indent=2))
print(f"\nSaved → {out_path}")
