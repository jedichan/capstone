"""Multi-camera entity evaluation against MEVID ground-truth identity labels.

Two-layer evaluation
--------------------
Layer 1 — MEVID full-corpus entity eval  (1 754 tracklets, 54 identities, 7 occasions)
    Skips YOLO/detection entirely.  Uses pre-extracted MEVID crops as input,
    feeds them into our cross-camera matching algorithm, compares resulting
    GlobalEntity assignments against GT identity labels.

    Metrics:
      * Pair-level Precision / Recall / F1
          - Positive pair: same identity, different cameras (within same occasion)
          - TP: pipeline put them in the same GlobalEntity
          - FP: different identity but pipeline merged them
          - FN: same identity, different cameras, but pipeline missed the link
      * Entity Purity
          - For each GlobalEntity, fraction of constituent tracks
            belonging to the majority GT identity
      * Coverage
          - Fraction of GT multi-camera identities (per occasion) that have
            at least one correctly formed entity

Layer 2 — MEVA 15-min end-to-end sparse check  (qualitative)
    Loads existing pipeline results from results/meva_multicam_1clip.json,
    aligns to the 10 MEVID GT tracklets that fall within our 10:00-10:15
    window, and reports temporal overlap statistics.

Usage
-----
python tests/eval_multicam_entity.py \\
    --annot-zip  _data/mevid-v1-annotation-data.zip \\
    --crops-dir  _data/mevid-crops \\
    --pipeline-json results/meva_multicam_1clip.json \\
    --device cuda \\
    --out results/eval_multicam_entity.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Allow running from repo root without pip-install
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MEVA_FPS = 30.0       # MEVA KF1 video frame rate
MEVA_T0  = 0.0        # Frame 0 corresponds to t=0s (absolute from recording start)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_mevid_tracklets(annot_zip: Path) -> list[dict[str, Any]]:
    """Parse track_test_info.txt → list of tracklet dicts.

    Each dict:
        track_idx   int   row index (used as track_id in CameraResult)
        identity_id str   zero-padded 4-digit  e.g. "0268"
        occasion_id int   1-7
        cam_id      str   zero-padded 3-digit  e.g. "336"
        start_frame int
        end_frame   int
        t_start     float  seconds from recording start
        t_end       float
    """
    tracklets: list[dict] = []
    with zipfile.ZipFile(annot_zip) as zf:
        fname = next(n for n in zf.namelist() if "track_test_info" in n)
        with zf.open(fname) as f:
            for idx, line in enumerate(f):
                vals = [float(v) for v in line.split()]
                if len(vals) != 5:
                    continue
                sf, ef, pid, occ, cam = vals
                tracklets.append(dict(
                    track_idx   = idx,
                    identity_id = f"{int(pid):04d}",
                    occasion_id = int(occ),
                    cam_id      = f"{int(cam):03d}",
                    start_frame = int(sf),
                    end_frame   = int(ef),
                    t_start     = sf / MEVA_FPS,
                    t_end       = ef / MEVA_FPS,
                ))
    logger.info("Loaded %d MEVID tracklets from %s", len(tracklets), annot_zip.name)
    return tracklets


def load_crops(tkl: dict, crops_dir: Path, num_crops: int = 5) -> list[np.ndarray]:
    """Load up to *num_crops* JPEG images for one MEVID tracklet."""
    person_dir = crops_dir / tkl["identity_id"]
    if not person_dir.exists():
        return []
    prefix = f"{tkl['identity_id']}O{int(tkl['occasion_id']):03d}C{tkl['cam_id']}T"
    jpgs = sorted(person_dir.glob(f"{prefix}*F*.jpg"))
    if not jpgs:
        return []
    step = max(1, len(jpgs) // num_crops)
    imgs = []
    for jp in jpgs[::step][:num_crops]:
        img = cv2.imread(str(jp))
        if img is not None:
            imgs.append(img)
    return imgs


# ---------------------------------------------------------------------------
# CameraResult construction from MEVID tracklets
# ---------------------------------------------------------------------------

def build_camera_results(
    tracklets: list[dict],
    embedder: Any,
    num_crops: int = 5,
    crops_dir: Path | None = None,
) -> list[Any]:
    """Build one CameraResult per camera from MEVID tracklets.

    Each MEVID tracklet row → one "person" track in its CameraResult.
    Embeddings are built from pre-extracted MEVID crops.
    """
    from video.core.schema.multi_camera import CameraResult

    cam_groups: dict[str, list[dict]] = defaultdict(list)
    for tkl in tracklets:
        cam_groups[tkl["cam_id"]].append(tkl)

    camera_results: list[CameraResult] = []
    total_embs = 0

    for cam_id, tkls in sorted(cam_groups.items()):
        tracks: list[dict] = []
        person_embeddings: dict[int, np.ndarray] = {}

        for tkl in tkls:
            tid = tkl["track_idx"]
            tracks.append(dict(
                track_id   = tid,
                class_name = "person",
                start_time = tkl["t_start"],
                end_time   = tkl["t_end"],
                camera_id  = f"G{cam_id}",
            ))
            if crops_dir is not None:
                imgs = load_crops(tkl, crops_dir, num_crops)
                if imgs:
                    emb = embedder.embed_crops(imgs)
                    v = emb.mean(axis=0)
                    n = np.linalg.norm(v)
                    person_embeddings[tid] = v / n if n > 0 else v
                    total_embs += 1

        camera_results.append(CameraResult(
            camera_id        = f"G{cam_id}",
            video_path       = "",
            tracks           = tracks,
            events           = [],
            clips            = [],
            meta             = {},
            person_crops     = {},
            person_embeddings= person_embeddings,
        ))
        logger.info(
            "  Camera G%s: %d tracklets, %d embeddings",
            cam_id, len(tkls), len(person_embeddings),
        )

    logger.info("Built %d CameraResults, %d total embeddings", len(camera_results), total_embs)
    return camera_results


# ---------------------------------------------------------------------------
# Layer 1 evaluation metrics
# ---------------------------------------------------------------------------

def build_gt_pair_set(
    tracklets: list[dict],
) -> set[frozenset[int]]:
    """Return set of frozenset({track_idx_a, track_idx_b}) for all GT positive pairs.

    A GT positive pair = same identity_id, different cam_id (within the supplied tracklets).
    """
    id_to_tracklets: dict[str, list[dict]] = defaultdict(list)
    for tkl in tracklets:
        id_to_tracklets[tkl["identity_id"]].append(tkl)

    pairs: set[frozenset[int]] = set()
    for pid, same_id_tkls in id_to_tracklets.items():
        for i in range(len(same_id_tkls)):
            for j in range(i + 1, len(same_id_tkls)):
                a, b = same_id_tkls[i], same_id_tkls[j]
                if a["cam_id"] != b["cam_id"]:
                    pairs.add(frozenset({a["track_idx"], b["track_idx"]}))
    return pairs


def evaluate_entities(
    entities: list[Any],
    gt_pairs: set[frozenset[int]],
    track_idx_to_identity: dict[int, str],
) -> dict[str, float]:
    """Compute pair-level P/R/F1 and entity purity.

    Parameters
    ----------
    entities:
        GlobalEntity list from match_across_cameras().
    gt_pairs:
        Set of frozenset({track_idx_a, track_idx_b}) that are GT positives.
    track_idx_to_identity:
        Maps track_idx → identity_id string.
    """
    import itertools

    # Predicted positive pairs: every (a, b) pair within the same GlobalEntity
    pred_pairs: set[frozenset[int]] = set()
    for ent in entities:
        tids = [app.track_id for app in ent.appearances]
        for a, b in itertools.combinations(tids, 2):
            pred_pairs.add(frozenset({a, b}))

    tp = len(pred_pairs & gt_pairs)
    fp = len(pred_pairs - gt_pairs)
    fn = len(gt_pairs - pred_pairs)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    # Entity purity
    purity_scores: list[float] = []
    for ent in entities:
        tids = [app.track_id for app in ent.appearances]
        if len(tids) < 2:
            continue
        id_counts: dict[str, int] = defaultdict(int)
        for tid in tids:
            pid = track_idx_to_identity.get(tid, "?")
            id_counts[pid] += 1
        majority = max(id_counts.values())
        purity_scores.append(majority / len(tids))

    purity = float(np.mean(purity_scores)) if purity_scores else 1.0

    return dict(
        TP=tp, FP=fp, FN=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        entity_purity=purity,
        n_entities=len(entities),
        n_gt_pairs=len(gt_pairs),
        n_pred_pairs=len(pred_pairs),
    )


def coverage_score(
    entities: list[Any],
    tracklets: list[dict],
    track_idx_to_identity: dict[int, str],
) -> float:
    """Fraction of GT multi-camera identities that have a pure entity covering them."""
    # Which identities appear on >=2 cameras?
    id_cameras: dict[str, set[str]] = defaultdict(set)
    for tkl in tracklets:
        id_cameras[tkl["identity_id"]].add(tkl["cam_id"])
    multicam_ids = {pid for pid, cams in id_cameras.items() if len(cams) >= 2}
    if not multicam_ids:
        return 0.0

    # For each GlobalEntity, collect its identity set (by majority)
    entity_identities: list[str] = []
    for ent in entities:
        tids = [app.track_id for app in ent.appearances]
        id_counts: dict[str, int] = defaultdict(int)
        for tid in tids:
            id_counts[track_idx_to_identity.get(tid, "?")] += 1
        if not id_counts:
            continue
        majority_id = max(id_counts, key=lambda x: id_counts[x])
        purity = id_counts[majority_id] / len(tids)
        if purity >= 0.5:
            entity_identities.append(majority_id)

    covered = multicam_ids & set(entity_identities)
    return len(covered) / len(multicam_ids)


# ---------------------------------------------------------------------------
# Layer 1: run per-occasion
# ---------------------------------------------------------------------------

def run_layer1(
    tracklets: list[dict],
    crops_dir: Path,
    embedder: Any,
    topology_prior: Any,
    max_transition_sec: float = 600.0,
    min_score: float = 0.55,
    num_crops: int = 5,
) -> dict[str, Any]:
    """Full Layer 1 evaluation across all 7 occasions."""
    from video.core.schema.multi_camera import CrossCameraConfig
    from video.factory.processors.cross_camera_matcher import match_across_cameras

    config = CrossCameraConfig(
        max_transition_sec     = max_transition_sec,
        cross_camera_min_score = min_score,
        topology_weight_reid   = 0.55,
        topology_weight_topo   = 0.45,
    )

    occasions = sorted({tkl["occasion_id"] for tkl in tracklets})
    logger.info("Occasions: %s", occasions)

    all_metrics: list[dict] = []

    for occ in occasions:
        occ_tracklets = [t for t in tracklets if t["occasion_id"] == occ]
        n_ids = len({t["identity_id"] for t in occ_tracklets})
        n_cams = len({t["cam_id"] for t in occ_tracklets})
        logger.info(
            "Occasion %d: %d tracklets, %d identities, %d cameras",
            occ, len(occ_tracklets), n_ids, n_cams,
        )

        # Build GT lookup for this occasion
        gt_pairs = build_gt_pair_set(occ_tracklets)
        track_idx_to_identity = {t["track_idx"]: t["identity_id"] for t in occ_tracklets}

        if not gt_pairs:
            logger.info("  No cross-camera GT pairs in occasion %d — skipping", occ)
            continue

        # Build CameraResult objects
        camera_results = build_camera_results(
            occ_tracklets, embedder, num_crops=num_crops, crops_dir=crops_dir,
        )

        # Run matching
        entities = match_across_cameras(
            camera_results, config, embedder,
            topology_prior=topology_prior,
        )

        # Evaluate
        metrics = evaluate_entities(entities, gt_pairs, track_idx_to_identity)
        cov = coverage_score(entities, occ_tracklets, track_idx_to_identity)
        metrics["coverage"] = cov
        metrics["occasion"] = occ
        metrics["n_tracklets"] = len(occ_tracklets)
        metrics["n_identities"] = n_ids
        metrics["n_cameras"] = n_cams

        logger.info(
            "  Occasion %d => P=%.3f R=%.3f F1=%.3f Purity=%.3f Coverage=%.3f "
            "(TP=%d FP=%d FN=%d entities=%d gt_pairs=%d)",
            occ,
            metrics["precision"], metrics["recall"], metrics["f1"],
            metrics["entity_purity"], metrics["coverage"],
            metrics["TP"], metrics["FP"], metrics["FN"],
            metrics["n_entities"], metrics["n_gt_pairs"],
        )
        all_metrics.append(metrics)

    if not all_metrics:
        return {}

    # Aggregate (macro-average over occasions)
    agg: dict[str, Any] = {}
    for key in ("precision", "recall", "f1", "entity_purity", "coverage"):
        agg[key] = float(np.mean([m[key] for m in all_metrics]))
    for key in ("TP", "FP", "FN", "n_gt_pairs", "n_pred_pairs"):
        agg[key] = sum(m[key] for m in all_metrics)
    agg["per_occasion"] = all_metrics
    return agg


# ---------------------------------------------------------------------------
# Layer 2: MEVA 15-min sparse check
# ---------------------------------------------------------------------------

def run_layer2(
    pipeline_json: Path,
    tracklets: list[dict],
    t_window_start: float = 36000.0,
    t_window_end: float   = 36900.0,
) -> dict[str, Any]:
    """Align pipeline results to the 10 MEVID tracklets in our 10:00-10:15 window."""
    if not pipeline_json.exists():
        logger.warning("Pipeline JSON not found: %s", pipeline_json)
        return {"error": "pipeline_json_not_found"}

    with open(pipeline_json, encoding="utf-8") as f:
        pipeline_data = json.load(f)

    entities = pipeline_data.get("global_entities", [])

    # Filter MEVID tracklets to our window and target cameras
    our_cams = {"329", "336", "436", "509"}
    gt_window = [
        t for t in tracklets
        if t["cam_id"] in our_cams
        and t["t_start"] < t_window_end
        and t["t_end"]   > t_window_start
    ]
    logger.info("Layer 2: %d MEVID GT tracklets in 10:00-10:15 window", len(gt_window))
    for t in gt_window:
        rel_s = t["t_start"] - t_window_start
        rel_e = t["t_end"]   - t_window_start
        logger.info(
            "  GT: identity=%s cam=G%s  clip_time=%.0f-%.0f s  (abs %.0f-%.0f)",
            t["identity_id"], t["cam_id"], rel_s, rel_e,
            t["t_start"], t["t_end"],
        )

    if not entities:
        logger.info("  No cross-camera entities in pipeline output")
        return {"gt_tracklets": gt_window, "entities": [], "matches": []}

    # For each pipeline entity, list its appearances
    logger.info("Pipeline entities found: %d", len(entities))
    matches: list[dict] = []
    for ent in entities:
        apps = ent.get("appearances", [])
        cams_in_entity = [a["camera_id"] for a in apps]
        logger.info("  Entity %s: cameras=%s", ent["global_entity_id"], cams_in_entity)
        for app in apps:
            t_start = app["start_time"]
            t_end   = app["end_time"]
            cam = app["camera_id"].lstrip("G")  # "G336" -> "336"
            # Check temporal overlap with any GT tracklet on same camera
            for gt in gt_window:
                if gt["cam_id"] == cam:
                    overlap = max(0, min(t_end, gt["t_end"]) - max(t_start, gt["t_start"]))
                    union   = max(t_end, gt["t_end"]) - min(t_start, gt["t_start"])
                    iou     = overlap / union if union > 0 else 0.0
                    if iou > 0.1:
                        matches.append(dict(
                            entity_id=ent["global_entity_id"],
                            camera=app["camera_id"],
                            pipeline_t=f"{t_start:.0f}-{t_end:.0f}",
                            gt_identity=gt["identity_id"],
                            gt_t=f"{gt['t_start']:.0f}-{gt['t_end']:.0f}",
                            iou=round(iou, 3),
                        ))

    for m in matches:
        logger.info(
            "  MATCH: entity=%s cam=%s pipeline=[%s] GT identity=%s gt=[%s] IoU=%.3f",
            m["entity_id"], m["camera"], m["pipeline_t"],
            m["gt_identity"], m["gt_t"], m["iou"],
        )

    return {
        "gt_tracklets_in_window": len(gt_window),
        "pipeline_entities": len(entities),
        "temporal_matches": matches,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Multi-camera entity eval vs MEVID GT")
    p.add_argument("--annot-zip",
                   default="_data/mevid-v1-annotation-data.zip")
    p.add_argument("--crops-dir",
                   default="_data/mevid-crops")
    p.add_argument("--topology-prior",
                   default="results/mevid_topology.json")
    p.add_argument("--pipeline-json",
                   default="results/meva_multicam_1clip.json",
                   help="Existing MEVA end-to-end pipeline output for Layer 2")
    p.add_argument("--max-transition-sec", type=float, default=600.0,
                   help="Time-gap limit for cross-camera candidate pairs")
    p.add_argument("--min-score", type=float, default=0.55,
                   help="Minimum combined score to accept a cross-camera assignment")
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--num-crops", type=int, default=5)
    p.add_argument("--out", default="results/eval_multicam_entity.json")
    args = p.parse_args()

    annot_zip    = Path(args.annot_zip)
    crops_dir    = Path(args.crops_dir)
    topo_path    = args.topology_prior
    pipeline_json = Path(args.pipeline_json)
    out_path     = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Load embedder + topology prior                                       #
    # ------------------------------------------------------------------ #
    from video.core.models.reid_embedder import ReIDEmbedder
    from video.core.models.camera_topology import CameraTopologyPrior

    logger.info("Loading ReIDEmbedder (device=%s)...", args.device)
    embedder = ReIDEmbedder(device=args.device)
    logger.info("Embedder: %s", embedder._backend)

    topology_prior = None
    if topo_path and Path(topo_path).exists():
        topology_prior = CameraTopologyPrior.load(topo_path)
        logger.info("Loaded topology prior: %s", topology_prior)

    # ------------------------------------------------------------------ #
    # Load MEVID tracklets                                                 #
    # ------------------------------------------------------------------ #
    tracklets = load_mevid_tracklets(annot_zip)

    # ------------------------------------------------------------------ #
    # Layer 1: per-occasion entity evaluation                              #
    # ------------------------------------------------------------------ #
    logger.info("=" * 60)
    logger.info("LAYER 1: MEVID full-corpus entity evaluation")
    logger.info("=" * 60)
    layer1 = run_layer1(
        tracklets, crops_dir, embedder, topology_prior,
        max_transition_sec=args.max_transition_sec,
        min_score=args.min_score,
        num_crops=args.num_crops,
    )

    # ------------------------------------------------------------------ #
    # Layer 2: MEVA sparse end-to-end check                               #
    # ------------------------------------------------------------------ #
    logger.info("=" * 60)
    logger.info("LAYER 2: MEVA 15-min end-to-end sparse check")
    logger.info("=" * 60)
    layer2 = run_layer2(pipeline_json, tracklets)

    # ------------------------------------------------------------------ #
    # Print summary                                                        #
    # ------------------------------------------------------------------ #
    print()
    print("=" * 65)
    print("Multi-Camera Entity Evaluation Summary")
    print("=" * 65)
    print()
    print("LAYER 1 — MEVID Full-Corpus Entity Matching")
    print("-" * 65)
    if layer1:
        print(f"  Occasions evaluated : {len(layer1.get('per_occasion', []))}")
        print(f"  GT cross-cam pairs  : {layer1.get('n_gt_pairs', 0)}")
        print(f"  Predicted pairs     : {layer1.get('n_pred_pairs', 0)}")
        print(f"  TP / FP / FN        : {layer1['TP']} / {layer1['FP']} / {layer1['FN']}")
        print(f"  Precision           : {layer1['precision']:.4f}")
        print(f"  Recall              : {layer1['recall']:.4f}")
        print(f"  F1                  : {layer1['f1']:.4f}")
        print(f"  Entity Purity       : {layer1['entity_purity']:.4f}")
        print(f"  Coverage            : {layer1['coverage']:.4f}")
        print()
        print("  Per-occasion breakdown:")
        for m in layer1.get("per_occasion", []):
            print(
                f"    Occ {m['occasion']:d}  "
                f"P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}  "
                f"Purity={m['entity_purity']:.3f}  "
                f"entities={m['n_entities']}  gt_pairs={m['n_gt_pairs']}"
            )
    print()
    print("LAYER 2 — MEVA 10:00-10:15 End-to-End Sparse Check")
    print("-" * 65)
    print(f"  GT tracklets in window : {layer2.get('gt_tracklets_in_window', 0)}")
    print(f"  Pipeline entities      : {layer2.get('pipeline_entities', 0)}")
    matches = layer2.get("temporal_matches", [])
    print(f"  Temporal matches (IoU>0.1): {len(matches)}")
    for m in matches:
        print(f"    entity={m['entity_id']}  cam={m['camera']}  "
              f"GT_id={m['gt_identity']}  IoU={m['iou']:.3f}")
    print("=" * 65)

    # ------------------------------------------------------------------ #
    # Save results                                                         #
    # ------------------------------------------------------------------ #
    result = {"layer1": layer1, "layer2": layer2}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
