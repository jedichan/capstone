"""Unit tests and demo for CameraTopologyPrior.

Run without any dataset:
    python tests/test_camera_topology.py

Tests:
  1. Cold-start prior: linear decay behaviour
  2. Online learning: GMM fitting after N observations
  3. Score monotonicity within a learned distribution
  4. Persistence (save/load round-trip)
  5. Directed asymmetry demo: cam1→cam2 vs cam2→cam1
  6. Integration with score_candidate_pairs (mock embeddings)
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import numpy as np

from video.core.models.camera_topology import (
    CAMERA_IDS_CAMPUS_GUIDE_V2,
    CameraTopologyPrior,
    campus_zone_of_camera,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_prior(cameras=None, max_transit_sec=300.0):
    cameras = cameras or ["cam1", "cam2", "cam3", "cam4"]
    return CameraTopologyPrior(cameras=cameras, max_transit_sec=max_transit_sec)


def _assert_close(a: float, b: float, tol: float = 0.05, label: str = ""):
    assert abs(a - b) <= tol, f"{label}: expected ~{b:.4f}, got {a:.4f}"


# ---------------------------------------------------------------------------
# Test 1: Cold-start prior
# ---------------------------------------------------------------------------

def test_cold_start_prior():
    prior = _make_prior()
    # Within 30 s → full score
    assert prior.score("cam1", "cam2", 0.0) == 1.0
    assert prior.score("cam1", "cam2", 30.0) == 1.0
    # At max_transit_sec → 0
    assert prior.score("cam1", "cam2", 300.0) == 0.0
    # Beyond max → 0
    assert prior.score("cam1", "cam2", 400.0) == 0.0
    # Linearly decreasing in (30, 300)
    s60 = prior.score("cam1", "cam2", 60.0)
    s180 = prior.score("cam1", "cam2", 180.0)
    assert s60 > s180 > 0.0, f"Expected decay: {s60:.3f} > {s180:.3f}"
    print("[PASS] test_cold_start_prior")


# ---------------------------------------------------------------------------
# Test 2: Online learning / GMM fitting
# ---------------------------------------------------------------------------

def test_gmm_fitting():
    prior = _make_prior()
    # Inject 20 observations centred on 45 s ± 5 s
    rng = np.random.default_rng(0)
    obs = rng.normal(loc=45.0, scale=5.0, size=20).clip(0, 300).tolist()
    for dt in obs:
        prior.observe("cam1", "cam2", dt)

    assert ("cam1", "cam2") in prior._fitted, "GMM should be fitted after 20 obs"

    # Score near mean should be high
    s_near = prior.score("cam1", "cam2", 45.0)
    # Score far from mean should be lower
    s_far = prior.score("cam1", "cam2", 200.0)
    assert s_near > s_far, f"Near={s_near:.3f} should > Far={s_far:.3f}"
    assert s_near > 0.5, f"Peak score should be > 0.5, got {s_near:.3f}"

    print(f"[PASS] test_gmm_fitting  (score@45s={s_near:.3f}, score@200s={s_far:.3f})")


# ---------------------------------------------------------------------------
# Test 3: Directed asymmetry
# ---------------------------------------------------------------------------

def test_directed_asymmetry():
    """cam1→cam2 ≠ cam2→cam1 when learned distributions differ."""
    prior = _make_prior()
    rng = np.random.default_rng(1)

    # cam1→cam2: fast corridor, ~30 s
    for dt in rng.normal(30.0, 3.0, 15).clip(0, 300).tolist():
        prior.observe("cam1", "cam2", dt)

    # cam2→cam1: long detour, ~120 s
    for dt in rng.normal(120.0, 10.0, 15).clip(0, 300).tolist():
        prior.observe("cam2", "cam1", dt)

    s_12_at_30 = prior.score("cam1", "cam2", 30.0)
    s_21_at_30 = prior.score("cam2", "cam1", 30.0)
    s_21_at_120 = prior.score("cam2", "cam1", 120.0)

    assert s_12_at_30 > s_21_at_30, (
        f"cam1→cam2 @30s ({s_12_at_30:.3f}) should beat cam2→cam1 @30s ({s_21_at_30:.3f})"
    )
    assert s_21_at_120 > s_21_at_30, (
        f"cam2→cam1 @120s ({s_21_at_120:.3f}) should beat @30s ({s_21_at_30:.3f})"
    )
    print(
        f"[PASS] test_directed_asymmetry  "
        f"(cam1→cam2@30s={s_12_at_30:.3f}  cam2→cam1@30s={s_21_at_30:.3f}  cam2→cam1@120s={s_21_at_120:.3f})"
    )


# ---------------------------------------------------------------------------
# Test 4: Persistence
# ---------------------------------------------------------------------------

def test_persistence():
    prior = _make_prior()
    rng = np.random.default_rng(2)
    for dt in rng.normal(60.0, 8.0, 10).clip(0, 300).tolist():
        prior.observe("cam1", "cam3", dt)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        tmp = Path(tf.name)

    try:
        prior.save(tmp)
        loaded = CameraTopologyPrior.load(tmp)

        assert loaded.cameras == prior.cameras
        orig_score = prior.score("cam1", "cam3", 60.0)
        loaded_score = loaded.score("cam1", "cam3", 60.0)
        _assert_close(loaded_score, orig_score, tol=0.05, label="save/load score")
        print(f"[PASS] test_persistence  (orig={orig_score:.3f}, loaded={loaded_score:.3f})")
    finally:
        tmp.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Test 5: transition_table output structure
# ---------------------------------------------------------------------------

def test_transition_table():
    prior = _make_prior(cameras=["A", "B", "C"])
    for dt in [10, 15, 20, 25, 30]:
        prior.observe("A", "B", float(dt))

    table = prior.transition_table()
    assert "A" in table
    assert "B" in table["A"]
    entry = table["A"]["B"]
    assert entry["n_observations"] == 5
    assert entry["mean_transit_sec"] is not None
    _assert_close(entry["mean_transit_sec"], 20.0, tol=1.0, label="mean_transit")
    print(f"[PASS] test_transition_table  (mean={entry['mean_transit_sec']:.1f}s)")


# ---------------------------------------------------------------------------
# Test 6: Integration — topology score improves a marginal cosine match
# ---------------------------------------------------------------------------

def test_topology_improves_score():
    """Simulate two candidate pairs with equal cosine similarity.

    Pair A: cam1→cam2, Δt=30s  (within learned transit window)
    Pair B: cam1→cam3, Δt=250s (far outside any window)

    With topology prior, Pair A should score higher.
    Without prior, both would score equally (same cosine).
    """
    prior = _make_prior()
    rng = np.random.default_rng(3)

    # Teach prior: cam1→cam2 transit ~30 s
    for dt in rng.normal(30.0, 5.0, 20).clip(0, 300).tolist():
        prior.observe("cam1", "cam2", dt)

    cosine = 0.72  # same for both
    score_a = 0.55 * cosine + 0.45 * prior.score("cam1", "cam2", 30.0)
    score_b = 0.55 * cosine + 0.45 * prior.score("cam1", "cam3", 250.0)

    assert score_a > score_b, (
        f"Topology should prefer Pair A ({score_a:.3f}) over Pair B ({score_b:.3f})"
    )
    print(
        f"[PASS] test_topology_improves_score  "
        f"(PairA={score_a:.3f}  PairB={score_b:.3f})"
    )


# ---------------------------------------------------------------------------
# Test 7: from_confirmed_matches factory
# ---------------------------------------------------------------------------

def test_from_confirmed_matches():
    appearances = [
        [
            {"camera_id": "cam1", "start_time": 0.0, "end_time": 10.0},
            {"camera_id": "cam2", "start_time": 45.0, "end_time": 55.0},
        ],
        [
            {"camera_id": "cam1", "start_time": 20.0, "end_time": 30.0},
            {"camera_id": "cam2", "start_time": 68.0, "end_time": 75.0},
        ],
        [
            {"camera_id": "cam2", "start_time": 5.0, "end_time": 12.0},
            {"camera_id": "cam3", "start_time": 40.0, "end_time": 50.0},
        ],
    ]
    prior = CameraTopologyPrior.from_confirmed_matches(
        appearances,
        cameras=["cam1", "cam2", "cam3"],
        max_transit_sec=300.0,
    )
    # cam1→cam2 should have 2 observations (~35s, ~38s)
    assert len(prior._obs.get(("cam1", "cam2"), [])) == 2
    assert len(prior._obs.get(("cam2", "cam3"), [])) == 1
    print("[PASS] test_from_confirmed_matches")


# ---------------------------------------------------------------------------
# Demo printout
# ---------------------------------------------------------------------------

def demo_topology_report():
    """Print a human-readable topology report for a synthetic 4-camera scenario."""
    cameras = ["entrance", "corridor_A", "corridor_B", "exit"]
    prior = CameraTopologyPrior(cameras=cameras, max_transit_sec=600.0)

    rng = np.random.default_rng(42)

    # entrance → corridor_A: quick, ~20 s
    for dt in rng.normal(20.0, 3.0, 30).clip(0, 600).tolist():
        prior.observe("entrance", "corridor_A", dt)

    # corridor_A → corridor_B: medium, ~45 s
    for dt in rng.normal(45.0, 8.0, 25).clip(0, 600).tolist():
        prior.observe("corridor_A", "corridor_B", dt)

    # corridor_B → exit: fast, ~15 s
    for dt in rng.normal(15.0, 4.0, 20).clip(0, 600).tolist():
        prior.observe("corridor_B", "exit", dt)

    # entrance → exit directly (rarely): ~300 s
    for dt in rng.normal(300.0, 30.0, 8).clip(0, 600).tolist():
        prior.observe("entrance", "exit", dt)

    print("\n" + "=" * 65)
    print("Camera Topology Prior - Demo Report")
    print("=" * 65)
    print(repr(prior))
    print()

    table = prior.transition_table()
    for cam_a, targets in table.items():
        for cam_b, stats in targets.items():
            if stats["n_observations"] == 0:
                continue
            mean_t = stats.get("mean_transit_sec")
            std_t = stats.get("std_transit_sec")
            fitted = "[GMM]" if stats["fitted"] else "[flat]"
            if mean_t is not None:
                print(
                    f"  {cam_a:15s} -> {cam_b:15s}  "
                    f"n={stats['n_observations']:3d}  "
                    f"mean={mean_t:6.1f}s  std={std_t:5.1f}s  {fitted}"
                )
            else:
                print(f"  {cam_a:15s} -> {cam_b:15s}  n=0")

    print()
    print("Score samples (entrance → corridor_A):")
    for dt in [5, 15, 20, 25, 45, 100, 300]:
        s = prior.score("entrance", "corridor_A", dt)
        bar = "#" * int(s * 30)
        print(f"  dt={dt:4d}s  score={s:.3f}  {bar}")

    print("=" * 65)


# ---------------------------------------------------------------------------
# Campus zone guide v2.0 seed
# ---------------------------------------------------------------------------

def test_campus_guide_v2_seed():
    """Synthetic campus map yields fitted pairs and sane ADMIN↔BUS scores."""
    prior = CameraTopologyPrior.from_campus_guide_v2(
        cameras=["G326", "G329", "G505"],
        rng=np.random.default_rng(0),
        samples_per_pair=8,
    )
    assert ("G326", "G505") in prior._fitted or ("G505", "G326") in prior._fitted
    # Mid-range ADMIN→BUS walk (~37 s) should beat nonsense gaps for fitted pairs
    s_mid = prior.score("G326", "G505", 37.0)
    s_bad = prior.score("G326", "G505", 280.0)
    assert s_mid > s_bad, f"mid={s_mid:.3f} should beat tail={s_bad:.3f}"

    full = CameraTopologyPrior.from_campus_guide_v2()
    assert len(full.cameras) == len(CAMERA_IDS_CAMPUS_GUIDE_V2)
    assert campus_zone_of_camera("G328") == "school"
    assert campus_zone_of_camera("G326") == "admin"

    alias = CameraTopologyPrior.from_campus_guide_v2(
        cameras=["cam_a", "cam_b"],
        logical_to_physical={"cam_a": "G326", "cam_b": "G505"},
        rng=np.random.default_rng(1),
    )
    assert alias.score("cam_a", "cam_b", 35.0) > 0.0
    print("[PASS] test_campus_guide_v2_seed")


# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_cold_start_prior()
    test_gmm_fitting()
    test_directed_asymmetry()
    test_persistence()
    test_transition_table()
    test_topology_improves_score()
    test_from_confirmed_matches()
    test_campus_guide_v2_seed()
    demo_topology_report()
    print("\nAll topology tests passed [OK]")
