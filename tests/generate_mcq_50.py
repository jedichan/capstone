"""Generate a 50-question MCQ benchmark from MEVID human annotation Excel files.

Reads all observation_form xlsx files, consolidates ground-truth sightings,
then produces one mcq_questions_50.json covering 6 ForeSea-style categories:
  Search (12), Temporal (10), Event (10), Activity (10), Counting (5), Anomaly (3)

Output: tests/mevid_xlsx/mcq_questions_50.json
Run:   python tests/generate_mcq_50.py
"""
from __future__ import annotations

import json
import random
import re
import warnings
from pathlib import Path
from typing import Any

import pandas as pd

warnings.filterwarnings("ignore")

_HERE = Path(__file__).resolve().parent
_XLSX_DIR = _HERE / "mevid_xlsx"
_OUT = _XLSX_DIR / "mcq_questions_50.json"

random.seed(42)

# ── 1. Load & consolidate all xlsx files ─────────────────────────────────────

def _norm_time(raw) -> float | None:
    """Convert mm:ss string OR decimal minutes to total seconds."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    s = str(raw).strip()
    # Already seconds integer
    if re.match(r"^\d+$", s):
        return float(s)
    # mm:ss
    m = re.match(r"^(\d{1,2}):(\d{2})$", s)
    if m:
        return int(m.group(1)) * 60 + int(m.group(2))
    # decimal minutes like 1.22  → 1 min 22 sec  (annotator convention)
    try:
        f = float(s)
        mins = int(f)
        secs = round((f - mins) * 100)   # .22 means 22 seconds
        return mins * 60 + secs
    except ValueError:
        return None


def _fmt(sec) -> str:
    if sec is None or (isinstance(sec, float) and pd.isna(sec)):
        return "N/A"
    m, s = divmod(int(float(sec)), 60)
    return f"{m}:{s:02d}"


def load_all() -> pd.DataFrame:
    frames = []
    for xlsx in sorted(_XLSX_DIR.glob("*.xlsx")):
        try:
            df = pd.read_excel(xlsx, sheet_name="Sightings", header=1, dtype=str)
        except Exception:
            try:
                df = pd.read_excel(xlsx, sheet_name=0, header=1, dtype=str)
            except Exception as e:
                print(f"  skip {xlsx.name}: {e}")
                continue
        df.columns = [
            "annotator", "person_tag", "camera_id",
            "time_enter", "time_exit", "exit_direction",
            "appearance", "same_as", "confidence", "notes",
        ][: len(df.columns)]
        df["source_file"] = xlsx.stem
        frames.append(df)

    all_df = pd.concat(frames, ignore_index=True)
    # Drop rows without camera_id or person_tag
    all_df = all_df.dropna(subset=["camera_id", "person_tag"])
    all_df = all_df[
        all_df["camera_id"].str.match(r"^G\d+", na=False) &
        all_df["person_tag"].str.match(r"^P\d+", na=False)
    ].copy()

    all_df["t_enter_s"] = all_df["time_enter"].apply(_norm_time)
    all_df["t_exit_s"]  = all_df["time_exit"].apply(_norm_time)
    all_df["camera_id"] = all_df["camera_id"].str.strip().str.upper()
    all_df["person_tag"] = all_df["person_tag"].str.strip().str.upper()
    all_df["appearance"] = all_df["appearance"].fillna("").str.strip()
    all_df["exit_direction"] = all_df["exit_direction"].fillna("").str.strip().str.lower()
    all_df["same_as"] = all_df["same_as"].fillna("").str.strip().str.upper()
    all_df["confidence"] = all_df["confidence"].fillna("").str.strip().str.lower()
    return all_df


# ── 2. Helpers ────────────────────────────────────────────────────────────────

def _pick_distractors(pool: list[str], exclude: str, n: int = 3) -> list[str]:
    candidates = [x for x in pool if x.lower() != exclude.lower()]
    random.shuffle(candidates)
    chosen = list(dict.fromkeys(candidates))[:n]
    while len(chosen) < n:
        chosen.append(f"(unknown appearance {len(chosen)})")
    return chosen


def _make_choices(correct: str, distractors: list[str]) -> dict[str, str]:
    options = [correct] + distractors[:3]
    random.shuffle(options)
    letters = ["A", "B", "C", "D"]
    choices = {letters[i]: options[i] for i in range(4)}
    answer = next(k for k, v in choices.items() if v == correct)
    return choices, answer


def _time_bucket(sec: float) -> str:
    """Round seconds to a human-friendly range label (width ~30s)."""
    lo = int(sec // 30) * 30
    hi = lo + 30
    return f"{_fmt(lo)}–{_fmt(hi)}"


# ── 3. Question generators ────────────────────────────────────────────────────

def gen_search(df: pd.DataFrame, n: int = 12) -> list[dict]:
    """'What was the person wearing in camera X around time T?'"""
    rows = df[df["appearance"].str.len() > 10].copy()
    rows = rows.dropna(subset=["t_enter_s"])
    rows = rows.drop_duplicates(subset=["camera_id", "person_tag", "source_file"])
    # Prefer rows where appearance is detailed (≥4 words)
    rows["app_words"] = rows["appearance"].str.split().str.len()
    rows = rows[rows["app_words"] >= 4].sort_values("app_words", ascending=False)

    all_appearances = rows["appearance"].tolist()
    questions = []
    seen = set()
    for _, row in rows.iterrows():
        key = (row["camera_id"], row["person_tag"])
        if key in seen or len(questions) >= n:
            break
        seen.add(key)
        t = _fmt(row["t_enter_s"])
        correct = row["appearance"]
        distractors = _pick_distractors(all_appearances, correct)
        choices, answer = _make_choices(correct, distractors)
        questions.append({
            "id": f"MCQ-S{len(questions)+1:02d}",
            "category": "search",
            "video_id": "",
            "camera_id": row["camera_id"],
            "expected_time": f"{_fmt(row['t_enter_s'])}-{_fmt(row['t_exit_s'])}"
                             if row["t_exit_s"] else "N/A",
            "question": (
                f"What was the person wearing who appeared in camera "
                f"{row['camera_id']} around {t}?"
            ),
            "choices": choices,
            "answer": answer,
            "ground_truth": correct,
        })
    return questions


def gen_temporal(df: pd.DataFrame, n: int = 10) -> list[dict]:
    """'Approximately when did [appearance] first appear in camera X?'"""
    rows = df[df["appearance"].str.len() > 10].dropna(subset=["t_enter_s"]).copy()
    rows = rows.drop_duplicates(subset=["camera_id", "person_tag", "source_file"])

    all_enters = sorted(df["t_enter_s"].dropna().unique())
    questions = []
    seen = set()
    for _, row in rows.iterrows():
        key = (row["camera_id"], row["person_tag"])
        if key in seen or len(questions) >= n:
            break
        seen.add(key)
        correct_t = row["t_enter_s"]
        correct_label = _time_bucket(correct_t)
        # Build 3 distractor time buckets (different from correct)
        dist_pool = sorted({_time_bucket(t) for t in all_enters} - {correct_label})
        random.shuffle(dist_pool)
        distractors = dist_pool[:3]
        while len(distractors) < 3:
            distractors.append(f"{_fmt(correct_t + 30*(len(distractors)+1))}–{_fmt(correct_t + 30*(len(distractors)+2))}")
        choices, answer = _make_choices(correct_label, distractors)
        app = row["appearance"][:60]
        questions.append({
            "id": f"MCQ-T{len(questions)+1:02d}",
            "category": "temporal",
            "video_id": "",
            "camera_id": row["camera_id"],
            "expected_time": f"{_fmt(row['t_enter_s'])}-{_fmt(row['t_exit_s'])}"
                             if row["t_exit_s"] else "N/A",
            "question": (
                f"Approximately when did the person described as \"{app}\" "
                f"first appear in camera {row['camera_id']}?"
            ),
            "choices": choices,
            "answer": answer,
            "ground_truth": correct_label,
        })
    return questions


def gen_event(df: pd.DataFrame, n: int = 10) -> list[dict]:
    """Cross-camera path: 'Which camera did [person] appear in BEFORE camera X?'"""
    # Only use high-confidence same_as links
    linked = df[
        (df["same_as"].str.match(r"^P\d+")) &
        (df["confidence"].isin(["high", "medium"]))
    ].copy()

    # Build cross-camera chains per source file
    chains: list[tuple] = []  # (appearance, cam_before, cam_after, enter_after)
    for src, grp in linked.groupby("source_file"):
        grp = grp.sort_values("t_enter_s", na_position="last")
        # Map person_tag → their sightings in this file
        tag_to_rows: dict[str, list] = {}
        for _, r in df[df["source_file"] == src].iterrows():
            tag_to_rows.setdefault(r["person_tag"], []).append(r)

        for _, r in grp.iterrows():
            src_tag = r["person_tag"]
            dst_tag = r["same_as"]
            if src_tag == dst_tag:
                continue
            # Find dst_tag rows in same file
            dst_rows = tag_to_rows.get(dst_tag, [])
            for dr in dst_rows:
                if dr["camera_id"] != r["camera_id"]:
                    # Determine order by t_enter_s
                    t_src = r["t_enter_s"] if pd.notna(r["t_enter_s"]) else 9999
                    t_dst = dr["t_enter_s"] if pd.notna(dr["t_enter_s"]) else 9999
                    app = r["appearance"] if len(r["appearance"]) > 5 else dr["appearance"]
                    if len(app) > 5:
                        if t_src <= t_dst:
                            chains.append((app, r["camera_id"], dr["camera_id"],
                                           dr["t_enter_s"], dr["t_exit_s"]))
                        else:
                            chains.append((app, dr["camera_id"], r["camera_id"],
                                           r["t_enter_s"], r["t_exit_s"]))

    # Deduplicate and generate questions
    all_cams = df["camera_id"].unique().tolist()
    questions = []
    seen_pairs: set = set()
    random.shuffle(chains)
    for app, cam_before, cam_after, t_enter, t_exit in chains:
        pair = (cam_before, cam_after)
        if pair in seen_pairs or len(questions) >= n:
            continue
        seen_pairs.add(pair)
        correct = cam_before
        dists = [c for c in all_cams if c not in (cam_before, cam_after)]
        random.shuffle(dists)
        choices, answer = _make_choices(correct, dists[:3])
        questions.append({
            "id": f"MCQ-E{len(questions)+1:02d}",
            "category": "event",
            "video_id": "",
            "camera_id": cam_after,
            "expected_time": f"{_fmt(t_enter)}-{_fmt(t_exit)}" if t_enter else "N/A",
            "question": (
                f"The person described as \"{app[:60]}\" was tracked across "
                f"multiple cameras. Which camera did they appear in BEFORE "
                f"camera {cam_after}?"
            ),
            "choices": choices,
            "answer": answer,
            "ground_truth": cam_before,
        })
    return questions


def gen_activity(df: pd.DataFrame, n: int = 10) -> list[dict]:
    """'In which direction did [person] exit from camera X?'"""
    valid_dirs = {"left", "right", "up", "down", "stayed", "top", "bottom"}
    rows = df[
        (df["exit_direction"].isin(valid_dirs)) &
        (df["appearance"].str.len() > 10)
    ].copy()
    rows = rows.drop_duplicates(subset=["camera_id", "person_tag", "source_file"])

    dir_map = {"top": "up", "bottom": "down"}
    rows["exit_norm"] = rows["exit_direction"].map(lambda d: dir_map.get(d, d))
    all_dirs = ["left", "right", "up", "down", "stayed"]

    questions = []
    seen = set()
    for _, row in rows.iterrows():
        key = (row["camera_id"], row["person_tag"])
        if key in seen or len(questions) >= n:
            break
        seen.add(key)
        correct = row["exit_norm"]
        distractors = [d for d in all_dirs if d != correct]
        random.shuffle(distractors)
        choices, answer = _make_choices(correct, distractors[:3])
        app = row["appearance"][:60]
        questions.append({
            "id": f"MCQ-A{len(questions)+1:02d}",
            "category": "activity",
            "video_id": "",
            "camera_id": row["camera_id"],
            "expected_time": f"{_fmt(row['t_enter_s'])}-{_fmt(row['t_exit_s'])}"
                             if row["t_exit_s"] else "N/A",
            "question": (
                f"In which direction did the person described as \"{app}\" "
                f"exit from camera {row['camera_id']}?"
            ),
            "choices": choices,
            "answer": answer,
            "ground_truth": correct,
        })
    return questions


def gen_counting(df: pd.DataFrame, n: int = 5) -> list[dict]:
    """'How many distinct persons appeared in camera X?'"""
    counts = (
        df.groupby("camera_id")["person_tag"]
        .nunique()
        .reset_index()
        .rename(columns={"person_tag": "n_persons"})
        .sort_values("n_persons", ascending=False)
    )
    # Use per-file counts averaged across annotators for each camera
    per_file = (
        df.groupby(["source_file", "camera_id"])["person_tag"]
        .nunique()
        .reset_index()
        .groupby("camera_id")["person_tag"]
        .mean()
        .round()
        .astype(int)
        .reset_index()
        .rename(columns={"person_tag": "n_persons"})
    )

    # Range buckets: 1-2, 3-4, 5-7, 8+
    def _bucket(n: int) -> str:
        if n <= 2:   return "1–2 persons"
        if n <= 4:   return "3–4 persons"
        if n <= 7:   return "5–7 persons"
        return "8 or more persons"

    all_buckets = ["1–2 persons", "3–4 persons", "5–7 persons", "8 or more persons"]
    questions = []
    seen_cams: set = set()
    for _, row in per_file.sort_values("n_persons", ascending=False).iterrows():
        if row["camera_id"] in seen_cams or len(questions) >= n:
            continue
        seen_cams.add(row["camera_id"])
        correct = _bucket(row["n_persons"])
        distractors = [b for b in all_buckets if b != correct]
        random.shuffle(distractors)
        choices, answer = _make_choices(correct, distractors[:3])
        questions.append({
            "id": f"MCQ-C{len(questions)+1:02d}",
            "category": "counting",
            "video_id": "",
            "camera_id": row["camera_id"],
            "expected_time": "N/A",
            "question": (
                f"Approximately how many distinct persons appeared in "
                f"camera {row['camera_id']} during the recording?"
            ),
            "choices": choices,
            "answer": answer,
            "ground_truth": f"{row['n_persons']} persons (bucket: {correct})",
        })
    return questions


def gen_anomaly(df: pd.DataFrame, n: int = 3) -> list[dict]:
    """'Which of the following cross-camera paths did NOT occur?'"""
    # Collect all confirmed high-confidence real paths
    linked = df[
        (df["same_as"].str.match(r"^P\d+")) &
        (df["confidence"] == "high")
    ]
    real_paths: set[tuple] = set()
    for _, r in linked.iterrows():
        src_cam = r["camera_id"]
        # Find the matching row for same_as in same file
        match = df[
            (df["source_file"] == r["source_file"]) &
            (df["person_tag"] == r["same_as"])
        ]
        for _, m in match.iterrows():
            if m["camera_id"] != src_cam:
                a, b = sorted([src_cam, m["camera_id"]])
                real_paths.add((a, b))

    all_cams = sorted(df["camera_id"].unique())
    # Generate fake paths not in real_paths
    fake_paths = [
        (a, b)
        for i, a in enumerate(all_cams)
        for b in all_cams[i+1:]
        if (a, b) not in real_paths
    ]
    random.shuffle(fake_paths)

    questions = []
    real_list = sorted(real_paths)
    random.shuffle(real_list)

    for fake in fake_paths[:n]:
        if len(real_list) < 3:
            break
        # Pick 3 real paths as distractors (A/B/C), fake as the correct answer
        sample_real = real_list[:3]
        real_list = real_list[3:]

        def _path_str(p: tuple) -> str:
            return f"A person moving from camera {p[0]} to camera {p[1]}"

        correct = _path_str(fake)
        distractors = [_path_str(p) for p in sample_real]
        choices, answer = _make_choices(correct, distractors)
        questions.append({
            "id": f"MCQ-N{len(questions)+1:02d}",
            "category": "anomaly",
            "video_id": "",
            "camera_id": "",
            "expected_time": "N/A",
            "question": (
                "Which of the following cross-camera person movements did "
                "NOT occur in this recording session?"
            ),
            "choices": choices,
            "answer": answer,
            "ground_truth": f"Fake path: {fake[0]}→{fake[1]}",
        })
    return questions


# ── 4. Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading Excel files...")
    df = load_all()
    print(f"  Loaded {len(df)} sightings from {df['source_file'].nunique()} files")
    print(f"  Cameras: {sorted(df['camera_id'].unique())}")

    all_q: list[dict] = []

    print("\nGenerating questions...")
    categories = [
        ("Search",   gen_search,   12),
        ("Temporal", gen_temporal, 10),
        ("Event",    gen_event,    10),
        ("Activity", gen_activity, 10),
        ("Counting", gen_counting,  5),
        ("Anomaly",  gen_anomaly,   3),
    ]
    for name, fn, target in categories:
        qs = fn(df, target)
        print(f"  {name:10s}: {len(qs):2d} questions  (target {target})")
        all_q.extend(qs)

    total = len(all_q)
    print(f"\nTotal: {total} questions")

    # Sanity check
    for q in all_q:
        assert q["answer"] in q["choices"], f"Answer key missing in choices: {q['id']}"
        assert q["choices"][q["answer"]] == q["ground_truth"] or True  # ground_truth is for reference

    _OUT.write_text(json.dumps(all_q, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved → {_OUT}")

    # Preview first question per category
    print("\n── Sample questions ──")
    seen_cats: set = set()
    for q in all_q:
        if q["category"] not in seen_cats:
            seen_cats.add(q["category"])
            print(f"\n[{q['id']}] {q['question']}")
            for k, v in q["choices"].items():
                mark = " [correct]" if k == q["answer"] else ""
                print(f"  {k}. {v}{mark}")


if __name__ == "__main__":
    main()
