"""
generate_queries_from_observations.py
--------------------------------------
Reads observation_form.xlsx (Sightings sheet) and automatically generates
agent_test.xlsx (Part1 sheet) in the format expected by agent_test_importer.py.

Query types generated:
  1. Existence      — "Is there a person [appearance] in camera [X]?"
  2. Appearance     — "Is there a person wearing [top] in camera [X]?"
  3. Direction      — "Did a person enter/exit from the [direction] in camera [X]?"
  4. Cross-camera   — "Did a person [appearance] appear in both [camA] and [camB]?"
  5. Negative (FP)  — Pairs from different sightings that have NO confirmed match

Usage:
    python scripts/generate_queries_from_observations.py \
        --obs agent/test/data/observation_form.xlsx \
        --out agent/test/data/agent_test.xlsx
"""

import argparse
import random
from pathlib import Path
from itertools import combinations
from collections import defaultdict

import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

# ── CLI ───────────────────────────────────────────────────────────────────────
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--obs", default="agent/test/data/observation_form.xlsx")
    p.add_argument("--out", default="agent/test/data/agent_test.xlsx")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-negatives", type=int, default=5,
                   help="Max negative (FP) cross-camera pairs to generate")
    return p.parse_args()


# ── Load sightings ─────────────────────────────────────────────────────────────
def load_sightings(xlsx_path: str) -> list[dict]:
    wb = openpyxl.load_workbook(xlsx_path, data_only=True)
    ws = wb["Sightings"]
    rows = list(ws.iter_rows(values_only=True))

    # Find header row (row with 'person_tag')
    header_row = None
    for i, row in enumerate(rows):
        if any(str(v or "").strip().lower() == "person_tag" for v in row):
            header_row = i
            break
    if header_row is None:
        raise ValueError("Could not find header row with 'person_tag' in Sightings sheet")

    headers = [str(v or "").strip().lower() for v in rows[header_row]]

    def get(row, col):
        try:
            idx = headers.index(col)
            val = row[idx]
            return str(val).strip() if val is not None else ""
        except (ValueError, IndexError):
            return ""

    sightings = []
    for row in rows[header_row + 1:]:
        if not any(v for v in row):
            continue
        tag = get(row, "person_tag")
        cam = get(row, "camera_id")
        if not tag or not cam:
            continue
        sightings.append({
            "annotator":       get(row, "annotator"),
            "person_tag":      tag,
            "camera_id":       cam,
            "time_enter":      get(row, "time_enter"),
            "time_exit":       get(row, "time_exit"),
            "exit_direction":  get(row, "exit_direction"),
            "appearance":      get(row, "appearance"),
            "same_as":         get(row, "same_as"),
            "confidence":      get(row, "confidence") or "high",
            "notes":           get(row, "notes"),
        })
    return sightings


# ── Derive video_id from camera_id ─────────────────────────────────────────────
# Maps camera_id → video_id (filename without .avi)
CAMERA_VIDEO_MAP = {
    # Group A — 16:35
    "G326": "2018-03-11.16-35-01.16-40-01.admin.G326.r13",
    "G328": "2018-03-11.16-35-01.16-40-01.school.G328.r13",
    "G329": "2018-03-11.16-35-01.16-40-01.admin.G329.r13",
    "G336": "2018-03-11.16-35-01.16-40-01.school.G336.r13",
    "G339": "2018-03-11.16-35-01.16-40-01.school.G339.r13",
    "G419": "2018-03-11.16-35-01.16-40-01.school.G419.r13",
    "G420": "2018-03-11.16-35-01.16-40-01.school.G420.r13",
    "G423": "2018-03-11.16-35-01.16-40-00.school.G423.r13",
    "G506": "2018-03-11.16-35-01.16-40-01.bus.G506.r13",
    "G638": "2018-03-11.16-35-01.16-40-01.school.G638.r13",
    # Group B — 16:15
    "G508": "2018-03-11.16-15-01.16-20-01.bus.G508.r13",
}

# Some cameras appear in BOTH groups — annotator should specify which slot.
# Fallback: Group A (16:35) version.
CAMERA_VIDEO_MAP_B = {
    "G423": "2018-03-11.16-15-01.16-20-00.school.G423.r13",
    "G336": "2018-03-11.16-15-01.16-20-01.school.G336.r13",
    "G329": "2018-03-11.16-15-01.16-20-01.admin.G329.r13",
    "G419": "2018-03-11.16-15-01.16-20-01.school.G419.r13",
    "G506": "2018-03-11.16-15-01.16-20-01.bus.G506.r13",
}


def video_id_for(camera_id: str, sighting: dict) -> str:
    # Use Group B map if the video_id appears in the annotator notes or time context
    # Simple heuristic: if another sighting for same cam+person has time > 5:00, it's Group B
    return CAMERA_VIDEO_MAP.get(camera_id, f"UNKNOWN_CAM_{camera_id}")


# ── Extract appearance attributes ─────────────────────────────────────────────
def extract_top(appearance: str) -> str:
    """Return the most distinctive part of appearance for query phrasing."""
    if not appearance:
        return ""
    parts = [p.strip() for p in appearance.split(",")]
    # First part is usually the top clothing — most distinctive
    return parts[0] if parts else appearance


def time_range(s: dict) -> str | None:
    if s["time_enter"] and s["time_exit"]:
        return f"{s['time_enter']}-{s['time_exit']}"
    if s["time_enter"]:
        return s["time_enter"]
    return None


# ── Generate queries ──────────────────────────────────────────────────────────
def generate(sightings: list[dict], max_negatives: int, seed: int) -> list[dict]:
    random.seed(seed)
    cases = []

    # Index: person_tag → list of sightings
    by_person = defaultdict(list)
    for s in sightings:
        by_person[s["person_tag"]].append(s)

    # Index: confirmed cross-camera pairs (same person_tag, different camera)
    confirmed_pairs = set()
    for tag, obs_list in by_person.items():
        cams = [o["camera_id"] for o in obs_list]
        for i, j in combinations(range(len(obs_list)), 2):
            if obs_list[i]["camera_id"] != obs_list[j]["camera_id"]:
                pair = tuple(sorted([obs_list[i]["camera_id"], obs_list[j]["camera_id"]]))
                confirmed_pairs.add(pair)

    def make(video_id, question, challenge, answer, expected_time,
             difficulty, video_quality, category):
        return {
            "video_id": video_id,
            "question": question,
            "recall_challenge": challenge,
            "expected_answer": answer,
            "expected_time": expected_time,
            "difficulty": difficulty,
            "video_quality": video_quality,
            "_category": category,
        }

    # ── 1. Existence queries  (one per sighting) ────────────────────────────
    for s in sightings:
        vid = video_id_for(s["camera_id"], s)
        top = extract_top(s["appearance"])
        t = time_range(s)
        q = (f"Is there a person visible in camera {s['camera_id']}?"
             if not top else
             f"Is there a person with {top} visible in camera {s['camera_id']}?")
        challenge = s["appearance"] if s["appearance"] else f"Person in {s['camera_id']}"
        cases.append(make(vid, q, challenge, "yes", t, "easy", "high", "existence"))

    # ── 2. Appearance attribute queries ────────────────────────────────────
    for s in sightings:
        if not s["appearance"]:
            continue
        vid = video_id_for(s["camera_id"], s)
        parts = [p.strip() for p in s["appearance"].split(",") if p.strip()]
        t = time_range(s)
        # Query on first attribute (top clothing)
        if len(parts) >= 1:
            q = f"Is there a person wearing {parts[0]} in camera {s['camera_id']}?"
            cases.append(make(vid, q, s["appearance"], "yes", t, "medium", "high", "appearance"))
        # Query on bag if mentioned
        bag_attrs = [p for p in parts if any(w in p.lower() for w in ["bag","backpack","tote","carry"])]
        if bag_attrs:
            q = f"Is there a person carrying {bag_attrs[0]} in camera {s['camera_id']}?"
            cases.append(make(vid, q, s["appearance"], "yes", t, "medium", "high", "appearance"))

    # ── 3. Direction / event queries ────────────────────────────────────────
    for s in sightings:
        if not s["exit_direction"] or s["exit_direction"].lower() == "stayed":
            continue
        vid = video_id_for(s["camera_id"], s)
        d = s["exit_direction"].lower()
        t = time_range(s)
        q = f"Did a person exit from the {d} side of camera {s['camera_id']}?"
        challenge = f"Person leaving frame {d}; {s['appearance']}"
        cases.append(make(vid, q, challenge, "yes", t, "medium", "high", "event"))

    # ── 4. Cross-camera positive queries ────────────────────────────────────
    for tag, obs_list in by_person.items():
        # Find all confirmed cross-camera pairs for this tag
        multi_cam = [(o["camera_id"], o) for o in obs_list]
        seen_pairs = set()
        for i, (cam_a, obs_a) in enumerate(multi_cam):
            for j, (cam_b, obs_b) in enumerate(multi_cam):
                if i >= j:
                    continue
                if cam_a == cam_b:
                    continue
                pair_key = tuple(sorted([cam_a, cam_b]))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                # Only generate if explicitly linked (same_as field or same person_tag)
                if obs_b.get("same_as") != tag and obs_a.get("same_as") != tag:
                    # Still generate if same person_tag — annotator used same tag intentionally
                    pass  # allow through

                vid_a = video_id_for(cam_a, obs_a)
                top = extract_top(obs_a["appearance"])
                conf = obs_b.get("confidence") or obs_a.get("confidence") or "high"
                difficulty = "hard" if conf == "low" else ("medium" if conf == "medium" else "hard")

                # Time on the SECOND camera (where they re-appeared)
                # Determine which is "first" by time
                def to_sec(t):
                    if not t: return 9999
                    parts = t.split(":")
                    try: return int(parts[0])*60 + int(parts[1])
                    except: return 9999
                first, second = (obs_a, obs_b) if to_sec(obs_a["time_enter"]) <= to_sec(obs_b["time_enter"]) else (obs_b, obs_a)
                first_cam  = first["camera_id"]
                second_cam = second["camera_id"]
                t_second = time_range(second)

                desc = top if top else f"person tagged {tag}"
                q = (f"Did a person with {desc} appear in camera {first_cam} and then "
                     f"appear again in camera {second_cam}?")
                challenge = (f"Same person ({tag}) first seen in {first_cam} at "
                             f"{first['time_enter'] or '?'}, then {second_cam} at "
                             f"{second['time_enter'] or '?'}. "
                             f"Appearance: {obs_a['appearance']}")
                vid = video_id_for(second_cam, second)
                cases.append(make(vid, q, challenge, "yes", t_second,
                                  difficulty, "high", "cross_camera"))

    # ── 5. Negative queries (FP — different people, not confirmed same) ──────
    # Collect all unique (person_tag, camera_id) sighting combos
    unique_sightings = list({(s["person_tag"], s["camera_id"]): s for s in sightings}.values())
    negative_candidates = []
    for i, s_a in enumerate(unique_sightings):
        for s_b in unique_sightings[i+1:]:
            if s_a["camera_id"] == s_b["camera_id"]:
                continue
            if s_a["person_tag"] == s_b["person_tag"]:
                continue
            # Check this pair is NOT a confirmed cross-camera match
            pair = tuple(sorted([s_a["camera_id"], s_b["camera_id"]]))
            if pair in confirmed_pairs:
                continue
            negative_candidates.append((s_a, s_b))

    random.shuffle(negative_candidates)
    for s_a, s_b in negative_candidates[:max_negatives]:
        top_a = extract_top(s_a["appearance"])
        top_b = extract_top(s_b["appearance"])
        desc = top_a if top_a else f"person {s_a['person_tag']}"
        vid = video_id_for(s_a["camera_id"], s_a)
        q = (f"Did a person with {desc} from camera {s_a['camera_id']} "
             f"also appear in camera {s_b['camera_id']}?")
        challenge = (f"NOT a match. {s_a['person_tag']} ({s_a['appearance']}) in {s_a['camera_id']} "
                     f"is different from {s_b['person_tag']} ({s_b['appearance']}) in {s_b['camera_id']}")
        cases.append(make(vid, q, challenge, "no", "N/A", "medium", "high", "negative"))

    return cases


# ── Write agent_test.xlsx ─────────────────────────────────────────────────────
CAT_COLOUR = {
    "existence":   "DEEAF1",
    "appearance":  "E2EFDA",
    "event":       "FFF2CC",
    "cross_camera":"F4CCFF",
    "negative":    "FFE0E0",
}

def fill(h): return PatternFill("solid", fgColor=h)
def fnt(bold=False, color="000000", size=10):
    return Font(name="Calibri", bold=bold, color=color, size=size)
thin = Side(style="thin", color="CCCCCC")
def bdr(): return Border(left=thin, right=thin, top=thin, bottom=thin)

def write_xlsx(cases: list[dict], out_path: str):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Part1"

    headers = ["video_id", "question", "recall_challenge",
               "expected_answer", "expected_time", "difficulty", "video_quality"]
    widths   = [52, 80, 70, 16, 18, 12, 14]

    # Header row
    for ci, (h, w) in enumerate(zip(headers, widths), 1):
        c = ws.cell(row=1, column=ci)
        c.value = h
        c.font = Font(name="Calibri", bold=True, color="FFFFFF", size=10)
        c.fill = fill("1F4E79")
        c.alignment = Alignment(horizontal="center", vertical="center")
        c.border = bdr()
        ws.column_dimensions[get_column_letter(ci)].width = w
    ws.row_dimensions[1].height = 24

    prev_vid = None
    for ri, case in enumerate(cases, 2):
        vid = case["video_id"]
        write_vid = vid if vid != prev_vid else None
        if vid:
            prev_vid = vid

        cat = case.get("_category", "")
        row_bg = CAT_COLOUR.get(cat, "FFFFFF")
        ans = case["expected_answer"]

        values = [write_vid, case["question"], case["recall_challenge"],
                  ans, case["expected_time"], case["difficulty"], case["video_quality"]]

        for ci, val in enumerate(values, 1):
            c = ws.cell(row=ri, column=ci)
            c.value = val
            if ci == 4:  # expected_answer
                c.fill = fill("C6EFCE" if ans == "yes" else "FFDCE1" if ans == "no" else row_bg)
                c.font = Font(name="Calibri", bold=True, size=10,
                              color="375623" if ans == "yes" else
                                    "C00000" if ans == "no" else "000000")
            else:
                c.fill = fill(row_bg)
                c.font = fnt(size=10)
            c.alignment = Alignment(vertical="center", wrap_text=True,
                                    horizontal="center" if ci in (4,5,6,7) else "left")
            c.border = bdr()
        ws.row_dimensions[ri].height = 38

    ws.freeze_panes = "A2"
    ws.sheet_view.zoomScale = 95

    # Legend sheet
    wl = wb.create_sheet("Legend (auto-generated)")
    wl.column_dimensions["A"].width = 18
    wl.column_dimensions["B"].width = 60
    for ri, (cat, bg) in enumerate(CAT_COLOUR.items(), 1):
        c1 = wl.cell(row=ri, column=1, value=cat)
        c1.fill = fill(bg)
        c1.font = fnt(bold=True)
        c1.border = bdr()
        c2 = wl.cell(row=ri, column=2,
                     value={"existence":   "Is there a [person/object] in [camera]?",
                            "appearance":  "Is there a person wearing [specific attribute] in [camera]?",
                            "event":       "Did a person exit from [direction] in [camera]?",
                            "cross_camera":"Did [person] appear in [camA] and then [camB]?",
                            "negative":    "Cross-camera pair NOT confirmed as same person (answer = no)"}.get(cat,""))
        c2.font = fnt()
        c2.border = bdr()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    wb.save(out_path)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = get_args()
    print(f"Loading observations from: {args.obs}")
    sightings = load_sightings(args.obs)
    print(f"  Loaded {len(sightings)} sighting rows")

    if not sightings:
        print("ERROR: No sightings found. Make sure the Sightings sheet is filled in.")
        return

    cases = generate(sightings, args.max_negatives, args.seed)
    print(f"  Generated {len(cases)} test cases")

    from collections import Counter
    cat_counts = Counter(c["_category"] for c in cases)
    ans_counts = Counter(c["expected_answer"] for c in cases)
    print(f"  By category: {dict(cat_counts)}")
    print(f"  By answer:   {dict(ans_counts)}")

    write_xlsx(cases, args.out)
    print(f"\nSaved: {args.out}")
    print("Next step:")
    print("  python agent/test/agent_test_importer.py --xlsx-path", args.out)


if __name__ == "__main__":
    # Need openpyxl.utils.get_column_letter in write_xlsx
    from openpyxl.utils import get_column_letter
    main()
