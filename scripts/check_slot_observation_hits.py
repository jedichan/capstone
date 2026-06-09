import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


CAM_RE = re.compile(r"\bG\d{3}\b", re.IGNORECASE)


@dataclass
class ObsRow:
    idx: int
    camera: str | None
    cls: str | None
    t_start: float | None
    t_end: float | None
    raw: dict[str, Any]


def _to_sec(v: Any) -> float | None:
    if v is None:
        return None
    s = str(v).strip()
    if not s or s.lower() in {"nan", "none"}:
        return None
    try:
        return float(s)
    except Exception:
        pass
    m = re.match(r"^(\d{1,2}):(\d{1,2})(?::(\d{1,2}))?$", s)
    if m:
        h = int(m.group(1))
        mm = int(m.group(2))
        ss = int(m.group(3) or 0)
        return float(h * 3600 + mm * 60 + ss)
    return None


def _pick_col(cols: list[str], keys: tuple[str, ...]) -> str | None:
    low = {c.lower(): c for c in cols}
    for k in keys:
        for lk, orig in low.items():
            if k in lk:
                return orig
    return None


def load_obs(xlsx_path: Path) -> list[ObsRow]:
    xl = pd.ExcelFile(xlsx_path)
    rows: list[ObsRow] = []
    for sheet in xl.sheet_names:
        df = pd.read_excel(xlsx_path, sheet_name=sheet, dtype=str)
        if df.empty:
            continue
        cols = [str(c) for c in df.columns]
        cam_col = _pick_col(cols, ("camera", "cam"))
        cls_col = _pick_col(cols, ("class", "type", "object", "category", "person", "vehicle"))
        st_col = _pick_col(cols, ("start", "from", "begin", "time"))
        ed_col = _pick_col(cols, ("end", "to", "finish", "time"))
        for _, r in df.iterrows():
            raw = {str(k): r[k] for k in df.columns}
            cam = None
            if cam_col:
                m = CAM_RE.search(str(r.get(cam_col, "")))
                if m:
                    cam = m.group(0).upper()
            if not cam:
                for v in raw.values():
                    m = CAM_RE.search(str(v))
                    if m:
                        cam = m.group(0).upper()
                        break
            if not cam:
                continue

            cls = None
            if cls_col:
                t = str(r.get(cls_col, "")).lower()
                if "person" in t or "people" in t or "pedestrian" in t:
                    cls = "person"
                elif "car" in t:
                    cls = "car"
                elif "truck" in t:
                    cls = "truck"
                elif "vehicle" in t:
                    cls = "vehicle"

            t_start = _to_sec(r.get(st_col)) if st_col else None
            t_end = _to_sec(r.get(ed_col)) if ed_col else None
            if t_start is not None and t_end is not None and t_end < t_start:
                t_start, t_end = t_end, t_start
            rows.append(ObsRow(len(rows), cam, cls, t_start, t_end, raw))
    return rows


def load_seed_events(seeds_dir: Path, slot: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for fp in sorted(seeds_dir.glob(f"*{slot}*events_vector_flat.json")):
        data = json.loads(fp.read_text(encoding="utf-8"))
        events = data.get("events", []) if isinstance(data, dict) else []
        for ev in events:
            if "camera_id" not in ev:
                ev["camera_id"] = data.get("camera_id")
            out.append(ev)
    return out


def overlap(a0: float, a1: float, b0: float, b1: float) -> bool:
    return min(a1, b1) >= max(a0, b0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", required=True)
    ap.add_argument("--seeds-dir", required=True)
    ap.add_argument("--slot", required=True)
    args = ap.parse_args()

    obs_rows = load_obs(Path(args.xlsx))
    events = load_seed_events(Path(args.seeds_dir), args.slot)

    by_cam: dict[str, list[dict[str, Any]]] = {}
    for e in events:
        cam = str(e.get("camera_id", "")).upper()
        if not cam:
            continue
        by_cam.setdefault(cam, []).append(e)

    total = len(obs_rows)
    cam_hit = 0
    class_hit = 0
    full_hit = 0
    samples: list[str] = []
    misses: list[str] = []

    for o in obs_rows:
        cands = by_cam.get(o.camera or "", [])
        if cands:
            cam_hit += 1
        else:
            misses.append(f"obs#{o.idx}: cam={o.camera} no camera match")
            continue

        cands2 = cands
        if o.cls and o.cls != "vehicle":
            cands2 = [e for e in cands if str(e.get("object_type") or e.get("class_name") or "").lower() == o.cls]
        elif o.cls == "vehicle":
            cands2 = [e for e in cands if str(e.get("object_type") or e.get("class_name") or "").lower() in {"car", "truck", "bus", "vehicle"}]
        if cands2:
            class_hit += 1
        else:
            misses.append(f"obs#{o.idx}: cam={o.camera} class={o.cls} no class match")
            continue

        has_time = o.t_start is not None and o.t_end is not None
        if not has_time:
            full_hit += 1
            if len(samples) < 8:
                samples.append(f"obs#{o.idx} cam={o.camera} cls={o.cls or '-'} time=NA -> matched")
            continue

        ok = False
        for e in cands2:
            s = float(e.get("start_time", 0.0))
            t = float(e.get("end_time", s))
            if overlap(o.t_start, o.t_end, s, t):
                ok = True
                break
        if ok:
            full_hit += 1
            if len(samples) < 8:
                samples.append(
                    f"obs#{o.idx} cam={o.camera} cls={o.cls or '-'} "
                    f"time={o.t_start:.1f}-{o.t_end:.1f} -> matched"
                )
        else:
            misses.append(
                f"obs#{o.idx}: cam={o.camera} cls={o.cls} "
                f"time={o.t_start}-{o.t_end} no temporal overlap"
            )

    print("=== slot observation hit check ===")
    print(f"slot={args.slot}")
    print(f"observation_rows_used={total}")
    print(f"seed_events={len(events)}")
    print(f"camera_hit={cam_hit}/{total}")
    print(f"camera+class_hit={class_hit}/{total}")
    print(f"full_hit(camera+class+time_if_present)={full_hit}/{total}")
    print("--- sample hits ---")
    for s in samples:
        print(s)
    print("--- first misses ---")
    for s in misses[:10]:
        print(s)


if __name__ == "__main__":
    main()
