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
    # HH:MM[:SS]
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
        for i, r in df.iterrows():
            raw = {str(k): r[k] for k in df.columns}
            # camera from explicit column or any cell text
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
            rows.append(
                ObsRow(
                    idx=len(rows),
                    camera=cam,
                    cls=cls,
                    t_start=t_start,
                    t_end=t_end,
                    raw=raw,
                )
            )
    return rows


def overlap(a0: float, a1: float, b0: float, b1: float) -> bool:
    return min(a1, b1) >= max(a0, b0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", required=True)
    ap.add_argument("--pipeline", required=True)
    args = ap.parse_args()

    obs_rows = load_obs(Path(args.xlsx))
    pipeline = json.loads(Path(args.pipeline).read_text(encoding="utf-8"))
    events = pipeline.get("events", [])

    by_cam: dict[str, list[dict[str, Any]]] = {}
    for e in events:
        cam = str(e.get("camera_id", "")).upper()
        by_cam.setdefault(cam, []).append(e)

    total = len(obs_rows)
    cam_hit = 0
    class_hit = 0
    time_hit = 0
    full_hit = 0
    samples: list[str] = []

    for o in obs_rows:
        cands = by_cam.get(o.camera or "", [])
        if cands:
            cam_hit += 1
        if not cands:
            continue
        cands2 = cands
        if o.cls and o.cls != "vehicle":
            cands2 = [e for e in cands if str(e.get("class_name", "")).lower() == o.cls]
        elif o.cls == "vehicle":
            cands2 = [e for e in cands if str(e.get("class_name", "")).lower() in {"car", "truck", "bus"}]
        if cands2:
            class_hit += 1
        else:
            continue
        # time check only if observation has explicit range
        has_time = o.t_start is not None and o.t_end is not None
        time_ok = False
        if not has_time:
            time_ok = True
        else:
            for e in cands2:
                s = float(e.get("start_time", 0.0))
                t = float(e.get("end_time", s))
                if overlap(o.t_start, o.t_end, s, t):
                    time_ok = True
                    break
        if time_ok:
            time_hit += 1
            full_hit += 1
            if len(samples) < 8:
                samples.append(
                    f"obs#{o.idx} cam={o.camera} cls={o.cls or '-'} "
                    f"time={o.t_start}-{o.t_end} -> matched"
                )

    print("=== 11-55 observation hit check ===")
    print(f"observation_rows_used={total}")
    print(f"pipeline_events={len(events)}")
    print(f"camera_hit={cam_hit}/{total}")
    print(f"camera+class_hit={class_hit}/{total}")
    print(f"full_hit(camera+class+time_if_present)={full_hit}/{total}")
    print("--- sample hits ---")
    for s in samples:
        print(s)


if __name__ == "__main__":
    main()
