import json
import sqlite3
import sqlite_vec
import struct
import concurrent.futures
from src.indexing.embedder import get_qwen_embedding

def serialize_f32(vector: list[float]) -> bytes:
    """Pack float list as binary blob for sqlite-vec."""
    return struct.pack(f"{len(vector)}f", *vector)


def _first_clause(text: str) -> str:
    """First segment before a clause separator (ASCII or fullwidth punctuation)."""
    if not text:
        return ""
    for sep in (",", ";", "|", "\uff0c", "\u3001"):
        if sep in text:
            return text.split(sep, 1)[0].strip()
    return text.strip()

def preprocess_event(raw_event: dict) -> dict:
    """Normalize raw JSON event fields for storage."""
    # bbox
    start_bbox = raw_event.get("start_bbox_xyxy", [0,0,0,0])
    end_bbox = raw_event.get("end_bbox_xyxy", [0,0,0,0])
    
    start_time = raw_event.get("start_time", 0.0)
    end_time = raw_event.get("end_time", 0.0)
    duration = end_time - start_time
    
    track_id = raw_event.get("entity_hint", "").replace("track_id=", "")
    
    keywords = raw_event.get("keywords", [])
    event_type = keywords[0] if keywords else ""
    keywords_json = json.dumps(keywords, ensure_ascii=False)
    
    obj_color = raw_event.get("object_color") or raw_event.get("object_color_cn", "")
    obj_type = raw_event.get("object_type", "")
    scene_zone = raw_event.get("scene_zone") or raw_event.get("scene_zone_cn", "")
    appearance = raw_event.get("appearance_notes") or raw_event.get("appearance_notes_cn", "")

    event_summary = f"{obj_color} {obj_type} in {scene_zone} {appearance}".strip()
    normalized_state = _first_clause(appearance)
    retrieval_text = f"{obj_color} {obj_type} {scene_zone} {appearance} {' '.join(keywords)}"
    
    return {
        "video_id": raw_event.get("video_id", ""),
        "camera_id": raw_event.get("camera_id", ""),
        "track_id": track_id,
        "global_id": f"{raw_event.get('video_id', '')}_{track_id}_{start_time}",
        "start_time": start_time,
        "end_time": end_time,
        "duration": duration,
        "source_clip_start_sec": raw_event.get("clip_start_sec", 0.0),
        "source_clip_end_sec": raw_event.get("clip_end_sec", 0.0),
        "object_type": obj_type,
        "object_color": obj_color,
        "scene_zone": scene_zone,
        "appearance_notes": appearance,
        "event_type": event_type,
        "event_text": raw_event.get("event_text") or raw_event.get("event_text_cn", ""),
        "event_summary": event_summary,
        "normalized_state": normalized_state,
        "keywords_json": keywords_json,
        "retrieval_text": retrieval_text,
        "start_bbox_x1": start_bbox[0],
        "start_bbox_y1": start_bbox[1],
        "start_bbox_x2": start_bbox[2],
        "start_bbox_y2": start_bbox[3],
        "end_bbox_x1": end_bbox[0],
        "end_bbox_y1": end_bbox[1],
        "end_bbox_x2": end_bbox[2],
        "end_bbox_y2": end_bbox[3]
    }

def process_batch(batch_events: list) -> list:
    """Process a batch: embeddings via API; returns list of (parsed_event, embedding)."""
    parsed_events = [preprocess_event(e) for e in batch_events]
    texts = [p["retrieval_text"] for p in parsed_events]
    
    try:
        # Dashscope batch limit (often 25 per call)
        embeddings = get_qwen_embedding(texts)
        return list(zip(parsed_events, embeddings))
    except Exception as e:
        print(f"Batch failed: {e}")
        return []

def ingest_data(json_path: str, db_path: str):
    """Load JSON, batch-embed in parallel, write SQLite + vectors."""
    print(f"Loading data: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    all_raw_events = []
    for video_data in data:
        all_raw_events.extend(video_data.get("events", []))
    
    total_events = len(all_raw_events)
    print(f"Found {total_events} events. Batch embedding and ingest...")
    
    BATCH_SIZE = 10
    batches = [all_raw_events[i:i + BATCH_SIZE] for i in range(0, total_events, BATCH_SIZE)]
    
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(process_batch, batch): i for i, batch in enumerate(batches)}
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            results.extend(future.result())
            completed += 1
            if completed % 10 == 0 or completed == len(batches):
                print(f"Embedding progress: {completed}/{len(batches)} batches ({min(completed*BATCH_SIZE, total_events)}/{total_events} rows)")
    
    print(f"Got embeddings for {len(results)} rows. Writing SQLite...")
    
    db = sqlite3.connect(db_path)
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    cursor = db.cursor()
    
    count = 0
    cursor.execute("BEGIN TRANSACTION")
    for parsed, vec in results:
        cursor.execute("""
            INSERT INTO episodic_events (
                video_id, camera_id, track_id, global_id, start_time, end_time, duration,
                source_clip_start_sec, source_clip_end_sec, object_type, object_color,
                scene_zone, appearance_notes, event_type, event_text, event_summary,
                normalized_state, keywords_json, retrieval_text,
                start_bbox_x1, start_bbox_y1, start_bbox_x2, start_bbox_y2,
                end_bbox_x1, end_bbox_y1, end_bbox_x2, end_bbox_y2
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            parsed["video_id"], parsed["camera_id"], parsed["track_id"], parsed["global_id"],
            parsed["start_time"], parsed["end_time"], parsed["duration"], parsed["source_clip_start_sec"],
            parsed["source_clip_end_sec"], parsed["object_type"], parsed["object_color"],
            parsed["scene_zone"], parsed["appearance_notes"], parsed["event_type"],
            parsed["event_text"], parsed["event_summary"], parsed["normalized_state"],
            parsed["keywords_json"], parsed["retrieval_text"],
            parsed["start_bbox_x1"], parsed["start_bbox_y1"], parsed["start_bbox_x2"], parsed["start_bbox_y2"],
            parsed["end_bbox_x1"], parsed["end_bbox_y1"], parsed["end_bbox_x2"], parsed["end_bbox_y2"]
        ))
        
        event_id = cursor.lastrowid
        
        cursor.execute(
            "INSERT INTO episodic_events_vec(rowid, embedding) VALUES (?, ?)",
            (event_id, serialize_f32(vec))
        )
        count += 1
            
    db.commit()
    db.close()
    print(f"Ingest done: wrote {count} rows to {db_path}")

if __name__ == "__main__":
    import sys
    import os
    from dotenv import load_dotenv
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
    
    try:
        load_dotenv()
    except Exception:
        pass
    
    mock_data_path = os.path.join(os.path.dirname(__file__), "../../../mvp_data/video_events_10k_mock.json")
    
    from pathlib import Path
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    db_path = str(BASE_DIR / "src" / "agent" / "memory" / "episodic" / "episodic_memory.db")
    
    if not os.environ.get("DASHSCOPE_API_KEY"):
        print("Error: set DASHSCOPE_API_KEY")
    else:
        ingest_data(mock_data_path, db_path)
