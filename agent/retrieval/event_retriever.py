import sqlite3
import sqlite_vec
import struct
import json
from typing import List, Dict, Any, Optional

from tools.llm import get_qwen_embedding
from db.config import get_graph_sqlite_db_path

def serialize_f32(vector: list[float]) -> bytes:
    return struct.pack(f"{len(vector)}f", *vector)

class EventRetriever:
    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or str(get_graph_sqlite_db_path())

    def _get_db_connection(self):
        db = sqlite3.connect(self.db_path)
        db.row_factory = sqlite3.Row  # rows as dict-like
        db.enable_load_extension(True)
        sqlite_vec.load(db)
        db.enable_load_extension(False)
        return db

    def get_event_detail(self, event_id: int) -> Optional[Dict[str, Any]]:
        """Fetch full event row by event_id."""
        db = self._get_db_connection()
        cursor = db.cursor()
        
        cursor.execute("SELECT * FROM episodic_events WHERE event_id = ?", (event_id,))
        row = cursor.fetchone()
        
        db.close()
        
        if row:
            return dict(row)
        return None

    def structured_search(self, 
                          video_id: str = None, 
                          object_type: str = None,
                          scene_zone_en: str = None,
                          min_duration: float = None,
                          start_time_after: float = None,
                          limit: int = 10) -> List[Dict[str, Any]]:
        """Structured field search (no vector index)."""
        db = self._get_db_connection()
        cursor = db.cursor()
        
        query = "SELECT * FROM episodic_events WHERE 1=1"
        params = []
        
        if video_id:
            query += " AND video_id = ?"
            params.append(video_id)
        if object_type:
            query += " AND object_type = ?"
            params.append(object_type)
        if scene_zone_en:
            query += " AND scene_zone_en = ?"
            params.append(scene_zone_en)
        if min_duration is not None:
            query += " AND duration_sec >= ?"
            params.append(min_duration)
        if start_time_after is not None:
            query += " AND start_time >= ?"
            params.append(start_time_after)
            
        query += " ORDER BY start_time ASC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        db.close()
        
        return [dict(row) for row in rows]

    def hybrid_event_search(self, 
                            query_text: str, 
                            top_k: int = 5,
                            video_id: str = None,
                            object_type: str = None,
                            scene_zone_en: str = None,
                            start_time_after: float = None,
                            end_time_before: float = None) -> List[Dict[str, Any]]:
        """
        Hybrid search: SQLite filters + sqlite-vec similarity.
        """
        try:
            query_vector = get_qwen_embedding(query_text)
        except Exception as e:
            print(f"Failed to embed query: {e}")
            return []

        db = self._get_db_connection()
        cursor = db.cursor()

        base_query = """
            SELECT 
                e.event_id,
                e.video_id,
                e.start_time,
                e.end_time,
                e.event_summary_en,
                v.distance
            FROM episodic_events_vec v
            JOIN episodic_events e ON v.rowid = e.event_id
            WHERE v.embedding MATCH ?
            AND k = ?
        """
        
        params = [serialize_f32(query_vector), top_k]
        
        if video_id:
            base_query += " AND e.video_id = ?"
            params.append(video_id)
            
        if object_type:
            base_query += " AND e.object_type = ?"
            params.append(object_type)
            
        if scene_zone_en:
            base_query += " AND e.scene_zone_en = ?"
            params.append(scene_zone_en)
            
        if start_time_after is not None:
            base_query += " AND e.start_time >= ?"
            params.append(start_time_after)
            
        if end_time_before is not None:
            base_query += " AND e.end_time <= ?"
            params.append(end_time_before)
            
        base_query += " ORDER BY v.distance ASC"
        
        cursor.execute(base_query, params)
        rows = cursor.fetchall()
        db.close()
        
        results = [dict(row) for row in rows]
        return results

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    try:
        load_dotenv()
    except Exception:
        pass
        
    if not os.environ.get("DASHSCOPE_API_KEY"):
        print("Tests require DASHSCOPE_API_KEY")
    else:
        retriever = EventRetriever()
        
        print("Test get_event_detail (event_id=1):")
        detail = retriever.get_event_detail(1)
        if detail:
            print(f"Fetched: {detail.get('event_summary_en')}")
        else:
            print("No row found (database may be empty).")
            
        print("\nTest hybrid_event_search ('find a parked car'):")
        results = retriever.hybrid_event_search("find a parked car", top_k=3)
        for idx, res in enumerate(results):
            print(f"[{idx+1}] ID: {res['event_id']}, video: {res['video_id']}, distance: {res['distance']:.4f}")
            print(f"    summary: {res['event_summary_en']}")
