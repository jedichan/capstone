import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import (
    DEFAULT_CHROMA_PATH,
    get_graph_chroma_child_collection,
    get_graph_chroma_event_collection,
    get_graph_chroma_global_entity_collection,
    get_graph_chroma_parent_collection,
    get_graph_chroma_video_collection,
)
from ..tools.llm import get_qwen_embedding


logger = logging.getLogger(__name__)


# DashScope text embedding currently rejects a single input outside [1, 8192].
# Keep a margin because APIs may count UTF-8/chars/tokens differently.
MAX_EMBEDDING_DOCUMENT_CHARS = 7600


class ChromaBuildError(RuntimeError):
    pass


@dataclass
class ChromaBuildConfig:
    chroma_path: Path = DEFAULT_CHROMA_PATH
    # 采用 default_factory 以便每次实例化时读取最新 AGENT_CHROMA_NAMESPACE / 显式 env 覆盖。
    child_collection: str = field(default_factory=get_graph_chroma_child_collection)
    parent_collection: str = field(default_factory=get_graph_chroma_parent_collection)
    event_collection: str = field(default_factory=get_graph_chroma_event_collection)
    video_collection: str = field(default_factory=get_graph_chroma_video_collection)
    global_entity_collection: str = field(default_factory=get_graph_chroma_global_entity_collection)
    reset_existing: bool = False
    sqlite_db_path: Path | None = None


class ChromaIndexBuilder:
    def __init__(self, config: ChromaBuildConfig) -> None:
        self.config = config
        self.chroma_path = config.chroma_path
        self.child_collection = config.child_collection
        self.parent_collection = config.parent_collection
        self.event_collection = config.event_collection
        self.video_collection = config.video_collection
        self.global_entity_collection = config.global_entity_collection
        self.sqlite_db_path = config.sqlite_db_path

    def build(self, seed_files: list[Path] | None = None, llm: Any = None) -> dict[str, Any]:
        import chromadb

        seed_files = seed_files or []
        self.chroma_path.mkdir(parents=True, exist_ok=True)
        events = self._load_seed_events(seed_files)
        child_records = self._build_child_records(events)
        parent_records = self._build_parent_records(child_records)
        event_records = self._build_event_records(events)

        # Tier 1: per-video discriminator summaries for coarse retrieval
        video_records = self._build_video_records(events, llm) if llm is not None else []

        # Cross-camera aggregation: one record per global entity across all cameras
        global_entity_records = self._build_global_entity_records(events)

        try:
            client = chromadb.PersistentClient(path=str(self.chroma_path))
            if self.config.reset_existing:
                self._delete_collection_if_exists(client, self.child_collection)
                self._delete_collection_if_exists(client, self.parent_collection)
                self._delete_collection_if_exists(client, self.event_collection)
                self._delete_collection_if_exists(client, self.video_collection)
                self._delete_collection_if_exists(client, self.global_entity_collection)

            child_collection = client.get_or_create_collection(
                name=self.child_collection,
                metadata={"hnsw:space": "cosine", "index_role": "child"},
            )
            parent_collection = client.get_or_create_collection(
                name=self.parent_collection,
                metadata={"hnsw:space": "cosine", "index_role": "parent"},
            )
            event_collection = client.get_or_create_collection(
                name=self.event_collection,
                metadata={"hnsw:space": "cosine", "index_role": "event"},
            )
            video_collection = client.get_or_create_collection(
                name=self.video_collection,
                metadata={"hnsw:space": "cosine", "index_role": "video"},
            )
            global_entity_collection = client.get_or_create_collection(
                name=self.global_entity_collection,
                metadata={"hnsw:space": "cosine", "index_role": "global_entity"},
            )

            if child_records:
                self._upsert_records(child_collection, child_records)
            if parent_records:
                self._upsert_records(parent_collection, parent_records)
            if event_records:
                self._upsert_records(event_collection, event_records)
            if video_records:
                self._upsert_records(video_collection, video_records)
            if global_entity_records:
                self._upsert_records(global_entity_collection, global_entity_records)
        except Exception as exc:
            logger.exception("Failed to build chroma indexes")
            raise ChromaBuildError(f"Build failed for {self.chroma_path}: {exc}") from exc

        result = {
            "chroma_path": str(self.chroma_path),
            "seed_files": [str(x) for x in seed_files],
            "child_collection": self.child_collection,
            "parent_collection": self.parent_collection,
            "event_collection": self.event_collection,
            "video_collection": self.video_collection,
            "global_entity_collection": self.global_entity_collection,
            "child_record_count": len(child_records),
            "parent_record_count": len(parent_records),
            "event_record_count": len(event_records),
            "video_record_count": len(video_records),
            "global_entity_record_count": len(global_entity_records),
            "chunk_strategy": {
                "child": "track-level (video_id + entity_hint)",
                "parent": "video-level time-bucketed (10min windows)",
                "event": "event-level (video_id + entity_hint + start_time + end_time)",
                "video": "video-level discriminator summary",
                "global_entity": "cross-camera entity-level (global_entity_id)",
            },
        }
        return result

    @staticmethod
    def _delete_collection_if_exists(client: Any, collection_name: str) -> None:
        try:
            client.delete_collection(collection_name)
        except Exception:
            logger.info("Skip deleting collection %s because it does not exist", collection_name)

    @staticmethod
    def _upsert_records(collection: Any, records: list[dict[str, Any]]) -> None:
        documents = [
            ChromaIndexBuilder._clip_embedding_document(record["document"])
            for record in records
        ]
        embeddings = get_qwen_embedding(documents)
        collection.upsert(
            ids=[record["id"] for record in records],
            documents=documents,
            metadatas=[record["metadata"] for record in records],
            embeddings=embeddings,
        )

    @staticmethod
    def _clip_embedding_document(document: Any) -> str:
        text = str(document or "").strip()
        if not text:
            return "empty document"
        if len(text) <= MAX_EMBEDDING_DOCUMENT_CHARS:
            return text
        return text[:MAX_EMBEDDING_DOCUMENT_CHARS].rstrip() + " [truncated for embedding]"

    def _load_seed_events(self, seed_files: list[Path]) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for seed_file in seed_files:
            if not seed_file.exists():
                raise ChromaBuildError(f"Seed file not found: {seed_file}")

            payload = json.loads(seed_file.read_text(encoding="utf-8"))
            # --- Camera-grouped multi-camera format ---
            # { "G328": {"video_id": "...", "events": [...]}, "G339": {...} }
            if isinstance(payload, dict) and self._is_camera_grouped(payload):
                for _camera_id, camera_entry in payload.items():
                    if not isinstance(camera_entry, dict):
                        continue
                    fallback = str(camera_entry.get("video_id", "")).strip() or str(_camera_id)
                    events.extend(self._normalize_events(camera_entry.get("events") or [], fallback))
                continue
            if isinstance(payload, dict) and isinstance(payload.get("events"), list):
                fallback = str(payload.get("video_id", "")).strip() or None
                events.extend(self._normalize_events(payload["events"], fallback))
                continue
            if isinstance(payload, list):
                for item in payload:
                    if isinstance(item, dict) and isinstance(item.get("events"), list):
                        fallback = str(item.get("video_id", "")).strip() or None
                        events.extend(self._normalize_events(item["events"], fallback))
                    elif isinstance(item, dict):
                        events.extend(self._normalize_events([item], None))
                continue
            if isinstance(payload, dict):
                events.extend(self._normalize_events([payload], None))
                continue
            raise ChromaBuildError(f"Unsupported seed JSON format: {seed_file}")
        return events

    @staticmethod
    def _normalize_events(raw_events: list[dict[str, Any]], fallback_video_id: str | None) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for event in raw_events:
            if not isinstance(event, dict):
                continue
            video_id = str(event.get("video_id") or fallback_video_id or event.get("camera_id") or "").strip()
            entity_hint = str(event.get("entity_hint") or event.get("track_id") or "na").strip()
            if not video_id:
                continue
            keywords = event.get("keywords")
            if not keywords:
                cls = event.get("class_name") or ""
                etype = event.get("event_type") or ""
                keywords = [k for k in [cls, etype] if k]
            # Derive global_entity_id from entity_hint for cross-camera tracking
            global_entity_id = event.get("global_entity_id")
            if not global_entity_id:
                global_entity_id = ChromaIndexBuilder._derive_global_entity_id(entity_hint)
            normalized.append(
                {
                    "video_id": video_id,
                    "camera_id": event.get("camera_id"),
                    "entity_hint": entity_hint,
                    "start_time": event.get("start_time"),
                    "end_time": event.get("end_time"),
                    "object_type": event.get("object_type") or event.get("class_name"),
                    "object_color": event.get("object_color_en") or event.get("object_color"),
                    "scene_zone": event.get("scene_zone_en") or event.get("scene_zone"),
                    "appearance_notes": event.get("appearance_notes_en") or event.get("appearance_notes"),
                    "event_text": (
                        event.get("event_text_en")
                        or event.get("event_summary_en")
                        or event.get("event_text")
                        or event.get("event_summary")
                        or event.get("description_for_llm")
                    ),
                    "keywords": ChromaIndexBuilder._normalize_keywords(keywords),
                    "cross_camera_ids": event.get("cross_camera_ids") or [],
                    "global_entity_id": global_entity_id,
                    "raw_event": event,
                }
            )
        return normalized

    @staticmethod
    def _normalize_keywords(raw_keywords: Any) -> list[str]:
        if isinstance(raw_keywords, list):
            return [str(item).strip() for item in raw_keywords if str(item).strip()]
        if isinstance(raw_keywords, str):
            text = raw_keywords.replace("|", ",").replace(";", ",")
            return [part.strip() for part in text.split(",") if part.strip()]
        return []

    # --- Multi-camera format detection & global entity derivation ---

    _GLOBAL_ENTITY_PATTERNS: tuple[str, ...] = (
        "_global_",       # person_global_1, vehicle_global_2, etc.
        "global_entity_", # alternative naming
    )

    @staticmethod
    def _is_camera_grouped(payload: dict[str, Any]) -> bool:
        """Detect multi-camera format: {camera_id: {video_id, events}, ...}.

        Heuristic: payload is a dict, does NOT have a direct "events" key,
        and at least one value is a dict that contains an "events" list.
        """
        if not isinstance(payload, dict):
            return False
        if "events" in payload:
            return False  # single-video format: {"video_id": ..., "events": [...]}
        for val in payload.values():
            if isinstance(val, dict) and isinstance(val.get("events"), list):
                return True
        return False

    @staticmethod
    def _derive_global_entity_id(entity_hint: str) -> str | None:
        """Derive ``global_entity_id`` from ``entity_hint`` when it matches a
        known global/cross-camera naming pattern (e.g. ``person_global_1``).

        Returns the entity_hint as-is when it looks like a global entity;
        returns None otherwise.
        """
        if not entity_hint:
            return None
        hint_lower = entity_hint.lower()
        for pattern in ChromaIndexBuilder._GLOBAL_ENTITY_PATTERNS:
            if pattern in hint_lower:
                return entity_hint
        return None

    @staticmethod
    def _dedupe_text(values: list[Any]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for value in values:
            text = str(value or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            out.append(text)
        return out

    @staticmethod
    def _safe_min(values: list[Any]) -> float | None:
        nums = [float(v) for v in values if isinstance(v, (int, float))]
        return min(nums) if nums else None

    @staticmethod
    def _safe_max(values: list[Any]) -> float | None:
        nums = [float(v) for v in values if isinstance(v, (int, float))]
        return max(nums) if nums else None

    @staticmethod
    def _format_time(value: float | None) -> str:
        if value is None:
            return "unknown"
        return f"{value:.3f}".rstrip("0").rstrip(".")

    def _build_child_records(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for event in events:
            key = (event["video_id"], event["entity_hint"])
            grouped.setdefault(key, []).append(event)

        records: list[dict[str, Any]] = []
        for (video_id, entity_hint), group in sorted(grouped.items()):
            start_time = self._safe_min([item.get("start_time") for item in group])
            end_time = self._safe_max([item.get("end_time") for item in group])
            appearance_notes = self._dedupe_text([item.get("appearance_notes") for item in group])
            scene_zones = self._dedupe_text([item.get("scene_zone") for item in group])
            event_texts = self._dedupe_text([item.get("event_text") for item in group])
            object_types = self._dedupe_text([item.get("object_type") for item in group])
            object_colors = self._dedupe_text([item.get("object_color") for item in group])
            keywords = self._dedupe_text(
                [keyword for item in group for keyword in item.get("keywords", [])]
            )
            child_id = f"{video_id}_{entity_hint}"
            parent_id = video_id
            document = self._build_child_document(
                video_id=video_id,
                entity_hint=entity_hint,
                appearance_notes=appearance_notes,
                scene_zones=scene_zones,
                event_texts=event_texts,
                keywords=keywords,
                start_time=start_time,
                end_time=end_time,
            )
            event_ids = self._lookup_event_ids_for_track(video_id, entity_hint)
            metadata = {
                "record_level": "child",
                "video_id": video_id,
                "parent_id": parent_id,
                "entity_hint": entity_hint,
                "event_id": event_ids[0] if event_ids else None,
                "object_type": object_types[0] if object_types else "unknown",
                "object_color": object_colors[0] if object_colors else "unknown",
                "scene_zone": scene_zones[0] if scene_zones else "unknown",
                "start_time": start_time if start_time is not None else -1.0,
                "end_time": end_time if end_time is not None else -1.0,
                "event_count": len(group),
                "keywords": ", ".join(keywords),
            }
            # Chroma rejects empty list metadata values; only include
            # event_ids when the backfill succeeded.
            if event_ids:
                metadata["event_ids"] = event_ids
            records.append(
                {
                    "id": child_id,
                    "document": document,
                    "metadata": metadata,
                }
            )
        return records

    @staticmethod
    def _build_child_document(
        *,
        video_id: str,
        entity_hint: str,
        appearance_notes: list[str],
        scene_zones: list[str],
        event_texts: list[str],
        keywords: list[str],
        start_time: float | None,
        end_time: float | None,
    ) -> str:
        sections = [
            f"Video {video_id}. Track {entity_hint}.",
            f"Time range {ChromaIndexBuilder._format_time(start_time)}s to {ChromaIndexBuilder._format_time(end_time)}s.",
        ]
        if appearance_notes:
            sections.append("Appearance notes: " + " ".join(appearance_notes))
        if scene_zones:
            sections.append("Located in: " + ", ".join(scene_zones) + ".")
        if event_texts:
            sections.append("Events: " + " ".join(event_texts))
        if keywords:
            sections.append("Keywords: " + ", ".join(keywords) + ".")
        return " ".join(sections)

    _PARENT_TIME_BUCKET_SEC: float = 600.0  # 10-minute parent windows

    def _build_parent_records(self, child_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Build parent records bucketed by 10-minute time windows instead of
        whole-video.  This keeps each parent document small enough for the
        embedding API while still providing coarse-grained temporal grouping."""
        grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
        for record in child_records:
            video_id = str(record["metadata"].get("video_id", "")).strip()
            if not video_id:
                continue
            start = record["metadata"].get("start_time")
            # Bucket by 10-min window; default to 0 if no start_time
            bucket = int(start // self._PARENT_TIME_BUCKET_SEC) if isinstance(start, (int, float)) else 0
            grouped.setdefault((video_id, bucket), []).append(record)

        records: list[dict[str, Any]] = []
        for (video_id, bucket), group in sorted(grouped.items()):
            bucket_start = bucket * self._PARENT_TIME_BUCKET_SEC
            bucket_end = bucket_start + self._PARENT_TIME_BUCKET_SEC
            start_time = self._safe_min([item["metadata"].get("start_time") for item in group])
            end_time = self._safe_max([item["metadata"].get("end_time") for item in group])
            child_ids = [item["id"] for item in group]
            scene_zones = self._dedupe_text([item["metadata"].get("scene_zone") for item in group])
            object_types = self._dedupe_text([item["metadata"].get("object_type") for item in group])
            object_colors = self._dedupe_text([item["metadata"].get("object_color") for item in group])
            parent_id = f"{video_id}_{int(bucket_start)}s"
            document = self._build_parent_document(
                video_id=video_id,
                child_records=group,
                scene_zones=scene_zones,
                object_types=object_types,
                object_colors=object_colors,
                start_time=start_time,
                end_time=end_time,
                bucket_start=bucket_start,
                bucket_end=bucket_end,
            )
            # Update child parent_id to the new bucketed id
            for child_record in group:
                child_record["metadata"]["parent_id"] = parent_id
            records.append(
                {
                    "id": parent_id,
                    "document": document,
                    "metadata": {
                        "record_level": "parent",
                        "video_id": video_id,
                        "child_count": len(group),
                        "start_time": start_time if start_time is not None else -1.0,
                        "end_time": end_time if end_time is not None else -1.0,
                        "scene_zones": ", ".join(scene_zones),
                        "object_types": ", ".join(object_types),
                        "object_colors": ", ".join(object_colors),
                        "child_ids_json": json.dumps(child_ids, ensure_ascii=False),
                        "bucket_start": bucket_start,
                        "bucket_end": bucket_end,
                    },
                }
            )
        return records

    def _build_video_records(
        self, events: list[dict[str, Any]], llm: Any
    ) -> list[dict[str, Any]]:
        """Build one Chroma record per video with an LLM-generated discriminator summary.

        Used by Tier 1 two-stage retrieval: query → video_collection → top-3 video_ids
        → child_collection (with ``where`` filter).
        """
        from ..tools.video_discriminator import generate_video_discriminator

        # Group events by video_id
        grouped: dict[str, list[dict[str, Any]]] = {}
        for ev in events:
            vid = str(ev.get("video_id", "")).strip()
            if vid:
                grouped.setdefault(vid, []).append(ev)

        records: list[dict[str, Any]] = []
        for video_id, group in sorted(grouped.items()):
            summary = generate_video_discriminator(video_id=video_id, events=group, llm=llm)
            records.append(
                {
                    "id": video_id,
                    "document": summary,
                    "metadata": {
                        "record_level": "video",
                        "video_id": video_id,
                        "event_count": len(group),
                    },
                }
            )
        return records

    # --- Cross-camera global entity collection ---

    def _build_global_entity_records(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Build one Chroma record per global entity aggregated across all cameras.

        Groups events by ``global_entity_id`` (derived from ``entity_hint``).
        Events without a global_entity_id are skipped.
        Each record summarizes the entity's full trajectory across all camera views,
        enabling cross-camera queries like "what did person_global_1 do?".
        """
        grouped: dict[str, list[dict[str, Any]]] = {}
        for event in events:
            ge_id = event.get("global_entity_id")
            if not ge_id:
                continue
            grouped.setdefault(ge_id, []).append(event)

        records: list[dict[str, Any]] = []
        for ge_id, group in sorted(grouped.items()):
            start_time = self._safe_min([item.get("start_time") for item in group])
            end_time = self._safe_max([item.get("end_time") for item in group])
            appearance_notes = self._dedupe_text([item.get("appearance_notes") for item in group])
            scene_zones = self._dedupe_text([item.get("scene_zone") for item in group])
            event_texts = self._dedupe_text([item.get("event_text") for item in group])
            object_types = self._dedupe_text([item.get("object_type") for item in group])
            object_colors = self._dedupe_text([item.get("object_color") for item in group])
            camera_ids = self._dedupe_text([item.get("camera_id") for item in group])
            video_ids = self._dedupe_text([item.get("video_id") for item in group])
            keywords = self._dedupe_text(
                [keyword for item in group for keyword in item.get("keywords", [])]
            )
            document = self._build_global_entity_document(
                global_entity_id=ge_id,
                camera_ids=camera_ids,
                video_ids=video_ids,
                appearance_notes=appearance_notes,
                scene_zones=scene_zones,
                event_texts=event_texts,
                object_types=object_types,
                object_colors=object_colors,
                keywords=keywords,
                start_time=start_time,
                end_time=end_time,
                event_count=len(group),
            )
            records.append(
                {
                    "id": ge_id,
                    "document": document,
                    "metadata": {
                        "record_level": "global_entity",
                        "global_entity_id": ge_id,
                        "camera_ids": ", ".join(camera_ids),
                        "video_ids": ", ".join(video_ids),
                        "object_type": object_types[0] if object_types else "unknown",
                        "object_color": object_colors[0] if object_colors else "unknown",
                        "scene_zone": ", ".join(scene_zones) if scene_zones else "unknown",
                        "start_time": start_time if start_time is not None else -1.0,
                        "end_time": end_time if end_time is not None else -1.0,
                        "event_count": len(group),
                        "camera_count": len(camera_ids),
                        "keywords": ", ".join(keywords),
                    },
                }
            )
        return records

    @staticmethod
    def _build_global_entity_document(
        *,
        global_entity_id: str,
        camera_ids: list[str],
        video_ids: list[str],
        appearance_notes: list[str],
        scene_zones: list[str],
        event_texts: list[str],
        object_types: list[str],
        object_colors: list[str],
        keywords: list[str],
        start_time: float | None,
        end_time: float | None,
        event_count: int,
    ) -> str:
        # Build per-camera timeline summary to strengthen cross-camera matching
        per_camera_timeline: dict[str, tuple[float, float]] = {}
        sections = [
            f"Global entity {global_entity_id} observed across {len(camera_ids)} cameras.",
            f"Cameras: {', '.join(camera_ids)}.",
            f"Videos: {', '.join(video_ids)}.",
            f"Time range {ChromaIndexBuilder._format_time(start_time)}s to {ChromaIndexBuilder._format_time(end_time)}s.",
            f"{event_count} events total.",
        ]
        # Try to infer camera-local span from event_text fragments when present in
        # the aggregated payload (best effort only).
        for text in event_texts:
            # This parser is intentionally conservative.
            # Example patterns: "in camera G506 from 82.0s to 228.0s"
            low = text.lower()
            if "camera" not in low or "from" not in low or "to" not in low:
                continue
            try:
                cam_tok = text.split("camera", 1)[1].strip().split()[0].strip(".,;:")
                start_tok = text.split("from", 1)[1].split("s", 1)[0].strip().split()[-1]
                end_tok = text.split("to", 1)[1].split("s", 1)[0].strip().split()[0]
                st = float(start_tok)
                et = float(end_tok)
                prev = per_camera_timeline.get(cam_tok)
                if prev is None:
                    per_camera_timeline[cam_tok] = (st, et)
                else:
                    per_camera_timeline[cam_tok] = (min(prev[0], st), max(prev[1], et))
            except Exception:
                continue
        if per_camera_timeline:
            hops = "; ".join(
                f"{cam}:{ChromaIndexBuilder._format_time(st)}-{ChromaIndexBuilder._format_time(et)}s"
                for cam, (st, et) in per_camera_timeline.items()
            )
            sections.append("Cross-camera timeline: " + hops + ".")
        if object_types:
            sections.append("Object types: " + ", ".join(object_types) + ".")
        if object_colors:
            sections.append("Object colors: " + ", ".join(object_colors) + ".")
        if scene_zones:
            sections.append("Located in: " + ", ".join(scene_zones) + ".")
        if appearance_notes:
            sections.append("Appearance notes: " + " ".join(appearance_notes))
        if event_texts:
            sections.append("Events: " + " ".join(event_texts))
        if keywords:
            sections.append("Keywords: " + ", ".join(keywords) + ".")
        return " ".join(sections)

    @staticmethod
    def _build_parent_document(
        *,
        video_id: str,
        child_records: list[dict[str, Any]],
        scene_zones: list[str],
        object_types: list[str],
        object_colors: list[str],
        start_time: float | None,
        end_time: float | None,
        bucket_start: float = 0.0,
        bucket_end: float = 0.0,
    ) -> str:
        sections = [
            f"Video {video_id}.",
            f"Window {ChromaIndexBuilder._format_time(bucket_start)}s-{ChromaIndexBuilder._format_time(bucket_end)}s.",
            f"Video time range {ChromaIndexBuilder._format_time(start_time)}s to {ChromaIndexBuilder._format_time(end_time)}s.",
            f"This parent record summarizes {len(child_records)} child tracks in this window.",
        ]
        if object_types:
            sections.append("Object types: " + ", ".join(object_types) + ".")
        if object_colors:
            sections.append("Object colors: " + ", ".join(object_colors) + ".")
        if scene_zones:
            sections.append("Scene zones: " + ", ".join(scene_zones) + ".")
        child_summaries = [record["document"] for record in child_records]
        if child_summaries:
            sections.append("Child track summaries: " + " ".join(child_summaries))
        return " ".join(sections)

    def _build_event_records(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Build one Chroma record per single event for fine-grained temporal retrieval.

        Record id format: ``{video_id}:{entity_hint}:{start_time}:{end_time}[:seq]``
        which mirrors the ``vector_ref_id`` emitted by ``sqlite_builder`` so
        that SQLite rows can be joined back to Chroma event hits if needed.
        """
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for event in events:
            key = (event["video_id"], event["entity_hint"])
            grouped.setdefault(key, []).append(event)

        records: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for (video_id, entity_hint), group in sorted(grouped.items()):
            ordered = sorted(
                group,
                key=lambda item: (
                    float(item.get("start_time"))
                    if isinstance(item.get("start_time"), (int, float))
                    else float("inf"),
                    float(item.get("end_time"))
                    if isinstance(item.get("end_time"), (int, float))
                    else float("inf"),
                ),
            )
            parent_track_id = f"{video_id}_{entity_hint}"
            for event_index, event in enumerate(ordered):
                start_time = event.get("start_time")
                end_time = event.get("end_time")
                base_id = (
                    f"{video_id}:{entity_hint}:"
                    f"{self._format_time(start_time) if isinstance(start_time, (int, float)) else 'na'}:"
                    f"{self._format_time(end_time) if isinstance(end_time, (int, float)) else 'na'}"
                )
                record_id = base_id
                dedup_counter = 1
                while record_id in seen_ids:
                    record_id = f"{base_id}:{dedup_counter}"
                    dedup_counter += 1
                seen_ids.add(record_id)

                appearance_notes = str(event.get("appearance_notes") or "").strip()
                scene_zone = str(event.get("scene_zone") or "").strip()
                object_type = str(event.get("object_type") or "").strip()
                object_color = str(event.get("object_color") or "").strip()
                event_text = str(event.get("event_text") or "").strip()
                keywords = [str(kw).strip() for kw in event.get("keywords", []) if str(kw).strip()]
                camera_id = str(event.get("camera_id") or "").strip()
                global_entity_id = str(event.get("global_entity_id") or "").strip()
                raw_cross = event.get("cross_camera_ids")
                if isinstance(raw_cross, list):
                    cross_camera_ids = [str(c).strip() for c in raw_cross if str(c).strip()]
                elif isinstance(raw_cross, str):
                    cross_camera_ids = [p.strip() for p in raw_cross.split(",") if p.strip()]
                else:
                    cross_camera_ids = []

                document = self._build_event_document(
                    video_id=video_id,
                    entity_hint=entity_hint,
                    camera_id=camera_id,
                    global_entity_id=global_entity_id,
                    cross_camera_ids=cross_camera_ids,
                    start_time=start_time,
                    end_time=end_time,
                    object_type=object_type,
                    object_color=object_color,
                    scene_zone=scene_zone,
                    appearance_notes=appearance_notes,
                    event_text=event_text,
                    keywords=keywords,
                )

                records.append(
                    {
                        "id": record_id,
                        "document": document,
                        "metadata": {
                            "record_level": "event",
                            "video_id": video_id,
                            "parent_track_id": parent_track_id,
                            "grand_parent_video_id": video_id,
                            "entity_hint": entity_hint,
                            "camera_id": camera_id or "unknown",
                            "global_entity_id": global_entity_id or "",
                            "cross_camera_ids": ", ".join(cross_camera_ids),
                            "event_index": event_index,
                            "event_id": self._lookup_event_id(video_id, entity_hint, start_time, end_time),
                            "object_type": object_type or "unknown",
                            "object_color": object_color or "unknown",
                            "scene_zone": scene_zone or "unknown",
                            "start_time": float(start_time) if isinstance(start_time, (int, float)) else -1.0,
                            "end_time": float(end_time) if isinstance(end_time, (int, float)) else -1.0,
                            "keywords": ", ".join(keywords),
                        },
                    }
                )
        return records

    @staticmethod
    def _build_event_document(
        *,
        video_id: str,
        entity_hint: str,
        camera_id: str,
        global_entity_id: str,
        cross_camera_ids: list[str],
        start_time: float | None,
        end_time: float | None,
        object_type: str,
        object_color: str,
        scene_zone: str,
        appearance_notes: str,
        event_text: str,
        keywords: list[str],
    ) -> str:
        sections = [
            f"Video {video_id}. Track {entity_hint}.",
            f"Event time range {ChromaIndexBuilder._format_time(start_time)}s to {ChromaIndexBuilder._format_time(end_time)}s.",
        ]
        if camera_id:
            sections.append(f"Camera {camera_id}.")
        if global_entity_id:
            sections.append(f"Global entity {global_entity_id}.")
        if cross_camera_ids:
            sections.append("Also linked to cameras: " + ", ".join(cross_camera_ids) + ".")
        subject_parts: list[str] = []
        if object_color:
            subject_parts.append(object_color)
        if object_type:
            subject_parts.append(object_type)
        if subject_parts:
            sections.append("Subject: " + " ".join(subject_parts) + ".")
        if scene_zone:
            sections.append(f"Located in: {scene_zone}.")
        if appearance_notes:
            sections.append(f"Appearance notes: {appearance_notes}")
        if event_text:
            sections.append(f"Event: {event_text}")
        if keywords:
            sections.append("Keywords: " + ", ".join(keywords) + ".")
        return " ".join(sections)

    # --- SQLite event_id backfill ---

    def _lookup_event_id(self, video_id: str, entity_hint: str, start_time: Any, end_time: Any) -> int | None:
        """Look up the SQLite ``event_id`` for a single event by its natural key.

        Returns ``None`` when the SQLite database is not available (e.g. Chroma
        built independently before SQLite).
        """
        if self.sqlite_db_path is None or not self.sqlite_db_path.exists():
            return None
        import sqlite3

        try:
            st = float(start_time) if isinstance(start_time, (int, float)) else None
            et = float(end_time) if isinstance(end_time, (int, float)) else None
        except (ValueError, TypeError):
            return None
        if st is None and et is None:
            return None

        try:
            with sqlite3.connect(str(self.sqlite_db_path)) as conn:
                conditions = ["video_id = ?", "(entity_hint = ? OR track_id = ?)"]
                params: list[Any] = [video_id, entity_hint, entity_hint]
                if st is not None:
                    conditions.append("ABS(start_time - ?) < 0.01")
                    params.append(st)
                if et is not None:
                    conditions.append("ABS(end_time - ?) < 0.01")
                    params.append(et)
                where = " AND ".join(conditions)
                row = conn.execute(
                    f"SELECT event_id FROM episodic_events WHERE {where} LIMIT 1", params
                ).fetchone()
                if row is not None:
                    return int(row[0])
        except Exception:
            pass
        return None

    def _lookup_event_ids_for_track(self, video_id: str, entity_hint: str) -> list[int]:
        """Look up all SQLite ``event_id`` values for a given track."""
        if self.sqlite_db_path is None or not self.sqlite_db_path.exists():
            return []
        import sqlite3

        try:
            with sqlite3.connect(str(self.sqlite_db_path)) as conn:
                rows = conn.execute(
                    "SELECT event_id FROM episodic_events "
                    "WHERE video_id = ? AND (entity_hint = ? OR track_id = ?) "
                    "ORDER BY start_time ASC",
                    (video_id, entity_hint, entity_hint),
                ).fetchall()
                return [int(r[0]) for r in rows]
        except Exception:
            return []
