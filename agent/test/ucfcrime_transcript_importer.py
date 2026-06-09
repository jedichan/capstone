from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_TRANSCRIPT_JSON = ROOT_DIR / "agent" / "test" / "data" / "UCFCrime_Test.json"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "agent" / "test" / "generated" / "ucfcrime_events_vector_flat"
DEFAULT_MANIFEST_PATH = ROOT_DIR / "agent" / "test" / "generated" / "ucfcrime_events_manifest.json"


@dataclass
class UcfCrimeTranscriptImportConfig:
    transcript_json_path: Path = DEFAULT_TRANSCRIPT_JSON
    output_dir: Path = DEFAULT_OUTPUT_DIR
    manifest_path: Path = DEFAULT_MANIFEST_PATH


class UcfCrimeTranscriptImportError(RuntimeError):
    pass


OBJECT_RULES = [
    ("police", ["police", "officer", "uniform"]),
    ("car", ["car", "vehicle", "truck", "suv"]),
    ("person", ["man", "woman", "people", "person", "adult", "lady"]),
    ("dog", ["dog", "puppy"]),
    ("child", ["baby", "child"]),
    ("bicycle", ["bicycle", "bike", "tricycle"]),
]

COLOR_RULES = [
    "black",
    "white",
    "blue",
    "red",
    "green",
    "yellow",
    "gray",
    "silver",
    "pink",
    "purple",
    "blond",
]

ZONE_RULES = [
    ("road", ["road", "street", "intersection"]),
    ("room", ["room", "wall", "door", "sofa", "table"]),
    ("store", ["store", "supermarket", "counter"]),
    ("yard", ["yard"]),
    ("bedside", ["bed", "crib", "pillow"]),
]

STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "to",
    "of",
    "in",
    "on",
    "with",
    "by",
    "for",
    "then",
    "there",
    "was",
    "were",
    "is",
    "are",
    "it",
    "his",
    "her",
    "their",
    "they",
    "them",
    "this",
    "that",
    "into",
    "from",
    "after",
    "before",
    "over",
    "under",
    "next",
    "just",
}


class UcfCrimeTranscriptImporter:
    def __init__(self, config: UcfCrimeTranscriptImportConfig) -> None:
        self.config = config

    def build(self) -> dict[str, Any]:
        if not self.config.transcript_json_path.exists():
            raise UcfCrimeTranscriptImportError(
                f"Transcript json not found: {self.config.transcript_json_path}"
            )

        payload = json.loads(self.config.transcript_json_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise UcfCrimeTranscriptImportError("Transcript json must be a dict keyed by video_id")

        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        manifest_items: list[dict[str, Any]] = []
        total_events = 0
        for video_id, item in payload.items():
            normalized = self._build_video_events(video_id, item)
            out_path = self.config.output_dir / f"{video_id}_events_vector_flat.json"
            out_path.write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")
            manifest_items.append(
                {
                    "video_id": video_id,
                    "duration": normalized.get("duration"),
                    "event_count": len(normalized.get("events", [])),
                    "output_path": str(out_path),
                }
            )
            total_events += len(normalized.get("events", []))

        manifest = {
            "source_json": str(self.config.transcript_json_path),
            "output_dir": str(self.config.output_dir),
            "video_count": len(manifest_items),
            "total_event_count": total_events,
            "items": manifest_items,
        }
        self.config.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return manifest

    def _build_video_events(self, video_id: str, item: Any) -> dict[str, Any]:
        if not isinstance(item, dict):
            raise UcfCrimeTranscriptImportError(f"Bad transcript item for video {video_id}")
        duration = self._safe_float(item.get("duration")) or 0.0
        timestamps = item.get("timestamps") or []
        sentences = item.get("sentences") or []
        if len(timestamps) != len(sentences):
            raise UcfCrimeTranscriptImportError(
                f"Timestamps and sentences length mismatch for video {video_id}"
            )

        events: list[dict[str, Any]] = []
        for idx, (time_pair, sentence) in enumerate(zip(timestamps, sentences), start=1):
            start_time, end_time = self._normalize_timestamp_pair(time_pair)
            text = self._normalize_sentence(sentence)
            if not text:
                continue
            object_type = self._infer_object_type(text)
            events.append(
                {
                    "video_id": video_id,
                    "clip_start_sec": 0.0,
                    "clip_end_sec": duration,
                    "start_time": start_time,
                    "end_time": end_time,
                    "object_type": object_type,
                    "object_color": self._infer_color(text, object_type),
                    "appearance_notes": self._build_appearance_notes(text),
                    "scene_zone": self._infer_scene_zone(text),
                    "event_text": self._build_event_text(start_time, end_time, text),
                    "keywords": self._extract_keywords(text),
                    "start_bbox_xyxy": [None, None, None, None],
                    "end_bbox_xyxy": [None, None, None, None],
                    "entity_hint": f"segment_{idx}",
                }
            )
        return {
            "video_id": video_id,
            "duration": duration,
            "events": events,
        }

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        try:
            return float(value)
        except Exception:
            return None

    @staticmethod
    def _normalize_timestamp_pair(value: Any) -> tuple[float | None, float | None]:
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            start_time = UcfCrimeTranscriptImporter._safe_float(value[0])
            end_time = UcfCrimeTranscriptImporter._safe_float(value[1])
            return start_time, end_time
        return None, None

    @staticmethod
    def _normalize_sentence(text: Any) -> str:
        cleaned = str(text or "").replace("##", " ").strip()
        cleaned = re.sub(r"\s+", " ", cleaned)
        if cleaned and cleaned[-1] not in ".!?":
            cleaned += "."
        return cleaned

    @staticmethod
    def _infer_object_type(text: str) -> str:
        lower = text.lower()
        for label, patterns in OBJECT_RULES:
            if any(token in lower for token in patterns):
                return label
        return "unknown"

    @staticmethod
    def _infer_color(text: str, object_type: str) -> str:
        lower = text.lower()
        object_phrases = {
            "car": ["car", "vehicle", "truck", "suv"],
            "dog": ["dog", "puppy"],
            "person": ["man", "woman", "person", "people", "lady", "adult"],
            "child": ["baby", "child"],
            "police": ["police", "officer", "uniform"],
            "bicycle": ["bicycle", "bike", "tricycle"],
        }
        for color in COLOR_RULES:
            normalized_color = "yellow" if color == "blond" else color
            for noun in object_phrases.get(object_type, []):
                if f"{color} {noun}" in lower:
                    return normalized_color
        for color in COLOR_RULES:
            if color in lower:
                return "yellow" if color == "blond" else color
        return "unknown"

    @staticmethod
    def _infer_scene_zone(text: str) -> str:
        lower = text.lower()
        for zone, patterns in ZONE_RULES:
            if any(token in lower for token in patterns):
                return zone
        return "unknown"

    @staticmethod
    def _build_appearance_notes(text: str) -> str:
        lower = text.lower()
        if any(token in lower for token in ["hit", "beat", "push", "burn", "crush", "fight", "struggle"]):
            prefix = "High activity;"
        elif any(token in lower for token in ["stand", "look", "watch", "wait", "parked"]):
            prefix = "Low activity;"
        else:
            prefix = "Observed event;"
        return f"{prefix} {text}"

    @staticmethod
    def _build_event_text(start_time: float | None, end_time: float | None, text: str) -> str:
        if isinstance(start_time, (int, float)) and isinstance(end_time, (int, float)):
            return f"From {start_time:.1f}s to {end_time:.1f}s, {text[0].lower() + text[1:] if len(text) > 1 else text.lower()}"
        return text

    @staticmethod
    def _extract_keywords(text: str, limit: int = 6) -> list[str]:
        tokens = re.findall(r"[a-z0-9_]+", text.lower())
        out: list[str] = []
        for token in tokens:
            if len(token) <= 2 or token in STOPWORDS:
                continue
            if token not in out:
                out.append(token)
            if len(out) >= limit:
                break
        return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert UCF Crime transcript json into events_vector_flat-like files"
    )
    parser.add_argument(
        "--transcript-json",
        type=str,
        default=str(DEFAULT_TRANSCRIPT_JSON),
        help="Source transcript json path",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for normalized per-video json files",
    )
    parser.add_argument(
        "--manifest-path",
        type=str,
        default=str(DEFAULT_MANIFEST_PATH),
        help="Manifest output path",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = UcfCrimeTranscriptImportConfig(
        transcript_json_path=Path(args.transcript_json).expanduser().resolve(),
        output_dir=Path(args.output_dir).expanduser().resolve(),
        manifest_path=Path(args.manifest_path).expanduser().resolve(),
    )
    importer = UcfCrimeTranscriptImporter(config)
    result = importer.build()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
