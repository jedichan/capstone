"""Repository root and conventional paths (config, default output dirs, etc.)."""

from pathlib import Path


def repo_root() -> Path:
    """This file is video/common/paths.py; two levels up is the repo root."""
    return Path(__file__).resolve().parents[2]


def botsort_reid_config_path() -> Path:
    return repo_root() / "config" / "trackers" / "botsort_reid.yaml"


def pipeline_output_dir() -> Path:
    return repo_root() / "pipeline_output"


def yolo_model_dir() -> Path:
    return repo_root() / "_model"
