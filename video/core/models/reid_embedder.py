"""Pedestrian Re-ID feature extractor.

Defaults to torchreid OSNet x0_5 (lightweight CNN); falls back to torchvision MobileNetV2 if unavailable.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_REID_INPUT_SIZE = (256, 128)  # (H, W)

try:
    from torchreid.reid import models as _torchreid_models
    _HAS_TORCHREID = True
except ImportError:
    _HAS_TORCHREID = False


def _resolve_device(device: str) -> str:
    """Resolve device at runtime; fall back to CPU when unavailable.

    torch may mis-report MPS on newer macOS; we probe with a small tensor when device is mps.
    """
    if device == "mps":
        try:
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                torch.zeros(1).to("mps")
                return "mps"
        except Exception:
            pass
        logger.warning("MPS unavailable; falling back to CPU")
        return "cpu"
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            logger.warning("CUDA unavailable; falling back to CPU")
            return "cpu"
    return device


class ReIDEmbedder:
    """Wraps Re-ID inference; exposes embed_crops and cosine_similarity.

    backend:
        "torchreid_osnet" — torchreid OSNet x0_5 (default)
        "torchvision"     — MobileNetV2 (ImageNet)
    """

    def __init__(
        self,
        config_file: str | Path | None = None,
        weights: str | Path | None = None,
        device: str = "cpu",
        input_size: tuple[int, int] = _REID_INPUT_SIZE,
        backend: str | None = None,
    ) -> None:
        self.device = _resolve_device(device)
        self.input_size = input_size  # (H, W)
        self._backend = self._resolve_backend(backend, config_file, weights)
        self._model = self._load_model(config_file, weights, self.device)

    # ------------------------------------------------------------------
    # Backend resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_backend(
        backend: str | None,
        config_file: str | Path | None,
        weights: str | Path | None,
    ) -> str:
        if backend is not None:
            return backend
        if _HAS_TORCHREID:
            return "torchreid_osnet"
        return "torchvision"

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(
        self,
        config_file: str | Path | None,
        weights: str | Path | None,
        device: str,
    ) -> torch.nn.Module:
        if self._backend == "torchreid_osnet":
            return self._load_torchreid_osnet(device)
        return self._load_torchvision(device)

    # osnet_x0_5  — lightweight  (~1.6M params, faster)
    # osnet_x0_75 — balanced     (~2.4M params)
    # osnet_x1_0  — full-size    (~2.2M params, best accuracy)
    _OSNET_VARIANT: str = "osnet_x1_0"

    @classmethod
    def _load_torchreid_osnet(cls, device: str) -> torch.nn.Module:
        model = _torchreid_models.build_model(
            name=cls._OSNET_VARIANT,
            num_classes=1000,
            pretrained=True,
            use_gpu=False,
        )
        model.to(device)
        model.eval()
        logger.info("torchreid %s loaded", cls._OSNET_VARIANT)
        return model

    @staticmethod
    def _load_torchvision(device: str) -> torch.nn.Module:
        from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

        base = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)
        # Drop classifier head; keep features + global avg pool → 1280-d
        model = torch.nn.Sequential(
            base.features,
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
        )
        model.to(device)
        model.eval()
        logger.info("Torchvision MobileNetV2 fallback loaded (1280-d features)")
        return model

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _preprocess(self, imgs: Sequence[np.ndarray]) -> torch.Tensor:
        """BGR ndarray list -> (N, 3, H, W) float32 tensor (preprocessed per backend)."""
        tensors: list[torch.Tensor] = []
        h, w = self.input_size
        for img in imgs:
            rgb = img[:, :, ::-1].copy()
            resized = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_CUBIC)
            t = torch.as_tensor(resized.astype("float32").transpose(2, 0, 1))
            if self._backend == "torchvision":
                t = t / 255.0
                t = torch.stack(
                    [
                        (t[0] - 0.485) / 0.229,
                        (t[1] - 0.456) / 0.224,
                        (t[2] - 0.406) / 0.225,
                    ]
                )
            elif self._backend == "torchreid_osnet":
                t = t / 255.0
                t = torch.stack(
                    [
                        (t[0] - 0.485) / 0.229,
                        (t[1] - 0.456) / 0.224,
                        (t[2] - 0.406) / 0.225,
                    ]
                )
            tensors.append(t)
        return torch.stack(tensors)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def embed_crops(self, crops: list[np.ndarray], batch_size: int = 64) -> np.ndarray:
        """Extract features in batches; returns L2-normalized vectors of shape (N, dim)."""
        if not crops:
            return np.empty((0, 0), dtype=np.float32)

        all_feats: list[torch.Tensor] = []
        for i in range(0, len(crops), batch_size):
            batch = crops[i : i + batch_size]
            inp = self._preprocess(batch).to(self.device)
            feats = self._model(inp)
            feats = F.normalize(feats, dim=1)
            all_feats.append(feats.cpu())

        return torch.cat(all_feats, dim=0).numpy()

    @staticmethod
    def cosine_similarity(feats_a: np.ndarray, feats_b: np.ndarray) -> np.ndarray:
        """(Na, dim) x (Nb, dim) -> (Na, Nb) cosine similarity matrix."""
        return feats_a @ feats_b.T
