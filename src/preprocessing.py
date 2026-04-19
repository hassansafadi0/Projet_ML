from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import zoom


TARGET_PATCH_SIZE: Tuple[int, int] = (64, 64)
MIN_PATCH_HEIGHT = 2
MIN_PATCH_WIDTH = 2


def _resize_patch(patch: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    zoom_factors = (
        target_size[0] / patch.shape[0],
        target_size[1] / patch.shape[1],
    )
    resized_patch = zoom(patch, zoom_factors, order=1)
    return resized_patch.astype(np.float32, copy=False)


def _normalize_patch(patch: np.ndarray) -> np.ndarray:
    patch_min = float(np.min(patch))
    patch_max = float(np.max(patch))

    if patch_max <= patch_min:
        return np.zeros_like(patch, dtype=np.float32)

    normalized_patch = (patch - patch_min) / (patch_max - patch_min)
    return normalized_patch.astype(np.float32, copy=False)


def preprocess_patch(patch: np.ndarray) -> Optional[np.ndarray]:
    """Resize and normalize one spectrogram patch for classical ML."""
    if patch is None or patch.size == 0:
        return None

    if patch.shape[0] < MIN_PATCH_HEIGHT or patch.shape[1] < MIN_PATCH_WIDTH:
        return None

    resized_patch = _resize_patch(patch, TARGET_PATCH_SIZE)
    return _normalize_patch(resized_patch)
