from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def extract_patch(spectrogram_values: np.ndarray, coords: Dict[str, int]) -> Optional[np.ndarray]:
    """Extract a patch safely from a spectrogram array."""
    if spectrogram_values.size == 0:
        return None

    max_y, max_x = spectrogram_values.shape
    x_start = int(np.clip(coords["x_start"], 0, max_x - 1))
    x_end = int(np.clip(max(coords["x_end"], x_start + 1), 1, max_x))
    y_low = int(np.clip(coords["y_low"], 0, max_y - 1))
    y_high = int(np.clip(max(coords["y_high"], y_low + 1), 1, max_y))

    patch = spectrogram_values[y_low:y_high, x_start:x_end]
    if patch.size == 0:
        return None

    return patch
