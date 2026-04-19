from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np


def save_dataset(X: np.ndarray, y: np.ndarray, path: str | Path) -> None:
    """Persist one dataset split to cache."""
    cache_path = Path(path)
    cache_path.mkdir(parents=True, exist_ok=True)
    np.save(cache_path / "X.npy", X)
    np.save(cache_path / "y.npy", y)


def load_dataset(path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load one dataset split from cache."""
    cache_path = Path(path)
    X = np.load(cache_path / "X.npy", allow_pickle=False)
    y = np.load(cache_path / "y.npy", allow_pickle=False)
    return X, y
