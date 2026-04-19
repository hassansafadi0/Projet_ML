from __future__ import annotations

from collections import Counter
from typing import Tuple

import numpy as np
from sklearn.utils import resample

from src.dataset_builder import CLASS_ORDER


def _print_distribution(y: np.ndarray, title: str) -> None:
    counts = Counter(y.tolist())
    print(title)
    for label in CLASS_ORDER:
        if label in counts:
            print(f"  {label}: {counts[label]}")


def balance_dataset(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Upsample minority classes so every class reaches the majority count."""
    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of samples.")

    if len(y) == 0:
        raise ValueError("Cannot balance an empty dataset.")

    unique_labels, label_counts = np.unique(y, return_counts=True)
    if len(unique_labels) < 2:
        raise ValueError("Balancing requires at least two classes.")

    max_count = int(np.max(label_counts))
    balanced_features = []
    balanced_labels = []

    _print_distribution(y, "Class distribution before balancing:")

    for label in unique_labels:
        class_features = X[y == label]
        class_labels = y[y == label]

        if len(class_features) == 0:
            continue

        if len(class_features) < max_count:
            resampled_features, resampled_labels = resample(
                class_features,
                class_labels,
                replace=True,
                n_samples=max_count,
                random_state=42,
            )
        else:
            resampled_features = class_features
            resampled_labels = class_labels

        balanced_features.append(resampled_features)
        balanced_labels.append(resampled_labels)

    X_balanced = np.vstack(balanced_features).astype(np.float32)
    y_balanced = np.concatenate(balanced_labels)

    shuffle_indices = np.random.default_rng(42).permutation(len(y_balanced))
    X_balanced = X_balanced[shuffle_indices]
    y_balanced = y_balanced[shuffle_indices]

    _print_distribution(y_balanced, "Class distribution after balancing:")
    return X_balanced, y_balanced
