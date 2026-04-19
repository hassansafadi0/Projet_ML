from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample

# Allow running this file directly with `python src/model.py`.
if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.dataset_builder import CLASS_ORDER

DEFAULT_PCA_COMPONENTS = 0


def _ordered_labels(labels: Sequence[str]) -> List[str]:
    label_set = set(labels)
    ordered = [label for label in CLASS_ORDER if label in label_set]
    extras = sorted(label for label in label_set if label not in CLASS_ORDER)
    return [*ordered, *extras]


def _print_distribution(labels: np.ndarray, title: str) -> None:
    unique_labels, counts = np.unique(labels, return_counts=True)
    print(title)
    for label, count in zip(unique_labels, counts):
        print(f"  {label}: {count}")


def _as_float32(array: np.ndarray) -> np.ndarray:
    return np.asarray(array, dtype=np.float32)


def oversample_minority(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    classes = np.unique(y)
    max_count = max(np.sum(y == current_class) for current_class in classes)

    X_resampled = []
    y_resampled = []

    for current_class in classes:
        class_mask = y == current_class
        X_class = X[class_mask]
        y_class = y[class_mask]

        if len(X_class) < max_count:
            X_class_resampled, y_class_resampled = resample(
                X_class,
                y_class,
                replace=True,
                n_samples=max_count,
                random_state=42,
            )
        else:
            X_class_resampled, y_class_resampled = X_class, y_class

        X_resampled.append(X_class_resampled)
        y_resampled.append(y_class_resampled)

    X_balanced = np.vstack(X_resampled)
    y_balanced = np.concatenate(y_resampled)
    return _as_float32(X_balanced), y_balanced


def plot_confusion_matrix(y_test: np.ndarray, y_pred: np.ndarray, labels: Sequence[str]) -> None:
    """Display the standard confusion matrix."""
    labels = _ordered_labels(labels)
    confusion = confusion_matrix(y_test, y_pred, labels=labels)

    figure, axis = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=labels).plot(
        ax=axis,
        cmap="Blues",
        colorbar=False,
        values_format="d",
    )
    axis.set_title("Confusion Matrix")
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    figure.tight_layout()


def plot_confusion_matrix_normalized(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    labels: Sequence[str],
) -> None:
    """Display the confusion matrix normalized by true labels."""
    labels = _ordered_labels(labels)
    normalized_confusion = confusion_matrix(
        y_test,
        y_pred,
        labels=labels,
        normalize="true",
    )

    figure, axis = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(
        confusion_matrix=normalized_confusion,
        display_labels=labels,
    ).plot(
        ax=axis,
        cmap="Blues",
        colorbar=False,
        values_format=".2f",
    )
    axis.set_title("Normalized Confusion Matrix")
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    figure.tight_layout()


def _build_model() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight={
            "Bp-40Down": 3,
            "Bm-Z": 2,
            "Bm-B": 1.5,
        },
        n_jobs=-1,
        random_state=42,
    )


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    all_labels: Sequence[str] | None = None,
    model_name: str = "logistic_regression",
    pca_components: int | None = DEFAULT_PCA_COMPONENTS,
) -> Tuple[object, Dict[str, float]]:
    """
    Train on the provided training set and evaluate on the untouched validation set.
    """
    del pca_components

    X_train = _as_float32(X_train)
    X_val = _as_float32(X_val)
    y_train = np.asarray(y_train)
    y_val = np.asarray(y_val)

    unique_labels = np.unique(y_train)
    if len(unique_labels) < 2:
        raise ValueError("At least two classes are required to train the model.")

    if len(X_val) != len(y_val):
        raise ValueError("X_val and y_val must contain the same number of samples.")

    if len(y_val) == 0:
        raise ValueError("Validation data must not be empty.")

    ordered_all_labels = _ordered_labels(
        all_labels if all_labels is not None else np.concatenate([y_train, y_val]).tolist()
    )

    normalized_model_name = model_name.strip().lower()
    if normalized_model_name != "logistic_regression":
        print(
            f"Requested model '{model_name}' is not used. "
            "Training with the fixed random_forest pipeline."
        )

    _print_distribution(y_train, "Training class distribution:")
    _print_distribution(y_val, "Validation class distribution:")
    print(f"All labels passed to training/reporting: {ordered_all_labels}")

    X_train, y_train = oversample_minority(X_train, y_train)
    print(f"Resampled training samples: {len(X_train)}")
    _print_distribution(y_train, "Training class distribution after oversampling:")

    model = _build_model()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)

    print("Model: random_forest")
    print(f"Train samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Features per patch: {X_train.shape[1]}")
    print(f"Input dtype: train={X_train.dtype}, validation={X_val.dtype}")
    print("StandardScaler: disabled")
    print("PCA: disabled")
    print(
        "RandomForestClassifier: "
        "n_estimators=300, max_depth=None, min_samples_split=2, "
        "min_samples_leaf=2, max_features=sqrt, "
        "class_weight={Bp-40Down: 5, Bm-Z: 2, Bm-B: 1.5}, n_jobs=-1"
    )
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification report:")
    print(
        classification_report(
            y_val,
            y_pred,
            labels=ordered_all_labels,
            zero_division=0,
        )
    )

    plot_confusion_matrix(y_val, y_pred, ordered_all_labels)
    plot_confusion_matrix_normalized(y_val, y_pred, ordered_all_labels)
    plt.show()

    metrics = {"accuracy": float(accuracy)}
    return model, metrics
