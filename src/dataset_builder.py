from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.annotation_processing import convert_annotation_to_pixels
from src.audio_loader import load_audio
from src.patch_extraction import extract_patch
from src.preprocessing import preprocess_patch
from src.spectrogram import compute_spectrogram


CLASS_LABELS: Dict[str, str] = {
    "bma": "Bm-A",
    "bmb": "Bm-B",
    "bmz": "Bm-Z",
    "bmd": "Bm-D",
    "bp20": "Bp-20",
    "bp20plus": "Bp-20Plus",
    "bpd": "Bp-40Down",
}

CLASS_ORDER: List[str] = [
    "Bm-A",
    "Bm-B",
    "Bm-Z",
    "Bm-D",
    "Bp-20",
    "Bp-20Plus",
    "Bp-40Down",
]

DEFAULT_NUM_WORKERS = max(1, min(4, os.cpu_count() or 1))


def _normalize_annotation_label(annotation_value: str) -> Optional[str]:
    annotation_key = str(annotation_value).strip().lower().replace("-", "")
    return CLASS_LABELS.get(annotation_key)


def _resolve_audio_path(audio_folder: str | Path, annotation_row: pd.Series) -> Optional[Path]:
    audio_folder = Path(audio_folder)
    dataset_name = str(annotation_row.get("dataset", "")).strip()
    filename = str(annotation_row.get("filename", "")).strip()

    if not filename:
        return None

    direct_path = audio_folder / dataset_name / filename
    if direct_path.exists():
        return direct_path

    matches = list(audio_folder.rglob(filename))
    if matches:
        return matches[0]

    return None


def _process_annotation_group(
    group_index: int,
    audio_root: str | Path,
    group_df: pd.DataFrame,
    max_duration: Optional[float] = None,
) -> Tuple[int, Optional[np.ndarray], List[str], int, int, int]:
    first_row = group_df.iloc[0]
    audio_path = _resolve_audio_path(audio_root, first_row)
    if audio_path is None:
        return group_index, None, [], len(group_df), 0, 0

    try:
        sample_rate, audio_signal = load_audio(audio_path, max_duration=max_duration)
        freq_axis, time_axis, spectrogram_values = compute_spectrogram(audio_signal, sample_rate)
    except (FileNotFoundError, OSError, ValueError):
        return group_index, None, [], len(group_df), 0, 0

    feature_rows: List[np.ndarray] = []
    labels: List[str] = []
    skipped_invalid_annotations = 0
    skipped_invalid_patches = 0

    for annotation_row in group_df.itertuples(index=False):
        label = _normalize_annotation_label(annotation_row.annotation)
        if label is None:
            skipped_invalid_annotations += 1
            continue

        patch_coords = convert_annotation_to_pixels(
            annotation_row,
            spectrogram_shape=spectrogram_values.shape,
            time_axis=time_axis,
            freq_axis=freq_axis,
        )
        if patch_coords is None:
            skipped_invalid_annotations += 1
            continue

        patch = extract_patch(spectrogram_values, patch_coords)
        processed_patch = preprocess_patch(patch)
        if processed_patch is None:
            skipped_invalid_patches += 1
            continue

        feature_rows.append(processed_patch.reshape(-1))
        labels.append(label)

    if not feature_rows:
        return group_index, None, labels, 0, skipped_invalid_annotations, skipped_invalid_patches

    return (
        group_index,
        np.asarray(feature_rows, dtype=np.float32),
        labels,
        0,
        skipped_invalid_annotations,
        skipped_invalid_patches,
    )


def _load_annotations_from_folder(annotations_root: str | Path) -> pd.DataFrame:
    annotations_root = Path(annotations_root)
    csv_files = sorted(annotations_root.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No annotation CSV files found in: {annotations_root}")

    dataframes: List[pd.DataFrame] = []
    for csv_file in csv_files:
        dataframe = pd.read_csv(csv_file)
        dataframe["source_csv"] = csv_file.name
        dataframes.append(dataframe)

    annotations_df = pd.concat(dataframes, ignore_index=True)
    return annotations_df.dropna(
        subset=[
            "dataset",
            "filename",
            "annotation",
            "low_frequency",
            "high_frequency",
            "start_datetime",
            "end_datetime",
        ]
    ).copy()


def _build_dataset_from_annotations(
    audio_root: str | Path,
    annotations_df: pd.DataFrame,
    max_duration: Optional[float] = None,
    num_workers: int = DEFAULT_NUM_WORKERS,
) -> Tuple[np.ndarray, np.ndarray]:
    feature_batches: List[np.ndarray] = []
    label_batches: List[np.ndarray] = []
    skipped_missing_audio = 0
    skipped_invalid_annotations = 0
    skipped_invalid_patches = 0

    grouped_annotations = annotations_df.groupby(["dataset", "filename"], sort=False)
    total_audio_files = grouped_annotations.ngroups
    build_start = perf_counter()
    group_items = list(grouped_annotations)
    ordered_results: List[Optional[Tuple[np.ndarray, List[str]]]] = [None] * total_audio_files

    def _log_progress(processed_groups: int, kept_patches: int) -> None:
        if processed_groups == 1 or processed_groups % 25 == 0 or processed_groups == total_audio_files:
            elapsed_seconds = perf_counter() - build_start
            processed_ratio = processed_groups / total_audio_files if total_audio_files else 1.0
            estimated_total = elapsed_seconds / processed_ratio if processed_ratio > 0 else elapsed_seconds
            remaining_seconds = max(0.0, estimated_total - elapsed_seconds)
            print(
                "Processed "
                f"{processed_groups}/{total_audio_files} audio files "
                f"({processed_ratio:.1%}) | "
                f"patches kept: {kept_patches} | "
                f"elapsed: {elapsed_seconds / 60.0:.1f} min | "
                f"ETA: {remaining_seconds / 60.0:.1f} min"
            )

    num_workers = max(1, int(num_workers))

    if num_workers == 1:
        for group_index, (_, group_df) in enumerate(group_items, start=1):
            (
                _,
                group_features,
                group_labels,
                group_missing_audio,
                group_invalid_annotations,
                group_invalid_patches,
            ) = _process_annotation_group(
                group_index=group_index,
                audio_root=audio_root,
                group_df=group_df,
                max_duration=max_duration,
            )
            skipped_missing_audio += group_missing_audio
            skipped_invalid_annotations += group_invalid_annotations
            skipped_invalid_patches += group_invalid_patches

            if group_features is not None and len(group_labels) > 0:
                feature_batches.append(group_features)
                label_batches.append(np.asarray(group_labels))

            kept_patches = sum(batch.shape[0] for batch in feature_batches)
            _log_progress(group_index, kept_patches)
    else:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_group_index = {
                executor.submit(
                    _process_annotation_group,
                    group_index,
                    audio_root,
                    group_df,
                    max_duration,
                ): group_index
                for group_index, (_, group_df) in enumerate(group_items, start=1)
            }

            kept_patches = 0
            processed_groups = 0

            for future in as_completed(future_to_group_index):
                (
                    group_index,
                    group_features,
                    group_labels,
                    group_missing_audio,
                    group_invalid_annotations,
                    group_invalid_patches,
                ) = future.result()
                processed_groups += 1
                skipped_missing_audio += group_missing_audio
                skipped_invalid_annotations += group_invalid_annotations
                skipped_invalid_patches += group_invalid_patches

                if group_features is not None and len(group_labels) > 0:
                    ordered_results[group_index - 1] = (group_features, group_labels)
                    kept_patches += group_features.shape[0]

                _log_progress(processed_groups, kept_patches)

        for group_result in ordered_results:
            if group_result is None:
                continue

            group_features, group_labels = group_result
            feature_batches.append(group_features)
            label_batches.append(np.asarray(group_labels))

    if not feature_batches:
        raise ValueError("No valid spectrogram patches were extracted from the provided dataset.")

    total_patches = sum(batch.shape[0] for batch in feature_batches)
    print(f"Built raw dataset with {total_patches} valid patches.")
    print(f"Skipped annotations with missing audio: {skipped_missing_audio}")
    print(f"Skipped invalid annotations: {skipped_invalid_annotations}")
    print(f"Skipped empty or very small patches: {skipped_invalid_patches}")

    X = np.concatenate(feature_batches, axis=0).astype(np.float32, copy=False)
    y = np.concatenate(label_batches, axis=0)
    return X, y


def build_dataset_from_split(
    audio_root: str | Path,
    annotations_root: str | Path,
    max_duration: Optional[float] = None,
    num_workers: int = DEFAULT_NUM_WORKERS,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build one dataset from all annotation CSVs under a split root.

    The spectrogram is computed once per audio file and reused for all matching
    annotations from that recording.
    """
    annotations_df = _load_annotations_from_folder(annotations_root)
    return _build_dataset_from_annotations(
        audio_root=audio_root,
        annotations_df=annotations_df,
        max_duration=max_duration,
        num_workers=num_workers,
    )


def build_dataset(
    audio_folder: str | Path,
    csv_file: str | Path,
    max_duration: Optional[float] = None,
    num_workers: int = DEFAULT_NUM_WORKERS,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Backward-compatible wrapper for building a dataset from a single CSV file.
    """
    csv_file = Path(csv_file)
    annotations_root = csv_file.parent
    annotations_df = _load_annotations_from_folder(annotations_root)
    annotations_df = annotations_df[annotations_df["source_csv"] == csv_file.name].copy()

    if annotations_df.empty:
        raise ValueError(f"No annotation rows found in CSV file: {csv_file}")

    return _build_dataset_from_annotations(
        audio_root=audio_folder,
        annotations_df=annotations_df,
        max_duration=max_duration,
        num_workers=num_workers,
    )
