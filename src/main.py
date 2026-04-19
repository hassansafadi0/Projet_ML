from __future__ import annotations

import argparse
from pathlib import Path
import sys

# pour éviter les problèmes d'importation lorsque ce script est exécuté directement
if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.cache_utils import load_dataset, save_dataset
from src.dataset_builder import CLASS_ORDER, DEFAULT_NUM_WORKERS, build_dataset_from_split
from src.model import DEFAULT_PCA_COMPONENTS, train_model


def _ordered_labels(labels: list[str]) -> list[str]:
    label_set = set(labels)
    ordered = [label for label in CLASS_ORDER if label in label_set]
    extras = sorted(label for label in label_set if label not in CLASS_ORDER)
    return [*ordered, *extras]


def parse_args() -> argparse.Namespace:
    default_train_audio_folder = Path("train/audio")
    default_train_annotations_folder = Path("train/annotations")
    default_val_audio_folder = Path("validation/audio")
    default_val_annotations_folder = Path("validation/annotations")
    default_cache_root = Path("cache")

    parser = argparse.ArgumentParser(
        description="Train a whale-vocalization classifier from spectrogram patches."
    )
    parser.add_argument(
        "--train-audio-folder",
        type=Path,
        default=default_train_audio_folder,
        help=f"Training audio root grouped by dataset. Default: {default_train_audio_folder}",
    )
    parser.add_argument(
        "--train-annotations-folder",
        type=Path,
        default=default_train_annotations_folder,
        help=f"Training annotations folder. Default: {default_train_annotations_folder}",
    )
    parser.add_argument(
        "--val-audio-folder",
        type=Path,
        default=default_val_audio_folder,
        help=f"Validation audio root grouped by dataset. Default: {default_val_audio_folder}",
    )
    parser.add_argument(
        "--val-annotations-folder",
        type=Path,
        default=default_val_annotations_folder,
        help=f"Validation annotations folder. Default: {default_val_annotations_folder}",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=default_cache_root,
        help=f"Dataset cache root. Default: {default_cache_root}",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=None,
        help="Optional maximum duration to load per audio file, in seconds.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="logistic_regression",
        choices=["sgd", "logistic_regression"],
        help="Classifier to train. Default: logistic_regression",
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=DEFAULT_PCA_COMPONENTS,
        help=(
            "Number of PCA components after scaling. "
            "Use 0 to disable PCA. Default: "
            f"{DEFAULT_PCA_COMPONENTS}"
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of worker threads used while building spectrogram patches. Default: auto",
    )
    parser.add_argument(
        "--use-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use cached datasets when available. Default: True",
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Force dataset rebuild and overwrite existing cache.",
    )
    return parser.parse_args()


def _validate_inputs(args: argparse.Namespace) -> None:
    if not args.train_audio_folder.exists():
        raise FileNotFoundError(f"Training audio folder does not exist: {args.train_audio_folder}")

    if not args.train_annotations_folder.exists():
        raise FileNotFoundError(
            f"Training annotations folder does not exist: {args.train_annotations_folder}"
        )

    if not args.val_audio_folder.exists():
        raise FileNotFoundError(f"Validation audio folder does not exist: {args.val_audio_folder}")

    if not args.val_annotations_folder.exists():
        raise FileNotFoundError(
            f"Validation annotations folder does not exist: {args.val_annotations_folder}"
        )


def _cache_exists(cache_path: Path) -> bool:
    return (cache_path / "X.npy").exists() and (cache_path / "y.npy").exists()


def _load_or_build_dataset(
    split_name: str,
    audio_root: Path,
    annotations_root: Path,
    cache_path: Path,
    max_duration: float | None,
    num_workers: int,
    use_cache: bool,
    rebuild_cache: bool,
):
    if use_cache and not rebuild_cache and _cache_exists(cache_path):
        print(f"Loading {split_name} dataset from cache: {cache_path}")
        return load_dataset(cache_path)

    print(f"Building {split_name} dataset...")
    X, y = build_dataset_from_split(
        audio_root=audio_root,
        annotations_root=annotations_root,
        max_duration=max_duration,
        num_workers=num_workers,
    )

    if use_cache:
        print(f"Saving {split_name} dataset cache to: {cache_path}")
        save_dataset(X, y, cache_path)

    return X, y


def main() -> None:
    args = parse_args()
    _validate_inputs(args)

    num_workers = args.num_workers if args.num_workers is not None else DEFAULT_NUM_WORKERS
    train_cache_path = args.cache_root / "train"
    validation_cache_path = args.cache_root / "validation"

    X_train, y_train = _load_or_build_dataset(
        split_name="training",
        audio_root=args.train_audio_folder,
        annotations_root=args.train_annotations_folder,
        cache_path=train_cache_path,
        max_duration=args.max_duration,
        num_workers=num_workers,
        use_cache=args.use_cache,
        rebuild_cache=args.rebuild_cache,
    )

    X_val, y_val = _load_or_build_dataset(
        split_name="validation",
        audio_root=args.val_audio_folder,
        annotations_root=args.val_annotations_folder,
        cache_path=validation_cache_path,
        max_duration=args.max_duration,
        num_workers=num_workers,
        use_cache=args.use_cache,
        rebuild_cache=args.rebuild_cache,
    )

    print(f"Training dataset size: {len(X_train)} patches")
    print(f"Validation dataset size: {len(X_val)} patches")
    print(f"Features per patch: {X_train.shape[1]}")

    all_labels = _ordered_labels([*y_train.tolist(), *y_val.tolist()])
    print(f"Unique labels before training: {all_labels}")

    _, metrics = train_model(
        X_train,
        y_train,
        X_val,
        y_val,
        all_labels=all_labels,
        model_name=args.model_name,
        pca_components=args.pca_components if args.pca_components > 0 else None,
    )
    print(f"Finished training. Final accuracy: {metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()
