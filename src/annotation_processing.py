from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd


def _annotation_value(annotation_row: object, field_name: str) -> object:
    if isinstance(annotation_row, pd.Series):
        return annotation_row[field_name]

    if hasattr(annotation_row, field_name):
        return getattr(annotation_row, field_name)

    return annotation_row[field_name]


def _parse_audio_start(filename: str) -> datetime:
    return datetime.strptime(Path(filename).stem, "%Y-%m-%dT%H-%M-%S_%f")


def _parse_timestamp(value: str) -> datetime:
    return pd.to_datetime(value, utc=True).to_pydatetime().replace(tzinfo=None)


def convert_annotation_to_pixels(
    annotation_row: object,
    spectrogram_shape: tuple[int, int],
    time_axis: np.ndarray,
    freq_axis: np.ndarray,
) -> Optional[Dict[str, int]]:
    """Convert one annotation row to clipped spectrogram coordinates."""
    try:
        audio_start = _parse_audio_start(str(_annotation_value(annotation_row, "filename")))
        start_time = _parse_timestamp(str(_annotation_value(annotation_row, "start_datetime")))
        end_time = _parse_timestamp(str(_annotation_value(annotation_row, "end_datetime")))
        low_frequency = float(_annotation_value(annotation_row, "low_frequency"))
        high_frequency = float(_annotation_value(annotation_row, "high_frequency"))
    except (AttributeError, KeyError, TypeError, ValueError):
        return None

    if end_time < start_time or high_frequency <= low_frequency:
        return None

    start_seconds = max(0.0, (start_time - audio_start).total_seconds())
    end_seconds = max(start_seconds, (end_time - audio_start).total_seconds())

    max_y, max_x = spectrogram_shape
    if max_y == 0 or max_x == 0:
        return None

    x_start = int(np.searchsorted(time_axis, start_seconds, side="left"))
    x_end = int(np.searchsorted(time_axis, end_seconds, side="right"))
    y_low = int(np.searchsorted(freq_axis, low_frequency, side="left"))
    y_high = int(np.searchsorted(freq_axis, high_frequency, side="right"))

    x_start = int(np.clip(x_start, 0, max_x - 1))
    x_end = int(np.clip(max(x_end, x_start + 1), 1, max_x))
    y_low = int(np.clip(y_low, 0, max_y - 1))
    y_high = int(np.clip(max(y_high, y_low + 1), 1, max_y))

    return {
        "x_start": x_start,
        "x_end": x_end,
        "y_low": y_low,
        "y_high": y_high,
    }
