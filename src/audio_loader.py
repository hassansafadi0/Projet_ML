from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple
import warnings

import numpy as np
from scipy.io import wavfile
from scipy.io.wavfile import WavFileWarning


def _to_mono(audio_array: np.ndarray) -> np.ndarray:
    """Convert multi-channel audio to a single mono channel."""
    if audio_array.ndim == 1:
        return audio_array

    return audio_array.mean(axis=1)


def _normalize_audio(audio_array: np.ndarray) -> np.ndarray:
    """Return audio as float32 in a stable numeric range."""
    if np.issubdtype(audio_array.dtype, np.floating):
        return audio_array.astype(np.float32, copy=False)

    if np.issubdtype(audio_array.dtype, np.integer):
        info = np.iinfo(audio_array.dtype)
        scale = max(abs(info.min), abs(info.max))
        if scale == 0:
            return audio_array.astype(np.float32)
        return audio_array.astype(np.float32) / scale

    return audio_array.astype(np.float32)


def load_audio(filepath: str | Path, max_duration: Optional[float] = None) -> Tuple[int, np.ndarray]:
    """Load a WAV file, optionally truncating it to the first max_duration seconds."""
    filepath = Path(filepath)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", WavFileWarning)
        sample_rate, audio_array = wavfile.read(filepath, mmap=True)

    if max_duration is not None:
        sample_limit = max(1, int(sample_rate * max_duration))
        audio_array = audio_array[:sample_limit]

    mono_audio = _to_mono(audio_array)
    normalized_audio = _normalize_audio(mono_audio)
    return sample_rate, normalized_audio
