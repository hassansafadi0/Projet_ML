from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.signal import spectrogram


def compute_spectrogram(signal: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a log-scaled spectrogram for a mono waveform."""
    nperseg = 256
    noverlap = min(int(round(nperseg * 0.98)), nperseg - 1)
    nfft = 512

    freq_axis, time_axis, spectrogram_values = spectrogram(
        signal,
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        detrend=False,
        scaling="density",
        mode="magnitude",
    )

    spectrogram_db = 20.0 * np.log10(spectrogram_values + 1e-10)
    return (
        freq_axis.astype(np.float32, copy=False),
        time_axis.astype(np.float32, copy=False),
        spectrogram_db.astype(np.float32, copy=False),
    )
