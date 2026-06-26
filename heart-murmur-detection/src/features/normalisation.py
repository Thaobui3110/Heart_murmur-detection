import numpy as np


def normalise_amplitude(signal: np.ndarray) -> np.ndarray:
    """
    Zero-mean, peak-amplitude normalise a PCG signal.

    Following McDonald et al. (PLOS Digital Health, 2024):
        x_norm = (x - mean(x)) / max(|x - mean(x)|)

    Parameters
    ----------
    signal : np.ndarray
        Raw PCG waveform, any dtype (typically int16 from .wav files).

    Returns
    -------
    np.ndarray
        Normalised waveform, float64, range [-1.0, 1.0].
        Returns zero array if input signal is silent (peak == 0).
    """
    signal = signal.astype(np.float64)
    signal = signal - np.mean(signal)
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = signal / peak
    return signal