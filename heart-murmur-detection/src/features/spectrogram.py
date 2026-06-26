import numpy as np
from scipy.signal import stft


def compute_log_spectrogram(signal: np.ndarray,
                             sr: int = 4000,
                             win_length_sec: float = 0.050,
                             hop_length_sec: float = 0.020) -> tuple:
    """
    Compute log-power spectrogram from a normalised PCG signal.

    Following McDonald et al. (PLOS Digital Health, 2024):
        - Hann window, length 50 ms, hop 20 ms
        - Frequency resolution: sr / n_fft = 4000 / 200 = 20 Hz
        - Feature rate: 1 / hop_length_sec = 50 frames/second
        - Log transform: log(|STFT|^2 + epsilon)

    Parameters
    ----------
    signal : np.ndarray
        Normalised PCG waveform (float64, range [-1, 1]).
    sr : int
        Sampling rate in Hz (must be 4000 for CirCor dataset).
    win_length_sec : float
        STFT window length in seconds (paper uses 0.050 = 50 ms).
    hop_length_sec : float
        STFT hop length in seconds (paper uses 0.020 = 20 ms).

    Returns
    -------
    S_log : np.ndarray
        Log-power spectrogram, shape (n_freq_bins, n_frames).
        n_freq_bins = n_fft // 2 + 1 = 101 (full 0–2000 Hz range).
    freqs : np.ndarray
        Frequency axis in Hz, length n_freq_bins.
    times : np.ndarray
        Time axis in seconds, length n_frames.
    """
    win_length = int(win_length_sec * sr)   # 200 samples
    hop_length = int(hop_length_sec * sr)   # 80 samples
    noverlap   = win_length - hop_length    # 120 samples

    freqs, times, Zxx = stft(signal, fs=sr, window='hann',
                              nperseg=win_length,
                              noverlap=noverlap,
                              nfft=win_length)

    S_power = np.abs(Zxx) ** 2
    S_log   = np.log(S_power + 1e-10)

    return S_log, freqs, times

def crop_frequency(S_log: np.ndarray,
                   freqs: np.ndarray,
                   max_freq: float = 800.0) -> tuple:
    """
    Crop spectrogram to retain only frequencies up to max_freq Hz.

    This is a Feature Selection step — discarding high-frequency bins
    that contain no heart sound information (speech, ambient noise,
    stethoscope artifacts above 800 Hz).

    Parameters
    ----------
    S_log : np.ndarray
        Log-power spectrogram, shape (n_freq_bins, n_frames).
    freqs : np.ndarray
        Frequency axis in Hz, length n_freq_bins.
    max_freq : float
        Maximum frequency to retain in Hz (default 800.0).

    Returns
    -------
    S_cropped : np.ndarray
        Cropped spectrogram, shape (n_retained_bins, n_frames).
        At 20 Hz resolution: n_retained_bins = 41 (0, 20, ..., 800 Hz).
    freqs_cropped : np.ndarray
        Cropped frequency axis in Hz, length n_retained_bins.
    """
    freq_mask = freqs <= max_freq
    return S_log[freq_mask, :], freqs[freq_mask]

def zscore_per_row(S_cropped: np.ndarray) -> np.ndarray:
    """
    Z-score normalise each frequency row independently across the time axis.

    Following McDonald et al. (PLOS Digital Health, 2024):
        S_norm[f, :] = (S[f, :] - mean(S[f, :])) / std(S[f, :])

    Applied per-row (per frequency bin), NOT per-column (per time frame).
    Each frequency bin is normalised independently across all time frames
    of that recording.

    Rationale: Murmurs contain much less time-frequency energy than S1/S2.
    This normalisation reduces the dynamic range between frequency bins,
    making murmur-related bins equally prominent as S1/S2 bins.

    Parameters
    ----------
    S_cropped : np.ndarray
        Cropped log-spectrogram, shape (n_freq_bins, n_frames).

    Returns
    -------
    S_norm : np.ndarray
        Z-score normalised spectrogram, same shape (n_freq_bins, n_frames).
        Each row has mean ≈ 0 and std ≈ 1.
        Constant rows (std == 0) are set to zero to avoid division by zero.
    """
    means = S_cropped.mean(axis=1, keepdims=True)
    stds  = S_cropped.std(axis=1, keepdims=True)
    stds[stds == 0] = 1.0   # silent rows → zero row after normalisation
    return (S_cropped - means) / stds


from pathlib import Path
from scipy.io import wavfile
from src.features.normalisation import normalise_amplitude


def extract_features(wav_path,
                     sr: int = 4000,
                     win_length_sec: float = 0.050,
                     hop_length_sec: float = 0.020,
                     max_freq: float = 800.0) -> tuple:
    """
    Full feature extraction pipeline for one PCG recording.

    Steps: load → amplitude normalise → log-spectrogram → crop → z-score

    Parameters
    ----------
    wav_path : str or Path
        Path to .wav file.
    sr : int
        Expected sampling rate (4000 Hz for CirCor dataset).
    win_length_sec : float
        STFT window length in seconds (paper: 0.050).
    hop_length_sec : float
        STFT hop length in seconds (paper: 0.020).
    max_freq : float
        Frequency cutoff in Hz (paper: 800.0).

    Returns
    -------
    features : np.ndarray
        Fully preprocessed spectrogram, shape (n_freq_bins, n_frames).
        n_freq_bins = 41 at default settings (0–800 Hz, 20 Hz resolution).
    freqs : np.ndarray
        Frequency axis in Hz, length n_freq_bins.
    times : np.ndarray
        Time axis in seconds, length n_frames.

    Raises
    ------
    ValueError
        If the loaded file has a different sampling rate than expected.
    """
    # Step 1: Load raw int16
    loaded_sr, signal = wavfile.read(Path(wav_path))
    if loaded_sr != sr:
        raise ValueError(f"Expected sr={sr}, got sr={loaded_sr} in {wav_path}")
    if signal.ndim > 1:
        signal = signal[:, 0]   # stereo → mono

    # Step 2: Amplitude normalisation
    signal = normalise_amplitude(signal)

    # Step 3: Log-spectrogram
    S_log, freqs, times = compute_log_spectrogram(
        signal, sr=sr,
        win_length_sec=win_length_sec,
        hop_length_sec=hop_length_sec
    )

    # Step 4: Frequency cropping
    S_crop, freqs = crop_frequency(S_log, freqs, max_freq=max_freq)

    # Step 5: Per-row z-score
    features = zscore_per_row(S_crop)

    return features, freqs, times


def extract_features_batch(wav_paths, show_progress: bool = True, **kwargs) -> list:
    """
    Extract features for a list of recordings.

    Parameters
    ----------
    wav_paths : list of str or Path
        List of paths to .wav files.
    show_progress : bool
        Print progress every 100 files.
    **kwargs
        Passed to extract_features().

    Returns
    -------
    list of (features, freqs, times) tuples.
    Failed files are returned as None with an error message printed.
    """
    results = []
    n = len(wav_paths)
    for i, path in enumerate(wav_paths):
        if show_progress and (i % 100 == 0):
            print(f"  [{i}/{n}] Processing {Path(path).name}...")
        try:
            result = extract_features(path, **kwargs)
            results.append(result)
        except Exception as e:
            print(f"  ERROR at {path}: {e}")
            results.append(None)
    return results