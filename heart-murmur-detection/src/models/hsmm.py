"""
Task 3.8 — Heart rate + systolic interval estimation (McDonald method).
Task 3.9 — State duration distributions (McDonald constants, not Springer fractions).

Key differences from Springer (2016):
1. HR estimated from RNN posteriors (1 - P(Diastole)) instead of homomorphic envelope
2. Systolic interval estimated from ACF of P(S1) + P(S2) separately
3. S1/S2 durations use absolute-second constants fit to CirCor paediatric data
4. Systole/Diastole means derived from measured systolic_interval, not fixed fractions
5. Diastole std is fixed (0.050s), not adaptive like Springer

References:
    McDonald et al., PLOS Digital Health 2024
    McDonald et al., CinC 2022
    segmenter.py — McDonald reference code (get_duration_distributions,
                   get_heart_rate, get_systolic_interval)
"""

import numpy as np

# ── State indices (must match labels.py) ──────────────────────────────────
S1, SYSTOLE, S2, DIASTOLE, MURMUR = 0, 1, 2, 3, 4

# ── McDonald absolute-second constants (fit to CirCor paediatric data) ────
# Source: segmenter.py — get_duration_distributions()
S1_MEAN_SEC  = 0.1163   # mean S1 duration
S1_STD_SEC   = 0.0196   # std  S1 duration
S2_MEAN_SEC  = 0.1032   # mean S2 duration
S2_STD_SEC   = 0.0195   # std  S2 duration

# Systole: mean = systolic_interval - S1_offset
SYS_OFFSET_SEC = 0.1279  # subtracted from systolic_interval to get systole mean
SYS_STD_SEC    = 0.0250  # fixed std for systole

# Diastole: mean = (heart_period - systolic_interval) - DIA_offset
DIA_OFFSET_SEC = 0.1053  # subtracted from diastolic_interval to get diastole mean
DIA_STD_SEC    = 0.0500  # fixed std for diastole (not adaptive like Springer)

# Minimum duration means (clamp to avoid degenerate distributions)
SYS_MIN_SEC = 0.070   # 70ms minimum systole mean
DIA_MIN_SEC = 0.100   # 100ms minimum diastole mean


# ── Task 3.8a — Heart rate estimation ─────────────────────────────────────

def _autocorr_argmax(signal, min_lag, max_lag):
    """Compute ACF and return lag of highest peak in [min_lag, max_lag].

    Matches ref code exactly:
    - NO zero-mean subtraction
    - normalise by acf[0]
    - use np.argmax directly (no find_peaks)
    """
    T = len(signal)
    acorr = np.correlate(signal, signal, mode='full')
    acorr = acorr[len(acorr) // 2:]   # positive lags: index = lag
    if acorr[0] > 0:
        acorr = acorr / acorr[0]

    max_lag = min(max_lag, T - 1)
    valid   = acorr[min_lag : max_lag + 1]
    rel_peak = int(np.argmax(valid))
    return min_lag + rel_peak


def estimate_heart_rate(posteriors, feature_rate=50,
                        min_bpm=30, max_bpm=180, default_bpm=80):
    """Estimate heart rate from RNN posteriors via autocorrelation.

    Matches ref code get_heart_rate():
    - Signal: sum of S1+Systole+S2+Murmur posteriors (= 1 - P(Diastole))
    - NO zero-mean subtraction
    - Use np.argmax directly (no find_peaks)
    - Search range: [60/max_bpm*fs, 60/min_bpm*fs] frames

    Parameters
    ----------
    posteriors : np.ndarray, shape (T, 5)
    feature_rate : int, default 50
    min_bpm, max_bpm : int, default 30–180
    default_bpm : int, fallback

    Returns
    -------
    heart_rate_bpm : float
    heart_period_sec : float
    heart_period_frames : int
    """
    # Ref code: np.sum(posterior[:, [0,1,2,4]], axis=1)
    s = (posteriors[:, S1] + posteriors[:, SYSTOLE] +
         posteriors[:, S2]  + posteriors[:, MURMUR])

    min_lag = round(feature_rate * 60 / max_bpm)
    max_lag = round(feature_rate * 60 / min_bpm)

    best_lag = _autocorr_argmax(s, min_lag, max_lag)

    heart_period_sec    = best_lag / feature_rate
    heart_rate_bpm      = 60.0 / heart_period_sec
    heart_period_frames = best_lag

    # Sanity check
    if not (min_bpm <= heart_rate_bpm <= max_bpm):
        heart_rate_bpm      = float(default_bpm)
        heart_period_sec    = 60.0 / default_bpm
        heart_period_frames = int(round(heart_period_sec * feature_rate))

    return heart_rate_bpm, heart_period_sec, heart_period_frames


# ── Task 3.8b — Systolic interval estimation ──────────────────────────────

def estimate_systolic_interval(posteriors, heart_rate_bpm,
                               feature_rate=50, min_duration_ms=150):
    """Estimate systolic interval from ACF of S1+S2 posteriors.

    Matches ref code get_systolic_interval() exactly:
    - Signal: P(S1) + P(S2) only (indices 0 and 2)
    - Search range: [150ms, heart_cycle/2] — NOT BPM-based
    - Peak finding: np.argmax directly (no find_peaks)
    - Output: the lag itself IS the systolic interval (no Bazett formula)

    Parameters
    ----------
    posteriors : np.ndarray, shape (T, 5)
    heart_rate_bpm : float
        Must be estimated first via estimate_heart_rate().
    feature_rate : int, default 50
    min_duration_ms : int, default 150 (ms) — minimum systolic interval

    Returns
    -------
    systolic_interval_sec : float
    """
    heart_cycle_frames    = (60.0 / heart_rate_bpm) * feature_rate
    max_systolic_frames   = int(heart_cycle_frames / 2)   # max = half heart cycle
    min_systolic_frames   = round(min_duration_ms * 1e-3 * feature_rate)

    # Ref code: posterior[:, [0, 2]] = S1 + S2 only
    s = posteriors[:, S1] + posteriors[:, S2]

    best_lag = _autocorr_argmax(s, min_systolic_frames, max_systolic_frames)

    return best_lag / feature_rate


# ── Task 3.9 — Duration distributions (McDonald method) ───────────────────

def compute_duration_distributions(heart_rate_bpm, systolic_interval_sec,
                                   feature_rate=50, d_max=None,
                                   state_subset=None):
    """Compute Gaussian duration distributions using McDonald's constants.

    Key differences from Springer:
    - S1/S2: absolute-second constants (fit to CirCor paediatric data)
    - Systole: mean = systolic_interval - offset (measured from signal)
    - Diastole: mean = (heart_period - systolic_interval) - offset
    - Diastole std: fixed 50ms (not adaptive)

    Parameters
    ----------
    heart_rate_bpm : float
        Estimated heart rate (BPM) from estimate_heart_rate().
    systolic_interval_sec : float
        Estimated systolic interval (seconds) from estimate_systolic_interval().
    feature_rate : int, default 50
    d_max : int, optional. Defaults to 2 × heart_period_frames.
    state_subset : list of str, optional. Defaults to all 5 states.

    Returns
    -------
    dict {state_name: np.ndarray shape (d_max,)}
        Log-probabilities of duration d=1,2,...,d_max frames.
        Index 0 corresponds to d=1 frame.
    """
    if state_subset is None:
        state_subset = ['S1', 'Systole', 'S2', 'Diastole', 'Murmur']

    heart_period_sec    = 60.0 / heart_rate_bpm
    heart_period_frames = int(round(heart_period_sec * feature_rate))

    if d_max is None:
        d_max = max(10, int(heart_period_frames * 2))

    # ── Compute mean/std per state (in frames) ────────────────────────────

    # S1: absolute constant
    mu_s1    = S1_MEAN_SEC  * feature_rate
    sig_s1   = S1_STD_SEC   * feature_rate

    # S2: absolute constant
    mu_s2    = S2_MEAN_SEC  * feature_rate
    sig_s2   = S2_STD_SEC   * feature_rate

    # Systole: derived from measured systolic_interval
    mean_sys_sec = max(systolic_interval_sec - SYS_OFFSET_SEC, SYS_MIN_SEC)
    mu_sys       = mean_sys_sec * feature_rate
    sig_sys      = SYS_STD_SEC  * feature_rate

    # Diastole: remainder of heart period after systolic interval
    diastolic_interval_sec = heart_period_sec - systolic_interval_sec
    mean_dia_sec           = max(diastolic_interval_sec - DIA_OFFSET_SEC, DIA_MIN_SEC)
    mu_dia                 = mean_dia_sec * feature_rate
    sig_dia                = DIA_STD_SEC  * feature_rate

    # Murmur: same as Systole (will be split when building topologies)
    mu_mur  = mu_sys
    sig_mur = sig_sys

    state_params = {
        'S1':       (mu_s1,  sig_s1),
        'Systole':  (mu_sys, sig_sys),
        'S2':       (mu_s2,  sig_s2),
        'Diastole': (mu_dia, sig_dia),
        'Murmur':   (mu_mur, sig_mur),
    }

    # ── Build log-probability arrays ──────────────────────────────────────
    durations     = np.arange(1, d_max + 1, dtype=np.float64)
    log_dur_dists = {}

    for state_name in state_subset:
        mu, sigma = state_params[state_name]
        sigma     = max(sigma, 1.0)          # avoid degenerate distributions

        pdf     = np.exp(-0.5 * ((durations - mu) / sigma) ** 2)
        pdf_sum = pdf.sum()

        if pdf_sum > 0:
            pdf = pdf / pdf_sum
        else:
            pdf        = np.zeros(d_max)
            mu_idx     = max(0, min(d_max - 1, int(round(mu)) - 1))
            pdf[mu_idx] = 1.0

        log_dur_dists[state_name] = np.log(pdf + 1e-10)

    return log_dur_dists, state_params


# ── Convenience wrapper ────────────────────────────────────────────────────

def get_hsmm_params(posteriors, feature_rate=50):
    """Full pipeline: posteriors → HR + systolic interval → duration dists.

    Matches ref code double_duration_viterbi() parameter estimation:
    1. get_heart_rate()       → heart_rate_bpm
    2. get_systolic_interval() → systolic_interval_sec  (needs HR as input)
    3. get_duration_distributions() → log_dur_dists

    Returns
    -------
    heart_rate_bpm : float
    systolic_interval_sec : float
    log_dur_dists : dict {state_name: np.ndarray}
    state_params : dict {state_name: (mu_frames, sig_frames)}
    """
    heart_rate_bpm, _, _ = estimate_heart_rate(
        posteriors, feature_rate=feature_rate
    )
    systolic_interval_sec = estimate_systolic_interval(
        posteriors, heart_rate_bpm, feature_rate=feature_rate
    )
    log_dur_dists, state_params = compute_duration_distributions(
        heart_rate_bpm, systolic_interval_sec, feature_rate=feature_rate
    )
    return heart_rate_bpm, systolic_interval_sec, log_dur_dists, state_params
