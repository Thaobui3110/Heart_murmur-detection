"""
Tasks 3.11–3.12 — 4 parallel HSMM topologies + confidence computation.

Implements the 4 HSMM models from McDonald et al., matching segmenter.py:
    segment_healthy_signal()
    segment_holosystolic_murmur()
    segment_early_systolic_murmur()
    segment_mid_systolic_murmur()
    compute_segmentation_confidence()

And the top-level runner matching double_duration_viterbi().

State index convention (must match labels.py and hsmm.py):
    S1=0, Systole=1, S2=2, Diastole=3, Murmur=4

References:
    McDonald et al., PLOS Digital Health 2024
    segmenter.py — segment_*() functions
"""

import copy
import numpy as np
import scipy.stats as sci_stat

from src.models.viterbi import hsmm_viterbi, build_duration_matrix
from src.models.hsmm import get_hsmm_params


# ── Transition matrices ────────────────────────────────────────────────────

# ω₁ and ω₂: 4-state cycle (S1→Sys→S2→Dia→S1)
TRANS_4STATE = np.array([
    [0, 1, 0, 0],   # S1      → Systole
    [0, 0, 1, 0],   # Systole → S2
    [0, 0, 0, 1],   # S2      → Diastole
    [1, 0, 0, 0],   # Diastole→ S1
], dtype=np.float32)

# ω₃: early-systolic (S1→Murmur→Systole→S2→Diastole→S1)
# State order: S1(0), Systole(1), S2(2), Diastole(3), Murmur(4)
TRANS_EARLY = np.array([
    [0, 0, 0, 0, 1],   # S1      → Murmur
    [0, 0, 1, 0, 0],   # Systole → S2
    [0, 0, 0, 1, 0],   # S2      → Diastole
    [1, 0, 0, 0, 0],   # Diastole→ S1
    [0, 1, 0, 0, 0],   # Murmur  → Systole
], dtype=np.float32)

# ω₄: mid-systolic (S1→Systole→Murmur→S2→Diastole→S1)
# Matches ref code segmenter.py segment_mid_systolic_murmur() exactly.
# S1 can start with Systole OR Murmur; Systole can go to S2 OR Murmur.
# Murmur → Systole (not S2) — allows Sys→Mur→Sys patterns.
TRANS_MID = np.array([
    [0, 1, 0, 0, 1],   # S1      → Systole OR Murmur
    [0, 0, 1, 0, 1],   # Systole → S2 OR Murmur   ← fixed
    [0, 0, 0, 1, 0],   # S2      → Diastole
    [1, 0, 0, 0, 0],   # Diastole→ S1
    [0, 1, 0, 0, 0],   # Murmur  → Systole         ← fixed (was →S2)
], dtype=np.float32)


# ── Helpers ────────────────────────────────────────────────────────────────

def _get_scipy_distribs(log_dur_dists, feature_rate=50):
    """Convert log_dur_dists dict → list of scipy.stats.norm objects.

    Matches ref code get_duration_distributions() return format.
    Order: [S1, Systole, S2, Diastole]
    """
    def _to_scipy(state_name):
        log_p = log_dur_dists[state_name]
        d     = np.arange(1, len(log_p) + 1, dtype=np.float64)
        p     = np.exp(log_p)
        mu    = np.sum(d * p)
        sigma = np.sqrt(np.sum((d - mu) ** 2 * p))
        sigma = max(sigma, 1.0)
        return sci_stat.norm(loc=mu, scale=sigma)

    return [_to_scipy(s) for s in ['S1', 'Systole', 'S2', 'Diastole']]


def _get_duration_matrix_from_dists(dists, d_max):
    """Build (D_max, N) duration matrix from list of scipy dists.

    Matches ref code get_duration_matrix().
    """
    d = np.arange(1, d_max + 1, dtype=np.float64)
    cols = [dist.pdf(d) for dist in dists]
    mat  = np.stack(cols).T          # (D_max, N)
    col_sums = mat.sum(axis=0)       # (N,)
    col_sums  = np.where(col_sums > 0, col_sums, 1.0)
    return (mat / col_sums).astype(np.float32)


def compute_confidence(posteriors, states):
    """Confidence = mean posterior at Viterbi path states.

    Matches ref code compute_segmentation_confidence():
        posteriors[np.arange(T), states].mean()

    Uses the MODIFIED posteriors passed to Viterbi (not the original 5-state).
    For ω₂: obs[:, 1] = posteriors_5[:, 4] (Murmur) was swapped in before
    calling Viterbi. So when path says state=1, we look up obs[:, 1] which
    already IS the Murmur posterior. Result is numerically identical to tracing
    through original posteriors with a state remapping. Matches ref code exactly.

    Parameters
    ----------
    posteriors : np.ndarray, shape (T, N)
        The MODIFIED posteriors actually used for decoding
        (e.g. 4-state for ω₁/ω₂, 5-state for ω₃/ω₄).
    states : np.ndarray, shape (T,)
        Viterbi path indices into the N columns of posteriors.

    Returns
    -------
    float in [0, 1]
    """
    T = len(states)
    return float(posteriors[np.arange(T), states].mean())


# ── 4 topology segmentation functions ─────────────────────────────────────

def segment_healthy(posteriors_5, log_dur_dists, d_max):
    """ω₁ — Normal signal, 4-state model.

    Matches ref code segment_healthy_signal():
    - Use first 4 channels: S1, Systole, S2, Diastole (Murmur DISCARDED)
    - Transition: S1→Sys→S2→Dia→S1

    Parameters
    ----------
    posteriors_5 : np.ndarray, shape (T, 5)  — original RNN output
    log_dur_dists : dict from compute_duration_distributions()
    d_max : int

    Returns
    -------
    states : np.ndarray (T,) with values in {0,1,2,3}
    confidence : float
    obs_used : np.ndarray (T, 4) — the posteriors actually used
    """
    # 4-state posteriors: keep S1, Systole, S2, Diastole (drop Murmur)
    obs = np.zeros((posteriors_5.shape[0], 4), dtype=posteriors_5.dtype)
    obs[:, 0] = posteriors_5[:, 0]   # S1
    obs[:, 1] = posteriors_5[:, 1]   # Systole
    obs[:, 2] = posteriors_5[:, 2]   # S2
    obs[:, 3] = posteriors_5[:, 3]   # Diastole

    dur_mat = _get_duration_matrix_from_dists(
        _get_scipy_distribs(log_dur_dists), d_max
    )
    states = hsmm_viterbi(obs, dur_mat, d_max, TRANS_4STATE)
    conf   = compute_confidence(obs, states)
    return states, conf, obs


def segment_holosystolic(posteriors_5, log_dur_dists, d_max):
    """ω₂ — Holosystolic murmur, 4-state model.

    Matches ref code segment_holosystolic_murmur():
    - Murmur posterior REPLACES Systole posterior at every frame
    - Transition: same 4-state cycle S1→Sys→S2→Dia→S1
      (but "Systole" channel now carries Murmur probability)

    Returns
    -------
    states : np.ndarray (T,) with values in {0,1,2,3}
        State 1 = "Systole" position in cycle, but driven by Murmur posterior
    """
    obs = np.zeros((posteriors_5.shape[0], 4), dtype=posteriors_5.dtype)
    obs[:, 0] = posteriors_5[:, 0]   # S1
    obs[:, 1] = posteriors_5[:, 4]   # Murmur REPLACES Systole ← key change
    obs[:, 2] = posteriors_5[:, 2]   # S2
    obs[:, 3] = posteriors_5[:, 3]   # Diastole

    dur_mat = _get_duration_matrix_from_dists(
        _get_scipy_distribs(log_dur_dists), d_max
    )
    states = hsmm_viterbi(obs, dur_mat, d_max, TRANS_4STATE)
    conf   = compute_confidence(obs, states)
    return states, conf, obs


def segment_early_systolic(posteriors_5, log_dur_dists, d_max):
    """ω₃ — Early-systolic murmur, 5-state model.

    Matches ref code segment_early_systolic_murmur():
    - 5 states: S1(0) → Murmur(4) → Systole(1) → S2(2) → Diastole(3) → S1
    - Duration: original Systole dist HALVED for both Murmur and Systole
      (so total murmur+systole ≈ original systole)
    """
    # Halve the Systole distribution for both Murmur and Systole slots
    orig_sys_dists = _get_scipy_distribs(log_dur_dists)  # [S1, Sys, S2, Dia]
    orig_sys       = orig_sys_dists[1]                   # scipy norm for Systole
    half_sys       = sci_stat.norm(loc=orig_sys.mean() / 2,
                                   scale=orig_sys.std()  / 2)

    # 5-state duration dists: [S1, Systole_half, S2, Diastole, Murmur_half]
    dists_5 = [orig_sys_dists[0],  # S1
               half_sys,            # Systole (halved)
               orig_sys_dists[2],  # S2
               orig_sys_dists[3],  # Diastole
               half_sys]            # Murmur (same as halved Systole)

    d      = np.arange(1, d_max + 1, dtype=np.float64)
    cols   = [dist.pdf(d) for dist in dists_5]
    mat    = np.stack(cols).T          # (D_max, 5)
    col_sums = mat.sum(axis=0)
    col_sums  = np.where(col_sums > 0, col_sums, 1.0)
    dur_mat   = (mat / col_sums).astype(np.float32)

    # Use full 5-state posteriors as-is
    obs    = posteriors_5.astype(np.float32)
    states = hsmm_viterbi(obs, dur_mat, d_max, TRANS_EARLY)
    conf   = compute_confidence(obs, states)
    return states, conf, obs


def segment_mid_systolic(posteriors_5, log_dur_dists, d_max):
    """ω₄ — Mid-systolic murmur, 5-state model.

    Matches ref code segment_mid_systolic_murmur():
    - 5 states: S1(0) → Systole(1) → Murmur(4) → S2(2) → Diastole(3) → S1
    - Duration: Systole QUARTERED (pre-murmur), Murmur HALVED
    """
    orig_sys_dists = _get_scipy_distribs(log_dur_dists)
    orig_sys       = orig_sys_dists[1]
    half_sys       = sci_stat.norm(loc=orig_sys.mean() / 2,
                                   scale=orig_sys.std()  / 2)
    quarter_sys    = sci_stat.norm(loc=orig_sys.mean() / 4,
                                   scale=orig_sys.std()  / 4)

    # 5-state duration dists: [S1, Systole_quarter, S2, Diastole, Murmur_half]
    dists_5 = [orig_sys_dists[0],  # S1
               quarter_sys,         # Systole (quartered — pre-murmur part)
               orig_sys_dists[2],  # S2
               orig_sys_dists[3],  # Diastole
               half_sys]            # Murmur (halved)

    d      = np.arange(1, d_max + 1, dtype=np.float64)
    cols   = [dist.pdf(d) for dist in dists_5]
    mat    = np.stack(cols).T
    col_sums = mat.sum(axis=0)
    col_sums  = np.where(col_sums > 0, col_sums, 1.0)
    dur_mat   = (mat / col_sums).astype(np.float32)

    obs    = posteriors_5.astype(np.float32)
    states = hsmm_viterbi(obs, dur_mat, d_max, TRANS_MID)
    conf   = compute_confidence(obs, states)
    return states, conf, obs


# ── Top-level runner ───────────────────────────────────────────────────────

def run_parallel_hsmm(posteriors_5, feature_rate=50,
                      murmur_models=('Holosystolic', 'Early-systolic', 'Mid-systolic')):
    """Run all 4 HSMM topologies and return confidences.

    Matches ref code double_duration_viterbi().

    Parameters
    ----------
    posteriors_5 : np.ndarray, shape (T, 5)
        RNN posterior probabilities (softmax output).
    feature_rate : int, default 50
    murmur_models : tuple of str
        Which murmur topologies to run.

    Returns
    -------
    dict with keys:
        'healthy_states'  : np.ndarray (T,) — ω₁ Viterbi path
        'healthy_conf'    : float            — C(ω₁)
        'murmur_states'   : np.ndarray (T,) — best murmur topology path
        'murmur_conf'     : float            — C(best murmur topology)
        'murmur_model'    : str              — 'Holosystolic'/'Early-systolic'/'Mid-systolic'
        'all_confs'       : dict {topology: confidence}
        'log_dur_dists'   : dict
        'heart_rate_bpm'  : float
        'systolic_interval_sec': float
        'd_max'           : int
    """
    # Step 1: Estimate heart rate and duration distributions
    hr_bpm, sys_interval, log_dur_dists, _ = get_hsmm_params(
        posteriors_5, feature_rate=feature_rate
    )
    heart_period_frames = int(round(60.0 / hr_bpm * feature_rate))
    d_max = heart_period_frames  # max duration = 1 full heart cycle

    # Step 2: Run ω₁ (healthy)
    h_states, h_conf, _ = segment_healthy(posteriors_5, log_dur_dists, d_max)

    # Step 3: Run murmur topologies
    murmur_funcs = {
        'Holosystolic':   segment_holosystolic,
        'Early-systolic': segment_early_systolic,
        'Mid-systolic':   segment_mid_systolic,
    }

    all_confs   = {'Healthy': h_conf}
    best_conf   = -1.0
    best_model  = None
    best_states = None

    for model_name in murmur_models:
        fn     = murmur_funcs[model_name]
        states, conf, _ = fn(posteriors_5, log_dur_dists, d_max)
        all_confs[model_name] = conf
        if conf > best_conf:
            best_conf   = conf
            best_model  = model_name
            best_states = states

    return {
        'healthy_states':        h_states,
        'healthy_conf':          h_conf,
        'murmur_states':         best_states,
        'murmur_conf':           best_conf,
        'murmur_model':          best_model,
        'all_confs':             all_confs,
        'log_dur_dists':         log_dur_dists,
        'heart_rate_bpm':        hr_bpm,
        'systolic_interval_sec': sys_interval,
        'd_max':                 d_max,
    }
