"""
Task 3.10 — Duration-dependent Viterbi algorithm for HSMM.

Implements the same algorithm as viterbi_hmm.pyx (Cython) in the McDonald
reference code, but in pure NumPy/Python with vectorised inner loops.

Interface matches segmenter.py compute_segmentation():
    states = hsmm_viterbi(posteriors, duration_matrix, max_duration, transition_matrix)

Algorithm: HSMM Viterbi with duration-dependent emission.

    log δ_t(j) = max_{i,d} [
        log δ_{t-d}(i) + log a_{ij} + log p_j(d) + cumlog_obs[t,j] - cumlog_obs[t-d,j]
    ]

Vectorisation strategy:
- Outer loop: t (time steps, T iterations)
- Inner loop d: vectorised over all durations using NumPy slicing
- Inner loop i: vectorised over all previous states using matrix ops

References:
    Springer et al., IEEE TBME 2016
    McDonald et al., segmenter.py / viterbi_hmm.pyx
"""

import numpy as np


def hsmm_viterbi(posteriors, duration_matrix, max_duration, transition_matrix):
    """Duration-dependent Viterbi for HSMM segmentation.

    Matches interface of viterbi_hmm.hsmm_viterbi() in McDonald ref code.

    Parameters
    ----------
    posteriors : np.ndarray, shape (T, N)
        Observation probabilities. posteriors[t, j] = P(o_t | state=j).
    duration_matrix : np.ndarray, shape (D_max, N)
        Duration probabilities. duration_matrix[d-1, j] = P(duration=d | state=j).
        Each column sums to 1.
    max_duration : int
        D_max — maximum duration in frames.
    transition_matrix : np.ndarray, shape (N, N)
        transition_matrix[i, j] = P(next_state=j | prev_state=i).
        Typically 0/1 for HSMM. Diagonal must be 0.

    Returns
    -------
    states : np.ndarray, shape (T,), dtype int32
        Optimal state sequence. Values in {0, ..., N-1}.
    """
    T, N = posteriors.shape
    D    = min(int(max_duration), T)

    NEG_INF = -1e30

    # ── Log-transform ─────────────────────────────────────────────────────
    log_obs  = np.log(np.clip(posteriors,       1e-10, 1.0))  # (T, N)
    log_dur  = np.log(np.clip(duration_matrix,  1e-10, 1.0))  # (D, N)

    # log_trans[i, j]: -inf where transition is forbidden
    with np.errstate(divide='ignore'):
        log_trans = np.where(transition_matrix > 0,
                             np.log(np.clip(transition_matrix, 1e-10, 1.0)),
                             NEG_INF)                          # (N, N)

    # ── Cumulative log-observations ───────────────────────────────────────
    # cumlog[t, j] = Σ_{s=0}^{t} log b_j(o_s)
    # Σ_{s=a}^{b} log b_j = cumlog[b, j] - (cumlog[a-1, j] if a>0 else 0)
    cumlog = np.cumsum(log_obs, axis=0)     # (T, N)
    # Prepend a row of zeros so cumlog[t-d, j] is valid for t-d=-1 → use index 0
    cumlog_ext = np.vstack([np.zeros((1, N)), cumlog])  # (T+1, N), index 0=before frame 0

    # ── Viterbi tables ────────────────────────────────────────────────────
    delta = np.full((T, N), NEG_INF, dtype=np.float64)  # (T, N)
    psi_d = np.zeros((T, N), dtype=np.int32)             # best duration
    psi_i = np.zeros((T, N), dtype=np.int32)             # best prev state (-1 = no prev)

    # ── Main recursion ────────────────────────────────────────────────────
    for t in range(T):
        # Durations to consider: d = 1..min(D, t+1)
        d_max_t = min(D, t + 1)
        d_arr   = np.arange(1, d_max_t + 1)  # shape (d_max_t,)

        # t_start for each duration: t_start = t - d + 1
        t_starts = t - d_arr + 1              # shape (d_max_t,)
        t_prevs  = t_starts - 1              # last frame of previous segment (-1 means none)

        # Observation sums for each (d, j):
        # obs_sum[d_idx, j] = Σ_{s=t_start}^{t} log b_j(o_s)
        #                    = cumlog_ext[t+1, j] - cumlog_ext[t_start, j]
        # cumlog_ext index: t+1 for "up to t", t_start for "up to t_start-1"
        obs_sums = (cumlog_ext[t + 1, :]               # (N,) broadcast
                    - cumlog_ext[t_starts, :])          # (d_max_t, N)
        # obs_sums[d_idx, j] = observation sum for duration d_arr[d_idx] ending at t

        # Duration log-probs: log_dur[d-1, j]
        log_p_d = log_dur[d_arr - 1, :]    # (d_max_t, N)

        # For segments with no previous (t_prev < 0):
        # val[d_idx, j] = obs_sums[d_idx, j] + log_p_d[d_idx, j]  (no transition)
        no_prev_mask = (t_prevs < 0)        # (d_max_t,) bool

        # For segments with a previous state (t_prev >= 0):
        # val[d_idx, j] = max_i [ delta[t_prev, i] + log_trans[i, j] ] + log_p_d + obs_sum
        # = (max over i of delta[t_prev, i] + log_trans[i, j]) + log_p_d + obs_sum
        # This is a "max-plus" matrix-vector product

        # Compute candidate values for each (d, j)
        best_vals = np.full((d_max_t, N), NEG_INF, dtype=np.float64)
        best_is   = np.zeros((d_max_t, N), dtype=np.int32)

        for d_idx in range(d_max_t):
            t_prev = t_prevs[d_idx]

            if t_prev < 0:
                # First segment: no previous state
                val = obs_sums[d_idx, :] + log_p_d[d_idx, :]  # (N,)
                best_vals[d_idx, :] = val
                best_is[d_idx, :]   = -1
            else:
                # For each j, find best i: delta[t_prev, i] + log_trans[i, j]
                # trans_scores[i, j] = delta[t_prev, i] + log_trans[i, j]
                delta_prev = delta[t_prev, :]          # (N,)
                # Add NEG_INF guard for invalid previous states
                valid = delta_prev > NEG_INF / 2
                if not np.any(valid):
                    continue

                trans_scores = (delta_prev[:, np.newaxis]
                                + log_trans)           # (N, N): [i, j]
                best_i_per_j  = np.argmax(trans_scores, axis=0)  # (N,)
                best_trans_val = trans_scores[best_i_per_j,
                                              np.arange(N)]       # (N,)

                val = best_trans_val + log_p_d[d_idx, :] + obs_sums[d_idx, :]
                best_vals[d_idx, :] = val
                best_is[d_idx, :]   = best_i_per_j

        # For each state j, pick the best duration
        best_d_per_j = np.argmax(best_vals, axis=0)   # (N,)
        for j in range(N):
            d_idx = best_d_per_j[j]
            delta[t, j]  = best_vals[d_idx, j]
            psi_d[t, j]  = d_arr[d_idx]
            psi_i[t, j]  = best_is[d_idx, j]

    # ── Termination + Backtracking ─────────────────────────────────────────
    states = np.zeros(T, dtype=np.int32)

    t = T - 1
    j = int(np.argmax(delta[t, :]))

    while t >= 0:
        d       = int(psi_d[t, j])
        t_start = t - d + 1
        prev_i  = int(psi_i[t, j])

        states[max(0, t_start): t + 1] = j

        if prev_i == -1 or t_start <= 0:
            if t_start > 0:
                states[:t_start] = j
            break

        t = t_start - 1
        j = prev_i

    return states


def build_duration_matrix(log_dur_dists, state_order, d_max):
    """Convert log_dur_dists dict → duration_matrix array (linear probs).

    Parameters
    ----------
    log_dur_dists : dict {state_name: np.ndarray (d_max,)}
    state_order : list of str   e.g. ['S1', 'Systole', 'S2', 'Diastole']
    d_max : int

    Returns
    -------
    duration_matrix : np.ndarray, shape (d_max, N)
        Linear probabilities, each column sums to 1.
    """
    N = len(state_order)
    duration_matrix = np.zeros((d_max, N), dtype=np.float64)

    for j, state_name in enumerate(state_order):
        log_p = log_dur_dists[state_name][:d_max]
        p     = np.exp(log_p)
        col_sum = p.sum()
        if col_sum > 0:
            duration_matrix[:len(p), j] = p / col_sum
        else:
            duration_matrix[0, j] = 1.0

    return duration_matrix
