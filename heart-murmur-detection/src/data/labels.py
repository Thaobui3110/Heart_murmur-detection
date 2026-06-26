"""
Task 3.1 — Ground-truth label generation (5-state with Murmur).

Converts 4-state TSV segmentation annotations into 5-state frame-level
labels by relabelling portions of Systole as Murmur based on clinician
timing annotations (Early-systolic, Mid-systolic, Holosystolic, Late-systolic).

State index convention (0-indexed, used throughout Phase 3):
    S1 = 0, Systole = 1, S2 = 2, Diastole = 3, Murmur = 4
    Unannotated / padding = -1  (masked from loss via ignore_index)

TSV file convention (1-indexed, CirCor dataset):
    0 = Unannotated, 1 = S1, 2 = Systole, 3 = S2, 4 = Diastole

References:
    McDonald et al., PLOS Digital Health 2024, Section "Recurrent Neural Network"
    McDonald et al., CinC 2022, Section 2.1
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional


# ── State constants ──────────────────────────────────────────────────────
S1 = 0
SYSTOLE = 1
S2 = 2
DIASTOLE = 3
MURMUR = 4
UNANNOTATED = -1

STATE_NAMES = {S1: 'S1', SYSTOLE: 'Systole', S2: 'S2',
               DIASTOLE: 'Diastole', MURMUR: 'Murmur',
               UNANNOTATED: 'Unannotated'}

# Mapping from TSV 1-indexed states to our 0-indexed states
_TSV_STATE_MAP = {
    0: UNANNOTATED,   # TSV state 0 = Unannotated (discovered in Phase 1)
    1: S1,            # TSV state 1 = S1
    2: SYSTOLE,       # TSV state 2 = Systole
    3: S2,            # TSV state 3 = S2
    4: DIASTOLE,      # TSV state 4 = Diastole
}


# ── TSV loading ──────────────────────────────────────────────────────────

def load_segmentation_tsv(tsv_path):
    """Load segmentation annotation from a .tsv file.

    Parameters
    ----------
    tsv_path : str or Path
        Path to the .tsv file (3 columns: start_time, end_time, state).

    Returns
    -------
    list of tuple (float, float, int)
        Each tuple is (start_sec, end_sec, tsv_state) where tsv_state
        uses the TSV convention (0=Unannotated, 1=S1, 2=Systole, 3=S2, 4=Diastole).
    """
    tsv_path = Path(tsv_path)
    segments = []
    with open(tsv_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) != 3:
                continue
            try:
                start = float(parts[0])
                end = float(parts[1])
                state = int(parts[2])
                segments.append((start, end, state))
            except ValueError:
                continue  # skip header or malformed lines
    return segments


# ── Frame-level label creation ───────────────────────────────────────────

def create_frame_labels(segments, duration_frames, feature_rate=50):
    """Convert time-based segments to frame-level label array (4-state, no murmur).

    Parameters
    ----------
    segments : list of (start_sec, end_sec, tsv_state)
        Output of load_segmentation_tsv().
    duration_frames : int
        Total number of frames (T) for this recording's spectrogram.
    feature_rate : int
        Frames per second (default 50, matching hop=20ms).

    Returns
    -------
    np.ndarray, shape (duration_frames,), dtype int8
        Frame-level labels. Values in {-1, 0, 1, 2, 3}.
        -1 = unannotated or outside annotated segments.
    """
    labels = np.full(duration_frames, UNANNOTATED, dtype=np.int8)

    for start_sec, end_sec, tsv_state in segments:
        # Convert time to frame indices
        frame_start = int(start_sec * feature_rate)
        frame_end = int(end_sec * feature_rate)

        # Clip to valid range
        frame_start = max(0, frame_start)
        frame_end = min(duration_frames, frame_end)

        if frame_start >= frame_end:
            continue

        # Map TSV state to our convention
        our_state = _TSV_STATE_MAP.get(tsv_state, UNANNOTATED)
        labels[frame_start:frame_end] = our_state

    return labels


# ── Murmur relabelling ───────────────────────────────────────────────────

def apply_murmur_relabelling(labels, timing):
    """Relabel Systole frames as Murmur based on clinician timing annotation.

    Modifies `labels` IN-PLACE and returns it.

    Parameters
    ----------
    labels : np.ndarray, shape (T,)
        Frame-level labels with Systole=1. Will be modified in-place.
    timing : str
        One of: 'Holosystolic', 'Early-systolic', 'Mid-systolic', 'Late-systolic'.
        If None, nan, or unrecognised, no relabelling is performed.

    Returns
    -------
    np.ndarray
        The same array, modified in-place.
    """
    # Validate timing
    if timing is None or (isinstance(timing, float) and np.isnan(timing)):
        return labels
    timing = str(timing).strip()
    if timing not in ('Holosystolic', 'Early-systolic', 'Mid-systolic', 'Late-systolic'):
        return labels

    # Find contiguous Systole segments
    systole_segments = _find_contiguous_segments(labels, SYSTOLE)

    for s_start, s_end in systole_segments:
        s_len = s_end - s_start  # length in frames

        if s_len <= 0:
            continue

        if timing == 'Holosystolic':
            # Entire systole → murmur
            labels[s_start:s_end] = MURMUR

        elif timing == 'Early-systolic':
            # First 50% → murmur
            mid = s_start + s_len // 2
            if mid > s_start:  # at least 1 frame
                labels[s_start:mid] = MURMUR

        elif timing == 'Mid-systolic':
            # Middle 50% → murmur
            quarter = s_len // 4
            m_start = s_start + quarter
            m_end = s_end - quarter
            if m_end > m_start:  # at least 1 frame
                labels[m_start:m_end] = MURMUR

        elif timing == 'Late-systolic':
            # Last 50% → murmur
            mid = s_start + s_len // 2
            if s_end > mid:  # at least 1 frame
                labels[mid:s_end] = MURMUR

    return labels


def _find_contiguous_segments(labels, state):
    """Find start and end indices of contiguous runs of a given state.

    Returns
    -------
    list of (int, int)
        Each tuple is (start_idx, end_idx) where labels[start:end] == state.
    """
    segments = []
    in_segment = False
    seg_start = 0

    for i in range(len(labels)):
        if labels[i] == state:
            if not in_segment:
                seg_start = i
                in_segment = True
        else:
            if in_segment:
                segments.append((seg_start, i))
                in_segment = False

    # Handle segment that extends to the end
    if in_segment:
        segments.append((seg_start, len(labels)))

    return segments


# ── Per-recording pipeline ───────────────────────────────────────────────

def _get_recording_location(recording_id):
    """Extract auscultation location from recording ID.

    Examples:
        '2530_MV' → 'MV'
        '50260_MV_1' → 'MV'
    """
    parts = recording_id.split('_')
    # Location is always the second part (after patient_id)
    if len(parts) >= 2:
        return parts[1]
    return None


def _should_apply_murmur(recording_id, patients_df, recordings_df):
    """Determine whether to apply murmur relabelling for this recording.

    Returns
    -------
    (bool, str or None)
        (should_relabel, timing_string)
        should_relabel is True only if:
        - Patient has Murmur='Present'
        - Patient has systolic murmur timing annotation
        - Recording's auscultation location is listed in 'Murmur locations'
    """
    # Get patient_id from recording_id
    rec_row = recordings_df[recordings_df['recording_id'] == recording_id]
    if rec_row.empty:
        return False, None
    patient_id = rec_row.iloc[0]['patient_id']
    rec_location = _get_recording_location(recording_id)

    # Get patient info
    pat_row = patients_df[patients_df['patient_id'] == patient_id]
    if pat_row.empty:
        return False, None
    pat = pat_row.iloc[0]

    # Check if patient has murmur
    if pat.get('murmur') != 'Present':
        return False, None

    # Check systolic timing
    timing = pat.get('systolic_murmur_timing')
    if timing is None or (isinstance(timing, float) and np.isnan(timing)):
        return False, None
    timing = str(timing).strip()
    if timing not in ('Holosystolic', 'Early-systolic', 'Mid-systolic', 'Late-systolic'):
        return False, None

    # Check if this recording's location is in 'Murmur locations'
    murmur_locs = pat.get('murmur_locations')
    if murmur_locs is None or (isinstance(murmur_locs, float) and np.isnan(murmur_locs)):
        return False, None

    murmur_locs_list = [loc.strip() for loc in str(murmur_locs).split('+')]
    if rec_location not in murmur_locs_list:
        return False, None

    return True, timing


def create_labels_for_recording(recording_id, data_root, patients_df,
                                 recordings_df, spectrogram_T,
                                 feature_rate=50):
    """Full pipeline: TSV → 4-state labels → 5-state labels for one recording.

    Parameters
    ----------
    recording_id : str
        e.g. '2530_MV', '9979_TV'
    data_root : str or Path
        Path to data/raw/training_data/
    patients_df : pd.DataFrame
        From patients.csv
    recordings_df : pd.DataFrame
        From recordings.csv
    spectrogram_T : int
        Number of time frames in this recording's spectrogram.
        Labels array will have exactly this length.
    feature_rate : int
        Frames per second (default 50).

    Returns
    -------
    np.ndarray, shape (spectrogram_T,), dtype int8
        Frame-level labels with values in {-1, 0, 1, 2, 3, 4}.
    """
    data_root = Path(data_root)

    # Load TSV
    tsv_path = data_root / f"{recording_id}.tsv"
    if not tsv_path.exists():
        raise FileNotFoundError(f"TSV not found: {tsv_path}")

    segments = load_segmentation_tsv(tsv_path)

    # Create 4-state frame labels
    labels = create_frame_labels(segments, spectrogram_T, feature_rate)

    # Apply murmur relabelling if appropriate
    should_relabel, timing = _should_apply_murmur(
        recording_id, patients_df, recordings_df
    )
    if should_relabel:
        labels = apply_murmur_relabelling(labels, timing)

    return labels


# ── Batch processing ─────────────────────────────────────────────────────

def create_all_labels(recordings_df, patients_df, data_root,
                       spectrogram_dir, output_dir, feature_rate=50,
                       verbose=True):
    """Create and save 5-state labels for all recordings.

    Parameters
    ----------
    recordings_df : pd.DataFrame
        From recordings.csv. Must have 'recording_id' column.
    patients_df : pd.DataFrame
        From patients.csv. Must have 'patient_id', 'Murmur',
        'Systolic murmur timing', 'Murmur locations' columns.
    data_root : str or Path
        Path to data/raw/training_data/
    spectrogram_dir : str or Path
        Path to pre-computed spectrograms (to get T for each recording).
        If a spectrogram .npy doesn't exist, the recording is skipped.
    output_dir : str or Path
        Where to save label .npy files.
    feature_rate : int
        Frames per second (default 50).
    verbose : bool
        Print progress.

    Returns
    -------
    dict
        Summary statistics: counts per state, number processed/skipped,
        distribution of murmur relabelling.
    """
    data_root = Path(data_root)
    spectrogram_dir = Path(spectrogram_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Statistics tracking
    stats = {
        'processed': 0,
        'skipped_no_spectrogram': 0,
        'skipped_no_tsv': 0,
        'skipped_error': 0,
        'relabelled_murmur': 0,
        'frame_counts': {s: 0 for s in [UNANNOTATED, S1, SYSTOLE, S2, DIASTOLE, MURMUR]},
        'timing_counts': {},
    }

    recording_ids = recordings_df['recording_id'].tolist()

    for i, rec_id in enumerate(recording_ids):
        if verbose and (i + 1) % 500 == 0:
            print(f"  [{i+1}/{len(recording_ids)}] Processing labels...")

        # Check spectrogram exists (to get T)
        spec_path = spectrogram_dir / f"{rec_id}.npy"
        if not spec_path.exists():
            stats['skipped_no_spectrogram'] += 1
            continue

        # Get T from spectrogram shape
        spec = np.load(spec_path)
        spectrogram_T = spec.shape[1]  # (41, T)

        try:
            labels = create_labels_for_recording(
                rec_id, data_root, patients_df, recordings_df,
                spectrogram_T, feature_rate
            )

            # Track statistics
            for state_val in [UNANNOTATED, S1, SYSTOLE, S2, DIASTOLE, MURMUR]:
                stats['frame_counts'][state_val] += int(np.sum(labels == state_val))

            if MURMUR in labels:
                stats['relabelled_murmur'] += 1
                # Track which timing was used
                should_relabel, timing = _should_apply_murmur(
                    rec_id, patients_df, recordings_df
                )
                if timing:
                    stats['timing_counts'][timing] = stats['timing_counts'].get(timing, 0) + 1

            # Save
            np.save(output_dir / f"{rec_id}.npy", labels)
            stats['processed'] += 1

        except FileNotFoundError:
            stats['skipped_no_tsv'] += 1
        except Exception as e:
            stats['skipped_error'] += 1
            if verbose:
                print(f"  ERROR on {rec_id}: {e}")

    if verbose:
        print(f"\n  Done: {stats['processed']} processed, "
              f"{stats['skipped_no_spectrogram']} skipped (no spectrogram), "
              f"{stats['skipped_no_tsv']} skipped (no TSV), "
              f"{stats['skipped_error']} errors")
        print(f"  Recordings with Murmur frames: {stats['relabelled_murmur']}")
        print(f"\n  Frame counts:")
        total_labelled = sum(v for k, v in stats['frame_counts'].items() if k != UNANNOTATED)
        for state_val, name in STATE_NAMES.items():
            count = stats['frame_counts'][state_val]
            if total_labelled > 0 and state_val != UNANNOTATED:
                pct = 100 * count / total_labelled
                print(f"    {name:>12s}: {count:>10,d}  ({pct:5.1f}% of labelled)")
            else:
                print(f"    {name:>12s}: {count:>10,d}")
        if stats['timing_counts']:
            print(f"\n  Murmur timing distribution:")
            for timing, count in sorted(stats['timing_counts'].items()):
                print(f"    {timing:>20s}: {count}")

    return stats


# ── Verification helpers ─────────────────────────────────────────────────

def verify_label_spectrogram_alignment(label_dir, spectrogram_dir,
                                        recording_ids, n_check=10):
    """Verify that label and spectrogram arrays have matching T dimensions.

    Parameters
    ----------
    label_dir, spectrogram_dir : Path
    recording_ids : list of str
    n_check : int
        Number of recordings to check (set to len(recording_ids) for all).

    Returns
    -------
    list of str
        Recording IDs with mismatched T (should be empty).
    """
    label_dir = Path(label_dir)
    spectrogram_dir = Path(spectrogram_dir)
    mismatches = []

    for rec_id in recording_ids[:n_check]:
        lab_path = label_dir / f"{rec_id}.npy"
        spec_path = spectrogram_dir / f"{rec_id}.npy"

        if not lab_path.exists() or not spec_path.exists():
            continue

        lab = np.load(lab_path)
        spec = np.load(spec_path)

        T_label = lab.shape[0]
        T_spec = spec.shape[1]  # spectrogram is (41, T)

        if T_label != T_spec:
            mismatches.append(rec_id)
            print(f"  MISMATCH {rec_id}: label T={T_label}, spec T={T_spec}")

    if not mismatches:
        print(f"  All {min(n_check, len(recording_ids))} checked recordings have matching T.")

    return mismatches


def inspect_recording_labels(recording_id, label_dir, feature_rate=50):
    """Print detailed label breakdown for one recording (for manual inspection).

    Parameters
    ----------
    recording_id : str
    label_dir : str or Path
    feature_rate : int
    """
    label_dir = Path(label_dir)
    lab_path = label_dir / f"{recording_id}.npy"
    if not lab_path.exists():
        print(f"  Label file not found: {lab_path}")
        return

    labels = np.load(lab_path)
    T = len(labels)
    duration_sec = T / feature_rate

    print(f"\n  Recording: {recording_id}")
    print(f"  T = {T} frames ({duration_sec:.1f} seconds)")
    print(f"  Label distribution:")

    for state_val in [UNANNOTATED, S1, SYSTOLE, S2, DIASTOLE, MURMUR]:
        count = int(np.sum(labels == state_val))
        pct = 100 * count / T if T > 0 else 0
        name = STATE_NAMES[state_val]
        print(f"    {name:>12s}: {count:>6d} frames ({pct:5.1f}%)")

    has_murmur = np.any(labels == MURMUR)
    print(f"  Contains Murmur: {has_murmur}")

    # Show segment sequence (first 20 transitions)
    if T > 0:
        transitions = []
        current_state = labels[0]
        seg_start = 0
        for i in range(1, T):
            if labels[i] != current_state:
                transitions.append((seg_start, i, current_state))
                current_state = labels[i]
                seg_start = i
        transitions.append((seg_start, T, current_state))

        print(f"  Number of segments: {len(transitions)}")
        n_show = min(20, len(transitions))
        print(f"  First {n_show} segments:")
        for start, end, state in transitions[:n_show]:
            name = STATE_NAMES.get(state, f'?{state}')
            dur_ms = (end - start) / feature_rate * 1000
            print(f"    [{start:>5d}–{end:>5d}] {name:>12s}  ({dur_ms:.0f} ms)")