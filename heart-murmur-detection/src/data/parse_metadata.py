"""
Parse CirCor DigiScope metadata into clean DataFrames.
Phase 1 - Task 1.1
"""

import pandas as pd
import numpy as np
import os
from collections import Counter
from scipy.io import wavfile


def load_patient_df(data_dir="data/raw/training_data"):
    """
    Load patient-level metadata from training_data.csv.
    Returns DataFrame: 1 row per patient (942 rows), cleaned columns.
    """
    csv_path = os.path.join(os.path.dirname(data_dir), "training_data.csv")
    df = pd.read_csv(csv_path)

    # --- Clean column names: lowercase, underscores ---
    df = df.rename(columns={
        "Patient ID": "patient_id",
        "Recording locations:": "recording_locations",
    })
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )

    # --- Derived: num_recordings from "AV+PV+TV+MV" ---
    df["num_recordings"] = df["recording_locations"].str.split("+").str.len()

    # --- Age as ordered categorical ---
    age_order = ["Neonate", "Infant", "Child", "Adolescent", "Young Adult"]
    df["age"] = pd.Categorical(df["age"], categories=age_order, ordered=True)

    # --- Pregnancy status: pandas reads True/False as bool already ---
    # Just ensure it stays as boolean (no conversion needed)
    df["pregnancy_status"] = df["pregnancy_status"].astype("boolean")

    # --- Murmur grading: "I/VI" -> 1, "II/VI" -> 2, "III/VI" -> 3 ---
    grade_map = {"I/VI": 1, "II/VI": 2, "III/VI": 3}
    df["systolic_murmur_grading"] = df["systolic_murmur_grading"].map(grade_map)
    df["diastolic_murmur_grading"] = df["diastolic_murmur_grading"].map(grade_map)

    return df


def _build_wav_path(data_dir, pid, loc, occurrence, total_at_loc):
    """
    Build correct .wav/.tsv path based on naming convention:
    - 1 recording at location  -> {pid}_{loc}.wav
    - N recordings at location -> {pid}_{loc}_1.wav, {pid}_{loc}_2.wav, ...
    """
    if total_at_loc == 1:
        stem = f"{pid}_{loc}"
    else:
        stem = f"{pid}_{loc}_{occurrence}"
    
    wav_path = os.path.join(data_dir, f"{stem}.wav")
    tsv_path = os.path.join(data_dir, f"{stem}.tsv")
    return wav_path, tsv_path


def load_recording_df(data_dir="data/raw/training_data", patient_df=None):
    """
    Build recording-level DataFrame: 1 row per .wav file.
    Handles patients with multiple recordings at the same location
    (e.g. AV+AV+PV+PV+TV+MV -> AV_1, AV_2, PV_1, PV_2, TV, MV).
    """
    if patient_df is None:
        patient_df = load_patient_df(data_dir)

    records = []
    missing_files = []
    total = len(patient_df)

    for i, (_, row) in enumerate(patient_df.iterrows()):
        pid = row["patient_id"]
        locations_list = row["recording_locations"].split("+")
        loc_counts = Counter(locations_list)

        # Track which occurrence we're on for each location
        loc_occurrence = {}

        for loc in locations_list:
            total_at_loc = loc_counts[loc]
            
            if total_at_loc == 1:
                occurrence = 1  # not used in filename
            else:
                loc_occurrence[loc] = loc_occurrence.get(loc, 0) + 1
                occurrence = loc_occurrence[loc]

            wav_path, tsv_path = _build_wav_path(
                data_dir, pid, loc, occurrence, total_at_loc
            )

            # Read wav to get duration
            duration = np.nan
            if os.path.exists(wav_path):
                sr, audio = wavfile.read(wav_path)
                duration = len(audio) / sr
            else:
                missing_files.append(wav_path)

            records.append({
                "patient_id": pid,
                "location": loc,
                "wav_path": wav_path,
                "tsv_path": tsv_path,
                "duration_seconds": duration,
                "murmur": row["murmur"],
            })

        if (i + 1) % 200 == 0:
            print(f"  Processed {i + 1}/{total} patients...")

    if missing_files:
        print(f"  WARNING: {len(missing_files)} files not found")

    return pd.DataFrame(records)


def save_metadata(data_dir="data/raw/training_data",
                  output_dir="data/metadata"):
    """Load, build, and save both DataFrames to CSV."""
    os.makedirs(output_dir, exist_ok=True)

    print("Loading patient metadata...")
    patient_df = load_patient_df(data_dir)
    patient_df.to_csv(os.path.join(output_dir, "patients.csv"), index=False)
    print(f"  Saved patients.csv: {patient_df.shape}")

    print("Building recording metadata (reading .wav durations)...")
    recording_df = load_recording_df(data_dir, patient_df)
    recording_df.to_csv(os.path.join(output_dir, "recordings.csv"), index=False)
    print(f"  Saved recordings.csv: {recording_df.shape}")

    return patient_df, recording_df