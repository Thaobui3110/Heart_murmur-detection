"""
src/evaluation/analysis.py
Error analysis và subgroup analysis cho murmur detection.
"""

import numpy as np
import pandas as pd


def identify_errors(df_pat, label_col='true_murmur', pred_col='pred_murmur'):
    """
    Tách FP và FN từ patient-level predictions.

    Returns
    -------
    fn_df : pd.DataFrame — Present patients bị đoán sai
    fp_df : pd.DataFrame — Absent patients bị đoán là Present
    """
    # False Negative: Present thực tế nhưng không được đoán là Present
    fn_df = df_pat[
        (df_pat[label_col] == 'Present') &
        (df_pat[pred_col] != 'Present')
    ].copy()

    # False Positive: Absent thực tế nhưng bị đoán là Present
    fp_df = df_pat[
        (df_pat[label_col] == 'Absent') &
        (df_pat[pred_col] == 'Present')
    ].copy()

    print(f'False Negatives (Present missed): {len(fn_df)}')
    print(f'  → Predicted Absent:  {(fn_df[pred_col] == "Absent").sum()}')
    print(f'  → Predicted Unknown: {(fn_df[pred_col] == "Unknown").sum()}')
    print(f'\nFalse Positives (Absent → Present): {len(fp_df)}')

    return fn_df, fp_df