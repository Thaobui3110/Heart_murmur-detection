"""
src/evaluation/metrics.py
Metrics cho murmur detection task (weighted accuracy, per-class metrics).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix as sk_confusion_matrix


# Trọng số của PhysioNet Challenge 2022
MURMUR_WEIGHTS = {'Present': 5, 'Unknown': 3, 'Absent': 1}
CLASSES = ['Present', 'Unknown', 'Absent']


def compute_confusion_matrix(df_patients, label_col='true_label', pred_col='predicted_label'):
    """
    Tính confusion matrix ở mức bệnh nhân.

    Returns
    -------
    cm : np.ndarray shape (3, 3)
        cm[i, j] = số bệnh nhân có nhãn thực i và dự đoán j
        Thứ tự rows/cols: ['Present', 'Unknown', 'Absent']
    """
    cm = sk_confusion_matrix(
        df_patients[label_col],
        df_patients[pred_col],
        labels=CLASSES
    )
    return cm


def compute_weighted_accuracy(cm, weights=None):
    """
    Tính weighted accuracy theo công thức PhysioNet Challenge 2022.

    weighted_accuracy = sum(w_i * n_correct_i) / sum(w_i * n_total_i)
    """
    if weights is None:
        weights = [MURMUR_WEIGHTS[c] for c in CLASSES]

    weights = np.array(weights)
    n_correct = np.diag(cm)           # TP từng class
    n_total   = cm.sum(axis=1)        # Tổng thực tế từng class

    return np.dot(weights, n_correct) / np.dot(weights, n_total)


def compute_per_class_metrics(cm, labels=None):
    """
    Tính sensitivity, specificity, PPV, F1 cho từng class.

    Parameters
    ----------
    cm : np.ndarray (3, 3)
        Confusion matrix, thứ tự CLASSES = ['Present', 'Unknown', 'Absent']

    Returns
    -------
    df : pd.DataFrame
        Index = class names, Columns = [Sensitivity, Specificity, PPV, F1, Support]
    """
    if labels is None:
        labels = CLASSES

    n = cm.shape[0]
    metrics = {}

    for i, cls in enumerate(labels):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp          # Actual positives not predicted positive
        fp = cm[:, i].sum() - tp          # Predicted positive but actually negative
        tn = cm.sum() - tp - fn - fp

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        ppv         = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1          = (2 * ppv * sensitivity / (ppv + sensitivity)
                       if (ppv + sensitivity) > 0 else 0.0)
        support     = int(cm[i, :].sum())

        metrics[cls] = {
            'Sensitivity': sensitivity,
            'Specificity': specificity,
            'PPV':         ppv,
            'F1':          f1,
            'Support':     support,
        }

    return pd.DataFrame(metrics).T