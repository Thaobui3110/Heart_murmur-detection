"""
Consistent plot styling for the entire project.
Import this in every notebook before plotting.
"""

import matplotlib.pyplot as plt

# === Murmur class colors — used across ALL phases ===
MURMUR_COLORS = {
    "Present": "#E74C3C",   # red
    "Unknown": "#F39C12",   # amber
    "Absent":  "#2ECC71",   # green
}
MURMUR_ORDER = ["Present", "Unknown", "Absent"]

# === Heart sound state colors — for segmentation overlays ===
STATE_COLORS = {
    0: "#CCCCCC",   # Unannotated — gray
    1: "#3498DB",   # S1 — blue
    2: "#E74C3C",   # Systole — red
    3: "#2ECC71",   # S2 — green
    4: "#F1C40F",   # Diastole — yellow
}
STATE_LABELS = {0: "Unannotated", 1: "S1", 2: "Systole", 3: "S2", 4: "Diastole"}
STATE_LABELS = {1: "S1", 2: "Systole", 3: "S2", 4: "Diastole"}

# === Auscultation location colors ===
LOCATION_COLORS = {
    "AV": "#3498DB",
    "PV": "#E74C3C",
    "TV": "#2ECC71",
    "MV": "#F39C12",
    "Phc": "#9B59B6",
}

# === Global plot style ===
def setup_style():
    """Call once at the top of every notebook."""
    plt.rcParams.update({
        "figure.figsize": (8, 5),
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })