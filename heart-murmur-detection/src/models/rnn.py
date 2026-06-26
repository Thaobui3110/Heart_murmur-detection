"""
Task 3.3 — Bidirectional GRU + FC head architecture.

Implements MurmurRNN: the neural network that converts log-spectrogram frames
into 5-state posterior probabilities P(q_t = ξ_i | x_{1:T}, θ).

Architecture (from McDonald et al. PLOS Digital Health 2024 / CinC 2022 Table 1):
    Input  : (B, T, 41)   — batch of padded spectrograms
    BiGRU  : 3 layers, hidden=60, bidirectional → output (B, T, 120)
    Dropout: 0.1
    FC1    : Linear(120→60) + Tanh + Dropout(0.1)
    FC2    : Linear(60→40)  + Tanh
    Output : Linear(40→5)  — raw logits, NO softmax
    Output : (B, T, 5)

State index convention (matches labels.py):
    S1=0, Systole=1, S2=2, Diastole=3, Murmur=4

Usage:
    model = MurmurRNN()
    logits = model(features, lengths)      # (B, T_max, 5) — for training
    probs  = torch.softmax(logits, dim=-1) # (B, T_max, 5) — for inference
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class MurmurRNN(nn.Module):
    """Bidirectional GRU network for 5-state heart sound segmentation.

    Parameters
    ----------
    input_size : int
        Number of input features per frame (default 41 = frequency bins).
    hidden_size : int
        GRU hidden size per direction (default 60; total output = 120).
    num_layers : int
        Number of stacked GRU layers (default 3).
    num_classes : int
        Number of output states (default 5: S1, Systole, S2, Diastole, Murmur).
    dropout : float
        Dropout probability applied between GRU layers and between GRU→FC (default 0.1).
    """

    def __init__(self, input_size=41, hidden_size=60, num_layers=3,
                 num_classes=5, dropout=0.1):
        super(MurmurRNN, self).__init__()

        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.num_classes = num_classes

        # ── Bidirectional GRU ──────────────────────────────────────────
        # dropout applies between GRU layers (not after the last layer)
        # output shape: (B, T, hidden_size * 2) = (B, T, 120)
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # ── Dropout after GRU output (before FC layers) ────────────────
        self.dropout = nn.Dropout(p=dropout)

        # ── Fully connected layers ─────────────────────────────────────
        # FC1: 120 → 60 + Tanh
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.act1 = nn.Tanh()

        # FC2: 60 → 40 + Tanh
        self.fc2 = nn.Linear(hidden_size, 40)
        self.act2 = nn.Tanh()

        # Output: 40 → 5 (raw logits, no softmax)
        self.output_layer = nn.Linear(40, num_classes)

    def forward(self, features, lengths):
        """Forward pass.

        Parameters
        ----------
        features : FloatTensor, shape (B, T_max, 41)
            Padded batch of spectrograms (from DataLoader).
        lengths : list of int, length B
            Original (unpadded) sequence lengths, sorted descending.

        Returns
        -------
        logits : FloatTensor, shape (B, T_max, 5)
            Raw logits at every time step. Apply softmax for probabilities.
        """
        # ── Pack padded sequences (GRU ignores padding) ────────────────
        packed = pack_padded_sequence(
            features, lengths,
            batch_first=True,
            enforce_sorted=True,   # lengths must be descending (collate_fn guarantees this)
        )

        # ── BiGRU forward ──────────────────────────────────────────────
        packed_out, _ = self.gru(packed)

        # ── Unpack → (B, T_max, 120) ───────────────────────────────────
        gru_out, _ = pad_packed_sequence(packed_out, batch_first=True)
        # gru_out shape: (B, T_actual_max, 120)
        # Note: T_actual_max may be slightly < T_max due to packing/unpacking
        # We need to pad back to original T_max for consistent loss computation

        B, T_actual, H = gru_out.shape
        T_max = features.shape[1]

        if T_actual < T_max:
            # Pad with zeros to restore original T_max
            pad = torch.zeros(B, T_max - T_actual, H,
                              device=gru_out.device, dtype=gru_out.dtype)
            gru_out = torch.cat([gru_out, pad], dim=1)  # (B, T_max, 120)

        # ── Dropout after GRU ─────────────────────────────────────────
        x = self.dropout(gru_out)            # (B, T_max, 120)

        # ── FC layers ─────────────────────────────────────────────────
        x = self.act1(self.fc1(x))           # (B, T_max, 60)
        x = self.dropout(x)                  # dropout between FC1 and FC2
        x = self.act2(self.fc2(x))           # (B, T_max, 40)

        # ── Output logits ──────────────────────────────────────────────
        logits = self.output_layer(x)        # (B, T_max, 5)

        return logits

    def predict_proba(self, features, lengths):
        """Run inference and return softmax probabilities.

        Parameters
        ----------
        features : FloatTensor, shape (B, T_max, 41) or (T, 41) for single recording
        lengths : list of int or int

        Returns
        -------
        probs : FloatTensor, shape (B, T_max, 5)
            Posterior probabilities summing to 1.0 at each timestep.
        """
        self.eval()
        with torch.no_grad():
            if features.dim() == 2:
                # Single recording: (T, 41) → (1, T, 41)
                features = features.unsqueeze(0)
                lengths = [lengths] if isinstance(lengths, int) else lengths

            logits = self.forward(features, lengths)
            probs  = torch.softmax(logits, dim=-1)
        return probs


def count_parameters(model):
    """Count total and trainable parameters in a model."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def build_model(seed=42):
    """Build MurmurRNN with fixed random seed for reproducibility."""
    torch.manual_seed(seed)
    model = MurmurRNN(
        input_size=41,
        hidden_size=60,
        num_layers=3,
        num_classes=5,
        dropout=0.1,
    )
    return model
