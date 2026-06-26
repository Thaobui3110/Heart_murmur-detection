"""
Task 3.2 — Data loading and batching for variable-length PCG sequences.
v3: In-memory cache support — eliminates per-batch disk I/O during training.

Key insight: reading 16 .npy files per batch × 160 batches = 2560 disk reads
per epoch. Loading everything into RAM once (~1.5 GB) makes each epoch
10-20× faster.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm


def load_dataset_to_ram(recording_ids, spectrogram_dir, label_dir):
    """Load ALL spectrograms and labels into RAM once.

    Parameters
    ----------
    recording_ids : list of str  (all 3163 IDs)
    spectrogram_dir, label_dir : str or Path

    Returns
    -------
    spec_cache  : dict {recording_id: np.ndarray (41, T)}
    label_cache : dict {recording_id: np.ndarray (T,)}
    lengths_dict: dict {recording_id: int T}
    """
    spectrogram_dir = Path(spectrogram_dir)
    label_dir       = Path(label_dir)

    spec_cache   = {}
    label_cache  = {}
    lengths_dict = {}

    for rec_id in tqdm(recording_ids, desc="Loading dataset to RAM"):
        spec  = np.load(spectrogram_dir / f"{rec_id}.npy")   # (41, T)
        label = np.load(label_dir       / f"{rec_id}.npy")   # (T,)
        spec_cache[rec_id]   = spec
        label_cache[rec_id]  = label
        lengths_dict[rec_id] = spec.shape[1]

    total_mb = (sum(v.nbytes for v in spec_cache.values()) +
                sum(v.nbytes for v in label_cache.values())) / 1e6
    print(f"RAM used: ~{total_mb:.0f} MB — {len(spec_cache)} recordings cached")

    return spec_cache, label_cache, lengths_dict


class PCGDataset(Dataset):
    """PyTorch Dataset for PCG spectrograms and frame-level labels.

    Parameters
    ----------
    recording_ids : list of str
    spectrogram_dir : str or Path  (used only if cache not provided)
    label_dir : str or Path        (used only if cache not provided)
    lengths_dict : dict {recording_id: T}, optional
    spec_cache : dict {recording_id: np.ndarray}, optional
        If provided, __getitem__ reads from RAM instead of disk.
    label_cache : dict {recording_id: np.ndarray}, optional
    """

    def __init__(self, recording_ids, spectrogram_dir, label_dir,
                 lengths_dict=None, spec_cache=None, label_cache=None):
        self.recording_ids   = list(recording_ids)
        self.spectrogram_dir = Path(spectrogram_dir)
        self.label_dir       = Path(label_dir)
        self.spec_cache      = spec_cache
        self.label_cache     = label_cache

        # Build lengths — from cache (instant) or dict (instant) or disk (slow)
        if lengths_dict is not None:
            self._lengths = [lengths_dict[r] for r in self.recording_ids]
        elif spec_cache is not None:
            self._lengths = [spec_cache[r].shape[1] for r in self.recording_ids]
        else:
            self._lengths = []
            for rec_id in self.recording_ids:
                spec = np.load(self.spectrogram_dir / f"{rec_id}.npy",
                               mmap_mode='r')
                self._lengths.append(spec.shape[1])

    def __len__(self):
        return len(self.recording_ids)

    def __getitem__(self, idx):
        rec_id = self.recording_ids[idx]

        # Read from RAM cache if available, else from disk
        if self.spec_cache is not None:
            spec      = self.spec_cache[rec_id]       # (41, T) — from RAM
            labels_np = self.label_cache[rec_id]      # (T,)    — from RAM
        else:
            spec      = np.load(self.spectrogram_dir / f"{rec_id}.npy")
            labels_np = np.load(self.label_dir       / f"{rec_id}.npy")

        features = torch.FloatTensor(spec.T)                        # (T, 41)
        labels   = torch.LongTensor(labels_np.astype(np.int64))     # (T,)

        return {
            'features':     features,
            'labels':       labels,
            'length':       features.shape[0],
            'recording_id': rec_id,
        }

    def get_length(self, idx):
        return self._lengths[idx]


def pcg_collate_fn(batch):
    """Collate variable-length samples into a padded batch (sorted descending)."""
    batch = sorted(batch, key=lambda x: x['length'], reverse=True)

    lengths       = [item['length'] for item in batch]
    recording_ids = [item['recording_id'] for item in batch]
    B             = len(batch)
    T_max         = lengths[0]

    features_padded = torch.zeros(B, T_max, 41)
    labels_padded   = torch.full((B, T_max), -1, dtype=torch.long)

    for i, item in enumerate(batch):
        T = item['length']
        features_padded[i, :T, :] = item['features']
        labels_padded[i,   :T]    = item['labels']

    return {
        'features':      features_padded,
        'labels':        labels_padded,
        'lengths':       lengths,
        'recording_ids': recording_ids,
    }


def create_dataloader(recording_ids, spectrogram_dir, label_dir,
                      batch_size=16, shuffle=True, num_workers=0,
                      lengths_dict=None, spec_cache=None, label_cache=None):
    """Create DataLoader. Pass spec_cache+label_cache for RAM-based loading."""
    dataset = PCGDataset(
        recording_ids, spectrogram_dir, label_dir,
        lengths_dict=lengths_dict,
        spec_cache=spec_cache,
        label_cache=label_cache,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=pcg_collate_fn,
        num_workers=num_workers,
        pin_memory=False,
    )