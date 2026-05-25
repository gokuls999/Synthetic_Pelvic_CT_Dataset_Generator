"""PyTorch Dataset over the cached pelvic slices.

Each cache file is a per-volume .npz with `slices` of shape (Z, H, W) in [-1, 1].
The dataset returns individual axial slices (or short Z-windows) plus a condition
dict {region_id, z_pos} so the model can learn slice continuity.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class PelvicSliceDataset(Dataset):
    """Yield 2D pelvic CT slices with (region_id, z_pos) conditioning.

    Args:
        labels_csv: path produced by `pseudo_labels.make_pseudo_labels`.
        slice_size: square side length (must match what was cached).
        return_index: if True also returns the (volume_idx, z_idx) tuple.
    """

    def __init__(self, labels_csv: str | Path, slice_size: int = 256,
                 return_index: bool = False):
        self.df = pd.read_csv(labels_csv)
        self.slice_size = slice_size
        self.return_index = return_index

        # Pre-index per-slice (volume_idx, z_idx) without loading pixels.
        # We need to know the per-volume slice count -> read npz headers lazily.
        from .progress import pbar

        self._slice_counts = []
        self._cache_paths = []
        self._region_ids = []
        for _, row in pbar(list(self.df.iterrows()), total=len(self.df),
                           desc="Indexing slice cache", unit="vol", leave=False):
            self._cache_paths.append(row["cache_path"])
            self._region_ids.append(int(row["region_id"]))
            with np.load(row["cache_path"]) as npz:
                self._slice_counts.append(int(npz["slices"].shape[0]))

        # Flat index of (vol_idx, z_idx)
        self._index = []
        for vi, n in enumerate(self._slice_counts):
            for zi in range(n):
                self._index.append((vi, zi))

        # Small per-volume LRU cache so we don't reopen npz for every slice.
        self._cache_vol_idx = -1
        self._cache_arr = None

    def __len__(self):
        return len(self._index)

    def _load_volume(self, vi: int) -> np.ndarray:
        if vi == self._cache_vol_idx:
            return self._cache_arr
        with np.load(self._cache_paths[vi]) as npz:
            arr = np.asarray(npz["slices"]).astype(np.float32)
        self._cache_vol_idx = vi
        self._cache_arr = arr
        return arr

    def __getitem__(self, idx):
        vi, zi = self._index[idx]
        arr = self._load_volume(vi)
        slc = arr[zi]                                      # (H, W)
        if slc.shape[0] != self.slice_size:
            # Defensive: shouldn't trigger if cache matches config.
            from scipy.ndimage import zoom as _z
            slc = _z(slc, self.slice_size / slc.shape[0], order=1).astype(np.float32)
        slc = torch.from_numpy(slc).unsqueeze(0)           # (1, H, W)

        n = self._slice_counts[vi]
        z_pos = (zi / max(n - 1, 1)) * 2.0 - 1.0           # normalized [-1, 1]
        cond = {
            "region_id": torch.tensor(self._region_ids[vi], dtype=torch.long),
            "z_pos": torch.tensor(z_pos, dtype=torch.float32),
        }
        if self.return_index:
            return slc, cond, (vi, zi)
        return slc, cond
