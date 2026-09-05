"""Load labeled pose-sequence samples exported by the web capture tool.

The exported JSON already contains **normalized** feature vectors (the web app's
`normalizePose` is the single source of truth), so this loader does no
normalization — it just validates shapes and stacks them into tensors. That is
what keeps the browser (inference) and Python (training) perfectly in sync.
"""

from __future__ import annotations

import glob
import json
import os
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

LABELS = {"normal": 0, "hostile": 1}
LABEL_NAMES = ["normal", "hostile"]


def _iter_files(paths: Iterable[str]) -> list[str]:
    files: list[str] = []
    for p in paths:
        if os.path.isdir(p):
            files.extend(sorted(glob.glob(os.path.join(p, "*.json"))))
        else:
            files.append(p)
    if not files:
        raise FileNotFoundError(f"No sample JSON files found in: {list(paths)}")
    return files


class PoseSequenceDataset(Dataset):
    """Sequences of shape (seq_len, feature_dim) with integer labels."""

    def __init__(self, paths: Iterable[str]):
        self.X: list[np.ndarray] = []
        self.y: list[int] = []
        self.seq_len: int | None = None
        self.feature_dim: int | None = None

        for path in _iter_files(paths):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            seq_len = data["seqLen"]
            feature_dim = data["featureDim"]
            if self.seq_len is None:
                self.seq_len, self.feature_dim = seq_len, feature_dim
            elif (seq_len, feature_dim) != (self.seq_len, self.feature_dim):
                raise ValueError(
                    f"{path}: shape ({seq_len},{feature_dim}) != "
                    f"expected ({self.seq_len},{self.feature_dim})"
                )

            for sample in data["samples"]:
                seq = np.asarray(sample["seq"], dtype=np.float32)
                if seq.shape != (seq_len, feature_dim):
                    raise ValueError(
                        f"{path}: sample shape {seq.shape} != "
                        f"({seq_len},{feature_dim})"
                    )
                if sample["label"] not in LABELS:
                    raise ValueError(f"{path}: unknown label {sample['label']!r}")
                self.X.append(seq)
                self.y.append(LABELS[sample["label"]])

        if not self.X:
            raise ValueError("Loaded 0 samples — record some clips first.")

        self.X_arr = np.stack(self.X)  # (N, T, F)
        self.y_arr = np.asarray(self.y, dtype=np.int64)

    def class_counts(self) -> dict[str, int]:
        return {
            name: int((self.y_arr == idx).sum())
            for name, idx in LABELS.items()
        }

    def __len__(self) -> int:
        return len(self.y_arr)

    def __getitem__(self, i: int):
        return torch.from_numpy(self.X_arr[i]), int(self.y_arr[i])
