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

# BlazePose 33-keypoint left/right swap map, used for horizontal-flip
# augmentation. Each pair is mirrored; unlisted indices (e.g. nose=0) map to
# themselves. Features are laid out [x0,y0,x1,y1,...], so keypoint i occupies
# columns 2i, 2i+1.
_LR_PAIRS = [
    (1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16),
    (17, 18), (19, 20), (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32),
]


def _build_flip_index(num_kp: int) -> np.ndarray:
    swap = list(range(num_kp))
    for a, b in _LR_PAIRS:
        if a < num_kp and b < num_kp:
            swap[a], swap[b] = b, a
    return np.asarray(swap, dtype=np.int64)


def augment_sequence(
    seq: np.ndarray,
    num_kp: int,
    rng: np.random.Generator,
    flip_p: float = 0.5,
    jitter_std: float = 0.02,
    scale_range: tuple[float, float] = (0.9, 1.1),
) -> np.ndarray:
    """Label-preserving augmentation of one normalized (T, F) sequence.

    - Horizontal flip (negate x + swap left/right keypoints): a punch mirrored
      is still a punch, and doubles effective coverage of handedness.
    - Scale jitter: robustness to body-size / distance estimation error.
    - Gaussian jitter: robustness to landmark noise.
    These operate in the already-normalized (torso-centered) space, matching
    what the model sees at inference.
    """
    out = seq.reshape(seq.shape[0], num_kp, 2).copy()

    if rng.random() < flip_p:
        flip_idx = _build_flip_index(num_kp)
        out[:, :, 0] *= -1.0  # mirror x about the torso center
        out = out[:, flip_idx, :]

    out *= rng.uniform(*scale_range)
    out += rng.normal(0.0, jitter_std, size=out.shape)

    return out.reshape(seq.shape).astype(np.float32)


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


class SplitView(Dataset):
    """A train/val/test view over a base dataset, augmenting only when asked.

    Keeping augmentation on the split (not the base) means the training view
    gets fresh random augmentation each epoch while val/test stay clean.
    """

    def __init__(
        self,
        base: PoseSequenceDataset,
        indices: list[int],
        augment: bool,
        seed: int = 0,
    ):
        self.base = base
        self.indices = indices
        self.augment = augment
        self.num_kp = (base.feature_dim or 0) // 2
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = self.indices[i]
        x = self.base.X_arr[idx]
        if self.augment:
            x = augment_sequence(x, self.num_kp, self.rng)
        return torch.from_numpy(np.ascontiguousarray(x)), int(self.base.y_arr[idx])
