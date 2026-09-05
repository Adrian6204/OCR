"""Temporal action classifier: a small GRU over pose-keypoint sequences.

A GRU is the pragmatic choice for a portfolio proof-of-concept — it learns the
temporal signature of a motion (a punch is a fast extend-and-retract, a wave is
periodic) from far less data and compute than an ST-GCN or a 3D-CNN, and exports
cleanly to ONNX for in-browser inference.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PoseGRU(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden: int = 64,
        layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, feature_dim)
        out, _ = self.gru(x)
        last = out[:, -1, :]  # final timestep summary
        return self.head(last)  # (batch, num_classes) logits
