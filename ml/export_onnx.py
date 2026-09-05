"""Export a trained checkpoint to ONNX for in-browser inference.

    python export_onnx.py pose_gru.pt

Writes to ../public/violence.onnx so the web app can fetch it at /violence.onnx.
"""

from __future__ import annotations

import argparse
import os

import torch

from model import PoseGRU

DEFAULT_OUT = os.path.join(os.path.dirname(__file__), "..", "public", "violence.onnx")


def export(checkpoint: str, out_path: str = DEFAULT_OUT) -> None:
    ckpt = torch.load(checkpoint, map_location="cpu")
    model = PoseGRU(
        feature_dim=ckpt["feature_dim"],
        hidden=ckpt["hidden"],
        layers=ckpt["layers"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    seq_len = ckpt["seq_len"]
    feature_dim = ckpt["feature_dim"]
    dummy = torch.randn(1, seq_len, feature_dim)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        out_path,
        input_names=["sequence"],
        output_names=["logits"],
        dynamic_axes={"sequence": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    print(f"Exported ONNX → {os.path.abspath(out_path)}  "
          f"(input [batch, {seq_len}, {feature_dim}], labels {ckpt['labels']})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", help="path to .pt checkpoint from train.py")
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()
    export(args.checkpoint, args.out)
