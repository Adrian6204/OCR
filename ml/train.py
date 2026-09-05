"""Train the pose-sequence violence classifier.

Usage:
    python train.py --data ./data --epochs 60
    python train.py --data samples1.json samples2.json --epochs 80 --export

`--data` accepts JSON files and/or directories of JSON files exported from the
web capture tool (`/capture`). With `--export`, writes ONNX to the web app's
public/ dir on completion.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from dataset import LABEL_NAMES, PoseSequenceDataset
from model import PoseGRU


def stratified_split(labels: np.ndarray, val_frac: float, seed: int):
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_frac)))
        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())
    return train_idx, val_idx


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[float, float]:
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    crit = nn.CrossEntropyLoss()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += crit(logits, y).item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        total += len(y)
    return loss_sum / max(1, total), correct / max(1, total)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True, help="JSON files or dirs")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="pose_gru.pt")
    ap.add_argument("--export", action="store_true", help="export ONNX when done")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = PoseSequenceDataset(args.data)
    print(f"Loaded {len(ds)} samples {ds.class_counts()}  "
          f"seq_len={ds.seq_len} feature_dim={ds.feature_dim}  device={device}")
    if min(ds.class_counts().values()) < 5:
        print("⚠ Fewer than 5 samples in a class — expect an unreliable model. "
              "Record more via /capture.")

    train_idx, val_idx = stratified_split(ds.y_arr, args.val_frac, args.seed)
    train_loader = DataLoader(Subset(ds, train_idx), batch_size=args.batch_size,
                              shuffle=True)
    val_loader = DataLoader(Subset(ds, val_idx), batch_size=args.batch_size)

    # Class weighting guards against imbalance.
    counts = np.bincount(ds.y_arr[train_idx], minlength=2).astype(np.float32)
    weights = torch.tensor(counts.sum() / (2 * np.maximum(counts, 1)),
                           dtype=torch.float32, device=device)

    model = PoseGRU(ds.feature_dim, hidden=args.hidden, layers=args.layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss(weight=weights)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()

        val_loss, val_acc = evaluate(model, val_loader, device)
        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(
                {"state_dict": model.state_dict(),
                 "feature_dim": ds.feature_dim,
                 "seq_len": ds.seq_len,
                 "hidden": args.hidden,
                 "layers": args.layers,
                 "labels": LABEL_NAMES},
                args.out,
            )
        if epoch % 5 == 0 or epoch == args.epochs:
            print(f"epoch {epoch:3d}  val_loss {val_loss:.3f}  "
                  f"val_acc {val_acc:.3f}  best {best_acc:.3f}")

    print(f"Best val accuracy {best_acc:.3f} — saved {args.out}")

    if args.export:
        from export_onnx import export
        export(args.out)


if __name__ == "__main__":
    main()
