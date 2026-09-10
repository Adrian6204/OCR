"""Train the pose-sequence violence classifier.

Usage:
    python train.py --data ./data --epochs 80 --export
    python train.py --data samples1.json samples2.json --no-augment

`--data` accepts JSON files and/or directories of JSON files exported from the
web capture tool (`/capture`) or produced by `preprocess_dataset.py`. With
`--export`, writes ONNX to the web app's public/ dir on completion.

Includes train-time augmentation, a stratified train/val/test split, early
stopping, LR scheduling, and precision/recall/F1 + a confusion matrix saved to
metrics.json.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import LABEL_NAMES, PoseSequenceDataset, SplitView
from model import PoseGRU


def stratified_split(labels: np.ndarray, val_frac: float, test_frac: float, seed: int):
    rng = np.random.default_rng(seed)
    train, val, test = [], [], []
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_frac)))
        n_test = int(round(len(idx) * test_frac))
        val.extend(idx[:n_val].tolist())
        test.extend(idx[n_val : n_val + n_test].tolist())
        train.extend(idx[n_val + n_test :].tolist())
    return train, val, test


@torch.no_grad()
def evaluate(model, loader, device, crit):
    model.eval()
    loss_sum = 0.0
    total = 0
    preds, gts = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += crit(logits, y).item() * len(y)
        total += len(y)
        preds.extend(logits.argmax(1).cpu().tolist())
        gts.extend(y.cpu().tolist())
    return loss_sum / max(1, total), np.asarray(preds), np.asarray(gts)


def metrics_for(preds: np.ndarray, gts: np.ndarray) -> dict:
    """Binary metrics for the positive (hostile=1) class + confusion matrix."""
    if len(gts) == 0:
        return {}
    tp = int(((preds == 1) & (gts == 1)).sum())
    fp = int(((preds == 1) & (gts == 0)).sum())
    fn = int(((preds == 0) & (gts == 1)).sum())
    tn = int(((preds == 0) & (gts == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    acc = (tp + tn) / max(1, len(gts))
    return {
        "accuracy": round(acc, 4),
        "precision_hostile": round(precision, 4),
        "recall_hostile": round(recall, 4),
        "f1_hostile": round(f1, 4),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True, help="JSON files or dirs")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--patience", type=int, default=15, help="early-stop patience")
    ap.add_argument("--no-augment", action="store_true", help="disable augmentation")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="pose_gru.pt")
    ap.add_argument("--metrics", default="metrics.json")
    ap.add_argument("--export", action="store_true", help="export ONNX when done")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = PoseSequenceDataset(args.data)
    print(f"Loaded {len(ds)} samples {ds.class_counts()}  "
          f"seq_len={ds.seq_len} feature_dim={ds.feature_dim}  device={device}")
    if min(ds.class_counts().values()) < 5:
        print("⚠ Fewer than 5 samples in a class — expect an unreliable model. "
              "Record/preprocess more data.")

    train_idx, val_idx, test_idx = stratified_split(
        ds.y_arr, args.val_frac, args.test_frac, args.seed
    )
    augment = not args.no_augment
    train_loader = DataLoader(
        SplitView(ds, train_idx, augment=augment, seed=args.seed),
        batch_size=args.batch_size, shuffle=True,
    )
    val_loader = DataLoader(SplitView(ds, val_idx, augment=False), batch_size=args.batch_size)
    test_loader = (
        DataLoader(SplitView(ds, test_idx, augment=False), batch_size=args.batch_size)
        if test_idx else None
    )
    print(f"Split: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}  "
          f"augment={augment}")

    counts = np.bincount(ds.y_arr[train_idx], minlength=2).astype(np.float32)
    weights = torch.tensor(counts.sum() / (2 * np.maximum(counts, 1)),
                           dtype=torch.float32, device=device)

    model = PoseGRU(ds.feature_dim, hidden=args.hidden, layers=args.layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)
    crit = nn.CrossEntropyLoss(weight=weights)

    best_val = float("inf")
    best_f1 = 0.0
    since_improve = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()

        val_loss, val_preds, val_gts = evaluate(model, val_loader, device, crit)
        vm = metrics_for(val_preds, val_gts)
        sched.step(val_loss)

        improved = val_loss < best_val - 1e-4
        if improved:
            best_val = val_loss
            best_f1 = vm.get("f1_hostile", 0.0)
            since_improve = 0
            torch.save(
                {"state_dict": model.state_dict(),
                 "feature_dim": ds.feature_dim,
                 "seq_len": ds.seq_len,
                 "hidden": args.hidden,
                 "layers": args.layers,
                 "labels": LABEL_NAMES},
                args.out,
            )
        else:
            since_improve += 1

        if epoch % 5 == 0 or epoch == args.epochs or improved:
            print(f"epoch {epoch:3d}  val_loss {val_loss:.3f}  "
                  f"val_acc {vm.get('accuracy', 0):.3f}  val_f1 {vm.get('f1_hostile', 0):.3f}"
                  f"{'  *' if improved else ''}")

        if since_improve >= args.patience:
            print(f"Early stop at epoch {epoch} (no val improvement in {args.patience}).")
            break

    print(f"Best val loss {best_val:.3f}  (f1 {best_f1:.3f}) — saved {args.out}")

    # Final report on the held-out test set using the best checkpoint.
    report = {"val_best_loss": round(best_val, 4), "val_best_f1": round(best_f1, 4)}
    if test_loader is not None:
        ckpt = torch.load(args.out, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        _, test_preds, test_gts = evaluate(model, test_loader, device, crit)
        report["test"] = metrics_for(test_preds, test_gts)
        print(f"Test metrics: {report['test']}")

    with open(args.metrics, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {args.metrics}")

    if args.export:
        from export_onnx import export
        export(args.out)


if __name__ == "__main__":
    main()
