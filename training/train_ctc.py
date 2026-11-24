"""Train a CRNN + CTC captcha model and export it for the server.

  # the original dataset (labels in the notebook's format)
  python -m training.train_ctc --data words_captcha/spec_train_val.txt --name words-v2 --out runs/words-v2

  # real customer captchas plus synthetic padding, validated on real ones only
  python -m training.train_ctc \
      --data datasets/acme/labels.txt datasets/synth-train/labels.txt \
      --val datasets/acme-val/labels.txt --name acme-v1 --out runs/acme-v1

The run folder gets best.pt (PyTorch weights) and model/ (model.onnx + model.json).
Copy or link run/model to models/current and restart the server to serve it.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from captcha2text.model import ctc_greedy

from .common import CRNN, CaptchaDataset, collate, export_onnx, read_labels


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model: CRNN, loader: DataLoader, device: torch.device, charset: str) -> tuple[float, list]:
    model.eval()
    correct, total, wrong = 0, 0, []
    for images, _, _, texts in loader:
        logits = model(images.to(device)).float().cpu().numpy()
        for row, expected in zip(logits, texts):
            got, _ = ctc_greedy(row, charset)
            total += 1
            if got == expected:
                correct += 1
            elif len(wrong) < 10:
                wrong.append((expected, got))
    return correct / max(total, 1), wrong


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, nargs="+", required=True, help="labels files to train on")
    p.add_argument("--val", type=Path, nargs="*", default=[], help="labels files to validate on")
    p.add_argument("--val-fraction", type=float, default=0.1, help="used when --val is not given")
    p.add_argument("--name", required=True, help="model name shown by /v1/health, e.g. words-v2")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--charset", help="default: every character found in the labels")
    p.add_argument("--lowercase", action="store_true", help="lowercase all labels first")
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--height", type=int, default=48)
    p.add_argument("--channels", type=int, default=3, choices=[1, 3])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=8, help="stop after this many epochs without improvement")
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    def load(files):
        items = [it for f in files for it in read_labels(f)]
        return [(path, text.lower()) for path, text in items] if args.lowercase else items

    train_items = load(args.data)
    if args.val:
        val_items = load(args.val)
    else:
        random.shuffle(train_items)
        cut = max(1, int(len(train_items) * args.val_fraction))
        val_items, train_items = train_items[:cut], train_items[cut:]

    charset = args.charset or "".join(sorted({c for _, t in train_items + val_items for c in t}))
    unknown = {c for _, t in train_items + val_items for c in t} - set(charset)
    if unknown:
        raise SystemExit(f"Labels use characters missing from --charset: {''.join(sorted(unknown))}")
    max_len = max(len(t) for _, t in train_items)
    steps = args.width // 4
    if 2 * max_len + 1 > steps:
        raise SystemExit(f"--width {args.width} gives {steps} steps, too few for {max_len}-character labels")

    device = pick_device(args.device)
    print(f"{len(train_items)} train / {len(val_items)} val, charset {charset!r} ({len(charset)}), "
          f"max length {max_len}, device {device}")

    ds = dict(charset=charset, width=args.width, height=args.height, channels=args.channels)
    train_loader = DataLoader(
        CaptchaDataset(train_items, train=not args.no_augment, **ds), batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, collate_fn=collate, persistent_workers=args.workers > 0, drop_last=True,
    )
    val_loader = DataLoader(
        CaptchaDataset(val_items, train=False, **ds), batch_size=256, shuffle=False,
        num_workers=args.workers, collate_fn=collate, persistent_workers=args.workers > 0,
    )

    model = CRNN(num_classes=len(charset) + 1, channels=args.channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(train_loader), pct_start=0.15
    )
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    loss_device = torch.device("cpu") if device.type == "mps" else device  # CTC loss has no MPS kernel

    args.out.mkdir(parents=True, exist_ok=True)
    config = {"name": args.name, "charset": charset, "width": args.width, "height": args.height,
              "channels": args.channels}
    best, stale, history = -1.0, 0, []

    for epoch in range(1, args.epochs + 1):
        model.train()
        started, total_loss = time.time(), 0.0
        for images, targets, lengths, _ in train_loader:
            logits = model(images.to(device))
            log_probs = logits.log_softmax(2).permute(1, 0, 2).to(loss_device)  # (T, N, C)
            input_lengths = torch.full((log_probs.size(1),), log_probs.size(0), dtype=torch.long)
            loss = ctc(log_probs, targets, input_lengths, lengths)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        acc, wrong = evaluate(model, val_loader, device, charset)
        history.append({"epoch": epoch, "loss": total_loss / len(train_loader), "val_accuracy": acc})
        print(f"epoch {epoch:>3}  loss {total_loss / len(train_loader):.4f}  val exact match {acc:.2%}  "
              f"({time.time() - started:.0f}s)  e.g. {wrong[:3]}", flush=True)

        if acc > best:
            best, stale = acc, 0
            torch.save({"state_dict": model.state_dict(), "config": config}, args.out / "best.pt")
        else:
            stale += 1
            if stale >= args.patience:
                print(f"No improvement for {args.patience} epochs, stopping.")
                break

    checkpoint = torch.load(args.out / "best.pt", map_location="cpu")
    model = CRNN(num_classes=len(charset) + 1, channels=args.channels)
    model.load_state_dict(checkpoint["state_dict"])
    model_dir = export_onnx(
        model, config, args.out / "model",
        extra={"val_accuracy": round(best, 4), "train_samples": len(train_items), "val_samples": len(val_items),
               "data": [str(f) for f in args.data]},
    )
    (args.out / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    print(f"Best val exact match {best:.2%}. Exported {model_dir}/model.onnx and model.json")


if __name__ == "__main__":
    main()
