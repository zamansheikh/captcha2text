"""Dataset, CRNN model and ONNX export shared by the training scripts."""

from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch import nn
from torch.utils.data import Dataset

from captcha2text.images import decode_image, preprocess


def read_labels(labels_file: Path, image_dir: Path | None = None) -> list[tuple[Path, str]]:
    """Read "<name> <text>" lines. A name without an extension means <name>.png."""
    labels_file = Path(labels_file)
    image_dir = Path(image_dir) if image_dir else labels_file.parent
    items = []
    for line in labels_file.read_text().splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        name, text = parts[0], parts[1]
        path = image_dir / name
        if path.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            path = image_dir / f"{name}.png"
        items.append((path, text))
    return items


def augment(rgb: np.ndarray) -> np.ndarray:
    """Small random changes so the model doesn't memorise exact pixels."""
    img = Image.fromarray(rgb)
    w, h = img.size
    angle = random.uniform(-4, 4)
    dx, dy = random.uniform(-0.04, 0.04) * w, random.uniform(-0.06, 0.06) * h
    img = img.rotate(angle, resample=Image.BILINEAR, translate=(dx, dy), fillcolor=(255, 255, 255))
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.8, 1.2))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.7, 1.3))
    if random.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.2, 0.9)))
    out = np.asarray(img, dtype=np.float32)
    if random.random() < 0.3:
        out = out + np.random.normal(0, random.uniform(2, 10), out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


class CaptchaDataset(Dataset):
    def __init__(self, items, charset: str, width: int, height: int, channels: int, train: bool):
        self.items = items
        self.index = {c: i + 1 for i, c in enumerate(charset)}  # 0 is the CTC blank
        self.width, self.height, self.channels = width, height, channels
        self.train = train

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i):
        path, text = self.items[i]
        rgb = decode_image(Path(path).read_bytes())
        if self.train:
            rgb = augment(rgb)
        x = preprocess(rgb, self.width, self.height, self.channels, "NCHW")[0]
        target = [self.index[c] for c in text]
        return torch.from_numpy(x), torch.tensor(target, dtype=torch.long), text


def collate(batch):
    images, targets, texts = zip(*batch)
    lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    return torch.stack(images), torch.cat(targets), lengths, list(texts)


def conv(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class CRNN(nn.Module):
    """CNN reads columns of the image, a bidirectional GRU reads them left to right,
    and CTC lines the output up with the text. One forward pass per image."""

    def __init__(self, num_classes: int, channels: int = 3):
        super().__init__()
        self.cnn = nn.Sequential(
            conv(channels, 32), nn.MaxPool2d(2, 2),            # H/2  W/2
            conv(32, 64), nn.MaxPool2d(2, 2),                  # H/4  W/4
            conv(64, 128), conv(128, 128), nn.MaxPool2d((2, 1)),  # H/8  W/4
            conv(128, 256), conv(256, 256), nn.MaxPool2d((2, 1)),  # H/16 W/4
            nn.Dropout2d(0.1),
        )
        self.rnn = nn.GRU(256, 128, num_layers=2, bidirectional=True, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.cnn(x)                 # (N, 256, H', T)
        f = f.mean(dim=2)               # (N, 256, T)
        f = f.permute(0, 2, 1)          # (N, T, 256)
        f, _ = self.rnn(f)
        return self.fc(f)               # (N, T, classes) logits


def export_onnx(model: CRNN, config: dict, out_dir: Path, extra: dict | None = None) -> Path:
    """Write model.onnx + model.json in the layout the server loads."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = model.eval().cpu()
    dummy = torch.zeros(1, config["channels"], config["height"], config["width"])
    torch.onnx.export(
        model,
        (dummy,),
        str(out_dir / "model.onnx"),
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    spec = {
        "name": config["name"],
        "input": {
            "width": config["width"],
            "height": config["height"],
            "channels": config["channels"],
            "layout": "NCHW",
        },
        "decoder": "ctc",
        "charset": config["charset"],
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **(extra or {}),
    }
    (out_dir / "model.json").write_text(json.dumps(spec, indent=2) + "\n")
    return out_dir
