"""Loads an ONNX model and turns an image into text.

A model folder holds two files:

  model.onnx   the network
  model.json   how to feed it and how to read its output

model.json fields:

  name         shown by GET /v1/health, e.g. "words-v1"
  input        width       pixels, or null to keep the image's aspect ratio
               height      pixels
               channels    3 (RGB) or 1 (grayscale)
               layout      "NHWC" or "NCHW"
               resample    "bilinear" (same as tf.image.resize, default) or "lanczos" (PIL)
               mean, std   after scaling pixels to 0..1, apply (x - mean) / std (default 0, 1)
               max_width   cap for width=null (default 2048)
  decoder      "ctc"  - the first output holds per-step scores for every class; class 0 is
                        the CTC blank and class i is charset[i - 1]. Used by CRNN models
                        from training/ and by ddddocr models.
               "ids"  - outputs "ids" (T,) and "probs" (T,) already decoded inside the
                        graph. Used by the notebook's attention model.
  output_layout  ctc only: "NTC" (batch, time, classes; default) or "TNC"
  charset      ctc only: a string, or a list when some entries are longer than one character
  allowed      ctc only, optional: the characters the answer may contain. Every other class
               is ruled out before the best class is picked at each step.
  lowercase    optional: lowercase the answer (default false)
  tokens       ids only: tokens[id] is the string for that id ("" for padding)
  end_token    ids only: the token that ends the answer, default "<end>"

Other fields (training notes, licence, source) are allowed and ignored.
"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .images import preprocess


@dataclass(frozen=True)
class ModelSpec:
    name: str
    width: int | None
    height: int
    channels: int = 3
    layout: str = "NHWC"
    resample: str = "bilinear"
    mean: float = 0.0
    std: float = 1.0
    max_width: int = 2048
    decoder: str = "ctc"
    output_layout: str = "NTC"
    charset: tuple[str, ...] = ()
    allowed: str | None = None
    lowercase: bool = False
    tokens: list[str] = field(default_factory=list)
    end_token: str = "<end>"

    @classmethod
    def load(cls, path: Path) -> "ModelSpec":
        raw = json.loads(Path(path).read_text())
        inp = raw.get("input", {})
        width = inp.get("width")
        spec = cls(
            name=raw["name"],
            width=int(width) if width is not None else None,
            height=int(inp["height"]),
            channels=int(inp.get("channels", 3)),
            layout=inp.get("layout", "NHWC"),
            resample=inp.get("resample", "bilinear"),
            mean=float(inp.get("mean", 0.0)),
            std=float(inp.get("std", 1.0)),
            max_width=int(inp.get("max_width", 2048)),
            decoder=raw.get("decoder", "ctc"),
            output_layout=raw.get("output_layout", "NTC"),
            charset=tuple(raw.get("charset", "")),
            allowed=raw.get("allowed"),
            lowercase=bool(raw.get("lowercase", False)),
            tokens=list(raw.get("tokens", [])),
            end_token=raw.get("end_token", "<end>"),
        )
        if spec.decoder not in ("ctc", "ids"):
            raise ValueError(f"{path}: decoder must be ctc or ids, got {spec.decoder!r}")
        if spec.decoder == "ctc" and not spec.charset:
            raise ValueError(f"{path}: a ctc model needs a charset")
        if spec.decoder == "ids" and not spec.tokens:
            raise ValueError(f"{path}: an ids model needs tokens")
        if spec.output_layout not in ("NTC", "TNC"):
            raise ValueError(f"{path}: output_layout must be NTC or TNC")
        if spec.allowed is not None and not set(spec.allowed) & set(spec.charset):
            raise ValueError(f"{path}: none of the allowed characters are in the charset")
        return spec

    def class_mask(self) -> np.ndarray | None:
        """Boolean mask over classes (blank included) that may be picked, or None for all."""
        if self.allowed is None:
            return None
        allowed = set(self.allowed)
        return np.array([True] + [c in allowed for c in self.charset])


@dataclass(frozen=True)
class Solution:
    text: str
    confidence: float
    processing_ms: int


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def ctc_greedy(logits: np.ndarray, charset, mask: np.ndarray | None = None) -> tuple[str, float]:
    """Best-path CTC decoding of (T, classes) logits.

    `mask` rules out classes before picking. Confidence is the product of each output
    character's peak probability.
    """
    logits = logits.astype(np.float32)
    if mask is not None:
        logits = np.where(mask[None, :], logits, -np.inf)
    probs = softmax(logits, axis=-1)
    best = probs.argmax(axis=-1)
    peak = probs.max(axis=-1)
    chars: list[str] = []
    confs: list[float] = []
    prev = 0
    for t, k in enumerate(best):
        k = int(k)
        if k != 0 and k != prev:
            chars.append(charset[k - 1])
            confs.append(float(peak[t]))
        elif k != 0 and k == prev:
            confs[-1] = max(confs[-1], float(peak[t]))
        prev = k
    confidence = float(np.prod(confs)) if confs else 0.0
    return "".join(chars), confidence


def ids_greedy(ids: np.ndarray, probs: np.ndarray, tokens: list[str], end_token: str) -> tuple[str, float]:
    """Read token ids produced by a model that decodes inside its own graph."""
    out: list[str] = []
    confidence = 1.0
    for i, p in zip(ids.reshape(-1), probs.reshape(-1)):
        i = int(i)
        token = tokens[i] if 0 <= i < len(tokens) else ""
        confidence *= float(p)
        if token == end_token:
            break
        out.append(token)
    text = "".join(out)
    return text, (confidence if text else 0.0)


class Solver:
    def __init__(self, model_dir: Path, threads: int = 0):
        import onnxruntime as ort

        model_dir = Path(model_dir)
        self.spec = ModelSpec.load(model_dir / "model.json")
        self._mask = self.spec.class_mask()
        options = ort.SessionOptions()
        if threads > 0:
            options.intra_op_num_threads = threads
        options.log_severity_level = 3  # some exported models declare output shapes loosely
        self.session = ort.InferenceSession(
            str(model_dir / "model.onnx"), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

    @property
    def name(self) -> str:
        return self.spec.name

    def solve(self, rgb: np.ndarray) -> Solution:
        started = time.perf_counter()
        s = self.spec
        x = preprocess(rgb, s.width, s.height, s.channels, s.layout, s.resample, s.mean, s.std, s.max_width)
        outputs = self.session.run(None, {self.input_name: x})

        if s.decoder == "ctc":
            by_name = dict(zip(self.output_names, outputs))
            scores = by_name.get("logits", outputs[0])
            scores = scores[:, 0, :] if s.output_layout == "TNC" else scores[0]
            text, confidence = ctc_greedy(scores, s.charset, self._mask)
        else:
            by_name = dict(zip(self.output_names, outputs))
            text, confidence = ids_greedy(by_name["ids"], by_name["probs"], s.tokens, s.end_token)

        if s.lowercase:
            text = text.lower()
        ms = round((time.perf_counter() - started) * 1000)
        return Solution(text=text, confidence=round(confidence, 4), processing_ms=ms)


_MODEL_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class ModelNotFound(Exception):
    pass


def model_exists(models_dir: Path, name: str) -> bool:
    return bool(_MODEL_NAME.match(name)) and (Path(models_dir) / name / "model.json").is_file()


def list_models(models_dir: Path) -> list[dict]:
    """Model folders with a few facts from each model.json, for the admin panel."""
    out = []
    root = Path(models_dir)
    if not root.is_dir():
        return out
    for folder in sorted(root.iterdir()):
        if not model_exists(root, folder.name):
            continue
        try:
            raw = json.loads((folder / "model.json").read_text())
        except (OSError, ValueError):
            continue
        source = raw.get("source") or {}
        out.append({
            "folder": folder.name,
            "name": raw.get("name", folder.name),
            "decoder": raw.get("decoder", "ctc"),
            "link_to": folder.resolve().name if folder.is_symlink() else None,
            "allowed": raw.get("allowed"),
            "source": source.get("project") if isinstance(source, dict) else str(source),
            "val_accuracy": raw.get("val_accuracy"),
        })
    return out


class ModelRegistry:
    """Model folders under models_dir, loaded on first use and kept in memory.

    Each API key can name a model folder; keys without one use the default. Folders that
    resolve to the same place (models/current -> models/words-v1) share one loaded model.
    Changing a symlink takes effect after a restart.
    """

    def __init__(self, models_dir: Path, default: str, threads: int = 0, loaded: dict | None = None):
        self.models_dir = Path(models_dir)
        self.default_name = default
        self.threads = threads
        self._by_name: dict = dict(loaded or {})
        self._by_path: dict = {}
        self._lock = threading.Lock()

    def get(self, name: str | None = None):
        name = name or self.default_name
        solver = self._by_name.get(name)
        if solver is not None:
            return solver
        with self._lock:
            if name in self._by_name:
                return self._by_name[name]
            if not model_exists(self.models_dir, name):
                raise ModelNotFound(name)
            folder = self.models_dir / name
            real = folder.resolve()
            solver = self._by_path.get(real) or Solver(folder, self.threads)
            self._by_path[real] = solver
            self._by_name[name] = solver
            return solver

    @property
    def default(self):
        return self.get(None)
