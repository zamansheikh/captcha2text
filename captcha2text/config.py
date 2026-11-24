"""Server settings, read from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _int(name: str, default: int) -> int:
    value = os.environ.get(name, "").strip()
    return int(value) if value else default


@dataclass(frozen=True)
class Settings:
    # Folder of model folders; each holds model.onnx and model.json.
    models_dir: Path = Path("models")
    # Model used by API keys that have no model of their own. Usually a symlink.
    default_model: str = "current"
    # Model used for requests with type=math. Usually a symlink.
    math_model: str = "math"
    # SQLite file with API keys and the usage log.
    db_path: Path = Path("data/captcha2text.db")
    # Largest image file accepted, in bytes (the docs promise 2 MB).
    max_image_bytes: int = 2 * 1024 * 1024
    # Largest decoded image accepted, in pixels. Stops tiny files that decode to huge images.
    max_image_pixels: int = 4096 * 4096
    # Requests per minute for new keys when --rpm is not given.
    default_rpm: int = 60
    # ONNX Runtime threads per inference. 0 lets ONNX Runtime pick.
    ort_threads: int = 0

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            models_dir=Path(os.environ.get("C2T_MODELS_DIR", "models")),
            default_model=os.environ.get("C2T_DEFAULT_MODEL", "current"),
            math_model=os.environ.get("C2T_MATH_MODEL", "math"),
            db_path=Path(os.environ.get("C2T_DB_PATH", "data/captcha2text.db")),
            max_image_bytes=_int("C2T_MAX_IMAGE_BYTES", 2 * 1024 * 1024),
            max_image_pixels=_int("C2T_MAX_IMAGE_PIXELS", 4096 * 4096),
            default_rpm=_int("C2T_DEFAULT_RPM", 60),
            ort_threads=_int("C2T_ORT_THREADS", 0),
        )
