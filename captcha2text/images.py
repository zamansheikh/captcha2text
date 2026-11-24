"""Image decoding and preprocessing.

The server and the training scripts both call `preprocess`, so a model always sees
images prepared the same way it was trained on.

`resize_bilinear` matches TensorFlow 2's `tf.image.resize(..., method="bilinear")`
(half-pixel centers, no antialiasing), which is what captcha_reader.ipynb trained with.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, UnidentifiedImageError

ALLOWED_FORMATS = {"PNG", "JPEG"}
LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)


class ImageError(Exception):
    """An image the API refuses. `code` is one of the documented error codes."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def decode_image(data: bytes, max_pixels: int = 4096 * 4096) -> np.ndarray:
    """Decode PNG or JPEG bytes to an RGB uint8 array of shape (height, width, 3)."""
    if not data:
        raise ImageError("invalid_image", "The image file is empty.")
    try:
        img = Image.open(io.BytesIO(data))
    except Image.DecompressionBombError:
        raise ImageError("image_too_large", "The image has too many pixels.") from None
    except (UnidentifiedImageError, OSError, ValueError):
        raise ImageError("invalid_image", "The file is not a PNG or JPEG image.") from None

    if img.format not in ALLOWED_FORMATS:
        raise ImageError("invalid_image", "The file is not a PNG or JPEG image.")

    width, height = img.size
    if width * height > max_pixels:
        raise ImageError(
            "image_too_large",
            f"The image is {width}x{height} pixels. The limit is {max_pixels:,} pixels.",
        )

    try:
        img.load()
    except (OSError, ValueError, Image.DecompressionBombError):
        raise ImageError("invalid_image", "The image file is damaged or cut short.") from None

    return to_rgb(img)


def to_rgb(img: Image.Image) -> np.ndarray:
    """Convert any PIL image to RGB. Transparent areas become white."""
    if img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        img = Image.alpha_composite(white, rgba)
    return np.asarray(img.convert("RGB"), dtype=np.uint8)


def _interpolation(out_size: int, in_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scale = in_size / out_size
    src = (np.arange(out_size, dtype=np.float64) + 0.5) * scale - 0.5
    src_floor = np.floor(src)
    lower = np.maximum(src_floor, 0).astype(np.int64)
    upper = np.minimum(np.ceil(src), in_size - 1).astype(np.int64)
    lerp = (src - src_floor).astype(np.float32)
    return lower, upper, lerp


def resize_bilinear(img: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize an (H, W, C) array to (height, width, C) float32, like tf.image.resize."""
    img = img.astype(np.float32, copy=False)
    in_h, in_w = img.shape[:2]
    y0, y1, wy = _interpolation(height, in_h)
    x0, x1, wx = _interpolation(width, in_w)

    top, bottom = img[y0], img[y1]
    wx = wx[None, :, None]
    top = top[:, x0] + (top[:, x1] - top[:, x0]) * wx
    bottom = bottom[:, x0] + (bottom[:, x1] - bottom[:, x0]) * wx
    return top + (bottom - top) * wy[:, None, None]


def auto_width(rgb: np.ndarray, height: int, max_width: int = 2048) -> int:
    """Width that keeps the aspect ratio at the given height."""
    h, w = rgb.shape[:2]
    return min(max_width, max(8, int(w * (height / h))))


def preprocess(
    rgb: np.ndarray,
    width: int | None,
    height: int,
    channels: int = 3,
    layout: str = "NHWC",
    resample: str = "bilinear",
    mean: float = 0.0,
    std: float = 1.0,
    max_width: int = 2048,
) -> np.ndarray:
    """RGB uint8 (H, W, 3) -> float32 batch of one.

    width=None keeps the aspect ratio. resample="bilinear" matches tf.image.resize;
    "lanczos" uses PIL (what ddddocr uses), with PIL's own grayscale conversion.
    Pixels are scaled to 0..1, then (x - mean) / std.
    """
    if channels not in (1, 3):
        raise ValueError(f"channels must be 1 or 3, got {channels}")
    if width is None:
        width = auto_width(rgb, height, max_width)

    if resample == "lanczos":
        img = Image.fromarray(rgb).resize((width, height), Image.LANCZOS)
        if channels == 1:
            img = img.convert("L")
        x = np.asarray(img, dtype=np.float32) / 255.0
        if channels == 1:
            x = x[..., None]
    elif resample == "bilinear":
        x = resize_bilinear(rgb, height, width) / 255.0
        if channels == 1:
            x = (x @ LUMA)[..., None]
    else:
        raise ValueError(f"resample must be bilinear or lanczos, got {resample}")

    if mean != 0.0 or std != 1.0:
        x = (x - mean) / std
    if layout == "NCHW":
        x = x.transpose(2, 0, 1)
    elif layout != "NHWC":
        raise ValueError(f"layout must be NHWC or NCHW, got {layout}")
    return np.ascontiguousarray(x[None], dtype=np.float32)
