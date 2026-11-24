from __future__ import annotations

import io
from dataclasses import dataclass, field

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from captcha2text.api import create_app
from captcha2text.config import Settings
from captcha2text.keys import KeyStore
from captcha2text.model import Solution


@dataclass
class FakeSolver:
    """Stands in for the ONNX model so the HTTP layer can be tested alone."""

    text: str = "kerry"
    confidence: float = 0.98
    error: Exception | None = None
    seen: list = field(default_factory=list)
    name: str = "fake-v1"

    def solve(self, rgb):
        self.seen.append(rgb.shape)
        if self.error:
            raise self.error
        return Solution(text=self.text, confidence=self.confidence, processing_ms=3)


def image_bytes(fmt: str = "PNG", size=(200, 80), mode="RGB") -> bytes:
    buf = io.BytesIO()
    Image.new(mode, size, "white" if mode != "RGBA" else (0, 0, 0, 0)).save(buf, format=fmt)
    return buf.getvalue()


@pytest.fixture
def settings(tmp_path):
    return Settings(models_dir=tmp_path / "models", db_path=tmp_path / "keys.db", max_image_bytes=200_000)


@pytest.fixture
def store(settings):
    return KeyStore(settings.db_path)


@pytest.fixture
def api_key(store):
    _, plain = store.create("Test customer", rpm=1000)
    return plain


@pytest.fixture
def solver():
    return FakeSolver()


@pytest.fixture
def client(settings, solver, store):
    app = create_app(settings, solver=solver, keys=store)
    with TestClient(app) as c:
        yield c


@pytest.fixture
def auth(api_key):
    return {"Authorization": f"Bearer {api_key}"}
