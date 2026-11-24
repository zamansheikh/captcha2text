"""Math captchas: the evaluator, and type=math through the API."""

from __future__ import annotations

import base64

import pytest
from conftest import FakeSolver, image_bytes
from fastapi.testclient import TestClient

from captcha2text.api import create_app
from captcha2text.mathexpr import solve_expression
from captcha2text.model import ModelRegistry


@pytest.mark.parametrize("text, value", [
    ("4+3-5", 2), ("4 + 3 - 5", 2), ("9-12", -3), ("2×7", 14), ("2x3+1", 7), ("1+2*3", 7),
    ("10-2×3+4", 8), ("7", 7), ("0+0", 0), ("12X2", 24),
])
def test_sums(text, value):
    assert solve_expression(text) == value


@pytest.mark.parametrize("text", ["", "4+35-", "+4", "4++3", "4+-3", "4/2", "4+a", "1234567+1", "4.5+1", "(1+2)"])
def test_not_sums(text):
    assert solve_expression(text) is None


@pytest.fixture
def math_client(settings, store):
    registry = ModelRegistry(settings.models_dir, "current", loaded={
        "current": FakeSolver(text="kerry", name="ddddocr-beta-v1"),
        "math": FakeSolver(text="4+3-5", name="math-v1"),
    })
    with TestClient(create_app(settings, keys=store, models=registry)) as c:
        c.registry = registry
        yield c


def _files():
    return {"image": ("c.png", image_bytes(), "image/png")}


def test_math_as_form_field(math_client, auth):
    r = math_client.post("/v1/solve", headers=auth, files=_files(), data={"type": "math"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["text"] == "4+3-5" and body["answer"] == "2"


def test_math_as_json_and_query(math_client, auth):
    b64 = base64.b64encode(image_bytes()).decode()
    r = math_client.post("/v1/solve", headers=auth, json={"image": b64, "type": "MATH"})
    assert r.json()["answer"] == "2"
    r = math_client.post("/v1/solve?type=math", headers=auth, files=_files())
    assert r.json()["answer"] == "2"


def test_math_plain_text_is_the_answer(math_client, auth):
    r = math_client.post("/v1/solve", headers={**auth, "Accept": "text/plain"}, files=_files(), data={"type": "math"})
    assert r.text == "2"


def test_text_captchas_unchanged(math_client, auth):
    body = math_client.post("/v1/solve", headers=auth, files=_files()).json()
    assert body["text"] == "kerry" and "answer" not in body


def test_bad_type(math_client, auth):
    r = math_client.post("/v1/solve", headers=auth, files=_files(), data={"type": "algebra"})
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_type"


def test_unreadable_sum(math_client, auth):
    math_client.registry.get("math").text = "4+35-"
    r = math_client.post("/v1/solve", headers=auth, files=_files(), data={"type": "math"})
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_expression"
    assert "4+35-" in r.json()["error"]["message"]


def test_math_model_missing(settings, store, auth):
    registry = ModelRegistry(settings.models_dir, "current", loaded={"current": FakeSolver()})
    with TestClient(create_app(settings, keys=store, models=registry)) as c:
        r = c.post("/v1/solve", headers=auth, files=_files(), data={"type": "math"})
    assert r.status_code == 503 and "Math captchas" in r.json()["error"]["message"]


def test_usage_records_math_model(math_client, auth, store):
    math_client.post("/v1/solve", headers=auth, files=_files(), data={"type": "math"})
    [model] = [row[0] for row in store._conn().execute("SELECT model FROM usage")]
    assert model == "math-v1"
