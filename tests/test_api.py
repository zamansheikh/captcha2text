"""Every response shape and error code promised in docs/API.md."""

from __future__ import annotations

import base64
import re

from conftest import FakeSolver, image_bytes

from captcha2text.api import create_app
from fastapi.testclient import TestClient

REQUEST_ID = re.compile(r"^req_[0-9A-Z]{12}$")


def assert_error(resp, status: int, code: str):
    assert resp.status_code == status, resp.text
    body = resp.json()
    assert body["success"] is False
    assert body["error"]["code"] == code
    assert body["error"]["message"]
    assert REQUEST_ID.match(body["request_id"])
    assert resp.headers["X-Request-Id"] == body["request_id"]
    return body


# ---- success ---------------------------------------------------------------------------


def test_health_needs_no_key(client):
    resp = client.get("/v1/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "model": "fake-v1"}


def test_solve_file_upload(client, auth, solver):
    resp = client.post("/v1/solve", headers=auth, files={"image": ("c.png", image_bytes(), "image/png")})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["success"] is True
    assert body["text"] == "kerry"
    assert body["confidence"] == 0.98
    assert isinstance(body["processing_ms"], int)
    assert REQUEST_ID.match(body["request_id"])
    assert solver.seen == [(80, 200, 3)]


def test_solve_jpeg(client, auth):
    resp = client.post("/v1/solve", headers=auth, files={"image": ("c.jpg", image_bytes("JPEG"), "image/jpeg")})
    assert resp.status_code == 200, resp.text


def test_solve_transparent_png(client, auth):
    data = image_bytes(mode="RGBA")
    resp = client.post("/v1/solve", headers=auth, files={"image": ("c.png", data, "image/png")})
    assert resp.status_code == 200, resp.text


def test_solve_base64_json(client, auth):
    b64 = base64.b64encode(image_bytes()).decode()
    resp = client.post("/v1/solve", headers=auth, json={"image": b64})
    assert resp.status_code == 200, resp.text
    assert resp.json()["text"] == "kerry"


def test_solve_base64_data_url_with_newlines(client, auth):
    b64 = base64.encodebytes(image_bytes()).decode()  # wrapped at 76 chars, like `base64` CLI
    resp = client.post("/v1/solve", headers=auth, json={"image": "data:image/png;base64," + b64})
    assert resp.status_code == 200, resp.text


def test_plain_text_response(client, auth):
    resp = client.post(
        "/v1/solve",
        headers={**auth, "Accept": "text/plain"},
        files={"image": ("c.png", image_bytes(), "image/png")},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/plain")
    assert resp.text == "kerry"


def test_errors_stay_json_with_text_accept(client, auth):
    resp = client.post("/v1/solve", headers={**auth, "Accept": "text/plain"}, json={})
    assert_error(resp, 400, "missing_image")


# ---- auth ------------------------------------------------------------------------------


def test_missing_key(client):
    resp = client.post("/v1/solve", files={"image": ("c.png", image_bytes(), "image/png")})
    assert_error(resp, 401, "missing_api_key")
    assert resp.headers["WWW-Authenticate"].startswith("Bearer")


def test_wrong_key(client):
    resp = client.post(
        "/v1/solve",
        headers={"Authorization": "Bearer c2t_live_nope"},
        files={"image": ("c.png", image_bytes(), "image/png")},
    )
    assert_error(resp, 401, "invalid_api_key")


def test_wrong_scheme(client, api_key):
    resp = client.post(
        "/v1/solve", headers={"Authorization": api_key}, files={"image": ("c.png", image_bytes(), "image/png")}
    )
    assert_error(resp, 401, "invalid_api_key")


def test_revoked_key(client, store):
    key, plain = store.create("Gone", rpm=10)
    store.revoke(key.id)
    resp = client.post(
        "/v1/solve",
        headers={"Authorization": f"Bearer {plain}"},
        files={"image": ("c.png", image_bytes(), "image/png")},
    )
    assert_error(resp, 401, "invalid_api_key")


# ---- bad input -------------------------------------------------------------------------


def test_missing_image_field(client, auth):
    resp = client.post("/v1/solve", headers=auth, files={"picture": ("c.png", image_bytes(), "image/png")})
    assert_error(resp, 400, "missing_image")


def test_missing_image_json(client, auth):
    assert_error(client.post("/v1/solve", headers=auth, json={"img": "x"}), 400, "missing_image")


def test_invalid_json(client, auth):
    resp = client.post("/v1/solve", headers={**auth, "Content-Type": "application/json"}, content=b"{nope")
    assert_error(resp, 400, "invalid_body")


def test_not_an_image(client, auth):
    resp = client.post("/v1/solve", headers=auth, files={"image": ("c.png", b"hello", "image/png")})
    assert_error(resp, 400, "invalid_image")


def test_gif_refused(client, auth):
    resp = client.post("/v1/solve", headers=auth, files={"image": ("c.gif", image_bytes("GIF"), "image/gif")})
    assert_error(resp, 400, "invalid_image")


def test_bad_base64(client, auth):
    assert_error(client.post("/v1/solve", headers=auth, json={"image": "***"}), 400, "invalid_image")


def test_image_field_as_text(client, auth):
    resp = client.post("/v1/solve", headers=auth, data={"image": "abc"}, files={"x": ("x.txt", b"x")})
    assert_error(resp, 400, "invalid_image")


def test_empty_file(client, auth):
    resp = client.post("/v1/solve", headers=auth, files={"image": ("c.png", b"", "image/png")})
    assert_error(resp, 400, "invalid_image")


def test_too_large_file(client, auth, settings):
    big = b"\0" * (settings.max_image_bytes + 1)
    resp = client.post("/v1/solve", headers=auth, files={"image": ("c.png", big, "image/png")})
    assert_error(resp, 413, "image_too_large")


def test_too_large_body(client, auth, settings):
    resp = client.post(
        "/v1/solve",
        headers={**auth, "Content-Type": "application/json"},
        content=b"x" * (settings.max_image_bytes * 2),
    )
    assert_error(resp, 413, "image_too_large")


def test_too_many_pixels(settings, solver, store, api_key):
    from dataclasses import replace

    app = create_app(replace(settings, max_image_pixels=1000), solver=solver, keys=store)
    with TestClient(app) as c:
        resp = c.post(
            "/v1/solve",
            headers={"Authorization": f"Bearer {api_key}"},
            files={"image": ("c.png", image_bytes(size=(100, 100)), "image/png")},
        )
    assert_error(resp, 413, "image_too_large")


def test_wrong_content_type(client, auth):
    resp = client.post("/v1/solve", headers={**auth, "Content-Type": "image/png"}, content=image_bytes())
    assert_error(resp, 415, "unsupported_media_type")


# ---- model results ---------------------------------------------------------------------


def test_no_text_found(client, auth, solver):
    solver.text = ""
    resp = client.post("/v1/solve", headers=auth, files={"image": ("c.png", image_bytes(), "image/png")})
    assert_error(resp, 422, "no_text_found")


def test_model_crash_is_500(client, auth, solver):
    solver.error = RuntimeError("boom")
    resp = client.post("/v1/solve", headers=auth, files={"image": ("c.png", image_bytes(), "image/png")})
    body = assert_error(resp, 500, "internal_error")
    assert "boom" not in body["error"]["message"]


# ---- limits and logging ----------------------------------------------------------------


def test_rate_limit(client, store):
    _, plain = store.create("Slow", rpm=2)
    headers = {"Authorization": f"Bearer {plain}"}
    files = {"image": ("c.png", image_bytes(), "image/png")}
    assert client.post("/v1/solve", headers=headers, files=files).status_code == 200
    assert client.post("/v1/solve", headers=headers, files=files).status_code == 200
    resp = client.post("/v1/solve", headers=headers, files=files)
    assert_error(resp, 429, "rate_limited")
    assert int(resp.headers["Retry-After"]) >= 1


def test_usage_is_logged(client, store, auth):
    client.post("/v1/solve", headers=auth, files={"image": ("c.png", image_bytes(), "image/png")})
    client.post("/v1/solve", headers=auth, json={})
    rows = {r["name"]: r for r in store.usage_summary(days=1)}
    assert rows["Test customer"]["total"] == 2
    assert rows["Test customer"]["solved"] == 1
    assert rows["Test customer"]["failed"] == 1


def test_each_key_uses_its_own_model(settings, store):
    from captcha2text.model import ModelRegistry

    registry = ModelRegistry(
        settings.models_dir, "current",
        loaded={"current": FakeSolver(text="general", name="ddddocr-v1"), "acme-v1": FakeSolver(text="acme", name="acme-v1")},
    )
    _, default_key = store.create("Anyone", rpm=100)
    _, acme_key = store.create("Acme", rpm=100, model="acme-v1")
    _, broken_key = store.create("Broken", rpm=100, model="gone-v1")
    files = {"image": ("c.png", image_bytes(), "image/png")}

    with TestClient(create_app(settings, keys=store, models=registry)) as c:
        assert c.get("/v1/health").json()["model"] == "ddddocr-v1"
        r = c.post("/v1/solve", headers={"Authorization": f"Bearer {default_key}"}, files=files)
        assert r.json()["text"] == "general"
        r = c.post("/v1/solve", headers={"Authorization": f"Bearer {acme_key}"}, files=files)
        assert r.json()["text"] == "acme"
        r = c.post("/v1/solve", headers={"Authorization": f"Bearer {broken_key}"}, files=files)
        assert_error(r, 503, "service_unavailable")

    models_used = [row[0] for row in store._conn().execute("SELECT model FROM usage ORDER BY id")]
    assert models_used == ["ddddocr-v1", "acme-v1", None]


def test_unknown_path_uses_error_shape(client):
    assert_error(client.get("/v1/nope"), 404, "not_found")
    assert_error(client.get("/v1/solve"), 405, "method_not_allowed")
