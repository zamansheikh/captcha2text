"""Landing page, docs, request form and the admin panel API."""

from __future__ import annotations

import json

import pytest
from conftest import image_bytes

from captcha2text.admin import AdminStore, hash_password, verify_password
from captcha2text.keys import main as keys_cli

PASSWORD = "correct horse battery staple"


@pytest.fixture
def admins(settings):
    store = AdminStore(settings.db_path)
    store.create_admin("admin", PASSWORD)
    return store


@pytest.fixture
def signed_in(client, admins):
    r = client.post("/admin/api/login", json={"username": "admin", "password": PASSWORD})
    assert r.status_code == 200, r.text
    return client


def _model_folder(root, name):
    (root / name).mkdir(parents=True)
    (root / name / "model.json").write_text(json.dumps({"name": name, "decoder": "ctc", "charset": "ab"}))


# ---- pages -------------------------------------------------------------------------------


def test_pages_have_security_headers(client):
    for path in ("/", "/docs", "/admin"):
        r = client.get(path)
        assert r.status_code == 200, path
        assert r.headers["content-type"].startswith("text/html")
        csp = r.headers["content-security-policy"]
        assert "script-src 'self' 'sha256-" in csp and "frame-ancestors 'none'" in csp
        assert r.text.lstrip().lower().startswith("<!doctype html>")
    assert client.get("/admin").headers["cache-control"] == "no-store"
    assert "noindex" in client.get("/admin").headers["x-robots-tag"]
    assert "Disallow: /admin" in client.get("/robots.txt").text
    assert client.get("/favicon.svg").headers["content-type"].startswith("image/svg+xml")


def test_inline_script_hashes_match(client):
    import base64
    import hashlib
    import re

    r = client.get("/")
    script = re.search(r"<script>(.*?)</script>", r.text, re.S).group(1)
    digest = base64.b64encode(hashlib.sha256(script.encode()).digest()).decode()
    assert f"'sha256-{digest}'" in r.headers["content-security-policy"]


# ---- request access ----------------------------------------------------------------------


def test_request_access_is_stored(client, settings):
    r = client.post("/request-access", json={"name": "Rahim", "email": "rahim@example.com", "company": "Acme",
                                             "captcha": "5 warped letters", "volume": "Under 10,000"})
    assert r.status_code == 200, r.text
    [req] = AdminStore(settings.db_path).list_requests()
    assert req["email"] == "rahim@example.com" and req["status"] == "new"


def test_request_access_validation(client, settings):
    assert client.post("/request-access", json={"name": "X"}).status_code == 400
    assert client.post("/request-access", json={"name": "X", "email": "nope"}).status_code == 400
    assert client.post("/request-access", json={"name": "X" * 101, "email": "a@b.co"}).status_code == 400
    assert client.post("/request-access", content=b"name=x", headers={"Content-Type": "application/x-www-form-urlencoded"}).status_code == 415
    r = client.post("/request-access", json={"name": "X", "email": "a@b.co"}, headers={"Origin": "https://evil.example"})
    assert r.status_code == 403
    assert AdminStore(settings.db_path).list_requests() == []


def test_request_access_honeypot_and_rate_limit(client, settings):
    assert client.post("/request-access", json={"name": "Bot", "email": "b@b.co", "nickname": "x"}).status_code == 200
    assert AdminStore(settings.db_path).list_requests() == []
    codes = [client.post("/request-access", json={"name": f"P{i}", "email": "p@b.co"}).status_code for i in range(7)]
    assert codes == [200] * 5 + [429, 429]


# ---- sign in -----------------------------------------------------------------------------


def test_admin_api_needs_sign_in(client, admins):
    for method, path in [("GET", "/admin/api/keys"), ("GET", "/admin/api/overview"), ("GET", "/admin/api/me")]:
        r = client.request(method, path)
        assert r.status_code == 401 and r.json()["error"]["code"] == "not_signed_in"


def test_wrong_password_then_lockout(client, admins):
    for _ in range(4):
        r = client.post("/admin/api/login", json={"username": "admin", "password": "wrong"})
        assert r.status_code == 401
    r = client.post("/admin/api/login", json={"username": "admin", "password": "wrong"})
    assert r.status_code == 423
    r = client.post("/admin/api/login", json={"username": "admin", "password": PASSWORD})
    assert r.status_code == 423  # still locked, even with the right password
    assert client.post("/admin/api/login", json={"username": "ghost", "password": "x"}).status_code == 401


def test_login_cookie_and_logout(client, admins):
    r = client.post("/admin/api/login", json={"username": "Admin", "password": PASSWORD})
    cookie = r.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=strict" in cookie
    assert client.get("/admin/api/me").json() == {"username": "admin"}
    assert client.post("/admin/api/logout", json={}).status_code == 200
    assert client.get("/admin/api/me").status_code == 401


def test_cross_site_changes_are_refused(signed_in):
    r = signed_in.post("/admin/api/keys", json={"name": "X"}, headers={"Origin": "https://evil.example"})
    assert r.status_code == 403
    # Two Origin headers, ours first: still refused.
    r = signed_in.post(
        "/admin/api/keys", content=b'{"name": "X"}',
        headers=[("Content-Type", "application/json"), ("Origin", "http://testserver"), ("Origin", "https://evil.example")],
    )
    assert r.status_code == 403
    assert signed_in.get("/admin/api/keys").json()["keys"] == []
    r = signed_in.post("/admin/api/keys", content=b"name=X", headers={"Content-Type": "application/x-www-form-urlencoded"})
    assert r.status_code == 415


# ---- keys --------------------------------------------------------------------------------


def test_create_use_edit_revoke_key(signed_in, settings):
    _model_folder(settings.models_dir, "acme-v1")
    r = signed_in.post("/admin/api/keys", json={"name": "Acme", "rpm": 30})
    assert r.status_code == 201, r.text
    plain, key = r.json()["api_key"], r.json()["key"]
    assert plain.startswith("c2t_live_") and key["model"] is None

    files = {"image": ("c.png", image_bytes(), "image/png")}
    solved = signed_in.post("/v1/solve", headers={"Authorization": f"Bearer {plain}"}, files=files)
    assert solved.status_code == 200

    [listed] = signed_in.get("/admin/api/keys").json()["keys"]
    assert listed["solved_30d"] == 1 and listed["last_used"] and "api_key" not in listed

    r = signed_in.patch(f"/admin/api/keys/{key['id']}", json={"rpm": 120, "model": "acme-v1"})
    assert r.json()["key"]["rpm"] == 120 and r.json()["key"]["model"] == "acme-v1"
    assert signed_in.patch(f"/admin/api/keys/{key['id']}", json={"model": "../etc"}).status_code == 400
    assert signed_in.patch(f"/admin/api/keys/{key['id']}", json={"rpm": -1}).status_code == 400
    assert signed_in.patch(f"/admin/api/keys/{key['id']}", json={"model": "default"}).json()["key"]["model"] is None

    assert signed_in.post(f"/admin/api/keys/{key['id']}/revoke", json={}).json()["key"]["active"] is False
    denied = signed_in.post("/v1/solve", headers={"Authorization": f"Bearer {plain}"}, files=files)
    assert denied.status_code == 401
    assert signed_in.post("/admin/api/keys/999/revoke", json={}).status_code == 404


def test_key_from_request_marks_it_done(signed_in, settings, admins):
    signed_in.post("/request-access", json={"name": "Karim", "email": "k@example.com"})
    [req] = signed_in.get("/admin/api/requests").json()["requests"]
    r = signed_in.post("/admin/api/keys", json={"name": "Karim", "request_id": req["id"]})
    [req] = signed_in.get("/admin/api/requests").json()["requests"]
    assert req["status"] == "done" and req["key_id"] == r.json()["key"]["id"]
    assert signed_in.post(f"/admin/api/requests/{req['id']}", json={"status": "new"}).status_code == 200
    assert signed_in.post(f"/admin/api/requests/{req['id']}", json={"status": "bogus"}).status_code == 400


def test_overview_and_models(signed_in, settings):
    _model_folder(settings.models_dir, "acme-v1")
    signed_in.post("/admin/api/keys", json={"name": "A", "model": "acme-v1"})
    ov = signed_in.get("/admin/api/overview").json()
    assert ov["active_keys"] == 1 and len(ov["daily"]) == 14 and ov["default_model"] == "fake-v1"
    models = signed_in.get("/admin/api/models").json()
    assert [m["folder"] for m in models["models"]] == ["acme-v1"] and models["models"][0]["keys"] == 1


# ---- passwords and CLI -------------------------------------------------------------------


def test_password_hashing():
    h = hash_password("s3cret-password")
    assert h.startswith("scrypt$") and verify_password("s3cret-password", h)
    assert not verify_password("other", h) and not verify_password("x", "garbage")


def test_admin_cli(tmp_path, capsys):
    db = str(tmp_path / "k.db")
    assert keys_cli(["--db", db, "admin-create", "zaman"]) == 0
    out = capsys.readouterr().out
    password = out.strip().splitlines()[-1].strip()
    assert AdminStore(tmp_path / "k.db").check_login("zaman", password) == (1, None)
    assert keys_cli(["--db", db, "admin-create", "zaman"]) == 1  # already exists
    assert keys_cli(["--db", db, "admin-create", "Bad Name!"]) == 1
    assert keys_cli(["--db", db, "admin-password", "zaman"]) == 0
    assert AdminStore(tmp_path / "k.db").check_login("zaman", password)[0] is None
    assert keys_cli(["--db", db, "admin-list"]) == 0
