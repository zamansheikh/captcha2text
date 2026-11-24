"""Admin panel backend: admin logins, sessions, access requests and the /admin/api routes.

Admins are created on the server with `c2t-keys admin-create <username>`. Passwords are
stored as scrypt hashes. A login sets an HttpOnly, SameSite=Strict session cookie; the
session lives in SQLite and lasts 12 hours. Five wrong passwords lock an account for
15 minutes, and each IP gets 20 login attempts per 15 minutes.

The landing page's "Request an API key" form posts to /request-access; requests show up in
the panel, where an admin turns one into a key.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import re
import secrets
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response
from starlette.concurrency import run_in_threadpool

from .errors import APIError
from .model import list_models, model_exists
from .ratelimit import WindowLimiter

SESSION_COOKIE = "c2t_admin"
SESSION_HOURS = 12
LOCK_AFTER = 5
LOCK_MINUTES = 15

SCHEMA = """
CREATE TABLE IF NOT EXISTS admins (
    id             INTEGER PRIMARY KEY,
    username       TEXT    NOT NULL UNIQUE,
    password_hash  TEXT    NOT NULL,
    created_at     TEXT    NOT NULL,
    last_login_at  TEXT,
    failed_logins  INTEGER NOT NULL DEFAULT 0,
    locked_until   TEXT
);
CREATE TABLE IF NOT EXISTS admin_sessions (
    token_hash  TEXT    PRIMARY KEY,
    admin_id    INTEGER NOT NULL REFERENCES admins(id),
    created_at  TEXT    NOT NULL,
    expires_at  TEXT    NOT NULL
);
CREATE TABLE IF NOT EXISTS access_requests (
    id          INTEGER PRIMARY KEY,
    created_at  TEXT    NOT NULL,
    name        TEXT    NOT NULL,
    email       TEXT    NOT NULL,
    company     TEXT,
    website     TEXT,
    captcha     TEXT,
    volume      TEXT,
    status      TEXT    NOT NULL DEFAULT 'new',
    key_id      INTEGER,
    handled_at  TEXT
);
"""

_USERNAME = re.compile(r"^[a-z0-9][a-z0-9._-]{1,31}$")
_EMAIL = re.compile(r"^[^@\s]{1,64}@[^@\s]+\.[^@\s]{2,}$")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


# ---- passwords ---------------------------------------------------------------------------

_N, _R, _P = 2**14, 8, 1


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=_N, r=_R, p=_P, dklen=32)
    return "scrypt${}${}${}${}${}".format(
        _N, _R, _P, base64.b64encode(salt).decode(), base64.b64encode(digest).decode()
    )


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, n, r, p, salt, digest = stored.split("$")
        if scheme != "scrypt":
            return False
        got = hashlib.scrypt(
            password.encode(), salt=base64.b64decode(salt), n=int(n), r=int(r), p=int(p), dklen=32
        )
        return hmac.compare_digest(got, base64.b64decode(digest))
    except (ValueError, TypeError):
        return False


# A real hash to check against when the username doesn't exist, so timing doesn't reveal it.
_DUMMY_HASH = hash_password(secrets.token_urlsafe(16))


def generate_password() -> str:
    return secrets.token_urlsafe(18)


# ---- storage -----------------------------------------------------------------------------

class AdminStore:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        with self._conn() as conn:
            conn.executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.db_path, timeout=10)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=10000")
            self._local.conn = conn
        return conn

    # admins
    def create_admin(self, username: str, password: str) -> int:
        if not _USERNAME.match(username):
            raise ValueError("Usernames are 2-32 characters: a-z, 0-9, dot, dash, underscore.")
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO admins (username, password_hash, created_at) VALUES (?, ?, ?)",
                (username, hash_password(password), _iso(_now())),
            )
        return cur.lastrowid

    def set_password(self, username: str, password: str) -> bool:
        with self._conn() as conn:
            cur = conn.execute(
                "UPDATE admins SET password_hash = ?, failed_logins = 0, locked_until = NULL WHERE username = ?",
                (hash_password(password), username),
            )
            if cur.rowcount:
                conn.execute(
                    "DELETE FROM admin_sessions WHERE admin_id = (SELECT id FROM admins WHERE username = ?)",
                    (username,),
                )
        return cur.rowcount == 1

    def list_admins(self) -> list[sqlite3.Row]:
        return self._conn().execute("SELECT id, username, created_at, last_login_at FROM admins ORDER BY id").fetchall()

    def check_login(self, username: str, password: str) -> tuple[int | None, str | None]:
        """Returns (admin id, None) on success, or (None, reason) where reason is 'locked' or 'wrong'."""
        row = self._conn().execute("SELECT * FROM admins WHERE username = ?", (username,)).fetchone()
        ok = verify_password(password, row["password_hash"] if row else _DUMMY_HASH)
        if row is None:
            return None, "wrong"
        now = _now()
        if row["locked_until"] and row["locked_until"] > _iso(now):
            return None, "locked"
        with self._conn() as conn:
            if ok:
                conn.execute(
                    "UPDATE admins SET failed_logins = 0, locked_until = NULL, last_login_at = ? WHERE id = ?",
                    (_iso(now), row["id"]),
                )
                return row["id"], None
            failed = row["failed_logins"] + 1
            locked = _iso(now + timedelta(minutes=LOCK_MINUTES)) if failed >= LOCK_AFTER else None
            conn.execute(
                "UPDATE admins SET failed_logins = ?, locked_until = ? WHERE id = ?",
                (0 if locked else failed, locked, row["id"]),
            )
        return None, "locked" if locked else "wrong"

    # sessions
    def create_session(self, admin_id: int) -> str:
        token = secrets.token_urlsafe(32)
        now = _now()
        with self._conn() as conn:
            conn.execute("DELETE FROM admin_sessions WHERE expires_at < ?", (_iso(now),))
            conn.execute(
                "INSERT INTO admin_sessions (token_hash, admin_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
                (hashlib.sha256(token.encode()).hexdigest(), admin_id, _iso(now),
                 _iso(now + timedelta(hours=SESSION_HOURS))),
            )
        return token

    def session_admin(self, token: str) -> sqlite3.Row | None:
        return self._conn().execute(
            """SELECT a.id, a.username FROM admin_sessions s JOIN admins a ON a.id = s.admin_id
               WHERE s.token_hash = ? AND s.expires_at > ?""",
            (hashlib.sha256(token.encode()).hexdigest(), _iso(_now())),
        ).fetchone()

    def delete_session(self, token: str) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM admin_sessions WHERE token_hash = ?", (hashlib.sha256(token.encode()).hexdigest(),))

    # access requests
    def add_request(self, fields: dict[str, str]) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO access_requests (created_at, name, email, company, website, captcha, volume)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (_iso(_now()), fields["name"], fields["email"], fields.get("company"), fields.get("website"),
                 fields.get("captcha"), fields.get("volume")),
            )
        return cur.lastrowid

    def count_requests(self, status: str = "new") -> int:
        return self._conn().execute("SELECT COUNT(*) FROM access_requests WHERE status = ?", (status,)).fetchone()[0]

    def list_requests(self) -> list[dict]:
        rows = self._conn().execute(
            "SELECT * FROM access_requests ORDER BY status = 'new' DESC, id DESC LIMIT 200"
        ).fetchall()
        return [dict(r) for r in rows]

    def set_request_status(self, request_id: int, status: str, key_id: int | None = None) -> bool:
        with self._conn() as conn:
            cur = conn.execute(
                "UPDATE access_requests SET status = ?, key_id = COALESCE(?, key_id), handled_at = ? WHERE id = ?",
                (status, key_id, _iso(_now()), request_id),
            )
        return cur.rowcount == 1


# ---- HTTP --------------------------------------------------------------------------------

def client_ip(request: Request) -> str:
    """The visitor's IP. Behind Cloudflare + nginx this is CF-Connecting-IP; it can be forged by
    someone who reaches the server directly, so it only feeds rate limits, never decisions."""
    return (
        request.headers.get("cf-connecting-ip")
        or request.headers.get("x-real-ip")
        or (request.client.host if request.client else "unknown")
    )


def _is_https(request: Request) -> bool:
    return request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https"


def _check_same_origin(request: Request) -> None:
    """Changes must come from our own pages: JSON body and, when sent, a matching Origin."""
    if request.method in ("POST", "PATCH", "DELETE"):
        if not request.headers.get("content-type", "").startswith("application/json"):
            raise APIError(415, "unsupported_media_type", "Send JSON.")
        host = request.headers.get("host", "")
        # Check every Origin header: a client could send two, and only one has to be ours.
        for origin in request.headers.getlist("origin"):
            if origin.split("://", 1)[-1] != host:
                raise APIError(403, "bad_origin", "Requests must come from this site.")


async def _json(request: Request) -> dict[str, Any]:
    body = await request.body()
    if len(body) > 16_384:
        raise APIError(413, "body_too_large", "The request is too large.")
    try:
        data = json.loads(body or b"{}")
    except ValueError:
        raise APIError(400, "invalid_body", "The request body is not valid JSON.") from None
    if not isinstance(data, dict):
        raise APIError(400, "invalid_body", "Send a JSON object.")
    return data


def _text(data: dict, field: str, max_len: int, required: bool = False) -> str | None:
    value = data.get(field)
    if value is None or (isinstance(value, str) and not value.strip()):
        if required:
            raise APIError(400, "invalid_field", f"{field} is required.")
        return None
    if not isinstance(value, str):
        raise APIError(400, "invalid_field", f"{field} must be text.")
    value = " ".join(value.split())
    if len(value) > max_len:
        raise APIError(400, "invalid_field", f"{field} can be at most {max_len} characters.")
    return value


def _rpm(value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 100_000:
        raise APIError(400, "invalid_field", "rpm must be a whole number from 0 (no limit) to 100000.")
    return value


def _key_json(key, usage: dict | None) -> dict:
    u = usage or {}
    return {
        "id": key.id, "name": key.name, "prefix": key.prefix, "rpm": key.rpm, "model": key.model,
        "created_at": key.created_at, "revoked_at": key.revoked_at, "active": key.active,
        "requests_30d": u.get("total", 0), "solved_30d": u.get("solved", 0), "failed_30d": u.get("failed", 0),
        "last_used": u.get("last_used"),
    }


def build_router(settings) -> APIRouter:
    router = APIRouter()
    login_limiter = WindowLimiter(limit=20, window=15 * 60)
    request_limiter = WindowLimiter(limit=5, window=60 * 60)

    def store(request: Request) -> AdminStore:
        return request.app.state.admins

    async def require_admin(request: Request):
        _check_same_origin(request)
        token = request.cookies.get(SESSION_COOKIE)
        admin = await run_in_threadpool(store(request).session_admin, token) if token else None
        if admin is None:
            raise APIError(401, "not_signed_in", "Sign in to continue.")
        return admin

    def no_store(data: Any, status: int = 200) -> JSONResponse:
        return JSONResponse(data, status_code=status, headers={"Cache-Control": "no-store"})

    # -- public: request access ------------------------------------------------------------

    @router.post("/request-access")
    async def request_access(request: Request) -> Response:
        _check_same_origin(request)
        data = await _json(request)
        if data.get("nickname"):  # hidden field; only bots fill it in
            return no_store({"success": True})
        if not request_limiter.hit(client_ip(request)):
            raise APIError(429, "rate_limited", "Too many requests from your network. Email us instead.")
        fields = {
            "name": _text(data, "name", 100, required=True),
            "email": _text(data, "email", 200, required=True),
            "company": _text(data, "company", 150),
            "website": _text(data, "website", 300),
            "captcha": _text(data, "captcha", 1000),
            "volume": _text(data, "volume", 100),
        }
        if not _EMAIL.match(fields["email"]):
            raise APIError(400, "invalid_field", "Enter a valid email address.")
        admins = store(request)
        if await run_in_threadpool(admins.count_requests, "new") >= 500:
            raise APIError(503, "service_unavailable", "We can't take requests right now. Email us instead.")
        await run_in_threadpool(admins.add_request, fields)
        return no_store({"success": True})

    # -- sign in / out ---------------------------------------------------------------------

    @router.post("/admin/api/login")
    async def login(request: Request) -> Response:
        _check_same_origin(request)
        if not login_limiter.hit(client_ip(request)):
            raise APIError(429, "rate_limited", "Too many sign-in attempts. Try again in 15 minutes.")
        data = await _json(request)
        username = (_text(data, "username", 64, required=True) or "").lower()
        password = data.get("password")
        if not isinstance(password, str) or not password:
            raise APIError(400, "invalid_field", "password is required.")
        admins = store(request)
        admin_id, reason = await run_in_threadpool(admins.check_login, username, password)
        if admin_id is None:
            if reason == "locked":
                raise APIError(423, "locked", f"Too many wrong passwords. Try again in {LOCK_MINUTES} minutes.")
            raise APIError(401, "wrong_login", "Wrong username or password.")
        token = await run_in_threadpool(admins.create_session, admin_id)
        resp = no_store({"success": True, "username": username})
        resp.set_cookie(
            SESSION_COOKIE, token, max_age=SESSION_HOURS * 3600, httponly=True, samesite="strict",
            secure=_is_https(request), path="/",
        )
        return resp

    @router.post("/admin/api/logout")
    async def logout(request: Request) -> Response:
        _check_same_origin(request)
        token = request.cookies.get(SESSION_COOKIE)
        if token:
            await run_in_threadpool(store(request).delete_session, token)
        resp = no_store({"success": True})
        resp.delete_cookie(SESSION_COOKIE, path="/")
        return resp

    @router.get("/admin/api/me")
    async def me(request: Request) -> Response:
        admin = await require_admin(request)
        return no_store({"username": admin["username"]})

    # -- overview ---------------------------------------------------------------------------

    @router.get("/admin/api/overview")
    async def overview(request: Request) -> Response:
        await require_admin(request)
        keys = request.app.state.keys

        def collect():
            all_keys = keys.list()
            daily = keys.usage_daily(14)
            usage = keys.usage_by_key(30)
            return {
                "active_keys": sum(k.active for k in all_keys),
                "revoked_keys": sum(not k.active for k in all_keys),
                "solved_30d": sum(u["solved"] for u in usage.values()),
                "failed_30d": sum(u["failed"] for u in usage.values()),
                "today": daily[-1],
                "daily": daily,
                "new_requests": store(request).count_requests("new"),
                "default_model": request.app.state.models.default.name,
            }

        return no_store(await run_in_threadpool(collect))

    # -- keys -------------------------------------------------------------------------------

    @router.get("/admin/api/keys")
    async def list_keys(request: Request) -> Response:
        await require_admin(request)
        keys = request.app.state.keys

        def collect():
            usage = keys.usage_by_key(30)
            return [_key_json(k, usage.get(k.id)) for k in reversed(keys.list())]

        return no_store({"keys": await run_in_threadpool(collect)})

    def _model_field(data: dict) -> str | None:
        model = data.get("model")
        if model in (None, "", "default"):
            return None
        if not isinstance(model, str) or not model_exists(settings.models_dir, model):
            raise APIError(400, "invalid_field", "That model folder doesn't exist on the server.")
        return model

    @router.post("/admin/api/keys")
    async def create_key(request: Request) -> Response:
        await require_admin(request)
        data = await _json(request)
        name = _text(data, "name", 100, required=True)
        rpm = _rpm(data.get("rpm", settings.default_rpm))
        model = _model_field(data)
        from_request = data.get("request_id")
        key, plain = await run_in_threadpool(request.app.state.keys.create, name, rpm, model)
        if isinstance(from_request, int):
            await run_in_threadpool(store(request).set_request_status, from_request, "done", key.id)
        return no_store({"key": _key_json(key, None), "api_key": plain}, status=201)

    @router.patch("/admin/api/keys/{key_id}")
    async def update_key(key_id: int, request: Request) -> Response:
        await require_admin(request)
        data = await _json(request)
        keys = request.app.state.keys
        if await run_in_threadpool(keys.get, key_id) is None:
            raise APIError(404, "not_found", "No key with that id.")
        if "rpm" in data:
            await run_in_threadpool(keys.set_rpm, key_id, _rpm(data["rpm"]))
        if "model" in data:
            await run_in_threadpool(keys.set_model, key_id, _model_field(data))
        key = await run_in_threadpool(keys.get, key_id)
        return no_store({"key": _key_json(key, None)})

    @router.post("/admin/api/keys/{key_id}/revoke")
    async def revoke_key(key_id: int, request: Request) -> Response:
        await require_admin(request)
        keys = request.app.state.keys
        if await run_in_threadpool(keys.get, key_id) is None:
            raise APIError(404, "not_found", "No key with that id.")
        await run_in_threadpool(keys.revoke, key_id)
        return no_store({"key": _key_json(await run_in_threadpool(keys.get, key_id), None)})

    # -- access requests ----------------------------------------------------------------------

    @router.get("/admin/api/requests")
    async def list_requests(request: Request) -> Response:
        await require_admin(request)
        return no_store({"requests": await run_in_threadpool(store(request).list_requests)})

    @router.post("/admin/api/requests/{request_id}")
    async def update_request(request_id: int, request: Request) -> Response:
        await require_admin(request)
        data = await _json(request)
        status = data.get("status")
        if status not in ("new", "done", "dismissed"):
            raise APIError(400, "invalid_field", "status must be new, done or dismissed.")
        if not await run_in_threadpool(store(request).set_request_status, request_id, status):
            raise APIError(404, "not_found", "No request with that id.")
        return no_store({"success": True})

    # -- models -------------------------------------------------------------------------------

    @router.get("/admin/api/models")
    async def models(request: Request) -> Response:
        await require_admin(request)
        keys = request.app.state.keys

        def collect():
            counts: dict[str | None, int] = {}
            for k in keys.list():
                if k.active:
                    counts[k.model] = counts.get(k.model, 0) + 1
            found = list_models(settings.models_dir)
            for m in found:
                m["keys"] = counts.get(m["folder"], 0)
                m["default"] = m["folder"] == settings.default_model
            return {"models": found, "default_folder": settings.default_model, "default_keys": counts.get(None, 0)}

        return no_store(await run_in_threadpool(collect))

    return router
