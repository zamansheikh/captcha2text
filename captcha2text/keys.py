"""API keys and the usage log, stored in SQLite.

Only a SHA-256 hash of each key is stored. The plain key is shown once, when it is created.

Command line:

  c2t-keys create --name "Acme Ltd" --rpm 60 [--model acme-v1]
  c2t-keys list
  c2t-keys set-model 3 acme-v1        (or "default" to use the default model again)
  c2t-keys revoke 3
  c2t-keys usage --days 30
"""

from __future__ import annotations

import argparse
import hashlib
import secrets
import sqlite3
import sys
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

KEY_PREFIX = "c2t_live_"

SCHEMA = """
CREATE TABLE IF NOT EXISTS api_keys (
    id          INTEGER PRIMARY KEY,
    name        TEXT    NOT NULL,
    key_hash    TEXT    NOT NULL UNIQUE,
    key_prefix  TEXT    NOT NULL,
    rpm         INTEGER NOT NULL,
    created_at  TEXT    NOT NULL,
    revoked_at  TEXT,
    model       TEXT
);
CREATE TABLE IF NOT EXISTS usage (
    id            INTEGER PRIMARY KEY,
    key_id        INTEGER NOT NULL REFERENCES api_keys(id),
    request_id    TEXT    NOT NULL,
    created_at    TEXT    NOT NULL,
    status        INTEGER NOT NULL,
    error_code    TEXT,
    processing_ms INTEGER,
    model         TEXT
);
CREATE INDEX IF NOT EXISTS usage_key_time ON usage(key_id, created_at);
"""

# Columns added after the first release, for databases created before them.
MIGRATIONS = [("api_keys", "model", "TEXT"), ("usage", "model", "TEXT")]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def generate_key() -> str:
    return KEY_PREFIX + secrets.token_urlsafe(24)


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


@dataclass(frozen=True)
class ApiKey:
    id: int
    name: str
    prefix: str
    rpm: int
    created_at: str
    revoked_at: str | None
    model: str | None = None  # None: use the server's default model

    @property
    def active(self) -> bool:
        return self.revoked_at is None


class KeyStore:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        with self._conn() as conn:
            conn.executescript(SCHEMA)
            for table, column, kind in MIGRATIONS:
                columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
                if column not in columns:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {kind}")

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.db_path, timeout=10)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=10000")
            self._local.conn = conn
        return conn

    @staticmethod
    def _row(row: sqlite3.Row) -> ApiKey:
        return ApiKey(
            row["id"], row["name"], row["key_prefix"], row["rpm"], row["created_at"], row["revoked_at"], row["model"]
        )

    def create(self, name: str, rpm: int, model: str | None = None) -> tuple[ApiKey, str]:
        key = generate_key()
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO api_keys (name, key_hash, key_prefix, rpm, created_at, model) VALUES (?, ?, ?, ?, ?, ?)",
                (name, hash_key(key), key[: len(KEY_PREFIX) + 4], rpm, _now(), model),
            )
            row = conn.execute("SELECT * FROM api_keys WHERE id = ?", (cur.lastrowid,)).fetchone()
        return self._row(row), key

    def find(self, key: str) -> ApiKey | None:
        row = self._conn().execute("SELECT * FROM api_keys WHERE key_hash = ?", (hash_key(key),)).fetchone()
        return self._row(row) if row else None

    def list(self) -> list[ApiKey]:
        return [self._row(r) for r in self._conn().execute("SELECT * FROM api_keys ORDER BY id")]

    def revoke(self, key_id: int) -> bool:
        with self._conn() as conn:
            cur = conn.execute(
                "UPDATE api_keys SET revoked_at = ? WHERE id = ? AND revoked_at IS NULL", (_now(), key_id)
            )
        return cur.rowcount == 1

    def set_rpm(self, key_id: int, rpm: int) -> bool:
        with self._conn() as conn:
            cur = conn.execute("UPDATE api_keys SET rpm = ? WHERE id = ?", (rpm, key_id))
        return cur.rowcount == 1

    def set_model(self, key_id: int, model: str | None) -> bool:
        with self._conn() as conn:
            cur = conn.execute("UPDATE api_keys SET model = ? WHERE id = ?", (model, key_id))
        return cur.rowcount == 1

    def log_usage(
        self, key_id: int, request_id: str, status: int, error_code: str | None, ms: int | None,
        model: str | None = None,
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO usage (key_id, request_id, created_at, status, error_code, processing_ms, model)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (key_id, request_id, _now(), status, error_code, ms, model),
            )

    def get(self, key_id: int) -> ApiKey | None:
        row = self._conn().execute("SELECT * FROM api_keys WHERE id = ?", (key_id,)).fetchone()
        return self._row(row) if row else None

    def usage_by_key(self, days: int) -> dict[int, dict]:
        """key id -> requests, solved, failed in the last `days` days, and when it was last used."""
        since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(timespec="seconds")
        rows = self._conn().execute(
            """
            SELECT key_id,
                   SUM(created_at >= ?)                     AS total,
                   SUM(created_at >= ? AND status = 200)    AS solved,
                   SUM(created_at >= ? AND status <> 200)   AS failed,
                   MAX(created_at)                          AS last_used
            FROM usage GROUP BY key_id
            """,
            (since, since, since),
        ).fetchall()
        return {r["key_id"]: {"total": r["total"] or 0, "solved": r["solved"] or 0, "failed": r["failed"] or 0,
                              "last_used": r["last_used"]} for r in rows}

    def usage_daily(self, days: int) -> list[dict]:
        """One row per UTC day for the last `days` days, oldest first, including empty days."""
        today = datetime.now(timezone.utc).date()
        first = today - timedelta(days=days - 1)
        rows = self._conn().execute(
            """
            SELECT substr(created_at, 1, 10) AS day,
                   SUM(status = 200) AS solved, SUM(status <> 200) AS failed
            FROM usage WHERE created_at >= ? GROUP BY day
            """,
            (first.isoformat(),),
        ).fetchall()
        by_day = {r["day"]: r for r in rows}
        out = []
        for i in range(days):
            day = (first + timedelta(days=i)).isoformat()
            r = by_day.get(day)
            out.append({"day": day, "solved": (r["solved"] or 0) if r else 0, "failed": (r["failed"] or 0) if r else 0})
        return out

    def usage_summary(self, days: int) -> list[sqlite3.Row]:
        since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(timespec="seconds")
        return self._conn().execute(
            """
            SELECT k.id, k.name, k.key_prefix,
                   COUNT(u.id)                                   AS total,
                   COALESCE(SUM(u.status = 200), 0)              AS solved,
                   COALESCE(SUM(u.status <> 200), 0)             AS failed,
                   CAST(AVG(u.processing_ms) AS INTEGER)         AS avg_ms
            FROM api_keys k
            LEFT JOIN usage u ON u.key_id = k.id AND u.created_at >= ?
            GROUP BY k.id ORDER BY k.id
            """,
            (since,),
        ).fetchall()


def main(argv: list[str] | None = None) -> int:
    from .config import Settings
    from .model import model_exists

    settings = Settings.from_env()
    parser = argparse.ArgumentParser(prog="c2t-keys", description="Manage Captcha2Text API keys.")
    parser.add_argument("--db", type=Path, default=settings.db_path, help=f"SQLite file (default {settings.db_path})")
    parser.add_argument("--models", type=Path, default=settings.models_dir,
                        help=f"model folders, to check --model names (default {settings.models_dir})")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("create", help="create a key and print it once")
    p.add_argument("--name", required=True, help="customer or app name")
    p.add_argument("--rpm", type=int, default=settings.default_rpm, help="requests per minute, 0 = no limit")
    p.add_argument("--model", help="model folder this key uses (default: the server's default model)")

    p = sub.add_parser("set-model", help="choose the model a key uses")
    p.add_argument("id", type=int)
    p.add_argument("model", help='a model folder name, or "default"')

    sub.add_parser("list", help="list keys")

    p = sub.add_parser("revoke", help="revoke a key; it stops working at once")
    p.add_argument("id", type=int)

    p = sub.add_parser("set-rpm", help="change a key's requests-per-minute limit")
    p.add_argument("id", type=int)
    p.add_argument("rpm", type=int)

    p = sub.add_parser("usage", help="requests per key")
    p.add_argument("--days", type=int, default=30)

    p = sub.add_parser("admin-create", help="create an admin panel login; prints a generated password once")
    p.add_argument("username")
    p.add_argument("--password-stdin", action="store_true", help="read the password from stdin instead")

    p = sub.add_parser("admin-password", help="set a new password (signs the admin out everywhere)")
    p.add_argument("username")
    p.add_argument("--password-stdin", action="store_true", help="read the password from stdin instead")

    sub.add_parser("admin-list", help="list admin panel logins")

    args = parser.parse_args(argv)
    store = KeyStore(args.db)

    if args.cmd.startswith("admin-"):
        from .admin import AdminStore, generate_password

        admins = AdminStore(args.db)
        if args.cmd == "admin-list":
            for a in admins.list_admins():
                print(f"{a['id']:>4}  {a['username']:<20} created {a['created_at']}  last sign-in {a['last_login_at'] or '-'}")
            return 0
        password = sys.stdin.readline().rstrip("\n") if args.password_stdin else generate_password()
        if len(password) < 12:
            print("Passwords must be at least 12 characters.", file=sys.stderr)
            return 1
        if args.cmd == "admin-create":
            try:
                admins.create_admin(args.username, password)
            except (ValueError, sqlite3.IntegrityError) as e:
                print(f"Could not create {args.username}: {e}", file=sys.stderr)
                return 1
        elif not admins.set_password(args.username, password):
            print(f"No admin named {args.username}.", file=sys.stderr)
            return 1
        if not args.password_stdin:
            print(f"Admin {args.username} can sign in at /admin with this password (shown once):\n\n  {password}\n")
        else:
            print(f"Password set for {args.username}.")
        return 0

    def check_model(name: str) -> None:
        if not model_exists(args.models, name):
            raise SystemExit(f"No model folder {args.models / name} with a model.json.")

    if args.cmd == "create":
        if args.model:
            check_model(args.model)
        key, plain = store.create(args.name, args.rpm, args.model)
        model = key.model or "the default model"
        print(f"Created key {key.id} for {key.name} ({key.rpm} requests/minute, {model}).")
        print("Give this key to the customer. It is not stored and cannot be shown again:\n")
        print(f"  {plain}\n")
    elif args.cmd == "set-model":
        model = None if args.model == "default" else args.model
        if model:
            check_model(model)
        if not store.set_model(args.id, model):
            print(f"No key with id {args.id}.", file=sys.stderr)
            return 1
        print(f"Key {args.id} now uses {model or 'the default model'}.")
    elif args.cmd == "list":
        rows = store.list()
        if not rows:
            print("No keys yet. Create one with: c2t-keys create --name NAME")
        for k in rows:
            state = "active" if k.active else f"revoked {k.revoked_at}"
            print(f"{k.id:>4}  {k.prefix}…  {k.rpm:>5} rpm  {k.model or '(default)':<18} {state:<32} {k.name}")
    elif args.cmd == "revoke":
        if not store.revoke(args.id):
            print(f"No active key with id {args.id}.", file=sys.stderr)
            return 1
        print(f"Revoked key {args.id}.")
    elif args.cmd == "set-rpm":
        if not store.set_rpm(args.id, args.rpm):
            print(f"No key with id {args.id}.", file=sys.stderr)
            return 1
        print(f"Key {args.id} now allows {args.rpm} requests/minute.")
    elif args.cmd == "usage":
        print(f"Last {args.days} days")
        print(f"{'id':>4}  {'key':<14} {'total':>8} {'solved':>8} {'failed':>8} {'avg ms':>7}  name")
        for r in store.usage_summary(args.days):
            avg = r["avg_ms"] if r["avg_ms"] is not None else "-"
            print(
                f"{r['id']:>4}  {r['key_prefix'] + '…':<14} {r['total']:>8} {r['solved']:>8}"
                f" {r['failed']:>8} {avg:>7}  {r['name']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
