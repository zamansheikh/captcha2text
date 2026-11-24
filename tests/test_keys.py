from __future__ import annotations

import json
import sqlite3

import pytest

from captcha2text.keys import KeyStore, main


def test_old_database_gets_model_columns(tmp_path):
    db = tmp_path / "old.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE api_keys (id INTEGER PRIMARY KEY, name TEXT NOT NULL, key_hash TEXT NOT NULL UNIQUE,
            key_prefix TEXT NOT NULL, rpm INTEGER NOT NULL, created_at TEXT NOT NULL, revoked_at TEXT);
        CREATE TABLE usage (id INTEGER PRIMARY KEY, key_id INTEGER NOT NULL, request_id TEXT NOT NULL,
            created_at TEXT NOT NULL, status INTEGER NOT NULL, error_code TEXT, processing_ms INTEGER);
        INSERT INTO api_keys (name, key_hash, key_prefix, rpm, created_at) VALUES ('Old', 'h', 'c2t_live_abcd', 5, 'x');
        """
    )
    conn.commit()
    conn.close()

    store = KeyStore(db)
    [old] = store.list()
    assert old.name == "Old" and old.model is None
    assert store.set_model(old.id, "acme-v1")
    assert store.list()[0].model == "acme-v1"
    store.log_usage(old.id, "req_X", 200, None, 3, "acme-v1")


def _model_folder(root, name):
    (root / name).mkdir(parents=True)
    (root / name / "model.json").write_text(json.dumps({"name": name}))


def test_cli_model_assignment(tmp_path, capsys):
    db, models = tmp_path / "k.db", tmp_path / "models"
    _model_folder(models, "acme-v1")
    base = ["--db", str(db), "--models", str(models)]

    assert main(base + ["create", "--name", "Acme", "--model", "acme-v1"]) == 0
    with pytest.raises(SystemExit, match="No model folder"):
        main(base + ["create", "--name", "Bad", "--model", "nope"])
    with pytest.raises(SystemExit, match="No model folder"):
        main(base + ["set-model", "1", "../etc"])

    assert main(base + ["set-model", "1", "default"]) == 0
    assert KeyStore(db).list()[0].model is None
    assert main(base + ["set-model", "1", "acme-v1"]) == 0
    assert main(base + ["list"]) == 0
    assert "acme-v1" in capsys.readouterr().out
