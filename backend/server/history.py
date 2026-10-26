"""Small local result repository, independent of browser storage and model code."""

import json
import os
from pathlib import Path
import sqlite3


def database_path():
    return Path(os.environ.get("MLBENCH_HISTORY_DB", str(Path(__file__).resolve().parents[2] / "tmp" / "experiments.sqlite3")))


def save_record(record):
    path = database_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path, timeout=10) as connection:
        connection.execute("CREATE TABLE IF NOT EXISTS experiments (id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
        connection.execute("INSERT OR REPLACE INTO experiments VALUES (?, ?)",
                           (record["id"], json.dumps(record, ensure_ascii=False, allow_nan=False)))


def load_records():
    path = database_path()
    if not path.exists():
        return {}
    with sqlite3.connect(path, timeout=10) as connection:
        connection.execute("CREATE TABLE IF NOT EXISTS experiments (id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
        records = {}
        for identity, payload in connection.execute("SELECT id, payload FROM experiments"):
            try:
                record = json.loads(payload)
            except (json.JSONDecodeError, TypeError):
                continue
            if record.get("status") in {"running", "queued"}:
                record.update(status="failed", stage="failed", progress=100,
                              error="后端在运行期间重启；请按原配置重新运行。")
            records[identity] = record
        return records
