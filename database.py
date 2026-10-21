"""SQLite persistence. All queries use bound parameters."""
import json
import sqlite3
from contextlib import contextmanager

SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    id                TEXT PRIMARY KEY,
    original_filename TEXT NOT NULL,
    stored_filename   TEXT NOT NULL,
    derived_filename  TEXT,
    uploaded_at       TEXT NOT NULL,
    sha256            TEXT NOT NULL,
    file_size         INTEGER NOT NULL,
    mime_type         TEXT,
    ai_status         TEXT NOT NULL,
    ai_label          TEXT,
    ai_score          REAL,
    real_score        REAL,
    ai_prediction     TEXT,
    ai_model_id       TEXT,
    ai_model_revision TEXT,
    caution_score     INTEGER NOT NULL,
    summary_level     TEXT NOT NULL,
    metadata_json     TEXT NOT NULL,
    analysis_json     TEXT NOT NULL,
    report_json       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_analyses_sha256 ON analyses(sha256);
CREATE INDEX IF NOT EXISTS idx_analyses_uploaded ON analyses(uploaded_at);
"""

_SUMMARY_COLUMNS = ("id, original_filename, stored_filename, derived_filename, uploaded_at, sha256, "
                    "file_size, mime_type, ai_status, ai_label, ai_score, real_score, ai_prediction, caution_score, "
                    "summary_level")


class Database:
    def __init__(self, path):
        self.path = str(path)
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def insert(self, record):
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO analyses (id, original_filename, stored_filename, derived_filename, "
                "uploaded_at, sha256, file_size, mime_type, ai_status, ai_label, ai_score, real_score, "
                "ai_prediction, ai_model_id, ai_model_revision, caution_score, summary_level, "
                "metadata_json, analysis_json, report_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (record["id"], record["original_filename"], record["stored_filename"],
                 record.get("derived_filename"), record["uploaded_at"], record["sha256"],
                 record["file_size"], record.get("mime_type"), record["ai_status"],
                 record.get("ai_label"), record.get("ai_score"), record.get("real_score"),
                 record.get("ai_prediction"), record.get("ai_model_id"), record.get("ai_model_revision"),
                 record["caution_score"],
                 record["summary_level"], json.dumps(record["metadata"]),
                 json.dumps(record["analysis"]), json.dumps(record["report"])))

    def get(self, record_id):
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM analyses WHERE id = ?", (record_id,)).fetchone()
        if row is None:
            return None
        data = dict(row)
        for col in ("metadata_json", "analysis_json", "report_json"):
            data[col[:-5]] = json.loads(data.pop(col))
        return data

    def list(self, search=None, limit=200):
        sql = f"SELECT {_SUMMARY_COLUMNS} FROM analyses"
        params = []
        if search:
            sql += " WHERE original_filename LIKE ? ESCAPE '\\' OR sha256 LIKE ? ESCAPE '\\' OR id LIKE ? ESCAPE '\\'"
            like = "%" + search.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
            params = [like, like, like]
        sql += " ORDER BY uploaded_at DESC LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            return [dict(r) for r in conn.execute(sql, params).fetchall()]

    def find_by_hash(self, sha256, exclude_id=None):
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, original_filename, uploaded_at FROM analyses WHERE sha256 = ? AND id != ? "
                "ORDER BY uploaded_at", (sha256, exclude_id or "")).fetchall()
        return [dict(r) for r in rows]

    def stats(self):
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS total, "
                "SUM(CASE WHEN ai_status = 'ok' THEN 1 ELSE 0 END) AS ai_ran, "
                "SUM(CASE WHEN summary_level IN ('review','conflicting') THEN 1 ELSE 0 END) AS needs_review, "
                "COUNT(DISTINCT sha256) AS unique_files FROM analyses").fetchone()
        return {k: (row[k] or 0) for k in row.keys()}

    def delete(self, record_id):
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM analyses WHERE id = ?", (record_id,))
            return cur.rowcount > 0
