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

-- The one stored, hashed report file per analysis (generated once, served on every download).
CREATE TABLE IF NOT EXISTS report_files (
    report_id        TEXT PRIMARY KEY,
    file_type        TEXT NOT NULL,
    stored_filename  TEXT NOT NULL,
    sha256           TEXT NOT NULL,
    original_sha256  TEXT NOT NULL,
    created_at       TEXT NOT NULL,
    app_version      TEXT NOT NULL
);

-- Append-only audit trail. Rows are never deleted; when an analysis is deleted its rows are
-- marked record_deleted so the history stays consistent.
CREATE TABLE IF NOT EXISTS audit_log (
    audit_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    report_id         TEXT NOT NULL,
    event_type        TEXT NOT NULL CHECK (event_type IN
                          ('report_generated', 'report_downloaded', 'report_verified')),
    file_type         TEXT NOT NULL,
    report_sha256     TEXT,
    original_sha256   TEXT NOT NULL,
    outcome           TEXT NOT NULL,
    detail            TEXT,
    created_at        TEXT NOT NULL,
    app_version       TEXT NOT NULL,
    record_deleted    INTEGER NOT NULL DEFAULT 0,
    record_deleted_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_audit_report ON audit_log(report_id);
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

    # ------------------------------------------------------------ report files
    def get_report_file(self, report_id):
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM report_files WHERE report_id = ?", (report_id,)).fetchone()
        return dict(row) if row else None

    def insert_report_file(self, row):
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO report_files (report_id, file_type, stored_filename, sha256, original_sha256, "
                "created_at, app_version) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (row["report_id"], row["file_type"], row["stored_filename"], row["sha256"],
                 row["original_sha256"], row["created_at"], row["app_version"]))

    def delete_report_file(self, report_id):
        with self._connect() as conn:
            conn.execute("DELETE FROM report_files WHERE report_id = ?", (report_id,))

    # --------------------------------------------------------------- audit log
    def add_audit(self, event):
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO audit_log (report_id, event_type, file_type, report_sha256, original_sha256, "
                "outcome, detail, created_at, app_version) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (event["report_id"], event["event_type"], event["file_type"], event.get("report_sha256"),
                 event["original_sha256"], event["outcome"], event.get("detail"), event["created_at"],
                 event["app_version"]))
            return cur.lastrowid

    def audit_for(self, report_id):
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM audit_log WHERE report_id = ? ORDER BY audit_id",
                                (report_id,)).fetchall()
        return [dict(r) for r in rows]

    def mark_audit_deleted(self, report_id, when):
        with self._connect() as conn:
            conn.execute("UPDATE audit_log SET record_deleted = 1, record_deleted_at = ? WHERE report_id = ?",
                         (when, report_id))

    def delete(self, record_id):
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM analyses WHERE id = ?", (record_id,))
            return cur.rowcount > 0
