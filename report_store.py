"""Tamper-evident storage of the generated PDF report.

The PDF is generated once, written to a restricted folder under a fixed internal name, and its
SHA-256 is computed from the bytes on disk and recorded in SQLite (report_files + audit_log,
together with the SHA-256 of the original uploaded image). Every download re-hashes the stored
file first and is refused if it no longer matches. Users can also re-verify the stored file or
check a PDF they hold.

This is tamper-EVIDENT, not tamper-proof: a change to the file is detected only as long as the
recorded hash itself has not also been changed, i.e. against someone who cannot edit the database.
It does not establish legal chain of custody.
"""
import logging
from datetime import datetime, timezone
from pathlib import Path

import reports
from integrity import sha256_bytes, sha256_file

log = logging.getLogger(__name__)

FILE_TYPE = "PDF"
HASH_NOTE = ("SHA-256 of the generated PDF report file (computed from the stored file bytes when the PDF was "
             "generated). A hash cannot be embedded inside the file it describes, so it is recorded separately "
             "in the application's audit log and shown here.")


class ReportIntegrityError(Exception):
    """The stored report no longer matches its recorded hash (or is missing)."""


def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _path(config, row):
    base = Path(config["REPORTS_DIR"]).resolve()
    target = (base / row["stored_filename"]).resolve()
    if target.parent != base:  # never follow a path outside the reports folder
        raise ReportIntegrityError("Invalid stored report path.")
    return target


def _audit(db, record, event_type, outcome, report_sha256=None, detail=None):
    db.add_audit({"report_id": record["id"], "event_type": event_type, "file_type": FILE_TYPE,
                  "report_sha256": report_sha256, "original_sha256": record["sha256"], "outcome": outcome,
                  "detail": detail, "created_at": _now(), "app_version": reports.APP_VERSION})


def ensure_pdf(record, *, config, db):
    """Return the report_files row, generating and recording the PDF if it does not exist yet."""
    existing = db.get_report_file(record["id"])
    if existing:
        return existing
    path = Path(config["REPORTS_DIR"]) / f"{record['id']}.pdf"
    path.unlink(missing_ok=True)  # stale file without a database row: never trust it
    with open(path, "xb") as fh:
        fh.write(reports.to_pdf(record["report"]))
    digest = sha256_file(path)  # hash the final bytes as stored on disk
    row = {"report_id": record["id"], "file_type": FILE_TYPE, "stored_filename": path.name, "sha256": digest,
           "original_sha256": record["sha256"], "created_at": _now(), "app_version": reports.APP_VERSION}
    db.insert_report_file(row)
    _audit(db, record, "report_generated", "ok", digest, f"stored as {path.name}")
    log.info("Generated PDF for %s (sha256=%s)", record["id"], digest)
    return row


def check(record, *, config, db):
    """Compare the stored PDF with its recorded hash. Returns (status, row, current_hash)."""
    row = db.get_report_file(record["id"])
    if row is None:
        return "not_generated", None, None
    path = _path(config, row)
    if not path.exists():
        return "missing", row, None
    current = sha256_file(path)
    return ("match" if current == row["sha256"] else "mismatch"), row, current


def open_for_download(record, *, config, db):
    """Path of the stored PDF after a successful hash check; logs the download or the refusal."""
    ensure_pdf(record, config=config, db=db)
    status, row, current = check(record, config=config, db=db)
    if status != "match":
        _audit(db, record, "report_downloaded", "refused", current,
               f"stored file {status}; recorded {row['sha256'] if row else None}")
        log.error("Refused download of %s: stored PDF %s", record["id"], status)
        raise ReportIntegrityError(
            "The stored PDF report does not match its recorded SHA-256 hash, so it was not served. "
            "It may have been altered or removed after it was generated." if status == "mismatch" else
            "The stored PDF report file is missing, so it was not served.")
    _audit(db, record, "report_downloaded", "ok", current)
    return _path(config, row)


def verify_stored(record, *, config, db):
    """Re-hash the stored PDF and log the result. Returns a result dict."""
    status, row, current = check(record, config=config, db=db)
    if status == "not_generated":
        return {"status": status, "recorded": None, "computed": None, "source": "stored file"}
    _audit(db, record, "report_verified", status, current, "re-hash of stored file")
    return {"status": status, "recorded": row["sha256"], "computed": current, "source": "stored file"}


def verify_uploaded(record, data, filename, *, db):
    """Hash a PDF supplied by the user and compare it with the recorded hash. Returns a result dict."""
    row = db.get_report_file(record["id"])
    if row is None:
        return {"status": "not_generated", "recorded": None, "computed": None, "source": "uploaded file"}
    digest = sha256_bytes(data)
    status = "match" if digest == row["sha256"] else "mismatch"
    safe_name = Path((filename or "").replace("\\", "/")).name[:120]
    _audit(db, record, "report_verified", status, digest, f"uploaded file '{safe_name}' ({len(data)} bytes)")
    return {"status": status, "recorded": row["sha256"], "computed": digest, "source": "uploaded file"}


def remove(record, *, config, db):
    """Delete the stored PDF and its row; keep the audit rows but mark them as belonging to a deleted record."""
    row = db.get_report_file(record["id"])
    if row:
        try:
            _path(config, row).unlink(missing_ok=True)
        except ReportIntegrityError:
            pass
        db.delete_report_file(record["id"])
    db.mark_audit_deleted(record["id"], _now())


def integrity_block(record, *, db):
    """What JSON/TXT reports and the report page show about the PDF file hash."""
    row = db.get_report_file(record["id"])
    if row is None:
        return {"pdf_sha256": None, "note": "The PDF report has not been generated yet; it is generated and hashed "
                                            "on first download."}
    return {"pdf_sha256": row["sha256"], "pdf_generated_at": row["created_at"],
            "original_image_sha256": row["original_sha256"], "note": HASH_NOTE,
            "limitation": "Tamper-evident, not tamper-proof: changes to the PDF are detected only while the "
                          "recorded hash in the database is intact. This is not a legal chain of custody."}
