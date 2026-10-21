"""Orchestrates one analysis: validate -> store -> hash -> metadata -> heuristics -> AI -> cross-check -> report."""
import io
import logging
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

import consistency
import heuristics
import report_store
import reports
from integrity import md5_bytes, sha256_bytes, sha256_file, validate_upload
from metadata import extract_metadata

log = logging.getLogger(__name__)

REPORT_ID_RE = re.compile(r"^DEI-\d{8}-[0-9A-F]{12}$")
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")


def new_report_id(now):
    return f"DEI-{now:%Y%m%d}-{uuid.uuid4().hex[:12].upper()}"


def display_filename(filename):
    """Original name for display/records only: no directory parts or control characters."""
    name = Path((filename or "").replace("\\", "/")).name
    return _CONTROL_CHARS.sub("", name)[:200] or "unnamed"


def analyse_upload(filename, data, *, config, db, detector):
    """Run the full pipeline and persist it. Raises integrity.ValidationError for bad uploads."""
    received = datetime.now(timezone.utc)
    # Fingerprint the bytes exactly as received, before anything else touches them.
    sha256 = sha256_bytes(data)
    md5 = md5_bytes(data)

    file_info = validate_upload(filename, data, config["MAX_CONTENT_LENGTH"], config["MAX_IMAGE_PIXELS"])
    report_id = new_report_id(received)
    stored_name = f"{report_id}.{file_info['storage_extension']}"
    stored_path = Path(config["UPLOAD_DIR"]) / stored_name
    ela_name = f"{report_id}_ela.png"
    ela_path = Path(config["DERIVED_DIR"]) / ela_name

    with open(stored_path, "xb") as fh:
        fh.write(data)
    try:
        stored_ok = sha256_file(stored_path) == sha256
        metadata = extract_metadata(data, filename)
        with Image.open(io.BytesIO(data)) as img:
            heur = heuristics.analyse(img, metadata, file_info, ela_path=ela_path)
            ai = detector.predict(img)

        identity = {"sha256": sha256, "md5": md5, "stored_copy_verified": stored_ok,
                    "previous_matches": db.find_by_hash(sha256)}
        cross = consistency.analyse(ai, metadata, file_info, heur, identity)
        report = reports.build_report(
            report_id=report_id, original_filename=display_filename(filename),
            received_at=received, file_info=file_info, metadata=metadata, identity=identity,
            ai=ai, heuristics_result=heur, consistency_result=cross,
            detector_status=detector.status())

        db.insert({
            "id": report_id,
            "original_filename": display_filename(filename),
            "stored_filename": stored_name,
            "derived_filename": ela_name if heur.get("ela") else None,
            "uploaded_at": report["file"]["received_at"],
            "sha256": sha256,
            "file_size": len(data),
            "mime_type": metadata["image"].get("mime_type"),
            "ai_status": ai["status"],
            "ai_label": ai.get("predicted_label"),
            "ai_score": ai.get("ai_score"),
            "real_score": ai.get("real_score"),
            "ai_prediction": ai.get("prediction_label"),
            "ai_model_id": ai.get("model_id"),
            "ai_model_revision": ai.get("commit_hash") or ai.get("revision"),
            "caution_score": heur["caution_score"],
            "summary_level": cross["summary"]["level"],
            "metadata": metadata,
            "analysis": {"ai": ai, "heuristics": heur, "consistency": cross, "identity": identity},
            "report": report,
        })
    except Exception:
        # Do not leave orphaned evidence files when the record could not be saved.
        stored_path.unlink(missing_ok=True)
        ela_path.unlink(missing_ok=True)
        raise
    log.info("Analysed %s as %s (sha256=%s, ai=%s)", display_filename(filename), report_id, sha256, ai["status"])
    # Generate and hash the PDF once, now. If this fails the analysis is still kept; the PDF is then
    # generated (and recorded) on its first download.
    try:
        report_store.ensure_pdf(db.get(report_id), config=config, db=db)
    except Exception:
        log.exception("PDF generation failed for %s", report_id)
    return report_id


def delete_analysis(record_id, *, config, db):
    """Delete the database record and its stored files. Returns False if not found."""
    record = db.get(record_id)
    if record is None:
        return False
    upload_dir = Path(config["UPLOAD_DIR"]).resolve()
    derived_dir = Path(config["DERIVED_DIR"]).resolve()
    paths = [(upload_dir, record["stored_filename"])]
    if record.get("derived_filename"):
        paths.append((derived_dir, record["derived_filename"]))
    report_store.remove(record, config=config, db=db)
    db.delete(record_id)
    for base, name in paths:
        target = (base / name).resolve()
        if target.parent == base:  # never follow a path outside the storage folder
            target.unlink(missing_ok=True)
    log.info("Deleted analysis %s", record_id)
    return True
