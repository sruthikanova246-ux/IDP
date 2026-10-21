"""Flask routes and application factory for the Digital Evidence Integrity System."""
import hmac
import json
import logging
import secrets
from logging.handlers import RotatingFileHandler
from pathlib import Path

from flask import (Flask, Response, abort, flash, jsonify, redirect, render_template, request,
                   send_from_directory, session, url_for)
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

import analyzer
import reports
from ai_detector import AIDetector, CONFIDENCE_NOTE, SCORE_NOTE
from config import load_config
from database import Database
from heuristics import WEIGHTS
from integrity import ALLOWED_EXTENSIONS, ValidationError, sha256_file

log = logging.getLogger("dei")

# Formats a browser can render inline for the preview.
BROWSER_PREVIEWABLE = {"JPEG", "MPO", "PNG", "WEBP", "GIF", "BMP"}


def create_app(overrides=None, detector=None):
    cfg = load_config(overrides)
    app = Flask(__name__)
    app.config.update(cfg)
    app.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax")

    _setup_logging(app)
    app.db = Database(app.config["DATABASE_PATH"])
    app.detector = detector or AIDetector(app.config["AI_MODEL_ID"], app.config["AI_MODEL_REVISION"],
                                          enabled=app.config["AI_ENABLED"],
                                          local_files_only=app.config["AI_LOCAL_FILES_ONLY"])
    if app.config["AI_PRELOAD"] and not app.config["TESTING"]:
        app.detector.start_background_load()

    _register_helpers(app)
    _register_routes(app)
    _register_errors(app)
    return app


def _setup_logging(app):
    if app.config["TESTING"] or any(isinstance(h, RotatingFileHandler) for h in log.handlers):
        return
    handler = RotatingFileHandler(app.config["LOG_PATH"], maxBytes=1_000_000, backupCount=3, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    for name in ("dei", "analyzer", "ai_detector"):
        logging.getLogger(name).addHandler(handler)
        logging.getLogger(name).setLevel(logging.INFO)


# --------------------------------------------------------------------- helpers
EVALUATION_FILE = Path(__file__).resolve().parent / "evaluation" / "results.json"


def _load_evaluation():
    """Measured detector results (evaluation/compare_models.py); None if not generated."""
    try:
        data = json.loads(EVALUATION_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    data.pop("per_image", None)
    return data


def _csrf_token():
    if "_csrf" not in session:
        session["_csrf"] = secrets.token_urlsafe(32)
    return session["_csrf"]


def _check_csrf():
    sent = request.form.get("csrf_token") or request.headers.get("X-CSRF-Token") or ""
    if not hmac.compare_digest(sent, session.get("_csrf", "")):
        abort(400, description="The form expired or was not submitted from this application. "
                               "Reload the page and try again.")


def _wants_json():
    return request.headers.get("X-Requested-With") == "fetch"


def _valid_id_or_404(record_id):
    if not analyzer.REPORT_ID_RE.match(record_id or ""):
        abort(404)


def _register_helpers(app):
    app.jinja_env.globals.update(csrf_token=_csrf_token)

    @app.template_filter("pct")
    def pct_filter(value, digits=1):
        return "n/a" if value is None else f"{value * 100:.{digits}f}%"

    @app.template_filter("filesize")
    def filesize_filter(n):
        for unit in ("bytes", "KB", "MB", "GB"):
            if n < 1024 or unit == "GB":
                return f"{n:,} bytes" if unit == "bytes" else f"{n:.1f} {unit}"
            n /= 1024

    @app.template_filter("when")
    def when_filter(iso_value):
        return (iso_value or "").replace("T", " ").replace("Z", " UTC")

    @app.context_processor
    def inject():
        return {"model_status": app.detector.status()}

    @app.after_request
    def security_headers(resp):
        resp.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' blob:; style-src 'self'; script-src 'self'; "
            "object-src 'none'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'")
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["X-Frame-Options"] = "DENY"
        resp.headers["Referrer-Policy"] = "no-referrer"
        return resp


# ---------------------------------------------------------------------- routes
def _register_routes(app):
    def record_or_404(record_id):
        _valid_id_or_404(record_id)
        record = app.db.get(record_id)
        if record is None:
            abort(404)
        return record

    def stored_copy_check(record):
        path = Path(app.config["UPLOAD_DIR"]) / record["stored_filename"]
        if not path.exists():
            return "missing"
        return "verified" if sha256_file(path) == record["sha256"] else "mismatch"

    @app.get("/")
    def index():
        return render_template("index.html", stats=app.db.stats(), recent=app.db.list(limit=5))

    @app.get("/upload")
    def upload():
        return render_template("upload.html", allowed=sorted(ALLOWED_EXTENSIONS),
                               max_mb=app.config["MAX_UPLOAD_MB"])

    @app.post("/upload")
    def upload_post():
        _check_csrf()
        file = request.files.get("evidence")
        if file is None or not file.filename:
            return _upload_error("Choose an image file to analyse.")
        data = file.read()
        try:
            record_id = analyzer.analyse_upload(file.filename, data, config=app.config,
                                                db=app.db, detector=app.detector)
        except ValidationError as exc:
            return _upload_error(str(exc))
        target = url_for("analysis", record_id=record_id)
        if _wants_json():
            return jsonify({"ok": True, "redirect": target})
        return redirect(target)

    def _upload_error(message):
        if _wants_json():
            return jsonify({"ok": False, "error": message}), 400
        flash(message, "error")
        return redirect(url_for("upload"))

    @app.get("/analysis/<record_id>")
    def analysis(record_id):
        record = record_or_404(record_id)
        return render_template("analysis.html", r=record, rep=record["report"],
                               previewable=record["report"]["file"]["detected_format"] in BROWSER_PREVIEWABLE)

    @app.get("/report/<record_id>")
    def report(record_id):
        record = record_or_404(record_id)
        return render_template("report.html", r=record, rep=record["report"],
                               storage_check=stored_copy_check(record),
                               previewable=record["report"]["file"]["detected_format"] in BROWSER_PREVIEWABLE)

    @app.get("/report/<record_id>/download/<fmt>")
    def download(record_id, fmt):
        record = record_or_404(record_id)
        rep = record["report"]
        if fmt == "json":
            body, mime = reports.to_json(rep).encode("utf-8"), "application/json"
        elif fmt == "txt":
            body, mime = reports.to_txt(rep).encode("utf-8"), "text/plain"
        elif fmt == "pdf":
            body, mime = reports.to_pdf(rep), "application/pdf"
        else:
            abort(404)
        return Response(body, mimetype=mime, headers={
            "Content-Disposition": f'attachment; filename="{record_id}.{fmt}"'})

    @app.get("/evidence/<record_id>/<kind>")
    def evidence_file(record_id, kind):
        record = record_or_404(record_id)
        if kind == "image":
            directory, name = app.config["UPLOAD_DIR"], record["stored_filename"]
        elif kind == "ela" and record.get("derived_filename"):
            directory, name = app.config["DERIVED_DIR"], record["derived_filename"]
        else:
            abort(404)
        resp = send_from_directory(directory, name, max_age=0)
        resp.headers["Content-Disposition"] = "inline"
        return resp

    @app.get("/history")
    def history():
        q = (request.args.get("q") or "").strip()[:100]
        return render_template("history.html", records=app.db.list(search=q or None), q=q)

    @app.post("/analysis/<record_id>/delete")
    def delete(record_id):
        _check_csrf()
        _valid_id_or_404(record_id)
        if not analyzer.delete_analysis(record_id, config=app.config, db=app.db):
            abort(404)
        flash(f"Report {record_id} and its stored files were deleted.", "success")
        return redirect(url_for("history"))

    @app.get("/methodology")
    def methodology():
        return render_template("methodology.html", evaluation=_load_evaluation(), limitations=reports.LIMITATIONS,
                               custody_note=reports.CUSTODY_NOTE, weights=WEIGHTS,
                               score_note=SCORE_NOTE,
                               confidence_note=CONFIDENCE_NOTE)

    @app.get("/api/model-status")
    def model_status():
        return jsonify(app.detector.status())

    @app.post("/api/model-load")
    def model_load():
        _check_csrf()
        app.detector.request_load()
        return jsonify(app.detector.status())


def _register_errors(app):
    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_exc):
        message = f"File is too large. Maximum upload size is {app.config['MAX_UPLOAD_MB']} MB."
        if _wants_json():
            return jsonify({"ok": False, "error": message}), 413
        return render_template("error.html", code=413, message=message), 413

    @app.errorhandler(HTTPException)
    def http_error(exc):
        if _wants_json():
            return jsonify({"ok": False, "error": exc.description}), exc.code
        return render_template("error.html", code=exc.code, message=exc.description), exc.code

    @app.errorhandler(Exception)
    def unexpected(exc):
        log.exception("Unhandled error")
        message = "An unexpected error occurred. Details were written to the server log."
        if _wants_json():
            return jsonify({"ok": False, "error": message}), 500
        return render_template("error.html", code=500, message=message), 500


if __name__ == "__main__":
    application = create_app()
    host, port = application.config["HOST"], application.config["PORT"]
    if host not in ("127.0.0.1", "localhost"):
        print(f"WARNING: binding to {host} exposes the app to your network.")
    print(f"Digital Evidence Integrity System running at http://{host}:{port}/  (Ctrl+C to stop)")
    application.run(host=host, port=port, debug=False, threaded=True)
