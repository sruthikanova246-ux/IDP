"""Tamper-evident PDF report: stored once, hashed from the file, checked before download, audited."""
import hashlib
import json

from conftest import csrf, make_app, upload
from integrity import sha256_bytes, sha256_file


def _setup(tmp_path, image):
    app = make_app(tmp_path)
    client = app.test_client()
    resp = upload(client, "scene.jpg", image)
    assert resp.status_code == 200
    rid = resp.get_json()["redirect"].rsplit("/", 1)[-1]
    return app, client, rid


def _pdf_path(app, rid):
    row = app.db.get_report_file(rid)
    return app.config["REPORTS_DIR"] / row["stored_filename"], row


def test_pdf_hash_from_stored_file_and_stable_across_downloads(tmp_path, jpeg_with_exif):
    app, client, rid = _setup(tmp_path, jpeg_with_exif)
    path, row = _pdf_path(app, rid)
    assert path.exists() and path.parent == app.config["REPORTS_DIR"]
    assert row["sha256"] == sha256_file(path)                      # hash of the stored bytes
    assert row["original_sha256"] == sha256_bytes(jpeg_with_exif)

    first = client.get(f"/report/{rid}/download/pdf")
    second = client.get(f"/report/{rid}/download/pdf")
    assert first.status_code == second.status_code == 200
    assert first.data == second.data == path.read_bytes()         # same stored file, not regenerated
    assert hashlib.sha256(first.data).hexdigest() == row["sha256"]
    assert first.data.startswith(b"%PDF")

    # the recorded hash is shown on the report page and in the TXT/JSON exports
    assert row["sha256"] in client.get(f"/report/{rid}").get_data(as_text=True)
    exported = json.loads(client.get(f"/report/{rid}/download/json").data)
    assert exported["report_file_integrity"]["pdf_sha256"] == row["sha256"]
    assert "SHA-256 of the generated PDF report file" in exported["report_file_integrity"]["note"]
    txt = client.get(f"/report/{rid}/download/txt").get_data(as_text=True)
    assert f"PDF report SHA-256:   {row['sha256']}" in txt


def test_audit_rows_record_both_hashes(tmp_path, jpeg_with_exif):
    app, client, rid = _setup(tmp_path, jpeg_with_exif)
    _, row = _pdf_path(app, rid)
    client.get(f"/report/{rid}/download/pdf")
    client.post(f"/report/{rid}/verify", data={"csrf_token": csrf(client)})
    rows = app.db.audit_for(rid)
    assert [r["event_type"] for r in rows] == ["report_generated", "report_downloaded", "report_verified"]
    for r in rows:
        assert r["report_sha256"] == row["sha256"]
        assert r["original_sha256"] == sha256_bytes(jpeg_with_exif)
        assert r["file_type"] == "PDF" and r["outcome"] in ("ok", "match")
        assert r["created_at"].endswith("Z") and r["app_version"]


def test_one_changed_byte_blocks_download_and_fails_verification(tmp_path, jpeg_with_exif):
    app, client, rid = _setup(tmp_path, jpeg_with_exif)
    path, row = _pdf_path(app, rid)
    original = client.get(f"/report/{rid}/download/pdf").data

    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0x01                                    # flip one bit in one byte
    path.write_bytes(bytes(data))

    resp = client.get(f"/report/{rid}/download/pdf")
    assert resp.status_code == 409
    assert "does not match its recorded SHA-256" in resp.get_data(as_text=True)

    page = client.post(f"/report/{rid}/verify", data={"csrf_token": csrf(client)}, follow_redirects=True)
    assert "does not match the recorded hash" in page.get_data(as_text=True)

    outcomes = [(r["event_type"], r["outcome"]) for r in app.db.audit_for(rid)]
    assert ("report_downloaded", "refused") in outcomes
    assert ("report_verified", "mismatch") in outcomes
    assert app.db.get_report_file(rid)["sha256"] == row["sha256"]  # the recorded value is never overwritten

    # a user-supplied copy is checked against the recorded hash too
    token = csrf(client)
    ok = client.post(f"/report/{rid}/verify", data={"csrf_token": token, "report_file": (__import__("io").BytesIO(original), "copy.pdf")},
                     content_type="multipart/form-data", follow_redirects=True)
    assert "uploaded file matches the recorded hash" in ok.get_data(as_text=True)
    bad = client.post(f"/report/{rid}/verify", data={"csrf_token": token, "report_file": (__import__("io").BytesIO(bytes(data)), "edited.pdf")},
                      content_type="multipart/form-data", follow_redirects=True)
    assert "uploaded file does not match the recorded hash" in bad.get_data(as_text=True)


def test_missing_pdf_is_refused(tmp_path, png_no_exif):
    app, client, rid = _setup(tmp_path, png_no_exif)
    path, _ = _pdf_path(app, rid)
    path.unlink()
    resp = client.get(f"/report/{rid}/download/pdf")
    assert resp.status_code == 409 and "missing" in resp.get_data(as_text=True)


def test_delete_removes_pdf_and_marks_audit(tmp_path, jpeg_with_exif):
    app, client, rid = _setup(tmp_path, jpeg_with_exif)
    path, _ = _pdf_path(app, rid)
    client.get(f"/report/{rid}/download/pdf")
    resp = client.post(f"/analysis/{rid}/delete", data={"csrf_token": csrf(client)})
    assert resp.status_code == 302
    assert not path.exists() and app.db.get_report_file(rid) is None
    rows = app.db.audit_for(rid)
    assert rows and all(r["record_deleted"] == 1 and r["record_deleted_at"] for r in rows)
    for url in ("/", "/history", "/methodology", "/upload"):
        assert client.get(url).status_code == 200
    assert client.get(f"/report/{rid}/download/pdf").status_code == 404
    # a new analysis afterwards still gets its own PDF
    resp = upload(client, "again.jpg", jpeg_with_exif)
    new_id = resp.get_json()["redirect"].rsplit("/", 1)[-1]
    assert app.db.get_report_file(new_id) is not None


def test_record_without_pdf_still_opens_and_generates_on_download(tmp_path, png_no_exif):
    """Records created before this feature have no PDF row; they must open and get one on first download."""
    app, client, rid = _setup(tmp_path, png_no_exif)
    path, _ = _pdf_path(app, rid)
    path.unlink()
    app.db.delete_report_file(rid)                                 # simulate a pre-feature record
    page = client.get(f"/report/{rid}")
    assert page.status_code == 200 and "not been generated yet" in page.get_data(as_text=True)
    assert json.loads(client.get(f"/report/{rid}/download/json").data)["report_file_integrity"]["pdf_sha256"] is None
    resp = client.get(f"/report/{rid}/download/pdf")
    assert resp.status_code == 200
    row = app.db.get_report_file(rid)
    assert row and hashlib.sha256(resp.data).hexdigest() == row["sha256"]
