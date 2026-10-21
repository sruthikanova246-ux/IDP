"""End-to-end workflow tests through the Flask test client."""
import json

from conftest import csrf, make_app, upload
from integrity import sha256_bytes, sha256_file


def _record_id(resp):
    assert resp.status_code == 200, resp.get_json()
    return resp.get_json()["redirect"].rsplit("/", 1)[-1]


def test_valid_upload_persists_and_hashes(tmp_path, jpeg_with_exif):
    app = make_app(tmp_path)
    client = app.test_client()
    rid = _record_id(upload(client, "scene.jpg", jpeg_with_exif))
    rec = app.db.get(rid)
    assert rec["original_filename"] == "scene.jpg"
    assert rec["sha256"] == sha256_bytes(jpeg_with_exif)
    stored = app.config["UPLOAD_DIR"] / rec["stored_filename"]
    assert stored.name == f"{rid}.jpg" and sha256_file(stored) == rec["sha256"]
    assert rec["metadata"]["key_fields"]["camera_make"] == "Canon"
    # AI disabled in this app: must be reported, never faked
    assert rec["ai_status"] == "disabled" and rec["ai_score"] is None and rec["real_score"] is None
    page = client.get(f"/analysis/{rid}").get_data(as_text=True)
    assert "AI analysis unavailable" in page and rec["sha256"] in page


def test_invalid_file_rejected(tmp_path):
    client = make_app(tmp_path).test_client()
    resp = upload(client, "fake.png", b"definitely not a png")
    assert resp.status_code == 400 and "could not be decoded" in resp.get_json()["error"]
    resp = upload(client, "script.html", b"<script>alert(1)</script>")
    assert resp.status_code == 400 and "Unsupported file extension" in resp.get_json()["error"]
    assert list((tmp_path / "uploads").iterdir()) == []


def test_oversized_file_rejected(tmp_path, png_no_exif):
    client = make_app(tmp_path, MAX_UPLOAD_MB=1).test_client()
    resp = upload(client, "big.png", png_no_exif + b"\x00" * (1024 * 1024 + 10))
    assert resp.status_code == 413 and "too large" in resp.get_json()["error"]


def test_csrf_required(tmp_path, png_no_exif):
    client = make_app(tmp_path).test_client()
    csrf(client)
    resp = upload(client, "a.png", png_no_exif, token="wrong")
    assert resp.status_code == 400


def test_report_downloads(tmp_path, jpeg_with_exif):
    app = make_app(tmp_path)
    client = app.test_client()
    rid = _record_id(upload(client, "scene.jpg", jpeg_with_exif))
    j = client.get(f"/report/{rid}/download/json")
    assert j.status_code == 200 and "attachment" in j.headers["Content-Disposition"]
    rep = json.loads(j.data)
    assert rep["report_id"] == rid and rep["file"]["sha256"] == sha256_bytes(jpeg_with_exif)
    assert rep["ai_detection"]["status"] == "disabled"
    t = client.get(f"/report/{rid}/download/txt").get_data(as_text=True)
    assert rid in t and "AI analysis unavailable" in t and "Canon EOS 5D Mark IV" in t
    p = client.get(f"/report/{rid}/download/pdf")
    assert p.status_code == 200 and p.data.startswith(b"%PDF-1.4") and p.data.rstrip().endswith(b"%%EOF")
    xref_offset = int(p.data.rsplit(b"startxref", 1)[1].split()[0])
    assert p.data[xref_offset:xref_offset + 4] == b"xref"
    assert rid.encode() in p.data and b"SHA-256" in p.data
    assert client.get(f"/report/{rid}/download/exe").status_code == 404
    html = client.get(f"/report/{rid}").get_data(as_text=True)
    assert "Re-hashed on opening this report" in html


def test_history_persists_across_restart(tmp_path, png_no_exif):
    app1 = make_app(tmp_path)
    rid = _record_id(upload(app1.test_client(), "first.png", png_no_exif))
    app2 = make_app(tmp_path)  # new app instance, same database = server restart
    client = app2.test_client()
    assert rid in client.get("/history").get_data(as_text=True)
    assert client.get(f"/report/{rid}").status_code == 200
    assert rid in client.get("/history?q=first").get_data(as_text=True)
    assert rid not in client.get("/history?q=nomatch").get_data(as_text=True)


def test_duplicate_file_detected(tmp_path, png_no_exif):
    app = make_app(tmp_path)
    client = app.test_client()
    first = _record_id(upload(client, "a.png", png_no_exif))
    second = _record_id(upload(client, "b.png", png_no_exif))
    obs = app.db.get(second)["report"]["consistency"]["observations"]
    assert any(first in o["detail"] for o in obs if o["id"].startswith("duplicate_"))


def test_delete_removes_record_and_files(tmp_path, jpeg_with_exif):
    app = make_app(tmp_path)
    client = app.test_client()
    rid = _record_id(upload(client, "scene.jpg", jpeg_with_exif))
    rec = app.db.get(rid)
    stored = app.config["UPLOAD_DIR"] / rec["stored_filename"]
    ela = app.config["DERIVED_DIR"] / rec["derived_filename"]
    assert stored.exists() and ela.exists()
    resp = client.post(f"/analysis/{rid}/delete", data={"csrf_token": csrf(client)})
    assert resp.status_code == 302
    assert app.db.get(rid) is None and not stored.exists() and not ela.exists()
    assert client.get(f"/report/{rid}").status_code == 404


def test_bad_ids_and_pages(tmp_path):
    client = make_app(tmp_path).test_client()
    for url in ("/report/../../etc", "/report/DEI-1-x", "/evidence/..%2Fsecret_key/image", "/analysis/nope"):
        assert client.get(url).status_code == 404
    for url in ("/", "/upload", "/history", "/methodology", "/api/model-status"):
        resp = client.get(url)
        assert resp.status_code == 200
        assert "default-src 'self'" in resp.headers["Content-Security-Policy"]
    assert "No analyses stored yet" in client.get("/history").get_data(as_text=True)
