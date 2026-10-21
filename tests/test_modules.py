"""Unit tests for hashing, validation, metadata, heuristics and consistency rules."""
import hashlib

import pytest

import consistency
import heuristics
from integrity import ValidationError, sha256_bytes, sha256_file, validate_upload
from metadata import extract_metadata

MAX = 20 * 1024 * 1024
PIX = 80_000_000


def test_sha256_known_vector_and_repeatability(tmp_path):
    assert sha256_bytes(b"abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    data = bytes(range(256)) * 5000
    path = tmp_path / "f.bin"
    path.write_bytes(data)
    assert sha256_file(path) == sha256_bytes(data) == sha256_bytes(data) == hashlib.sha256(data).hexdigest()
    assert sha256_bytes(data + b"\x00") != sha256_bytes(data)


def test_validate_accepts_real_images(jpeg_with_exif, png_no_exif):
    info = validate_upload("photo.JPG", jpeg_with_exif, MAX, PIX)
    assert info["detected_format"] == "JPEG" and info["extension_matches_content"]
    assert validate_upload("x.png", png_no_exif, MAX, PIX)["detected_format"] == "PNG"


@pytest.mark.parametrize("name,data,msg", [
    ("evil.exe", b"MZ\x90\x00", "Unsupported file extension"),
    ("notes.txt", b"hello", "Unsupported file extension"),
    ("fake.jpg", b"this is not an image", "could not be decoded"),
    ("empty.png", b"", "empty"),
    ("noext", b"abc", "Unsupported file extension"),
])
def test_validate_rejects_bad_files(name, data, msg):
    with pytest.raises(ValidationError, match=msg):
        validate_upload(name, data, MAX, PIX)


def test_validate_rejects_oversize_and_truncated(jpeg_with_exif):
    with pytest.raises(ValidationError, match="too large"):
        validate_upload("a.jpg", jpeg_with_exif, 100, PIX)
    with pytest.raises(ValidationError, match="pixel safety limit"):
        validate_upload("a.jpg", jpeg_with_exif, MAX, 1000)
    with pytest.raises(ValidationError):
        validate_upload("a.jpg", jpeg_with_exif[: len(jpeg_with_exif) // 3], MAX, PIX)


def test_extension_mismatch_detected(png_no_exif):
    info = validate_upload("renamed.jpg", png_no_exif, MAX, PIX)
    assert info["detected_format"] == "PNG" and not info["extension_matches_content"]


def test_exif_extraction(jpeg_with_exif):
    md = extract_metadata(jpeg_with_exif, "photo.jpg")
    kf = md["key_fields"]
    assert md["exif_present"]
    assert kf["camera_make"] == "Canon"
    assert kf["camera_model"] == "Canon EOS 5D Mark IV"
    assert kf["software"] == "Adobe Photoshop 25.0"
    assert kf["capture_datetime"] == "2024:05:01 09:00:00"
    assert md["image"]["width"] == 320 and md["image"]["color_mode"] == "RGB"


def test_missing_exif_handled(png_no_exif):
    md = extract_metadata(png_no_exif, "plain.png")
    assert md["exif_present"] is False
    assert md["key_fields"]["camera_make"] is None
    assert md["errors"] == []


def _run(data, name, ai):
    from io import BytesIO
    from PIL import Image
    info = validate_upload(name, data, MAX, PIX)
    md = extract_metadata(data, name)
    with Image.open(BytesIO(data)) as img:
        heur = heuristics.analyse(img, md, info)
    ident = {"sha256": sha256_bytes(data), "stored_copy_verified": True, "previous_matches": []}
    return heur, consistency.analyse(ai, md, info, heur, ident)


UNAVAILABLE = {"status": "unavailable", "message": "test"}


def test_heuristics_editing_and_timeline(jpeg_with_exif):
    heur, cross = _run(jpeg_with_exif, "photo.jpg", UNAVAILABLE)
    ids = {f["id"] for f in heur["findings"]}
    assert {"editing_software", "modified_after_capture", "camera_metadata"} <= ids
    assert heur["caution_score"] == 25  # 15 editing + 10 modified after capture
    obs = {o["id"] for o in cross["observations"]}
    assert "editing_is_a_clue" in obs and "ai_unavailable" in obs
    assert cross["summary"]["level"] == "limited"


def test_missing_metadata_is_not_manipulation(png_no_exif):
    heur, cross = _run(png_no_exif, "plain.png", UNAVAILABLE)
    assert heur["caution_band"] == "low"
    obs = {o["id"]: o for o in cross["observations"]}
    assert "missing_exif_context" in obs
    assert "not proof of manipulation" in obs["missing_exif_context"]["title"]


def test_generator_signature_and_ai_conflict(png_generator):
    ai_not_flagged = {"status": "ok", "ai_score": 0.1, "real_score": 0.9, "predicted_class": "not_ai_generated",
                      "model_input_size": "224x224"}
    heur, cross = _run(png_generator, "gen.png", ai_not_flagged)
    assert any(f["id"] == "generator_signature" for f in heur["findings"])
    assert heur["caution_band"] == "elevated"
    assert any(o["id"] == "metadata_generated_vs_ai_not_flagged" for o in cross["observations"])
    assert cross["summary"]["level"] == "conflicting"
    assert "does not verify authenticity" in cross["summary"]["text"]


def test_hash_observation_explains_scope(png_no_exif):
    _, cross = _run(png_no_exif, "plain.png", UNAVAILABLE)
    hash_obs = next(o for o in cross["observations"] if o["id"] == "hash_identity")
    assert "says nothing about whether the depicted scene is genuine" in hash_obs["detail"]
