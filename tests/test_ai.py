"""AI detector tests: real inference with the pretrained model, and the model-unavailable fallback."""
import json

from ai_detector import AIDetector
from conftest import gradient, make_app, upload


def test_model_unavailable_fallback(tmp_path, png_no_exif):
    missing = AIDetector(str(tmp_path / "no-such-model"), None, enabled=True, local_files_only=True)
    app = make_app(tmp_path, detector=missing, AI_ENABLED=True)
    client = app.test_client()
    resp = upload(client, "a.png", png_no_exif)
    assert resp.status_code == 200
    rid = resp.get_json()["redirect"].rsplit("/", 1)[-1]
    rec = app.db.get(rid)
    assert rec["ai_status"] == "unavailable"
    assert rec["ai_score"] is None and rec["real_score"] is None and rec["ai_prediction"] is None
    ai = rec["report"]["ai_detection"]
    assert "scores" not in ai and "prediction_label" not in ai
    assert rec["metadata"]["image"]["format"] == "PNG"  # metadata analysis still works
    assert rec["summary_level"] == "limited"
    txt = client.get(f"/report/{rid}/download/txt").get_data(as_text=True)
    assert "AI analysis unavailable. No AI prediction was produced." in txt
    assert "AI analysis unavailable" in client.get(f"/analysis/{rid}").get_data(as_text=True)


def test_real_inference_outputs(real_detector):
    result = real_detector.predict(gradient((300, 300)))
    assert result["status"] == "ok"
    assert {s["label"] for s in result["scores"]} == {"ai", "hum"}
    total = sum(s["softmax"] for s in result["scores"])
    assert abs(total - 1.0) < 1e-4
    assert result["ai_score"] == next(s["softmax"] for s in result["scores"] if s["label"] == "ai")
    assert result["real_score"] == next(s["softmax"] for s in result["scores"] if s["label"] == "hum")
    expected = "Likely AI-generated" if result["ai_score"] > result["real_score"] else "Likely real"
    assert result["prediction_label"] == expected
    assert result["confidence"] is None  # never invented
    assert result["model_input_size"] == "224x224"
    # deterministic: same image -> same output
    again = real_detector.predict(gradient((300, 300)))
    assert again["scores"] == result["scores"]


def test_real_inference_end_to_end(tmp_path, real_detector, jpeg_with_exif):
    app = make_app(tmp_path, detector=real_detector, AI_ENABLED=True)
    client = app.test_client()
    resp = upload(client, "scene.jpg", jpeg_with_exif)
    rid = resp.get_json()["redirect"].rsplit("/", 1)[-1]
    rec = app.db.get(rid)
    assert rec["ai_status"] == "ok"
    assert rec["ai_model_id"] == "Ateeqq/ai-vs-human-image-detector"
    assert abs(rec["ai_score"] + rec["real_score"] - 1.0) < 1e-4
    page = client.get(f"/analysis/{rid}").get_data(as_text=True)
    assert f"{rec['ai_score'] * 100:.1f}%" in page and f"{rec['real_score'] * 100:.1f}%" in page
    assert "Not independently calibrated" in page
    rep = json.loads(client.get(f"/report/{rid}/download/json").data)
    assert rep["ai_detection"]["ai_score"] == rec["ai_score"]
    txt = client.get(f"/report/{rid}/download/txt").get_data(as_text=True)
    assert f"AI-generated score:  {rec['ai_score'] * 100:.2f}%" in txt
