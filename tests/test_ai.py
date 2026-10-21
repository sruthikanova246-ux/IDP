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
    assert {s["label"] for s in result["scores"]} == {"artificial", "human"}
    total = sum(s["score"] for s in result["scores"])
    assert abs(total - 1.0) < 1e-4
    # the AI score is the model's own output for the 'artificial' class - not relabelled or flipped
    assert abs(result["ai_score"] - next(s["score"] for s in result["scores"] if s["label"] == "artificial")) < 1e-6
    assert abs(result["real_score"] - next(s["score"] for s in result["scores"] if s["label"] == "human")) < 1e-5
    expected = "Likely AI-generated" if result["ai_score"] > 0.5 else "Likely real"
    assert result["prediction_label"] == expected
    assert result["confidence"] is None  # never invented
    assert result["model_input_size"] == "224x224"
    assert result["model_id"] == "Smogy/SMOGY-Ai-images-detector"
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
    assert rec["ai_model_id"] == "Smogy/SMOGY-Ai-images-detector"
    assert rec["ai_model_revision"] == "fb9aadb85642d707f05b40191689d60f75b9a3e6"
    assert abs(rec["ai_score"] + rec["real_score"] - 1.0) < 1e-4
    page = client.get(f"/analysis/{rid}").get_data(as_text=True)
    assert f"{rec['ai_score'] * 100:.1f}%" in page and f"{rec['real_score'] * 100:.1f}%" in page
    assert "Not independently calibrated" in page
    rep = json.loads(client.get(f"/report/{rid}/download/json").data)
    assert rep["ai_detection"]["ai_score"] == rec["ai_score"]
    txt = client.get(f"/report/{rid}/download/txt").get_data(as_text=True)
    assert f"AI-generated score:  {rec['ai_score'] * 100:.2f}%" in txt


def test_label_mapping_rules():
    """The AI score must come from the label that means AI - never from position or a flipped label."""
    from ai_detector import _label_mapping
    m = _label_mapping({0: "artificial", 1: "human"}, 2, None)
    assert m["kind"] == "softmax" and m["ai_indices"] == [0]
    m = _label_mapping({0: "human", 1: "artificial"}, 2, None)   # order swapped -> index follows the label
    assert m["ai_indices"] == [1]
    m = _label_mapping({0: "dalle", 1: "real", 2: "sd"}, 3, None)  # AI = 1 - P(real)
    assert m["ai_indices"] == [0, 2]
    assert _label_mapping({0: "LABEL_0"}, 1, None)["kind"] == "unmapped"  # unreviewed sigmoid: no guess
    assert _label_mapping({0: "cat", 1: "dog"}, 2, None)["kind"] == "unmapped"


def test_inconclusive_band_is_a_display_rule():
    from ai_detector import decide
    assert decide(0.9) == "ai" and decide(0.1) == "real" and decide(0.5) == "real"
    assert decide(0.5, (0.35, 0.65)) == "inconclusive" and decide(0.7, (0.35, 0.65)) == "ai"
