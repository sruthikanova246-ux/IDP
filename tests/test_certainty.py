"""Model certainty: shown only for stored ok results, equal to the stored top score, never estimated."""
import json
import sqlite3

import pytest

import certainty
from ai_detector import CONFIDENCE_NOTE, PREDICTION_LABELS, decide
from conftest import make_app, upload
from config import BASE_DIR, DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION


class FakeDetector:
    """Returns a fixed two-class softmax result shaped like AIDetector.predict (no model needed)."""

    def __init__(self, ai_score, band=None, model_id=DEFAULT_MODEL_ID, revision=DEFAULT_MODEL_REVISION):
        self.ai_score, self.band, self.model_id, self.revision = ai_score, band, model_id, revision

    def status(self):
        return {"enabled": True, "state": "ready", "model_id": self.model_id, "revision": self.revision,
                "known_limitations": ["test limitation"], "decision_rule": "test rule", "band": self.band}

    def start_background_load(self):
        pass

    def request_load(self):
        pass

    def predict(self, img):
        ai, real = round(self.ai_score, 6), round(1 - self.ai_score, 6)
        scores = [{"index": 0, "label": "artificial", "logit": 1.0, "score": ai},
                  {"index": 1, "label": "human", "logit": -1.0, "score": real}]
        top = max(scores, key=lambda s: s["score"])
        call = decide(ai, self.band)
        return {"model_id": self.model_id, "revision": self.revision, "status": "ok",
                "commit_hash": self.revision, "architecture": "Fake", "display_name": "Fake",
                "output_type": "softmax", "output_mapping": "test mapping", "scores": scores,
                "inference_ms": 1.0, "model_input_size": "224x224", "score_note": "test note",
                "decision_rule": "test rule", "inconclusive_band": list(self.band) if self.band else None,
                "known_limitations": ["test limitation"], "confidence": None, "confidence_note": CONFIDENCE_NOTE,
                "ai_score": ai, "real_score": real,
                "predicted_class": {"ai": "ai_generated", "real": "not_ai_generated"}.get(call, "inconclusive"),
                "prediction_label": PREDICTION_LABELS[call], "top_score": top["score"], "top_label": top["label"]}


def write_eval(path, rows, model_id=DEFAULT_MODEL_ID, revision=DEFAULT_MODEL_REVISION, band=None):
    path.write_text(json.dumps({"generated_at": "2026-10-21T00:00:00Z", "chosen_model": model_id,
                                "chosen_revision": revision, "chosen_band": band,
                                "per_image": {"test": {model_id: rows}, "val": {model_id: []}}}), encoding="utf-8")
    return path


def rows(n_correct, n_wrong, score=0.97):
    """Test rows whose top score is `score`: correct ones labelled 'ai', wrong ones labelled 'real'."""
    return ([{"file": f"ai/{i}.png", "label": "ai", "split": "test", "ai_score": score} for i in range(n_correct)]
            + [{"file": f"real/{i}.jpg", "label": "real", "split": "test", "ai_score": score} for i in range(n_wrong)])


def analyse(tmp_path, detector, data, evaluation_file):
    app = make_app(tmp_path, detector=detector, AI_ENABLED=True, EVALUATION_FILE=evaluation_file)
    client = app.test_client()
    resp = upload(client, "img.png", data)
    assert resp.status_code == 200, resp.get_json()
    rid = resp.get_json()["redirect"].rsplit("/", 1)[-1]
    return app, client, rid


def downloads(client, rid):
    txt = client.get(f"/report/{rid}/download/txt").get_data(as_text=True)
    js = json.loads(client.get(f"/report/{rid}/download/json").data)
    pdf = client.get(f"/report/{rid}/download/pdf").data
    return txt, js, pdf


def test_block_shows_stored_top_score(tmp_path, png_no_exif):
    app, client, rid = analyse(tmp_path, FakeDetector(0.8432), png_no_exif, tmp_path / "missing.json")
    rec = app.db.get(rid)
    stored_top = rec["analysis"]["ai"]["top_score"]
    cert = rec["report"]["certainty"]
    assert stored_top == 0.8432 and cert["model_certainty"] == stored_top
    assert cert["label"] == "Model certainty" and cert["certainty_note"] == certainty.CERTAINTY_NOTE
    assert "empirical_accuracy_bin" not in cert  # no evaluation data: never estimated
    for url in (f"/analysis/{rid}", f"/report/{rid}"):
        page = client.get(url).get_data(as_text=True)
        assert f'data-certainty="{stored_top}"' in page
        assert "84.3%" in page and 'aria-label="Model certainty 84.3%"' in page
        assert "It can be high on wrong results." in page
        assert "Not independently calibrated" in page and "data-empirical" not in page
    # The verdict and scores are untouched.
    assert rec["ai_prediction"] == "Likely AI-generated" and rec["ai_score"] == 0.8432


def test_downloads_carry_the_same_value(tmp_path, png_no_exif):
    app, client, rid = analyse(tmp_path, FakeDetector(0.1234), png_no_exif, tmp_path / "missing.json")
    txt, js, pdf = downloads(client, rid)
    stored_top = app.db.get(rid)["analysis"]["ai"]["top_score"]
    assert stored_top == 0.8766
    assert js["certainty"]["model_certainty"] == stored_top
    assert js["certainty"]["certainty_note"] == certainty.CERTAINTY_NOTE
    line = "Model certainty:     87.7%"
    assert line in txt and line.encode("latin-1") in pdf
    assert "Calibration:         Not independently calibrated" in txt
    assert b"Calibration:         Not independently calibrated" in pdf
    assert "measured probability of being correct" in txt


def test_inconclusive_keeps_label_and_certainty(tmp_path, png_no_exif):
    app, client, rid = analyse(tmp_path, FakeDetector(0.6, band=(0.3, 0.7)), png_no_exif, tmp_path / "missing.json")
    rec = app.db.get(rid)
    assert rec["ai_prediction"] == "Inconclusive"
    assert rec["report"]["certainty"]["model_certainty"] == 0.6
    page = client.get(f"/analysis/{rid}").get_data(as_text=True)
    assert "Status: Inconclusive" in page and "60.0%" in page


def test_no_block_when_ai_unavailable(tmp_path, png_no_exif):
    app = make_app(tmp_path)  # AI disabled
    client = app.test_client()
    rid = upload(client, "a.png", png_no_exif).get_json()["redirect"].rsplit("/", 1)[-1]
    rec = app.db.get(rid)
    assert "certainty" not in rec["report"]
    page = client.get(f"/analysis/{rid}").get_data(as_text=True)
    assert "AI analysis unavailable" in page and "data-certainty" not in page and "Model certainty" not in page
    txt, js, pdf = downloads(client, rid)
    assert "certainty" not in js and "Model certainty" not in txt and b"Model certainty" not in pdf


def test_old_record_without_value_still_opens(tmp_path, png_no_exif):
    app, client, rid = analyse(tmp_path, FakeDetector(0.9), png_no_exif, tmp_path / "missing.json")
    rec = app.db.get(rid)
    report, analysis = rec["report"], rec["analysis"]
    del report["certainty"]
    report["schema_version"] = "1.1"
    for ai in (report["ai_detection"], analysis["ai"]):
        ai.pop("top_score"), ai.pop("top_label")
    with sqlite3.connect(app.config["DATABASE_PATH"]) as conn:
        conn.execute("UPDATE analyses SET report_json = ?, analysis_json = ? WHERE id = ?",
                     (json.dumps(report), json.dumps(analysis), rid))
    for url in (f"/analysis/{rid}", f"/report/{rid}"):
        resp = client.get(url)
        page = resp.get_data(as_text=True)
        assert resp.status_code == 200 and "data-certainty" not in page and "Not independently calibrated" in page
    txt = client.get(f"/report/{rid}/download/txt").get_data(as_text=True)
    assert "Model certainty" not in txt and "Calibration:         Not independently calibrated" in txt
    assert "certainty" not in json.loads(client.get(f"/report/{rid}/download/json").data)


def test_empirical_line_with_enough_test_images(tmp_path, png_no_exif):
    evaluation = write_eval(tmp_path / "eval.json", rows(9, 3) + rows(5, 0, score=0.6))
    app, client, rid = analyse(tmp_path, FakeDetector(0.95), png_no_exif, evaluation)
    emp = app.db.get(rid)["report"]["certainty"]["empirical_accuracy_bin"]
    # Only the 12 rows in the 90-100% bin count: 9 correct ('ai' predicted, labelled 'ai'), 3 wrong.
    assert emp["range"] == [0.9, 1.0] and emp["n"] == 12 and emp["correct"] == 9 and emp["accuracy"] == 0.75
    text = "On our test set, predictions with this certainty (90-100%) were correct 75% of the time (n = 12)"
    page = client.get(f"/analysis/{rid}").get_data(as_text=True)
    assert text in page and certainty.CALIBRATED_ON_TEST in page
    txt, js, pdf = downloads(client, rid)
    assert js["certainty"]["empirical_accuracy_bin"]["n"] == 12
    assert js["certainty"]["calibration_note"] == certainty.CALIBRATED_ON_TEST
    assert " ".join(txt.split()).count(text) == 1
    # PDF text strings escape parentheses.
    assert rb"On our test set, predictions with this certainty \(90-100%\) were correct 75%" in pdf
    assert rb"of the time \(n = 12\)." in pdf


@pytest.mark.parametrize("case", ["too_few", "other_model", "no_file", "other_bin"])
def test_empirical_line_omitted(tmp_path, png_no_exif, case):
    evaluation = tmp_path / "eval.json"
    if case == "too_few":
        write_eval(evaluation, rows(9, 0))                      # n = 9 < 10
    elif case == "other_model":
        write_eval(evaluation, rows(20, 0), revision="0" * 40)  # evaluated a different revision
    elif case == "other_bin":
        write_eval(evaluation, rows(20, 0, score=0.75))         # data only in the 70-90% bin
    app, client, rid = analyse(tmp_path, FakeDetector(0.95), png_no_exif, evaluation)
    cert = app.db.get(rid)["report"]["certainty"]
    assert "empirical_accuracy_bin" not in cert and cert["calibration_note"] == certainty.NOT_CALIBRATED
    page = client.get(f"/analysis/{rid}").get_data(as_text=True)
    assert "data-empirical" not in page and "On our test set" not in page
    assert "Not independently calibrated" in page


def test_empirical_bin_matches_project_evaluation():
    """Against the real evaluation/results.json, the bin equals an independent recount of the test rows."""
    path = BASE_DIR / "evaluation" / "results.json"
    if not path.exists():
        pytest.skip("evaluation/results.json not generated")
    data = json.loads(path.read_text(encoding="utf-8"))
    ai = {"status": "ok", "model_id": data["chosen_model"], "revision": data["chosen_revision"],
          "commit_hash": data["chosen_revision"], "output_type": "softmax", "scores": [{}, {}]}
    test_rows = data["per_image"]["test"][data["chosen_model"]]
    in_bin = [r for r in test_rows if max(r["ai_score"], 1 - r["ai_score"]) >= 0.9]
    correct = sum((("ai" if r["ai_score"] > 0.5 else "real") == r["label"]) for r in in_bin)
    emp = certainty.empirical_bin(0.999, ai, path)
    assert emp["n"] == len(in_bin) and emp["correct"] == correct
    # Bins below 90% hold fewer than 10 test images for this model, so no line is shown there.
    assert certainty.empirical_bin(0.6, ai, path) is None and certainty.empirical_bin(0.8, ai, path) is None


def test_real_model_top_score_is_its_top_class_score(real_detector):
    from conftest import gradient
    result = real_detector.predict(gradient((300, 300)))
    assert result["top_score"] == max(s["score"] for s in result["scores"])
    assert result["top_score"] == max(result["ai_score"], result["real_score"]) or \
        abs(result["top_score"] - max(result["ai_score"], result["real_score"])) < 1e-5
