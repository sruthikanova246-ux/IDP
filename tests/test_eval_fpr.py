"""Regression guard: false-positive rate of the configured model on real photographs.

Runs the configured detector (exactly as the app does) over the real photographs in the TEST
split of tests/eval_data/ and fails if too many are labelled "Likely AI-generated".

MAX_TEST_FPR is documented in README.md ("Evaluation"). It is a regression limit set from the
measured result, not a quality claim: the measured rate on this small set is reported in
evaluation/results.json and on the Methodology page.

Skipped (with the reason) when the evaluation images or the model weights are not available:
  python tests/eval_data/build_eval_set.py   # download the evaluation images
  python download_model.py                   # download the model
"""
import csv
from pathlib import Path

import pytest
from PIL import Image

from ai_detector import AIDetector
from config import DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION

EVAL_DIR = Path(__file__).resolve().parent / "eval_data"
MAX_TEST_FPR = 0.10  # measured 0/40 (0%) on 2026-10-21; limit allows at most 4 of 40 before failing


def _real_test_rows():
    labels = EVAL_DIR / "labels.csv"
    if not labels.exists():
        pytest.skip("tests/eval_data/labels.csv is missing")
    with open(labels, newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["label"] == "real" and r["split"] == "test"]
    present = [r for r in rows if (EVAL_DIR / "images" / r["file"]).exists()]
    if len(present) < len(rows):
        pytest.skip(f"evaluation images missing ({len(present)}/{len(rows)} present); "
                    "run python tests/eval_data/build_eval_set.py")
    return present


def test_false_positive_rate_on_real_photos():
    rows = _real_test_rows()
    detector = AIDetector(DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION, enabled=True, local_files_only=True)
    if not detector.load():
        pytest.skip(f"model not available locally: {detector.error}")
    flagged = []
    for r in rows:
        with Image.open(EVAL_DIR / "images" / r["file"]) as img:
            img.load()
            res = detector.predict(img)
        assert res["status"] == "ok", res.get("message")
        if res["predicted_class"] == "ai_generated":
            flagged.append((r["file"], round(res["ai_score"], 3)))
    fpr = len(flagged) / len(rows)
    assert fpr <= MAX_TEST_FPR, (f"false-positive rate {fpr:.1%} on {len(rows)} real test photos exceeds "
                                 f"{MAX_TEST_FPR:.0%}; flagged: {flagged}")
