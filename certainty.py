"""Model certainty: the detector's top class score, plus (optionally) how often the model was right
at that certainty on the project's labelled test split.

Model certainty is read from the stored detector result (``top_score``); it is never recomputed,
estimated or combined with any other score. The empirical line is computed from the per-image scores
in evaluation/results.json, and only for the exact model and revision that were evaluated.
"""
import json
import logging

from ai_detector import decide

log = logging.getLogger(__name__)

LABEL = "Model certainty"
CERTAINTY_NOTE = ("How strongly the model favours its predicted class. This is the model's own score, not a "
                  "measured probability of being correct. It can be high on wrong results.")
NOT_CALIBRATED = "Not independently calibrated"
CALIBRATED_ON_TEST = "Certainty calibrated only against our small test set"

# Certainty bins for the empirical line: [lo, hi), the last one includes 1.0.
BINS = ((0.5, 0.7), (0.7, 0.9), (0.9, 1.0))
MIN_BIN_N = 10


def fmt(value):
    """The one display format for certainty (web page, TXT and PDF)."""
    return f"{value * 100:.1f}%"


def _bin_for(value):
    for lo, hi in BINS:
        if lo <= value < hi or (hi == 1.0 and value == 1.0):
            return lo, hi
    return None  # below 50%: only possible for models with more than two classes


def empirical_bin(top_score, ai, evaluation_file):
    """Test-split accuracy of predictions whose certainty falls in the same bin, or None.

    None when there is no evaluation file, it evaluated a different model or revision, the output is not
    a two-class softmax (per-image results store only the AI score, so the top score of other output
    types cannot be reconstructed), or the bin holds fewer than MIN_BIN_N test images.
    """
    if ai.get("output_type") != "softmax" or len(ai.get("scores") or []) != 2:
        return None
    try:
        data = json.loads(evaluation_file.read_text(encoding="utf-8"))
    except (OSError, ValueError, AttributeError):
        return None
    model_id, revision = data.get("chosen_model"), data.get("chosen_revision")
    if model_id != ai.get("model_id") or revision not in (ai.get("commit_hash"), ai.get("revision")):
        return None
    rows = ((data.get("per_image") or {}).get("test") or {}).get(model_id) or []
    target = _bin_for(top_score)
    if target is None:
        return None
    band = data.get("chosen_band")
    n = correct = 0
    for row in rows:
        score = row["ai_score"]
        if _bin_for(max(score, 1 - score)) != target:
            continue
        n += 1
        correct += decide(score, band) == row["label"]  # "inconclusive" never counts as correct
    if n < MIN_BIN_N:
        return None
    return {"range": list(target), "correct": correct, "n": n, "accuracy": round(correct / n, 4),
            "split": "test", "source": "evaluation/results.json", "evaluated_at": data.get("generated_at")}


def build(ai, evaluation_file):
    """The certainty block stored in a new report, or None when there is nothing real to show."""
    top = ai.get("top_score")
    if ai.get("status") != "ok" or ai.get("ai_score") is None or top is None:
        return None
    block = {"label": LABEL, "model_certainty": top, "certainty_note": CERTAINTY_NOTE,
             "calibration_note": NOT_CALIBRATED}
    emp = empirical_bin(top, ai, evaluation_file)
    if emp:
        block["empirical_accuracy_bin"] = emp
        block["calibration_note"] = CALIBRATED_ON_TEST
    return block


def empirical_text(emp):
    lo, hi = emp["range"]
    return (f"On our test set, predictions with this certainty ({lo * 100:.0f}-{hi * 100:.0f}%) were correct "
            f"{emp['accuracy'] * 100:.0f}% of the time (n = {emp['n']})")
