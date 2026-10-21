"""Compare candidate detectors and write evaluation/results.json.

Decision rules (fixed before any results were seen):

Target (set by the project owner): on the held-out TEST split, balanced accuracy >= 70% and
real-photo accuracy >= 70%, with "Inconclusive" counted as a non-answer (never correct).

1. Candidates are scored on the VALIDATION split only. Eligible models have validation
   real-photo accuracy >= 70% and AI-image accuracy >= 50%. Among eligible models choose the
   highest validation balanced accuracy; ties -> lower false-positive rate -> higher AUC.
   If no model is eligible, choose the highest validation balanced accuracy and report that
   the target is unlikely to be met.
2. The optional "Inconclusive" band is chosen on the VALIDATION split only, for the selected
   model, from symmetric bands around 50%. Inconclusive answers never count as correct, so a
   band cannot raise accuracy; its only benefit is fewer wrong calls. Choose the band with the
   fewest wrong calls ((FP + FN) / N) that keeps validation balanced accuracy and real-photo
   accuracy >= 70% and leaves at most 25% of images inconclusive; prefer no band unless a band
   strictly reduces wrong calls.
3. The TEST split is scored exactly once, for the chosen model, to report the result. The
   previous model is also scored on it once, after the choice is frozen, as the "before" figure.

  python -m evaluation.compare_models
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate_model import evaluate, metrics, split_report  # noqa: E402
from evaluation.candidates import CANDIDATES  # noqa: E402

OUT = Path(__file__).resolve().parent / "results.json"
BANDS = [None, (0.4, 0.6), (0.35, 0.65), (0.3, 0.7), (0.25, 0.75), (0.2, 0.8)]
TARGET = 0.70
MIN_AI_ACCURACY_FOR_ELIGIBILITY = 0.50
MAX_INCONCLUSIVE = 0.25
BASELINE_ID = "Ateeqq/ai-vs-human-image-detector"  # model in use before this evaluation


def wrong_rate(m):
    return (m["confusion"]["real"]["ai"] + m["confusion"]["ai"]["real"]) / (m["n_real"] + m["n_ai"])


def choose_model(ok):
    eligible = [r for r in ok if r["val"]["real_accuracy"] >= TARGET
                and r["val"]["ai_accuracy"] >= MIN_AI_ACCURACY_FOR_ELIGIBILITY]
    pool = eligible or ok
    chosen = min(pool, key=lambda r: (-r["val"]["balanced_accuracy"], r["val"]["false_positive_rate"],
                                      -(r["val"]["auc"] or 0)))
    note = ("Chosen by the validation rule (eligible: real-photo accuracy >= 70% and AI-image accuracy "
            ">= 50% on validation; highest balanced accuracy)." if eligible else
            "No model met the validation eligibility rule; the highest validation balanced accuracy was "
            "chosen and the target is unlikely to be met.")
    return chosen, note, [r["id"] for r in eligible]


def choose_band(val_rows):
    trials = []
    for band in BANDS:
        m = metrics(val_rows, band)
        trials.append({"band": band, "wrong_rate": round(wrong_rate(m), 4),
                       "inconclusive_share": m["inconclusive_share"], "fpr": m["false_positive_rate"],
                       "fnr": m["false_negative_rate"], "balanced_accuracy": m["balanced_accuracy"],
                       "real_accuracy": m["real_accuracy"]})
    no_band = trials[0]
    allowed = [t for t in trials[1:] if t["inconclusive_share"] <= MAX_INCONCLUSIVE
               and t["balanced_accuracy"] >= TARGET and t["real_accuracy"] >= TARGET
               and t["wrong_rate"] < no_band["wrong_rate"]]
    best = min(allowed, key=lambda t: (t["wrong_rate"], t["inconclusive_share"])) if allowed else no_band
    return best["band"], trials


def main():
    results = []
    for cand in CANDIDATES:
        print(f"Evaluating {cand['id']} on the validation split ...", flush=True)
        try:
            res = evaluate(cand["id"], cand["revision"], splits={"val"})
        except Exception as exc:
            print(f"  failed: {exc}")
            results.append({**cand, "error": f"{type(exc).__name__}: {str(exc)[:300]}"})
            continue
        v = res["val"]
        print(f"  balanced={v['balanced_accuracy']} real={v['real_accuracy']} ai={v['ai_accuracy']} "
              f"auc={v['auc']}", flush=True)
        results.append({**cand, **res})

    ok = [r for r in results if "error" not in r]
    chosen, selection_note, eligible = choose_model(ok)
    band, band_trials = choose_band(chosen["per_image"])

    # The test split is scored exactly once, for the chosen model only.
    print(f"\nScoring the TEST split once with {chosen['id']} (band={band}) ...", flush=True)
    final = evaluate(chosen["id"], chosen["revision"], splits={"test"})
    test = split_report(final["per_image"], "test", band)
    test_no_band = split_report(final["per_image"], "test", None)
    t = test["all"]
    target_met = t["balanced_accuracy"] >= TARGET and t["real_accuracy"] >= TARGET

    # "Before" figure for the report: the previous model on the same test split, scored after the
    # choice above is frozen, so it cannot influence it.
    baseline = None
    if chosen["id"] != BASELINE_ID:
        base = next(c for c in CANDIDATES if c["id"] == BASELINE_ID)
        print(f"Scoring the TEST split once with the previous model {BASELINE_ID} (for before/after) ...", flush=True)
        base_eval = evaluate(base["id"], base["revision"], splits={"test"})
        baseline = {"id": BASELINE_ID, "revision": base["revision"],
                    "test": split_report(base_eval["per_image"], "test", None),
                    "per_image": base_eval["per_image"]}

    summary = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rules": __doc__.split("Decision rules (fixed before any results were seen):")[1].split("  python")[0].strip(),
        "dataset": {"n_real": chosen["val"]["n_real"] + t["n_real"], "n_ai": chosen["val"]["n_ai"] + t["n_ai"],
                    "val": {k: chosen["val"][k] for k in ("n_real", "n_ai")},
                    "test": {k: t[k] for k in ("n_real", "n_ai")},
                    "source": "Wikimedia Commons (see tests/eval_data/labels.csv)"},
        "target": {"balanced_accuracy": TARGET, "real_accuracy": TARGET, "met_on_test": target_met},
        "chosen_model": chosen["id"], "chosen_revision": chosen["revision"], "selection_note": selection_note,
        "eligible_models": eligible,
        "band_trials_val": band_trials, "chosen_band": band,
        "chosen_test": test, "chosen_test_without_band": test_no_band,
        "baseline_test": None if baseline is None else {k: v for k, v in baseline.items() if k != "per_image"},
        "candidates": [{k: v for k, v in r.items() if k != "per_image"} for r in results],
        "per_image": {"val": {r["id"]: r["per_image"] for r in ok}, "test": {chosen["id"]: final["per_image"],
                               **({BASELINE_ID: baseline["per_image"]} if baseline else {})}},
    }
    OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nValidation split (all candidates):")
    print("%-52s %7s %7s %7s %7s %6s" % ("model", "bal", "real", "ai", "FPR", "AUC"))
    for r in ok:
        v = r["val"]
        print("%-52s %7.3f %7.3f %7.3f %7.3f %6.3f" % (r["id"][:52], v["balanced_accuracy"], v["real_accuracy"],
                                                       v["ai_accuracy"], v["false_positive_rate"], v["auc"]))
    print(f"\nChosen: {chosen['id']} -- {selection_note} band={band}")
    print("Band trials (val):", json.dumps(band_trials, indent=1))
    print("TEST (chosen model, scored once):", json.dumps(test, indent=1))
    print("TEST without band:", json.dumps(test_no_band["all"], indent=1))
    if baseline:
        print("BASELINE TEST:", json.dumps(baseline["test"]["all"], indent=1))
    print("TARGET MET ON TEST:", target_met)
    return 0


if __name__ == "__main__":
    sys.exit(main())
