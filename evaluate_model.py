"""Evaluate an AI-image detector on the labelled set in tests/eval_data/.

Runs every image through the application's own AIDetector (same preprocessing as production)
and reports, per split: confusion matrix, false-positive rate on real photos, false-negative rate
on AI images, ROC AUC and score distributions, plus breakdowns by stratum and variant.

  python evaluate_model.py                                   # the configured model
  python evaluate_model.py --model ORG/NAME --revision SHA   # any image-classification model
  python evaluate_model.py --out results.json                # also save per-image scores
"""
import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

from PIL import Image

from ai_detector import AIDetector, decide
from config import DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION

EVAL_DIR = Path(__file__).resolve().parent / "tests" / "eval_data"


def load_rows(splits=None):
    with open(EVAL_DIR / "labels.csv", newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if splits is None or r["split"] in splits]
    return [r for r in rows if (EVAL_DIR / "images" / r["file"]).exists()]


def score_images(detector, rows):
    out = []
    for r in rows:
        with Image.open(EVAL_DIR / "images" / r["file"]) as img:
            img.load()
            res = detector.predict(img)
        if res["status"] != "ok" or res.get("ai_score") is None:
            raise RuntimeError(f"{r['file']}: {res.get('status')} {res.get('message')}")
        out.append({**r, "ai_score": res["ai_score"]})
    return out


def auc(scored):
    """Probability that a random AI image scores higher than a random real one (ties count half)."""
    pos = [s["ai_score"] for s in scored if s["label"] == "ai"]
    neg = [s["ai_score"] for s in scored if s["label"] == "real"]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def metrics(scored, band=None):
    """Counts with the 50% argmax rule; with a band, scores inside it are 'inconclusive'."""
    real = [s for s in scored if s["label"] == "real"]
    ai = [s for s in scored if s["label"] == "ai"]

    def call(score):
        return decide(score, band)

    tally = {"real": {"ai": 0, "real": 0, "inconclusive": 0}, "ai": {"ai": 0, "real": 0, "inconclusive": 0}}
    for s in scored:
        tally[s["label"]][call(s["ai_score"])] += 1

    def dist(items):
        vals = sorted(s["ai_score"] for s in items)
        if not vals:
            return None
        q = statistics.quantiles(vals, n=4) if len(vals) > 1 else [vals[0]] * 3
        return {"min": round(vals[0], 4), "q1": round(q[0], 4), "median": round(q[1], 4),
                "q3": round(q[2], 4), "max": round(vals[-1], 4)}

    n_real, n_ai = len(real), len(ai)

    def rate(num, den):
        return round(num / den, 4) if den else None

    # "Inconclusive" is a non-answer: it never counts as correct.
    real_acc = rate(tally["real"]["real"], n_real)
    ai_acc = rate(tally["ai"]["ai"], n_ai)
    answered_real = n_real - tally["real"]["inconclusive"]
    answered_ai = n_ai - tally["ai"]["inconclusive"]
    real_acc_ans = rate(tally["real"]["real"], answered_real)
    ai_acc_ans = rate(tally["ai"]["ai"], answered_ai)
    return {
        "n_real": n_real, "n_ai": n_ai,
        "confusion": tally,  # rows = true label, columns = predicted
        "real_accuracy": real_acc,
        "ai_accuracy": ai_acc,
        "balanced_accuracy": round((real_acc + ai_acc) / 2, 4) if None not in (real_acc, ai_acc) else None,
        "false_positive_rate": rate(tally["real"]["ai"], n_real),
        "false_negative_rate": rate(tally["ai"]["real"], n_ai),
        "inconclusive_share": rate(tally["real"]["inconclusive"] + tally["ai"]["inconclusive"], n_real + n_ai),
        "inconclusive_rate_real": rate(tally["real"]["inconclusive"], n_real),
        "inconclusive_rate_ai": rate(tally["ai"]["inconclusive"], n_ai),
        # accuracy among images that received an answer (excluding inconclusive) - shown for transparency only
        "answered_only": {
            "n_real": answered_real, "n_ai": answered_ai,
            "real_accuracy": real_acc_ans, "ai_accuracy": ai_acc_ans,
            "balanced_accuracy": (round((real_acc_ans + ai_acc_ans) / 2, 4)
                                  if None not in (real_acc_ans, ai_acc_ans) else None),
        },
        "auc": None if auc(scored) is None else round(auc(scored), 4),
        "ai_score_real_photos": dist(real),
        "ai_score_ai_images": dist(ai),
    }


def split_report(scored, split, band=None):
    """Metrics for one split: everything, originals only (real originals + AI), and the stress group."""
    part = [s for s in scored if s["split"] == split]
    if not part:
        return None
    originals = [s for s in part if s["label"] == "ai" or s["variant"] == "original"]
    stress = [s for s in part if s["label"] == "real" and s["variant"] != "original"]
    return {"all": metrics(part, band), "originals": metrics(originals, band),
            "stress_real_only": metrics(stress, band) if stress else None}


def breakdown(scored, key):
    groups = {}
    for s in scored:
        groups.setdefault((s["label"], s[key]), []).append(s["ai_score"])
    return {f"{label} | {name}": {"n": len(v), "flagged_ai": sum(x > 0.5 for x in v),
                                  "median_ai_score": round(statistics.median(v), 4)}
            for (label, name), v in sorted(groups.items())}


def evaluate(model_id, revision, band=None, splits=None):
    det = AIDetector(model_id, revision, enabled=True)
    started = time.time()
    if not det.load():
        raise RuntimeError(det.error)
    scored = score_images(det, load_rows(splits))
    result = {"model_id": model_id, "revision": revision, "labels": det.info["labels"],
              "seconds": round(time.time() - started, 1), "band": band}
    for split in ("train", "val", "test"):
        rep = split_report(scored, split, band)
        if rep:
            result[split] = rep["all"]
            result[f"{split}_detail"] = rep
    result["by_stratum"] = breakdown(scored, "stratum")
    result["by_variant"] = breakdown([s for s in scored if s["label"] == "real"], "variant")
    result["per_image"] = [{"file": s["file"], "label": s["label"], "split": s["split"], "variant": s["variant"],
                            "stratum": s["stratum"], "ai_score": round(s["ai_score"], 6)} for s in scored]
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL_ID)
    ap.add_argument("--revision", default=DEFAULT_MODEL_REVISION)
    ap.add_argument("--band", type=float, nargs=2, metavar=("LOW", "HIGH"), help="inconclusive band, e.g. 0.35 0.65")
    ap.add_argument("--split", choices=["train", "val", "test"], action="append",
                    help="limit to these splits (repeatable); default: all")
    ap.add_argument("--out")
    args = ap.parse_args()
    if not (EVAL_DIR / "labels.csv").exists():
        print("tests/eval_data/labels.csv missing; run tests/eval_data/build_eval_set.py")
        return 1
    res = evaluate(args.model, args.revision, tuple(args.band) if args.band else None,
                   set(args.split) if args.split else None)
    summary = {k: v for k, v in res.items() if k != "per_image"}
    print(json.dumps(summary, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
