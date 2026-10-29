"""Raw Boolean Cohen kappa point estimates for the ten validated categories.

Selected original binary_cohen_kappa calculation; no composites, sensitivity
recoding, or bootstrap intervals. Public CSV replaces private annotation exports.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

try:
    from .takeaway_evidence import ROOT, boolean
    from .message_behavior_analysis import LABELS
except ImportError:
    from takeaway_evidence import ROOT, boolean
    from message_behavior_analysis import LABELS

THRESHOLD = 0.6


class InterannotatorError(ValueError):
    pass


def validate_threshold(threshold: float) -> None:
    if not math.isfinite(threshold) or not -1.0 <= threshold <= 1.0:
        raise InterannotatorError("threshold must be finite and between -1 and 1")


def validate_boolean_values(values: list[bool], name: str) -> None:
    # Public CSV adapter emits Python bools; no NumPy dependency is needed.
    if any(not isinstance(value, bool) for value in values):
        raise InterannotatorError(f"{name} must contain only Boolean values")


def binary_cohen_kappa(
    rater_a_values: list[bool], rater_b_values: list[bool], *,
    rater_a: str, rater_b: str, threshold: float = THRESHOLD,
) -> dict:
    """Return a transparent binary Cohen-kappa calculation."""
    validate_threshold(threshold)
    if len(rater_a_values) != len(rater_b_values):
        raise InterannotatorError("kappa inputs have different lengths")
    validate_boolean_values(rater_a_values, "rater_a_values")
    validate_boolean_values(rater_b_values, "rater_b_values")
    n = len(rater_a_values)
    if not n:
        return {"n": 0, "rater_a": rater_a, "rater_b": rater_b,
                "a_positive": 0, "b_positive": 0,
                "a_positive_rate": None, "b_positive_rate": None,
                "both_positive": 0, "a_positive_b_negative": 0,
                "a_negative_b_positive": 0, "both_negative": 0,
                "observed_agreement": None, "expected_agreement": None,
                "cohen_kappa": None, "kappa_defined": False,
                "undefined_reason": "no_items", "threshold": threshold,
                "meets_threshold": False, "classification": "undefined"}
    counts = Counter(zip(rater_a_values, rater_b_values))
    both_positive = counts[(True, True)]
    a_positive_b_negative = counts[(True, False)]
    a_negative_b_positive = counts[(False, True)]
    both_negative = counts[(False, False)]
    if both_positive + a_positive_b_negative + a_negative_b_positive + both_negative != n:
        raise InterannotatorError("binary contingency cells do not sum to n")
    a_positive = both_positive + a_positive_b_negative
    b_positive = both_positive + a_negative_b_positive
    a_positive_rate = a_positive / n
    b_positive_rate = b_positive / n
    observed = (both_positive + both_negative) / n
    expected = (a_positive_rate * b_positive_rate
                + (1 - a_positive_rate) * (1 - b_positive_rate))
    denominator = 1 - expected
    if math.isclose(denominator, 0.0, abs_tol=1e-15):
        kappa = None
        undefined_reason = "no_marginal_variation"
        classification = "undefined"
        meets = False
    else:
        kappa = (observed - expected) / denominator
        undefined_reason = None
        meets = kappa >= threshold
        classification = "meets_kappa_threshold" if meets else "below_kappa_threshold"
    return {"n": n, "rater_a": rater_a, "rater_b": rater_b,
            "a_positive": a_positive, "b_positive": b_positive,
            "a_positive_rate": a_positive_rate, "b_positive_rate": b_positive_rate,
            "both_positive": both_positive, "a_positive_b_negative": a_positive_b_negative,
            "a_negative_b_positive": a_negative_b_positive, "both_negative": both_negative,
            "observed_agreement": observed, "expected_agreement": expected,
            "cohen_kappa": kappa, "kappa_defined": kappa is not None,
            "undefined_reason": undefined_reason, "threshold": threshold,
            "meets_threshold": meets, "classification": classification}


def load_validation(path: Path):
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for rater in ("human_1", "human_2", "model"):
            row[rater] = boolean(row[rater])
    return rows


def analyze(rows, expected_items=100):
    by_label = {label: {} for label in LABELS}
    for row in rows:
        label, item = row["label"], row["id"]
        if label not in by_label or not item or item in by_label[label]:
            raise InterannotatorError("unknown label, blank id, or duplicate item/label")
        by_label[label][item] = row
    ids = set(by_label[LABELS[0]])
    if len(ids) != expected_items or any(set(items) != ids for items in by_label.values()):
        raise InterannotatorError(f"require the same {expected_items} items for all ten labels")
    comparisons, retained = {}, []
    for label, items in by_label.items():
        records = [items[item] for item in sorted(ids)]
        comparisons[label] = {}
        for a, b in (("human_1", "human_2"), ("model", "human_1"), ("model", "human_2")):
            comparisons[label][f"{a}_vs_{b}"] = binary_cohen_kappa(
                [r[a] for r in records], [r[b] for r in records], rater_a=a, rater_b=b)
        if all(r["meets_threshold"] for r in comparisons[label].values()):
            retained.append(label)
    return {"items": len(ids), "labels": len(LABELS), "decisions_per_rater": len(rows),
            "threshold": THRESHOLD, "comparisons": comparisons, "retained_labels": retained}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "analysis/inputs/validation_annotations.csv")
    args = parser.parse_args()
    print(json.dumps(analyze(load_validation(args.input)), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()