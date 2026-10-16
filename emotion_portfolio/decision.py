"""Validation-based threshold calibration, reconstructed from thesis §3.5.

New portfolio implementation. No fitted parameters or thesis predictions included.
"""
import math

LABELS = (
    "anger", "brain dysfunction (forget)", "emptiness", "hopelessness",
    "loneliness", "sadness", "suicide intent", "worthlessness",
)
SUICIDE_INDEX = LABELS.index("suicide intent")
CORE_INDICES = tuple(LABELS.index(x) for x in
                     ("hopelessness", "worthlessness", "sadness"))


def validate_rows(rows, *, binary=False, width=8):
    """Reject empty, ragged, nonfinite and out-of-range inputs."""
    rows = [list(row) for row in rows]
    if not rows:
        raise ValueError("At least one row is required")
    for row in rows:
        if len(row) != width:
            raise ValueError(f"Each row must contain {width} entries")
        for value in row:
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("Values must be finite numbers")
            if binary and value not in (0, 1):
                raise ValueError("Targets must be binary")
            if not 0 <= value <= 1:
                raise ValueError("Values must be in [0, 1]")
    return rows


def validate_thresholds(thresholds):
    return validate_rows([thresholds])[0]


def binary_stats(targets, predictions):
    tp = sum(y == 1 and p == 1 for y, p in zip(targets, predictions))
    fp = sum(y == 0 and p == 1 for y, p in zip(targets, predictions))
    fn = sum(y == 1 and p == 0 for y, p in zip(targets, predictions))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1}


def calibrate(targets, probabilities, precision_floor=0.70):
    """Fit on validation data only; never tune on test predictions.

    Most labels maximize F1. Suicide Intent maximizes recall under the
    precision floor, then F1. If no feasible point exists, use max F1 and
    flag fallback. Exact ties use the lowest grid threshold deterministically.
    Labels without positive validation examples keep 0.5 and are flagged.
    """
    targets = validate_rows(targets, binary=True)
    probabilities = validate_rows(probabilities)
    if len(targets) != len(probabilities):
        raise ValueError("Targets and probabilities must have equal row counts")
    if not math.isfinite(precision_floor) or not 0 <= precision_floor <= 1:
        raise ValueError("precision_floor must be in [0, 1]")
    thresholds, details = [], []
    for j, name in enumerate(LABELS):
        ys = [row[j] for row in targets]
        ps = [row[j] for row in probabilities]
        if not any(ys):
            thresholds.append(0.5)
            details.append({"label": name, "status": "no_validation_positives",
                            "threshold": 0.5})
            continue
        candidates = []
        for n in range(5, 96):
            threshold = n / 100
            stats = binary_stats(ys, [int(p >= threshold) for p in ps])
            candidates.append({"threshold": threshold, **stats})
        status = "max_f1"
        if j == SUICIDE_INDEX:
            feasible = [c for c in candidates if c["precision"] >= precision_floor]
            if feasible:
                best = max(feasible, key=lambda c: (c["recall"], c["f1"], -c["threshold"]))
                status = "precision_constrained_max_recall"
            else:
                best = max(candidates, key=lambda c: (c["f1"], -c["threshold"]))
                status = "precision_floor_unmet_fallback"
        else:
            best = max(candidates, key=lambda c: (c["f1"], -c["threshold"]))
        thresholds.append(best["threshold"])
        details.append({"label": name, "status": status, **best})
    return thresholds, details


def apply_thresholds(probabilities, thresholds):
    rows = validate_rows(probabilities)
    thresholds = validate_thresholds(thresholds)
    return [[int(p >= t) for p, t in zip(row, thresholds)] for row in rows]


def risk_levels(probabilities, thresholds):
    """Thesis priority cascade: Suicide Intent > three core labels > Green.

    Research category names only. Green does not mean clinically safe.
    """
    predictions = apply_thresholds(probabilities, thresholds)
    return ["Red" if row[SUICIDE_INDEX] else
            "Amber" if any(row[j] for j in CORE_INDICES) else "Green"
            for row in predictions]
