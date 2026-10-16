"""Multi-label metrics for binary targets and fixed-threshold predictions."""
from .decision import binary_stats, validate_rows


def evaluate(targets, predictions):
    targets = validate_rows(targets, binary=True)
    predictions = validate_rows(predictions, binary=True)
    if len(targets) != len(predictions):
        raise ValueError("Targets and predictions must have equal row counts")
    per_label = [binary_stats([r[j] for r in targets], [r[j] for r in predictions])
                 for j in range(8)]
    overall = binary_stats([x for r in targets for x in r],
                           [x for r in predictions for x in r])
    errors = sum(y != p for ys, ps in zip(targets, predictions) for y, p in zip(ys, ps))
    return {"micro_f1": overall["f1"], "micro_precision": overall["precision"],
            "micro_recall": overall["recall"],
            "macro_f1": sum(s["f1"] for s in per_label) / 8,
            "false_positive_assignments": overall["fp"],
            "hamming_loss": errors / (len(targets) * 8), "per_label": per_label}
