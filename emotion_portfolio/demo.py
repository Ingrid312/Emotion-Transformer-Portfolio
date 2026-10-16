"""Invented numeric demonstration. Does not run the thesis model."""
import json
from .decision import calibrate, apply_thresholds, risk_levels
from .metrics import evaluate


def main():
    # Fixed synthetic fixtures, not Reddit texts or original prediction arrays.
    validation_y = [[int((i + j) % 3 == 0) for j in range(8)] for i in range(12)]
    validation_p = [[0.58 if y else 0.22 for y in row] for row in validation_y]
    thresholds, details = calibrate(validation_y, validation_p)
    test_y = [[1, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 1, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 1, 0]]
    test_p = [[0.8 if y else 0.1 for y in row] for row in test_y]
    print("SYNTHETIC NUMERIC DEMO — not thesis results or model inference")
    print(json.dumps({"thresholds": thresholds,
                      "calibration": details,
                      "held_out_synthetic_metrics": evaluate(test_y, apply_thresholds(test_p, thresholds)),
                      "research_categories": risk_levels(test_p, thresholds)}, indent=2))


if __name__ == "__main__":
    main()
