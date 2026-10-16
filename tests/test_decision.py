import math
import unittest
from emotion_portfolio.decision import calibrate, apply_thresholds, risk_levels
from emotion_portfolio.metrics import evaluate


def expand(values):
    return [[v] * 8 for v in values]


class DecisionTests(unittest.TestCase):
    def test_precision_floor_selects_recall_under_constraint(self):
        ys = expand([1, 1, 0, 0])
        ps = expand([0.9, 0.6, 0.7, 0.1])
        thresholds, details = calibrate(ys, ps)
        self.assertEqual(thresholds[6], 0.71)
        self.assertEqual(details[6]["recall"], 0.5)
        self.assertGreaterEqual(details[6]["precision"], 0.7)
        self.assertAlmostEqual(thresholds[0], 0.11)

    def test_infeasible_precision_is_explicit(self):
        _, details = calibrate(expand([1, 0]), expand([0.2, 0.9]))
        self.assertEqual(details[6]["status"], "precision_floor_unmet_fallback")

    def test_no_positive_validation_label(self):
        thresholds, details = calibrate(expand([0, 0]), expand([0.2, 0.8]))
        self.assertEqual(thresholds, [0.5] * 8)
        self.assertTrue(all(d["status"] == "no_validation_positives" for d in details))

    def test_priority_and_exact_threshold(self):
        self.assertEqual(risk_levels([[0.5] * 8], [0.5] * 8), ["Red"])
        anger_only = [0.9, 0, 0, 0, 0, 0, 0, 0]
        core_only = [0, 0, 0, 0.9, 0, 0, 0, 0]
        self.assertEqual(risk_levels([anger_only, core_only], [0.5] * 8), ["Green", "Amber"])

    def test_known_metrics(self):
        y = [[1, 1, 0, 0, 0, 0, 0, 0]]
        p = [[1, 0, 1, 0, 0, 0, 0, 0]]
        result = evaluate(y, p)
        self.assertEqual(result["micro_f1"], 0.5)
        self.assertEqual(result["false_positive_assignments"], 1)
        self.assertEqual(result["hamming_loss"], 0.25)

    def test_rejects_invalid_input(self):
        for rows in ([], [[0.2]], [[math.nan] * 8], [[1.2] * 8]):
            with self.assertRaises(ValueError):
                apply_thresholds(rows, [0.5] * 8)
        with self.assertRaises(ValueError):
            calibrate(expand([1]), expand([0.4, 0.8]))
        with self.assertRaises(ValueError):
            apply_thresholds(expand([0.4]), [math.nan] * 8)


if __name__ == "__main__":
    unittest.main()
