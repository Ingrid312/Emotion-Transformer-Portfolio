"""Check saved aggregate CSV reports and thresholds without loading weights.

Checks internal metric consistency; it does not rerun or independently validate
the underlying predictions. Inferred counts use reported precision/recall.
"""
import csv
import hashlib
import json
import math
from pathlib import Path

LABELS = ["anger", "brain dysfunction (forget)", "emptiness", "hopelessness",
          "loneliness", "sadness", "suicide intent", "worthlessness"]
ROOT = Path(__file__).resolve().parents[1]


def load_report(path, flat=False):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if flat:
        return {r["Label/Metric Name"]: {"precision": float(r["Precision"]),
                   "recall":float(r["Recall"]), "f1-score":float(r["F1-Score"]),
                   "support":float(r["Support"])} for r in rows}
    return {r[""]: {key:float(r[key]) for key in ("precision","recall","f1-score","support")} for r in rows}


def check_report(rows):
    assert all(name in rows for name in LABELS)
    per_label, tp_total, fp_total, fn_total = [], 0, 0, 0
    for name in LABELS:
        r = rows[name]
        p, recall, f1, support = (r[k] for k in ("precision","recall","f1-score","support"))
        assert 0 <= p <= 1 and 0 <= recall <= 1 and 0 <= f1 <= 1
        assert p > 0, "Cannot uniquely infer false positives from a zero-precision aggregate"
        assert support > 0 and support.is_integer()
        tp_float = recall * support
        tp = round(tp_float)
        assert abs(tp_float-tp) < 1e-6
        fp_float = tp/p-tp if p else 0
        fp, fn = round(fp_float), int(support)-tp
        assert abs(fp_float-fp) < 1e-6 and fp >= 0 and fn >= 0
        f1_from_counts = 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0
        assert math.isclose(f1, f1_from_counts, abs_tol=1e-10)
        per_label.append({"label":name, **r, "inferred_tp":tp,"inferred_fp":fp,"inferred_fn":fn})
        tp_total += tp; fp_total += fp; fn_total += fn
    micro = rows["micro avg"]
    expected = {"precision":tp_total/(tp_total+fp_total), "recall":tp_total/(tp_total+fn_total),
                "f1-score":2*tp_total/(2*tp_total+fp_total+fn_total),
                "support":float(tp_total+fn_total)}
    for key,value in expected.items():
        assert math.isclose(micro[key],value,abs_tol=1e-10),(key,micro[key],value)
    for key in ("precision","recall","f1-score"):
        assert math.isclose(rows["macro avg"][key],sum(r[key] for r in per_label)/8,abs_tol=1e-10)
        if "weighted avg" in rows:
            assert math.isclose(rows["weighted avg"][key],
                                sum(r[key]*r["support"] for r in per_label)/(tp_total+fn_total),abs_tol=1e-10)
    return {"micro":micro,"macro":rows["macro avg"],"positive_label_assignments":tp_total+fn_total,
            "inferred_tp":tp_total,"inferred_fp":fp_total,"inferred_fn":fn_total,"per_label":per_label}


def main():
    paths = {"hierarchical": ROOT/"results/original_hierarchical_performance.csv",
             "flat_bert": ROOT/"results/original_flat_bert_performance.csv"}
    checked = {key:check_report(load_report(path,flat=(key=="flat_bert"))) for key,path in paths.items()}
    thresholds_path = ROOT/"artifacts/thresholds.json"
    thresholds = json.loads(thresholds_path.read_text())
    assert thresholds["labels"] == LABELS
    assert len(thresholds["thresholds"]) == 8
    assert all(math.isfinite(x) and 0 <= x <= 1 for x in thresholds["thresholds"])
    prepared = json.loads((ROOT/"results/recovered_preprocessing.json").read_text())
    assert checked["hierarchical"]["positive_label_assignments"]==prepared["splits"]["test"]["positive_label_assignments"]
    assert [r["support"] for r in checked["hierarchical"]["per_label"]] == [
        prepared["splits"]["test"]["per_label_counts"][name] for name in LABELS]
    original_test_count = prepared["splits"]["test"]["records"]
    checked["hierarchical"]["evaluation_posts_from_recovered_notebook"] = original_test_count
    checked["hierarchical"]["hamming_loss_inferred_using_notebook_sample_count"] = (
        checked["hierarchical"]["inferred_fp"]+checked["hierarchical"]["inferred_fn"])/(original_test_count*8)
    checked["flat_bert"]["evaluation_posts_from_saved_notebook_log"] = 604
    checked["flat_bert"]["evaluation_role"] = "Development set named test, used for epoch selection"
    result = {"status":"saved_artifact_internal_consistency_checked_not_neural_inference",
              "reports":checked,
              "thresholds":{"file":"artifacts/thresholds.json","label_order_matches":True,
                            "values":thresholds["thresholds"],"checkpoint_pairing_verified":False},
              "source_sha256":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in [*paths.values(),thresholds_path]},
              "limits":["No checkpoint was loaded and no neural predictions were rerun.",
                        "Aggregate consistency does not independently establish prediction correctness.",
                        "The two historical reports use different evaluation populations.",
                        "The 1,313 pre-calibration FP count is thesis-reported, not independently supplied as predictions."]}
    out = ROOT/"results/saved_artifact_verification.json"
    out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({key:{"micro_f1":r["micro"]["f1-score"],"macro_f1":r["macro"]["f1-score"],
                           "inferred_fp":r["inferred_fp"]} for key,r in checked.items()},indent=2))
    print("Eight thresholds and both report summaries passed internal consistency checks.")


if __name__ == "__main__":
    main()
