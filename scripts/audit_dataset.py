"""Audit local dataset splits without exporting post text or identifiers.

Usage: python scripts/audit_dataset.py /path/to/Dataset > results/data_audit.json
The three input files remain local and are excluded by .gitignore.
"""
import argparse
from collections import Counter
import hashlib
from itertools import combinations
import json
from pathlib import Path

LABELS = ["anger", "brain dysfunction (forget)", "emptiness", "hopelessness",
          "loneliness", "sadness", "suicide intent", "worthlessness"]


def read_records(path):
    text = path.read_text(encoding="utf-8-sig")
    try:
        records = json.loads(text)
        format_name = "JSON array"
    except json.JSONDecodeError:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
        format_name = "JSON Lines"
    if not isinstance(records, list) or not records or not all(isinstance(r, dict) for r in records):
        raise ValueError("Expected a nonempty JSON array or JSON Lines objects")
    return records, format_name


def audit(directory):
    splits, report = {}, {"source": "User-supplied dataset files; aggregated audit only",
                         "label_order": LABELS, "splits": {}}
    for split in ("train", "val", "test"):
        path = directory / (split + ".json")
        rows, format_name = read_records(path)
        splits[split] = rows
        label_counts = Counter(e for row in rows for e in row.get("emotions", []))
        ids = [r.get("id") for r in rows if r.get("id") is not None]
        texts = [r["text"].strip() for r in rows if isinstance(r.get("text"), str) and r["text"].strip()]
        report["splits"][split] = {
            "file_name": path.name,
            "format": format_name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "records": len(rows),
            "fields": sorted(set().union(*(set(r) for r in rows))),
            "positive_label_assignments": sum(label_counts.values()),
            "per_label_counts": {name: label_counts[name] for name in LABELS},
            "missing_or_blank_text": len(rows) - len(texts),
            "missing_ids": len(rows) - len(ids),
            "duplicate_id_excess_records": len(ids) - len(set(ids)),
            "duplicate_stripped_text_excess_records": len(texts) - len(set(texts)),
            "records_with_unknown_labels": sum(any(e not in LABELS for e in r.get("emotions", [])) for r in rows),
            "label_id_mismatches": sum(str(r.get("label_id", "")).zfill(8) !=
                                       "".join(str(int(e in r.get("emotions", []))) for e in LABELS) for r in rows),
        }
    report["total_records"] = sum(len(rows) for rows in splits.values())
    report["cross_split_overlap"] = {}
    for left, right in combinations(splits, 2):
        id_sets = [{r["id"] for r in splits[key] if r.get("id") is not None} for key in (left, right)]
        text_sets = [{r["text"].strip() for r in splits[key] if isinstance(r.get("text"), str) and r["text"].strip()}
                     for key in (left, right)]
        report["cross_split_overlap"][left + "_vs_" + right] = {
            "shared_ids": len(id_sets[0] & id_sets[1]),
            "shared_unique_stripped_texts": len(text_sets[0] & text_sets[1]),
        }
    report["limits"] = ["Text duplicate checks strip outer whitespace only; semantic duplicates are not checked.",
                         "These file splits are not confirmed as the split used for the dissertation metrics.",
                         "No text, post IDs, dates or per-record labels are exported."]
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.data_directory), indent=2))
