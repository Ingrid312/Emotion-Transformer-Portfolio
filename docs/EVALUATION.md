# Evaluation and reproducibility

The full thesis model experiment has not been rerun. Source data, all three training notebooks, the original calibrated thresholds and two neural-model evaluation CSVs have now been supplied. The recovered hierarchical preprocessing was executed locally and produced the 3,860 / 965 / 1,207 split and 4,425 positive test label assignments described by the source materials. Original checkpoint bytes and per-example probabilities remain unavailable. The saved CSVs pass internal consistency checks, but historical baseline notebooks use different evaluation splits. See [data verification](DATA.md), [notebook guide](NOTEBOOKS.md) and [artifact verification](ARTIFACTS.md).

## What is verified here

The standard-library tests check hand-computable multi-label metrics, input validation, threshold equality, constrained threshold selection, infeasible constraints, missing validation positives and risk-priority rules. The synthetic demo separates invented validation and test arrays. These checks verify decision logic only.

Recovered preprocessing and the three traditional-model pipelines were also executed on the prepared split. Their actual aggregate outputs are saved separately in `results/recovered_preprocessing.json` and `results/shared_split_traditional_*`. This does not reproduce the historical neural-model scores.

The supplied saved neural reports and thresholds were checked with `scripts/verify_saved_reports.py`. Label order, finite threshold values, per-label F1, inferred counts and macro/micro/weighted averages agree. This is aggregate validation, not independent model evaluation.

## How to reproduce the research results later

1. Run `scripts/prepare_experiment.py` on authorized local source files and record checksums, label order and split statistics. The 906-versus-1,207 discrepancy is explained by source versus cleaned/re-split data. For exact historical split identity, obtain the original split files or indices.
2. Use the three prepared public notebooks on the same split. Their shared-split and validation-selection adaptations constitute a new experiment, not a reproduction of the old baseline scores.
3. Record Python, PyTorch, Transformers and CUDA versions, GPU type and model revision. Train using the recorded settings.
4. Save the best validation checkpoint, validation probabilities and binary targets. Fit thresholds using validation data only.
5. Freeze thresholds; run one held-out test evaluation and export per-label precision, recall and F1, Micro/Macro-F1, Hamming loss and FP totals.
6. Evaluate both the same checkpoint at threshold 0.5 and at calibrated thresholds. Compare the corresponding FP assignment totals. FP is not the number of incorrectly classified posts.
7. Recompute risk stability under threshold changes of ±0.02 and ±0.05. Never present the rule-derived risk categories as independently validated clinical labels.

## Claims requiring care

- The thesis lists a higher overall Flat BERT F1, but recovered baseline logs do not support a controlled comparison on the same held-out split. The Flat BERT table's Micro/Macro pair also comes from different logged epochs.
- The 8.45% FP reduction compares calibration settings of the hierarchical model, not hierarchical versus Flat BERT.
- Micro precision is 0.7601, while Micro-F1 is 0.807; these are different metrics.
- 0.849 is the recall for one label, not overall accuracy or a clinical sensitivity estimate.
- Threshold precision constraints are fitted on validation data; they do not guarantee the same precision in deployment.
- The report does not supply component ablations or repeated-run confidence intervals.
- Under threshold perturbation, reported risk assignments remain unchanged for 95.36%–99.59% of test examples. This measures internal decision stability, not external validity.
