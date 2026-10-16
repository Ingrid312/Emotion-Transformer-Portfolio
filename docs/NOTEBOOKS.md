# Recovered source notebooks

The primary hierarchical model was found in the supplied `test7.ipynb`. It contains cleaning, splitting, model definition, training, threshold calibration, evidence ranking and risk analysis. Public notebooks remove stored outputs, execution counts and source metadata, including private Windows usernames and displayed dataset examples.

## Public files

| File | Source and changes | Current verification |
|---|---|---|
| `scripts/prepare_experiment.py` | Original cleaning and split logic from source cells 4–7 and 10 (one-based); new local input/output wrapper and aggregate report | Executed locally; counts and label prevalence match saved source outputs |
| `notebooks/model.ipynb` | Selected primary-flow source cells 13, 15–27, 33 and 41; added headings/setup and the small fixes listed below | Python syntax checked; training not rerun |
| `notebooks/baseline.ipynb` | Supplied traditional-model code with its input directory changed to the hierarchical model's prepared split | Source code body executed locally; aggregate metrics included |
| `notebooks/standard-bert.ipynb` | Supplied Flat BERT architecture/training, adapted to shared train/validation/test files and validation-only checkpoint selection | Python syntax checked; adapted model not rerun |

The notebook source cell numbers above identify positions in the original upload, not execution counts. Auxiliary experimental cells with alternative splits, manually entered plotting scores, model redefinitions and raw example exports were excluded from the public main-model notebook. The original uploaded files remain unchanged.

Public training fixes: scheduler length now counts the final accumulation group in each epoch; the last partial accumulation group is normalized by its actual size in both neural-model notebooks; single-text hierarchical inference uses the same sentence-position sampling as its dataset loader. These changes apply only to future runs and do not retroactively verify or alter reported thesis scores.

## Prepare and run

Use a separate Python environment and install `requirements-training.txt`. This is a dependency list, not a recovered original version lockfile. Model execution requires access to pretrained model files and appropriate compute. No model files are downloaded or trained merely by opening a notebook.

Place the locally authorized `train.json`, `val.json` and `test.json` under `Dataset`, then run from the repository root:

```bash
python scripts/prepare_experiment.py Dataset
jupyter lab
```

Open `notebooks/model.ipynb` in a fresh kernel and run in order. The model notebook reads the prepared split rather than repeating every exploratory cleaning cell. It trains, saves `outputs_hier_strategy4/best.pt`, calibrates validation thresholds, evaluates test predictions and generates research risk/evidence outputs. These are source-code capabilities, not newly reproduced results.

The two adapted baseline notebooks read the same prepared split. Traditional baseline code has been executed locally. Flat BERT selects its checkpoint using the separate validation set and then evaluates test once, but has not been rerun. Fresh neural-model training is needed before claiming a controlled comparison with the new traditional scores. Local outputs and data stay excluded by `.gitignore`.

## What the recovered preprocessing explains

The source's cleaning removes five duplicate **cleaned-text plus label-vector** records, leaving 6,032 rows. With seed 123, it holds out 20% for test, then holds out 20% of the remainder for validation. The resulting proportions are approximately **64% / 16% / 20%**, yielding 3,860 / 965 / 1,207 posts. The stratification key combines Suicide Intent presence with positive-label count capped at three.

Concatenating the supplied files in train/val/test order and applying this source logic reproduced those counts, all per-label split counts, and 4,425 positive test label assignments. This supports the data-preparation account; original split identifiers and trained predictions were not supplied, so byte-for-byte historical split identity and performance reproduction are not claimed.

## Historical baseline records

The original traditional-model notebook's stored output uses 4,814 training posts and 605 test posts. The original Flat BERT output uses 5,433 training posts and a 604-post set named `test`. It evaluates that set each epoch and selects its checkpoint on the same set, so that set acts as development data rather than an independent final test set.

The Flat BERT log's best Micro-F1 is 0.8343036 at epoch 6, where logged Macro-F1 is 0.7994. Epoch 8 has Micro-F1 0.8341 and Macro-F1 0.8020. Thus the thesis table's rounded 0.834 / 0.802 pair is not supported as the same best-Micro-F1 checkpoint by this saved log. Historical numbers are retained as reported evidence, not verified as a fair ranking. The adapted notebooks deliberately do not reuse those numbers as their own results.

## Remaining source limitations

- The supplied hierarchical notebook has saved preprocessing output but no stored output for its main training/calibration cells. Later plots contain manually entered scores. Those plots are not independent execution evidence.
- For posts exceeding eight sentences, the original `HierDataset` samples eight positions across the post while the original single-text evidence function keeps the first eight. The public notebook aligns the evidence function to the dataset's sampling; historical predictions are not regenerated by this change.
- `BD_PHRASES` is declared but not used by `context_feats`; the actual cognitive feature checks `BD_TOKENS`. The quote detector also treats apostrophes as quotation cues. Feature behavior needs focused evaluation.
- The source's `AsymmetricLoss` is asymmetric focal-weighted BCE with focusing parameters 0 and 4; it is not a full implementation of every feature of the cited ASL algorithm, such as negative probability clipping.
- One duplicated cleaned text with different label vectors remains after source deduplication. The locally recovered split has zero exact cleaned-text overlap across train/validation/test; semantic duplicates have not been assessed.
- Original weights, per-example prediction arrays and the Flask application files remain unavailable. The subsequently supplied calibrated thresholds and two aggregate neural evaluation reports are now included; see `ARTIFACTS.md`.

Aggregate historical evidence is in `results/notebook_audit.json`. Locally executed preprocessing statistics are in `results/recovered_preprocessing.json`.

The new traditional run has Micro/Macro-F1 of 0.7261/0.6790 for SVM, 0.7308/0.6959 for Logistic Regression and 0.6697/0.4557 for Naive Bayes. Full aggregate metrics and environment versions are in `results/shared_split_traditional_metrics.csv` and `results/shared_split_traditional_run.json`. The source code body was executed with the local prepared-data path; stored notebook outputs remain cleared for privacy.
