# Dataset verification

The supplied DepressionEmo files were inspected locally. Only aggregate counts and file checksums are included in this repository; the three dataset files remain excluded from the public package.

| File | Actual format | Posts | Positive label assignments |
|---|---|---:|---:|
| `train.json` | JSON Lines | 4,225 | 15,378 |
| `val.json` | JSON Lines | 906 | 3,355 |
| `test.json` | JSON Lines | 906 | 3,274 |
| Total | | 6,037 | 22,007 |

The split is approximately 70% / 15% / 15%. Despite the `.json` suffix, each nonempty line is an independent JSON object; loading the entire file with `json.load()` will fail. The audit script handles both JSON arrays and JSON Lines.

Each record has `id`, `title`, `post`, `text`, `upvotes`, `date`, `emotions` and `label_id`. Across the supplied files, all eight label names are recognized, no text is missing or blank, and the zero-padded eight-digit `label_id` agrees with `emotions` for every record.

## Duplicate checks

There are no duplicate post IDs within or across splits. However, training has **four excess records with duplicate text**, and **two unique text strings occur in both training and test**. Text comparison strips only outer whitespace; this is not a semantic duplicate check. ID uniqueness alone therefore does not exclude text leakage. No data has been deleted or repartitioned by the audit.

## Difference from the dissertation experiment

The supplied test file has **906 posts and 3,274 positive label assignments**. The dissertation/defense materials report test risk-category counts summing to **1,207 posts** (45 + 825 + 337), and **4,425 positive test label assignments**. These do not match the supplied default split.

The subsequently supplied `test7.ipynb` confirms the cleaning and re-splitting procedure. It removes five duplicate cleaned-text plus label-vector records, retaining 6,032 posts. It holds out 20% for test and then 20% of the remainder for validation, using seed 123 and a stratification key combining Suicide Intent presence with capped positive-label count.

| Prepared split | Posts | Positive label assignments |
|---|---:|---:|
| Train | 3,860 | 14,087 |
| Validation | 965 | 3,479 |
| Test | 1,207 | 4,425 |

The recovered preprocessing was executed on the supplied files concatenated in train/val/test order. Its counts and per-label prevalence match the saved notebook output; the test assignment total matches the dissertation. Original historical split identifiers were not supplied, so exact row-for-row historical identity is not independently established. Neural-model performance has not been rerun; new traditional-model results are recorded separately.

Source deduplication retains identical cleaned texts when their label vectors differ. One such excess record remains; in the locally recovered split it does not produce any exact cleaned-text overlap across train/validation/test. Raw-split overlap findings above should not be attributed to this newly prepared split.

See `results/recovered_preprocessing.json` and [notebook guide](NOTEBOOKS.md). Do not describe the original 906-post source test file as the test split for the thesis's 1,207-post experiment.

## Repeat the local audit

Place the authorized files in a local `Dataset` directory, then run from the repository root:

```bash
python scripts/audit_dataset.py Dataset > results/data_audit.json
```

The report contains counts and SHA-256 checksums, with no raw text or post identifiers. It does not train a model or reproduce the dissertation evaluation. Detailed output is in [data_audit.json](../results/data_audit.json).
