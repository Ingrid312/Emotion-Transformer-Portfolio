# Context-Gated Hierarchical Transformer for Emotion Detection

**NLP Research Portfolio | Multi-label Emotion Detection**

基于预训练 RoBERTa 的八标签情绪识别毕业设计。研究重点是句子级证据、上下文特征和决策阈值，而非生成式问答。

This research portfolio presents my dissertation on multi-label emotion detection. A RoBERTa sentence encoder, context gating and a document-level Transformer produce eight emotion scores and sentence-level evidence.

![Saved evaluation report scores and reported calibration counts](assets/reported_results.png)

*Scores are taken from the supplied evaluation CSVs. The two neural models used different evaluation populations, so the figure is not a controlled model ranking. The pre-calibration count of 1,313 is thesis-reported; the calibrated count of 1,202 is consistent with the supplied hierarchical CSV.*

## Prototype screenshots from the dissertation

The images below are extracted directly from the submitted dissertation. They document the original **MindSight research prototype**; they are not screenshots of the numeric demo in this repository. The original application source and trained checkpoint are not included in this release.

### System interface

<img src="assets/thesis_system_interface.png" alt="MindSight prototype showing text input, the assessment result and label-specific evidence" width="520">

*Source: dissertation p. 62, Figure 4.6. The page includes text input, the evidence-count control, an assessment panel and evidence sentences. The Green banner at the top belongs to the introductory risk-level carousel; the example's prediction is the Red badge in the assessment panel.*

### Prediction and evidence sentences

![MindSight Amber assessment and label-specific evidence sentences](assets/thesis_prediction_evidence.png)

*Source: dissertation p. 63, Figure 4.7. This example shows an Amber research category and evidence for Loneliness and Emptiness. The original screenshot includes short example-text excerpts; no source usernames or post identifiers are shown. The displayed “Conf” values are reproduced as shown in the thesis, without an independent calibration claim.*

### Model architecture

![Original model architecture showing RoBERTa, context gating, rule prior, hierarchical encoding and threshold calibration](assets/thesis_model_architecture.png)

*Source: dissertation p. 30, Figure 3.1. The diagram links sentence encoding and seven-dimensional linguistic features to context gating, hierarchical aggregation, label-wise attention and calibrated research categories.*

## What the project does

- Cleans and splits English social-media posts into sentence sequences.
- Encodes up to **8 sentences × 64 tokens** using RoBERTa-base.
- Uses **7 contextual features**: negation, quotation, third-person pronouns, cognitive cues, memory cues, polarity and subjectivity.
- Aggregates sentence representations using a **2-layer, 8-head Transformer**, then uses label-wise attention for prediction and evidence ranking.
- Calibrates one decision threshold per label using the validation set.
- Maps predictions to Red / Amber / Green research screening categories.

## Verified source data

The supplied source files contain **4,225 training, 906 validation and 906 test posts** (6,037 total). They use JSON Lines despite their `.json` filenames. A local audit found no blank text, unknown labels or `label_id` inconsistencies. Training includes four excess duplicate-text records, and two unique texts occur in both training and test, despite distinct post IDs.

The recovered hierarchical notebook explains the experiment split: cleaning retains **6,032 posts**, then seed-123 stratified splitting yields **3,860 train / 965 validation / 1,207 test** posts. Running that preprocessing locally on the supplied files reproduces these counts, label prevalence and **4,425 positive test label assignments**. The raw 906-post test file and the prepared 1,207-post test split serve different stages. Model training and prediction scores have not been rerun. See [data verification](docs/DATA.md).

## Saved evaluation reports

The original hierarchical and Flat BERT evaluation CSVs and eight-label threshold file are now included. Label order, per-label metrics and aggregate summaries have been checked for internal consistency. These are **saved experimental reports, not a new neural-model reproduction**: trained checkpoint bytes and per-example probabilities were not supplied.

| Saved report | Micro-F1 | Macro-F1 | Evaluation posts from notebook |
|---|---:|---:|---:|
| Hierarchical Transformer | 0.8073 | 0.7725 | 1,207 |
| Flat BERT | 0.8343 | 0.7994 | 604, used for epoch selection |

The hierarchical report's positive-label support is 4,425 and matches the recovered test split for every label. Its Suicide Intent precision/recall/F1 are **0.7507 / 0.8490 / 0.7969**. The Flat BERT report has 2,260 positive label assignments and agrees with its best-Micro-F1 logged epoch. Because the evaluation sets differ and Flat BERT selected epochs on its 604-post set, these numbers do not establish a controlled performance difference. Public baseline notebooks use the shared prepared split for future runs.

The README uses the saved CSV values: the dissertation's Macro-F1 values of 0.773 (hierarchical) and 0.802 (Flat BERT) differ from those saved reports. The original thesis table remains recorded separately in `results/thesis_reported.json`; the Flat BERT value 0.802 appears at a different logged epoch. See [artifact verification](docs/ARTIFACTS.md).

### Locally rerun traditional baselines

The adapted traditional baseline code was executed on **3,860 prepared training posts and 1,207 test posts**. These are new results, kept separate from the historical thesis table:

| Model | Micro-F1 | Macro-F1 |
|---|---:|---:|
| TF-IDF + Naive Bayes | 0.6697 | 0.4557 |
| TF-IDF + Logistic Regression | 0.7308 | 0.6959 |
| TF-IDF + SVM | 0.7261 | 0.6790 |

Detailed aggregate metrics and runtime versions are included in `results/shared_split_traditional_metrics.csv` and `results/shared_split_traditional_run.json`. The neural models still require fresh training before any controlled comparison with these new results.

| Decision metric | Default threshold 0.5 | Calibrated thresholds |
|---|---:|---:|
| False-positive label assignments | 1,313 | 1,202 |
| Micro precision | 0.7450 | 0.7601 |
| Macro precision | 0.7109 | 0.7284 |
| Hamming loss | 0.1970 | 0.1883 |

The thesis reports an **8.45%** decrease in false-positive label assignments. The calibrated CSV independently supports the aggregate count **1,202**, inferred from its precision, recall and support; the pre-calibration 1,313 count still comes from the thesis. This is not a count of misclassified posts, and no raw prediction arrays were supplied for independent recomputation.

## Repository status

| Material | Status |
|---|---|
| Method and results | Summarized from the supplied dissertation and defense slides |
| Source data audit | Verified local split counts, label consistency and text duplication; aggregate report included |
| Recovered preprocessing | Executed locally: 6,032 cleaned posts, split 3,860 / 965 / 1,207 |
| Prototype screenshots and architecture figure | Extracted from dissertation pp. 30, 62 and 63; historical project evidence |
| Calibration, metric and risk-mapping utilities | Newly written for this portfolio; tested on synthetic fixtures |
| Synthetic demonstration | Runs with the Python standard library; no model or dataset required |
| Hierarchical training source | Recovered from the supplied core notebook; primary flow included as `notebooks/model.ipynb` |
| Traditional baseline notebook | Included with shared-split adaptation; its code body was executed locally and aggregate metrics are included |
| Flat BERT notebook | Included with shared-split and validation-selection adaptations; neural training has not been rerun |
| Original evaluation reports | Included as aggregate CSVs; internal consistency checked |
| Original calibrated thresholds | Included in `artifacts/thresholds.json`; exact float values retained |
| Model weights and original predictions | Not included; checkpoint pairing and neural predictions not independently verified |
| Live model service | Not included |

The standalone decision utilities are newly written for this portfolio, while the training notebooks are recovered from the subsequently supplied project notebooks. The initially supplied `bert.py`, `analysis.py` and dataset README are upstream reference materials and are not presented as the hierarchical model's source. See [provenance](docs/PROVENANCE.md).

Saved thresholds belong to the original experiment. A fresh training run must generate and use its own validation-calibrated thresholds; do not apply this archived threshold file to arbitrary new weights.

## Run the recovered training workflow

Keep the authorized source files locally under `Dataset`, then use a separate environment:

```bash
python -m pip install -r requirements-training.txt
python scripts/prepare_experiment.py Dataset
jupyter lab
```

Open `notebooks/model.ipynb` in a fresh kernel and run in order. The two baseline notebooks read the same prepared files; Flat BERT selects checkpoints on validation and evaluates test afterward. The dependency list is not the original environment lockfile. Preprocessing and traditional baseline code were executed; notebook source was checked statically. Neural training/inference and historical scores are not newly verified. See [notebook guide](docs/NOTEBOOKS.md).

## Run the verified demonstration

Python 3.10 or newer; no third-party packages required:

```bash
python -m emotion_portfolio.demo
python -m unittest discover -s tests -v
```

The demo uses **invented numeric probabilities and labels**, calibrates thresholds, and prints metrics and risk categories. It does not accept user text or perform AI inference. Its output is unrelated to the reported thesis scores.

To use the calibration code with real predictions, provide validation-only labels/probabilities, freeze the resulting thresholds, then evaluate on a held-out test set. The label order is fixed in `emotion_portfolio/decision.py`. See [method](docs/METHOD.md) and [evaluation](docs/EVALUATION.md).

## Files

- `emotion_portfolio/decision.py`: input checks, threshold selection and risk mapping.
- `emotion_portfolio/metrics.py`: multi-label evaluation without external dependencies.
- `emotion_portfolio/demo.py`: synthetic numeric demonstration.
- `results/thesis_reported.json`: source-labeled thesis results.
- `results/data_audit.json`: aggregate audit of the newly supplied source data.
- `scripts/audit_dataset.py`: repeatable local JSON / JSON Lines data audit.
- `scripts/prepare_experiment.py`: recovered text cleaning and experiment split.
- `notebooks/`: hierarchical, traditional and Flat BERT training workflows.
- `results/recovered_preprocessing.json`: locally executed preprocessing statistics.
- `results/notebook_audit.json`: aggregate observations from original notebook logs.
- `results/shared_split_traditional_*`: newly executed traditional baseline metrics and runtime information.
- `results/original_*_performance.csv`: original saved neural-model evaluation reports.
- `artifacts/thresholds.json`: original eight-label threshold file, without model weights.
- `scripts/verify_saved_reports.py`: repeatable checks of saved aggregate artifacts.
- `assets/reported_results.png`: plot of reported results.
- `assets/thesis_*.png`: original prototype screenshots and architecture figure extracted from the dissertation.
- `docs/`: method, provenance, evaluation limits and model card.
- `tests/`: calibration edge cases, metric checks and priority rules.

## Data and intended use

DepressionEmo contains **6,037 Reddit posts** and eight emotion labels. The dataset is attributed to Rahman et al. (2024), not collected or annotated by me. No dataset files or trained weights are redistributed here. The original thesis screenshots contain limited example-text excerpts; they are included as project evidence, not as a dataset release.

The upstream repository requires prior written permission to use the dataset in a web application, mobile app or commercial tool. This release is a research portfolio and code demonstration. Its screening categories are not a clinical assessment. Attention rankings indicate model focus; they do not prove causal explanations.

## References

- Rahman et al. (2024), *DepressionEmo: A novel dataset for multilabel classification of depression emotions*. [Journal DOI](https://doi.org/10.1016/j.jad.2024.08.013), [preprint](https://arxiv.org/abs/2401.04655), [dataset and terms](https://github.com/abuBakarSiddiqurRahman/DepressionEmo).
- [RoBERTa-base model card](https://huggingface.co/FacebookAI/roberta-base).
- Project evidence: *Depression-Related Emotion Multi-Label Detection on Long Social Media Posts using Transformer*, supplied dissertation and defense slides. Personal identifiers and original filenames are omitted from this public portfolio. Source locations are listed in `docs/PROVENANCE.md`.

Dataset, pretrained models and third-party materials retain their own terms. This release does not grant a license to those materials.
