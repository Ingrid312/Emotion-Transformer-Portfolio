# Provenance

## Primary project evidence

The project evidence consists of a dissertation (89 pages) and defense slides (19 pages). Personal names, student identifiers and original filenames are omitted from this public portfolio. References below identify sources by document type and page number.

| Claim | Dissertation page | Defense slide PDF page |
|---|---:|---:|
| Cleaning and seven context features | 31–33 | 10 |
| Encoder, gating, rule prior and label attention | 33–37 | 8–10 |
| Training settings | 38–39, 50 | — |
| Threshold calibration and cascade | 39–42 | 11 |
| Comparative F1 scores | 52–53 | 12 |
| Calibration and false positives | 54–55 | 15 |
| Per-label recall and precision | 56–57 | 14 |
| Risk stability | 60–61 | 16 |
| Interface description | 61–64 | — |
| Core code screenshots | 77–89 | — |

The thesis describes a Flask interface and `best.pt` / `thresholds.json` loading. The application source and checkpoint were not attached. No live service is recreated or claimed in this release.

The original eight-label `thresholds.json` and two neural evaluation CSVs were subsequently supplied and are included under `artifacts` and `results`. Their numeric contents are unchanged. Internal consistency checks and source hashes are documented in `ARTIFACTS.md`. A private checkpoint screenshot was inspected but is not redistributed; no model weights were loaded.

## Extracted figures

These three embedded PDF images are exported at their original pixel dimensions and included in the README. Their visible contents have not been redrawn or changed.

| Asset | Source | Original pixels |
|---|---|---:|
| `assets/thesis_model_architecture.png` | Dissertation p. 30, Figure 3.1 | 1816 × 1275 |
| `assets/thesis_system_interface.png` | Dissertation p. 62, Figure 4.6 | 871 × 1363 |
| `assets/thesis_prediction_evidence.png` | Dissertation p. 63, Figure 4.7 | 1122 × 872 |

The screenshots document the historical MindSight prototype. They do not demonstrate that the application is runnable from this release. Short example-text excerpts remain visible in the original screenshots, without usernames or post identifiers. No standalone dataset records are provided. Screenshot “Conf” values are not independently validated calibration results. In Figure 4.6, the introductory Green carousel banner and the Red prediction badge serve different interface roles.

## Upstream materials

The three source data files (`train.json`, `val.json`, `test.json`) and project notebooks (`test7.ipynb`, `baseline.ipynb`, `standard-bert.ipynb`) were subsequently supplied. The public package includes sanitized, selected/adapted notebook code and aggregate statistics, not raw data records or notebook outputs. The recovered preprocessing explains the hierarchical test count. Notebook source selection and baseline adaptations are documented in `docs/NOTEBOOKS.md`.

The attached README describes the DepressionEmo authors' dataset and baselines; its attribution and restrictions are retained through links in this repository. The attached `bert.py` implements label-combination classification with CrossEntropyLoss; it is not the thesis's independent eight-sigmoid hierarchical model. `file_io.py` is also missing from those attachments. Publishing these as the complete original graduation project would be misleading.

For that reason, upstream scripts and standalone dataset records are not copied into this portfolio. Limited example-text excerpts remain visible only within the original thesis screenshots described above. Consult the [original repository](https://github.com/abuBakarSiddiqurRahman/DepressionEmo).

## Newly prepared portfolio materials

README, method summary, evaluation notes, model card, numeric utilities, tests and the results chart were prepared with AI assistance for this portfolio. Utilities follow the described decision procedure but are newly written; they are not recovered original code. Numerical research results are transcribed from the supplied thesis. The full PDFs are not included in the public package because a concise summary is sufficient and avoids redistributing source text and identifiers.
