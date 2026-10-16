# Model card

**Task:** English multi-label emotion classification over eight DepressionEmo labels.

**Model family:** RoBERTa-base sentence encoder with context gating, hierarchical Transformer aggregation and label-wise attention. This is an encoder-based pretrained-language-model project, not a generative LLM, chatbot, RAG system or agent.

**Portfolio release:** Recovered training notebooks, executable preprocessing, original evaluation CSVs and thresholds, documentation and standalone decision utilities. No trained model is distributed; text inference requires model training or a compatible original checkpoint. The standard-library demo remains synthetic. Saved thresholds are archived original parameters, not defaults for new training runs.

**Source and prepared data:** DepressionEmo by Rahman et al., 6,037 Reddit posts. Raw source files have 4,225 / 906 / 906 train/validation/test posts. Recovered hierarchical preprocessing removes five duplicate cleaned-text plus label-vector rows and splits 6,032 posts into 3,860 / 965 / 1,207, with 4,425 positive test label assignments. These aggregate results were reproduced locally and match saved notebook statistics. Exact historical split identity and trained performance are not independently reproduced. Raw source files have text overlap; the locally recovered prepared split has no exact cleaned-text overlap across splits. See [data verification](DATA.md).

**Saved hierarchical evaluation report:** Micro-F1 0.807333616; Macro-F1 0.772465839; Suicide Intent precision 0.750741840 and recall 0.848993289. Internal aggregate consistency and label support have been checked, but neural predictions were not rerun. The thesis table's 0.773 Macro-F1 differs from the supplied report. See [artifact verification](ARTIFACTS.md).

**Intended use:** Academic review, method discussion and demonstration of evaluation logic.

**Limitations:** English Reddit domain, overlapping and partly model-assisted dataset labels, fixed sentence truncation, no external clinical validation, no component ablation results, and no repeated-run uncertainty estimates. Risk categories are rule-derived research outputs. A Green category does not establish absence of risk. Attention ranking does not establish causation.

**Application restrictions:** The dataset authors require written permission for dataset use in web/mobile applications or commercial tools. Refer to current upstream terms before any such work.
