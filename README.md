# fms-ehrs

Modeling codebase for [`../input-representation-benchmark`](../input-representation-benchmark). This repository implements clinical event tokenization, causal decoder training, hidden-state extraction, linear probe evaluation, and prediction aggregation.

Read the two README files when you reproduce a benchmark run.

## Setup

Clone `fms-ehrs` next to `input-representation-benchmark` and create the shared Conda environment.

```bash
conda env create -f ../input-representation-benchmark/environment.yml
conda activate input-rep
pip install -e .
```

To run the Experiment 3 CLIF-to-MEDS collation adapter, install the optional Cocoa dependency.

```bash
pip install -e ".[clif-cocoa]"
```

## Benchmark Overview

The benchmark trains 156 causal decoder models on full MIMIC-IV v3.1 hospitalizations across two architectures (`Llama-3.2`-style and `Qwen3`-style), three random seeds (`42`, `43`, `44`), and three experiments.

- **Experiment 1 (Quantization and fusion, 72 runs):** Crosses six quantization schemes (`deciles`, `ventiles`, `5-10-5` reference-anchored ventiles, `trentiles`, `10-10-10` reference-anchored trentiles, and `centiles`) with `unfused` and `fused` code-value layouts (12 configurations).
- **Experiment 2 (Numeric and temporal encoding, 72 runs):** Crosses four numeric encodings (`discrete`, `soft`, `xval`, `xval_affine`) with three temporal encodings (`time_tokens`, `event_order`, `time_rope`) (12 configurations).
- **Experiment 3 (Native MIMIC versus CLIF schema, 12 runs):** Compares native MIMIC-IV MEDS with Common Longitudinal Intensive Care Unit Data Format (CLIF 2.1.0) on a matched ICU cohort (`deciles + unfused + discrete + time_tokens`).

Stage 2 extracts the final-layer hidden state at the last valid token in the first 24 hours (`[0h, 24h]`). Stage 3 fits locked linear probes (`LogisticRegression(C=0.01)` and `Ridge(alpha=100.0)`) on clinical outcomes during hours 24 to 48 (`(24h, 48h]`, 37 tasks in Experiments 1 and 2, and 36 tasks in Experiment 3).

## Pipeline Entrypoints

| Stage | Script | Purpose |
| --- | --- | --- |
| 0 (Exp 3 prep) | `fms_ehrs/scripts/collate_clif_with_cocoa.py` | Collate CLIF 2.1.0 tables via Cocoa and align admission splits with the native MIMIC-IV ICU cohort |
| 0 | `fms_ehrs/scripts/tokenize_w_config.py` | Convert MEDS event tables into integer token timelines, vocabularies, and training-split numeric statistics |
| 1 (Exp 1) | `fms_ehrs/scripts/tune_model.py` | Train packed causal decoders across quantization and fusion configurations |
| 1 (Exp 2 & 3) | `fms_ehrs/scripts/train_representation.py` | Train windowed causal decoders across numeric, temporal, and schema configurations |
| 2 | `fms_ehrs/scripts/extract_hidden_states.py` | Extract the final valid hidden state from each 24-hour admission prefix |
| 3 | `fms_ehrs/scripts/transfer_rep_based_preds.py` | Fit locked L2-regularized logistic and Ridge probes on frozen Stage 2 features |
| Summary | `fms_ehrs/scripts/aggregate_version_preds.py` | Aggregate Stage 3 probe predictions across outcomes and seeds |
| Analysis | `fms_ehrs/scripts/eval_token_ce.py` | Compute token-level cross-entropy on held-out sequences |

## Tokenizer Configurations

| Config | Experiment | Scope |
| --- | --- | --- |
| `fms_ehrs/config/mimic-meds-ed.yaml` | Experiments 1 & 2 | MIMIC-IV emergency department prefix streams and inpatient MEDS events |
| `fms_ehrs/config/mimic-meds.yaml` | Reference | Inpatient MIMIC-IV MEDS events without emergency department prefix streams |
| `fms_ehrs/config/mimic-meds-exp3-full.yaml` | Experiment 3 (Native) | Native MIMIC-IV MEDS events across the ten shared clinical event categories |
| `fms_ehrs/config/cocoa-exp3-clif-full.yaml` | Experiment 3 (CLIF) | Harmonized CLIF 2.1.0 events across the ten shared clinical event categories |

## Stage File Contracts

| File | Producer | Consumer |
| --- | --- | --- |
| `<data_version>-tokenized/train/vocab.gzip` | `tokenize_w_config.py` | Stage 1 training and Stage 2 extraction |
| `<data_version>-tokenized/train/numeric_stats.json` | `tokenize_w_config.py` | `xval` and `xval_affine` numeric wrappers |
| `<data_version>_first_24h-tokenized/<split>/tokens_timelines.parquet` | `tokenize_w_config.py` | Stage 2 feature extraction |
| `<data_version>_first_24h-tokenized/<split>/tokens_timelines_outcomes.parquet` | Benchmark outcome scripts | Stage 3 linear probes |
| `<model_dir>/checkpoint-*` | `tune_model.py`, `train_representation.py` | Stage 2 feature extraction |
| `<model_dir>/representation_mechanics.pt` | `train_representation.py` | Wrapper reconstruction during Stage 2 extraction |
| `<split>/features-<model_stem>.npy` | `extract_hidden_states.py` | Stage 3 linear probes |
| `test/*-preds-*.pkl` | `transfer_rep_based_preds.py` | Metric aggregation and bootstrap evaluation |

## Repository Layout

| Path | Contents |
| --- | --- |
| `fms_ehrs/framework/` | Core modules for tokenization, datasets, model wrappers, temporal encodings, and evaluation |
| `fms_ehrs/config/` | YAML configurations for Stage 0 tokenization |
| `fms_ehrs/scripts/` | Command-line entrypoints for Stages 0 through 3 |
| `fms_ehrs/tests/` | Unit tests and dry-run contract checks |
| `notes/` | Developer guide for adding new events and outcomes |

## Testing

Run the unit test suite and dry-run script contract checks from the repository root.

```bash
pytest fms_ehrs/tests/unit
bash fms_ehrs/tests/dryrun/run_all.sh
```
