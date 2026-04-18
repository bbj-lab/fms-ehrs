# fms-ehrs

`fms-ehrs` runs the model steps used by
`../input-representation-benchmark`.
It turns event tables into token sequences, trains models, extracts hidden
model feature vectors, and runs prediction tasks.
The benchmark repository controls experiment scheduling and final statistics
assembly.

## Active scripts

- `fms_ehrs/scripts/tokenize_w_config.py`
- `fms_ehrs/scripts/tune_model.py`
- `fms_ehrs/scripts/train_representation.py`
- `fms_ehrs/scripts/extract_hidden_states.py`
- `fms_ehrs/scripts/transfer_rep_based_preds.py`
- `fms_ehrs/scripts/aggregate_version_preds.py`
- `fms_ehrs/scripts/eval_token_ce.py`

Older scripts were moved to `deprecated/`.

## Current benchmark snapshot

The current benchmark trains 28 model settings under the same one-epoch
training limit:

- Experiment 1 tests numeric bin size, reference-range anchoring, and whether
  code and value are merged into one token.
- Experiment 2 tests value methods (`discrete`, `soft`, `xval`,
  `xval_affine`) and time methods (`none`, `age`, `rope`).
- Experiment 3 tests vocabulary mapping arms (`native`, `clif_mapped`,
  `rand_mapped`, `freq_mapped`) with the `discrete + rope` setting.

The full benchmark defines 30 outcomes. Each experiment evaluates 29 outcomes
because the ICU outcome differs between Experiments 1-2 and Experiment 3.

## What this repo is responsible for

- tokenize MEDS event tables from YAML configuration files
- train sequence models
- rebuild value support modules during extraction when needed
- extract final model feature vectors from first-24-hour token timelines
- fit prediction models and save prediction payloads
- aggregate prediction payloads into metrics, confidence intervals, and paired
  comparison tables

## Benchmark hand-offs

| Benchmark step | Script in this repo |
| --- | --- |
| Stage 0 | `fms_ehrs/scripts/tokenize_w_config.py` |
| Exp1 Stage 1 | `fms_ehrs/scripts/tune_model.py` |
| Exp2/Exp3 Stage 1 | `fms_ehrs/scripts/train_representation.py` |
| Stage 2 | `fms_ehrs/scripts/extract_hidden_states.py` |
| Stage 3 | `fms_ehrs/scripts/transfer_rep_based_preds.py` |
| stats backend for benchmark postprocessing | `fms_ehrs/scripts/aggregate_version_preds.py` |

## Active tokenizer configs

- `fms_ehrs/config/mimic-meds.yaml`
- `fms_ehrs/config/mimic-meds-ed.yaml`
- `fms_ehrs/config/mimic-meds-exp3-icu.yaml`

Older CLIF configs live under `deprecated/config/`.

For current Experiment 3 runs, `mimic-meds-exp3-icu.yaml` tokenizes LAB and
VITAL event blocks.

## Artifact contract

| Artifact | Produced by | Used by |
| --- | --- | --- |
| `<data_version>-tokenized/train/vocab.gzip` | `tokenize_w_config.py` | training and extraction |
| `<data_version>-tokenized/train/numeric_stats.json` | `tokenize_w_config.py` | `xval` / `xval_affine` value modules |
| `<data_version>_first_24h-tokenized/<split>/tokens_timelines.parquet` | tokenization | extraction |
| `<data_version>_first_24h-tokenized/<split>/tokens_timelines_outcomes.parquet` | benchmark-side outcome joiners | Stage 3 |
| `<model_dir>/checkpoint-*` | `tune_model.py` or `train_representation.py` | extraction |
| `<model_dir>/representation_mechanics.pt` | `train_representation.py` | value module rebuild |
| `<data_version>_first_24h-tokenized/<split>/features-<model>.npy` | `extract_hidden_states.py` | downstream probes |
| `<data_version>_first_24h-tokenized/test/*-preds-*.pkl` | `transfer_rep_based_preds.py` | `aggregate_version_preds.py` and benchmark-side stats refresh |

## Reporting assumptions in this repo

- First-24-hour tokenized timelines are the extraction surface for prediction
  features.
- `xval` and `xval_affine` runs depend on both `numeric_stats.json` and
  `representation_mechanics.pt`.
- `aggregate_version_preds.py` writes per-family metrics and paired tables.
  The benchmark repository then builds combined reporting tables.

## Reproducibility notes

- This repository covers the model-side path: tokenization, training,
  extraction, and prediction output generation.
- For the paper's reported statistics files, figure inputs, and metric audit
  surfaces, see the `Statistics files for Reproducibility` section in
  `../input-representation-benchmark/README.md`.

## Directory map

| Path | Role |
| --- | --- |
| `fms_ehrs/framework/` | active library modules |
| `fms_ehrs/config/` | active MEDS configs |
| `fms_ehrs/scripts/` | active runnable scripts |
| `notes/` | short maintained notes |
| `fms_ehrs/tests/unit/` | unit and contract tests |
| `fms_ehrs/tests/dryrun/` | dry-run checks for active scripts |
| `docs/` | structure and surface-inventory docs |
| `deprecated/` | archived scripts, configs, notes, launchers, and diagrams |

`slurm/` is now a pointer directory. Archived launchers are in `deprecated/slurm/`.

## Installation

```bash
uv venv --python="$(which python3)" venv
. venv/bin/activate
uv pip install --torch-backend=cu128 --link-mode=copy -e .
```

## Docs

- `fms_ehrs/scripts/README.md`: active script inventory
- `fms_ehrs/tests/README.md`: unit and dry-run audit layout
- `docs/layout.md`: repo layout
- `docs/surface_inventory.md`: active/utility/deprecated classification
- `notes/README.md`: maintained notes
- `deprecated/README.md`: archived material
- `../input-representation-benchmark/README.md`: benchmark-level run path
