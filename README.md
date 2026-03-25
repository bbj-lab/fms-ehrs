# fms-ehrs

`fms-ehrs` provides the model-side execution path used by `../input-representation-benchmark`.
It handles tokenization, model training, hidden-state extraction, and downstream prediction jobs.
The benchmark repo controls experiment orchestration, statistics refresh, and manuscript updates.

## Active command-line entrypoints

- `fms_ehrs/scripts/tokenize_w_config.py`
- `fms_ehrs/scripts/tune_model.py`
- `fms_ehrs/scripts/train_representation.py`
- `fms_ehrs/scripts/extract_hidden_states.py`
- `fms_ehrs/scripts/transfer_rep_based_preds.py`
- `fms_ehrs/scripts/aggregate_version_preds.py`
- `fms_ehrs/scripts/eval_token_ce.py`

Older scripts were moved to `deprecated/`.

## What this repo is responsible for

- tokenize MEDS event tables from YAML config files
- train discrete and wrapper-based sequence models
- rebuild wrapper models during extraction
- extract final hidden states from 24-hour tokenized timelines
- fit downstream prediction models and save prediction payloads
- aggregate prediction payloads into metrics, confidence intervals, and pairwise tables

## Benchmark hand-offs

| Benchmark step | Entry point in this repo |
| --- | --- |
| Stage 0 | `fms_ehrs/scripts/tokenize_w_config.py` |
| Exp1 Stage 1 | `fms_ehrs/scripts/tune_model.py` |
| Exp2/Exp3 Stage 1 | `fms_ehrs/scripts/train_representation.py` |
| Stage 2 | `fms_ehrs/scripts/extract_hidden_states.py` |
| Stage 3 | `fms_ehrs/scripts/transfer_rep_based_preds.py` |
| aligned stats backend | `fms_ehrs/scripts/aggregate_version_preds.py` |

## Active tokenizer configs

- `fms_ehrs/config/mimic-meds.yaml`
- `fms_ehrs/config/mimic-meds-ed.yaml`
- `fms_ehrs/config/mimic-meds-exp3-icu.yaml`

Older CLIF configs live under `deprecated/config/`.

## Artifact contract

| Artifact | Produced by | Used by |
| --- | --- | --- |
| `<data_version>-tokenized/train/vocab.gzip` | `tokenize_w_config.py` | training and extraction |
| `<data_version>-tokenized/train/numeric_stats.json` | `tokenize_w_config.py` | `xval` / `xval_affine` wrappers |
| `<data_version>_first_24h-tokenized/<split>/tokens_timelines.parquet` | tokenization | extraction |
| `<data_version>_first_24h-tokenized/<split>/tokens_timelines_outcomes.parquet` | benchmark-side outcome joiners | Stage 3 |
| `<model_dir>/checkpoint-*` | `tune_model.py` or `train_representation.py` | extraction |
| `<model_dir>/representation_mechanics.pt` | `train_representation.py` | wrapper reconstruction |
| `<data_version>_first_24h-tokenized/<split>/features-<model>.npy` | `extract_hidden_states.py` | downstream probes |
| `<data_version>_first_24h-tokenized/test/*-preds-*.pkl` | `transfer_rep_based_preds.py` | `aggregate_version_preds.py` and benchmark-side stats refresh |

## Directory map

| Path | Role |
| --- | --- |
| `fms_ehrs/framework/` | active library modules |
| `fms_ehrs/config/` | active MEDS configs |
| `fms_ehrs/scripts/` | active script entrypoints |
| `notes/` | short maintained notes |
| `fms_ehrs/tests/unit/` | unit and contract tests |
| `fms_ehrs/tests/dryrun/` | dry-run wrappers for active scripts |
| `fms_ehrs/tests/` | compatibility wrappers for historical test imports |
| `tests/` | compatibility wrapper tests |
| `docs/` | structure and surface-inventory docs |
| `deprecated/` | archived scripts, configs, notes, launchers, and diagrams |

`slurm/` is now a pointer directory. Archived launchers are in `deprecated/slurm/`.

## Installation

```bash
uv venv --python="$(which python3)" venv
. venv/bin/activate
uv pip install --torch-backend=cu128 --link-mode=copy -e .
```

There is no maintained `requirements.txt`.

## Docs

- `fms_ehrs/scripts/README.md`: active script inventory
- `fms_ehrs/tests/README.md`: unit and dry-run audit layout
- `docs/layout.md`: repo layout
- `docs/surface_inventory.md`: active/utility/deprecated classification
- `notes/README.md`: maintained notes
- `deprecated/README.md`: archived material
- `../input-representation-benchmark/README.md`: benchmark-level run path

If this README and the benchmark repo disagree on orchestration, follow the benchmark repo and then update this file.
