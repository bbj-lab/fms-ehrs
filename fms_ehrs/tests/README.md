# fms-ehrs Pipeline Tests

This directory verifies the active scripts in `fms_ehrs/scripts/`:

```text
Stage 0
  tokenize_w_config.py
    -> <data_version>-tokenized/{train,val,test}/tokens_timelines.parquet
    -> <data_version>-tokenized/train/vocab.gzip
    -> <data_version>-tokenized/train/numeric_stats.json

Stage 1
  tune_model.py                  (Exp1)
  train_representation.py        (Exp2/Exp3)
    -> artifacts/runs/models/.../checkpoint-*
    -> representation_mechanics.pt (wrapper models)

Stage 2
  extract_hidden_states.py
    -> <data_version>-tokenized/<split>/features-<model>.npy

Stage 3
  transfer_rep_based_preds.py
    -> <data_version>-tokenized/test/*-preds-*.pkl

Stats backend
  aggregate_version_preds.py
    -> aggregated metric and pairwise tables

Mechanistic analysis
  eval_token_ce.py
    -> token-level CE summaries
```

## Layout

- `unit/`: script-coverage contracts and script-level unit tests.
- `dryrun/`: one dry-run wrapper per active script.

## Unit tests

Use the same environment as benchmark runs:

```bash
conda activate input-rep
pytest fms_ehrs/tests/unit
```

If `pytest` is unavailable, run the script-coverage checks directly:

```bash
python - <<'PY'
import sys
sys.path.insert(0, ".")
from fms_ehrs.tests.unit import test_script_contracts as t
t.test_every_active_script_has_dryrun_wrapper()
t.test_dryrun_manifest_matches_active_scripts()
print("fms script contract checks passed")
PY
```

## Dry-run wrappers

Compile-only mode (default, fastest):

```bash
bash fms_ehrs/tests/dryrun/run_all.sh
```

Execute-mode for wrappers configured with safe runtime calls:

```bash
FMS_DRYRUN_EXECUTE_MODE=1 bash fms_ehrs/tests/dryrun/run_all.sh
```

Execute-mode needs runtime dependencies in the active environment (for example `polars`).
