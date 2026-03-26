# Surface Inventory (Model Execution Reorg)

This inventory classifies the `fms-ehrs` surface used by the input-representation benchmark.

## Pipeline-Critical

- `fms_ehrs/scripts/tokenize_w_config.py`
- `fms_ehrs/scripts/tune_model.py`
- `fms_ehrs/scripts/train_representation.py`
- `fms_ehrs/scripts/extract_hidden_states.py`
- `fms_ehrs/scripts/transfer_rep_based_preds.py`
- `fms_ehrs/scripts/aggregate_version_preds.py`
- `fms_ehrs/scripts/eval_token_ce.py`
- `fms_ehrs/config/mimic-meds.yaml`
- `fms_ehrs/config/mimic-meds-ed.yaml`
- `fms_ehrs/config/mimic-meds-exp3-icu.yaml`
- `fms_ehrs/framework/`
- `fms_ehrs/tests/unit/test_prediction_aggregation.py`
- `fms_ehrs/tests/unit/test_train_representation.py`
- `fms_ehrs/tests/unit/test_script_contracts.py`
- `fms_ehrs/tests/dryrun/`
- `pyproject.toml`

## Active Utilities

- `fms_ehrs/tests/unit/test_muon_optimizer.py`
- `fms_ehrs/tests/unit/test_soft_target_loss.py`
- `fms_ehrs/tests/unit/test_value_encoders.py`
- `fms_ehrs/tests/unit/test_xval.py`
- `fms_ehrs/tests/unit/test_windowed_padded_dataset.py`
- `notes/`
- `env.def` (container build helper; optional for benchmark pipeline)

## Deprecated

- `deprecated/misc/modified_whiskey.sql`
- `deprecated/misc/modified_cohort_id.sql`
- `deprecated/img/`

`fms_ehrs/misc/` is now a compatibility stub directory for archived CLIs.

## Compatibility Risk Notes

- Keep package name and import path `fms_ehrs` unchanged.
- Keep `fms_ehrs/scripts/*.py` entrypoint filenames stable.
- Keep `fms_ehrs/config/*.yaml` stable because benchmark SLURM launchers pass these paths directly.
