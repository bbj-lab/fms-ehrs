# Active Script Inventory

This directory now contains only the active script entrypoints on the live path.

## Active scripts

- `tokenize_w_config.py`: tokenize split parquet trees from a YAML config
- `tune_model.py`: packed training path used by Exp1
- `train_representation.py`: padded/windowed training path used by Exp2 and Exp3
- `extract_hidden_states.py`: extract final hidden states from tokenized 24-hour timelines
- `transfer_rep_based_preds.py`: fit downstream models from extracted features and save predictions
- `aggregate_version_preds.py`: aggregate saved predictions into metrics, confidence intervals, and pairwise tables
- `eval_token_ce.py`: token cross-entropy analysis used by the mechanistic section

Archived CLIs, older analysis scripts, and CLIF/UCMC command sets were moved to `../../deprecated/scripts/`.

Checks for these entrypoints live in:

- `../tests/unit/test_script_contracts.py`
- `../tests/dryrun/run_all.sh`
