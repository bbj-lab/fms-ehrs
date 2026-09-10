# Active script inventory

Script roles, output filenames, and benchmark hand-offs are in
[`../../README.md`](../../README.md).

Active entrypoints:

| Script | Stage |
| --- | --- |
| `collate_clif_with_cocoa.py` | Exp3 pre-tokenization: freeze native H_ICU splits and adapt Cocoa-collated CLIF events |
| `tokenize_w_config.py` | 0 |
| `tune_model.py` | 1 (Exp1) |
| `train_representation.py` | 1 (Exp2/Exp3) |
| `extract_hidden_states.py` | 2 |
| `transfer_rep_based_preds.py` | 3 |
| `aggregate_version_preds.py` | stats backend |
| `eval_token_ce.py` | mechanistic analysis |

Tests: [`../tests/README.md`](../tests/README.md). Archived CLIs live under
[`../../deprecated/scripts/`](../../deprecated/scripts/).
