# Command-Line Scripts

Stage file contracts and benchmark configurations are documented in [`../../README.md`](../../README.md).

| Script | Stage | Description |
| --- | --- | --- |
| `collate_clif_with_cocoa.py` | Stage 0 (Exp 3 prep) | Collate raw CLIF 2.1.0 tables via Cocoa and align admission splits with the native MIMIC-IV ICU cohort |
| `tokenize_w_config.py` | Stage 0 | Tokenize MEDS event tables into integer timelines, vocabularies, and training-split numeric statistics |
| `tune_model.py` | Stage 1 (Exp 1) | Train packed causal decoders across quantization and fusion configurations |
| `train_representation.py` | Stage 1 (Exp 2 & 3) | Train windowed causal decoders across numeric, temporal, and schema configurations |
| `extract_hidden_states.py` | Stage 2 | Extract the final valid hidden state from each first-24-hour admission prefix |
| `transfer_rep_based_preds.py` | Stage 3 | Fit locked L2-regularized logistic and Ridge probes on frozen Stage 2 features |
| `aggregate_version_preds.py` | Summary | Aggregate Stage 3 probe predictions across outcomes and random seeds |
| `eval_token_ce.py` | Analysis | Compute token-level cross-entropy on held-out sequences |

See [`../tests/README.md`](../tests/README.md) for unit and dry-run test commands.
