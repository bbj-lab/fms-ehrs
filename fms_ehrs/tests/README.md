# Test Suite

Run the unit tests and dry-run CLI contract checks from the repository root.

```bash
conda activate input-rep
pytest fms_ehrs/tests/unit
bash fms_ehrs/tests/dryrun/run_all.sh
```

- `unit/`: Tests for tokenization determinism, value encoders, soft-target loss, xVal mechanics, temporal encodings, windowed datasets, CLIF collation, prediction aggregation, and CLI script contracts.
- `dryrun/`: Wrapper scripts that validate CLI syntax and argument handling for every entrypoint in `fms_ehrs/scripts/`. Set `FMS_DRYRUN_EXECUTE_MODE=1` to run execute-mode checks.
