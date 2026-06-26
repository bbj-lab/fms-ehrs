# fms-ehrs tests

Output filenames and stage hand-offs are in [`../../README.md`](../../README.md).

```bash
conda activate input-rep
pytest fms_ehrs/tests/unit
bash fms_ehrs/tests/dryrun/run_all.sh
```

- `unit/`: script contracts and library tests. `test_windowed_padded_dataset.py`
  covers continuation windows and admission-level relative time.
- `dryrun/`: compile-only by default; set `FMS_DRYRUN_EXECUTE_MODE=1` for safe
  execute-mode wrappers.

If `pytest` is unavailable:

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
