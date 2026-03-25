# Layout (Model Execution)

This is the primary navigation layout for `fms-ehrs`.

## Main directories

- `fms_ehrs/framework/`: core library modules.
- `fms_ehrs/scripts/`: active CLI entrypoints used by benchmark stages.
- `fms_ehrs/config/`: active tokenizer/model config files used by the benchmark.
- `fms_ehrs/tests/unit/`: unit and contract tests.
- `fms_ehrs/tests/dryrun/`: one dry-run wrapper per active script.
- `fms_ehrs/tests/`: compatibility wrappers for legacy test imports.
- `notes/`: maintained notes.
- `docs/`: inventory and structure docs.
- `deprecated/`: retired CLIF/UCMC launchers, scripts, configs, and notes.

## Active path constraints

- Package/import name stays `fms_ehrs`.
- Active script filenames in `fms_ehrs/scripts/` stay stable.
- Active YAML names in `fms_ehrs/config/` stay stable.

## Reorg policy

- Move benchmark-irrelevant residue to `deprecated/`.
- Keep active script entrypoints fixed for cross-repo compatibility.
