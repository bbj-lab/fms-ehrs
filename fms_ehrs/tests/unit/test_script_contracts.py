from __future__ import annotations

import json
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _active_scripts(root: Path) -> list[Path]:
    scripts_dir = root / "fms_ehrs" / "scripts"
    return sorted(p for p in scripts_dir.glob("*.py") if p.name != "__init__.py")


def _dryrun_name(rel_path: str) -> str:
    return f"run_{rel_path.replace('/', '__').replace('.py', '')}.sh"


def test_every_active_script_has_dryrun_wrapper() -> None:
    root = _repo_root()
    dryrun_root = root / "fms_ehrs" / "tests" / "dryrun"
    for script_path in _active_scripts(root):
        rel = script_path.relative_to(root).as_posix()
        dryrun = dryrun_root / _dryrun_name(rel)
        assert dryrun.exists(), f"missing dry-run wrapper for {rel}"


def test_dryrun_manifest_matches_active_scripts() -> None:
    root = _repo_root()
    scripts = [p.relative_to(root).as_posix() for p in _active_scripts(root)]
    manifest_path = root / "fms_ehrs" / "tests" / "dryrun" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    listed = sorted(entry["script"] for entry in manifest["scripts"])
    assert listed == scripts
