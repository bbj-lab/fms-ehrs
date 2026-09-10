#!/usr/bin/env python3

"""
Single source of truth for naming and verifying per-model artifacts.

Every downstream artifact (hidden-state features, probe predictions, token-level
cross entropy) is keyed by a model stem. That stem must identify the *training
run*, not the checkpoint directory it points at: campaigns publish their final
checkpoints as symlinks, and several runs legitimately resolve to the same
checkpoint basename (for example ``checkpoint-372270``). Deriving a stem from a
resolved path therefore collapses distinct runs onto one filename and silently
overwrites results.

The helpers here keep the link name, reject stem collisions before jobs are
submitted, and record provenance beside each feature file so a mismatch fails
loudly instead of being reused.
"""

from __future__ import annotations

import collections
import json
import os
import pathlib
import re
import typing

Pathlike: typing.TypeAlias = str | os.PathLike


def sanitize_model_stem(stem: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("_")


def model_artifact_stem(model_loc: Pathlike) -> str:
    """Return the stable artifact stem for a model location.

    Uses the given path's own name (never the symlink target) so that two runs
    sharing a checkpoint keep distinct artifacts. ``name`` rather than ``stem``
    avoids truncating run names that contain a dot.
    """
    model_loc = pathlib.Path(model_loc)
    if model_loc.name.startswith("model-") and model_loc.parent.name:
        return sanitize_model_stem(f"{model_loc.parent.name}-{model_loc.name}")
    return sanitize_model_stem(model_loc.name)


def feature_filename(model_stem: str, *, all_layers: bool = False) -> str:
    return "features{x}-{m}.npy".format(x="-all-layers" if all_layers else "", m=model_stem)


def provenance_path(feature_path: Pathlike) -> pathlib.Path:
    feature_path = pathlib.Path(feature_path)
    return feature_path.with_name(f"{feature_path.name}.provenance.json")


def assert_unique_model_stems(model_locs: typing.Iterable[Pathlike]) -> dict[str, str]:
    """Fail when two model locations would write to the same artifact stem.

    Returns the stem -> model location mapping so callers can reuse it.
    """
    grouped: dict[str, list[str]] = collections.defaultdict(list)
    for model_loc in model_locs:
        grouped[model_artifact_stem(model_loc)].append(str(model_loc))
    collisions = {stem: locs for stem, locs in grouped.items() if len(locs) > 1}
    if collisions:
        detail = "\n".join(
            f"  {stem}:\n    - " + "\n    - ".join(locs) for stem, locs in sorted(collisions.items())
        )
        raise ValueError(
            "Model locations collide onto the same artifact stem; distinct runs would "
            f"overwrite each other:\n{detail}"
        )
    return {stem: locs[0] for stem, locs in grouped.items()}


def build_provenance(
    *,
    model_loc: Pathlike,
    data_version: str,
    split: str,
    all_layers: bool,
    shape: typing.Sequence[int],
    dtype: str,
) -> dict:
    model_loc = pathlib.Path(model_loc)
    return {
        "model_loc": str(model_loc),
        "model_stem": model_artifact_stem(model_loc),
        "model_target": str(model_loc.resolve()),
        "data_version": data_version,
        "split": split,
        "all_layers": bool(all_layers),
        "shape": list(shape),
        "dtype": dtype,
    }


def write_provenance(feature_path: Pathlike, provenance: dict) -> pathlib.Path:
    path = provenance_path(feature_path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def read_provenance(feature_path: Pathlike) -> dict | None:
    path = provenance_path(feature_path)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def provenance_mismatch(recorded: dict | None, expected: dict) -> str | None:
    """Return a human-readable reason the existing artifact is not reusable."""
    if recorded is None:
        return "no provenance record beside the existing feature file"
    for key in ("model_loc", "data_version", "split", "all_layers"):
        if recorded.get(key) != expected.get(key):
            return f"{key} differs (recorded {recorded.get(key)!r}, expected {expected.get(key)!r})"
    return None
