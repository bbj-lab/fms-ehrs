#!/usr/bin/env python3
"""Freeze and materialize the Exp3 CLIF arm through Cocoa."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fms_ehrs.framework.cocoa_meds import (
    freeze_exp3_splits,
    materialize_clif_meds,
    validate_collation_config,
    validate_exp3_meds_contract,
)


def _path(value: str) -> Path:
    return Path(value).expanduser()


def _freeze_command(args: argparse.Namespace) -> int:
    manifest = freeze_exp3_splits(
        cohort_dir=args.cohort_dir,
        raw_meds_dir=args.raw_meds_dir,
        output_path=args.output,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def _materialize_command(args: argparse.Namespace) -> int:
    manifest = materialize_clif_meds(
        raw_data_home=args.raw_data_home,
        collation_config=args.collation_config,
        frozen_manifest_path=args.frozen_splits_manifest,
        output_meds_dir=args.output_meds_dir,
        work_dir=args.work_dir,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def _validate_command(args: argparse.Namespace) -> int:
    if args.collation_config is not None:
        validate_collation_config(args.collation_config)
    validation = validate_exp3_meds_contract(
        meds_dir=args.meds_dir,
        frozen_manifest_path=args.frozen_splits_manifest,
        require_all_families=not args.allow_empty_families,
    )
    print(json.dumps(validation, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    freeze = commands.add_parser(
        "freeze-splits",
        help="Verify the raw arm and write a frozen split manifest.",
    )
    freeze.add_argument("--cohort-dir", type=_path, required=True)
    freeze.add_argument("--raw-meds-dir", type=_path, required=True)
    freeze.add_argument("--output", type=_path, required=True)
    freeze.set_defaults(func=_freeze_command)

    materialize = commands.add_parser(
        "materialize",
        help="Collate CLIF with Cocoa and adapt it to benchmark MEDS splits.",
    )
    materialize.add_argument("--raw-data-home", type=_path, required=True)
    materialize.add_argument("--collation-config", type=_path, required=True)
    materialize.add_argument("--frozen-splits-manifest", type=_path, required=True)
    materialize.add_argument("--output-meds-dir", type=_path, required=True)
    materialize.add_argument("--work-dir", type=_path, required=True)
    materialize.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only the requested output and work directories.",
    )
    materialize.set_defaults(func=_materialize_command)

    validate = commands.add_parser(
        "validate",
        help="Validate a materialized CLIF MEDS arm against frozen native splits.",
    )
    validate.add_argument("--meds-dir", type=_path, required=True)
    validate.add_argument("--frozen-splits-manifest", type=_path, required=True)
    validate.add_argument("--collation-config", type=_path)
    validate.add_argument(
        "--allow-empty-families",
        action="store_true",
        help="Only for synthetic fixtures; production validation requires all families.",
    )
    validate.set_defaults(func=_validate_command)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
