#!/usr/bin/env python3

"""
Canonical entrypoint for aggregating saved prediction files into paper-facing
metric and pairwise comparison tables.

This wrapper preserves a clearer public name while delegating to the legacy
module path for backward compatibility.
"""

from __future__ import annotations

from fms_ehrs.scripts.aggregate_version_preds import main


if __name__ == "__main__":
    raise SystemExit(main())
