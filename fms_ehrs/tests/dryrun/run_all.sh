#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for f in "${SCRIPT_DIR}"/run_*.sh; do
  if [[ "$(basename "${f}")" == "run_all.sh" ]]; then
    continue
  fi
  echo "[dryrun] ${f}"
  bash "${f}"
done
echo "[dryrun] complete"
