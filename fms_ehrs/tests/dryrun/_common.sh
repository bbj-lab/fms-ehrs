#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
DRYRUN_EXECUTE_MODE="${FMS_DRYRUN_EXECUTE_MODE:-0}"

run_compile() {
  local rel="$1"
  python3 -m py_compile "${REPO_ROOT}/${rel}"
}

run_execute() {
  local rel="$1"
  shift
  python3 "${REPO_ROOT}/${rel}" "$@"
}

run_script_dryrun() {
  local rel="$1"
  local mode="$2"
  shift 2

  if [[ "${DRYRUN_EXECUTE_MODE}" != "1" ]]; then
    run_compile "${rel}"
    return
  fi

  if [[ "${mode}" == "compile" ]]; then
    run_compile "${rel}"
  else
    run_execute "${rel}" "$@"
  fi
}
