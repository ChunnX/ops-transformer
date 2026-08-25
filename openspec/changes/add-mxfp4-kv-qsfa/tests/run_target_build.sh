#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${QSFA_TARGET_BUILD_COMMAND:-}" ]]; then
  echo "BLOCKED: set QSFA_TARGET_BUILD_COMMAND to the exact target Ascend C build command before running TEST-BUILD" >&2
  exit 2
fi

echo "Running owner/CI supplied target build command"
eval "${QSFA_TARGET_BUILD_COMMAND}"
