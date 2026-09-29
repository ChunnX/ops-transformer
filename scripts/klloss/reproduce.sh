#!/usr/bin/env bash
# Build from this checkout, then run the expected-timeout/control matrix serially.
set -Eeuo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
"${SLIG_PYTHON}" "${SLIG_REPO_ROOT}/scripts/klloss/reproduce.py" "$@"
