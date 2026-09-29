#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
probe_variant="${1:-all}"
probe_rounds="${2:-2}"
case "${probe_variant}" in pipeline|same|split|all) ;; *) echo 'Usage: compile.sh [pipeline|same|split|all] [rounds]' >&2; exit 2;; esac
[[ "${probe_rounds}" =~ ^[1-9][0-9]*$ ]] && (( probe_rounds <= 65536 )) || { echo 'rounds must be 1..65536' >&2; exit 2; }
probe_log="${SLIG_WORK_DIR}/logs/build_${probe_variant}_r${probe_rounds}_$(date +%Y%m%d_%H%M%S)_$$.log"
"${SLIG_PYTHON}" "${SLIG_REPO_ROOT}/scripts/klloss/build.py" "${probe_variant}" --rounds "${probe_rounds}" 2>&1 | tee "${probe_log}"
