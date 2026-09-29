#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
[[ "${SLIG_TIMEOUT_SECONDS:-300}" =~ ^[1-9][0-9]*$ ]] || { echo 'SLIG_TIMEOUT_SECONDS must be a positive integer' >&2; exit 2; }
probe_variant="${1:-same}"
case "${probe_variant}" in pipeline|same|split) ;; *) echo 'Usage: run.sh pipeline [options] | run.sh same|split rounds [options]' >&2; exit 2;; esac
if (( $# )); then shift; fi
if [[ "${probe_variant}" == pipeline ]]; then
    probe_name=pipeline
else
    probe_rounds="${1:-2}"
    if (( $# )); then shift; fi
    [[ "${probe_rounds}" =~ ^[1-9][0-9]*$ ]] && (( probe_rounds <= 65536 )) || { echo 'rounds must be 1..65536' >&2; exit 2; }
    probe_name="${probe_variant}_r${probe_rounds}"
fi
probe_case="${SLIG_WORK_DIR}/${probe_name}"
[[ -f "${probe_case}/manifest.json" ]] || { echo "Build ${probe_name} first" >&2; exit 2; }
export ASCEND_CUSTOM_OPP_PATH="${probe_case}/install/packages/vendors/klloss_repro"
export LD_LIBRARY_PATH="${ASCEND_CUSTOM_OPP_PATH}/op_api/lib:${LD_LIBRARY_PATH:-}"
probe_log="${SLIG_WORK_DIR}/logs/run_${probe_name}_$(date +%Y%m%d_%H%M%S)_$$.log"
echo "[SLIG] log=${probe_log}; fresh process; default device=npu:0"
timeout --signal=TERM --kill-after=15s "${SLIG_TIMEOUT_SECONDS:-300}s" \
    "${SLIG_PYTHON}" "${SLIG_REPO_ROOT}/scripts/klloss/run.py" "${probe_case}/manifest.json" "$@" 2>&1 | tee "${probe_log}"
