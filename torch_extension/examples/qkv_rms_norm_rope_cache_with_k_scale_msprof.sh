#!/usr/bin/env bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=${REPO:-$(cd "${SCRIPT_DIR}/../.." && pwd)}
CANN_ENV=${CANN_ENV:-/usr/local/Ascend/cann-9.1.0/set_env.sh}
PYTHON=${PYTHON:-python3}
TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-/home/s00620103/torch_extensions/qkv_rms_norm_rope_cache_with_k_scale}
OUTPUT_ROOT=${OUTPUT_ROOT:-/home/s00620103/msprof/qkv_rms_norm_rope_cache_with_k_scale/$(date +%Y%m%d_%H%M%S)}
AIC_METRICS=${AIC_METRICS:-PipeUtilization}
MAX_DIFF_LINES=${MAX_DIFF_LINES:-0}
EXAMPLE=${EXAMPLE:-torch_extension/examples/qkv_rms_norm_rope_cache_with_k_scale_example.py}

DEFAULT_CASES=(
    n64_kv8_perf_t16384_dpad_tnd_to_ntd
    n16_kv2_perf_t16384_dpad_tnd_to_ntd
    n8_kv1_perf_t16384_dpad_tnd_to_ntd
)

CASES=()
EXTRA_ARGS=()

usage() {
    cat <<'EOF'
Usage:
  qkv_rms_norm_rope_cache_with_k_scale_msprof.sh [--output DIR] [--case CASE ...] [--max-diff-lines N]
                                                 [--aic-metrics GROUP] [-- EXTRA_ARGS...]

Default cases:
  n64_kv8_perf_t16384_dpad_tnd_to_ntd  # T=16384, Nq=64, Nk=Nv=8
  n16_kv2_perf_t16384_dpad_tnd_to_ntd  # T=16384, Nq=16, Nk=Nv=2
  n8_kv1_perf_t16384_dpad_tnd_to_ntd   # T=16384, Nq=8,  Nk=Nv=1

Environment overrides:
  REPO, CANN_ENV, PYTHON, TORCH_EXTENSIONS_DIR, OUTPUT_ROOT, AIC_METRICS, MAX_DIFF_LINES, EXAMPLE
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --case)
            CASES+=("$2")
            shift 2
            ;;
        --output)
            OUTPUT_ROOT="$2"
            shift 2
            ;;
        --max-diff-lines)
            MAX_DIFF_LINES="$2"
            shift 2
            ;;
        --aic-metrics)
            AIC_METRICS="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        --)
            shift
            EXTRA_ARGS+=("$@")
            break
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ ! -f "${CANN_ENV}" ]]; then
    echo "CANN env file not found: ${CANN_ENV}" >&2
    exit 2
fi

# shellcheck source=/dev/null
source "${CANN_ENV}"

PYTHON_BIN=$(command -v "${PYTHON}" || true)
if [[ -z "${PYTHON_BIN}" ]]; then
    echo "python not found: ${PYTHON}" >&2
    exit 2
fi

if ! command -v msprof >/dev/null 2>&1; then
    echo "msprof not found after sourcing ${CANN_ENV}" >&2
    exit 2
fi

if [[ "${#CASES[@]}" -eq 0 ]]; then
    CASES=("${DEFAULT_CASES[@]}")
fi

mkdir -p "${TORCH_EXTENSIONS_DIR}" "${OUTPUT_ROOT}"
export TORCH_EXTENSIONS_DIR
export LD_LIBRARY_PATH="/usr/local/Ascend/cann-9.1.0/opp/vendors/custom_transformer/op_api/lib:${LD_LIBRARY_PATH:-}"

cd "${REPO}"

for case_name in "${CASES[@]}"; do
    case_output="${OUTPUT_ROOT}/${case_name}"
    log_file="${OUTPUT_ROOT}/${case_name}.log"
    mkdir -p "${case_output}"
    echo "[MSPROF] case=${case_name} output=${case_output}"
    msprof --output="${case_output}" --ascendcl=on --runtime-api=on --task-time=on --ai-core=on \
        --aic-metrics="${AIC_METRICS}" --python-path="${PYTHON_BIN}" \
        "${PYTHON_BIN}" "${EXAMPLE}" --case "${case_name}" --max-diff-lines "${MAX_DIFF_LINES}" \
        --perf-warmup-all \
        "${EXTRA_ARGS[@]}" 2>&1 | tee "${log_file}"
done

echo "[MSPROF] all cases complete: ${OUTPUT_ROOT}"
