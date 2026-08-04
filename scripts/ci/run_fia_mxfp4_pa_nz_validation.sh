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
# Build and run the reproducible validation gates for FIA MXFP4 mode-6 PA_NZ.
#
# It intentionally separates three levels of proof:
#   1. Python contract/address oracle;
#   2. Ascend950 kernel compilation and Arch35 host/tiling gtest cases;
#   3. an externally supplied device numerical command, when available.
# The script never reports device numerical success unless that command ran.

set -euo pipefail

readonly REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
readonly OP_NAME="fused_infer_attention_score"
readonly SOC="${SOC:-ascend950}"
readonly JOBS="${JOBS:-8}"
readonly REF_TEST="${REPO_ROOT}/attention/fused_infer_attention_score/tests/ut/op_host/arch35/test_mxfp4_pa_nz_reference.py"
readonly GTEST_FILTER="FusedInferAttentionScore/FusedInferAttentionScoreArch35TilingTest.param/FusedInferAttentionScore_950_mxfp4_pa_nz_*"

require_device=false
for argument in "$@"; do
    case "${argument}" in
        --require-device) require_device=true ;;
        *)
            echo "unknown argument: ${argument}" >&2
            echo "usage: $0 [--require-device]" >&2
            exit 64
            ;;
    esac
done

if [[ "${SOC}" != "ascend950" ]]; then
    echo "This change is Arch35-only; set SOC=ascend950." >&2
    exit 64
fi

load_cann_env() {
    local candidate
    for candidate in "${ASCEND_TOOLKIT_HOME:-}" "${ASCEND_CANN_PACKAGE_PATH:-}" \
        "/usr/local/Ascend/ascend-toolkit/latest"; do
        if [[ -n "${candidate}" && -f "${candidate}/set_env.sh" ]]; then
            # CANN's environment script can reference variables not yet defined.
            set +u
            # shellcheck disable=SC1090
            source "${candidate}/set_env.sh"
            set -u
            return
        fi
    done
    if [[ -z "${ASCEND_HOME_PATH:-}" && -z "${ASCEND_CANN_PACKAGE_PATH:-}" ]]; then
        echo "CANN environment not found. Set ASCEND_TOOLKIT_HOME or source set_env.sh before running." >&2
        exit 2
    fi
}

find_host_ut() {
    local found=()
    while IFS= read -r path; do
        found+=("${path}")
    done < <(find "${REPO_ROOT}/build" -type f -name transformer_op_host_ut -perm -111 2>/dev/null)
    if [[ ${#found[@]} -ne 1 ]]; then
        echo "expected exactly one transformer_op_host_ut under ${REPO_ROOT}/build; found ${#found[@]}" >&2
        printf '%s\n' "${found[@]:-}" >&2
        exit 2
    fi
    printf '%s\n' "${found[0]}"
}

load_cann_env

echo '[1/4] PA_NZ producer/address/formula oracle'
python3 "${REF_TEST}"

echo '[2/4] Ascend950 FIA kernel compilation'
bash "${REPO_ROOT}/build.sh" --opkernel --soc="${SOC}" --ops="${OP_NAME}" -j"${JOBS}"

echo '[3/4] Build Arch35 FIA host tiling UT without auto-executing the full suite'
bash "${REPO_ROOT}/build.sh" --ophost_test --soc="${SOC}" --ops="${OP_NAME}" --noexec -j"${JOBS}"

echo '[4/4] Run only the four MXFP4 PA_NZ host/tiling cases'
HOST_UT="$(find_host_ut)"
"${HOST_UT}" --gtest_filter="${GTEST_FILTER}" --gtest_color=yes

if [[ -n "${FIA_DEVICE_TEST_CMD:-}" ]]; then
    echo 'Running caller-provided device numerical test command'
    bash -lc "${FIA_DEVICE_TEST_CMD}"
elif [[ "${require_device}" == true ]]; then
    echo 'FIA_DEVICE_TEST_CMD is required for --require-device.' >&2
    echo 'It must execute a device test with non-identity blockTable, cross-page partial segment,' >&2
    echo 'G>=2, N>=2, symmetric plus FP16/BF16 asymmetric offsets.' >&2
    exit 2
else
    echo 'Host and compile gates passed. Device numerical test was not requested.'
    echo 'For release evidence, rerun with --require-device and FIA_DEVICE_TEST_CMD=<device-test-command>.'
fi
