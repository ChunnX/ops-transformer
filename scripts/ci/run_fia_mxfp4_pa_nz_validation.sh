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
# The final gate is an ACLNN direct-call example, not a caller-supplied command.
# It executes the newly built custom operator and compares NPU output with a CPU
# golden for FP16 symmetric, FP16 asymmetric, and BF16 asymmetric PA_NZ cases.

set -euo pipefail

readonly REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
readonly OP_NAME="fused_infer_attention_score"
readonly SOC="${SOC:-ascend950}"
readonly JOBS="${JOBS:-8}"
readonly REF_TEST="${REPO_ROOT}/attention/fused_infer_attention_score/tests/ut/op_host/arch35/test_mxfp4_pa_nz_reference.py"
readonly GTEST_FILTER="FusedInferAttentionScore/FusedInferAttentionScoreArch35TilingTest.param/FusedInferAttentionScore_950_mxfp4_pa_nz_*"
readonly EXAMPLE_NAME="fused_infer_attention_score_v5_antiquant_mxfp4_pa_nz"
readonly VENDOR_NAME="${FIA_VENDOR_NAME:-custom}"

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

require_custom_package() {
    # build.sh --run_example ... cust links and dispatches through this directory.
    # It must be the custom package produced from the same checkout, for example:
    #   /work/custom_opp/vendors/custom_transformer
    local custom_opp_path="${FIA_CUSTOM_OPP_PATH:-${ASCEND_CUSTOM_OPP_PATH%%:*}}"
    if [[ -z "${custom_opp_path}" ]]; then
        echo "Set FIA_CUSTOM_OPP_PATH to the installed custom vendor directory, e.g." >&2
        echo "  /work/custom_opp/vendors/${VENDOR_NAME}_transformer" >&2
        exit 2
    fi
    if [[ ! -f "${custom_opp_path}/op_api/lib/libcust_opapi.so" ]]; then
        echo "${custom_opp_path} is not an installed ${VENDOR_NAME}_transformer custom package." >&2
        exit 2
    fi
    export ASCEND_CUSTOM_OPP_PATH="${custom_opp_path}:${ASCEND_CUSTOM_OPP_PATH:-}"
    export LD_LIBRARY_PATH="${custom_opp_path}/op_api/lib:${LD_LIBRARY_PATH:-}"
    printf 'Using custom operator package: %s\n' "${custom_opp_path}"
}

load_cann_env

echo '[1/5] PA_NZ producer/address/formula oracle'
python3 "${REF_TEST}"

echo '[2/5] Ascend950 FIA kernel compilation'
bash "${REPO_ROOT}/build.sh" --opkernel --soc="${SOC}" --ops="${OP_NAME}" -j"${JOBS}"

echo '[3/5] Build Arch35 FIA host tiling UT without auto-executing the full suite'
bash "${REPO_ROOT}/build.sh" --ophost_test --soc="${SOC}" --ops="${OP_NAME}" --noexec -j"${JOBS}"

echo '[4/5] Run only the four MXFP4 PA_NZ host/tiling cases'
HOST_UT="$(find_host_ut)"
"${HOST_UT}" --gtest_filter="${GTEST_FILTER}" --gtest_color=yes

echo '[5/5] ACLNN direct PA_NZ numerical example (custom package)'
require_custom_package
bash "${REPO_ROOT}/build.sh" --run_example "${OP_NAME}" eager cust --vendor_name="${VENDOR_NAME}" --soc="${SOC}" \
    --example_name="${EXAMPLE_NAME}"
