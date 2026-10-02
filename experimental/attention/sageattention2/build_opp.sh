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
# SageAttention2 标准 OPP 自定义算子包构建脚本
# -----------------------------------------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../../.." && pwd)

echo "================================================================================"
echo "  SageAttention2 OPP Custom Operator Package Build Pipeline"
echo "  Repository Root: ${REPO_ROOT}"
echo "  Target Operator: SageAttention2 (INT8 QK^T + FP8 PV, Ascend 950PR)"
echo "================================================================================"

# 1. 探测与配置 CANN Toolkit 环境
if [ -z "${ASCEND_HOME_PATH:-}" ] && [ -z "${ASCEND_TOOLKIT_HOME:-}" ]; then
    for cand in "/usr/local/Ascend/ascend-toolkit/latest" "${HOME}/Ascend/ascend-toolkit/latest" "/usr/local/Ascend/latest"; do
        if [ -d "$cand" ]; then
            export ASCEND_TOOLKIT_HOME="$cand"
            export ASCEND_HOME_PATH="$cand"
            if [ -f "$cand/set_env.sh" ]; then
                echo "[INFO] Sourcing CANN environment: $cand/set_env.sh"
                # shellcheck source=/dev/null
                source "$cand/set_env.sh" || true
            fi
            break
        fi
    done
fi

# 2. 构造默认构建参数（支持传入参数覆盖或补充）
DEFAULT_ARGS=(
    "--pkg"
    "--experimental"
    "--soc=ascend950"
    "--ops=sageattention2"
    "--build-type=Release"
    "-O3"
)

if [ $# -eq 0 ]; then
    BUILD_ARGS=("${DEFAULT_ARGS[@]}")
else
    BUILD_ARGS=("$@")
fi

echo "[INFO] Executing build command in ${REPO_ROOT}:"
echo "       bash ./build.sh ${BUILD_ARGS[*]}"
echo ""

cd "${REPO_ROOT}"
bash ./build.sh "${BUILD_ARGS[@]}"

# 3. 校验并展示产物
RUN_FILE=""
for cand in "${REPO_ROOT}/build_out"/cann-ops-transformer-custom_*.run \
            "${REPO_ROOT}/build"/cann-ops-transformer-custom_*.run; do
    if [ -f "$cand" ]; then
        RUN_FILE="$cand"
        break
    fi
done

echo ""
echo "================================================================================"
if [ -n "$RUN_FILE" ] && [ -f "$RUN_FILE" ]; then
    echo "  [SUCCESS] SageAttention2 OPP Custom Operator Package Built Successfully!"
    echo "  Package Location: ${RUN_FILE}"
    echo "  File Size       : $(ls -lh "$RUN_FILE" | awk '{print $5}')"
    echo ""
    echo "  To install the package, execute:"
    echo "    bash ${SCRIPT_DIR}/install_opp.sh"
    echo "  or directly run:"
    echo "    bash ${RUN_FILE} --quiet --install-for-all"
else
    echo "  [NOTICE] Build completed. Please verify build_out/ directory."
fi
echo "================================================================================"
