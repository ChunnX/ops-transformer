#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# =====================================================================
# build_mla.sh - 一键编译 flash_mla_with_kvcache 算子族（主算子 + metadata AICPU 算子，Ascend 950）
#
# 用途：
#   封装仓根 build.sh，以固定参数构建 flash_mla_with_kvcache 算子族的
#   run 包（含 kernel bin）：
#     bash build.sh --pkg --soc=ascend950 \
#       --ops=flash_mla_with_kvcache,flash_mla_with_kvcache_metadata -j20
#   其中 flash_mla_with_kvcache_metadata 为纯 AICPU（仅 op_kernel_aicpu）
#   算子；如遇缺少 aic-*-ops-info.ini 的 OpFileNotExistsError，与 AICPU
#   kernel 编译相关，参见 docs/zh/install/compile.md 与 aicpu 开发指南
#   （docs/zh/develop/aicpu_develop_guide.md）。
#   -j20 由本脚本强制统一（不得被外部覆盖），并剥离用户透传的
#   --ops= / --soc= / -j / -jN，防止与固定标志冲突（双标志冲突）。
#
# 用法示例：
#   bash build_mla.sh                 # 首次全量编译（非 incremental）
#   bash build_mla.sh --incremental   # 增量编译，复用 build/output 目录
#   bash build_mla.sh --pkg-type=deb  # 其余参数原样透传给 build.sh
#
# 产物：
#   build_out/*.run （run 包），build/（kernel bin 与编译中间产物）
# =====================================================================
set -euo pipefail

ROOT=$(dirname "$(readlink -f "$0")")

# ---------- CANN 环境探测与引入 ----------
if ! command -v bisheng >/dev/null 2>&1; then
    echo "[build_mla] bisheng not found in PATH, trying to source setenv.bash ..."
    candidate_envs=()
    if [[ -n "${ASCEND_CANN_PACKAGE_PATH:-}" ]]; then
        candidate_envs+=("${ASCEND_CANN_PACKAGE_PATH}/bin/setenv.bash")
    fi
    candidate_envs+=("${HOME}/Ascend/ascend-toolkit/latest/bin/setenv.bash")
    candidate_envs+=("/usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash")

    sourced_env=""
    for env in "${candidate_envs[@]}"; do
        if [[ -f "${env}" ]]; then
            # shellcheck disable=SC1090
            source "${env}"
            sourced_env="${env}"
            break
        fi
    done

    if ! command -v bisheng >/dev/null 2>&1; then
        echo "[build_mla][ERROR] bisheng not found after sourcing setenv.bash." >&2
        echo "[build_mla][ERROR] candidates in priority order:" >&2
        for env in "${candidate_envs[@]}"; do
            echo "[build_mla][ERROR]   ${env}" >&2
        done
        echo "[build_mla][ERROR] Please install CANN toolkit and (re)set ASCEND_CANN_PACKAGE_PATH." >&2
        exit 1
    fi
    echo "[build_mla] bisheng ready via ${sourced_env}"
fi

# ---------- 编译参数（固定值，不对外部开放） ----------
# flash_mla_with_kvcache: 主算子；flash_mla_with_kvcache_metadata: 纯 AICPU
# metadata 算子（仅 op_kernel_aicpu）。
SOC=ascend950
OPS=flash_mla_with_kvcache,flash_mla_with_kvcache_metadata

# ---------- 透传参数剥离（防双标志冲突） ----------
pass_args=()
for arg in "$@"; do
    case "${arg}" in
        --ops=* | --soc=* | -j | -j[0-9]*)
            echo "[build_mla] ignore overridden arg: ${arg} (forced: --ops=${OPS} --soc=${SOC} -j20)"
            ;;
        *)
            pass_args+=("${arg}")
            ;;
    esac
done

# ---------- 执行 ----------
trap 'echo "[build_mla] FAILED at line ${LINENO}"' ERR

if ! bash "${ROOT}/build.sh" --pkg "--soc=${SOC}" "--ops=${OPS}" -j20 "${pass_args[@]}"; then
    rc=$?
    echo "[build_mla][ERROR] build.sh exited with code ${rc}" >&2
    if [[ -f "${ROOT}/build/failed_ops.log" ]]; then
        echo "[build_mla][ERROR] tail -50 build/failed_ops.log:" >&2
        tail -50 "${ROOT}/build/failed_ops.log" >&2
    fi
    echo "[build_mla][ERROR] build artifacts under ${ROOT}/build/ may be incomplete." >&2
    exit 1
fi

# ---------- 成功提示：产物路径 ----------
echo ""
echo "[build_mla] SUCCESS"
echo "[build_mla] artifacts:"
for pkg in "${ROOT}"/build_out/*.run; do
    [[ -f "${pkg}" ]] && echo "  run package : ${pkg}"
done
for pkg in "${ROOT}"/output/*.run; do
    [[ -f "${pkg}" ]] && echo "  run package : ${pkg}"
done
for bin in "${ROOT}"/build/ops_transformer_kernel* "${ROOT}"/build/*kernel*.bin; do
    [[ -e "${bin}" ]] && echo "  kernel bin  : ${bin}"
done
echo "[build_mla] build dir  : ${ROOT}/build/"
