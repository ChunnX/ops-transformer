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
set -euo pipefail

ROOT=$(dirname "$(readlink -f "$0")")
INSTALL_ROOT="${MLA_INSTALL_ROOT:-/workspace/Custom/flash_mla}"

# 清理
rm -rf "${INSTALL_ROOT}" "${ROOT}/build" "${ROOT}/build_out"

# 编译
bash "${ROOT}/build.sh" --pkg --soc=ascend950 --ops=flash_mla_with_kvcache,flash_mla_with_kvcache_metadata -j20 "$@"

# 安装
PKG=$(ls -t "${ROOT}"/build_out/cann-ops-transformer-custom_linux-x86_64.run | head -1)
if [[ -z "${PKG}" ]]; then
    echo "[build_mla] ERROR: run package not found under ${ROOT}/build_out/" >&2
    exit 1
fi
bash "${PKG}" --quiet --install-path="${INSTALL_ROOT}"

# source 环境 & 跑测试
set +u; source "${INSTALL_ROOT}/vendors/custom_transformer/bin/set_env.bash"; set -u
python3 "${ROOT}/test.py" --repo-root "${MLA_TESTKIT_ROOT:-/workspace/Code/perf/ops-transformer-testkit}"
