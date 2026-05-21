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
# CCU dispatch setup/teardown <<<>>> 直调工程：一键编译 + 多进程（默认 2p）运行。
#   用法：bash run.sh [RANK_SIZE]   RANK_SIZE 为进程数/卡数，缺省 2。
CURRENT_DIR=$(
    cd $(dirname ${BASH_SOURCE:-$0})
    pwd
)
SOC_VERSION="ascend950pr_9599"
CANN_ENV="/home/0513/env/set_env.sh"
RANK_SIZE=${1:-2}   # 进程数（每进程独占一张卡），默认 2p

# 导入 CANN 环境
source "${CANN_ENV}"
_ASCEND_INSTALL_PATH="${ASCEND_HOME_PATH}"
export ASCEND_TOOLKIT_HOME="${_ASCEND_INSTALL_PATH}"
export ASCEND_HOME_PATH="${_ASCEND_INSTALL_PATH}"

set -e
# 编译
rm -rf "${CURRENT_DIR}/build" "${CURRENT_DIR}/out"
mkdir -p "${CURRENT_DIR}/build"
cmake -B "${CURRENT_DIR}/build" -S "${CURRENT_DIR}" \
    -DSOC_VERSION=${SOC_VERSION} \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_INSTALL_PREFIX="${CURRENT_DIR}/out" \
    -DASCEND_CANN_PACKAGE_PATH="${_ASCEND_INSTALL_PATH}"
cmake --build "${CURRENT_DIR}/build" -j
cmake --install "${CURRENT_DIR}/build"
set +e

# 多进程运行：rank_id = 0..RANK_SIZE-1，各进程独占同号 NPU 卡
export LD_LIBRARY_PATH="${CURRENT_DIR}/out/lib:${CURRENT_DIR}/out/lib64:${_ASCEND_INSTALL_PATH}/lib64:${LD_LIBRARY_PATH}"
echo "========== 运行 <<<>>> 直调测试：${RANK_SIZE}p（${RANK_SIZE} 进程 / ${RANK_SIZE} 卡）=========="
pids=()
for ((rank_id = 0; rank_id < RANK_SIZE; rank_id++)); do
    "${CURRENT_DIR}/out/bin/test_moe_dispatch_ccu_direct" --rank_id ${rank_id} &
    pids+=($!)
done

FAIL=0
for ((i = 0; i < RANK_SIZE; i++)); do
    if wait "${pids[$i]}"; then
        echo "[rank $i] 退出码 0"
    else
        echo "[rank $i] 退出码非 0"
        FAIL=1
    fi
done

echo "=================================================="
if [ ${FAIL} -eq 0 ]; then
    echo "全部 ${RANK_SIZE}p 端到端 <<<>>> 直调通过"
else
    echo "存在失败 rank"
fi
exit ${FAIL}
