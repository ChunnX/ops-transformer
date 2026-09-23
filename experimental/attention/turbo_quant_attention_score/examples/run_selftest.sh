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
# TurboQuantAttentionScore 自验证脚本（Atlas 800T A2 / Ascend 910B3）
#
# 用法：
#   REPO=~/ops-transformer bash run_selftest.sh          # 增量构建 + 编译自测 + 跑全部 case
#   REPO=~/ops-transformer FULL_BUILD=1 bash run_selftest.sh   # 全量构建（首次克隆用）
#   REPO=~/ops-transformer SKIP_BUILD=1 bash run_selftest.sh   # 跳过算子构建，只跑自测
#
# 前置：已 source CANN set_env.sh（脚本内会自动 source），NPU 空闲。
# 产物：stdout 即自验证日志（含 [MEM] / [PERF] / [PASS|FAIL] 行与 summary）。
#
# 注意（性能计时前提）：[PERF] 用 aclrt 事件测算子执行时间，但 host 侧 tiling 与下发
# 也在同一时间线上，受机器负载影响。实测：同一份二进制、同一构建产物，在不同时刻曾测出
# 2 倍差异（decode_kv4k 1.38ms vs 2.55ms）。故**性能数据必须在无其他负载时采集，并复核
# 一次**：若两次结果差异明显，取机器空闲那一次，并在报告中注明采集时的机器状态。
# ----------------------------------------------------------------------------
set -eo pipefail
# set_env.sh 内部引用了未定义的 LD_LIBRARY_PATH/PYTHONPATH/CMAKE_PREFIX_PATH，先给出默认值
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PYTHONPATH:-}"
export CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:-}"

REPO="${REPO:-$HOME/ops-transformer}"
OP="experimental/attention/turbo_quant_attention_score"
ATC_HOME="${ATC_HOME:-$HOME/Ascend/ascend-toolkit}"
JOBS="${JOBS:-16}"
SKIP_BUILD="${SKIP_BUILD:-0}"
FULL_BUILD="${FULL_BUILD:-0}"

echo "==== [env] date=$(date '+%F %T') host=$(hostname) ===="
echo "==== [env] repo=$REPO ===="
source "$ATC_HOME/set_env.sh"
echo "==== [env] CANN: $(cat "$ATC_HOME/latest/compiler/version.info" | tr '\n' ' ') ===="
npu-smi info || true

cd "$REPO"

# ---- 1. 构建算子（生成 custom_transformer 自定义算子包） ----
PKG_ROOT="$REPO/build/_CPack_Packages/Linux/External/cann-ops-transformer-custom_linux-aarch64.run/packages/vendors/custom_transformer"
if [ "$SKIP_BUILD" != "1" ]; then
  # 修改 kernel 后必须清掉该算子的中间产物，否则增量构建不会重拷源码
  rm -rf build/binary/ascend910b/src/turbo_quant_attention_score \
         build/binary/ascend910b/bin/turbo_quant_attention_score \
         build/binary/ascend910b/gen/turbo_quant_attention_score_ascend910b_0.done \
         build/binary/ascend910b/gen/kernel_meta_TurboQuantAttentionScore_* \
         build/binary/ascend910b/gen/compile_stop.flag
  # experimental/ 目录下的算子必须带 --experimental，否则不参与构建
  EXTRA="--incremental"
  if [ "$FULL_BUILD" = "1" ]; then EXTRA=""; fi
  echo "==== [build] bash build.sh --experimental --ops=turbo_quant_attention_score $EXTRA -j$JOBS ===="
  bash build.sh --experimental --ops=turbo_quant_attention_score $EXTRA -j"$JOBS"
fi

test -d "$PKG_ROOT" || { echo "[FATAL] 算子包不存在：$PKG_ROOT"; exit 1; }
echo "==== [env] custom op package=$PKG_ROOT ===="

# ASCEND_CUSTOM_OPP_PATH 必须指向 vendor 目录本身（不是 packages 根），
# 否则报 "binary_info_config.json does not support opType" (161001)
export ASCEND_CUSTOM_OPP_PATH="$PKG_ROOT"
export LD_LIBRARY_PATH="$PKG_ROOT/op_api/lib:$ATC_HOME/latest/lib64:${LD_LIBRARY_PATH:-}"

# ---- 2. 编译设备端自测 harness ----
cd "$REPO/$OP/examples"
echo "==== [compile] test_aclnn_turbo_quant_attention_score.cpp ===="
g++ -O2 -std=c++17 test_aclnn_turbo_quant_attention_score.cpp -o tqas_test \
    -I"$ATC_HOME/latest/include" -I"$PKG_ROOT/op_api/include" \
    -L"$ATC_HOME/latest/lib64" -L"$PKG_ROOT/op_api/lib" \
    -lascendcl -lcust_opapi -lnnopbase -lpthread

# ---- 3. 执行自验证（精度 + 性能 + 内存） ----
echo "==== [run] tqas_test ===="
./tqas_test
rc=$?
echo "==== [done] exit=$rc ===="
exit $rc
