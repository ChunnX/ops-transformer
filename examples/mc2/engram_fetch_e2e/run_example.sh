#!/bin/bash
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# the CANN Open Software License Agreement Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.hiascend.com/license/
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
# ----------------------------------------------------------------------------

# engram_fetch 端到端验证用例一键运行脚本(用法详见 bash run_example.sh --help):
#   默认以 2 张 NPU 组成 HCCL 组, 依次对 sliced(fetch_mode=0, 切片跨NPU通信) 与
#   local(fetch_mode=1, 全量表冗余纯本地取数) 两种模式执行推理+训练端到端验证。

set -o pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

FETCH_MODE="both"
NPROC=2
PYTHON=${PYTHON:-python3}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --fetch-mode)
            FETCH_MODE="$2"; shift 2 ;;
        --fetch-mode=*)
            FETCH_MODE="${1#*=}"; shift ;;
        --nproc)
            NPROC="$2"; shift 2 ;;
        --nproc=*)
            NPROC="${1#*=}"; shift ;;
        --python)
            PYTHON="$2"; shift 2 ;;
        --python=*)
            PYTHON="${1#*=}"; shift ;;
        -h|--help)
            cat <<'EOF'
用法:
  bash run_example.sh [选项] [其余参数原样透传给 test_engram_fetch_e2e.py]

选项:
  --fetch-mode sliced|local|both  待验证的取数模式, 默认 both(依次验证两种模式)
  --nproc N                       每 N 张 NPU 一个 HCCL 组, 默认 2
  --python PATH                   python 解释器, 默认 python3

其余参数(如 --phase/--dtype/--num-entries/--hidden/--num-tokens/--seed)会透传给测试脚本,
详见: python3 test_engram_fetch_e2e.py --help

示例:
  bash run_example.sh                                # 2 卡验证 sliced + local 两种模式(infer+train)
  bash run_example.sh --fetch-mode local             # 仅验证 local 模式
  bash run_example.sh --fetch-mode sliced --nproc 4 --dtype fp16 --num-entries 8192
EOF
            exit 0 ;;
        *)
            EXTRA_ARGS+=("$1"); shift ;;
    esac
done

if [[ "$FETCH_MODE" != "sliced" && "$FETCH_MODE" != "local" && "$FETCH_MODE" != "both" ]]; then
    echo "[ERROR] 无效的 --fetch-mode: ${FETCH_MODE} (可选: sliced|local|both)"
    exit 1
fi
if ! [[ "$NPROC" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] 无效的 --nproc: ${NPROC} (需为正整数)"
    exit 1
fi

echo "[INFO] python: ${PYTHON}, nproc: ${NPROC}, fetch-mode: ${FETCH_MODE}"
echo "[INFO] 透传参数: ${EXTRA_ARGS[*]:-（无）}"

# 前置依赖检查
if ! ${PYTHON} -c "import torch, torch_npu, cann_ops_transformer" 2>/dev/null; then
    echo "[ERROR] 依赖检查失败, 请确认已安装 torch / torch_npu / cann_ops_transformer(PTA whl)"
    exit 1
fi

if [[ "$FETCH_MODE" == "both" ]]; then
    MODES=("sliced" "local")
else
    MODES=("$FETCH_MODE")
fi

FAILED_MODES=()
for mode in "${MODES[@]}"; do
    echo ""
    echo "================================ [engram_fetch/${mode}] 开始端到端验证 ================================"
    if ${PYTHON} -m torch.distributed.run --standalone --nproc_per_node="${NPROC}" \
        "${SCRIPT_DIR}/test_engram_fetch_e2e.py" --fetch-mode "${mode}" "${EXTRA_ARGS[@]}"; then
        echo "================================ [engram_fetch/${mode}] 验证通过 ===================================="
    else
        echo "================================ [engram_fetch/${mode}] 验证失败 ===================================="
        FAILED_MODES+=("${mode}")
    fi
done

echo ""
if [[ ${#FAILED_MODES[@]} -eq 0 ]]; then
    echo "[SUMMARY] 全部模式验证通过: ${MODES[*]}"
    exit 0
fi
echo "[SUMMARY] 验证失败的模式: ${FAILED_MODES[*]}"
exit 1
