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
DTYPES=""
NPROC=2
PYTHON=${PYTHON:-python3}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --fetch-mode)
            FETCH_MODE="$2"; shift 2 ;;
        --fetch-mode=*)
            FETCH_MODE="${1#*=}"; shift ;;
        --dtypes)
            DTYPES="$2"; shift 2 ;;
        --dtypes=*)
            DTYPES="${1#*=}"; shift ;;
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
  --dtypes LIST|all               轮跑多种表数据类型(逗号分隔, 或 all=bf16,fp16,fp32,fp8e4m3,fp8e5m2),
                                  与 --fetch-mode 组成矩阵; 使用后不得再透传 --dtype
  --nproc N                       每 N 张 NPU 一个 HCCL 组, 默认 2
  --python PATH                   python 解释器, 默认 python3

其余参数(如 --phase/--dtype/--num-entries/--hidden/--num-tokens/--seed/--with-sf/--iters)会透传给测试脚本,
详见: python3 test_engram_fetch_e2e.py --help

示例:
  bash run_example.sh                                # 2 卡验证 sliced + local 两种模式(infer+train)
  bash run_example.sh --fetch-mode local             # 仅验证 local 模式
  bash run_example.sh --fetch-mode sliced --nproc 4 --dtype fp16 --num-entries 8192
  bash run_example.sh --dtypes all                   # 轮跑全部 dtype(bf16/fp16/fp32/fp8e4m3/fp8e5m2)
  bash run_example.sh --dtypes bf16,fp8e4m3 --with-sf   # 轮跑 bf16+fp32sf 与 fp8+E8M0sf 链路
  bash run_example.sh --fetch-mode both --phase perf --num-entries 513800 --hidden 576 \
      --num-tokens 8192 --iters 20                   # 客户规格(51.38万词表×576维)取数性能测试
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
if [[ -n "$DTYPES" ]]; then
    if [[ "$DTYPES" == "all" ]]; then
        DTYPES="bf16,fp16,fp32,fp8e4m3,fp8e5m2"
    fi
    for arg in "${EXTRA_ARGS[@]}"; do
        if [[ "$arg" == "--dtype" || "$arg" == --dtype=* ]]; then
            echo "[ERROR] 同时指定 --dtypes 与透传 --dtype, 请二选一"
            exit 1
        fi
    done
    IFS=',' read -r -a DTYPE_LIST <<< "$DTYPES"
else
    DTYPE_LIST=("")
fi

echo "[INFO] python: ${PYTHON}, nproc: ${NPROC}, fetch-mode: ${FETCH_MODE}, dtypes: ${DTYPES:-（单dtype, 见透传参数）}"
echo "[INFO] 透传参数: ${EXTRA_ARGS[*]:-（无）}"

if ! ${PYTHON} -c "import torch, torch_npu, cann_ops_transformer" 2>/dev/null; then
    echo "[ERROR] 依赖检查失败, 请确认已安装 torch / torch_npu / cann_ops_transformer(PTA whl)"
    exit 1
fi

if [[ "$FETCH_MODE" == "both" ]]; then
    MODES=("sliced" "local")
else
    MODES=("$FETCH_MODE")
fi

FAILED_CASES=()
for mode in "${MODES[@]}"; do
    for dtype in "${DTYPE_LIST[@]}"; do
        label="${mode}"
        dtype_args=()
        if [[ -n "$dtype" ]]; then
            label="${mode}/${dtype}"
            dtype_args=(--dtype "$dtype")
        fi
        echo ""
        echo "================================ [engram_fetch/${label}] 开始端到端验证 ================================"
        if ${PYTHON} -m torch.distributed.run --standalone --nproc_per_node="${NPROC}" \
            "${SCRIPT_DIR}/test_engram_fetch_e2e.py" --fetch-mode "${mode}" "${dtype_args[@]}" "${EXTRA_ARGS[@]}"; then
            echo "================================ [engram_fetch/${label}] 验证通过 ===================================="
        else
            echo "================================ [engram_fetch/${label}] 验证失败 ===================================="
            FAILED_CASES+=("${label}")
        fi
    done
done

echo ""
if [[ ${#FAILED_CASES[@]} -eq 0 ]]; then
    echo "[SUMMARY] 全部用例验证通过"
    exit 0
fi
echo "[SUMMARY] 验证失败的用例: ${FAILED_CASES[*]}"
exit 1
