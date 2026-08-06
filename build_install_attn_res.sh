#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# build_install_attn_res.sh — 一键编译安装 attn_res_fwd / attention_res_grad 的 torch 插件与算子包
#
# 用法:
#   bash build_install_attn_res.sh [安装路径] [--no-test] [--soc=ascend910b]
#     安装路径默认: 当前脚本目录下的 install/
#     --no-test: 安装完成后不执行精度测试（默认执行 BF16 精度测试）
#     --soc=xxx: 指定算子包编译 SoC（默认自动检测本机芯片，如 ascend910b/ascend910_93/ascend950）
#
# 流程:
#   1. 编译并安装 torch_extension wheel（cann_ops_transformer，使用当前 python 环境）
#   2. 编译正反向算子包（attn_res_fwd, attention_res_grad，SoC 自动检测或 --soc 指定）
#   3. 安装算子包到指定路径
#   4. source 该路径下 set_env.bash，设置 ASCEND_CUSTOM_OPP_PATH / LD_LIBRARY_PATH
#   5. 执行精度测试（attention/attn_res_fwd/test_attn_res_bound.py，可加 --no-test 跳过）
#
# 注意:
#   - 算子包构建期间请勿 source workspace CANN set_env.sh（beta.3 缺 asc/impl/tensor_api）
#   - 精度测试运行前会自动 source workspace CANN set_env.sh
# -----------------------------------------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPS_ROOT="${SCRIPT_DIR}"     # ops-transformer
TORCH_EXT_DIR="${OPS_ROOT}/torch_extension"
OPS="attn_res_fwd,attention_res_grad"
WORKSPACE_CANN_ENV="/workspace/z00889690/Ascend_env/cann-9.1.0-beta.3/set_env.sh"

# 自动检测本机芯片并映射到 build.sh 支持的 SoC（可被 --soc= 覆盖）
detect_soc() {
    local chip
    chip=$(npu-smi info 2>/dev/null | grep -oE '910B[0-9]+|910_9[0-9]+|950[A-Za-z0-9]*' | head -1)
    case "${chip}" in
        910B*) echo "ascend910b" ;;
        910_9*) echo "ascend910_93" ;;
        950*) echo "ascend950" ;;
        *) echo "" ;;
    esac
}

RUN_TEST=1
INSTALL_PATH=""
SOC=""
for arg in "$@"; do
    case "${arg}" in
        --no-test) RUN_TEST=0 ;;
        --soc=*) SOC="${arg#--soc=}" ;;
        *) INSTALL_PATH="${arg}" ;;
    esac
done
INSTALL_PATH="${INSTALL_PATH:-${SCRIPT_DIR}/install}"

if [[ -z "${SOC}" ]]; then
    SOC="$(detect_soc)"
    if [[ -n "${SOC}" ]]; then
        echo "检测到本机芯片: $(npu-smi info 2>/dev/null | grep -oE '910B[0-9]+|910_9[0-9]+|950[A-Za-z0-9]*' | head -1) -> --soc=${SOC}"
    else
        SOC="ascend910b"
        echo "未检测到 NPU 芯片信息，使用默认 --soc=${SOC}"
    fi
else
    echo "使用指定的 --soc=${SOC}"
fi

echo "=== [1/4] 编译并安装 torch_extension ==="
cd "${TORCH_EXT_DIR}"
python3 -c "import build" 2>/dev/null || python3 -m pip install build
rm -rf build dist
python3 -m build --wheel -n
python3 -m pip install dist/*.whl --force-reinstall --no-deps
echo "cann_ops_transformer 安装完成: $(python3 -c 'import cann_ops_transformer; print(cann_ops_transformer.__file__)')"

echo "=== [2/4] 编译正反向算子包 (${OPS}, --soc=${SOC}) ==="
cd "${OPS_ROOT}"
bash build.sh --pkg --ops="${OPS}" --soc="${SOC}"

echo "=== [3/4] 安装算子包到 ${INSTALL_PATH} 并设置环境变量 ==="
RUN_FILE="${OPS_ROOT}/build/cann-ops-transformer-custom_linux-aarch64.run"
if [[ ! -f "${RUN_FILE}" ]]; then
    echo "ERROR: 未找到算子包 ${RUN_FILE}" >&2
    exit 1
fi
mkdir -p "${INSTALL_PATH}"
"${RUN_FILE}" --install-path="${INSTALL_PATH}" --force

SET_ENV="${INSTALL_PATH}/vendors/custom_transformer/bin/set_env.bash"
if [[ ! -f "${SET_ENV}" ]]; then
    echo "ERROR: 未找到 ${SET_ENV}" >&2
    exit 1
fi
set +u   # set_env.bash 内部可能引用未定义变量，与 set -u 冲突
source "${SET_ENV}"
set -u

if [[ "${RUN_TEST}" -eq 1 ]]; then
    echo ""
    echo "=== [4/4] 执行精度测试 (BF16 H=7168 seed=0) ==="
    if [[ -f "${WORKSPACE_CANN_ENV}" ]]; then
        # 构建阶段不能 source（缺 tensor_api），测试运行阶段需要完整 CANN 环境
        set +u
        source "${WORKSPACE_CANN_ENV}"
        set -u
    else
        echo "WARNING: 未找到 workspace CANN 环境 ${WORKSPACE_CANN_ENV}，测试可能需要现有环境变量"
    fi
    cd "${OPS_ROOT}/attention/attn_res_fwd"
    if ! python3 -c "import numpy, torch, torch_npu, cann_ops_transformer" 2>/dev/null; then
        echo "WARNING: 当前 python3 缺少测试依赖 (numpy/torch/torch_npu/cann_ops_transformer)，跳过精度测试"
        echo "         请确保在正确的 python 环境（如 conda kdy）下运行本脚本"
    else
        python3 test_attn_res_bound.py --dtype bf16 --H 7168 --seed 0
    fi
else
    echo ""
    echo "=== [4/4] 跳过精度测试 (--no-test) ==="
fi

echo ""
echo "==================== 完成 ===================="
echo "torch 插件已安装: cann_ops_transformer"
echo "算子包已安装到:   ${INSTALL_PATH}"
echo "ASCEND_CUSTOM_OPP_PATH=${ASCEND_CUSTOM_OPP_PATH:-}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
echo "运行前建议 source:"
echo "  source ${SET_ENV}"
echo "==============================================="
