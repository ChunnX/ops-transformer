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
#
# build_opp.sh: CANN 官方标准自定义 OPP 算子构建与出包工具
# -----------------------------------------------------------------------------------------------------------

set -euo pipefail

OP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${OP_DIR}/../../.." && pwd)"

# 默认构建配置
SOC="ascend950"
OUTPUT_DIR="${OP_DIR}/build/opp"
DRY_RUN=false
MOCK_MODE=false
CLEAN_BUILD=false
VERBOSE=false
EXTRA_ARGS=()

# 颜色控制
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

print_usage() {
    echo "Usage: $0 [options] [-- <extra CANN build.sh args>]"
    echo ""
    echo "Options:"
    echo "  --soc=<soc>              Target SoC version (default: ascend950)"
    echo "  --output-dir=<path>      Output package directory (default: build/opp)"
    echo "  --dry-run                Perform Dry-Run without requiring NPU hardware or CANN toolkit"
    echo "  --mock                   Compile local Mock OPP package (equivalent to --dry-run)"
    echo "  --clean                  Clean output/build directories before building"
    echo "  --verbose                Enable verbose build logging"
    echo "  -h, --help               Display this help message"
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        --soc=*)
            SOC="${1#*=}"
            shift
            ;;
        --output-dir=*)
            OUTPUT_DIR="${1#*=}"
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            MOCK_MODE=true
            shift
            ;;
        --mock)
            MOCK_MODE=true
            shift
            ;;
        --clean)
            CLEAN_BUILD=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        --)
            shift
            EXTRA_ARGS+=("$@")
            break
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

echo -e "${CYAN}================================================================================${NC}"
echo -e "${CYAN}SageAttention2 CANN Custom OPP Package Builder (build_opp.sh)${NC}"
echo -e "${CYAN}Operator directory: ${OP_DIR}${NC}"
echo -e "${CYAN}Target SoC:         ${SOC}${NC}"
echo -e "${CYAN}Output directory:   ${OUTPUT_DIR}${NC}"
echo -e "${CYAN}================================================================================${NC}"

if [ "${CLEAN_BUILD}" = true ]; then
    echo -e "${BLUE}>>> Cleaning build and output directories...${NC}"
    rm -rf "${OUTPUT_DIR}"
fi

# 检查 CANN 环境
CANN_ROOT="${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}"
if [ ! -d "${CANN_ROOT}" ] && [ -d "/usr/local/Ascend/cann" ]; then
    CANN_ROOT="/usr/local/Ascend/cann"
fi

HAS_CANN=false
HAS_BISHENG=false

if [ -f "${CANN_ROOT}/set_env.sh" ]; then
    HAS_CANN=true
fi

if [ -n "${ASCEND_HOME_PATH:-}" ] && [ -x "${ASCEND_HOME_PATH}/bin/bisheng" ]; then
    HAS_BISHENG=true
elif command -v bisheng >/dev/null 2>&1; then
    HAS_BISHENG=true
fi

# 如果未强制指定 --mock/--dry-run，且环境具备 CANN 与 BiSheng，走官方出包流程
if [ "${MOCK_MODE}" = false ] && [ "${DRY_RUN}" = false ] && [ "${HAS_CANN}" = true ] && [ "${HAS_BISHENG}" = true ]; then
    echo -e "${GREEN}>>> Active CANN & BiSheng environment detected. Running official vendor packaging...${NC}"
    set +u
    source "${CANN_ROOT}/set_env.sh"
    set -u

    export CATLASS_ROOT="${CATLASS_ROOT:-${REPO_DIR}/../catlass}"
    if [ ! -d "${CATLASS_ROOT}" ] && [ -d "${REPO_DIR}/third_party/catlass" ]; then
        export CATLASS_ROOT="${REPO_DIR}/third_party/catlass"
    fi

    cd "${REPO_DIR}"
    echo "Running: bash build.sh --pkg --experimental --soc=${SOC} --ops=sageattention2 ${EXTRA_ARGS[*]:-}"
    bash build.sh --pkg --experimental --soc="${SOC}" --ops=sageattention2 "${EXTRA_ARGS[@]}"

    # 安装/展开 run 包至指定 output_dir
    # （此处位于脚本顶层而非函数内，bash 的 local 在函数外直接报错退出，故用普通变量）
    RUN_PKG=$(find "${REPO_DIR}/build_out" -name "cann-ops-transformer-custom_*.run" | head -n 1 || true)
    if [ -n "${RUN_PKG}" ] && [ -f "${RUN_PKG}" ]; then
        echo -e "${GREEN}>>> Unpacking ${RUN_PKG} into ${OUTPUT_DIR}...${NC}"
        bash "${RUN_PKG}" --quiet --install-path="${OUTPUT_DIR}"
    fi

    echo -e "${GREEN}>>> Official CANN OPP Build & Packaging Completed Successfully!${NC}"
    exit 0
fi

# 无卡环境、本地开发或显式指定 --dry-run/--mock: 执行高拟真 Mock 编译与标准 Vendor 目录封包
echo -e "${YELLOW}>>> Running in Local Dry-Run / Mock Build Mode...${NC}"

# 检测本地 C++ 编译器
CXX_COMPILER="g++"
if command -v clang++ >/dev/null 2>&1; then
    CXX_COMPILER="clang++"
fi
echo -e "Using Host C++ Compiler: ${BLUE}${CXX_COMPILER}${NC}"

# 规划官方标准自定义 OPP 目录结构
# 规范结构参见 CANN 自定义算子开发指南：
# vendors/
#   config.ini
#   customize/
#     op_api/lib/libcust_opapi.so
#     op_api/include/aclnn_sage_attention2.h
#     op_proto/inc/sage_attention2_def.h
#     op_proto/lib/libcust_opapi.so
#     op_impl/ai_core/tbe/config/${SOC}/sageattention2.ini
#     op_impl/ai_core/tbe/kernel/${SOC}/sageattention2.o
#     op_impl/ai_core/tbe/op_tiling/lib/liboptiling.so
#     bin/set_env.bash

VENDORS_DIR="${OUTPUT_DIR}/vendors"
CUSTOM_DIR="${VENDORS_DIR}/customize"

mkdir -p "${CUSTOM_DIR}/op_api/lib"
mkdir -p "${CUSTOM_DIR}/op_api/include"
mkdir -p "${CUSTOM_DIR}/op_proto/inc"
mkdir -p "${CUSTOM_DIR}/op_proto/lib"
mkdir -p "${CUSTOM_DIR}/op_impl/ai_core/tbe/config/${SOC}"
mkdir -p "${CUSTOM_DIR}/op_impl/ai_core/tbe/kernel/${SOC}"
mkdir -p "${CUSTOM_DIR}/op_impl/ai_core/tbe/op_tiling/lib"
mkdir -p "${CUSTOM_DIR}/bin"

echo -e "${BLUE}1. Compiling op_host dynamic libraries (liboptiling.so & libcust_opapi.so)...${NC}"

SHARED_FLAGS="-shared -fPIC -O2"
if [[ "$OSTYPE" == "darwin"* ]]; then
    SHARED_FLAGS="-shared -fPIC -O2 -undefined dynamic_lookup"
fi

# 编译 libcust_opapi.so (包含 ACLNN Host 接口、量化核以及原型推导)
${CXX_COMPILER} -std=c++17 ${SHARED_FLAGS} \
    "${OP_DIR}/aclnn/sage_attn_host.cpp" \
    "${OP_DIR}/quant/quant_int8.cpp" \
    "${OP_DIR}/quant/quant_fp8.cpp" \
    "${OP_DIR}/op_host/sage_attention2_def.cpp" \
    "${OP_DIR}/op_host/sage_attention2_infershape.cpp" \
    -I"${OP_DIR}/op_host" \
    -I"${OP_DIR}/aclnn" \
    -I"${OP_DIR}/quant" \
    -I"${OP_DIR}/sageattn_fp8/ascendc" \
    -I"${OP_DIR}/sageattn_fp8/catlass" \
    -o "${CUSTOM_DIR}/op_api/lib/libcust_opapi.so"

# 软链接或同步一份到 op_proto/lib
cp -f "${CUSTOM_DIR}/op_api/lib/libcust_opapi.so" "${CUSTOM_DIR}/op_proto/lib/libcust_opapi.so"

# 编译 liboptiling.so (包含算子 Tiling 算法与 InferShape)
${CXX_COMPILER} -std=c++17 ${SHARED_FLAGS} \
    "${OP_DIR}/op_host/sage_attention2_def.cpp" \
    "${OP_DIR}/op_host/sage_attention2_infershape.cpp" \
    "${OP_DIR}/aclnn/sage_attn_host.cpp" \
    "${OP_DIR}/quant/quant_int8.cpp" \
    "${OP_DIR}/quant/quant_fp8.cpp" \
    -I"${OP_DIR}/op_host" \
    -I"${OP_DIR}/aclnn" \
    -I"${OP_DIR}/quant" \
    -I"${OP_DIR}/sageattn_fp8/ascendc" \
    -I"${OP_DIR}/sageattn_fp8/catlass" \
    -o "${CUSTOM_DIR}/op_impl/ai_core/tbe/op_tiling/lib/liboptiling.so"

# 如果是 macOS，同时生成对应名称或软连接以保证平台兼容性
if [[ "$OSTYPE" == "darwin"* ]]; then
    cp -f "${CUSTOM_DIR}/op_api/lib/libcust_opapi.so" "${CUSTOM_DIR}/op_api/lib/libcust_opapi.dylib" || true
    cp -f "${CUSTOM_DIR}/op_impl/ai_core/tbe/op_tiling/lib/liboptiling.so" "${CUSTOM_DIR}/op_impl/ai_core/tbe/op_tiling/lib/liboptiling.dylib" || true
fi

echo -e "${BLUE}2. Compiling op_kernel object binary for ${SOC}...${NC}"
${CXX_COMPILER} -std=c++17 -c \
    "${OP_DIR}/op_kernel/sageattention2.cpp" \
    -I"${OP_DIR}/sageattn_fp8/ascendc" \
    -I"${OP_DIR}/sageattn_fp8/catlass" \
    -I"${OP_DIR}/quant" \
    -o "${CUSTOM_DIR}/op_impl/ai_core/tbe/kernel/${SOC}/sageattention2.o"

echo -e "${BLUE}3. Packaging headers and configuration files...${NC}"

# 安装头文件
if [ -f "${OP_DIR}/op_api/aclnn_sage_attention2.h" ]; then
    cp -f "${OP_DIR}/op_api/aclnn_sage_attention2.h" "${CUSTOM_DIR}/op_api/include/"
fi
cp -f "${OP_DIR}/aclnn/sage_attn_proto.h" "${CUSTOM_DIR}/op_api/include/"
cp -f "${OP_DIR}/op_host/sage_attention2_def.h" "${CUSTOM_DIR}/op_proto/inc/"

# 生成 vendors/config.ini
cat << 'EOF' > "${VENDORS_DIR}/config.ini"
[common]
load_priority=customize
EOF

# 生成 op_impl 算子配置文件
cat << EOF > "${CUSTOM_DIR}/op_impl/ai_core/tbe/config/${SOC}/sageattention2.ini"
[SageAttention2]
opFile.value=sageattention2
opInterface.value=sage_attention2
dynamicFormat.flag=true
dynamicRankSupport.flag=true
dynamicShapeSupport.flag=true
needCheckSupport.flag=false
precisionReduce.flag=true
soc.version=${SOC}
deterministic.flag=false
EOF

# 生成环境变量导入脚本 set_env.bash
cat << 'EOF' > "${CUSTOM_DIR}/bin/set_env.bash"
#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUSTOM_VENDOR_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export ASCEND_CUSTOM_OPP_PATH="${CUSTOM_VENDOR_DIR}:${ASCEND_CUSTOM_OPP_PATH:-}"
export LD_LIBRARY_PATH="${CUSTOM_VENDOR_DIR}/customize/op_api/lib:${CUSTOM_VENDOR_DIR}/customize/op_impl/ai_core/tbe/op_tiling/lib:${LD_LIBRARY_PATH:-}"
EOF
chmod +x "${CUSTOM_DIR}/bin/set_env.bash"

# 4. 验证打包产物完整性
echo -e "${BLUE}4. Verifying package integrity and layout...${NC}"
test -f "${VENDORS_DIR}/config.ini" || { echo -e "${RED}[ERROR] config.ini missing!${NC}"; exit 1; }
test -f "${CUSTOM_DIR}/op_api/lib/libcust_opapi.so" || { echo -e "${RED}[ERROR] libcust_opapi.so missing!${NC}"; exit 1; }
test -f "${CUSTOM_DIR}/op_impl/ai_core/tbe/op_tiling/lib/liboptiling.so" || { echo -e "${RED}[ERROR] liboptiling.so missing!${NC}"; exit 1; }
test -f "${CUSTOM_DIR}/op_impl/ai_core/tbe/kernel/${SOC}/sageattention2.o" || { echo -e "${RED}[ERROR] sageattention2.o missing!${NC}"; exit 1; }
test -f "${CUSTOM_DIR}/op_impl/ai_core/tbe/config/${SOC}/sageattention2.ini" || { echo -e "${RED}[ERROR] sageattention2.ini missing!${NC}"; exit 1; }

echo -e "${GREEN}================================================================================${NC}"
echo -e "${GREEN}>>> CANN Custom OPP Package Successfully Built!${NC}"
echo -e "${GREEN}Package location: ${OUTPUT_DIR}/vendors${NC}"
echo -e "${GREEN}Verified artifacts:${NC}"
echo -e "  - ${OUTPUT_DIR}/vendors/config.ini"
echo -e "  - ${OUTPUT_DIR}/vendors/customize/op_api/lib/libcust_opapi.so"
echo -e "  - ${OUTPUT_DIR}/vendors/customize/op_impl/ai_core/tbe/op_tiling/lib/liboptiling.so"
echo -e "  - ${OUTPUT_DIR}/vendors/customize/op_impl/ai_core/tbe/kernel/${SOC}/sageattention2.o"
echo -e "  - ${OUTPUT_DIR}/vendors/customize/op_impl/ai_core/tbe/config/${SOC}/sageattention2.ini"
echo -e "  - ${OUTPUT_DIR}/vendors/customize/bin/set_env.bash"
echo -e "${GREEN}================================================================================${NC}"

exit 0
