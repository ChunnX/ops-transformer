#!/usr/bin/env bash
# build_opp.sh: CANN 标准自定义 OPP 算子 vendor 树构建（BiSheng 容器内执行）
#
# 全部产物由 BiSheng 编译：op_host/op_api/quant 的 host 动态库走 BiSheng 宿主模式，
# 设备核走 BiSheng 交叉编译（ascend950 → dav-3510），不存在任何宿主编译器代偿路径。
# 运行环境：算子门禁容器（experimental/attention/sageattention2/docker/），
# 即 run.sh opp 的实现体；交付态整仓 run 包另由仓库根 build.sh 出包（见 README）。

set -euo pipefail

OP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${OP_DIR}/build/opp"
SOC="ascend950"            # 官方 vendor 目录/ini 的 SoC 标识
NPU_ARCH="dav-3510"        # BiSheng 设备架构（ascend950 对应 dav-3510）
CLEAN_BUILD=false
VERBOSE=false
EXTRA_ARGS=()

print_usage() {
    echo "Usage: $0 [options] [-- <extra args>]"
    echo ""
    echo "Options:"
    echo "  --soc=<soc>              Target SoC version (default: ascend950; BiSheng arch dav-3510)"
    echo "  --output-dir=<path>      Output package directory (default: build/opp)"
    echo "  --clean                  Clean output directory before building"
    echo "  --verbose                Enable verbose build logging"
    echo "  -h, --help               Display this help message"
}

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

echo "================================================================================"
echo "SageAttention2 CANN Custom OPP Package Builder (BiSheng, container-only)"
echo "Operator directory: ${OP_DIR}"
echo "Target SoC:         ${SOC} (npu-arch ${NPU_ARCH})"
echo "Output directory:   ${OUTPUT_DIR}"
echo "================================================================================"

# BiSheng / CANN 环境是硬前置（容器内由 run.sh 注入；裸宿主直接报错，不做代偿编译）
if [ -z "${ASCEND_HOME_PATH:-}" ]; then
    # shellcheck source=/dev/null
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
BISHENG="${ASCEND_HOME_PATH}/bin/bisheng"
if [ ! -x "${BISHENG}" ]; then
    if command -v bisheng >/dev/null 2>&1; then
        BISHENG="bisheng"
    else
        echo "[ERROR] BiSheng compiler not found (ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-unset})." >&2
        echo "        请在算子门禁容器内执行（bash run.sh opp），或 source CANN set_env.sh。" >&2
        exit 1
    fi
fi

# CATLASS 解析（唯一定义点与 run.sh 容器侧一致：CATLASS_ROOT 覆盖 → init.sh 置备位置）
CATLASS_ROOT="${CATLASS_ROOT:-${OP_DIR}/docker/third_party/catlass}"
CATLASS_DIR="${CATLASS_ROOT}/include"
if [ ! -d "${CATLASS_DIR}" ]; then
    echo "[ERROR] CATLASS include directory not found: ${CATLASS_DIR} (export CATLASS_ROOT to override)." >&2
    echo "        请先执行 bash docker/init.sh 置备 CATLASS。" >&2
    exit 1
fi

if [ "${CLEAN_BUILD}" = true ]; then
    echo ">>> Cleaning output directory..."
    rm -rf "${OUTPUT_DIR}"
fi

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

HOST_CXX_FLAGS=(-std=c++17 -O2 -fPIC -Wall -Wextra)
COMMON_INCLUDES=(
    -I"${OP_DIR}/op_host"
    -I"${OP_DIR}/aclnn"
    -I"${OP_DIR}/quant"
    -I"${OP_DIR}/sageattn_fp8/ascendc"
    -I"${OP_DIR}/sageattn_fp8/catlass"
    -I"${CATLASS_DIR}"
)
[ "${VERBOSE}" = true ] && HOST_CXX_FLAGS+=(-v)

QUANT_SOURCES=("${OP_DIR}/quant/quant_int8.cpp" "${OP_DIR}/quant/quant_fp8.cpp")

echo "1. Compiling op_host dynamic libraries with BiSheng (libcust_opapi.so & liboptiling.so)..."

# libcust_opapi.so (包含 ACLNN Host 接口、量化核以及原型推导)
"${BISHENG}" "${HOST_CXX_FLAGS[@]}" -shared \
    "${OP_DIR}/aclnn/sage_attn_host.cpp" \
    "${QUANT_SOURCES[@]}" \
    "${OP_DIR}/op_host/sage_attention2_def.cpp" \
    "${OP_DIR}/op_host/sage_attention2_infershape.cpp" \
    "${COMMON_INCLUDES[@]}" \
    -o "${CUSTOM_DIR}/op_api/lib/libcust_opapi.so"

# 同步一份到 op_proto/lib
cp -f "${CUSTOM_DIR}/op_api/lib/libcust_opapi.so" "${CUSTOM_DIR}/op_proto/lib/libcust_opapi.so"

# liboptiling.so (包含算子 Tiling 算法与 InferShape)
"${BISHENG}" "${HOST_CXX_FLAGS[@]}" -shared \
    "${OP_DIR}/op_host/sage_attention2_def.cpp" \
    "${OP_DIR}/op_host/sage_attention2_infershape.cpp" \
    "${OP_DIR}/aclnn/sage_attn_host.cpp" \
    "${QUANT_SOURCES[@]}" \
    "${COMMON_INCLUDES[@]}" \
    -o "${CUSTOM_DIR}/op_impl/ai_core/tbe/op_tiling/lib/liboptiling.so"

echo "2. Compiling op_kernel device object for ${SOC} with BiSheng (-xasc ${NPU_ARCH})..."
"${BISHENG}" --npu-arch="${NPU_ARCH}" -std=c++17 -xasc -c \
    "${OP_DIR}/op_kernel/sageattention2.cpp" \
    -I"${ASCEND_HOME_PATH}/include" \
    -I"${OP_DIR}/sageattn_fp8/ascendc" \
    -I"${OP_DIR}/sageattn_fp8/catlass" \
    -I"${OP_DIR}/quant" \
    -I"${CATLASS_DIR}" \
    -o "${CUSTOM_DIR}/op_impl/ai_core/tbe/kernel/${SOC}/sageattention2.o"

echo "3. Packaging headers and configuration files..."

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
echo "4. Verifying package integrity and layout..."
test -f "${VENDORS_DIR}/config.ini" || { echo "[ERROR] config.ini missing!"; exit 1; }
test -f "${CUSTOM_DIR}/op_api/lib/libcust_opapi.so" || { echo "[ERROR] libcust_opapi.so missing!"; exit 1; }
test -f "${CUSTOM_DIR}/op_impl/ai_core/tbe/op_tiling/lib/liboptiling.so" || { echo "[ERROR] liboptiling.so missing!"; exit 1; }
test -f "${CUSTOM_DIR}/op_impl/ai_core/tbe/kernel/${SOC}/sageattention2.o" || { echo "[ERROR] sageattention2.o missing!"; exit 1; }
test -f "${CUSTOM_DIR}/op_impl/ai_core/tbe/config/${SOC}/sageattention2.ini" || { echo "[ERROR] sageattention2.ini missing!"; exit 1; }

echo "================================================================================"
echo ">>> CANN Custom OPP Package Successfully Built (all artifacts by BiSheng)!"
echo "Package location: ${OUTPUT_DIR}/vendors"
echo "Verified artifacts:"
echo "  - ${OUTPUT_DIR}/vendors/config.ini"
echo "  - ${OUTPUT_DIR}/vendors/customize/op_api/lib/libcust_opapi.so"
echo "  - ${OUTPUT_DIR}/vendors/customize/op_impl/ai_core/tbe/op_tiling/lib/liboptiling.so"
echo "  - ${OUTPUT_DIR}/vendors/customize/op_impl/ai_core/tbe/kernel/${SOC}/sageattention2.o"
echo "  - ${OUTPUT_DIR}/vendors/customize/op_impl/ai_core/tbe/config/${SOC}/sageattention2.ini"
echo "  - ${OUTPUT_DIR}/vendors/customize/bin/set_env.bash"
echo "================================================================================"

exit 0
