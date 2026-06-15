#!/bin/bash
# ============================================================
# aclnn_fused_qkv_projection 算子测试脚本
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ASCEND_PATH="${ASCEND_PATH:-/ddhome/qkv_xpb/Ascend/cann-9.1.0-beta.1}"
VENDOR_NAME="${VENDOR_NAME:-custom}"
VENDOR_DIR="${ASCEND_PATH}/opp/vendors/${VENDOR_NAME}_transformer"
TEST_BIN="/tmp/test_fused_qkv"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

pass() { echo -e "${GREEN}[PASS]${NC} $1"; }
fail() { echo -e "${RED}[FAIL]${NC} $1"; }
info() { echo -e "${YELLOW}[INFO]${NC} $1"; }

# ---- 1. 检查环境 ----
info "检查 NPU 设备..."
if [ -c /dev/davinci0 ]; then
    info "  设备 /dev/davinci0 存在"
else
    fail "  设备 /dev/davinci0 不存在"
fi

info "CANN 路径: ${ASCEND_PATH}"
info "Vendor 路径: ${VENDOR_DIR}"

if [ ! -f "${VENDOR_DIR}/op_api/lib/libcust_opapi.so" ]; then
    fail "  找不到 libcust_opapi.so，请先安装 .run 包"
fi

# ---- 2. 编译测试程序 ----
info "编译测试程序..."
g++ -std=c++17 \
    -I${ASCEND_PATH}/include \
    -I${VENDOR_DIR}/op_api/include \
    -L${ASCEND_PATH}/lib64 \
    -L${ASCEND_PATH}/aarch64-linux/lib64 \
    -L${ASCEND_PATH}/aarch64-linux/devlib \
    -L${VENDOR_DIR}/op_api/lib \
    -o ${TEST_BIN} \
    ${SCRIPT_DIR}/test_aclnn_aclnn_fused_qkv_projection.cpp \
    -Wl,--start-group \
    -lascendcl -lacl_op_compiler -lcust_opapi -lnnopbase \
    -lgraph_base -lge_common_base -lops_base -lregister \
    -Wl,--end-group \
    -Wl,-rpath,${ASCEND_PATH}/aarch64-linux/devlib:${ASCEND_PATH}/aarch64-linux/lib64:${ASCEND_PATH}/lib64:${VENDOR_DIR}/op_api/lib

RUNTIME_LD_PATH="${ASCEND_PATH}/aarch64-linux/devlib:${ASCEND_PATH}/lib64:${VENDOR_DIR}/op_api/lib"

if [ $? -eq 0 ]; then
    pass "编译成功"
else
    fail "编译失败"
    exit 1
fi

# ---- 3. 运行测试 ----
info "运行测试 (超时 30s)..."
echo "----------------------------------------"
LD_LIBRARY_PATH=${RUNTIME_LD_PATH} \
    timeout 30 ${TEST_BIN} 2>&1
EXIT_CODE=$?
echo "----------------------------------------"

if [ ${EXIT_CODE} -eq 0 ]; then
    pass "测试通过!"
elif [ ${EXIT_CODE} -eq 124 ]; then
    fail "测试超时 (30s)"
else
    fail "测试失败，退出码: ${EXIT_CODE}"
fi
