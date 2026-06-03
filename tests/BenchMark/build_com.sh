#!/bin/bash
set -eo pipefail

# ===================== --help 帮助说明 =====================
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo -e "\n========== Ascend 算子编译脚本 =========="
    echo -e "使用方法："
    echo -e "  1. 默认运行（使用脚本内置参数）："
    echo -e ""
    echo -e "  2. 自定义参数运行："
    echo -e "     [CANN_PATH] [WORK_DIR] [CCE_FILES] [BIN_NAME] [TYPE] [AICORE_ARCH]"
    echo -e ""
    echo -e "参数说明（按顺序）："
    echo -e "  1. CANN_PATH     CANN 安装路径（默认：/home/00949060/Ascend/v122_b150_patch）"
    echo -e "  2. WORK_DIR      算子编译工作目录（默认：v122_WORK）"
    echo -e "  3. CCE_FILES     待编译的 CCE 文件（多个用空格分隔，默认：main.cce DoExpTest.cce）"
    echo -e "  4. BIN_NAME      编译输出可执行文件名（默认：test）"
    echo -e "  5. TYPE          算子类型：vec、cube（默认：vec）"
    echo -e "  6. AICORE_ARCH   SOC VERSION（默认：920r1）"
    echo -e ""
    echo -e "示例："
    echo -e "  \"\" \"\" \"my_test\" \"main.cce\""
    echo -e "  /home/user/Ascend work \"a.cce b.cce\" test vec 920r1"
    echo -e "==========================================================\n"
    exit 0
fi

######################################################################################
# 【全局可配置参数 - 默认值】
# 支持外部传参覆盖：./build_com.sh [CANN_PATH] [WORK_DIR] [CCE_FILES] [BIN_NAME] [TYPE] [AICORE_ARCH]
######################################################################################
# 默认值定义
DEFAULT_CANN_PATH="/home/g00886479/workspace/cann/1215_B150/"
DEFAULT_WORK_DIR=$(pwd)
DEFAULT_CCE_FILES="main.cce DoExpTest.cce"
DEFAULT_BIN_NAME="test"
DEFAULT_TYPE="vec"
DEFAULT_AICORE_ARCH="920r1"

# ===================== 外部参数解析 =====================
# 规则：不传参用默认，传参则覆盖
# 传参顺序：$1 CANN路径  $2 编译目录  $3 CCE文件  $4 输出文件名  $5 算子类型  $6 芯片版本
export CANN_INSTALL_PATH="${1:-$DEFAULT_CANN_PATH}"
export WORK_DIR="${2:-$DEFAULT_WORK_DIR}"
export CCE_FILES="${3:-$DEFAULT_CCE_FILES}"
export BIN_NAME="${4:-$DEFAULT_BIN_NAME}"
export TYPE="${5:-$DEFAULT_TYPE}"
export AICORE_ARCH="${6:-$DEFAULT_AICORE_ARCH}"

# 环境变量路径
# CANN_SETENV="${CANN_INSTALL_PATH}/latest/bin/setenv.bash"
CANN_SETENV="${CANN_INSTALL_PATH}/cann/set_env.sh"
######################################################################################

# 彩色日志输出
RED="\033[31m"
GREEN="\033[32m"
YELLOW="\033[33m"
BLUE="\033[34m"
NC="\033[0m"

info() { echo -e "${GREEN}[INFO] $*${NC}"; }
warn() { echo -e "${YELLOW}[WARN] $*${NC}"; }
error() { echo -e "${RED}[ERROR] $*${NC}"; exit 1; }

# 启动目录
COMPILE_PATH="${WORK_DIR}"

# ===================== 打印最终使用参数 =====================
info "========== Ascend 算子编译脚本 =========="
info "使用参数如下："
info "CANN路径:    ${CANN_INSTALL_PATH}"
info "编译目录:    ${WORK_DIR}"
info "编译文件:    ${CCE_FILES}"
info "输出程序:    ${BIN_NAME}"
info "算子类型:    ${TYPE}"
info "芯片版本:    ${AICORE_ARCH}"
echo "--------------------------------------------------------"

# ===================== 1. 前置检查 =====================
[ -d "${CANN_INSTALL_PATH}" ] || error "CANN 路径不存在：${CANN_INSTALL_PATH}"
[ -f "${CANN_SETENV}" ] || error "CANN 环境脚本不存在"
[ -n "${CCE_FILES}" ] || error "未指定待编译的 CCE 文件"

# ===================== 2. 编译算子 =====================
info  "\n${BLUE}========== 开始编译算子 ==========${NC}"
#mkdir -p "${COMPILE_PATH}"
cd "${COMPILE_PATH}" || error "进入编译目录失败"

# 加载 CANN 环境
info "加载 CANN 环境变量"
source "${CANN_SETENV}" || true

# 算子编译
info "编译中：${CCE_FILES} -> ${BIN_NAME}"
bisheng -O2 -std=c++17 --cce-aicore-arch=dav-${AICORE_ARCH}-${TYPE} \
-I ${DEFAULT_WORK_DIR}/common/  \
-I ${CANN_INSTALL_PATH}/cann/x86_64-linux/include/ \
-I ${CANN_INSTALL_PATH}/cann/x86_64-linux/ascendc/include/basic_api/impl \
-I ${CANN_INSTALL_PATH}/cann/x86_64-linux/ascendc/include/basic_api \
-I ${CANN_INSTALL_PATH}/cann/x86_64-linux/asc/include/basic_api \
-I ${CANN_INSTALL_PATH}/cann/x86_64-linux/asc \
-I ${CANN_INSTALL_PATH}/cann/x86_64-linux/asc/include \
-L ${CANN_INSTALL_PATH}/cann/x86_64-linux/lib64/ \
-lascendc_runtime \
-lascendcl \
-lruntime \
-lregister \
-lerror_manager \
-lprofapi \
-lunified_dlog \
-lmmpa \
-lascend_dump \
-ltiling_api \
-lplatform \
-ldl \
-lc_sec \
-lstdc++ \
${CCE_FILES} -o "${BIN_NAME}"

info "编译完成：${COMPILE_PATH}/${BIN_NAME}"

# ===================== 完成 =====================
info  "\n${GREEN}========== 编译成功！ ==========${NC}"
info "编译目录：${COMPILE_PATH}"
info "程序输出：${BIN_NAME}"