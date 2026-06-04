#!/bin/bash
set -eo pipefail

# ===================== --help 帮助说明 =====================
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo -e "\n========== v122 Ascend ESL 仿真脚本 =========="
    echo -e "使用方法："
    echo -e "  1. 默认运行（使用脚本内置参数）："
    echo -e ""
    echo -e "  2. 自定义参数运行："
    echo -e "     [CANN_PATH] [ESL_DIR] [BIN_NAME] [SIM_ARGS]"
    echo -e ""
    echo -e "参数说明（按顺序）："
    echo -e "  1. CANN_PATH           CANN 安装路径（默认：/home/Ascend/v122_b150_patch）"
    echo -e "  3. ESL_DIR             ESL 仿真运行目录（默认：ESL）"
    echo -e "  4. BIN_NAME            编译输出可执行文件名（默认：test）"
    echo -e "  5. SIM_ARGS            算子仿真运行参数（默认：1 1 2）"
    echo -e ""
    echo -e "示例："
    echo -e "  \"\" \"\" \"\" my_test \"2 2 4\""
    echo -e "  /home/user/Ascend work esl test \"3 3 3\" Ascend910_9695"
    echo -e "==========================================================\n"
    exit 0
fi


DEFAULT_CANN_PATH="/home/Ascend/v122_b150_patch"
DEFAULT_ESL_DIR="ESL"
DEFAULT_BIN_NAME="test"
DEFAULT_SIM_ARGS="1 1 2"

# ===================== 外部参数解析 =====================
# 规则：不传参用默认，传参则覆盖
# 传参顺序：$1 CANN路径  $2 编译目录  $3 ESL目录  $4 输出文件名  $5 运行参数
export CANN_INSTALL_PATH="${1:-$DEFAULT_CANN_PATH}"
export ESL_DIR="${2:-$DEFAULT_ESL_DIR}"
export BIN_NAME="${3:-$DEFAULT_BIN_NAME}"
export SIM_ARGS="${4:-$DEFAULT_SIM_ARGS}"

# 记录脚本启动目录
SCRIPT_PWD="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ESL_PATH="${SCRIPT_PWD}/${ESL_DIR}"

# ===================== 打印最终使用参数 =====================
echo "========== v122 Ascend ESL 仿真脚本 =========="
echo "使用参数如下："
echo "CANN路径:        ${CANN_INSTALL_PATH}"
echo "编译目录:         ${WORK_DIR}"
echo "ESL目录:         ${ESL_DIR}"
echo "二进制程序名称:    ${BIN_NAME}"
echo "运行参数:         ${SIM_ARGS}"
echo "--------------------------------------------------------"

# 环境变量路径
#CANN_SETENV="${CANN_INSTALL_PATH}/latest/bin/setenv.bash"
CANN_SETENV="${CANN_INSTALL_PATH}/cann/set_env.sh"

# 加载 CANN 环境
echo "加载 CANN 环境变量"
source "${CANN_SETENV}" || true

cd "${ESL_PATH}" || echo "进入 ESL 目录失败"

# ===================== 3. 运行环境配置 =====================
echo "配置运行环境"
export LD_LIBRARY_PATH="${ESL_PATH}:${LD_LIBRARY_PATH}"
# export LD_LIBRARY_PATH="/home/g00886479/workspace/ND2NZ/esl_v121:${LD_LIBRARY_PATH}"

# 文件句柄
ulimit -n 65536

echo "执行：./${BIN_NAME} ${SIM_ARGS}"
./${BIN_NAME} ${SIM_ARGS}

# ===================== 完成 =====================
echo  "\n${GREEN}========== 仿真执行成功！ ==========${NC}"
echo "仿真目录：${ESL_PATH}"