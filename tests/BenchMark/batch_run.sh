#!/bin/bash
set -eo pipefail

# ===================== 帮助文档函数 =====================
show_help() {
    cat << EOF
用法: $0 [模式] [参数...]

简介:
    用于在不同运行模式下自动切换文件后缀并执行编译/仿真/测试流程。
    支持 ESL 仿真模式

模式选择:
    esl                     ESL 仿真模式
    -h, --help              显示此帮助信息

模式详细说明:

ESL 模式 (esl)
    用法: $0 esl [AICORE_ARCH] <MODEL_VERSION> <CSV_FILE>
    说明:
        - 将当前目录下的 .cpp 文件重命名为 .cce
        - 调用 ./compile_simulator_run.sh 执行 ESL 编译和仿真
        - 可选参数 AICORE_ARCH：若传入则直接使用，否则根据 MODEL_VERSION 自动匹配
    参数:
        AICORE_ARCH     可选，AI Core 架构 (例如: 920r1, c220)
        MODEL_VERSION   模型版本 (例如: Ascend910_9695)
        CSV_FILE        输入测试用例 CSV 文件路径

示例:
    查看帮助:
        $0 --help

    ESL 模式 (自动匹配 AICORE_ARCH):
        $0 Ascend910_9695 testcase_1.csv

    ESL 模式 (指定 AICORE_ARCH):
        $0 920r1 Ascend910_9695 testcase_1.csv

EOF
}

ascend910b_list=("ascend910b1" "ascend910b2" "ascend910b2c" "ascend910b3" "ascend910b4" "ascend910b4-1" "ascend910_9391" "ascend910_9381" "ascend910_9372" "ascend910_9392" "ascend910_9382" "ascend910_9362")
ascend910_list=("ascend910a" "ascend910proa" "ascend910b" "ascend910prob" "ascend910premiuma")
ascend310p_list=("ascend310p1" "ascend310p3" "ascend310p5" "ascend310p7" "ascend310p3vir01" "ascend310p3vir02" "ascend310p3vir04" "ascend310p3vir08")
ascend310b_list=("ascend310b1" "ascend310b2" "ascend310b3" "ascend310b4")
ascend910_95_list=("ascend910_9599" "ascend910_9589" "ascend910_958b" "ascend910_957b" "ascend910_957d" "ascend910_950z" "ascend910_958a")
ascend910_96_list=("ascend910_9695" "ascend910_9691" "ascend910_9685" "ascend910_9689")
ascend910_55_list=("ascend910_5591")

# 检查是否请求帮助
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    show_help
    exit 0
fi
# 检查参数
if [ $# -lt 1 ]; then
    echo "Usage: $0 <mode> [params...]"
    echo "  - esl mode: $0 [AICORE_ARCH] <MODEL_VERSION> <CSV_FILE>"
    exit 1
fi


if [ $# -ne 4 ] && [ $# -ne 3 ]; then
    echo "【ESL错误】参数数量不正确"
    echo "用法：$0 [AICORE_ARCH] <MODEL_VERSION> <CSV_FILE> <TARGET_API>"
    exit 1
fi

# 根据参数数量决定变量赋值
if [ $# -eq 4 ]; then
    AICORE_ARCH=$1
    MODEL_VERSION=$2
    CSV_FILE=$3
    TARGET_API=$4
else
    MODEL_VERSION=$1
    CSV_FILE=$2
    TARGET_API=$3
fi

# 如果没有传入 AICORE_ARCH，则按照原逻辑进行匹配
if [ -z "$AICORE_ARCH" ]; then
    input_version=$(echo "$MODEL_VERSION" | tr '[:upper:]' '[:lower:]')
    # 判断是否在910b列表
    if [[ " ${ascend910b_list[@]} " =~ " ${input_version} " ]]; then
        AICORE_ARCH=c220
    # 判断是否在910列表
    elif [[ " ${ascend910_list[@]} " =~ " ${input_version} " ]]; then
        AICORE_ARCH=c100
    # 判断是否在310p列表
    elif [[ " ${ascend310p_list[@]} " =~ " ${input_version} " ]]; then
        AICORE_ARCH=m200
    # 判断是否在310b列表
    elif [[ " ${ascend310b_list[@]} " =~ " ${input_version} " ]]; then
        AICORE_ARCH=m300
    # 判断是否在910_95列表
    elif [[ " ${ascend910_95_list[@]} " =~ " ${input_version} " ]]; then
        AICORE_ARCH=c310
    # 判断是否在910_55列表
    elif [[ " ${ascend910_55_list[@]} " =~ " ${input_version} " ]]; then
        AICORE_ARCH=310r6
    # 判断是否在910_96列表
    elif [[ " ${ascend910_96_list[@]} " =~ " ${input_version} " ]]; then
        AICORE_ARCH=920r1
    else
        echo "【错误】$MODEL_VERSION 不在已知产品列表中"
        exit 1
    fi
fi

ROOT_DIR=${PWD}
CANN_INSTALL_PATH="/home/g00886479/workspace/cann_v121/"
BIN_NAME="common_benchMark"

WORK_DIR=${ROOT_DIR}/src/${TARGET_API}
KERNEL_TYPE="cube"
ESL_MODEL_PATH="esl"

CSV_FILE=${WORK_DIR}/${CSV_FILE}
# 校验CSV文件是否存在
if [ ! -f "$CSV_FILE" ]; then
    echo "【ESL错误】CSV文件不存在：$CSV_FILE"
    exit 1
fi

# 校验通过
echo "   ESL分支校验通过"
echo "   芯片类型：$AICORE_ARCH"
echo "   Model类型：$MODEL_VERSION"
echo "   CSV文件：$CSV_FILE"

# # 扩展名替换，增加提示
# echo -e "\n   切换文件后缀：.cpp -> .cce"
# for file in *.cpp; do
#     if [ -f "$file" ]; then
#         new_file="${file%.cpp}.cce"
#         mv "$file" "$new_file"
#         echo "   重命名：$file -> $new_file"
#     fi
# done

# 定义颜色输出
BLUE='\033[0;34m'
NC='\033[0m'

# 定义日志函数
error() {
    echo -e "[ERROR]$1" >&2
    exit 1
}

# ===================== 步骤1: 检查build_com.sh和build_sim.sh是否存在 =====================
echo "\nStep 1: 检查依赖脚本..."
if [ ! -f "build_com.sh" ] || [ ! -f "build_sim.sh" ] ; then
    error "编译或仿真脚本不存在于当前目录，请检查"
fi
chmod +x build_com.sh
chmod +x build_sim.sh

# ===================== 步骤2: 调用build_com.sh =====================
echo "\nStep 2: 运行编译..."
# 示例：build_com.sh [CANN_PATH] [WORK_DIR] [CCE_FILES] [BIN_NAME] [TYPE] [AICORE_ARCH]
sudo bash build_com.sh ${CANN_INSTALL_PATH} ${WORK_DIR} "kernel.cce" ${BIN_NAME} ${KERNEL_TYPE} "${AICORE_ARCH}"

# ===================== 步骤3: 仿真环境 =====================
prepare_simulate_env() {
    local DEFAULT_WORK_DIR=$(pwd)
    local CANN_INSTALL_PATH=$1
    local MODEL_VERSION=$2
    local COMPILE_PATH="${3:-$DEFAULT_WORK_DIR}"
    local BIN_NAME=$4
    local ESL_PATH=$5

    # 2. 前置检查
    [ -d "${CANN_INSTALL_PATH}" ] || error "CANN 路径不存在：${CANN_INSTALL_PATH}"
    RUNTIME_SO="${CANN_INSTALL_PATH}/cann/x86_64-linux/simulator/${MODEL_VERSION}/lib/libruntime_camodel.so"
    [ -f "${RUNTIME_SO}" ] || error "运行时库缺失"
    [ -f "${COMPILE_PATH}/${BIN_NAME}" ] || error "编译的二进制程序不存在：${COMPILE_PATH}/${BIN_NAME}"

    # 3. 准备 ESL 仿真环境
    echo "\n${BLUE}========== 准备 ESL 仿真环境 ==========${NC}"
    mkdir -p "${ESL_PATH}"
    cd "${ESL_PATH}" || error "进入 ESL 目录失败"

    # 解压模型
    echo "解压模型"
    sudo unzip -o "*.zip" >/dev/null 2>&1

    # 5. 拷贝依赖库
    echo "拷贝运行时库"
    sudo cp -f "${RUNTIME_SO}" ./

    # 6. 创建软连接
    echo "创建库软连接"
    rm -f libruntime.so libascend_hal.so
    ln -sf libruntime_camodel.so libruntime.so
    ln -sf libnpu_drv_camodel.so libascend_hal.so

    # 校验软连接
    [ -L libruntime.so ] || error "libruntime.so 软连接失败"
    [ -L libascend_hal.so ] || error "libascend_hal.so 软连接失败"

    # 拷贝程序
    echo "\n${BLUE}========== 运行算子仿真 ==========${NC}"
    cp -f "${COMPILE_PATH}/${BIN_NAME}" ./
    chmod +x "${BIN_NAME}"
    cd ..
}

prepare_simulate_env "${CANN_INSTALL_PATH}" "${MODEL_VERSION}" ${WORK_DIR} "${BIN_NAME}" ${ESL_MODEL_PATH}

# ===================== 步骤4: 调用build_sim.sh =====================
echo "\nStep 3: 处理仿真..."
# 示例：build_sim.sh [CANN_PATH] [WORK_DIR] [ESL_DIR] [BIN_NAME] [SIM_ARGS] [MODEL_VERSION]
# sudo bash ./build_sim.sh ${CANN_INSTALL_PATH} ${ESL_MODEL_PATH} ${BIN_NAME} "$CSV_FILE 1"

# 提取不带路径的文件名（含后缀）
filename_with_ext="${CSV_FILE##*/}"
CSV_FILE_NAME="${filename_with_ext%.*}"
TESTCASE_DIR=${WORK_DIR}/${CSV_FILE_NAME}

if [ -d "${TESTCASE_DIR}" ]; then
    echo "Directory${TESTCASE_DIR} exists. Clearing it..."
    sudo rm -rf "${TESTCASE_DIR}"/*
fi
mkdir -p "${TESTCASE_DIR}"

CSV_LINE_COUNT=$(wc -l < "$CSV_FILE")
for ((i=1; i<=CSV_LINE_COUNT; i++));
do
    line_number=$((i - 1))
    echo "Processing line $line_number..."
    # 调用build_sim.sh
    sudo bash ./build_sim.sh ${CANN_INSTALL_PATH} ${ESL_MODEL_PATH} ${BIN_NAME} "$CSV_FILE $line_number"

    # 后处理，计算带宽
    python3 run_test.py "$CSV_FILE" "$ESL_MODEL_PATH" "${CSV_FILE_NAME}_${line_number}" "${TESTCASE_DIR}/result.csv"

    # 仿真日志保留
    case_dir="${TESTCASE_DIR}/case_${line_number}"
    if [ -d "${case_dir}" ]; then
        echo "Directory${case_dir} exists. Clearing it..."
        rm -rf "${case_dir}"/*
    fi
    mkdir -p "${case_dir}"
    sudo mv "${ESL_MODEL_PATH}/log" "${case_dir}/"
    sudo mv "${ESL_MODEL_PATH}/log_ca" "${case_dir}/"
    sudo mv "peon/${ESL_MODEL_PATH}" "${case_dir}/"
done
