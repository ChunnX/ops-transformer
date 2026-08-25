#!/bin/bash
# 用法:
#   ./run_fia_antiquant.sh                              全量编译+安装+运行全部 case
#   ./run_fia_antiquant.sh --case=XXX                   只运行指定 case(不编译)
#   ./run_fia_antiquant.sh --case=AAA,BBB               运行多个 case(不编译)
#   ./run_fia_antiquant.sh --tiling-key=564399360       指定 tiling key 编译+安装+运行全部
#   ./run_fia_antiquant.sh --tiling-key=564399360,2711883009  多 key 编译+安装+运行全部
#   ./run_fia_antiquant.sh --contiguous-only            只跑连续基线
#   ./run_fia_antiquant.sh --tiling-key=564399360 --case=AAA,BBB  编译+运行指定 case
#
# 环境激活 (路径按实际环境替换)
source PATH_TO_CANN/cann/set_env.sh
source PATH_TO_CONDA/etc/profile.d/conda.sh
conda activate qjr
export ASCEND_CANN_PACKAGE_PATH=${ASCEND_HOME_PATH}
export ASCEND_GLOBAL_LOG_LEVEL=0
export ASCEND_SLOG_PINTT_TO_STDOUT=1

# 若不在 HwHiAiUser 组则通过 sg 切换
if ! id -nG | grep -qw HwHiAiUser; then
    exec sg HwHiAiUser -c "bash $0 $*"
fi

REPO_ROOT="PATH_TO_REPO_ROOT"
MY_TEST_ROOT="PATH_TO_TEST_ROOT"
VENDOR_PATH="PATH_TO_VENDOR/custom_transformer"
CASE_ID="SMOKE_ALL"
TILING_KEY=""
EXTRA_ARGS=""

for arg in "$@"; do
    case "$arg" in
        --tiling-key=*) TILING_KEY="${arg#*=}" ;;
        --case=*) CASE_ID="${arg#*=}" ;;
        --contiguous-only) EXTRA_ARGS="${EXTRA_ARGS} --contiguous-only" ;;
    esac
done

# 编译
echo "[build] 编译 fused_infer_attention_score..."
cd "${REPO_ROOT}"

BUILD_CMD="bash build.sh --pkg --soc=ascend950 --ops=fused_infer_attention_score --build-type=Debug -O0 -j$(nproc)"
if [ -n "${TILING_KEY}" ]; then
    # 多 key 必须用分号传 cmake list (build.sh 的 TILING_KEY 是单值变量, 多个 --tiling_key= 会后者覆盖前者;
    # config.cmake 的 string(REPLACE ";" "::" EP_TILING_KEY) 会处理分号列表)。逗号分隔自动转分号。
    TK_SEMICOLON="${TILING_KEY//,/;}"
    BUILD_CMD="${BUILD_CMD} --tiling_key=${TK_SEMICOLON}"
    echo "[build] tiling_key=${TILING_KEY} (cmake list: ${TK_SEMICOLON})"
else
    echo "[build] 全量编译(无 tiling_key)"
fi

${BUILD_CMD} &> "${MY_TEST_ROOT}/log/build.log"

if [ $? -ne 0 ]; then
    echo "[build] 失败, 查看 log/build.log"
    exit 1
fi

# 安装
echo "[install] 安装 .run 包..."
RUN_PKG=$(ls "${REPO_ROOT}"/build_out/cann-ops-transformer-custom_linux-*.run 2>/dev/null | head -1)
if [ -z "${RUN_PKG}" ]; then
    echo "[install] 未找到 .run 包, 查看 log/build.log"
    exit 1
fi
"${RUN_PKG}" --install-path="PATH_TO_VENDOR" \
    &> "${MY_TEST_ROOT}/log/install.log"
if [ $? -ne 0 ]; then
    echo "[install] 失败, 查看 log/install.log"
    exit 1
fi
echo "[build] 完成"

# 测试
if [ ! -f "${VENDOR_PATH}/bin/set_env.bash" ]; then
    echo "[error] vendors 未安装"
    exit 1
fi

source "${VENDOR_PATH}/bin/set_env.bash"
# CANN 9.2.0 的 libnnopbase.so 已自带 NnopbaseSupportTensorV2 符号, 无需 LD_PRELOAD 旧 patch
rm -rf ${HOME}/.cache/torch_extensions 2>/dev/null

echo "[test] 运行 case: ${CASE_ID} ${EXTRA_ARGS} ..."
python "${MY_TEST_ROOT}/test_fia_antiquant_noncontiguous.py" --case "${CASE_ID}" ${EXTRA_ARGS} \
    &> "${MY_TEST_ROOT}/log/test_case.log"

echo "[test] 结果摘要:"
grep -E "PASS|FAIL|ERROR|Summary" "${MY_TEST_ROOT}/log/test_case.log" 2>/dev/null | tail -30
