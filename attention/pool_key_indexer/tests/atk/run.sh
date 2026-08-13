#!/bin/bash
# ============================================================================
# PoolKeyIndexer ATK 双标杆测试一键运行脚本
#
# 用法:
#   ./run.sh                              # 默认: pyaclnn, 前10条, 不重新生成
#   ./run.sh -g                           # 强制重新生成用例
#   ./run.sh -n 100                       # 执行前100条
#   ./run.sh -r 10-50                     # 执行范围 [10, 50)
#   ./run.sh -l "1,3,5,10"                # 执行指定用例列表
#   ./run.sh -a                           # 执行全部用例
#   ./run.sh -c 20 -l "1000"              # 复检模式: case 1000, 20轮 + 自动CV分析
#
# 参数说明:
#   -g          强制重新生成用例
#   -n NUM      执行前 NUM 条用例 (默认: 10)
#   -r RANGE    执行范围, 格式 START-END
#   -l LIST     执行白名单, 逗号分隔
#   -a          执行全部用例
#   -d DEVICES  NPU 设备列表 (默认: 0)
#   -c LOOPS    复检模式: 循环执行 LOOPS 次
#   -mt NUM     CPU 多线程数 (默认: 5)
#   -h          显示帮助
#
# 优先级: -a > -l > -r > -n
# ============================================================================

WORK_DIR="$(cd "$(dirname "$0")" && pwd)"
YAML_FILE="${WORK_DIR}/pool_key_indexer.yaml"
GENERATOR_FILE="${WORK_DIR}/pool_key_indexer_generator.py"
API_FILE="${WORK_DIR}/pool_key_indexer_api.py"
ACLNN_API_FILE="${WORK_DIR}/pool_key_indexer_aclnn_api.py"
JSON_FILE="${WORK_DIR}/result/pool_key_indexer/json/all_pool_key_indexer.json"

export ASCEND_OPP_PATH="${ASCEND_OPP_PATH:-/usr/local/Ascend/cann/opp}"
export ASCEND_CUSTOM_OPP_PATH="${ASCEND_CUSTOM_OPP_PATH:-/usr/local/Ascend/cann/opp/vendors/custom_transformer}"
export ATK_TASK_OUTPUT_PATH="${WORK_DIR}"

GEN=false
NUM=10
RANGE=""
LIST=""
ALL=false
RECHECK=0
DEVICES="0"
MT_NUM=5

while getopts ":gn:r:l:c:d:mt:ah" opt; do
    case ${opt} in
        g  ) GEN=true ;;
        n  ) NUM="${OPTARG}" ;;
        r  ) RANGE="${OPTARG}" ;;
        l  ) LIST="${OPTARG}" ;;
        c  ) RECHECK="${OPTARG}" ;;
        d  ) DEVICES="${OPTARG}" ;;
        mt ) MT_NUM="${OPTARG}" ;;
        a  ) ALL=true ;;
        h  ) sed -n '3,30p' "$0"; exit 0 ;;
        \? ) echo "ERROR: Unknown option -${OPTARG}."; exit 1 ;;
        :  ) echo "ERROR: -${OPTARG} requires an argument."; exit 1 ;;
    esac
done

cd "${WORK_DIR}"

if [ "${GEN}" = "true" ]; then
    echo "[Step 1] Generate test cases (forced)..."
    atk case -f "${YAML_FILE}" -p "${GENERATOR_FILE}"
elif [ ! -f "${JSON_FILE}" ]; then
    echo "[Step 1] Generate test cases (JSON not found)..."
    atk case -f "${YAML_FILE}" -p "${GENERATOR_FILE}"
else
    echo "[Step 1] Skip case generation (use -g to force)"
fi

CASE_ARGS=()
if [ "${ALL}" = "true" ]; then
    echo "[Step 2] Running ALL cases..."
elif [ -n "${LIST}" ]; then
    echo "[Step 2] Running whitelist: [${LIST}]"
    CASE_ARGS+=(-wl "[${LIST}]")
elif [ -n "${RANGE}" ]; then
    RANGE_START="${RANGE%-*}"
    RANGE_END="${RANGE#*-}"
    echo "[Step 2] Running range: [${RANGE_START}, ${RANGE_END})"
    CASE_ARGS+=(-s "${RANGE_START}" -e "${RANGE_END}")
else
    echo "[Step 2] Running first ${NUM} cases..."
    CASE_ARGS+=(-s 0 -e "${NUM}")
fi

RECHECK_ARGS=()
TASK="accuracy"
if [ "${RECHECK}" -gt 0 ] 2>/dev/null; then
    echo "  Recheck: ${RECHECK} loops"
    RECHECK_ARGS+=(--disable_id_seed --loop_nums "${RECHECK}" --save_data output -to 3600)
    TASK="accuracy_lt"
fi

echo "  Task: ${TASK}, MT: ${MT_NUM}"
echo "=========================================="

export OMP_NUM_THREADS="${MT_NUM}"
export MKL_NUM_THREADS="${MT_NUM}"

atk node --backend pyaclnn --devices ${DEVICES} --name operator \
    node --backend cpu --name golden \
    task -c "${JSON_FILE}" \
    -p "${ACLNN_API_FILE}" \
    --task "${TASK}" "${CASE_ARGS[@]}" -cp "${RECHECK_ARGS[@]}" -mt "${MT_NUM}"

echo "=========================================="
echo " ATK test completed"
echo " Report: ${WORK_DIR}/atk_output/"
echo "=========================================="
