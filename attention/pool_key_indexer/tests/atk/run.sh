#!/bin/bash
# ============================================================================
# PoolKeyIndexer ATK 双标杆测试一键运行脚本
#
# 测试拓扑(pki_double_benchmark):
#   - operator (pyaclnn): aclnn 接口直调算子(原始精度输入);
#   - npu_golden (npu)  : NPU 组合标杆(与 pytest _reference_npu 一致);
#   - 真值 (cpu_benchmark): ATK benchmark 任务自动生成(CPU 金标升精度)。
#   精度判定:
#     indices: 算子 vs 真值 逐行有效索引集合比对(100% 行通过, 与 pytest 一致);
#     values : 按索引排序后 算子/标杆 vs 真值 双标杆比例门禁。
#
# 用法:
#   ./run.sh                              # 默认: 设备 2, 前 10 条, 不重新生成
#   ./run.sh -g                           # 强制重新生成用例
#   ./run.sh -n 100                       # 执行前 100 条
#   ./run.sh -r 10-50                     # 执行范围 [10, 50)
#   ./run.sh -l "1,3,5,10"                # 执行指定用例列表
#   ./run.sh -a                           # 执行全部用例
#   ./run.sh -c 20 -l "1000"              # 复检模式: case 1000, 20 轮(accuracy_lt)
#   ./run.sh -d 3                         # 指定 NPU 设备(默认 2; 0/1 告警态勿用)
#   ./run.sh -q ...                       # 量化场景(quant_mode=0/1, 仅 950):
#                                           pool_key_indexer_quant.yaml, 拓扑为
#                                           operator(pyaclnn) + golden(cpu 金标)
#
# 参数说明:
#   -g          强制重新生成用例
#   -n NUM      执行前 NUM 条用例 (默认: 10)
#   -r RANGE    执行范围, 格式 START-END
#   -l LIST     执行白名单, 逗号分隔
#   -a          执行全部用例
#   -d DEVICES  NPU 设备号 (默认: 2, 算子与标杆共用)
#   -c LOOPS    复检模式: 循环执行 LOOPS 次(accuracy_lt, 每轮独立重新生成数据)
#   -mt NUM     CPU 多线程数 (默认: 5)
#   -q          量化场景(独立用例集/独立 JSON, 不影响非量化)
#   -h          显示帮助
#
# 优先级: -a > -l > -r > -n
# ============================================================================

WORK_DIR="$(cd "$(dirname "$0")" && pwd)"
YAML_FILE="${WORK_DIR}/pool_key_indexer.yaml"
GENERATOR_FILE="${WORK_DIR}/pool_key_indexer_generator.py"
JSON_FILE="${WORK_DIR}/result/pool_key_indexer/json/all_pool_key_indexer.json"
PLUGIN_DIR="${WORK_DIR}"

# 环境变量:
#   ASCEND_CUSTOM_OPP_PATH 指向 CANN 安装根目录, 使
#   ${ASCEND_CUSTOM_OPP_PATH}/vendors/custom_transformer/op_api/lib/libcust_opapi.so
#   可被 pyaclnn 后端定位(ATK 的 so 查找约定);
#   ASCEND_OPP_PATH 沿用当前 shell 环境(缺省时回退到 toolkit 的 opp)。
CANN_HOME="${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/ascend-toolkit/latest}"
export ASCEND_CUSTOM_OPP_PATH="${ASCEND_CUSTOM_OPP_PATH:-${CANN_HOME}}"
export ASCEND_OPP_PATH="${ASCEND_OPP_PATH:-${CANN_HOME}/opp}"
# ATK 输出目录(含 celery 文件 broker): 默认工作目录; 若 /home 空间不足可
# 通过 PKI_ATK_OUTPUT_DIR 指到其他分区(如 /tmp), 避免 broker .msg 写入
# 失败导致任务挂起/崩溃
export ATK_TASK_OUTPUT_PATH="${PKI_ATK_OUTPUT_DIR:-${WORK_DIR}}"

GEN=false
NUM=10
RANGE=""
LIST=""
ALL=false
RECHECK=0
DEVICES="2"
MT_NUM=5
QUANT=false
BLACKLIST=""

while getopts ":gn:r:l:c:b:d:mt:aqh" opt; do
    case ${opt} in
        g  ) GEN=true ;;
        n  ) NUM="${OPTARG}" ;;
        r  ) RANGE="${OPTARG}" ;;
        l  ) LIST="${OPTARG}" ;;
        b  ) BLACKLIST="${OPTARG}" ;;
        c  ) RECHECK="${OPTARG}" ;;
        d  ) DEVICES="${OPTARG}" ;;
        mt ) MT_NUM="${OPTARG}" ;;
        a  ) ALL=true ;;
        q  ) QUANT=true ;;
        h  ) sed -n '3,44p' "$0"; exit 0 ;;
        \? ) echo "ERROR: Unknown option -${OPTARG}."; exit 1 ;;
        :  ) echo "ERROR: -${OPTARG} requires an argument."; exit 1 ;;
    esac
done

cd "${WORK_DIR}"

# 量化模式: 独立用例集 + CPU 金标拓扑(NPU 组合标杆暂不支持 descale)
if [ "${QUANT}" = "true" ]; then
    YAML_FILE="${WORK_DIR}/pool_key_indexer_quant.yaml"
    GENERATOR_FILE="${WORK_DIR}/pool_key_indexer_quant_generator.py"
    JSON_FILE="${WORK_DIR}/result/pool_key_indexer_quant/json/all_pool_key_indexer_quant.json"
fi

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
    if [ -n "${BLACKLIST}" ]; then
        echo "  Blacklist (excluded): [${BLACKLIST}]"
        CASE_ARGS+=(-bl "[${BLACKLIST}]")
    fi
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

TASK="accuracy"
if [ "${RECHECK}" -gt 0 ] 2>/dev/null; then
    echo "  Recheck: ${RECHECK} loops (accuracy_lt, per-loop regenerated data)"
    RECHECK_ARGS+=(--loop_nums "${RECHECK}" -to 3600)
    TASK="accuracy_lt"
fi

echo "  Task: ${TASK}, Devices: ${DEVICES}, MT: ${MT_NUM}, Mode: $([ "${QUANT}" = "true" ] && echo quant || echo non-quant)"
echo "=========================================="

export OMP_NUM_THREADS="${MT_NUM}"
export MKL_NUM_THREADS="${MT_NUM}"

# 节点拓扑: operator(pyaclnn, 主节点) + 标杆节点;
# 真值任务由精度标准名(pki_double_benchmark 含 "benchmark")自动触发。
# -p 传插件目录: 同时加载 API 适配器(BaseApi/AclnnBaseApi)与精度比较器。
#   非量化: npu_golden(npu 组合标杆, 原始精度输入);
#   量化  : golden(cpu 金标 —— NPU 组合标杆暂不支持 descale, PKIApi 对
#           量化输入统一路由 CPU 金标, 拓扑直连 cpu 免去 NPU 往返)。
if [ "${QUANT}" = "true" ]; then
    atk node --backend pyaclnn --devices ${DEVICES} --name operator \
        node --backend cpu --name golden \
        task -c "${JSON_FILE}" \
        -p "${PLUGIN_DIR}" \
        --task "${TASK}" "${CASE_ARGS[@]}" -cp ${RECHECK_ARGS[@]+"${RECHECK_ARGS[@]}"} -mt "${MT_NUM}"
else
    atk node --backend pyaclnn --devices ${DEVICES} --name operator \
        node --backend npu --devices ${DEVICES} --name npu_golden \
        task -c "${JSON_FILE}" \
        -p "${PLUGIN_DIR}" \
        --task "${TASK}" "${CASE_ARGS[@]}" -cp ${RECHECK_ARGS[@]+"${RECHECK_ARGS[@]}"} -mt "${MT_NUM}"
fi

echo "=========================================="
echo " ATK test completed"
echo " Report: ${WORK_DIR}/atk_output/"
echo "=========================================="
