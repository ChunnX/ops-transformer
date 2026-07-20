#!/bin/bash

# 脚本路径
TEST_CHUNK_GATED_DELTA_RULE_SINGLE_SCRIPT="test_chunk_gated_delta_rule_single.py"
OP_NAME="ChunkGatedDeltaRule"
PROF_RUNS=5
PROF_WARMUP=1

# ====================== 结果输出目录 ======================
RESULT_DIR="output"
mkdir -p "${RESULT_DIR}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${RESULT_DIR}/run_${TIMESTAMP}.log"
CSV_FILE="${RESULT_DIR}/result_${TIMESTAMP}.csv"

# ====================== msprof profile (一次跑所有用例) ======================
run_profile() {
    local test_mode="$1"
    echo "===== 执行 ${test_mode} 模式性能profiling (USE_GRAPH=${USE_GRAPH}) ====="

    # 1. 正常跑一遍生成 CSV (含 status, durations 留空)
    echo "===== 第1步: 正常精度测试 ====="
    TEST_MODE=${test_mode} USE_GRAPH=${USE_GRAPH} CSV_FILE="${CSV_FILE}" \
        python3 -m pytest -rA -s $TEST_CHUNK_GATED_DELTA_RULE_SINGLE_SCRIPT \
        -v -m ci -W ignore::UserWarning -W ignore::DeprecationWarning 2>&1 | tee "${LOG_FILE}"
    local exit_code=${PIPESTATUS[0]}

    # 2. 一次 msprof 跑所有用例 (ENABLE_PROF=true, 每用例跑5次算子)
    echo "===== 第2步: msprof 性能采集 (每用例${PROF_RUNS}次) ====="
    local prof_dir="${RESULT_DIR}/prof/prof_${TIMESTAMP}"
    mkdir -p "${prof_dir}"
    TEST_MODE=${test_mode} USE_GRAPH=${USE_GRAPH} ENABLE_PROF=true \
        msprof --output="${prof_dir}" \
        --summary-format=csv --export=on \
        python3 -m pytest -rA -s $TEST_CHUNK_GATED_DELTA_RULE_SINGLE_SCRIPT \
        -v -m ci -W ignore::UserWarning -W ignore::DeprecationWarning 2>&1 | tee -a "${LOG_FILE}"

    # 3. 从 op_summary 提取算子耗时, 每${PROF_RUNS}行取平均, 回填到 CSV
    echo "===== 第3步: 提取耗时并回填CSV ====="
    local summary_file=$(find "${prof_dir}" -name "op_summary_*.csv" | head -1)
    if [ -z "${summary_file}" ]; then
        echo "错误: 未找到 op_summary CSV 文件"
        return 1
    fi

    python3 -c "
import csv

OP_NAME = '${OP_NAME}'
PROF_RUNS = ${PROF_RUNS}
PROF_WARMUP = ${PROF_WARMUP}
CSV_FILE = '${CSV_FILE}'
summary_file = '${summary_file}'

# 从 op_summary 提取所有 ChunkGatedDeltaRule 的 Task Duration
times = []
with open(summary_file) as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row.get('OP Type', '').strip() == OP_NAME:
            t = row['Task Duration(us)'].strip()
            if t:
                times.append(float(t))

print(f'op_summary 中 {OP_NAME} 记录数: {len(times)}')

# 每用例 PROF_WARMUP+PROF_RUNS 条, 丢弃前 PROF_WARMUP 条, 取后 PROF_RUNS 条平均
group_size = PROF_WARMUP + PROF_RUNS
durations = []
for i in range(0, len(times), group_size):
    group = times[i+PROF_WARMUP:i+group_size]
    avg = sum(group) / len(group)
    durations.append(f'{avg:.3f}')

print(f'用例数: {len(durations)}')

# 回填到 CSV
rows = []
with open(CSV_FILE, 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    fields = reader.fieldnames
    for i, row in enumerate(reader):
        if i < len(durations):
            row['durations'] = durations[i]
        rows.append(row)

with open(CSV_FILE, 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

print('durations 回填完成')
"

    echo "执行日志: ${LOG_FILE}"
    echo "CSV: ${CSV_FILE}"
    echo "Profiling 数据: ${prof_dir}"
    return ${exit_code}
}

# ====================== 执行区======================

# 算子调测
run_single() {
    echo "===== 执行单算子用例调测 (USE_GRAPH=${USE_GRAPH}) ====="
    TEST_MODE=single USE_GRAPH=${USE_GRAPH} CSV_FILE="${CSV_FILE}" \
        python3 -m pytest -rA -s $TEST_CHUNK_GATED_DELTA_RULE_SINGLE_SCRIPT \
        -v -m ci -W ignore::UserWarning -W ignore::DeprecationWarning 2>&1 | tee "${LOG_FILE}"
    local exit_code=${PIPESTATUS[0]}
    echo "执行日志: ${LOG_FILE}"
    echo "CSV: ${CSV_FILE}"
    return ${exit_code}
}

# RDV测试
run_rdv() {
    echo "===== 执行RDV参数集测试 (USE_GRAPH=${USE_GRAPH}) ====="
    TEST_MODE=rdv USE_GRAPH=${USE_GRAPH} CSV_FILE="${CSV_FILE}" \
        python3 -m pytest -rA -s $TEST_CHUNK_GATED_DELTA_RULE_SINGLE_SCRIPT \
        -v -m ci -W ignore::UserWarning -W ignore::DeprecationWarning 2>&1 | tee "${LOG_FILE}"
    local exit_code=${PIPESTATUS[0]}
    echo "执行日志: ${LOG_FILE}"
    echo "CSV: ${CSV_FILE}"
    return ${exit_code}
}

# 显示帮助信息
show_help() {
    echo "用法: $0 <模式> [graph] [prof]"
    echo "参数说明："
    echo "  模式:"
    echo "    single    执行单算子用例调测"
    echo "    rdv       执行RDV参数集测试"
    echo "  graph       可选，启用aclgraph模式"
    echo "  prof        可选，启用msprof性能采集（逐用例统计算子耗时）"
    echo "  help        显示本帮助信息"
    echo "示例："
    echo "  $0 single          # single模式（eager）"
    echo "  $0 single graph    # single模式（aclgraph）"
    echo "  $0 single prof     # single模式+msprof性能采集"
    echo "  $0 single graph prof # single模式+aclgraph+msprof性能采集"
    echo "  $0 rdv             # rdv模式（eager）"
    echo "  $0 rdv graph       # rdv模式（aclgraph）"
    echo "  $0 rdv prof        # rdv模式+msprof性能采集"
    echo "  $0 rdv graph prof  # rdv模式+aclgraph+msprof性能采集"
}

# ====================== 主逻辑 ======================
# 检查传入的参数数量
if [ $# -lt 1 ] || [ $# -gt 3 ]; then
    echo "错误：参数数量不正确（需1-3个参数）"
    show_help
    exit 1
fi

# 解析第一个参数：模式
TEST_MODE_ARG="$1"
USE_GRAPH=false
ENABLE_PROF=false

# 解析剩余参数
for arg in "${@:2}"; do
    case "$arg" in
        graph)
            USE_GRAPH=true
            ;;
        prof)
            ENABLE_PROF=true
            ;;
        *)
            echo "错误：未知参数 '$arg'，仅支持 'graph' 或 'prof'"
            show_help
            exit 1
            ;;
    esac
done

# 根据参数执行对应函数
case "$TEST_MODE_ARG" in
    single)
        if [ "${ENABLE_PROF}" == "true" ]; then
            run_profile "single"
        else
            run_single
        fi
        ;;
    rdv)
        if [ "${ENABLE_PROF}" == "true" ]; then
            run_profile "rdv"
        else
            run_rdv
        fi
        ;;
    help)
        show_help
        ;;
    *)
        echo "错误：未知参数 '$TEST_MODE_ARG'，仅支持 single/rdv/help"
        show_help
        exit 1
        ;;
esac

exit 0
