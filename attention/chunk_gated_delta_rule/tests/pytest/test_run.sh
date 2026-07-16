#!/bin/bash

# 脚本路径
TEST_CHUNK_GATED_DELTA_RULE_SINGLE_SCRIPT="test_chunk_gated_delta_rule_single.py"
OP_NAME="ChunkGatedDeltaRule"

# ====================== 结果输出目录 ======================
RESULT_DIR="output"
mkdir -p "${RESULT_DIR}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${RESULT_DIR}/run_${TIMESTAMP}.log"
CSV_FILE="${RESULT_DIR}/result_${TIMESTAMP}.csv"

# ====================== 从 op_summary CSV 提取算子耗时(ms) ======================
extract_op_time() {
    local prof_dir="$1"
    local summary_file=$(find "${prof_dir}" -name "op_summary_*.csv" | head -1)
    if [ -z "${summary_file}" ]; then
        echo ""
        return
    fi
    local time_us=$(python3 -c "
import csv
with open('${summary_file}') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row.get('OP Type', '') == '${OP_NAME}':
            print(row['Task Duration(us)'].strip())
            break
" 2>/dev/null)
    if [ -n "${time_us}" ]; then
        python3 -c "print(f'{float(${time_us}):.3f}')"
    else
        echo ""
    fi
}

# ====================== 逐用例 msprof profile ======================
run_profile() {
    local test_mode="$1"
    echo "===== 执行 ${test_mode} 模式性能profiling (USE_GRAPH=${USE_GRAPH}) ====="

    # 先正常跑一遍生成 CSV (durations 留空)
    TEST_MODE=${test_mode} USE_GRAPH=${USE_GRAPH} CSV_FILE="${CSV_FILE}" \
        python3 -m pytest -rA -s $TEST_CHUNK_GATED_DELTA_RULE_SINGLE_SCRIPT \
        -v -m ci -W ignore::UserWarning -W ignore::DeprecationWarning 2>&1 | tee "${LOG_FILE}"
    local exit_code=${PIPESTATUS[0]}

    # 逐用例跑 msprof 获取耗时
    echo "===== 开始逐用例 msprof 性能采集 ====="
    local prof_base="${RESULT_DIR}/prof/prof_${TIMESTAMP}"
    mkdir -p "${prof_base}"

    # 读取 CSV 中的用例列表 (跳过表头)
    local case_indices=()
    local case_names=()
    local idx=0
    while IFS= read -r line; do
        local name=$(echo "${line}" | cut -d',' -f1)
        if [ -n "${name}" ]; then
            case_names+=("${name}")
            case_indices+=("${idx}")
        fi
        idx=$((idx + 1))
    done < <(tail -n +2 "${CSV_FILE}")

    local durations_line=""
    for i in "${!case_names[@]}"; do
        local name="${case_names[$i]}"
        local case_idx="${case_indices[$i]}"
        local pytest_filter="param_combinations${case_idx}"
        echo "  [msprof] 用例: ${name} (${pytest_filter})"

        local times_line=""
        local run_idx
        for run_idx in 1 2 3 4 5; do
            local prof_out="${prof_base}/${name}_run${run_idx}"
            mkdir -p "${prof_out}"
            TEST_MODE=${test_mode} USE_GRAPH=${USE_GRAPH} \
                msprof --output="${prof_out}" --aic-mode=task-based --task-time=on \
                --summary-format=csv --export=on \
                python3 -m pytest -rA -s $TEST_CHUNK_GATED_DELTA_RULE_SINGLE_SCRIPT \
                -v -m ci -W ignore::UserWarning -W ignore::DeprecationWarning \
                -k "${pytest_filter}" 2>&1 | tail -1
            local op_time=$(extract_op_time "${prof_out}")
            echo "    run${run_idx}: ${op_time} us"
            if [ -n "${times_line}" ]; then
                times_line+=","
            fi
            times_line+="${op_time}"
        done

        local avg_time=$(python3 -c "
times = [float(t) for t in '${times_line}'.split(',') if t]
print(f'{sum(times)/len(times):.3f}' if times else '')
")
        echo "  [msprof] ${name} 算子耗时(5次平均): ${avg_time} us"
        if [ -n "${durations_line}" ]; then
            durations_line+=","
        fi
        durations_line+="${avg_time}"
    done

    # 回填 durations 到 CSV
    python3 -c "
import csv
durations = '${durations_line}'.split(',')
rows = []
with open('${CSV_FILE}', 'r') as f:
    reader = csv.DictReader(f)
    fields = reader.fieldnames
    for i, row in enumerate(reader):
        if i < len(durations):
            row['durations'] = durations[i]
        rows.append(row)
with open('${CSV_FILE}', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
print('durations 回填完成')
"

    echo "执行日志: ${LOG_FILE}"
    echo "CSV: ${CSV_FILE}"
    echo "Profiling 数据: ${prof_base}"
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
    echo "  graph       可选，启用静态图模式"
    echo "  prof        可选，启用msprof性能采集（逐用例统计算子耗时）"
    echo "  help        显示本帮助信息"
    echo "示例："
    echo "  $0 single          # single模式（eager）"
    echo "  $0 single graph    # single模式（静态图）"
    echo "  $0 single prof     # single模式+msprof性能采集"
    echo "  $0 single graph prof # single模式+静态图+msprof性能采集"
    echo "  $0 rdv             # rdv模式（eager）"
    echo "  $0 rdv graph       # rdv模式（静态图）"
    echo "  $0 rdv prof        # rdv模式+msprof性能采集"
    echo "  $0 rdv graph prof  # rdv模式+静态图+msprof性能采集"
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
