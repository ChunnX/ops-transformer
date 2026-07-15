#!/bin/bash

# 脚本路径
TEST_CHUNK_GATED_DELTA_RULE_SINGLE_SCRIPT="test_chunk_gated_delta_rule_single.py"

# 结果输出目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="${SCRIPT_DIR}/output/${TIMESTAMP}"
LOG_FILE="${RESULTS_DIR}/test_run.log"
CSV_FILE="${RESULTS_DIR}/test_results.csv"
PERF_DIR="${RESULTS_DIR}/msprof_data"

# ====================== 执行区======================

# 算子调测
run_single() {
    mkdir -p "${RESULTS_DIR}"
    echo "===== 执行单算子用例调测 ====="
    CSV_REPORT_PATH="${CSV_FILE}" TEST_MODE=single python3 -m pytest -rA -s \
        $TEST_CHUNK_GATED_DELTA_RULE_SINGLE_SCRIPT -v -m ci \
        -W ignore::UserWarning -W ignore::DeprecationWarning 2>&1 | tee "${LOG_FILE}"
    return ${PIPESTATUS[0]}
}

# RDV测试
run_rdv() {
    mkdir -p "${RESULTS_DIR}"
    echo "===== 执行RDV参数集测试 ====="
    CSV_REPORT_PATH="${CSV_FILE}" TEST_MODE=rdv python3 -m pytest -rA -s \
        $TEST_CHUNK_GATED_DELTA_RULE_SINGLE_SCRIPT -v -m ci \
        -W ignore::UserWarning -W ignore::DeprecationWarning 2>&1 | tee "${LOG_FILE}"
    return ${PIPESTATUS[0]}
}

# 性能测试（逐例 msprof 采集）
run_perf() {
    local mode="$1"
    mkdir -p "${RESULTS_DIR}" "${PERF_DIR}"

    echo "===== 执行性能测试（逐例msprof采集）模式: ${mode} ====="

    # 1. 收集所有用例 node id
    local node_ids
    node_ids=$(TEST_MODE="${mode}" python3 -m pytest \
        $TEST_CHUNK_GATED_DELTA_RULE_SINGLE_SCRIPT --collect-only -q \
        -m ci -W ignore::UserWarning -W ignore::DeprecationWarning 2>/dev/null \
        | grep "::test_chunk_gated_delta_rule")

    local total
    total=$(echo "${node_ids}" | grep -c "::")
    echo "共发现 ${total} 条用例，开始逐例 msprof 采集..."
    echo ""

    local pass=0
    local fail=0
    local idx=0

    while IFS= read -r node_id; do
        [ -z "${node_id}" ] && continue
        idx=$((idx + 1))
        local case_name="case_$(printf '%03d' "${idx}")"
        local case_prof_dir="${PERF_DIR}/${case_name}"

        echo "===== [${idx}/${total}] ${case_name} ====="
        echo "  node_id: ${node_id}"

        rm -rf "${case_prof_dir}"
        mkdir -p "${case_prof_dir}"

        local case_csv="${case_prof_dir}/result.csv"
        CSV_REPORT_PATH="${case_csv}" TEST_MODE="${mode}" PERF_RUNS=5 \
            msprof --output="${case_prof_dir}" \
            python3 -m pytest -rA -s "${node_id}" -v -m ci \
            -W ignore::UserWarning -W ignore::DeprecationWarning 2>&1 | tee -a "${LOG_FILE}"

        local status=${PIPESTATUS[0]}
        if [ "${status}" -eq 0 ]; then
            pass=$((pass + 1))
            echo "  [PASS] ${case_name}"
        else
            fail=$((fail + 1))
            echo "  [FAIL] ${case_name} (exit=${status})"
        fi
        echo ""

    done <<< "${node_ids}"

    # 2. 合并用例结果与性能数据到 test_results.csv
    echo "========== 合并性能数据 =========="
    python3 "${SCRIPT_DIR}/collect_perf_data.py" \
        --prof-dir "${PERF_DIR}" \
        --output "${CSV_FILE}" 2>&1 | tee -a "${LOG_FILE}"

    echo ""
    echo "========== 性能测试汇总 =========="
    echo "总计: ${total}  通过: ${pass}  失败: ${fail}"

    return 0
}

# 打印结果文件路径
print_results() {
    echo ""
    echo "===== 结果文件 ====="
    echo "日志文件: ${LOG_FILE}"
    echo "CSV结果:  ${CSV_FILE}"
}

# 打印性能结果文件路径
print_perf_results() {
    echo ""
    echo "===== 结果文件 ====="
    echo "日志文件:   ${LOG_FILE}"
    echo "CSV结果:    ${CSV_FILE}"
    echo "msprof数据: ${PERF_DIR}"
}

# 显示帮助信息
show_help() {
    echo "用法: $0 [参数]"
    echo "参数说明："
    echo "  single    执行单算子用例调测"
    echo "  rdv       执行RDV参数集测试"
    echo "  perf      逐例msprof性能采集（single参数集）"
    echo "  perf_rdv  逐例msprof性能采集（rdv参数集）"
    echo "  help      显示本帮助信息"
    echo "示例："
    echo "  $0 single    # 执行single模式"
    echo "  $0 rdv       # 执行rdv模式"
    echo "  $0 perf      # 逐例msprof性能采集（single参数集）"
    echo "  $0 perf_rdv  # 逐例msprof性能采集（rdv参数集）"
}

# ====================== 主逻辑 ======================
# 检查传入的参数数量
if [ $# -ne 1 ]; then
    echo "错误：必须传入且仅传入一个参数（single/rdv/perf/perf_rdv/help）"
    show_help
    exit 1
fi

# 根据参数执行对应函数
case "$1" in
    single)
        run_single
        EXIT_CODE=$?
        print_results
        exit ${EXIT_CODE}
        ;;
    rdv)
        run_rdv
        EXIT_CODE=$?
        print_results
        exit ${EXIT_CODE}
        ;;
    perf)
        run_perf "single"
        EXIT_CODE=$?
        print_perf_results
        exit ${EXIT_CODE}
        ;;
    perf_rdv)
        run_perf "rdv"
        EXIT_CODE=$?
        print_perf_results
        exit ${EXIT_CODE}
        ;;
    help)
        show_help
        ;;
    *)
        echo "错误：未知参数 '$1'，仅支持 single/rdv/perf/perf_rdv/help"
        show_help
        exit 1
        ;;
esac

exit 0
