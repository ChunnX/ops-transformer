#!/bin/bash
# 修复：移除set -e（避免非关键命令失败退出），保留set -u（未定义变量警告）和pipefail
set -uo pipefail

# ===================== 核心配置区 =====================
ALL_OPS=(
    "bmm_reducescatter_alltoall"
    "alltoall_allgather_bmm"
    "alltoallv_gmm"
    "gmm_alltoallv"
    "all_gather_base_mm"
    "mm_reduce_scatter_base"
    "moe_distribute_combine"
    "moe_distribute_combine_v2"
    "moe_update_expert"
    "moe_combine_add_rms_norm"
    "distribute_barrier"
    "matmul"
)

DYNAMIC_OPS=(
    "bmm_reducescatter_alltoall"
    "alltoall_allgather_bmm"
    "alltoallv_gmm"
    "gmm_alltoallv"
    "all_gather_base_mm"
    "mm_reduce_scatter_base"
)

# 基础配置
TIMEOUT_SEC=300
TIMEOUT_EXIT_CODE=124
RANK_MIN=0
RANK_MAX=15
LOG_ROOT="./log"
TIME_STAMP=$(date +%Y%m%d%H%M%S)
LOG_DIR="${LOG_ROOT}/${TIME_STAMP}"
PASS_LOG="${LOG_DIR}/pass.log"
FAIL_LOG="${LOG_DIR}/fail.log"
REPORT_LOG="${LOG_DIR}/report.log"

# ==================== 全局变量 ====================
TARGET_OP1=""
TARGET_OP2=""
RESTART_FAIL_REPORT=""
RUN_MODE="full"

# ==================== 统计变量（确保全部初始化） ====================
total_cases=0
passed_cases=0
failed_cases=0
timeout_cases=0
error_cases=0
fail_rank_cases=0
missing_rank_cases=0
declare -a failed_commands=()
declare -a restart_commands=()
# 修复：预初始化MISSING_RANKS，避免未定义
MISSING_RANKS=""

# ==================== 工具函数 ====================
show_help() {
    echo "用法："
    echo "  1. 全量运行所有用例："
    echo "     $0"
    echo "  2. 单独重跑指定OP1+OP2组合："
    echo "     $0 --op1 <OP1值> --op2 <OP2值>"
    echo "  3. 批量重跑指定report.log中的失败用例："
    echo "     $0 --restart_fail <report.log路径>"
    echo "  4. 查看帮助："
    echo "     $0 --help"
    echo ""
    echo "可选OP值列表："
    echo "  ${ALL_OPS[*]}"
    echo ""
    echo "示例："
    echo "  $0 --op1 matmul --op2 distribute_barrier"
    echo "  $0 --restart_fail log/20251205103335/report.log"
}

init_log() {
    mkdir -p "${LOG_DIR}"
    > "${PASS_LOG}"
    > "${FAIL_LOG}"
    > "${REPORT_LOG}"
    echo "===== 测试开始 [$(date +%Y-%m-%d\ %H:%M:%S)] =====" | tee -a "${REPORT_LOG}"
    echo "日志存储目录：${LOG_DIR}" | tee -a "${REPORT_LOG}"
}

validate_op() {
    local op="$1"
    local op_name="$2"

    if [[ -z "${op}" ]]; then
        echo "错误：${op_name}不能为空！"
        show_help
        exit 1
    fi

    local is_valid=0
    for valid_op in "${ALL_OPS[@]}"; do
        if [[ "${valid_op}" == "${op}" ]]; then
            is_valid=1
            break
        fi
    done

    if [[ ${is_valid} -eq 0 ]]; then
        echo "错误：${op_name}='${op}' 不是合法值！"
        echo "合法OP值：${ALL_OPS[*]}"
        exit 1
    fi
}

extract_fail_commands() {
    echo "正在从 ${RESTART_FAIL_REPORT} 提取失败用例命令..."

    local temp_cmd_file=$(mktemp)
    grep -E "^- 【用例[0-9]+】\[.*\] .*python multi_stream.py" "${RESTART_FAIL_REPORT}" \
        | sed -E 's/^- *【用例[0-9]+】\[.*\] *//g' \
        | grep -v "^$" > "${temp_cmd_file}"

    mapfile -t restart_commands < "${temp_cmd_file}"
    rm -f "${temp_cmd_file}"

    if [[ ${#restart_commands[@]} -eq 0 ]]; then
        echo "警告：未从 ${RESTART_FAIL_REPORT} 中提取到任何失败用例命令！"
        exit 0
    fi

    echo "成功提取到 ${#restart_commands[@]} 个失败用例命令，准备重跑..."
    for cmd in "${restart_commands[@]}"; do
        echo "  - ${cmd}"
    done
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --op1)
                TARGET_OP1="$2"
                shift 2
                ;;
            --op2)
                TARGET_OP2="$2"
                shift 2
                ;;
            --restart_fail)
                RESTART_FAIL_REPORT="$2"
                shift 2
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                echo "错误：未知参数 '$1'"
                show_help
                exit 1
                ;;
        esac
    done

    if [[ -n "${RESTART_FAIL_REPORT}" ]]; then
        if [[ ! -f "${RESTART_FAIL_REPORT}" ]]; then
            echo "错误：report.log文件不存在！路径：${RESTART_FAIL_REPORT}"
            exit 1
        fi
        extract_fail_commands
        RUN_MODE="restart_fail"
    elif [[ -n "${TARGET_OP1}" || -n "${TARGET_OP2}" ]]; then
        if [[ -z "${TARGET_OP1}" || -z "${TARGET_OP2}" ]]; then
            echo "错误：--op1和--op2必须同时指定！"
            show_help
            exit 1
        fi
        validate_op "${TARGET_OP1}" "OP1"
        validate_op "${TARGET_OP2}" "OP2"
        RUN_MODE="single"
    else
        RUN_MODE="full"
    fi
}

is_dynamic_op() {
    local target_op="$1"
    for op in "${DYNAMIC_OPS[@]}"; do
        if [[ "${op}" == "${target_op}" ]]; then
            return 0
        fi
    done
    return 1
}

check_rank_pass() {
    local rank=$1
    local output_file=$2
    if grep -qE "^[[:space:]]*rank ${rank} test result: PASS" "${output_file}"; then
        return 0
    else
        return 1
    fi
}

is_test_passed() {
    local exit_code=$1
    local output_file=$2
    # 修复：初始化missing_ranks为空数组，避免未定义
    local missing_ranks=()
    # 重置MISSING_RANKS
    MISSING_RANKS=""

    if [[ ${exit_code} -eq ${TIMEOUT_EXIT_CODE} ]]; then
        ((timeout_cases++))
        return 1
    fi

    if [[ ${exit_code} -ne 0 ]]; then
        ((error_cases++))
        return 1
    fi

    if grep -qE "^[[:space:]]*rank [0-9]+ test result: FAIL" "${output_file}"; then
        ((fail_rank_cases++))
        return 1
    fi

    for ((rank=${RANK_MIN}; rank<=${RANK_MAX}; rank++)); do
        if ! check_rank_pass ${rank} "${output_file}"; then
            missing_ranks+=(${rank})
        fi
    done
    if [[ ${#missing_ranks[@]} -gt 0 ]]; then
        ((missing_rank_cases++))
        # 修复：用${missing_ranks[*]:-}兜底空值
        MISSING_RANKS="${missing_ranks[*]:-}"
        return 1
    fi

    return 0
}

run_single_command() {
    local cmd="$1"
    local case_num="$2"
    # 修复：重置MISSING_RANKS为默认空
    MISSING_RANKS=""

    local temp_output=$(mktemp)

    echo -e "\n========================================"
    echo "【用例${case_num}】执行命令: ${cmd}"
    echo "【超时限制】${TIMEOUT_SEC}秒（5分钟）| 【检测rank范围】${RANK_MIN}-${RANK_MAX}"
    echo -e "========================================\n" | tee -a "${REPORT_LOG}"

    # 修复：timeout命令后显式获取退出码，避免pipefail干扰
    timeout ${TIMEOUT_SEC}s ${cmd} 2>&1 | tee -a "${REPORT_LOG}" > "${temp_output}"
    local exit_code=$?

    if is_test_passed ${exit_code} "${temp_output}"; then
        # 修复：安全自增方式
        passed_cases=$((passed_cases + 1))
        local result="PASSED"
        local fail_type=""
    else
        echo "fail1"
        echo $failed_cases
        # 修复：安全自增方式（替代((failed_cases++))）
        failed_cases=$((failed_cases + 1))
        echo "fail2"
        local result=""
        local fail_type=""
        if [[ ${exit_code} -eq ${TIMEOUT_EXIT_CODE} ]]; then
            result="FAILED (超时，强制终止)"
            fail_type="[超时]"
        elif [[ ${exit_code} -ne 0 ]]; then
            result="FAILED (脚本错误，退出码:${exit_code})"
            fail_type="[脚本错误]"
        elif grep -qE "^[[:space:]]*rank [0-9]+ test result: FAIL" "${temp_output}"; then
            result="FAILED (存在rank输出FAIL)"
            fail_type="[存在FAIL]"
        else
            # 修复：用${MISSING_RANKS:-}兜底未定义
            result="FAILED (缺少rank ${MISSING_RANKS:-}的PASS输出)"
            fail_type="[缺失rank PASS:${MISSING_RANKS:-}]"
        fi
        # 修复：用${fail_type:-}和${cmd:-}兜底空值
        failed_commands+=("【用例${case_num}】${fail_type:-} ${cmd:-}")
    fi

    echo -e "\n【用例${case_num}】测试结果: ${result}" | tee -a "${REPORT_LOG}"
    echo -e "----------------------------------------\n" | tee -a "${REPORT_LOG}"

    if [[ ${result} == "PASSED" ]]; then
        echo -e "===== 用例${case_num} [PASS] ${cmd} =====" >> "${PASS_LOG}"
        cat "${temp_output}" >> "${PASS_LOG}"
        echo -e "\n========================================\n" >> "${PASS_LOG}"
    else
        echo -e "===== 用例${case_num} [FAIL] ${cmd} =====" >> "${FAIL_LOG}"
        cat "${temp_output}" >> "${FAIL_LOG}"
        echo -e "\n========================================\n" >> "${FAIL_LOG}"
    fi

    rm -f "${temp_output}"
}

run_single_case() {
    local op1="$1"
    local op2="$2"
    total_cases=$((total_cases + 1))

    local base_cmd="python multi_stream.py --op1 ${op1} --op2 ${op2}"
    if is_dynamic_op "${op1}" && is_dynamic_op "${op2}"; then
        base_cmd="python multi_stream.py --dynamic --op1 ${op1} --op2 ${op2}"
    fi

    run_single_command "${base_cmd}" "${total_cases}"
}

# ==================== 主逻辑 ====================
parse_args "$@"
init_log

case "${RUN_MODE}" in
    full)
        echo "===== 全量测试模式：遍历所有OP1+OP2组合 =====" | tee -a "${REPORT_LOG}"
        for op1 in "${ALL_OPS[@]}"; do
            for op2 in "${ALL_OPS[@]}"; do
                run_single_case "${op1}" "${op2}"
            done
        done
        ;;
    single)
        echo "===== 单例测试模式：OP1=${TARGET_OP1} OP2=${TARGET_OP2} =====" | tee -a "${REPORT_LOG}"
        run_single_case "${TARGET_OP1}" "${TARGET_OP2}"
        ;;
    restart_fail)
        echo "===== 批量重跑模式：共${#restart_commands[@]}个失败用例 =====" | tee -a "${REPORT_LOG}"
        total_cases=${#restart_commands[@]}
        case_idx=1
        for cmd in "${restart_commands[@]}"; do
            run_single_command "${cmd}" "${case_idx}"
            case_idx=$((case_idx + 1))
        done
        ;;
esac

# 生成最终统计报告
echo -e "\n========================================" | tee -a "${REPORT_LOG}"
echo -e "              测试统计结果              " | tee -a "${REPORT_LOG}"
echo -e "========================================" | tee -a "${REPORT_LOG}"
echo "运行模式:       ${RUN_MODE^^}" | tee -a "${REPORT_LOG}"
echo "总用例数:       ${total_cases}" | tee -a "${REPORT_LOG}"
echo "通过用例数:     ${passed_cases}" | tee -a "${REPORT_LOG}"
echo "未通过用例数:   ${failed_cases}" | tee -a "${REPORT_LOG}"
if [[ "${RUN_MODE}" == "full" ]]; then
    echo "  ├─ 超时用例数:         ${timeout_cases}" | tee -a "${REPORT_LOG}"
    echo "  ├─ 脚本错误用例数:     ${error_cases}" | tee -a "${REPORT_LOG}"
    echo "  ├─ 存在FAIL的用例数:   ${fail_rank_cases}" | tee -a "${REPORT_LOG}"
    echo "  └─ 缺少rank PASS用例数: ${missing_rank_cases}" | tee -a "${REPORT_LOG}"
fi
echo -e "========================================" | tee -a "${REPORT_LOG}"

if [[ ${failed_cases} -gt 0 ]]; then
    echo -e "\n【未通过的用例详情】:" | tee -a "${REPORT_LOG}"
    for failed_cmd in "${failed_commands[@]}"; do
        echo "- ${failed_cmd}" | tee -a "${REPORT_LOG}"
    done
else
    if [[ "${RUN_MODE}" == "single" ]]; then
        echo -e "\n🎉 指定用例 OP1=${TARGET_OP1} OP2=${TARGET_OP2} 测试通过！" | tee -a "${REPORT_LOG}"
    elif [[ "${RUN_MODE}" == "restart_fail" ]]; then
        echo -e "\n🎉 所有重跑的失败用例均通过测试！" | tee -a "${REPORT_LOG}"
    else
        echo -e "\n🎉 所有用例均通过测试！（所有rank 0-15均输出PASS）" | tee -a "${REPORT_LOG}"
    fi
fi

echo -e "\n===== 测试结束 [$(date +%Y-%m-%d\ %H:%M:%S)] =====" | tee -a "${REPORT_LOG}"
echo -e "\n测试完成！日志汇总："
echo "  - 成功用例日志：${PASS_LOG}"
echo "  - 失败用例日志：${FAIL_LOG}"
echo "  - 测试统计报告：${REPORT_LOG}"