#!/usr/bin/env bash

set -uo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TEST_ROOT="${REPO_ROOT}/experimental/attention/quant_block_sparse_attn/tests"
PYTEST_ROOT="${TEST_ROOT}/pytest/bsa_fullquant_mxfp8_test"
TTK_ROOT="${TEST_ROOT}/ttk/bsa_fullquant_mxfp8_test"
ASSETS_ROOT="${TEST_ROOT}/assets"

RUN_MODE="${1:-all}"
RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULT_ROOT="${QBSA_RESULT_ROOT:-${REPO_ROOT}/qbsa_test_results/${RUN_TIMESTAMP}}"
PREPARED_CSV_ROOT="${RESULT_ROOT}/prepared_csv"
CACHE_ROOT="${RESULT_ROOT}/cache"

PYTEST_REDLINE_SOURCE="${PYTEST_ROOT}/qbsa_mxfp8_redline.csv"
TTK_REDLINE_SOURCE="${TTK_ROOT}/qbsa_mxfp8_redline.csv"
PYTEST_REDLINE_CSV="${PREPARED_CSV_ROOT}/pytest_mxfp8_redline_without_last_2.csv"
TTK_REDLINE_CSV="${PREPARED_CSV_ROOT}/ttk_mxfp8_redline_without_last_2.csv"
PYTEST_STC_CSV="${PYTEST_ROOT}/qbsa_mxfp8_stc.csv"
TTK_STC_CSV="${TTK_ROOT}/qbsa_mxfp8_stc.csv"
PYTEST_GENERALIZED_CSV="${PYTEST_ROOT}/qbsa_mxfp8_generalized_2000.csv"
TTK_GENERALIZED_CSV="${TTK_ROOT}/qbsa_mxfp8_generalized_2000.csv"

mkdir -p "${RESULT_ROOT}" "${PREPARED_CSV_ROOT}" "${CACHE_ROOT}"
cd "${REPO_ROOT}"

print_usage() {
    echo "Usage: bash $0 [mode]"
    echo
    echo "Modes:"
    echo "  all                 Run the complete requested regression matrix (default)."
    echo "  redline             Run TTK and pytest redline, eager + ACLGraph."
    echo "  stc                 Run TTK and pytest STC, eager + ACLGraph."
    echo "  generalized         Run TTK eager and pytest eager generalized 2000 cases."
    echo "  ttk-redline         Run TTK redline, eager + ACLGraph."
    echo "  pytest-redline      Run pytest redline, eager + ACLGraph."
    echo "  ttk-stc             Run TTK STC, eager + ACLGraph."
    echo "  pytest-stc          Run pytest STC, eager + ACLGraph."
    echo "  ttk-generalized     Run TTK eager generalized 2000 cases."
    echo "  pytest-generalized  Run pytest eager generalized 2000 cases."
    echo
    echo "The redline suites automatically exclude their final two cases."
    echo "Each suite prepares inputs and CPU Golden once, then reuses them."
    echo "Set QBSA_RESULT_ROOT to override the output/cache directory."
}

require_file() {
    local path="$1"
    if [[ ! -f "${path}" ]]; then
        echo "[ERROR] Input file does not exist: ${path}" >&2
        return 1
    fi
}

copy_without_last_two_cases() {
    local source_csv="$1"
    local target_csv="$2"
    local description="$3"

    require_file "${source_csv}" || return 1
    local line_count
    line_count="$(wc -l <"${source_csv}")"
    if [[ ${line_count} -lt 4 ]]; then
        echo "[ERROR] ${description} must contain a header and at least three cases: ${source_csv}" >&2
        return 1
    fi

    head -n -2 "${source_csv}" >"${target_csv}"
    echo "[INFO] ${description}: retained $((line_count - 3)) cases; excluded the final two."
    echo "       prepared CSV: ${target_csv}"
}

prepare_redline_csvs() {
    copy_without_last_two_cases \
        "${TTK_REDLINE_SOURCE}" \
        "${TTK_REDLINE_CSV}" \
        "TTK MXFP8 redline" || return 1
    copy_without_last_two_cases \
        "${PYTEST_REDLINE_SOURCE}" \
        "${PYTEST_REDLINE_CSV}" \
        "Pytest MXFP8 redline"
}

run_logged() {
    local suite_name="$1"
    local log_file="$2"
    shift 2

    echo "[START] ${suite_name}"
    echo "        log: ${log_file}"
    "$@" >"${log_file}" 2>&1
    local status=$?

    if [[ ${status} -eq 0 ]]; then
        echo "[PASS]  ${suite_name}"
    else
        echo "[FAIL]  ${suite_name}, exit code=${status}"
        echo "        inspect: ${log_file}"
    fi
    return ${status}
}

run_pytest_matrix() {
    local suite_key="$1"
    local suite_label="$2"
    local case_csv="$3"
    local run_aclgraph="$4"
    local cache_dir="${CACHE_ROOT}/pytest_${suite_key}"

    require_file "${case_csv}" || return 1
    mkdir -p "${cache_dir}"

    run_logged \
        "Pytest ${suite_label} prepare input + CPU Golden" \
        "${RESULT_ROOT}/pytest_${suite_key}_prepare.log" \
        python3 "${PYTEST_ROOT}/qbsa_mxfp8_golden.py" \
        --csv \
        --case_files "${case_csv}" \
        --cache-dir "${cache_dir}" \
        --mode gen,cpu || return 1

    local matrix_status=0
    run_logged \
        "Pytest ${suite_label} eager (reuse input + Golden)" \
        "${RESULT_ROOT}/pytest_${suite_key}_eager.log" \
        python3 "${PYTEST_ROOT}/qbsa_mxfp8_golden.py" \
        --csv \
        --case_files "${case_csv}" \
        --cache-dir "${cache_dir}" \
        --mode npu,compare || matrix_status=1

    if [[ "${run_aclgraph}" == "true" ]]; then
        run_logged \
            "Pytest ${suite_label} ACLGraph (reuse input + Golden)" \
            "${RESULT_ROOT}/pytest_${suite_key}_aclgraph.log" \
            python3 "${PYTEST_ROOT}/qbsa_mxfp8_golden.py" \
            --csv \
            --case_files "${case_csv}" \
            --cache-dir "${cache_dir}" \
            --mode npu,compare \
            --aclgraph || matrix_status=1
    fi

    return ${matrix_status}
}

run_ttk_matrix() {
    local suite_key="$1"
    local suite_label="$2"
    local case_csv="$3"
    local run_aclgraph="$4"
    local manual_data_dir="${CACHE_ROOT}/ttk_${suite_key}_manual_data"

    require_file "${case_csv}" || return 1
    mkdir -p "${manual_data_dir}"

    run_logged \
        "TTK ${suite_label} prepare input + CPU Golden" \
        "${RESULT_ROOT}/ttk_${suite_key}_prepare.log" \
        python3 -m ttk e2e \
        -i "${case_csv}" \
        --plugin "${ASSETS_ROOT}" \
        --no-prof \
        --dump in,golden \
        --dump-format bin \
        --manual-data-dirs "${manual_data_dir}" || return 1

    local matrix_status=0
    local eager_result_csv="${RESULT_ROOT}/ttk_${suite_key}_eager_results.csv"
    run_logged \
        "TTK ${suite_label} eager (reuse input + Golden)" \
        "${RESULT_ROOT}/ttk_${suite_key}_eager.log" \
        python3 -m ttk e2e \
        -i "${case_csv}" \
        -o "${eager_result_csv}" \
        --csv-preserve \
        --plugin "${ASSETS_ROOT}" \
        --manual-data-dirs "${manual_data_dir}" \
        --device-whitelist=0 \
        --pc 1 \
        --warmup false \
        --run 1 || matrix_status=1
    echo "        result CSV: ${eager_result_csv}"

    if [[ "${run_aclgraph}" == "true" ]]; then
        local aclgraph_result_csv="${RESULT_ROOT}/ttk_${suite_key}_aclgraph_results.csv"
        run_logged \
            "TTK ${suite_label} ACLGraph (reuse input + Golden)" \
            "${RESULT_ROOT}/ttk_${suite_key}_aclgraph.log" \
            python3 -m ttk e2e \
            -i "${case_csv}" \
            -o "${aclgraph_result_csv}" \
            --csv-preserve \
            --plugin "${ASSETS_ROOT}" \
            --manual-data-dirs "${manual_data_dir}" \
            --aclgraph \
            --device-whitelist=0 \
            --pc 1 \
            --warmup false \
            --run 1 || matrix_status=1
        echo "        result CSV: ${aclgraph_result_csv}"
    fi

    return ${matrix_status}
}

run_ttk_redline() {
    run_ttk_matrix "redline" "MXFP8 redline without final two cases" "${TTK_REDLINE_CSV}" true
}

run_pytest_redline() {
    run_pytest_matrix "redline" "MXFP8 redline without final two cases" "${PYTEST_REDLINE_CSV}" true
}

run_ttk_stc() {
    run_ttk_matrix "stc" "MXFP8 STC" "${TTK_STC_CSV}" true
}

run_pytest_stc() {
    run_pytest_matrix "stc" "MXFP8 STC" "${PYTEST_STC_CSV}" true
}

run_ttk_generalized() {
    run_ttk_matrix "generalized_2000" "MXFP8 generalized 2000" "${TTK_GENERALIZED_CSV}" false
}

run_pytest_generalized() {
    run_pytest_matrix "generalized_2000" "MXFP8 generalized 2000" "${PYTEST_GENERALIZED_CSV}" false
}

overall_status=0

case "${RUN_MODE}" in
    all)
        prepare_redline_csvs || exit 1
        run_ttk_redline || overall_status=1
        run_pytest_redline || overall_status=1
        run_ttk_stc || overall_status=1
        run_pytest_stc || overall_status=1
        run_ttk_generalized || overall_status=1
        run_pytest_generalized || overall_status=1
        ;;
    redline)
        prepare_redline_csvs || exit 1
        run_ttk_redline || overall_status=1
        run_pytest_redline || overall_status=1
        ;;
    stc)
        run_ttk_stc || overall_status=1
        run_pytest_stc || overall_status=1
        ;;
    generalized)
        run_ttk_generalized || overall_status=1
        run_pytest_generalized || overall_status=1
        ;;
    ttk-redline)
        copy_without_last_two_cases \
            "${TTK_REDLINE_SOURCE}" "${TTK_REDLINE_CSV}" "TTK MXFP8 redline" || exit 1
        run_ttk_redline || overall_status=1
        ;;
    pytest-redline)
        copy_without_last_two_cases \
            "${PYTEST_REDLINE_SOURCE}" "${PYTEST_REDLINE_CSV}" "Pytest MXFP8 redline" || exit 1
        run_pytest_redline || overall_status=1
        ;;
    ttk-stc)
        run_ttk_stc || overall_status=1
        ;;
    pytest-stc)
        run_pytest_stc || overall_status=1
        ;;
    ttk-generalized)
        run_ttk_generalized || overall_status=1
        ;;
    pytest-generalized)
        run_pytest_generalized || overall_status=1
        ;;
    -h|--help|help)
        print_usage
        exit 0
        ;;
    *)
        echo "[ERROR] Unknown mode: ${RUN_MODE}" >&2
        print_usage >&2
        exit 2
        ;;
esac

echo
echo "Results directory: ${RESULT_ROOT}"
exit ${overall_status}
