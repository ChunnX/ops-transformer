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

PYTEST_STC_CSV="${PYTEST_ROOT}/qbsa_mxfp8_stc.csv"
TTK_STC_CSV="${TTK_ROOT}/qbsa_mxfp8_stc.csv"
TTK_GENERALIZED_CSV="${TTK_ROOT}/qbsa_mxfp8_generalized_2000.csv"

mkdir -p "${RESULT_ROOT}"
cd "${REPO_ROOT}"

print_usage() {
    echo "Usage: bash $0 [all|pytest-stc|ttk-stc|ttk-generalized]"
    echo
    echo "Modes:"
    echo "  all              Run all three suites in sequence (default)."
    echo "  pytest-stc       Run Pytest MXFP8 STC cases."
    echo "  ttk-stc          Run TTK MXFP8 STC cases."
    echo "  ttk-generalized  Run TTK MXFP8 generalized 2000 cases."
    echo
    echo "Set QBSA_RESULT_ROOT to override the output directory."
}

require_file() {
    local path="$1"
    if [[ ! -f "${path}" ]]; then
        echo "[ERROR] Input file does not exist: ${path}" >&2
        return 1
    fi
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

run_pytest_stc() {
    require_file "${PYTEST_STC_CSV}" || return 1
    run_logged \
        "Pytest MXFP8 STC" \
        "${RESULT_ROOT}/pytest_mxfp8_stc.log" \
        python3 "${PYTEST_ROOT}/qbsa_mxfp8_golden.py" \
        --csv \
        --case_files "${PYTEST_STC_CSV}" \
        --mode all
}

run_ttk_stc() {
    require_file "${TTK_STC_CSV}" || return 1
    local result_csv="${RESULT_ROOT}/ttk_mxfp8_stc_results.csv"
    run_logged \
        "TTK MXFP8 STC" \
        "${RESULT_ROOT}/ttk_mxfp8_stc.log" \
        python3 -m ttk e2e \
        -i "${TTK_STC_CSV}" \
        -o "${result_csv}" \
        --csv-preserve \
        --plugin "${ASSETS_ROOT}" \
        --device-whitelist=0 \
        --pc 1 \
        --warmup false \
        --run 1
    local status=$?
    echo "        result CSV: ${result_csv}"
    return ${status}
}

run_ttk_generalized() {
    require_file "${TTK_GENERALIZED_CSV}" || return 1
    local result_csv="${RESULT_ROOT}/ttk_mxfp8_generalized_2000_results.csv"
    run_logged \
        "TTK MXFP8 generalized 2000" \
        "${RESULT_ROOT}/ttk_mxfp8_generalized_2000.log" \
        python3 -m ttk e2e \
        -i "${TTK_GENERALIZED_CSV}" \
        -o "${result_csv}" \
        --csv-preserve \
        --plugin "${ASSETS_ROOT}" \
        --device-whitelist=0 \
        --pc 1 \
        --warmup false \
        --run 1
    local status=$?
    echo "        result CSV: ${result_csv}"
    return ${status}
}

overall_status=0

case "${RUN_MODE}" in
    all)
        run_pytest_stc || overall_status=1
        run_ttk_stc || overall_status=1
        run_ttk_generalized || overall_status=1
        ;;
    pytest-stc)
        run_pytest_stc || overall_status=1
        ;;
    ttk-stc)
        run_ttk_stc || overall_status=1
        ;;
    ttk-generalized)
        run_ttk_generalized || overall_status=1
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
