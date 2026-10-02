#!/usr/bin/env bash
# profile_single_op.sh: SageAttention2 single operator profiling runner (msprof / sim)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./common.sh
source "${SCRIPT_DIR}/common.sh"

ROOT_DIR="$(get_sage_root_dir)"
OUTPUT_DIR="${ROOT_DIR}/prof_output"
CASE_NAME="${1:-P-01}"
PYTHON_BIN="$(detect_python_bin)"

run_msprof_profile() {
    local target_case="$1"
    local out_dir="$2"
    local py_bin="$3"

    mkdir -p "${out_dir}"
    log_info "Found msprof in PATH, executing hardware profiling..."
    log_info "Target case: ${target_case}"
    log_info "Output directory: ${out_dir}"

    msprof --output="${out_dir}" \
           --application="${py_bin} ${SCRIPT_DIR}/benchmark.py --device npu --cases ${target_case} --metric-policy test_perf" \
           --aic-mode=PipeUtilization \
           --l2-cache=on \
           --aicore-metrics=MemoryL0 \
           --ai-core=on

    log_success "Hardware profiling data collected at: ${out_dir}"
}

run_sim_profile() {
    local target_case="$1"
    local py_bin="$2"

    log_warn "msprof command not found in current environment (host development mode)."
    log_info "Running benchmark with Python profiling simulation instead..."
    "${py_bin}" "${SCRIPT_DIR}/benchmark.py" --device cpu --cases "${target_case}" --quick
}

main() {
    log_info "SageAttention2 Single Operator Profiling Pipeline (msprof)"
    log_info "Target Case      : ${CASE_NAME}"
    log_info "Output Directory : ${OUTPUT_DIR}"

    if command -v msprof &>/dev/null; then
        run_msprof_profile "${CASE_NAME}" "${OUTPUT_DIR}" "${PYTHON_BIN}"
    else
        run_sim_profile "${CASE_NAME}" "${PYTHON_BIN}"
    fi
}

main "$@"
