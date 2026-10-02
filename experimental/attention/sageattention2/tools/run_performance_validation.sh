#!/usr/bin/env bash
# run_performance_validation.sh: SageAttention2 benchmark execution and gate verification
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./common.sh
source "${SCRIPT_DIR}/common.sh"

ROOT_DIR="$(get_sage_root_dir)"
PYTHON_BIN="$(detect_python_bin)"

compile_and_run_aclnn() {
    local bin_target="/tmp/benchmark_aclnn_fia"
    log_step "[1/2] Compiling and running C++ aclnn vs FIA baseline suite..."

    clang++ -std=c++17 -Wall -Wextra -Werror \
        -I "${ROOT_DIR}/op_host" \
        -I "${ROOT_DIR}" \
        -I "${ROOT_DIR}/op_api" \
        "${ROOT_DIR}/benchmarks/benchmark_aclnn_fia.cpp" \
        -o "${bin_target}"

    log_info "Running with default timing policy (Warmup=25, Repeats=50)..."
    "${bin_target}"

    log_info "Running with test_perf timing policy (Warmup=5, Repeats=20)..."
    SAGEATTN_PERF_WARMUP=5 SAGEATTN_PERF_ITERS=20 "${bin_target}"
}

run_python_benchmarks() {
    local py_bin="$1"
    log_step "[2/2] Running Python Performance Benchmark Suite..."

    if ! has_torch "${py_bin}"; then
        log_warn "PyTorch not available in Python environment (${py_bin}), skipping Python benchmark."
        return 0
    fi

    "${py_bin}" "${SCRIPT_DIR}/benchmark.py" --cases standard --quick

    if has_npu "${py_bin}"; then
        log_info "Running dual-policy on NPU (primary frozen 25/50, appendix test_perf 5/20)..."
        "${py_bin}" "${SCRIPT_DIR}/benchmark.py" --cases standard --metric-policy both \
            --output-json "${ROOT_DIR}/build/bench_dual_policy.json"
    else
        log_info "NPU unavailable: skipping dual-policy run (host simulation mode)."
    fi
}

main() {
    log_info "SageAttention2 End-to-End Performance Benchmark Runner"
    compile_and_run_aclnn
    run_python_benchmarks "${PYTHON_BIN}"
    log_success "All Performance Benchmark Validations Completed Successfully!"
}

main "$@"
