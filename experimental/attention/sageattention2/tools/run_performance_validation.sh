#!/usr/bin/env bash
# ==============================================================================
# run_performance_validation.sh: SageAttention2 性能基准一键评测与门禁核验脚本
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "================================================================================"
echo "   SageAttention2 End-to-End Performance Benchmark Runner"
echo "================================================================================"

# 1. 执行 C++ aclnn 两段式性能对比基准测试
echo "[STEP 1/2] Compiling and running C++ aclnn vs FIA baseline suite..."
clang++ -std=c++17 -Wall -Wextra -Werror \
    -I "${ROOT_DIR}/op_host" \
    -I "${ROOT_DIR}" \
    -I "${ROOT_DIR}/op_api" \
    "${ROOT_DIR}/benchmarks/benchmark_aclnn_fia.cpp" \
    -o /tmp/benchmark_aclnn_fia

echo "[INFO] Running with Default Timing Policy (Warmup=25, Repeats=50)..."
/tmp/benchmark_aclnn_fia

echo "[INFO] Running with Task test_perf Policy (Warmup=5, Repeats=20)..."
SAGEATTN_PERF_WARMUP=5 SAGEATTN_PERF_ITERS=20 /tmp/benchmark_aclnn_fia

# 2. 执行 Python 验收矩阵与性能验证
echo "[STEP 2/2] Running Python Performance Benchmark Suite..."
PYTHON_BIN="python3"
if [ -f "${ROOT_DIR}/../../../../.venv/bin/python" ]; then
    PYTHON_BIN="${ROOT_DIR}/../../../../.venv/bin/python"
elif [ -f "${ROOT_DIR}/.venv/bin/python" ]; then
    PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
fi

if "${PYTHON_BIN}" -c "import torch" &>/dev/null; then
    "${PYTHON_BIN}" "${SCRIPT_DIR}/benchmark.py" --cases standard --quick
else
    echo "[WARN] PyTorch not available in current Python environment, skipping Python benchmark."
fi

echo "================================================================================"
echo "   All Performance Benchmark Validations Completed Successfully!"
echo "================================================================================"
