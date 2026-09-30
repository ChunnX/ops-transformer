#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PACKAGE_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
BUILD_DIR="$PACKAGE_DIR/build"
mkdir -p "$BUILD_DIR"

echo "Running SageAttention2 Benchmark Suite..."
echo "[1/3] PyTorch Benchmark Runner..."
python3 "$PACKAGE_DIR/tests/benchmark_pytorch_runner.py" "$BUILD_DIR/pytorch_benchmark.json"

echo "[2/3] C++ aclnn Benchmark Runner..."
if [ -f "$BUILD_DIR/benchmark_fia_vs_sage" ]; then
    "$BUILD_DIR/benchmark_fia_vs_sage" --warmup 50 --repeats 100 --output "$BUILD_DIR/aclnn_benchmark.json"
else
    echo "Notice: $BUILD_DIR/benchmark_fia_vs_sage not yet built. Skipping C++ executable run."
fi

echo "[3/3] MSPROF Microarchitecture Gate Verification..."
python3 "$SCRIPT_DIR/msprof_analysis.py" --output-json "$BUILD_DIR/microarch_metrics.json"

echo "Benchmark suite completed successfully."
