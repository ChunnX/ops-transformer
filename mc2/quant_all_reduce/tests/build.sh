#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build_standalone"
PROF="${PROF:-OFF}"

echo "=== QuantAllReduce Standalone Profiler Build ==="
echo "PROF=${PROF}"

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

cmake -DCMAKE_BUILD_TYPE=Debug \
      -DENABLE_PROFILING=${PROF} \
      "${SCRIPT_DIR}/standalone"

make -j$(nproc)

echo ""
echo "=== Build complete ==="
echo "Binary: ${BUILD_DIR}/standalone_profiler"
echo "Kernel: ${BUILD_DIR}/libquant_all_reduce_kernels.so"
echo ""
echo "Run example:"
echo "  cd ${BUILD_DIR}"
echo "  export LD_LIBRARY_PATH=\$(pwd):\$ASCEND_HOME_PATH/lib64:\$LD_LIBRARY_PATH"
echo "  ./standalone_profiler --bs 1024 --hidden_size 2048 --rankNum 2"
echo ""
echo "With profiler:"
echo "  PROF=ON bash ${SCRIPT_DIR}/build.sh"
