#!/usr/bin/env bash
set -euo pipefail
MODE="${1:?usage: build_and_run_aclnn.sh null|non-null}"
if [[ "${MODE}" != "null" && "${MODE}" != "non-null" ]]; then
  echo "mode must be null or non-null" >&2
  exit 2
fi
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CANN_PATH="${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}"
BIN="${SCRIPT_DIR}/test_aclnn_s8s4_offset"
g++ -std=c++17 "${SCRIPT_DIR}/test_aclnn_s8s4_offset.cpp" \
  -I"${CANN_PATH}/include" -I"${CANN_PATH}/include/aclnnop" \
  -L"${CANN_PATH}/lib64" -Wl,-rpath,"${CANN_PATH}/lib64" \
  -lopapi_transformer -lopapi_math -lopapi -lascendcl -lnnopbase -lc_sec -lpthread \
  -o "${BIN}"
"${BIN}" "${MODE}"
