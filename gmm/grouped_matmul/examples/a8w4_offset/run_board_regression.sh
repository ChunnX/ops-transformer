#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"
VISIBLE_DEVICE="${A8W4_VISIBLE_DEVICE:-0}"
RANDOM_ROUNDS="${A8W4_RANDOM_ROUNDS:-20}"
INCLUDE_FIXED_CASES="${A8W4_INCLUDE_FIXED_CASES:-1}"
RESULT_ROOT="${A8W4_BOARD_RESULT_DIR:-${REPO_ROOT}/build_out/a8w4_board_regression}"

case "${VISIBLE_DEVICE}" in
  1|5|6|7)
    echo "device ${VISIBLE_DEVICE} is reserved by another user; choose 0, 2, 3, or 4" >&2
    exit 2
    ;;
esac

run_matrix() {
  local api="$1"
  local format="$2"
  local result_dir="${RESULT_ROOT}/${api}_${format}"
  echo "[BOARD] api=${api} format=${format} visible_device=${VISIBLE_DEVICE}"
  env \
    ASCEND_RT_VISIBLE_DEVICES="${VISIBLE_DEVICE}" \
    A8W4_API="${api}" \
    A8W4_WEIGHT_FORMAT="${format}" \
    A8W4_RANDOM_ROUNDS="${RANDOM_ROUNDS}" \
    A8W4_INCLUDE_FIXED_CASES="${INCLUDE_FIXED_CASES}" \
    A8W4_MATRIX_RESULT_DIR="${result_dir}" \
    bash "${SCRIPT_DIR}/run_case_matrix.sh"
}

# The two V5 runs are the existing board paths. WeightNz/NZ is the newly
# connected frontend path. All three execute the same installed custom package.
run_matrix V5 ND
run_matrix V5 NZ
run_matrix WeightNz NZ

echo "board_regression_root=${RESULT_ROOT}"
