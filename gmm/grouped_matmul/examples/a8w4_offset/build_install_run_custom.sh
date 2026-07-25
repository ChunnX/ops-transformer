#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-all}"
case "${MODE}" in
  build|install|test-build|run-only|run|all) ;;
  *) echo "usage: $0 [build|install|test-build|run-only|run|all]" >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"
CANN_PATH="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann}"
VENDOR_NAME="${VENDOR_NAME:-a8w4test}"
VENDOR_DIR="${CANN_PATH}/opp/vendors/${VENDOR_NAME}_transformer"
PACKAGE="${REPO_ROOT}/build_out/cann-ops-transformer-${VENDOR_NAME}_linux-aarch64.run"
TEST_BIN="${REPO_ROOT}/build_out/test_aclnn_a8w4_offset_custom"
JOBS="${JOBS:-$(nproc)}"

build_package() {
  local pywrap="${REPO_ROOT}/third_party/pywrap"
  local build_path="${PATH}"
  if [[ -x "${pywrap}/python3" ]]; then
    build_path="${pywrap}:${build_path}"
  fi
  PATH="${build_path}" bash "${REPO_ROOT}/build.sh" \
    --pkg --soc=ascend950 --ops=grouped_matmul \
    --vendor_name="${VENDOR_NAME}" -j"${JOBS}"
}

install_package() {
  [[ -x "${PACKAGE}" ]] || { echo "package not found: ${PACKAGE}" >&2; exit 1; }
  # install.sh appends vendors/<vendor>_transformer itself. The root must be
  # .../opp, not .../opp/vendors, or an erroneous vendors/vendors is created.
  "${PACKAGE}" --quiet --install-path="${CANN_PATH}/opp" --force
}

build_test() {
  mkdir -p "$(dirname -- "${TEST_BIN}")"
  g++ -std=c++17 "${SCRIPT_DIR}/test_aclnn_a8w4_offset.cpp" \
    -I"${CANN_PATH}/include" -I"${CANN_PATH}/include/aclnnop" \
    -I"${VENDOR_DIR}/op_api/include" -I"${VENDOR_DIR}/op_api/include/aclnnop" \
    -L"${VENDOR_DIR}/op_api/lib" -Wl,-rpath,"${VENDOR_DIR}/op_api/lib" \
    -L"${CANN_PATH}/lib64" -Wl,-rpath,"${CANN_PATH}/lib64" \
    -Wl,-rpath-link,"${CANN_PATH}/aarch64-linux/devlib" \
    -Wl,--allow-shlib-undefined \
    -lcust_opapi -lopapi_math -lopapi -lascendcl -lnnopbase -lc_sec -lpthread \
    -o "${TEST_BIN}"
}

run_test_binary() {
  [[ -d "${VENDOR_DIR}" ]] || { echo "vendor package not installed: ${VENDOR_DIR}" >&2; exit 1; }
  local driver_root="${ASCEND_DRIVER_PATH:-/usr/local/Ascend/driver}"
  local driver_dirs=(
    "${driver_root}/lib64"
    "${driver_root}/lib64/common"
    "${driver_root}/lib64/driver"
  )
  local driver_ld_path=""
  local driver_dir
  local has_driver_hal=0
  for driver_dir in "${driver_dirs[@]}"; do
    if [[ -d "${driver_dir}" ]]; then
      driver_ld_path="${driver_ld_path:+${driver_ld_path}:}${driver_dir}"
    fi
    if [[ -r "${driver_dir}/libascend_hal.so" ]]; then
      has_driver_hal=1
    fi
  done
  if [[ ${has_driver_hal} -eq 0 ]]; then
    IFS=: read -r -a inherited_lib_dirs <<< "${LD_LIBRARY_PATH:-}"
    for driver_dir in "${inherited_lib_dirs[@]}"; do
      [[ "${driver_dir}" == *"/devlib"* ]] && continue
      if [[ -r "${driver_dir}/libascend_hal.so" ]]; then
        has_driver_hal=1
        break
      fi
    done
  fi
  if [[ ${has_driver_hal} -eq 0 ]]; then
    echo "Ascend driver runtime library libascend_hal.so is not mounted." >&2
    echo "Checked ${driver_root}/lib64{,/common,/driver} and LD_LIBRARY_PATH." >&2
    echo "CANN devlib contains compile stubs only and cannot be used for a board regression." >&2
    exit 127
  fi
  export ASCEND_CUSTOM_OPP_PATH="${VENDOR_DIR}${ASCEND_CUSTOM_OPP_PATH:+:${ASCEND_CUSTOM_OPP_PATH}}"
  export LD_LIBRARY_PATH="${VENDOR_DIR}/op_api/lib:${CANN_PATH}/lib64${driver_ld_path:+:${driver_ld_path}}:${LD_LIBRARY_PATH:-}"
  "${TEST_BIN}" non-null
}

run_test() {
  build_test
  run_test_binary
}

case "${MODE}" in
  build) build_package ;;
  install) install_package ;;
  test-build) build_test ;;
  run-only) run_test_binary ;;
  run) run_test ;;
  all) build_package; install_package; run_test ;;
esac
