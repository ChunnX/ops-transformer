#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# See LICENSE in the root of the software repository for the full text of the License.

# End-to-end Ascend 910B precision test for GroupedMatmulQuant.
#
# Optional environment variables:
#   CANN_ENV_SCRIPT  CANN environment script.
#                    Default: /usr/local/Ascend/ascend-toolkit/set_env.sh
#   PYTHON_BIN       Python executable. Default: python3
#   VENDOR_NAME      Custom operator vendor name. Default: gmm_quant
#   SOC_VERSION      build.sh SoC value. Default: ascend910b
#   OPP_INSTALL_ROOT Custom operator installation root.
#                    Default: ${ASCEND_HOME_PATH}/opp
#   LOG_DIR          Test log directory.
#                    Default: <repo>/build_out/grouped_matmul_quant_precision_logs

set -Eeuo pipefail

log()
{
    printf '\n[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

die()
{
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

on_error()
{
    local exit_code=$?
    printf 'ERROR: command failed at line %s (exit code %s)\n' "${BASH_LINENO[0]}" "${exit_code}" >&2
    exit "${exit_code}"
}

trap on_error ERR

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
EXTENSION_DIR="${REPO_ROOT}/experimental/npu_ops_transformer_ext"
TEST_DIR="${SCRIPT_DIR}/tests"

CANN_ENV_SCRIPT="${CANN_ENV_SCRIPT:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENDOR_NAME="${VENDOR_NAME:-gmm_quant}"
SOC_VERSION="${SOC_VERSION:-ascend910b}"

[[ -f "${REPO_ROOT}/build.sh" ]] || die "Cannot locate repository root: ${REPO_ROOT}"
[[ -f "${CANN_ENV_SCRIPT}" ]] || die "CANN environment script not found: ${CANN_ENV_SCRIPT}"
command -v "${PYTHON_BIN}" >/dev/null 2>&1 || die "Python executable not found: ${PYTHON_BIN}"

log "Loading CANN environment: ${CANN_ENV_SCRIPT}"
# shellcheck disable=SC1090
set +u
source "${CANN_ENV_SCRIPT}"
set -u

[[ -n "${ASCEND_HOME_PATH:-}" ]] || die "ASCEND_HOME_PATH is empty after sourcing ${CANN_ENV_SCRIPT}"
OPP_INSTALL_ROOT="${OPP_INSTALL_ROOT:-${ASCEND_HOME_PATH}/opp}"
VENDOR_ENV_SCRIPT="${OPP_INSTALL_ROOT}/vendors/${VENDOR_NAME}_transformer/bin/set_env.bash"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/build_out/grouped_matmul_quant_precision_logs}"

log "Checking Python, torch_npu, and NPU availability"
"${PYTHON_BIN}" --version
command -v npu-smi >/dev/null 2>&1 || die "npu-smi was not found in PATH"
npu-smi info
"${PYTHON_BIN}" - <<'PY'
import torch
import torch_npu

print("torch:", torch.__version__)
print("torch_npu:", torch_npu.__version__)
print("NPU available:", torch.npu.is_available())
print("NPU count:", torch.npu.device_count())
if not torch.npu.is_available() or torch.npu.device_count() < 1:
    raise RuntimeError("No available Ascend NPU was detected")
PY

log "Building GroupedMatmulQuant custom operator package"
cd "${REPO_ROOT}"
bash build.sh \
    --pkg \
    --experimental \
    --ops=grouped_matmul_quant \
    --soc="${SOC_VERSION}" \
    --vendor_name="${VENDOR_NAME}"

shopt -s nullglob
packages=("${REPO_ROOT}"/build_out/cann-ops-transformer-"${VENDOR_NAME}"_linux*.run)
shopt -u nullglob
(( ${#packages[@]} > 0 )) || die "Custom operator package was not generated under ${REPO_ROOT}/build_out"

package_path="${packages[0]}"
for candidate in "${packages[@]:1}"; do
    if [[ "${candidate}" -nt "${package_path}" ]]; then
        package_path="${candidate}"
    fi
done

log "Installing custom operator package: ${package_path}"
chmod +x "${package_path}"
"${package_path}" --install-path="${OPP_INSTALL_ROOT}"

[[ -f "${VENDOR_ENV_SCRIPT}" ]] || die "Vendor environment script not found: ${VENDOR_ENV_SCRIPT}"
# shellcheck disable=SC1090
set +u
source "${VENDOR_ENV_SCRIPT}"
set -u
[[ -n "${ASCEND_CUSTOM_OPP_PATH:-}" ]] || die "ASCEND_CUSTOM_OPP_PATH is empty after vendor activation"
printf 'ASCEND_CUSTOM_OPP_PATH=%s\n' "${ASCEND_CUSTOM_OPP_PATH}"

log "Installing Python test dependencies"
"${PYTHON_BIN}" -m pip install pytest numpy

log "Building the PyTorch registration extension for grouped_matmul_quant only"
mkdir -p "${LOG_DIR}"
cd "${EXTENSION_DIR}"
"${PYTHON_BIN}" setup.py clean
export NPU_OPS_TRANSFORMER_EXT_OPS=grouped_matmul_quant
"${PYTHON_BIN}" -m pip install \
    --verbose \
    --force-reinstall \
    --no-build-isolation \
    --no-deps \
    -e . \
    2>&1 | tee "${LOG_DIR}/grouped_matmul_quant_extension_build.log"

log "Checking the operator schema and PrivateUse1 implementation"
"${PYTHON_BIN}" - <<'PY'
import torch
import torch_npu
import npu_ops_transformer_ext

qualified_name = "npu_ops_transformer_ext::grouped_matmul_quant"
operator = torch.ops.npu_ops_transformer_ext.grouped_matmul_quant
has_npu_kernel = torch._C._dispatch_has_kernel_for_dispatch_key(qualified_name, "PrivateUse1")

print("package:", npu_ops_transformer_ext.__file__)
print("extension:", npu_ops_transformer_ext._C.__file__)
print("operator:", operator)
print("PrivateUse1 kernel:", has_npu_kernel)
if not has_npu_kernel:
    raise RuntimeError(f"{qualified_name} has no PrivateUse1 implementation")
PY

log "Running GroupedMatmulQuant CPU reference and Ascend NPU precision cases"
cd "${TEST_DIR}"
"${PYTHON_BIN}" -m pytest -sv test_grouped_matmul_quant.py \
    2>&1 | tee "${LOG_DIR}/grouped_matmul_quant_precision.log"

log "Precision test completed successfully"
printf 'Logs:\n'
printf '  %s\n' "${LOG_DIR}/grouped_matmul_quant_extension_build.log"
printf '  %s\n' "${LOG_DIR}/grouped_matmul_quant_precision.log"
