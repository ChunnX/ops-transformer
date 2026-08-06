#!/bin/bash
# AttnResFwd Torch 测试：CANN toolkit + custom_transformer OPP + torch_extension PYTHONPATH
# Usage: source cann_env.sh
#
# 默认 CANN 基线：cann-9.2.0-weekly（精度调试主线）
# 可覆盖：export CANN_PKG=cann-9.1.0-beta.3-20260722

set -euo pipefail

CANN_PKG="${CANN_PKG:-cann-9.2.0-weekly.20260715.01-20260715}"
CANN_PKG_DIR="/data/kdy/cann/pkg/${CANN_PKG}"
CANN_PKG_SETENV="${CANN_PKG_DIR}/ascend-toolkit/set_env.sh"

export ATTN_RES_FWD_CANN_PKG="${CANN_PKG}"
export ATTN_RES_FWD_CANN_SETENV="${CANN_PKG_SETENV}"

if [[ ! -f "${CANN_PKG_SETENV}" ]]; then
    echo "ERROR: CANN set_env not found: ${CANN_PKG_SETENV}" >&2
    echo "Hint: export CANN_PKG=cann-9.1.0-beta.3-20260722" >&2
    return 1 2>/dev/null || exit 1
fi

# shellcheck disable=SC1090
source "${CANN_PKG_SETENV}"

CUSTOM_SETENV_CANDIDATES=(
    "${CUSTOM_INSTALL_PATH:+${CUSTOM_INSTALL_PATH}/vendors/custom_transformer/bin/set_env.bash}"
    "/data/kdy/cann/custompkgbase/vendors/custom_transformer/bin/set_env.bash"
    "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)/custompkg_install/vendors/custom_transformer/bin/set_env.bash"
    "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)/build_out/vendors/custom_transformer/bin/set_env.bash"
)

# custom set_env.bash 可能读写 ASCEND_CUSTOM_OPP_PATH；先占位避免 set -u 报错
export ASCEND_CUSTOM_OPP_PATH="${ASCEND_CUSTOM_OPP_PATH:-}"

for path in "${CUSTOM_SETENV_CANDIDATES[@]}"; do
    if [[ -f "${path}" ]]; then
        # shellcheck disable=SC1090
        source "${path}"
        export ATTN_RES_FWD_CUSTOM_SETENV="${path}"
        break
    fi
done

# JIT OpBuilder 编译使用 CANN set_env 设置的 ASCEND_HOME_PATH（勿覆盖为 ascend-toolkit/latest）
export ASCEND_GLOBAL_LOG_LEVEL="${ASCEND_GLOBAL_LOG_LEVEL:-4}"
export ASCEND_SLOG_PRINT_TO_STDOUT="${ASCEND_SLOG_PRINT_TO_STDOUT:-0}"
export ASCEND_GLOBAL_EVENT_ENABLE="${ASCEND_GLOBAL_EVENT_ENABLE:-0}"

TORCH_EXT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../torch_extension" && pwd)"
export PYTHONPATH="${TORCH_EXT_ROOT}:${PYTHONPATH:-}"
