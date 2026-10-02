#!/usr/bin/env bash
# -----------------------------------------------------------------------------------------------------------
# SageAttention2 标准 OPP 自定义算子包安装脚本
# -----------------------------------------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../../.." && pwd)

echo "================================================================================"
echo "  SageAttention2 OPP Custom Operator Package Installer"
echo "  Target Operator: SageAttention2 (INT8 QK^T + FP8 PV, Ascend 950PR)"
echo "================================================================================"

RUN_FILE=""

# 1. 查找待安装的 .run 包文件
if [ $# -ge 1 ] && [[ "$1" == *.run ]] && [ -f "$1" ]; then
    RUN_FILE=$(realpath "$1")
    shift
fi

if [ -z "$RUN_FILE" ]; then
    # 按照优先级在常用构建输出路径中检索
    for cand in "${REPO_ROOT}/build_out"/cann-ops-transformer-custom_*.run \
                "${REPO_ROOT}/build"/cann-ops-transformer-custom_*.run \
                "${REPO_ROOT}/../hwdev/build"/cann-ops-transformer-custom_*.run \
                "${REPO_ROOT}/../hwdev/build/g_w_e"/cann-ops-transformer-custom_*.run; do
        if [ -f "$cand" ]; then
            RUN_FILE="$cand"
            break
        fi
    done
fi

if [ -z "$RUN_FILE" ] || [ ! -f "$RUN_FILE" ]; then
    echo "[ERROR] No valid cann-ops-transformer-custom_*.run package found!"
    echo "        Please build the package first by running:"
    echo "          bash ${SCRIPT_DIR}/build_opp.sh"
    echo "        Or specify the .run package path directly:"
    echo "          bash ${SCRIPT_DIR}/install_opp.sh /path/to/cann-ops-transformer-custom_linux-aarch64.run"
    exit 1
fi

echo "[INFO] Found OPP Custom Package: ${RUN_FILE}"
echo "       Package Size: $(ls -lh "$RUN_FILE" | awk '{print $5}')"

# 2. 准备安装选项
INSTALL_ARGS=("--quiet" "--install-for-all")
if [ $# -gt 0 ]; then
    INSTALL_ARGS=("$@")
fi

echo "[INFO] Executing installer with flags: ${INSTALL_ARGS[*]}"
bash "${RUN_FILE}" "${INSTALL_ARGS[@]}"

# 3. 验证安装后路径与环境变量建议
echo ""
echo "================================================================================"
echo "  [SUCCESS] SageAttention2 OPP Custom Package Installed Successfully!"
echo ""
echo "  Verification Checklist:"
if [ -n "${ASCEND_OPP_PATH:-}" ] && [ -d "${ASCEND_OPP_PATH}/vendors/customize" ]; then
    echo "  - ASCEND_OPP_PATH customize directory verified: ${ASCEND_OPP_PATH}/vendors/customize"
elif [ -d "/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize" ]; then
    echo "  - Standard Toolkit customize directory verified: /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize"
fi
echo ""
echo "  Environment configuration (if not set in current shell):"
echo "    export ASCEND_CUSTOM_OPP_PATH=\${ASCEND_OPP_PATH:-/usr/local/Ascend/ascend-toolkit/latest/opp}/vendors/customize"
echo "================================================================================"
