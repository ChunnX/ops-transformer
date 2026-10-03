#!/usr/bin/env bash
# install_opp.sh: CANN 标准自定义 OPP 算子部署与环境变量导出（无编译职责）
#
# 编译统一由 build_opp.sh 在 BiSheng 容器内完成；本脚本只做部署：
#   - 默认部署到算子本地 build/opp_installed（算子 .gitignore 已兜住）；
#   - 若仓库根存在官方整包产物 cann-ops-transformer-custom_*.run，优先以官方 run 包安装
#     （真机交付形态，语义不变）；
#   - --export-env 输出可直接 eval 的环境变量注入命令。

set -euo pipefail

OP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${OP_DIR}/../../.." && pwd)"

INSTALL_PATH=""
REBUILD=false
QUIET=false
EXPORT_ENV_ONLY=false
SOC="ascend950"
PASSTHROUGH_ARGS=()

print_usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --install-path=<path>    Custom installation path (default: build/opp_installed,"
    echo "                             or first entry of ASCEND_CUSTOM_OPP_PATH when set)"
    echo "  --rebuild                Force re-running build_opp.sh before deployment"
    echo "  --export-env             Print shell export commands for evaluation: eval \$($0 --export-env)"
    echo "  --soc=<soc>              Target SoC version (default: ascend950)"
    echo "  --quiet                  Suppress verbose progress output"
    echo "  -h, --help               Display this help message"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --install-path=*)
            INSTALL_PATH="${1#*=}"
            shift
            ;;
        --rebuild)
            REBUILD=true
            shift
            ;;
        --export-env)
            EXPORT_ENV_ONLY=true
            shift
            ;;
        --soc=*)
            # SOC 统一经下方 --soc="${SOC}" 单点传入 build_opp.sh（此处若再入
            # PASSTHROUGH_ARGS 会把同一旗标注入两次）
            SOC="${1#*=}"
            shift
            ;;
        --quiet)
            QUIET=true
            shift
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            PASSTHROUGH_ARGS+=("$1")
            shift
            ;;
    esac
done

# 确定默认安装目录：显式指定 > ASCEND_CUSTOM_OPP_PATH 首项 > 算子本地 build/opp_installed
if [ -z "${INSTALL_PATH}" ]; then
    if [ -n "${ASCEND_CUSTOM_OPP_PATH:-}" ]; then
        FIRST_PATH="${ASCEND_CUSTOM_OPP_PATH%%:*}"
        INSTALL_PATH="${FIRST_PATH%/vendors*}"
    else
        INSTALL_PATH="${OP_DIR}/build/opp_installed"
    fi
fi

# 获取规范绝对路径
mkdir -p "${INSTALL_PATH}"
INSTALL_PATH="$(cd "${INSTALL_PATH}" && pwd)"

# 若请求仅导出环境变量，直接输出标准命令并退出
if [ "${EXPORT_ENV_ONLY}" = true ]; then
    echo "export ASCEND_CUSTOM_OPP_PATH=\"${INSTALL_PATH}:\${ASCEND_CUSTOM_OPP_PATH:-}\""
    echo "export LD_LIBRARY_PATH=\"${INSTALL_PATH}/vendors/customize/op_api/lib:${INSTALL_PATH}/vendors/customize/op_impl/ai_core/tbe/op_tiling/lib:\${LD_LIBRARY_PATH:-}\""
    exit 0
fi

if [ "${QUIET}" = false ]; then
    echo "================================================================================"
    echo "SageAttention2 CANN Custom OPP Installer"
    echo "Operator directory: ${OP_DIR}"
    echo "Target installation: ${INSTALL_PATH}"
    echo "Target SoC:          ${SOC}"
    echo "================================================================================"
fi

PKG_SRC_DIR="${OP_DIR}/build/opp"

# 1. 检查或触发 build_opp.sh（BiSheng 容器内）出包
if [ "${REBUILD}" = true ] || [ ! -d "${PKG_SRC_DIR}/vendors/customize" ]; then
    if [ "${QUIET}" = false ]; then
        echo ">>> OPP package not found or rebuild requested. Invoking build_opp.sh..."
    fi
    bash "${OP_DIR}/build_opp.sh" --output-dir="${PKG_SRC_DIR}" --soc="${SOC}" "${PASSTHROUGH_ARGS[@]}"
fi

# 2. 如果存在官方整包出包的 .run 安装包，优先调用官方 run 脚本（真机交付形态）
# （find 只输出已存在的路径，非空判定即为存在性判定）
RUN_PKG=$(find "${REPO_DIR}/build_out" -name "cann-ops-transformer-custom_*.run" 2>/dev/null | head -n 1 || true)
if [ -n "${RUN_PKG}" ]; then
    if [ "${QUIET}" = false ]; then
        echo ">>> Executing official CANN custom opp run package: ${RUN_PKG}..."
    fi
    bash "${RUN_PKG}" --quiet --install-path="${INSTALL_PATH}"
    CONFIG_FILE="${INSTALL_PATH}/vendors/config.ini"
    if [ ! -f "${CONFIG_FILE}" ]; then
        cat << 'EOF' > "${CONFIG_FILE}"
[common]
load_priority=custom_transformer,customize
EOF
    fi
else
    # 本地标准 Vendor 目录部署
    if [ "${QUIET}" = false ]; then
        echo ">>> Deploying vendor tree to ${INSTALL_PATH}/vendors..."
    fi

    mkdir -p "${INSTALL_PATH}/vendors/customize"

    # 同步 op_api, op_proto, op_impl, bin
    cp -rf "${PKG_SRC_DIR}/vendors/customize/"* "${INSTALL_PATH}/vendors/customize/"

    # 配置 vendors/config.ini 优先级
    CONFIG_FILE="${INSTALL_PATH}/vendors/config.ini"
    if [ ! -f "${CONFIG_FILE}" ]; then
        cat << 'EOF' > "${CONFIG_FILE}"
[common]
load_priority=customize
EOF
    else
        # 确保 load_priority 包含 customize
        if ! grep -q "customize" "${CONFIG_FILE}"; then
            sed -i 's/load_priority=/load_priority=customize,/' "${CONFIG_FILE}" 2>/dev/null || true
        fi
    fi

    # 更新部署目录下的 set_env.bash 路径
    cat << EOF > "${INSTALL_PATH}/vendors/customize/bin/set_env.bash"
#!/usr/bin/env bash
export ASCEND_CUSTOM_OPP_PATH="${INSTALL_PATH}:\${ASCEND_CUSTOM_OPP_PATH:-}"
export LD_LIBRARY_PATH="${INSTALL_PATH}/vendors/customize/op_api/lib:${INSTALL_PATH}/vendors/customize/op_impl/ai_core/tbe/op_tiling/lib:\${LD_LIBRARY_PATH:-}"
EOF
    chmod +x "${INSTALL_PATH}/vendors/customize/bin/set_env.bash"
fi

# 3. 部署后自检验证
if [ "${QUIET}" = false ]; then
    echo ">>> Verifying deployed OPP components..."
fi

test -f "${INSTALL_PATH}/vendors/config.ini" || { echo "[ERROR] ${INSTALL_PATH}/vendors/config.ini missing!"; exit 1; }
test -f "${INSTALL_PATH}/vendors/customize/op_api/lib/libcust_opapi.so" || { echo "[ERROR] libcust_opapi.so missing!"; exit 1; }
test -f "${INSTALL_PATH}/vendors/customize/op_impl/ai_core/tbe/op_tiling/lib/liboptiling.so" || { echo "[ERROR] liboptiling.so missing!"; exit 1; }
test -f "${INSTALL_PATH}/vendors/customize/bin/set_env.bash" || { echo "[ERROR] set_env.bash missing!"; exit 1; }

if [ "${QUIET}" = false ]; then
    echo "================================================================================"
    echo ">>> SageAttention2 CANN Custom OPP Successfully Installed!"
    echo "Installation directory: ${INSTALL_PATH}/vendors"
    echo "Active vendor:          customize"
    echo "To enable environment in your current shell, execute:"
    echo "  source \"${INSTALL_PATH}/vendors/customize/bin/set_env.bash\""
    echo "or evaluate dynamically:"
    echo "  eval \$(${OP_DIR}/install_opp.sh --install-path=\"${INSTALL_PATH}\" --export-env)"
    echo "================================================================================"
fi

exit 0
