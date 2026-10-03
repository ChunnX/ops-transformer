#!/usr/bin/env bash
# install_opp.sh: CANN 官方标准自定义 OPP 算子部署与环境变量导出工具

set -euo pipefail

OP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${OP_DIR}/../../.." && pwd)"

INSTALL_PATH=""
DRY_RUN=false
MOCK_MODE=false
REBUILD=false
QUIET=false
EXPORT_ENV_ONLY=false
SOC="ascend950"
PASSTHROUGH_ARGS=()

# 颜色控制
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

print_usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --install-path=<path>    Custom installation path (default: auto-detected or build/opp_installed)"
    echo "  --dry-run                Perform Dry-Run deployment without modifying system files"
    echo "  --mock                   Mock installation mode (same as --dry-run)"
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
        --dry-run)
            DRY_RUN=true
            MOCK_MODE=true
            PASSTHROUGH_ARGS+=("--dry-run")
            shift
            ;;
        --mock)
            MOCK_MODE=true
            PASSTHROUGH_ARGS+=("--mock")
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
            SOC="${1#*=}"
            PASSTHROUGH_ARGS+=("$1")
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

# 确定默认安装目录
if [ -z "${INSTALL_PATH}" ]; then
    if [ -n "${ASCEND_HOME_PATH:-}" ] && [ -d "${ASCEND_HOME_PATH}/opp" ] && [ "${MOCK_MODE}" = false ]; then
        INSTALL_PATH="${ASCEND_HOME_PATH}/opp"
    elif [ -n "${ASCEND_CUSTOM_OPP_PATH:-}" ]; then
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
    echo -e "${CYAN}================================================================================${NC}"
    echo -e "${CYAN}SageAttention2 CANN Custom OPP Installer (install_opp.sh)${NC}"
    echo -e "${CYAN}Operator directory: ${OP_DIR}${NC}"
    echo -e "${CYAN}Target installation: ${INSTALL_PATH}${NC}"
    echo -e "${CYAN}Target SoC:          ${SOC}${NC}"
    echo -e "${CYAN}================================================================================${NC}"
fi

PKG_SRC_DIR="${OP_DIR}/build/opp"

# 1. 检查或触发 build_opp.sh 出包
if [ "${REBUILD}" = true ] || [ ! -d "${PKG_SRC_DIR}/vendors/customize" ]; then
    if [ "${QUIET}" = false ]; then
        echo -e "${BLUE}>>> OPP package not found or rebuild requested. Invoking build_opp.sh...${NC}"
    fi
    bash "${OP_DIR}/build_opp.sh" --output-dir="${PKG_SRC_DIR}" --soc="${SOC}" "${PASSTHROUGH_ARGS[@]}"
fi

# 2. 如果存在官方出包生成的 .run 安装包且处于非 mock 模式，优先调用官方 run 脚本
RUN_PKG=$(find "${REPO_DIR}/build_out" -name "cann-ops-transformer-custom_*.run" 2>/dev/null | head -n 1 || true)
if [ -n "${RUN_PKG}" ] && [ -f "${RUN_PKG}" ] && [ "${MOCK_MODE}" = false ] && [ "${DRY_RUN}" = false ]; then
    if [ "${QUIET}" = false ]; then
        echo -e "${GREEN}>>> Executing official CANN custom opp run package: ${RUN_PKG}...${NC}"
    fi
    bash "${RUN_PKG}" --quiet --install-path="${INSTALL_PATH}"
else
    # 本地标准 Vendor 目录部署
    if [ "${QUIET}" = false ]; then
        echo -e "${BLUE}>>> Deploying vendor tree to ${INSTALL_PATH}/vendors...${NC}"
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
            sed -i.bak 's/load_priority=/load_priority=customize,/' "${CONFIG_FILE}" 2>/dev/null || \
            sed -i '' 's/load_priority=/load_priority=customize,/' "${CONFIG_FILE}" 2>/dev/null || true
            rm -f "${CONFIG_FILE}.bak"
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
    echo -e "${BLUE}>>> Verifying deployed OPP components...${NC}"
fi

test -f "${INSTALL_PATH}/vendors/config.ini" || test -f "${INSTALL_PATH}/vendors/custom_transformer/config.ini" || { echo -e "${RED}[ERROR] ${INSTALL_PATH}/vendors/config.ini missing!${NC}"; exit 1; }
test -f "${INSTALL_PATH}/vendors/customize/op_api/lib/libcust_opapi.so" || test -f "${INSTALL_PATH}/vendors/custom_transformer/op_api/lib/libcust_opapi.so" || { echo -e "${RED}[ERROR] libcust_opapi.so missing!${NC}"; exit 1; }
test -f "${INSTALL_PATH}/vendors/customize/op_impl/ai_core/tbe/op_tiling/lib/liboptiling.so" || test -f "${INSTALL_PATH}/vendors/custom_transformer/op_impl/ai_core/tbe/op_tiling/lib/liboptiling.so" || test -f "${INSTALL_PATH}/vendors/custom_transformer/op_impl/ai_core/tbe/op_tiling/liboptiling.so" || test -f "${INSTALL_PATH}/vendors/customize/op_impl/ai_core/tbe/op_tiling/liboptiling.so" || { echo -e "${RED}[ERROR] liboptiling.so missing!${NC}"; exit 1; }
test -f "${INSTALL_PATH}/vendors/customize/bin/set_env.bash" || test -f "${INSTALL_PATH}/vendors/custom_transformer/bin/set_env.bash" || { echo -e "${RED}[ERROR] set_env.bash missing!${NC}"; exit 1; }

if [ "${QUIET}" = false ]; then
    echo -e "${GREEN}================================================================================${NC}"
    echo -e "${GREEN}>>> SageAttention2 CANN Custom OPP Successfully Installed!${NC}"
    echo -e "${GREEN}Installation directory: ${INSTALL_PATH}/vendors${NC}"
    echo -e "${GREEN}Active vendor:          customize${NC}"
    echo -e "${GREEN}To enable environment in your current shell, execute:${NC}"
    echo -e "  ${YELLOW}source \"${INSTALL_PATH}/vendors/customize/bin/set_env.bash\"${NC}"
    echo -e "or evaluate dynamically:"
    echo -e "  ${YELLOW}eval \$(\"${OP_DIR}/install_opp.sh\" --install-path=\"${INSTALL_PATH}\" --export-env)${NC}"
    echo -e "${GREEN}================================================================================${NC}"
fi

exit 0
