#!/usr/bin/env bash
# run.sh — SageAttention2 门禁统一入口（唯一宿主依赖：Docker + compose 插件）。
#
# 宿主侧：仅做一键初始化校验与容器包装（docker/sage run），不在宿主做任何编译；
# 容器侧：全部编译与测试统一由 BiSheng 承载——C++ 宿主单测走 BiSheng 宿主模式，
# 设备核走 BiSheng 交叉编译（dav-3510），不存在 clang++/g++ 等任何其他编译器路径。
#
# 用法：bash run.sh [host|kernel|python|opp|all]
#   host   BiSheng 编译并运行 C++ 宿主单测（test_aclnn_host / test_op_host / test_tiling_struct）
#   kernel BiSheng 交叉编译设备核 → build/sageattention2_bisheng.o
#   python 容器内 python 门禁（tiling_struct / quant / kernel_twin / gqa_chunk）
#   opp    OPP vendor 树 BiSheng 出包（build_opp.sh）+ 部署（install_opp.sh）
#   all    host → kernel → python → opp

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${SCRIPT_DIR}"
BUILD_DIR="${WORK_DIR}/build"

# ---------------------------------------------------------------- 宿主侧：容器包装
if [ "${SAGE_IN_CONTAINER:-}" != "1" ]; then
    command -v docker >/dev/null 2>&1 \
        || { echo "ERROR: 宿主未安装 docker（门禁唯一宿主依赖，含 compose 插件）。" >&2; exit 1; }
    # 一键初始化兜底：CATLASS（docker/third_party/）未置备时自动补齐（幂等）
    if [ ! -f "${WORK_DIR}/docker/third_party/catlass/include/catlass/catlass.hpp" ]; then
        bash "${WORK_DIR}/docker/init.sh"
    fi
    exec bash "${WORK_DIR}/docker/sage" run bash run.sh "$@"
fi

# ---------------------------------------------------------------- 容器侧
# CANN / BiSheng 环境解析（唯一定义点）：镜像 ENV 已导出 ASCEND_HOME_PATH；
# 此处兜底探测不同 CANN 包布局（ascend-toolkit/latest 或 cann-<version>）。
if [ -z "${ASCEND_HOME_PATH:-}" ] || [ ! -x "${ASCEND_HOME_PATH}/bin/bisheng" ]; then
    ASCEND_HOME_PATH=""
    for cand in /usr/local/Ascend/ascend-toolkit/latest /usr/local/Ascend/cann-*; do
        if [ -x "${cand}/bin/bisheng" ]; then
            ASCEND_HOME_PATH="${cand}"
            break
        fi
    done
fi
if [ -z "${ASCEND_HOME_PATH:-}" ] || [ ! -x "${ASCEND_HOME_PATH}/bin/bisheng" ]; then
    echo "ERROR: 容器内未找到 BiSheng 编译器（ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-unset}）。" >&2
    exit 1
fi
export ASCEND_HOME_PATH
case ":${PATH}:" in
    *":${ASCEND_HOME_PATH}/bin:"*) ;;
    *) export PATH="${ASCEND_HOME_PATH}/bin:${PATH}" ;;
esac
BISHENG="${ASCEND_HOME_PATH}/bin/bisheng"

# CATLASS 解析（唯一定义点，与 cmake/SageAttnDeps.cmake 同链）：CATLASS_ROOT 覆盖
# → init.sh 置备的 docker/third_party/catlass
CATLASS_ROOT="${CATLASS_ROOT:-${WORK_DIR}/docker/third_party/catlass}"
if [ ! -f "${CATLASS_ROOT}/include/catlass/catlass.hpp" ]; then
    echo "ERROR: CATLASS 未置备（${CATLASS_ROOT}）。请在宿主执行 bash docker/init.sh。" >&2
    exit 1
fi
export CATLASS_ROOT
CATLASS_INC="${CATLASS_ROOT}/include"

echo "================================================================="
echo "SageAttention2 Build & Test Runner (container / BiSheng only)"
echo "Working directory : ${WORK_DIR}"
echo "BiSheng           : ${BISHENG}"
echo "CATLASS_ROOT      : ${CATLASS_ROOT}"
echo "================================================================="

HOST_CXX_FLAGS=(-std=c++17 -O2 -Wall -Wextra)
COMMON_INCLUDES=(
    -I"${WORK_DIR}/sageattn_fp8/ascendc"
    -I"${WORK_DIR}/sageattn_fp8/catlass"
    -I"${WORK_DIR}/quant"
    -I"${CATLASS_INC}"
)

run_host_tests() {
    echo ">>> [host] C++ 宿主单测（BiSheng 宿主模式，唯一编译器）..."
    mkdir -p "${BUILD_DIR}"

    echo "-- 编译 test_aclnn_host"
    "${BISHENG}" "${HOST_CXX_FLAGS[@]}" \
        "${WORK_DIR}/tests/test_aclnn_host.cpp" \
        "${WORK_DIR}/aclnn/sage_attn_host.cpp" \
        -I"${WORK_DIR}/aclnn" \
        "${COMMON_INCLUDES[@]}" \
        -o "${BUILD_DIR}/test_aclnn_host"
    "${BUILD_DIR}/test_aclnn_host"

    echo "-- 编译 test_op_host (REG_OP & InferShape)"
    "${BISHENG}" "${HOST_CXX_FLAGS[@]}" \
        "${WORK_DIR}/tests/test_op_host.cpp" \
        "${WORK_DIR}/op_host/sage_attention2_def.cpp" \
        "${WORK_DIR}/op_host/sage_attention2_infershape.cpp" \
        -I"${WORK_DIR}/op_host" \
        -I"${WORK_DIR}/aclnn" \
        -o "${BUILD_DIR}/test_op_host"
    "${BUILD_DIR}/test_op_host"

    echo "-- 编译 test_tiling_struct"
    "${BISHENG}" "${HOST_CXX_FLAGS[@]}" \
        "${WORK_DIR}/tests/test_tiling_struct.cpp" \
        "${COMMON_INCLUDES[@]}" \
        -o "${BUILD_DIR}/test_tiling_struct"
    "${BUILD_DIR}/test_tiling_struct"

    echo ">>> [host] C++ 宿主单测全部通过"
}

compile_kernel_bisheng() {
    echo ">>> [kernel] BiSheng 交叉编译设备核 → Ascend 950 (dav-3510)..."
    mkdir -p "${BUILD_DIR}"
    "${BISHENG}" --npu-arch=dav-3510 -std=c++17 -xasc -c \
        "${WORK_DIR}/op_kernel/sageattention2.cpp" \
        -I"${ASCEND_HOME_PATH}/include" \
        "${COMMON_INCLUDES[@]}" \
        -o "${BUILD_DIR}/sageattention2_bisheng.o"
    echo ">>> [kernel] 设备核编译成功: ${BUILD_DIR}/sageattention2_bisheng.o"
}

run_python_tests() {
    echo ">>> [python] Python 门禁（容器内 python + torch，测试动态库由 BiSheng 编译）..."
    local PY_BIN="${SAGE_PYTHON:-python3}"
    echo "Using python: $("${PY_BIN}" -c 'import sys; print(sys.executable, sys.version.split()[0])')"

    echo "-- test_tiling_struct.py"
    "${PY_BIN}" -m unittest "${WORK_DIR}/tests/test_tiling_struct.py"

    echo "-- test_quant.py"
    "${PY_BIN}" "${WORK_DIR}/tests/test_quant.py"

    echo "-- test_kernel_twin.py (宿主孪生执行器与 kernel 宿主串行分支 vs golden)"
    "${PY_BIN}" "${WORK_DIR}/tests/test_kernel_twin.py"

    echo "-- test_gqa_chunk.py (GQA 8:1 分块打包双路径位级一致性)"
    "${PY_BIN}" "${WORK_DIR}/tests/test_gqa_chunk.py"

    echo ">>> [python] Python 门禁全部通过"
}

run_opp_tests() {
    echo ">>> [opp] CANN 标准自定义 OPP 出包（build_opp.sh, 全 BiSheng）+ 部署（install_opp.sh）..."
    bash "${WORK_DIR}/build_opp.sh" --clean
    bash "${WORK_DIR}/install_opp.sh" --quiet
    echo ">>> [opp] OPP 出包与部署成功"
}

TARGET="${1:-all}"

case "${TARGET}" in
    host)
        run_host_tests
        ;;
    kernel)
        compile_kernel_bisheng
        ;;
    python)
        run_python_tests
        ;;
    opp)
        run_opp_tests
        ;;
    all)
        run_host_tests
        compile_kernel_bisheng
        run_python_tests
        run_opp_tests
        echo "================================================================="
        echo "SageAttention2 全部门禁通过（host / kernel / python / opp, BiSheng 容器）"
        echo "================================================================="
        ;;
    *)
        echo "Unknown target: ${TARGET}"
        echo "Usage: $0 [all|host|kernel|python|opp]"
        exit 1
        ;;
esac
