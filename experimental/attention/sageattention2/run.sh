#!/usr/bin/env bash
# run.sh — SageAttention2 门禁统一入口。
# 宿主侧只做容器包装（依赖仅 Docker + compose 插件）；容器内全部编译与测试统一由 BiSheng 承载
# （C++ 宿主单测走 BiSheng 宿主模式，设备核走 BiSheng 交叉编译 dav-3510）。
# 首次使用先执行 ./docker/sage init 置备 .env / CATLASS / 基础镜像 / golden。
#
# 用法：bash run.sh [host|kernel|python|opp|all]
#   host   BiSheng 编译并运行 C++ 宿主单测（test_aclnn_host / test_op_host / test_tiling_struct）
#   kernel BiSheng 交叉编译设备核 → build/sageattention2_bisheng.o
#   python 容器内 python 门禁（tiling_struct / quant / kernel_twin / gqa_chunk）
#   opp    OPP vendor 树 BiSheng 出包（build_opp.sh）+ 部署（install_opp.sh）
#   all    host → kernel → python → opp

set -euo pipefail

WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${WORK_DIR}/build"

if [ "${SAGE_IN_CONTAINER:-}" != "1" ]; then
  exec bash "${WORK_DIR}/docker/sage" run bash run.sh "$@"
fi

# 容器侧：ASCEND_HOME_PATH 由镜像 ENV 导出，CATLASS 由 sage init 置备
export PATH="${ASCEND_HOME_PATH}/bin:${PATH}"
BISHENG="${ASCEND_HOME_PATH}/bin/bisheng"
CATLASS_ROOT="${CATLASS_ROOT:-${WORK_DIR}/docker/third_party/catlass}"
export CATLASS_ROOT
CATLASS_INC="${CATLASS_ROOT}/include"

# bisheng 宿主模式链接不自动携带 C++ 运行时，链接型编译需显式 -lstdc++；CANN 库统一使用 CXX11_ABI=0
HOST_CXX_FLAGS=(-std=c++17 -O2 -D_GLIBCXX_USE_CXX11_ABI=0 -Wall -Wextra -Wno-ignored-qualifiers -Wno-reorder -Wno-unused-parameter -lstdc++)
COMMON_INCLUDES=(
  -I"${ASCEND_HOME_PATH}/include"
  -I"${ASCEND_HOME_PATH}/include/aclnn"
  -I"${WORK_DIR}/sageattn_fp8/ascendc"
  -I"${WORK_DIR}/sageattn_fp8/catlass"
  -I"${WORK_DIR}/quant"
  -I"${CATLASS_INC}"
)

run_host_tests() {
  echo ">>> [host] C++ 宿主单测（BiSheng 宿主模式）..."
  mkdir -p "${BUILD_DIR}"

  echo "-- 编译 test_aclnn_host"
  "${BISHENG}" "${HOST_CXX_FLAGS[@]}" \
    "${WORK_DIR}/tests/test_aclnn_host.cpp" \
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
    "${COMMON_INCLUDES[@]}" \
    -L"${ASCEND_HOME_PATH}/lib64" \
    -lgraph -lgraph_base -lmetadef -lopp_registry \
    -Wl,-rpath,"${ASCEND_HOME_PATH}/lib64" \
    -o "${BUILD_DIR}/test_op_host"
  "${BUILD_DIR}/test_op_host"

  echo "-- 编译 test_tiling_struct"
  "${BISHENG}" "${HOST_CXX_FLAGS[@]}" \
    "${WORK_DIR}/tests/test_tiling_struct.cpp" \
    "${COMMON_INCLUDES[@]}" \
    -o "${BUILD_DIR}/test_tiling_struct"
  "${BUILD_DIR}/test_tiling_struct"

  echo ">>> [host] 全部通过"
}

compile_kernel_bisheng() {
  echo ">>> [kernel] BiSheng 交叉编译设备核 → dav-3510..."
  mkdir -p "${BUILD_DIR}"
  # -DCATLASS_ARCH=3510 与 build_opp.sh / tools/sim 同口径
  # （真实消费方：catlass.hpp 3510 档常量 + CATLASS_DEVICE_SIMD_VF 宏，见 build_opp.sh 同注）
  "${BISHENG}" --npu-arch=dav-3510 -std=c++17 -xasc -DCATLASS_ARCH=3510 \
    -DENABLE_CV_COMM_VIA_SSBUF=1 --cce-no-dcache-flush -c \
    "${WORK_DIR}/op_kernel/sageattention2.cpp" \
    -I"${ASCEND_HOME_PATH}/include" \
    "${COMMON_INCLUDES[@]}" \
    -o "${BUILD_DIR}/sageattention2_bisheng.o"
  echo ">>> [kernel] 编译成功: ${BUILD_DIR}/sageattention2_bisheng.o"
}

run_python_tests() {
  echo ">>> [python] Python 门禁..."
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

  echo ">>> [python] 全部通过"
}

run_opp_tests() {
  echo ">>> [opp] OPP 出包（build_opp.sh）+ 部署（install_opp.sh）..."
  bash "${WORK_DIR}/build_opp.sh" --clean
  bash "${WORK_DIR}/install_opp.sh" --quiet
  echo ">>> [opp] 出包与部署成功"
}

case "${1:-all}" in
host)
  run_host_tests
  ;;
kernel)
  compile_kernel_bisheng
  ;;
python)
  run_python_tests
  ;;
test)
  run_host_tests
  run_python_tests
  echo "=== SageAttention2 宿主与 Python 门禁全部通过 ==="
  ;;
opp)
  run_opp_tests
  ;;
all)
  run_host_tests
  compile_kernel_bisheng
  run_python_tests
  run_opp_tests
  echo "=== SageAttention2 全部门禁通过（host / kernel / python / opp, BiSheng 容器）==="
  ;;
*)
  echo "Unknown target: ${1}" >&2
  echo "Usage: $0 [all|test|host|kernel|python|opp]" >&2
  exit 1
  ;;
esac
