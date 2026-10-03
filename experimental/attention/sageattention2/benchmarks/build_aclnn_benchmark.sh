#!/usr/bin/env bash
# build_aclnn_benchmark.sh: 独立 ACLNN 基准工具构建（BiSheng 容器内执行）
#
# 编译统一使用 BiSheng（宿主模式）并链接容器内 CANN 真机库。
# 容器内直接基于原生 CANN 头文件与库文件构建。

set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
if [ -z "${ASCEND_HOME_PATH:-}" ]; then
    # shellcheck source=/dev/null
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
cann="${ASCEND_HOME_PATH}"

if [ ! -f "${cann}/include/acl/acl.h" ]; then
    echo "ERROR: CANN environment not found at ${cann}." >&2
    echo "       本脚本须在算子门禁容器内执行（bash run.sh shell 或 docker/sage run）。" >&2
    exit 1
fi

BISHENG="${cann}/bin/bisheng"
[ -x "${BISHENG}" ] || BISHENG="bisheng"

custom="$root/build/opp/vendors/customize/op_api/lib"
out_dir="$root/build/benchmarks"
mkdir -p "$out_dir"

echo "================================================================="
echo "Building SageAttention2 Standalone ACLNN Benchmark Tool"
echo "Root directory:     ${root}"
echo "Compiler:           ${BISHENG} (host mode)"
echo "CANN:               ${cann}"
echo "Target binary:      ${out_dir}/benchmark_aclnn"
echo "================================================================="

EXTRA_LIBS="-lascendcl -lopapi -lnnopbase -ldl -lm"
if [ -f "${custom}/libcust_opapi.so" ]; then
    EXTRA_LIBS="-lcust_opapi ${EXTRA_LIBS}"
fi

CATLASS_INC=""
if [ -d "${root}/docker/third_party/catlass/include" ]; then
    CATLASS_INC="${root}/docker/third_party/catlass/include"
elif [ -d "${root}/third_party/catlass/include" ]; then
    CATLASS_INC="${root}/third_party/catlass/include"
fi

"${BISHENG}" -O2 -std=c++17 -Wall -Wextra -Werror -lstdc++ \
    "${root}/benchmarks/benchmark_aclnn.cpp" \
    -I"${cann}/include" \
    -I"${root}/aclnn" \
    -I"${root}/op_api" \
    -I"${root}/quant" \
    -I"${CATLASS_INC}" \
    -L"${custom}" -L"${cann}/lib64" \
    -Wl,-rpath,"${custom}" -Wl,-rpath,"${cann}/lib64" \
    ${EXTRA_LIBS} \
    -o "${out_dir}/benchmark_aclnn"

echo "Build successful! Generated binary: ${out_dir}/benchmark_aclnn"

# 冒烟自检：若当前环境透传了物理 NPU 则执行微量测试；若无物理 NPU 则验证二进制构建成功即可。
set +e
smoke_out=$("${out_dir}/benchmark_aclnn" --smoke-test --output "${out_dir}/smoke_test.json" 2>&1)
smoke_rc=$?
set -e
if [ "${smoke_rc}" -eq 0 ]; then
    echo "Smoke test passed successfully on real NPU! Generated: ${out_dir}/smoke_test.json"
elif echo "${smoke_out}" | grep -q "500000"; then
    echo "Notice: No physical NPU device detected in container (expected on CPU/host). Benchmark binary build verified."
else
    echo "${smoke_out}"
    echo "Smoke test failed with unexpected exit code ${smoke_rc}"
    exit "${smoke_rc}"
fi
echo "================================================================="
