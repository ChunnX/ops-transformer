#!/usr/bin/env bash
# build_aclnn_benchmark.sh: 独立 ACLNN 基准工具构建（BiSheng 容器内执行）
#
# 编译统一使用 BiSheng（宿主模式）并链接容器内 CANN 真机库；不再提供无 CANN 环境的
# 宿主 mock 编译分支。benchmark_aclnn.cpp 的 SIMULATION 语义保留为纯运行时模式：
# 容器无 NPU 设备透传时工具仍可运行（合成延迟、NaN speedup、verdict=SIMULATION、
# 退出码 2），该报告不作为验收证据。

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

EXTRA_LIBS="-lascendcl -lopapi -lnnopbase -ldl"
if [ -f "${custom}/libcust_opapi.so" ]; then
    EXTRA_LIBS="-lcust_opapi ${EXTRA_LIBS}"
fi

"${BISHENG}" -O2 -std=c++17 -Wall -Wextra -Werror \
    "${root}/benchmarks/benchmark_aclnn.cpp" \
    -I"${cann}/include" \
    -I"${root}/aclnn" \
    -I"${root}/op_api" \
    -L"${custom}" -L"${cann}/lib64" \
    -Wl,-rpath,"${custom}" -Wl,-rpath,"${cann}/lib64" \
    ${EXTRA_LIBS} \
    -o "${out_dir}/benchmark_aclnn"

echo "Build successful! Generated binary: ${out_dir}/benchmark_aclnn"

# 冒烟自检：容器无 NPU 透传时工具走 SIMULATION 运行时模式（退出码 2，见文件头注释）；
# 0 = 真机运行完成，0/2 之外均为失败。
set +e
"${out_dir}/benchmark_aclnn" --smoke-test --output "${out_dir}/smoke_test.json"
smoke_rc=$?
set -e
if [ "${smoke_rc}" -ne 0 ] && [ "${smoke_rc}" -ne 2 ]; then
    echo "Smoke test failed with exit code ${smoke_rc}"
    exit "${smoke_rc}"
fi
if [ "${smoke_rc}" -eq 2 ]; then
    echo "Smoke test ran in SIMULATION mode (exit 2): report carries NaN speedup, no verdict."
fi
echo "Smoke test passed successfully! Generated: ${out_dir}/smoke_test.json"
echo "================================================================="
