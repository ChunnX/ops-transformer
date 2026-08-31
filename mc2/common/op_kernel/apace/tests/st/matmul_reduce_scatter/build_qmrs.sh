#!/bin/bash
# build_qmrs.sh — 编译 apace_qmm_rs_urma_st (迁移后首次编译验证)
set -e
source /home/l00947526/Ascend806/cann-9.2.0/set_env.sh 2>/dev/null || source /usr/local/Ascend/cann-9.2.0/set_env.sh

SCRIPT_DIR="/home/y00845930/dev/ops-transformer/mc2/common/op_kernel/apace/tests/st/matmul_reduce_scatter"
cd "$SCRIPT_DIR"

echo "==== [1/2] cmake configure ===="
rm -rf build
cmake -S .. -B build 2>&1 | tail -15

echo "==== [2/2] build apace_qmm_rs_urma_st ===="
cmake --build build --target apace_qmm_rs_urma_st --parallel 4 2>&1 | tail -30

ls -la build/matmul_reduce_scatter/apace_qmm_rs_urma_st && echo "BUILD OK"
