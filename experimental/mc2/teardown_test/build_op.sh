#!/bin/bash
# 构建并安装 teardown 算子包（registry-invoke）。用法: bash build_op.sh <op_dir_name>
#   <op_dir_name> = dispatch_teardown | combine_teardown
set -e
OP=${1:?op dir name: dispatch_teardown|combine_teardown}
export ASCEND_DATE=${ASCEND_DATE:-0625}
source /home/${ASCEND_DATE}/env/set_env.sh 2>/dev/null || true
cd /home/developer/ops-transformer
echo "=== build.sh --pkg --experimental --soc=ascend950 --ops=$OP ==="
bash build.sh --pkg --experimental --soc=ascend950 --ops=$OP -j16
echo "=== install .run ==="
RUN=$(ls -t build_out/cann-ops-transformer-custom_linux-*.run 2>/dev/null | head -1)
echo "run pkg: $RUN"
./$RUN --force
echo "=== installed; vendors/custom_transformer ==="
ls -la ${ASCEND_OPP_PATH}/vendors/ 2>/dev/null || true
