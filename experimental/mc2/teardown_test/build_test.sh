#!/bin/bash
# cmake 构建 teardown_test 单卡 aclnn 驱动 exe。用法: bash build_test.sh
set -e
export ASCEND_DATE=${ASCEND_DATE:-0625}
CANN=/home/${ASCEND_DATE}/env/ascend-toolkit/latest
source /home/${ASCEND_DATE}/env/set_env.sh 2>/dev/null || true
export ASCEND_HOME_PATH=$CANN
export ASCEND_CUSTOM_OPP_PATH=${ASCEND_OPP_PATH}/vendors/custom_transformer
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
rm -rf build
cmake -B build -DASCEND_CANN_PACKAGE_PATH=$CANN
cmake --build build -j8
echo "=== built ==="
ls -la build/test_*teardown 2>/dev/null || true
