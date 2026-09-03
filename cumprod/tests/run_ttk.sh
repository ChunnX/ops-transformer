#!/usr/bin/env bash
# Cumprod TTK 验证入口：GE 通路（geir）+ aclnn 通路。
# 用法: bash run_ttk.sh [geir|aclnn|all]
set -e
MODE=${1:-all}
HERE=$(cd "$(dirname "$0")" && pwd)
OP_PKG=${OP_PKG:-/tmp/opencode/cumprod_pkg/vendors/custom_transformer}
TTK_HOME=${TTK_HOME:-/workspace/l00634032/ops-test-kit}
CANN=${CANN:-/workspace/0805-imgrawdecodeposthandle/Ascend/cann-9.2.0}

export ASCEND_HOME_PATH=$CANN
export ASCEND_OPP_PATH=$CANN/opp
export ASCEND_CUSTOM_OPP_PATH=$OP_PKG
export LD_LIBRARY_PATH=$CANN/lib64:$OP_PKG/op_api/lib:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:/opt/buildtools/python-3.12.9/lib
export PYTHONPATH=/opt/buildtools/python-3.12.9/lib/python3.12/site-packages:$CANN/python/site-packages
export PATH=/opt/buildtools/python-3.12.9/bin:$CANN/bin:$PATH

cd "$TTK_HOME"
PLUGIN="$HERE/ttk/ttk_golden_plugin.py"

if [[ "$MODE" == "geir" || "$MODE" == "all" ]]; then
  echo "==== TTK geir (GE 通路) ===="
  GEIR_INPUT=${GEIR_INPUT:-$HERE/ttk/geir_cumprod.csv}
  GEIR_OUT=${GEIR_OUT:-$HERE/ttk/geir_cumprod_result.csv}
  python3 -m ttk geir -i "$GEIR_INPUT" --dev 0 --task-prof=false -b \
      --plugin "$PLUGIN" -o "$GEIR_OUT" ${GEIR_EXTRA:-}
fi
if [[ "$MODE" == "aclnn" || "$MODE" == "all" ]]; then
  echo "==== TTK aclnn (aclnn 通路) ===="
  python3 -m ttk aclnn -i "$HERE/ttk/aclnn_cumprod.csv" --dev 0 \
      --plugin "$PLUGIN" -o "$HERE/ttk/aclnn_cumprod_result.csv"
fi
