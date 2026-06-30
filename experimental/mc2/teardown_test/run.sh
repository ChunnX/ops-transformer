#!/bin/bash
# teardown_test 单 case 编排：env → golden 造数 → 清场 → aclnn 单卡跑 → 比对
# 用法: bash run.sh <op:dispatch|combine> <bs> <topk> <h> <E> <R> [seed]
# 前置：算子包已 build.sh --pkg 装好；test exe 已 cmake 编好（见 build_test.sh）。
set -e
OP=${1:?op: dispatch|combine}
BS=${2:?bs}; TOPK=${3:?topk}; H=${4:?h}; E=${5:?E}; R=${6:?R}; SEED=${7:-1234}

# ---- 环境 ----
PY=${PY:-/root/miniconda3/envs/mc2_py310/bin/python}
export ASCEND_DATE=${ASCEND_DATE:-0625}
CANN=/home/${ASCEND_DATE}/env/ascend-toolkit/latest
source /home/${ASCEND_DATE}/env/set_env.sh 2>/dev/null || source ${CANN}/../set_env.sh 2>/dev/null || true
export ASCEND_CUSTOM_OPP_PATH=${ASCEND_OPP_PATH}/vendors/custom_transformer
export LD_LIBRARY_PATH=${CANN}/lib64:${CANN}/x86_64-linux/lib64:${ASCEND_CUSTOM_OPP_PATH}/op_api/lib:${LD_LIBRARY_PATH}

HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE"
DATADIR="$HERE/data_${OP}_${BS}_${TOPK}_${H}_${E}_${R}"
mkdir -p "$DATADIR"

echo "=================== [$OP] bs=$BS topk=$TOPK h=$H E=$E R=$R seed=$SEED ==================="

# ---- golden 造数 ----
echo "--- golden ---"
$PY golden_${OP}.py $BS $TOPK $H $E $R "$DATADIR" $SEED

# ---- 清场（约束：跑前 npu-smi，有进程全杀）----
echo "--- clear NPU ---"
npu-smi info | grep -E "Process id|python|test_" || true
pkill -9 python3 2>/dev/null || true
pkill -9 python 2>/dev/null || true
pkill -9 test_dispatch_teardown 2>/dev/null || true
pkill -9 test_combine_teardown 2>/dev/null || true
sleep 3
npu-smi info | grep -E "No running|Process id" | head -8 || true

# ---- aclnn 单卡跑 ----
echo "--- run npu (aclnn) ---"
EXE="$HERE/build/test_${OP}_teardown"
if [ ! -x "$EXE" ]; then echo "[ERR] $EXE 不存在，先 bash build_test.sh"; exit 2; fi
timeout 120 "$EXE" $BS $TOPK $H $E $R "$DATADIR"

# ---- 比对 ----
echo "--- compare ---"
$PY compare_${OP}.py $BS $TOPK $H $E $R "$DATADIR"
