#!/bin/bash
# 链式组合验证：dispatch_teardown 在 NPU 上真实产出的 expand_idx → 直接喂给 combine_teardown，
#   证明两算子按设计契约组合（dispatch 产 expand_idx，combine 消费它定位 token）。
# 用法: bash run_chain.sh <bs> <topk> <h> <E> <R> [seed]
set -e
BS=${1:?bs}; TOPK=${2:?topk}; H=${3:?h}; E=${4:?E}; R=${5:?R}; SEED=${6:-1234}
PY=${PY:-/root/miniconda3/envs/mc2_py310/bin/python}
export ASCEND_DATE=${ASCEND_DATE:-0625}
CANN=/home/${ASCEND_DATE}/env/ascend-toolkit/latest
source /home/${ASCEND_DATE}/env/set_env.sh 2>/dev/null || true
export ASCEND_CUSTOM_OPP_PATH=${ASCEND_OPP_PATH}/vendors/custom_transformer
export LD_LIBRARY_PATH=${CANN}/lib64:${CANN}/x86_64-linux/lib64:${ASCEND_CUSTOM_OPP_PATH}/op_api/lib:${LD_LIBRARY_PATH}
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
D1="$HERE/data_chain_disp_${BS}_${TOPK}_${H}_${E}_${R}"
D2="$HERE/data_chain_comb_${BS}_${TOPK}_${H}_${E}_${R}"
mkdir -p "$D1" "$D2"

echo "=================== CHAIN bs=$BS topk=$TOPK h=$H E=$E R=$R seed=$SEED ==================="
echo "--- clear NPU ---"
pkill -9 test_dispatch_teardown 2>/dev/null || true; pkill -9 test_combine_teardown 2>/dev/null || true
pkill -9 python3 2>/dev/null || true; sleep 2; npu-smi info | grep -E "No running" | head -4 || true

echo "--- 1) dispatch: golden 造数 + NPU 跑 → 产 expand_idx.npu.bin ---"
$PY golden_dispatch.py $BS $TOPK $H $E $R "$D1" $SEED >/dev/null
timeout 120 "$HERE/build/test_dispatch_teardown" $BS $TOPK $H $E $R "$D1" | grep -E "ok|ERR"

echo "--- 2) combine: golden 造数（同 seed → 同 expert_ids/expand_idx）---"
$PY golden_combine.py $BS $TOPK $H $E $R "$D2" $SEED >/dev/null

echo "--- 3) 一致性 + 喂值：dispatch 的 NPU expand_idx → combine 输入 ---"
if cmp -s "$D1/expert_ids.bin" "$D2/expert_ids.bin"; then echo "    expert_ids 一致(同 seed) OK"; else echo "    [WARN] expert_ids 不一致"; fi
if cmp -s "$D1/expand_idx.npu.bin" "$D2/expand_idx.bin"; then echo "    dispatch(NPU)产出 expand_idx == combine golden expand_idx，bit 一致 OK"; else echo "    [WARN] expand_idx 与 golden 不一致"; fi
cp "$D1/expand_idx.npu.bin" "$D2/expand_idx.bin"   # 显式用 dispatch 的真实产出喂 combine

echo "--- 4) combine: 用 dispatch 真实产出的 expand_idx 跑 NPU ---"
timeout 120 "$HERE/build/test_combine_teardown" $BS $TOPK $H $E $R "$D2" | grep -E "ok|ERR"

echo "--- 5) 比对 combine 输出 x ---"
$PY compare_combine.py $BS $TOPK $H $E $R "$D2"
