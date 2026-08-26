#!/bin/bash
# 空核 PTO 的 Pipe 流水图（MindStudio Insight → 时间线）。
# 用法：bash run_fag_msprof_op.sh
#
# 流水图只要 --aic-metrics=PipeTimeline。
# Default 的 visualize_data.bin 只能看详情/Roofline，不是流水图。
source /root/miniconda3/etc/profile.d/conda.sh
conda activate kjc
source /home/kjc/kjc_cann_0819/cann-9.2.0/set_env.sh
source /home/kjc/kjc_tmp/fag_hdr_ablation/hdr_s1_entry_empty/vendors/custom_transformer/bin/set_env.bash

export PYTHONPATH=/home/kjc/ops-transformer/torch_extension:${ASCEND_HOME_PATH}/python/site-packages
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64:${LD_LIBRARY_PATH}
export TMPDIR=/home/kjc/kjc_tmp

cd "$(dirname "$0")"
OUT=/home/kjc/kjc_tmp/fag_msprof_op/pipe
mkdir -p "$OUT"

msprof op --output="$OUT" --aic-metrics=PipeTimeline --kernel-name=FlashAttnGrad --warm-up=10 \
  python -u ./test_flash_attn_grad-binary-msprof.py \
  --mode perf --side pypto --filter bn2_n36 --xlsx ./fag_test_case_kernel.xlsx \
  --warmup 0 --iters 1 --no-smoke

echo "流水图请打开这个（chrome://tracing 或 Insight 时间线）："
ls -l "$OUT"/OPPROF_*/trace.json
echo "visualize_data.bin 在 PTO 上经常是空的，不要用它："
ls -l "$OUT"/OPPROF_*/visualize_data.bin
