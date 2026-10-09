#!/bin/bash
# 环境勘察（在容器内执行）
WORK=/home/wchao00893980/deepseek1009

echo "=== 1. 新建/确认工作目录 ==="
mkdir -p "$WORK"
ls -ld "$WORK"

echo
echo "=== 2. NPU ==="
npu-smi info 2>&1 | head -14

echo
echo "=== 3. 已有的 scatter 相关目录（上次会话留下的） ==="
ls -la /home/wchao00893980/scatter1009 2>/dev/null | head -15
echo "--- scatter_probe_20261009 ---"
ls -la /home/wchao00893980/scatter_probe_20261009 2>/dev/null | head -15
echo "--- scatter_a2_repro_20260912 ---"
ls -la /home/wchao00893980/scatter_a2_repro_20260912 2>/dev/null | head -15

echo
echo "=== 4. conda 环境里的 torch ==="
source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate wangchao 2>/dev/null
echo "python: $(which python)"
python -c "import torch, torch_npu; print('torch', torch.__version__, '/ torch_npu', torch_npu.__version__)" 2>&1 | tail -3

echo
echo "=== 5. CANN ==="
ls /usr/local/Ascend/ascend-toolkit/latest 2>/dev/null | head
head -3 /usr/local/Ascend/ascend-toolkit/latest/version.cfg 2>/dev/null

echo
echo "=== 6. 可复用的 ops-transformer checkout ==="
for d in /home/wchao00893980/scatter_a2_repro_20260912/ops-transformer \
         /home/wchao00893980/gbsa_a2a3_1009/ops-transformer \
         /home/wchao00893980/gbsa0927/ops-transformer; do
  if [ -d "$d/.git" ]; then
    echo "--- $d"
    git -C "$d" log --oneline -2 2>&1 | head -2
    git -C "$d" remote -v 2>&1 | head -2
    echo "    dirty files: $(git -C "$d" status --short 2>/dev/null | wc -l)"
  else
    echo "--- $d : 不是 git 仓库或不存在"
  fi
done

echo
echo "=== 7. 磁盘 ==="
df -h / | tail -1
du -sh /home/wchao00893980/*/ops-transformer 2>/dev/null | head -10
