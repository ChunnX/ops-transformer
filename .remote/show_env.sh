#!/bin/bash
# 查容器里的环境变量
echo "=== 所有含 cache/dump/prof/L2 的环境变量 ==="
env | grep -iE "cache|dump|profil|l2" | sort

echo
echo "=== 完整 env（前 50 行） ==="
env | sort | head -50

echo
echo "=== torch_npu / CANN 相关 ==="
env | grep -iE "ascend|cann|torch|npu" | sort

echo
echo "=== 是否有 DISABLE_L2_CACHE / ASCEND_*DUMP* / prof 开关 ==="
for v in DISABLE_L2_CACHE ASCEND_GLOBAL_LOG_LEVEL ASCEND_SLOG_PRINT_TO_STDOUT ASCEND_OPP_KERNEL_CACHE TASK_QUEUE_ENABLE; do
  echo "  $v=${!v:-<未设置>}"
done
