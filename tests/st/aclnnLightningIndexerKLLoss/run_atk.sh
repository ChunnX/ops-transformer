#!/bin/bash
# ATK ST 运行脚本 — lightning_indexer_kl_loss
# NPU torch 接口精度验证：CPU 低精度（api_type） vs NPU torch 接口（aclnn_api_type）
# 用法: bash run_atk.sh [seed]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SEED="${1:-1234}"

# 环境前置（若已在 shell 中 source 过可注释掉）
# source /data/miniconda3/etc/profile.d/conda.sh && conda activate jzh
# source /data/jzh/toolkit/cann-9.0.0/cann/set_env.sh
export PYTHONPATH="/data/jzh/ops-transformer/torch_extension:${PYTHONPATH:-}"
export ASCEND_GLOBAL_LOG_LEVEL=0
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export TMP=/tmp
export TMPDIR=/tmp

echo "=== ATK ST: aclnnLightningIndexerKLLoss ==="
echo "  workdir : $SCRIPT_DIR"
echo "  seed    : $SEED"
echo ""

# 清理旧缓存（避免 sqlite3 UNIQUE 冲突）
rm -rf __pycache__ atk_output atk.db celerybeat-schedule db.sqlite3 atk.log

# 1. 生成用例
echo "--- [1/2] 生成用例 ---"
atk case -f aclnn_lightning_indexer_kl_loss.yaml -p ./generate_lightning_indexer_kl_loss.py -s "$SEED"

CASE_FILE="result/aclnn_lightning_indexer_kl_loss/json/all_aclnn_lightning_indexer_kl_loss.json"
if [ ! -f "$CASE_FILE" ]; then
  echo "ERROR: 用例文件未生成: $CASE_FILE"
  exit 1
fi

# 2. accuracy：CPU 低精度（api_type） vs NPU torch 接口（aclnn_api_type）
echo ""
echo "--- [2/2] accuracy (CPU vs NPU torch) ---"
atk node --backend npu --devices 0 -o "$(pwd)" \
  node --backend cpu \
  task -c "$CASE_FILE" --task accuracy -p "$(pwd)" -s 0 -e -10

echo ""
echo "=== ATK ST done ==="


atk node --backend npu --devices 1 -o "$(pwd)" \
  node --backend cpu \
  task -c result/aclnn_lightning_indexer_kl_loss/json/all_aclnn_lightning_indexer_kl_loss.json \
  --task accuracy -p "$(pwd)" -s 0 -e -1 --save_data output

atk node --backend npu --devices 1 -o "$(pwd)" \
node --backend cpu \
task -c result/aclnn_lightning_indexer_kl_loss/json/all_aclnn_lightning_indexer_kl_loss.json \
--task performance_device -p "$(pwd)" -s 0 -e 10

mssanitizer --log-file "$(pwd)/mss.log" -- \
atk node --backend npu --devices 1 -o "$(pwd)" \
node --backend cpu \
task -c result/aclnn_lightning_indexer_kl_loss/json/all_aclnn_lightning_indexer_kl_loss.json \
--task accuracy -p "$(pwd)" -s 0 -e -1 -ms -msl "$(pwd)/mss.log"