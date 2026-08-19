#!/bin/bash
# PoolKeyIndexer pytest 运行脚本
# 用法: bash run.sh [pytest_args...]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================"
echo " PoolKeyIndexer pytest 测试"
echo "============================================"

# 确保 cann_ops_transformer 可 import
export PYTHONPATH="${PYTHONPATH:-}:/usr/local/lib/python3.10/site-packages"

python -m pytest "$@" -v -s --tb=short 2>&1 | tee pytest_output.log

echo "============================================"
echo " 测试完成，日志: pytest_output.log"
echo "============================================"
