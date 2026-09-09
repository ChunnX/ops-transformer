#!/usr/bin/env bash
source /usr/local/Ascend/cann-9.2.0/set_env.sh
set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PYTHON=/home/x00949573/.conda/envs/xwr/bin/python
TESTCASE_DIR=${1:-"$SCRIPT_DIR/pt_path"}
RESULT_XLSX=${2:-"$SCRIPT_DIR/result.xlsx"}

mapfile -t CASE_FILES < <(find "$TESTCASE_DIR" -maxdepth 1 -name '*.pt' | sort)
[ ${#CASE_FILES[@]} -gt 0 ] || { echo "未找到PT用例: $TESTCASE_DIR"; exit 1; }
pass=0; fail=0
for case_file in "${CASE_FILES[@]}"; do
  echo "执行: $(basename "$case_file")"
  if QLIV2_TESTCASE_DIR="$TESTCASE_DIR" QLIV2_TESTCASE_PATH="$case_file" \
     QLIV2_RESULT_PATH="$RESULT_XLSX" "$PYTHON" -m pytest -q \
       "$SCRIPT_DIR/test_quant_lightning_indexer_v2_batch.py"; then
    pass=$((pass+1))
  else
    fail=$((fail+1))
  fi
done
echo "总计=${#CASE_FILES[@]} 通过=$pass 失败=$fail"
[ "$fail" -eq 0 ]
