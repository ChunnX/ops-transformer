#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PYTHON=${PYTHON:-python3}
PYTHON_SITE=$($PYTHON -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
export PYTHONPATH="$PYTHON_SITE${PYTHONPATH:+:$PYTHONPATH}"

COMMAND=${1:-help}
shift || true
EXCEL=""; SHEET="Sheet1"; PT_PATH="$SCRIPT_DIR/pt_path"
RESULT="$SCRIPT_DIR/result.xlsx"; CASES=""; INDEXES=""; SAVE_PT=""
while [ $# -gt 0 ]; do
  case "$1" in
    -E|--excel) EXCEL="$2"; shift 2 ;;
    -S|--sheet) SHEET="$2"; shift 2 ;;
    -P|--pt-path) PT_PATH="$2"; shift 2 ;;
    -O|--output) RESULT="$2"; shift 2 ;;
    -C|--cases) CASES="$2"; shift 2 ;;
    -I|--indexes) INDEXES="$2"; shift 2 ;;
    --save-pt) SAVE_PT="$2"; shift 2 ;;
    -M|--run-mode) [ "$2" = eager ] || { echo "DSL QLI仅支持eager"; exit 2; }; shift 2 ;;
    -h|--help) COMMAND=help; shift ;;
    *) echo "未知选项: $1"; exit 2 ;;
  esac
done

case "$COMMAND" in
  single)
    QLIV2_CASE_NAMES="$CASES" QLIV2_SINGLE_SAVE_PT_DIR="$SAVE_PT" \
    QLIV2_SINGLE_RESULT_PATH="$RESULT" "$PYTHON" -m pytest -rA -s \
      "$SCRIPT_DIR/test_quant_lightning_indexer_v2_single.py" -v -m ci ;;
  batch)
    if [ -n "$EXCEL" ]; then
      "$PYTHON" "$SCRIPT_DIR/batch/quant_lightning_indexer_v2_pt_save.py" \
        "$EXCEL" "$PT_PATH" --sheet "$SHEET"
    fi
    QLIV2_TESTCASE_DIR="$PT_PATH" QLIV2_CASE_NAMES="$CASES" \
    QLIV2_CASE_INDEXES="$INDEXES" QLIV2_RESULT_PATH="$RESULT" \
      "$PYTHON" -m pytest -rA -s "$SCRIPT_DIR/test_quant_lightning_indexer_v2_batch.py" -v -m ci ;;
  batch_exec)
    [ -n "$EXCEL" ] || { echo "batch_exec必须指定--excel"; exit 2; }
    FILES=$("$PYTHON" "$SCRIPT_DIR/batch/list_pt_from_excel.py" "$EXCEL" "$PT_PATH" --sheet "$SHEET")
    QLIV2_TESTCASE_DIR="$PT_PATH" QLIV2_PT_FILE_LIST="$FILES" QLIV2_RESULT_PATH="$RESULT" \
      "$PYTHON" -m pytest -rA -s "$SCRIPT_DIR/test_quant_lightning_indexer_v2_batch.py" -v -m ci ;;
  help)
    echo "用法: $0 single|batch|batch_exec [-E excel] [-S sheet] [-P pt_dir] [-O result.xlsx] [-C names] [-I indexes] [--save-pt dir]" ;;
  *) echo "未知命令: $COMMAND"; exit 2 ;;
esac
