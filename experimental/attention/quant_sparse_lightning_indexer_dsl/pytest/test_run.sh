#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python3}
PYTHON_SITE=$($PYTHON -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')

MODE=${1:-single}
shift || true
EXCEL=""
PT_PATH="./pt_path"
SHEET="TestCases"
CASES=""
INDEXES=""
DEVICE_ID=0
while getopts "E:P:S:C:I:D:" opt; do
  case ${opt} in
    E) EXCEL=${OPTARG} ;;
    P) PT_PATH=${OPTARG} ;;
    S) SHEET=${OPTARG} ;;
    C) CASES=${OPTARG} ;;
    I) INDEXES=${OPTARG} ;;
    D) DEVICE_ID=${OPTARG} ;;
  esac
done
export QSLI_DEVICE_ID=${DEVICE_ID}
export QSLI_CASE_NAMES=${CASES}
export QSLI_CASE_INDEXES=${INDEXES}
ROOT=$(cd ../.. && pwd)
export PYTHONPATH=${PYTHON_SITE}:${ROOT}${PYTHONPATH:+:${PYTHONPATH}}

case ${MODE} in
  single)
    "$PYTHON" -m pytest -rA -s test_quant_sparse_lightning_indexer_single.py -v -m ci
    ;;
  batch)
    if [[ -n ${EXCEL} ]]; then
      "$PYTHON" batch/quant_sparse_lightning_indexer_pt_save.py "${EXCEL}" "${PT_PATH}" --sheet "${SHEET}"
    fi
    QSLI_TESTCASE_DIR=${PT_PATH} "$PYTHON" -m pytest -rA -s test_quant_sparse_lightning_indexer_batch.py -v -m ci
    ;;
  batch_exec)
    [[ -n ${EXCEL} ]] || { echo "batch_exec requires -E" >&2; exit 2; }
    QSLI_PT_FILE_LIST=$("$PYTHON" batch/list_pt_from_excel.py "${EXCEL}" "${PT_PATH}" --sheet "${SHEET}")
    export QSLI_PT_FILE_LIST
    QSLI_TESTCASE_DIR=${PT_PATH} "$PYTHON" -m pytest -rA -s test_quant_sparse_lightning_indexer_batch.py -v -m ci
    ;;
  *) echo "usage: bash test_run.sh {single|batch|batch_exec} [-E xlsx] [-P pt_dir] [-S sheet] [-C names] [-I indexes] [-D device]" >&2; exit 2 ;;
esac
