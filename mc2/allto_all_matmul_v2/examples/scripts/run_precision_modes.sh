#!/bin/bash
# run precision_mode 0/1/2 for a single bf16 case and report precision metrics
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
PY=/root/miniconda3/envs/a2a_mm_v2/bin/python

for pm in 0 1 2; do
  echo "############################################################"
  echo "CASE: x1=e4m3fn x2=e4m3fn y=bf16 precision_mode=$pm"
  echo "############################################################"
  log=$(mktemp)
  if timeout 600 "$PY" test_golden_alltoallmm.py --x1dtype e4m3fn --x2dtype e4m3fn --ydtype bf16 --bias --precision-mode "$pm" >"$log" 2>&1; then
    grep -E "rel_err: max|abs_err: max|\[Rank [01]\]|所有 rank|存在失败|\[PASS\]|\[FAIL\]" "$log" | tail -20
  else
    echo "[ERROR] script failed/timeout"
    grep -E "\[ERROR\]|Traceback|RuntimeError|507015|aicore" "$log" | tail -15
  fi
  rm -f "$log"
  sleep 2
done
