#!/bin/bash
# 8 dtype combos: x1/x2 in {e4m3fn, e5m2} x y in {bf16, fp16}, all with bias
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
PY=/root/miniconda3/envs/a2a_mm_v2/bin/python

combos=(
  "e4m3fn e4m3fn bf16"
  "e4m3fn e4m3fn fp16"
  "e4m3fn e5m2   bf16"
  "e4m3fn e5m2   fp16"
  "e5m2   e4m3fn bf16"
  "e5m2   e4m3fn fp16"
  "e5m2   e5m2   bf16"
  "e5m2   e5m2   fp16"
)

PASS=0; FAIL=0
declare -a RESULTS
for c in "${combos[@]}"; do
  read -r x1 x2 y <<< "$c"
  echo "=============================================================="
  echo "CASE: x1=$x1 x2=$x2 y=$y (bias)"
  echo "=============================================================="
  log=$(mktemp)
  if timeout 600 "$PY" test_golden_alltoallmm.py --x1dtype "$x1" --x2dtype "$x2" --ydtype "$y" --bias >"$log" 2>&1; then
    if grep -q "所有 rank 精度对比通过" "$log"; then
      echo "[RESULT] PASS"
      PASS=$((PASS+1)); RESULTS+=("PASS  x1=$x1 x2=$x2 y=$y")
    else
      echo "[RESULT] FAIL (no pass marker)"
      FAIL=$((FAIL+1)); RESULTS+=("FAIL  x1=$x1 x2=$x2 y=$y")
    fi
  else
    echo "[RESULT] FAIL (script error/timeout)"
    FAIL=$((FAIL+1)); RESULTS+=("FAIL  x1=$x1 x2=$x2 y=$y")
  fi
  # show tail on failure
  if [ "${RESULTS[-1]}" == FAIL* ] 2>/dev/null || [[ "${RESULTS[-1]}" == FAIL* ]]; then
    grep -E "\[ERROR\]|\[FAIL\]|Traceback|RuntimeError|507015|aicore" "$log" | tail -15
  fi
  rm -f "$log"
  sleep 2
done

echo ""
echo "=============================================================="
echo "SUMMARY: PASS=$PASS FAIL=$FAIL total=${#combos[@]}"
echo "=============================================================="
for r in "${RESULTS[@]}"; do echo "  $r"; done
[ "$FAIL" -eq 0 ]
