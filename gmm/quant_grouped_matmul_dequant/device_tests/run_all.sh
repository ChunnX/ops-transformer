#!/bin/bash
# ============================================================
# Run every shape family (LIGHT + HEAVY) on device sequentially.
# One top-level summary + per-shape SUMMARY/STATS under <out_dir>/<shape>/.
#
# Usage:
#   bash device_tests/run_all.sh <out_dir>
#   SHAPES="decode_like k_heavy"   bash device_tests/run_all.sh <out_dir>
#   TIER=LIGHT                     bash device_tests/run_all.sh <out_dir>
#   TIER=HEAVY                     bash device_tests/run_all.sh <out_dir>
#
# CANN + torch_npu + qgmmd op package must already be installed in the
# shell. If not, source them first:
#   eval "$(pyenv init -)" && eval "$(pyenv virtualenv-init -)" \
#     && pyenv activate .venv3.11
#   source ~/.local/ascend-toolkit/set_env.sh
#   source ~/.local/ascend-toolkit/vendors/custom_transformer/bin/set_env.bash
# ============================================================
set -e

if [ -z "$1" ]; then
    echo "Usage: bash $0 <out_dir>" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# gmm/quant_grouped_matmul_dequant/device_tests/ → 3 levels up = repo root.
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

OUT_DIR="$(realpath "$1")"
TIER="${TIER:-ALL}"
RUN_DEV="gmm/quant_grouped_matmul_dequant/device_tests/run_device.py"

# Default shape list = every SHAPES name listed in configs.py for the
# requested TIER (preserve order). With TIER=ALL, includes LIGHT + HEAVY +
# QWEN3 in that order.
if [ -z "${SHAPES:-}" ]; then
    SHAPES="$(python3 -c "
import sys
sys.path.insert(0, '${SCRIPT_DIR}')
from configs import SHAPES_LIGHT, SHAPES_HEAVY, SHAPES_QWEN3
tier = '${TIER}'
if tier == 'ALL':
    pool = SHAPES_LIGHT + SHAPES_HEAVY + SHAPES_QWEN3
elif tier == 'LIGHT':
    pool = SHAPES_LIGHT
elif tier == 'HEAVY':
    pool = SHAPES_HEAVY
elif tier == 'QWEN3':
    pool = SHAPES_QWEN3
else:
    pool = []
print(' '.join(n for n, *_ in pool))
")"
fi

mkdir -p "$OUT_DIR"
OVERALL="$OUT_DIR/OVERALL.txt"
: > "$OVERALL"

echo "qgmmd device batch → $OUT_DIR  TIER=$TIER" | tee -a "$OVERALL"
echo "Shapes: $SHAPES" | tee -a "$OVERALL"
echo "" | tee -a "$OVERALL"

fail_any=0
for shape in $SHAPES; do
    echo "=== $shape ===" | tee -a "$OVERALL"
    t0=$(date +%s)
    parent_out="$OUT_DIR/$shape"
    run_log="$(mktemp)"
    if python3 "$RUN_DEV" --shape "$shape" --tier "$TIER" --out "$parent_out" \
            2>&1 | tee -a "$run_log"; then
        dt=$(( $(date +%s) - t0 ))
        run_dir="$(grep -oE 'Output dir: .*' "$run_log" | head -1 | sed 's/^Output dir: //')"
        if [ -n "$run_dir" ] && [ -f "$run_dir/SUMMARY.txt" ]; then
            tail -1 "$run_dir/SUMMARY.txt" | tee -a "$OVERALL"
        fi
        echo "  (group wall: ${dt}s)" | tee -a "$OVERALL"
    else
        dt=$(( $(date +%s) - t0 ))
        echo "  FAILURES in $shape (group wall: ${dt}s)" | tee -a "$OVERALL"
        fail_any=1
    fi
    rm -f "$run_log"
    echo "" | tee -a "$OVERALL"
done

echo "Batch done at $(date)" | tee -a "$OVERALL"
echo "Top-level summary: $OVERALL"
exit $fail_any
