#!/usr/bin/env bash
# ==============================================================================
# profile_single_op.sh: 昇腾 Atlas 950PR 单算子硬件性能画像收集脚本 (msprof)
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="${ROOT_DIR}/prof_output"
CASE_NAME="${1:-P-01}"

echo "================================================================================"
echo "   SageAttention2 Single Operator Profiling Pipeline (msprof)"
echo "   Target Case  : ${CASE_NAME}"
echo "   Output Directory: ${OUTPUT_DIR}"
echo "================================================================================"

mkdir -p "${OUTPUT_DIR}"

if command -v msprof &>/dev/null; then
    echo "[INFO] Found msprof in PATH, executing hardware profiling..."
    msprof --output="${OUTPUT_DIR}" \
           --application="python3 ${SCRIPT_DIR}/benchmark.py --device npu --cases ${CASE_NAME} --metric-policy test_perf" \
           --aic-mode=PipeUtilization \
           --l2-cache=on \
           --aicore-metrics=MemoryL0 \
           --ai-core=on
    echo "[SUCCESS] Profiling data collected at: ${OUTPUT_DIR}"
else
    echo "[WARN] msprof command not found in current environment (host development mode)."
    echo "[INFO] Running benchmark with Python profiling simulation instead..."
    python3 "${SCRIPT_DIR}/benchmark.py" --device cpu --cases "${CASE_NAME}" --quick
fi
