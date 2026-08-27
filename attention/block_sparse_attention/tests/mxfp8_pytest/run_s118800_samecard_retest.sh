#!/bin/bash
# Same-card retest: FIA MXFP8 (stock CANN) / BSA MXFP8 / BSA MXFP4
# S=118800 B=1 N=1 D=128 keep=1 (FIA sparse=0)
set -eo pipefail

OPS_SRC=/home/icebearPolar/ops-transformer_0703
INSTALL_PATH=/home/icebearPolar/cann_bsa_0703
OPP_VENDOR="${INSTALL_PATH}/vendors/custom_transformer"
MSPROF_BIN=/usr/local/Ascend/ascend-toolkit/latest/bin/msprof
MSPROF_TIMEOUT=7200
DEVICE_HOST=${DEVICE_HOST:-3}
ROOT="${OPS_SRC}/attention/block_sparse_attention/tests/mxfp8_pytest/s118800_samecard_20260827"
SUMMARY="${ROOT}/summary_all.txt"

source /usr/local/Ascend/cann/set_env.sh
source /home/icebearPolar/xxp/bin/activate
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export PYTHONUNBUFFERED=1
export ASCEND_RT_VISIBLE_DEVICES="${DEVICE_HOST}"
export PYTHONPATH="${OPS_SRC}/torch_extension:${PYTHONPATH:-}"

mkdir -p "${ROOT}"
{
  echo "time=$(date -Iseconds) DEVICE_HOST=${DEVICE_HOST} INSTALL_PATH=${INSTALL_PATH}"
  echo "S=118800 B=1 N=1 D=128 keep=1 FIA_sparse=0"
  echo "msprof warm-up=5 launch-count=10 aic-metrics=PipeUtilization,ArithmeticUtilization,Default"
  echo "FIA=stock CANN  BSA=cann_bsa_0703"
  echo "===="
} | tee "${SUMMARY}"

extract_csv() {
  local csv=$1 kernel=$2
  python3 - "$csv" "$kernel" <<'PY'
import csv, sys
p, kernel = sys.argv[1], sys.argv[2]
with open(p, newline="") as f:
    rows = list(csv.DictReader(f))
best = None
for r in rows:
    name = r.get("Op Name") or r.get("OpName") or ""
    if kernel not in name:
        continue
    d = r.get("Task Duration(us)") or ""
    try:
        v = float(d)
    except Exception:
        continue
    if best is None or v > best[0]:
        best = (v, name, r.get("Block Dim") or "",
                r.get("Current Freq") or r.get("Current Freq(MHz)") or "",
                r.get("Rated Freq") or r.get("Rated Freq(MHz)") or "")
if best:
    print(f"CSV Task Duration(us)={best[0]} Op={best[1]} BlockDim={best[2]} Freq={best[3]}/{best[4]}")
else:
    print("CSV no matching op")
PY
}

run_msprof() {
  local tag=$1 kernel=$2 case_py=$3
  local case_dir="${ROOT}/${tag}"
  mkdir -p "${case_dir}"
  echo "=== msprof ${tag} kernel=${kernel} ===" | tee -a "${SUMMARY}"
  set +e
  timeout "${MSPROF_TIMEOUT}" "${MSPROF_BIN}" op --output="${case_dir}" \
    --aic-metrics=PipeUtilization,ArithmeticUtilization,Default \
    --kernel-name="${kernel}" \
    --warm-up=5 --launch-count=10 --kill=off \
    python3 "${case_py}" \
    > "${case_dir}/msprof_stdout.log" 2>&1
  local rc=$?
  set -e
  local dur
  dur=$(awk -v k="${kernel}" '$0 ~ "Op Name: "k{p=1} p&&/Task Duration/{print; exit}' \
    "${case_dir}/msprof_stdout.log" || true)
  local csv_dur=""
  local opprof
  opprof=$(ls -td "${case_dir}"/OPPROF_* 2>/dev/null | head -1 || true)
  if [ -n "${opprof}" ] && [ -f "${opprof}/OpBasicInfo.csv" ]; then
    csv_dur=$(extract_csv "${opprof}/OpBasicInfo.csv" "${kernel}")
  fi
  {
    echo "[rc=${rc}] ${tag}"
    echo "${csv_dur}"
    echo "${dur}"
  } | tee -a "${SUMMARY}"
  if grep -Eiq "CreateSnapshot|207001|Out of memory|Profiling data parse failed" \
      "${case_dir}/msprof_stdout.log" 2>/dev/null; then
    echo "[WARN] ${tag} possible snapshot/OOM" | tee -a "${SUMMARY}"
  fi
  return 0
}

# --- 1) FIA MXFP8 stock CANN ---
unset ATK_CUSTOM_OPP_PATH ASCEND_CUSTOM_OPP_PATH
export FIA_DEVICE=0 FIA_S=118800 FIA_SPARSE=0 FIA_B=1 FIA_N1=1 FIA_N2=1 FIA_D=128 FIA_OUT=bfloat16
run_msprof fia_mxfp8_sp0 FusedInferAttentionScore \
  "${OPS_SRC}/attention/fused_infer_attention_score/tests/pytest/fia_fullquant_mxfp8_test/fia_mxfp8_msprof_one.py"

# --- 2/3) BSA custom opp ---
source "${OPP_VENDOR}/bin/set_env.bash"
export ATK_CUSTOM_OPP_PATH="${OPP_VENDOR}/op_api/lib/libcust_opapi.so"
export BSA_DEVICE=0 BSA_S=118800 BSA_KEEP=1.0 BSA_B=1 BSA_N1=1 BSA_N2=1 BSA_D=128 BSA_OUT=bfloat16

run_msprof bsa_mxfp8_keep1 BlockSparseAttention \
  "${OPS_SRC}/attention/block_sparse_attention/tests/mxfp8_pytest/bsa_mxfp8_msprof_one.py"

run_msprof bsa_mxfp4_keep1 BlockSparseAttention \
  "${OPS_SRC}/attention/block_sparse_attention/tests/mxfp4_pytest/bsa_mxfp4_msprof_one.py"

echo "ALL_DONE $(date -Iseconds)" | tee -a "${SUMMARY}"
