#!/bin/bash
# BSA FP16 msprof op 长序列扫点（与 MXFP8/MXFP4 同矩阵）。
# B=1 N=1 D=128 BNSD QKV/out=fp16 quant_mode=0 block=(128,128)
# 稀疏度 50%/80% → BSA_KEEP=0.5/0.2
set -eo pipefail

HOME_BASE=${HOME_BASE:-/home/npu_user5/xuxuepeng}
OPS_SRC=${OPS_SRC:-${HOME_BASE}/ops-transformer_0703}
CASE_PY="${OPS_SRC}/attention/block_sparse_attention/tests/fp16_pytest/bsa_fp16_msprof_one.py"
INSTALL_PATH=${INSTALL_PATH:-${HOME_BASE}/cann_bsa_0817}
OPP_VENDOR="${INSTALL_PATH}/vendors/custom_transformer_transformer"
MSPROF_BIN=${MSPROF_BIN:-/usr/local/Ascend/ascend-toolkit/latest/bin/msprof}
MSPROF_TIMEOUT=${MSPROF_TIMEOUT:-7200}
DEVICE=${BSA_DEVICE:-4}
CONTINUE=${CONTINUE:-0}
SMOKE=${SMOKE:-0}

SEQ_LIST=${BSA_PERF_S_LIST:-"8422 26352 49408 72000 75348 75600 82800 108000 118800 162000 186300 243000 1048576"}
KEEP_LIST=${BSA_PERF_KEEP_LIST:-"0.5 0.2"}

source "${HOME_BASE}/venv311/bin/activate"
source /usr/local/Ascend/cann-9.1.0-beta.1/share/info/ascendnpu-ir/bin/set_env.sh
source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash
source "${OPP_VENDOR}/bin/set_env.bash"
export ATK_CUSTOM_OPP_PATH="${OPP_VENDOR}/op_api/lib/libcust_opapi.so"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export PYTHONUNBUFFERED=1
export ASCEND_RT_VISIBLE_DEVICES="${DEVICE}"
export BSA_DEVICE=0
export BSA_B=1 BSA_N1=1 BSA_N2=1 BSA_D=128 BSA_OUT=float16

cd "$(dirname "$0")"
ROOT=${BSA_MSPROF_ROOT:-"bsa_msprof_perf_fp16_longseq"}
ALL_SUMMARY="${ROOT}/summary_all.txt"
mkdir -p "${ROOT}"

if [ "${CONTINUE}" != "1" ] && [ "${SMOKE}" != "1" ]; then
  rm -rf "${ROOT}"
  mkdir -p "${ROOT}"
  : > "${ALL_SUMMARY}"
fi
{
  echo "time=$(date -Iseconds) INSTALL_PATH=${INSTALL_PATH} DEVICE_HOST=${DEVICE}"
  echo "SEQ_LIST=${SEQ_LIST}"
  echo "KEEP_LIST=${KEEP_LIST}  # 0.5=稀疏度50%保留50%; 0.2=稀疏度80%保留20%"
  echo "layout=BNSD D=128 N=1 QKV/out=fp16 quantMode=0 block=128,128"
  echo "msprof warm-up=5 launch-count=10 aic-metrics=PipeUtilization,ArithmeticUtilization,Default"
  echo "===="
} >> "${ALL_SUMMARY}"

is_ws_fail() {
  local logf=$1
  grep -Eiq \
    "CreateSnapshot|207001|Out of memory|out of memory|ACL_ERROR_RT_MEMORY|MemoryError" \
    "${logf}" 2>/dev/null
}

est_gb() {
  python3 - "$1" "$2" <<'PY'
import math, sys
S, keep = int(sys.argv[1]), float(sys.argv[2])
B, N, D = 1, 1, 128
bx = by = 128
xb, yb = math.ceil(S / bx), math.ceil(S / by)
ws = B * N * xb * yb * 4 + B * N * xb * 4 + 16 * (1024 ** 2)
act = B * (N * S * D * 3) * 2  # fp16 QKV
mask = B * N * xb * yb
out = B * N * S * D * 2
runtime = 2 * (1024 ** 3)
snap = 1.5
print((ws + act + mask + out + runtime) * snap / (1024 ** 3))
PY
}

run_one() {
  local S=$1 KEEP=$2
  local tag="S${S}_keep${KEEP}"
  local case_dir="${ROOT}/${tag}"
  if [ "${CONTINUE}" = "1" ] && { grep -q "Task Duration" "${case_dir}/msprof_stdout.log" 2>/dev/null \
       || grep -qE "^SKIP_" "${case_dir}/skip.txt" 2>/dev/null; }; then
    echo "SKIP_DONE ${tag}" | tee -a "${ALL_SUMMARY}"
    return 0
  fi
  mkdir -p "${case_dir}"
  local gb
  gb=$(est_gb "${S}" "${KEEP}")
  echo "=== msprof ${tag} estGB=${gb} ===" | tee -a "${ALL_SUMMARY}"
  export BSA_S="${S}" BSA_KEEP="${KEEP}"
  local warm=5 launch=10
  if [ "${SMOKE}" = "1" ]; then
    warm=0
    launch=1
  fi
  set +e
  timeout "${MSPROF_TIMEOUT}" "${MSPROF_BIN}" op --output="${case_dir}" \
    --aic-metrics=PipeUtilization,ArithmeticUtilization,Default \
    --kernel-name=BlockSparseAttention \
    --warm-up="${warm}" --launch-count="${launch}" --kill=off \
    python3 "${CASE_PY}" \
    > "${case_dir}/msprof_stdout.log" 2>&1
  local rc=$?
  set -e
  local dur
  dur=$(awk '/Op Name: BlockSparseAttention/{p=1} p&&/Task Duration/{print; exit}' \
    "${case_dir}/msprof_stdout.log" || true)
  local csv_dur=""
  local opprof
  opprof=$(ls -td "${case_dir}"/OPPROF_* 2>/dev/null | head -1 || true)
  if [ -n "${opprof}" ] && [ -f "${opprof}/OpBasicInfo.csv" ]; then
    csv_dur=$(python3 - "${opprof}/OpBasicInfo.csv" <<'PY'
import csv, sys
p = sys.argv[1]
with open(p, newline="") as f:
    rows = list(csv.DictReader(f))
best = None
for r in rows:
    name = (r.get("Op Name") or r.get("OpName") or "")
    if "BlockSparseAttention" not in name:
        continue
    d = r.get("Task Duration(us)") or r.get("Task Duration(us) ") or ""
    try:
        v = float(d)
    except Exception:
        continue
    if best is None or v > best[0]:
        best = (v, name, r.get("Block Dim") or r.get("BlockDim") or "",
                r.get("Current Freq(MHz)") or "", r.get("Rated Freq(MHz)") or "")
if best:
    print(f"CSV Task Duration(us)={best[0]} Op={best[1]} BlockDim={best[2]} Freq={best[3]}/{best[4]}")
PY
    )
  fi
  if { [ -n "${dur}" ] || [ -n "${csv_dur}" ]; } && [ "${rc}" = "0" ]; then
    {
      echo "[OK] ${tag} rc=${rc} estGB=${gb}"
      echo "${csv_dur}"
      echo "${dur}"
    } | tee -a "${ALL_SUMMARY}"
  else
    local tagf="SKIP_FAIL"
    if is_ws_fail "${case_dir}/msprof_stdout.log"; then
      tagf="SKIP_WS"
    fi
    echo "${tagf}" > "${case_dir}/skip.txt"
    local reason
    reason=$(grep -Eih "CreateSnapshot|207001|Out of memory|ACL_ERROR|GRAPH_FAILED|Error|error" \
      "${case_dir}/msprof_stdout.log" | head -4 || true)
    {
      echo "[${tagf}] ${tag} rc=${rc} estGB=${gb}"
      echo "${reason}"
    } | tee -a "${ALL_SUMMARY}"
  fi
}

if [ "${SMOKE}" = "1" ]; then
  run_one 8422 0.5
  exit 0
fi

for KEEP in ${KEEP_LIST}; do
  for S in ${SEQ_LIST}; do
    run_one "${S}" "${KEEP}"
  done
done

echo "DONE time=$(date -Iseconds)" | tee -a "${ALL_SUMMARY}"
