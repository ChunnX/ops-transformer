#!/bin/bash
# FIA MXFP8 OCP msprof op 长序列扫点。
# B=1 Nq=Nkv=1 D=128 TND 非 PA bf16
# sparse_mode=0 (dense 无 mask) 与 3 (rightDownCausal, 2048x2048)
# 过长序列：SKIP_EST 预估超预算则跳过；运行时 OOM → SKIP_WS；同一 sparse 更大 S 不再跑。
set -eo pipefail

HOME_BASE=${HOME_BASE:-/home/npu_user5/xuxuepeng}
OPS_SRC=${OPS_SRC:-${HOME_BASE}/ops-transformer_0703}
CASE_PY="${OPS_SRC}/attention/fused_infer_attention_score/tests/pytest/fia_fullquant_mxfp8_test/fia_mxfp8_msprof_one.py"
INSTALL_PATH=${INSTALL_PATH:-${HOME_BASE}/cann_fia_0817}
OPP_VENDOR="${INSTALL_PATH}/vendors/custom_transformer_transformer"
MSPROF_BIN=${MSPROF_BIN:-/usr/local/Ascend/ascend-toolkit/latest/bin/msprof}
MSPROF_TIMEOUT=${MSPROF_TIMEOUT:-7200}
DEVICE=${FIA_DEVICE:-1}
CONTINUE=${CONTINUE:-0}
SMOKE=${SMOKE:-0}
HBM_BUDGET_GB=${HBM_BUDGET_GB:-108}

SEQ_LIST=${FIA_PERF_S_LIST:-"8422 26352 49408 72000 75348 75600 82800 108000 118800 162000 186300 243000 1048576"}
SPARSE_LIST=${FIA_PERF_SPARSE_LIST:-"0 3"}
KERNEL_NAME=${FIA_KERNEL_NAME:-FusedInferAttentionScore}

source "${HOME_BASE}/venv311/bin/activate"
source /usr/local/Ascend/cann-9.1.0-beta.1/share/info/ascendnpu-ir/bin/set_env.sh
source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash
# 板载 CANN FIA tiling 不认 e8m0 (dtype 37)；MxFP8 必须走本仓库编出来的 custom FIA。
# 不要挂 BSA 的 cann_bsa_0817，避免 Host/Kernel 混装。
if [ ! -f "${OPP_VENDOR}/bin/set_env.bash" ]; then
  echo "FATAL: custom FIA opp not found at ${OPP_VENDOR}" >&2
  echo "Compile+install first: bash build.sh --pkg --soc=ascend950 --ops=fused_infer_attention_score --vendor_name=custom_transformer -j32" >&2
  echo "then: .run --install-path=${INSTALL_PATH}" >&2
  exit 1
fi
source "${OPP_VENDOR}/bin/set_env.bash"
export ATK_CUSTOM_OPP_PATH="${OPP_VENDOR}/op_api/lib/libcust_opapi.so"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export PYTHONUNBUFFERED=1
export ASCEND_RT_VISIBLE_DEVICES="${DEVICE}"
export FIA_DEVICE=0
export FIA_B=1 FIA_N1=1 FIA_N2=1 FIA_D=128 FIA_OUT=bfloat16

cd "$(dirname "$0")"
ROOT=${FIA_MSPROF_ROOT:-"fia_msprof_perf_mxfp8_longseq"}
ALL_SUMMARY="${ROOT}/summary_all.txt"
mkdir -p "${ROOT}"

if [ "${CONTINUE}" != "1" ] && [ "${SMOKE}" != "1" ]; then
  rm -rf "${ROOT}"
  mkdir -p "${ROOT}"
  : > "${ALL_SUMMARY}"
fi
{
  echo "time=$(date -Iseconds) DEVICE_HOST=${DEVICE} HBM_BUDGET_GB=${HBM_BUDGET_GB}"
  echo "SEQ_LIST=${SEQ_LIST}"
  echo "SPARSE_LIST=${SPARSE_LIST}  # 0=dense; 3=rightDownCausal"
  echo "layout=TND non-PA D=128 N=1 out=bf16 MxFP8 OCP q/k=6 v=8"
  echo "msprof warm-up=5 launch-count=10 kernel=${KERNEL_NAME} aic-metrics=PipeUtilization,ArithmeticUtilization,Default"
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
S, sparse = int(sys.argv[1]), int(sys.argv[2])
B, N, D, cores = 1, 1, 128, 28
act = B * (N * S * D * 3)  # e4m3 QKV
out = B * N * S * D * 2
qk_s = 2 * B * N * S * (D // 32)
sg = math.ceil(S / 32)
if sg % 2:
    sg += 1
v_s = (sg // 2) * N * D * 2
mask = 2048 * 2048 if sparse == 3 else 0
# FIA workspace 保守估计：每核若干 fp32 SxD 缓冲；过长序列会先被 SKIP_EST
ws_fia = cores * 4 * S * D * 4
runtime = 4 * (1024 ** 3)
snap = 1.8
print((ws_fia + act + out + qk_s + v_s + mask + runtime) * snap / (1024 ** 3))
PY
}

declare -A SKIP_LARGER
SKIP_LARGER[0]=0
SKIP_LARGER[3]=0

run_one() {
  local S=$1 SP=$2
  local tag="S${S}_sp${SP}"
  local case_dir="${ROOT}/${tag}"
  if [ "${CONTINUE}" = "1" ] && { grep -q "Task Duration" "${case_dir}/msprof_stdout.log" 2>/dev/null \
       || grep -qE "^SKIP_" "${case_dir}/skip.txt" 2>/dev/null; }; then
    echo "SKIP_DONE ${tag}" | tee -a "${ALL_SUMMARY}"
    return 0
  fi
  mkdir -p "${case_dir}"
  local gb
  gb=$(est_gb "${S}" "${SP}")
  echo "=== msprof ${tag} estGB=${gb} ===" | tee -a "${ALL_SUMMARY}"

  if awk -v e="${gb}" -v b="${HBM_BUDGET_GB}" 'BEGIN{exit !(e+0 > b+0)}'; then
    echo "SKIP_EST" > "${case_dir}/skip.txt"
    echo "[SKIP_EST] ${tag} estGB=${gb} > budget=${HBM_BUDGET_GB}" | tee -a "${ALL_SUMMARY}"
    SKIP_LARGER["${SP}"]=1
    return 0
  fi
  if [ "${SKIP_LARGER[${SP}]}" = "1" ]; then
    echo "SKIP_PREV" > "${case_dir}/skip.txt"
    echo "[SKIP_PREV] ${tag} earlier S already SKIP_EST/SKIP_WS for sparse=${SP}" | tee -a "${ALL_SUMMARY}"
    return 0
  fi

  export FIA_S="${S}" FIA_SPARSE="${SP}"
  local warm=5 launch=10
  if [ "${SMOKE}" = "1" ]; then
    warm=0
    launch=1
  fi
  set +e
  timeout "${MSPROF_TIMEOUT}" "${MSPROF_BIN}" op --output="${case_dir}" \
    --aic-metrics=PipeUtilization,ArithmeticUtilization,Default \
    --kernel-name="${KERNEL_NAME}" \
    --warm-up="${warm}" --launch-count="${launch}" --kill=off \
    python3 "${CASE_PY}" \
    > "${case_dir}/msprof_stdout.log" 2>&1
  local rc=$?
  set -e
  local dur
  dur=$(awk -v kn="${KERNEL_NAME}" '
    $0 ~ "Op Name: " kn {p=1}
    p && /Task Duration/ {print; exit}
  ' "${case_dir}/msprof_stdout.log" || true)
  local csv_dur=""
  local opprof
  opprof=$(ls -td "${case_dir}"/OPPROF_* 2>/dev/null | head -1 || true)
  if [ -n "${opprof}" ] && [ -f "${opprof}/OpBasicInfo.csv" ]; then
    csv_dur=$(KERNEL_NAME="${KERNEL_NAME}" python3 - "${opprof}/OpBasicInfo.csv" <<'PY'
import csv, os, sys
p = sys.argv[1]
kn = os.environ.get("KERNEL_NAME", "FusedInferAttentionScore")
with open(p, newline="") as f:
    rows = list(csv.DictReader(f))
best = None
for r in rows:
    name = (r.get("Op Name") or r.get("OpName") or r.get("OP Type") or "")
    if kn not in name:
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
      SKIP_LARGER["${SP}"]=1
    fi
    echo "${tagf}" > "${case_dir}/skip.txt"
    local reason
    reason=$(grep -Eih "CreateSnapshot|207001|Out of memory|MemoryError|ACL_ERROR|GRAPH_FAILED|Error|error" \
      "${case_dir}/msprof_stdout.log" | head -4 || true)
    {
      echo "[${tagf}] ${tag} rc=${rc} estGB=${gb}"
      echo "${reason}"
    } | tee -a "${ALL_SUMMARY}"
  fi
}

if [ "${SMOKE}" = "1" ]; then
  run_one 8422 0
  exit 0
fi

for SP in ${SPARSE_LIST}; do
  SKIP_LARGER["${SP}"]=0
  for S in ${SEQ_LIST}; do
    run_one "${S}" "${SP}"
  done
done

echo "DONE time=$(date -Iseconds)" | tee -a "${ALL_SUMMARY}"
