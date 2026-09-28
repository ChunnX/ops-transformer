#!/usr/bin/env bash
# One-click TTK/CAMODEL run for the tiny formal-operator cases.
#
# Run ONLY inside the shared local-resource-runner lock. Each case gets a hard
# 180 s wall-clock timeout; there is no warmup and no performance matrix.
#
# Prerequisites:
#   bash compile_formal_ops.sh      (JIT host build)
#   bash install_formal_opp.sh      (independent opp tree; idempotent)
#   python3 build_tiny_cases.py --out <case-root>
#   python3 build_quant_cases.py --out <case-root>   (optional quant case)
#   python3 build_smla_linked_cases.py --out <mla-root> --metadata-npy <npy>
#     (optional tiny sparse_flash_mla case; a real metadata tensor is required.
#      Generate it with aicpu_metadata_host/, which runs the real
#      SparseFlashMlaMetadata AICPU kernel on the host.)
#
# Vendor registration (verified against the working parent-session probe):
#   * ASCEND_OPP_PATH points at the task-owned OPP root, whose
#     vendors/config.ini `load_priority` names the installed vendor directory.
#   * ASCEND_CUSTOM_OPP_PATH must stay unset: TTK prefers the custom path over
#     the vendor config, and with the custom path the `_impl_prefix` becomes
#     empty so `impl.dynamic` cannot be imported.
#
# Usage:
#   bash run_tiny_sim.sh [case-root] [selector]
#     selector is either a cmp_ratio ("128") or a case-name substring
#     ("hca_b1_boundary").  PRO0921_FIRST_ONLY=1 keeps only the first case
#     (the HCA boundary probe).  PRO0921_READBACK_XCACHE=1 enables the local
#     input read-back hook (see run_case_ttk.py).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_ROOT="$(cd "${HERE}/../../../../.." && pwd)"
CASE_ROOT="${1:-/tmp/pro0921_latent_cases}"
SELECTOR="${2:-}"
CANN_ROOT="${CANN_ROOT:-/home/jinxh/Ascend_0722/cann-9.2.0}"
INSTALL_ROOT="${INSTALL_ROOT:-${TASK_ROOT}/sim-opp}"
OPS_TEST_KIT="${OPS_TEST_KIT:-/home/jinxh/ops-test-kit}"
PY_DEPS="${PRO0921_PY_DEPS:-${TASK_ROOT}/sim-python-deps}"
SIM_LIB="${CANN_ROOT}/tools/simulator/Ascend950PR_9589/lib"
SIM_PRELOAD="${SIM_LIB}/libpem_davinci.so:${SIM_LIB}/libstars.so:${SIM_LIB}/libffts_model.so:${SIM_LIB}/libmcu_loop.so:${SIM_LIB}/libmcu_wrapper.so:${SIM_LIB}/libmodel_top.so:${SIM_LIB}/libnpu_drv.so"
PYTHON_BIN="${PRO0921_PYTHON:-/home/jinxh/.pyenv/versions/3.11.13/bin/python3}"
CASE_TIMEOUT="${PRO0921_CASE_TIMEOUT_S:-170}"
OUTER_TIMEOUT=$((CASE_TIMEOUT + 10))          # hard 180 s per case by default
FIRST_ONLY="${PRO0921_FIRST_ONLY:-0}"
READBACK="${PRO0921_READBACK_XCACHE:-0}"

[[ -x "${PYTHON_BIN}" ]] || { echo "FATAL: Python missing: ${PYTHON_BIN}" >&2; exit 2; }
if [[ ! -f "${CANN_ROOT}/set_env.sh" ]]; then
  echo "FATAL: ${CANN_ROOT}/set_env.sh not found" >&2
  exit 2
fi

# Fixed task-owned tree; never fall back to an ambient toolkit installation.
OPP_ROOT="${PRO0921_OPP_ROOT:-${INSTALL_ROOT}}"
[[ -e "${OPP_ROOT}/scene.info" && -e "${OPP_ROOT}/built-in" && -d "${OPP_ROOT}/vendors" ]] || {
  echo "FATAL: incomplete task OPP: ${OPP_ROOT}" >&2; exit 2;
}
SCENE_OS="$(sed -n 's/^os=//p' "${OPP_ROOT}/scene.info" | head -1)"
SCENE_ARCH="$(sed -n 's/^arch=//p' "${OPP_ROOT}/scene.info" | head -1)"
PLATFORM_INI="${OPP_ROOT}/../${SCENE_ARCH}-${SCENE_OS}/data/platform_config/Ascend950PR_9589.ini"
if [[ -n "${SCENE_OS}" && -n "${SCENE_ARCH}" && ! -f "${PLATFORM_INI}" ]]; then
  echo "FATAL: ${PLATFORM_INI} not found; platform_config is required by get_npu_hw_info" >&2
  exit 2
fi
# TTK picks the vendor from <opp>/vendors/config.ini load_priority; resolve the
# concrete directory (the build installs "<vendor>_transformer").
VENDOR_DIR="${OPP_ROOT}/vendors/pro0921latent_transformer"
[[ -d "${VENDOR_DIR}" ]] || { echo "FATAL: vendor missing: ${VENDOR_DIR}" >&2; exit 2; }
TBE_DIR="${VENDOR_DIR}/op_impl/ai_core/tbe"

# Explicit toolkit path only.
if [[ "${PRO0921_DISABLE_PROXY:-0}" == "1" ]]; then
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
fi
# shellcheck disable=SC1091
source "${CANN_ROOT}/set_env.sh"
# The custom-OPP path would override the vendor config and blank the impl
# prefix; the working probe unset it (and these stale toolkit selectors).
unset ASCEND_CUSTOM_OPP_PATH ASCEND_CUSTOM_PATH ASCEND_TOOLKIT_HOME ASCEND_HOME_PATH
export ASCEND_OPP_PATH="${OPP_ROOT}"
export LD_LIBRARY_PATH="${CANN_ROOT}/tools/simulator/Ascend950PR_9589/lib:${LD_LIBRARY_PATH:-}"
# PYTHONPATH: vendor TBE first, then TTK, the toolkit python package, the
# toolkit built-in TBE, and the task-owned python deps (en_dtypes 0.0.4 for the
# hifloat8 quant case).  The deps dir is prepended last so it can supply
# packages the toolkit does not ship, without shadowing the toolkit.
export PYTHONPATH="${TBE_DIR}:${OPS_TEST_KIT}:${CANN_ROOT}/python/site-packages:${CANN_ROOT}/opp/built-in/op_impl/ai_core/tbe:${PY_DEPS}"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
unset TILINGKEY_PAR_COMPILE ASCENDC_PAR_COMPILE_JOB
export TILINGKEY_PARALLEL_JOB=4

# Keep full instruction traces off for bounded smoke; preserve simulator stats.
export CAMODEL_CONFIG_PATH="${CASE_ROOT}/camodel-config"
"${PYTHON_BIN}" - "${SIM_LIB}" "${CAMODEL_CONFIG_PATH}" <<'PYCONFIG'
import json
import os
import sys
from pathlib import Path
source, target = map(Path, sys.argv[1:])
target.mkdir(parents=True, exist_ok=True)
config = json.loads((source / "config.json").read_text())
config["LOG"]["core_enable_mask"] = ["0x1" if os.environ.get("PRO0921_TRACE_CORE0") == "1" else "0x0"]
stars = json.loads((source / "config_stars.json").read_text())
assert stars["model_top"]["num_aic"] == 32 and stars["model_top"]["num_aiv"] == 64
(target / "config.json").write_text(json.dumps(config, indent=2))
(target / "config_stars.json").write_text(json.dumps(stars, indent=2))
PYCONFIG

echo "python    : ${PYTHON_BIN}"
echo "opp root  : ${ASCEND_OPP_PATH}"
echo "vendor    : ${VENDOR_DIR}"
echo "py deps   : ${PY_DEPS}"
echo "case root : ${CASE_ROOT}"
echo "budget    : ${OUTER_TIMEOUT}s outer / ${CASE_TIMEOUT}s inner, no warmup"
echo "readback  : ${READBACK}"

mapfile -t CASE_DIRS < <("${PYTHON_BIN}" - "${CASE_ROOT}" "${SELECTOR}" "${FIRST_ONLY}" <<'PYEOF'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
selector = sys.argv[2]
first_only = sys.argv[3] == "1"
manifest = []
for name in ("cases.json", "quant_cases.json", "smla_cases.json"):
    path = root / name
    if not path.exists():
        continue
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        # smla_cases.json wraps the list under "cases".
        data = data.get("cases", [])
    manifest.extend(data)
if first_only:
    manifest = manifest[:1]
exact_names = {entry["name"] for entry in manifest}
for entry in manifest:
    if selector:
        if selector.isdigit():
            if str(entry["cmp_ratio"]) != selector:
                continue
        elif selector in exact_names and selector != entry["name"]:
            continue
        elif selector not in entry["name"]:
            continue
    print(entry["dir"])
PYEOF
)

if [[ "${#CASE_DIRS[@]}" -eq 0 ]]; then
  echo "FATAL: no cases under ${CASE_ROOT} for selector '${SELECTOR}' (build them first)" >&2
  exit 2
fi

failures=0
for case_dir in "${CASE_DIRS[@]}"; do
  name="$(basename "${case_dir}")"
  echo "=== CASE_START ${name}"
  # truncate instead of delete: `rm -f` may be blocked by a restricted sandbox,
  # and a stale result.csv is harmless because check_result is only reached on rc == 0.
  : > "${case_dir}/exit_code.txt"
  : > "${case_dir}/result.csv"
  mkdir -p "${case_dir}/ascend-log" "${case_dir}/test-data" "${case_dir}/camodel-log"
  set +e
  (
    cd "${case_dir}"     # the TTK subprocess must run with cwd == case dir
    LD_PRELOAD="${SIM_PRELOAD}" \
    PRO0921_CASE_DIR="${case_dir}" \
    PRO0921_CASE_NAME="${name}" \
    PRO0921_PLUGIN_FILE="${HERE}/plugin.py" \
    PRO0921_CASE_TIMEOUT_S="${CASE_TIMEOUT}" \
    PRO0921_READBACK_XCACHE="${READBACK}" \
    ASCEND_PROCESS_LOG_PATH="${case_dir}/ascend-log" \
    TEST_DATA_ROOT_PATH="${case_dir}/test-data" \
    CAMODEL_LOG_PATH="${case_dir}/camodel-log" \
      timeout --foreground --signal=TERM --kill-after=20s "${OUTER_TIMEOUT}" \
      "${PYTHON_BIN}" "${HERE}/run_case_ttk.py" > "${case_dir}/run.log" 2>&1
  )
  rc=$?
  set -e
  echo "${rc}" > "${case_dir}/exit_code.txt"
  echo "=== CASE_EXIT ${name} rc=${rc}"
  if [[ "${rc}" -ne 0 ]]; then
    echo "FAIL ${name}: nonzero exit (124/137 == timeout). first error lines:"
    grep -iE "error|fail|trap|exception|timeout" "${case_dir}/run.log" | head -20 || true
    failures=$((failures + 1))
    continue
  fi
  verdict="$("${PYTHON_BIN}" "${HERE}/check_result.py" "${case_dir}")"
  echo "${verdict}"
  case "${verdict}" in
    PASS:*)
      if [[ "${READBACK}" == "1" && -f "${case_dir}/readback_spec.json" ]]; then
        if ! "${PYTHON_BIN}" "${HERE}/check_xcache_readback.py" "${case_dir}"; then
          failures=$((failures + 1))
        fi
      fi
      ;;
    *) failures=$((failures + 1)) ;;
  esac
done

if [[ "${failures}" -ne 0 ]]; then
  echo "PRO0921_TINY_SIM=FAIL failures=${failures}"
  exit 1
fi
echo "PRO0921_TINY_SIM=PASS cases=${#CASE_DIRS[@]}"
