#!/usr/bin/env bash
# Install this task's freshly built formal JIT package under the resource lock.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPS_ROOT="$(cd "${HERE}/../../../.." && pwd)"
TASK_ROOT="$(cd "${OPS_ROOT}/.." && pwd)"
CANN_ROOT="${CANN_ROOT:-/home/jinxh/Ascend_0722/cann-9.2.0}"
OPP_ROOT="${PRO0921_OPP_ROOT:-${TASK_ROOT}/sim-opp}"
RUN_PKG="${1:-${OPS_ROOT}/build/cann-ops-transformer-pro0921latent_linux-x86_64.run}"
case "${OPP_ROOT}" in
  "${TASK_ROOT}"/*) ;;
  *) echo "FATAL: OPP must remain under this task: ${TASK_ROOT}" >&2; exit 2 ;;
esac
[[ -f "${RUN_PKG}" && -f "${CANN_ROOT}/opp/scene.info" ]] || {
  echo "FATAL: package or CANN scene.info missing" >&2; exit 2;
}
# Always install the supplied package. Reusing an old staging tree would hide changes.
bash "${RUN_PKG}" --quiet --install-path="${OPP_ROOT}"
VENDOR="pro0921latent_transformer"
[[ -d "${OPP_ROOT}/vendors/${VENDOR}/op_impl/ai_core/tbe" ]] || {
  echo "FATAL: expected vendor missing" >&2; exit 2;
}
ln -sfnT "${CANN_ROOT}/opp/built-in" "${OPP_ROOT}/built-in"
ln -sfnT "${CANN_ROOT}/opp/scene.info" "${OPP_ROOT}/scene.info"
printf 'load_priority=%s\n' "${VENDOR}" > "${OPP_ROOT}/vendors/config.ini"
mkdir -p "$(dirname "${OPP_ROOT}")/x86_64-linux/data"
ln -sfnT "${CANN_ROOT}/x86_64-linux/data/platform_config" \
  "$(dirname "${OPP_ROOT}")/x86_64-linux/data/platform_config"
echo "PRO0921_OPP_ROOT=${OPP_ROOT}"
echo "PRO0921_VENDOR_DIR=${OPP_ROOT}/vendors/${VENDOR}"
