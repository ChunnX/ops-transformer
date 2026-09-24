#!/usr/bin/env bash
set -eo pipefail
vendor_tools_dir=$(cd "$(dirname "$0")" && pwd)
vendor_python_site=${1:?Usage: build_torch.sh PYTHON_SITE [VENDOR] [WHEEL_DIR]}
vendor_name=${2:-codex_gmm}
vendor_wheel_dir=${3:-$(dirname "$vendor_python_site")/wheels}
vendor_repo_root=$(cd "$vendor_tools_dir/../../.." && pwd)
source "${ASCEND_HOME_PATH:-/usr/local/Ascend/cann}/set_env.sh"
exec python3 "$vendor_repo_root/experimental/tools/build_torch_vendor.py" \
  --ops gmm_k_dim,gmm_local_exp,gmm_local_exp_with_zero --vendor "$vendor_name" \
  --output-dir "$vendor_wheel_dir" --install-dir "$vendor_python_site"
