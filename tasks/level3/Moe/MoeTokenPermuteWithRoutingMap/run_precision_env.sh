#!/usr/bin/env bash
# 在启动 Python 之前设置 ASCEND_CUSTOM_OPP_PATH / LD_LIBRARY_PATH，避免 precision 阶段
# 先于 _set_custom_opp_path 执行 import torch_npu 导致 dlsym 找不到 aclnnMoeTokenPermuteWithRoutingMap。
# 用法（先 conda activate cann90、source set_env.sh）：
#   bash tasks/level3/Moe/MoeTokenPermuteWithRoutingMap/run_precision_env.sh \
#     -task_name MoeTokenPermuteWithRoutingMap -stages compile precision -n_case 10
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS="$(cd "$HERE/../../../.." && pwd)"
RT="$WS/scripts/run_test.py"
if [[ ! -f "$RT" ]]; then
  echo "run_precision_env: 未找到 $RT" >&2
  exit 1
fi
mapfile -t hits < <(find "$WS/runs/lvl3/Moe/MoeTokenPermuteWithRoutingMap" \
  -path '*/packages/vendors/*/op_api/lib/libcust_opapi.so' -type f 2>/dev/null || true)
if [[ ${#hits[@]} -eq 0 ]]; then
  echo "run_precision_env: 未找到 libcust_opapi.so，请先在本 workspace 下完成 compile。" >&2
  exit 1
fi
SO="${hits[0]}"
latest=0
for h in "${hits[@]}"; do
  t=$(stat -c %Y "$h" 2>/dev/null || echo 0)
  if [[ "$t" -ge "$latest" ]]; then
    latest="$t"
    SO="$h"
  fi
done
OP_API_LIB="$(cd "$(dirname "$SO")" && pwd)"
VENDOR_ROOT="$(cd "$OP_API_LIB/../.." && pwd)"
export ASCEND_CUSTOM_OPP_PATH="${VENDOR_ROOT}/"
export LD_LIBRARY_PATH="${OP_API_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
exec python "$RT" "$@"
