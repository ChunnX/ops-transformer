#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# AttnResFwd pytest 入口（conda kdy311 + CANN + ACLNN / Torch 接口）

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env_kdy.sh"

PYTEST=(python -m pytest -rA -s -W ignore::UserWarning)
echo "Using Python: $(which python) ($(python -V 2>&1))"

setup_npu() {
    python - <<'PY'
import npu_device
print(npu_device.format_status_summary())
dev = npu_device.resolve_device_id()
print(f"Selected NPU device: {dev}")
PY
}

setup_cann() {
    # shellcheck disable=SC1091
    source "${SCRIPT_DIR}/cann_env.sh"
    echo "ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-unset}"
    echo "Custom OPP: ${ATTN_RES_FWD_CUSTOM_SETENV:-not found}"
}

usage() {
    echo "Usage: $0 [cpu|npu|prod|pick10|fails|resident|extended|torch|sweep|grid_tn|lite|all|help]"
    echo "  cpu       - CPU golden self-test"
    echo "  npu/prod  - NPU 生产 shape (CI, ACLNN example)"
    echo "  pick10    - NPU pick10 扫参"
    echo "  fails     - NPU 历史失败用例回归"
    echo "  resident  - NPU 驻留分支"
    echo "  extended  - pick10 + fails"
    echo "  torch     - cann_ops_transformer.attn_res_fwd 精度（CI + 失败回归 + resident）"
    echo "  sweep     - Torch 正向 H 扫描（T=1, N=12, seed=0）"
    echo "  grid_tn   - 泛化精度 N=1..12 × T 采样[1,16384]（H=7168，data_compare）"
    echo "  lite      - 单 shape 多实现精度+性能（attenres_lite.py，见 ATTENRES_LITE.md）"
    echo "  all       - cpu + ci npu (ACLNN L0)"
    echo ""
    echo "Environment:"
    echo "  conda ${CONDA_ENV_NAME:-kdy311} (override: CONDA_ENV_NAME=xxx)"
    echo "  CANN_PKG=${CANN_PKG:-cann-9.2.0-weekly.20260715.01-20260715} (override in cann_env.sh)"
    echo "  DEVICE_ID=0        手动指定 NPU（默认自动选空闲卡）"
    echo "  FORCE_AUTO_DEVICE=1  忽略 DEVICE_ID，强制自动选卡"
}

run_cpu() {
    echo "===== AttnResFwd CPU golden ====="
    "${PYTEST[@]}" test_attn_res_fwd_l0.py::test_attn_res_fwd_cpu_golden_l0 -v
}

run_prod() {
    echo "===== AttnResFwd NPU production (ACLNN example) ====="
    setup_npu
    "${PYTEST[@]}" test_attn_res_fwd_l0.py::test_attn_res_fwd_npu_precision_l0 -v -m ci
}

run_pick10() {
    echo "===== AttnResFwd NPU pick10 (not implemented in L0) ====="
    echo "Use: bash test_run.sh torch -k pick10"
    exit 1
}

run_fails() {
    echo "===== AttnResFwd NPU previous fails (use torch) ====="
    setup_cann
    setup_npu
    "${PYTEST[@]}" test_attn_res_fwd_torch.py -k previous_fails -v -m extended
}

run_resident() {
    echo "===== AttnResFwd Torch resident ====="
    setup_cann
    setup_npu
    "${PYTEST[@]}" test_attn_res_fwd_torch.py -k resident -v -m resident
}

run_extended() {
    echo "===== AttnResFwd Torch extended (pick10 + fails) ====="
    setup_cann
    setup_npu
    "${PYTEST[@]}" test_attn_res_fwd_torch.py -v -m extended
}

run_torch() {
    echo "===== AttnResFwd Torch (cann_ops_transformer) ====="
    setup_cann
    setup_npu
    "${PYTEST[@]}" test_attn_res_fwd_torch.py -v
}

run_sweep() {
    echo "===== AttnResFwd Torch forward H sweep ====="
    setup_cann
    setup_npu
    "${PYTEST[@]}" test_attn_res_fwd_torch.py -v -m sweep
}

run_grid_tn() {
    echo "===== AttnResFwd Torch grid_tn (N=1..12 × T sample [1,16384], H=7168) ====="
    setup_cann
    setup_npu
    "${PYTEST[@]}" test_attn_res_fwd_torch.py -v -m grid_tn
}

run_lite() {
    echo "===== AttnResFwd attenres_lite (single-shape precision + profiler) ====="
    echo "Doc: ${SCRIPT_DIR}/ATTENRES_LITE.md"
    setup_cann
    setup_npu
    python attenres_lite.py "$@"
}

run_all() {
    run_cpu
    run_prod
}

case "${1:-all}" in
    cpu) run_cpu ;;
    npu|prod) run_prod ;;
    pick10) run_pick10 ;;
    fails) run_fails ;;
    resident) run_resident ;;
    extended) run_extended ;;
    torch) run_torch ;;
    sweep) run_sweep ;;
    grid_tn) run_grid_tn ;;
    lite)
        shift || true
        run_lite "$@"
        ;;
    all) run_all ;;
    help|-h|--help) usage ;;
    *)
        usage
        exit 1
        ;;
esac
