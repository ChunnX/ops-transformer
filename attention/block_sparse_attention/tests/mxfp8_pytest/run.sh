#!/bin/bash
# BSA MXFP8 0817 精度。必须在容器内、source 自定义算子 set_env.bash 后执行。
#
# 默认: 全量 test_bsa_mxfp8_golden.py
# 合入前回归（OCP 1–3: varlen + act_s + trap + gap + D=64 layout）:
#   ./run.sh --reg
# -k 与 test_bsa_mxfp8_golden.MXFP8_OCP_REG_K 保持一致，必须整段一个参数、不要换行。
set -e
cd "$(dirname "$0")"
export BSA_CPU_ONLY=0
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
PY="${PYTHON:-python}"

MXFP8_OCP_REG_K='tnd_varlen or actq or scale_stride_trap or gap_sparse or (d64 and (bsnd_mha or tnd_mha or tnd_gqa) and not kvs4096 and not kvs4097)'

if [ "${1:-}" = "--reg" ] || [ "${1:-}" = "reg" ]; then
  shift
  exec "$PY" -m pytest test_bsa_mxfp8_golden.py -v -s --tb=short -k "$MXFP8_OCP_REG_K" "$@"
fi
exec "$PY" -m pytest test_bsa_mxfp8_golden.py -v -s "$@"
