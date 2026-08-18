#!/bin/bash
# BSA MXFP8 0817 精度冒烟。必须在容器内、source 自定义算子 set_env.bash 后执行。
set -e
cd "$(dirname "$0")"
export BSA_CPU_ONLY=0
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
PY="${PYTHON:-python}"
exec "$PY" -m pytest test_bsa_mxfp8_golden.py -v -s "$@"
