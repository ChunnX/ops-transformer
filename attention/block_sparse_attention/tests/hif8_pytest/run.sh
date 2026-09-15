#!/bin/bash

# Change these two paths when moving to another server.
CANN=/usr/local/Ascend/cann
CUSTOM=/home/liwenguihw/work/custom_install/in_0824

source "$CANN/set_env.sh"
source "$CUSTOM/vendors/custom_transformer/bin/set_env.bash"
cd "$(dirname "$0")" || exit 2

export PYTHONPATH="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])'):${PYTHONPATH:-}"
export ASCEND_LAUNCH_BLOCKING=1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export BSA_CPU_ONLY=0
# Avoid importing/JIT-building every cann_ops_transformer operator while
# pytest is collecting this focused BSA suite.
export CANN_OPS_TRANSFORMER_LOAD_ONLY=block_sparse_attention
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-6}
export BSA_HIF8_IGNORE_VENDOR_PERM=1
export BSA_HIF8_RUN_GENERATED_600=1
export BSA_HIF8_CASE_START=0
export BSA_HIF8_CASE_COUNT=600
LOG=${LOG:-/home/liwenguihw/work/run.log}

python -u -m pytest test_bsa_hif8_golden.py::TestBsaHif8Golden::test_generated_600_cases -v -s \
    > "$LOG" 2>&1
