#!/usr/bin/env bash
set -eo pipefail

# Change CANN/CUSTOM when moving this directory to another server.
CANN=${CANN:-/usr/local/Ascend/cann}
CUSTOM=${CUSTOM:-/home/liwenguihw/work/custom_install/in_0824}
ROOT="$(cd "$(dirname "$0")" && pwd)"
CASE_START=${CASE_START:-1}
# ATK uses a half-open interval [start, end), so 1..2 runs only case 1.
CASE_END=${CASE_END:-2}

# Do not start a second ATK task from this directory while one is running.
exec 9>"$ROOT/.run.lock"
if ! flock -n 9; then
    echo "BSA HIF8 ATK is already running from $ROOT"
    exit 1
fi

source "$CANN/set_env.sh"
source "$CUSTOM/vendors/custom_transformer/bin/set_env.bash"
SYSTEM_ASCEND_OPP_PATH="$ASCEND_OPP_PATH"
# The system OPP vendor registry is root-owned on this host.  Use the
# user-readable overlay used by compile.sh so ATK case generation can import
# TBE without opening /usr/local/Ascend/.../opp/vendors/config.ini.
if [ -d /home/liwenguihw/work/cann_user_overlay/opp ]; then
    export ASCEND_OPP_PATH=/home/liwenguihw/work/cann_user_overlay/opp
fi
cd "$ROOT"

export PYTHONPATH="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])'):${PYTHONPATH:-}"
export ASCEND_LAUNCH_BLOCKING=1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export BSA_CPU_ONLY=0
export ATK_CUSTOM_OPP_PATH="$CUSTOM/vendors/custom_transformer/op_api/lib/libcust_opapi.so"

# Generate the ATK case JSON, then execute the HIF8 pyaclnn plugin.
atk case -f op_block_sparse_attention.yaml -p generator_blocksparseattention.py
# Runtime tiling libraries live under the real CANN OPP tree.  The overlay is
# needed only by ATK's Python case generator to avoid the unreadable vendor
# registry on this host.
export ASCEND_OPP_PATH="$SYSTEM_ASCEND_OPP_PATH"
atk task -c ./result/op_block_sparse_attention/json/all_op_block_sparse_attention.json \
    -n nodes.yaml \
    -p hif8_atk_test_bsa.py \
    -to 9999 -s "$CASE_START" -e "$CASE_END" -mt 1
