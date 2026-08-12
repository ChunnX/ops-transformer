#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}
PTA_LIBRARY=${PTA_LIBRARY:-$SCRIPT_DIR/build/pta/liballto_allv_grouped_mat_mul_shmem_pta.so}
PTA_TIMEOUT_SECONDS=${PTA_TIMEOUT_SECONDS:-600}

: "${CUST_OPAPI_LIBRARY:?CUST_OPAPI_LIBRARY must be the absolute path to libcust_opapi.so}"
CUST_OPAPI_LIBRARY=$(readlink -f "$CUST_OPAPI_LIBRARY")
if [[ ! -f "$CUST_OPAPI_LIBRARY" ]]; then
    echo "CUST_OPAPI_LIBRARY does not exist: $CUST_OPAPI_LIBRARY" >&2
    exit 1
fi
VENDOR_DIR=$(dirname "$(dirname "$(dirname "$CUST_OPAPI_LIBRARY")")")
BINARY_INFO_CONFIG="$VENDOR_DIR/op_impl/ai_core/tbe/kernel/config/ascend950/binary_info_config.json"
if [[ ! -r "$BINARY_INFO_CONFIG" ]]; then
    echo "Missing Ascend950 binary_info_config.json: $BINARY_INFO_CONFIG" >&2
    echo "Rebuild and reinstall the custom package with --soc=ascend950 before running PTA." >&2
    exit 1
fi
if ! grep -q '"AlltoAllvGroupedMatMulShmem"' "$BINARY_INFO_CONFIG"; then
    echo "AlltoAllvGroupedMatMulShmem is absent from: $BINARY_INFO_CONFIG" >&2
    echo "The installed custom package does not contain the SHMEM Ascend950 kernel registration." >&2
    exit 1
fi
case ":${ASCEND_CUSTOM_OPP_PATH:-}:" in
    *":$VENDOR_DIR:"*) ;;
    *) ASCEND_CUSTOM_OPP_PATH="$VENDOR_DIR${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}" ;;
esac
export ASCEND_CUSTOM_OPP_PATH
echo "Custom OPP directory: $VENDOR_DIR"
echo "Ascend950 binary config: $BINARY_INFO_CONFIG"

prepend_runtime_dir() {
    local directory=$1
    if [[ ! -d "$directory" ]]; then
        return
    fi
    case ":${LD_LIBRARY_PATH:-}:" in
        *":$directory:"*) ;;
        *) LD_LIBRARY_PATH="$directory${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
    esac
}

# The source operator loads SHMEM's set_env.sh before launching ranks.  The
# standalone PTA bridge must provide the same runtime search path because
# libshmem loads its bootstrap/HYBM dependencies while aclshmemx_init_attr runs.
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}
SHMEM_RUNTIME_LIBRARY_DIR=""
if [[ -n "${SHMEM_HOME_PATH:-}" ]]; then
    if [[ -f "$SHMEM_HOME_PATH/set_env.sh" ]]; then
        # shellcheck disable=SC1090
        source "$SHMEM_HOME_PATH/set_env.sh" -soc_type Ascend950
    fi
    prepend_runtime_dir "$SHMEM_HOME_PATH/shmem/lib"
    if [[ -f "$SHMEM_HOME_PATH/shmem/lib/libshmem.so" ]]; then
        SHMEM_RUNTIME_LIBRARY_DIR="$SHMEM_HOME_PATH/shmem/lib"
    fi
elif [[ -n "${SHMEM_HOME:-}" ]]; then
    if [[ -f "$SHMEM_HOME/set_env.sh" ]]; then
        # shellcheck disable=SC1090
        source "$SHMEM_HOME/set_env.sh" -soc_type Ascend950
    fi
    prepend_runtime_dir "$SHMEM_HOME/lib"
    prepend_runtime_dir "$SHMEM_HOME/build/lib"
    if [[ -f "$SHMEM_HOME/build/lib/libshmem.so" ]]; then
        SHMEM_RUNTIME_LIBRARY_DIR="$SHMEM_HOME/build/lib"
    elif [[ -f "$SHMEM_HOME/lib/libshmem.so" ]]; then
        SHMEM_RUNTIME_LIBRARY_DIR="$SHMEM_HOME/lib"
    fi
else
    echo "SHMEM_HOME_PATH (source SDK layout) or SHMEM_HOME (HCCL-integrated layout) is required" >&2
    exit 1
fi
prepend_runtime_dir "/usr/local/Ascend/driver/lib64/driver"
export LD_LIBRARY_PATH
export SHMEM_LOG_LEVEL=${SHMEM_LOG_LEVEL:-INFO}

if [[ -z "$SHMEM_RUNTIME_LIBRARY_DIR" ]]; then
    echo "SHMEM runtime library directory does not contain libshmem.so; check SHMEM_HOME_PATH/SHMEM_HOME" >&2
    exit 1
fi
echo "SHMEM runtime library directory: $SHMEM_RUNTIME_LIBRARY_DIR"
echo "SHMEM log level: $SHMEM_LOG_LEVEL"

if [[ ! -f "$PTA_LIBRARY" ]]; then
    echo "Missing PTA library: $PTA_LIBRARY" >&2
    echo "Run $SCRIPT_DIR/build_pta.sh first or set PTA_LIBRARY." >&2
    exit 1
fi
if ! command -v timeout >/dev/null 2>&1; then
    echo "GNU timeout is required to bound SHMEM collective failures." >&2
    exit 1
fi

timeout --signal=TERM --kill-after=30 "${PTA_TIMEOUT_SECONDS}s" \
    "$PYTHON_BIN" "$SCRIPT_DIR/test_allto_allv_grouped_mat_mul_shmem.py" \
    --library "$PTA_LIBRARY" "$@"
