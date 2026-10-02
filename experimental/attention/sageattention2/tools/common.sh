#!/usr/bin/env bash
# common.sh: Reusable environment helpers and logging functions for tools
set -euo pipefail

get_tools_dir() {
    cd "$(dirname "${BASH_SOURCE[0]}")" && pwd
}

get_sage_root_dir() {
    local tools_dir
    tools_dir="$(get_tools_dir)"
    cd "${tools_dir}/.." && pwd
}

detect_python_bin() {
    local sage_root
    sage_root="$(get_sage_root_dir)"
    local python_bin="python3"
    if [ -f "${sage_root}/.venv/bin/python" ]; then
        python_bin="${sage_root}/.venv/bin/python"
    elif [ -f "${sage_root}/../../../../.venv/bin/python" ]; then
        python_bin="${sage_root}/../../../../.venv/bin/python"
    elif command -v python3 &>/dev/null; then
        python_bin="python3"
    elif command -v python &>/dev/null; then
        python_bin="python"
    fi
    echo "${python_bin}"
}

log_info() {
    echo "[INFO] $*"
}

log_warn() {
    echo "[WARN] $*"
}

log_step() {
    echo "[STEP] $*"
}

log_success() {
    echo "[SUCCESS] $*"
}

has_torch() {
    local py_bin="$1"
    "${py_bin}" -c "import torch" &>/dev/null
}

has_npu() {
    local py_bin="$1"
    "${py_bin}" -c "import torch, torch_npu; assert torch.npu.is_available()" &>/dev/null
}
