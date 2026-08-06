#!/bin/bash
# 激活 conda 环境 kdy311（AttnResFwd Torch 测试默认 Python）
# Usage: source env_kdy.sh

# CANN 9.x + torch_npu 2.10 需 Python 3.11（kdy311）；py3.8 的 kdy 最高仅 torch-npu 2.4
CONDA_ENV_NAME="${CONDA_ENV_NAME:-kdy311}"

_activate_kdy() {
    if [[ -n "${CONDA_PREFIX:-}" ]] && [[ "$(basename "${CONDA_PREFIX}")" == "${CONDA_ENV_NAME}" ]]; then
        return 0
    fi

    local bases=(
        "${CONDA_BASE:-}"
        "/data/miniconda3"
        "/root/miniconda3"
        "${HOME}/miniconda3"
        "${HOME}/anaconda3"
        "/opt/conda"
    )

    for base in "${bases[@]}"; do
        [[ -z "${base}" ]] && continue
        if [[ -f "${base}/etc/profile.d/conda.sh" ]]; then
            # shellcheck disable=SC1091
            source "${base}/etc/profile.d/conda.sh"
            conda activate "${CONDA_ENV_NAME}" 2>/dev/null && return 0
        fi
        if [[ -x "${base}/envs/${CONDA_ENV_NAME}/bin/python" ]]; then
            export PATH="${base}/envs/${CONDA_ENV_NAME}/bin:${PATH}"
            return 0
        fi
    done

    echo "ERROR: conda env '${CONDA_ENV_NAME}' not found. Set CONDA_BASE or install env ${CONDA_ENV_NAME}." >&2
    return 1
}

_activate_kdy
