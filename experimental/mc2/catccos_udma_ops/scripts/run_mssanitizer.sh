#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if ! command -v mssanitizer >/dev/null 2>&1; then
    echo "mssanitizer was not found in PATH; source the CANN environment first." >&2
    exit 127
fi

# memcheck catches local GM/MTE violations; initcheck complements the NaN
# poison by finding uninitialized local memory. Override for platform support,
# for example MSSANITIZER_TOOLS=memcheck.
IFS=',' read -r -a tools <<< "${MSSANITIZER_TOOLS:-memcheck,initcheck}"
command=(mssanitizer)
for tool in "${tools[@]}"; do
    command+=(-t "${tool}")
done
# run_memory_safety.sh deliberately launches allocator/workspace/output NaN
# poison kernels before the fused operator.  Without a kernel filter,
# msSanitizer instruments those Fill kernels as well and obscures the result we
# care about.  Both delivered device kernels contain "Grouped" in their kernel
# names, while the setup kernels do not.  Keep an override for tool versions or
# locally renamed kernels that need a more precise pattern.
kernel_name=${MSSANITIZER_KERNEL_NAME:-Grouped}
if [[ -z "${kernel_name}" ]]; then
    echo "MSSANITIZER_KERNEL_NAME must not be empty." >&2
    exit 2
fi
command+=(--kernel-name="${kernel_name}")
command+=(-- bash "${SCRIPT_DIR}/run_memory_safety.sh" "$@")
exec "${command[@]}"
