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
command+=(-- bash "${SCRIPT_DIR}/run_memory_safety.sh" "$@")
exec "${command[@]}"
