#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
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
# Both delivered device kernels contain "Grouped" in their kernel names. Keep
# an override for tool versions or locally renamed kernels that need a more
# precise pattern.
kernel_name=${MSSANITIZER_KERNEL_NAME:-Grouped}
if [[ -z "${kernel_name}" ]]; then
    echo "MSSANITIZER_KERNEL_NAME must not be empty." >&2
    exit 2
fi
command+=(--kernel-name="${kernel_name}")

# msSanitizer performs the memory instrumentation itself. Do not reuse the NaN
# poison/red-zone validation path here: it introduces Fill kernels and several
# target launches. One selected case is launched exactly once per rank, with an
# internally allocated output. Route metadata uses Device tensors by default;
# set MSSANITIZER_ROUTE_INPUT=list (or pass the corresponding CLI option) to
# exercise the Python-list path instead.
route_input=${MSSANITIZER_ROUTE_INPUT:-tensor}
case "${route_input}" in
    tensor|list) ;;
    *)
        echo "MSSANITIZER_ROUTE_INPUT must be either tensor or list." >&2
        exit 2
        ;;
esac

has_cases=0
for argument in "$@"; do
    case "${argument}" in
        --cases|--cases=*) has_cases=1 ;;
    esac
done

worker=(
    python3 "${PROJECT_DIR}/validation/run_accuracy.py"
    --mssanitizer
    --mssanitizer-route-input "${route_input}"
)
if [[ ${has_cases} -eq 0 ]]; then
    worker+=(--cases "${PROJECT_DIR}/validation/cases/memory_safety.yaml")
fi
worker+=("$@")
command+=(-- "${worker[@]}")
exec "${command[@]}"
