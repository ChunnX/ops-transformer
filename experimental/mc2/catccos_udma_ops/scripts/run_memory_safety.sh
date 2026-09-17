#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
export CATCCOS_UDMA_VALIDATE_ROUTES=${CATCCOS_UDMA_VALIDATE_ROUTES:-1}

has_cases=0
for argument in "$@"; do
    case "${argument}" in
        --cases|--cases=*) has_cases=1 ;;
    esac
done

arguments=(--memory-safety)
if [[ ${has_cases} -eq 0 ]]; then
    arguments+=(--cases "${PROJECT_DIR}/validation/cases/memory_safety.yaml")
fi
exec python3 "${PROJECT_DIR}/validation/run_accuracy.py" "${arguments[@]}" "$@"
