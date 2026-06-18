#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License"). See LICENSE in the root of the repository.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
RUNNER="${REPO_ROOT}/scripts/smla_generalization_test_runner.py"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DEVICE_ID="${DEVICE_ID:-0}"
CASE_NUM="${CASE_NUM:-700}"
RESULT_DIR="${RESULT_DIR:-${SCRIPT_DIR}/result}"

mkdir -p "${RESULT_DIR}"

RUN_ARGS=(
    --repo-root "${REPO_ROOT}"
    --device "${DEVICE_ID}"
    --total "${CASE_NUM}"
    --out "${RESULT_DIR}/smla_700_cases.jsonl"
    --summary-out "${RESULT_DIR}/smla_700_cases.summary.json"
)

if [[ -n "${CASE_START:-}" ]]; then
    RUN_ARGS+=(--case-start "${CASE_START}")
fi

if [[ -n "${CASE_END:-}" ]]; then
    RUN_ARGS+=(--case-end "${CASE_END}")
fi

if [[ "${CONTINUOUS_KV_ONLY:-0}" != "0" ]]; then
    RUN_ARGS+=(--continuous-kv-only)
fi

if [[ "${STOP_ON_FAIL:-0}" != "0" ]]; then
    RUN_ARGS+=(--stop-on-fail)
fi

"${PYTHON_BIN}" "${RUNNER}" "${RUN_ARGS[@]}" "$@"
