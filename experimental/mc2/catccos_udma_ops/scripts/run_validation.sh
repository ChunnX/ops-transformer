#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
echo "run_validation.sh is deprecated and forwards to run_accuracy.sh." >&2
exec bash "${SCRIPT_DIR}/run_accuracy.sh" "$@"
