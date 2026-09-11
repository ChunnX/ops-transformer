#!/bin/bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
exec "${PYTHON_BIN:-python3}" -m common.fd_performance_run "$@"
