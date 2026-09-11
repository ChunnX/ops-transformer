#!/bin/bash
# Run all three MLA operators sequentially through their single-case perf paths.
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec bash "$script_dir/run_fd_perf.sh" --op all "$@"
