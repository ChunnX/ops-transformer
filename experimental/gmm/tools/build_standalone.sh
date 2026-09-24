#!/usr/bin/env bash
set -eo pipefail
gmm_tools_dir=$(cd "$(dirname "$0")" && pwd)
gmm_build_dir=${1:?Usage: build_standalone.sh BUILD_DIR [ascend910b|ascend910_93]}
source "${ASCEND_HOME_PATH:-/usr/local/Ascend/cann}/set_env.sh"
mkdir -p "$gmm_build_dir"
gmm_build_dir=$(cd "$gmm_build_dir" && pwd)
cmake -S "$gmm_tools_dir/standalone" -B "$gmm_build_dir" -DCMAKE_BUILD_TYPE=Release -DASCEND_COMPUTE_UNIT="${2:-ascend910b}"
cmake --build "$gmm_build_dir" --target all binary -j8
cmake --install "$gmm_build_dir" --prefix "$gmm_build_dir/stage"
cmake --build "$gmm_build_dir" --target package -j8
