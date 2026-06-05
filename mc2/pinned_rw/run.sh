#!/bin/bash
#
# 验证 Device 能否读写 Host-pinned mapped 内存
#
# 用法:
#   ./run.sh              # 默认 dav-3510 (950PR / 950DT / 910PR)
#   ./run.sh dav-2201     # 910B / 910C / 910_93

set -e

ARCH="${1:-dav-3510}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CANN_PATH="${CANN_PATH:-/home/developer/Ascend/cann-9.0.0}"
if [ -f "${CANN_PATH}/set_env.sh" ]; then
    source "${CANN_PATH}/set_env.sh"
fi

mkdir -p output

echo "=== Building ($ARCH) ==="
mkdir -p build && cd build
cmake -DCMAKE_ASC_ARCHITECTURES="$ARCH" .. && make -j

echo "=== Running on NPU ==="
./demo

echo ""
echo "=== Result ==="
echo "PASS = Device can read/write host-pinned mapped memory via RegisterV2(MAPPED)"
echo "FAIL = Device cannot access host-pinned mapped memory on this chip/CANN"