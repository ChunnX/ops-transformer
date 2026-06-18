#!/bin/bash
# 集成构建: setup(CCU 写法) + teardown(AIV 写法) -> 一个 launcher。全程在本目录。
set -e
DIR=$(cd $(dirname ${BASH_SOURCE:-$0}); pwd)
cd $DIR
[ -n "$ASCEND_HOME_PATH" ] || { echo "[ERROR] 先 source /home/0617/env/set_env.sh"; exit 1; }
A=$ASCEND_HOME_PATH

echo "[1/2] 编 CCU setup 算子(ccu::) -> setup_ccu/libdispatch_setup_ccu.so"
( cd setup_ccu && make clean >/dev/null 2>&1; make >/dev/null )

echo "[2/2] 编 AIV teardown(<<<>>>) + 集成 launcher(同一个 cmake)"
rm -rf build
cmake -B build -DSOC_VERSION=ascend950pr_9599 -DCMAKE_BUILD_TYPE=Debug -DASCEND_CANN_PACKAGE_PATH=$A >/dev/null
cmake --build build -j16

BIN=$DIR/build/dispatch_integrated
echo ""
echo "[OK] 集成 launcher 编译成功 -> $BIN"
echo "运行(需 device): export LD_LIBRARY_PATH=$DIR/build/lib:$DIR/setup_ccu:\$LD_LIBRARY_PATH && $BIN"

if [[ "$1" == "run" ]]; then
    export LD_LIBRARY_PATH=$DIR/build/lib:$DIR/setup_ccu:$LD_LIBRARY_PATH
    echo "=== 运行集成 launcher ==="
    $BIN
fi
