# CCU AllGather 第一阶段

基于 CCU 通信的 AllGather 通信验证（2 卡，FP16，M=256, K=256）。

核心参考：asc-comm `02_allgather_add`，CCU 通信层逐字节复用。

## 编译

```bash
source $ASCEND_HOME_PATH/set_env.sh
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

## 运行

```bash
export HCCL_OP_EXPANSION_MODE=CCU_SCHED
./build/ccu_all_gather_matmul
```

或一键编译+运行：

```bash
bash run.sh
```

## 期望输出

recvBuf 前 8 个元素 == 1.0（rank0 数据），后 8 个 == 2.0（rank1 数据），两卡一致。
