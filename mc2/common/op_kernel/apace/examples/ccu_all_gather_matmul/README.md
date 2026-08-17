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

## 输出

程序不打印数值，结果落地 bin 文件：

```
output/0/npu_out.bin   # rank0 的 recvBuf（256KB）
output/1/npu_out.bin   # rank1 的 recvBuf（256KB）
```

Python 校验：

```python
import numpy as np
data = np.fromfile("output/0/npu_out.bin", dtype=np.float16)
assert np.all(data[:65536] == 1.0)   # rank0 数据
assert np.all(data[65536:] == 2.0)   # rank1 数据
```

## 技术说明

Host 侧用 `uint16_t` 承载 FP16 数据（与 apace ST 一致），不使用 `half` 类型，避免 `__truncsfhf2`/`__extendhfsf2` 软浮点符号依赖。

## 编译验证记录

- 2025-08-17：编译通过 + 运行验证 PASS（2 卡 AllGather 通信正确，recvBuf 前半 1.0 后半 2.0）
