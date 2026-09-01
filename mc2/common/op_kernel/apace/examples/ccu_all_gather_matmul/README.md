# CCU AllGather + Matmul 融合算子（第三阶段）

2 卡 AllGather + Blaze Matmul 通算并行流水（FP16 A/B → BF16 C，M=512, tileM=256, commTurn=2, K=N=256）。

## 编译验证指南

### 1. 前置准备

```bash
# 初始化 tensor_api 子模块（首次必须，否则 Blaze 头文件缺失）
cd ~/code/ops-tensor
git submodule update --init --recursive include/tensor_api

# 确认 CANN 环境
echo $ASCEND_HOME_PATH   # 应指向 CANN 安装目录
source $ASCEND_HOME_PATH/set_env.sh
```

> 若 `OPS_TENSOR_ROOT` 不在 `~/code/ops-tensor`，修改 `CMakeLists.txt` L28 的路径。

### 2. 编译 + 运行

```bash
cd ~/code/ops-transformer-0817/mc2/common/op_kernel/apace/examples/ccu_all_gather_matmul
bash run.sh
```

`run.sh` 自动完成：source 环境 → `export HCCL_OP_EXPANSION_MODE=CCU_SCHED` → cmake 编译 → 运行。

### 3. 手动分步（调试用）

```bash
source $ASCEND_HOME_PATH/set_env.sh
export HCCL_OP_EXPANSION_MODE=CCU_SCHED

rm -rf build
cmake -S . -B build
cmake --build build --target ccu_all_gather_matmul --parallel 4

./build/ccu_all_gather_matmul
```

### 4. 精度验证

```python
import numpy as np
raw = np.fromfile("output/0/npu_out.bin", dtype=np.uint16)
c = np.frombuffer((raw.astype(np.uint32) << 16).tobytes(), dtype=np.float32)
# tile0（前 512×256=131072 元素）：A=1.0, C=256×1.0=256.0
assert np.all(c[:131072] == 256.0)
# tile1（后 131072 元素）：A=2.0, C=256×2.0=512.0
assert np.all(c[131072:] == 512.0)
```

### 5. 关键编译参数

| 参数 | 值 | 说明 |
|---|---|---|
| `--npu-arch` | `dav-3510` | Ascend 950 |
| `HCCL_OP_EXPANSION_MODE` | `CCU_SCHED` | CCU 调度模式（运行时必须） |
| 链接库 | `hcomm asccomm_ccu_dataplane c_sec ascendcl acl_rt ascendc_runtime pthread dl platform tiling_api register` | |

## 运行环境要求

- `HCCL_OP_EXPANSION_MODE=CCU_SCHED` 必须设置（`run.sh` 已包含，手动运行需自行 export）
- `ASCEND_HOME_PATH` 指向 CANN 安装目录
- 至少 2 张 NPU 卡（2 rank 单进程多线程）

## 已知问题

- **ops-tensor 路径硬编码**：CMakeLists.txt L28 为 `$ENV{HOME}/code/ops-tensor`，换环境需手动修改
- **tensor_api submodule 权限**：WSL 下 `.git` 可能不可写，需 `sudo`
- **链接 `dl`**：`dlopen` 需显式链接 `libdl`（已配置）

## 风险点验证状态

| 风险 | 说明 | 状态 |
|---|---|---|
| P10 | AIC 写 CCU Variable 地址空间 | ✅ 上板验证通过 |
| P11 | AIC 使用 WriteGmByPassDCache/ReadGmByPassDCache/Nop | ✅ 上板验证通过 |
| P14 | CCU kernel 循环内 channel NotifyRecord/NotifyWait 多次调用 | ✅ 上板验证通过（GroupCopy Loop Entity 需每轮重建） |

## 验证记录

- 2025-08-17：第一阶段 AllGather 通信验证 PASS
- 2025-08-18：第二阶段 AllGather+Matmul 串联验证 PASS（编译修复后）
- 2025-08-18：第三阶段通算并行流水验证 PASS（tile0=256.0, tile1=512.0，P10/P11/P14 全部通过）
