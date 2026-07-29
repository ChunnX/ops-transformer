# block_sparse_attention_grad 确定性方案文档

## 1. 问题背景

block_sparse_attention_grad 算子在 arch22 (Atlas A2) 平台上，当使用 GQA (Grouped Query Attention) 时，
多个 cube 核会并行计算不同 Q head 的 dK/dV。由于同一 group 内的 Q head 共享相同的 K/V，
不同核通过 `Fixpipe` + `SetAtomicAdd` 累加到同一 GM 地址，导致结果非确定性。

## 2. 根因分析

通过对照实验确认：**`Fixpipe` + `SetAtomicAdd` 对同一 GM 地址的多次累加在硬件层面不完全 bit-exact**。

实验方法：将 blockMmad3 的 `isAtomicAdd` 设为 `false`，确定性从 3/7 提升到 7/7（bit-exact），
但精度下降（GQA 场景丢失跨 Q-head 累加数据）。

不确定性产生的两个场景：
1. **跨 Q-head 累加**（GQA, group_size > 1）：不同 Q head 共享同一 KV，写入同一地址
2. **跨 Q-block 累加**（S1 > blockShapeX）：同一 Q head 的不同 Q block 写入同一 KV 地址

## 3. 最终方案：Per-core-per-Q-head-in-group 独立槽位

### 3.1 核心思路

为每个 `(cube_core, qHeadInGroup)` 分配独立的 workspace 槽位：

- **消除跨 Q-head 的 atomicAdd**：不同 Q head 写入不同槽位，由 AIV VecDeterReduce 按序归约
- **保留跨 Q-block 的 atomicAdd**：同一 Q head 的不同 Q block 按固定顺序执行，atomicAdd 确定性有保证
- `isAtomicAdd = true` 仅用于同一 Q head 内的跨 Q block 累加

### 3.2 数据流

```
VecPre:
  ├── 清零 shared workspace (dQ/dK/dV)
  └── 清零 per-core-per-Q-head workspace (ZeroDetWorkspace)

AIC (每个 cube 核):
  ├── gDk/gDv 指向 per-core-per-Q-head slot:
  │     detDkBase + coreIdx * groupSize * dkvSize * sizeof(float)
  │                 + qHeadInGroup * dkvSize * sizeof(float)
  ├── blockMmad3 用 atomicAdd 写入 (仅跨 Q-block 累加)
  └── CrossCoreSetFlag(CUBE2POST)

AIV (CUBE2POST + SyncAll 后):
  ├── VecDeterReduce: 按序读取 aicNum * groupSize 个 slot，FP32 累加到 shared workspace
  └── VecPost: 从 shared workspace 读取，scale + cast，写入 output GM
```

### 3.3 Workspace 布局

```
[libapiSize | sOutSize | dPOutSize | dQOutSize | dKOutSize | dVOutSize | gradSize | detDkWorkspace | detDvWorkspace]
```

- detDkWorkspace = aicNum * groupSize * dkvSize * sizeof(float)
- detDvWorkspace = aicNum * groupSize * dkvSize * sizeof(float)
- 每个 AIC core 的 slot 布局:
  ```
  [core0_qhead0: dkvSize | core0_qhead1: dkvSize | ... | core1_qhead0: dkvSize | ...]
  ```

## 4. 代码修改清单

### 4.1 Tiling (op_host/block_sparse_attention_grad_tiling.h)
- 新增字段: `detDkWorkspaceSize`, `detDvWorkspaceSize`, `aicNumForDet`, `groupSizeForDet`

### 4.2 Tiling 实现 (op_host/arch22/block_sparse_attention_grad_tiling_arch22.cpp)
- CalculateWorkSpace: `detDkWorkspaceSize = aicNum * groupSize * dkvSize * sizeof(float)`
- FillTilingData: 设置新增字段

### 4.3 Kernel (op_kernel/arc22/block_sparse_attention_grad_kernel.h)

**AIC operator()**:
- gDk/gDv 指向 per-core-per-Q-head slot: `detDkBaseCore + qHeadInGroup * dkvSize * sizeof(float)`
- 每次 blockMmad3 调用前通过 `SetGlobalBuffer` 更新 gDk/gDv 基地址
- `qHeadInGroup = preTaskInfo.curHeadIdx % groupSize`

**VecPre**:
- 新增 `ZeroDetWorkspace()`: 清零 aicNum * groupSize * dkvSize 个元素

**VecDeterReduceKernel (新增)**:
- 读取 `totalSlots = aicNum * groupSize` 个 per-core slot
- 按序 FP32 累加到 shared workspace
- 使用 MTE2/V/MTE3 管道同步

**VecDeterReduce (新增)**:
- 调用 VecDeterReduceKernel 对 dk 和 dv 分别归约

**AIV operator()**:
- 在 VecPost 前调用 VecDeterReduce

## 5. 测试结果

### 5.1 测试覆盖

共 25 组 shape，覆盖：

| 维度 | 覆盖范围 |
|------|---------|
| 格式 | BNSD (15组) + TND (10组) |
| 注意力类型 | GQA (18组) + MHA (7组) |
| group_size | 1, 2, 3, 4, 6, 8 |
| Batch | 1, 2, 3, 4, 8 |
| NumHeads | 4, 8, 12, 16, 24 |
| S1 | 64, 128, 256, 384, 512 |
| S2 | 64, 128, 256, 384, 512 |
| sparsity | 0.2, 0.3, 0.5, 0.8, 1.0 |

### 5.2 最终结果

| 指标 | 结果 |
|------|------|
| 精度 (vs CPU golden, rtol=1e-3, atol=1e-3) | **25/25 PASS** |
| 确定性 (bit-exact, torch.equal) | **25/25 PASS** |
| 确定性 (1e-3 容差) | **25/25 PASS** |
| max_diff (跨迭代) | **0.000000** |

### 5.3 测试 Shape 明细

#### GQA BNSD (11组)

| # | B | N | KV | group | S1 | S2 | sparsity | 精度 | 确定性 |
|---|---|---|----|----|----|----|----------|------|--------|
| 1 | 2 | 8 | 2 | 4 | 128 | 128 | 0.5 | PASS | PASS |
| 2 | 4 | 16 | 4 | 4 | 256 | 256 | 0.5 | PASS | PASS |
| 3 | 8 | 12 | 4 | 3 | 512 | 512 | 0.3 | PASS | PASS |
| 4 | 2 | 8 | 2 | 4 | 128 | 256 | 1.0 | PASS | PASS |
| 5 | 1 | 16 | 8 | 2 | 256 | 384 | 0.3 | PASS | PASS |
| 6 | 2 | 12 | 2 | 6 | 128 | 128 | 0.5 | PASS | PASS |
| 7 | 1 | 16 | 2 | 8 | 256 | 256 | 0.5 | PASS | PASS |
| 8 | 2 | 8 | 4 | 2 | 384 | 384 | 0.5 | PASS | PASS |
| 9 | 4 | 8 | 2 | 4 | 64 | 64 | 0.5 | PASS | PASS |
| 10 | 2 | 8 | 2 | 4 | 256 | 256 | 0.2 | PASS | PASS |
| 11 | 2 | 8 | 2 | 4 | 256 | 256 | 0.8 | PASS | PASS |

#### GQA TND (7组)

| # | B | N | KV | group | S1_max | S2_max | sparsity | 精度 | 确定性 |
|---|---|---|----|----|--------|--------|----------|------|--------|
| 12 | 2 | 8 | 2 | 4 | 128 | 128 | 0.5 | PASS | PASS |
| 13 | 4 | 16 | 4 | 4 | 256 | 256 | 0.5 | PASS | PASS |
| 14 | 8 | 24 | 4 | 6 | 512 | 512 | 0.3 | PASS | PASS |
| 15 | 3 | 12 | 4 | 3 | 256 | 384 | 1.0 | PASS | PASS |
| 16 | 2 | 12 | 2 | 6 | 128 | 128 | 0.5 | PASS | PASS |
| 17 | 2 | 8 | 4 | 2 | 256 | 384 | 0.3 | PASS | PASS |
| 18 | 1 | 16 | 2 | 8 | 256 | 256 | 0.5 | PASS | PASS |

#### MHA BNSD (4组)

| # | B | N | KV | group | S1 | S2 | sparsity | 精度 | 确定性 |
|---|---|---|----|----|----|----|----------|------|--------|
| 19 | 2 | 8 | 8 | 1 | 128 | 128 | 0.5 | PASS | PASS |
| 20 | 1 | 4 | 4 | 1 | 256 | 256 | 1.0 | PASS | PASS |
| 21 | 4 | 8 | 8 | 1 | 64 | 64 | 0.5 | PASS | PASS |
| 22 | 2 | 8 | 8 | 1 | 256 | 256 | 0.3 | PASS | PASS |

#### MHA TND (3组)

| # | B | N | KV | group | S1_max | S2_max | sparsity | 精度 | 确定性 |
|---|---|---|----|----|--------|--------|----------|------|--------|
| 23 | 3 | 8 | 8 | 1 | 128 | 128 | 0.5 | PASS | PASS |
| 24 | 2 | 4 | 4 | 1 | 256 | 256 | 1.0 | PASS | PASS |
| 25 | 2 | 8 | 8 | 1 | 256 | 256 | 0.3 | PASS | PASS |

### 5.4 原有测试脚本验证

| 测试脚本 | 格式 | 类型 | 结果 |
|---------|------|------|------|
| bsag_gqa_bnsd.py | BNSD | GQA | 3次迭代 × 3项 = 9/9 True |
| bsag_gqa_tnd_test.py | TND | GQA | 3次迭代 × 3项 = 9/9 True |
| bsag_mha_tnd_test.py | TND | MHA | 10次迭代 × 3项 = 30/30 True |

## 6. CPU Golden 对齐说明

CPU golden 计算需与 NPU 计算路径对齐才能保证精度比对一致：

1. **dS/P 经 FP16 中转**：NPU 将 dS 和 P 存储为 FP16 在 workspace 中，Mmad 读取 FP16 输入并在 FP32 累加。
   CPU golden 需模拟此过程：`dS.to(FP16).to(FP32)`。
2. **Scale 在累加后统一应用**：NPU 在 VecPost 中对累加后的 FP32 结果统一乘 scale，再 cast 到 FP16。
   CPU golden 需先无 scale 累加，最后统一乘 scale。

## 7. 已知限制

1. **Workspace 开销**: per-core-per-Q-head workspace 需要 `aicNum * groupSize * dkvSize * sizeof(float) * 2`。
   对于大规模序列（如 kvTotalSeqlen=4096, kvHeads=8, headDim=128, group=4），20 核约需 3.2GB 额外 workspace。
2. **性能影响**: VecDeterReduce 引入额外的 GM 读取和 V 计算（aicNum * groupSize 次）。
   后续可优化为 ping-pong 流水或分批归约。

## 8. 调试方法

### 8.1 编译安装
```bash
source /opt/nzh/0608/cann-9.1.0-beta.1/set_env.sh
cd /home/lsy/ops-transformer_1787
bash build.sh --ops=block_sparse_attention_grad --soc=ascend910b --pkg
./build/cann-ops-transformer-custom_linux-aarch64.run --quiet
```

### 8.2 运行测试
```bash
export LD_LIBRARY_PATH=/opt/nzh/0608/cann-9.1.0-beta.1/aarch64-linux/lib64:/opt/nzh/0608/cann-9.1.0-beta.1/opp/vendors/custom_transformer/op_api/lib/:${LD_LIBRARY_PATH}
export ASCEND_OPP_PATH=/opt/nzh/0608/cann-9.1.0-beta.1/opp
export ASCEND_HOME_PATH=/opt/nzh/0608/cann-9.1.0-beta.1

cd attention/block_sparse_attention_grad/DevelopTest

# 多 shape 测试 (25组, 5次迭代/组)
python3 multi_shape_test.py

# 原有单 shape 测试
python3 bsag_gqa_bnsd.py
python3 bsag_gqa_tnd_test.py
python3 bsag_mha_tnd_test.py
```

### 8.3 测试脚本说明

| 脚本 | 说明 |
|------|------|
| multi_shape_test.py | 25组shape综合测试，含精度+确定性(bit-exact)验证 |
| bsag_gqa_bnsd.py | GQA BNSD 单shape测试 |
| bsag_gqa_tnd_test.py | GQA TND 单shape测试 |
| bsag_mha_tnd_test.py | MHA TND 单shape测试, 10次迭代 |
| bsa_tnd_test.py | TND 前向测试 (仅CPU golden) |

## 9. 迭代历史

| 版本 | 方案 | 确定性 | 精度 | 问题 |
|------|------|--------|------|------|
| v1 | per-core workspace + atomicAdd | 3/7 bit-exact | 7/7 | Fixpipe atomicAdd 不完全 bit-exact |
| v2 | per-core workspace, isAtomicAdd=false | 7/7 bit-exact | 4/7 | 丢失跨 Q-block 累加 |
| v3 (最终) | per-core-per-Q-head slots + atomicAdd(仅跨Q-block) | **25/25 bit-exact** | **25/25** | 无 |
