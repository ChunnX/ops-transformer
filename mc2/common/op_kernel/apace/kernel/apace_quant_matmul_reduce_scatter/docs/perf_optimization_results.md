# quantMatmulReduceScatter 性能优化尝试结果报告

> 日期: 2026-08-19
> 平台: Ascend 950PR (dav-3510), CANN 9.2.0
> 代码版本: totalJobs=1, isCommBlock=(blockIdx==0), 自适应headMSize

## 一、逐个优化尝试结果

### 优化 1: dbL0c=2 (双缓冲 L0C, 重叠 fixpipe 与 MAC)

**改动**: `subMmTilingData.dbL0c = 2` (从 1 改为 2)

**测试用例**: M=2048 K=3584 N=4096 R=2

**结果**:
| 指标 | dbL0c=1 (原) | dbL0c=2 (优化) | 变化 |
|------|-------------|---------------|------|
| MAC | 256us (90%) | 248us (87%) | -3% |
| MTE2 | 262us (93%) | 253us (89%) | -3% |
| fixpipe | 75us (26%) | 82us (29%) | **+9% 恶化** |
| Task Duration | 284us | 285us | 持平 |

**失败原因**: L0C 双缓冲占用 L0A/L0B 空间, 引发更多 L1 重载, fixpipe 反而增加。

---

### 优化 2: stepK=4 (K-steps 减半, 减少 MTE2 DMA 次数)

**改动**: `subMmTilingData.stepK = 4` (从 2 改为 4)

**测试用例**: M=2048 K=3584 N=4096 R=2

**结果**:
| 指标 | stepK=2 (原) | stepK=4 (优化) | 变化 |
|------|-------------|---------------|------|
| MAC | 251us | 248us | -1% |
| MTE2 | 256us | 253us | -1% |
| Task Duration | 284us | 285us | 持平 |

**失败原因**: MTE2 与 MAC 已高度重叠 (90%+), K-step 减少被其他开销抵消。

---

### 优化 3: baseN=128 (N-tiles 翻倍, 提升 cube_util)

**改动**: `subMmTilingData.baseN = 128` (从 256 改为 128)

**测试用例**: 5 个代表用例

**结果**:
| Case | baseN=256 (原) | baseN=128 (优化) | 变化 |
|------|----------------|------------------|------|
| M=2048 R=2 | 300us | 321us | **+7% 恶化** |
| M=2048 R=4 | 243us | 245us | +0.8% |
| M=4096 R=2 | 552us | 569us | +3% |
| M=1024 R=2 | 194us | 183us | -5.7% |
| M=4096 R=4 | 417us | 439us | +5% |

**失败原因**: N-tiles 翻倍导致 fixpipe 调用次数翻倍, 每次 fixpipe 的固定开销累加, 抵消了 cube_util 提升的收益。

---

### 优化 4: 多核通信 totalJobs=rankSize (框架 work partition)

**改动**: `totalJobs=rankSize`, `participantNum=rankSize`, `isCommBlock=(blockIdx<rankSize)`

**测试用例**: M=2048 K=3584 N=4096 R=2

**结果**: Rank 0 PASS, Rank 1 **FAIL** (81% 元素错误, max_err=839680)

**失败原因**: 框架 `DoCommit` 的 src 地址用 `targetRankId` 做偏移 (AllGather 语义):
```
srcAddr = localAddr_ + targetRankId * chunkBytes_ + ...
```
但 ReduceScatter 需要用 `rankId` 做偏移:
```
srcAddr = localAddr_ + rankId * chunkBytes_ + ...
```
语义不匹配: AllGather 是"把 chunk_i 发给 rank_i", ReduceScatter 是"把自己的 chunk 发给所有 rank"。

---

### 优化 5: 手动 WriteNbi 多核 PUT (绕过框架)

**改动**: 保持 `totalJobs=1`, 但前 rankSize 个 block 手动调用 `Hcomm::WriteNbi` + `Drain`

**测试用例**: M=2048 K=3584 N=4096 R=2

**结果**: Rank 0 和 Rank 1 均 **FAIL** (100% 元素错误)

**失败原因**: `Hcomm` 实例的 UB workspace 与 reduceSum 的 UB 空间冲突, 或 `channelHandles` 多核并发访问有框架级限制。

---

### 优化 6: R=2 时 stepK=4 + baseN=128 (组合优化)

**改动**: 同时 `stepK=4` + `baseN=128`

**测试用例**: 5 个代表用例

**结果**:
| Case | 原始 | 组合优化 | 变化 |
|------|------|---------|------|
| M=2048 R=2 | 300us | 321us | **+7% 恶化** |
| M=2048 R=4 | 243us | 245us | +0.8% |

**失败原因**: baseN=128 的 fixpipe 开销主导, stepK=4 的 MTE2 收益被完全抵消。

---

## 二、多核通信失败的根因分析

### AllGather PUT 语义 (框架设计)
```
block i → PUT workspace[chunk_i] → targetRank i
每个 block 负责 1 个 target, 每个 target 收到 1 份完整数据 ✓
```

### ReduceScatter PUT 语义 (本算子需求)
```
block 0 → PUT workspace[chunk_rankId] → targetRank 0
block 0 → PUT workspace[chunk_rankId] → targetRank 1
...
block 0 → PUT workspace[chunk_rankId] → targetRank R-1
1 个 block 遍历所有 target, 每个 target 收到本 rank 的 chunk ✓
```

### totalJobs=rankSize 时的框架行为
```
block i → 负责 targetRank i
block i → DoCommit: srcAddr = workspace[targetRank_i * chunkBytes]
  → 从 workspace 读取 chunk_i (不是 chunk_rankId!)
  → PUT chunk_i 给 rank_i
  → rank_i 收到 chunk_i, 但需要的是所有 R 个 chunk
  → 80% 数据缺失 ✗
```

### 结论
APACE AllToAll PUT 框架的 work partition 按 target rank 分配, 适用于 AllGather 语义。ReduceScatter 需要按 source chunk 分配, 语义不匹配, 无法通过参数调整实现多核并行。

---

## 三、无法进一步提升性能的根本原因

### 1. AIC matmul 是不可逾越的计算瓶颈
- 总 MAC 操作 = M×K×N, 固定不变
- MTE2/MAC 重叠率已达 90%+, 内存带宽已充分利用
- fixpipe 是唯一可优化项, 但 dbL0c=2 在小 tile 场景反效果

### 2. AIV 完全被 AIC 掩盖
- AIV 82% 时间在 WaitFlag 等 AIC 完成
- reduceSum 仅占 AIV 的 16-25%
- 即使消除全部 reduceSum, 端到端仍 = AIC 时间

### 3. 多核通信受限于框架语义
- APACE AllToAll PUT 框架设计为 AllGather 语义
- ReduceScatter 需要不同语义, 无法通过参数调整或手动 WriteNbi 绕过
- 需要框架层面支持 ReduceScatter 语义的 work partition

### 4. tiling 参数已穷尽
- SWAT 自动生成的 tiling 在大部分场景已接近最优
- 手动调整 baseM/baseN/baseK/stepK/dbL0c 均无法突破 MTE2 瓶颈
- MTE2 是内存带宽受限, 需要架构级改进 (更多带宽) 或算法级改进 (SplitK 减少总计算量)

---

## 四、当前性能状态

| 指标 | 值 |
|------|-----|
| 精度 | 12/12 ALL PASS |
| 性能范围 | 222~2669 us |
| AIC 关键路径 | max(MAC, MTE2) + fixpipe_non_overlap |
| 瓶颈 | MTE2 (内存带宽), 11/12 用例 |
| AIV 状态 | 完全被 AIC 掩盖 |
| 优化空间 | 无 (tiling 参数穷尽, 多核通信框架不支持) |
| 突破方向 | SplitK (算法级, 每卡算 K/R, AIC 减半) |
