# AlltoAll MX Quant Matmul 设计笔记

## 1. 问题背景

MC2 AlltoAll 分布式量化矩阵乘法中，BlockMmad 需要从 AlltoAll 通信输出缓冲区读取 A/scaleA 数据。通信后 AlltoAll 接收缓冲区包含所有 rank 的数据，按 `[rank][rankM][rankKPerRank]` 全局布局。

### 精度问题现象（全1输入 pass，非全1输入 fail）

全1输入时，所有 A/scaleA 数据均为 1，即使读取到错误 rank 的数据，值仍为 1，不暴露问题。非全1输入时，读取错误 rank 的 scaleA/A 数据导致反量化结果错误。

### 根因

AlltoAll 缓冲区为全局布局 `[rankDim][rankM][per_rank_dim]`，rank 间 stride 为 `rankM × per_rank_dim`。但代码中 `aRankStride` 和 `scaleARankStride` 使用了 `tileM × per_rank_dim`（偏小 4 倍），导致 rank 1/2/3 读取到错误位置（实际读取 rank 0 的其他行数据）。

同时，`aOffset`（task 间步长）和 `aGM`（tail 偏移）错误地乘以了 `rankDim`，假设每个 task 在缓冲区中占据 `rankDim × tileM × rankKPerRank` 大小的独立区域，而实际上 task 间步长仅为 `tileM × rankKPerRank`（在 rank0 段内的行偏移）。

## 2. 缓冲区布局分析

### 全局布局（A 和 ScaleA 相同）

A AlltoAll 分多次通信（每个 tile 一组 task），ScaleA AlltoAll 一次完成所有通信。但**最终数据布局相同**：

```
A 缓冲区 (commOutGM_):
┌─────────────────────────────┐
│ rank0: [rankM=2048][rankKPerRank] │  offset = 0
├─────────────────────────────┤
│ rank1: [rankM=2048][rankKPerRank] │  offset = rankM × rankKPerRank
├─────────────────────────────┤
│ rank2: [rankM=2048][rankKPerRank] │  offset = 2 × rankM × rankKPerRank
├─────────────────────────────┤
│ rank3: [rankM=2048][rankKPerRank] │  offset = 3 × rankM × rankKPerRank
└─────────────────────────────┘

ScaleA 缓冲区 (commX1ScaleGM1_):
┌──────────────────────────────────┐
│ rank0: [rankM=2048][scaleKPerRank] │  offset = 0
├──────────────────────────────────┤
│ rank1: [rankM=2048][scaleKPerRank] │  offset = rankM × scaleKPerRank
├──────────────────────────────────┤
│ rank2: [rankM=2048][scaleKPerRank] │  offset = 2 × rankM × scaleKPerRank
├──────────────────────────────────┤
│ rank3: [rankM=2048][scaleKPerRank] │  offset = 3 × rankM × scaleKPerRank
└──────────────────────────────────┘
```

### task 在缓冲区中的位置

每个 task 处理 `tileM=512` 行。task i 在每个 rank 的数据段内偏移 `i × tileM` 行：

- task 0: 各 rank 行 [0, 512)
- task 1: 各 rank 行 [512, 1024)
- task 2: 各 rank 行 [1024, 1536)
- task 3: 各 rank 行 [1536, 2048)

因此：
- `aOffset = tileM × rankKPerRank × sizeof`（task 在 rank0 段内的字节偏移）
- `x1ScaleOffset = tileM × scaleKPerRank × sizeof`（task 在 ScaleA rank0 段内的字节偏移）
- `aRankStride = rankM × rankKPerRank`（rank 间 stride，元素数）
- `scaleARankStride = rankM × scaleKPerRank`（ScaleA rank 间 stride，元素数）

### BlockMmad 中的地址计算

```
aGmBaseAddr_ = aGM + taskIndex × aOffset     → 指向 rank0 段内 task 的起始行
scaleAGmBaseAddr_ = x1Scale + taskIndex × x1ScaleOffset  → 同理

curAGmAddr = aGmBaseAddr_ + rankIdx × aRankStride_       → 指向 rankIdx 段内 task 的起始行
curScaleAGmAddr = scaleAGmBaseAddr_ + rankIdx × scaleARankStride_  → 同理

CopyAInL1: gmARank = [rankM, rankKPerRank] at curAGmAddr
           Slice(mPos_=0, localIter0×kL1) → 读取 task 在该 rank 中的 A tile

CopyScalesInL1: gmScaleARank = [rankM, scaleKPerRank] at curScaleAGmAddr
                Slice(mPos_=0, scaleKStart) → 读取 task 在该 rank 中的 ScaleA tile
```

`mPos_` 由 BlockScheduler 给出，范围 [0, tileM)，因为 problemShape.M = tileM。由于 `aGmBaseAddr_`/`scaleAGmBaseAddr_` 已通过 `aOffset`/`x1ScaleOffset` 跳到 task 的起始行，`mPos_=0` 即读取正确的行。

## 3. 修复方案

### 核心思路

保持原始 `block_mmad_qbmm_mx.h` 的单层 `iter0` 循环结构，`k_ = totalK = rankDim × rankKPerRank`。循环内通过 `rankIdx = iter0 / localKL1Iter_` 和 `localIter0 = iter0 % localKL1Iter_` 跟踪当前 rank。跨 rank 边界时更新 `gmA`/`gmScaleA` 的 GM 地址指针。

假设：`rankKPerRank` 可被 `kL1_` 整除。

### 修改文件清单

| # | 文件 | 改动 |
|---|------|------|
| 1 | `dista/blaze_ext/gemm/policy/dispatch_policy_alltoall.h` | 无需改动 |
| 2 | `dista/blaze_ext/gemm/block/block_mmad_qbmm_mx_alltoall.h` | 基于 blaze 重写，rank 迭代逻辑 |
| 3 | `dista/blaze_ext/gemm/kernel/kernel_qbmm_mx_alltoall.h` | 基于 blaze 重写，增加 AlltoAllParams |
| 4 | `dista/kernel/all_to_all_matmul/all_to_all_mx_quant_matmul_arch35.h` | 修复全局布局偏移 |

### 3.1 block_mmad_qbmm_mx_alltoall.h — 核心改动

基于 `blaze/gemm/block/block_mmad_qbmm_mx.h`，非改动部分与原文件完全一致。

**新增成员变量**：
```cpp
uint32_t rankDim_{1};
uint64_t rankKPerRank_{0};
uint64_t scaleKPerRank_{0};
uint64_t localKL1Iter_{0};
uint64_t aRankStride_{0};        // = rankM × rankKPerRank
uint64_t scaleARankStride_{0};   // = rankM × scaleKPerRank
uint64_t mPos_{0};
__gm__ AType *aGmBaseAddr_{nullptr};
__gm__ fp8_e8m0_t *scaleAGmBaseAddr_{nullptr};
```

**operator() 循环改造**：
```cpp
for (uint64_t iter0 = 0; iter0 < kL1Iter_; ++iter0) {
    uint64_t rankIdx = iter0 / localKL1Iter_;
    uint64_t localIter0 = iter0 % localKL1Iter_;

    if (rankIdx != curRankIdx) {
        curAGmAddr = aGmBaseAddr_ + (rankIdx * aRankStride_ >> sizeShift);
        curScaleAGmAddr = scaleAGmBaseAddr_ + rankIdx * scaleARankStride_;
        curRankIdx = rankIdx;
    }
    // ...
}
```

**CopyAInL1**: 用 `curAGmAddr` 构建 `[rankM, rankKPerRank]` tensor，Slice `(mPos_, localIter0 × kL1_)`

**CopyScalesInL1**: 用 `curScaleAGmAddr` 构建 `[rankM, scaleKPerRank]` tensor，Slice `(mPos_, scaleKStart)`

### 3.2 all_to_all_mx_quant_matmul_arch35.h — 全局布局偏移修复

**ProcessTile 关键修改**：
```cpp
// aOffset: task 间步长（在 rank0 段内偏移 tileM 行）
tileCtx_.aOffset = tileMMultiRankK * sizeof(DTYPE_X1);
// 而非: tileMMultiRankK * rankDim * sizeof(DTYPE_X1)  ← 错误：多乘了 rankDim

// computeRankM: 传全局 rankM，用于 aRankStride/scaleARankStride 计算
tileCtx_.computeRankM = mc2Tiling_.rankM;
// 而非: mc2Tiling_.tileM  ← 错误：aRankStride/scaleARankStride 偏小 4 倍
```

**ProcessTail 同理**：
```cpp
tileCtx_.aGM = commOutGM_ + tileCntMultitileMMultiRankK * sizeof(DTYPE_X1);
// 而非: ... * rankDim * sizeof(DTYPE_X1)  ← 错误

tileCtx_.aOffset = tailMMultiRankK * sizeof(DTYPE_X1);
// 而非: ... * rankDim * sizeof(DTYPE_X1)  ← 错误

tileCtx_.computeRankM = mc2Tiling_.rankM;
// 而非: mc2Tiling_.tailM  ← 错误
```

**效果**：
- `aRankStride = rankM × rankKPerRank = 2048 × 2048 = 4194304`
- `scaleARankStride = rankM × scaleKPerRank = 2048 × scaleKPerRank`
- `aOffset = tileM × rankKPerRank × sizeof`（task 在 rank0 段内的字节偏移）
- `x1ScaleOffset = tileM × scaleKPerRank × sizeof`（task 在 ScaleA rank0 段内的字节偏移，已正确）

## 4. 数据流示意

```
全局 A 缓冲区 (commOutGM_):
┌────────────────────────────────────┐
│ rank0: [2048 rows][rankKPerRank]    │  offset = 0
│   task0: rows [0,512)              │
│   task1: rows [512,1024)           │
│   task2: rows [1024,1536)          │
│   task3: rows [1536,2048)          │
├────────────────────────────────────┤
│ rank1: [2048 rows][rankKPerRank]    │  offset = rankM × rankKPerRank
│   ...                              │
├────────────────────────────────────┤
│ rank2: ...                          │
├────────────────────────────────────┤
│ rank3: ...                          │
└────────────────────────────────────┘

task1 的 aGmBaseAddr_ = commOutGM_ + 512 × rankKPerRank × sizeof
                        (指向 rank0 段内 task1 的起始行)

task1 rank1 的 curAGmAddr = aGmBaseAddr_ + 1 × (2048 × rankKPerRank)
                           = commOutGM_ + (512 + 2048) × rankKPerRank × sizeof
                           (指向 rank1 段内 task1 的起始行 ✓)

BlockMmad operator() 循环:
iter0=0    → rankIdx=0, localIter0=0 → curAGmAddr = base + 0
iter0=N    → rankIdx=1, localIter0=0 → curAGmAddr = base + aRankStride  ← 跨 rank
iter0=2N   → rankIdx=2, localIter0=0 → curAGmAddr = base + 2×aRankStride
iter0=3N   → rankIdx=3, localIter0=0 → curAGmAddr = base + 3×aRankStride
```

## 5. 不改动的文件

- `blaze/gemm/block/block_scheduler_qbmm.h` — 直接 include
- `blaze/gemm/tile/tile_trait.h`, `pad_mx_kl1.h`, `layout_utils.h`, `common_utils.h` — 不改
- `dista/single_core/aiv_comm/all_to_all/ccu/hccl_communicator.h` — 不改
- `dista/kernel/all_to_all_matmul/blaze_mx_quant_matmul.h` — 无需改动（接口兼容）

## 6. 勘误

### 2026-07-02: 缓冲区布局误解（已修正）

曾错误认为 A 缓冲区为 per-task 独立布局 `[rankDim][tileM][rankKPerRank]`，据此：
1. 将 `computeRankM` 从 `rankM` 改为 `tileM`/`tailM`
2. 将 `aOffset` 设为 `rankDim × tileM × rankKPerRank × sizeof`
3. 将 `aGM` tail 偏移乘以 `rankDim`

**实际情况**：A 和 ScaleA 缓冲区都是全局布局 `[rankDim][rankM][per_rank_dim]`。A 虽然分多次通信，但最终数据排布与 ScaleA（一次通信完成）相同。

**修复**：
- `computeRankM` 应传全局 `rankM`（2048），使 `aRankStride`/`scaleARankStride` 正确
- `aOffset` 应为 `tileM × rankKPerRank × sizeof`（task 在 rank0 段内的行偏移），不应乘 `rankDim`
- `aGM` tail 偏移同理，不应乘 `rankDim`

**为什么全1能过**：全局布局下错误 stride 导致读取其他 rank 的数据，但全1时所有 rank 数据值相同，不影响结果。

### 2026-07-03: ScaleA/ScaleB 搬运优化

#### 问题

原实现将 `scaleKL1` 除以 `rankDim`（`l1.scaleKL1 = mmTile.scaleKL1 / rankDim`），导致：
1. `scaleKL1_ / kL1_` 变为 1 或 0，ScaleA 每 kL1 迭代都触发 GM→L1 搬运（而非按复用周期搬运）
2. ScaleB 受连带影响，也被迫每 iteration 搬运
3. 若 `scaleKL1_ / kL1_ == 0`，`localIter0 % 0` 为除零未定义行为
4. MTE2 事务数比预期多 rankDim 倍，MTE2 耗时异常

#### 期望搬运策略

ScaleA 和 ScaleB 的数据布局不同，应采用不同搬运策略：

| 数据 | GM 布局 | 搬运长度 | 搬运时机 | 搬运次数 |
|------|---------|---------|---------|---------|
| ScaleA | `[rankDim][rankM][scaleKPerRank]`（不连续）| `scaleKPerRank` | `localIter0 == 0`（rank 边界）| `rankDim` |
| ScaleB | `[scaleK_total][N]`（连续）| `scaleKL1`（原始 tiling）| `iter0 % (scaleKL1/kL1) == 0` | `kL1Iter/(scaleKL1/kL1)` |

ScaleA 由于 rankDim 份不连续，每次进入新 rank（`localIter0 == 0`）时搬运整份 `scaleKPerRank`。
ScaleB 数据连续，按原始 blaze 的 tiling 复用节奏搬运。

#### 采用方案：选项 A（统一搬运时机）

ScaleA/ScaleB 共享双缓冲 flag 和 `scaleL1BufId`，搬运时机统一为 `localIter0 == 0`。

**原因**：在 `scaleKL1 >= scaleKPerRank`（即 `scaleKL1/kL1 >= localKL1Iter`）假设下，rank 边界频率 ≤ ScaleB 自然搬运频率，不会造成 ScaleB 多余搬运。实现最简，无新增 flag/struct/L1 布局变更。

#### 改动

| # | 文件 | 改动 |
|---|------|------|
| 1 | `blaze_mx_quant_matmul.h` | `l1.scaleKL1 = mmTile.scaleKL1`（恢复原始值，去掉 `/ rankDim`）|
| 2 | `block_mmad Init()` | `scaleAL1OneBuffer_` 基于 `scaleKPerRank_` 而非 `scaleKL1_` |
| 3 | `CopyScalesInL1()` | 条件 `localIter0 == 0`；ScaleA 从 rank 头部搬整份 `scaleKPerRank_`；ScaleB 用全局 K offset |
| 4 | `Iterate()` | `coordScaleKL1 = localIter0 * scaleKL1Len`（去掉 `%` 取模）|
| 5 | `operator()` | flag 释放条件 `localIter0+1 == localKL1Iter_`；调用新增 `iter0` 参数 |

#### 搬运次数验证（rankDim=4, localKL1Iter=4, scaleKL1/kL1=4）

| iter0 | rankIdx | localIter0 | 搬 ScaleA+ScaleB? |
|-------|---------|-----------|-------------------|
| 0 | 0 | 0 | YES |
| 1-3 | 0 | 1-3 | - |
| 4 | 1 | 0 | YES |
| 5-7 | 1 | 1-3 | - |
| 8 | 2 | 0 | YES |
| 9-11 | 2 | 1-3 | - |
| 12 | 3 | 0 | YES |
| 13-15 | 3 | 1-3 | - |

- ScaleA: 4 次 = rankDim
- ScaleB: 4 次 = kL1Iter/(scaleKL1/kL1) = 16/4 = 4（恰好一致，无额外开销）

### 2026-07-03: 代码精简

移除 `fullLoadMode != 0` 死代码（`fullLoadMode` 固定为 0）、移除未使用的 `gmA`/`gmScaleA` 参数、删除自定义 `CeilDiv`、提取重复的 `scaleKPerRank` 计算。总计精简约 108 行。
