# RadixTopK 迁移分析与性能报告

## 1. 项目概述

将 math 仓的 RadixTopK 算子迁移到 lightning_indexer，替代原有的 Sort+MergeSort 路径，验证精度和性能。

### 环境
- **NPU**: Ascend 910B3, NPU 1
- **CANN**: cann-9.0.0, `source /home/tjj/setup_env.sh`
- **构建**: `bash build.sh --pkg --soc=ascend910b --ops=lightning_indexer -j16`
- **安装路径**: `/home/tjj/0623/cann-9.0.0/opp/vendors/custom_transformer/vendors/custom_transformer/`
- **ref 仓库**: `/home/tjj/ops-transformer-ref` (Sort 路径)

### 测试 Shape

| 参数 | 小 shape | 大 shape |
|------|---------|---------|
| batch_size | 18 | 1 |
| q_seq | 3 | 8192 |
| k_seq | 3072 | 8192 |
| q_head_num | 16 | 64 |
| k_head_num | 1 | 1 |
| head_dim | 128 | 128 |
| sparse_count (kValue) | 2048 | 512 |
| sparse_mode | 3 | 3 |
| layout_query | BSND | TND |
| layout_key | PA_BSND | PA_BSND |
| radixTotalBatch | 54 | 8192 |
| sortLen (per batch) | 3072 | 8192 |

---

## 2. 架构对比

### 分核策略

| | math/BSA (tile-split) | lightning_indexer (round-robin) |
|---|---|---|
| **策略** | 所有核处理同一 batch，切分 sortLen 的 tiles | 每核处理不同 batch（query token） |
| **batch 并行** | 串行（for batchId） | 并行（batchId += aivNum） |
| **tile 并行** | 并行（tiles / coreNum） | 串行（单核处理全部 tiles） |
| **SyncAll** | 16 次/batch | 0 次 |
| **每核 V_S sync** | 少（tileNum/40 × 3 × 8） | 多（tileNum × 3 × 8 × batches/core） |
| **适用场景** | batch 少、sortLen 大 | batch 多、sortLen 适中 |

### 为什么不一致

lightning_indexer 大 shape 有 8192 个 batch（query token），如果用 math/BSA 的 tile-split：
```
8192 batch × 16 SyncAll/batch = 131,072 次 SyncAll
```
即使每次 SyncAll 仅 1μs，也要 ~131ms，远超当前 7.5ms。因此 round-robin 是唯一可行方案。

### RadixTopK "batch" 的定义

```
radixTotalBatch = batch_size × q_seq × k_head_num
```

每个 "batch" 是一个 query token，需要独立对 sortLen（=k_seq）个 score 值选出 top-kValue 个索引。

### math 仓分核策略代码段

**Tiling 阶段：tile 切分计算** — `radix_top_k_tiling.cpp:207-221`

```cpp
void CalcTileDistribution(dataNum, formerTileLen) {
    totalTileNum = CeilDiv(dataNum, formerTileLen);           // 总 tile 数
    formerTileNum = CeilDiv(totalTileNum, totalCoreNum_);     // 主核 tile 数（多1个）
    tailTileNum = totalTileNum / totalCoreNum_;               // 尾核 tile 数
    formerCoreNum = totalTileNum % totalCoreNum_;             // 主核数量
    tailCoreNum = tailTileNum ? (totalCoreNum_ - formerCoreNum) : 0;
    tailTileLen = dataNum - formerTileLen * (totalProcessedTiles - 1);  // 最后 tile 长度
}
```

**Kernel 阶段：读取分核参数** — `radix_top_k_common.h:200-220`

```cpp
void InitBaseParams() {
    blockIdx_ = GetBlockIdx();
    coreNum_ = tiling_.coreNum;
    formerCoreNum_ = tiling_.formerCoreNum;
    // 当前核是主核还是尾核
    tileNum_ = blockIdx_ < formerCoreNum_ ? formerTileNum_ : tailTileNum_;
    // 最后一个核处理最后一个 tile（可能不满）
    tailTileLen_ = blockIdx_ != coreNum_ - 1 ? formerTileLen_ : tiling_.tailTileLen;
}
```

**Kernel 阶段：计算 GM 偏移** — `radix_top_k_common.h:643-656`

```cpp
uint64_t CalcBlockOffset(batchId) {
    uint64_t blockOffset = batchId * sortLen_;  // batch 偏移
    if (blockIdx_ < formerCoreNum_) {
        blockOffset += blockIdx_ * formerTileNum_ * formerTileLen_;
    } else {
        blockOffset += formerCoreNum_ * formerTileNum_ * formerTileLen_ +
                      (blockIdx_ - formerCoreNum_) * tailTileNum_ * formerTileLen_;
    }
    return blockOffset;  // 本核在 GM 中的起始读取位置
}
```

**Kernel 阶段：batch 串行 + SyncAll** — `radix_top_k_ub.h:78-85`

```cpp
void Process() {
    for (int32_t batchId = 0; batchId < batch_; batchId++) {
        SubProcess(batchId);  // 所有核处理同一 batch
        SyncAll();            // 同步后继续下一个 batch
    }
}
```

**Kernel 阶段：每 round 的 SyncAll** — `radix_top_k_ub.h:263-305`

```cpp
void SubProcess(batchId) {
    for (int32_t roundId = round_ - 1; roundId >= 0; roundId--) {
        // 每核处理自己的 tiles [0, tileNum_)
        for (tileId = 0; tileId < tileNum_; tileId++) {
            CopyIn -> Twiddle -> CalcCumsumHistogram  // 统计直方图
        }
        CopyOut2Ws(globalHist -> GM);  // 写入 GM workspace
        SyncAll();                     // <- 同步1: histogram 可见
        Update(roundId);               // <- 跨核归约 + 找 boundary
        SyncAll();                     // <- 同步2: boundary 共识
    }
    SyncAll();
    TileTopK();  // 提取结果
}
```

### BSA 分核策略代码段

**Tile 切分** — `bsa_radix_topk_service.h:178-197`

```cpp
// InitParams() 中，按 GetBlockIdx() 切分 tiles
uint32_t coreNum = constInfo.aivNum;
uint64_t baseTilesPerCore = totalTileNum_ / coreNum;
uint64_t extraCores = totalTileNum_ % coreNum;
this->formerCoreNum_ = extraCores;
this->tailCoreNum_ = coreNum - extraCores;
this->formerTileNum_ = baseTilesPerCore + (extraCores > 0 ? 1 : 0);
this->tailTileNum_ = baseTilesPerCore;

uint64_t myTileNum = (GetBlockIdx() < formerCoreNum_) ? formerTileNum_ : tailTileNum_;
this->tileStartId_ = (GetBlockIdx() < formerCoreNum_)
    ? GetBlockIdx() * formerTileNum_
    : formerCoreNum_ * formerTileNum_ + (GetBlockIdx() - formerCoreNum_) * tailTileNum_;
```

**Batch 串行 + SyncAll** — `bsa_select_block_mask_base.h:258-310`

```cpp
// Process() 中，所有核同步遍历每个 batch
for (uint32_t batchIdx = 0; batchIdx < constInfo.batchSize; batchIdx++) {
    for (uint32_t headIdx = 0; headIdx < constInfo.numHeads; headIdx++) {
        // ... matmul/softmax ...
        SyncAll();
        radixTopKService.ProcessRadixTopKAndWriteMask(batchIdx, headIdx);
        SyncAll();  // <- 所有核完成当前 batch 后才继续
    }
}
```

**每 round 的 SyncAll** — `bsa_radix_topk_service.h:370-377`

```cpp
// ProcessRadixTopKAndWriteMask() 的 radix loop 中
for (int32_t roundId = round_ - 1; roundId >= 0; roundId--) {
    // ... 每核处理自己的 tiles [tileStartId_, tileStartId_+tileNum_) ...
    SyncAll();  // <- histogram 写入 GM 后同步
    earlyExit = Update(roundId);  // <- 跨核归约 globalHist
    SyncAll();  // <- boundary consensus
}
SyncAll();  // <- 最终同步
```

**跨核 globalHist 归约** — `bsa_radix_topk_service.h:832-854`

```cpp
// ReduceGlobalHist(): 从 GM 读入所有核的 globalHist，向量化 Add 归约
CopyInGlobalHist(globalHist);  // MTE2: GM->UB, 读取 coreNum×4 个 int32
// 向量化二分归约
while (remain > coresInBlock) {
    Add(globalHist, globalHist, globalHist[splitPoint * numValue_], ...);
    remain = splitPoint;
    splitPoint /= 2;
}
VToSSync();
// 标量求和剩余部分
for (int32_t i = 0; i < remain; i++) {
    sum += globalHist.GetValue(binMask + i * numValue_);
}
```

**跨核 coreTopK offset** — `bsa_radix_topk_service.h:894-907`

```cpp
// ComputeCumSumPrev(): 读前驱核的 boundaryBinCumSum，算本核输出偏移
DataCopyPad(resTensor, boundaryBinCumSumGm_, ...);  // GM->UB
MTE2ToSSync();
cumSumValuePrev = 0;
for (int32_t coreId = 0; coreId < GetBlockIdx(); coreId++) {
    cumSumValuePrev += resTensor.GetValue(coreId);  // 前缀和
}
```

### lightning_indexer 分核策略代码段

**Round-robin batch 分配** — `lightning_indexer_kernel.h:734-770`

```cpp
// ProcessRadixTopK: 每核处理不同的 batch（round-robin）
for (uint32_t batchId = aivIdx; batchId < totalBatch; batchId += aivTotal) {
    // 单核独立处理一个 batch 的完整 RadixTopK
    InlineRadixTopK(pipe, scorePtr, idxPtr, ...,
                    0,      // coreId = 0（单核）
                    1,      // coreNum = 1
                    batchId);
}
```

**单核 tile 切分** — `radix_topk_inline.h:42-48`

```cpp
// 单核处理全部 tiles（coreNum=1）
uint32_t totalTiles = (sortLen + tileLen - 1) / tileLen;
uint32_t tileNum_ = totalTiles;  // 全部 tiles 归一个核
uint32_t tailTileLen_ = sortLen - (tileNum_ - 1) * tileLen;
```

**ProcessMain AIC/AIV 分核** — `lightning_indexer_kernel.h:293-366`

```cpp
// SplitCore: 按 baseBlockNum 切分 AIC 核
uint32_t totalBlockNum = GetTotalBaseBlockNum();
uint32_t minBlockPerCore = totalBlockNum / coreNum;
uint32_t deal1MoreBlockCoreNum = totalBlockNum % coreNum;
// 每个核分配 [bN2Start, bN2End] × [gS1Start, gS1End] × [s2Start, s2End]
```

**AIC/AIV 配对** — `lightning_indexer_kernel.h:718-731`

```cpp
if ASCEND_IS_AIC {
    CrossCoreWaitFlag(constInfo.syncV1C1);
    matmulService.ComputeMm1(runInfo);      // AIC 做 matmul
    CrossCoreSetFlag<..., PIPE_FIX>(constInfo.syncC1V1);  // 通知 AIV
} else {
    CrossCoreWaitFlag(constInfo.syncC1V1);
    vectorService.ProcessVec(runInfo);       // AIV 做 reduce + scoreB16Gm 写入
    CrossCoreSetFlag<..., PIPE_MTE2>(constInfo.syncV1C1);  // 通知 AIC
}
```

**scoreB16Gm 行索引** — `lightning_indexer_service_vector.h:403-405`

```cpp
// 按 cuS1Idx 确定 scoreB16Gm 行（对应 ProcessRadixTopK 的 batchId）
uint64_t scoreRowIdx = static_cast<uint64_t>(info.indiceOutOffset) / constInfo_.sparseCount + cuS1Idx;
uint64_t scoreOffset = scoreRowIdx * constInfo_.radixSortLenAlign + cuBaseS2Idx;
```

### 三家分核策略对比总结

| | math 仓 | BSA | lightning_indexer |
|---|---|---|---|
| **tile 切分** | tiling 预计算 `formerCoreNum`/`formerTileNum` | `InitParams` 按 `GetBlockIdx` 分 tiles | `InlineRadixTopK` 中 `tileNum_ = totalTiles`（全归一个核） |
| **batch 遍历** | `for (batchId)` 串行 + SyncAll | `for (batchIdx)` 串行 + SyncAll | `for (batchId = aivIdx; ...; += aivTotal)` round-robin |
| **核间同步** | 每 round 2 次 SyncAll + GM 归约 | 每 round 2 次 SyncAll + GM 归约 | 0 次 SyncAll（每核独立） |
| **coreId/coreNum** | `GetBlockIdx()` / `coreNum_` | `GetBlockIdx()` / `aivNum` | `0` / `1`（传给 InlineRadixTopK） |
| **globalHist 归约** | GM workspace + 向量化 Add | GM workspace + 向量化 Add | 无（单核标量累加器 `gh[4]`） |
| **coreTopK offset** | GM workspace + 前缀和 | GM workspace + 前缀和 | 无（单核 `writeOffset` 标量累加） |

---

## 3. 910B3 架构限制

| 限制 | 影响 |
|------|------|
| `Not`/`ShiftRight` 只支持 int16 | TwiddleInB16 用 int16 |
| `Compares` 不支持 bfloat16_t | 需 Cast 到 float 后比较 |
| `Cast(float→int32, CAST_RINT)` | bitwise reinterpret |
| `Adds` 不支持 int32 tensor-tensor | 用 `Add` |
| **V op 2048-element 限制** | GatherMask/Compares 对 float/int32 最多 2048 元素/call，限制 tileLen ≤ 2048 |
| UB→UB DataCopy 2048-element 限制 | RDX_UB2UB macro 分段拷贝 |

---

## 4. 优化历程

### 性能进展

| 版本 | 大 shape | 小 shape | vs Sort 大 | vs Sort 小 |
|------|---------|---------|-----------|-----------|
| 原始 a606ae9b2 | 25.0ms | 0.84ms | +276% | +53% |
| P1: tileLen 512→2048 | 22.1ms | 0.60ms | +232% | +9% |
| P2: 向量化 extraction | 19.9ms | 0.55ms | +199% | +0% |
| P3: 消除冗余 sync | 19.8ms | 0.55ms | +198% | +0% |
| **+ debug cleanup + Mins** | **7.53ms** | **0.47ms** | **+13%** | **+7%** |

### P1: tileLen 512→2048
- **改动**: `kernel.h` 中 `tileLen = (sortLen < 2048) ? sortLen : 2048`
- **效果**: sortLen=8192 时 tile 数 16→4，减少 V op launch 开销
- **限制**: tileLen=2560 精度崩塌（V op 2048-element 硬限制）

### P2: 向量化 extraction index 写入
- **改动**: `radix_topk_inline.h` 中用 `Adds + DataCopyPad` 替代 `indexGm.SetValue` 逐元素写
- **参考**: math 仓 `radix_top_k_common.h:460-485` 的 `SubTopKAndCopyOut`
- **效果**: extraction 从 ~60μs/batch 降到 ~1μs/batch

### P3: 消除 round loop 冗余 sync
- **改动**: 将 `tileHist.SetValue` 移入 bin loop（BSA 模式），消除 `binCounts` 数组和 3 对冗余 V_S/S_V sync
- **效果**: 6→3 sync pairs/tile/round（50% 减少），实测收益小（sync 开销低于预期）

### P4: 移除 debug 残留 + 向量化 +1e30 替换
- **问题**: debug 代码 `Duplicate(reduceOutInner, -1e30)` 覆盖了正常 reduce 结果
- **改动**:
  1. 移除 debug `Duplicate(reduceOutInner, -1e30)` 覆盖
  2. 移除 scoreB16Gm init（-1e30 填充循环 + SyncAll）
  3. 移除 ProcessMain 后 `PipeBarrier<PIPE_ALL>`
  4. 用 `Mins(reduceOutInner, reduceOutInner, 1.0e29f)` 向量化替代 512 次 scalar GetValue/SetValue loop
- **效果**: AIC 117.9μs→25.5μs（-78%），AIV scalar 78.5μs→17.0μs（-78%）

---

## 5. msprof 性能数据

### 小 shape 对比（优化后）

| 指标 | RadixTopK | Sort (ref) | 差值 | 差异 |
|------|-----------|------------|------|------|
| **Task Duration** | **61.8μs** | **35.9μs** | +25.9μs | +72% |
| AIC time | 25.5μs | 27.4μs | -1.9μs | -7% |
| AIV time | 50.7μs | 32.3μs | +18.4μs | +57% |
| AIV scalar | 17.0μs | 11.4μs | +5.5μs | +48% |
| AIV vec | 16.8μs | 13.7μs | +3.1μs | +23% |
| AIV mte3 | 2.1μs | 0.8μs | +1.3μs | +157% |
| cube_util | 41.2% | 76.2% | -35.0% | -46% |

### 大 shape 对比（优化后）

| 指标 | RadixTopK | Sort (ref) | 差值 | 差异 |
|------|-----------|------------|------|------|
| **Task Duration** | **6.82ms** | **5.90ms** | +0.92ms | +15.5% |
| AIC time | 3.33ms | 5.28ms | -1.95ms | -37.0% |
| AIV time | 6.78ms | 5.90ms | +0.88ms | +14.9% |
| AIV scalar | 2.55ms | 1.28ms | +1.27ms | +99.8% |
| AIV vec | 4.78ms | 5.05ms | -0.28ms | -5.5% |
| AIV mte3 | 0.42ms | 0.03ms | +0.39ms | +1219% |
| cube_util | 48.8% | 89.4% | -40.6% | -45% |

### 关键发现

1. **AIC 时间 Radix 比 Sort 快 37%**（大 shape）— Sort 路径的 AIC 需要额外做 Sort+MergeSort
2. **AIV vec Radix 比 Sort 快 5.5%**（大 shape）— RadixTopK 的 V 计算（Cast+Twiddle+Compares+GatherMask）比硬件排序指令更高效
3. **AIV scalar 是唯一显著瓶颈**（+99.8%）— RadixTopK 的 V_S_SYNC + GetValue/SetValue 标量操作
4. **cube_utilization 仅 48.8%** — AIV scalar 阻塞了 AIC pipeline

---

## 6. V_S Sync 分析

### 为什么需要 V_S sync

910B 架构上 S pipe（Scalar）和 V pipe（Vector）是独立执行单元。`GatherMask` 的 `rsvdCnt`（匹配元素计数）是 V pipe 硬件寄存器输出，只能被 S pipe 读取，必须通过 `V_S` event sync。

### 三家实现对比

math 仓、BSA、我们的实现在 `GatherMask` 后都有 `VToSSync()` — **这是硬件限制，无法消除**。

```cpp
// 所有三个实现都有这个模式：
GatherMask(..., rsvdCnt);   // V pipe 产生 rsvdCnt
VToSSync();                  // V→S：S pipe 读 rsvdCnt
cumSumOne += rsvdCnt;        // S pipe 标量累加
tileHist.SetValue(...);      // S pipe 写 UB
```

### AIV scalar 差异的真正来源

不是 V_S sync 数量（三家相同），而是**每核处理的 tile×batch 总量不同**：

| | BSA | 我们 | math 仓 |
|---|---|---|---|
| 架构 | 多核切分 tiles | round-robin 切分 batches | 多核切分 tiles |
| 每核 V_S sync 总量 | (tileNum/40)×3×8 = ~3 | tileNum×3×8×205 = ~5535 | 同 BSA |

### tileHist 读取方式差异

math 仓 `HandleLastRoundBoundary` 用 `ReduceSum`（V pipe）替代逐 tile `GetValue`：
```cpp
// math: 向量化求和
Cast(tileHistFp32, resTensor, CAST_NONE, tileNum_);
ReduceSum(tileHistFp32, tileHistFp32, tileHistFp32, curSegLen);
VToSSync();  // 只需 1 次
```

但最后一轮分配 `remainCoreBoundaryNum` 时 math 仓也用 `GetValue`。区别在于 math 仓 `tileNum/coreNum` 很小（大 shape ~0.4 tile/核），几乎不执行。

---

## 7. 精度问题

### 已修复的 bug

| Bug | 根因 | 修复 |
|-----|------|------|
| TND radixTotalBatch 错误 | `bSize × s2Size × n2Size` 应为 `query.shape[0] × n2Size` | tiling.cpp 从 query shape dim(0) 计算 |
| TND batch mapping 错误 | `bIdx=0` 恒定，`s1Idx=batchId`（全局索引） | 用 `actualSeqLengthsGmQ` 做行→batch 映射 |
| isAccumSeqS1/S2 未初始化 | BSND 下未定义值导致 bIdx>0 的 actS1=0 | 显式 `false for BSND, true for TND` |
| ProcessVec radixCuRealAcSeq 错误 | `actS2Size` 代替 `actS1Size` | 改为 `actS2Size - (actS1Size - cuS1Idx) + 1` |
| +1e30 标量替换覆盖正常数据 | debug `Duplicate(reduceOutInner, -1e30)` 残留 | 移除 debug 代码 |
| 非确定性输出 | scoreB16Gm 未初始化位置含 GM 垃圾 | 已用 Mins 向量化替代（移除 init 后需验证） |

### 已知未修复问题

**预存 bug: first-s1Idx-per-AIV 失败**

- **现象**: `actS1 < s1BaseSize` 时每个 AIV 的第一个 s1Idx 的 scoreB16Gm 写入不生效
- **根因**: ProcessVec 的 DataCopyPad (MTE3) 到 scoreB16Gm 的写入对第一个 s1Idx 不生效（pipeline 冷启动）
- **影响**: 原始稳定版 a606ae9b2 也有此 bug，非本次引入
- **验证**: -1e30 填充测试确认 DataCopyPad 未写入 positions 0-255
- **状态**: 添加 V→MTE3 sync、PipeBarrier<PIPE_ALL>、MTE3→V sync 均未修复

### 泛化测试结果（10 shape, sm=3 only）

| # | B | Q | K | H | sc | layout | 结果 | 精度 |
|---|---|---|---|---|---|--------|------|------|
| 1 | 1 | 1 | 2048 | 4 | 128 | BSND | PASS | 100% |
| 2 | 18 | 8 | 128 | 1 | 64 | BSND | FAIL | 85.2% |
| 3 | 4 | 3 | 2048 | 1 | 512 | TND | PASS | 83.3% |
| 4 | 1 | 1 | 1024 | 8 | 64 | BSND | PASS | 100% |
| 5 | 4 | 8 | 3072 | 4 | 256 | BSND | PASS | 100% |
| 6 | 1 | 16 | 512 | 64 | 256 | BSND | PASS | 100% |
| 7 | 8 | 3 | 1024 | 8 | 64 | BSND | PASS | 100% |
| 8 | 8 | 32 | 8192 | 4 | 256 | BSND | PASS | 100% |
| 9 | 1 | 1 | 512 | 4 | 256 | BSND | PASS | 100% |
| 10 | 8 | 16 | 8192 | 64 | 256 | BSND | FAIL | 95.7% |

8 PASS / 2 FAIL。FAIL 原因：小 K + 小 sparse_count 的 bf16 tie-breaking（case 2），预存 first-s1Idx bug（case 10）。

---

## 8. 输出排序

RadixTopK 输出**未排序**，按 tile 顺序 + tile 内 GT-then-EQ 顺序排列。与 math 仓一致（math 仓 `sorted` 标志注释为 "当前仅支持 false"）。精度比较脚本在比较前对两边都做 `np.sort()`，不要求输出排序。

---

## 9. 相关文件

| 文件 | 说明 |
|------|------|
| `attention/lightning_indexer/op_kernel/arch22/radix_topk_inline.h` | RadixTopK 主逻辑（InlineRadixTopK 函数） |
| `attention/lightning_indexer/op_kernel/arch22/lightning_indexer_kernel.h` | ProcessRadixTopK 调度，TND batch mapping |
| `attention/lightning_indexer/op_kernel/arch22/lightning_indexer_service_vector.h` | ProcessVec scoreB16Gm 写入，+1e30 替换 |
| `attention/lightning_indexer/op_host/lightning_indexer_tiling.cpp` | radixTotalBatch 计算 |
| `C:\Users\Administrator\ops-math\math\radix_top_k\op_kernel\radix_top_k_ub.h` | math 仓 UB 变体参考 |
| `C:\Users\Administrator\ops-math\math\radix_top_k\op_kernel\radix_top_k_common.h` | math 仓基类（SubTopKAndCopyOut, CopyOutResult） |
| `C:\Users\Administrator\ops-transformer\attention\bsa_select_block_mask\op_kernel\bsa_radix_topk_service.h` | BSA 参考（tile-split 多核，GM workspace） |

---

## 10. 下一步优化方向

| 方向 | 预期收益 | 难度 | 说明 |
|------|---------|------|------|
| 修复 first-s1Idx bug | 精度修复 | 中 | 需深入 AscendC MTE3 pipeline 行为 |
| 混合分核策略 | 小 shape 性能提升 | 高 | totalBatch < aivNum 时用 tile-split |
| 减少 radix rounds | V_S sync 总量降低 | 中 | early exit 已有，可探索动态 round 数 |
| 向量化 tileHist 读取 | AIV scalar 降低 | 中 | 用 ReduceSum 替代逐 tile GetValue（math 仓模式） |
| bf16 scoreCache | 减少 UB 占用 + Cast 开销 | 高 | 需重构 xBuf 布局，之前尝试精度问题 |
