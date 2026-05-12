# fused_infer_attention_score 批量一致性分析

## 1. 批量一致性定义

**批量一致性（Batch Consistency）** 要求：对于某一特定输入样本，其输出结果必须**二进制严格一致（bit-exact）**，不随该样本在 batch 中的索引位置变化，也不随 batch 整体大小变化。

例如，给定两组输入：
- Group A: `[b1, b2, b3]`
- Group B: `[b4, b5, b6, b7, b1]`

b1 在两组中的 batch index 和 batch 大小均不同，但 b1 的输出必须 `memcmp` 完全相等。

---

## 2. 算子架构概览

`fused_infer_attention_score` 有三条主要的 tiling 策略路径，通过 `tilingKey` 区分：

| 前缀 KEY | 策略 | 核心调度方式 |
|----------|------|------------|
| `1000000000000000000` | **PFA** (PromptFlashAttention) | 每个 batch 独立调度 |
| `100000000000000000` | **IFA** (IncreFlashAttention) | 先按 batch 分区，batch 内按 head/seq 分区 |
| `5000000000000000000` | **SplitFuse / FAInfer** | 跨 batch 任务展平 + 统一四维遍历 |

SplitFuse 内部有三个子路径：

| 子路径 | 模板参数 | 关键特征 |
|--------|---------|---------|
| non-FD | `IS_FD = false` | 每个 task 是完整的 (B,N1,S1,all_KV) 计算，无跨核 KV 切分 |
| **FD (Flash Decode)** | `IS_FD = true` | KV 序列可跨核切分，CombineScale 归约合并 |
| Decoding | `FAInferKernelDecoding` | qSeqlen=1，按 KV head 维度切分 |

---

## 3. 各路径 bit-exact 批量一致性分析

### 3.1 PFA 路径 — ✅ 通过

每个 batch 完全独立处理，batch index 仅用于全局内存寻址。增删其他 batch 对本 batch 计算无任何影响。

**结论：bit-exact 一致。**

### 3.2 IFA 路径 — ✅ 通过

task 先按 batch 维度分区（`batchIdx = taskIdx / tasksPerBatch`），再在 batch 内分解。每个 task 完全在其所属 batch 内完成，不存在跨核 KV 切分。

**结论：bit-exact 一致。**

### 3.3 SplitFuse non-FD 路径 — ✅ 通过

核心调度逻辑（`flash_attention_regular.h:312-370`）：

```cpp
for (taskIdx = coreIdx; taskIdx < totalTaskNum; taskIdx += coreNum) {
    // 累加各 batch 任务数确定 curBatchTmp
    // batch 内分解为 qSBlockIdx, qNBlockIdx
    runMainLoop(coreIdx, curBatchTmp, qNBlockIdxCur, qSBlockIdxCur,
                /*isSplitKV=*/false, 0, 0, ...);
}
```

每个 task 是一个**原子化的完整计算单元**：处理一个 `(B, N1, S1)` tile 的**全部** KV 序列。不存在跨核 KV 切分（`isSplitKV = false` 硬编码）。`totalTaskNum` 虽然随 batch 组合变化，但 task 分配到不同 core 仅仅是**调度位置**的改变 — 计算参数 `(BIdx, qNBlockIdx, qSBlockIdx)` 确定后，`runMainLoop` 内部执行的计算是完全相同的，不受在哪个 core 上运行的影响。

**结论：bit-exact 一致。**

### 3.4 Decoding 路径 — ✅ 通过

```cpp
curBatch = taskIdx / firstBatchTaskNum;
```

简单除法确定 batch，无跨 batch 干扰。

**结论：bit-exact 一致。**

---

## 4. SplitFuse FD 路径详细分析 — ❌ 不通过

### 4.1 核心问题：切分边界随 totalTaskNum 漂移

FD 路径的 Host 端 tiling 入口（`flash_attention_infer_tiling.h:619-630`）：

```cpp
void FAInferTiling::splitBN2S1GS2(FAInferTilingData &faTilingData) {
    uint64_t totalTaskNum = 0;
    for (int32_t batchIdx = 0; batchIdx < faInfo_.batch; batchIdx++) {
        BatchParams p = getBatchParams(batchIdx, groupSize);
        totalTaskNum += faInfo_.numHeads * p.qSeqlen * p.kvSeqlen;  // ① 跨所有 batch 求和
    }
    uint64_t perCoreTaskNum = (totalTaskNum + blockNum_ - 1) / blockNum_;  // ② 计算每核任务量
    fillCoreInfoForFlashDecode(faTilingData, groupSize, perCoreTaskNum);   // ③ 按四维顺序连续分配
    fillSplitInfoForFlashDecode(faTilingData, groupSize);                  // ④ 记录 KV 切分信息
}
```

`fillCoreInfoForFlashDecode` 按 `(B, N1, S1, S2)` 顺序连续遍历整个任务空间，每个 core 分配 `perCoreTaskNum` 个"工作单元"（`workUnit = remainingQ × remainingKV`）。

**关键依赖链**：

```
batch 组合变化 → totalTaskNum 变化 → perCoreTaskNum 变化 → core 分配边界漂移
    → 同一 (B,N1,S1) tile 的 KV 切分边界改变
    → isSplitKV 状态可能从 false 变为 true（或反之）
    → 切分份数可能从 N 变为 M
    → CombineScale 的输入数据不同
    → 浮点运算顺序不同
    → 输出不 bit-exact
```

### 4.2 具体反例

假设 2 个 AI Core，`numHeads=1, kvHeads=1`。

**场景 A**：batch = [b0, b1]

| Batch | qSeqlen | kvSeqlen | taskCount (= numHeads × q × kv) |
|-------|---------|----------|----------------------------------|
| b0    | 1       | 512      | 512 |
| b1    | 1       | 512      | 512 |

- `totalTaskNum = 512 + 512 = 1024`
- `perCoreTaskNum = ceil(1024 / 2) = 512`
- Core 0: b0 的全部 512 tasks → b0 的 `isSplitKV = false`
- Core 1: b1 的全部 512 tasks → b1 的 `isSplitKV = false`
- b1 的输出由 `runMainLoop` 直接写入，**无 CombineScale 参与**

**场景 B**：batch = [b0, b1, **b2**]（新增 b2）

| Batch | qSeqlen | kvSeqlen | taskCount |
|-------|---------|----------|-----------|
| b0    | 1       | 512      | 512 |
| b1    | 1       | 512      | 512 |
| b2    | 1       | 100      | 100 |

- `totalTaskNum = 512 + 512 + 100 = 1124`
- `perCoreTaskNum = ceil(1124 / 2) = 562`
- Core 0: b0 全部(512) + b1 的前 50 个 S2 task → b1 的 `isSplitKV = true`
- Core 1: b1 剩余 462 个 S2 task + b2 全部(100)
- b1 的输出由 **CombineScale 归约合并** 2 个 core 的部分结果

**对比**：b1 的输入数据（Q/K/V）在场景 A 和场景 B 中完全相同，但：
- 场景 A：`runMainLoop` 在线 softmax 处理全部 512 个 KV token，直接输出
- 场景 B：Core 0 处理 KV `[0, 50)` 得部分结果 (O₀, lse₀)，Core 1 处理 KV `[50, 512)` 得部分结果 (O₁, lse₁)，CombineScale 合并：

```
global_max = max(lse₀_max, lse₁_max)
w₀ = exp(lse₀ - global_max) / (exp(lse₀ - global_max) + exp(lse₁ - global_max))
w₁ = exp(lse₁ - global_max) / (exp(lse₀ - global_max) + exp(lse₁ - global_max))
O = w₀·O₀ + w₁·O₁
```

由于浮点运算的**非结合性**（`(a+b)+c ≠ a+(b+c)` 在浮点下），两种计算路径产生的结果**不可能 bit-exact 一致**。

### 4.3 根本原因

| 层级 | 问题 |
|------|------|
| **Tiling 层** | `totalTaskNum` 是所有 batch 的加权求和，任何 batch 的增删改都会改变它 |
| **Tiling 层** | `perCoreTaskNum` 从 `totalTaskNum` 派生，决定了 core 之间的切分边界 |
| **Tiling 层** | `fillCoreInfoForFlashDecode` 的连续遍历使得切分边界可以在任意 `(B,N1,S1,S2)` 坐标处 |
| **Kernel 层** | `isSplitKV = true` 时，`runMainLoop` 仅计算部分 softmax，写 workspace |
| **Epilogue 层** | `CombineScale` 对不同份数/不同边界的部分结果进行浮点归约，无法保证 bit-exact |

### 4.4 无效的规避路径

以下方式**不能**使 FD 路径达到 bit-exact 一致：

1. **仅改变 batch 顺序（不改变 batch 集合）**：如果 batch 集合相同但顺序不同，`totalTaskNum` 不变，`perCoreTaskNum` 不变。但 batch 在四维空间中的**起始位置**发生变化，切分边界相对于该 batch 的偏移会完全改变。例如 `[b0, b1, b2]` 变为 `[b1, b0, b2]`，b0 原本从 task offset 0 开始，现在从 T1 开始。

2. **增加零长度 batch**：如果新增 batch 的 `qSeqlen=0` 或 `kvSeqlen=0`，则 `tasksPerBatch=0`，`totalTaskNum` 不变。此时切分边界也不变，但这是平凡情况。

### 4.5 非 FD 路径为何通过

对比 non-FD 路径：虽然 `totalTaskNum` 也变化，但 `isSplitKV = false` **硬编码**。这意味着每个 task 是自包含的，`runMainLoop` 在单个 core 上完成整个 `(B,N1,S1)` tile 的在线 softmax 计算。task 分配到哪个 core 只是**调度问题**，不影响计算本身。输出直接写入 batch 对应的全局内存区域，无需跨核归约。

---

## 5. 最终结论

### 总体判断

| 路径 | 批量 bit-exact 一致性 | 原因 |
|------|----------------------|------|
| **PFA** | ✅ 通过 | 各 batch 独立调度 |
| **IFA** | ✅ 通过 | task 先按 batch 分区 |
| **SplitFuse non-FD** | ✅ 通过 | `isSplitKV=false` 硬编码，每个 task 计算完整 |
| **SplitFuse FD** | **❌ 不通过** | `totalTaskNum` 跨 batch 依赖 → 切分边界漂移 → CombineScale 归约无法 bit-exact |
| **Decoding** | ✅ 通过 | batch 索引简单除法 |

### FD 路径不通过的触发条件

以下**任一**条件满足时，特定 batch 的输出会受影响：

1. batch 集合中**增删任意非零序列长度的 batch** → `totalTaskNum` 变化 → `perCoreTaskNum` 变化 → 切分边界全局漂移
2. batch 集合中任一 batch 的序列长度变化 → 同上
3. batch 集合不变但顺序改变 → batch 在任务空间中的起始偏移改变 → 切分边界相对该 batch 漂移（`totalTaskNum` 不变但分配仍不同，因为核心分配从不同 batch 内的不同偏移开始）

### 例外（理论上可能 bit-exact 但不可依赖）

- 所有 batch 恰好对齐 core 边界（每个 batch 的 task 数是 `perCoreTaskNum` 的整数倍）
- 所有 batch 的 `qSeqlen=0` 或 `kvSeqlen=0`（无有效计算）

这些情况在实际使用中不可控，不应作为一致性保证的依据。

### arch32 vs arch35 差异

| 架构 | 代表硬件 | FD 触发条件 |
|------|---------|------------|
| **arch32** | Ascend 910B | FD 强制要求 `pagedCacheFlag == true`（必须使用 block_table 参数）。非 paged attention 模式（BSH layout）下 `flashDecodeFlag` 不会置为 true，走 non-FD SplitFuse 路径，**bit-exact 一致** |
| **arch35** | Ascend 9xx | FD 由 `loopTimes < 0.4 * aicNum` 触发，**不强制**要求 paged attention。非 paged 模式下 `qSeqlen=1` + 长 KV 也可能触发 FD，**不 bit-exact 一致** |

**arch32 FD 条件** (`fused_infer_attention_score_tiling.cpp:1440-1450`)：
- `pagedCacheFlag == true`
- `numTasks = batch * kvHeads <= 0.4 * aicNum && minKVSeqlen >= 1024` (isShortSeq)
- 或 `numTasks <= 0.8 * aicNum && minKVSeqlen >= aicNum * 512` (isLongSeq)

**arch35 FD 条件** (`arch35/fused_infer_attention_score_tiling_impl.cpp:727-757`)：
- `qSeqlen == 1`，`maxActualseq >= 256`
- `loopTimes = batch * kvHeads * ceil(gSize / sOuterFactor_) < 0.4 * aicNum`
- MHA（gSize=1）：仅需 loopTimes < 0.4\*aicNum
- GQA：还需 maxActualseq >= 2048

---

## 6. 验证方法

基于上述分析，验证脚本 `test_batch_consistency_fia.py` 需覆盖：

1. **非 FD 场景**（所有平台应 PASS）：Prompt（qSeqlen>1）、短 KV（kvSeqlen<256）
2. **非 paged FD 场景**（arch35 应 FAIL，arch32 应 PASS）：qSeqlen=1 + 长 KV + GQA/MHA
3. **Paged FD 场景**（两个平台均应 FAIL）：使用 `block_table` 参数 + 长 KV

验证脚本自动检测平台架构（910* → arch32，其他 → arch35）并调整预期结果。

---

## 7. 关键源文件索引

| 文件 | 关键内容 |
|------|---------|
| `op_kernel/fused_infer_attention_score.cpp` | 总调度，按 tilingKey 分发 |
| `op_kernel/flash_attention_regular.h:312-370` | non-FD 路径：task 轮询调度，`isSplitKV=false` |
| `op_kernel/flash_attention_regular.h:246-310` | FD 路径：四维遍历 + isSplitKV 检测 |
| `op_kernel/flash_attention_regular.h:435-1013` | `runMainLoop`：在线 softmax 计算 |
| `op_kernel/flash_attention_regular_decode.h` | Decoding kernel |
| `op_kernel/attn_infra/epilogue/block/CombineScale.hpp` | KV 切分归约合并 |
| `op_host/flash_attention_infer_tiling.h:619-630` | **`splitBN2S1GS2`**：跨 batch 任务展平 + perCoreTaskNum 计算 |
| `op_host/flash_attention_infer_tiling.h:387-511` | **`fillCoreInfoForFlashDecode`**：四维连续分配核心范围 |
| `op_host/flash_attention_infer_tiling.h:514-617` | `fillSplitInfoForFlashDecode`：记录 KV 切分信息 |
| `op_host/fused_infer_attention_score_tiling.cpp:1440-1450` | **arch32 FD 触发条件**：`pagedCacheFlag==true` + isLongSeq/isShortSeq |
| `op_host/arch35/fused_infer_attention_score_tiling_impl.cpp:727-757` | **arch35 FD 触发条件**：`CheckFlashDecode`，不要求 pagedCache |
| `op_host/arch35/fused_infer_attention_score_tiling_impl.cpp:1087-1153` | **arch35 SplitPolicy**：FD 触发入口
