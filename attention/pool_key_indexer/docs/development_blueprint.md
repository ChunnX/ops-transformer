# PoolKeyIndexer 算子开发蓝图

## 一、总体策略

```
步骤1: 退化场景适配 ──→ 2补: pool_key 0轴非连续修复(已实施) ──→ 步骤2: pool_size/pool_tail_k扩展 ──→ 步骤3: 索引展开与追加 ──→ 步骤4: A5量化支持 ──→ 步骤5: torch_extension接口适配
(quantMode=-1,          (tiling读运行时stride +                    (参考LIV2 cmp_ratio/               (Expand + Append)           (quantMode=0/1)           (参考LIV2 OpBuilder模式
 pool_tail_k=0,          arch22接线KeyNd2NzForPA)                    cmp_residual_k,                   输出维度:                    arch35 only              JIT编译 + PrivateUse1 dispatch)
 pool_size=1)                                                       不做索引展开)                       topk+pool_size-1)

  8补: 910B编译修复(已实施) — 15类编译问题修复，跨步骤1-3 + 模板TilingKey + torch_extension
  8补2: 运行时崩溃修复(已实施) — workspace 16MB系统保留区 + CalcWorkspaceSize重写 + ValueDepend转IntArrayRef + ACLNN_CMD参数顺序 + register_fake移除
```

**核心原则**：每步可独立编译验证，前步通过后才进入下一步。

**规格参考**：`docs/aclnnKpoolIndexer.md` 全程参照；步骤5 额外参照 `torch_extension/cann_ops_transformer` 中 `lightning_indexer` / `quant_lightning_indexer` 的 OpBuilder 适配模式。

---

## 二、步骤1 — 退化场景适配（当前阶段）

### 2.1 目标

`quantMode=-1, pool_tail_k=0, pool_size=1` 时，PKI 与 LightningIndexer 行为完全等价。

### 2.2 当前状态

| 组件 | 状态 | 说明 |
|------|------|------|
| op_host (def/infershape/tiling) | ✅ 已完成 | 校验完整，aclnn 已改自动生成 |
| common.h | ✅ 已修复 | 恢复 FIA_SYNC_MODE2/BUFFER_SIZE_BYTE_*/isLDOpen/splitMFlag |
| arch35 kernel.h | ✅ 已修复 | 删多余`}`、恢复splitMFlag计算/isLDOpen读取、LI_SYNC_MODE4→PKI_SYNC_MODE4 |
| arch35 service_cube.h | ✅ 已修复 | splitMFlag 恢复、命名修复 |
| arch35 service_vector.h | ✅ 已修复 | ProcessTopK 从 LI 重写、InitBuffers 修复、ReinterpretCast 修复 |
| arch22 kernel.h | ✅ 已修复 | negInf 定义、valueOutGmTmp uint32 |
| arch22 service_vector.h | ✅ 已修复 | CleanInvalidOutput 重写、ProcessLD/ProcessVec 修复 |

### 2.3 退化场景等价性分析

| 计算路径 | LI 行为 | PKI 退化行为 | 等价? |
|---------|---------|-------------|--------|
| Q⊗K^T (Cube Mmad) | S2=token数 | pool_size=1 → S2=pool数=token数 | ✅ |
| ReLU + Weight 聚合 | G头求和 | 相同 | ✅ |
| Causal Mask | sparseMode==3 | maskMode==3 (值相同) | ✅ |
| TopK 选择 | topk 个 | sparseCount=topk/1=topk | ✅ |
| 索引输出 | topk 个 | topk+1-1=topk 个 | ✅ |
| 值输出 | K_T (fp16/bf16) | **float** | ⚠️ 有意适配 |
| INVALID_VAL | uint16 (0xFC00/0xFF80) | uint32 (0xFF800000 float -inf) | ⚠️ 有意适配 |

**有意差异**（非错误）：
- 输出类型 float：规格要求 `sparseValuesOut` 为 FLOAT
- INVALID_VAL：float -inf 适配 float 输出
- `UIntToFloatReturnValue`：将 uint32 score 转换为 float 输出

### 2.4 待验证项

| 验证项 | 方法 |
|--------|------|
| 编译通过 | arch35 + arch22 编译（FP16/BF16 两种 dtype） |
| 功能正确性 | BSND/TND 两种 layout 跑通，结果与 LI 对齐 |
| maskMode=0/3 | 两种 mask 模式均验证 |
| returnValue true/false | 两种模式均验证 |

### 2.5 遗留风险

1. **arch35 ProcessTopK 重写**：从 LI 拷贝后适配了 float/returnValue/uint32，需编译验证控制流完整性
2. **arch22 CleanInvalidOutput**：重写后 uint32 操作需验证 InitGlobalMemory 支持
3. **splitMFlag 恢复**：`gSize==64 && sparseCount<=2048` 条件在退化场景下是否触发（gSize=N1, sparseCount=topk）

---

## 二·补、pool_key 0轴非连续寻址修复（已实施）

### 2补.1 问题背景

`pool_key` 在 PA（PageAttention）场景下需要支持 0 轴非连续——物理 block 之间可能存在 padding，`stride[0]` 大于 `blockSize * N2 * D` 的名义值。`op_def` 已对所有架构设置 `IgnoreContiguous()`，但 tiling 未读取运行时 stride，而是用 shape 维度乘积算了一个连续 stride 填入 `keyStride0`，导致 PA 场景 0 轴非连续输入地址计算错误。

参考实现：QLI/LIV2 arch35 `lightning_indexer_v2_tiling.cpp:210-215, 572-600, 995-999`。

### 2补.2 实施内容

#### tiling 层（op_host）

**`pool_key_indexer_tiling.h`**：
- 添加 `#include <vector>`
- `PoolKeyIndexerTiling` 类添加私有成员 `std::vector<uint32_t> keyStridesVec_`
- 添加私有方法 `ge::graphStatus CheckKeyContiguous(const PoolKeyIndexerTilingInfo &info) const`

**`pool_key_indexer_tiling.cpp`**：
1. `ParseAndCheckParams` 中 key shape 解析之后，添加 `GetDynamicInputStride(PKI_POOL_KEY_INDEX, 0)` 读取运行时 stride 到 `keyStridesVec_`
2. `keyStride0` 填值修改：
   - PA 路径：`keyStride0 = keyStridesVec_[0]`（有 stride 时）或 `0`（无 stride 时）——替代原来的 `blockSize * N2 * D`
   - BSND 路径：`keyStride0 = 0`——kernel `KeyNd2Nz` 用 `tensorKeyOffset` 连续寻址，不用此字段
   - TND 路径：`keyStride0 = 0`——同上
3. 新增 `CheckKeyContiguous()` 方法：
   - PA：`checkStartIdx = 1`（仅允许 0 轴非连续，从 axis 1 开始检查）
   - BSND/TND：`checkStartIdx = 0`（所有轴必须连续）
   - 按 shape 算 expectedStrides，逐轴比对 `keyStridesVec_[i] != expected[i]`
4. `ParseAndCheckParams` 末尾调用 `CheckKeyContiguous()`

#### kernel 层 — arch22（op_kernel/arch22）

**`pool_key_indexer_kernel.h` `InitTilingData`**：
- 补 `constInfo.keyStride0 = tilingData->keyStride0`（arch22 原来未读取此字段）

**`pool_key_indexer_service_cube.h` `KeyNd2NzForPA`**：
- 地址计算从硬编码 `kCacheBlockSize * kHeadNum * headDim` 改为 `keyStride0`
  ```cpp
  // 改前：blkTable.GetValue(...) * kCacheBlockSize * kHeadNum * headDim
  // 改后：blkTable.GetValue(...) * keyStride0
  ```

#### 无需修改

| 项 | 理由 |
|---|---|
| `op_def` 所有 `IgnoreContiguous()` | 已正确，所有架构保留运行时 stride |
| `common.h` `ConstInfo::keyStride0` | 字段已存在（共用） |
| arch35 `kernel.h:155` | 已读 `keyStride0` |
| arch35 `service_cube.h:279` | 已用 `keyStride0` 寻址 |

### 2补.3 验证方法

1. **golden 构造**：参考 LIV2 golden `lightning_indexer_v2_golden.py:846-855` 的 blockFusion 方式——构造 0 轴 stride 带 padding 的 pool_key，view 回原始 shape
2. **正确性**：非连续输入结果 == 连续输入结果（相同数据内容）
3. **错误路径**：构造非 0 轴非连续输入 → `CheckKeyContiguous` 报错拦截
4. **所有架构**：arch22 + arch35 均验证

### 2补.4 关联事项（步骤4范畴）

`k_descale` 同样设置了 `IgnoreContiguous()` 且 tiling 未接线 `keyDequantScaleStride0`，修复方式相同但需区分 `quant_mode`（shape 不同）。详见 §5.4。

---

## 三、步骤2 — pool_size 和 pool_tail_k 扩展（不做索引展开）

### 3.1 设计思路

参考 LIV2 的 `cmp_ratio` / `cmp_residual_k` 设计模式：

| 概念 | LIV2 | PKI | 对应关系 |
|------|------|-----|---------|
| 压缩比 | cmp_ratio (r) | pool_size | 连续 r/pool_size 个 token → 1 个压缩单元 |
| 残余 token | cmp_residual_k (res_k) | pool_tail_k | 尾部不完整单元的有效 token 数 |
| 原始序列长度 | L_orig = S2·r + res_k | L_orig = S2·pool_size + pool_tail_k | 结构性参数 |
| 压缩空间 TopK | 在 S2 空间 | 在 S2 空间 (sparseCount=topk/pool_size) | 一致 |
| 因果掩码 end(j) | (j+1)·r - 1 | (j+1)·pool_size - 1 | 一致 |

**PKI 与 LIV2 完全对应**：PKI 的 `poolKey` 与 LIV2 的 `key` 一样，输入已经是压缩/聚合后的（S2 维度是压缩块/pool 数，不是原始 token 数）。两者的 cmp_ratio/pool_size、cmp_residual_k/pool_tail_k、因果掩码修正逻辑完全一致，可直接照搬 LIV2 的实现。

### 3.2 关键修改点

#### 3.2.1 op_host 修改

**tiling.cpp `ParseAndCheckParams`**：
```
info.sparseCount = info.topk / info.poolSize;  // 已有
// 新增：pool_size > 1 时的校验
- pool_size 与 topk 的整除关系（已有）
- pool_tail_k[i] ∈ [0, pool_size-1]（已有值校验）
// 新增：因果掩码位置修正参数
- L_orig[i] = S2 * pool_size + pool_tail_k[i]  （PA/BSND 场景，每个 batch）
- global_offset = L_orig - S1  （causal 场景的 query 全局偏移）
```

**tiling.h `ConstInfo`/`TilingData`**：
```
// 已有：poolSize, topk, sparseCount, maskMode, poolTailK（需传入 kernel）
// 新增传入 kernel：
- poolTailK 指针/GM 地址（每个 batch 的尾部 token 数）
```

**SetTilingData**：将 `poolSize` 和 `poolTailK` 的 GM 偏移写入 tiling data。

#### 3.2.2 arch35 kernel 修改

**kernel.h `CalcRunInfo`**：
```
// 因果掩码位置修正（参考 LIV2 的 cmp_ratio 逻辑）
// maskMode == 3 (causal) 时：
//   end(j) = (j+1) * pool_size - 1
//   global_pos_Q(i) = (L_orig - S1) + i
//   有效条件: end(j) <= global_pos_Q(i)
// pool_size=1 时退化为: j <= (L_orig - S1) + i，与 LI 一致
```

**service_cube.h `ComputeMm1`**：
- 无需修改（poolKey 与 LIV2 的 key 语义一致，S2 维度是压缩/pool 数，Cube matmul 直接使用）

**service_vector.h**：
- `trunkLen` / `topkCount_` 已使用 `constInfo_.sparseCount`（在 tiling.cpp:175 计算 `= topk/poolSize`，kernel 仅读取 tilingData），退化时等价
- TopK 在 S2 空间操作，输出 sparseCount 个 pool 索引
- **不展开为 token 索引**（步骤3 才做）

#### 3.2.3 arch22 kernel 修改
- 同 arch35，因果掩码位置修正
- `CleanInvalidOutput` 的填充范围按 sparseCount 调整

### 3.3 pool_tail_k 因果掩码修正

```
对于 batch b:
  L_orig[b] = S2 * pool_size + pool_tail_k[b]   // 原始 token 总数
  offset[b] = L_orig[b] - S1                     // query 全局偏移

  对于 query 位置 i, pool 位置 j:
    pool 覆盖的 token 范围: [j*pool_size, (j+1)*pool_size - 1]
    因果条件: (j+1)*pool_size - 1 <= offset[b] + i
    即: j <= (offset[b] + i + 1) / pool_size - 1

  特殊处理最后一个 pool（有 pool_tail_k 个有效 token）:
    最后 pool 的 end = S2*pool_size + pool_tail_k - 1 = L_orig - 1
```

### 3.4 输出维度（步骤2 不展开）

| 输出 | 退化场景(pool_size=1) | 步骤2(pool_size>1, 不展开) |
|------|----------------------|---------------------------|
| sparse_indices 最后一维 | topk + 0 = topk | **sparseCount = topk/pool_size**（pool 级别） |
| sparse_values 最后一维 | topk/1 = topk | **sparseCount = topk/pool_size** |

**注意**：步骤2 输出的是 pool 索引（未展开），输出维度与规格不同（规格是 topk+pool_size-1）。步骤3 才展开到规格要求的维度。因此步骤2 的输出 shape 需要临时调整为 sparseCount 维度，步骤3 再改回。

### 3.5 验证方法
- pool_size=2/4/8/16，pool_tail_k=0~pool_size-1
- maskMode=0/3
- 对比：手写 golden data（pool 级别 TopK 结果）
- 退化验证：pool_size=1 时结果与步骤1 一致

### 3.6 风险点
1. **pool_tail_k 传入 kernel**：当前 `poolTailK` 在 Init 中接收但未传递给 service。需要在 ConstInfo 中添加 poolTailK GM 偏移，kernel 运行时从 GM 读取每个 batch 的 tail_k 值
2. **因果掩码精度**：`L_orig` 计算涉及乘法，需确认 int32/int64 精度
3. **TND+PA 场景**：pool_tail_k 的前缀和处理需与 actual_seq_k 一致

---

## 四、步骤3 — 索引展开与追加

### 4.1 设计思路

将步骤2 输出的 pool 索引展开为原始 token 索引，并追加尾块有效 token。**无效位置（因果掩码遮挡、尾块不足）统一填 -1。**

```
输出结构 (总维度 = topk + poolSize - 1):

位置:  [0 ........................ validExpand-1] [validExpand ... topk-1] [topk ... topk+visibleTailK-1] [topk+visibleTailK ... topk+poolSize-2]
内容:   有效展开的 token 索引                      -1 (无效)              因果可见的尾块 token 索引      -1 (不可见/不足)

其中:
  validExpand = validS2Len * poolSize       (因果掩码后的有效 pool 数 × poolSize)
  topk = sparseCount * poolSize
  visibleTailK = 因果掩码下可见的尾块 token 数（maskMode=0 时 = poolTailK，maskMode=3 时需计算）
  尾块无效 = poolSize - 1 - visibleTailK 个
```

**关键原则**：先全部初始化为 -1，再覆盖有效部分，无效位置自然保持 -1。

### 4.2 关键修改点

#### 4.2.1 输出 shape
- sparse_indices 最后一维 = `topk + pool_size - 1`（规格要求，infershape/tiling 已正确）
- sparse_values 最后一维 = `topk / pool_size` = sparseCount

#### 4.2.2 arch35 service_vector.h 新增展开逻辑

在 `ProcessTopK` 之后、DataCopy 输出之前，新增 `ExpandAndAppendIndices` 函数。

**向量化设计核心思路**：

展开操作 `tokenIndices[k*P + p] = poolIndices[k] * P + p` 可分解为：

```
Step 0: Duplicate — tokenIndices 全部填 -1（无效占位）
Step 1: Muls  — baseIndices[k] = poolIndices[k] * poolSize     （仅处理 validS2Len 个有效索引）
Step 2: Brcb  — expandedBases[k*P + p] = baseIndices[k]        （广播，每个元素重复 P 次）
Step 3: Add   — tokenIndices = expandedBases + cyclicOffset     （加循环偏移 [0,1,...,P-1] 循环）
Step 4: 尾块  — Duplicate 起始值 + Add 偏移模板
```

无效位置（validS2Len 之后的展开部分、尾块不足部分）保持 Step 0 填充的 -1。

```cpp
__aicore__ void ExpandAndAppendIndices(
    LocalTensor<int32_t>& poolIndices,    // [sparseCount] TopK 输出的 pool 索引
    LocalTensor<int32_t>& tokenIndices,   // [topk+poolSize-1] 输出 token 索引
    LocalTensor<int32_t>& workBuf,         // 临时 buffer
    uint32_t sparseCount,
    uint32_t poolSize,
    uint32_t validS2Len,                   // 因果掩码后的有效 pool 数
    int32_t poolTailK,
    int32_t L_orig,
    uint32_t curS1Idx,                     // 当前 query 位置（用于因果掩码可见性判断）
    uint32_t curS1Size)                    // 当前 batch 的 S1 长度
{
    uint32_t topk = sparseCount * poolSize;
    uint32_t totalOut = topk + poolSize - 1;
    uint32_t validExpand = validS2Len * poolSize;

    // ---- Step 0: 全部初始化为 -1 ----
    Duplicate<int32_t>(tokenIndices, -1, AlignUp(totalOut, 8U));

    if (validS2Len == 0) { return; }  // 无有效 pool 索引，全部 -1

    // ---- Step 1: 构造偏移模板 [0, 1, ..., poolSize-1] ----
    LocalTensor<int32_t> offsetTpl = workBuf;
    for (uint32_t p = 0; p < poolSize; p++) {
        offsetTpl.SetValue(p, static_cast<int32_t>(p));
    }
    // padding 部分 (poolSize ~ AlignUp(poolSize,8)-1) 保持 0

    if (poolSize % 8 == 0) {
        // ============ Brcb 路径 (poolSize 是 8 的倍数) ============
        // 构造循环偏移模式 [0,1,...,P-1, 0,1,...,P-1, ...]
        LocalTensor<int32_t> cyclicOffset = workBuf[AlignUp(poolSize, 8U)];
        for (uint32_t k = 0; k < validS2Len; k++) {
            DataCopy(cyclicOffset[k * poolSize], offsetTpl, poolSize);
        }
        // Muls: base = poolIndices * poolSize
        LocalTensor<int32_t> baseIndices = workBuf[AlignUp(poolSize, 8U) + AlignUp(validExpand, 8U)];
        Muls<int32_t>(baseIndices, poolIndices,
                      static_cast<int32_t>(poolSize), AlignUp(validS2Len, 8U));
        // Brcb: 广播每个 base 重复 poolSize 次
        Brcb(tokenIndices, baseIndices, poolSize, validS2Len);
        // Add: 加循环偏移
        Add<int32_t>(tokenIndices, tokenIndices, cyclicOffset, AlignUp(validExpand, 8U));
    } else {
        // ============ Fallback 路径 (poolSize 非 8 的倍数) ============
        uint32_t alignedPoolSize = AlignUp(poolSize, 8U);
        uint32_t outOff = 0;
        for (uint32_t k = 0; k < validS2Len; k++) {
            int32_t base = poolIndices.GetValue(k) * static_cast<int32_t>(poolSize);
            // Duplicate 填充 base（alignedPoolSize 个，满足对齐）
            Duplicate<int32_t>(tokenIndices[outOff], base, alignedPoolSize);
            // Add 偏移模板（padding 部分 base+0=base，不影响有效结果）
            Add<int32_t>(tokenIndices[outOff], tokenIndices[outOff], offsetTpl, alignedPoolSize);
            outOff += poolSize;  // 只前进 poolSize（padding 被下次覆盖或保持 -1）
        }
    }
    // [validExpand, topk) 保持 -1（Step 0 已填充）

    // ---- Step 2: 追加尾块有效部分（受因果掩码约束） ----
    // pool_tail_k 是物理尾部 token 数（不受因果掩码影响），需判断因果可见性
    if (poolTailK > 0) {
        uint32_t tailPos = topk;
        // 当前 query 位置的全局位置
        int32_t globalPosQ = L_orig - static_cast<int32_t>(curS1Size)
                             + static_cast<int32_t>(curS1Idx);
        // 展开部分的 token 上界 = S2 * poolSize = topk
        int32_t s2TokenEnd = static_cast<int32_t>(topk);
        // 可见尾块 token 数
        int32_t visibleTailK = Max(0, Min(poolTailK, globalPosQ - s2TokenEnd + 1));

        if (visibleTailK > 0) {
            // 偏移模板 [0, 1, ..., visibleTailK-1]
            LocalTensor<int32_t> tailOffsetTpl = workBuf;  // 复用 offsetTpl 区域
            for (int32_t t = 0; t < visibleTailK; t++) {
                tailOffsetTpl.SetValue(t, t);
            }
            // Duplicate 起始值
            Duplicate<int32_t>(tokenIndices[tailPos],
                               L_orig - poolTailK, AlignUp(visibleTailK, 8U));
            // Add 偏移 → [L_orig-poolTailK, ..., L_orig-poolTailK+visibleTailK-1]
            Add<int32_t>(tokenIndices[tailPos],
                         tokenIndices[tailPos], tailOffsetTpl, AlignUp(visibleTailK, 8U));
        }
        // [topk + visibleTailK, totalOut) 保持 -1（Step 0 已填充）
    }
}
```

**指令开销分析**：

| 步骤 | 指令 | Brcb 路径 | Fallback 路径 |
|------|------|-----------|--------------|
| Step 0: 初始化 -1 | Duplicate | 1 | 1 |
| 偏移模板构造 | SetValue | poolSize (≤128) | poolSize (≤128) |
| 循环偏移复制 | DataCopy | validS2Len | — |
| Muls | Muls | 1 | validS2Len/8 批 |
| 广播 | Brcb | 1 | — |
| 加偏移 | Add | 1 | validS2Len × 2 (Duplicate+Add) |
| 尾块 | Duplicate + Add | 2 | 2 |
| **总计** | | **~6 + validS2Len** | **~3 + validS2Len × 2.25** |

对比标量方案 `O(validS2Len * poolSize)` 次标量赋值，两种向量化方案均显著优于标量。

#### 4.2.3 arch22 同样新增展开逻辑
- 在 Vector 核的输出阶段执行，逻辑与 arch35 一致
- arch22 的向量指令 API 与 arch35 略有差异，需适配
- arch22 的 Brcb 广播因子对齐要求可能不同（16 的倍数），需确认

#### 4.2.4 infershape/tiling
- 当前 infershape 已正确输出 `topk + pool_size - 1` 和 `topk / pool_size`
- 步骤2 未修改输出维度，步骤3 无需恢复

### 4.3 验证方法
- pool_size=2/4/8/16，pool_tail_k=0~pool_size-1
- 验证输出维度 = topk + pool_size - 1
- 验证有效展开索引正确性
- 验证无效位置填充 -1（因果掩码遮挡 + 尾块不足）
- 验证尾块追加正确性
- maskMode=0/3 均需验证
- Brcb 路径和 Fallback 路径均需验证（poolSize 为 8 的倍数和非倍数）
- 退化验证：poolSize=1 时不执行展开，输出 = sparseCount = topk（与步骤2一致）

### 4.4 风险点
1. **UB 空间**：workBuf 需容纳偏移模板 + 循环偏移 + baseIndices。Brcb 路径最大需 `AlignUp(poolSize,8) + AlignUp(validExpand,8) + AlignUp(validS2Len,8)` 个 int32。Fallback 路径只需偏移模板。需评估最大场景
2. **Brcb 对齐约束**：广播因子需为 8 的倍数（arch35）或 16 的倍数（arch22），非对齐时走 Fallback
3. **validS2Len 传递**：需从 ProcessTopK 的因果掩码计算中获取 validS2Len，传入 ExpandAndAppendIndices
4. **尾块因果可见性**：pool_tail_k 是物理尾部 token 数（不受因果掩码影响）。步骤3 需根据 `globalPosQ = L_orig - S1 + curS1Idx` 计算可见尾块数 `visibleTailK = Max(0, Min(poolTailK, globalPosQ - topk + 1))`，只追加可见部分。maskMode=0 时 visibleTailK = poolTailK（全部可见）
5. **Duplicate/Add 对齐**：所有向量指令的元素数需 AlignUp 到 8（int32 的 32B 对齐），padding 部分不影响有效结果
6. **int32 向量指令支持**：确认 arch35/arch22 的 `Muls`/`Add`/`Brcb`/`Duplicate` 对 int32 类型的支持

---

## 五、步骤4 — A5 量化支持（quantMode=0/1）

> **2026-08-28 修订**：基于 `attention/quant_lightning_indexer_v2`（QLIv2）实际代码调研，**推翻原"Cube 侧 FP8→FP16 反量化后再 Mmad"的设想**，改为 QLIv2 的两条原生路径。两个语义决策已拍板（见 §5.2）。
>
> **2026-08-28 实施完成**：Phase 0-5 全部落地并在 A5（Ascend950PR）验证通过。
> - 量化套件 `tests/pytest/test_quant.py`：**22/22 passed**（mode=0/1 × BSND/TND/PA × mask 0/3 × PA 非连续 × E8M0=0xFF NaN 传播 × 错误路径）
> - 量化泛化套件 `tests/pytest/test_quant_general.py`：**91/91 passed**（4 布局桶 × 2 mode × 8 例确定性随机 + 27 例定向边界：S2 跨 128 边界、0 池 batch、零长度 query、topk 极值含 3072、n1 极值 1/64、PA 非连续、pa_bs=1024、tail 边界、batch 5/6 维）
> - 非量化回归：precision+smoke 138 passed、GE 图模式 68 passed（无回归）
> - 编译回归：ascend950 + ascend910b（共用 tiling）均通过
> - 实施中的关键修正（相对本蓝图原稿）：① dstNzC0Stride 对齐单位保持 BLOCK_CUBE=16（QLIv2 同款，仅 KeyNd2NzForPA 的 dst 写偏移用类型因子 32）；② kScale L1 乒乓步长按 mBaseSizeMax 运行时计算（非编译期常量）；③ k_descale 乒乓搬运条件是 `(s2Idx-s2Start)%16==0`（不是 isFirstS2InnerLoop，多 s2 块场景滚动覆盖）；④ kScaleUB 需 Duplicate 0 初始化（尾块不足 s2BaseSize 时防垃圾分数）
> - 泛化测试追加修正（2026-08-28 二轮）：⑤ **量化路径禁用 splitM**（`isQuant` 时 splitMFlag=false）——splitM 分支的 LoadData2D dst 布局按 fp16 M 切分推导（dstOffset=i*64*16 等），fp8 的 L1 NZ 行深变化后该分支布局错位（fp8+n1=64+PA+sparse<256 组合实测分数错乱出负值）；QLIv2 无 splitM 机制可参照，正确性优先，量化场景走常规单块路径；⑥ **mxFP8 多 s2 块的 LoadKScaleToL1 双重偏移**——非 PA 分支以 tensorKeyScaleOffset（含 s2Idx 前缀）为基 + 块内相对偏移，调用方此前误传绝对池号导致 s2Idx>0 时 scale 读错位（S2=300 实测分数爆炸到 1e16）
>
> **2026-08-29 泛化第二层（扩展包络）完成**：`test_quant_general2.py` 71 例（ps 24/64/128、topk 3072/4096、S2 至 700 跨 5+ 块、E8M0 宽编码 100..160、descale 大幅差、深 batch、pa_bs 至 512）全过。追加修复 3 个 bug：
> ⑦ **量化 + 禁 splitM 后 mBaseSize 必须 L0 对齐钳制**：s1gL0RealSize 可达 256 直写 128 行 L0A/L0C 越界（aicore l1 error 507015 偶发崩溃，plog 显示 AIC 核 l1 error info 非零）；钳制条件是 mBaseSize=gSize×s1PerL0（s1PerL0 偶数）——半区（mBaseSize/2）必须含整数个 s1 行（dualDstCtl 按半区拆给两 AIV，Vec 按 gSize 行距读取），n1=40/48 实测半区错位导致集合失败；
> ⑧ **GetKeyScale PA 分支块边界重写**：原 QLIv2 式按 kCacheBlockSize 步进写 UB（pa_bs>s2BaseSize 时单块写入跨段越界），改为按池号连续写段内 + GM 源按物理块跳转（段偏移与读取位置一致）；
> ⑨ **mode=0 qScale 融合改为 UB 预乘 + BroadcastLane**：Gather 对寄存器 Mul 产物（weight×qScale）的 lane 布局不兼容（gSize≥32 时取错 lane 丢 g 贡献），改为标量预乘落 UB（weightScaleUB，行距 128 元素 bank 对齐）+ `LoadAlign<DIST_BRC_B32>` 从 UB 广播第 i 个 weight（QLIv2 BroadcastLane 模式）；
> ⑩ **泛化第二层采样修正**：topk 采样必须落在合法档位（[1,2048]∪{3072,4096,5120,6144,7168,8192}），sparse×ps 常落入非法间隙（如 2816），重试直到合法。
> 验证：量化三层 184/184 两轮全过；非量化回归 precision+smoke+GE 图 206 passed 无回归。

### 5.1 QLIv2 调研结论与总体架构

QLIv2 实际架构（与"反量化成 FP16 再 Mmad"不同）：

| QLIv2 quant_mode | Cube 侧 | scale 消费点 | PKI 对应 |
|------------------|---------|-------------|---------|
| 1 (FP8 per-token) | **原生 FP8×FP8 Mmad**（L0C fp32 累加），scale 不上 Cube | **Vector 侧融合**：`BatchMulWeightAndReduceSum(..., qScale, kScale)`——qScale 乘入 weight、kScale 乘入聚合和 | quantMode=0 |
| 3 (mxFP8) | **LoadData-Mx 硬件反量化**：数据+E8M0 scale 同入 L0（scale 按 2 个 E8M0 打包成 bf16 单位 Dn2Nz 搬入 L1） | Cube L0 内硬件完成，Vector 无 scale | quantMode=1 |

数学依据：quantMode=0 的 scale 为 per-token-head 标量，可从 D 维点积中完全提出：`relu(Σ_d q·k) · dq · dk`（scale 恒正，ReLU 与乘法可交换，等价精确）。quantMode=1 的 scale 为 per-32-D-group，不可提出，必须走硬件 MX 路径。

**PKI 现有代码对两条路径的适配度**（调研确认）：

| 项 | 状态 | 说明 |
|----|------|------|
| `MulWeightAndReduceSum`（arch35 vf/pool_key_indexer_vector1.h:228） | ✅ 预留 | 已声明 regQScale/regKScale 寄存器（从 QLIv2 简化而来未使用），qk 输入已是 float，ReLU 在 vector 侧做——mode=0 加 scale 是"回归"式改动 |
| Fixpipe float 路径（arch35 service_cube.h:399-415） | ✅ 就绪 | `Fixpipe<float,float>` dualDstCtl 双目标已存在，FP8 Mmad 的 L0C float 结果直接复用 |
| 模板 Tiling Key FP8 组合（template_tiling_key.h:168-177） | ✅ 已注册 | QUANT_MODE 0/1/2 全枚举已含，entry 的 FP8 dtype 分支已存在（pool_key_indexer.cpp:61-64） |
| entry 按 QUANT_MODE 分发 SCALE_T | ❌ 缺失 | 当前 FP8 分支未区分 mode，PkiType 无 SCALE_T 参数 |
| descale GM 接线 | ❌ 缺失 | kernel.h:417 已接收 qDescale/kDescale 形参但未消费 |

### 5.2 语义决策记录（已拍板）

| 决策点 | 结论 | 依据 |
|--------|------|------|
| mxFP8 语义 | **与 QLIv2 一致**：每 32 个 D 元素一组 E8M0 scale，descale shape = query/key 去 D shape + **[D/64, 2] 尾维**（E8M0 按 2 打包） | QLIv2 tiling.cpp:1158-1199（MX shape 校验）、golden.py:104-105 |
| E8M0=0xFF | **不拦截，NaN 传播**（0xFF 按 E8M0 约定为 NaN） | QLIv2 golden.py:199-203（`raw==255 → NaN`）；host/kernel 均无 0xFF 校验 |

E8M0 解码约定（kernel/golden 共用）：`scale = 2^(raw - 127)`；`raw == 255 → NaN`；`raw == 0 → 2^-127`。

⚠️ **连带影响——规格文档冲突**：`docs/aclnnKpoolIndexer.md:305-339` 现写 quantMode=1 的 descale shape 与 mode=0 相同（q:(B,S1,N1)、k:(B,S2,N2)/PA:(blockNum,blockSize,N2)），与 QLIv2 语义冲突。**Phase 0 需修订规格**（descale shape 按 mode 区分，见 §5.4）。issue.md 中"每32元素共享一个scale"的公式与 QLIv2 语义一致，无需改。

### 5.3 两种量化模式的计算路径

**quantMode=0（FP8 per-token-head，descale=FLOAT）**：

```
scale shape: q=(B,S1,N1)/(T1,N1)，k=(B,S2,N2)/(T2,N2)/PA:(blockNum,blockSize,N2)
Cube:   原生 FP8×FP8 Mmad（scale-free）→ L0C float → Fixpipe float（现有路径）
Vector: relu(qk) 聚合时融合 scale：
        out = (Σ_g relu(Σ_d q·k) · w_g · dq) · dk    ← dq 乘入 weight，dk 乘入聚合和
参考: QLIv2 QUANT_MODE_FP8 → vector1.h:1183 MulWeightAndReduceSumOptionalScaleImpl<true>
```

**quantMode=1（mxFP8，descale=FLOAT8_E8M0）**：

```
scale shape: q=(B,S1,N1,D/64,2)/(T1,N1,D/64,2)
             k=(B,S2,N2,D/64,2)/(T2,N2,D/64,2)/PA:(blockNum,blockSize,N2,D/64,2)
Cube:   LoadData-Mx 变体（数据+scale 同入 L0，硬件按 32 元素组反量化）→ Mmad → L0C float → Fixpipe float
        scale 搬运：Dn2Nz（2 个 E8M0 打包 1 个 bf16），scalePerToken = headDim/32
Vector: 无 scale（现有路径不变）
参考: QLIv2 QUANT_MODE_MXFP8 → service_cube.h:480-547（LoadQScale/LoadKScaleToL1）、563-586（LoadData-Mx）
```

weights 在两种模式下均为 FP16/BF16（不量化）。arch22 不支持 FP8，quantMode 只能是 -1（tiling 层架构门禁拦截，见 §5.4）。

### 5.4 op_host 现状与缺口

**已有校验**（tiling.cpp:243-337, 532-621，覆盖较全）：

| 校验项 | 状态 | 位置 |
|--------|------|------|
| quantMode 合法值 (-1/0/1) | ✅ | :155 |
| descale 空与非空匹配（-1→null，0/1→非null） | ✅ | :259-267, 280-285, 310-315 |
| query dtype（-1→FP16/BF16，0/1→FP8_E4M3FN） | ✅ | :269-272, 287-290, 317-320 |
| descale dtype（0→FLOAT，1→FLOAT8_E8M0） | ✅ | :294-301, 324-331 |
| weights dtype（-1→同query，0/1→FP16/BF16） | ✅ | :274-276, 302-306, 332-336 |
| descale shape 与 query/poolKey 逐维一致（含 PA 三维） | ✅（仅按 mode=0 语义） | :532-621 |
| def.cpp 950 config dtype 组合表（4组合） | ✅ | def.cpp:112-145 |

**缺口（4项，随本步骤补齐）**：

1. **架构门禁缺失**：无 "quantMode 0/1 仅 DAV_3510" 校验（当前被 tiling.cpp:247-257 全局 fail-fast 掩盖，解除时必须补）
2. **keyDequantScaleStride0 死字段**：tiling 无赋值点，`GetDynamicInputStride(PKI_K_DESCALE_INDEX, 0)` 从未调用（QLIv2 参考 tiling.cpp:228-236）
3. **CheckKeyContiguous 不覆盖 k_descale**：k_descale 已设 `IgnoreContiguous()`（def.cpp:137），PA 0轴非连续时其余轴需校验连续（QLIv2 有 scaleNonContiguous 检查）
4. **mode=1 的 descale shape 校验缺失**：现校验按 mode=0 语义（(B,S2,N2)），mode=1 需改为 +[D/64,2] 尾维，并补 headDim%64==0（headDim=128 恒真，校验保留防御）

`k_descale` 的 `IgnoreContiguous()` 设置合理（与 QLIv2 一致）：k_descale 与 pool_key 在 token 维度一一对应，PA 场景必须跟随 pool_key 的 0 轴非连续。

**keyDequantScaleStride0 填值**（参考 QLIv2 tiling.cpp:1374-1381；PKI 无 FP4，比 QLIv2 简单）：

| 场景 | mode=0 (per-token) | mode=1 (mxFP8) |
|------|--------------------|----------------|
| PA（有运行时 stride） | `keyDequantScaleStridesVec_[0]` | `keyDequantScaleStridesVec_[0]` |
| PA（无 stride，连续推导） | `blockSize * N2` | `blockSize * N2 * (D/32)` |
| BSND/TND | 0（连续寻址，不用） | 0（连续寻址，不用） |

**kernel 侧 k_descale 寻址**（mode=0，Vector 侧）：非 PA 连续寻址；PA 走 `block_table × keyDequantScaleStride0`（参考 QLIv2 GetKeyScale，service_vector.h:432-485）。mode=1 的 scale 寻址在 Cube 侧 LoadKScaleToL1 内（同公式）。

### 5.5 实施计划（6 Phase，每阶段独立编译验证）

**Phase 0 — op_host + 规格修订**
- fail-fast（tiling.cpp:247-257）→ 架构门禁：仅 `GetCurNpuArch()==DAV_3510` 放行 quantMode 0/1
- descale shape 校验按 mode 分裂：mode=0 维持现状；mode=1 改为 +[D/64,2] 尾维（BSND:(B,S1,N1,D/64,2)，TND:(T1,N1,D/64,2)，PA:(blockNum,blockSize,N2,D/64,2)），补 headDim%64==0
- `keyDequantScaleStridesVec_` 接线 + `keyDequantScaleStride0` 填值（按 §5.4 表）
- CheckKeyContiguous 扩 k_descale（PA 仅查 0 轴起，非 PA 全连续）
- 修订 docs/aclnnKpoolIndexer.md 的 mode=1 descale shape（消除 §5.2 所述冲突）

**Phase 1 — kernel 框架**
- `PkiType` 增加 `SCALE_T` 模板参数（mode0→float，mode1→fp8_e8m0_t）+ `isMxFp8` trait（参考 QLIv2 common.h:38-58）
- entry arch35 FP8 分支按 QUANT_MODE tpl 值分发 SCALE_T（pool_key_indexer.cpp:61-64）
- kernel.h：descale GM 接线（mode=1 Cube 侧按 bfloat16_t reinterpret——2×E8M0 打包；mode=0 Vector 侧）
- RunInfo 增 `tensorQScaleOffset/tensorKeyScaleOffset`（参考 QLIv2 kernel.h:690-715 的 scale offset 公式）

**Phase 2 — Cube 侧**
- FP8 NZ 布局因子：`KeyNd2NzForPA` dst offset `×FP8_BLOCK_CUBE(32)`（现 service_cube.h:294 是 FP16_BLOCK_CUBE=16）；`LoadData kStep=CeilDiv(headDim,32)`（现 /16）
- L1/L0 buffer 按 sizeof(Q_T) 自动折半（InitBuffers 按类型分配，FP8 实例化自动生效，验证即可）
- mode=1 MX 路径移植：`bufQScaleL1_/bufKeyScaleL1_` + `LoadQScaleToL1/LoadKScaleToL1`（Dn2Nz，`scalePerToken=D/32`，GM offset `/FP8_TWO`，PA 按 block_table×keyDequantScaleStride0）+ LoadData-Mx 变体（LoadData2DMxParams，`yStep=D/64`）+ L0 dst type `mx_fp8_e4m3_t`
- Mmad/ComputeL0c/Fixpipe 不变

**Phase 3 — Vector 侧（仅 mode=0）**
- `InitVecInputTensor` 增 qScaleGm/kScaleGm；ProcessVec1 增两处搬运：
  - q_descale 与 weights 同 pattern 同 offset（shape 相同，QLIv2 直接复用 weightGmOffset，service_vector.h:523-531）
  - k_descale：非 PA 连续寻址；PA 走 `block_table × keyDequantScaleStride0`（参考 QLIv2 GetKeyScale，service_vector.h:432-485）
- `MulWeightAndReduceSum` 增 qScale/kScale 参数：qScale 乘入 regW 后 Gather 广播，kScale 乘入 regSum（量化 uint32 前）
- UB 预算复核：kScaleBuf ~16KB + qScaleBuf（arch35 UB 248KB，需核算与现有 128KB mm1ResUB 的余量）

**Phase 4 — torch_extension**
- `FixPkiAclDtypes`：mode=1 校验 descale dtype 为 float8_e8m0fnu + TensorWrapper `ACL_FLOAT8_E8M0`；mode=0 直接透传（§6.3.2 预留项落地）

**Phase 5 — 测试**
- golden 扩展：mode=0 per-token 参考实现；mode=1 移植 QLIv2 `e8m0_raw_to_float`（含 0xFF→NaN）+ per-32-group 反量化
- 用例矩阵：BSND/TND/PA × mask 0/3 × quant 0/1 × returnValue × E8M0=0xFF 用例（NaN 传播）× k_descale 0轴非连续 blockFusion
- FP8 vs FP16 直接计算精度对比；arch22 拦截回归；quantMode=-1 无回归

**验证顺序**：Phase 0 编译 → 1+2 编译 → mode=0 端到端 → mode=1 端到端 → Phase 4 → 全量回归

### 5.6 风险点

1. **LoadData-Mx 的格式约束**（mode=1）：LoadData2DMxParams 的 yStep/srcStride 单位是打包后的 bf16（2×E8M0），与数据侧 32 元素一组的映射需逐参数核对 QLIv2 原实现（service_cube.h:563-586）
2. **FP8 NZ 布局偏移**：L1 NZ 布局中 fp8 与 fp16 的 C0 粒度不同（32 vs 16），`KeyNd2NzForPA` 的 `s2L1Offset * FP16_BLOCK_CUBE` 类偏移必须全部按类型分支
3. **Vector scale 融合的数值口径**（mode=0）：dq 乘入 weight 后与 weight 的 FP16/BF16 精度交互——QLIv2 先 Cast weight 到 fp32 再乘 scale，PKI 需保持同一顺序避免精度分叉
4. **k_descale PA 寻址**（mode=0）：block_table 查表逐块搬运的同步（QLIv2 GetKeyScale 的 S_MTE2 硬同步模式需照搬）
5. **UB 预算**（mode=0 vector）：qScale/kScale 乒乓 buffer 新增占用，需与 128KB mm1ResUB + 现有 vector 缓冲核算 248KB 上限
6. **arch22 拦截**：quantMode 0/1 在 A2/A3 必须被 tiling 层架构门禁拒绝（Phase 0），防止错误结果静默产出
7. **E8M0=0xFF NaN 传播**：TopK 对 NaN score 的排序行为需与 golden 一致（golden 按 NaN 参与排序，不特判）

---

## 六、步骤5 — torch_extension 接口适配

### 6.1 目标与定位

将 `aclnnPoolKeyIndexer` 封装为 `cann_ops_transformer` 库中的 PyTorch 算子，使上层模型代码可通过 `cann_ops_transformer.pool_key_indexer(...)` 直接调用，并支持 `torch.compile` / FakeTensor / Autograd 的图模式推导。

**适配位置**：`ops-transformer/torch_extension/cann_ops_transformer`（OpBuilder + JIT 编译体系），**不是**算子目录下既有的 `torch_ops_extension/`（基于 `NpuExtension` + `EXEC_NPU_CMD_V1` 的旧式 custom 命名空间封装，二者并存，步骤5 以 `cann_ops_transformer` 体系为准）。

**参考实现**：
- 主参考：`ops/lightning_indexer.py` + `ops/csrc/lightning_indexer.cpp`（LIV2 非量化路径）
- 量化路径参考：`ops/quant_lightning_indexer.py` + `ops/csrc/quant_lightning_indexer.cpp`（FP8 dtype 适配、`TensorWrapper` + `FixAclDtypes` 模式）

**与 LIV2 的关键差异**：

| 维度 | LIV2 | PKI | 影响 |
|------|------|-----|------|
| aclnn 接口 | 两段式 + 独立 `aclnnLightningIndexerV2Metadata` | **仅两段式，无 metadata 子算子** | 步骤5 无需 metadata 算子，仅注册单算子 |
| 输出 values dtype | bfloat16 | **float**（规格要求） | 输出 tensor 用 `at::kFloat` |
| 输出 indices 维度 | topk | **topk + pool_size - 1** | infershape 需按 pool_size 计算 |
| 输出 values 维度 | topk | **topk / pool_size**（pool 级，非展开） | 与 indices 维度不同 |
| return_value=False | values 为空 | values 为空（`{0}` shape） | 一致 |
| 量化接口分离 | 独立 `quant_lightning_indexer` 算子 | **单算子，quant_mode 驱动** | 步骤5 只注册一个算子，quant_mode 作属性 |

### 6.2 新增文件清单

| 文件 | 职责 | 参考 |
|------|------|------|
| `torch_extension/cann_ops_transformer/ops/csrc/pool_key_indexer.cpp` | C++ 后端：构造输出 tensor + `ACLNN_CMD(aclnnPoolKeyIndexer, ...)` 下发 | `csrc/lightning_indexer.cpp` |
| `torch_extension/cann_ops_transformer/ops/pool_key_indexer.py` | Python 前端：`OpBuilder` + schema + Meta + PrivateUse1 dispatch（`register_fake` 已移除，Meta impl 足够） | `ops/lightning_indexer.py` |
| `torch_extension/cann_ops_transformer/ops/graph_convert/graph_convert_pool_key_indexer.py` | torchair 图模式 GE Converter：`register_fx_node_ge_converter` + `torchair.ge.custom_op("PoolKeyIndexer")` | `graph_convert/graph_convert_lightning_indexer.py`（LIV2 stub）、旧 `torch_ops_extension` converter（已移植） |
| `torch_extension/cann_ops_transformer/ops/__init__.py`（修改） | 导出 `pool_key_indexer` + `convert_pool_key_indexer` | 在末尾追加 import |

**依赖的公共组件（无需新建，已存在）**：
- `op_builder/builder.py`：`OpBuilder` 基类（JIT `load`、`register_schema`、`register_meta`、`include_paths`、`cxx_args`、`extra_ldflags`）
- `common/inc/aclnn_common.h`：`ACLNN_CMD` 宏、`ConvertType`、`TensorWrapper`、`get_valid_tensor`、`DecodeDevice`、`ApplyDeterministicConfig`

### 6.3 关键修改点

#### 6.3.1 C++ 后端 `ops/csrc/pool_key_indexer.cpp`

参考 `lightning_indexer.cpp` 的 `ConstructLightningIndexerOutputTensor` + `LightningIndexer` 结构，PKI 无需 metadata 函数，仅实现一个主函数。

**输出 shape 推导**（注意 BSND/TND 差异，且 indices 与 values 末维不同）：

```cpp
std::tuple<at::Tensor, at::Tensor> ConstructPoolKeyIndexerOutput(
    const at::Tensor &query, int64_t topk, int64_t poolSize,
    const std::string &layoutQ, bool returnValue)
{
    int64_t indicesLen = topk + poolSize - 1;   // 展开后 token 数
    int64_t valuesLen  = topk / poolSize;      // pool 级分数数（sparseCount）
    at::SmallVector<int64_t, SIZE> shape;
    if (layoutQ == "BSND") {
        shape = {query.size(0), query.size(1), /*indicesLen 或 valuesLen*/};
    } else {  // TND
        shape = {query.size(0), /*indicesLen 或 valuesLen*/};
    }
    at::Tensor sparseIndices = at::empty(shapeWith(indicesLen), query.options().dtype(at::kInt));
    at::Tensor sparseValues  = returnValue
        ? at::empty(shapeWith(valuesLen), query.options().dtype(at::kFloat))
        : at::empty({0}, query.options().dtype(at::kFloat));
    return {sparseIndices, sparseValues};
}
```

**主函数签名**（严格对齐 aclnn 两段式参数顺序：inputs → attrs → outputs）：

```cpp
std::tuple<at::Tensor, at::Tensor> PoolKeyIndexer(
    const at::Tensor &query, const at::Tensor &poolKey, const at::Tensor &weights,
    const at::Tensor &poolTailK,
    const c10::optional<at::Tensor> &actualSeqQ,
    const c10::optional<at::Tensor> &actualSeqK,
    const c10::optional<at::Tensor> &blockTable,
    const c10::optional<at::Tensor> &qDescale,
    const c10::optional<at::Tensor> &kDescale,
    c10::string_view layoutQ, c10::string_view layoutK,
    int64_t topk, int64_t poolSize, int64_t maskMode, int64_t quantMode,
    bool returnValue)
{
    TORCH_CHECK(query.numel() > 0, "Tensor query is empty.");
    TORCH_CHECK(poolKey.numel() > 0, "Tensor pool_key is empty.");

    std::string queryLayoutStr(layoutQ), keyLayoutStr(layoutK);
    auto [sparseIndicesOut, sparseValuesOut] = ConstructPoolKeyIndexerOutput(
        query, topk, poolSize, queryLayoutStr, returnValue);

    // ValueDepend(OPTIONAL) inputs: aclnn expects aclIntArray*, convert tensor→IntArrayRef
    at::Tensor poolTailKCpu = poolTailK.to(at::kLong).cpu().contiguous();
    at::IntArrayRef poolTailKArr(poolTailKCpu.data_ptr<int64_t>(), poolTailKCpu.numel());

    at::Tensor actualSeqQCpu;
    c10::optional<at::IntArrayRef> actualSeqQArr = c10::nullopt;
    if (actualSeqQ.has_value() && actualSeqQ.value().defined()) {
        actualSeqQCpu = actualSeqQ.value().to(at::kLong).cpu().contiguous();
        actualSeqQArr = at::IntArrayRef(actualSeqQCpu.data_ptr<int64_t>(), actualSeqQCpu.numel());
    }
    // actualSeqK 同理 ...

    char *queryLayoutPtr = const_cast<char *>(queryLayoutStr.c_str());
    char *keyLayoutPtr = const_cast<char *>(keyLayoutStr.c_str());

    // 注意参数顺序：attrs 为 topk, poolSize, layoutQ, layoutK, maskMode, quantMode, returnValue
    ACLNN_CMD(aclnnPoolKeyIndexer,
              query, poolKey, weights, poolTailKArr,
              actualSeqQArr, actualSeqKArr, blockTable,
              qDescale, kDescale,
              topk, poolSize, queryLayoutPtr, keyLayoutPtr,
              maskMode, quantMode, returnValue,
              sparseIndicesOut, sparseValuesOut);
    return {sparseIndicesOut, sparseValuesOut};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("pool_key_indexer", &PoolKeyIndexer, "pool_key_indexer");
}
```

> **注**：本样例未含 `c10::OptionalDeviceGuard`，与 LIV2/QLI/flash_attn 等 lightning 家族算子保持一致（依赖 `query.options()` 携带设备信息分配输出张量）。开发规范 §3.1 建议加 DeviceGuard，但当前算子家族约定不使用。
>
> **关键设计**：
> - `pool_tail_k` / `actual_seq_q` / `actual_seq_k` 是 `ValueDepend(OPTIONAL)` 输入，aclnn API 期望 `aclIntArray*`（非 `aclTensor*`），C++ wrapper 需将 1-D int64 tensor 转 CPU + `contiguous()` + 构造 `at::IntArrayRef`
> - `block_table` / `q_descale` / `k_descale` 直接传 `c10::optional<at::Tensor>`，`ConvertType` 自动处理 optional→`aclTensor*`(或 nullptr)
> - ACLNN_CMD 属性参数顺序为 `topk, poolSize, layoutQ, layoutK, maskMode, quantMode, returnValue`（对齐 aclnn IR 声明）

#### 6.3.2 FP8 量化 dtype 适配（依赖步骤4，可延后）

参考 `quant_lightning_indexer.cpp` 的 `TensorWrapper` + `FixQLIV2AclDtypes` 模式。PKI 的 quant_mode 语义与 LIV2 不同（PKI: -1/0/1，LIV2: 0..5），需实现 PKI 专属的 dtype 修正：

```cpp
inline void FixPkiAclDtypes(int64_t quantMode,
                            TensorWrapper &queryW, TensorWrapper &keyW,
                            TensorWrapper &qScaleW, TensorWrapper &kScaleW)
{
    if (quantMode == 0) {
        // FP8 per-token-head: query/key=FP8_E4M3FN, descale=FLOAT，aclnn 已自动识别
    } else if (quantMode == 1) {
        // mxFP8: descale=FLOAT8_E8M0，需显式设置 scale dtype
        TORCH_CHECK(qScaleW.tensor_.scalar_type() == at::kFloat8_e8m0fnu,
                    "quant_mode=1 requires q_descale dtype float8_e8m0fnu");
        TORCH_CHECK(kScaleW.tensor_.scalar_type() == at::kFloat8_e8m0fnu,
                    "quant_mode=1 requires k_descale dtype float8_e8m0fnu");
        qScaleW.dtype = ACL_FLOAT8_E8M0;
        kScaleW.dtype = ACL_FLOAT8_E8M0;
    }
    // quant_mode=-1: descale 必须为 None，不进入此函数
}
```

**量化路径下 `ACLNN_CMD` 传参改用 `TensorWrapper`**（使修正后的 dtype 生效），非量化路径直接传 `at::Tensor`。

#### 6.3.3 Python 前端 `ops/pool_key_indexer.py`

参考 `lightning_indexer.py`，PKI 无 metadata 子算子，schema 仅一条：

```python
class PoolKeyIndexerOpBuilder(OpBuilder):
    def __init__(self):
        super().__init__("pool_key_indexer")

    def sources(self):
        return ["ops/csrc/pool_key_indexer.cpp"]

    def schema(self) -> str:
        # 必选 tensor + 必选 scalar 在前，optional tensor 在 '*' 后
        return ("pool_key_indexer(Tensor query, Tensor pool_key, Tensor weights, "
                "Tensor pool_tail_k, *, "
                "Tensor? actual_seq_q=None, Tensor? actual_seq_k=None, "
                "Tensor? block_table=None, Tensor? q_descale=None, Tensor? k_descale=None, "
                'str layout_q="BSND", str layout_k="BSND", '
                "int topk=128, int pool_size=1, int mask_mode=0, int quant_mode=-1, "
                "bool return_value=False) -> (Tensor, Tensor)")

    def register_meta(self):
        @impl(AS_LIBRARY, self.name, "Meta")
        def pool_key_indexer_meta(query, pool_key, weights, pool_tail_k, *,
                                  actual_seq_q=None, actual_seq_k=None,
                                  block_table=None, q_descale=None, k_descale=None,
                                  layout_q="BSND", layout_k="BSND",
                                  topk=128, pool_size=1, mask_mode=0,
                                  quant_mode=-1, return_value=False):
            indices_len = topk + pool_size - 1
            values_len  = topk // pool_size
            if layout_q == "BSND":
                idx_shape = [query.shape[0], query.shape[1], indices_len]
                val_shape = [query.shape[0], query.shape[1], values_len]
            else:  # TND
                idx_shape = [query.shape[0], indices_len]
                val_shape = [query.shape[0], values_len]
            sparse_indices = torch.empty(idx_shape, dtype=torch.int32, device="meta")
            sparse_values  = (torch.empty(val_shape, dtype=torch.float, device="meta")
                              if return_value else torch.empty([0], dtype=torch.float, device="meta"))
            return (sparse_indices, sparse_values)

pool_key_indexer_op_builder = PoolKeyIndexerOpBuilder()

@impl(AS_LIBRARY, pool_key_indexer_op_builder.name, "PrivateUse1")
def pool_key_indexer(query, pool_key, weights, pool_tail_k, *,
                     actual_seq_q=None, actual_seq_k=None,
                     block_table=None, q_descale=None, k_descale=None,
                     layout_q="BSND", layout_k="BSND",
                     topk=128, pool_size=1, mask_mode=0,
                     quant_mode=-1, return_value=False):
    op_module = pool_key_indexer_op_builder.load()
    return op_module.pool_key_indexer(
        query, pool_key, weights, pool_tail_k,
        actual_seq_q, actual_seq_k, block_table, q_descale, k_descale,
        layout_q, layout_k, topk, pool_size, mask_mode, quant_mode, return_value)
```

**与 LIV2 的差异**：
- 无 `*_metadata` 算子（PKI aclnn 接口无 metadata 分离，不同于 LIV2 的 `lightning_indexer_metadata`）
- `register_fake` 已移除（与 LIV2 主算子一致，Meta impl 足够提供 shape 推导；如后续需 torch.compile 支持可补回）
- 已添加 `graph_convert_pool_key_indexer.py`，注册真实可用的 GE converter（移植自旧 `torch_ops_extension`，不同于 LIV2 的 RuntimeError stub）

#### 6.3.4 graph_convert Converter `ops/graph_convert/graph_convert_pool_key_indexer.py`

参考 `graph_convert_lightning_indexer.py` 的结构，但 PKI 的 GE converter **实际可用**（移植自旧 `torch_ops_extension` 的 `torchair.ge.custom_op` 实现），而非 LIV2 的 `raise RuntimeError` stub：

```python
@register_fx_node_ge_converter(
    torch.ops.cann_ops_transformer.pool_key_indexer.default
)
def convert_pool_key_indexer(query, pool_key, weights, pool_tail_k, *,
                             actual_seq_q=None, actual_seq_k=None,
                             block_table=None, q_descale=None, k_descale=None,
                             layout_q="BSND", layout_k="BSND",
                             topk=128, pool_size=1, mask_mode=0,
                             quant_mode=-1, return_value=False,
                             meta_outputs=None):
    return torchair.ge.custom_op(
        "PoolKeyIndexer",
        inputs={...},  # 9 个输入 tensor
        attrs={...},   # 7 个属性
        outputs=["sparse_indices_out", "sparse_values_out"])
```

**与 LIV2 的差异**：LIV2 的 `graph_convert_lightning_indexer.py` 仅注册 metadata 子算子且 raise RuntimeError（主算子未注册 GE converter）。PKI 为主算子注册了**真实可用的** GE converter（`torchair.ge.custom_op`），直接映射到 GE 图的 `PoolKeyIndexer` 算子节点。

#### 6.3.5 注册导出 `ops/__init__.py`

在末尾追加：
```python
from .pool_key_indexer import pool_key_indexer
from .graph_convert.graph_convert_pool_key_indexer import convert_pool_key_indexer
```

### 6.4 顶层调用方式

适配完成后，模型侧调用方式：
```python
import cann_ops_transformer
sparse_indices, sparse_values = cann_ops_transformer.pool_key_indexer(
    query, pool_key, weights, pool_tail_k,
    actual_seq_q=cu_seqlens_q, actual_seq_k=cu_seqlens_k,
    block_table=block_table,           # PA 场景
    q_descale=q_scale, k_descale=k_scale,  # 量化场景
    layout_q="BSND", layout_k="BSND",
    topk=2048, pool_size=16, mask_mode=3, quant_mode=-1, return_value=True)
```

### 6.5 与既有 `torch_ops_extension/` 的关系（已删除）

算子目录下原有的 `torch_ops_extension/custom_ops/`（旧式 `torch.ops.custom.npu_pool_key_indexer`，基于 `NpuExtension` + `EXEC_NPU_CMD_V1`）已**整体删除**。原因：

1. **功能完全覆盖**：`cann_ops_transformer` 路径已补全 graph_convert converter（移植自旧 `torch_ops_extension` 的 `torchair.ge.custom_op` 实现），eager + torch.compile + torchair 图模式三条路径均可用
2. **命名违规**：旧封装使用 `npu_pool_key_indexer`（带 `npu_` 前缀），违反 `torch_extension_guidelines.md` §2.1「对外 api 接口及算子名一律不带 `npu_` 前缀」
3. **重复维护**：两套封装底层调用同一个 `aclnnPoolKeyIndexer`，存在维护成本和一致性风险

| 路径 | 入口 | 编译方式 | 适用场景 | 状态 |
|------|------|---------|---------|------|
| `cann_ops_transformer`（步骤5） | `cann_ops_transformer.pool_key_indexer` | `OpBuilder` JIT + `ACLNN_CMD` | eager + torch.compile + torchair 图模式 | ✅ 唯一路径 |
| ~~`torch_ops_extension/`~~（已删除） | ~~`torch.ops.custom.npu_pool_key_indexer`~~ | ~~`NpuExtension` 预编译 + `EXEC_NPU_CMD_V1`~~ | ~~torchair 静态图~~ | ❌ 已删除 |

### 6.6 验证方法

1. **JIT 编译**：首次调用 `cann_ops_transformer.pool_key_indexer(...)` 触发 ninja JIT 编译，无报错
2. **功能对齐**：eager 路径输出与 `tests/atk` pyaclnn 路径结果一致（相同输入、相同 rtol）
3. **Meta 推导**：`torch.compile` 模式下 FakeTensor 推导出的 output shape 与实际一致
4. **退化验证**：`pool_size=1, quant_mode=-1` 时输出维度 = `topk`（与步骤1 一致）
5. **量化路径**（依赖步骤4）：`quant_mode=0/1`，FP8 输入经 `TensorWrapper` dtype 修正后正确下发
6. **return_value**：True/False 两种模式，False 时 values shape 为 `{0}`

### 6.7 风险点

1. **schema 顺序**：`ACLNN_CMD` 的传参顺序必须与 `aclnnPoolKeyIndexerGetWorkspaceSize` 的 IR 声明严格一致（inputs → attrs → outputs），否则 `GetWorkspaceSize` 返回非 0
2. **optional tensor 处理**：`block_table` / `q_descale` / `k_descale` 直接传 `c10::optional<at::Tensor>`，`ConvertType` 自动处理 optional→`aclTensor*`(或 nullptr)；`pool_tail_k` / `actual_seq_q` / `actual_seq_k` 是 ValueDepend 输入，需转 `IntArrayRef`（aclnn 期望 `aclIntArray*`）
3. **layout 字符串生命周期**：`c10::string_view` → `std::string` → `const_cast<char*>` 需保证字符串在 `ACLNN_CMD` 执行期间存活（参考 LIV2 的局部变量做法）
4. **FP8 dtype 修正时机**：`FixPkiAclDtypes` 依赖步骤4 的 kernel 反量化实现完成；步骤4 未完成前，步骤5 可先只支持 `quant_mode=-1`，FP8 路径留 TODO
5. **输出 dtype 一致性**：`sparse_values` 必须为 FLOAT（规格要求），不能跟随 query dtype，C++ 构造时显式用 `at::kFloat`
6. **与旧式封装共存**：避免 `cann_ops_transformer` 与 `custom` 命名空间算子重名导致 dispatcher 冲突（命名空间不同，无冲突）

---

## 七、步骤间依赖与里程碑

```
步骤1 (退化适配)
  ├── 编译验证 ✅
  ├── 功能验证 (BSND/TND, maskMode 0/3, returnValue true/false)
  └── 里程碑M1: 退化场景与LI对齐

2补 (pool_key 0轴非连续修复)
  ├── 依赖M1
  ├── tiling 读取运行时 stride + CheckKeyContiguous
  ├── arch22 InitTilingData 补读 + KeyNd2NzForPA 改用 keyStride0
  ├── 所有架构均支持 PA 0轴非连续寻址
  └── 里程碑M1.5: PA 非连续输入正确

步骤2 (pool_size/pool_tail_k)
  ├── 依赖M1.5
  ├── 因果掩码修正 (参考LIV2 cmp_ratio)
  ├── pool_tail_k 传入kernel
  ├── 里程碑M2: pool级TopK正确 (输出pool索引)

步骤3 (索引展开与追加)
  ├── 依赖M2
  ├── Expand + Append 逻辑
  ├── 输出shape恢复为规格
  ├── 里程碑M3: 输出维度和内容符合规格

步骤4 (A5量化)
  ├── 依赖M3
  ├── Phase 0: tiling 架构门禁 + k_descale stride接线 + shape校验分裂 + 规格修订
  ├── Phase 1-2: 模板 SCALE_T 分发 + Cube 原生FP8 Mmad / LoadData-Mx (mode=1)
  ├── Phase 3: Vector scale 融合 (mode=0)
  ├── Phase 4: torch_extension FixPkiAclDtypes
  ├── Phase 5: golden/用例矩阵 (含 E8M0=0xFF NaN 传播)
  ├── 不影响arch22 (架构门禁拦截)
  └── 里程碑M4: 全功能完成 (规格全覆盖) ✅ 2026-08-28 达成 (量化 22/22, 非量化无回归)

步骤5 (torch_extension接口适配)
  ├── 依赖M3（非量化路径可独立适配）
  ├── 量化路径依赖M4
  ├── C++ 后端 + Python 前端 (OpBuilder JIT)
  ├── Meta 推导 + PrivateUse1 dispatch
  └── 里程碑M5: torch.compile 可用，eager 输出与 ATK 对齐
```

---

## 八、关键设计决策记录

| 决策 | 理由 |
|------|------|
| 输出类型 float（非 K_T） | 规格要求 sparseValuesOut 为 FLOAT |
| INVALID_VAL = 0xFF800000 (uint32) | 适配 float -inf |
| splitMFlag 恢复为运行时 | 与 LI 一致，退化场景下正确触发 |
| isLDOpen 恒 false（未启用） | 与 LI v1/LIV2/QLI 一致。arch22 用 isSparseCountOver2K+ProcessLD 做跨核归约，arch35 用 isLDOpen 门控 S2 切分（恒 false，未启用）。arch22/arch35 不一致是与 LI v1 一致的设计模式 |
| preTokens/nextTokens 删除 | PKI 不支持，arch22 删除引用 |
| poolKey 与 key 一致 | 两者输入都是压缩/聚合后的，S2 是压缩空间长度，kernel 无需内部聚合 |
| 步骤2 不展开索引 | 先验证 pool 级 TopK 正确性，再展开 |
| 2补: 所有架构均支持 pool_key 0轴非连续（不区分 arch35/arch22） | arch22 用户也可能使用 PA 非连续 kv_cache；且 LIV2 仅在 950 设置但 PKI 已对所有 config 设置 IgnoreContiguous，需全架构接线 |
| 2补: BSND/TND keyStride0 置 0 | kernel `KeyNd2Nz` 用 `tensorKeyOffset` 连续寻址，不读 keyStride0；只有 PA 的 `KeyNd2NzForPA` 需要 |
| 步骤4 仅 arch35 | A2/A3 不支持 FP8 |
| 步骤5 适配 cann_ops_transformer 而非改造 torch_ops_extension | OpBuilder JIT 模式支持 torch.compile，与 LIV2 体系一致；旧式封装违反命名规范（npu_ 前缀），功能被 cann_ops_transformer 完全覆盖后整体删除 |
| 步骤5 graph_convert 移植真实 GE converter | 旧 torch_ops_extension 的 `torchair.ge.custom_op("PoolKeyIndexer")` 已移植到 cann_ops_transformer graph_convert，图模式可用（不同于 LIV2 的 RuntimeError stub） |
| 步骤5 无独立 metadata 子算子 | PKI aclnn 接口无 metadata 分离（不同于 LIV2），单算子即可 |
| 步骤5 输出 values dtype 固定 FLOAT | 规格要求，不跟随 query dtype，C++ 显式 `at::kFloat` |
| 步骤5 register_fake 移除 | 与 LIV2 主算子一致，Meta impl 足够提供 shape 推导。register_fake 与 Meta 冲突，移除后 eager 模式正常 |
| 8补2: ValueDepend 输入转 IntArrayRef | `pool_tail_k`/`actual_seq_q`/`actual_seq_k` 在 op_def 中声明为 ValueDepend(OPTIONAL)，aclnn API 期望 aclIntArray*（非 aclTensor*），C++ wrapper 需将 tensor 转 CPU+contiguous+IntArrayRef |
| 8补2: workspace 总量含 16MB 系统保留区 | arch22 的 `GetUserWorkspace()` 从 base+16MB 开始，host tiling 必须加 `GetLibApiWorkSpaceSize()`，否则 kernel 越界踩踏系统区 |
| 8补: quant_mode +1 偏移映射 | def.cpp 属性 `-1` 是用户接口约定不改；tiling.h `int32_t` 保持一致；TilingData `uint32_t` 和模板 Tiling Key 通过 `+1` 统一映射为非负值，实现用户接口与 kernel 内部编码解耦 |
| 8补: 模板常量加 TPL_ 前缀 | `PKI_TPL_QUANT_*` / `PKI_TPL_MASK_*` 消除与 tiling.h `constexpr` 同名冲突，防止 `#define` 静默覆盖 |
| 8补: ExpandAndAppendIndices 值传递 | `LocalTensor` 是轻量包装器（指针+偏移），按值传递无性能开销，且解决临时值无法绑定左值引用问题 |
| 步骤4: mxFP8 语义与 QLIv2 对齐 | descale 每 32 个 D 元素一组 E8M0，shape 带 [D/64,2] 尾维；规格文档 aclnnKpoolIndexer.md 需同步修订（Phase 0） |
| 步骤4: E8M0=0xFF 不拦截 | 按 NaN 传播（与 QLIv2 golden 一致，host/kernel 均无 0xFF 校验） |
| 步骤4: 双路径而非反量化成FP16 | mode=0 原生FP8 Mmad + Vector scale 融合（scale 可从点积提出，数学精确）；mode=1 LoadData-Mx 硬件反量化（scale 不可提出）。推翻原蓝图的 "DequantQuery 先转FP16 再 Mmad" 设想 |

---

## 八·补、910B 编译修复（已实施）

在 ascend910B 上执行 `bash build.sh --pkg --soc=ascend910b --vendor_name=custom --ops=pool_key_indexer` 编译时发现并修复了 15 类编译问题。详细记录见 `docs/build_fix_record.md`。

### 修复类别总览

| 类别 | 文件 | 根因 | 修复方式 |
|------|------|------|---------|
| #1 宏参数过多 | infershape.cpp | `OP_LOGE_FOR_INVALID_VALUE_WITH_REASON` 传 6 参，宏仅接受 4 参 | 移除多余格式化参数，信息编码到 reason 字符串 |
| #2 常量重复定义 | tiling.h | `PKI_TOPK_6K` 在同文件定义两次 | 删除重复定义 |
| #3 Shape 类型不匹配 | tiling.cpp | `GetInputShape()` 返回 `StorageShape*`，代码用 `Shape*` | 改为 `StorageShape*` + `GetStorageShape().` 访问 |
| #4 TilingData 字段访问 | tiling.cpp | `TILING_DATA_FIELD_DEF` 生成私有字段+getter/setter，代码直接访问公有字段 | 改用 `set_` / `get_` 方法 |
| #5 Platform API 不存在 | tiling.cpp | `GetNpuArch`/`GetCoreNumPerAicore`/`CalcTschBlockDim`(静态) 不存在 | 改用 `GetCurNpuArch`/`GetCoreNumAiv`/`GetCoreNumAic`/`GetCoreMemSize(2参)`/实例方法 |
| #6 Workspace API 不存在 | tiling.cpp | `GetWorkspaceSize/SetWorkspaceSize` 不存在 | 改用 `GetWorkspaceSizes(1)` 返回指针 |
| #7 格式化字符串类型 | tiling.cpp | `%lu` 用于 `int64_t`（有符号） | 改为 `%ld` |
| #8 模板 Tiling Key 宏不存在 | template_tiling_key.h | `ASCENDC_TPL_INT_DECL/SEL` 不存在，导致文本泄漏 | 改用 `ASCENDC_TPL_UINT_DECL/SEL` + `ASCENDC_TPL_4_BW` + `ASCENDC_TPL_UI_LIST` |
| #9 未声明标识符 | arch22/kernel.h | Init 参数名与函数体引用名不一致 | 统一为 snake_case |
| #10 左值引用绑定临时值 | arch22/service_vector.h | `ExpandAndAppendIndices` 引用参数接收临时 `LocalTensor` | 改为值传递 |
| #11 宿主编译缺语句终结符 | template_tiling_key.h | `ASCENDC_TPL_ARGS_DECL/SEL` 宿主模式展开为 `static` 变量缺 `;` | 宏调用末尾补 `;` |
| #12 缺失 GET_TPL_TILING_KEY | tiling.cpp | tiling 未调用 `GET_TPL_TILING_KEY` + `SetTilingKey` | 补充 8 参数 Tiling Key 计算和 SetTilingKey |
| #13 quant_mode 值冲突+同名重定义 | tiling.h + template_tiling_key.h + tiling.cpp | `int32_t(-1)` → `uint32_t(0xFFFFFFFF)` + `PKI_QUANT_NONE` 同名双重定义 | 模板常量加 `TPL_` 前缀 + `quantMode + 1` 偏移映射 |
| #14 左值引用绑定临时值 | arch35/service_vector.h | 同 #10（arch35 遗漏） | 改为值传递 |
| #15 mask 常量同名冲突 | template_tiling_key.h | `PKI_MASK_*` 同名双重定义 | 模板常量加 `TPL_` 前缀 |

### quant_mode 值映射设计

```
用户接口 (def.cpp)     host 内部 (tiling.h)      TilingData / 模板 (template_tiling_key.h)
                      info.quantMode            tplQuantMode = quantMode + 1
─────────────────     ─────────────────────     ─────────────────────────────────
quant_mode = -1   →   -1                  →   0  (PKI_TPL_QUANT_NONE)
quant_mode = 0    →    0                  →   1  (PKI_TPL_QUANT_FP8_PER_TOKEN)
quant_mode = 1    →    1                  →   2  (PKI_TPL_QUANT_MXFP8)
```

**设计原则**：def.cpp 属性值（`-1`）是用户接口约定不改；tiling.h `int32_t` 保持与属性一致；TilingData `uint32_t` 和模板 Tiling Key 通过 `+1` 偏移统一映射，实现用户接口与 kernel 内部编码解耦。

### 修复文件清单

| 文件 | 修改内容 |
|------|----------|
| `op_host/pool_key_indexer_infershape.cpp` | 宏参数从 6 改为 4 |
| `op_host/pool_key_indexer_tiling.h` | 删除重复 `PKI_TOPK_6K` |
| `op_host/pool_key_indexer_tiling.cpp` | StorageShape + set_/get_ + Platform API + Workspace API + %ld + GET_TPL_TILING_KEY + SetTilingKey + quant_mode +1 映射 |
| `op_kernel/pool_key_indexer_template_tiling_key.h` | UINT_DECL + TPL_ 前缀 + 末尾分号 |
| `op_kernel/arch22/pool_key_indexer_kernel.h` | 参数名 snake_case 对齐 |
| `op_kernel/arch22/pool_key_indexer_service_vector.h` | `ExpandAndAppendIndices` 值传递 |
| `op_kernel/arch35/pool_key_indexer_service_vector.h` | `ExpandAndAppendIndices` 值传递 |
| `op_host/pool_key_indexer_def.cpp` | 无需修改（属性值 -1 通过 tiling.cpp +1 映射） |

### 参考模式来源

- `attention/sparse_flash_mla` — `ASCENDC_TPL_UINT_DECL` 正确用法 + `GET_TPL_TILING_KEY` 调用模式
- `attention/flash_attention_score` — `ASCENDC_TPL_UINT_DECL` 多值列表 + Platform API
- `ffn/ffn` — `GetCoreNumAiv/Aic`、`GetCoreMemSize` 2 参、`CalcTschBlockDim` 实例方法
- `common/src/op_host/tiling_util.cpp` — `GetCurNpuArch()` 正确用法
- `attention/lightning_indexer/op_host/lightning_indexer_tiling.cpp` — dtype 到模板值的 if-else 映射模式

---

## 八·补2、运行时崩溃修复（已实施）

> 2026-08-20 在 910B3 + CANN 9.1.0 环境，编译修复后算子所有用例运行即崩溃（EZ9999 `MTE accesses an invalid GM address`），pytest 12 用例 0 通过。详细记录见 `docs/build_fix_record.md` 末尾"运行时崩溃修复记录"章节。

### 根因

**host tiling 的 workspace 总量漏加了 16MB 系统保留区。**

- arch22（`__NPU_ARCH__ == 2201`）上 `GetUserWorkspace()` 实现为 `workspace + RESERVED_WORKSPACE`，其中 `RESERVED_WORKSPACE = 16MB`（`GetLibApiWorkSpaceSize()`），即 kernel 实际可用的 user workspace 从总分配的 +16MB 处开始
- LIV2 的 `DoTiling` 首行就是 `workspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize()` 再加上自身区域
- PKI 的 `CalcWorkspaceSize` 只计算了自身区域，且公式与 kernel 侧布局完全不符（host 按旧布局只算出 409600 字节；kernel 按自身公式需要 ~10.5MB）
- 各核按 `aiCoreIdx * 区域大小` 偏移写入时越界踩踏 16MB 保留区，导致崩溃点不固定、pc 漂移、mssanitizer 假阴性

### 修复内容

| 文件 | 修改 |
|------|------|
| `op_host/pool_key_indexer_tiling.cpp` `CalcWorkspaceSize` | 重写为镜像 kernel 侧布局（`scoreGm` + `ldScore` + `ldParam` 三段），用 `aicNum × 2 × mBaseSizeAlign × 512 × 4B` 等正确公式 |
| `op_host/pool_key_indexer_tiling.cpp` `DoTiling` | 补 `workspaceSize += ascendcPlatform.GetLibApiWorkSpaceSize()`（16MB 系统保留区） |
| `op_host/pool_key_indexer_tiling.cpp` 文件末尾 | 补 `IMPL_OP_OPTILING(PoolKeyIndexer).Tiling(TilingPoolKeyIndexer)` 注册 |
| `op_host/CMakeLists.txt` | `op_host_aclnnInner` → `op_host_aclnn`（与 LIV2/PKI 一致） |
| `torch_extension/.../csrc/pool_key_indexer.cpp` | ValueDepend 输入转 `IntArrayRef` + ACLNN_CMD 参数顺序修正 + `get_valid_tensor` 移除 |
| `torch_extension/.../pool_key_indexer.py` | 移除 `register_fake`（与 LIV2 主算子一致，Meta impl 足够） |
| `tests/pytest/simple_test.py` | 新增简化测试脚本（2 用例：BSND + TND） |
| `tests/atk/pool_key_indexer_golden.py` | `actual_seq_k` 语义修正（累计 pool 数，非 token 数） |

### C++ 后端关键修复

1. **ValueDepend 输入转 `IntArrayRef`**：`pool_tail_k` / `actual_seq_q` / `actual_seq_k` 在 op_def 中声明为 `ValueDepend(OPTIONAL)` + `DT_INT64`，aclnn 自动生成的 API 期望 `aclIntArray*`（非 `aclTensor*`）。修复将 1-D int64 tensor 转到 CPU、转 `at::kLong`、取 `contiguous()`，再构造 `at::IntArrayRef` 传给 `ACLNN_CMD`
2. **ACLNN_CMD 参数顺序修正**：属性顺序从 `layoutQ, layoutK, topk, poolSize, maskMode, quantMode, returnValue` 改为 `topk, poolSize, layoutQ, layoutK, maskMode, quantMode, returnValue`（对齐 aclnn IR 声明）
3. **`block_table` / `q_descale` / `k_descale` 直接传 `c10::optional<at::Tensor>`**：不再用 `get_valid_tensor` 转换，`ConvertType` 直接处理 optional

### 验证结果

| 项目 | 结果 |
|------|------|
| `simple_test.py` 2 用例 | 全部 PASSED（历史 0 通过） |
| 9 组 shape 变体（1核/8核/64核、S2=4096、topk=2048/4096、B=2、pool16+causal、G=1） | 9/9 通过 |
| 重复运行 ×3 | 稳定通过 |
| 官方 pytest `test_pool_key_indexer.py` | 9/12 通过（3 个 TND 失败为 golden 脚本侧 `actual_seq_k` 语义问题） |
| LIV2 同包回归 | 无影响 |

### 经验教训

1. **MIX 类算子 workspace 总量 = `GetLibApiWorkSpaceSize()`（系统保留区，arch22=16MB）+ 算子自用区域**，二者缺一不可
2. host/kernel 的 workspace 布局公式必须逐字段镜像维护
3. 越界踩踏系统保留区的故障表现极具迷惑性：崩溃核不固定、pc 漂移、拦截式检测工具假阴性。`mssanitizer` 的 `MEMORY_RECORD` 是对照"kernel 实际访问地址 vs 运行时分配边界"的最有效手段
4. ValueDepend 输入在 aclnn 层面期望 `aclIntArray*`，C++ wrapper 需将 tensor 转 `IntArrayRef`

---

## 九、规格约束速查

### 输入约束

| 参数 | 数据类型 | 维度 | 约束 |
|------|---------|------|------|
| query | FP16/BF16/FP8_E4M3FN(950) | BSND:(B,S1,N1,D) / TND:(T1,N1,D) | 不支持空tensor，N1≤64，headDim=128 |
| poolKey | 同query | PA:(blockNum,blockSize,N2,D) / BSND:(B,S2,N2,D) / TND:(T2,N2,D) | 不支持空tensor，N2=1，支持0轴非连续 |
| weights | FP16/BF16 | BSND:(B,S1,N1) / TND:(T1,N1) | 不支持空tensor |
| poolTailK | INT64 | (B,) | 取值[0,pool_size-1] |
| actualSeqLengthsQueryOptional | INT64 | (B,) | TND必传，前缀和 |
| actualSeqLengthsKeyOptional | INT64 | (B,) | TND/PA必传，TND前缀和/PA非前缀和 |
| blockTableOptional | INT32 | (B,maxBlockNumPerSeq) | PA必传 |
| qDescaleOptional | FLOAT/FLOAT8_E8M0 | 同query维度/同key维度 | 量化必传 |
| kDescaleOptional | 同上 | 同key维度 | 量化必传 |

### 属性约束

| 属性 | 类型 | 约束 |
|------|------|------|
| topk | INT64 | [1,2048] ∪ {3072,4096,5120,6144,7168,8192} |
| poolSize | INT64 | [1,128] |
| maskMode | INT64 | 0(defaultMask) / 3(rightDownCausal) |
| quantMode | INT64 | -1(不量化) / 0(FP8 per-token-head) / 1(mxFP8) |
| returnValue | BOOL | 是否输出sparseValuesOut |

### 输出约束

| 输出 | 数据类型 | 维度 |
|------|---------|------|
| sparseIndicesOut | INT32 | BSND:(B,S1,topk+pool_size-1) / TND:(T1,topk+pool_size-1) |
| sparseValuesOut | FLOAT | BSND:(B,S1,topk//pool_size) / TND:(T1,topk//pool_size) |

### 维度一致性约束

- poolKey 的 N2 必须为 1，D 必须与 query 的 headDim 一致（128）
- weights/qDescaleOptional 的各维度必须与 query 对应维度一致
- kDescaleOptional 的各维度必须与 poolKey 对应维度一致
- poolTailK/actualSeqLengthsKeyOptional 的元素数必须等于 B
- 前缀和末值：actualSeqLengthsQueryOptional 最后元素 = query 的 T1；actualSeqLengthsKeyOptional(TND) 最后元素 = poolKey 的 T2
- 输出 sparseIndicesOut/sparseValuesOut 的 B、S1（或 T1）必须与 query 对应维度一致

### 平台支持

| 平台 | FP16/BF16 | FP8_E4M3FN | 量化 |
|------|-----------|------------|------|
| Ascend 950PR/950DT (arch35) | ✅ | ✅ | ✅ (quantMode 0/1) |
| Atlas A3 (arch22) | ✅ | ❌ | ❌ |
| Atlas A2 (arch22) | ✅ | ❌ | ❌ |
