# PoolKeyIndexer 算子开发蓝图

## 一、总体策略

```
步骤1: 退化场景适配 ──→ 2补: pool_key 0轴非连续修复(已实施) ──→ 步骤2: pool_size/pool_tail_k扩展 ──→ 步骤3: 索引展开与追加 ──→ 步骤4: A5量化支持 ──→ 步骤5: torch_extension接口适配
(quantMode=-1,          (tiling读运行时stride +                    (参考LIV2 cmp_ratio/               (Expand + Append)           (quantMode=0/1)           (参考LIV2 OpBuilder模式
 pool_tail_k=0,          arch22接线KeyNd2NzForPA)                    cmp_residual_k,                   输出维度:                    arch35 only              JIT编译 + PrivateUse1 dispatch)
 pool_size=1)                                                       不做索引展开)                       topk+pool_size-1)
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
- `trunkLen` / `topkCount_` 已使用 `sparseCount`（= topk/pool_size），退化时等价
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

### 5.1 设计思路

在 arch35 (Ascend 950) 上实现 FP8 反量化：

| quantMode | 输入类型 | 反量化系数类型 | 反量化方式 |
|-----------|---------|-------------|-----------|
| -1 | FP16/BF16 | 无 | 直接使用 |
| 0 | FP8_E4M3FN | FLOAT | per-token-head 反量化 |
| 1 | FP8_E4M3FN | FLOAT8_E8M0 | mxFP8 块反量化 |

### 5.2 反量化计算

```
quantMode=0 (FP8 per-token-head):
  Q = Dequant(query, q_descale)    // q_descale shape: (B,S1,N1) 或 (T1,N1)
  K_pool = Dequant(poolKey, k_descale)  // k_descale shape: (B,S2,N2) 或 (T2,N2) 或 (blockNum,blockSize,N2)
  // 反量化公式: value_fp16 = fp8_value * scale
  // 每个 token-head 有独立的 scale

quantMode=1 (mxFP8):
  // 反量化系数为 FLOAT8_E8M0（指数格式）
  // 反量化公式: value_fp16 = fp8_value * 2^(scale_exp - 127)
  // 每 32 个元素共享一个 scale（block 量化）
```

### 5.3 关键修改点

#### 5.3.1 op_host（已有，无需修改）
- def.cpp 950 配置已定义 FP8_E4M3FN 和 FLOAT8_E8M0 数据类型
- tiling.cpp 已有 quantMode 驱动的 dtype/descale 校验
- ValueDepend + ExtendCfgInfo 已配置

#### 5.3.2 arch35 kernel 新增反量化

**kernel.h `Init`**：
- 已接收 `qDescale`/`kDescale` 参数，需传递给 service
- 在 ConstInfo 中添加 `quantMode`（已有）和 descale GM 偏移

**service_cube.h 新增 `DequantInput`**：
```cpp
// 在 ComputeMm1 之前，对 Q 和 K 做 FP8 → FP16 反量化
__aicore__ void DequantQuery(LocalTensor<half>& qOut, LocalTensor<fp8_e4m3fn>& qIn,
                              LocalTensor<float>& qScale, uint32_t count) {
    // quantMode=0: per-token-head, scale shape (...,N1)
    //   for each token t, head h: qOut[t,h,:] = qIn[t,h,:].ToFloat() * qScale[t,h]
    // quantMode=1: mxFP8, scale shape 每32元素1个
    //   for each block of 32: qOut[block] = qIn[block].ToFloat() * 2^(scale[block]-127)
}
```

**关键实现**：
- FP8 → FP16 转换：使用 `Cast` 指令
- per-token-head 乘法：`Mul` 指令，scale 广播到 D 维度
- mxFP8 指数转换：`Float32 → half`，先做 `2^(exp-127)` 再乘

**反量化位置**（计算流水）：
```
Cube 核:
  1. Load Q FP8 from GM → UB
  2. Dequant Q: FP8 × scale → FP16 (新增)
  3. Load K FP8 from GM
  4. Dequant K: FP8 × scale → FP16 (新增)
  5. Mmad(Q_fp16, K_fp16) → Score (现有)
```

#### 5.3.3 arch35 service_vector.h
- `MulWeightAndReduceSum` 中的权重 W 仍为 FP16/BF16（规格要求），无需反量化
- score 类型从 uint32 不变（反量化后的 FP16 计算，结果仍量化为 uint32）

#### 5.3.4 arch22 不修改
- A2/A3 不支持 FP8，quantMode 只能是 -1

### 5.4 tiling 修改

**CalcWorkspaceSize**：
- 新增反量化中间 buffer（FP16 Q 和 K 的 UB 空间）

**k_descale 0 轴非连续接线**（参考 QLI `quant_lightning_indexer_v2_tiling.cpp:228-236, 1270-1295, 1375-1380`）：

`k_descale` 的 `IgnoreContiguous()` 设置合理（与 QLI 一致），因为 k_descale 与 pool_key 在 token 维度一一对应，PA 场景必须跟随 pool_key 的 0 轴非连续。当前 tiling 未读取 k_descale 运行时 stride，需修复：

| 改动 | 位置 | 参考QLI |
|------|------|---------|
| 添加 `keyDequantScaleStridesVec_` 成员 | `tiling.h` `PoolKeyIndexerTiling` | `:180, :264` |
| 读取运行时 stride | `tiling.cpp` `ParseAndCheckParams` | `:228-236` |
| 校验 k_descale 非连续（PA 仅 0 轴） | `CheckKeyContiguous()` 内 | `:1270-1295` |
| 填值 `keyDequantScaleStride0` | `CalcTilingParams` | `:1375-1380` |

填值逻辑需区分 quant_mode（shape 不同）：
- `quant_mode=0` (per-token-head): k_descale shape = (B, S2, N2, 1)
- `quant_mode=1` (mxFP8): k_descale shape = (B, S2, N2, D/group, group)
- PA 路径用 `keyDequantScaleStridesVec_[0]`；mxFP8 fallback 用 `blockSize * (headDim / MX_SCALE_GROUP_SIZE)`；其余置 0

**kernel 侧 k_descale 寻址**：arch35/arch22 反量化路径需用 `keyDequantScaleStride0` 按 block_table 寻址 k_descale（当前 kernel 无反量化逻辑，随步骤4一起实现）。

### 5.5 验证方法
- quantMode=0/1，FP8_E4M3FN 输入
- 对比：FP8 反量化后与直接 FP16 输入的结果（精度误差在允许范围内）
- q_descale/k_descale 的各种 shape（BSND/TND/PA_BBND）
- k_descale 0 轴非连续场景（构造 blockFusion tensor，参考 QLI golden）
- 退化验证：quantMode=-1 时不受影响

### 5.6 风险点
1. **反量化性能**：FP8→FP16 转换增加计算量和 UB 占用，需评估对流水的影响
2. **mxFP8 指数处理**：FLOAT8_E8M0 是纯指数格式，需确认 Cast/Mul 指令支持
3. **双缓冲影响**：反量化打乱 Q/K 的双缓冲流水，需重新设计流水
4. **k_descale 非连续寻址**：PA 场景 k_descale 需按 `keyDequantScaleStride0` 寻址，且需区分 quant_mode 的不同 shape

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
| `torch_extension/cann_ops_transformer/ops/pool_key_indexer.py` | Python 前端：`OpBuilder` + schema + Meta + register_fake + PrivateUse1 dispatch | `ops/lightning_indexer.py` |
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
    TORCH_CHECK(poolKey.numel() > 0, "Tensor poolKey is empty.");

    std::string layoutQStr(layoutQ), layoutKStr(layoutK);
    auto [sparseIndices, sparseValues] = ConstructPoolKeyIndexerOutput(
        query, topk, poolSize, layoutQStr, returnValue);

    auto device = query.device();
    auto actualSeqQVal = get_valid_tensor(actualSeqQ, device);
    auto actualSeqKVal = get_valid_tensor(actualSeqK, device);
    auto blockTableVal = get_valid_tensor(blockTable, device);
    auto qDescaleVal   = get_valid_tensor(qDescale, device);
    auto kDescaleVal   = get_valid_tensor(kDescale, device);

    char *layoutQPtr = const_cast<char *>(layoutQStr.c_str());
    char *layoutKPtr = const_cast<char *>(layoutKStr.c_str());

    ACLNN_CMD(aclnnPoolKeyIndexer,
              query, poolKey, weights, poolTailK,
              actualSeqQVal, actualSeqKVal, blockTableVal,
              qDescaleVal, kDescaleVal,
              layoutQPtr, layoutKPtr,
              topk, poolSize, maskMode, quantMode, returnValue,
              sparseIndices, sparseValues);
    return {sparseIndices, sparseValues};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("pool_key_indexer", &PoolKeyIndexer, "pool_key_indexer");
}
```

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
- 已添加 `@torch.library.register_fake("cann_ops_transformer::pool_key_indexer")` 支持 torch.compile FakeTensor 推导
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
2. **optional tensor 处理**：`get_valid_tensor` 将 None 转为空 int32 tensor，需确认 aclnn 侧对空 tensor 的容忍度（LIV2 已验证该模式可行）
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
  ├── arch35 反量化逻辑
  ├── 不影响arch22
  └── 里程碑M4: 全功能完成 (规格全覆盖)

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
| splitMFlag/isLDOpen 恢复为运行时 | 与 LI 一致，退化场景下正确触发 |
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

---

## 九、规格约束速查

### 输入约束

| 参数 | 数据类型 | 维度 | 约束 |
|------|---------|------|------|
| query | FP16/BF16/FP8_E4M3FN(950) | BSND:(B,S1,N1,D) / TND:(T1,N1,D) | 不支持空tensor，N1≤64，headDim=128 |
| poolKey | 同query | PA:(blockNum,blockSize,N2,D) / BSND:(B,S2,N2,D) / TND:(T2,N2,D) | 不支持空tensor，N2=1，支持0轴非连续 |
| weights | FP16/BF16 | BSND:(B,S1,N1) / TND:(T1,N1) | 不支持空tensor |
| poolTailK | INT32 | (B,) | 取值[0,pool_size-1] |
| actualSeqLengthsQueryOptional | INT32 | (B,) | TND必传，前缀和 |
| actualSeqLengthsKeyOptional | INT32 | (B,) | TND/PA必传，TND前缀和/PA非前缀和 |
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
