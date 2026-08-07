# PoolKeyIndexer 算子开发蓝图

## 一、总体策略

```
步骤1: 退化场景适配 ──→ 步骤2: pool_size/pool_tail_k扩展 ──→ 步骤3: 索引展开与追加 ──→ 步骤4: A5量化支持
(quantMode=-1,          (参考LIV2 cmp_ratio/               (Expand + Append)           (quantMode=0/1)
 pool_tail_k=0,          cmp_residual_k,                   输出维度:                    arch35 only
 pool_size=1)            不做索引展开)                       topk+pool_size-1)
```

**核心原则**：每步可独立编译验证，前步通过后才进入下一步。

**规格参考**：`docs/aclnnKpoolIndexer.md` 全程参照。

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

将步骤2 输出的 pool 索引展开为原始 token 索引，并追加尾块有效 token。

```
输入(TopK结果): [pool_idx_0, pool_idx_1, ..., pool_idx_{sparseCount-1}]

展开: 每个 pool_idx 展开为 pool_size 个 token 索引
  pool_idx_j → [j*pool_size, j*pool_size+1, ..., j*pool_size+pool_size-1]

追加: 最后追加 pool_tail_k 个尾部 token 索引
  [L_orig - pool_tail_k, ..., L_orig - 1]

输出: [展开的 topk 个索引] + [pool_tail_k 个尾索引]
     = topk + pool_tail_k 个索引（最大 topk + pool_size - 1）
```

### 4.2 关键修改点

#### 4.2.1 输出 shape 恢复为规格
- sparse_indices 最后一维 = `topk + pool_size - 1`
- sparse_values 最后一维 = `topk / pool_size` = sparseCount

#### 4.2.2 arch35 service_vector.h 新增展开逻辑

在 `ProcessTopK` 之后、输出之前，新增 `ExpandAndAppendIndices` 函数。

**向量化设计核心思路**：

展开操作 `tokenIndices[k*P + p] = poolIndices[k] * P + p` 可分解为三步向量运算：

```
Step 1: Muls  — baseIndices[k] = poolIndices[k] * poolSize        （1条指令处理 sparseCount 个）
Step 2: Brcb  — expandedBases[k*P + p] = baseIndices[k]           （1条指令广播，每个元素重复 P 次）
Step 3: Add   — tokenIndices = expandedBases + cyclicOffset        （1条指令处理 sparseCount*P 个）
```

其中 `cyclicOffset` 是 `[0,1,...,P-1, 0,1,...,P-1, ...]` 的循环偏移模式，可预先构造。

```cpp
__aicore__ void ExpandAndAppendIndices(
    LocalTensor<int32_t>& poolIndices,   // [sparseCount] TopK 输出的 pool 索引
    LocalTensor<int32_t>& tokenIndices, // [topk+poolSize-1] 输出 token 索引
    LocalTensor<int32_t>& workBuf,       // 临时 buffer (>= sparseCount*poolSize + poolSize)
    uint32_t sparseCount,
    uint32_t poolSize,
    int32_t poolTailK,
    int32_t L_orig)
{
    uint32_t totalExpand = sparseCount * poolSize;

    // ---- 1. 构造循环偏移模板 [0, 1, ..., poolSize-1] ----
    // poolSize <= 128，用标量 SetValue 构造一次即可
    LocalTensor<int32_t> offsetTpl = workBuf;                    // [0, poolSize)
    for (uint32_t p = 0; p < poolSize; p++) {
        offsetTpl.SetValue(p, static_cast<int32_t>(p));
    }

    // ---- 2. 构造循环偏移模式 [0,1,...,P-1, 0,1,...,P-1, ...] ----
    // 用 DataCopy 将 offsetTpl 复制 sparseCount 份
    LocalTensor<int32_t> cyclicOffset = workBuf[poolSize];       // [0, totalExpand)
    for (uint32_t k = 0; k < sparseCount; k++) {
        DataCopy(cyclicOffset[k * poolSize], offsetTpl, poolSize);
    }

    // ---- 3. Muls: base = poolIndices * poolSize ----
    LocalTensor<int32_t> baseIndices = workBuf[poolSize + totalExpand];
    Muls(baseIndices, poolIndices, static_cast<int32_t>(poolSize), sparseCount);

    // ---- 4. Brcb: 广播每个 base 重复 poolSize 次 ----
    // Brcb 将 [b0, b1, ...] 广播为 [b0,b0,..(P次), b1,b1,..(P次), ...]
    LocalTensor<int32_t> expandedBases = tokenIndices;           // 直接写入输出位置
    Brcb(expandedBases, baseIndices, poolSize, sparseCount);     // 广播因子 = poolSize

    // ---- 5. Add: expandedBases + cyclicOffset = 最终 token 索引 ----
    Add(tokenIndices, expandedBases, cyclicOffset, totalExpand);

    // ---- 6. 追加尾块 [L_orig - poolTailK, ..., L_orig - 1] ----
    if (poolTailK > 0) {
        uint32_t tailOffset = totalExpand;
        // Duplicate 填充起始值，Add 偏移
        LocalTensor<int32_t> tailOffsetTpl = workBuf;            // 复用 offsetTpl 区域
        for (int32_t t = 0; t < poolTailK; t++) {
            tailOffsetTpl.SetValue(t, t);
        }
        Duplicate<int32_t>(tokenIndices[tailOffset],
                           L_orig - poolTailK, poolTailK);
        Add(tokenIndices[tailOffset], tokenIndices[tailOffset],
            tailOffsetTpl, poolTailK);
    }
}
```

**指令开销分析**：

| 步骤 | 指令 | 数量 | 说明 |
|------|------|------|------|
| 偏移模板构造 | SetValue | poolSize (≤128) | 一次性，标量 |
| 循环偏移复制 | DataCopy | sparseCount | 每次复制 poolSize 个 |
| Muls | Muls | 1 | 批量标量乘 |
| 广播 | Brcb | 1 | 批量广播 |
| Add | Add | 1 | 批量向量加 |
| 尾块 Duplicate | Duplicate | 1 | 填充起始值 |
| 尾块 Add | Add | 1 | 加偏移 |

对比标量方案 `O(sparseCount * poolSize)` 次标量操作，向量化方案仅需 `O(sparseCount)` 次 DataCopy + 常数条向量指令。

**Brcb 约束与 Fallback**：

`Brcb` 指令的广播因子需要满足对齐要求（通常为 8 的倍数）。当 `poolSize` 不是 8 的倍数时，使用 Fallback 方案：

```cpp
// Fallback: 分块处理，每块用 Duplicate + Add 替代 Brcb
constexpr uint32_t VEC_BLK = 8;  // int32 对齐 32B
for (uint32_t k = 0; k < sparseCount; k += VEC_BLK) {
    uint32_t blkLen = Min(VEC_BLK, sparseCount - k);
    // Muls: base = poolIndices[k:k+blkLen] * poolSize
    Muls(baseBlk, poolIndices[k], poolSize, blkLen);
    // 对每个 base 用 Duplicate 填充 + Add 偏移
    for (uint32_t i = 0; i < blkLen; i++) {
        Duplicate(tokenIndices[outOff], baseBlk.GetValue(i), poolSize);
        Add(tokenIndices[outOff], tokenIndices[outOff], offsetTpl, poolSize);
        outOff += poolSize;
    }
}
```

Fallback 内层仍有标量循环，但每个元素内部是 `Duplicate + Add`（2条向量指令），远优于纯标量逐元素赋值。

#### 4.2.3 arch22 同样新增展开逻辑
- 在 Vector 核的输出阶段执行，逻辑与 arch35 一致
- arch22 的向量指令 API 与 arch35 略有差异，需适配

#### 4.2.4 infershape 修正
- 当前 infershape 已正确输出 `topk + pool_size - 1` 和 `topk / pool_size`
- 步骤2 临时改为 sparseCount 的部分需在步骤3 恢复

### 4.3 验证方法
- pool_size=2/4/8/16，pool_tail_k=0~pool_size-1
- 验证输出维度 = topk + pool_size - 1
- 验证展开索引正确性
- 验证尾块追加正确性
- maskMode=3 时验证因果掩码后的展开
- Brcb 路径和 Fallback 路径均需验证（poolSize 为 8 的倍数和非倍数）

### 4.4 风险点
1. **UB 空间**：workBuf 需容纳 `poolSize + 2*sparseCount*poolSize` 个 int32，最大约 `128 + 2*8192*128 ≈ 2MB`，可能超 UB。需分批处理或复用输出 buffer
2. **Brcb 对齐约束**：广播因子需为 8 的倍数（arch35）或 16 的倍数（arch22），非对齐时走 Fallback
3. **无效索引填充**：当 sparseCount < topk/pool_size 时（S2 < sparseCount），剩余位置填 -1
4. **DataCopy 对齐**：输出 int32 索引，需对齐 32B
5. **int32 向量指令支持**：确认 arch35/arch22 的 `Muls`/`Add`/`Brcb` 对 int32 类型的支持

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
- `keyDequantScaleStride0` 已在 ConstInfo 中，需在 tiling 中计算

### 5.5 验证方法
- quantMode=0/1，FP8_E4M3FN 输入
- 对比：FP8 反量化后与直接 FP16 输入的结果（精度误差在允许范围内）
- q_descale/k_descale 的各种 shape（BSND/TND/PA_BBND）
- 退化验证：quantMode=-1 时不受影响

### 5.6 风险点
1. **反量化性能**：FP8→FP16 转换增加计算量和 UB 占用，需评估对流水的影响
2. **mxFP8 指数处理**：FLOAT8_E8M0 是纯指数格式，需确认 Cast/Mul 指令支持
3. **双缓冲影响**：反量化打乱 Q/K 的双缓冲流水，需重新设计流水
4. **descale 读取**：k_descale 在 PA 场景的 stride 计算（keyDequantScaleStride0）

---

## 六、步骤间依赖与里程碑

```
步骤1 (退化适配)
  ├── 编译验证 ✅
  ├── 功能验证 (BSND/TND, maskMode 0/3, returnValue true/false)
  └── 里程碑M1: 退化场景与LI对齐

步骤2 (pool_size/pool_tail_k)
  ├── 依赖M1
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
```

---

## 七、关键设计决策记录

| 决策 | 理由 |
|------|------|
| 输出类型 float（非 K_T） | 规格要求 sparseValuesOut 为 FLOAT |
| INVALID_VAL = 0xFF800000 (uint32) | 适配 float -inf |
| splitMFlag/isLDOpen 恢复为运行时 | 与 LI 一致，退化场景下正确触发 |
| preTokens/nextTokens 删除 | PKI 不支持，arch22 删除引用 |
| poolKey 与 key 一致 | 两者输入都是压缩/聚合后的，S2 是压缩空间长度，kernel 无需内部聚合 |
| 步骤2 不展开索引 | 先验证 pool 级 TopK 正确性，再展开 |
| 步骤4 仅 arch35 | A2/A3 不支持 FP8 |

---

## 八、规格约束速查

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
