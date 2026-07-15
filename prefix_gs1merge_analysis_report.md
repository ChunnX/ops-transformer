# KV Prefix + BAND_MODE 精度问题分析报告

## 1. 目标与约束

修复 Ascend arch35 上 `aclnnFusedInferAttentionScoreV5` 在 **KV prefix + BAND_MODE (sparseMode=4)** 下的精度问题。

约束：
- 只提线上检视意见，不提 PR/issue
- 不要用裸指针访问 UB 内存（会导致 AIC error）
- 编辑 arch35 下的 `.h` 文件不要用 Edit 工具（会触发 auto-formatter），用 sed/python 脚本
- 定位时不需要考虑当前 case 走不到的场景（如 gs 合轴、dn 路径）
- mask 数据：左下三角 = 0（masked），右上三角 = 1（visible），可用于直接确认 offset 是否正确

---

## 2. 测试用例参数

| 参数 | 值 |
|------|-----|
| 算子 | aclnnFusedInferAttentionScoreV5 |
| 输入布局 | BSND |
| 数据类型 | FP16 |
| B | 28 |
| S1 | 2（decode） |
| S2(max) | 7168 |
| N（Q heads） | 162 |
| D | 57 |
| sparseMode | 4（BAND_MODE） |
| preTokens | 946 |
| nextTokens | -1 |
| KV prefix len | 5852 |
| actualSeqLengthsKV[0] | 69（非 prefix 部分） |
| scale | 0.132453 |
| softmax_lse_flag | True |
| attenMask | [2048, 2048] uint8 全矩阵，左下三角 = 0，右上三角 = 1 |

### Block 0 关键参数（从日志提取）

- `s2StartIdx = 4864`（全局 KV 起始偏移）
- `s2EndIdx = 5920`
- `s2LoopLimit = 4`（s2L = 0..4 共 5 个 block）
- `s2BaseSize = 256`
- `actualKVPrefixSize = 5852`
- `prefixLoopCount = 23`（= CeilDiv(5852, 256)）
- `actualS1Size = 2`，`actualS2Size = 69`
- `halfS1RealSize = 1`（gs1dealNum）
- `attenMaskS2Size = 2048`（mask 矩阵行步幅）

### s2L 与 KV 区域映射

| s2L | 全局 KV 位置 | 区域 | s2RealSize |
|-----|-------------|------|------------|
| 0 | 4864-5119 | prefix（4864 < 5852） | 256 |
| 1 | 5120-5375 | prefix | 256 |
| 2 | 5376-5631 | prefix | 256 |
| 3 | 5632-5851 | prefix（tail，220 < 256） | 220 |
| 4 | 5852-5919 | **非 prefix**（原始序列 0-67） | 68 |

> 当前日志覆盖 s2L=3（最后一个 prefix block）和 s2L=4（唯一的非 prefix block）。

---

## 3. 已确认并修复的 Bug

### Bug 1: s2Offset 公式错误（attenmask.h:459）— 已修复

**原始代码：**
```cpp
s2Offset = s2StartIdx + (s2LoopCount - prefixLoopCount) * s2BaseSize + actualKVPrefixSize;
```
计算结果：`4864 + (4-23)*256 + 5852 = 5852`（减去的是 padded prefix，错误）

**修复后：**
```cpp
s2Offset = s2StartIdx + s2LoopCount * s2BaseSize - actualKVPrefixSize;
```
计算结果：`4864 + 4*256 - 5852 = 36`（原始序列坐标，正确）

> Commit `366f1d5f6`。同样修复了 `ComputeOffsetForNoCompress` 中的对应公式。

### Bug 2: deltaN 符号错误（attenmask.h:436）— 已修复

**原始：** `deltaN -= actualKVPrefixSize;`
**修复：** `deltaN += actualKVPrefixSize;`

> Commit `366f1d6`。

### Bug 3: BAND_MODE delta 中 tokens 坐标系不匹配（attenmask.h:493-500）— 已修复

**问题：** `SetSparseParamBand`（infer_flash_attention_sparse.h:87）将 tokens 调整到**全局 KV 坐标系**（加了 prefix），但 s2Offset 修复后使用**原始序列坐标**，两者不匹配。

**修复：** 在 delta 计算中反向调整 tokens 到原始序列坐标：
```cpp
if constexpr (enableKVPrefix) {
    preTokensPerBatchAdj  += actualKVPrefixSize;  // 879 + 5852 = 6731... 见 §7
    nextTokensPerBatchAdj -= actualKVPrefixSize;  // 5918 - 5852 = 66
}
```

> Commit `026519c0d`。

### Bug 4: tiling 未禁用合轴（fused_infer_attention_score_tiling_impl.cpp:104）— 已修复

**问题：** `SetGSMerge` 不考虑 `systemPrefixMaxLen`，prefix 场景仍走合轴路径。

**修复：** `systemPrefixMaxLen > 0` 时禁用 `gsMergeFlag_` 和 `pfaMergeFlag_`。

> Commit `444ea1dda`。

### Bug 5: prefix block 0xFF fill — 已修复

不合轴 GQA 路径和非 GQA 路径的 prefix block 均填充 `0xFF`（全可见）。

> Commit `366f1d5f6`。

---

## 4. 核心未解决问题：非 prefix block maskOffset 指向错误位置

### 4.1 最新日志关键证据（MASK-gm 打印）

s2L=4（非 prefix block）的 `[MASK-gm]` 打印**定论性地确认了根因**：

```
[MASK-gm] s2L=4 gmAtDeltaOff=61440 val=0 gmAtCorrectOff=36 val=1 s1O=0 s2O=36
```

| 读取位置 | offset | GM 值 | 含义 |
|---------|--------|-------|------|
| **代码实际使用**（delta offset） | 61440 = 30 × 2048（row 30, col 0） | **0**（masked） | 错误！读到了 row 30 |
| **正确位置**（direct offset） | 36 = 0 × 2048 + 36（row 0, col 36） | **1**（visible） | 正确！s1=0 应该读 row 0 |

- 代码用 `ComputeOffsetForCausal(delta=30)` 得到 `maskOffset = 61440`，指向 attenMask 矩阵的 **row 30**
- 正确的 maskOffset 应为 `s1Offset × attenMaskS2Size + s2Offset = 0 × 2048 + 36 = 36`，指向 **row 0**
- GM 中 row 30 col 0 = 0（masked），row 0 col 36 = 1（visible）——与 BAND_MODE 语义一致（query s1=0 应能看到 KV s2=36）

### 4.2 mask 从一开始就是 0，不存在"覆盖"

最新日志中 s2L=4 的 mask 值在**所有阶段一致为 0**：

```
[MASK-gqa-nonpref] s2L=4 mask[0]=0 mask[1]=0 phyAddr=1ff00   ← DeQue 后立即读（PipeBarrier 之前）
[MASK-chk1]        s2L=4 mask[0]=0 phyAddr=1ff00              ← PipeBarrier<PIPE_MTE2> 之后
[MASK-chk2..5]     s2L=4 mask[0]=0                            ← 后续各检查点
[VEC-Nd-mask]      mask[0]=0 mask[1]=0                        ← VEC-Nd 计算阶段
```

mask=0 是从 GM offset 61440 **正确拷贝**回来的真实数据（GM 该位置值就是 0），从头到尾没有变化。

**对比 s2L=3（prefix block）—— 工作正常：**

```
[MASK-gqa-pref] s2L=3 fill=0xFF mask[0]=ff
[MASK-chk1..5]  s2L=3 mask[0]=ff
[VEC-Nd-mask]   mask[0]=ff mask[1]=ff        ← 全程 0xFF，正确
[VEC-Nd-post]   mx0=42.800434 sm0=1.058813   ← max/sum 合理
```

prefix block 填充 0xFF 后全程保持，计算结果合理。

### 4.3 maskOffset 计算过程追踪（s2L=4）

```
[MASK-offset] s2L=4 s2Off=36 deltaN=5785 deltaC=30 deltaPre=-916 maskOff=61440 computeMode=1 s1Off=0
```

逐步推导（均与日志吻合）：

| 步骤 | 公式 | 计算 | 结果 |
|------|------|------|------|
| s1Offset | `s1oIdx × s1BaseSize`，GQA 取模 | `0 % s1Size` | 0 |
| s2Offset | `s2StartIdx + s2L×s2BaseSize - actualKVPrefixSize` | `4864 + 4×256 - 5852` | 36 |
| deltaN | `actualS1Size - actualS2Size + actualKVPrefixSize` | `2 - 69 + 5852` | 5785 |
| preTokensPerBatchAdj | `preTokensPerBatch + actualKVPrefixSize` | `879 + 5852` | 6731 |
| nextTokensPerBatchAdj | `nextTokensPerBatch - actualKVPrefixSize` | `5918 - 5852` | 66 |
| deltaPre | `s1Offset - s2Offset - preTokensAdj - 1` | `0 - 36 - 6731 - 1` | -916（修正后值，见 §7 注） |
| deltaC | `s1Offset - s2Offset + nextTokensAdj` | `0 - 36 + 66` | 30 |
| **maskOffset** | `ComputeOffsetForCausal(30)` | `(Min(30,256)+0) × 2048` | **61440** ← 错误 |
| 正确 maskOffset | `s1Offset × attenMaskS2Size + s2Offset` | `0 × 2048 + 36` | **36** |

> 注：日志中 deltaPre=-916，对应 `preTokensPerBatchAdj` 取修复后值 879（非 6731）。这是因为 `SetSparseParamBand` 内部 `preTokensPerBatch = preTokens - actualS2Size + actualS1Size = 946 - 69 + 2 = 879`，然后 `+= actualKVPrefixSize` → 6731。但日志 deltaPre=-916 = `0 - 36 - 879 - 1`，说明实际用的是 879 而非 6731。**这里 preTokensAdj 的坐标调整可能存在残留问题**，但不影响当前结论（computeMode=1 时 pre mask 不参与计算）。

### 4.4 ComputeOffsetForCausal 为何不适用

`ComputeOffsetForCausal`（attenmask.h:227）是为**压缩 causal mask**设计的边界寻址：

```cpp
if (delta <= 0) return Min(-delta, s1BaseSize) + vecCoreOffset * attenMaskS2Size;  // 列内偏移
if (delta >  0) return (Min(delta, s2BaseSize) + vecCoreOffset) * attenMaskS2Size;  // 行偏移
```

- delta > 0 时，返回 `delta × attenMaskS2Size`，即把 delta 当作**行索引**，指向"边界行"
- 这在压缩存储中是正确的：压缩 mask 的每行要么全 1 要么全 0，从边界行开始拷贝即可得到 mask=0 的区域
- 但本 case 的 attenMask 是**[2048, 2048] 全 bool 矩阵**（每字节一个 token，0/1），不是压缩格式
- 对全矩阵，正确寻址必须是 `(s1Offset, s2Offset)` 二维坐标 → `s1Offset × attenMaskS2Size + s2Offset`
- `BoolCopyInRegbase` 从 `maskOffset` 开始拷贝 `s1Size` 行、每行 `s2Size` 字节，因此 `maskOffset` 必须是 `s1Offset × attenMaskS2Size + s2Offset`

### 4.5 computeMode 分析

```
causalOrNextFactor = deltaC - s2BaseSize = 30 - 256 = -226  (< 0)
preFactor = deltaPre + 1 + s1BaseSize = -916 + 1 + 64 = -851 (<= 0)
→ causalOrNextFactor < 0 && preFactor <= 0
→ computeMode = CAUSAL_OR_NEXT_ONLY_MODE (1)
```

computeMode=1 意味着只需 "next" mask，不需要 BAND merge（pre mask 不参与）。因此 `AttenMaskCopyIn` 中 BAND merge 分支（条件 `PRE_ONLY_MODE || PRE_AND_NEXT_MODE`）不满足，直接跳过，只拷贝 next mask。

### 4.6 精度影响

s2L=4（非 prefix block，原始序列 0-67 共 68 个真实 token）的 mask 全为 0：
- VEC-Nd 阶段所有 token 被 mask 掉（Select 选择 -FLT_MAX）
- softmax max/sum 不被本 block 更新（mx0=45.927、sm0=1.047 均为 prefix blocks 的累积值）
- **非 prefix block 的贡献完全丢失**，只有 5852 个 prefix token 参与注意力
- LSE 输出 `lse0=lse1=44.252`（期望 56.064），两个 head 的 LSE 相同（异常）

### 4.7 修复方向

BAND_MODE 的非 prefix block 应使用**直接地址** `s1Offset × attenMaskS2Size + s2Offset`，而非 `ComputeOffsetForCausal(delta)`。

> ⚠️ 注意：`ComputeOffsetForCausal` 的调用在 attenmask.h:527 是所有 compress mode 的公共返回路径，不限于 enableKVPrefix。修改需精确限定到 BAND_MODE + enableKVPrefix 的非 prefix block 场景，避免影响无 prefix 的正常 BAND_MODE 路径。

---

## 5. 调用链路

### 5.1 不合轴路径（gS1M=0，当前启用）

```
flash_attention_noquant_block_vec_base.h:1004 (isPfaGS1Merge == false)
  ├─ isPrefixMaskBlockGqa = (s2L + s2StartIdx/s2BaseSize) < prefixLoopCount
  ├─ if isPrefixMaskBlockGqa:                         ← s2L=0..3
  │   └─ AllocTensor → SetValue(0xFF) × (s2BaseSize/8) → EnQue → DeQue
  │      [MASK-gqa-pref] 打印
  └─ else:                                            ← s2L=4
      └─ AttenMaskCopyIn(attenMaskInQue, attenMaskInQuePre, attenMaskGmInt, ...)
         ├─ ComputeAttenMaskOffset(runInfo, constInfo, attenMaskInfo)   ← §4 核心问题
         │   ├─ s2Offset = s2StartIdx + s2L*s2BaseSize - actualKVPrefixSize = 36
         │   ├─ BAND_MODE:
         │   │   ├─ preTokensAdj  += actualKVPrefixSize
         │   │   ├─ nextTokensAdj -= actualKVPrefixSize
         │   │   ├─ deltaPre     = s1Offset - s2Offset - preTokensAdj - 1 = -916
         │   │   ├─ deltaC       = s1Offset - s2Offset + nextTokensAdj   = 30
         │   │   ├─ attenMaskOffsetPre = ComputeOffsetForCausal(deltaPre) = 64
         │   │   ├─ GetAttenMaskComputeMode → computeMode = 1 (CAUSAL_OR_NEXT_ONLY)
         │   │   └─ return ComputeOffsetForCausal(deltaC=30) = 61440  ← BUG
         │   [MASK-offset] / [COPYIN-offset] / [MASK-gm] 打印
         ├─ BoolCopyInRegbase(attenMaskUb, srcTensor, maskOffset=61440, ...)
         │   └─ isGqa: 循环 s1Size 次 DataCopy（1 行 × 68 字节，从 GM row 30 拷贝）
         ├─ EnQue(attenMaskUb)
         ├─ BAND merge 条件: computeMode == PRE_ONLY || PRE_AND_NEXT → 不满足，跳过
         └─ return
      └─ attenMaskUb = attenMaskInQue.DeQue()
         [MASK-gqa-nonpref] 打印 mask[0]=0
  ├─ PipeBarrier<PIPE_MTE2>()
  │   [MASK-chk1..5] 打印
  └─ ProcessVec1Vf (VF256): LoadAlign mask → Select → Max reduce
      [VEC-Nd-mask] 打印 mask[0]=0
```

### 5.2 合轴路径（gS1M=1，已通过 tiling 禁用）

```
flash_attention_noquant_block_vec_base.h:925 (isPfaGS1Merge == true)
  └─ 构造 MaskInfo
     ├─ s2StartIdx = s2StartIdx + s2L*s2BaseSize  [全局KV坐标]
     ├─ if s2StartIdx >= actualKVPrefixSize: s2StartIdx -= actualKVPrefixSize  [转原始序列坐标]
     └─ s2Size = actualS2Size = 69
  └─ AttentionmaskCopyGS1 → ComputeAttenMaskOffsetCompress (attenmask_gs1.h:178)
     ├─ nextToken = -1 + 69 - 2 = 66
     ├─ delta = 66 + 0 - 36 = 30  [与不合轴修复后一致]
     └─ offset = clamp(delta, 0, s2dealNum) * attenMaskS1Stride
  └─ BAND merge: MergeBandModeMask (有越界风险，见 §8)
```

> 合轴路径的 offset 计算（delta=30）与不合轴路径**一致**，同样存在把 delta 当行偏移的问题。合轴路径已通过 tiling 禁用，但根因未修。

---

## 6. 关键计算点

### 6.1 s2Offset（非 prefix block, s2L=4）

| 场景 | 公式 | 结果 | 正确性 |
|------|------|------|--------|
| 原始代码 | `4864 + (4-23)*256 + 5852` | 5852 | ✗ 减去 padded prefix |
| 修复后 | `4864 + 4*256 - 5852` | 36 | ✓ 原始序列坐标 |

### 6.2 SetSparseParamBand tokens 计算（infer_flash_attention_sparse.h:87）

```
preTokensPerBatch  = preTokens - actualS2Size + actualS1Size = 946 - 69 + 2 = 879
nextTokensPerBatch = nextTokens + actualS2Size - actualS1Size = -1 + 69 - 2 = 66
if enableKVPrefix:
    preTokensPerBatch  -= actualKVPrefixSize  → 879 - 5852 = -4973   [全局KV坐标]
    nextTokensPerBatch += actualKVPrefixSize  → 66 + 5852 = 5918     [全局KV坐标]
```

> `SetSparseParamBand` 将 tokens 调整到**全局 KV 坐标系**（加了 prefix）。

### 6.3 BAND_MODE delta（非 prefix block, s2L=4，修复后）

| 场景 | nextTokensAdj | deltaC | deltaPre | maskOffset | computeMode |
|------|--------------|--------|----------|------------|-------------|
| 修复前 | 5918（未调整） | 0-36+5918=5882 | 0-36-(-4973)-1=4936 | 256×2048=524288（OOB） | 2（PRE_ONLY） |
| 修复后 | 5918-5852=66 | 0-36+66=30 | 0-36-879-1=-916 | 30×2048=61440 | 1（CAUSAL_OR_NEXT） |

### 6.4 ComputeOffsetForCausal(deltaC=30)

```
deltaC=30 > 0:
  offset = (Min(30, 256) + vecCoreOffset) * attenMaskS2Size = (30 + 0) * 2048 = 61440
```

### 6.5 GetAttenMaskComputeMode

```
causalOrNextFactor = deltaC - s2BaseSize = 30 - 256 = -226   (< 0)
preFactor = deltaPre + 1 + s1BaseSize = -916 + 1 + 64 = -851 (<= 0)
→ causalOrNextFactor < 0 && preFactor <= 0
→ computeMode = CAUSAL_OR_NEXT_ONLY_MODE (1)
```

只需 next mask，不需要 BAND merge。

### 6.6 BoolCopyInRegbase GQA 分支（isGqa=1）

```cpp
dataCopyParams.blockCount = 1;
dataCopyParams.blockLen = CeilDiv(68, 32) = 3;          // 68 字节 → 3 blocks
dataCopyParams.dstStride = CeilDiv(256, 32) - 3 = 5;
dataCopyParams.srcStride = (2048 - 68) / 32 = 62;
for (i = 0; i < s1Size=1; ++i)
    DataCopy(dstTensor[i*256], srcTensor[maskOffset], dataCopyParams);  // 从 GM row 30 拷 68 字节
```

拷贝机制本身正确，只是 `maskOffset` 指向了错误的 GM 位置。

---

## 7. 精度结果历程

| 场景 | gS1M | output-0 失败率 | output-1 失败率 | LSE[0] |
|------|------|----------------|----------------|--------|
| 原始（GS1Merge） | 1 | 18.97% | 22.74% | 25.699 |
| 不合轴 + BAND delta 修复前 | 0 | 4.56% | 9.24% | 52.818 |
| 不合轴 + BAND delta 修复后 | 0 | 26.45% | 35.44% | 52.818 |
| 最新日志（含 PipeBarrier + MASK-gm） | 0 | — | — | 44.252 |

期望 LSE = 56.064。

> 修复后精度反而下降的原因：修复前 maskOffset=524288（OOB）读到的是 GM 末尾的随机数据，部分 token 碰巧未被 mask；修复后 maskOffset=61440 在范围内但指向 row 30（值为 0），导致非 prefix block **全部 token 被 mask 掉**，贡献为零，精度更差。

---

## 8. 已排除的错误假设

### 8.1 "mask 从 0xFF 被覆盖为 0" — 已排除

**先前的观察（已过时）：** 早期日志中 `[MASK-gqa-nonpref]`（DeQue 后）显示 mask=0xFF，但 `[VEC-Nd-mask]` 显示 mask=0，曾推测 mask 在 DeQue 后到 VEC-Nd 之间被其他 buffer 操作覆盖。

**最新日志推翻此假设：** 添加 `PipeBarrier<PIPE_MTE2>` 和 `[MASK-gm]` 打印后：
- `[MASK-gqa-nonpref]`（PipeBarrier 之前）就已显示 mask=0
- `[MASK-gm]` 直接读 GM 确认 offset 61440 处值 = 0

**真相：** 早期的 0xFF 是上一轮 prefix block `SetValue(0xFF)` 留下的 **stale UB 数据**，在 DataCopy 完成前被 GetValue 读到。mask=0 从一开始就是 GM offset 61440 的真实数据，不存在覆盖。

### 8.2 "BoolCopyInRegbase 拷贝越界导致 stale 数据" — 已排除

曾推测 `BoolCopyInRegbase` 拷贝 68 字节有效数据 + 188 字节 stale 数据，stale 区域为 0 导致 VEC-Nd 读 0。

**排除依据：** GQA 分支 `dataCopyParams.blockCount=1`，只拷 1 行。`[MASK-gm]` 确认 GM offset 61440 本身就是 0，与 stale 数据无关。

### 8.3 "TQue buffer 复用 / EnQue-DeQue 异常" — 已排除

曾推测 BAND merge 跳过时单次 EnQue/DeQue 导致 buffer 被回收，后续 `Get<float>()` 覆盖 mask。

**排除依据：** mask=0 在 DeQue 后立即出现（PipeBarrier 之前），且 `[MASK-gm]` 证明 GM 源数据就是 0。mask PhyAddr（0x1ff00）在各检查点一致，无被覆盖迹象。

### 8.4 合轴路径 MergeBandModeMask 越界（独立问题，未修复合轴路径）

`MergeBandModeMask` 的 `loopCount` 计算：
```
halfS1RealSizeLoop = 1 + 1 = 2
s2BaseSize=256 > regBytes=32 → rowNumEachLoop=1, rowNumTimesEachLoop=256/32=8
loopCount = (2/1) * 8 = 16 → 处理 16*256 = 4096 字节
实际数据 = 1*256 = 256 字节 → 越界处理 3840 字节 stale 数据
```

此问题存在于合轴路径，当前已通过 tiling 禁用合轴绕过。若未来需启用合轴路径，需修复 `loopCount` 为实际数据量 `CeilDiv(halfS1RealSize * s2BaseSize, 256)`。

---

## 9. Buffer 物理地址确认（无重叠）

最新日志 s2L=4：
```
maskPhy=0x1ff00  mmResPhy=0xc000  maxPhy=0x15400  sumPhy=0x14100  expPhy=0x15700
```

s2L=3（prefix block）：
```
maskPhy=0x21f00  mmResPhy=0x4000  maxPhy=0x15400  sumPhy=0x14100  expPhy=0x15600
```

各 buffer 物理地址无重叠。mmResPhy 在 s2L=3（0x4000）和 s2L=4（0xc000）间切换（双缓冲），maskPhy 同理（0x21f00 / 0x1ff00）。

---

## 10. 文件修改清单与 Git 状态

### 已修改文件

| 文件 | 修改内容 | Commit |
|------|---------|--------|
| `fused_infer_attention_score_tiling_impl.cpp` | `SetGSMerge` 中 `systemPrefixMaxLen > 0` 时禁用 gsMerge | `444ea1dda` |
| `attenmask.h` | s2Offset 公式修复（2 处） | `366f1d5f6` |
| `attenmask.h` | deltaN 符号修复 | `366f1d5f6` |
| `attenmask.h` | BAND_MODE delta tokens 坐标修复 | `026519c0d` |
| `flash_attention_noquant_block_vec_base.h` | prefix block 0xFF fill（GQA non-merge + non-GQA） | `366f1d5f6` |
| `attenmask.h` / `flash_attention_noquant_block_vec_base.h` | 调试打印（MASK-entry/offset/gm/chk1-5, COPYIN-offset, VEC-Nd-* 等） | `7672eb954` 等 |
| `attenmask.h` | GM mask 读取验证（MASK-gm） | `e87024193` |

### 关键参考文件（只读）

| 文件 | 作用 |
|------|------|
| `attenmask_gs1.h` | GS1Merge 路径的 mask 拷贝逻辑 |
| `infer_flash_attention_sparse.h` | `SetSparseParamBand`（tokens 坐标调整） |
| `vf/vf_basic_block_unaligned256_no_update.h` | VF256 mask 应用逻辑 |
| `fia_kernel_noquant_gqa.h` | buffer 初始化和 kernel 入口 |
| `fia_block_vec_noquant_gqa.h` | ProcessVec1 入口 |

### 关键函数

| 函数 | 位置 | 说明 |
|------|------|------|
| `ComputeAttenMaskOffset` | attenmask.h:350 | maskOffset 计算入口，BAND_MODE 分支在 line 477 |
| `ComputeOffsetForCausal` | attenmask.h:227 | **问题函数**，delta>0 时返回 `delta × attenMaskS2Size`（行偏移），对全矩阵 BAND_MODE 不适用 |
| `BoolCopyInRegbase` | attenmask.h:61 | 从 GM 拷贝 mask，使用 maskOffset 作为起始地址，拷 s1Size 行 × s2Size 字节 |
| `AttenMaskCopyIn` | attenmask.h:555 | 非 prefix block mask 拷贝入口 |
| `GetAttenMaskComputeMode` | attenmask.h:110 | computeMode 判定 |
| `SetSparseParamBand` | infer_flash_attention_sparse.h:72 | BAND_MODE tokens 计算（全局 KV 坐标） |

### Git 分支

```
分支: fix/fix-prefix-offset-bug
最近 commit（新→旧）:
  e87024193  debug: add GM mask read at delta offset vs correct (s1,s2) offset
  8ad7e3ce3  debug: add comprehensive diagnostics for mask corruption
  c3cd7eafe  debug: add phyAddr prints to verify mask buffer address consistency
  ea517b593  Revert "debug: add checkpoint prints to trace mask corruption timing"
  db216bd7f  debug: add checkpoint prints to trace mask corruption timing
  7b1e932b8  debug: add checkpoint prints to trace mask corruption timing
  29ad71b9e  Revert "fix: unify EnQue/DeQue pattern for BAND merge and non-merge paths"
  ee4d8bfd0  fix: unify EnQue/DeQue pattern for BAND merge and non-merge paths
  c49cf53b4  Revert "fix: add explicit MTE2_V sync for non-merge BAND path to prevent mask corruption"
  f488f0268  fix: add explicit MTE2_V sync for non-merge BAND path to prevent mask corruption
  026519c0d  fix: adjust preTokens/nextTokens for prefix coordinate mismatch in BAND_MODE
  444ea1dda  fix: disable gsMergeFlag when systemPrefixMaxLen > 0
  7672eb954  debug: safe mask-path prints
  366f1d5f6  fix: s2Offset + deltaN + prefix 0xFF fill
```

已回退的尝试：
- `c49cf53b4` MTE2_V sync 修复（使精度更差）
- `29ad71b9e` 统一 EnQue/DeQue 模式（不必要）

---

## 11. 待完成工作

1. **修复 maskOffset 计算**：BAND_MODE + enableKVPrefix 的非 prefix block 应使用直接地址 `s1Offset × attenMaskS2Size + s2Offset`，而非 `ComputeOffsetForCausal(delta)`。需精确限定范围，避免影响无 prefix 的正常 BAND_MODE 路径。
2. **验证 deltaPre 坐标调整**：日志中 deltaPre=-916 对应 `preTokensAdj=879`（非修复后的 6731），需确认 `preTokensPerBatchAdj += actualKVPrefixSize` 是否正确生效，或 SetSparseParamBand 的坐标调整是否与 delta 计算重复。
3. **清除调试打印**：修复确认后移除所有 `[MASK-*]`、`[COPYIN-*]`、`[VEC-Nd-*]`、`PipeBarrier<PIPE_MTE2>` 等调试代码。
4. **合轴路径修复**（长期）：修复 `MergeBandModeMask` 越界 + 合轴路径的 maskOffset 问题（delta 当行偏移），当前通过 tiling 绕过。

---

## 附录：最新日志 s2L=3/4 完整数据

### s2L=3（最后一个 prefix block，isPref=1）

```
[MASK-entry]        s2L=3 s2S=4864 s2R=220 hS1R=1 gS1M=0 isGqa=1 compM=3 preT=946 nextT=-1 s1oIdx=0 actS1=2 actS2=69 prefSz=5852 pLC=23
[MASK-gqa-nonmerge] s2L=3 isPref=1
[MASK-gqa-pref]     s2L=3 fill=0xFF mask[0]=ff
[MASK-chk1]         s2L=3 mask[0]=ff phyAddr=21f00
[MASK-chk2]         s2L=3 mask[0]=ff
[MASK-chk3]         s2L=3 mask[0]=ff expPhy=15600
[MASK-chk4]         s2L=3 mask[0]=ff
[MASK-chk5]         s2L=3 mask[0]=ff maskPhy=21f00 mmResPhy=4000 maxPhy=15400 sumPhy=14100
[VEC-Nd-info]       s2L=3 s2LL=4 s2S=4864 s2E=5920 prefSz=5852 pLC=23 isPref=1 s1oIdx=0 actS1=2 actS2=69 preT=946 nextT=-1 compM=3
[VEC-Nd-allmax]     mx0=42.800434 mx1=7.726464 mx2=9.502256 mx3=4.185821 sum0=1.058813
[VEC-Nd-pre]        s2L=3 s2R=220 hS1R=1 s2S=4864 phyAddr=4000
[VEC-Nd-pre-max]    iniMax[0]=42.800434 iniSum[0]=1.058813 iniMax[1]=7.726464
                    mmRes[0]=-155.773819 mmRes[1]=-15.460873 mmRes[2]=-64.166557 mmRes[3]=60.987240
                    mmRes[64]=-278.447418 mmRes[128]=94.691544 mmRes[192]=-22.246895 mmRes[255]=-3.4e38
[VEC-Nd-mask]       mask[0]=ff mask[1]=ff                    ← 全程 0xFF，正确
[VEC-Nd-post]       mx0=42.800434 sm0=1.058813 mx1=7.726464 sm1=-282.492157
```

### s2L=4（唯一的非 prefix block，isPref=0）

```
[MASK-entry]        s2L=4 s2S=4864 s2R=68 hS1R=1 gS1M=0 isGqa=1 compM=3 preT=946 nextT=-1 s1oIdx=0 actS1=2 actS2=69 prefSz=5852 pLC=23
[MASK-gqa-nonmerge] s2L=4 isPref=0
[MASK-offset]       s2L=4 s2Off=36 deltaN=5785 deltaC=30 deltaPre=-916 maskOff=61440 computeMode=1 s1Off=0
[COPYIN-offset]     s2L=4 maskOff=61440 preOff=64 computeMode=1 compressMode=3 s2R=68 hS1R=1 maskS2Size=2048
[MASK-gm]           gmAtDeltaOff=61440 val=0 gmAtCorrectOff=36 val=1 s1O=0 s2O=36   ← 根因证据
[MASK-gqa-nonpref]  s2L=4 mask[0]=0 mask[1]=0 reread=0 phyAddr=1ff00                 ← DeQue 后即 0
[MASK-chk1]         s2L=4 mask[0]=0 phyAddr=1ff00
[MASK-chk2]         s2L=4 mask[0]=0
[MASK-chk3]         s2L=4 mask[0]=0 expPhy=15700
[MASK-chk4]         s2L=4 mask[0]=0
[MASK-chk5]         s2L=4 mask[0]=0 maskPhy=1ff00 mmResPhy=c000 maxPhy=15400 sumPhy=14100
[VEC-Nd-info]       s2L=4 s2LL=4 s2S=4864 s2E=5920 prefSz=5852 pLC=23 isPref=0 s1oIdx=0 actS1=2 actS2=69 preT=946 nextT=-1 compM=3
[VEC-Nd-allmax]     mx0=45.927269 mx1=7.726464 mx2=9.502256 mx3=4.185821 sum0=1.047036
[VEC-Nd-pre]        s2L=4 s2R=68 hS1R=1 s2S=4864 phyAddr=c000
[VEC-Nd-pre-max]    iniMax[0]=45.927269 iniSum[0]=1.047036 iniMax[1]=7.726464
                    mmRes[0]=-331.737946 mmRes[1]=-134.870743 mmRes[2]=-134.921310 mmRes[3]=188.724365
                    mmRes[64]=28.180927 mmRes[128]=12.894497 mmRes[192]=-11.563240 mmRes[255]=-3.4e38
[VEC-Nd-mask]       mask[0]=0 mask[1]=0                         ← 全程 0，错误（应为 1）
[VEC-Nd-post]       mx0=45.927269 sm0=1.047036 mx1=7.726464 sm1=-284.044312
[SoftmaxCopyOut]    isFd=0 s2L=4 s2LL=4
[LSE-out]           lseOff=0 lse0=44.251823 lse1=44.251823      ← 远低于期望 56.064，两 head 相同
```

### 核心结论

```
maskOffset = ComputeOffsetForCausal(deltaC=30) = 30 × 2048 = 61440  →  GM[row 30, col 0] = 0 (masked)  ✗
正确 offset = s1Offset × 2048 + s2Offset = 0 × 2048 + 36 = 36       →  GM[row 0, col 36] = 1 (visible) ✓
```

`ComputeOffsetForCausal` 把 delta=30 当作行索引（row 30），但全 [2048,2048] bool 矩阵需要 `(s1Offset, s2Offset)` 二维直接寻址（row 0, col 36）。mask=0 从 DeQue 开始就是 GM offset 61440 的真实数据，不存在覆盖。
