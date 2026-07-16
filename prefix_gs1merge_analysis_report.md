# KV Prefix + BAND_MODE 精度问题分析报告

## 1. 目标与约束

修复 Ascend arch35 上 `aclnnFusedInferAttentionScoreV5` 在 **KV prefix + BAND_MODE (sparseMode=4)** 下的精度问题。

约束：
- 只提到线上分支，不提 PR/issue
- 不要本地编译
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

---

## 12. 直接寻址修复后的新问题分析（2026-07-16）

### 12.1 修改概述

将 `attenmask.h:535-538` 的 `ret` 和 `attenmask.h:501-503` 的 `attenMaskOffsetPre` 从 `ComputeOffsetForCausal(deltaCausalOrNext)` 改为直接寻址：
```cpp
ret = (s1Offset + runInfo.vecCoreOffset) * attenMaskInfo.attenMaskS2Size + s2Offset;
```

### 12.2 修改前后对比

| 指标 | 修改前 | 修改后 |
|------|--------|--------|
| maskOff (s2L=4) | 61440 (row 30) | **36** (row 0, col 36) |
| mask[0] | 0 (masked) | **1** (visible) |
| output-0 失败率 | 34.29% | 14.33% (↓改善) |
| output-1 失败率 | 50.53% | 25.69% (↓改善) |
| LSE[0] (Block 0) | 44.25 | **0.00** (↓恶化) |
| sm1 (head 1 sum) | -284.04 (有限) | **-inf** (恶化) |
| output-1 尾部 9063-9072 | 正常 | **inf** (恶化) |

**结论：起始点位精度改善（前 113 个元素全对），但引入了 sm1=-inf → LSE=0/inf 的新问题。**

### 12.3 根因分析：两个叠加的 Bug

#### Bug A: s2Offset 偏移 36（tail prefix block 未处理）

**公式（attenmask.h:459）：**
```
s2Offset = s2StartIdx + s2LoopCount * s2BaseSize - actualKVPrefixSize
         = 4864 + 4 × 256 - 5852 = 36
```

**实际 KV 布局：**
```
s2L=0: global 4864-5119  (prefix, 256 tokens)
s2L=1: global 5120-5375  (prefix, 256 tokens)
s2L=2: global 5376-5631  (prefix, 256 tokens)
s2L=3: global 5632-5851  (prefix, 220 tokens ← tail block! 不是 256)
s2L=4: global 5852-5919  (non-prefix, 68 tokens)
```

`s2LoopCount × s2BaseSize = 4 × 256 = 1024`，但实际从 s2StartIdx 到非 prefix 起点的距离是 `5852 - 4864 = 988`。差额 `1024 - 988 = 36 = 256 - 220`（tail prefix block 的 padding）。

| 计算方式 | 结果 | 正确性 |
|---------|------|--------|
| 当前公式 `4864 + 4×256 - 5852` | **36** | ✗ 多了 36 |
| 正确值 `5852 - 5852` | **0** | ✓ |

**s2Offset=36 导致 mask 从 col 36 开始读，而非 col 0。**

#### Bug B: 直接寻址绕过了 BAND_MODE nextTokens 约束

**BAND_MODE 语义（preTokens=946, nextTokens=-1）：**
- Query s1 可见 KV s2 当且仅当：`s2 >= s1 - preTokens` AND `s2 <= s1 + nextTokens`
- Query 0: `s2 >= -946` AND `s2 <= -1` → **无 KV 可见**（全部 masked）
- Query 1: `s2 >= -945` AND `s2 <= 0` → **仅 KV 0 可见**

**attenMask 矩阵是因果 mask（上三角=1, 下三角=0）：**
- `mask[s1][s2] = 1` 当 `s2 >= s1`（可见）
- `mask[s1][s2] = 0` 当 `s2 < s1`（masked）

**直接寻址读取的是原始因果 mask，不含 BAND 约束：**
```
直接寻址: maskOffset = s1Offset × 2048 + s2Offset = 0 × 2048 + 36 = 36
GM[row 0, col 36] = 1 (因果 mask: col 36 >= row 0 → visible)
```

但 BAND_MODE nextTokens=-1 要求 query 0 看不到任何 KV。因果 mask 的 1（visible）在此处是**错误的**。

**旧代码 `ComputeOffsetForCausal(deltaC=30)` 的意外效果：**
```
deltaC=30 > 0 → offset = 30 × 2048 = 61440 (row 30)
GM[row 30, col 0..29] = 0 (下三角, masked)  ← 意外地 mask 掉了前 30 个 token
GM[row 30, col 30..67] = 1 (上三角, visible) ← 后 38 个 token 仍可见
```

旧代码虽然 offset 错误，但 row 30 的下三角部分恰好 mask 掉了前 30 个 token，部分接近 BAND_MODE 的正确行为（全部 masked）。新代码直接寻址读 row 0，全上三角=1，完全无 mask。

**两种代码的 mask 对比（s2L=4, s1=0, 68 tokens）：**

| Token 位置 | BAND_MODE 正确值 | 旧代码 (row 30) | 新代码 (row 0) |
|-----------|-----------------|----------------|---------------|
| 0-29      | 0 (masked)      | 0 (masked) ✓   | 1 (visible) ✗ |
| 30-67     | 0 (masked)      | 1 (visible) ✗  | 1 (visible) ✗ |

**两种代码都是错的，但新代码更错**（0/68 正确 vs 30/68 正确）。

### 12.4 sm1=-inf 的产生机制

**执行顺序（Block 0，流水线交错）：**
```
s2L=0 (prefix, mask=0xFF) → VEC-Nd: iniSum[0]=-inf ← 初始状态已是 -inf
...
s2L=4 (non-prefix, mask=1) → VEC-Nd: 处理 68 个全可见 token
  mmRes[192] = -FLT_MAX (buffer 中的垃圾值，本应被 mask 掉)
  mask=1 → -FLT_MAX 参与 softmax 计算
  → sum 累加结果仍为 -inf（因为初始 sum 已是 -inf）
SoftmaxCopyOut → LSE = max + log(sum) = 56.064 + log(-inf) → 0.0
```

**初始 sum=-inf 的来源：**
softmax max/sum buffer 在 head group 切换时未正确重置。上一个 head group 的 s2L=4（mask=1，全可见）处理垃圾 mmRes 值后，sum 变为 -inf，残留到下一个 head group 的初始状态。

修改前 mask=0（全 masked）→ s2L=4 不更新 max/sum → 残留的 sum 保持有限值（如 1.0）→ 不影响后续计算。
修改后 mask=1（全 visible）→ s2L=4 处理垃圾 mmRes → sum 变为/保持 -inf → 传播到所有后续 head group。

### 12.5 LSE=0 和 output-1 尾部 inf 的原因

- **LSE=0**: `LSE = max + log(sum)`。sum=-inf → log(-inf)=-inf → LSE=-inf → 输出截断为 0.0
- **output-1 尾部 inf**: 部分 head group 的 LSE 计算溢出为 inf（max 过大 + sum 异常）

### 12.6 修复方向

**必须同时修复 Bug A 和 Bug B：**

#### Bug A 修复：s2Offset 公式

不能用 `s2LoopCount × s2BaseSize`（假设所有 block 都是 256），需要用实际 KV 累积偏移：
```cpp
// 方案1: 使用实际全局 KV 位置（需要 tiling 传入正确的偏移）
int64_t actualGlobalS2 = runInfo.s2StartIdx + runInfo.s2LoopCount * constInfo.s2BaseSize;
// 修正 tail prefix block 的 padding
if (s2L >= prefixLoopCount) {
    // 非 prefix block: 减去 prefix 总长（含 tail block 的实际大小，不是 prefixLoopCount × s2BaseSize）
    s2Offset = actualGlobalS2 - constInfo.actualKVPrefixSize;
    // 需要 actualGlobalS2 正确反映 tail block 的 220 而非 256
}
```

或更根本地：tiling 阶段计算并传入每个 s2L 对应的实际全局 KV 起始位置。

#### Bug B 修复：BAND_MODE mask 约束

直接寻址 `s1Offset × attenMaskS2Size + s2Offset` 读取的是原始因果 mask，**不含 BAND 约束**。正确做法：

**方案1（推荐）**：保留 `ComputeOffsetForCausal(deltaCausalOrNext)` 的 delta-based 寻址。delta > 0 时读 row delta 的下三角（mask=0），实现 nextTokens 约束。但需先修复 s2Offset（Bug A），使 delta 值正确。

**方案2**：直接寻址 + 后续 BAND merge。读原始因果 mask 后，用 `MergeBandModeMask` 根据 preTokens/nextTokens 裁剪。但当前 computeMode=1 时 BAND merge 被跳过，需要修改 computeMode 判定逻辑。

**方案3**：直接寻址 + 手动 mask 裁剪。在 `AttenMaskCopyIn` 中，对 enableKVPrefix + BAND_MODE 的非 prefix block，读取 mask 后根据 deltaC/deltaPre 手动置 0。

### 12.7 之前分析报告的修正

之前 §4.7 的结论：
> BAND_MODE 的非 prefix block 应使用直接地址 `s1Offset × attenMaskS2Size + s2Offset`

**此结论不完整。** 直接寻址只修复了 GM 读取位置（从错误的 row 30 改为 row 0），但**丢失了 BAND_MODE 的 nextTokens 约束**。因果 mask 的 1（visible）在 BAND_MODE nextTokens=-1 时应为 0（masked）。

正确结论：**直接寻址是必要的（修复 GM 位置），但不充分。还需在直接寻址的基础上应用 BAND 约束（preTokens/nextTokens），或保留 delta-based 寻址但修复 s2Offset。**

### 12.8 关键日志证据

**修改后 Block 0 s2L=4（mask 正确读取但值错误）：**
```
[MASK-offset]      s2L=4 s2Off=36 deltaC=30 maskOff=36 computeMode=1 s1Off=0
[MASK-gm]          gmAtDeltaOff=36 val=1 gmAtCorrectOff=36 val=1  ← 因果 mask=1, 但 BAND 应为 0
[MASK-gqa-nonpref] mask[0]=1 mask[1]=1                             ← 全 visible（错误）
[VEC-Nd-mask]      mask[0]=1 mask[1]=1
[VEC-Nd-post]      mx0=56.064 sm0=1.000419 mx1=1.835 sm1=-inf      ← sm1=-inf
[LSE-out]          lse0=0.000000 lse1=0.000000                     ← LSE=0（应 56.064）
```

**修改后 Block 0 s2L=0（初始状态已污染）：**
```
[VEC-Nd-allmax]    mx0=2.289532 mx1=1.835438 sum0=-inf             ← sum 初始已 -inf
[VEC-Nd-pre-max]   iniMax[0]=2.289532 iniSum[0]=-inf iniMax[1]=1.835
```

**修改前 Block 0 s2L=0（初始状态有限）：**
```
[VEC-Nd-allmax]    mx0=66.072792 mx1=7.726464 sum0=1.000000        ← sum 初始 1.0（有限）
```

### 12.9 精度对比总结

| 场景 | mask 值 | 可见 token 数 | sm1 | LSE | 正确性 |
|------|---------|-------------|-----|-----|--------|
| BAND_MODE 正确 | 0 (全 masked) | 0/68 | 有限 | 56.064 | ✓ |
| 旧代码 (row 30) | 0×30 + 1×38 | 38/68 | -284 (有限) | 44.25 | 部分正确 |
| 新代码 (row 0) | 1×68 | 68/68 | **-inf** | **0** | 更差 |
| 新代码+Bug A 修复 (row 0, col 0) | 1×68 | 68/68 | -inf | 0 | **仍错** |

> 即使修复 Bug A（s2Offset=0），直接寻址仍读因果 mask=1（全 visible），BAND 约束未应用。必须同时修复 Bug B。

---

## 13. 正确的修复方向：保留 delta-based 寻址 + 修复 s2Offset（2026-07-16 更新）

### 13.1 核心认知纠正

**之前 §12 的分析有误。** §12.2 中 "BAND_MODE 正确值 = 全 masked（0/68 可见）" 是错误的。

**关键事实：非 prefix 场景下，BAND_MODE 的 delta-based `ComputeOffsetForCausal` 代码是正确的**（用户确认）。因此 prefix 场景的非 prefix block 应该**复用完全相同的 delta-based 逻辑**，而不是换成直接寻址。

### 13.2 非 prefix 场景 BAND_MODE 工作原理（已验证正确）

**参数：** preTokens=946, nextTokens=-1, S1=2, S2=69, 因果 mask（上三角=1）

**SetSparseParamBand 坐标变换：**
```
preTokensPerBatch  = preTokens - S2 + S1 = 946 - 69 + 2 = 879
nextTokensPerBatch = nextTokens + S2 - S1 = -1 + 69 - 2 = 66
```
此变换将 BAND 约束从 "绝对 token 位置" 转为 "因果 mask 相对位置"，使 `ComputeOffsetForCausal` 能从因果 mask 中正确提取 BAND 区域。

**delta 计算（非 prefix, s2StartIdx=0, s2L=0）：**
```
s2Offset = 0 + 0 × 256 = 0
s1Offset = 0
deltaCausalOrNext = 0 - 0 + 66 = 66
deltaPre = 0 - 0 - 879 - 1 = -880
```

**ComputeOffsetForCausal(deltaC=66):**
```
delta > 0 → offset = 66 × 2048 = 135168 (row 66)
```
从因果 mask 的 row 66 读取 68 字节：
- Col 0-65: 0（下三角，masked）→ 前 66 个 token 被 mask
- Col 66-67: 1（上三角，visible）→ 后 2 个 token 可见

**这就是 BAND_MODE nextTokens 约束的正确实现。** delta 值决定了因果 mask 中 "next 边界" 的位置，`ComputeOffsetForCausal` 利用因果 mask 的三角结构实现 BAND 裁剪。

**computeMode 判定：**
```
causalOrNextFactor = 66 - 256 = -190 (< 0)  → 需要 next mask
preFactor = -880 + 1 + 64 = -815 (<= 0)     → pre 约束自动满足
→ CAUSAL_OR_NEXT_ONLY_MODE (1)  → 只需 next mask，不需 BAND merge
```

### 13.3 prefix 场景的正确做法

**原则：prefix 场景的非 prefix block 应与非 prefix 场景使用完全相同的 delta-based 寻址逻辑。**

需要两个条件：
1. **s2Offset 正确**（原始序列坐标）
2. **pre/next tokens 正确**（原始序列坐标，即反转 SetSparseParamBand 的 prefix 调整）

#### 条件 1: s2Offset 正确公式

**当前公式（Bug A）：**
```cpp
s2Offset = s2StartIdx + s2LoopCount * s2BaseSize - actualKVPrefixSize;
// = 4864 + 4×256 - 5852 = 36  ✗
```

**问题：** `s2LoopCount × s2BaseSize` 假设所有 block 都是 256，但 tail prefix block（s2L=3）只有 220 个 token。多算了 `256 - 220 = 36`。

**正确公式：**
```cpp
s2Offset = s2StartIdx + (s2LoopCount - prefixLoopCount) * s2BaseSize;
// = 4864 + (4 - 23) × 256 = 4864 - 4864 = 0  ✓
```

**推导：**
```
s2StartIdx = 4864 = 19 × 256  (本 core 从全局 block 19 开始)
prefixLoopCount = 23           (全局 prefix block 数)
s2LoopCount = 4                (本 core 内的第 5 个 block，全局 block 23)

全局 block 23 的 padded 位置 = 23 × 256 = 5888
全局 block 23 的实际位置 = actualKVPrefixSize = 5852
差值 (tail padding) = 5888 - 5852 = 36

s2Offset = 全局位置 - actualKVPrefixSize - tail_padding
         = (s2StartIdx + s2LoopCount × s2BaseSize) - actualKVPrefixSize - (prefixLoopCount × s2BaseSize - actualKVPrefixSize)
         = s2StartIdx + s2LoopCount × s2BaseSize - prefixLoopCount × s2BaseSize
         = s2StartIdx + (s2LoopCount - prefixLoopCount) × s2BaseSize
```

**验证（所有 block）：**

| s2L | 全局 block | 类型 | s2Offset 公式 | s2Offset 值 | 正确性 |
|-----|-----------|------|--------------|------------|--------|
| 0 | 19 | prefix | `4864 + 0×256` = 4864 | 4864 (全局) | ✓ |
| 1 | 20 | prefix | `4864 + 1×256` = 5120 | 5120 (全局) | ✓ |
| 2 | 21 | prefix | `4864 + 2×256` = 5376 | 5376 (全局) | ✓ |
| 3 | 22 | prefix | `4864 + 3×256` = 5632 | 5632 (全局) | ✓ |
| 4 | 23 | non-prefix | `4864 + (4-23)×256` = 0 | 0 (原始) | ✓ |

> prefix block 用全局坐标（不减 prefix），非 prefix block 用原始序列坐标（减去 padded prefix）。这与 attenMask GM 的坐标一致：prefix block 不读 GM（填 0xFF），非 prefix block 从 GM 的原始序列位置读取。

#### 条件 2: pre/next tokens 调整（已正确实现）

**SetSparseParamBand** 将 tokens 转到全局 KV 坐标：
```
preTokensPerBatch  = 879 - 5852 = -4973  (全局)
nextTokensPerBatch = 66 + 5852 = 5918    (全局)
```

**attenmask.h** 反转此调整，回到原始序列坐标：
```cpp
preTokensPerBatchAdj  += actualKVPrefixSize;   // -4973 + 5852 = 879  ✓
nextTokensPerBatchAdj -= actualKVPrefixSize;   // 5918 - 5852 = 66    ✓
```

调整后的值 (879, 66) 与非 prefix 场景完全一致。**此部分代码正确，无需修改。**

### 13.4 正确修复后的完整计算

**s2Offset = 0（修复 Bug A 后）：**
```
s1Offset = 0
s2Offset = 0
preTokensPerBatchAdj = 879
nextTokensPerBatchAdj = 66

deltaCausalOrNext = 0 - 0 + 66 = 66
deltaPre = 0 - 0 - 879 - 1 = -880

maskOffset = ComputeOffsetForCausal(66) = 66 × 2048 = 135168  (row 66)
attenMaskOffsetPre = ComputeOffsetForCausal(-880) = 64  (column offset)
```

**与非 prefix 场景完全一致！** delta=66，从 row 66 读取因果 mask，前 66 个 token masked，后 2 个 visible。computeMode=1（CAUSAL_OR_NEXT_ONLY），不需 BAND merge。

### 13.5 需要回退的代码

**attenmask.h:535-538**（直接寻址覆盖 ret）：
```cpp
// 需要删除此段，让 ret 保持 ComputeOffsetForCausal(deltaCausalOrNext) 的结果
if constexpr (enableKVPrefix) {
    if (attenMaskInfo.compressMode == static_cast<uint8_t>(AttenMaskCompressMode::BAND_MODE)) {
        ret = (s1Offset + runInfo.vecCoreOffset) * attenMaskInfo.attenMaskS2Size + s2Offset;  // ← 删除
    }
}
```

**attenmask.h:501-503**（直接寻址覆盖 attenMaskOffsetPre）：
```cpp
// 需要改为 ComputeOffsetForCausal(deltaPre)，与非 prefix 路径一致
if constexpr (enableKVPrefix) {
    attenMaskInfo.attenMaskOffsetPre = (s1Offset + runInfo.vecCoreOffset) *
        attenMaskInfo.attenMaskS2Size + s2Offset;  // ← 改为 ComputeOffsetForCausal(deltaPre, ...)
}
```

### 13.6 需要修复的代码

**attenmask.h:459**（s2Offset 公式）：
```cpp
// 当前（Bug A）:
s2Offset = runInfo.s2StartIdx + runInfo.s2LoopCount * constInfo.s2BaseSize - constInfo.actualKVPrefixSize;

// 修复后:
s2Offset = runInfo.s2StartIdx + (runInfo.s2LoopCount - constInfo.prefixLoopCount) * constInfo.s2BaseSize;
```

### 13.7 修复前后对比

| 指标 | 旧代码 (row 30) | 直接寻址 (row 0) | **正确修复 (row 66)** |
|------|----------------|-----------------|---------------------|
| s2Offset | 36 | 36 | **0** |
| deltaC | 30 | 30 | **66** |
| maskOffset | 61440 (row 30) | 36 (row 0) | **135168 (row 66)** |
| mask[0..65] | 0 (masked) | 1 (visible) | **0 (masked)** |
| mask[66..67] | 1 (visible) | 1 (visible) | **1 (visible)** |
| BAND 约束 | 部分错误 | 完全丢失 | **正确** |
| sm1 | -284 (有限) | -inf | **有限（正确）** |

### 13.8 为什么直接寻址是错的

**直接寻址** `(s1Offset, s2Offset) = (0, 0)` 读取因果 mask 的 **row 0**：
- Row 0: col 0-67 全部 = 1（上三角，visible）
- 结果：全部 68 个 token 可见

**但 BAND_MODE nextTokens 约束要求**：通过 delta=66 将读取位置移到 **row 66**，利用因果 mask 的三角结构实现裁剪：
- Row 66: col 0-65 = 0（下三角，masked），col 66-67 = 1（上三角，visible）
- 结果：前 66 个 token masked，后 2 个 visible

**`ComputeOffsetForCausal` 的核心作用**是将 BAND 的 nextTokens 约束映射到因果 mask 的行偏移。直接寻址绕过了这个映射，读到的 mask 不含 BAND 约束。

### 13.9 总结

| 问题 | 根因 | 修复 |
|------|------|------|
| maskOffset 指向错误位置 | s2Offset=36（tail block padding） | s2Offset 公式改为 `(s2L - prefixLoopCount) × s2BaseSize` |
| BAND 约束丢失 | 直接寻址绕过 `ComputeOffsetForCausal` | 回退直接寻址，保留 delta-based 寻址 |
| pre/next tokens 坐标 | SetSparseParamBand 转全局，attenmask.h 反转 | 已正确，无需修改 |

**正确修复 = 回退直接寻址 + 修复 s2Offset 公式。** 修复后，prefix 场景的非 prefix block 与非 prefix 场景使用完全相同的 delta-based BAND_MODE 逻辑。

---

## 14. 重新分析：prefix 区域也需要 BAND mask（2026-07-16 最终版）

### 14.1 核心认知纠正（再次）

**§12 和 §13 的分析仍有错误。** 之前认为 "prefix block 填 0xFF 是正确的" 是错误的。

**正确理解（用户确认）：**
- prefix 场景只是 s2 有前缀，**该 mask 的部分还是需要 mask**
- mask 规则和**非 prefix 的 case 完全一样**
- prefix token 也受 BAND_MODE 的 preTokens/nextTokens 约束

### 14.2 BAND_MODE 对 prefix token 的约束

**参数：** preTokens=946, nextTokens=-1, S1=2, S2_orig=69, prefix_len=5852

**SetSparseParamBand 变换（含 prefix 调整，全局 KV 坐标）：**
```
preTokensPerBatch  = preTokens - S2 + S1 - prefix = 946 - 69 + 2 - 5852 = -4973
nextTokensPerBatch = nextTokens + S2 - S1 + prefix = -1 + 69 - 2 + 5852 = 5918
```

**attenmask.h 反转（回到原始序列坐标）：**
```
preTokensPerBatchAdj  = -4973 + 5852 = 879
nextTokensPerBatchAdj = 5918 - 5852 = 66
```

**BAND 可见条件（原始序列坐标，相对位置）：**
```
visible iff: s1 - preT_adj <= s2_orig <= s1 + nextT_adj
           = s1 - 879 <= s2_orig <= s1 + 66
```

| Query | 可见 s2_orig 范围 | prefix 可见数 | non-prefix 可见数 | 总计 |
|-------|-----------------|--------------|-------------------|------|
| s1=0 | [-879, 66] | 879 (s2_orig [-879, -1]) | 67 (s2_orig [0, 66]) | 946 |
| s1=1 | [-878, 67] | 878 (s2_orig [-878, -1]) | 68 (s2_orig [0, 67]) | 946 |

### 14.3 各 prefix block 的正确 mask 状态

| s2L | 全局 KV 范围 | s2_orig 范围 | block 大小 | Q0 可见数 | 当前(0xFF) | 正确？ |
|-----|-------------|-------------|-----------|----------|-----------|--------|
| 0 | 4864-5119 | [-988, -733] | 256 | **214** (s2_orig >= -879) | 256 (全可见) | **✗ 错误** |
| 1 | 5120-5375 | [-732, -477] | 256 | 256 (全可见) | 256 | ✓ |
| 2 | 5376-5631 | [-476, -221] | 256 | 256 (全可见) | 256 | ✓ |
| 3 | 5632-5851 | [-220, -1] | 220 | 220 (全可见) | 220 | ✓ |
| 4 | 5852-5919 | [0, 67] | 68 | **67** (s2_orig <= 66) | 0 (全masked) | **✗ 错误** |

**两个有问题的 block：**
1. **s2L=0（第一个 prefix block）**：前 42 个 token（s2_orig [-988, -880]）应被 mask，但当前填 0xFF（全可见）
2. **s2L=4（非 prefix block）**：应有 67 个可见，但当前 s2Offset=36 + 直接寻址导致全可见或全 masked

### 14.4 s2L=0 的 mask 问题

**s2L=0 的 s2_orig 范围：[-988, -733]**

BAND 约束：s2_orig >= s1 - preT = 0 - 879 = **-879**
- s2_orig [-988, -880]：**masked**（42 个 token，在 preTokens 边界之外）
- s2_orig [-879, -733]：**visible**（214 个 token）

**当前代码：** force-fill 0xFF（全可见）→ 42 个本应 masked 的 token 被错误地参与注意力计算

**问题：** prefix token 的 s2_orig < 0，而 mask 矩阵是 [2048, 2048] 按 [0, 2047] 索引。无法从 GM 读取负偏移的 mask。

**解决方向：**
- **方案 A**：在 UB 中手动构造 mask。对 prefix block，根据 preTokens/nextTokens 约束计算每个 token 的可见性，填 0xFF 或 0x00
- **方案 B**：使用 SetSparseParamBand 的全局坐标。在全局坐标下，prefix block 的 s2Offset > 0（如 s2L=0 全局 s2Offset=4864）。但 mask 矩阵 [2048, 2048] 无法覆盖全局位置 4864
- **方案 C**：对 s2L=0 使用 delta-based 寻址。如果 s2Offset 使用原始坐标（-988），deltaPre=108 > 0 → PRE_ONLY_MODE，需要读取 pre mask。但 s2Offset=-988 < 0，无法从 GM 读取

### 14.5 s2L=4（非 prefix block）的 mask 问题

**正确 s2Offset = 0（修复 tail block padding Bug A）**

delta 计算（与非 prefix case 完全一致）：
```
s1Offset = 0, s2Offset = 0
deltaC = 0 - 0 + 66 = 66
deltaPre = 0 - 0 - 879 - 1 = -880
causalOrNextFactor = 66 - 256 = -190 (< 0)
preFactor = -880 + 1 + 64 = -815 (<= 0)
computeMode = CAUSAL_OR_NEXT_ONLY (1)
maskOffset = ComputeOffsetForCausal(66) = 66 × 2048 = 135168
```

**与非 prefix case 完全相同的 delta-based 逻辑。** 用户确认非 prefix case 正确，因此 s2L=4 修复后也正确。

### 14.6 完整修复方案

| 问题 | 当前状态 | 正确修复 |
|------|---------|---------|
| s2L=0 prefix mask | force-fill 0xFF（42 token 错误可见） | 根据 preTokens 约束计算 mask（214 可见，42 masked） |
| s2L=1..3 prefix mask | force-fill 0xFF | ✓ 正确（全可见） |
| s2L=4 s2Offset | 36（tail padding bug） | 改为 `(s2L - prefixLoopCount) × s2BaseSize = 0` |
| s2L=4 寻址方式 | 直接寻址（绕过 BAND） | 回退为 delta-based `ComputeOffsetForCausal` |
| pre/next tokens 调整 | SetSparseParamBand 全局 + attenmask.h 反转 | ✓ 正确 |

### 14.7 关键约束

1. **非 prefix case 的 BAND_MODE 代码是正确的**（用户确认）。prefix case 的非 prefix block 应复用完全相同的逻辑
2. **prefix block 也需要 BAND mask**（用户确认）。不能简单填 0xFF
3. **mask 矩阵 [2048, 2048] 只覆盖原始序列 [0, 68]**。prefix token（s2_orig < 0）无法从 GM 读取 mask
4. **s2L=0 是唯一需要部分 mask 的 prefix block**。其余 prefix block（s2L=1..3）全可见，填 0xFF 正确

### 14.8 s2L=0 mask 处理的具体方案

s2L=0 的特殊性：s2_orig [-988, -733]，preT=879
- 前 42 token（s2_orig [-988, -880]）：s2_orig < -879 → masked
- 后 214 token（s2_orig [-879, -733]）：s2_orig >= -879 → visible

**由于无法从 GM 读取负偏移 mask，需要在 UB 中构造：**

```cpp
// 伪代码
if (isPrefixBlock && s2L == 0) {  // 或更通用的条件：deltaPre > 0
    // 需要部分 mask 的 prefix block
    int64_t s2OrigStart = s2StartIdx + s2L * s2BaseSize - actualKVPrefixSize;  // = -988
    int64_t preBoundary = s1Offset - preTokensPerBatchAdj;  // = 0 - 879 = -879
    int64_t maskCount = preBoundary - s2OrigStart;  // = -879 - (-988) = 109... 
    // 注意：这里需要考虑 halfS1RealSize 和具体的 block 结构
    // 前 maskCount 个 token 填 0x00 (masked)，其余填 0xFF (visible)
}
```

**更通用的方案：** 不硬编码 s2L==0，而是检查 deltaPre 是否 > 0（pre 约束激活）。如果 deltaPre > 0 且是 prefix block，则在 UB 中根据 pre 边界构造 mask。

### 14.9 总结

**三个需要修复的问题（按严重程度排序）：**

1. **s2L=0 prefix block 部分mask**（新发现）：前 42 token 应 masked 但当前全可见。导致 42 个超出 preTokens 范围的 prefix token 错误参与注意力
2. **s2L=4 s2Offset bug**（Bug A）：36 应为 0。导致 delta 值错误
3. **s2L=4 直接寻址**（Bug B）：绕过 BAND_MODE 约束。应回退为 delta-based

**修复原则：**
- 非 prefix block（s2L=4）：使用与非 prefix case **完全相同**的 delta-based 逻辑
- Prefix block（s2L=0..3）：不能简单填 0xFF。需要根据 BAND 约束计算 mask。大部分 prefix block 全可见（0xFF 正确），但 s2L=0 需要部分 mask
- Prefix block 的 mask 无法从 GM 读取（s2_orig < 0），需要在 UB 中构造
