# FIA Antiquant 算子 0/1 轴非连续支持方案

## 1. 背景与目标

### 1.1 参考PR

参考 [ops-transformer PR #6430](https://gitcode.com/cann/ops-transformer/pull/6430)（标题：`mxfp8 PA场景支持0/1轴非连续`），该 PR 在**全量化(mxfp8 fullquant) FA 算子**上实现了 PA(Paged Attention) 场景下 kvcache、krope、kvscale 的 0/1 轴非连续支持。

### 1.2 本次目标

在**伪量化(antiquant) FA 算子**上实现相同功能。本接口参与适配的参数共 **4 个**：

| 参数 | 说明 |
|------|------|
| `k` | key 张量 |
| `v` | value 张量 |
| `k_descale` | key_antiquant_scale（key 反量化缩放因子） |
| `v_descale` | value_antiquant_scale（value 反量化缩放因子） |

> **注意**：k_rope 不在本次范围内（参考 PR 含 kRope，本方案不含）。

### 1.3 功能定义

- **PA 场景**：支持 `k`/`v`/`k_descale`/`v_descale` 的 0 轴(Bn)和 1 轴(N2)非连续
- **非 PA 场景**：不支持非连续（报错拦截）
- 非连续仅限 0/1 轴，2+ 轴仍要求连续

---

## 2. 架构差异：Fullquant vs Antiquant

| 维度 | Fullquant (PR#6430 已做) | Antiquant (本次要做) |
|------|--------------------------|----------------------|
| ConstInfo 类型 | `CommonConstInfo` (新架构) | `ConstInfo<isInfer, hasRope>` (旧 regbase) |
| Tiling 数据 | `FiaBaseParams` + `stridesParams` | `InputParamsRegbase` + `StridesParams` |
| KV 偏移计算 | `OffsetCalculator::GetOffset()` | 手动公式：`ComputeOffsetForAntiquant` + `SetAntiqParamCommon` + `CopyKV` |
| Scale 访问 | `FaGmTensor` + `OffsetCalculator` | `InitQuant()` 直接 `SetGlobalBuffer`，无 stride |
| 需改 kernel 文件 | (已改完) | 4 个 kernel + 2 个 tiling/define + 2 个 host |

### 2.1 Antiquant offset 计算链路

```
ComputeOffsetForAntiquant()     →  runInfo.keyOffset / valueOffset
        ↓
SetAntiqParamCommon()           →  taskParam.kvGmOffset = kvOffset + subBlockIdx * dealSize * kvStep
        ↓                                  taskParam.kvStep = dSize(BNSD) 或 n2Size*dSize(BSH/TND)
AntiquantBaseAPI()              →  kvOffset = kvGmOffset + copyLoopIdx * copySplitS * kvStep
        ↓
CopyKV() / CopyKVPageAttention() →  DataCopyPad(srcGm[offset], srcStride=...)
```

当前全用**连续公式**，需在 stride≠0 时替换为用户传入的 `bnStride`/`n2Stride`。

---

## 3. 现状盘点：基础设施已就位

以下组件**已存在**，无需新增：

### 3.1 Tiling 数据结构

| 文件 | 行号 | 内容 | 状态 |
|------|------|------|------|
| `attention/common/op_kernel/arch35/flash_attention_score_tiling_regbase.h` | 127-148 | `StridesParams` 类 (bnStride/n2Stride + getter/setter) | ✓ |
| 同上 | 231-234 | `InputParamsRegbase.keyStrides`/`valueStrides`/`kRopeStrides` | ✓ |
| `attention/common/op_kernel/arch35/util_regbase.h` | 312-318 | `COMMON_CONST_INFO` 宏含 `keyBnStride`/`keyN2Stride`/`valueBnStride`/`valueN2Stride`/`keyRopeBnStride`/`keyRopeN2Stride` | ✓ |
| 同上 | 426-457 | `ConstInfo<isInfer,hasRope>` 通过宏展开拥有上述字段 | ✓ |

### 3.2 Host 侧解析

| 文件 | 行号 | 内容 | 状态 |
|------|------|------|------|
| `fia_tiling_info.h` | 236-247, 322-324 | `FiaTilingInfo` 含 `keyStrides`/`valueStrides`/`kRopeStrides`/`kScaleStrides`/`vScaleStrides` + `keyBnStride` 等 + `keyNonContigDim` 等 | ✓ |
| `fused_infer_attention_score_tiling_info_parser.h` | 36, 71-72, 144-158 | `GetStrides()`/`GetKvIsContiguous()`/`GetKvStrideValues()` + `keyStrides_` 等成员 | ✓ |
| `fused_infer_attention_score_tiling_info_parser.cpp` | 250-265 | `GetStrides()` 读取 `context_->GetDynamicInputStride` / `GetOptionalInputStride` | ✓ |
| 同上 | 269+ | `GetKvIsContiguous()` 调用 `CheckTensorContiguousLocal()` | ✓ |
| 同上 | 300+ | `GetKvStrideValues()` 提取 `keyBnStride_`/`keyN2Stride_` | ✓ |
| 同上 | 1582-1606 | `GenerateInfo()` 拷贝 strides 到 fiaInfo | ✓ |
| `fused_infer_attention_score_tiling_impl.cpp` | 1818-1830 | `SetFATilingData()` 写 key/value/kRope strides 到 `InputParamsRegbase` | ✓(仅k/v/kRope) |

### 3.3 算子定义 & Op API

| 文件 | 行号 | 内容 | 状态 |
|------|------|------|------|
| `fused_infer_attention_score_def.cpp` | 714, 784, 1446, 1586 | key/value/key_antiquant_scale/value_antiquant_scale 使用 `IgnoreContiguous()` | ✓ |
| `aclnn_fused_infer_attention_score_v5.cpp` | 186-200 | `CheckTensorContiguous()` 函数 | ✓ |
| 同上 | 250-254 | V2 弱符号回退 | ✓ |

### 3.4 Checker

| 文件 | 行号 | 内容 | 状态 |
|------|------|------|------|
| `base_checker.h` | 46-47 | `CheckTensorContiguous()` 声明 | ✓ |
| `base_checker.cpp` | 59-84 | `CheckTensorContiguous()` 实现（遍历 strides 找首个不连续维度） | ✓ |
| `dequant_checker.cpp` | 926-1000, 1075-1200, 1354-1400 | fullquant 场景的 0/1 轴非连续检查 | ✓(fullquant) |

---

## 4. 需要修改的文件清单

### 4.1 Kernel 侧（4 个核心文件）

#### 文件1: `flash_attention_score_antiquant_kernel.h`

**路径**: `attention/fused_infer_attention_score/op_kernel/arch35/flash_attention_score_antiquant_kernel.h`

**修改点**: `ComputeConstexpr()` (line 170-384)

**现状**: `constInfo` 已有 `keyBnStride`/`keyN2Stride`/`valueBnStride`/`valueN2Stride` 字段（来自 COMMON_CONST_INFO 宏），但**从未赋值**。

**修改**: 在 `ComputeConstexpr()` 末尾读取 tiling 中的 strides：

```cpp
// 在 line 378 (isPa block) 之后、line 380 (transposeLayout) 之前添加
if (inputParamsRegbase.keyStrides.get_bnStride() != 0) {
    constInfo.keyBnStride = inputParamsRegbase.keyStrides.get_bnStride();
    constInfo.keyN2Stride = inputParamsRegbase.keyStrides.get_n2Stride();
    constInfo.valueBnStride = inputParamsRegbase.valueStrides.get_bnStride();
    constInfo.valueN2Stride = inputParamsRegbase.valueStrides.get_n2Stride();
}
```

> **注意**: 当 stride=0 时表示连续，使用默认的轴乘积公式，不覆盖。

---

#### 文件2: `infer_flash_attention_kvcache.h`

**路径**: `attention/fused_infer_attention_score/op_kernel/arch35/infer_flash_attention_kvcache.h`

**修改点**: `ComputeOffsetForAntiquant()` (line 930-981)

**现状**:
- BSH/TND: `valueOffset = valueCoreOffset + sInnerLoopIdx * s2BaseN2Dv`（其中 `s2BaseN2Dv = s2BaseSize * n2D = s2BaseSize * n2Size * dSize`）
- BNSD: `valueOffset = valueCoreOffset + sInnerLoopIdx * s2BaseDv`（其中 `s2BaseDv = s2BaseSize * dSize`）

**问题**: `sInnerLoopIdx * s2BaseN2Dv` 假定 S2 轴是连续的，即每前进一个 s2BaseSize 行，偏移增加 `n2Size * dSize` 个元素。非连续时用户传入的 `n2Stride` 可能不等于 `n2Size * dSize`。

**修改思路**: 将 `s2BaseN2Dv` 替换为"每个 s2 base block 的实际 stride"。

```cpp
// BSH/TND layout, 非 prefix 分支 (line 962):
// 原始: runInfo.valueOffset = runParam.valueCoreOffset + sInnerLoopIdx * constInfo.s2BaseN2Dv;
// 修改:
uint64_t s2BaseStride = (constInfo.valueN2Stride != 0) ? 
    s2BaseSize * constInfo.valueN2Stride : constInfo.s2BaseN2Dv;
runInfo.valueOffset = runParam.valueCoreOffset + sInnerLoopIdx * s2BaseStride;
runInfo.keyOffset = runInfo.valueOffset;
// 同理 keyOffset 也要用 keyN2Stride
// 实际上 keyOffset = valueOffset，所以 key/value 的 stride 必须一致？
// 需要确认：如果 key 和 value 的 stride 不同，需分别计算
```

**待确认问题**: `keyOffset` 当前直接等于 `valueOffset`（line 966/975: `runInfo.keyOffset = runInfo.valueOffset`）。如果 key 和 value 有不同的非连续 stride，需要分开计算。需要检查 host 侧是否保证 key/value stride 一致，还是需要支持各自独立。

**Prefix 分支** (line 939-959): prefix 使用的是 prefixGm，不受用户 stride 影响，**不需要修改**。

**FD 分支** (line 963-964/972-973): `flashDecodeS2Idx * sInnerLoopSize * n2D` 中的 `n2D` 也需考虑 stride。

---

#### 文件3: `flash_attention_score_antiquant_block_vec.h`

**路径**: `attention/fused_infer_attention_score/op_kernel/arch35/flash_attention_score_antiquant_block_vec.h`

**修改点3a**: `setConstAntiTaskParam()` (line 557-587)

**现状** (line 567-571):
```cpp
if constexpr (layout == LayOutTypeEnum::LAYOUT_BNSD) {
    taskParam.kvStep = constInfo.dSize;
} else {
    taskParam.kvStep = taskParam.kvHeadNum * constInfo.dSize;  // = n2Size * dSize
}
```

**修改**: kvStep 表示每前进一行 KV 的元素步长。非连续时 n2Stride 可能 ≠ n2Size * dSize。

```cpp
if constexpr (layout == LayOutTypeEnum::LAYOUT_BNSD) {
    taskParam.kvStep = constInfo.dSize;  // BNSD: 行内 D 连续，stride 仅影响 B/N2 轴
} else {
    // BSH/TND: n2Stride 是 N2 维的实际步长
    taskParam.kvStep = (constInfo.keyN2Stride != 0) ? constInfo.keyN2Stride : 
                       taskParam.kvHeadNum * constInfo.dSize;
}
```

> **注意**: `kvStep` 用于 key 和 value 两个路径（`AntiquantKey` 和 `AntiquantValue`）。如果 key 和 value stride 不同，需要为 value 单独设置一个 `kvStepV`，或者分别传递 stride 到 taskParam。

**待确认**: 需要为 key 和 value 分别维护 stride 参数，还是 host 侧保证一致？PR#6430 中 fullquant 的 key/value stride 是独立传递的。

**修改点3b**: `SetAntiqParamCommon()` (line 616-657)

**现状** (line 635):
```cpp
taskParam.kvGmOffset = kvOffset + constInfo.subBlockIdx * GetRealDealSize(runInfo.s2RealSize) * taskParam.kvStep;
```

**分析**: `kvGmOffset` 是当前 s2 loop 的基础 GM 偏移。`kvOffset`（即 `runInfo.keyOffset`/`valueOffset`）已在 `ComputeOffsetForAntiquant` 中计算（已考虑 stride）。这里追加的是 `subBlockIdx * dealSize * kvStep`，用于双 vec 核切分。`kvStep` 在此表示行间步长，需与 stride 对齐。

如果 key/value stride 不同，需要分别传入。建议在 `AntiquantKey()` 调用时用 key stride，`AntiquantValue()` 调用时用 value stride。但 `SetAntiqParamCommon` 是公共函数，需增加参数或在 taskParam 中区分。

**修改方案**: 在 `AntiquantTaskParamBaseAPI` 中新增 `kvN2Stride` 字段，`SetAntiqParamCommon` 根据当前是 key 还是 value 使用对应 stride。或者更简单的方案：在调用 `SetAntiqParamCommon` 前设置 `taskParam.kvStep` 为正确的 key 或 value stride。

**修改点3c**: `InitQuant()` (line 456-500)

**现状**: 直接 `SetGlobalBuffer` 设置 `keyAntiqScaleGm`/`valueAntiqScaleGm`，无 stride 信息。

**分析**: k_descale/v_descale 的 stride 影响的是**反量化参数的读取偏移**，不是 KV 数据本身。当前 antiquant scale 的读取在 `CopyAntiqParam()` (processor.h:416) 中通过 `taskParam.antiqParamOffset = runInfo.n2oIdx * taskParam.headDim` 计算偏移，这是连续公式。

**需要传递 k_descale/v_descale 的 stride 到 taskParam**，并在 `CopyAntiqParam` 中使用。

---

#### 文件4: `flash_attention_score_antiquant_processor.h`

**路径**: `attention/fused_infer_attention_score/op_kernel/arch35/flash_attention_score_antiquant_processor.h`

**修改点4a**: `AntiquantTaskParamBaseAPI` struct (line 52-87)

**新增字段**:
```cpp
uint64_t kvBnStride;    // KV 的 Bn 轴 stride（0 表示连续）
uint64_t kvN2Stride;    // KV 的 N2 轴 stride（0 表示连续）
uint64_t scaleBnStride; // descale 的 Bn 轴 stride
uint64_t scaleN2Stride; // descale 的 N2 轴 stride
```

**修改点4b**: `CopyKV()` (line 543-584)

**现状** (line 570-575):
```cpp
if constexpr (layout == LayOutTypeEnum::LAYOUT_BNSD) {
    copyInParams.srcStride = 0;  // BNSD: 行间无间隔
} else {
    copyInParams.srcStride = copyInParams.blockLen * (kvHeadNum - 1);  // BSH/TND: 跳过其他 head
}
```

**分析**: `srcStride` 是 DataCopyPad 的源跨步，表示每复制一行后跳过多少元素。当前 BSH/TND 的 `srcStride = blockLen * (kvHeadNum - 1)` 是因为一行 KV 包含所有 head（n2Size * dSize），复制单 head 后要跳过其余 head。

非连续时，用户传入的 `n2Stride` 是 N2 维的实际步长。如果 `n2Stride != n2Size * dSize`，则 `srcStride` 需调整为 `n2Stride - dSize`（每行 head 维度数据复制后，跳到下一个 head 的步长）。

但 **BNSD 场景下 srcStride=0**，因为 BNSD 是 `[B, N, S, D]`，复制连续的 S 行，D 维是连续的。BNSD 的非连续发生在 B/N 轴（0/1轴），不是行内。所以 BNSD 的 `CopyKV` **不需要改 srcStride**，但需要改 `offset` 计算（已在 `AntiquantBaseAPI` 的 `kvOffset` 中处理）。

**修改**:
```cpp
// BSH/TND:
if (taskParam.kvN2Stride != 0) {
    copyInParams.srcStride = (taskParam.kvN2Stride / sizeof(KV_T)) - headDim;
    // 或者: copyInParams.srcStride = taskParam.kvN2Stride - copyInParams.blockLen;
} else {
    copyInParams.srcStride = copyInParams.blockLen * (kvHeadNum - 1);
}
```

> **注意**: srcStride 的单位是"块"（元素数/typeElementSize），不是字节。需仔细核对单位。

**修改点4c**: `CopyKVPageAttention()` (line 609-649)

**现状**: PA 场景的 offset 公式：
- BSH: `curOffset = (idInBlockTable * kvCacheBlockSize + reaminRowCnt) * kvHeadNum * headDim + n2Idx * headDim`
- BNSD: `curOffset = idInBlockTable * kvHeadNum * kvCacheBlockSize * headDim + n2Idx * kvCacheBlockSize * headDim + reaminRowCnt * headDim`

**分析**: PA 场景的 offset 是通过 block table 查找物理 block，然后 block 内 offset 按 head 布局计算。这里 `kvHeadNum * headDim` 是一个 block 内一行所有 head 的数据量，对应 N2 维的 stride。

如果 `n2Stride != n2Size * dSize`，则：
- BSH: `curOffset = (idInBlockTable * kvCacheBlockSize + reaminRowCnt) * kvN2Stride + n2Idx * headDim` (用 kvN2Stride 替代 kvHeadNum*headDim)

但 **PA 场景的 block 内布局是否受用户 stride 影响**需要确认。PA 场景下 KV 存储在 paged block 中，block 内是物理连续的，用户传入的 stride 描述的是**逻辑到物理的映射**。如果 block 内 stride 和 block 间 stride 机制不同，需分别处理。

> **待确认**: PA 场景下 n2Stride 是 block 内的 stride 还是 block 间的 stride？参考 fullquant 的做法是传给 `OffsetCalculator::Init()`，由 offset calculator 统一处理。Antiquant 路径没有 OffsetCalculator，需自行处理。

**修改点4d**: `CopyAntiqParam()` (line 416-442)

**现状** (line 440):
```cpp
DataCopyPad(dstLocal, srcGm[taskParam.antiqParamOffset], copyInParams, copyInPadParams);
```

其中 `antiqParamOffset = runInfo.n2oIdx * taskParam.headDim`。

**分析**: 如果 k_descale/v_descale 的 N2 轴非连续，`antiqParamOffset` 需用 `n2oIdx * scaleN2Stride` 替代 `n2oIdx * headDim`。

**修改**:
```cpp
uint64_t antiqOffset;
if (taskParam.scaleN2Stride != 0) {
    antiqOffset = taskParam.bIdx * taskParam.scaleBnStride + taskParam.n2Idx * taskParam.scaleN2Stride;
} else {
    antiqOffset = taskParam.antiqParamOffset;  // 原始逻辑
}
DataCopyPad(dstLocal, srcGm[antiqOffset], copyInParams, copyInPadParams);
```

**修改点4e**: `CopyAntiquantParamsPageAttention()` (line 508-541)

**现状** (line 533-534):
```cpp
uint64_t srcOffset = idInBlockTable * taskParam.kvCacheBlockSize * useKvHeadNum +
                     taskParam.kvCacheBlockSize * useN2Idx + remainElmenCnt;
```

**分析**: PA 场景下 descale 的 offset 也需考虑 stride。

---

### 4.2 Tiling 数据结构（2 个文件）

#### 文件5: `flash_attention_score_tiling_regbase.h`

**路径**: `attention/common/op_kernel/arch35/flash_attention_score_tiling_regbase.h`

**修改**: 在 `InputParamsRegbase` (line 231-234) 新增 k_descale/v_descale 的 stride 字段：

```cpp
// 增加strides参数
StridesParams keyStrides;
StridesParams valueStrides;
StridesParams kRopeStrides;
StridesParams kScaleStrides;   // 新增：k_descale stride
StridesParams vScaleStrides;   // 新增：v_descale stride
```

并在对应 get/set 区域添加 getter/setter。

#### 文件6: `util_regbase.h`

**路径**: `attention/common/op_kernel/arch35/util_regbase.h`

**修改**: 在 `COMMON_CONST_INFO` 宏 (line 312-318) 新增 descale stride 字段：

```cpp
/* strides */
uint64_t keyBnStride;
uint64_t keyN2Stride;
uint64_t valueBnStride;
uint64_t valueN2Stride;
uint64_t keyRopeBnStride;
uint64_t keyRopeN2Stride;
uint64_t kScaleBnStride;    // 新增
uint64_t kScaleN2Stride;    // 新增
uint64_t vScaleBnStride;    // 新增
uint64_t vScaleN2Stride     // 新增（注意末尾无逗号，因为是宏最后一行）
```

> **注意**: 如果 `CVSharedParams` 结构体 (line 507-552) 也镜像了 stride 字段，需同步新增。

---

### 4.3 Host 侧（2 个文件）

#### 文件7: `fused_infer_attention_score_tiling_impl.cpp`

**路径**: `attention/fused_infer_attention_score/op_host/arch35/fused_infer_attention_score_tiling_impl.cpp`

**修改**: `SetFATilingData()` (line 1818-1830) 补充写 k_descale/v_descale strides：

```cpp
if (fiaInfo.hasViewStride) {
    inputParams.keyStrides.set_bnStride(fiaInfo.keyBnStride);
    inputParams.keyStrides.set_n2Stride(fiaInfo.keyN2Stride);
    inputParams.valueStrides.set_bnStride(fiaInfo.valueBnStride);
    inputParams.valueStrides.set_n2Stride(fiaInfo.valueN2Stride);
    if (fiaInfo.kRopeStrides != nullptr) {
        inputParams.kRopeStrides.set_bnStride(fiaInfo.kRopeBnStride);
        inputParams.kRopeStrides.set_n2Stride(fiaInfo.kRopeN2Stride);
    }
    // 新增：k_descale / v_descale strides
    if (fiaInfo.kScaleStrides != nullptr) {
        inputParams.kScaleStrides.set_bnStride(fiaInfo.kScaleBnStride);
        inputParams.kScaleStrides.set_n2Stride(fiaInfo.kScaleN2Stride);
    }
    if (fiaInfo.vScaleStrides != nullptr) {
        inputParams.vScaleStrides.set_bnStride(fiaInfo.vScaleBnStride);
        inputParams.vScaleStrides.set_n2Stride(fiaInfo.vScaleN2Stride);
    }
}
```

> **待确认**: `fiaInfo` 中是否已有 `kScaleBnStride`/`kScaleN2Stride`/`vScaleBnStride`/`vScaleN2Stride` 字段？`fia_tiling_info.h` 中有 `kScaleStrides`/`vScaleStrides` (line 239-240) 但需确认是否有展开的 Bn/N2 字段。如果没有，需在 `fia_tiling_info.h` 新增，并在 `GetKvStrideValues()` 中提取。

#### 文件8: `dequant_checker.cpp`

**路径**: `attention/fused_infer_attention_score/op_host/checkers/dequant_checker.cpp`

**修改**: 在 antiquant 相关检查函数中新增 0/1 轴非连续校验。

**现状**: `CheckFeatureForAntiquant()` (line 2181) 调用 `CheckFeatureLayoutForAntiquant`/`CheckFeatureQuerySForAntiquant`/`CheckFeaturePAForAntiquant`/`CheckFeatureRopeForAntiquant`，**无 stride 检查**。

**修改**: 新增 `CheckStrideForAntiquant()` 方法，参照 fullquant 的做法：
1. 非 PA 场景：kv/kvscale 必须连续，否则报错
2. PA 场景：kv/kvscale 仅允许 0/1 轴非连续，2+ 轴非连续报错

```cpp
ge::graphStatus DequantChecker::CheckStrideForAntiquant(const FiaTilingInfo &fiaInfo)
{
    if (!fiaInfo.hasViewStride) {
        return ge::GRAPH_SUCCESS;
    }
    // 非 PA 场景不支持非连续
    if (!fiaInfo.pageAttentionFlag) {
        int32_t dimIndex = 0;
        // 检查 k/v/k_descale/v_descale 是否连续
        OP_CHECK_IF((CheckTensorContiguous(...) != ge::GRAPH_SUCCESS),
            OP_LOGE(fiaInfo.opName, "In non-PA antiquant scenarios, non-contiguous tensors are not supported."),
            return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }
    // PA 场景仅支持 0/1 轴非连续
    int32_t dimIndex = 0;
    // 对 k/v/k_descale/v_descale 分别检查
    OP_CHECK_IF(((CheckTensorContiguous(...) != ge::GRAPH_SUCCESS) && (dimIndex != 0 && dimIndex != 1)),
        OP_LOGE(fiaInfo.opName, "In PA antiquant scenarios, only dimensions 0/1 non-contiguous are supported."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}
```

并在 `CheckFeatureForAntiquant()` 中调用。

---

## 5. 数据流详图

### 5.1 非连续 stride 传递链路

```
Host 侧:
  context_->GetDynamicInputStride(KEY_INDEX, 0)     → keyStrides_
  context_->GetOptionalInputStride(KEY_ANTIQUANT_SCALE_INDEX) → kScaleStrides_
  context_->GetOptionalInputStride(VALUE_ANTIQUANT_SCALE_INDEX) → vScaleStrides_
          ↓
  GetKvStrideValues() 提取 keyBnStride_/keyN2Stride_ 等
          ↓
  GenerateInfo() → fiaInfo.keyBnStride / fiaInfo.kScaleBnStride 等
          ↓
  SetFATilingData() → inputParamsRegbase.keyStrides / kScaleStrides 等
          ↓
Kernel 侧:
  ComputeConstexpr() → constInfo.keyBnStride / kScaleBnStride 等
          ↓
  ComputeOffsetForAntiquant() → runInfo.keyOffset (用 n2Stride 替代 n2Size*dSize)
          ↓
  setConstAntiTaskParam() → taskParam.kvStep (用 n2Stride)
          ↓
  SetAntiqParamCommon() → taskParam.kvGmOffset
          ↓
  AntiquantBaseAPI() → kvOffset = kvGmOffset + copyLoopIdx * copySplitS * kvStep
          ↓
  CopyKV() → DataCopyPad(srcGm[offset], srcStride=用 n2Stride 算)
  CopyAntiqParam() → DataCopyPad(srcGm[scaleOffset], ...) (用 scaleN2Stride 算)
```

### 5.2 PA 场景 offset 对比

| 场景 | 连续（当前） | 非连续（目标） |
|------|-------------|---------------|
| BSH block内 | `(blockId*blockSize+row) * n2*d + n2Idx*d` | `(blockId*blockSize+row) * n2Stride + n2Idx*d` |
| BNSD block内 | `blockId*n2*blockSize*d + n2Idx*blockSize*d + row*d` | 需分析 BnStride/N2Stride 如何映射到 block 内 |

> **PA 场景复杂点**: paged attention 的 block 是物理内存块，block 内数据布局是固定的。用户的 stride 描述的是逻辑张量的步长，在 PA 场景下可能只影响 **block 间**的寻址，不影响 block 内。需要确认 stride 是逻辑级还是物理级。

---

## 6. 待确认问题

以下问题需要在实施前讨论确认：

### Q1: key 和 value 的 stride 是否可以不同？

**现状**: `ComputeOffsetForAntiquant` 中 `keyOffset = valueOffset`（直接赋值）。
**影响**: 如果 key/value 的 n2Stride 不同，需分别计算 keyOffset 和 valueOffset。
**建议**: 参照 fullquant PR，key/value stride 独立传递。需修改 `ComputeOffsetForAntiquant` 分开计算。
**优先级**: 高

### Q2: kvStep 在 key/value stride 不同时如何处理？

**现状**: `taskParam.kvStep` 是单一值，用于 `AntiquantBaseAPI` 的 offset 计算和 `CopyKV` 的 srcStride 计算。
**影响**: 如果 key 和 value 用不同 stride，需在 `AntiquantKey()` 和 `AntiquantValue()` 调用前分别设置 `taskParam.kvStep`。
**建议**: `AntiquantKey()` 调用 `SetAntiqParamCommon` 时用 key stride，`AntiquantValue()` 调用时用 value stride。因为 `SetAntiqParamCommon` 已经接收不同 offset（keyOffset vs valueOffset），只需确保 kvStep 也对应即可。
**优先级**: 高

### Q3: PA 场景下 stride 是 block 间还是 block 内？

**现状**: PA 场景的 `CopyKVPageAttention` 通过 block table 查找物理 block，block 内 offset 按固定公式计算。
**影响**: 如果 stride 仅影响逻辑到物理 block 的映射（即哪些 block 属于哪个 batch/head），则 block 内 offset 公式不变。如果 stride 影响 block 内数据布局，则需修改 block 内 offset。
**参考**: fullquant PR 将 stride 传给 `OffsetCalculator::Init()`，由 offset calculator 统一处理 block 内/间寻址。Antiquant 没有 OffsetCalculator，需自行实现等价逻辑。
**优先级**: 高

### Q4: k_descale/v_descale 的 stride 含义

**现状**: antiquant scale（k_descale/v_descale）的 shape 在 PA 场景为 `[Bn, N2, ...]` 或 `[Bn, N2, Bs, ...]`。当前 `CopyAntiqParam` 用 `antiqParamOffset = n2oIdx * headDim` 计算。
**影响**: 如果 descale 的 N2 轴非连续，`antiqParamOffset` 需用 `n2oIdx * scaleN2Stride`。
**待确认**: descale 在 PA 场景下的 shape 和 stride 语义是否与 KV 相同？
**优先级**: 中

### Q5: `fia_tiling_info.h` 是否需新增 kScaleBnStride 等字段？

**现状**: `fia_tiling_info.h` 有 `kScaleStrides` (Stride 指针) 但可能没有展开的 `kScaleBnStride`/`kScaleN2Stride` 整数字段。
**影响**: `SetFATilingData` 需要这些字段来写入 tiling data。
**建议**: 参照 `keyBnStride`/`keyN2Stride` (line 242-243) 的做法，在 `fia_tiling_info.h` 新增 `kScaleBnStride`/`kScaleN2Stride`/`vScaleBnStride`/`vScaleN2Stride`，并在 `GetKvStrideValues()` 中提取。
**优先级**: 中

### Q6: `CVSharedParams` 是否需同步新增 stride 字段？

**现状**: `util_regbase.h` 的 `CVSharedParams<true, true>` (line 507-552) 镜像了 `keyBnStride` 等 6 个 stride 字段。
**影响**: 如果 antiquant 路径使用 CVSharedParams 传递 stride，需同步新增 kScale/vScale stride。
**优先级**: 低（需确认是否使用）

### Q7: srcStride 单位问题

**现状**: `DataCopyPad` 的 `copyInParams.srcStride` 单位是"块数"（元素数/typeElementSize），不是字节也不是元素数。
**影响**: 用户传入的 stride 单位是元素数，需转换为 DataCopyPad 的块单位。
**公式**: `srcStride_in_blocks = (n2Stride_in_elements - headDim) / typeElementSize`
**优先级**: 中

### Q8: BNSD layout 非连续的 offset 计算

**现状**: BNSD 的 `CopyKV` srcStride=0（行间无间隔），offset 通过 `AntiquantBaseAPI` 的 `kvOffset = kvGmOffset + copyLoopIdx * copySplitS * kvStep` 计算，其中 `kvStep = dSize`。
**影响**: BNSD 的非连续在 B/N 轴（0/1轴），`kvGmOffset` 已在 `ComputeOffsetForAntiquant` 中用 bnStride/n2Stride 计算。但 `kvStep` 是行内步长（dSize），BNSD 行内 D 连续，所以 `kvStep` 不变。`CopyKV` 的 srcStride 也不变（0）。
**结论**: BNSD layout **可能不需要改 CopyKV**，只需改 offset 计算。但需确认 `valueCoreOffset` 是否已包含 bnStride/n2Stride 的逻辑。
**优先级**: 中

---

## 7. 实施顺序建议

```
Phase 1: Tiling 数据结构扩展（文件5、6）
  ├── InputParamsRegbase 新增 kScaleStrides/vScaleStrides
  ├── COMMON_CONST_INFO 宏新增 kScale/vScale stride 字段
  └── (若需) fia_tiling_info.h 新增 kScaleBnStride 等 + GetKvStrideValues 提取

Phase 2: Host 侧（文件7、8）
  ├── SetFATilingData 写 kScale/vScale strides
  └── dequant_checker 新增 antiquant stride 校验

Phase 3: Kernel 侧接线（文件1）
  └── ComputeConstexpr 读取 strides 到 constInfo

Phase 4: Kernel 侧 offset 计算（文件2、3）
  ├── ComputeOffsetForAntiquant 用 stride 替代连续公式
  ├── setConstAntiTaskParam 设置 kvStep（含 key/value 分别处理）
  └── SetAntiqParamCommon 传递 stride 到 taskParam

Phase 5: Kernel 侧数据搬运（文件3、4）
  ├── AntiquantTaskParamBaseAPI 新增 stride 字段
  ├── CopyKV srcStride / offset 用 stride
  ├── CopyKVPageAttention offset 用 stride
  ├── CopyAntiqParam offset 用 scaleN2Stride
  └── CopyAntiquantParamsPageAttention offset 用 stride

Phase 6: 测试验证
  ├── 冒烟测试（连续场景不退化）
  ├── PA + 0轴非连续
  ├── PA + 1轴非连续
  └── 非 PA + 非连续（应报错）
```

---

## 8. 风险与注意事项

1. **连续场景不能退化**: stride=0 时走原逻辑，不引入额外开销
2. **key/value stride 独立性**: 当前 `keyOffset = valueOffset`，需拆分
3. **PA block 内/间寻址**: 需确认 stride 语义，避免 block 内 offset 错误
4. **DataCopyPad srcStride 单位**: 元素数→块数的转换
5. **双 vec 核切分**: `subBlockIdx * dealSize * kvStep` 中的 kvStep 需用正确 stride
6. **FD(flash decode) 场景**: `flashDecodeS2Idx * sInnerLoopSize * n2D` 也需考虑 stride
7. **编译验证**: COMMON_CONST_INFO 宏修改影响所有 ConstInfo 特化，需确保编译通过

---

## 9. 文件清单汇总

| # | 文件路径 | 修改内容 | Phase |
|---|---------|---------|-------|
| 1 | `attention/fused_infer_attention_score/op_kernel/arch35/flash_attention_score_antiquant_kernel.h` | `ComputeConstexpr()` 读取 stride | 3 |
| 2 | `attention/fused_infer_attention_score/op_kernel/arch35/infer_flash_attention_kvcache.h` | `ComputeOffsetForAntiquant()` 用 stride | 4 |
| 3 | `attention/fused_infer_attention_score/op_kernel/arch35/flash_attention_score_antiquant_block_vec.h` | `setConstAntiTaskParam`/`SetAntiqParamCommon`/`InitQuant` | 4-5 |
| 4 | `attention/fused_infer_attention_score/op_kernel/arch35/flash_attention_score_antiquant_processor.h` | `AntiquantTaskParamBaseAPI`/`CopyKV`/`CopyKVPageAttention`/`CopyAntiqParam` | 5 |
| 5 | `attention/common/op_kernel/arch35/flash_attention_score_tiling_regbase.h` | `InputParamsRegbase` 新增 kScale/vScale stride | 1 |
| 6 | `attention/common/op_kernel/arch35/util_regbase.h` | `COMMON_CONST_INFO` 宏新增 stride 字段 | 1 |
| 7 | `attention/fused_infer_attention_score/op_host/arch35/fused_infer_attention_score_tiling_impl.cpp` | `SetFATilingData()` 写 kScale/vScale strides | 2 |
| 8 | `attention/fused_infer_attention_score/op_host/checkers/dequant_checker.cpp` | 新增 antiquant stride 校验 | 2 |
| (9) | `attention/fused_infer_attention_score/op_host/fia_tiling_info.h` | (若需) 新增 kScaleBnStride 等字段 | 1 |
| (10) | `attention/fused_infer_attention_score/op_host/fused_infer_attention_score_tiling_info_parser.cpp` | (若需) GetKvStrideValues 提取 kScale/vScale stride | 1 |
