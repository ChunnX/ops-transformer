# FIA Antiquant 算子 PA 场景 0/1 轴非连续支持方案

## 1. 目标

在伪量化(antiquant) FA 算子上，支持 PA 场景下 `k`/`v`/`k_descale`/`v_descale` 的 0轴(Bn)和1轴(N2)非连续。

支持 **3 种 PA 布局**：

| 布局 | shape | antiquant paKvShapeType |
|------|-------|------------------------|
| PA_BNBD | `[Bn, N2, Bs, D]` | KV_CACHE_BNSD=1 |
| PA_BBND | `[Bn, Bs, N2, D]` | KV_CACHE_BSH=0 |
| PA_NZ | `[Bn, N2, D1, Bs, D0]` | KV_CACHE_NZ=2 |

约束：非 PA 不支持非连续；仅 0/1 轴可非连续，block 内(2+轴)必须连续。

---

## 2. 三种布局的 stride 语义

| | PA_BNBD `[Bn,N2,Bs,D]` | PA_BBND `[Bn,Bs,N2,D]` | PA_NZ `[Bn,N2,D1,Bs,D0]` |
|---|---|---|---|
| n2Stride 连续值 | Bs×D | D | D1×Bs×D0 = D×Bs |
| n2Stride 最小约束 | ≥ Bs×D | ≥ D | ≥ D×Bs |
| bnStride 连续值 | N2×Bs×D | Bs×N2×D | N2×D×Bs |
| bnStride 约束 | ≥ N2×n2Stride | ≥ Bs×N2×n2Stride | ≥ N2×n2Stride |
| bsStride | D (固定) | N2×n2Stride (**受n2Stride影响**) | D0 (固定) |
| kvStep(非PA, BSH/TND) | D (不变) | N2×n2Stride | N/A |

> BBND 特殊性：行步长 bsStride = N2 × n2Stride，非连续时随 n2Stride 变化。

> srcStride 单位：arch35 上 `DataCopyExtParams.srcStride` 为 `int64_t`，单位是**字节**。

---

## 3. 文件1: `flash_attention_score_tiling_regbase.h`

**路径**: `attention/common/op_kernel/arch35/flash_attention_score_tiling_regbase.h`

### 修改点1: InputParamsRegbase 新增字段 (line 231-234)

**现有代码**:
```cpp
    // 增加strides参数
    StridesParams keyStrides;
    StridesParams valueStrides;
    StridesParams kRopeStrides;
```

**改为**:
```cpp
    // 增加strides参数
    StridesParams keyStrides;
    StridesParams valueStrides;
    StridesParams kRopeStrides;
    StridesParams kScaleStrides;
    StridesParams vScaleStrides;
```

### 修改点2: 新增 getter/setter (line 872, 在 `}` 闭合 `InputParamsRegbase` 类之前)

**现有代码** (line 864-873):
```cpp
    void set_kRopeStrides(StridesParams kRopeStrides)
    {
        this->kRopeStrides = kRopeStrides;
    }

    StridesParams get_kRopeStrides() const
    {
        return kRopeStrides;
    }
};
```

**改为**:
```cpp
    void set_kRopeStrides(StridesParams kRopeStrides)
    {
        this->kRopeStrides = kRopeStrides;
    }

    StridesParams get_kRopeStrides() const
    {
        return kRopeStrides;
    }

    void set_kScaleStrides(StridesParams kScaleStrides)
    {
        this->kScaleStrides = kScaleStrides;
    }

    StridesParams get_kScaleStrides() const
    {
        return kScaleStrides;
    }

    void set_vScaleStrides(StridesParams vScaleStrides)
    {
        this->vScaleStrides = vScaleStrides;
    }

    StridesParams get_vScaleStrides() const
    {
        return vScaleStrides;
    }
};
```

---

## 4. 文件2: `util_regbase.h`

**路径**: `attention/common/op_kernel/arch35/util_regbase.h`

### 修改点1: COMMON_CONST_INFO 宏 (line 312-318)

**现有代码**:
```cpp
    /* strides */                                                                                                      \
    uint64_t keyBnStride;                                                                                              \
    uint64_t keyN2Stride;                                                                                              \
    uint64_t valueBnStride;                                                                                            \
    uint64_t valueN2Stride;                                                                                            \
    uint64_t keyRopeBnStride;                                                                                          \
    uint64_t keyRopeN2Stride
```

**改为**:
```cpp
    /* strides */                                                                                                      \
    uint64_t keyBnStride;                                                                                              \
    uint64_t keyN2Stride;                                                                                              \
    uint64_t valueBnStride;                                                                                            \
    uint64_t valueN2Stride;                                                                                            \
    uint64_t keyRopeBnStride;                                                                                          \
    uint64_t keyRopeN2Stride;                                                                                          \
    uint64_t kScaleBnStride;                                                                                           \
    uint64_t kScaleN2Stride;                                                                                           \
    uint64_t vScaleBnStride;                                                                                           \
    uint64_t vScaleN2Stride
```

### 修改点2: CVSharedParams<true, true> (line 540-545)

**现有代码**:
```cpp
    uint64_t keyBnStride;
    uint64_t keyN2Stride;
    uint64_t valueBnStride;
    uint64_t valueN2Stride;
    uint64_t keyRopeBnStride;
    uint64_t keyRopeN2Stride;
```

**改为**:
```cpp
    uint64_t keyBnStride;
    uint64_t keyN2Stride;
    uint64_t valueBnStride;
    uint64_t valueN2Stride;
    uint64_t keyRopeBnStride;
    uint64_t keyRopeN2Stride;
    uint64_t kScaleBnStride;
    uint64_t kScaleN2Stride;
    uint64_t vScaleBnStride;
    uint64_t vScaleN2Stride;
```

---

## 5. 文件3: `fused_infer_attention_score_tiling_impl.cpp`

**路径**: `attention/fused_infer_attention_score/op_host/arch35/fused_infer_attention_score_tiling_impl.cpp`

### 修改点: SetFATilingData (line 1822-1831)

**现有代码**:
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
    }
```

**改为**:
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
        if (fiaInfo.kScaleStrides != nullptr && fiaInfo.kScaleStrides->GetDimNum() > 0) {
            inputParams.kScaleStrides.set_bnStride(fiaInfo.kScaleStrides->GetStride(0));
            inputParams.kScaleStrides.set_n2Stride(
                fiaInfo.kScaleStrides->GetDimNum() > 1 ? fiaInfo.kScaleStrides->GetStride(1) : 0);
        }
        if (fiaInfo.vScaleStrides != nullptr && fiaInfo.vScaleStrides->GetDimNum() > 0) {
            inputParams.vScaleStrides.set_bnStride(fiaInfo.vScaleStrides->GetStride(0));
            inputParams.vScaleStrides.set_n2Stride(
                fiaInfo.vScaleStrides->GetDimNum() > 1 ? fiaInfo.vScaleStrides->GetStride(1) : 0);
        }
    }
```

---

## 6. 文件4: `dequant_checker.cpp` / `dequant_checker.h`

**路径**: `attention/fused_infer_attention_score/op_host/checkers/`

### 修改点1: `dequant_checker.h` 新增声明 (line 124 后)

**现有代码** (line 120-124):
```cpp
    ge::graphStatus CheckFeatureForAntiquant(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckFeatureLayoutForAntiquant(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckFeatureQuerySForAntiquant(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckFeaturePAForAntiquant(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckFeatureRopeForAntiquant(const FiaTilingInfo &fiaInfo) const;
```

**改为** (新增一行):
```cpp
    ge::graphStatus CheckFeatureForAntiquant(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckFeatureLayoutForAntiquant(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckFeatureQuerySForAntiquant(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckFeaturePAForAntiquant(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckFeatureRopeForAntiquant(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckStrideForAntiquant(const FiaTilingInfo &fiaInfo);
```

### 修改点2: `dequant_checker.cpp` CheckFeatureForAntiquant (line 2181-2190)

**现有代码**:
```cpp
ge::graphStatus DequantChecker::CheckFeatureForAntiquant(const FiaTilingInfo &fiaInfo)
{
    if (ge::GRAPH_SUCCESS != CheckFeatureLayoutForAntiquant(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckFeatureQuerySForAntiquant(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckFeaturePAForAntiquant(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckFeatureRopeForAntiquant(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}
```

**改为**:
```cpp
ge::graphStatus DequantChecker::CheckFeatureForAntiquant(const FiaTilingInfo &fiaInfo)
{
    if (ge::GRAPH_SUCCESS != CheckFeatureLayoutForAntiquant(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckFeatureQuerySForAntiquant(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckFeaturePAForAntiquant(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckFeatureRopeForAntiquant(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckStrideForAntiquant(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}
```

### 修改点3: 新增 `CheckStrideForAntiquant` 函数 (在 line 2190 后插入)

```cpp
ge::graphStatus DequantChecker::CheckStrideForAntiquant(const FiaTilingInfo &fiaInfo)
{
    if (!fiaInfo.hasViewStride) {
        return ge::GRAPH_SUCCESS;
    }
    const gert::Shape keyShape = fiaInfo.opParamInfo.key.shape->GetStorageShape();
    const uint32_t keyDimNum = keyShape.GetDimNum();
    const gert::Shape valueShape = fiaInfo.opParamInfo.value.shape->GetStorageShape();
    const uint32_t valueDimNum = valueShape.GetDimNum();
    int32_t dimIndex = 0;

    if (!fiaInfo.pageAttentionFlag) {
        OP_CHECK_IF((CheckTensorContiguous(keyDimNum, keyShape, fiaInfo.keyStrides, dimIndex) != ge::GRAPH_SUCCESS),
            OP_LOGE(fiaInfo.opName, "In non-PA antiquant scenarios, non-contiguous key is not supported."),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF((CheckTensorContiguous(valueDimNum, valueShape, fiaInfo.valueStrides, dimIndex) != ge::GRAPH_SUCCESS),
            OP_LOGE(fiaInfo.opName, "In non-PA antiquant scenarios, non-contiguous value is not supported."),
            return ge::GRAPH_FAILED);
        if (fiaInfo.opParamInfo.keyAntiquantScale.tensor != nullptr) {
            const auto &scaleShape = fiaInfo.opParamInfo.keyAntiquantScale.tensor->GetStorageShape();
            OP_CHECK_IF((CheckTensorContiguous(scaleShape.GetDimNum(), scaleShape, fiaInfo.kScaleStrides, dimIndex) != ge::GRAPH_SUCCESS),
                OP_LOGE(fiaInfo.opName, "In non-PA antiquant scenarios, non-contiguous k_descale is not supported."),
                return ge::GRAPH_FAILED);
        }
        if (fiaInfo.opParamInfo.valueAntiquantScale.tensor != nullptr) {
            const auto &scaleShape = fiaInfo.opParamInfo.valueAntiquantScale.tensor->GetStorageShape();
            OP_CHECK_IF((CheckTensorContiguous(scaleShape.GetDimNum(), scaleShape, fiaInfo.vScaleStrides, dimIndex) != ge::GRAPH_SUCCESS),
                OP_LOGE(fiaInfo.opName, "In non-PA antiquant scenarios, non-contiguous v_descale is not supported."),
                return ge::GRAPH_FAILED);
        }
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(((CheckTensorContiguous(keyDimNum, keyShape, fiaInfo.keyStrides, dimIndex) != ge::GRAPH_SUCCESS) &&
                 (dimIndex != 0 && dimIndex != 1)),
        OP_LOGE(fiaInfo.opName, "In PA antiquant scenarios, key only supports non-contiguous in dimensions 0/1, but got dimension %d.", dimIndex),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(((CheckTensorContiguous(valueDimNum, valueShape, fiaInfo.valueStrides, dimIndex) != ge::GRAPH_SUCCESS) &&
                 (dimIndex != 0 && dimIndex != 1)),
        OP_LOGE(fiaInfo.opName, "In PA antiquant scenarios, value only supports non-contiguous in dimensions 0/1, but got dimension %d.", dimIndex),
        return ge::GRAPH_FAILED);
    if (fiaInfo.opParamInfo.keyAntiquantScale.tensor != nullptr) {
        const auto &scaleShape = fiaInfo.opParamInfo.keyAntiquantScale.tensor->GetStorageShape();
        OP_CHECK_IF(((CheckTensorContiguous(scaleShape.GetDimNum(), scaleShape, fiaInfo.kScaleStrides, dimIndex) != ge::GRAPH_SUCCESS) &&
                     (dimIndex != 0 && dimIndex != 1)),
            OP_LOGE(fiaInfo.opName, "In PA antiquant scenarios, k_descale only supports non-contiguous in dimensions 0/1, but got dimension %d.", dimIndex),
            return ge::GRAPH_FAILED);
    }
    if (fiaInfo.opParamInfo.valueAntiquantScale.tensor != nullptr) {
        const auto &scaleShape = fiaInfo.opParamInfo.valueAntiquantScale.tensor->GetStorageShape();
        OP_CHECK_IF(((CheckTensorContiguous(scaleShape.GetDimNum(), scaleShape, fiaInfo.vScaleStrides, dimIndex) != ge::GRAPH_SUCCESS) &&
                     (dimIndex != 0 && dimIndex != 1)),
            OP_LOGE(fiaInfo.opName, "In PA antiquant scenarios, v_descale only supports non-contiguous in dimensions 0/1, but got dimension %d.", dimIndex),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}
```

---

## 7. 文件5: `flash_attention_score_antiquant_kernel.h`

**路径**: `attention/fused_infer_attention_score/op_kernel/arch35/flash_attention_score_antiquant_kernel.h`

### 修改点: ComputeConstexpr (line 378-380 之间)

**现有代码** (line 373-383):
```cpp
    if constexpr (isPa) {
        this->constInfo.blockTableDim2 = inputParamsRegbase.blockTableDim2;
        this->constInfo.blockSize = inputParamsRegbase.blockSize;
        this->constInfo.paLayoutType = inputParamsRegbase.paLayoutType;
        this->constInfo.paBlockNumSum = inputParamsRegbase.paBlockNumSum;
    }

    this->constInfo.transposeLayout = inputParamsRegbase.transposeLayout;
```

**改为** (在 `}` 和空行之间插入):
```cpp
    if constexpr (isPa) {
        this->constInfo.blockTableDim2 = inputParamsRegbase.blockTableDim2;
        this->constInfo.blockSize = inputParamsRegbase.blockSize;
        this->constInfo.paLayoutType = inputParamsRegbase.paLayoutType;
        this->constInfo.paBlockNumSum = inputParamsRegbase.paBlockNumSum;
    }
    if (inputParamsRegbase.keyStrides.get_bnStride() != 0) {
        constInfo.keyBnStride = inputParamsRegbase.keyStrides.get_bnStride();
        constInfo.keyN2Stride = inputParamsRegbase.keyStrides.get_n2Stride();
    }
    if (inputParamsRegbase.valueStrides.get_bnStride() != 0) {
        constInfo.valueBnStride = inputParamsRegbase.valueStrides.get_bnStride();
        constInfo.valueN2Stride = inputParamsRegbase.valueStrides.get_n2Stride();
    }
    if (inputParamsRegbase.kScaleStrides.get_bnStride() != 0) {
        constInfo.kScaleBnStride = inputParamsRegbase.kScaleStrides.get_bnStride();
        constInfo.kScaleN2Stride = inputParamsRegbase.kScaleStrides.get_n2Stride();
    }
    if (inputParamsRegbase.vScaleStrides.get_bnStride() != 0) {
        constInfo.vScaleBnStride = inputParamsRegbase.vScaleStrides.get_bnStride();
        constInfo.vScaleN2Stride = inputParamsRegbase.vScaleStrides.get_n2Stride();
    }

    this->constInfo.transposeLayout = inputParamsRegbase.transposeLayout;
```

---

## 8. 文件6: `infer_flash_attention_kvcache.h`

**路径**: `attention/fused_infer_attention_score/op_kernel/arch35/infer_flash_attention_kvcache.h`

### 修改点1: CalcValueCoreOffsetBSH (line 294-303)

**现有代码**:
```cpp
TEMPLATE_INTF
__aicore__ inline void CalcValueCoreOffsetBSH(RunParamStr<isInfer> &runParam,
                                              const ConstInfo<isInfer, hasRope> &constInfo, int32_t bIdx)
{
    uint64_t valueInnerOffsetSize = 0;
    if (constInfo.isKvContinuous == 1) {
        valueInnerOffsetSize = bIdx * constInfo.n2S2Dv + runParam.kvLeftPaddingSize * constInfo.n2Dv;
    }
    runParam.valueCoreOffset = valueInnerOffsetSize + runParam.n2oIdx * constInfo.dSizeV;
}
```

**改为**:
```cpp
TEMPLATE_INTF
__aicore__ inline void CalcValueCoreOffsetBSH(RunParamStr<isInfer> &runParam,
                                              const ConstInfo<isInfer, hasRope> &constInfo, int32_t bIdx)
{
    uint64_t valueInnerOffsetSize = 0;
    uint64_t headStride = (constInfo.valueN2Stride != 0) ? constInfo.valueN2Stride : constInfo.dSizeV;
    if (constInfo.isKvContinuous == 1) {
        uint64_t batchStride = (constInfo.valueBnStride != 0) ? constInfo.valueBnStride : constInfo.n2S2Dv;
        valueInnerOffsetSize = bIdx * batchStride + runParam.kvLeftPaddingSize * constInfo.n2Dv;
    }
    runParam.valueCoreOffset = valueInnerOffsetSize + runParam.n2oIdx * headStride;
}
```

### 修改点2: CalcValueCoreOffsetTND (line 305-317)

**现有代码**:
```cpp
TEMPLATE_INTF
__aicore__ inline void CalcValueCoreOffsetTND(RunParamStr<isInfer> &runParam,
                                              const ConstInfo<isInfer, hasRope> &constInfo, int32_t bIdx,
                                              __gm__ int64_t *actualSeqKvlenAddr)
{
    uint64_t valueInnerOffsetSize = 0;
    if constexpr (!isPa) {
        valueInnerOffsetSize = (bIdx == 0) ? 0 : actualSeqKvlenAddr[bIdx - 1] * constInfo.n2Dv;
    } else {
        valueInnerOffsetSize = bIdx * constInfo.n2S2Dv;
    }
    runParam.valueCoreOffset = valueInnerOffsetSize + runParam.n2oIdx * constInfo.dSizeV;
}
```

**改为**:
```cpp
TEMPLATE_INTF
__aicore__ inline void CalcValueCoreOffsetTND(RunParamStr<isInfer> &runParam,
                                              const ConstInfo<isInfer, hasRope> &constInfo, int32_t bIdx,
                                              __gm__ int64_t *actualSeqKvlenAddr)
{
    uint64_t valueInnerOffsetSize = 0;
    uint64_t headStride = (constInfo.valueN2Stride != 0) ? constInfo.valueN2Stride : constInfo.dSizeV;
    if constexpr (!isPa) {
        valueInnerOffsetSize = (bIdx == 0) ? 0 : actualSeqKvlenAddr[bIdx - 1] * constInfo.n2Dv;
    } else {
        uint64_t batchStride = (constInfo.valueBnStride != 0) ? constInfo.valueBnStride : constInfo.n2S2Dv;
        valueInnerOffsetSize = bIdx * batchStride;
    }
    runParam.valueCoreOffset = valueInnerOffsetSize + runParam.n2oIdx * headStride;
}
```

### 修改点3: CalcValueCoreOffsetBNSD (line 337-351)

**现有代码**:
```cpp
TEMPLATE_INTF
__aicore__ inline void CalcValueCoreOffsetBNSD(RunParamStr<isInfer> &runParam,
                                               const ConstInfo<isInfer, hasRope> &constInfo, int32_t bIdx)
{
    uint64_t headStrideV = 0;
    uint64_t valueInnerOffsetSize = 0;
    if (constInfo.isKvContinuous == 1) {
        headStrideV = constInfo.s2Dv;
        valueInnerOffsetSize = bIdx * constInfo.n2Size * headStrideV + runParam.kvLeftPaddingSize * constInfo.dSizeV;
    } else {
        headStrideV = constInfo.dSizeV * runParam.s2InCurrentBatch;
        valueInnerOffsetSize = 0;
    }
    runParam.valueCoreOffset = valueInnerOffsetSize + runParam.n2oIdx * headStrideV;
}
```

**改为**:
```cpp
TEMPLATE_INTF
__aicore__ inline void CalcValueCoreOffsetBNSD(RunParamStr<isInfer> &runParam,
                                               const ConstInfo<isInfer, hasRope> &constInfo, int32_t bIdx)
{
    uint64_t headStrideV = 0;
    uint64_t valueInnerOffsetSize = 0;
    if (constInfo.isKvContinuous == 1) {
        headStrideV = (constInfo.valueN2Stride != 0) ? constInfo.valueN2Stride : constInfo.s2Dv;
        uint64_t batchStride = (constInfo.valueBnStride != 0) ? constInfo.valueBnStride : constInfo.n2Size * headStrideV;
        valueInnerOffsetSize = bIdx * batchStride + runParam.kvLeftPaddingSize * constInfo.dSizeV;
    } else {
        headStrideV = constInfo.dSizeV * runParam.s2InCurrentBatch;
        valueInnerOffsetSize = 0;
    }
    runParam.valueCoreOffset = valueInnerOffsetSize + runParam.n2oIdx * headStrideV;
}
```

### 修改点4: SyncKeyCoreOffset (line 353-362)

**现有代码**:
```cpp
TEMPLATE_INTF
__aicore__ inline void SyncKeyCoreOffset(RunParamStr<isInfer> &runParam, const ConstInfo<isInfer, hasRope> &constInfo,
                                         int32_t bIdx, __gm__ int64_t *actualSeqKvlenAddr)
{
    if (unlikely(constInfo.dSize != constInfo.dSizeV)) {
        GetKeyCoreOffsetParam<TEMPLATE_INTF_ARGS>(runParam, constInfo, bIdx, actualSeqKvlenAddr);
    } else {
        runParam.keyCoreOffset = runParam.valueCoreOffset;
    }
}
```

**改为**:
```cpp
TEMPLATE_INTF
__aicore__ inline void SyncKeyCoreOffset(RunParamStr<isInfer> &runParam, const ConstInfo<isInfer, hasRope> &constInfo,
                                         int32_t bIdx, __gm__ int64_t *actualSeqKvlenAddr)
{
    if (unlikely(constInfo.dSize != constInfo.dSizeV) ||
        constInfo.keyBnStride != 0 || constInfo.keyN2Stride != 0 ||
        constInfo.valueBnStride != 0 || constInfo.valueN2Stride != 0) {
        GetKeyCoreOffsetParam<TEMPLATE_INTF_ARGS>(runParam, constInfo, bIdx, actualSeqKvlenAddr);
    } else {
        runParam.keyCoreOffset = runParam.valueCoreOffset;
    }
}
```

> `GetKeyCoreOffsetParam` (line 258-292) 也需同步修改：用 `keyBnStride`/`keyN2Stride` 替代连续值，与 `CalcValueCoreOffsetBSH/BNSD/TND` 对称。

### 修改点5: ComputeOffsetForAntiquant (line 960-979)

**现有代码** (非 prefix 分支):
```cpp
    } else {
        if constexpr (layout == LayOutTypeEnum::LAYOUT_BSH || layout == LayOutTypeEnum::LAYOUT_TND) {
            runInfo.valueOffset = runParam.valueCoreOffset + sInnerLoopIdx * constInfo.s2BaseN2Dv;
            if constexpr (isFd) {
                runInfo.valueOffset += runInfo.flashDecodeS2Idx * constInfo.sInnerLoopSize * constInfo.n2D;
            }
            runInfo.keyOffset = runInfo.valueOffset;
            if constexpr (hasRope) {
                runInfo.kRopeOffset = runParam.kRopeNBGOffset + sInnerLoopIdx * constInfo.s2BaseN2DR;
            }
        } else {
            runInfo.valueOffset = runParam.valueCoreOffset + sInnerLoopIdx * constInfo.s2BaseDv;
            if constexpr (isFd) {
                runInfo.valueOffset += runInfo.flashDecodeS2Idx * constInfo.sInnerLoopSize * constInfo.dSize;
            }
            runInfo.keyOffset = runInfo.valueOffset;
            if constexpr (hasRope) {
                runInfo.kRopeOffset = runParam.kRopeNBGOffset + sInnerLoopIdx * constInfo.s2BaseDR;
            }
        }
    }
```

**改为**:
```cpp
    } else {
        if constexpr (layout == LayOutTypeEnum::LAYOUT_BSH || layout == LayOutTypeEnum::LAYOUT_TND) {
            uint64_t s2BaseStrideV = constInfo.s2BaseN2Dv;
            if (constInfo.valueN2Stride != 0) {
                s2BaseStrideV = s2BaseSize * constInfo.n2Size * constInfo.valueN2Stride;
            }
            runInfo.valueOffset = runParam.valueCoreOffset + sInnerLoopIdx * s2BaseStrideV;
            if constexpr (isFd) {
                uint64_t fdStride = (constInfo.valueN2Stride != 0)
                    ? constInfo.sInnerLoopSize * constInfo.n2Size * constInfo.valueN2Stride
                    : constInfo.sInnerLoopSize * constInfo.n2D;
                runInfo.valueOffset += runInfo.flashDecodeS2Idx * fdStride;
            }
            uint64_t s2BaseStrideK = constInfo.s2BaseN2D;
            if (constInfo.keyN2Stride != 0) {
                s2BaseStrideK = s2BaseSize * constInfo.n2Size * constInfo.keyN2Stride;
            }
            runInfo.keyOffset = runParam.keyCoreOffset + sInnerLoopIdx * s2BaseStrideK;
            if constexpr (hasRope) {
                runInfo.kRopeOffset = runParam.kRopeNBGOffset + sInnerLoopIdx * constInfo.s2BaseN2DR;
            }
        } else {
            runInfo.valueOffset = runParam.valueCoreOffset + sInnerLoopIdx * constInfo.s2BaseDv;
            if constexpr (isFd) {
                runInfo.valueOffset += runInfo.flashDecodeS2Idx * constInfo.sInnerLoopSize * constInfo.dSize;
            }
            runInfo.keyOffset = runParam.keyCoreOffset + sInnerLoopIdx * constInfo.s2BaseD;
            if constexpr (hasRope) {
                runInfo.kRopeOffset = runParam.kRopeNBGOffset + sInnerLoopIdx * constInfo.s2BaseDR;
            }
        }
    }
```

> BNSD 的 s2BaseDv/s2BaseD 不受 n2Stride 影响（bsStride=D 固定），保持原值。keyOffset 改为用 `keyCoreOffset` 独立计算。

---

## 9. 文件7: `flash_attention_score_antiquant_block_vec.h`

**路径**: `attention/fused_infer_attention_score/op_kernel/arch35/flash_attention_score_antiquant_block_vec.h`

### 修改点1: AntiquantKey (line 594)

**现有代码**:
```cpp
    SetAntiqParamCommon(runInfo, runInfo.keyOffset, constInfo);
```

**改为**:
```cpp
    if (constInfo.keyN2Stride != 0 && layout != LayOutTypeEnum::LAYOUT_BNSD) {
        taskParam.kvStep = constInfo.n2Size * constInfo.keyN2Stride;
    }
    SetAntiqParamCommon(runInfo, runInfo.keyOffset, constInfo, true);
```

### 修改点2: AntiquantValue (line 944)

**现有代码**:
```cpp
    SetAntiqParamCommon(runInfo, runInfo.valueOffset, constInfo);
```

**改为**:
```cpp
    if (constInfo.valueN2Stride != 0 && layout != LayOutTypeEnum::LAYOUT_BNSD) {
        taskParam.kvStep = constInfo.n2Size * constInfo.valueN2Stride;
    }
    SetAntiqParamCommon(runInfo, runInfo.valueOffset, constInfo, false);
```

### 修改点3: SetAntiqParamCommon 签名和函数体 (line 616-657)

**现有代码**:
```cpp
ANTIQUANT_TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void
FABlockVecAntiquant<ANTIQUANT_TEMPLATE_ARGS>::SetAntiqParamCommon(const RunInfo<isInfer> &runInfo, int64_t kvOffset,
                                                                  ConstInfo<isInfer, hasRope> &constInfo)
{
    if (isBeforeHalf) {
        taskParam.copyTotalS = GetRealDealSize(runInfo.s2RealSize); // 2 is Vec num
    } else {
        taskParam.copyTotalS = runInfo.s2RealSize - (GetRealDealSize(runInfo.s2RealSize)); // 2 is Vec num
    }
    uint32_t curSequence = constInfo.s2BaseSize * runInfo.s2LoopCount + runInfo.kvLeftPaddingSize +
                           constInfo.subBlockIdx * GetRealDealSize(runInfo.s2RealSize);
    if constexpr (isInfer) {
        curSequence += runInfo.s2StartIdx;
    }
    taskParam.flashDecodeS2Idx = runInfo.flashDecodeS2Idx;
    if constexpr (isFd) {
        curSequence += taskParam.flashDecodeS2Idx * taskParam.sInnerLoopSize;
    }
    taskParam.kvGmOffset = kvOffset + constInfo.subBlockIdx * GetRealDealSize(runInfo.s2RealSize) * taskParam.kvStep;
    taskParam.s2BatchOffset = curSequence;
    taskParam.kvPaddingBeginOffset = runInfo.kvLeftPaddingSize;
    taskParam.antiqParamOffset = runInfo.n2oIdx * taskParam.headDim;
    taskParam.bIdx = runInfo.boIdx;
    taskParam.n2Idx = runInfo.n2oIdx;
    taskParam.s2Idx = runInfo.s2LoopCount;
    if constexpr (isInfer) {
        taskParam.s2Idx += runInfo.s2StartIdx / constInfo.s2BaseSize;
        if constexpr (enableKVPrefix) {
            taskParam.isPrefixLoop = ((runInfo.s2LoopCount + runInfo.s2StartIdx / constInfo.s2BaseSize) <
                                      constInfo.prefixLoopCount); // 判断当前s2LoopCount是否处于prefixLoopCount
            if (taskParam.isPrefixLoop) {
                taskParam.kvGmOffset = runInfo.prefixOffset +
                                       constInfo.subBlockIdx * GetRealDealSize(runInfo.s2RealSize) * taskParam.kvStep;
            } else {
                taskParam.kvGmOffset =
                    kvOffset + constInfo.subBlockIdx * GetRealDealSize(runInfo.s2RealSize) * taskParam.kvStep;
                taskParam.s2Idx -= constInfo.prefixLoopCount;
            }
        }
    }
}
```

**改为** (新增 `bool isKey` 参数 + stride 传递):
```cpp
ANTIQUANT_TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void
FABlockVecAntiquant<ANTIQUANT_TEMPLATE_ARGS>::SetAntiqParamCommon(const RunInfo<isInfer> &runInfo, int64_t kvOffset,
                                                                  ConstInfo<isInfer, hasRope> &constInfo, bool isKey)
{
    if (isBeforeHalf) {
        taskParam.copyTotalS = GetRealDealSize(runInfo.s2RealSize); // 2 is Vec num
    } else {
        taskParam.copyTotalS = runInfo.s2RealSize - (GetRealDealSize(runInfo.s2RealSize)); // 2 is Vec num
    }
    uint32_t curSequence = constInfo.s2BaseSize * runInfo.s2LoopCount + runInfo.kvLeftPaddingSize +
                           constInfo.subBlockIdx * GetRealDealSize(runInfo.s2RealSize);
    if constexpr (isInfer) {
        curSequence += runInfo.s2StartIdx;
    }
    taskParam.flashDecodeS2Idx = runInfo.flashDecodeS2Idx;
    if constexpr (isFd) {
        curSequence += taskParam.flashDecodeS2Idx * taskParam.sInnerLoopSize;
    }
    taskParam.kvGmOffset = kvOffset + constInfo.subBlockIdx * GetRealDealSize(runInfo.s2RealSize) * taskParam.kvStep;
    taskParam.s2BatchOffset = curSequence;
    taskParam.kvPaddingBeginOffset = runInfo.kvLeftPaddingSize;
    taskParam.antiqParamOffset = runInfo.n2oIdx * taskParam.headDim;
    taskParam.bIdx = runInfo.boIdx;
    taskParam.n2Idx = runInfo.n2oIdx;
    taskParam.s2Idx = runInfo.s2LoopCount;
    if (isKey) {
        taskParam.kvBnStride = constInfo.keyBnStride;
        taskParam.kvN2Stride = constInfo.keyN2Stride;
        taskParam.scaleBnStride = constInfo.kScaleBnStride;
        taskParam.scaleN2Stride = constInfo.kScaleN2Stride;
    } else {
        taskParam.kvBnStride = constInfo.valueBnStride;
        taskParam.kvN2Stride = constInfo.valueN2Stride;
        taskParam.scaleBnStride = constInfo.vScaleBnStride;
        taskParam.scaleN2Stride = constInfo.vScaleN2Stride;
    }
    if constexpr (isInfer) {
        taskParam.s2Idx += runInfo.s2StartIdx / constInfo.s2BaseSize;
        if constexpr (enableKVPrefix) {
            taskParam.isPrefixLoop = ((runInfo.s2LoopCount + runInfo.s2StartIdx / constInfo.s2BaseSize) <
                                      constInfo.prefixLoopCount); // 判断当前s2LoopCount是否处于prefixLoopCount
            if (taskParam.isPrefixLoop) {
                taskParam.kvGmOffset = runInfo.prefixOffset +
                                       constInfo.subBlockIdx * GetRealDealSize(runInfo.s2RealSize) * taskParam.kvStep;
            } else {
                taskParam.kvGmOffset =
                    kvOffset + constInfo.subBlockIdx * GetRealDealSize(runInfo.s2RealSize) * taskParam.kvStep;
                taskParam.s2Idx -= constInfo.prefixLoopCount;
            }
        }
    }
}
```

> 需同步修改 `flash_attention_score_antiquant_block_vec.h` 中 `SetAntiqParamCommon` 的声明。

---

## 10. 文件8: `flash_attention_score_antiquant_processor.h`

**路径**: `attention/fused_infer_attention_score/op_kernel/arch35/flash_attention_score_antiquant_processor.h`

### 修改点1: AntiquantTaskParamBaseAPI 新增字段 (line 86, `bool isPrefixLoop = 0;` 后)

**现有代码** (line 84-87):
```cpp
    uint64_t kvPaddingBeginOffset;

    bool isPrefixLoop = 0;
};
```

**改为**:
```cpp
    uint64_t kvPaddingBeginOffset;

    bool isPrefixLoop = 0;
    uint64_t kvBnStride = 0;
    uint64_t kvN2Stride = 0;
    uint64_t scaleBnStride = 0;
    uint64_t scaleN2Stride = 0;
};
```

### 修改点2: CopyKV srcStride (line 564-576)

**现有代码**:
```cpp
    if constexpr (PAGE_ATTENTION) {
        if (paKvShapeType == 0) {
            copyInParams.srcStride = copyInParams.blockLen * (kvHeadNum - 1);
        } else {
            copyInParams.srcStride = 0;
        }
    } else {
        if constexpr (layout == LayOutTypeEnum::LAYOUT_BNSD) {
            copyInParams.srcStride = 0;
        } else {
            copyInParams.srcStride = copyInParams.blockLen * (kvHeadNum - 1);
        }
    }
```

> `CopyKV` 的签名目前没有 taskParam 参数，无法获取 stride。需修改签名传入 `taskParam` 或单独传入 `kvN2Stride`。

**改为** (需先修改 `CopyKV` 签名增加 `uint64_t kvN2Stride` 参数):
```cpp
    if constexpr (PAGE_ATTENTION) {
        if (paKvShapeType == 0) {
            if (kvN2Stride != 0) {
                copyInParams.srcStride = static_cast<int64_t>(kvHeadNum - 1) * kvN2Stride * sizeof(KV_T);
            } else {
                copyInParams.srcStride = copyInParams.blockLen * (kvHeadNum - 1);
            }
        } else {
            copyInParams.srcStride = 0;
        }
    } else {
        if constexpr (layout == LayOutTypeEnum::LAYOUT_BNSD) {
            copyInParams.srcStride = 0;
        } else {
            if (kvN2Stride != 0) {
                copyInParams.srcStride = static_cast<int64_t>(kvHeadNum - 1) * kvN2Stride * sizeof(KV_T);
            } else {
                copyInParams.srcStride = copyInParams.blockLen * (kvHeadNum - 1);
            }
        }
    }
```

> 所有调用 `CopyKV` 的地方需传入 `taskParam.kvN2Stride`：
> - `AntiquantBaseAPI` line 489: `CopyKV(tmpUb, kvGm, kvOffset, dealRowCount, taskParam.headDim, taskParam.kvHeadNum, taskParam.paKvShapeType, taskParam.kvN2Stride);`
> - `CopyKVPageAttention` line 642-643: 同理

### 修改点3: CopyKVPageAttention offset (line 625-644)

**现有代码**:
```cpp
        uint64_t curOffset = 0;
        if (taskParam.paKvShapeType == static_cast<uint32_t>(KvCacheLayout::KV_CACHE_NZ)) { // NZ
            curOffset = idInBlockTable * taskParam.kvHeadNum * taskParam.headDim * taskParam.kvCacheBlockSize +
                        (uint64_t)(taskParam.n2Idx * taskParam.headDim * taskParam.kvCacheBlockSize) +
                        reaminRowCnt * typeElementSize;
            CopyKVPaNz(dstLocal[copyFinishRowCnt * typeElementSize], srcGm, curOffset, copyRowCnt, dealRowCount,
                       taskParam);
        } else {
            if (taskParam.paKvShapeType == static_cast<uint32_t>(KvCacheLayout::KV_CACHE_BSH)) { // BBH
                curOffset = (idInBlockTable * taskParam.kvCacheBlockSize + reaminRowCnt) * taskParam.kvHeadNum *
                                taskParam.headDim +
                            (uint64_t)(taskParam.n2Idx * taskParam.headDim);
            } else { // BNBD
                curOffset = idInBlockTable * taskParam.kvHeadNum * taskParam.kvCacheBlockSize * taskParam.headDim +
                            (uint64_t)(taskParam.n2Idx * taskParam.kvCacheBlockSize * taskParam.headDim) +
                            reaminRowCnt * taskParam.headDim;
            }
            CopyKV(dstLocal[copyFinishRowCnt * dBaseSize], srcGm, curOffset, copyRowCnt, taskParam.headDim,
                   taskParam.kvHeadNum, taskParam.paKvShapeType);
        }
```

**改为**:
```cpp
        uint64_t curOffset = 0;
        if (taskParam.paKvShapeType == static_cast<uint32_t>(KvCacheLayout::KV_CACHE_NZ)) { // NZ
            uint64_t nzHeadStride = taskParam.headDim * taskParam.kvCacheBlockSize;
            uint64_t nzBlockStride = taskParam.kvHeadNum * nzHeadStride;
            if (taskParam.kvN2Stride != 0) {
                nzHeadStride = taskParam.kvN2Stride;
                nzBlockStride = (taskParam.kvBnStride != 0) ? taskParam.kvBnStride : taskParam.kvHeadNum * nzHeadStride;
            }
            curOffset = idInBlockTable * nzBlockStride +
                        (uint64_t)(taskParam.n2Idx * nzHeadStride) +
                        reaminRowCnt * typeElementSize;
            CopyKVPaNz(dstLocal[copyFinishRowCnt * typeElementSize], srcGm, curOffset, copyRowCnt, dealRowCount,
                       taskParam);
        } else {
            if (taskParam.paKvShapeType == static_cast<uint32_t>(KvCacheLayout::KV_CACHE_BSH)) { // BBH
                uint64_t bsStride = taskParam.kvHeadNum * taskParam.headDim;
                uint64_t bnStride = taskParam.kvCacheBlockSize * bsStride;
                if (taskParam.kvN2Stride != 0) {
                    bsStride = taskParam.kvHeadNum * taskParam.kvN2Stride;
                    bnStride = (taskParam.kvBnStride != 0) ? taskParam.kvBnStride : taskParam.kvCacheBlockSize * bsStride;
                }
                curOffset = idInBlockTable * bnStride +
                            reaminRowCnt * bsStride +
                            (uint64_t)(taskParam.n2Idx * taskParam.headDim);
            } else { // BNBD
                uint64_t headStride = taskParam.kvCacheBlockSize * taskParam.headDim;
                uint64_t blockStride = taskParam.kvHeadNum * headStride;
                if (taskParam.kvN2Stride != 0) {
                    headStride = taskParam.kvN2Stride;
                    blockStride = (taskParam.kvBnStride != 0) ? taskParam.kvBnStride : taskParam.kvHeadNum * headStride;
                }
                curOffset = idInBlockTable * blockStride +
                            (uint64_t)(taskParam.n2Idx * headStride) +
                            reaminRowCnt * taskParam.headDim;
            }
            CopyKV(dstLocal[copyFinishRowCnt * dBaseSize], srcGm, curOffset, copyRowCnt, taskParam.headDim,
                   taskParam.kvHeadNum, taskParam.paKvShapeType, taskParam.kvN2Stride);
        }
```

### 修改点4: CopyAntiqParam (line 440)

**现有代码**:
```cpp
        DataCopyPad(dstLocal, srcGm[taskParam.antiqParamOffset], copyInParams, copyInPadParams);
```

**改为**:
```cpp
        uint64_t antiqOffset = taskParam.antiqParamOffset;
        if (taskParam.scaleN2Stride != 0) {
            antiqOffset = taskParam.n2Idx * taskParam.scaleN2Stride;
        }
        DataCopyPad(dstLocal, srcGm[antiqOffset], copyInParams, copyInPadParams);
```

### 修改点5: CopyAntiquantParamsPageAttention (line 533-534)

**现有代码**:
```cpp
        uint64_t srcOffset = idInBlockTable * taskParam.kvCacheBlockSize * useKvHeadNum +
                             taskParam.kvCacheBlockSize * useN2Idx + remainElmenCnt;
```

**改为**:
```cpp
        uint64_t blockStride = taskParam.kvCacheBlockSize * useKvHeadNum;
        uint64_t headStride = taskParam.kvCacheBlockSize;
        if (taskParam.scaleBnStride != 0) {
            blockStride = taskParam.scaleBnStride;
        }
        if (taskParam.scaleN2Stride != 0) {
            headStride = taskParam.scaleN2Stride;
        }
        uint64_t srcOffset = idInBlockTable * blockStride +
                             useN2Idx * headStride + remainElmenCnt;
```

### 修改点6: LoadAntiquantParamsPerToken scaleOffset (line 339-374)

**现有代码** (per-head 分支, line 340-356):
```cpp
    if (taskParam.isPerHead) {
        if constexpr (enableKVPrefix) {
            if (taskParam.isPrefixLoop) {
                scaleOffset = taskParam.bIdx * taskParam.kvHeadNum * taskParam.antiqSeqSize +
                              taskParam.n2Idx * taskParam.antiqSeqSize + taskParam.s2Idx * taskParam.singleSInnerSize +
                              taskParam.kvPaddingBeginOffset + subBlockIdx * GetRealDealSize(s2RealSize);
            } else {
                scaleOffset = taskParam.bIdx * taskParam.kvHeadNum * taskParam.antiqSeqSize +
                              taskParam.n2Idx * taskParam.antiqSeqSize + taskParam.s2Idx * taskParam.singleSInnerSize +
                              (taskParam.antiqSeqSize - taskParam.seqSize) + taskParam.kvPaddingBeginOffset +
                              subBlockIdx * GetRealDealSize(s2RealSize);
            }
        } else {
            scaleOffset = taskParam.bIdx * taskParam.kvHeadNum * taskParam.seqSize +
                          taskParam.n2Idx * taskParam.seqSize + taskParam.s2Idx * taskParam.singleSInnerSize +
                          taskParam.kvPaddingBeginOffset + subBlockIdx * GetRealDealSize(s2RealSize);
        }
    } else {
```

**改为**:
```cpp
    if (taskParam.isPerHead) {
        uint64_t batchStride = taskParam.kvHeadNum * taskParam.antiqSeqSize;
        uint64_t headStride = taskParam.antiqSeqSize;
        uint64_t seqStride = taskParam.singleSInnerSize;
        if (taskParam.scaleBnStride != 0) {
            batchStride = taskParam.scaleBnStride;
        }
        if (taskParam.scaleN2Stride != 0) {
            headStride = taskParam.scaleN2Stride;
        }
        if constexpr (enableKVPrefix) {
            if (taskParam.isPrefixLoop) {
                scaleOffset = taskParam.bIdx * batchStride +
                              taskParam.n2Idx * headStride + taskParam.s2Idx * seqStride +
                              taskParam.kvPaddingBeginOffset + subBlockIdx * GetRealDealSize(s2RealSize);
            } else {
                scaleOffset = taskParam.bIdx * batchStride +
                              taskParam.n2Idx * headStride + taskParam.s2Idx * seqStride +
                              (taskParam.antiqSeqSize - taskParam.seqSize) + taskParam.kvPaddingBeginOffset +
                              subBlockIdx * GetRealDealSize(s2RealSize);
            }
        } else {
            uint64_t batchStrideNP = taskParam.kvHeadNum * taskParam.seqSize;
            if (taskParam.scaleBnStride != 0) {
                batchStrideNP = taskParam.scaleBnStride;
            }
            uint64_t headStrideNP = taskParam.seqSize;
            if (taskParam.scaleN2Stride != 0) {
                headStrideNP = taskParam.scaleN2Stride;
            }
            scaleOffset = taskParam.bIdx * batchStrideNP +
                          taskParam.n2Idx * headStrideNP + taskParam.s2Idx * seqStride +
                          taskParam.kvPaddingBeginOffset + subBlockIdx * GetRealDealSize(s2RealSize);
        }
    } else {
```

> 非 per-head 分支 (line 357-370) 同理：`bIdx * antiqSeqSize` → 用 `scaleBnStride` 替代。
> `LoadAntiquantParamsPerTokenGroup` (line 257+) 同理修改 `bIdx * kvHeadNum * antiqSeqSize * grpNum` 和 `n2Idx * antiqSeqSize * grpNum`。

---

## 11. 实施顺序

```
Phase 1: Tiling 结构 (文件1,2) — InputParamsRegbase + COMMON_CONST_INFO + CVSharedParams 新增 kScale/vScale
Phase 2: Host (文件3,4) — SetFATilingData 写 stride + CheckStrideForAntiquant
Phase 3: Kernel 接线 (文件5) — ComputeConstexpr 读取 stride
Phase 4: Offset 计算 (文件6,7) — CalcValueCoreOffset* + SyncKeyCoreOffset + ComputeOffsetForAntiquant + SetAntiqParamCommon
Phase 5: 数据搬运 (文件8) — AntiquantTaskParamBaseAPI + CopyKV + CopyKVPageAttention + CopyAntiqParam + CopyAntiquantParamsPageAttention + LoadAntiquantParams*
Phase 6: 测试 — 冒烟(连续不退化) → PA_BNBD/BBND/NZ 0轴/1轴非连续 → 非 PA 报错
```

---

## 12. 核心风险

1. **连续不退化**：stride=0 走原逻辑，constInfo 字段默认 0
2. **BBND bsStride 依赖 n2Stride**：行步长 = N2 × n2Stride，CopyKVPageAttention 和 kvStep 需体现
3. **srcStride 单位**：arch35 是字节，需 `* sizeof(KV_T)` 转换
4. **CopyKV 签名变更**：需新增 `kvN2Stride` 参数，所有调用点同步修改
5. **SetAntiqParamCommon 签名变更**：需新增 `bool isKey` 参数区分 key/value stride

---

## 13. 文件清单

| # | 文件 | 修改点数 | Phase |
|---|------|---------|-------|
| 1 | `common/op_kernel/arch35/flash_attention_score_tiling_regbase.h` | 2 | 1 |
| 2 | `common/op_kernel/arch35/util_regbase.h` | 2 | 1 |
| 3 | `op_host/arch35/fused_infer_attention_score_tiling_impl.cpp` | 1 | 2 |
| 4 | `op_host/checkers/dequant_checker.cpp` + `.h` | 3 | 2 |
| 5 | `op_kernel/arch35/flash_attention_score_antiquant_kernel.h` | 1 | 3 |
| 6 | `op_kernel/arch35/infer_flash_attention_kvcache.h` | 5 | 4 |
| 7 | `op_kernel/arch35/flash_attention_score_antiquant_block_vec.h` | 3 | 4-5 |
| 8 | `op_kernel/arch35/flash_attention_score_antiquant_processor.h` | 6 | 5 |
