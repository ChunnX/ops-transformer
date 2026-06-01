# FlashAttentionScore BAND模式完整链路分析（ARCH32架构）

## 概述

本文档详细分析了 `flash_attention_score` 算子中BAND（滑窗）模式的完整执行链路，专注于 **ARCH32 架构** 的实现，包括模式入口、触发条件、参数传递和内核实现等关键环节。

---

## 一、BAND模式定义与类型

### 1.1 稀疏模式枚举定义（ARCH32）

**文件位置**: `op_host/arch32/flash_attention_score_tiling_general.cpp:156-167`

```cpp
enum SparseMode : uint8_t {
    NO_MASK = 0,           // 无掩码
    ALL_MASK,              // 全掩码
    LEFT_UP_CAUSAL,        // 左上因果（decoder）
    RIGHT_DOWN_CAUSAL,     // 右下因果（encoder）
    BAND,                  // 标准BAND模式（滑窗）
    PREFIX,                // Prefix 模式
    PREFIX_COMPRESS,       // Prefix 压缩模式
    RIGHT_DOWN_CAUSAL_BAND, // 右下因果+滑窗模式
    BAND_LEFT_UP_CAUSAL    // 滑窗+左上因果模式
};
```

### 1.2 注意力掩码压缩模式

**文件位置**: `op_host/arch32/flash_attention_score_tiling_general.cpp:168-176`

```cpp
enum AttenMaskCompressMode : uint8_t {
    NO_COMPRESS_MODE = 0,           // 无压缩
    LEFT_UP_CAUSAL_MODE,            // 左上因果压缩
    RIGHT_DOWN_CAUSAL_MODE,         // 右下因果压缩
    BAND_MODE,                      // BAND 模式压缩
    PREFIX_MODE,                    // Prefix 模式压缩
    RIGHT_DOWN_CAUSAL_BAND_MODE = 5,// 右下因果+滑窗压缩
    BAND_LEFT_UP_CAUSAL_MODE        // 滑窗+左上因果压缩
};
```

### 1.3 内核侧稀疏模式枚举

**文件位置**: `op_kernel/arch32/flash_attention_score_common.h:75-86`

```cpp
enum class SparseModeEnum {
    ALL = 0,
    NONE = 1,
    ANY = 2,
    CAUSAL = 3,
    BAND = 4,                    // 标准BAND模式
    PREFIX = 5,
    BAND_COMPRESS = 6,           // BAND压缩模式
    RIGHT_DOWN_CAUSAL = 7,
    RIGHT_DOWN_CAUSAL_BAND = 8,  // 右下因果+滑窗模式
    BAND_LEFT_UP_CAUSAL = 9      // 左上因果+滑窗模式
};
```

### 1.4 实现模式枚举

**文件位置**: `op_host/arch32/flash_attention_score_tiling_general.cpp:178-181`

```cpp
enum ImplMode : uint8_t {
    AA_HIGH_PRECISION = 0,                // 高精度模式
    AA_HIGH_PERFORMANCE = 1,              // 高性能模式
    AA_INVALID_LINE_HIGH_PRECISION = 2    // 无效行高精度模式
};
```

---

## 二、BAND模式入口与触发条件

### 2.1 主入口点

**文件位置**: `op_host/flash_attention_score_tiling.cpp:345-372`

```cpp
ASCENDC_EXTERN_C ge::graphStatus TilingFlashAttentionScore(gert::TilingContext *context)
{
    OP_LOGW(context, "Start registering tiling.");
    if (CheckParams(context) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    auto platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfoPtr == nullptr,
        OP_LOGE(context, "platformInfoPtr is null"),
        return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);

    // 检查架构类型
    if (ascendcPlatform.GetCurNpuArch() == NpuArch::DAV_3510) {
        OP_LOGW(context, "Current npu arch is dav-3510.");
        if (IsEmptyInputRegbase(context)) {
            return ge::GRAPH_SUCCESS;
        }
    } else {
        OP_LOGW(context, "Current npu arch is not dav-3510.");
        // ARCH32 架构走这里
        if (IsEmptyInput(context)) {
            return ge::GRAPH_SUCCESS;
        }
    }

    // 进入 Tiling 注册流程
    auto resultCode = TilingRegistryArch::GetInstance().DoTilingImpl(context);
    return resultCode;
}
```

### 2.2 Tiling处理入口（ARCH32）

**文件位置**: `op_host/arch32/flash_attention_score_tiling_general.cpp`

ARCH32 架构的 Tiling 处理主要通过 `FlashAttentionScoreTilingBase` 类实现。

---

## 三、BAND模式识别与参数解析

### 3.1 属性解析

**文件位置**: `op_host/arch32/flash_attention_score_tiling_general.cpp:1037-1096`

```cpp
bool FlashAttentionScoreTilingBase::ParseAttrParams()
{
    auto attrs = context_->GetAttrs();
    size_t idx = 0;

    // 获取基本属性
    auto scaleValuePtr = attrs->GetAttrPointer<float>(idx++);
    auto keepProbPtr = attrs->GetAttrPointer<float>(idx++);
    auto preTokensPtr = attrs->GetAttrPointer<int64_t>(idx++);      // preTokens参数
    auto nextTokensPtr = attrs->GetAttrPointer<int64_t>(idx++);     // nextTokens参数
    auto n1SizePtr = attrs->GetAttrPointer<uint32_t>(idx++);
    inputLayout_ = attrs->GetAttrPointer<char>(idx++);

    preTokens_ = *preTokensPtr;
    nextTokens_ = *nextTokensPtr;
    keepProb_ = *keepProbPtr;
    scaleValue_ = *scaleValuePtr;
    n1Size_ = *n1SizePtr;

    // 参数边界检查
    OP_CHECK_IF(n1Size_ == 0, OPS_REPORT_VECTOR_INNER_ERR(opName_, "head num is zero."), return false);
    OP_CHECK_IF(keepProb_ <= 0.0 || keepProb_ > 1.0,
               OPS_REPORT_VECTOR_INNER_ERR(opName_, "keepProb value must be in range of (0, 1]."), return false);

    hasDropOut_ = (keepProb_ < 1.0f);

    // 获取 innerPrecise 属性
    if (attrs->GetAttrNum() > idx) {
        auto implModePtr = attrs->GetAttrPointer<uint8_t>(idx++);
        if (static_cast<ImplMode>(*implModePtr) == ImplMode::AA_INVALID_LINE_HIGH_PRECISION) {
            implMode_ = ImplMode::AA_INVALID_LINE_HIGH_PRECISION;
        }
        isHighPercision_ = true;
    }

    // 获取 sparse_mode 属性（关键）
    if (attrs->GetAttrNum() > idx) {
        auto sparseModePtr = attrs->GetAttrPointer<int64_t>(idx++);
        sparseMode_ = *sparseModePtr;

        // 根据稀疏模式设置掩码压缩模式
        if (sparseMode_ == static_cast<int64_t>(LEFT_UP_CAUSAL)) {
            attenMaskCompressMode_ = LEFT_UP_CAUSAL_MODE;
        } else if (sparseMode_ == static_cast<int64_t>(RIGHT_DOWN_CAUSAL)) {
            attenMaskCompressMode_ = RIGHT_DOWN_CAUSAL_MODE;
        } else if (sparseMode_ == static_cast<int64_t>(BAND)) {
            attenMaskCompressMode_ = BAND_MODE;  // BAND 模式
        } else if (sparseMode_ == static_cast<int64_t>(RIGHT_DOWN_CAUSAL_BAND)) {
            attenMaskCompressMode_ = RIGHT_DOWN_CAUSAL_BAND_MODE;
        } else if (sparseMode_ == static_cast<int64_t>(BAND_LEFT_UP_CAUSAL)) {
            attenMaskCompressMode_ = BAND_LEFT_UP_CAUSAL_MODE;
        } else if (sparseMode_ == static_cast<int64_t>(PREFIX_COMPRESS)) {
            attenMaskCompressMode_ = PREFIX_MODE;
        }
        OP_LOGD(context_, "The current value of attenMaskCompressMode is %u.", attenMaskCompressMode_);
    }

    // ...其他属性解析
    return true;
}
```

### 3.2 稀疏信息获取与模式判断

**文件位置**: `op_host/arch32/flash_attention_score_tiling_general.cpp:1338-1382`

```cpp
bool FlashAttentionScoreTilingBase::GetSparseInfo(SparseEnum &sparseType)
{
    OP_LOGD(context_, "check sparse info: preTokens[%ld], nextTokens[%ld], s1[%ld], s2[%ld], hasAttenMask[%d].",
              preTokens_, nextTokens_, s1Size_, s2Size_, hasAttenMask_);

    if (sparseMode_ > static_cast<int64_t>(PREFIX_COMPRESS)) {
        OP_LOGE(context_, "Not support sparse mode of %ld.", sparseMode_);
        return false;
    }

    if (!hasAttenMask_) {
        return true;
    }

    if (layoutType_ == LayoutType::TND) {
        return true;
    }

    if (sparseMode_ == static_cast<int64_t>(NO_MASK)) {
        if (preTokens_ >= s1Size_ && nextTokens_ == 0) {
            sparseType = SparseEnum::CAUSAL;
            preTokens_ = s1Size_;
        } else {
            if (preTokens_ >= s1Size_ && nextTokens_ >= s2Size_) {
                return true;
            }
            // 进入BAND模式检查
            if (!SparseBandModeCheck(s1Size_, s2Size_, s1Size_, s2Size_, sparseType)) {
                return false;
            }
        }
    } else {
        if (!SparseModeProcess(sparseType)) {
            return false;
        }
    }
    return true;
}
```

### 3.3 BAND模式检查逻辑

**文件位置**: `op_host/arch32/flash_attention_score_tiling_general.cpp:730-795`

```cpp
bool FlashAttentionScoreTilingBase::SparseBandModeCheck(int64_t maxS1Value, int64_t maxS2Value,
                                                          int64_t minS1Value, int64_t minS2Value,
                                                          SparseEnum &sparseType)
{
    // 对于BAND模式，需要转换preTokens和nextTokens的参考点
    int64_t oriPreTokens = (sparseMode_ == static_cast<int64_t>(BAND)) ?
                           (preTokens_ + s2Size_ - s1Size_) : preTokens_;
    int64_t oriNextTokens = (sparseMode_ == static_cast<int64_t>(BAND)) ?
                            (nextTokens_ + s1Size_ - s2Size_) : nextTokens_;

    if (preTokens_ >= 0 && nextTokens_ >= 0) {
        if (preTokens_ >= maxS1Value && nextTokens_ >= maxS2Value) {
            OP_LOGW(context_,
                      "PreTokens[%ld] and nextTokens[%ld] config error, should not both greater than maxS1Val[%ld] "
                      "maxS2Val[%ld].",
                      oriPreTokens, oriNextTokens, maxS1Value, maxS2Value);
            return true;
        }
        // 设置稀疏有效大小
        s1SparseValidSize_ = std::min(AlignUp(preTokens_, HIGH_PERF_BLOCK_SIZE), s1Size_);
        s2SparseValidSize_ = std::min(AlignUp(nextTokens_, HIGH_PERF_BLOCK_SIZE), s2Size_);
        isSparseValidSizeAligned_ = true;
        sparseType = SparseEnum::BAND;
        // 启用BAND无效行高精度实现模式
        EnableBandInvalidLineImplMode();
        return true;
    }

    if (preTokens_ < 0 && nextTokens_ < 0) {
        OP_LOGE(context_, "PreTokens[%ld] and nextTokens[%ld] config error, there is no valid data block.",
                  oriPreTokens, oriNextTokens);
        return false;
    }

    if (preTokens_ < 0 && nextTokens_ >= 0) {
        int64_t absPreTokens = std::abs(preTokens_);
        if (nextTokens_ >= absPreTokens && absPreTokens < minS2Value) {
            EnableBandInvalidLineImplMode();
            s1SparseValidSize_ = std::min(AlignUp(preTokens_, HIGH_PERF_BLOCK_SIZE), 0L);
            s2SparseValidSize_ = std::min(AlignUp(nextTokens_, HIGH_PERF_BLOCK_SIZE), s2Size_);
            isSparseValidSizeAligned_ = true;
            sparseType = SparseEnum::BAND;
            return true;
        } else {
            OP_LOGE(context_,
                      "PreTokens[%ld] and nextTokens[%ld] config error with S1[%ld], there is no valid data block.",
                      oriPreTokens, oriNextTokens, minS1Value);
            return false;
        }
    }

    if (preTokens_ >= 0 && nextTokens_ < 0) {
        int64_t absNextTokens = std::abs(nextTokens_);
        if (absNextTokens <= preTokens_ && absNextTokens < minS1Value) {
            EnableBandInvalidLineImplMode();
            s1SparseValidSize_ = std::min(AlignUp(preTokens_, HIGH_PERF_BLOCK_SIZE), s1Size_);
            s2SparseValidSize_ = std::min(AlignUp(nextTokens_, HIGH_PERF_BLOCK_SIZE), 0L);
            isSparseValidSizeAligned_ = true;
            sparseType = SparseEnum::BAND;
            return true;
        } else {
            OP_LOGE(context_,
                      "PreTokens[%ld] and nextTokens[%ld] config error with S2[%ld], there is no valid data block.",
                      oriPreTokens, oriNextTokens, minS2Value);
            return false;
        }
    }
    return true;
}
```

### 3.4 BAND无效行实现模式启用

**文件位置**: `op_host/arch32/flash_attention_score_tiling_general.cpp:697-706`

```cpp
void FlashAttentionScoreTilingBase::EnableBandInvalidLineImplMode()
{
    if (implMode_ == ImplMode::AA_INVALID_LINE_HIGH_PRECISION) {
        return;
    }
    // pretoken and nexttoken are already valid values (leftup vertex) after adjusted
    if (preTokens_ < (s1Size_ - s2Size_) || nextTokens_ < 0) {
        implMode_ = ImplMode::AA_INVALID_LINE_HIGH_PRECISION;
        OP_LOGI(context_, "Enable invalid line impl mode.");
        return;
    }
}
```

### 3.5 PreToken和NextToken调整

**文件位置**: `op_host/arch32/flash_attention_score_tiling_general.cpp:680-728`

```cpp
bool FlashAttentionScoreTilingBase::PretokenAndNexttokenAdjustment(SparseEnum &sparseType)
{
    // ...其他模式处理

    } else if (sparseMode_ == static_cast<int64_t>(RIGHT_DOWN_CAUSAL)) {
        if (s1Size_ == s2Size_) {
            if (preTokens_ != s1Size_ || nextTokens_ != 0) {
                OP_LOGW(context_,
                          "preTokens[%ld] and nextTokens[%ld] not match sparseMode[%ld], "
                          "preTokens will be reset max int value and nextTokens will be reset 0.",
                          preTokens_, nextTokens_, sparseMode_);
                preTokens_ = s1Size_;
                nextTokens_ = 0;
            }
            sparseType = SparseEnum::CAUSAL;
        } else {
            // 不等S序列长度时转换为BAND模式
            preTokens_ = s1Size_;
            nextTokens_ = s2Size_ - s1Size_;
            OP_LOGD(context_,
                      "Unequal s, sparseType rightDownCasual reset to band, and reset preTokens[%ld] "
                      "and nextTokens[%ld].",
                      preTokens_, nextTokens_);
            sparseType = SparseEnum::BAND;
            // 检查需要启用 AA_INVALID_LINE_HIGH_PRECISION
            EnableBandInvalidLineImplMode();
            s1SparseValidSize_ = preTokens_;
            s2SparseValidSize_ = std::min(AlignUp(nextTokens_, HIGH_PERF_BLOCK_SIZE), s2Size_);
            isSparseValidSizeAligned_ = true;
        }
    } else if (sparseMode_ == static_cast<int64_t>(BAND)) {
        // unequal s, pretoken and nexttoken count from rigthDown vertex, need to change to leftUp vertex
        if (s1Size_ != s2Size_) {
            preTokens_ = s1Size_ - s2Size_ + preTokens_;
            nextTokens_ = s2Size_ - s1Size_ + nextTokens_;
        }
        if (!SparseBandModeCheck(s1Size_, s2Size_, s1Size_, s2Size_, sparseType)) {
            return false;
        }
    }
    return true;
}
```

### 3.6 稀疏模式处理

**文件位置**: `op_host/arch32/flash_attention_score_tiling_general.cpp:730-795`

```cpp
bool FlashAttentionScoreTilingBase::SparseModeProcess(SparseEnum &sparseType)
{
    if (!PretokenAndNexttokenAdjustment(sparseType)) {
        return false;
    }

    if (sparseMode_ == static_cast<int64_t>(PREFIX) || sparseMode_ == static_cast<int64_t>(PREFIX_COMPRESS)) {
        std::ostringstream failReason;
        sparseType = GetPrefixNList(failReason);
        if (sparseType != SparseEnum::PREFIX && sparseMode_ == static_cast<int64_t>(PREFIX_COMPRESS)) {
            OP_LOGE(context_, "[%s] %s.", templateName_, failReason.str().c_str());
            return false;
        }

        if (sparseType == SparseEnum::PREFIX && sparseMode_ == static_cast<int64_t>(PREFIX) &&
            attenMaskShapeType_ != static_cast<uint8_t>(AttenMaskShapeType::ATTEN_B_N2_G_S1_S2) &&
            attenMaskShapeType_ != static_cast<uint8_t>(AttenMaskShapeType::ATTEN_B_1_1_S1_S2) &&
            bSize_ != 1) {
            OP_LOGE(context_, "Prefix mode get invalid atten_mask shape, should be [BNSS] or [B1SS].");
            return false;
        }
    }
    return true;
}
```

---

## 四、BAND模式核心参数设置

### 4.1 稀疏Tiling信息设置

**文件位置**: `op_host/arch32/flash_attention_score_tiling_general.cpp:1112-1118`

```cpp
void FlashAttentionScoreTilingBase::SetSparseTilingInfo(SparseEnum &sparseType)
{
    tilingData_.inputParams.sparseType = static_cast<uint8_t>(sparseType);
    tilingData_.inputParams.attenMaskCompressMode = attenMaskCompressMode_;
    tilingData_.inputParams.preTokens = preTokens_;
    tilingData_.inputParams.nextTokens = nextTokens_;
}
```

### 4.2 核心参数设置到TilingData

**文件位置**: `op_host/arch32/flash_attention_score_tiling_general.cpp`

ARCH32 架构将所有 BAND 相关参数设置到 `FlashAttentionScoreTilingData` 结构中：

```cpp
struct InputParams {
    // ...其他参数
    int64_t s1SparseValidSize;     // S1 稀疏有效大小
    int64_t s2SparseValidSize;     // S2 稀疏有效大小
    uint8_t sparseType;            // 稀疏类型
    uint8_t attenMaskCompressMode; // 掩码压缩模式
    int64_t preTokens;             // 向前 Token 数
    int64_t nextTokens;            // 向后 Token 数
    // ...
};
```

---

## 五、内核实现中的BAND模式处理（ARCH32）

### 5.1 BAND模式S2范围计算（标准布局）

**文件位置**: `op_kernel/arch32/flash_attention_score_s1s2_bn2gs1.h:712-724`

```cpp
} else if (this->tilingData->inputParams.sparseType == static_cast<uint8_t>(SparseModeEnum::BAND)) {
    // 对角线往外扩散场景, s1和s2可能不同
    this->s2StartIdx = Max(
        this->s1oIdx * this->tilingData->coreParams.s1BaseSize - this->tilingData->coreParams.s1SparseValidSize, 0);
    if ((this->s1oIdx + 1) * this->tilingData->coreParams.s1BaseSize +
        this->tilingData->coreParams.s2SparseValidSize >= this->s2Size) {
        this->s2EndIdx = actualS2Len;
    } else {
        this->s2EndIdx = Min((this->s1oIdx + 1) * this->tilingData->coreParams.s1BaseSize +
                                this->tilingData->coreParams.s2SparseValidSize,
                            actualS2Len);
    }
}
```

### 5.2 BAND模式S2范围计算（变长布局）

**文件位置**: `op_kernel/arch32/flash_attention_var_len_score.h:408-417`

```cpp
} else if (this->tilingData->inputParams.sparseType == static_cast<uint8_t>(SparseModeEnum::BAND)) {
    this->s2StartIdx = Max(
        this->s1oIdx * this->tilingData->coreParams.s1BaseSize - this->tilingData->coreParams.s1SparseValidSize, 0);
    if ((this->s1oIdx + 1) * this->tilingData->coreParams.s1BaseSize +
        this->tilingData->coreParams.s2SparseValidSize >= this->s2Size) {
        this->s2EndIdx = actualS2Len;
    } else {
        this->s2EndIdx = Min((this->s1oIdx + 1) * this->tilingData->coreParams.s1BaseSize +
                                this->tilingData->coreParams.s2SparseValidSize,
                            actualS2Len);
    }
}
```

### 5.3 BAND压缩模式实现

**文件位置**: `op_kernel/arch32/flash_attention_var_len_score.h:418-429`

```cpp
} else if (this->tilingData->inputParams.sparseType == static_cast<uint8_t>(SparseModeEnum::BAND_COMPRESS)) {
    this->s2StartIdx = Max(this->s1oIdx * this->tilingData->coreParams.s1BaseSize - actualS1Len +
                               Max(actualS2Len - this->tilingData->inputParams.preTokens, 0),
                           0);
    if ((this->s1oIdx + 1) * this->tilingData->coreParams.s1BaseSize +
        this->tilingData->inputParams.nextTokens >= this->s2Size) {
        this->s2EndIdx = actualS2Len;
    } else {
        this->s2EndIdx = (this->s1oIdx + 1) * this->tilingData->coreParams.s1BaseSize +
                         this->tilingData->inputParams.nextTokens;
    }
    this->s2EndIdx = Min(this->s2EndIdx, actualS2Len);
}
```

### 5.4 复合BAND模式实现（变长布局）

**文件位置**: `op_kernel/arch32/flash_attention_var_len_score_sab.h:370-413`

```cpp
} else if (this->tilingData->inputParams.sparseType == static_cast<uint8_t>(SparseModeEnum::BAND)) {
    this->s2StartIdx = Max(
        this->s1oIdx * this->cubeS1BaseSize - this->tilingData->coreParams.s1SparseValidSize, 0);
    if ((this->s1oIdx + 1) * this->cubeS1BaseSize +
        this->tilingData->coreParams.s2SparseValidSize >= this->s2Size) {
        this->s2EndIdx = actualS2Len;
    } else {
        this->s2EndIdx = Min((this->s1oIdx + 1) * this->cubeS1BaseSize +
                                this->tilingData->coreParams.s2SparseValidSize,
                            actualS2Len);
    }
} else if (this->tilingData->inputParams.sparseType == static_cast<uint8_t>(SparseModeEnum::BAND_COMPRESS)) {
    this->s2StartIdx = Max(this->s1oIdx * this->cubeS1BaseSize - actualS1Len +
                               Max(actualS2Len - this->tilingData->inputParams.preTokens, 0),
                           0);
    if ((this->s1oIdx + 1) * this->cubeS1BaseSize +
        this->tilingData->inputParams.nextTokens >= this->s2Size) {
        this->s2EndIdx = actualS2Len;
    } else {
        this->s2EndIdx = (this->s1oIdx + 1) * this->cubeS1BaseSize +
                         this->tilingData->inputParams.nextTokens;
    }
    this->s2EndIdx = Min(this->s2EndIdx, actualS2Len);
} else if (this->tilingData->inputParams.sparseType ==
               static_cast<uint8_t>(SparseModeEnum::RIGHT_DOWN_CAUSAL_BAND)) {
    if (this->boIdx == this->tilingData->inputParams.bandIndex) {
        this->s2StartIdx = 0;
        this->s2EndIdx = Min((this->s1oIdx + 1) * this->cubeS1BaseSize + actualS2Len +
                                Max(actualS2Len - actualS1Len, 0),
                            actualS2Len);
    } else {
        this->s2StartIdx = 0;
        this->s2EndIdx = Min((this->s1oIdx + 1) * this->cubeS1BaseSize + actualS2Len - actualS1Len,
                            actualS2Len);
    }
} else if (this->tilingData->inputParams.sparseType ==
               static_cast<uint8_t>(SparseModeEnum::BAND_LEFT_UP_CAUSAL)) {
    if (this->boIdx == this->tilingData->inputParams.bandIndex) {
        this->s2StartIdx = 0;
        this->s2EndIdx = Min((this->s1oIdx + 1) * this->cubeS1BaseSize + actualS2Len -
                                Max(actualS1Len - this->tilingData->inputParams.nextTokens, 0),
                            actualS2Len);
    } else {
        this->s2StartIdx = 0;
        this->s2EndIdx = Min((this->s1oIdx + 1) * this->cubeS1BaseSize, actualS1Len);
    }
}
```

---

## 六、TilingKey生成中的BAND模式

**文件位置**: `op_kernel/arch32/flash_attention_score_template_tiling_key.h:98-106`

```cpp
// Sparse类型枚举:
//          0：ALL
//          1：NONE
//          2：ANY
//          3：CAUSAL
//          4：BAND
//          5：PREFIX
//          6：BAND_COMPRESS
//          7：RIGHT_DOWN_CAUSAL
//          8：RIGHT_DOWN_CAUSAL_BAND
//          9：BAND_LEFT_UP_CAUSAL
ASCENDC_TPL_UINT_DECL(Sparse, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
```

---

## 七、BAND模式相关核心参数总结（ARCH32）

### 7.1 输入属性参数

| 参数名 | 类型 | 说明 | 代码位置 |
|--------|------|------|----------|
| `sparse_mode` | int64_t | 稀疏模式，4表示BAND模式 | `flash_attention_score_tiling_general.cpp:1080` |
| `pre_tokens` | int64_t | 向前看token数量 | `flash_attention_score_tiling_general.cpp:1040` |
| `next_tokens` | int64_t | 向后看token数量 | `flash_attention_score_tiling_general.cpp:1041` |
| `input_layout` | string | 输入数据布局格式 | `flash_attention_score_tiling_general.cpp:1042` |
| `inner_precise` | int64_t | 内部精度模式 | `flash_attention_score_tiling_general.cpp:1071` |

### 7.2 中间计算参数

| 参数名 | 类型 | 说明 | 代码位置 |
|--------|------|------|----------|
| `s1SparseValidSize` | int64_t | S1维度稀疏有效大小 | `flash_attention_score_tiling_general.cpp:743` |
| `s2SparseValidSize` | int64_t | S2维度稀疏有效大小 | `flash_attention_score_tiling_general.cpp:744` |
| `attenMaskCompressMode` | uint8_t | 注意力掩码压缩模式 | `flash_attention_score_tiling_general.cpp:1086` |
| `sparseType` | SparseEnum | 稀疏类型枚举 | `flash_attention_score_tiling_general.cpp:710` |
| `implMode` | ImplMode | 实现模式（高精度/高性能） | `flash_attention_score_tiling_general.cpp:701` |
| `bandIndex` | int64_t | BAND索引（用于复合模式） | `flash_attention_score_tiling.h:773` |

### 7.3 核心处理参数

| 参数名 | 类型 | 说明 | 代码位置 |
|--------|------|------|----------|
| `s1Size` | int64_t | Query序列长度 | `flash_attention_score_tiling_general.cpp` |
| `s2Size` | int64_t | Key/Value序列长度 | `flash_attention_score_tiling_general.cpp` |
| `s1BaseSize` | int64_t | S1基本块大小 | `flash_attention_score_tiling.h` |
| `s2BaseSize` | int64_t | S2基本块大小 | `flash_attention_score_tiling.h` |
| `cubeS1BaseSize` | int64_t | Cube S1基本大小（某些布局） | `flash_attention_var_len_score_sab.h` |

### 7.4 内核执行参数

| 参数名 | 类型 | 说明 | 代码位置 |
|--------|------|------|----------|
| `s2StartIdx` | int64_t | S2行起始索引 | `flash_attention_var_len_score.h:409` |
| `s2EndIdx` | int64_t | S2行结束索引 | `flash_attention_var_len_score.h:416` |
| `s1oIdx` | int64_t | S1外层索引 | `flash_attention_var_len_score.h:409` |
| `actualS1Len` | int64_t | 实际S1长度（变长场景） | `flash_attention_var_len_score.h:410` |
| `actualS2Len` | int64_t | 实际S2长度（变长场景） | `flash_attention_var_len_score.h:415` |

---

## 八、完整执行流程图（ARCH32）

```
用户调用FlashAttentionScore算子
         ↓
[flash_attention_score_tiling.cpp:345] TilingFlashAttentionScore入口
         ↓
检查NPU架构类型（非DAV_3510走ARCH32路径）
         ↓
[flash_attention_score_tiling_general.cpp] DoOpTiling
         ↓
[flash_attention_score_tiling_general.cpp:1037] ParseAttrParams解析属性
    - preTokens (向前token数)
    - nextTokens (向后token数)
    - sparse_mode (稀疏模式)
    - 设置 attenMaskCompressMode = BAND_MODE
         ↓
[flash_attention_score_tiling_general.cpp:1338] GetSparseInfo获取稀疏信息
         ↓
[sparse_mode == 4] 判断是否为BAND模式
         ↓
[flash_attention_score_tiling_general.cpp:730] SparseModeProcess处理稀疏模式
         ↓
[flash_attention_score_tiling_general.cpp:680] PretokenAndNexttokenAdjustment调整参数
    - S1≠S2时转换参考点
    - preTokens = s1Size - s2Size + preTokens
    - nextTokens = s2Size - s1Size + nextTokens
         ↓
[flash_attention_score_tiling_general.cpp:730] SparseBandModeCheck检查BAND模式
    - 计算s1SparseValidSize
    - 计算s2SparseValidSize
    - EnableBandInvalidLineImplMode可能启用高精度模式
         ↓
[flash_attention_score_tiling_general.cpp:1112] SetSparseTilingInfo设置稀疏信息
    - sparseType = SparseEnum::BAND
    - attenMaskCompressMode = BAND_MODE
    - preTokens, nextTokens
    - s1SparseValidSize, s2SparseValidSize
         ↓
设置TilingData参数到tilingData_.inputParams
         ↓
生成TilingKey (包含Sparse=4标识)
         ↓
[flash_attention_var_len_score.h:408] 内核执行
    - 计算s2StartIdx和s2EndIdx
    - 应用BAND模式的稀疏计算
    - s2StartIdx = Max(s1oIdx * s1BaseSize - s1SparseValidSize, 0)
    - s2EndIdx = Min((s1oIdx + 1) * s1BaseSize + s2SparseValidSize, actualS2Len)
         ↓
返回FlashAttentionScore结果
```

---

## 九、关键触发条件总结（ARCH32）

### 9.1 直接BAND模式触发

- **sparse_mode == 4** (SparseMode::BAND)
- **sparse_mode == 6** (SparseMode::BAND_COMPRESS)
- **sparse_mode == 8** (SparseMode::RIGHT_DOWN_CAUSAL_BAND)
- **sparse_mode == 9** (SparseMode::BAND_LEFT_UP_CAUSAL)

### 9.2 间接BAND模式转换

**文件位置**: `flash_attention_score_tiling_general.cpp:700-716`

```cpp
} else if (sparseMode_ == static_cast<int64_t>(RIGHT_DOWN_CAUSAL)) {
    if (s1Size_ == s2Size_) {
        // 标准因果模式
        sparseType = SparseEnum::CAUSAL;
    } else {
        // 不等S序列长度时转换为BAND模式
        preTokens_ = s1Size_;
        nextTokens_ = s2Size_ - s1Size_;
        sparseType = SparseEnum::BAND;
        EnableBandInvalidLineImplMode();
    }
}
```

### 9.3 NO_MASK模式下的BAND转换

当 `sparse_mode == 0` 且不满足全掩码或因果模式条件时，会调用 `SparseBandModeCheck` 进行BAND模式判断。

---

## 十、重要常量和限制（ARCH32）

### 10.1 高性能块大小

**文件位置**: `op_host/arch32/flash_attention_score_tiling_general.cpp`

```cpp
static const int64_t HIGH_PERF_BLOCK_SIZE = 128L;
```

### 10.2 序列长度限制

- **dSize最大值**: 768 (`HEAD_DIM_MAX_VALUE`)
- **S1/S2最大值**: `std::numeric_limits<int32_t>::max()`
- **对齐要求**: 128字节对齐

### 10.3 模式兼容性

- BAND模式支持所有布局类型（BSH, SBH, BNSD, BSND, TND）
- 支持的数据类型：FP16, BF16, FP32, FP8
- 复合模式需要额外的 `bandIndex` 参数

---

## 十一、注意事项（ARCH32）

### 11.1 参数配置要点

1. **preTokens和nextTokens的配置**
   - 对于BAND模式，这两个参数定义了滑窗的范围
   - 负值表示向不同方向扩展
   - 需要根据实际序列长度合理配置

2. **S1≠S2场景处理**
   - 会自动进行坐标转换，确保以左上顶点为参考
   - 可能触发 `AA_INVALID_LINE_HIGH_PRECISION` 模式

3. **数据布局选择**
   - ARCH32 支持多种布局，性能和内存占用各不相同
   - BNSD 布局通常性能最优

### 11.2 性能优化建议

1. **序列长度对齐**: 尽量按128字节对齐序列长度
2. **合理设置稀疏范围**: 避免过大或过小的preTokens/nextTokens值
3. **布局选择**: 根据实际场景选择最优的数据布局

### 11.3 调试信息

关键日志输出位置：
- `flash_attention_score_tiling_general.cpp`: 属性解析日志
- `flash_attention_score_tiling_general.cpp:703`: 无效行模式启用日志

---

## 十二、总结（ARCH32专用）

BAND模式是FlashAttentionScore算子中实现滑窗注意力机制的核心功能，ARCH32 架构提供了完整的实现支持。整个链路从属性解析、模式判断、参数计算到内核实现形成完整闭环，支持多种复合模式和优化策略，为不同场景的Transformer模型提供高效的注意力计算支持。

**ARCH32 关键特性**：
- 灵活的滑窗范围控制
- 自动参数调整和坐标转换
- 支持复合模式（如RIGHT_DOWN_CAUSAL_BAND）
- 多种实现模式（高精度/高性能）
- 完整的参数校验和错误处理
- 支持标准布局和变长序列场景

**主要文件索引（ARCH32）**：
- 算子定义: `op_graph/flash_attention_score_proto.h`
- Tiling处理: `op_host/arch32/flash_attention_score_tiling_general.cpp`
- 内核实现: `op_kernel/arch32/flash_attention_var_len_score.h`
- 内核实现: `op_kernel/arch32/flash_attention_score_s1s2_bn2gs1.h`
- 通用定义: `op_kernel/arch32/flash_attention_score_common.h`
- TilingKey定义: `op_kernel/arch32/flash_attention_score_template_tiling_key.h`

**与 ARCH35 的主要区别**：
- ARCH32 使用 `FlashAttentionScoreTilingBase` 基类
- ARCH35 使用 `FlashAttentionScoreTilingRegbase` 基类
- 内核实现路径不同，但算法逻辑基本一致
- 参数传递和存储结构略有差异