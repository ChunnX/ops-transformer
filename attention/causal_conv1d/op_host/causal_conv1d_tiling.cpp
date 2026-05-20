/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file causal_conv1d_tiling.cpp
 * \brief
 */

#include <algorithm>
#include <limits>

#include "log/log.h"
#include "op_common/op_host/util/platform_util.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "util/math_util.h"
#include "../op_kernel/causal_conv1d_tiling_data.h"
#include "../op_kernel/causal_conv1d_tiling_key.h"

namespace optiling {
namespace CausalConv1d {

using namespace Ops::Transformer::OpTiling;

constexpr uint32_t X_INDEX = 0;
constexpr uint32_t WEIGHT_INDEX = 1;
constexpr uint32_t BIAS_INDEX = 2;
constexpr uint32_t CONV_STATES_INDEX = 3;
constexpr uint32_t QUERY_START_LOC_INDEX = 4;
constexpr uint32_t CACHE_INDICES_INDEX = 5;
constexpr uint32_t INITIAL_STATE_MODE_INDEX = 6;
constexpr uint32_t NUM_ACCEPTED_TOKENS_INDEX = 7;


constexpr int64_t ASCENDC_RESERVED_WORKSPACE_SIZE = 16 * 1024 * 1024;

constexpr int64_t DIM_ALIGN_BYTES = 32;


static int64_t FloorPow2(int64_t n)
{
    if (n <= 0) {
        return 0;
    }
    int64_t result = 1;
    while (result <= n) {
        result <<= 1;
    }
    return result >> 1;
}

static int64_t FloorPow2Multiple(int64_t n, int64_t base)
{
    if (n < base || base <= 0) {
        return base > 0 ? base : 0;
    }
    return FloorPow2(n / base) * base;
}

struct CausalConv1dCompileInfo {};

struct CausalConv1dAttrInfo {
    int64_t activationMode = 0;
    int64_t padSlotId = -1;
    int64_t runMode = 0;
};

struct CausalConv1dInputInfo {
    int64_t dim = 0;
    int64_t cuSeqlen = 0;
    int64_t seqLen = 0;
    int64_t batch = 0;
    int64_t kernelWidth = 0;
    int64_t stateLen = 0;
    int64_t numCacheLines = 0;
    uint32_t inputModeKey = CAUSAL_CONV1D_TPL_INPUT_MODE_FN_VARLEN;
    int64_t elemBytes = 0;
    bool hasCacheIndices = false;
    bool hasInitialStateMode = false;
    bool hasNumAcceptedTokens = false;
    bool hasBias = false;
};

// ============================================================
// CausalConv1dTiling
// ============================================================

class CausalConv1dTiling {
public:
    explicit CausalConv1dTiling(gert::TilingContext *context) : context_(context) {}
    ge::graphStatus DoTiling();

private:
    gert::TilingContext *context_;
    const int64_t *qslData_ = nullptr;
    CausalConv1dInputInfo inputInfo_;
    uint64_t ubSize_ = 0;
    uint32_t coreNum_ = 0;
    uint64_t cacheLineSize_ = 0;
    int64_t ubLimitedBaseDim_ = 0;
    int64_t baseDim_ = 0;
    int64_t tokenBlockSize_ = 0;
    int64_t tokenBlockCnt_ = 0;
    ge::graphStatus GetPlatformInfo();
    ge::graphStatus SetWorkspaceSize(size_t workspaceSize);
    ge::graphStatus SetupFnInitStateWorkspace();
    ge::graphStatus GetAttrsInfo(CausalConv1dAttrInfo &attrInfo);
    ge::graphStatus ParseCausalConv1dInputInfo(const CausalConv1dAttrInfo &attrInfo);

    ge::graphStatus SetTilingData(const CausalConv1dAttrInfo &attrInfo);
    void ChooseBaseDim(int64_t cuSeqlen, int64_t dim);
    static constexpr bool IsFnInputMode(uint32_t inputModeKey);
    static constexpr bool IsVarlenInputModeKey(uint32_t key) {
        return key == CAUSAL_CONV1D_TPL_INPUT_MODE_FN_VARLEN ||
               key == CAUSAL_CONV1D_TPL_INPUT_MODE_UPDATE_VARLEN;
    }
    static constexpr uint32_t NormalizeWidthTilingKey(uint32_t inputModeKey, int32_t kernelWidth);

    ge::graphStatus ValidateDimSize(int64_t dim, ge::DataType xDtype);
    ge::graphStatus ResolveQslData(int64_t qslSize, int64_t cuSeqlen);
    ge::graphStatus ParseXShape(bool isFnMode);
    ge::graphStatus ParseWeightShape();
    ge::graphStatus ParseConvStatesShape();
    ge::graphStatus ParseQueryStartLoc(bool isFnMode);
    ge::graphStatus ParseCacheIndices(int64_t padSlotId);
    ge::graphStatus ParseInitialStateMode(bool isFnMode);
    ge::graphStatus ParseNumAcceptedTokens(bool isFnMode);
    ge::graphStatus ParseBias();

    int64_t ComputeUbLimitedBaseDim();
};

// ============================================================
// Key normalization
// ============================================================

constexpr bool CausalConv1dTiling::IsFnInputMode(uint32_t inputModeKey)
{
    return (inputModeKey == CAUSAL_CONV1D_TPL_INPUT_MODE_FN_VARLEN) ||
           (inputModeKey == CAUSAL_CONV1D_TPL_INPUT_MODE_FN_BATCH);
}

constexpr uint32_t CausalConv1dTiling::NormalizeWidthTilingKey(uint32_t inputModeKey, int32_t kernelWidth)
{
    if (!IsFnInputMode(inputModeKey)) {
        return CAUSAL_CONV1D_TPL_WIDTH_RUNTIME;
    }
    switch (kernelWidth) {
        case 2:
            return CAUSAL_CONV1D_TPL_WIDTH_2;
        case 3:
            return CAUSAL_CONV1D_TPL_WIDTH_3;
        case 4:
            return CAUSAL_CONV1D_TPL_WIDTH_4;
        default:
            return CAUSAL_CONV1D_TPL_WIDTH_RUNTIME;
    }
}

// ============================================================
// Platform info
// ============================================================

ge::graphStatus CausalConv1dTiling::GetPlatformInfo()
{
    fe::PlatFormInfos *platformInfoPtr = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum_ = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum_ == 0, OP_LOGE(context_, "coreNum_ is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize_);
    OP_CHECK_IF(ubSize_ == 0, OP_LOGE(context_, "ubSize_ is 0"), return ge::GRAPH_FAILED);
    cacheLineSize_ = Ops::Base::GetCacheLineSize(context_);
    OP_CHECK_IF(cacheLineSize_ == 0, OP_LOGE(context_, "cacheLineSize_ is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CausalConv1dTiling::SetWorkspaceSize(size_t workspaceSize)
{
    size_t *currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = workspaceSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CausalConv1dTiling::SetupFnInitStateWorkspace()
{
    if (!IsFnInputMode(inputInfo_.inputModeKey) || !inputInfo_.hasInitialStateMode) {
        return SetWorkspaceSize(0);
    }
    const int64_t snapshotBytes = inputInfo_.numCacheLines * inputInfo_.stateLen *
                                  inputInfo_.dim * inputInfo_.elemBytes;
    OP_CHECK_IF(SetWorkspaceSize(static_cast<size_t>(ASCENDC_RESERVED_WORKSPACE_SIZE + snapshotBytes)) !=
                    ge::GRAPH_SUCCESS,
                OP_LOGE(context_, "SetWorkspaceSize error"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->SetScheduleMode(1) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_, "SetScheduleMode(1) error"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// ============================================================
// Attrs
// ============================================================

ge::graphStatus CausalConv1dTiling::GetAttrsInfo(CausalConv1dAttrInfo &attrInfo)
{
    constexpr int32_t kActivationModeAttrIndex = 0;
    constexpr int32_t kPadSlotIdAttrIndex = 1;
    constexpr int32_t kRunModeAttrIndex = 2;

    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    const int64_t *activationModePtr = attrs->GetAttrPointer<int64_t>(kActivationModeAttrIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context_, activationModePtr);
    attrInfo.activationMode = *activationModePtr;
    OP_CHECK_IF(attrInfo.activationMode != 0 && attrInfo.activationMode != 1,
                OP_LOGE(context_, "activationMode only supports 0/1"),
                return ge::GRAPH_FAILED);

    const int64_t *padSlotIdPtr = attrs->GetAttrPointer<int64_t>(kPadSlotIdAttrIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context_, padSlotIdPtr);
    attrInfo.padSlotId = *padSlotIdPtr;

    const int64_t *runModePtr = attrs->GetAttrPointer<int64_t>(kRunModeAttrIndex);
    attrInfo.runMode = (runModePtr != nullptr) ? *runModePtr : 0;
    OP_CHECK_IF(attrInfo.runMode != 0 && attrInfo.runMode != 1, OP_LOGE(context_, "run_mode only supports 0/1"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// ============================================================
// Validation
// ============================================================

ge::graphStatus CausalConv1dTiling::ValidateDimSize(int64_t dim, ge::DataType xDtype)
{
    const int64_t dtypeSize = static_cast<int64_t>(ge::GetSizeByDataType(xDtype));
    OP_CHECK_IF((dim * dtypeSize) % DIM_ALIGN_BYTES != 0,
                OP_LOGE(context_,
                        "dim must satisfy (dim * dtypeSize) %% %ld == 0 for causal_conv1d; "
                        "got dim=%ld, dtypeSize=%ld.",
                        DIM_ALIGN_BYTES, dim, dtypeSize),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CausalConv1dTiling::ResolveQslData(int64_t qslSize, int64_t cuSeqlen)
{
    const gert::Tensor *qslTensor = context_->GetOptionalInputTensor(QUERY_START_LOC_INDEX);
    qslData_ = (qslTensor != nullptr) ? qslTensor->GetData<int64_t>() : nullptr;
    if (qslData_ == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(qslData_[0] != 0, OP_LOGE(context_, "queryStartLoc[0]=%ld must be 0", qslData_[0]), return ge::GRAPH_FAILED);
    OP_CHECK_IF(qslData_[qslSize - 1] != cuSeqlen,
                OP_LOGE(context_, "queryStartLoc[last] must equal cuSeqlen, got %ld vs %ld",
                        qslData_[qslSize - 1], cuSeqlen),
                return ge::GRAPH_FAILED);
    for (int64_t i = 0; i + 1 < qslSize; ++i) {
        const int64_t cur = qslData_[i];
        const int64_t nxt = qslData_[i + 1];
        OP_CHECK_IF(cur < 0 || cur > cuSeqlen,
                    OP_LOGE(context_, "queryStartLoc[%ld] out of range: %ld (cuSeqlen=%ld)", i, cur, cuSeqlen),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(nxt < 0 || nxt > cuSeqlen,
                    OP_LOGE(context_, "queryStartLoc[%ld] out of range: %ld (cuSeqlen=%ld)", i + 1, nxt, cuSeqlen),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(nxt < cur,
                    OP_LOGE(context_,
                            "queryStartLoc must be non-decreasing, got queryStartLoc[%ld]=%ld queryStartLoc[%ld]=%ld",
                            i, cur, i + 1, nxt),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

// ============================================================
// Shape & dtype info
// ============================================================

ge::graphStatus CausalConv1dTiling::ParseXShape(bool isFnMode)
{
    auto xShapePtr = context_->GetInputShape(X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShapePtr);
    auto xShape = EnsureNotScalar(xShapePtr->GetStorageShape());

    if (xShape.GetDimNum() == 2) {
        if (!isFnMode) {
            inputInfo_.inputModeKey = CAUSAL_CONV1D_TPL_INPUT_MODE_UPDATE_BATCH;
            inputInfo_.batch = xShape.GetDim(0);
            inputInfo_.dim = xShape.GetDim(1);
            inputInfo_.seqLen = 1;
            inputInfo_.cuSeqlen = inputInfo_.batch;
            OP_CHECK_IF(inputInfo_.batch <= 0 || inputInfo_.dim <= 0,
                        OP_LOGE(context_, "invalid x shape for 2D decode mode, batch=%ld dim=%ld",
                                inputInfo_.batch, inputInfo_.dim),
                        return ge::GRAPH_FAILED);
        } else {
            inputInfo_.inputModeKey = CAUSAL_CONV1D_TPL_INPUT_MODE_FN_VARLEN;
            inputInfo_.cuSeqlen = xShape.GetDim(0);
            inputInfo_.dim = xShape.GetDim(1);
            inputInfo_.seqLen = 0;
            OP_CHECK_IF(inputInfo_.dim <= 0 || inputInfo_.cuSeqlen < 0,
                        OP_LOGE(context_, "invalid x shape for 2D varlen mode, dim=%ld cuSeqlen=%ld", inputInfo_.dim, inputInfo_.cuSeqlen),
                        return ge::GRAPH_FAILED);
        }
    } else if (xShape.GetDimNum() == 3) {
        inputInfo_.inputModeKey = isFnMode ? CAUSAL_CONV1D_TPL_INPUT_MODE_FN_BATCH
                                           : CAUSAL_CONV1D_TPL_INPUT_MODE_UPDATE_BATCH;
        inputInfo_.batch = xShape.GetDim(0);
        inputInfo_.seqLen = xShape.GetDim(1);
        inputInfo_.dim = xShape.GetDim(2);
        inputInfo_.cuSeqlen = inputInfo_.batch * inputInfo_.seqLen;
        OP_CHECK_IF(inputInfo_.batch <= 0 || inputInfo_.dim <= 0 || inputInfo_.seqLen <= 0,
                    OP_LOGE(context_, "invalid x shape for 3D batch mode, batch=%ld dim=%ld seqLen=%ld",
                            inputInfo_.batch, inputInfo_.dim, inputInfo_.seqLen),
                    return ge::GRAPH_FAILED);
    } else {
        OP_LOGE(context_, "x must be 2D (cu_seqlen, dim) or 3D (batch, seqlen, dim)");
        return ge::GRAPH_FAILED;
    }
    auto xDesc = context_->GetInputDesc(X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xDesc);
    const ge::DataType xDtype = xDesc->GetDataType();
    OP_CHECK_IF(xDtype != ge::DT_BF16 && xDtype != ge::DT_FLOAT16,
                OP_LOGE(context_, "x dtype only supports bf16/fp16, got %d", static_cast<int>(xDtype)),
                return ge::GRAPH_FAILED);
    inputInfo_.elemBytes = static_cast<int64_t>(ge::GetSizeByDataType(xDtype));
    OP_CHECK_IF(ValidateDimSize(inputInfo_.dim, xDtype) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_, "dim alignment validation failed"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(inputInfo_.cuSeqlen > static_cast<int64_t>(std::numeric_limits<int32_t>::max()),
                OP_LOGE(context_, "cuSeqlen is too large for int32 indexing, got %ld", inputInfo_.cuSeqlen),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CausalConv1dTiling::ParseWeightShape()
{
    auto wShapePtr = context_->GetInputShape(WEIGHT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, wShapePtr);
    auto wShape = EnsureNotScalar(wShapePtr->GetStorageShape());
    OP_CHECK_IF(wShape.GetDimNum() != 2, OP_LOGE(context_, "weight must be 2D: (kernelWidth, dim), got %ldD", wShape.GetDimNum()), return ge::GRAPH_FAILED);
    inputInfo_.kernelWidth = wShape.GetDim(0);
    const int64_t wDim = wShape.GetDim(1);
    OP_CHECK_IF(wDim != inputInfo_.dim, OP_LOGE(context_, "weight.shape[1]=%ld must equal dim=%ld", wDim, inputInfo_.dim), return ge::GRAPH_FAILED);
    OP_CHECK_IF(inputInfo_.kernelWidth < 2 || inputInfo_.kernelWidth > 4,
                OP_LOGE(context_, "Only support kernelWidth in [2,4] now, actually is %ld.", inputInfo_.kernelWidth),
                return ge::GRAPH_FAILED);

    auto wDesc = context_->GetInputDesc(WEIGHT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, wDesc);
    auto xDesc = context_->GetInputDesc(X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xDesc);
    OP_CHECK_IF(wDesc->GetDataType() != xDesc->GetDataType(),
                OP_LOGE(context_, "weight dtype must equal x dtype, got %d vs %d", static_cast<int>(wDesc->GetDataType()), static_cast<int>(xDesc->GetDataType())),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CausalConv1dTiling::ParseConvStatesShape()
{
    auto convStatesShapePtr = context_->GetInputShape(CONV_STATES_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, convStatesShapePtr);
    auto convStatesShape = EnsureNotScalar(convStatesShapePtr->GetStorageShape());
    OP_CHECK_IF(convStatesShape.GetDimNum() != 3, OP_LOGE(context_, "convStates must be 3D: (num_cache_lines, state_len, dim)"),
                return ge::GRAPH_FAILED);
    inputInfo_.numCacheLines = convStatesShape.GetDim(0);
    inputInfo_.stateLen = convStatesShape.GetDim(1);
    const int64_t convStatesDim = convStatesShape.GetDim(2);
    OP_CHECK_IF(inputInfo_.numCacheLines <= 0,
                OP_LOGE(context_, "convStates.shape[0] (num_cache_lines=%ld) must be > 0", inputInfo_.numCacheLines),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(convStatesDim != inputInfo_.dim, OP_LOGE(context_, "convStates.shape[2]=%ld must equal dim=%ld", convStatesDim, inputInfo_.dim), return ge::GRAPH_FAILED);
    OP_CHECK_IF(inputInfo_.stateLen < (inputInfo_.kernelWidth - 1),
                OP_LOGE(context_, "convStates.shape[1]=%ld must be >= kernelWidth-1=%ld", inputInfo_.stateLen, inputInfo_.kernelWidth - 1),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(inputInfo_.numCacheLines < inputInfo_.batch,
                OP_LOGE(context_,
                        "convStates.shape[0] (num_cache_lines=%ld) must be >= batch=%ld",
                        inputInfo_.numCacheLines, inputInfo_.batch),
                return ge::GRAPH_FAILED);

    auto convStatesDesc = context_->GetInputDesc(CONV_STATES_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, convStatesDesc);
    auto xDesc = context_->GetInputDesc(X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xDesc);
    OP_CHECK_IF(convStatesDesc->GetDataType() != xDesc->GetDataType(),
                OP_LOGE(context_, "convStates dtype must equal x dtype, got %d vs %d", static_cast<int>(convStatesDesc->GetDataType()), static_cast<int>(xDesc->GetDataType())),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CausalConv1dTiling::ParseQueryStartLoc(bool isFnMode)
{
    // Step 1: determine qsl existence and size from shape
    auto qslShapePtr = context_->GetOptionalInputShape(QUERY_START_LOC_INDEX);
    const gert::CompileTimeTensorDesc *qslDesc = context_->GetOptionalInputDesc(QUERY_START_LOC_INDEX);
    bool qslExist = false;
    int64_t qslSize = 0;
    if (qslShapePtr != nullptr) {
        const auto qslStorageShape = qslShapePtr->GetStorageShape();
        const int64_t qslDimNum = qslStorageShape.GetDimNum();
        qslExist = !(qslDimNum == 0 || (qslDimNum == 1 && qslStorageShape.GetDim(0) <= 0));
        if (qslExist) {
            auto qslShape = EnsureNotScalar(qslStorageShape);
            OP_CHECK_IF(qslShape.GetDimNum() != 1, OP_LOGE(context_, "queryStartLoc must be 1D, got %ldD", qslShape.GetDimNum()),
                        return ge::GRAPH_FAILED);
            qslSize = qslShape.GetDim(0);
            OP_CHECK_IF(qslSize < 1, OP_LOGE(context_, "queryStartLoc.size must be >= 1, got %ld", qslSize), return ge::GRAPH_FAILED);
            OP_CHECK_NULL_WITH_CONTEXT(context_, qslDesc);
            OP_CHECK_IF(qslDesc->GetDataType() != ge::DT_INT64, OP_LOGE(context_, "queryStartLoc dtype must be int64, got %d", static_cast<int>(qslDesc->GetDataType())),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(ResolveQslData(qslSize, inputInfo_.cuSeqlen) != ge::GRAPH_SUCCESS,
                        OP_LOGE(context_, "ResolveQslData error"), return ge::GRAPH_FAILED);
        }
    }

    if (qslExist) {
        if (!isFnMode && !IsVarlenInputModeKey(inputInfo_.inputModeKey) &&
            (qslSize - 1) != inputInfo_.batch) {
            inputInfo_.inputModeKey = CAUSAL_CONV1D_TPL_INPUT_MODE_UPDATE_VARLEN;
            inputInfo_.seqLen = 0;
        }
        if (IsVarlenInputModeKey(inputInfo_.inputModeKey)) {
            inputInfo_.batch = qslSize - 1;
        } else {
            OP_CHECK_IF(qslSize != inputInfo_.batch + 1,
                        OP_LOGE(context_, "queryStartLoc.size=%ld must equal batch=%ld + 1", qslSize, inputInfo_.batch),
                        return ge::GRAPH_FAILED);
        }
    } else {
        OP_CHECK_IF(IsVarlenInputModeKey(inputInfo_.inputModeKey),
                    OP_LOGE(context_, "queryStartLoc is required in varlen mode"),
                    return ge::GRAPH_FAILED);
        qslSize = inputInfo_.batch + 1;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CausalConv1dTiling::ParseCacheIndices(int64_t padSlotId)
{
    auto cIdxShapePtr = context_->GetOptionalInputShape(CACHE_INDICES_INDEX);
    if (cIdxShapePtr == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    const auto cIdxStorageShape = cIdxShapePtr->GetStorageShape();
    const int64_t cIdxDimNum = cIdxStorageShape.GetDimNum();
    inputInfo_.hasCacheIndices = (cIdxDimNum > 0) && (cIdxDimNum != 1 || cIdxStorageShape.GetDim(0) > 0);
    if (!inputInfo_.hasCacheIndices) {
        return ge::GRAPH_SUCCESS;
    }
    auto cIdxShape = EnsureNotScalar(cIdxStorageShape);
    OP_CHECK_IF(cIdxShape.GetDimNum() != 1, OP_LOGE(context_, "cacheIndices must be 1D, got %ldD", cIdxShape.GetDimNum()), return ge::GRAPH_FAILED);
    OP_CHECK_IF(cIdxShape.GetDim(0) != inputInfo_.batch,
                OP_LOGE(context_, "cacheIndices.size=%ld must equal batch=%ld", cIdxShape.GetDim(0), inputInfo_.batch),
                return ge::GRAPH_FAILED);

    auto cIdxDesc = context_->GetOptionalInputDesc(CACHE_INDICES_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, cIdxDesc);
    OP_CHECK_IF(cIdxDesc->GetDataType() != ge::DT_INT64, OP_LOGE(context_, "cacheIndices dtype must be int64, got %d", static_cast<int>(cIdxDesc->GetDataType())),
                return ge::GRAPH_FAILED);

    const gert::Tensor *cIdxTensor = context_->GetOptionalInputTensor(CACHE_INDICES_INDEX);
    const int64_t *cIdxData = (cIdxTensor != nullptr) ? cIdxTensor->GetData<int64_t>() : nullptr;
    if (cIdxData == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    for (int64_t i = 0; i < inputInfo_.batch; ++i) {
        const int64_t v = cIdxData[i];
        if (v == padSlotId) {
            continue;
        }
        OP_CHECK_IF(v > static_cast<int64_t>(std::numeric_limits<int32_t>::max()),
                    OP_LOGE(context_, "cacheIndices[%ld]=%ld exceeds int32 range", i, v),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(v < 0 || v >= inputInfo_.numCacheLines,
                    OP_LOGE(context_,
                            "cacheIndices[%ld]=%ld out of range [0, num_cache_lines=%ld), padSlotId=%ld", i,
                            v, inputInfo_.numCacheLines, padSlotId),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CausalConv1dTiling::ParseInitialStateMode(bool isFnMode)
{
    auto initStShapePtr = context_->GetOptionalInputShape(INITIAL_STATE_MODE_INDEX);
    if (initStShapePtr == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    const auto initStStorageShape = initStShapePtr->GetStorageShape();
    const int64_t initStDimNum = initStStorageShape.GetDimNum();
    inputInfo_.hasInitialStateMode = (initStDimNum > 0) && (initStDimNum != 1 || initStStorageShape.GetDim(0) > 0);
    if (!inputInfo_.hasInitialStateMode) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(!isFnMode,
                OP_LOGE(context_, "initialStateMode is only supported in run_mode=0 (fn/prefill)"),
                return ge::GRAPH_FAILED);
    auto initStShape = EnsureNotScalar(initStStorageShape);
    OP_CHECK_IF(initStShape.GetDimNum() != 1, OP_LOGE(context_, "initialStateMode must be 1D, got %ldD", initStShape.GetDimNum()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(initStShape.GetDim(0) != inputInfo_.batch,
                OP_LOGE(context_, "initialStateMode.size=%ld must equal batch=%ld", initStShape.GetDim(0), inputInfo_.batch),
                return ge::GRAPH_FAILED);

    auto initStDesc = context_->GetOptionalInputDesc(INITIAL_STATE_MODE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, initStDesc);
    OP_CHECK_IF(initStDesc->GetDataType() != ge::DT_INT64, OP_LOGE(context_, "initialStateMode dtype must be int64, got %d", static_cast<int>(initStDesc->GetDataType())),
                return ge::GRAPH_FAILED);

    const gert::Tensor *initStTensor = context_->GetOptionalInputTensor(INITIAL_STATE_MODE_INDEX);
    const int64_t *initStData = (initStTensor != nullptr) ? initStTensor->GetData<int64_t>() : nullptr;
    if (initStData == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    for (int64_t i = 0; i < inputInfo_.batch; ++i) {
        const int64_t v = initStData[i];
        OP_CHECK_IF(v != 0 && v != 1,
                    OP_LOGE(context_, "initialStateMode[%ld]=%ld is invalid (only supports 0/1)", i, v),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CausalConv1dTiling::ParseNumAcceptedTokens(bool isFnMode)
{
    auto numAccShapePtr = context_->GetOptionalInputShape(NUM_ACCEPTED_TOKENS_INDEX);
    if (numAccShapePtr == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    const auto numAccStorageShape = numAccShapePtr->GetStorageShape();
    const int64_t numAccDimNum = numAccStorageShape.GetDimNum();
    inputInfo_.hasNumAcceptedTokens = (numAccDimNum > 0) && (numAccDimNum != 1 || numAccStorageShape.GetDim(0) > 0);
    if (!inputInfo_.hasNumAcceptedTokens) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(isFnMode,
                OP_LOGE(context_, "numAcceptedTokens is only supported in run_mode=1 (decode/update)"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(inputInfo_.kernelWidth != 4,
                OP_LOGE(context_, "numAcceptedTokens is only supported for kernelWidth=4 currently"),
                return ge::GRAPH_FAILED);
    auto numAccShape = EnsureNotScalar(numAccStorageShape);
    OP_CHECK_IF(numAccShape.GetDimNum() != 1, OP_LOGE(context_, "numAcceptedTokens must be 1D, got %ldD", numAccShape.GetDimNum()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(numAccShape.GetDim(0) != inputInfo_.batch,
                OP_LOGE(context_, "numAcceptedTokens.size=%ld must equal batch=%ld", numAccShape.GetDim(0), inputInfo_.batch),
                return ge::GRAPH_FAILED);

    if (!IsVarlenInputModeKey(inputInfo_.inputModeKey)) {
        const int64_t reqStateLen = (inputInfo_.kernelWidth - 1) + (inputInfo_.seqLen - 1);
        OP_CHECK_IF(inputInfo_.stateLen < reqStateLen,
                    OP_LOGE(context_,
                            "spec decode requires stateLen >= (kernelWidth-1) + (seqlen-1), got stateLen=%ld req=%ld",
                            inputInfo_.stateLen, reqStateLen),
                    return ge::GRAPH_FAILED);
    }

    auto numAccDesc = context_->GetOptionalInputDesc(NUM_ACCEPTED_TOKENS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, numAccDesc);
    OP_CHECK_IF(numAccDesc->GetDataType() != ge::DT_INT64, OP_LOGE(context_, "numAcceptedTokens dtype must be int64, got %d", static_cast<int>(numAccDesc->GetDataType())),
                return ge::GRAPH_FAILED);

    const gert::Tensor *numAccTensor = context_->GetOptionalInputTensor(NUM_ACCEPTED_TOKENS_INDEX);
    const int64_t *numAccData = (numAccTensor != nullptr) ? numAccTensor->GetData<int64_t>() : nullptr;
    if (numAccData == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    for (int64_t i = 0; i < inputInfo_.batch; ++i) {
        const int64_t a = numAccData[i];
        OP_CHECK_IF(a < 0, OP_LOGE(context_, "numAcceptedTokens[%ld]=%ld is invalid (must be >= 0)", i, a),
                    return ge::GRAPH_FAILED);

        if (!IsVarlenInputModeKey(inputInfo_.inputModeKey)) {
            OP_CHECK_IF(a > inputInfo_.seqLen,
                        OP_LOGE(context_, "numAcceptedTokens[%ld]=%ld exceeds seqlen=%ld", i, a, inputInfo_.seqLen),
                        return ge::GRAPH_FAILED);
        } else if (qslData_ != nullptr) {
            const int64_t lenI = qslData_[i + 1] - qslData_[i];
            OP_CHECK_IF(a > lenI,
                        OP_LOGE(context_, "numAcceptedTokens[%ld]=%ld exceeds varlen segment length=%ld", i,
                                a, lenI),
                        return ge::GRAPH_FAILED);
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CausalConv1dTiling::ParseBias()
{
    auto biasShapePtr = context_->GetOptionalInputShape(BIAS_INDEX);
    if (biasShapePtr == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    const auto biasStorageShape = biasShapePtr->GetStorageShape();
    const int64_t biasDimNum = biasStorageShape.GetDimNum();
    inputInfo_.hasBias = (biasDimNum > 0) && (biasDimNum != 1 || biasStorageShape.GetDim(0) > 0);
    if (!inputInfo_.hasBias) {
        return ge::GRAPH_SUCCESS;
    }
    auto biasShape = EnsureNotScalar(biasStorageShape);
    OP_CHECK_IF(biasShape.GetDimNum() != 1, OP_LOGE(context_, "bias must be 1D: (dim,), got %ldD", biasShape.GetDimNum()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(biasShape.GetDim(0) != inputInfo_.dim, OP_LOGE(context_, "bias.size=%ld must equal dim=%ld", biasShape.GetDim(0), inputInfo_.dim),
                return ge::GRAPH_FAILED);

    auto biasDesc = context_->GetOptionalInputDesc(BIAS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, biasDesc);
    auto xDesc = context_->GetInputDesc(X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xDesc);
    OP_CHECK_IF(biasDesc->GetDataType() != xDesc->GetDataType(),
                OP_LOGE(context_, "bias dtype must equal x dtype, got %d vs %d", static_cast<int>(biasDesc->GetDataType()), static_cast<int>(xDesc->GetDataType())),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// ============================================================
// Shape & dtype info (orchestrator)
// ============================================================

ge::graphStatus CausalConv1dTiling::ParseCausalConv1dInputInfo(const CausalConv1dAttrInfo &attrInfo)
{
    const bool isFnMode = (attrInfo.runMode == 0);

    // Parse order and data flow (all functions write to inputInfo_ directly):
    //
    //  ┌─ ParseXShape ──► dim, cuSeqlen, seqLen, batch, inputModeKey
    //  ├─ ParseQueryStartLoc ──► may refine batch, seqLen, inputModeKey from x shape
    //  │
    //  ├─ ParseWeightShape ──► kernelWidth (needs dim)
    //  ├─ ParseConvStatesShape ──► stateLen, numCacheLines (needs dim, kernelWidth, final batch)
    //  └─ ParseCacheIndices ──► hasCacheIndices (needs batch, numCacheLines)
    //
    //  ┌─ ParseInitialStateMode ──► hasInitialStateMode (needs batch)
    //  ├─ ParseNumAcceptedTokens ──► hasNumAcceptedTokens (needs batch, kernelWidth, stateLen, seqLen, inputModeKey)
    //  └─ ParseBias ───────────────► hasBias (needs dim)

    // Phase 1: x shape + batch refinement
    OP_CHECK_IF(ParseXShape(isFnMode) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_, "ParseXShape error"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ParseQueryStartLoc(isFnMode) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_, "ParseQueryStartLoc error"), return ge::GRAPH_FAILED);

    // Phase 2: weight → conv_states → cache_indices (state management chain)
    OP_CHECK_IF(ParseWeightShape() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_, "ParseWeightShape error"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ParseConvStatesShape() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_, "ParseConvStatesShape error"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ParseCacheIndices(attrInfo.padSlotId) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_, "ParseCacheIndices error"), return ge::GRAPH_FAILED);

    // Phase 3: remaining optional inputs
    OP_CHECK_IF(ParseInitialStateMode(isFnMode) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_, "ParseInitialStateMode error"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ParseNumAcceptedTokens(isFnMode) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_, "ParseNumAcceptedTokens error"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ParseBias() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_, "ParseBias error"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// ============================================================
// Tiling planning
// ============================================================

int64_t CausalConv1dTiling::ComputeUbLimitedBaseDim()
{
    constexpr int64_t kOutBufCnt = 2;
    constexpr int64_t kCtxBufCnt = 3; // acc + tmp + curr

    const int64_t elemBytes = inputInfo_.elemBytes;
    const int64_t alignElems = static_cast<int64_t>(cacheLineSize_) / elemBytes;

    const int64_t ringBufCnt = inputInfo_.kernelWidth + 1;  // (W-1) history + 1 curr + 1 prefetch = RING_SLOTS
    const int64_t calcBufCnt = inputInfo_.kernelWidth + (inputInfo_.hasBias ? 1 : 0) + kCtxBufCnt;  // weight + bias + acc/tmp/curr
    const int64_t bytesPerElem = (ringBufCnt * elemBytes) + (kOutBufCnt * elemBytes) +
                                 (calcBufCnt * static_cast<int64_t>(sizeof(float)));
    const int64_t rawBaseDim = static_cast<int64_t>(ubSize_) / bytesPerElem;
    if (rawBaseDim <= 0) {
        return 0;
    }
    return FloorPow2Multiple(rawBaseDim, alignElems);
}

void CausalConv1dTiling::ChooseBaseDim(int64_t cuSeqlen, int64_t dim)
{
    if (dim <= ubLimitedBaseDim_) {
        baseDim_ = dim;
        int64_t group = static_cast<int64_t>(coreNum_);
        tokenBlockSize_ = Ops::Base::CeilDiv(cuSeqlen, group);
        tokenBlockCnt_ = Ops::Base::CeilDiv(cuSeqlen, tokenBlockSize_);
        return;
    }

    const int64_t alignElems = static_cast<int64_t>(cacheLineSize_) / inputInfo_.elemBytes;
    int64_t bestBaseDim = alignElems;
    int64_t maxActualCores = 0;

    for (int64_t chunks = ubLimitedBaseDim_ / alignElems; chunks >= 1; chunks >>= 1) {
        int64_t candidate = chunks * alignElems;
        int64_t baseDimCnt = Ops::Base::CeilDiv(dim, candidate);
        int64_t group = static_cast<int64_t>(coreNum_) / baseDimCnt;
        if (group <= 0) {
            break;
        }
        int64_t tokensPerCore = Ops::Base::CeilDiv(cuSeqlen, group);
        int64_t tokenBlocks = Ops::Base::CeilDiv(cuSeqlen, tokensPerCore);
        int64_t actualCores = tokenBlocks * baseDimCnt;
        if (actualCores > maxActualCores) {
            maxActualCores = actualCores;
            bestBaseDim = candidate;
            tokenBlockSize_ = tokensPerCore;
            tokenBlockCnt_ = tokenBlocks;
        }
    }

    baseDim_ = bestBaseDim;
}

} // namespace CausalConv1d

using namespace CausalConv1d;

// ============================================================
// Tiling entry
// ============================================================

ge::graphStatus CausalConv1dTiling::SetTilingData(const CausalConv1dAttrInfo &attrInfo)
{
    CausalConv1dTilingData *tiling = context_->GetTilingData<CausalConv1dTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, tiling);

    tiling->activationMode = attrInfo.activationMode;
    tiling->padSlotId = attrInfo.padSlotId;
    tiling->dim = inputInfo_.dim;
    tiling->cuSeqlen = inputInfo_.cuSeqlen;
    tiling->seqLen = inputInfo_.seqLen;
    tiling->batch = inputInfo_.batch;
    tiling->kernelWidth = inputInfo_.kernelWidth;
    tiling->stateLen = inputInfo_.stateLen;
    tiling->numCacheLines = inputInfo_.numCacheLines;
    tiling->hasCacheIndices = inputInfo_.hasCacheIndices ? 1 : 0;
    tiling->hasInitialStateMode = inputInfo_.hasInitialStateMode ? 1 : 0;
    tiling->hasNumAcceptedTokens = inputInfo_.hasNumAcceptedTokens ? 1 : 0;
    tiling->hasBias = inputInfo_.hasBias ? 1 : 0;
    tiling->baseDim = baseDim_;
    tiling->hasInitStateWorkspace = inputInfo_.hasInitialStateMode ? 1 : 0;
    tiling->tokenBlockSize = tokenBlockSize_;
    tiling->tokenBlockCnt = tokenBlockCnt_;

    OP_LOGD(context_,
            "Tiling: batch[%ld], dim[%ld], baseDim[%ld], tokenTiling[%ld,%ld], hasActivation[%d], hasBias[%d].",
            inputInfo_.batch, inputInfo_.dim, baseDim_,
            tokenBlockSize_, tokenBlockCnt_,
            static_cast<int32_t>(attrInfo.activationMode != 0), static_cast<int32_t>(tiling->hasBias));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CausalConv1dTiling::DoTiling()
{
    OP_CHECK_IF(GetPlatformInfo() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_, "GetPlatformInfo error"), return ge::GRAPH_FAILED);

    CausalConv1dAttrInfo attrInfo;
    OP_CHECK_IF(GetAttrsInfo(attrInfo) != ge::GRAPH_SUCCESS, OP_LOGE(context_, "GetAttrsInfo error"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(ParseCausalConv1dInputInfo(attrInfo) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_, "ParseCausalConv1dInputInfo error"), return ge::GRAPH_FAILED);

    ubLimitedBaseDim_ = ComputeUbLimitedBaseDim();
    OP_CHECK_IF(ubLimitedBaseDim_ <= 0, OP_LOGE(context_, "UB too small to form valid baseDim"), return ge::GRAPH_FAILED);

    const uint32_t inputModeKey = inputInfo_.inputModeKey;
    OP_CHECK_IF(inputInfo_.dim <= 0 || inputInfo_.batch <= 0,
                OP_LOGE(context_, "dim/batch must be positive"), return ge::GRAPH_FAILED);

    ChooseBaseDim(inputInfo_.cuSeqlen, inputInfo_.dim);

    uint32_t blockDim = static_cast<uint32_t>(tokenBlockCnt_ * Ops::Base::CeilDiv(inputInfo_.dim, baseDim_));

    const uint32_t widthKey = NormalizeWidthTilingKey(inputModeKey, static_cast<int32_t>(inputInfo_.kernelWidth));
    OP_CHECK_IF(SetupFnInitStateWorkspace() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_, "SetupWorkspace error"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(SetTilingData(attrInfo) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_, "SetTilingData error"), return ge::GRAPH_FAILED);

    context_->SetBlockDim(blockDim);
    const uint64_t tilingKey = GET_TPL_TILING_KEY(inputModeKey, widthKey);
    context_->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CausalConv1dTilingFunc(gert::TilingContext *context)
{
    CausalConv1dTiling tctx(context);
    return tctx.DoTiling();
}

static ge::graphStatus TilingParseForCausalConv1d(gert::TilingParseContext *context)
{
    OP_LOGD(context, "Enter TilingParseForCausalConv1d.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(CausalConv1d)
    .Tiling(CausalConv1dTilingFunc)
    .TilingParse<CausalConv1dCompileInfo>(TilingParseForCausalConv1d);

} // namespace optiling
