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
constexpr int64_t BF16_FP16_ELEM_BYTES = 2;
constexpr int64_t DIM_ALIGN_ELEMS = DIM_ALIGN_BYTES / BF16_FP16_ELEM_BYTES;
constexpr int64_t MAX_DIM_TILE_SIZE = 4096;
constexpr int64_t FN_UB_RESERVED_BYTES = 512;
constexpr int64_t RING_SLOT_CNT = 5;
constexpr int64_t FN_OUT_SLOT_CNT = 2;
constexpr int64_t FN_CALC_FP32_SLOT_CNT = 8;

constexpr int64_t MAX_FN_TOKEN_SEQ_RANGE_COUNT = 128;

struct CausalConv1dCompileInfo {
    uint64_t ubSize = 0;
    uint32_t coreNum = 0;
};

struct CausalConv1dAttrInfo {
    int64_t activationMode = 0;
    int64_t padSlotId = -1;
    int64_t runMode = 0;
};

struct DimTileChoice {
    int64_t baseDim = 0;
    int64_t baseDimCnt = 0;
    int64_t gridSize = 0;
};

struct VarlenTokenTileChoice {
    bool enabled = false;
    int64_t tokenBlockSize = 0;
    int64_t tokenBlockCnt = 0;
    int64_t gridSize = 0;
};

enum FnTilingCaseKind : int64_t {
    FN_TILING_CASE_INVALID = 0,
    FN_TILING_CASE_TOKEN_FIRST = 1,
    FN_TILING_CASE_TOKEN_DIM_CO_SPLIT = 2,
};

struct TokenCoreMappingChoice {
    int64_t tokenCoreBudget = 0;
    int64_t tokenBlocksPerCore = 0;
    int64_t tokenCoreTailCnt = 0;
    int64_t blockDim = 0;
};

struct FnTokenSeqRangePlan {
    bool enabled = false;
    int64_t rangeCount = 0;
    int64_t tokenTileStartSeq[MAX_FN_TOKEN_SEQ_RANGE_COUNT] = {};
    int64_t tokenTileEndSeq[MAX_FN_TOKEN_SEQ_RANGE_COUNT] = {};
};

struct FnHostPlan {
    FnTilingCaseKind caseKind = FN_TILING_CASE_INVALID;
    FnExecutionPlan executionPlan = FN_EXECUTION_PLAN_INVALID;
    DimTileChoice baseDimChoice;
    VarlenTokenTileChoice tokenBlockChoice;
    TokenCoreMappingChoice tokenCoreMapping;
    FnTokenSeqRangePlan tokenSeqRangePlan;
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
    uint32_t inputModeKey_ = CAUSAL_CONV1D_TPL_INPUT_MODE_FN_VARLEN;
    const int64_t *qslData_ = nullptr;
    CausalConv1dInputInfo inputInfo_;

    ge::graphStatus GetPlatformInfo(uint64_t &ubSize, uint32_t &coreNum);
    ge::graphStatus SetWorkspaceSize(size_t workspaceSize);
    ge::graphStatus GetAttrsInfo(CausalConv1dAttrInfo &attrInfo);
    ge::graphStatus ParseCausalConv1dInputInfo(const CausalConv1dAttrInfo &attrInfo);

    void SetTilingData(CausalConv1dTilingData &tiling, const CausalConv1dAttrInfo &attrInfo,
                       const DimTileChoice &baseDimChoice, bool isFn,
                       const FnHostPlan &fnHostPlan);
    DimTileChoice ChooseCanonicalUpdateBaseDimChoice(int64_t batch, int64_t dim, uint32_t coreNum);
    FnHostPlan ChooseFnHostPlan(uint64_t ubSize, uint32_t coreNum);
    FnTokenSeqRangePlan BuildFnTokenSeqRangePlan(const int64_t *qslData, int64_t batch,
                                                    int64_t tokenBlockSize, int64_t tokenBlockCnt);

    static constexpr bool IsFnInputMode(uint32_t inputModeKey);
    static constexpr bool IsVarlenInputModeKey(uint32_t key) {
        return key == CAUSAL_CONV1D_TPL_INPUT_MODE_FN_VARLEN ||
               key == CAUSAL_CONV1D_TPL_INPUT_MODE_UPDATE_VARLEN;
    }
    static constexpr uint32_t NormalizeFnPlanTilingKey(uint32_t inputModeKey, FnExecutionPlan fnExecutionPlan);
    static constexpr uint32_t NormalizeWidthTilingKey(uint32_t inputModeKey, int32_t kernelWidth);
    static const char *GetFnTilingCaseName(FnTilingCaseKind caseKind);

    ge::graphStatus ValidateAlignedDim(int64_t dim, ge::DataType xDtype);
    ge::graphStatus ResolveQslData(int64_t qslSize, int64_t cuSeqlen);
    ge::graphStatus ParseXShape(const gert::Shape &xShape, bool isFnMode,
                                int64_t &dim, int64_t &cuSeqlen, int64_t &seqLen,
                                int64_t &batch, uint32_t &inputModeKey);
    ge::graphStatus ParseWeightShape(int64_t dim, int64_t &kernelWidth);
    ge::graphStatus ParseConvStatesShape(int64_t dim, int64_t kernelWidth, int64_t batch,
                                         int64_t &stateLen, int64_t &numCacheLines);
    ge::graphStatus ParseQueryStartLoc(int64_t cuSeqlen, bool isFnMode,
                                       uint32_t &inputModeKey, int64_t &batch, int64_t &seqLen);
    ge::graphStatus ParseCacheIndices(int64_t batch, int64_t numCacheLines,
                                      int64_t padSlotId);
    ge::graphStatus ParseInitialStateMode(int64_t batch, bool isFnMode);
    ge::graphStatus ParseNumAcceptedTokens(int64_t batch, uint32_t inputModeKey, int64_t kernelWidth,
                                           int64_t stateLen, int64_t seqLen, bool isFnMode);
    ge::graphStatus ParseBias(int64_t dim);

    int64_t ResolveFnTokenCoreBudget(int64_t baseDimCnt, FnExecutionPlan fnExecutionPlan, uint32_t coreNum);
    int64_t ComputeFnUbLimitedBaseDim(uint64_t ubSize);
    DimTileChoice ChooseFnTokenFirstBaseDimChoice(int64_t dim);
    DimTileChoice ChooseFnTokenDimCoSplitBaseDimChoice(int64_t dim, uint64_t ubSize, uint32_t coreNum);
    TokenCoreMappingChoice BuildFnTokenCoreMappingChoice(int64_t tokenBlockCnt, int64_t baseDimCnt,
                                                           FnExecutionPlan fnExecutionPlan, uint32_t coreNum);
    VarlenTokenTileChoice ChooseFnTokenBlockChoice(int64_t cuSeqlen, int64_t baseDimCnt,
                                                     FnExecutionPlan fnExecutionPlan, uint32_t coreNum);
    VarlenTokenTileChoice ChooseUnifiedFnTokenBlockPlan(const DimTileChoice &baseDimChoice,
                                                          FnExecutionPlan fnExecutionPlan, uint32_t coreNum);
};

// ============================================================
// Key normalization
// ============================================================

constexpr bool CausalConv1dTiling::IsFnInputMode(uint32_t inputModeKey)
{
    return (inputModeKey == CAUSAL_CONV1D_TPL_INPUT_MODE_FN_VARLEN) ||
           (inputModeKey == CAUSAL_CONV1D_TPL_INPUT_MODE_FN_BATCH);
}

constexpr uint32_t CausalConv1dTiling::NormalizeFnPlanTilingKey(uint32_t inputModeKey,
                                                                  FnExecutionPlan fnExecutionPlan)
{
    if (!IsFnInputMode(inputModeKey)) {
        return CAUSAL_CONV1D_TPL_FN_PLAN_INVALID;
    }
    switch (fnExecutionPlan) {
        case FN_EXECUTION_PLAN_CUTBS:
            return CAUSAL_CONV1D_TPL_FN_PLAN_CUTBS;
        case FN_EXECUTION_PLAN_CUTBSD:
            return CAUSAL_CONV1D_TPL_FN_PLAN_CUTBSD;
        default:
            return CAUSAL_CONV1D_TPL_FN_PLAN_INVALID;
    }
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

const char *CausalConv1dTiling::GetFnTilingCaseName(FnTilingCaseKind caseKind)
{
    switch (caseKind) {
        case FN_TILING_CASE_TOKEN_FIRST:
            return "token_first";
        case FN_TILING_CASE_TOKEN_DIM_CO_SPLIT:
            return "token_dim_co_split";
        default:
            return "invalid";
    }
}

// ============================================================
// Platform info
// ============================================================

ge::graphStatus CausalConv1dTiling::GetPlatformInfo(uint64_t &ubSize, uint32_t &coreNum)
{
    fe::PlatFormInfos *platformInfoPtr = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context_, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context_, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CausalConv1dTiling::SetWorkspaceSize(size_t workspaceSize)
{
    size_t *currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = workspaceSize;
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

ge::graphStatus CausalConv1dTiling::ValidateAlignedDim(int64_t dim, ge::DataType xDtype)
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
    OP_CHECK_IF(qslData_[0] != 0, OP_LOGE(context_, "queryStartLoc[0] must be 0"), return ge::GRAPH_FAILED);
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

ge::graphStatus CausalConv1dTiling::ParseXShape(const gert::Shape &xShape, bool isFnMode,
                                                  int64_t &dim, int64_t &cuSeqlen, int64_t &seqLen,
                                                  int64_t &batch, uint32_t &inputModeKey)
{
    if (xShape.GetDimNum() == 2) {
        if (!isFnMode) {
            inputModeKey = CAUSAL_CONV1D_TPL_INPUT_MODE_UPDATE_BATCH;
            batch = xShape.GetDim(0);
            dim = xShape.GetDim(1);
            seqLen = 1;
            cuSeqlen = batch;
            OP_CHECK_IF(batch <= 0 || dim <= 0, OP_LOGE(context_, "invalid x shape for 2D decode mode"),
                        return ge::GRAPH_FAILED);
        } else {
            inputModeKey = CAUSAL_CONV1D_TPL_INPUT_MODE_FN_VARLEN;
            cuSeqlen = xShape.GetDim(0);
            dim = xShape.GetDim(1);
            seqLen = 0;
            OP_CHECK_IF(dim <= 0 || cuSeqlen < 0, OP_LOGE(context_, "invalid x shape for 2D varlen mode"),
                        return ge::GRAPH_FAILED);
        }
    } else if (xShape.GetDimNum() == 3) {
        inputModeKey = isFnMode ? CAUSAL_CONV1D_TPL_INPUT_MODE_FN_BATCH
                                : CAUSAL_CONV1D_TPL_INPUT_MODE_UPDATE_BATCH;
        batch = xShape.GetDim(0);
        seqLen = xShape.GetDim(1);
        dim = xShape.GetDim(2);
        cuSeqlen = batch * seqLen;
        OP_CHECK_IF(batch <= 0 || dim <= 0 || seqLen <= 0, OP_LOGE(context_, "invalid x shape for 3D batch mode"),
                    return ge::GRAPH_FAILED);
    } else {
        OP_LOGE(context_, "x must be 2D (cu_seqlen, dim) or 3D (batch, seqlen, dim)");
        return ge::GRAPH_FAILED;
    }
    auto xDesc = context_->GetInputDesc(X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xDesc);
    const ge::DataType xDtype = xDesc->GetDataType();
    OP_CHECK_IF(xDtype != ge::DT_BF16 && xDtype != ge::DT_FLOAT16,
                OP_LOGE(context_, "x dtype only supports bf16/fp16"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ValidateAlignedDim(dim, xDtype) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_, "dim alignment validation failed"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(cuSeqlen > static_cast<int64_t>(std::numeric_limits<int32_t>::max()),
                OP_LOGE(context_, "cuSeqlen is too large for int32 indexing, got %ld", cuSeqlen),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CausalConv1dTiling::ParseWeightShape(int64_t dim, int64_t &kernelWidth)
{
    auto wShapePtr = context_->GetInputShape(WEIGHT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, wShapePtr);
    auto wShape = EnsureNotScalar(wShapePtr->GetStorageShape());
    OP_CHECK_IF(wShape.GetDimNum() != 2, OP_LOGE(context_, "weight must be 2D: (kernelWidth, dim)"), return ge::GRAPH_FAILED);
    kernelWidth = wShape.GetDim(0);
    const int64_t wDim = wShape.GetDim(1);
    OP_CHECK_IF(wDim != dim, OP_LOGE(context_, "weight.shape[1] must equal dim"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(kernelWidth < 2 || kernelWidth > 4, OP_LOGE(context_, "Only support kernelWidth in [2,4] now, actually is %ld.", kernelWidth),
                return ge::GRAPH_FAILED);

    auto wDesc = context_->GetInputDesc(WEIGHT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, wDesc);
    auto xDesc = context_->GetInputDesc(X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xDesc);
    OP_CHECK_IF(wDesc->GetDataType() != xDesc->GetDataType(),
                OP_LOGE(context_, "weight dtype must equal x dtype"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CausalConv1dTiling::ParseConvStatesShape(int64_t dim, int64_t kernelWidth,
                                                           int64_t batch,
                                                           int64_t &stateLen, int64_t &numCacheLines)
{
    auto sShapePtr = context_->GetInputShape(CONV_STATES_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, sShapePtr);
    auto sShape = EnsureNotScalar(sShapePtr->GetStorageShape());
    OP_CHECK_IF(sShape.GetDimNum() != 3, OP_LOGE(context_, "convStates must be 3D: (num_cache_lines, state_len, dim)"),
                return ge::GRAPH_FAILED);
    numCacheLines = sShape.GetDim(0);
    stateLen = sShape.GetDim(1);
    const int64_t sDim = sShape.GetDim(2);
    OP_CHECK_IF(numCacheLines <= 0, OP_LOGE(context_, "convStates.shape[0] (num_cache_lines) must be > 0"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(sDim != dim, OP_LOGE(context_, "convStates.shape[2] must equal dim"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(stateLen < (kernelWidth - 1), OP_LOGE(context_, "convStates.shape[1] must be >= kernelWidth-1"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(numCacheLines < batch,
                OP_LOGE(context_,
                        "convStates.shape[0] (num_cache_lines=%ld) must be >= batch=%ld",
                        numCacheLines, batch),
                return ge::GRAPH_FAILED);

    auto sDesc = context_->GetInputDesc(CONV_STATES_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, sDesc);
    auto xDesc = context_->GetInputDesc(X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xDesc);
    OP_CHECK_IF(sDesc->GetDataType() != xDesc->GetDataType(),
                OP_LOGE(context_, "convStates dtype must equal x dtype"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CausalConv1dTiling::ParseQueryStartLoc(int64_t cuSeqlen, bool isFnMode,
                                                         uint32_t &inputModeKey, int64_t &batch, int64_t &seqLen)
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
            OP_CHECK_IF(qslShape.GetDimNum() != 1, OP_LOGE(context_, "queryStartLoc must be 1D"),
                        return ge::GRAPH_FAILED);
            qslSize = qslShape.GetDim(0);
            OP_CHECK_IF(qslSize < 1, OP_LOGE(context_, "queryStartLoc.size must be >= 1"), return ge::GRAPH_FAILED);
            OP_CHECK_NULL_WITH_CONTEXT(context_, qslDesc);
            OP_CHECK_IF(qslDesc->GetDataType() != ge::DT_INT64, OP_LOGE(context_, "queryStartLoc dtype must be int64"),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(ResolveQslData(qslSize, cuSeqlen) != ge::GRAPH_SUCCESS,
                        OP_LOGE(context_, "ResolveQslData error"), return ge::GRAPH_FAILED);
        }
    }

    if (qslExist) {
        // Step 2: when 2D decode [B,D] gets a qsl whose segment count differs from the
        // inferred batch, switch to varlen mode — the qsl segments define the real batches.
        if (!isFnMode && !IsVarlenInputModeKey(inputModeKey) && (qslSize - 1) != batch) {
            inputModeKey = CAUSAL_CONV1D_TPL_INPUT_MODE_UPDATE_VARLEN;
            seqLen = 0;
        }
        // Step 3: resolve batch ↔ qslSize.
        //   varlen:     batch is derived from qsl (batch = qslSize - 1)
        //   non-varlen: batch is known from x shape, qsl must match (qslSize == batch + 1)
        if (IsVarlenInputModeKey(inputModeKey)) {
            batch = qslSize - 1;
        } else {
            OP_CHECK_IF(qslSize != batch + 1, OP_LOGE(context_, "queryStartLoc.size must equal batch + 1"),
                        return ge::GRAPH_FAILED);
        }
    } else {
        // qsl absent: varlen mode requires it; non-varlen derives a synthetic size from batch.
        OP_CHECK_IF(IsVarlenInputModeKey(inputModeKey),
                    OP_LOGE(context_, "queryStartLoc is required in varlen mode"),
                    return ge::GRAPH_FAILED);
        qslSize = batch + 1;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CausalConv1dTiling::ParseCacheIndices(int64_t batch, int64_t numCacheLines,
                                                        int64_t padSlotId)
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
    OP_CHECK_IF(cIdxShape.GetDimNum() != 1, OP_LOGE(context_, "cacheIndices must be 1D"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(cIdxShape.GetDim(0) != batch, OP_LOGE(context_, "cacheIndices.size must equal batch"),
                return ge::GRAPH_FAILED);

    auto cIdxDesc = context_->GetOptionalInputDesc(CACHE_INDICES_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, cIdxDesc);
    OP_CHECK_IF(cIdxDesc->GetDataType() != ge::DT_INT64, OP_LOGE(context_, "cacheIndices dtype must be int64"),
                return ge::GRAPH_FAILED);

    const gert::Tensor *cIdxTensor = context_->GetOptionalInputTensor(CACHE_INDICES_INDEX);
    const int64_t *cIdxData = (cIdxTensor != nullptr) ? cIdxTensor->GetData<int64_t>() : nullptr;
    if (cIdxData != nullptr) {
        for (int64_t i = 0; i < batch; ++i) {
            const int64_t v = cIdxData[i];
            if (v == padSlotId) {
                continue;
            }
            OP_CHECK_IF(v > static_cast<int64_t>(std::numeric_limits<int32_t>::max()),
                        OP_LOGE(context_, "cacheIndices[%ld]=%ld exceeds int32 range", i, v),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(v < 0 || v >= numCacheLines,
                        OP_LOGE(context_,
                                "cacheIndices[%ld]=%ld out of range [0, num_cache_lines=%ld), padSlotId=%ld", i,
                                v, numCacheLines, padSlotId),
                        return ge::GRAPH_FAILED);
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CausalConv1dTiling::ParseInitialStateMode(int64_t batch, bool isFnMode)
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
    OP_CHECK_IF(initStShape.GetDimNum() != 1, OP_LOGE(context_, "initialStateMode must be 1D"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(initStShape.GetDim(0) != batch, OP_LOGE(context_, "initialStateMode.size must equal batch"),
                return ge::GRAPH_FAILED);

    auto initStDesc = context_->GetOptionalInputDesc(INITIAL_STATE_MODE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, initStDesc);
    OP_CHECK_IF(initStDesc->GetDataType() != ge::DT_INT64, OP_LOGE(context_, "initialStateMode dtype must be int64"),
                return ge::GRAPH_FAILED);

    const gert::Tensor *initStTensor = context_->GetOptionalInputTensor(INITIAL_STATE_MODE_INDEX);
    const int64_t *initStData = (initStTensor != nullptr) ? initStTensor->GetData<int64_t>() : nullptr;
    if (initStData != nullptr) {
        for (int64_t i = 0; i < batch; ++i) {
            const int64_t v = initStData[i];
            OP_CHECK_IF(v != 0 && v != 1,
                        OP_LOGE(context_, "initialStateMode[%ld]=%ld is invalid (only supports 0/1)", i, v),
                        return ge::GRAPH_FAILED);
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CausalConv1dTiling::ParseNumAcceptedTokens(int64_t batch, uint32_t inputModeKey, int64_t kernelWidth,
                                                             int64_t stateLen, int64_t seqLen, bool isFnMode)
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
    OP_CHECK_IF(kernelWidth != 4,
                OP_LOGE(context_, "numAcceptedTokens is only supported for kernelWidth=4 currently"),
                return ge::GRAPH_FAILED);
    auto numAccShape = EnsureNotScalar(numAccStorageShape);
    OP_CHECK_IF(numAccShape.GetDimNum() != 1, OP_LOGE(context_, "numAcceptedTokens must be 1D"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(numAccShape.GetDim(0) != batch, OP_LOGE(context_, "numAcceptedTokens.size must equal batch"),
                return ge::GRAPH_FAILED);

    if (!IsVarlenInputModeKey(inputModeKey)) {
        const int64_t reqStateLen = (kernelWidth - 1) + (seqLen - 1);
        OP_CHECK_IF(stateLen < reqStateLen,
                    OP_LOGE(context_,
                            "spec decode requires stateLen >= (kernelWidth-1) + (seqlen-1), got stateLen=%ld req=%ld",
                            stateLen, reqStateLen),
                    return ge::GRAPH_FAILED);
    }

    auto numAccDesc = context_->GetOptionalInputDesc(NUM_ACCEPTED_TOKENS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, numAccDesc);
    OP_CHECK_IF(numAccDesc->GetDataType() != ge::DT_INT64, OP_LOGE(context_, "numAcceptedTokens dtype must be int64"),
                return ge::GRAPH_FAILED);

    const gert::Tensor *numAccTensor = context_->GetOptionalInputTensor(NUM_ACCEPTED_TOKENS_INDEX);
    const int64_t *numAccData = (numAccTensor != nullptr) ? numAccTensor->GetData<int64_t>() : nullptr;
    if (numAccData != nullptr) {
        for (int64_t i = 0; i < batch; ++i) {
            const int64_t a = numAccData[i];
            OP_CHECK_IF(a < 0, OP_LOGE(context_, "numAcceptedTokens[%ld]=%ld is invalid (must be >= 0)", i, a),
                        return ge::GRAPH_FAILED);

            if (!IsVarlenInputModeKey(inputModeKey)) {
                OP_CHECK_IF(a > seqLen,
                            OP_LOGE(context_, "numAcceptedTokens[%ld]=%ld exceeds seqlen=%ld", i, a, seqLen),
                            return ge::GRAPH_FAILED);
            } else if (qslData_ != nullptr) {
                const int64_t lenI = qslData_[i + 1] - qslData_[i];
                OP_CHECK_IF(a > lenI,
                            OP_LOGE(context_, "numAcceptedTokens[%ld]=%ld exceeds varlen segment length=%ld", i,
                                    a, lenI),
                            return ge::GRAPH_FAILED);
            }
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CausalConv1dTiling::ParseBias(int64_t dim)
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
    OP_CHECK_IF(biasShape.GetDimNum() != 1, OP_LOGE(context_, "bias must be 1D: (dim,)"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(biasShape.GetDim(0) != dim, OP_LOGE(context_, "bias.size must equal dim"),
                return ge::GRAPH_FAILED);

    auto biasDesc = context_->GetOptionalInputDesc(BIAS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, biasDesc);
    auto xDesc = context_->GetInputDesc(X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xDesc);
    OP_CHECK_IF(biasDesc->GetDataType() != xDesc->GetDataType(),
                OP_LOGE(context_, "bias dtype must equal x dtype"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// ============================================================
// Shape & dtype info (orchestrator)
// ============================================================

ge::graphStatus CausalConv1dTiling::ParseCausalConv1dInputInfo(const CausalConv1dAttrInfo &attrInfo)
{
    const bool isFnMode = (attrInfo.runMode == 0);

    auto xShapePtr = context_->GetInputShape(X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShapePtr);
    auto xShape = EnsureNotScalar(xShapePtr->GetStorageShape());

    OP_CHECK_IF(ParseXShape(xShape, isFnMode, inputInfo_.dim, inputInfo_.cuSeqlen,
                            inputInfo_.seqLen, inputInfo_.batch,
                            inputInfo_.inputModeKey) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_, "ParseXShape error"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ParseWeightShape(inputInfo_.dim, inputInfo_.kernelWidth) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_, "ParseWeightShape error"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(ParseQueryStartLoc(inputInfo_.cuSeqlen, isFnMode, inputInfo_.inputModeKey,
                                   inputInfo_.batch, inputInfo_.seqLen) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_, "ParseQueryStartLoc error"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ParseConvStatesShape(inputInfo_.dim, inputInfo_.kernelWidth, inputInfo_.batch,
                                     inputInfo_.stateLen, inputInfo_.numCacheLines) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_, "ParseConvStatesShape error"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(ParseCacheIndices(inputInfo_.batch, inputInfo_.numCacheLines,
                                   attrInfo.padSlotId) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_, "ParseCacheIndices error"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(ParseInitialStateMode(inputInfo_.batch, isFnMode) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_, "ParseInitialStateMode error"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(ParseNumAcceptedTokens(inputInfo_.batch, inputInfo_.inputModeKey, inputInfo_.kernelWidth,
                                       inputInfo_.stateLen, inputInfo_.seqLen, isFnMode) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_, "ParseNumAcceptedTokens error"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(ParseBias(inputInfo_.dim) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_, "ParseBias error"), return ge::GRAPH_FAILED);

    inputModeKey_ = inputInfo_.inputModeKey;
    return ge::GRAPH_SUCCESS;
}

// ============================================================
// Tiling planning
// ============================================================

DimTileChoice CausalConv1dTiling::ChooseCanonicalUpdateBaseDimChoice(int64_t batch, int64_t dim, uint32_t coreNum)
{
    const int64_t candidates[] = {4096, 2048, 1024, 512, 384, 192};

    auto chooseOnce = [&](bool requireExactDiv) -> DimTileChoice {
        DimTileChoice bestOver;
        int64_t bestOverGap = std::numeric_limits<int64_t>::max();
        DimTileChoice bestUnder;

        for (int64_t baseDim : candidates) {
            if (baseDim <= 0) {
                continue;
            }
            if (requireExactDiv && (dim % baseDim != 0)) {
                continue;
            }

            const int64_t baseDimCnt = requireExactDiv ? (dim / baseDim) : Ops::Base::CeilDiv(dim, baseDim);
            const int64_t gridSize = batch * baseDimCnt;
            if (gridSize <= 0) {
                continue;
            }

            OP_LOGD(context_,
                    "DimTile(update) candidate[%s]: baseDim[%ld], baseDimCnt[%ld], gridSize[%ld], coreNum[%u].",
                    requireExactDiv ? "exact" : "tail", baseDim, baseDimCnt, gridSize, coreNum);
            if (gridSize >= static_cast<int64_t>(coreNum)) {
                const int64_t gap = gridSize - static_cast<int64_t>(coreNum);
                if (gap < bestOverGap) {
                    bestOver = {baseDim, baseDimCnt, gridSize};
                    bestOverGap = gap;
                }
            } else if (gridSize > bestUnder.gridSize ||
                       (gridSize == bestUnder.gridSize && baseDim < bestUnder.baseDim)) {
                bestUnder = {baseDim, baseDimCnt, gridSize};
            }
        }

        return (bestOver.baseDim != 0) ? bestOver : bestUnder;
    };

    DimTileChoice result = chooseOnce(true);
    if (result.baseDim == 0) {
        result = chooseOnce(false);
    }
    OP_LOGD(context_, "DimTile(update) chosen: baseDim[%ld], baseDimCnt[%ld], gridSize[%ld].", result.baseDim,
            result.baseDimCnt, result.gridSize);
    return result;
}

int64_t CausalConv1dTiling::ResolveFnTokenCoreBudget(int64_t baseDimCnt, FnExecutionPlan fnExecutionPlan,
                                                       uint32_t coreNum)
{
    if (baseDimCnt <= 0 || coreNum == 0 || fnExecutionPlan == FN_EXECUTION_PLAN_INVALID) {
        return 0;
    }

    int64_t tokenCoreBudget = static_cast<int64_t>(coreNum);
    if (fnExecutionPlan == FN_EXECUTION_PLAN_CUTBSD) {
        tokenCoreBudget = std::max<int64_t>(1, tokenCoreBudget / baseDimCnt);
    }
    return tokenCoreBudget;
}

int64_t CausalConv1dTiling::ComputeFnUbLimitedBaseDim(uint64_t ubSize)
{
    if (ubSize <= static_cast<uint64_t>(FN_UB_RESERVED_BYTES)) {
        return 0;
    }

    const int64_t bytesPerElem = (RING_SLOT_CNT * BF16_FP16_ELEM_BYTES) + (FN_OUT_SLOT_CNT * BF16_FP16_ELEM_BYTES) +
                                 (FN_CALC_FP32_SLOT_CNT * static_cast<int64_t>(sizeof(float)));
    const int64_t budgetBytes = static_cast<int64_t>(ubSize) - FN_UB_RESERVED_BYTES;
    const int64_t ubLimitedBaseDim = Ops::Base::FloorAlign(budgetBytes / bytesPerElem, DIM_ALIGN_ELEMS);
    return std::min<int64_t>(MAX_DIM_TILE_SIZE, ubLimitedBaseDim);
}

DimTileChoice CausalConv1dTiling::ChooseFnTokenFirstBaseDimChoice(int64_t dim)
{
    if (dim <= 0 || dim > MAX_DIM_TILE_SIZE) {
        return {};
    }
    return {dim, 1, 1};
}

DimTileChoice CausalConv1dTiling::ChooseFnTokenDimCoSplitBaseDimChoice(int64_t dim, uint64_t ubSize,
                                                                         uint32_t coreNum)
{
    if (dim <= 0) {
        return {};
    }

    const int64_t ubLimitedBaseDim = ComputeFnUbLimitedBaseDim(ubSize);
    if (ubLimitedBaseDim <= 0) {
        OP_LOGD(context_, "FnDimCoSplit: UB budget is too small to form a valid baseDim.");
        return {};
    }

    DimTileChoice result;
    result.baseDim = ubLimitedBaseDim;
    result.baseDimCnt = Ops::Base::CeilDiv(dim, result.baseDim);
    result.gridSize = result.baseDimCnt;

    if (coreNum == 0 || result.baseDimCnt <= 1 || result.baseDimCnt >= static_cast<int64_t>(coreNum) ||
        (coreNum % result.baseDimCnt == 0)) {
        OP_LOGD(context_,
                "FnDimCoSplit: dim[%ld], ubLimitedBaseDim[%ld], baseDimCnt[%ld], coreNum[%u], adjusted[%d].", dim,
                result.baseDim, result.baseDimCnt, coreNum, 0);
        return result;
    }

    int64_t adjustedBaseDimCnt = result.baseDimCnt;
    while (adjustedBaseDimCnt < static_cast<int64_t>(coreNum) && (coreNum % adjustedBaseDimCnt != 0)) {
        ++adjustedBaseDimCnt;
    }

    if (adjustedBaseDimCnt >= static_cast<int64_t>(coreNum)) {
        OP_LOGD(context_,
                "FnDimCoSplit: keep baseDimCnt[%ld] because no divisible adjustment exists under coreNum[%u].",
                result.baseDimCnt, coreNum);
        return result;
    }

    const int64_t adjustedBaseDim = Ops::Base::CeilAlign(Ops::Base::CeilDiv(dim, adjustedBaseDimCnt), DIM_ALIGN_ELEMS);
    if (adjustedBaseDim <= 0 || adjustedBaseDim > ubLimitedBaseDim || adjustedBaseDim > MAX_DIM_TILE_SIZE) {
        OP_LOGD(context_,
                "FnDimCoSplit: rejected adjusted baseDim[%ld] with baseDimCnt[%ld], ubLimitedBaseDim[%ld].",
                adjustedBaseDim, adjustedBaseDimCnt, ubLimitedBaseDim);
        return result;
    }

    result.baseDim = adjustedBaseDim;
    result.baseDimCnt = Ops::Base::CeilDiv(dim, result.baseDim);
    result.gridSize = result.baseDimCnt;
    OP_LOGD(context_,
            "FnDimCoSplit: dim[%ld], ubLimitedBaseDim[%ld], adjustedBaseDim[%ld], baseDimCnt[%ld], coreNum[%u].",
            dim, ubLimitedBaseDim, result.baseDim, result.baseDimCnt, coreNum);
    return result;
}

TokenCoreMappingChoice CausalConv1dTiling::BuildFnTokenCoreMappingChoice(int64_t tokenBlockCnt,
                                                                           int64_t baseDimCnt,
                                                                           FnExecutionPlan fnExecutionPlan,
                                                                           uint32_t coreNum)
{
    TokenCoreMappingChoice mapping;
    mapping.tokenCoreBudget = ResolveFnTokenCoreBudget(baseDimCnt, fnExecutionPlan, coreNum);
    if (tokenBlockCnt <= 0 || mapping.tokenCoreBudget <= 0 || baseDimCnt <= 0) {
        return mapping;
    }

    mapping.tokenBlocksPerCore = Ops::Base::CeilDiv(tokenBlockCnt, mapping.tokenCoreBudget);
    mapping.tokenCoreTailCnt =
        tokenBlockCnt - (std::max<int64_t>(0, mapping.tokenBlocksPerCore - 1) * mapping.tokenCoreBudget);
    if (mapping.tokenCoreTailCnt <= 0) {
        mapping.tokenCoreTailCnt = mapping.tokenCoreBudget;
    }
    mapping.blockDim = mapping.tokenCoreBudget * baseDimCnt;
    return mapping;
}

FnTokenSeqRangePlan CausalConv1dTiling::BuildFnTokenSeqRangePlan(const int64_t *qslData, int64_t batch,
                                                                   int64_t tokenBlockSize, int64_t tokenBlockCnt)
{
    FnTokenSeqRangePlan plan;
    if (qslData == nullptr || batch <= 0 || tokenBlockSize <= 0 || tokenBlockCnt <= 0 ||
        tokenBlockCnt > MAX_FN_TOKEN_SEQ_RANGE_COUNT) {
        return plan;
    }

    plan.enabled = true;
    plan.rangeCount = tokenBlockCnt;
    int64_t seq = 0;
    for (int64_t tokenTileId = 0; tokenTileId < tokenBlockCnt; ++tokenTileId) {
        const int64_t tokenStart = tokenTileId * tokenBlockSize;
        const int64_t tokenEnd = tokenStart + tokenBlockSize;

        while (seq < batch && qslData[seq + 1] <= tokenStart) {
            ++seq;
        }

        int64_t endSeq = seq;
        while (endSeq < batch && qslData[endSeq] < tokenEnd) {
            ++endSeq;
        }

        plan.tokenTileStartSeq[tokenTileId] = seq;
        plan.tokenTileEndSeq[tokenTileId] = endSeq;
    }
    return plan;
}

VarlenTokenTileChoice CausalConv1dTiling::ChooseFnTokenBlockChoice(int64_t cuSeqlen, int64_t baseDimCnt,
                                                                     FnExecutionPlan fnExecutionPlan,
                                                                     uint32_t coreNum)
{
    VarlenTokenTileChoice tokenBlockChoice;
    const int64_t tokenCoreBudget = ResolveFnTokenCoreBudget(baseDimCnt, fnExecutionPlan, coreNum);
    if (cuSeqlen <= 0 || tokenCoreBudget <= 0) {
        return tokenBlockChoice;
    }

    tokenBlockChoice.enabled = true;
    const int64_t idealBlockSize = Ops::Base::CeilDiv(cuSeqlen, tokenCoreBudget);
    tokenBlockChoice.tokenBlockSize = (idealBlockSize > 0) ? idealBlockSize : 1;
    tokenBlockChoice.tokenBlockCnt = Ops::Base::CeilDiv(cuSeqlen, tokenBlockChoice.tokenBlockSize);
    tokenBlockChoice.gridSize = tokenBlockChoice.tokenBlockCnt * baseDimCnt;
    return tokenBlockChoice;
}

VarlenTokenTileChoice CausalConv1dTiling::ChooseUnifiedFnTokenBlockPlan(
    const DimTileChoice &baseDimChoice, FnExecutionPlan fnExecutionPlan, uint32_t coreNum)
{
    VarlenTokenTileChoice tokenBlockChoice;
    if (!IsFnInputMode(inputModeKey_)) {
        return tokenBlockChoice;
    }
    if (inputInfo_.batch <= 0 || inputInfo_.cuSeqlen <= 0 ||
        baseDimChoice.baseDimCnt <= 0 || coreNum == 0 || fnExecutionPlan == FN_EXECUTION_PLAN_INVALID) {
        return tokenBlockChoice;
    }
    if (inputInfo_.hasNumAcceptedTokens) {
        OP_LOGD(context_, "Varlen token tiling disabled: speculative decode still uses the existing seq mapping.");
        return tokenBlockChoice;
    }

    tokenBlockChoice = ChooseFnTokenBlockChoice(inputInfo_.cuSeqlen, baseDimChoice.baseDimCnt, fnExecutionPlan, coreNum);

    OP_LOGD(context_,
            "FnTokenTile(plan=%ld): cuSeqlen[%ld], baseDimCnt[%ld], tokenBlockSize[%ld], "
            "tokenBlockCnt[%ld], gridSize[%ld].",
            static_cast<int64_t>(fnExecutionPlan), inputInfo_.cuSeqlen, baseDimChoice.baseDimCnt,
            tokenBlockChoice.tokenBlockSize, tokenBlockChoice.tokenBlockCnt, tokenBlockChoice.gridSize);
    return tokenBlockChoice;
}

FnHostPlan CausalConv1dTiling::ChooseFnHostPlan(uint64_t ubSize, uint32_t coreNum)
{
    FnHostPlan plan;
    if (!IsFnInputMode(inputModeKey_)) {
        return plan;
    }
    if (inputInfo_.batch <= 0 || inputInfo_.cuSeqlen <= 0 || inputInfo_.dim <= 0 || coreNum == 0) {
        return plan;
    }

    if (inputInfo_.dim <= MAX_DIM_TILE_SIZE) {
        plan.caseKind = FN_TILING_CASE_TOKEN_FIRST;
        plan.executionPlan = FN_EXECUTION_PLAN_CUTBS;
        plan.baseDimChoice = ChooseFnTokenFirstBaseDimChoice(inputInfo_.dim);
    } else {
        plan.caseKind = FN_TILING_CASE_TOKEN_DIM_CO_SPLIT;
        plan.executionPlan = FN_EXECUTION_PLAN_CUTBSD;
        plan.baseDimChoice = ChooseFnTokenDimCoSplitBaseDimChoice(inputInfo_.dim, ubSize, coreNum);
    }

    if (plan.baseDimChoice.baseDim <= 0 || plan.baseDimChoice.baseDimCnt <= 0) {
        return {};
    }

    plan.baseDimChoice.gridSize = inputInfo_.batch * plan.baseDimChoice.baseDimCnt;
    plan.tokenBlockChoice =
        ChooseUnifiedFnTokenBlockPlan(plan.baseDimChoice, plan.executionPlan, coreNum);
    if (!plan.tokenBlockChoice.enabled || plan.tokenBlockChoice.tokenBlockSize <= 0 ||
        plan.tokenBlockChoice.tokenBlockCnt <= 0 || plan.tokenBlockChoice.gridSize <= 0) {
        return {};
    }

    plan.tokenCoreMapping = BuildFnTokenCoreMappingChoice(plan.tokenBlockChoice.tokenBlockCnt,
                                                            plan.baseDimChoice.baseDimCnt, plan.executionPlan, coreNum);
    if (plan.tokenCoreMapping.tokenCoreBudget <= 0 || plan.tokenCoreMapping.blockDim <= 0) {
        return {};
    }
    if (plan.tokenCoreMapping.blockDim > static_cast<int64_t>(coreNum)) {
        plan.tokenCoreMapping.blockDim = static_cast<int64_t>(coreNum);
    }
    return plan;
}

} // namespace CausalConv1d

using namespace CausalConv1d;

// ============================================================
// Tiling entry
// ============================================================

void CausalConv1dTiling::SetTilingData(CausalConv1dTilingData &tiling,
                                         const CausalConv1dAttrInfo &attrInfo,
                                         const DimTileChoice &baseDimChoice, bool isFn,
                                         const FnHostPlan &fnHostPlan)
{
    tiling.activationMode = attrInfo.activationMode;
    tiling.padSlotId = attrInfo.padSlotId;
    tiling.dim = inputInfo_.dim;
    tiling.cuSeqlen = inputInfo_.cuSeqlen;
    tiling.seqLen = inputInfo_.seqLen;
    tiling.batch = inputInfo_.batch;
    tiling.kernelWidth = inputInfo_.kernelWidth;
    tiling.stateLen = inputInfo_.stateLen;
    tiling.numCacheLines = inputInfo_.numCacheLines;
    tiling.hasCacheIndices = inputInfo_.hasCacheIndices ? 1 : 0;
    tiling.hasInitialStateMode = inputInfo_.hasInitialStateMode ? 1 : 0;
    tiling.hasNumAcceptedTokens = inputInfo_.hasNumAcceptedTokens ? 1 : 0;
    tiling.hasBias = inputInfo_.hasBias ? 1 : 0;
    tiling.baseDim = baseDimChoice.baseDim;
    tiling.baseDimCnt = baseDimChoice.baseDimCnt;
    tiling.hasInitStateWorkspace = (isFn && inputInfo_.hasInitialStateMode) ? 1 : 0;
    if (isFn) {
        tiling.tokenBlockSize = fnHostPlan.tokenBlockChoice.tokenBlockSize;
        tiling.tokenBlockCnt = fnHostPlan.tokenBlockChoice.tokenBlockCnt;
        if (fnHostPlan.tokenSeqRangePlan.enabled) {
            tiling.hasExplicitTokenSeqRanges = 1;
            tiling.explicitTokenSeqRangeCount = fnHostPlan.tokenSeqRangePlan.rangeCount;
            for (int64_t i = 0; i < fnHostPlan.tokenSeqRangePlan.rangeCount; ++i) {
                tiling.tokenTileStartSeq[i] = fnHostPlan.tokenSeqRangePlan.tokenTileStartSeq[i];
                tiling.tokenTileEndSeq[i] = fnHostPlan.tokenSeqRangePlan.tokenTileEndSeq[i];
            }
        }
    }
}

ge::graphStatus CausalConv1dTiling::DoTiling()
{
    CausalConv1dTilingData *tiling = context_->GetTilingData<CausalConv1dTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, tiling);

    uint64_t ubSize = 0;
    uint32_t coreNum = 0;
    OP_CHECK_IF(GetPlatformInfo(ubSize, coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_, "GetPlatformInfo error"), return ge::GRAPH_FAILED);

    CausalConv1dAttrInfo attrInfo;
    OP_CHECK_IF(GetAttrsInfo(attrInfo) != ge::GRAPH_SUCCESS, OP_LOGE(context_, "GetAttrsInfo error"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(ParseCausalConv1dInputInfo(attrInfo) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_, "ParseCausalConv1dInputInfo error"), return ge::GRAPH_FAILED);

    const uint32_t inputModeKey = inputInfo_.inputModeKey;
    const bool isFn = IsFnInputMode(inputModeKey);
    OP_CHECK_IF(inputInfo_.dim <= 0 || inputInfo_.batch <= 0,
                OP_LOGE(context_, "dim/batch must be positive"), return ge::GRAPH_FAILED);

    const char *plannerModeTag = "update";
    DimTileChoice baseDimChoice;
    FnExecutionPlan fnExecutionPlan = FN_EXECUTION_PLAN_INVALID;
    FnHostPlan fnHostPlan;
    const int64_t *qslData = nullptr;
    if (isFn && (inputModeKey == CAUSAL_CONV1D_TPL_INPUT_MODE_FN_VARLEN)) {
        const gert::Tensor *qslTensor = context_->GetOptionalInputTensor(CausalConv1d::QUERY_START_LOC_INDEX);
        qslData = (qslTensor != nullptr) ? qslTensor->GetData<int64_t>() : nullptr;
    }

    if (isFn) {
        fnHostPlan = ChooseFnHostPlan(ubSize, coreNum);
        plannerModeTag = GetFnTilingCaseName(fnHostPlan.caseKind);
        baseDimChoice = fnHostPlan.baseDimChoice;
        fnExecutionPlan = fnHostPlan.executionPlan;
    } else {
        baseDimChoice = ChooseCanonicalUpdateBaseDimChoice(inputInfo_.batch, inputInfo_.dim, coreNum);
    }

    OP_CHECK_IF(baseDimChoice.baseDim <= 0 || baseDimChoice.baseDimCnt <= 0 || baseDimChoice.gridSize <= 0,
                OP_LOGE(context_, "invalid dim tile size selection"), return ge::GRAPH_FAILED);

    int64_t effectiveGridSize = baseDimChoice.gridSize;

    if (isFn) {
        OP_CHECK_IF(fnHostPlan.caseKind == FN_TILING_CASE_INVALID || fnExecutionPlan == FN_EXECUTION_PLAN_INVALID ||
                        !fnHostPlan.tokenBlockChoice.enabled || fnHostPlan.tokenBlockChoice.tokenBlockSize <= 0 ||
                        fnHostPlan.tokenBlockChoice.tokenBlockCnt <= 0 || fnHostPlan.tokenBlockChoice.gridSize <= 0 ||
                        fnHostPlan.tokenCoreMapping.tokenCoreBudget <= 0 || fnHostPlan.tokenCoreMapping.blockDim <= 0,
                    OP_LOGE(context_, "run_mode=0 must resolve a valid unified token tiling plan"),
                    return ge::GRAPH_FAILED);

        effectiveGridSize = fnHostPlan.tokenBlockChoice.gridSize;
        if (inputModeKey == CAUSAL_CONV1D_TPL_INPUT_MODE_FN_VARLEN) {
            fnHostPlan.tokenSeqRangePlan =
                BuildFnTokenSeqRangePlan(qslData, inputInfo_.batch,
                                         fnHostPlan.tokenBlockChoice.tokenBlockSize,
                                         fnHostPlan.tokenBlockChoice.tokenBlockCnt);
            if (!fnHostPlan.tokenSeqRangePlan.enabled &&
                qslData != nullptr &&
                fnHostPlan.tokenBlockChoice.tokenBlockCnt > MAX_FN_TOKEN_SEQ_RANGE_COUNT) {
                OP_LOGD(context_,
                        "FnTokenSeqRanges disabled: tokenBlockCnt[%ld] exceeds fixed tiling capacity[%ld].",
                        fnHostPlan.tokenBlockChoice.tokenBlockCnt, MAX_FN_TOKEN_SEQ_RANGE_COUNT);
            }
        }
        OP_LOGD(context_,
                "FnHostPlan(case=%s): inputModeKey[%u], dim[%ld], cuSeqlen[%ld], baseDim[%ld], baseDimCnt[%ld], "
                "tokenCoreBudget[%ld], tokenBlockSize[%ld], tokenBlockCnt[%ld], tokenBlocksPerCore[%ld], "
                "tokenCoreTailCnt[%ld], explicitSeqRanges[%ld], baseGrid[%ld], phase1Grid[%ld], mappedBlockDim[%ld].",
                plannerModeTag, inputModeKey, inputInfo_.dim, inputInfo_.cuSeqlen, baseDimChoice.baseDim,
                baseDimChoice.baseDimCnt, fnHostPlan.tokenCoreMapping.tokenCoreBudget,
                fnHostPlan.tokenBlockChoice.tokenBlockSize, fnHostPlan.tokenBlockChoice.tokenBlockCnt,
                fnHostPlan.tokenCoreMapping.tokenBlocksPerCore, fnHostPlan.tokenCoreMapping.tokenCoreTailCnt,
                static_cast<int64_t>(fnHostPlan.tokenSeqRangePlan.enabled),
                baseDimChoice.gridSize, fnHostPlan.tokenBlockChoice.gridSize, fnHostPlan.tokenCoreMapping.blockDim);
    }

    uint32_t blockDim =
        (effectiveGridSize < static_cast<int64_t>(coreNum)) ? static_cast<uint32_t>(effectiveGridSize) : coreNum;
    if (isFn) {
        const int64_t mappedBlockDim = std::min<int64_t>(effectiveGridSize, fnHostPlan.tokenCoreMapping.blockDim);
        OP_CHECK_IF(mappedBlockDim <= 0, OP_LOGE(context_, "invalid mapped blockDim for run_mode=0"),
                    return ge::GRAPH_FAILED);
        blockDim = static_cast<uint32_t>(mappedBlockDim);
    }

    const uint32_t fnPlanKey = NormalizeFnPlanTilingKey(inputModeKey, fnExecutionPlan);
    const uint32_t widthKey = NormalizeWidthTilingKey(inputModeKey, static_cast<int32_t>(inputInfo_.kernelWidth));
    if (isFn && inputInfo_.hasInitialStateMode) {
        constexpr int64_t kDtypeSize = 2;
        const int64_t snapshotWorkspaceSize = inputInfo_.numCacheLines * inputInfo_.stateLen *
                                              inputInfo_.dim * kDtypeSize;
        const int64_t workspaceSize = ASCENDC_RESERVED_WORKSPACE_SIZE + snapshotWorkspaceSize;
        OP_CHECK_IF(SetWorkspaceSize(static_cast<size_t>(workspaceSize)) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context_, "SetWorkspaceSize error"), return ge::GRAPH_FAILED);
        OP_CHECK_IF(context_->SetScheduleMode(1) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context_, "SetScheduleMode(1) error"), return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(SetWorkspaceSize(0) != ge::GRAPH_SUCCESS, OP_LOGE(context_, "SetWorkspaceSize error"),
                    return ge::GRAPH_FAILED);
    }

    SetTilingData(*tiling, attrInfo, baseDimChoice, isFn, fnHostPlan);

    OP_LOGD(context_,
            "Tiling result: mode[%s], batch[%ld], dim[%ld], baseDim[%ld], baseDimCnt[%ld], gridSize[%ld], "
            "effectiveGrid[%ld], blockDim[%u], coreNum[%u], tokenTiling[%ld,%ld], hasActivation[%d], hasBias[%d], "
            "fnPlan[%ld].",
            plannerModeTag, inputInfo_.batch, inputInfo_.dim, baseDimChoice.baseDim, baseDimChoice.baseDimCnt,
            baseDimChoice.gridSize, effectiveGridSize, blockDim, coreNum,
            fnHostPlan.tokenBlockChoice.tokenBlockSize, fnHostPlan.tokenBlockChoice.tokenBlockCnt,
            static_cast<int32_t>(attrInfo.activationMode != 0), static_cast<int32_t>(tiling->hasBias),
            static_cast<int64_t>(fnExecutionPlan));

    context_->SetBlockDim(blockDim);
    const uint64_t tilingKey = GET_TPL_TILING_KEY(inputModeKey, widthKey, fnPlanKey);
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
