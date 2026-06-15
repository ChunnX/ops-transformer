/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file inplace_partial_rotary_mul_grad_tiling.cpp
 * \brief Tiling implementation for InplacePartialRotaryMulGrad (arch35 / ascend950).
 */

#include "inplace_partial_rotary_mul_grad_tiling.h"
#include <vector>
#include <algorithm>
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "graph/utils/type_utils.h"

namespace {
constexpr uint32_t ROTARY_MODE_ATTR_IDX = 0;
constexpr uint32_t PARTIAL_SLICE_ATTR_IDX = 1;

constexpr int64_t INPUT_DY_INDEX = 0;
constexpr int64_t INPUT_COS_INDEX = 1;
constexpr int64_t INPUT_SIN_INDEX = 2;
constexpr int64_t INPUT_X_INDEX = 3;

constexpr int64_t DIM_NUM = 4;
constexpr int64_t D_LIMIT = 1024;
constexpr int64_t BLOCK_SIZE_BYTES = 32;
constexpr int64_t RESERVE_UB_SIZE = 16 * 1024;
constexpr int64_t MAX_COPY_BLOCK_COUNT = 4095;

constexpr int64_t MODE_HALF = 0;
constexpr int64_t MODE_INTERLEAVE = 1;
constexpr int64_t MODE_QUARTER = 2;
constexpr int64_t MODE_INTERLEAVE_HALF = 3;

const std::vector<ge::DataType> SUPPORT_DTYPE = {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16};

int64_t CeilDiv(int64_t a, int64_t b)
{
    return (a == 0) ? 0 : (a + b - 1) / b;
}

int64_t CeilAlign(int64_t a, int64_t b)
{
    return CeilDiv(a, b) * b;
}

int64_t DSplitCoef(int64_t mode)
{
    if (mode == MODE_QUARTER) {
        return 4;
    }
    if (mode == MODE_HALF || mode == MODE_INTERLEAVE_HALF) {
        return 2;
    }
    return 1; // interleave
}
} // namespace

namespace optiling {

int64_t InplacePartialRotaryMulGradTilingClass::DTypeCode(ge::DataType dt) const
{
    if (dt == ge::DT_FLOAT) {
        return INPLACE_PARTIAL_ROTARY_DTYPE_FP32;
    }
    if (dt == ge::DT_FLOAT16) {
        return INPLACE_PARTIAL_ROTARY_DTYPE_FP16;
    }
    return INPLACE_PARTIAL_ROTARY_DTYPE_BF16; // bfloat16
}

ge::graphStatus InplacePartialRotaryMulGradTilingClass::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo != nullptr) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        aicoreParams_.numBlocks = ascendcPlatform.GetCoreNumAiv();
        uint64_t ubSizePlatForm;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
        aicoreParams_.ubSize = ubSizePlatForm;
        socVersion_ = ascendcPlatform.GetSocVersion();
    } else {
        auto compileInfoPtr = context_->GetCompileInfo<InplacePartialRotaryMulGradCompileInfo>();
        if (compileInfoPtr == nullptr) {
            return ge::GRAPH_FAILED;
        }
        aicoreParams_.numBlocks = compileInfoPtr->numBlocks;
        aicoreParams_.ubSize = compileInfoPtr->ubSize;
        socVersion_ = compileInfoPtr->socVersion;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InplacePartialRotaryMulGradTilingClass::CheckParam()
{
    auto dyShapePtr = context_->GetInputShape(INPUT_DY_INDEX);
    auto cosShapePtr = context_->GetInputShape(INPUT_COS_INDEX);
    auto sinShapePtr = context_->GetInputShape(INPUT_SIN_INDEX);
    if (dyShapePtr == nullptr || cosShapePtr == nullptr || sinShapePtr == nullptr) {
        OP_LOGE(context_->GetNodeName(), "input shape is nullptr.");
        return ge::GRAPH_FAILED;
    }
    const auto& dyShape = dyShapePtr->GetStorageShape();
    const auto& cosShape = cosShapePtr->GetStorageShape();
    const auto& sinShape = sinShapePtr->GetStorageShape();

    if (dyShape.GetDimNum() != DIM_NUM) {
        OP_LOGE(context_->GetNodeName(), "dy must be 4D (BSND), got dimNum=%zu.", dyShape.GetDimNum());
        return ge::GRAPH_FAILED;
    }
    if (cosShape.GetDimNum() != DIM_NUM || sinShape.GetDimNum() != DIM_NUM) {
        OP_LOGE(context_->GetNodeName(), "cos/sin must be 4D (BSND).");
        return ge::GRAPH_FAILED;
    }
    if (cosShape != sinShape) {
        OP_LOGE(context_->GetNodeName(), "cos and sin must have the same shape.");
        return ge::GRAPH_FAILED;
    }

    // Only the x == nullptr scenario is supported.
    if (context_->GetInputShape(INPUT_X_INDEX) != nullptr) {
        OP_LOGE(context_->GetNodeName(),
            "only the x == nullptr scenario is supported (dcos/dsin are not computed).");
        return ge::GRAPH_FAILED;
    }

    auto dyDesc = context_->GetInputDesc(INPUT_DY_INDEX);
    auto cosDesc = context_->GetInputDesc(INPUT_COS_INDEX);
    if (dyDesc == nullptr || cosDesc == nullptr) {
        OP_LOGE(context_->GetNodeName(), "input desc is nullptr.");
        return ge::GRAPH_FAILED;
    }
    ge::DataType dyDtype = dyDesc->GetDataType();
    ge::DataType cosDtype = cosDesc->GetDataType();
    if (std::find(SUPPORT_DTYPE.begin(), SUPPORT_DTYPE.end(), dyDtype) == SUPPORT_DTYPE.end() ||
        std::find(SUPPORT_DTYPE.begin(), SUPPORT_DTYPE.end(), cosDtype) == SUPPORT_DTYPE.end()) {
        OP_LOGE(context_->GetNodeName(), "dy/cos dtype must be float16/float32/bfloat16.");
        return ge::GRAPH_FAILED;
    }
    // cos and sin share dtype.
    if (context_->GetInputDesc(INPUT_SIN_INDEX)->GetDataType() != cosDtype) {
        OP_LOGE(context_->GetNodeName(), "cos and sin must have the same dtype.");
        return ge::GRAPH_FAILED;
    }

    if (rotaryMode_ < MODE_HALF || rotaryMode_ > MODE_INTERLEAVE_HALF) {
        OP_LOGE(context_->GetNodeName(), "rotary_mode must be in [0, 3], got %ld.", rotaryMode_);
        return ge::GRAPH_FAILED;
    }

    b_ = dyShape.GetDim(0);
    s_ = dyShape.GetDim(1);
    n_ = dyShape.GetDim(2);
    d_ = dyShape.GetDim(3);
    if (d_ > D_LIMIT) {
        OP_LOGE(context_->GetNodeName(), "D must be <= %ld.", D_LIMIT);
        return ge::GRAPH_FAILED;
    }
    if (partialStart_ < 0 || partialStart_ > d_ || sliceLen_ < 0 || partialStart_ + sliceLen_ > d_) {
        OP_LOGE(context_->GetNodeName(),
            "partial_slice [%ld, %ld) out of range [0, %ld].", partialStart_, partialStart_ + sliceLen_, d_);
        return ge::GRAPH_FAILED;
    }
    if (sliceLen_ > 0) {
        // half / interleave / interleave-half require sliceLen % 2 == 0; quarter requires % 4.
        // (coef is only used for the UB layout; the divisibility requirement follows the docs.)
        int64_t alignReq = (rotaryMode_ == MODE_QUARTER) ? 4 : 2;
        if (sliceLen_ % alignReq != 0) {
            OP_LOGE(context_->GetNodeName(), "slice length %ld must be divisible by %ld for rotary_mode %ld.",
                sliceLen_, alignReq, rotaryMode_);
            return ge::GRAPH_FAILED;
        }
        // interleave (mode 1) CopyIn/CopyOut use InterleaveMode helpers with dstStride=0 /
        // srcStride=0 in UB, requiring the per-row UB stride (dAlignDy / dAlignCs / dAlignOut)
        // to exactly equal sliceLen (no padding between rows).
        if (rotaryMode_ == MODE_INTERLEAVE) {
            int64_t szDy = static_cast<int64_t>(ge::GetSizeByDataType(dyDtype));
            int64_t szCs = static_cast<int64_t>(ge::GetSizeByDataType(cosDtype));
            int64_t blockElemsDy = BLOCK_SIZE_BYTES / szDy;
            int64_t blockElemsCs = BLOCK_SIZE_BYTES / szCs;
            if (sliceLen_ % blockElemsDy != 0) {
                OP_LOGE(context_->GetNodeName(),
                    "interleave mode (1) requires sliceLen (%ld) to be a multiple of %ld "
                    "(block-aligned for dy dtype size=%ld).",
                    sliceLen_, blockElemsDy, szDy);
                return ge::GRAPH_FAILED;
            }
            if (sliceLen_ % blockElemsCs != 0) {
                OP_LOGE(context_->GetNodeName(),
                    "interleave mode (1) requires sliceLen (%ld) to be a multiple of %ld "
                    "(block-aligned for cos/sin dtype size=%ld).",
                    sliceLen_, blockElemsCs, szCs);
                return ge::GRAPH_FAILED;
            }
        }
        // interleave-half (mode 3) CopyOut uses DataCopy with srcStride=0 (via CopyOutInterleaveMode);
        // when sliceLen is not block-aligned, dAlignOut > sliceLen leaves UB row padding that srcStride=0
        // cannot skip, causing garbage bytes between rows to overwrite valid GM data.
        if (rotaryMode_ == MODE_INTERLEAVE_HALF) {
            int64_t szDy = static_cast<int64_t>(ge::GetSizeByDataType(dyDtype));
            int64_t blockElemsDy = BLOCK_SIZE_BYTES / szDy;
            if (sliceLen_ % blockElemsDy != 0) {
                OP_LOGE(context_->GetNodeName(),
                    "interleave-half mode (3) requires sliceLen (%ld) to be a multiple of %ld "
                    "(block-aligned for dy dtype size=%ld).",
                    sliceLen_, blockElemsDy, szDy);
                return ge::GRAPH_FAILED;
            }
        }
    }

    cosB_ = cosShape.GetDim(0);
    cosS_ = cosShape.GetDim(1);
    cosN_ = cosShape.GetDim(2);
    int64_t cosD = cosShape.GetDim(3);
    if (cosD != sliceLen_) {
        OP_LOGE(context_->GetNodeName(), "cos last dim %ld must equal slice length %ld.", cosD, sliceLen_);
        return ge::GRAPH_FAILED;
    }
    auto checkAxis = [](int64_t cosDim, int64_t dyDim, const char* name) {
        return cosDim == 1 || cosDim == dyDim;
    };
    if (!checkAxis(cosB_, b_, "B") || !checkAxis(cosS_, s_, "S") || !checkAxis(cosN_, n_, "N")) {
        OP_LOGE(context_->GetNodeName(),
            "cos B/S/N must each be 1 or equal to dy's B/S/N (BSND broadcast).");
        return ge::GRAPH_FAILED;
    }

    dyDtypeCode_ = DTypeCode(dyDtype);
    cosDtypeCode_ = DTypeCode(cosDtype);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InplacePartialRotaryMulGradTilingClass::GetShapeAttrsInfo()
{
    auto attrs = context_->GetAttrs();
    if (attrs == nullptr) {
        OP_LOGE(context_->GetNodeName(), "attrs is nullptr.");
        return ge::GRAPH_FAILED;
    }
    auto modePtr = attrs->GetAttrPointer<int64_t>(ROTARY_MODE_ATTR_IDX);
    rotaryMode_ = (modePtr == nullptr) ? MODE_HALF : *modePtr;

    auto partialSlicePtr = attrs->GetAttrPointer<int64_t>(PARTIAL_SLICE_ATTR_IDX);
    if (partialSlicePtr != nullptr) {
        partialStart_ = partialSlicePtr[0];
        sliceLen_ = partialSlicePtr[1] - partialSlicePtr[0];
    } else {
        partialStart_ = 0;
        sliceLen_ = 0;
    }

    auto ret = CheckParam();
    if (ret != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // Broadcast strides over the (B,S,N) grid, in units of sliceLen-element rows.
    cosStrideB_ = (cosB_ == 1) ? 0 : (cosS_ * cosN_);
    cosStrideS_ = (cosS_ == 1) ? 0 : cosN_;
    cosStrideN_ = (cosN_ == 1) ? 0 : 1;
    isFullBroadcast_ = (cosStrideB_ == 0 && cosStrideS_ == 0 && cosStrideN_ == 0) ? 1 : 0;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InplacePartialRotaryMulGradTilingClass::InitTilingData()
{
    if (tilingData_ == nullptr) {
        tilingData_ = context_->GetTilingData<InplacePartialRotaryMulGradTilingData>();
        if (tilingData_ == nullptr) {
            OP_LOGE(context_->GetNodeName(), "get tilingdata ptr failed");
            return ge::GRAPH_FAILED;
        }
    }
    if (memset_s(tilingData_, sizeof(InplacePartialRotaryMulGradTilingData), 0,
        sizeof(InplacePartialRotaryMulGradTilingData)) != EOK) {
        OP_LOGE(context_->GetNodeName(), "memset tilingdata failed");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InplacePartialRotaryMulGradTilingClass::DoOpTiling()
{
    if (InitTilingData() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    int64_t mTotal = b_ * s_ * n_;
    int64_t numAiv = aicoreParams_.numBlocks > 0 ? aicoreParams_.numBlocks : 1;
    int64_t usedCoreNum = std::min(numAiv, mTotal > 0 ? mTotal : 1);
    usedCoreNum = std::max<int64_t>(usedCoreNum, 1);
    int64_t blockFactorM = CeilDiv(mTotal, usedCoreNum);
    int64_t blockTailM = (usedCoreNum == 1) ? mTotal : (mTotal - (usedCoreNum - 1) * blockFactorM);
    if (blockTailM <= 0) {
        // more cores than rows: shrink usedCoreNum to CeilDiv and recompute
        usedCoreNum = std::max<int64_t>(CeilDiv(mTotal, blockFactorM > 0 ? blockFactorM : 1), 1);
        blockFactorM = CeilDiv(mTotal, usedCoreNum);
        blockTailM = (usedCoreNum == 1) ? mTotal : (mTotal - (usedCoreNum - 1) * blockFactorM);
    }

    int64_t ubFactorM = 1;
    if (sliceLen_ > 0) {
        int64_t coef = DSplitCoef(rotaryMode_);
        int64_t szDy = static_cast<int64_t>(ge::GetSizeByDataType(
            context_->GetInputDesc(INPUT_DY_INDEX)->GetDataType()));
        int64_t szCs = static_cast<int64_t>(ge::GetSizeByDataType(
            context_->GetInputDesc(INPUT_COS_INDEX)->GetDataType()));
        int64_t blockElemsDy = BLOCK_SIZE_BYTES / szDy;
        int64_t blockElemsCs = BLOCK_SIZE_BYTES / szCs;
        int64_t dAlignDy = CeilAlign(sliceLen_ / coef, blockElemsDy) * coef;
        int64_t dAlignCs = CeilAlign(sliceLen_ / coef, blockElemsCs) * coef;
        int64_t dAlignOut = (rotaryMode_ == MODE_INTERLEAVE_HALF) ? CeilAlign(sliceLen_, blockElemsDy) : dAlignDy;
        // dy-in + dy-out + cos + sin, all resident per tile.
        int64_t perTileRowBytes = dAlignDy * szDy + dAlignOut * szDy + dAlignCs * szCs + dAlignCs * szCs;
        int64_t availableUb = static_cast<int64_t>(aicoreParams_.ubSize) - RESERVE_UB_SIZE;
        // interleave (mode 1) VF reads paired even/odd VL chunks per row; the odd-VL read
        // on the last tile row can land past the nominal buffer end (single-buffered).
        // Reserve 2*VL_FLOAT32_SIZE elements per buffer as trailing pad (dy_in+dy_out+cos+sin).
        if (rotaryMode_ == MODE_INTERLEAVE) {
            constexpr int64_t VL_PAD_ELEMS = 128;  // 2 * VL_FLOAT32_SIZE
            int64_t padBytes = VL_PAD_ELEMS * (2 * szDy + 2 * szCs);
            availableUb -= padBytes;
        }
        if (perTileRowBytes > availableUb) {
            OP_LOGE(context_->GetNodeName(),
                "UB size insufficient for one tile row: need %ld bytes, available %ld bytes "
                "(ubSize=%lu, D=%ld, sliceLen=%ld, mode=%ld).",
                perTileRowBytes, availableUb, aicoreParams_.ubSize, d_, sliceLen_, rotaryMode_);
            return ge::GRAPH_FAILED;
        }
        ubFactorM = availableUb / perTileRowBytes;
        ubFactorM = std::min(ubFactorM, blockFactorM);
        ubFactorM = std::min(ubFactorM, MAX_COPY_BLOCK_COUNT);
        ubFactorM = std::max<int64_t>(ubFactorM, 1);
    }

    tilingData_->b = b_;
    tilingData_->s = s_;
    tilingData_->n = n_;
    tilingData_->d = d_;
    tilingData_->partialStart = partialStart_;
    tilingData_->sliceLen = sliceLen_;
    tilingData_->rotaryMode = rotaryMode_;
    tilingData_->dyDtypeCode = dyDtypeCode_;
    tilingData_->cosDtypeCode = cosDtypeCode_;
    tilingData_->cosStrideB = cosStrideB_;
    tilingData_->cosStrideS = cosStrideS_;
    tilingData_->cosStrideN = cosStrideN_;
    tilingData_->usedCoreNum = usedCoreNum;
    tilingData_->blockFactorM = blockFactorM;
    tilingData_->ubFactorM = ubFactorM;
    tilingData_->isFullBroadcast = isFullBroadcast_;

    context_->SetBlockDim(usedCoreNum);

    OP_LOGD(context_->GetNodeName(),
        "InplacePartialRotaryMulGrad tiling: mode=%ld slice=[%ld,%ld) M=%ld usedCoreNum=%ld blockFactorM=%ld "
        "ubFactorM=%ld isFullBroadcast=%ld dyDtype=%ld cosDtype=%ld",
        rotaryMode_, partialStart_, partialStart_ + sliceLen_, mTotal, usedCoreNum, blockFactorM, ubFactorM,
        isFullBroadcast_, dyDtypeCode_, cosDtypeCode_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Tiling4InplacePartialRotaryMulGrad(gert::TilingContext* context)
{
    return Ops::Transformer::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

ge::graphStatus TilingPrepareForInplacePartialRotaryMulGrad(gert::TilingParseContext* context)
{
    auto platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto compileInfoPtr = context->GetCompiledInfo<InplacePartialRotaryMulGradCompileInfo>();
    if (compileInfoPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfoPtr->numBlocks = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfoPtr->ubSize = ubSizePlatForm;
    compileInfoPtr->socVersion = ascendcPlatform.GetSocVersion();
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(InplacePartialRotaryMulGrad)
    .Tiling(Tiling4InplacePartialRotaryMulGrad)
    .TilingParse<InplacePartialRotaryMulGradCompileInfo>(TilingPrepareForInplacePartialRotaryMulGrad);
} // namespace optiling
