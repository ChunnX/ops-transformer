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
 * \file mmhc_pre_sinkhorn_sinkhorn_tiling.cpp
 * \brief MhcPreSinkhorn tiling implementation
 */

#include <sstream>
#include "mhc_pre_sinkhorn_tiling.h"

using namespace ge;
namespace optiling {
namespace {
constexpr uint64_t WORKSPACE_SIZE = 32;
int64_t CeilDiv(int64_t x, int64_t y)
{
    if (y != 0) {
        return (x + y - 1) / y;
    }
    return x;
}
int64_t DownAlign(int64_t x, int64_t y) {
    if (y == 0) {
        return x;
    }
    return (x / y) * y;
}
int64_t RoundUp(int64_t x, int64_t y) {
    return CeilDiv(x, y) * y;
}

constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t REPEAT_SIZE = 256;
constexpr int64_t UB_RESEVED_SIZE = 8192;
constexpr int64_t DOUBLE_BUFFER = 2;
constexpr uint64_t M_L1_MAX_SIZE = 256;
constexpr uint64_t K_MULIT_CORE_SPLIT_BASE_SIZE = 256;
constexpr uint64_t A_L1_SIZE = 128 * 256;
constexpr uint64_t K_L1_MAX_SIZE = 1024;
constexpr int64_t HC_MULT_ATTR_IDX = 0;
constexpr int64_t ITER_TIMES_ATTR_IDX = 1;
constexpr int64_t HC_EPS_ATTR_IDX = 2;
constexpr int64_t NORM_EPS_ATTR_IDX = 3;
constexpr int64_t NEED_BACKWARD_ATTR_IDX = 4;
constexpr int64_t DEFAULT_ITER_TIMES = 20;
}

ge::graphStatus MhcPreSinkhornTiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        auto compileInfoPtr = context_->GetCompileInfo<MhcPreSinkhornCompileInfo>();
        OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context_, "compile info is null"),
                      return ge::GRAPH_FAILED);
        aivCoreNum_ = compileInfoPtr->coreNum;
        ubSize_ = compileInfoPtr->ubSize;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        aivCoreNum_ = ascendcPlatform.GetCoreNumAiv();
        aicCoreNum_ = ascendcPlatform.GetCoreNumAic();
        uint64_t ubSizePlatForm;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
        ubSize_ = ubSizePlatForm;
        socVersion_ = ascendcPlatform.GetSocVersion();
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPreSinkhornTiling::GetAttr()
{
    auto* attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    auto hcMultAttr = attrs->GetAttrPointer<int64_t>(HC_MULT_ATTR_IDX);
    hcMult_ = hcMultAttr == nullptr ? 4 : *hcMultAttr;

    auto iterTimesAttr = attrs->GetAttrPointer<int64_t>(ITER_TIMES_ATTR_IDX);
    iterTimes_ = iterTimesAttr == nullptr ? DEFAULT_ITER_TIMES : *iterTimesAttr;

    auto hcEpsAttr = attrs->GetAttrPointer<float>(HC_EPS_ATTR_IDX);
    hcEps_ = hcEpsAttr == nullptr ? 1e-6f : *hcEpsAttr;

    auto normEpsAttr = attrs->GetAttrPointer<float>(NORM_EPS_ATTR_IDX);
    normEps_ = normEpsAttr == nullptr ? 1e-6f : *normEpsAttr;

    auto needBackwardAttr = attrs->GetAttrPointer<bool>(NEED_BACKWARD_ATTR_IDX);
    needGrad_ = needBackwardAttr == nullptr ? true : *needBackwardAttr;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPreSinkhornTiling::GetShapeAttrsInfoInner()
{
    auto xShape = context_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShape);
    size_t xDimNum = xShape->GetStorageShape().GetDimNum();
    if (xDimNum == 3) {
        bs_ = xShape->GetStorageShape().GetDim(0);
        hcMult_ = xShape->GetStorageShape().GetDim(1);
        d_ = xShape->GetStorageShape().GetDim(2);
    } else if (xDimNum == 4) {
        int64_t b = xShape->GetStorageShape().GetDim(0);
        int64_t s = xShape->GetStorageShape().GetDim(1);
        bs_ = b * s;
        hcMult_ = xShape->GetStorageShape().GetDim(2);
        d_ = xShape->GetStorageShape().GetDim(3);
    }

    auto shapePhi = context_->GetInputShape(1);
    hcMix_ = shapePhi->GetStorageShape().GetDim(0);
    OP_CHECK_IF(shapePhi->GetStorageShape().GetDim(1) != d_ * hcMult_,
                    OP_LOGE(context_->GetNodeName(),
                             "Phi dim 1 should be equal with d_ * hcMult_  %ld, but is %ld", d_ * hcMult_, shapePhi->GetStorageShape().GetDim(1)),
                    return ge::GRAPH_FAILED);

    auto shapeAlpha = context_->GetInputShape(2);
    int64_t scaleFirstDim = shapeAlpha->GetStorageShape().GetDim(0);
    OP_CHECK_IF(scaleFirstDim != 3,
                    OP_LOGE(context_->GetNodeName(),
                             "alpha size should be equal with 3, but is %ld", scaleFirstDim),
                    return ge::GRAPH_FAILED);

    auto shapeBias = context_->GetInputShape(3);
    int64_t baseFirstDim = shapeBias->GetStorageShape().GetDim(0);
    OP_CHECK_IF(baseFirstDim != hcMix_,
                    OP_LOGE(context_->GetNodeName(),
                             "bias size should be equal with mixhc, but is %ld", baseFirstDim),
                    return ge::GRAPH_FAILED);

    OP_CHECK_IF(GetAttr() != ge::GRAPH_SUCCESS,
                  OP_LOGE(context_->GetNodeName(), "get attr failed."),
                  return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}


ge::graphStatus MhcPreSinkhornTiling::CalcMKSplitCoreMembasePart2Tiling() 
{
    rowOfFormerBlock_ = CeilDiv(bs_, static_cast<int64_t>(aivCoreNum_));
    usedAivCoreNums_ = std::min(CeilDiv(bs_, rowOfFormerBlock_), static_cast<int64_t>(aivCoreNum_));
    rowOfTailBlock_ = bs_ - (usedAivCoreNums_ - 1) * rowOfFormerBlock_;

    int64_t minRowPerCore = 1;
    int64_t rowOnceLoop = std::min(rowOfFormerBlock_, minRowPerCore);
    int64_t kBlockNum = tilingData_.get_cubeBlockDimK();

    hcMultAlign_ = RoundUp(hcMult_, BLOCK_SIZE / sizeof(float));
    int64_t mix0OriginSize = kBlockNum * rowOnceLoop * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER;     
    int64_t mix1OriginSize = kBlockNum * rowOnceLoop * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER; 
    int64_t mix2OriginSize = kBlockNum * rowOnceLoop * hcMult_ * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER; 
    int64_t mix0Size = rowOnceLoop * hcMultAlign_ * sizeof(float);     
    int64_t mix1Size = rowOnceLoop * hcMultAlign_ * sizeof(float); 
    int64_t mix2Size = rowOnceLoop * hcMult_ * hcMultAlign_ * sizeof(float); 
    int64_t squareSumSize = kBlockNum * RoundUp(rowOnceLoop * 16, 16) * sizeof(float) * DOUBLE_BUFFER;
    int64_t rsqrtSize = RoundUp(rowOnceLoop, BLOCK_SIZE / sizeof(float)) * sizeof(float) * DOUBLE_BUFFER;
    int64_t xSize = rowOnceLoop * hcMult_ * RoundUp(d_, 16) * 2 * DOUBLE_BUFFER;
    int64_t ySize = rowOnceLoop * RoundUp(d_, 16) * 2 * DOUBLE_BUFFER;
    int64_t postSize = rowOnceLoop * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER;
    int64_t combFragSize = rowOnceLoop * hcMult_ * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER;
    int64_t base0Size = hcMultAlign_ * sizeof(float);
    int64_t base1Size = hcMultAlign_ * sizeof(float);
    int64_t base2Size = hcMult_ * hcMultAlign_ * sizeof(float);
    int64_t xCastSize = rowOnceLoop * hcMult_ * RoundUp(d_, 8) * sizeof(float);
    int64_t yCastSize = rowOnceLoop * RoundUp(d_, 8) * sizeof(float);
    int64_t rowBrcb0Size = RoundUp(rowOnceLoop, 8) * BLOCK_SIZE;
    int64_t hcBrcb1Size = RoundUp(rowOnceLoop * hcMultAlign_, 8) * BLOCK_SIZE;
    int64_t reduceBufSize = rowOnceLoop * hcMultAlign_ * sizeof(float);
    int64_t maskPatternSize = BLOCK_SIZE * 16;

    int64_t totalSize = mix0OriginSize + mix1OriginSize + mix2OriginSize + mix0Size + mix1Size + mix2Size + squareSumSize + rsqrtSize + 
                        xSize + ySize + postSize + combFragSize + base0Size + base1Size + base2Size + xCastSize + yCastSize + 
                        rowBrcb0Size + hcBrcb1Size + reduceBufSize + maskPatternSize;
    rowFactor_ = rowOnceLoop; 
    if (totalSize <= ubSize_) {
        dLoop_ = 1;
        dFactor_ = d_;
        tailDFactor_ = dFactor_;
    } else {
        int64_t usedUbSize = mix0OriginSize + mix1OriginSize + mix2OriginSize + mix0Size + mix1Size + mix2Size + squareSumSize + rsqrtSize
                             + postSize + combFragSize + base0Size + base1Size + base2Size + rowBrcb0Size + hcBrcb1Size + reduceBufSize + maskPatternSize;
        int64_t ubRemain = ubSize_ - usedUbSize;
        dFactor_ = d_;
        int64_t base = 2;
        while (1) {
            dFactor_ = CeilDiv(d_, base);
            xSize = rowOnceLoop * hcMult_ * RoundUp(dFactor_, 16) * 2 * DOUBLE_BUFFER;
            ySize = rowOnceLoop * RoundUp(dFactor_, 16) * 2 * DOUBLE_BUFFER;
            xCastSize = rowOnceLoop * hcMult_ * RoundUp(dFactor_, 8) * sizeof(float);
            yCastSize = rowOnceLoop * RoundUp(dFactor_, 8) * sizeof(float);
            int64_t targetSize = xSize + ySize + xCastSize + yCastSize;
            if (targetSize <= ubRemain) {
                break;
            }
            base++;
        }
        if (dFactor_ > 32) {
            dFactor_ = DownAlign(dFactor_, 32);
        }
        dLoop_ = CeilDiv(d_, dFactor_);
        tailDFactor_ = d_ % dFactor_ == 0 ? dFactor_ : d_ % dFactor_;
    }

    if (dFactor_ == d_) {
        while (rowFactor_ <= rowOfFormerBlock_) {
            mix0OriginSize = kBlockNum * rowFactor_ * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER;     
            mix1OriginSize = kBlockNum * rowFactor_ * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER; 
            mix2OriginSize = kBlockNum * rowFactor_ * hcMult_ * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER; 
            mix0Size = rowFactor_ * hcMultAlign_ * sizeof(float);     
            mix1Size = rowFactor_ * hcMultAlign_ * sizeof(float); 
            mix2Size = rowFactor_ * hcMult_ * hcMultAlign_ * sizeof(float); 
            squareSumSize = kBlockNum * RoundUp(rowFactor_ * 16, 16) * sizeof(float) * DOUBLE_BUFFER;
            rsqrtSize = RoundUp(rowFactor_, BLOCK_SIZE / sizeof(float)) * sizeof(float) * DOUBLE_BUFFER;
            xSize = rowFactor_ * hcMult_ * RoundUp(d_, 16) * 2 * DOUBLE_BUFFER;
            ySize = rowFactor_ * RoundUp(d_, 16) * 2 * DOUBLE_BUFFER;
            postSize = rowFactor_ * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER;
            combFragSize = rowFactor_ * hcMult_ * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER;
            xCastSize = rowFactor_ * hcMult_ * RoundUp(d_, 8) * sizeof(float);
            yCastSize = rowFactor_ * RoundUp(d_, 8) * sizeof(float);
            rowBrcb0Size = RoundUp(rowFactor_, 8) * BLOCK_SIZE;
            hcBrcb1Size = RoundUp(rowFactor_ * hcMultAlign_, 8) * BLOCK_SIZE;
            reduceBufSize = rowFactor_ * hcMultAlign_ * sizeof(float);
            maskPatternSize = BLOCK_SIZE;
            totalSize = mix0OriginSize + mix1OriginSize + mix2OriginSize + mix0Size + mix1Size + mix2Size + squareSumSize + rsqrtSize + 
                        xSize + ySize + postSize + combFragSize + base0Size + base1Size + base2Size + xCastSize + yCastSize + 
                        rowBrcb0Size + hcBrcb1Size + reduceBufSize + maskPatternSize;
            if (totalSize > ubSize_) {
                rowFactor_ = rowFactor_ - 1;
                break;
            }
            rowFactor_ = rowFactor_ + 1;
        }
        rowFactor_ = rowFactor_ > rowOfFormerBlock_ ? rowFactor_ - 1 : rowFactor_;
    }
    rowLoopOfFormerBlock_ = CeilDiv(rowOfFormerBlock_, rowFactor_);
    rowLoopOfTailBlock_ = CeilDiv(rowOfTailBlock_, rowFactor_);
    tailRowFactorOfFormerBlock_ = rowOfFormerBlock_ % rowFactor_ == 0 ? rowFactor_ : rowOfFormerBlock_ % rowFactor_;
    tailRowFactorOfTailBlock_ = rowOfTailBlock_ % rowFactor_ == 0 ? rowFactor_ : rowOfTailBlock_ % rowFactor_;  
    
    tilingData_.set_bs(bs_);
    tilingData_.set_hcMix(hcMix_);
    tilingData_.set_hcMult(hcMult_);
    tilingData_.set_d(d_);
    tilingData_.set_hcMultAlign(hcMultAlign_);
    tilingData_.set_rowOfFormerBlock(rowOfFormerBlock_);
    tilingData_.set_rowOfTailBlock(rowOfTailBlock_);
    tilingData_.set_rowLoopOfFormerBlock(rowLoopOfFormerBlock_);
    tilingData_.set_rowLoopOfTailBlock(rowLoopOfTailBlock_);
    tilingData_.set_stage2RowFactor(rowFactor_);
    tilingData_.set_secondUsedCoreNum(usedAivCoreNums_);
    tilingData_.set_tailRowFactorOfFormerBlock(tailRowFactorOfFormerBlock_);
    tilingData_.set_tailRowFactorOfTailBlock(tailRowFactorOfTailBlock_);
    tilingData_.set_dLoop(dLoop_);
    tilingData_.set_dFactor(dFactor_);
    tilingData_.set_tailDFactor(tailDFactor_);
    tilingData_.set_iterTimes(iterTimes_);
    tilingData_.set_hcEps(hcEps_);
    tilingData_.set_normEps(normEps_);
    tilingData_.set_needGrad(needGrad_ ? 1 : 0);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPreSinkhornTiling::CalcMembaseOpTiling() 
{
    rowOfFormerBlock_ = CeilDiv(bs_, static_cast<int64_t>(aivCoreNum_));
    usedAivCoreNums_ = std::min(CeilDiv(bs_, rowOfFormerBlock_), static_cast<int64_t>(aivCoreNum_));
    rowOfTailBlock_ = bs_ - (usedAivCoreNums_ - 1) * rowOfFormerBlock_;

    int64_t minRowPerCore = 1;
    int64_t rowOnceLoop = std::min(rowOfFormerBlock_, minRowPerCore);

    hcMultAlign_ = RoundUp(hcMult_, BLOCK_SIZE / sizeof(float));
    int64_t mix0Size = rowOnceLoop * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER;     
    int64_t mix1Size = rowOnceLoop * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER; 
    int64_t mix2Size = rowOnceLoop * hcMult_ * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER; 
    int64_t rsqrtSize = RoundUp(rowOnceLoop, BLOCK_SIZE / sizeof(float)) * sizeof(float) * DOUBLE_BUFFER;
    int64_t xSize = rowOnceLoop * hcMult_ * RoundUp(d_, 16) * 2 * DOUBLE_BUFFER;
    int64_t ySize = rowOnceLoop * hcMult_ * RoundUp(d_, 16) * 2 * DOUBLE_BUFFER;
    int64_t postSize = rowOnceLoop * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER;
    int64_t combFragSize = rowOnceLoop * hcMult_ * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER;
    int64_t base0Size = hcMultAlign_ * sizeof(float);
    int64_t base1Size = hcMultAlign_ * sizeof(float);
    int64_t base2Size = hcMult_ * hcMultAlign_ * sizeof(float);
    int64_t xCastSize = rowOnceLoop * hcMult_ * RoundUp(d_, 8) * sizeof(float);
    int64_t yCastSize = rowOnceLoop * RoundUp(d_, 8) * sizeof(float);
    int64_t rowBrcb0Size = RoundUp(rowOnceLoop, 8) * BLOCK_SIZE;
    int64_t hcBrcb1Size = RoundUp(rowOnceLoop * hcMultAlign_, 8) * BLOCK_SIZE;
    int64_t reduceBufSize = rowOnceLoop * hcMultAlign_ * sizeof(float);

    int64_t totalSize = mix0Size + mix1Size + mix2Size + rsqrtSize + xSize + ySize + postSize + combFragSize + 
                        base0Size + base1Size + base2Size + xCastSize + yCastSize + rowBrcb0Size + hcBrcb1Size + reduceBufSize;
    rowFactor_ = rowOnceLoop; 
    if (totalSize <= ubSize_) {
        dLoop_ = 1;
        dFactor_ = d_;
        tailDFactor_ = dFactor_;
    } else {
        int64_t usedUbSize = mix0Size + mix1Size + mix2Size + rsqrtSize + postSize + combFragSize + 
                             base0Size + base1Size + base2Size + rowBrcb0Size + hcBrcb1Size + reduceBufSize;
        int64_t ubRemain = ubSize_ - usedUbSize;
        dFactor_ = d_;
        int64_t base = 2;
        while (1) {
            dFactor_ = CeilDiv(d_, base);
            xSize = rowOnceLoop * hcMult_ * RoundUp(dFactor_, 16) * 2 * DOUBLE_BUFFER;
            ySize = rowOnceLoop * hcMult_ * RoundUp(dFactor_, 16) * 2 * DOUBLE_BUFFER;
            xCastSize = rowOnceLoop * hcMult_ * RoundUp(dFactor_, 8) * sizeof(float);
            yCastSize = rowOnceLoop * RoundUp(dFactor_, 8) * sizeof(float);
            int64_t targetSize = xSize + ySize + xCastSize + yCastSize;
            if (targetSize <= ubRemain) {
                break;
            }
            base++;
        }
        if (dFactor_ > 32) {
            dFactor_ = DownAlign(dFactor_, 32);
        }
        dLoop_ = CeilDiv(d_, dFactor_);
        tailDFactor_ = d_ % dFactor_ == 0 ? dFactor_ : d_ % dFactor_;
    }

    if (dFactor_ == d_) {
        while (rowFactor_ <= rowOfFormerBlock_) {
            mix0Size = rowFactor_ * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER;     
            mix1Size = rowFactor_ * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER; 
            mix2Size = rowFactor_ * hcMult_ * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER; 
            rsqrtSize = RoundUp(rowFactor_, BLOCK_SIZE / sizeof(float)) * sizeof(float) * DOUBLE_BUFFER;
            xSize = rowFactor_ * hcMult_ * RoundUp(d_, 16) * 2 * DOUBLE_BUFFER;
            ySize = rowFactor_ * hcMult_ * RoundUp(d_, 16) * 2 * DOUBLE_BUFFER;
            postSize = rowFactor_ * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER;
            combFragSize = rowFactor_ * hcMult_ * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER;
            xCastSize = rowFactor_ * hcMult_ * RoundUp(d_, 8) * sizeof(float);
            yCastSize = rowFactor_ * hcMult_ * RoundUp(d_, 8) * sizeof(float);
            rowBrcb0Size = RoundUp(rowFactor_, 8) * BLOCK_SIZE;
            hcBrcb1Size = RoundUp(rowFactor_ * hcMultAlign_, 8) * BLOCK_SIZE;
            reduceBufSize = rowFactor_ * hcMultAlign_ * sizeof(float);
            totalSize = mix0Size + mix1Size + mix2Size + rsqrtSize + xSize + ySize + postSize + combFragSize + 
                                base0Size + base1Size + base2Size + xCastSize + yCastSize + rowBrcb0Size + hcBrcb1Size + reduceBufSize;;
            if (totalSize > ubSize_) {
                rowFactor_ = rowFactor_ - 1;
                break;
            }
            rowFactor_ = rowFactor_ + 1;
        }
        rowFactor_ = rowFactor_ > rowOfFormerBlock_ ? rowFactor_ - 1 : rowFactor_;
    }
    rowLoopOfFormerBlock_ = CeilDiv(rowOfFormerBlock_, rowFactor_);
    rowLoopOfTailBlock_ = CeilDiv(rowOfTailBlock_, rowFactor_);
    tailRowFactorOfFormerBlock_ = rowOfFormerBlock_ % rowFactor_ == 0 ? rowFactor_ : rowOfFormerBlock_ % rowFactor_;
    tailRowFactorOfTailBlock_ = rowOfTailBlock_ % rowFactor_ == 0 ? rowFactor_ : rowOfTailBlock_ % rowFactor_;  

    tilingData_.set_bs(bs_);
    tilingData_.set_hcMix(hcMix_);
    tilingData_.set_hcMult(hcMult_);
    tilingData_.set_d(d_);
    tilingData_.set_hcMultAlign(hcMultAlign_);
    tilingData_.set_rowOfFormerBlock(rowOfFormerBlock_);
    tilingData_.set_rowOfTailBlock(rowOfTailBlock_);
    tilingData_.set_rowLoopOfFormerBlock(rowLoopOfFormerBlock_);
    tilingData_.set_rowLoopOfTailBlock(rowLoopOfTailBlock_);
    tilingData_.set_rowFactor(rowFactor_);
    tilingData_.set_tailRowFactorOfFormerBlock(tailRowFactorOfFormerBlock_);
    tilingData_.set_tailRowFactorOfTailBlock(tailRowFactorOfTailBlock_);
    tilingData_.set_dLoop(dLoop_);
    tilingData_.set_dFactor(dFactor_);
    tilingData_.set_tailDFactor(tailDFactor_);
    tilingData_.set_iterTimes(iterTimes_);
    tilingData_.set_hcEps(hcEps_);

    tilingData_.set_kBlockFactor(32);
    tilingData_.set_kFactor(512);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPreSinkhornTiling::CalcOpTiling() {
    uint64_t kSize = hcMult_ * d_;
    tilingData_.set_k(kSize);
    uint64_t mDimNum = std::min(aicCoreNum_, static_cast<uint64_t>(CeilDiv(bs_, M_L1_MAX_SIZE)));
    uint64_t singleCoreM = RoundUp(CeilDiv(bs_, mDimNum), AscendC::BLOCK_CUBE);
    uint64_t kDimNum = aicCoreNum_ / mDimNum;
    uint64_t splitKSize = RoundUp(CeilDiv(kSize, kDimNum), K_MULIT_CORE_SPLIT_BASE_SIZE);
    
    tilingData_.set_cubeBlockDimM(mDimNum);
    tilingData_.set_cubeBlockDimK(CeilDiv(kSize, splitKSize));
    tilingData_.set_multCoreSplitMSize(singleCoreM);
    tilingData_.set_mL1Size(std::min(M_L1_MAX_SIZE, singleCoreM));
    tilingData_.set_multCoreSplitKSize(splitKSize);
    tilingData_.set_kL1Size(std::min(A_L1_SIZE / singleCoreM, static_cast<uint64_t>(K_L1_MAX_SIZE)) / 128 * 128);
    
    tilingData_.set_cvLoopKSize(1024);

    tilingData_.set_cubeCoreNum(static_cast<int64_t>(aicCoreNum_));
    int64_t lineByteSize = (sizeof(int16_t) + sizeof(int32_t)) * DOUBLE_BUFFER * tilingData_.get_cvLoopKSize();
    int64_t stage1MFactorValue = ubSize_ / lineByteSize;
    tilingData_.set_stage1MFactor(stage1MFactorValue);
    if (kDimNum != 1) {
        return CalcMKSplitCoreMembasePart2Tiling();
    }
    return CalcMembaseOpTiling();
}


ge::graphStatus MhcPreSinkhornTiling::DoOpTiling()
{
    if (GetPlatformInfo() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }

    if (GetShapeAttrsInfoInner() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }

    if (CalcOpTiling() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }

    if (GetWorkspaceSize() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }

    if (PostTiling() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPreSinkhornTiling::GetWorkspaceSize()
{
    workspaceSize_ = 16 * 1024 * 1024 + 128 * 1024 * 1024;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPreSinkhornTiling::PostTiling()
{
    context_->SetTilingKey(0);
    context_->SetBlockDim(aicCoreNum_);
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = workspaceSize_;
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForMhcPreSinkhorn(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForMhcPreSinkhorn(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("MhcPreSinkhorn", "Tiling context is null"),
               return ge::GRAPH_FAILED);

    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OP_LOGE("TilingForMhcPreSinkhorn", "Tiling platformInfo is null"),
               return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);

    MhcPreSinkhornTiling mhcPreSinkhornTiling(context);
    return mhcPreSinkhornTiling.DoOpTiling();
}

IMPL_OP_OPTILING(MhcPreSinkhorn)
    .Tiling(TilingForMhcPreSinkhorn)
    .TilingParse<MhcPreSinkhornCompileInfo>(TilingPrepareForMhcPreSinkhorn);

}  // namespace optiling
