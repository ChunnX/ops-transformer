/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_fused_topk_tiling.cpp
 * \brief
 */

#include "platform/platform_info.h"
#include "log/log.h"
#include "err/ops_err.h"
#include "util/math_util.h"
#include "moe_fused_topk_tiling.h"

namespace {
constexpr uint32_t BYTE_BLOCK = 32;
constexpr int32_t X_INPUT_INDEX = 0;
constexpr int32_t ADD_NUM_INPUT_INDEX = 1;
constexpr int32_t MAPPING_NUM_INPUT_INDEX = 2;
constexpr int32_t MAPPING_TABLE_INPUT_INDEX = 3;

constexpr uint32_t DIM_INDEX0 = 0;
constexpr uint32_t DIM_INDEX1 = 1;

constexpr uint32_t ATTR_GROUP_NUM_INDEX = 0;
constexpr uint32_t ATTR_GROUP_TOPK_INDEX = 1;
constexpr uint32_t ATTR_TOP_N_INDEX = 2;
constexpr uint32_t ATTR_TOP_K_INDEX = 3;
constexpr uint32_t ATTR_ACTIVATE_TYPE_INDEX = 4;
constexpr uint32_t ATTR_IS_NORM_INDEX = 5;
constexpr uint32_t ATTR_SCALE_INDEX = 6;
constexpr uint32_t ATTR_ENABLE_EXPERT_MAPPING_INDEX = 7;

constexpr uint32_t RESERVED_UB = 16 * 1024;
constexpr uint32_t SORT_UNIT = 32;
constexpr uint32_t SYS_WORKSPACESIZE = 16 * 1024 * 1024;

constexpr bool TOPK_IS_REUSE_SOURCE = false;
constexpr bool TOPK_IS_INIT_INDEX = false;
constexpr bool TOPK_IS_LARGEST = true;
constexpr uint32_t FP32_DTYPE_SIZE = 4U;  // float
constexpr uint32_t INT32_DTYPE_SIZE = 4U; // int32
constexpr uint32_t FP32_BLOCK_ALIGN_NUM = 8U;
} // namespace

namespace optiling {
using Ops::Base::CeilAlign;
using Ops::Transformer::OpTiling::TilingBaseClass;

class MoeFusedTopkTilingBase : public TilingBaseClass {
public:
    explicit MoeFusedTopkTilingBase(gert::TilingContext *context) : TilingBaseClass(context) {}
    ~MoeFusedTopkTilingBase() override = default;

protected:
    bool IsCapable() override
    {
        return true;
    }
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override
    {
        return ge::GRAPH_SUCCESS;
    }
    uint64_t GetTilingKey() const override
    {
        return tilingKey_;
    }
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;

private:
    void ComputeTilingKey();
    void GetUsedCore();
    ge::graphStatus SplitUb();
    ge::graphStatus GetTopKTiling();
    void GetTmpBuffSize();
    void TilingDataPrint();

    uint32_t firstDimSize_ = 0;
    uint32_t secondDimSize_ = 0;
    uint32_t addNumDimSize_ = 0;
    uint32_t groupNum_ = 0;
    uint32_t groupTopk_;
    uint32_t topN_;
    uint32_t topK_ = 0;
    uint32_t activateType_ = 0;
    uint32_t isNorm_ = 0;
    float scale_ = 1.0;
    bool enableExpertMapping_ = false;
    uint32_t groupEles_ = 0;
    uint32_t expertNum_ = 0;
    uint32_t tableDim_ = 0;
    uint32_t ubSize_ = 0;
    uint32_t usedCoreNum_ = 0;
    uint32_t coreNum_ = 0;
    uint32_t batchPerCore_ = 1;
    uint32_t tailBatch_ = 0;
    uint32_t ubFactorElement_ = 0;
    uint32_t topkMaxValue_ = 0;
    uint32_t topkMinValue_ = 0;
    uint64_t tilingKey_ = 0;
    uint64_t workspacePerCore_ = 0;
    MoeFusedTopkTilingData tilingData_;
};

ge::graphStatus MoeFusedTopkTilingBase::GetShapeAttrsInfo()
{
    auto xShape = context_->GetInputShape(X_INPUT_INDEX)->GetStorageShape();
    auto addNumShape = context_->GetInputShape(ADD_NUM_INPUT_INDEX)->GetStorageShape();
    firstDimSize_ = xShape.GetDim(DIM_INDEX0);
    secondDimSize_ = xShape.GetDim(DIM_INDEX1);
    addNumDimSize_ = addNumShape.GetDim(DIM_INDEX0);

    auto attrs = context_->GetAttrs();
    groupNum_ = *(attrs->GetAttrPointer<uint32_t>(ATTR_GROUP_NUM_INDEX));
    groupTopk_ = *(attrs->GetAttrPointer<uint32_t>(ATTR_GROUP_TOPK_INDEX));
    topN_ = *(attrs->GetAttrPointer<uint32_t>(ATTR_TOP_N_INDEX));
    topK_ = *(attrs->GetAttrPointer<uint32_t>(ATTR_TOP_K_INDEX));
    activateType_ = *(attrs->GetAttrPointer<uint32_t>(ATTR_ACTIVATE_TYPE_INDEX));
    isNorm_ = static_cast<uint32_t>(*(attrs->GetAttrPointer<bool>(ATTR_IS_NORM_INDEX)));
    scale_ = *(attrs->GetAttrPointer<float>(ATTR_SCALE_INDEX));
    enableExpertMapping_ = *(attrs->GetAttrPointer<bool>(ATTR_ENABLE_EXPERT_MAPPING_INDEX));
    groupEles_ = groupNum_ == 0 ? secondDimSize_ : secondDimSize_ / groupNum_;

    if (enableExpertMapping_) {
        auto mappingTableShape = context_->GetInputShape(MAPPING_TABLE_INPUT_INDEX)->GetStorageShape();
        expertNum_ = mappingTableShape.GetDim(DIM_INDEX0);
        tableDim_ = mappingTableShape.GetDim(DIM_INDEX1);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeFusedTopkTilingBase::GetPlatformInfo()
{
    auto compileInfo = static_cast<const MoeFusedTopkCompileInfo*>(context_->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfo);
    coreNum_ = compileInfo->coreNum;
    ubSize_ = (compileInfo->ubSize - RESERVED_UB) / BYTE_BLOCK * BYTE_BLOCK;
    OP_CHECK_IF(coreNum_ == 0,
                OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(),
                                            "MoeFusedTopk GetHardwareInfo Failed, vectorCoreNum: %u", coreNum_),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ubSize_ == 0,
                OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(),
                                            "MoeFusedTopk GetHardwareInfo Failed, ubSize: %u", ubSize_),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeFusedTopkTilingBase::DoOpTiling()
{
    ComputeTilingKey();
    GetUsedCore();
    if (GetTopKTiling() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_->GetNodeName(), "GetTopKTiling Failed");
        return ge::GRAPH_FAILED;
    }
    GetTmpBuffSize();
    return SplitUb();
}

void MoeFusedTopkTilingBase::ComputeTilingKey()
{
    tilingKey_ = 0U;
    if (enableExpertMapping_) {
        tilingKey_ += 1U;
    }
}

void MoeFusedTopkTilingBase::GetUsedCore()
{
    if (firstDimSize_ <= coreNum_) {
        batchPerCore_ = 1;
        usedCoreNum_ = firstDimSize_;
        tailBatch_ = 0;
        return;
    }
    batchPerCore_ = firstDimSize_ / coreNum_;
    tailBatch_ = firstDimSize_ % coreNum_;
    usedCoreNum_ = coreNum_;
    return;
}

ge::graphStatus MoeFusedTopkTilingBase::SplitUb()
{
    uint32_t needUbSize = 0;
    uint32_t tilingDataSize = (sizeof(tilingData_) + BYTE_BLOCK - 1) / BYTE_BLOCK * BYTE_BLOCK;
    ubFactorElement_ = (ubSize_ - tilingDataSize) / BYTE_BLOCK;

    // tilingDataSize
    needUbSize += tilingDataSize;
    // xInQueue_
    needUbSize += CeilAlign(groupEles_, SORT_UNIT) * groupNum_ * FP32_DTYPE_SIZE;
    // addNumInQueue_
    needUbSize += CeilAlign(secondDimSize_, FP32_BLOCK_ALIGN_NUM) * FP32_DTYPE_SIZE;
    // yOutQueue_
    needUbSize += CeilAlign(topK_, FP32_BLOCK_ALIGN_NUM) * FP32_DTYPE_SIZE;
    // indicesOutQueue_
    needUbSize += CeilAlign(topK_, SORT_UNIT) * INT32_DTYPE_SIZE;
    // sigmoidBuf_
    needUbSize += CeilAlign(secondDimSize_, FP32_BLOCK_ALIGN_NUM) * FP32_DTYPE_SIZE;
    // sigmoidAddQueue_
    needUbSize += CeilAlign(secondDimSize_, FP32_BLOCK_ALIGN_NUM) * FP32_DTYPE_SIZE;
    // sortedQueue_
    needUbSize += CeilAlign(groupEles_, SORT_UNIT) * groupNum_ * sizeof(int64_t);
    // topkValueQueue_
    needUbSize += CeilAlign(secondDimSize_, SORT_UNIT) * FP32_DTYPE_SIZE;
    // assistQueue_
    needUbSize += CeilAlign(secondDimSize_, SORT_UNIT) * INT32_DTYPE_SIZE;
    // mappingNumQueue_
    needUbSize += CeilAlign(expertNum_ * INT32_DTYPE_SIZE, BYTE_BLOCK);
    // tempBuf_
    needUbSize += topkMinValue_;

    OP_CHECK_IF(needUbSize > ubSize_,
            OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(),
                                            "This case need minimum UB size is %u, which is out of total UB size: %u.",
                                            needUbSize, ubSize_),
            return ge::GRAPH_FAILED);
    OP_LOGD(context_->GetNodeName(), "This case need minimum UB size is %u.", needUbSize);
    
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeFusedTopkTilingBase::GetWorkspaceSize()
{
    workspacePerCore_ = secondDimSize_ * FP32_DTYPE_SIZE;
    size_t usedWorkspaceSize = SYS_WORKSPACESIZE + usedCoreNum_ * workspacePerCore_;
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = usedWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeFusedTopkTilingBase::PostTiling()
{
    context_->SetBlockDim(usedCoreNum_);
    tilingData_.set_firstDimSize(firstDimSize_);
    tilingData_.set_secondDimSize(secondDimSize_);
    tilingData_.set_addNumDimSize(addNumDimSize_);
    tilingData_.set_groupNum(groupNum_);
    tilingData_.set_groupTopk(groupTopk_);
    tilingData_.set_topN(topN_);
    tilingData_.set_topK(topK_);

    tilingData_.set_activateType(activateType_);
    tilingData_.set_isNorm(isNorm_);
    tilingData_.set_scale(scale_);
    tilingData_.set_groupEles(groupEles_);
    tilingData_.set_blockNum(usedCoreNum_);
    tilingData_.set_ubFactorElement(ubFactorElement_);
    tilingData_.set_batchPerCore(batchPerCore_);
    tilingData_.set_tailBatch(tailBatch_);

    tilingData_.set_expertNum(expertNum_);
    tilingData_.set_tableDim(tableDim_);
    tilingData_.set_topkMaxValue(topkMaxValue_);
    tilingData_.set_topkMinValue(topkMinValue_);
    tilingData_.set_workspacePerCore(workspacePerCore_);

    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    TilingDataPrint();
    return ge::GRAPH_SUCCESS;
}

void MoeFusedTopkTilingBase::TilingDataPrint()
{
    OP_LOGD(context_->GetNodeName(), "tilingKey:                      %lu", tilingKey_);
    OP_LOGD(context_->GetNodeName(), "usedCoreNum:                    %u", usedCoreNum_);
    OP_LOGD(context_->GetNodeName(), "firstDimSize:                   %u", tilingData_.get_firstDimSize());
    OP_LOGD(context_->GetNodeName(), "secondDimSize:                  %u", tilingData_.get_secondDimSize());
    OP_LOGD(context_->GetNodeName(), "addNumDimSize:                  %u", tilingData_.get_addNumDimSize());
    OP_LOGD(context_->GetNodeName(), "groupNum:                       %u", tilingData_.get_groupNum());
    OP_LOGD(context_->GetNodeName(), "groupTopk:                      %u", tilingData_.get_groupTopk());
    OP_LOGD(context_->GetNodeName(), "topN:                           %u", tilingData_.get_topN());
    OP_LOGD(context_->GetNodeName(), "topK:                           %u", tilingData_.get_topK());
    OP_LOGD(context_->GetNodeName(), "activateType:                   %u", tilingData_.get_activateType());
    OP_LOGD(context_->GetNodeName(), "isNorm:                         %d", tilingData_.get_isNorm());
    OP_LOGD(context_->GetNodeName(), "scale:                          %f", tilingData_.get_scale());
    OP_LOGD(context_->GetNodeName(), "groupEles:                      %u", tilingData_.get_groupEles());
    OP_LOGD(context_->GetNodeName(), "blockNum:                       %u", tilingData_.get_blockNum());
    OP_LOGD(context_->GetNodeName(), "ubFactorElement:                %u", tilingData_.get_ubFactorElement());
    OP_LOGD(context_->GetNodeName(), "batchPerCore:                   %u", tilingData_.get_batchPerCore());
    OP_LOGD(context_->GetNodeName(), "tailBatch:                      %u", tilingData_.get_tailBatch());
    OP_LOGD(context_->GetNodeName(), "expertNum:                      %u", tilingData_.get_expertNum());
    OP_LOGD(context_->GetNodeName(), "tableDim:                       %u", tilingData_.get_tableDim());
    OP_LOGD(context_->GetNodeName(), "topkMaxValue:                   %u", tilingData_.get_topkMaxValue());
    OP_LOGD(context_->GetNodeName(), "topkMinValue:                   %u", tilingData_.get_topkMinValue());
    OP_LOGD(context_->GetNodeName(), "workspacePerCore:               %lu", tilingData_.get_workspacePerCore());
}

ge::graphStatus MoeFusedTopkTilingBase::GetTopKTiling()
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());

    int32_t topkInner = (groupEles_ + SORT_UNIT - 1) / SORT_UNIT * SORT_UNIT;

    OP_CHECK_IF(!AscendC::TopKTilingFunc(ascendcPlatform, topkInner, groupNum_, topN_, FP32_DTYPE_SIZE, TOPK_IS_INIT_INDEX,
                        AscendC::TopKMode::TOPK_NORMAL, TOPK_IS_LARGEST, tilingData_.topkTilingData),
                    OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(),
                                                    "TopKTilingFunc Failed"),
                    return ge::GRAPH_FAILED);
    OP_CHECK_IF(!AscendC::GetTopKMaxMinTmpSize(ascendcPlatform, topkInner, groupNum_, TOPK_IS_REUSE_SOURCE, 
                        TOPK_IS_INIT_INDEX, AscendC::TopKMode::TOPK_NORMAL, TOPK_IS_LARGEST, FP32_DTYPE_SIZE,
                        topkMaxValue_, topkMinValue_),
                    OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(),
                                                    "GetTopKMaxMinTmpSize Failed"),
                    return ge::GRAPH_FAILED);
    
    return ge::GRAPH_SUCCESS;
}

void MoeFusedTopkTilingBase::GetTmpBuffSize()
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    uint32_t groupElesAlignSortCount = (groupNum_ + SORT_UNIT - 1) / SORT_UNIT * SORT_UNIT;
    uint32_t maxValue = 0;
    uint32_t minValue = 0;
    std::vector<int64_t> sigmoidVec = {static_cast<int64_t>(secondDimSize_)};
    ge::Shape sigmoidShape(sigmoidVec);
    AscendC::GetSigmoidMaxMinTmpSize(sigmoidShape, FP32_DTYPE_SIZE, false, maxValue, minValue);
    topkMinValue_ = std::max(topkMinValue_, minValue);
    topkMaxValue_ = std::max(topkMaxValue_, maxValue);

    uint32_t sortTmpSize = AscendC::GetSortTmpSize(ascendcPlatform, groupElesAlignSortCount, FP32_DTYPE_SIZE);
    topkMinValue_ = std::max(topkMinValue_, sortTmpSize);

    std::vector<int64_t> srcBroadCastVec = {static_cast<int64_t>(groupNum_), 1};
    std::vector<int64_t> dstBroadCastVec = {static_cast<int64_t>(groupNum_), static_cast<int64_t>(groupEles_)};
    ge::Shape srcBroadCastShape(srcBroadCastVec);
    ge::Shape dstBroadCastShape(dstBroadCastVec);
    AscendC::GetBroadCastMaxMinTmpSize(ascendcPlatform, srcBroadCastShape, dstBroadCastShape,
                                       FP32_DTYPE_SIZE, false, maxValue, minValue);
    topkMinValue_ = std::max(topkMinValue_, minValue);
    topkMaxValue_ = std::max(topkMaxValue_, maxValue);

    uint32_t secondDimSizeAlignSortCount = (secondDimSize_ + SORT_UNIT - 1) / SORT_UNIT * SORT_UNIT;
    sortTmpSize = AscendC::GetSortTmpSize(ascendcPlatform, secondDimSizeAlignSortCount, FP32_DTYPE_SIZE);
    topkMinValue_ = std::max(topkMinValue_, sortTmpSize);

    topkMinValue_ = std::max(topkMinValue_, secondDimSizeAlignSortCount * FP32_DTYPE_SIZE);
    topkMaxValue_ = std::max(topkMaxValue_, topkMinValue_);
}

REGISTER_OPS_TILING_TEMPLATE(MoeFusedTopk, MoeFusedTopkTilingBase, 10000); // If not 950, fallback to this.
} // namespace optiling
