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
 * \file moe_fused_topk_tiling_arch35.cpp
 * \brief Ascend950 tiling for MoeFusedTopk.
 */

#include "moe_fused_topk_tiling.h"
#include "../op_kernel/arch35/moe_fused_topk_arch35_tiling_def.h"
#include "err/ops_err.h"
#include "log/log.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"
#include "op_host/tiling_util.h"
#include "platform/platform_info.h"
#include "util/math_util.h"

namespace {
constexpr uint32_t BYTE_BLOCK = 32;
constexpr int32_t X_INPUT_INDEX = 0;
constexpr int32_t ADD_NUM_INPUT_INDEX = 1;
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
constexpr uint32_t FP32_DTYPE_SIZE = 4U;
constexpr uint32_t INT32_DTYPE_SIZE = 4U;
constexpr uint32_t FP32_BLOCK_ALIGN_NUM = 8U;
constexpr uint32_t FP16_BLOCK_ALIGN_NUM = 16U;
} // namespace

namespace optiling {
using Ops::Base::CeilAlign;

class MoeFusedTopkTilingArch35 : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit MoeFusedTopkTilingArch35(gert::TilingContext *context)
        : Ops::Transformer::OpTiling::TilingBaseClass(context)
    {}
    ~MoeFusedTopkTilingArch35() override = default;

protected:
    bool IsCapable() override { return Ops::Transformer::OpTiling::IsRegbaseSocVersion(context_); }
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override { return ge::GRAPH_SUCCESS; }
    uint64_t GetTilingKey() const override { return tilingKey_; }
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;

private:
    void GetTilingKey();
    void GetUsedCore();
    ge::graphStatus SplitUb();
    ge::graphStatus GetTopKTiling();
    void GetTmpBuffSize();
    void TilingDataPrint() const;

    uint32_t firstDimSize_ = 0;
    uint32_t secondDimSize_ = 0;
    uint32_t addNumDimSize_ = 0;
    uint32_t groupNum_ = 0;
    uint32_t groupTopk_ = 0;
    uint32_t topN_ = 0;
    uint32_t topK_ = 0;
    uint32_t activateType_ = 0;
    uint32_t isNorm_ = 0;
    float scale_ = 1.0f;
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
    MoeFusedTopkArch35TilingData *tilingDataPtr_ = nullptr;
};

ge::graphStatus MoeFusedTopkTilingArch35::GetShapeAttrsInfo()
{
    auto xShape = context_->GetInputShape(X_INPUT_INDEX)->GetStorageShape();
    auto addNumShape = context_->GetInputShape(ADD_NUM_INPUT_INDEX)->GetStorageShape();
    firstDimSize_ = static_cast<uint32_t>(xShape.GetDim(DIM_INDEX0));
    secondDimSize_ = static_cast<uint32_t>(xShape.GetDim(DIM_INDEX1));
    addNumDimSize_ = static_cast<uint32_t>(addNumShape.GetDim(DIM_INDEX0));

    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
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
        expertNum_ = static_cast<uint32_t>(mappingTableShape.GetDim(DIM_INDEX0));
        tableDim_ = static_cast<uint32_t>(mappingTableShape.GetDim(DIM_INDEX1));
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeFusedTopkTilingArch35::GetPlatformInfo()
{
    auto compileInfo = static_cast<const MoeFusedTopkCompileInfo *>(context_->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfo);
    coreNum_ = compileInfo->coreNum;
    ubSize_ = static_cast<uint32_t>((compileInfo->ubSize - RESERVED_UB) / BYTE_BLOCK * BYTE_BLOCK);
    OP_CHECK_IF(coreNum_ == 0,
                OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(),
                                            "MoeFusedTopk GetHardwareInfo Failed, vectorCoreNum: %u", coreNum_),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ubSize_ == 0,
                OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "MoeFusedTopk GetHardwareInfo Failed, ubSize: %u",
                                            ubSize_),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

void MoeFusedTopkTilingArch35::GetTilingKey()
{
    tilingKey_ = 0U;
    if (enableExpertMapping_) {
        tilingKey_ += 1U;
    }
}

void MoeFusedTopkTilingArch35::GetUsedCore()
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
}

ge::graphStatus MoeFusedTopkTilingArch35::GetTopKTiling()
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    int32_t topkInner = static_cast<int32_t>(CeilAlign(groupEles_, SORT_UNIT));

    OP_CHECK_IF(
        !AscendC::TopKTilingFunc(ascendcPlatform, topkInner, groupNum_, topN_, FP32_DTYPE_SIZE, TOPK_IS_INIT_INDEX,
                                 AscendC::TopKMode::TOPK_NORMAL, TOPK_IS_LARGEST, tilingDataPtr_->topkTilingData),
        OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "TopKTilingFunc Failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(!AscendC::GetTopKMaxMinTmpSize(ascendcPlatform, topkInner, groupNum_, TOPK_IS_REUSE_SOURCE,
                                               TOPK_IS_INIT_INDEX, AscendC::TopKMode::TOPK_NORMAL, TOPK_IS_LARGEST,
                                               FP32_DTYPE_SIZE, topkMaxValue_, topkMinValue_),
                OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "GetTopKMaxMinTmpSize Failed"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

void MoeFusedTopkTilingArch35::GetTmpBuffSize()
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    uint32_t maxValue = 0;
    uint32_t minValue = 0;
    std::vector<int64_t> sigmoidVec = {static_cast<int64_t>(secondDimSize_)};
    ge::Shape sigmoidShape(sigmoidVec);
    AscendC::GetSigmoidMaxMinTmpSize(sigmoidShape, FP32_DTYPE_SIZE, false, maxValue, minValue);
    topkMinValue_ = std::max(topkMinValue_, minValue);
    topkMaxValue_ = std::max(topkMaxValue_, maxValue);

    uint32_t groupSortTmpSize =
        AscendC::GetSortTmpSize(ascendcPlatform, CeilAlign(groupEles_, SORT_UNIT), FP32_DTYPE_SIZE);
    topkMinValue_ = std::max(topkMinValue_, groupSortTmpSize);
    topkMaxValue_ = std::max(topkMaxValue_, groupSortTmpSize);

    std::vector<int64_t> srcBroadCastVec = {static_cast<int64_t>(groupNum_), 1};
    std::vector<int64_t> dstBroadCastVec = {static_cast<int64_t>(groupNum_), static_cast<int64_t>(groupEles_)};
    ge::Shape srcBroadCastShape(srcBroadCastVec);
    ge::Shape dstBroadCastShape(dstBroadCastVec);
    AscendC::GetBroadCastMaxMinTmpSize(ascendcPlatform, srcBroadCastShape, dstBroadCastShape, FP32_DTYPE_SIZE, false,
                                       maxValue, minValue);
    topkMinValue_ = std::max(topkMinValue_, minValue);
    topkMaxValue_ = std::max(topkMaxValue_, maxValue);

    uint32_t secondDimSizeAlignSortCount = CeilAlign(secondDimSize_, SORT_UNIT);
    uint32_t expertSortTmpSize = AscendC::GetSortTmpSize(ascendcPlatform, secondDimSizeAlignSortCount, FP32_DTYPE_SIZE);
    topkMinValue_ = std::max(topkMinValue_, expertSortTmpSize);
    topkMaxValue_ = std::max(topkMaxValue_, expertSortTmpSize);

    // Ascend950 Sum uses shared scratch by repeat count; reserve it only for the arch35 kernel path.
    constexpr uint32_t VECTOR_FP32_ELEMS = 256U / FP32_DTYPE_SIZE;
    uint32_t sumRepeatCount = (topN_ + VECTOR_FP32_ELEMS - 1) / VECTOR_FP32_ELEMS;
    uint32_t sumTmpSize = CeilAlign(sumRepeatCount, SORT_UNIT) * groupNum_ * FP32_DTYPE_SIZE;
    topkMaxValue_ = std::max(topkMaxValue_, sumTmpSize);

    topkMinValue_ = std::max(topkMinValue_, secondDimSizeAlignSortCount * FP32_DTYPE_SIZE);
    topkMaxValue_ = std::max(topkMaxValue_, topkMinValue_);
}

ge::graphStatus MoeFusedTopkTilingArch35::SplitUb()
{
    uint32_t needUbSize = 0;
    uint32_t tilingDataSize = CeilAlign(sizeof(MoeFusedTopkArch35TilingData), static_cast<size_t>(BYTE_BLOCK));
    ubFactorElement_ = (ubSize_ - tilingDataSize) / BYTE_BLOCK;

    needUbSize += tilingDataSize;
    needUbSize += CeilAlign(groupEles_, SORT_UNIT) * groupNum_ * FP32_DTYPE_SIZE;
    // FP16/BF16 paths use the upper half of this FP32 buffer as a staging area before Cast.
    needUbSize += CeilAlign(secondDimSize_, FP16_BLOCK_ALIGN_NUM) * FP32_DTYPE_SIZE;
    needUbSize += CeilAlign(topK_, FP32_BLOCK_ALIGN_NUM) * FP32_DTYPE_SIZE;
    needUbSize += CeilAlign(topK_, SORT_UNIT) * INT32_DTYPE_SIZE;
    needUbSize += CeilAlign(secondDimSize_, FP32_BLOCK_ALIGN_NUM) * FP32_DTYPE_SIZE;
    needUbSize += CeilAlign(secondDimSize_, FP32_BLOCK_ALIGN_NUM) * FP32_DTYPE_SIZE;
    needUbSize += CeilAlign(groupEles_, SORT_UNIT) * groupNum_ * sizeof(int64_t);
    needUbSize += CeilAlign(secondDimSize_, SORT_UNIT) * FP32_DTYPE_SIZE;
    needUbSize += CeilAlign(secondDimSize_, SORT_UNIT) * INT32_DTYPE_SIZE;
    needUbSize += CeilAlign(expertNum_ * INT32_DTYPE_SIZE, BYTE_BLOCK);
    needUbSize += topkMaxValue_;

    OP_CHECK_IF(needUbSize > ubSize_,
                OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(),
                                            "This case need minimum UB size is %u, which is out of total UB size: %u.",
                                            needUbSize, ubSize_),
                return ge::GRAPH_FAILED);
    OP_LOGD(context_->GetNodeName(), "Ascend950 case need minimum UB size is %u.", needUbSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeFusedTopkTilingArch35::DoOpTiling()
{
    tilingDataPtr_ = context_->GetTilingData<MoeFusedTopkArch35TilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, tilingDataPtr_);

    GetTilingKey();
    GetUsedCore();
    if (GetTopKTiling() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_->GetNodeName(), "GetTopKTiling Failed");
        return ge::GRAPH_FAILED;
    }
    GetTmpBuffSize();
    return SplitUb();
}

ge::graphStatus MoeFusedTopkTilingArch35::GetWorkspaceSize()
{
    workspacePerCore_ = secondDimSize_ * FP32_DTYPE_SIZE;
    workspaceSize_ = SYS_WORKSPACESIZE + usedCoreNum_ * workspacePerCore_;
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = workspaceSize_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeFusedTopkTilingArch35::PostTiling()
{
    context_->SetBlockDim(usedCoreNum_);

    tilingDataPtr_->firstDimSize = firstDimSize_;
    tilingDataPtr_->secondDimSize = secondDimSize_;
    tilingDataPtr_->addNumDimSize = addNumDimSize_;
    tilingDataPtr_->groupNum = groupNum_;
    tilingDataPtr_->groupTopk = groupTopk_;
    tilingDataPtr_->topN = topN_;
    tilingDataPtr_->topK = topK_;

    tilingDataPtr_->activateType = activateType_;
    tilingDataPtr_->isNorm = isNorm_;
    tilingDataPtr_->scale = scale_;
    tilingDataPtr_->groupEles = groupEles_;
    tilingDataPtr_->blockNum = usedCoreNum_;
    tilingDataPtr_->ubFactorElement = ubFactorElement_;
    tilingDataPtr_->batchPerCore = batchPerCore_;
    tilingDataPtr_->tailBatch = tailBatch_;

    tilingDataPtr_->expertNum = expertNum_;
    tilingDataPtr_->tableDim = tableDim_;
    tilingDataPtr_->topkMaxValue = topkMaxValue_;
    tilingDataPtr_->topkMinValue = topkMinValue_;
    tilingDataPtr_->workspacePerCore = workspacePerCore_;

    TilingDataPrint();
    return ge::GRAPH_SUCCESS;
}

void MoeFusedTopkTilingArch35::TilingDataPrint() const
{
    OP_LOGD(context_->GetNodeName(), "arch35 tilingKey:           %lu", tilingKey_);
    OP_LOGD(context_->GetNodeName(), "arch35 usedCoreNum:         %u", usedCoreNum_);
    OP_LOGD(context_->GetNodeName(), "arch35 firstDimSize:        %u", tilingDataPtr_->firstDimSize);
    OP_LOGD(context_->GetNodeName(), "arch35 secondDimSize:       %u", tilingDataPtr_->secondDimSize);
    OP_LOGD(context_->GetNodeName(), "arch35 groupNum:            %u", tilingDataPtr_->groupNum);
    OP_LOGD(context_->GetNodeName(), "arch35 groupEles:           %u", tilingDataPtr_->groupEles);
    OP_LOGD(context_->GetNodeName(), "arch35 topkMaxValue:        %u", tilingDataPtr_->topkMaxValue);
    OP_LOGD(context_->GetNodeName(), "arch35 topkMinValue:        %u", tilingDataPtr_->topkMinValue);
    OP_LOGD(context_->GetNodeName(), "arch35 workspacePerCore:    %lu", tilingDataPtr_->workspacePerCore);
}

REGISTER_OPS_TILING_TEMPLATE(MoeFusedTopk, MoeFusedTopkTilingArch35, 1000);
} // namespace optiling
