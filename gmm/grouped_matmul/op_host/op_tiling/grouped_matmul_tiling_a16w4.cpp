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
 * \file grouped_matmul_tiling_a16w4.cpp
 * \brief
 */
#include "grouped_matmul_tiling_utils.h"

namespace optiling {
ge::graphStatus GMMTiling::CheckA16W4MsdEnable(uint64_t mSize, uint64_t antiquantGroupNum,
                                               const gert::TilingContext *context, const GMMCompileInfo *compileInfoPtr)
{
    if (!isSingleWeight_ || !isSingleX_ || !isSingleY_) {
        OP_LOGD(context->GetNodeName(),
                "A16W4 only support SingleWeight/SingleX/SingleY. current  isSingleWeight_: %d, isSingleX_: %d, "
                "isSingleY_: %d.",
                isSingleWeight_, isSingleX_, isSingleY_);
        return GRAPH_PARAM_INVALID;
    }
    // n轴在scale shape的第2维
    uint64_t nSize = context->GetDynamicInputTensor(ANTIQUANT_SCALE_INDEX, 0)->GetStorageShape().GetDim(2);
    uint64_t kSize = context->GetDynamicInputTensor(X_INDEX, 0)->GetStorageShape().GetDim(1);
    // 2: input shape info size
    std::array<int64_t, 2> kNList = {static_cast<int64_t>(kSize), static_cast<int64_t>(nSize)};
    int isInA16W4MsdWhiteList = A16W4_MSD_WHITE_LIST.count(kNList);
    uint64_t avgMSize = groupNum_ != 0 ? mSize / groupNum_ : 0;

    auto attr = context->GetAttrs();
    auto tuningConfigPtr =
        attr != nullptr ? (attr->GetAttrPointer<gert::ContinuousVector>(ATTR_INDEX_TUNING_CONFIG)) : nullptr;
    int64_t tuningConfig = (tuningConfigPtr != nullptr && tuningConfigPtr->GetSize() > 0)
                               ? (reinterpret_cast<const int64_t *>(tuningConfigPtr->GetData()))[0]
                               : 0;
    if (tuningConfig != 0L) {
        avgMSize = static_cast<uint32_t>(tuningConfig);
    }

    uint64_t withOffset = GetWithOffset(context);

    OP_LOGD(context->GetNodeName(),
            "A16W4 try to enable msd. antiquantGroupNum: %llu, whiteListFlag: %d, withOffset: %d avgM: %d. ",
            antiquantGroupNum, isInA16W4MsdWhiteList, withOffset, avgMSize);
    const auto transposeWeightPtr = attr->GetAttrPointer<bool>(ATTR_INDEX_TRANS_W);
    auto transposeWeight = transposeWeightPtr != nullptr ? *transposeWeightPtr : false;
    auto biasPtr = context->GetDynamicInputTensor(BIAS_INDEX, 0);
    bool hasBias = !(biasPtr == nullptr || biasPtr->GetStorageShape().GetShapeSize() == 0);
    // msd当前仅支持group size为32, 方案的m轴收益上限是16
    if (!(isInA16W4MsdWhiteList && antiquantGroupNum != 0 && kSize / antiquantGroupNum == 32 && !withOffset &&
          avgMSize <= 16 && context->GetOutputDesc(Y_INDEX)->GetDataType() == ge::DT_BF16 && transposeWeight &&
          !hasBias)) {
        return GRAPH_PARAM_INVALID;
    }
    return ge::GRAPH_SUCCESS;
}

uint64_t GMMTiling::GetWithOffset(const gert::TilingContext *context)
{
    auto offset = context->GetDynamicInputTensor(ANTIQUANT_OFFSET_INDEX, 0);
    uint64_t withOffset = 0;
    if (offset != nullptr) {
        auto &offsetShape = offset->GetStorageShape();
        if (offsetShape.GetDimNum() > 1 || offsetShape.GetDim(0) > 0) {
            withOffset = 1U;
        }
    }
    return withOffset;
}

ge::graphStatus GMMTiling::A16W4MsdTiling(gert::TilingContext *context, const GMMCompileInfo *compileInfoPtr)
{
    uint64_t mSize = context->GetDynamicInputTensor(X_INDEX, 0)->GetStorageShape().GetDim(0);
    uint64_t kSize = context->GetDynamicInputTensor(X_INDEX, 0)->GetStorageShape().GetDim(1);
    uint64_t antiquantGroupNum = context->GetDynamicInputTensor(ANTIQUANT_SCALE_INDEX, 0)->GetStorageShape().GetDim(1);
    if (CheckA16W4MsdEnable(mSize, antiquantGroupNum, context, compileInfoPtr) != ge::GRAPH_SUCCESS) {
        return GRAPH_PARAM_INVALID;
    }
    OP_LOGD(context->GetNodeName(), "A16W4 enable msd. groupListType: %u. ", groupListType_);
    GMMTilingData tilingDataA16W4;
    tilingDataA16W4.mmTilingData.set_dbL0B(1);
    tilingDataA16W4.mmTilingData.set_dbL0C(1);
    tilingDataA16W4.mmTilingData.set_stepKa(1);
    tilingDataA16W4.mmTilingData.set_stepKb(1);
    tilingDataA16W4.mmTilingData.set_depthA1(1);
    tilingDataA16W4.mmTilingData.set_depthB1(1);
    tilingDataA16W4.mmTilingData.set_baseK(512);    // MM tiling设置初始值512
    tilingDataA16W4.mmTilingData.set_baseN(256);    // MM tiling设置初始值256
    tilingDataA16W4.mmTilingData.set_stepM(1);
    tilingDataA16W4.mmTilingData.set_stepN(1);

    tilingDataA16W4.gmmBaseParams.set_coreNum(compileInfoPtr->aicNum);
    tilingDataA16W4.gmmBaseParams.set_groupNum(groupNum_);
    tilingDataA16W4.gmmBaseParams.set_totalInGroup(mSize);
    tilingDataA16W4.gmmBaseParams.set_k(kSize);
    tilingDataA16W4.gmmBaseParams.set_n(
        context->GetDynamicInputTensor(ANTIQUANT_SCALE_INDEX, 0)->GetStorageShape().GetDim(2));  // 第2维是n轴
    tilingDataA16W4.gmmBaseParams.set_vBaseM(0);
    tilingDataA16W4.gmmBaseParams.set_ubCalSize(0);
    tilingDataA16W4.gmmBaseParams.set_ubRestBytes(0);
    tilingDataA16W4.gmmBaseParams.set_parallNum(1);
    tilingDataA16W4.gmmBaseParams.set_quantGroupNum(antiquantGroupNum);
    tilingDataA16W4.gmmBaseParams.set_m(mSize);
    tilingDataA16W4.gmmBaseParams.set_withOffset(0);
    context->SetBlockDim(compileInfoPtr->aicNum);
    context->SetScheduleMode(1);  // set as batchmod for template using SyncAll
    context->SetTilingKey(GET_TPL_TILING_KEY(
        GMM_TPL_BF16, GMM_TPL_INT4, GMM_TPL_BF16, 0, 1, groupListType_, 0, GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE,
        GROUPED_MATMUL_A16W4_KERNEL_TEMPLATE_MSD_ANTIQUANT_GS32, GROUPED_MATMUL_AIV_AIC_RATIO_2, 0));
    tilingDataA16W4.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingDataA16W4.GetDataSize());

    size_t *workspaces = context->GetWorkspaceSizes(1);  // get second variable
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaces);     // check workspaces is not null
    workspaces[0] = SYS_WORKSPACE_SIZE;                  // default size
    // A矩阵原始空间需要占用2Byte的m*k
    uint64_t aUnfoldSize = 2 * mSize * kSize;
    // max需要再ub上做32B对齐。cache line为512B, 整体占用空间需要做cache line对齐
    uint64_t aMaxSize = CeilDiv(mSize * 32, 512) * 512;
    // 每核预留weight的空间大小为512*1024Byte，同时开2 db，保证流水
    uint64_t weightS8Size = compileInfoPtr->aicNum * 512 * 1024 * 2;
    // 每核预留C矩阵的shape为32 * 320。
    uint64_t singeCF32Size = compileInfoPtr->aicNum * 32 * 320 * sizeof(float);
    // 一轮计算k轴会计算1024的长度，groupsize = 32时，需要预留32块C矩阵空间，同时开db保证流水
    uint64_t totalCF32Size = singeCF32Size * 1024 / 32 * 2;
    workspaces[0] += static_cast<size_t>(aUnfoldSize + aMaxSize + weightS8Size + totalCF32Size);
    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling
