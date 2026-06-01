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
 * \file norm_flash_attention_tiling.cpp
 * \brief Tiling计算实现
 */

#include "norm_flash_attention_tiling.h"
#include "../../../common/include/tiling_base/tiling_templates_registry.h"
#include "register/op_def_registry.h"
#include "platform/platform_infos_def.h"
#include "../../../common/include/err/ops_err.h"

namespace optiling {

REGISTER_OPS_TILING_TEMPLATE(NormFlashAttention, NormFlashAttentionBaseTiling, 1000);

ge::graphStatus NormFlashAttentionBaseTiling::ParseInputAndAttr()
{
    auto queryTensor = context_->GetInputTensor(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, queryTensor);
    totalLength_ = queryTensor->GetStorageShape().GetShapeSize();

    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        OP_LOGE(context_->GetNodeName(), "get platform info failed");
        return ge::GRAPH_FAILED;
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint64_t ubSize, l1Size, l0CSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0CSize);
    blockDim_ = ascendcPlatform.GetCoreNumAic();

    return ge::GRAPH_SUCCESS;
}

void NormFlashAttentionBaseTiling::FillTilingData()
{
    tilingData_.set_coreNum(blockDim_);
    tilingData_.set_totalLength(totalLength_);
}

ge::graphStatus NormFlashAttentionBaseTiling::TilingProcess()
{
    tilingKey_ = 0UL;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NormFlashAttentionBaseTiling::DoOpTiling()
{
    auto inputDesc = context_->GetInputDesc(0);
    if (inputDesc == nullptr) {
        OP_LOGE(context_->GetNodeName(), "invalid input pointer");
        return ge::GRAPH_FAILED;
    }

    if (ParseInputAndAttr() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (TilingProcess() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    FillTilingData();
    PrintTilingData();

    return ge::GRAPH_SUCCESS;
}

void NormFlashAttentionBaseTiling::PrintTilingData()
{
    OP_LOGD(context_->GetNodeName(), "coreNum: [%d]", tilingData_.get_coreNum());
    OP_LOGD(context_->GetNodeName(), "totalLength: [%d]", tilingData_.get_totalLength());
}

uint64_t NormFlashAttentionBaseTiling::GetTilingKey() const
{
    return tilingKey_;
}

ge::graphStatus NormFlashAttentionBaseTiling::PostTiling()
{
    OP_CHECK_IF(tilingData_.GetDataSize() % sizeof(uint64_t) != 0,
        OP_LOGE(context_->GetNodeName(), "tiling data size[%zu] is not aligned to 8", tilingData_.GetDataSize()),
        return ge::GRAPH_FAILED);
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData());
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    context_->SetBlockDim(tilingData_.get_coreNum());
    context_->SetScheduleMode(1);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingFuncNormFlashAttention(gert::TilingContext* context)
{
    OP_CHECK_IF(context == nullptr,
        OPS_REPORT_CUBE_INNER_ERR("[NormFlashAttentionTilingFunc]", " context is null"),
        return ge::GRAPH_FAILED);

    return Ops::Transformer::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingPrepareNormFlashAttention(gert::TilingParseContext* context)
{
    OP_CHECK_IF(context == nullptr,
                OPS_REPORT_CUBE_INNER_ERR("[TilingPrepareNormFlashAttention]", "context is null"),
                return ge::GRAPH_FAILED);
    fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr,
                OPS_REPORT_CUBE_INNER_ERR(context->GetNodeName(), "platformInfoPtr is null"),
                return ge::GRAPH_FAILED);

    auto compileInfoPtr = context->GetCompiledInfo<NormFlashAttentionCompileInfo>();
    OP_CHECK_IF(compileInfoPtr == nullptr,
                OPS_REPORT_CUBE_INNER_ERR(context->GetNodeName(), "compileInfoPtr is null"),
                return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);

    compileInfoPtr->aicNum = ascendcPlatform.GetCoreNumAic();
    compileInfoPtr->aivNum = ascendcPlatform.GetCoreNumAiv();

    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfoPtr->l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, compileInfoPtr->l2Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, compileInfoPtr->l0ASize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, compileInfoPtr->l0BSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfoPtr->l0CSize);

    OP_LOGI(context->GetNodeName(),
            "parse compile info success l1Size:%lu, l2Size:%lu, coreNum:%lu",
            compileInfoPtr->l1Size,
            compileInfoPtr->l2Size,
            compileInfoPtr->aicNum);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(NormFlashAttention)
    .Tiling(TilingFuncNormFlashAttention)
    .TilingParse<NormFlashAttentionCompileInfo>(TilingPrepareNormFlashAttention);

} // namespace optiling
