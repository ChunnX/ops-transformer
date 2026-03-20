/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file rotate_dynamic_quant_tiling.cpp
 * \brief Tiling计算实现
 */

#include "rotate_dynamic_quant_tiling.h"
#include "../../../common/include/tiling_base/tiling_templates_registry.h"
#include "register/op_def_registry.h"
#include "platform/platform_infos_def.h"
#include "../../../common/include/err/ops_err.h"

namespace optiling {

REGISTER_OPS_TILING_TEMPLATE(RotateDynamicQuant, RotateDynamicQuantBaseTiling, 1000);

ge::graphStatus RotateDynamicQuantBaseTiling::GetInputShape()
{
    auto xTensor = context_->GetInputTensor(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xTensor);

    auto xShape = xTensor->GetStorageShape();
    mDim_ = static_cast<uint64_t>(xShape.GetDim(0));
    nDim_ = static_cast<uint64_t>(xShape.GetDim(1));
    totalLength_ = static_cast<uint64_t>(xShape.GetShapeSize());

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RotateDynamicQuantBaseTiling::ParseInputAndAttr()
{
    // 获取输入tensor信息
    OP_CHECK_RET(GetInputShape() != ge::GRAPH_SUCCESS, return ge::GRAPH_FAILED);

    // 获取平台信息
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

    // 获取dst_type属性值
    auto attrs = context_->GetAttrs();
    if (attrs != nullptr) {
        auto dstTypePtr = attrs->GetAttrPointer<int64_t>(0);
        if (dstTypePtr != nullptr) {
            dstType_ = *dstTypePtr;
        } else {
            dstType_ = 0;
        }
    } else {
        dstType_ = 0;
    }

    return ge::GRAPH_SUCCESS;
}

void RotateDynamicQuantBaseTiling::FillTilingData()
{
    tilingData_.set_coreNum(blockDim_);
    tilingData_.set_totalLength(totalLength_);
    tilingData_.set_mDim(mDim_);
    tilingData_.set_nDim(nDim_);
    tilingData_.set_dstType(dstType_);
}

ge::graphStatus RotateDynamicQuantBaseTiling::TilingProcess()
{
    // 计算workspace大小
    size_t userWorkspaceSize = 0;
    size_t systemWorkspaceSize = 16 * 1024 * 1024; // 16M

    // 设置tilingKey
    tilingKey_ = static_cast<uint64_t>(dstType_);

    workspaceSize_ = userWorkspaceSize + systemWorkspaceSize;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RotateDynamicQuantBaseTiling::DoOpTiling()
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

void RotateDynamicQuantBaseTiling::PrintTilingData()
{
    OP_LOGD(context_->GetNodeName(), "coreNum: [%u]", tilingData_.get_coreNum());
    OP_LOGD(context_->GetNodeName(), "totalLength: [%lu]", tilingData_.get_totalLength());
    OP_LOGD(context_->GetNodeName(), "mDim: [%lu]", tilingData_.get_mDim());
    OP_LOGD(context_->GetNodeName(), "nDim: [%lu]", tilingData_.get_nDim());
    OP_LOGD(context_->GetNodeName(), "dstType: [%ld]", tilingData_.get_dstType());
}

uint64_t RotateDynamicQuantBaseTiling::GetTilingKey() const
{
    return tilingKey_;
}

ge::graphStatus RotateDynamicQuantBaseTiling::PostTiling()
{
    OP_CHECK_IF(tilingData_.GetDataSize() % sizeof(uint64_t) != 0,
        OP_LOGE(context_->GetNodeName(), "tiling data size[%zu] is not aligned to 8", tilingData_.GetDataSize()),
        return ge::GRAPH_FAILED);
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData());
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    context_->SetBlockDim(tilingData_.get_coreNum());
    context_->SetScheduleMode(1);

    // 设置workspace
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_IF(workspaces == nullptr, OPS_REPORT_CUBE_INNER_ERR(context_->GetNodeName(), "workspaces is null"),
        return ge::GRAPH_FAILED);
    workspaces[0] = workspaceSize_;
    return ge::GRAPH_SUCCESS;
}

// Tiling函数入口
static ge::graphStatus TilingFuncRotateDynamicQuant(gert::TilingContext* context)
{
    OP_CHECK_IF(context == nullptr,
        OPS_REPORT_CUBE_INNER_ERR("[RotateDynamicQuantTilingFunc]", " context is null"),
        return ge::GRAPH_FAILED);

    return Ops::Transformer::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

// TilingPrepare函数
static ge::graphStatus TilingPrepareRotateDynamicQuant(gert::TilingParseContext* context)
{
    OP_CHECK_IF(context == nullptr,
                OPS_REPORT_CUBE_INNER_ERR("[TilingPrepareRotateDynamicQuant]", "context is null"),
                return ge::GRAPH_FAILED);
    fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr,
                OPS_REPORT_CUBE_INNER_ERR(context->GetNodeName(), "platformInfoPtr is null"),
                return ge::GRAPH_FAILED);

    auto compileInfoPtr = context->GetCompiledInfo<RotateDynamicQuantCompileInfo>();
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

IMPL_OP_OPTILING(RotateDynamicQuant)
    .Tiling(TilingFuncRotateDynamicQuant)
    .TilingParse<RotateDynamicQuantCompileInfo>(TilingPrepareRotateDynamicQuant);

} // namespace optiling
