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
 * \file moe_distribute_combine_tiling_arch22.cpp
 * \brief
 */

#include "moe_distribute_combine_tiling_arch22.h"

namespace optiling {
static ge::graphStatus MoeDistributeCombineA2TilingFuncImpl(gert::TilingContext* context, const CombineV2Config& config)
{
    const char *nodeName = context->GetNodeName();
    OP_LOGI(nodeName, "Enter MoeDistributeCombineA2 tiling func.");
    // 涉及SyncAll，设置batch mode模式，所有核同时启动 
    uint32_t batch_mode = 1U; 
    auto ret = context->SetScheduleMode(batch_mode); 
    GE_ASSERT_GRAPH_SUCCESS(ret);

    // tilingData
    MoeDistributeCombineA2TilingData *tilingData = context->GetTilingData<MoeDistributeCombineA2TilingData>();
    OP_TILING_CHECK(tilingData == nullptr, VECTOR_INNER_ERR_REPORT_TILING(nodeName, "tilingData is nullptr."),
        return ge::GRAPH_FAILED);
    MoeDistributeCombineA2Info &info = tilingData->moeDistributeCombineInfo;

    bool isLayered = false;
    OP_TILING_CHECK(MoeDistributeCombineCheckCommAlg(context, isLayered, config) != ge::GRAPH_SUCCESS,
        VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(), "MoeDistributeCombineA2 CheckCommAlg Failed"),
        return ge::GRAPH_FAILED);
    int32_t commQuantMode = 0;
    OP_TILING_CHECK(MoeDistributeCombineA2CheckShapeAndSetTiling(context, info, isLayered, config) != ge::GRAPH_SUCCESS,
        VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(), "MoeDistributeCombineA2 CheckShapeAndSetTiling Failed"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(MoeDistributeCombineA2CheckAttrAndSetTiling(context, info, commQuantMode, isLayered, config) != ge::GRAPH_SUCCESS,
        VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(), "MoeDistributeCombineA2 CheckAttrAndSetTiling Failed"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(MoeDistributeCombineA2GetPlatformInfoAndSetTiling(context, info) != ge::GRAPH_SUCCESS,
        VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(), "MoeDistributeCombineA2 GetPlatformInfoAndSetTiling Failed"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(MoeDistributeCombineA2CheckWinSize(context, nodeName, info, isLayered) != ge::GRAPH_SUCCESS,
        VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(), "MoeDistributeCombineA2 CheckWinSize Failed"),
        return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t numBlocks = ascendcPlatform.CalcTschBlockDim(aivNum, 0, aivNum);
    context->SetBlockDim(numBlocks);
    uint32_t aicpuBlockDim = info.epWorldSize > RANK_NUM_PER_NODE_A2 ? mc2tiling::AICPU_NUM_BLOCKS_A2 : 1;
    context->SetAicpuBlockDim(aicpuBlockDim);

    uint64_t tilingKey = MoeDistributeCombineA2CalcTilingKey(isLayered, commQuantMode);
    context->SetTilingKey(tilingKey);
    // 2. workspace
    size_t *workSpaces = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workSpaces == nullptr, VECTOR_INNER_ERR_REPORT_TILING(nodeName, "workSpaces is nullptr."),
        return ge::GRAPH_FAILED);
    // SYSTEM_NEED_WORKSPACE + userWorkspaceSize
    workSpaces[0] = SYSTEM_NEED_WORKSPACE + static_cast<size_t>(info.moeExpertNum * sizeof(uint32_t) * 2U);

    // 3. communication
    auto attrs = context->GetAttrs();
    auto group = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_GROUP_EP_INDEX));
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_WORLD_SIZE_INDEX);
    std::string algConfig = MoeDistributeCombineA2GetAlgConfig(*epWorldSizePtr, isLayered);
    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(group, static_cast<uint32_t>(18), algConfig); // opType=18
    OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(tilingData->mc2InitTiling) != 0,
        OP_LOGE(nodeName, "mc2CcTilingConfig mc2tiling GetTiling mc2InitTiling failed"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(tilingData->mc2CcTiling) != 0,
        OP_LOGE(nodeName, "mc2CcTilingConfig mc2tiling GetTiling mc2CcTiling failed"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeCombineV2TilingFuncA2A3::MoeDistributeCombineTilingFuncImpl(gert::TilingContext* context, const CombineV2Config& config)
{
    std::string socVersion = mc2tiling::GetSocVersion(context);
    ge::graphStatus ret;

    if (socVersion == "Ascend910B") {
        ret = MoeDistributeCombineA2TilingFuncImpl(context, config);
    } else {
        ret = MoeDistributeCombineA3TilingFuncImpl(context, config);
    }

    return ret;
}

struct MoeDistributeCombineCompileInfo {};
ge::graphStatus TilingParseForMoeDistributeCombineV2(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MoeDistributeCombineV2TilingFuncA2A3(gert::TilingContext* context)
{
    MoeDistributeCombineV2TilingFuncA2A3 impl;
    return impl.MoeDistributeCombineV2TilingFunc(gert::TilingContext* context);
}

IMPL_OP_OPTILING(MoeDistributeCombineV2)
    .Tiling(MoeDistributeCombineV2TilingFuncA2A3)
    .TilingParse<MoeDistributeCombineCompileInfo>(TilingParseForMoeDistributeCombineV2);

}