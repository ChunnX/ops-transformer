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
 * \file moe_distribute_dispatch_tiling_arch22.cpp
 * \brief
 */

#include "moe_distribute_dispatch_tiling_arch22.h"

#include <queue>
#include <vector>
#include <dlfcn.h>
#include <fcntl.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <cmath>
#include <cstdint>
#include <string>
#include <sys/types.h>
#include "tiling/tiling_api.h"
#include "mc2_log.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "op_host/op_tiling/mc2_tiling_utils.h"
#include "../../../op_kernel/moe_distribute_dispatch_v2_tiling.h"
#include "../../../op_kernel/moe_distribute_dispatch_v2_tiling_key.h"
using namespace Mc2Tiling;

namespace {

}

namespace optiling {

uint64_t MoeDistributeDispatchV2TilingFuncA2A3::CalTilingKey(const bool isScales, const uint32_t quantMode,
    const uint32_t tpWorldSize, const bool isSetFullMeshV2, bool isLayered)
{
    uint32_t templateDispatch = TILINGKEY_NO_FULLMESH;
    bool tp = false;
    uint32_t tilingKeyQuantMode = quantMode;
    bool scaleMode = false;
    uint64_t tilingKey;
    uint32_t commMode = TILINGKEY_TPL_MTE;
    if (isScales) {
        scaleMode = true;
    }
    if (isSetFullMeshV2) {
        templateDispatch = TILINGKEY_ENABLE_FULLMESH;
    }
    if (isLayered) {
        templateDispatch = TILINGKEY_ENABLE_HIERARCHY;
    }
    if (tpWorldSize == MAX_TP_WORLD_SIZE) {
        tp = true;
    }
    tilingKey = GET_TPL_TILING_KEY(tp, tilingKeyQuantMode, scaleMode, 
                                            templateDispatch, commMode, TILINGKEY_TPL_A3);
    return tilingKey;
}

ge::graphStatus MoeDistributeDispatchV2TilingFuncA2A3::MoeDistributeDispatchTilingFuncImpl(gert::TilingContext* context)
{
    const char *nodeName = context->GetNodeName();
    OP_LOGI(nodeName, "Enter MoeDistributeDispatchA2 tiling func.");

    // 涉及SyncAll，设置batch mode模式，所有核同时启动 
    uint32_t batch_mode = 1U; 
    auto ret = context->SetScheduleMode(batch_mode); 
    GE_ASSERT_GRAPH_SUCCESS(ret);

    // 1. tilingData
    MoeDistributeDispatchA2TilingData *tilingData = context->GetTilingData<MoeDistributeDispatchA2TilingData>();
    OP_TILING_CHECK(tilingData == nullptr, VECTOR_INNER_ERR_REPORT_TILING(nodeName, "tilingData is nullptr."),
        return ge::GRAPH_FAILED);
    MoeDistributeDispatchA2Info& info = tilingData->moeDistributeDispatchInfo;

    bool isLayered = false;
    OP_TILING_CHECK(MoeDistributeDispatchA2CheckCommAlg(context, isLayered) != ge::GRAPH_SUCCESS,
        VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(), "MoeDistributeDispatchA2 CheckCommAlg Failed"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(MoeDistributeDispatchA2CheckShapeAndSetTiling(context, info, isLayered) != ge::GRAPH_SUCCESS,
        VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(), "MoeDistributeDispatchA2 CheckShapeAndSetTiling Failed"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(MoeDistributeDispatchA2CheckAttrAndSetTiling(context, info, isLayered) != ge::GRAPH_SUCCESS,
        VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(), "MoeDistributeDispatchA2 CheckAttrAndSetTiling Failed"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(MoeDistributeDispatchA2GetPlatformInfoAndSetTiling(context, info) != ge::GRAPH_SUCCESS,
        VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(), "MoeDistributeDispatchA2 GetPlatformInfoAndSetTiling Failed"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(MoeDistributeDispatchA2CheckWinSize(context, nodeName, info, isLayered) != ge::GRAPH_SUCCESS,
        VECTOR_INNER_ERR_REPORT_TILING(context->GetNodeName(), "MoeDistributeDispatchA2 CheckWinSize Failed"),
        return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t numBlocks = ascendcPlatform.CalcTschBlockDim(aivNum, 0, aivNum);
    context->SetBlockDim(numBlocks);
    uint32_t aicpuBlockDim = info.epWorldSize > RANK_NUM_PER_NODE_A2 ? mc2tiling::AICPU_NUM_BLOCKS_A2 : 1;
    context->SetAicpuBlockDim(aicpuBlockDim);

    uint64_t tilingKey = MoeDistributeDispatchA2CalcTilingKey(context, isLayered);
    context->SetTilingKey(tilingKey);
    // 2. workspace
    size_t *workSpaces = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workSpaces == nullptr, VECTOR_INNER_ERR_REPORT_TILING(nodeName, "workSpaces is nullptr."),
        return ge::GRAPH_FAILED);
    workSpaces[0] = SYSTEM_NEED_WORKSPACE + USER_WORKSPACE_A2;

    // 3. communication
    auto attrs = context->GetAttrs();
    auto group = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_GROUP_EP_INDEX));
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_WORLD_SIZE_INDEX);
    std::string algConfig = MoeDistributeCombineA2GetAlgConfig(*epWorldSizePtr, isLayered);
    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(group, static_cast<uint32_t>(18), algConfig); // opType=18 BatchWrite
    OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(tilingData->mc2InitTiling) != 0,
        OP_LOGE(nodeName, "mc2CcTilingConfig mc2InitTiling GetTiling failed"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(tilingData->mc2CcTiling) != 0,
        OP_LOGE(nodeName, "mc2CcTilingConfig mc2CcTiling GetTiling failed"), return ge::GRAPH_FAILED);

    OP_LOGI(nodeName, "Leave MoeDistributeDispatchA2 tiling func.");
    return ge::GRAPH_SUCCESS;
}

bool MoeDistributeDispatchV2TilingFuncA2A3::CheckTensorDataType(const gert::TilingContext *context, const char *nodeName,
    const bool isScales, const uint32_t quantMode, const bool isActiveMask, const bool hasElasticInfo,
    const bool isPerformance, DispatchV2Config &config)
{
    auto xDesc = context->GetInputDesc(config.xIndex);
    OP_TILING_CHECK((xDesc->GetDataType() != ge::DT_BF16) && (xDesc->GetDataType() != ge::DT_FLOAT16),
        OP_LOGE(nodeName, "x dataType is invalid, dataType should be bf16 or float16, but is %s.",
        Ops::Base::ToString(xDesc->GetDataType()).c_str()), return false);

    if (quantMode == static_cast<uint32_t>(QuantModeA5::PERTOKEN_DYNAMIC_QUANT)) {
        auto dynamicScalesDesc = context->GetOutputDesc(OUTPUT_DYNAMIC_SCALES_INDEX);
        OP_TILING_CHECK(dynamicScalesDesc == nullptr, OP_LOGE(nodeName, "dynamicScalesDesc is null."),
            return false);
        OP_TILING_CHECK(dynamicScalesDesc->GetDataType() != ge::DT_FLOAT,
            OP_LOGE(nodeName, "dynamicScales dataType is invalid, dataType should be float, but is %s.",
            Ops::Base::ToString(dynamicScalesDesc->GetDataType()).c_str()), return false);
    }
    if (isScales) {
        auto scalesDesc = context->GetOptionalInputDesc(config.scalesIndex);
        OP_TILING_CHECK(scalesDesc->GetDataType() != ge::DT_FLOAT,
            OP_LOGE(nodeName, "scales dataType is invalid, dataType should be float, but is %s.",
            Ops::Base::ToString(scalesDesc->GetDataType()).c_str()), return false);
    }
    auto expandXDesc = context->GetOutputDesc(OUTPUT_EXPAND_X_INDEX);
    if (quantMode != static_cast<uint32_t>(QuantModeA5::NON_QUANT)) {
        OP_TILING_CHECK(expandXDesc->GetDataType() != ge::DT_INT8,
            OP_LOGE(nodeName, "expandX dataType is invalid, dataType should be int8, but is %s.",
            Ops::Base::ToString(expandXDesc->GetDataType()).c_str()), return false);
    } else {
        OP_TILING_CHECK(expandXDesc->GetDataType() != xDesc->GetDataType(),
            OP_LOGE(nodeName, "expandX dataType is invalid, dataType should be equal to x dataType %s, but is %s.",
            Ops::Base::ToString(xDesc->GetDataType()).c_str(), Ops::Base::ToString(expandXDesc->GetDataType()).c_str()),
            return false);
    }

    OP_TILING_CHECK(!CheckCommomOtherInputTensorDataType(context, nodeName, isActiveMask, hasElasticInfo, isPerformance, config),
        OP_LOGE(nodeName, "CheckCommomOtherInputTensorDataType failed."), return false);
    OP_TILING_CHECK(!CheckCommomOutputTensorDataType(context, nodeName),
        OP_LOGE(nodeName, "CheckCommomOutputTensorDataType failed."), return false);

    return true;
}

ge::graphStatus MoeDistributeDispatchV2TilingFuncA2A3::MoeDistributeDispatchV2TilingFunc(gert::TilingContext* context)
{
    std::string socVersion = mc2tiling::GetSocVersion(context);
    NpuArch npuArch = mc2tiling::GetNpuArch(context);
    ge::graphStatus ret;
    if (socVersion == "Ascend910B") {
        ret = MoeDistributeDispatchTilingFuncImpl(context);
    } else {
        DispatchV2Config config;
        config.xIndex = 0U; // 0: 根据dispatchV2算子原型标志位初始化groupEp索引
        config.expertIdsIndex = 1U; // 1: 根据dispatchV2算子原型标志位初始化expertIds索引
        config.scalesIndex = 2U; // 2: 根据dispatchV2算子原型标志位初始化scales索引
        config.xActiveMaskIndex = 3U; // 3: 根据dispatchV2算子原型标志位初始化xActiveMask索引
        config.expertScalesIndex = 4U; // 4: 根据dispatchV2算子原型标志位初始化expertScales索引
        config.elasticInfoIndex = 5U; // 5: 根据dispatchV2算子原型标志位初始化elasticInfo索引
        config.performanceInfoIndex = 6U; // 6: 根据dispatchV2算子原型标志位初始化performanceInfo索引
        config.attrGroupEpIndex = 0; // 0: 根据dispatchV2算子原型标志位初始化groupEp索引
        config.attrEpWorldSizeIndex = 1; // 1: 根据dispatchV2算子原型标志位初始化epWorldSize索引
        config.attrEpRankIdIndex = 2; // 2: 根据dispatchV2算子原型标志位初始化epRankId索引
        config.attrMoeExpertNumIndex = 3;  // 3: 根据dispatchV2算子原型标志位初始化moeExpertNum索引
        config.attrGroupTpIndex = 4; // 4: 根据dispatchV2算子原型标志位初始化groupTp索引
        config.attrTpWorldSizeIndex = 5; // 5: 根据dispatchV2算子原型标志位初始化tpWorldSize索引
        config.attrTpRankIdIndex = 6; // 6: 根据dispatchV2算子原型标志位初始化tpRankId索引
        config.attrExpertSharedTypeIndex = 7; // 7: 根据dispatchV2算子原型标志位初始化expertSharedType索引
        config.attrSharedExpertNumIndex = 8; // 8: 根据dispatchV2算子原型标志位初始化sharedExpertNum索引
        config.attrSharedExpertRankNumIndex = 9; // 9: 根据dispatchV2算子原型标志位初始化sharedExpertRankNum索引
        config.attrQuantModeIndex = 10; // 10: 根据dispatchV2算子原型标志位初始化quantMode索引
        config.attrGlobalBsIndex = 11; // 11: 根据dispatchV2算子原型标志位初始化globalBs索引
        config.attrExpertTokenNumsTypeIndex = 12; // 12: 根据dispatchV2算子原型标志位初始化expertTokenNumType索引
        config.attrCommAlgIndex = 13; // 13: 根据dispatchV2算子原型标志位初始化commAlg索引
        config.attrZeroExpertNumIndex = 14; // 14: 根据dispatchV2算子原型标志位初始化zeroExpertNumIndex索引
        config.attrCopyExpertNumIndex = 15; // 15: 根据dispatchV2算子原型标志位初始化copyExpertNumIndex索引
        config.attrConstExpertNumIndex = 16; // 16: 根据dispatchV2算子原型标志位初始化constExpertNumIndex索引
        config.isMc2Context = false;
        ret = MoeDistributeDispatchA3TilingFuncImplPublic(context, config);
    }
    return ret;
}

ge::graphStatus MoeDistributeDispatchV2TilingFuncA2A3::CheckQuantModeMatchScales(gert::TilingContext *context, const char *nodeName, bool isScales,
        uint32_t quantMode, DispatchV2Config &config)
{
    OP_TILING_CHECK(quantMode == static_cast<uint32_t>(QuantModeA5::STATIC_QUANT), OP_LOGE(nodeName, "cannot support static quant now."),
            return ge::GRAPH_FAILED);
    OP_TILING_CHECK((isScales && (quantMode == static_cast<uint32_t>(QuantModeA5::NON_QUANT))) || ((!isScales) && (quantMode == static_cast<uint32_t>(QuantModeA5::STATIC_QUANT))),
        OP_LOGE(nodeName, "quant mode and scales not match, isScales is %d, quantMode is %u.",
        static_cast<int32_t>(isScales), quantMode), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeDispatchV2TilingFuncA2A3::CheckCommAlgPtr(const char* commAlgPtr, const char *nodeName)
{
    OP_TILING_CHECK(((strlen(commAlgPtr) != 0) && (strcmp(commAlgPtr, "fullmesh_v1") != 0) &&
        (strcmp(commAlgPtr, "fullmesh_v2")  != 0) && (strcmp(commAlgPtr, "hierarchy") != 0)),
        OP_LOGE(nodeName, "Attr commAlg is invalid, current only support hierarchy, fullmesh_v1 and fullmesh_v2, but got commAlg = %s.", 
        commAlgPtr), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeDispatchV2TilingFuncA2A3::CheckQuantModePtr(const int64_t* quantModePtr)
{
    OP_TILING_CHECK((*quantModePtr < static_cast<int64_t>(QuantModeA5::NON_QUANT)) ||
    (*quantModePtr > static_cast<int64_t>(QuantModeA5::PERTOKEN_DYNAMIC_QUANT)),
    OP_LOGE(nodeName, "quantMode is invalid, only support [0, %ld], but got quantMode=%ld.",
    static_cast<int64_t>(QuantModeA5::PERTOKEN_DYNAMIC_QUANT), *quantModePtr), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

struct MoeDistributeDispatchCompileInfo {};
static ge::graphStatus TilingParseForMoeDistributeDispatchV2(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

static MoeDistributeDispatchV2TilingFuncImpl(gert::TilingContext* context)
{
    MoeDistributeDispatchV2TilingFuncA2A3 impl;
    return impl.MoeDistributeDispatchV2TilingFunc(context);
}

IMPL_OP_OPTILING(MoeDistributeDispatchV2)
    .Tiling(MoeDistributeDispatchV2TilingFunc)
    .TilingParse<MoeDistributeDispatchCompileInfo>(TilingParseForMoeDistributeDispatchV2);
} // namespace optiling
