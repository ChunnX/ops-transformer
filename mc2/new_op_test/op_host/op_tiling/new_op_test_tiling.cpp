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
 * \file new_op_test_tiling.cpp
 * \brief host侧tiling实现
 */
#include <register/op_def_registry.h>
#include "../../op_kernel/new_op_test_tiling_key.h"
#include "../../op_kernel/new_op_test_tiling_data.h"
#include "common/new_op_test_util_tiling.h"

namespace MC2Tiling {

using namespace AscendC;
using namespace ge;

/**
 * @brief 打印tilingData
 * @param context: 框架根据input，output，attrs等信息生成tiling需要的context
 * @param tilingData: 框架根据context的opName匹配tiling模板，计算产生的tilingData
 * @return
 */
static void PrintTilingDataInfo(gert::TilingContext *context, NewOpTestTilingData &tilingData)
{
    const char *nodeName = context->GetNodeName();
    OP_LOGD(nodeName, "bs is %lu in new_op_test.", tilingData.newOpTestTilingInfo.bs);
    OP_LOGD(nodeName, "hiddenSize is %u in new_op_test.", tilingData.newOpTestTilingInfo.hiddenSize);
    OP_LOGD(nodeName, "scaleHiddenSize is %lu in new_op_test.",
            tilingData.newOpTestTilingInfo.scaleHiddenSize);
    OP_LOGD(nodeName, "aivNum is %u in new_op_test.", tilingData.newOpTestTilingInfo.aivNum);
}

/**
 * @brief 设置hcomm参数
 * @param context: 框架根据input，output，attrs等信息生成tiling需要的context
 * @param tilingData: 框架根据context的opName匹配tiling模板，计算产生的tilingData
 * @param runInfo: 封装的doTiling所需要的参数
 * @return
 */
static ge::graphStatus SetHcommCfg(const gert::TilingContext *context, NewOpTestTilingData *tilingData,
                        const TilingRunInfo &runInfo)
{
    const char *nodeName = context->GetNodeName();
    OP_LOGD(nodeName, "group is %s in new_op_test.", runInfo.group.c_str());
    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(runInfo.group, OP_TYPE_ALL_TO_ALL,
                                                 "AlltoAll=level0:fullmesh;level1:pairwise");
    // MTE方式必要适配
    mc2CcTilingConfig.SetCommEngine(AIV_TYPE);
    OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(tilingData->mc2InitTiling) != 0,
        OP_LOGE(nodeName, "mc2CcTilingConfig mc2InitTiling GetTiling failed"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(tilingData->mc2CcTiling) != 0,
        OP_LOGE(nodeName, "mc2CcTilingConfig mc2CcTiling GetTiling failed"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 设置tilingData
 * @param context: 框架根据input，output，attrs等信息生成tiling需要的context
 * @param tilingData: 框架根据context的opName匹配tiling模板，计算产生的tilingData
 * @return
 */
static void SetTilingData(gert::TilingContext *context, NewOpTestTilingData &tilingData)
{
    fe::PlatFormInfos *platformInfoPtr = context->GetPlatformInfo();
    platform_ascendc::PlatformAscendC ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    // set tilingData
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    context->SetBlockDim(ascendcPlatform.CalcTschBlockDim(aivNum, 0, aivNum));
    tilingData.newOpTestTilingInfo.aivNum = aivNum;
    uint64_t xValueBS = context->GetInputShape(X_INDEX)->GetStorageShape().GetDim(DIM_ZERO);
    uint64_t xValueH = context->GetInputShape(X_INDEX)->GetStorageShape().GetDim(DIM_ONE);
    uint64_t scalesValueH = context->GetInputShape(SCALES_INDEX)->GetStorageShape().GetDim(DIM_ONE);
    // 3d场景，context->GetInputShape在函数CheckInputTensorDim中已经校验
    if (context->GetInputShape(X_INDEX)->GetStorageShape().GetDimNum() == THREE_DIMS) {
        xValueBS = xValueBS * context->GetInputShape(X_INDEX)->GetStorageShape().GetDim(DIM_ONE);
        xValueH = context->GetInputShape(X_INDEX)->GetStorageShape().GetDim(DIM_TWO);
        scalesValueH = context->GetInputShape(SCALES_INDEX)->GetStorageShape().GetDim(DIM_TWO);
    }
    tilingData.newOpTestTilingInfo.bs = xValueBS;
    tilingData.newOpTestTilingInfo.hiddenSize = xValueH;
    tilingData.newOpTestTilingInfo.scaleHiddenSize = scalesValueH;
    tilingData.newOpTestTilingInfo.totalWinSize = mc2tiling::Mc2TilingUtils::GetMaxWindowSize();
}

/**
 * @brief 设置tilingKey
 * @param context: 框架根据input，output，attrs等信息生成tiling需要的context
 * @return
 */
static void SetTilingKey(gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();
    // 设置tilingKey模板参数
    const uint64_t tilingKey = GET_TPL_TILING_KEY(MTE_COMM);
    context->SetTilingKey(tilingKey);
    OP_LOGD(nodeName, "tilingKey is [%lu] in new_op_test.", tilingKey);
}

/**
 * @brief new_op_test算子的tiling函数
 * @param context: 框架根据input，output，attrs等信息生成tiling需要的context
 * @return
 */
static ge::graphStatus NewOpTestTilingFunc(gert::TilingContext *context)
{
    OP_TILING_CHECK(context == nullptr,
                    OP_LOGE("new_op_test", "failed to get tiling context in new_op_test."),
                    return ge::GRAPH_FAILED);
    const char *nodeName = context->GetNodeName();
    OP_TILING_CHECK(nodeName == nullptr,
                    OP_LOGE("new_op_test", "failed to get nodeName in new_op_test."),
                    return ge::GRAPH_FAILED);

    NewOpTestTilingData *tilingData = context->GetTilingData<NewOpTestTilingData>();
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE(nodeName, "tilingData is nullptr in new_op_test."),
                    return ge::GRAPH_FAILED);

    TilingRunInfo runInfo = {};
    OP_TILING_CHECK(NewOpTestUtilTiling::CheckNpuArch(context) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "NpuArch is invalid in new_op_test."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(NewOpTestUtilTiling::CheckTilingFunc(context, runInfo, OpType::OP_NEW_OP_TEST) !=
                        ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "tiling check failed in new_op_test."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(SetHcommCfg(context, tilingData, runInfo) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "SetHCommCfg failed."), return ge::GRAPH_FAILED);
    SetTilingData(context, *tilingData);
    SetTilingKey(context);
    PrintTilingDataInfo(context, *tilingData);
    return ge::GRAPH_SUCCESS;
}

struct NewOpTestCompileInfo {};
 
ge::graphStatus TilingParseForNewOpTest(gert::TilingParseContext *context) {
    (void)context;
  	return ge::GRAPH_SUCCESS;
}
 
IMPL_OP_OPTILING(NewOpTest)
    .Tiling(NewOpTestTilingFunc)
    .TilingParse<NewOpTestCompileInfo>(TilingParseForNewOpTest);

} // namespace MC2Tiling
