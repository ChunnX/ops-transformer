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
 * \file qbmm_reduce_scatter_add_rms_norm_cast_tiling.cpp
 * \brief host侧tiling实现
 */

#include <register/op_def_registry.h>
#include "tiling/mc2_tiling_utils.h"
#include "util/math_util.h"
#include "qbmm_reduce_scatter_add_rms_norm_cast_tiling_check.h"
#include "../../op_kernel/qbmm_reduce_scatter_add_rms_norm_cast_tiling_data.h"
#include "qbmm_reduce_scatter_add_rms_norm_cast_tiling_check.h"

#include "mc2_log.h"
using namespace AscendC;
using namespace ge;
namespace MC2Tiling {
constexpr uint32_t OP_TYPE_ALL_TO_ALL = 8;
constexpr uint32_t AIV_TYPE = 3;
constexpr uint32_t BUFFER_NUM = 1;

// matmul tiling 切分
constexpr int32_t SINGLE_CORE_M = 128;
constexpr int32_t SINGLE_CORE_N = 256;
constexpr int32_t SINGLE_CORE_K = 2560;
constexpr int32_t BASE_M = 128;
constexpr int32_t BASE_N = 256;
constexpr int32_t BASE_K = 2560;
constexpr uint32_t SYSTEM_NEED_WORKSPACE = 16U * 1024 * 1024;

using namespace AscendC;
using namespace ge;



/**
 * @brief 设置hcomm参数
 * @param context: 框架根据input，output，attrs等信息生成tiling需要的context
 * @param tilingData: 框架根据context的opName匹配tiling模板，计算产生的tilingData
 * @return
 */
static ge::graphStatus SetHcommCfg(const gert::TilingContext *context, QbmmReduceScatterAddRmsNormCastTilingData *tilingData,
                        const std::string &group)
{
    const char *nodeName = context->GetNodeName();
    OP_LOGD(nodeName, "group is %s in qbmm_reduce_scatter_add_rms_norm_cast.", group.c_str());
    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(group, OP_TYPE_ALL_TO_ALL,
                                                 "AlltoAll=level0:fullmesh;level1:pairwise");
    // MTE方式必要适配
    mc2CcTilingConfig.SetCommEngine(AIV_TYPE);
    OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(tilingData->mc2InitTiling) != 0,
        OP_LOGE(nodeName, "mc2CcTilingConfig mc2InitTiling GetTiling failed"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(tilingData->mc2CcTiling) != 0,
        OP_LOGE(nodeName, "mc2CcTilingConfig mc2CcTiling GetTiling failed"), return ge::GRAPH_FAILED);
    OP_LOGD("QbmmReduceScatterAddRmsNormCast set HCCL success.");
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 设置tilingData
 * @param context: 框架根据input，output，attrs等信息生成tiling需要的context
 * @param tilingData: 框架根据context的opName匹配tiling模板，计算产生的tilingData
 * @return
 */
static void SetTilingData(gert::TilingContext *context, QbmmReduceScatterAddRmsNormCastTilingData &tilingData)
{
    const char *nodeName = context->GetNodeName();
    fe::PlatFormInfos *platformInfoPtr = context->GetPlatformInfo();
    platform_ascendc::PlatformAscendC ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    // set corenum
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    OP_LOGD(nodeName, "aicNum is %u, aivNum is %u in qbmm_reduce_scatter_add_rms_norm_cast.", aicNum, aivNum);
    context->SetBlockDim(aicNum);   // TODO:
    tilingData.qbmmReduceScatterAddRmsNormCastTilingInfo.aicNum = aicNum;
    tilingData.qbmmReduceScatterAddRmsNormCastTilingInfo.aivNum = aivNum;
    ge::DataType gammaDtype = context->GetInputDesc(GAMMA_INDEX)->GetDataType();
    if (gammaDtype == ge::DT_FLOAT) {
        tilingData.qbmmReduceScatterAddRmsNormCastTilingInfo.isGammaBf16 = false;
    } else if (gammaDtype == ge::DT_BF16) {
        tilingData.qbmmReduceScatterAddRmsNormCastTilingInfo.isGammaBf16 = true;
    }
    OP_LOGD(nodeName, "gamma type is %s\n.", Ops::Base::ToString(gammaDtype).c_str());
}

/**
 * @brief 设置tilingKey, 先使用老的方式生成tilingkey
 * @param context: 框架根据input，output，attrs等信息生成tiling需要的context
 * @return
 */
static void SetTilingKey(gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();
    ge::Format x2Format = static_cast<ge::Format>(ge::GetPrimaryFormat(context->GetInputDesc(X2_INDEX)->GetStorageFormat()));
    if (x2Format == ge::FORMAT_ND) {
        context->SetTilingKey(0);
    } else if (x2Format == ge::FORMAT_FRACTAL_NZ) {
        context->SetTilingKey(1);
    }
}

static ge::graphStatus SetWorkSpace(gert::TilingContext *context, QbmmReduceScatterAddRmsNormCastTilingData *tilingData)
{
    const char *nodeName = context->GetNodeName();

    const gert::StorageShape *x1Shape = context -> GetInputShape(0);
    size_t *workSpaces = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workSpaces == nullptr, OP_LOGE(nodeName, "workSpaces is nullptr."), return ge::GRAPH_FAILED);
    uint32_t M = tilingData->qbmmReduceScatterAddRmsNormCastTilingInfo.M;
    uint32_t N = tilingData->qbmmReduceScatterAddRmsNormCastTilingInfo.N;
    workSpaces[0] = SYSTEM_NEED_WORKSPACE + M * N * sizeof(bfloat16_t);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetTCubeTiling(
    gert::TilingContext *context, QbmmReduceScatterAddRmsNormCastTilingData *tilingData)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    matmul_tiling::MultiCoreMatmulTiling mmTiling(*ascendcPlatform);
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    const char *nodeName = context->GetNodeName();
    const gert::StorageShape *x1Shape = context -> GetInputShape(0);
    const gert::StorageShape *x2Shape = context -> GetInputShape(1);
    uint32_t M = x1Shape->GetStorageShape().GetDim(0);
    uint32_t Ka = x1Shape->GetStorageShape().GetDim(1);
    tilingData->qbmmReduceScatterAddRmsNormCastTilingInfo.M = M;
    tilingData->qbmmReduceScatterAddRmsNormCastTilingInfo.Ka = Ka;
    uint32_t N;

    const bool *transposeX2Ptr = attrs->GetAttrPointer<bool>(TRANSPOSE_X2_INDEX);
    bool isAtrans = false;
    bool isBtrans = *transposeX2Ptr;

    mmTiling.SetDim(1);
    mmTiling.SetAType(matmul_tiling::TPosition::GM,
        matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_INT8, isAtrans);
    ge::Format x2Format = static_cast<ge::Format>(ge::GetPrimaryFormat(context->GetInputDesc(X2_INDEX)->GetStorageFormat()));
    if (x2Format == ge::FORMAT_ND) {
        mmTiling.SetBType(matmul_tiling::TPosition::GM,
        matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_INT8, isBtrans);
        N = x2Shape->GetStorageShape().GetDim(1);
    } else if (x2Format == ge::FORMAT_FRACTAL_NZ) {
        mmTiling.SetBType(matmul_tiling::TPosition::GM,
        matmul_tiling::CubeFormat::NZ, matmul_tiling::DataType::DT_INT8, isBtrans);
        N = x2Shape->GetStorageShape().GetDim(0) * x2Shape->GetStorageShape().GetDim(3);
    }
    tilingData->qbmmReduceScatterAddRmsNormCastTilingInfo.N = N;
    mmTiling.SetCType(matmul_tiling::TPosition::GM,
        matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_INT32);
    mmTiling.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
        matmul_tiling::DataType::DT_INT32);
    mmTiling.SetOrgShape(M, N, Ka);
    mmTiling.SetSingleShape(SINGLE_CORE_M, SINGLE_CORE_N, SINGLE_CORE_K);
    mmTiling.SetFixSplit(BASE_M, BASE_N, -1);
    mmTiling.EnableBias(false);
    mmTiling.SetBufferSpace(-1, -1, -1);    // 默认使用该AI处理器所有空间

    OP_TILING_CHECK(mmTiling.GetTiling(tilingData->qbmmReduceScatterAddRmsNormCastTilingInfo.matmulTiling) == -1,
                    OP_LOGE(nodeName, "failed to get tiling matmulTiling."),
                    return ge::GRAPH_FAILED);

    
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief qbmm_reduce_scatter_add_rms_norm_cast算子的tiling函数
 * @param context: 框架根据input，output，attrs等信息生成tiling需要的context
 * @return
 */
static ge::graphStatus QbmmReduceScatterAddRmsNormCastTilingFunc(gert::TilingContext *context)
{
    // 1. tiling参数校验
    OP_LOGD("QbmmReduceScatterAddRmsNormCast tiling start.");
    // OP_TILING_CHECK(context == nullptr,
    //                 OP_LOGE("qbmm_reduce_scatter_add_rms_norm_cast", "failed to get tiling context in qbmm_reduce_scatter_add_rms_norm_cast."),
    //                 return ge::GRAPH_FAILED);
    const char *nodeName = context->GetNodeName();
    // OP_TILING_CHECK(nodeName == nullptr,
    //                 OP_LOGE("qbmm_reduce_scatter_add_rms_norm_cast", "failed to get nodeName in qbmm_reduce_scatter_add_rms_norm_cast."),
    //                 return ge::GRAPH_FAILED);

    QbmmReduceScatterAddRmsNormCastTilingData *tilingData = context->GetTilingData<QbmmReduceScatterAddRmsNormCastTilingData>();
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE(nodeName, "tilingData is nullptr in qbmm_reduce_scatter_add_rms_norm_cast."),
                    return ge::GRAPH_FAILED);


    // // 校验输入输出tensor的dim/dtype/format
    // OP_TILING_CHECK(QbmmReduceScatterAddRmsNormCastCheckTiling::TilingCheckQbmmReduceScatterAddRmsNormCast(context) !=
    //     ge::GRAPH_SUCCESS, OP_LOGE(nodeName, "Tiling check param failed."), return ge::GRAPH_FAILED);
    
    // 2. tiling参数设置
    // 获取group以便ranksize设置与通信设置
    std::string group = "";
    const char *groupPtr = context->GetAttrs()->GetAttrPointer<char>(GROUP_INDEX);
    group = std::string(groupPtr);
    OP_TILING_CHECK(SetHcommCfg(context, tilingData, group) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "SetHCommCfg failed."), return ge::GRAPH_FAILED);
    // 调用matmul做tiling切分
    OP_TILING_CHECK(SetTCubeTiling(context, tilingData) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "SetTCubeTiling failed."), return ge::GRAPH_FAILED);
    SetWorkSpace(context, tilingData);
    SetTilingData(context, *tilingData);
    SetTilingKey(context);
    OP_LOGD("QbmmReduceScatterAddRmsNormCast tiling end.");
    return ge::GRAPH_SUCCESS;
}
 
ge::graphStatus TilingParseForQbmmReduceScatterAddRmsNormCast(gert::TilingParseContext *context) {
    (void)context;
  	return ge::GRAPH_SUCCESS;
}
struct QbmmReduceScatterAddRmsNormCastCompileInfo {};
 
IMPL_OP_OPTILING(QbmmReduceScatterAddRmsNormCast)
    .Tiling(QbmmReduceScatterAddRmsNormCastTilingFunc)
    .TilingParse<QbmmReduceScatterAddRmsNormCastCompileInfo>(TilingParseForQbmmReduceScatterAddRmsNormCast);

} // namespace MC2Tiling
