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
#include "util/math_util.h"
#include "../../op_kernel/qbmm_reduce_scatter_add_rms_norm_cast_tiling_data.h"
#include "quant_matmul_mx_tiling_swat.h"  // 使用 SWAT tiling engine

#include "mc2_log.h"
using namespace AscendC;
using namespace ge;
namespace MC2Tiling {
constexpr uint32_t OP_TYPE_ALL_TO_ALL = 8;
constexpr uint32_t AIV_TYPE = 3;
constexpr uint32_t BUFFER_NUM = 1;
constexpr uint32_t SYSTEM_NEED_WORKSPACE = 16U * 1024 * 1024;

constexpr size_t X1_INDEX = 0;
constexpr size_t X2_INDEX = 1;
constexpr size_t Y_INDEX = 2;
constexpr size_t GAMMA_INDEX = 3;
constexpr size_t SCALE_INDEX = 4;
constexpr size_t BIAS_INDEX = 5;
constexpr size_t PER_TOKEN_SCALE_INDEX = 6;
constexpr size_t Y1_INDEX = 0;
constexpr size_t Y2_INDEX = 1;
constexpr size_t X_INDEX = 2;
constexpr size_t OUTPUT_INDEX = 0;
constexpr size_t GROUP_INDEX = 0;
constexpr size_t RANK_SIZE_INDEX = 1;
constexpr size_t TRANSPOSE_X2_INDEX = 2;
constexpr size_t OUT_PUT_DTYPE_INDEX = 3;
constexpr size_t EPSILON_INDEX = 4;
constexpr size_t DIM_ZERO = 0;
constexpr size_t DIM_ONE = 1;
constexpr size_t DIM_TWO = 2;
constexpr size_t DIM_THREE = 3;

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
// static void SetTilingData(gert::TilingContext *context, QbmmReduceScatterAddRmsNormCastTilingData &tilingData)
// {
//     const char *nodeName = context->GetNodeName();
//     fe::PlatFormInfos *platformInfoPtr = context->GetPlatformInfo();
//     platform_ascendc::PlatformAscendC ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
//     // set corenum
//     uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
//     uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
//     OP_LOGD(nodeName, "aicNum is %u, aivNum is %u in qbmm_reduce_scatter_add_rms_norm_cast.", aicNum, aivNum);
//     context->SetBlockDim(aicNum);   // TODO:
//     tilingData.qbmmReduceScatterAddRmsNormCastTilingInfo.aicNum = aicNum;
//     tilingData.qbmmReduceScatterAddRmsNormCastTilingInfo.aivNum = aivNum;
// }

/**
 * @brief 设置tilingKey, 先使用老的方式生成tilingkey
 * @param context: 框架根据input，output，attrs等信息生成tiling需要的context
 * @return
 */
static void SetTilingKey(gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();
    ge::Format x2Format = static_cast<ge::Format>(ge::GetPrimaryFormat(context->GetInputDesc(X2_INDEX)->GetStorageFormat()));
    OP_LOGD(nodeName, "SetTilingKey: x2Format=%d (ND=%d FRACTAL_NZ=%d)", x2Format, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ);
    if (x2Format == ge::FORMAT_ND) {
        context->SetTilingKey(0);
    } else if (x2Format == ge::FORMAT_FRACTAL_NZ) {
        context->SetTilingKey(1);
    }
}

static ge::graphStatus SetWorkSpace(gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();

    const gert::StorageShape *x1Shape = context->GetInputShape(X1_INDEX);
    const gert::StorageShape *x2Shape = context->GetInputShape(X2_INDEX);
    size_t *workSpaces = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workSpaces == nullptr, OP_LOGE(nodeName, "workSpaces is nullptr."), return ge::GRAPH_FAILED);
    
    uint32_t M = x1Shape->GetStorageShape().GetDim(DIM_ZERO);
    uint32_t N;
    ge::Format x2Format = static_cast<ge::Format>(ge::GetPrimaryFormat(context->GetInputDesc(X2_INDEX)->GetStorageFormat()));
    if (x2Format == ge::FORMAT_ND) {
        N = x2Shape->GetStorageShape().GetDim(DIM_ONE);
    } else if (x2Format == ge::FORMAT_FRACTAL_NZ) {
        N = x2Shape->GetStorageShape().GetDim(DIM_ZERO) * x2Shape->GetStorageShape().GetDim(DIM_THREE);  // Dim0*Dim3
    } 
    
    workSpaces[0] = SYSTEM_NEED_WORKSPACE + M * N * sizeof(float32_t);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetTilingData(
    gert::TilingContext *context, QbmmReduceScatterAddRmsNormCastTilingData *tilingData)
{
    const char *nodeName = context->GetNodeName();
    
    // 1. 获取输入 shape
    const gert::StorageShape *x1Shape = context->GetInputShape(X1_INDEX);
    const gert::StorageShape *x2Shape = context->GetInputShape(X2_INDEX);
    
    OP_TILING_CHECK(x1Shape == nullptr || x2Shape == nullptr,
        OP_LOGE(nodeName, "Get input shape failed."), return ge::GRAPH_FAILED);
    
    // 2. 解析 M, K, N
    uint64_t M = x1Shape->GetStorageShape().GetDim(DIM_ZERO);
    uint64_t K = x1Shape->GetStorageShape().GetDim(DIM_ONE);
    uint64_t N;
    
    ge::Format x2Format = static_cast<ge::Format>(ge::GetPrimaryFormat(context->GetInputDesc(X2_INDEX)->GetStorageFormat()));
    if (x2Format == ge::FORMAT_ND) {
        N = x2Shape->GetStorageShape().GetDim(DIM_ONE);
        printf("enter nd tiling mode\n");
    } else if (x2Format == ge::FORMAT_FRACTAL_NZ) {
        // NZ storage: (Nt, Kt, 16, 32) = (ceil(N/32), ceil(K/16), 16, 32), N = Nt*32
        auto d0 = x2Shape->GetStorageShape().GetDim(DIM_ZERO);
        auto d3 = x2Shape->GetStorageShape().GetDim(DIM_THREE);
        N = d0 * d3;
        printf("[NZ Tiling Debug] storage dims: [%lu,%lu,%lu,%lu], N=%lu\n",
               d0, x2Shape->GetStorageShape().GetDim(DIM_ONE),
               x2Shape->GetStorageShape().GetDim(DIM_TWO), d3, N);
    } else {
        OP_LOGE(nodeName, "Unsupported format: %d", x2Format);
        return ge::GRAPH_FAILED;
    }
    
    OP_LOGD(nodeName, "Input shape: M=%lu, K=%lu, N=%lu", M, K, N);
    
    // 3. 使用 SWAT tiling engine 计算 matmul tiling 参数
    QuantMatmulTilingSwat<mm::DataType::DT_FLOAT8_E4M3FN, mm::DataType::DT_FLOAT8_E4M3FN> tilingEngine;
    bool transB = (x2Format == ge::FORMAT_ND); // ND→transB=true, NZ→transB=false
    tilingEngine.GetTilingData(M, N, K, false, transB, tilingData->qbmmTilingdata);
    
    // 注意：GetTilingData 会自动填充 qbmmTilingdata 的所有字段，包括：
    // - m, n, k
    // - baseM, baseN, baseK
    // - scaleKL1, stepK
    // - mTailTile, nTailTile
    // - mBaseTailSplitCnt, nBaseTailSplitCnt
    // - mTailMain, nTailMain
    // - usedCoreNum, nBufferNum, dbL0c
    tilingData->qbmmTilingdata.baseM = 128;
    OP_LOGD(nodeName, "QuantMatmulTilingSwat computed tiling data successfully");
    OP_LOGD(nodeName, "  qbmmTilingdata.m=%u, n=%u, k=%u", 
            tilingData->qbmmTilingdata.m, tilingData->qbmmTilingdata.n, tilingData->qbmmTilingdata.k);
    OP_LOGD(nodeName, "  qbmmTilingdata.baseM=%u, baseN=%u, baseK=%u", 
            tilingData->qbmmTilingdata.baseM, tilingData->qbmmTilingdata.baseN, tilingData->qbmmTilingdata.baseK);
    OP_LOGD(nodeName, "  qbmmTilingdata.usedCoreNum=%u", tilingData->qbmmTilingdata.usedCoreNum);
    
    // 4. 填充通信和自定义参数（在 matmul tiling 之后）
    // rankSize（从 attrs 获取）
    const uint32_t* rankSizePtr = context->GetAttrs()->GetAttrPointer<uint32_t>(RANK_SIZE_INDEX);
    OP_TILING_CHECK(rankSizePtr == nullptr,
        OP_LOGE(nodeName, "rankSize is nullptr."), return ge::GRAPH_FAILED);
    uint32_t rankSize = *rankSizePtr;
    tilingData->qbmmReduceScatterAddRmsNormCastTilingInfo.rankSize = rankSize;
    
    // epsilon（从 attrs 获取）
    const float* epsilonPtr = context->GetAttrs()->GetAttrPointer<float>(EPSILON_INDEX);
    OP_TILING_CHECK(epsilonPtr == nullptr,
        OP_LOGE(nodeName, "epsilon is nullptr."), return ge::GRAPH_FAILED);
    float epsilon = *epsilonPtr;
    tilingData->qbmmReduceScatterAddRmsNormCastTilingInfo.epsilon = epsilon;
    
    // avgFactor（可选参数，用于 RMS Norm）
    tilingData->qbmmReduceScatterAddRmsNormCastTilingInfo.avgFactor = 1.0f / rankSize;
    
    // ubCalcM, ubCalcN（V核 UB 计算参数）
    tilingData->qbmmReduceScatterAddRmsNormCastTilingInfo.ubCalcM = tilingData->qbmmTilingdata.baseM;
    tilingData->qbmmReduceScatterAddRmsNormCastTilingInfo.ubCalcN = tilingData->qbmmTilingdata.baseN;
    tilingData->qbmmReduceScatterAddRmsNormCastTilingInfo.needUbBuffer = 1;
    
    // 获取平台信息（用于填充 aivNum/aicNum）
    fe::PlatFormInfos *platformInfoPtr = context->GetPlatformInfo();
    platform_ascendc::PlatformAscendC ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    tilingData->qbmmReduceScatterAddRmsNormCastTilingInfo.aivNum = ascendcPlatform.GetCoreNumAiv();
    tilingData->qbmmReduceScatterAddRmsNormCastTilingInfo.aicNum = ascendcPlatform.GetCoreNumAic();

    context->SetBlockDim(ascendcPlatform.GetCoreNumAic()); 
    
    OP_LOGD(nodeName, "Custom tiling parameters filled:");
    OP_LOGD(nodeName, "  rankSize=%u, epsilon=%f, avgFactor=%f", rankSize, epsilon, tilingData->qbmmReduceScatterAddRmsNormCastTilingInfo.avgFactor);
    OP_LOGD(nodeName, "  ubCalcM=%u, ubCalcN=%u, needUbBuffer=%u", 
            tilingData->qbmmReduceScatterAddRmsNormCastTilingInfo.ubCalcM,
            tilingData->qbmmReduceScatterAddRmsNormCastTilingInfo.ubCalcN,
            tilingData->qbmmReduceScatterAddRmsNormCastTilingInfo.needUbBuffer);
    OP_LOGD(nodeName, "  aivNum=%u, aicNum=%u", 
            tilingData->qbmmReduceScatterAddRmsNormCastTilingInfo.aivNum,
            tilingData->qbmmReduceScatterAddRmsNormCastTilingInfo.aicNum);
    
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
    SetWorkSpace(context);
    OP_TILING_CHECK(SetTilingData(context, tilingData) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "SetTilingData failed."), return ge::GRAPH_FAILED);
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
