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
 * \file quant_matmul_reduce_scatter_tiling.cpp
 * \brief host侧tiling实现
 */

#include <register/op_def_registry.h>
#include "util/math_util.h"
#include "../../op_kernel/quant_matmul_reduce_scatter_tiling_data.h"
#include "../../../3rd/quant_matmul/op_host/op_tiling/quant_matmul_mx_tiling_swat.h"  // 使用 SWAT tiling engine

#include "mc2_log.h"
using namespace AscendC;
using namespace ge;
namespace MC2Tiling {
constexpr uint32_t OP_TYPE_ALL_TO_ALL = 8;
constexpr uint32_t AIV_TYPE = 3;
constexpr uint32_t BUFFER_NUM = 1;
constexpr uint32_t SYSTEM_NEED_WORKSPACE = 16U * 1024 * 1024;
constexpr uint64_t CUBE_BLOCK_NUM = 16;
constexpr uint64_t MAX_BASIC_BLOCK_SIZE = 256;
constexpr uint64_t M_BLOCK_WINDOW_LEN = 4; // 与device侧Blaze::Gemm::WINDOW_LEN一致
constexpr uint64_t UB_TOTAL_BYTES = 248UL * 1024UL;
constexpr uint64_t UB_COST_FACTOR = 12UL; // sumfp32(4b)+localbf16(2b)+localfp32(4b)+sumbf16(2b)

constexpr size_t X1_INDEX = 0;
constexpr size_t X2_INDEX = 1;
constexpr size_t OUTPUT_INDEX = 0;

constexpr size_t GROUP_INDEX = 0;
constexpr size_t RANK_SIZE_INDEX = 1;
constexpr size_t TRANSPOSE_X2_INDEX = 2;
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
static ge::graphStatus SetHcommCfg(const gert::TilingContext *context, QuantMatmulReduceScatterTilingData *tilingData,
                        const std::string &group)
{
    const char *nodeName = context->GetNodeName();
    OP_LOGD(nodeName, "group is %s in quant_matmul_reduce_scatter.", group.c_str());
    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(group, OP_TYPE_ALL_TO_ALL,
                                                 "AlltoAll=level0:fullmesh;level1:pairwise");
    // MTE方式必要适配
    mc2CcTilingConfig.SetCommEngine(AIV_TYPE);
    OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(tilingData->mc2InitTiling) != 0,
        OP_LOGE(nodeName, "mc2CcTilingConfig mc2InitTiling GetTiling failed"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(tilingData->mc2CcTiling) != 0,
        OP_LOGE(nodeName, "mc2CcTilingConfig mc2CcTiling GetTiling failed"), return ge::GRAPH_FAILED);
    OP_LOGD("QuantMatmulReduceScatter set HCCL success.");
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 设置tilingData
 * @param context: 框架根据input，output，attrs等信息生成tiling需要的context
 * @param tilingData: 框架根据context的opName匹配tiling模板，计算产生的tilingData
 * @return
 */
// static void SetTilingData(gert::TilingContext *context, QuantMatmulReduceScatterTilingData &tilingData)
// {
//     const char *nodeName = context->GetNodeName();
//     fe::PlatFormInfos *platformInfoPtr = context->GetPlatformInfo();
//     platform_ascendc::PlatformAscendC ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
//     // set corenum
//     uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
//     uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
//     OP_LOGD(nodeName, "aicNum is %u, aivNum is %u in quant_matmul_reduce_scatter.", aicNum, aivNum);
//     context->SetBlockDim(aicNum);   // TODO:
//     tilingData.quantMatmulReduceScatterTilingInfo.aicNum = aicNum;
//     tilingData.quantMatmulReduceScatterTilingInfo.aivNum = aivNum;
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
    uint32_t N = 0;
    ge::Format x2Format = static_cast<ge::Format>(ge::GetPrimaryFormat(context->GetInputDesc(X2_INDEX)->GetStorageFormat()));
    if (x2Format == ge::FORMAT_ND) {
        N = x2Shape->GetStorageShape().GetDim(DIM_ONE);
    } else if (x2Format == ge::FORMAT_FRACTAL_NZ) {
        N = x2Shape->GetStorageShape().GetDim(DIM_ZERO) * x2Shape->GetStorageShape().GetDim(DIM_THREE);  // Dim0*Dim3
    } else {
        OP_LOGE(nodeName, "Unsupported x2 format: %d", x2Format);
        return ge::GRAPH_FAILED;
    }
    
    workSpaces[0] = SYSTEM_NEED_WORKSPACE + M * N * sizeof(float32_t);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetTilingData(
    gert::TilingContext *context, QuantMatmulReduceScatterTilingData *tilingData)
{
    const char *nodeName = context->GetNodeName();
    
    // 1. 获取输入 shape
    const gert::StorageShape *x1Shape = context->GetInputShape(X1_INDEX);
    const gert::StorageShape *x2Shape = context->GetInputShape(X2_INDEX);
    const gert::StorageShape *outputShape = context->GetOutputShape(OUTPUT_INDEX);

    OP_TILING_CHECK(x1Shape == nullptr || x2Shape == nullptr,
        OP_LOGE(nodeName, "Get input shape failed."), return ge::GRAPH_FAILED);
    
    // 2. 解析 M, K, N
    uint64_t M = x1Shape->GetStorageShape().GetDim(DIM_ZERO);
    uint64_t K = x1Shape->GetStorageShape().GetDim(DIM_ONE);
    uint64_t N = 0;
    
    ge::Format x2Format = static_cast<ge::Format>(ge::GetPrimaryFormat(context->GetInputDesc(X2_INDEX)->GetStorageFormat()));
    if (x2Format == ge::FORMAT_ND) {
        N = x2Shape->GetStorageShape().GetDim(DIM_ONE);
        printf("enter nd tiling mode\n");
    } else if (x2Format == ge::FORMAT_FRACTAL_NZ) {
        // // NZ storage: (Nt, Kt, 16, 32) = (ceil(N/32), ceil(K/16), 16, 32), N = Nt*32
        // auto d0 = x2Shape->GetStorageShape().GetDim(DIM_ZERO);
        // auto d3 = x2Shape->GetStorageShape().GetDim(DIM_THREE);
        // N = d0 * d3;
        // printf("[NZ Tiling Debug] storage dims: [%lu,%lu,%lu,%lu], N=%lu\n",
        //        d0, x2Shape->GetStorageShape().GetDim(DIM_ONE),
        //        x2Shape->GetStorageShape().GetDim(DIM_TWO), d3, N);
        // NZ 场景直接用output shape取N
        N = static_cast<uint64_t>(outputShape->GetStorageShape().GetDim(DIM_ONE));
        printf("[NZ Tiling Debug] N: %u \n", N);
    } else {
        OP_LOGE(nodeName, "Unsupported format: %d", x2Format);
        return ge::GRAPH_FAILED;
    }
    
    OP_LOGD(nodeName, "Input shape: M=%lu, K=%lu, N=%lu", M, K, N);
    
    // 3. 使用 SWAT tiling engine 计算 matmul tiling 参数
    // rankSize（从 attrs 获取）
    const uint32_t* rankSizePtr = context->GetAttrs()->GetAttrPointer<uint32_t>(RANK_SIZE_INDEX);
    OP_TILING_CHECK(rankSizePtr == nullptr,
        OP_LOGE(nodeName, "rankSize is nullptr."), return ge::GRAPH_FAILED);
    uint32_t rankSize = *rankSizePtr;
    OP_TILING_CHECK(rankSize == 0, OP_LOGE(nodeName, "rankSize is 0."), return ge::GRAPH_FAILED);
    tilingData->quantMatmulReduceScatterTilingInfo.rankSize = rankSize;

    QuantMatmulTilingSwat<mm::DataType::DT_FLOAT8_E4M3FN, mm::DataType::DT_FLOAT8_E4M3FN> tilingEngine;
    bool transB = (x2Format == ge::FORMAT_ND); // ND→transB=true, NZ→transB=false
    // tilingEngine.SetMaxBaseM(128);
    uint64_t tpSizeM = M / rankSize;
    OP_TILING_CHECK(M % rankSize != 0, OP_LOGE(nodeName, "M must be divisble by rankSize!"), return ge::GRAPH_FAILED);
    // baseM取tpSizeM的ceil(rankSize/4)倍, 使mBlockCnt = rankSize/ceil(rankSize/4) <= 窗口宽4:
    // 单窗口行线性走位, 每一轮分核都覆盖全部M块, alltoall全程喂满所有peer链路;
    // rankSize <= 4时倍数为1, 退化为原有逻辑
    uint64_t mSplitFactor = (rankSize + M_BLOCK_WINDOW_LEN - 1) / M_BLOCK_WINDOW_LEN;
    uint64_t dynMaxBaseM = std::min(MAX_BASIC_BLOCK_SIZE, (mSplitFactor * tpSizeM / CUBE_BLOCK_NUM) * CUBE_BLOCK_NUM);
    if (dynMaxBaseM < CUBE_BLOCK_NUM) {
        dynMaxBaseM = std::min(MAX_BASIC_BLOCK_SIZE, (M / 2 / CUBE_BLOCK_NUM) * CUBE_BLOCK_NUM);
    }
    if (dynMaxBaseM < CUBE_BLOCK_NUM) {
        dynMaxBaseM = std::min(MAX_BASIC_BLOCK_SIZE, M);
    }
    tilingEngine.SetMaxBaseM(dynMaxBaseM);
    tilingEngine.GetTilingData(M, N, K, false, transB, tilingData->qbmmTilingdata);
    
    // 注意：GetTilingData 会自动填充 qbmmTilingdata 的所有字段，包括：
    // - m, n, k
    // - baseM, baseN, baseK
    // - scaleKL1, stepK
    // - mTailTile, nTailTile
    // - mBaseTailSplitCnt, nBaseTailSplitCnt
    // - mTailMain, nTailMain
    // - usedCoreNum, nBufferNum, dbL0c
    OP_LOGD(nodeName, "QuantMatmulTilingSwat computed tiling data successfully");
    OP_LOGD(nodeName, "  qbmmTilingdata.m=%u, n=%u, k=%u", 
            tilingData->qbmmTilingdata.m, tilingData->qbmmTilingdata.n, tilingData->qbmmTilingdata.k);
    OP_LOGD(nodeName, "  qbmmTilingdata.baseM=%u, baseN=%u, baseK=%u", 
            tilingData->qbmmTilingdata.baseM, tilingData->qbmmTilingdata.baseN, tilingData->qbmmTilingdata.baseK);
    OP_LOGD(nodeName, "  qbmmTilingdata.usedCoreNum=%u", tilingData->qbmmTilingdata.usedCoreNum);
    
    // 4. 填充通信和自定义参数（在 matmul tiling 之后）
    // ubCalcM, ubCalcN（V核 UB 计算参数）
    tilingData->quantMatmulReduceScatterTilingInfo.ubCalcM = tilingData->qbmmTilingdata.baseM;
    tilingData->quantMatmulReduceScatterTilingInfo.ubCalcN = tilingData->qbmmTilingdata.baseN;
    tilingData->quantMatmulReduceScatterTilingInfo.needUbBuffer = 1;
    
    // 获取平台信息（用于填充 aivNum/aicNum）
    fe::PlatFormInfos *platformInfoPtr = context->GetPlatformInfo();
    platform_ascendc::PlatformAscendC ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    tilingData->quantMatmulReduceScatterTilingInfo.aivNum = ascendcPlatform.GetCoreNumAiv();
    tilingData->quantMatmulReduceScatterTilingInfo.aicNum = ascendcPlatform.GetCoreNumAic();

    context->SetBlockDim(ascendcPlatform.GetCoreNumAic()); 
    
    OP_LOGD(nodeName, "Custom tiling parameters filled:");
    OP_LOGD(nodeName, "  rankSize=%u, ubCalcM=%u, ubCalcN=%u, needUbBuffer=%u",
        rankSize,
        tilingData->quantMatmulReduceScatterTilingInfo.ubCalcM,
        tilingData->quantMatmulReduceScatterTilingInfo.ubCalcN,
        tilingData->quantMatmulReduceScatterTilingInfo.needUbBuffer);
    OP_LOGD(nodeName, "  aivNum=%u, aicNum=%u", 
        tilingData->quantMatmulReduceScatterTilingInfo.aivNum,
        tilingData->quantMatmulReduceScatterTilingInfo.aicNum);
    
    uint64_t rowPaddingBytes = Ops::Base::CeilDiv(static_cast<uint64_t>((N) * 2), static_cast<uint64_t>(32)) * 32;
    uint64_t paddingN = rowPaddingBytes / 2;
    uint64_t ubTotalCost = paddingN * UB_COST_FACTOR;
    if (ubTotalCost > UB_TOTAL_BYTES) {
        OP_LOGE(nodeName, "Exceeds UB limit when N = %lu", N);
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

/**
 * @brief quant_matmul_reduce_scatter算子的tiling函数
 * @param context: 框架根据input，output，attrs等信息生成tiling需要的context
 * @return
 */
static ge::graphStatus QuantMatmulReduceScatterTilingFunc(gert::TilingContext *context)
{
    // 1. tiling参数校验
    OP_LOGD("QuantMatmulReduceScatter tiling start.");
    // OP_TILING_CHECK(context == nullptr,
    //                 OP_LOGE("quant_matmul_reduce_scatter", "failed to get tiling context in quant_matmul_reduce_scatter."),
    //                 return ge::GRAPH_FAILED);
    const char *nodeName = context->GetNodeName();
    // OP_TILING_CHECK(nodeName == nullptr,
    //                 OP_LOGE("quant_matmul_reduce_scatter", "failed to get nodeName in quant_matmul_reduce_scatter."),
    //                 return ge::GRAPH_FAILED);

    QuantMatmulReduceScatterTilingData *tilingData = context->GetTilingData<QuantMatmulReduceScatterTilingData>();
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE(nodeName, "tilingData is nullptr in quant_matmul_reduce_scatter."),
                    return ge::GRAPH_FAILED);


    // // 校验输入输出tensor的dim/dtype/format
    // OP_TILING_CHECK(QuantMatmulReduceScatterCheckTiling::TilingCheckQuantMatmulReduceScatter(context) !=
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
    OP_LOGD("QuantMatmulReduceScatter tiling end.");
    return ge::GRAPH_SUCCESS;
}
 
ge::graphStatus TilingParseForQuantMatmulReduceScatter(gert::TilingParseContext *context) {
    (void)context;
  	return ge::GRAPH_SUCCESS;
}
struct QuantMatmulReduceScatterCompileInfo {};
 
IMPL_OP_OPTILING(QuantMatmulReduceScatter)
    .Tiling(QuantMatmulReduceScatterTilingFunc)
    .TilingParse<QuantMatmulReduceScatterCompileInfo>(TilingParseForQuantMatmulReduceScatter);

} // namespace MC2Tiling
