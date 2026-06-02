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
 * \file add_rms_norm_dynamic_quant_all_gather_qbmm_tiling.cpp
 * \brief host侧tiling实现
 */

#include <register/op_def_registry.h>
#include "util/math_util.h"
#include "../../op_kernel/tiling/all_gather_matmul_tiling_data.h"
#include "quant_matmul_mx_tiling_swat.h"

#include "mc2_log.h"
using namespace AscendC;
using namespace ge;
namespace MC2Tiling {

using namespace AscendC;
using namespace ge;

constexpr uint32_t AIV_TYPE = 3;
constexpr uint32_t OP_TYPE_ALL_TO_ALL = 6;

constexpr size_t X1_INDEX = 0;
constexpr size_t X2_INDEX = 1;
constexpr size_t RESIDUAL_INDEX = 2;
constexpr size_t Y_INDEX = 3;
constexpr size_t GAMMA_INDEX = 4;
constexpr size_t SCALE_INDEX = 5;
constexpr size_t SMOOTH_SCALE_INDEX = 6;
constexpr size_t BIAS_INDEX = 7;

constexpr size_t OUTPUT_INDEX = 0;
constexpr size_t Z_INDEX = 1;

constexpr size_t GROUP_INDEX = 0;
constexpr size_t RANK_SIZE_INDEX = 1;
constexpr size_t TRANSPOSE_X2_INDEX = 2;
constexpr size_t DTYPE_INDEX = 3;
constexpr size_t RESIDUAL_NORM_MODE_INDEX = 4;
constexpr size_t OPTIONAL_OUTPUT_INDEX = 5;
constexpr size_t DIM_ZERO = 0;
constexpr size_t DIM_ONE = 1;
constexpr size_t DIM_TWO = 2;
constexpr size_t NUM_THREE = 3;
constexpr size_t NUM_FOUR = 4;
constexpr size_t TWO_DIMS = 2;
constexpr size_t FOUR_DIMS = 4;

constexpr size_t TWO_BYTES = 2;
constexpr size_t FOUR_BYTES = 4;

constexpr uint64_t BASE_WORKSPACE_SIZE = 16UL * 1024UL * 1024UL;
constexpr uint64_t MTE_BLOCK_BYTES = 512UL;
constexpr uint32_t MTE_K_SPLIT_NUM = 2UL;
constexpr uint64_t UB_ALIGN = 32UL;
constexpr uint64_t MAX_K_VALUE = 5120UL;
constexpr uint64_t MAX_M_VALUE = 128UL;
constexpr uint32_t MTE_M_CORE_NUM = 2UL;

// tilingKey
constexpr uint32_t SOFT_SYNC_M = 128;
constexpr uint32_t TILE_K_M = 65;
constexpr uint32_t INIT_TILINGKEY = 10000UL;
constexpr uint64_t TILINGKEY_SMOOTH_SCALE = 1UL;
constexpr uint64_t TILINGKEY_OPTIONAL_OUTPUT = 10UL;
constexpr uint64_t TILINGKEY_TILE_K = 100UL;
constexpr uint64_t TILINGKEY_SOFT_SYNC = 1000UL;

// matmul tiling 切分
constexpr int32_t SINGLE_CORE_M = 128;
constexpr int32_t SINGLE_CORE_N = 256;
constexpr int32_t SINGLE_CORE_K = 512;

// addRmsNorm 参数设置
constexpr float EPSILON = 1e-6;


/**
 * @brief add_rms_norm_dynamic_quant_all_gather_qbmm算子的tiling函数
 * @param context: 框架根据input，output，attrs等信息生成tiling需要的context
 * @return
 */
static ge::graphStatus AddRmsNormDynamicQuantAllGatherQbmmTilingFunc(gert::TilingContext *context)
{
    OP_LOGD("AddRmsNormDynamicQuantAllGatherQbmm", "Tiling start");
    OP_TILING_CHECK(context == nullptr,
        OP_LOGE("add_rms_norm_dynamic_quant_all_gather_qbmm", "failed to get tiling context."),
        return ge::GRAPH_FAILED);
    const char *nodeName = context->GetNodeName();
    OP_TILING_CHECK(nodeName == nullptr,
        OP_LOGE("add_rms_norm_dynamic_quant_all_gather_qbmm", "failed to get nodeName."),
        return ge::GRAPH_FAILED);

    allGatherMatmulTilingData *tilingData = context->GetTilingData<allGatherMatmulTilingData>();
    OP_TILING_CHECK(tilingData == nullptr,
        OP_LOGE(nodeName, "tilingData is nullptr."), return ge::GRAPH_FAILED);

    // === 所有 tiling check 注释，测试阶段不校验 ===

    // 1. 从 input shape 获取 M, Ka, N
    const gert::StorageShape *x1Shape = context->GetInputShape(X1_INDEX);
    const gert::StorageShape *x2Shape = context->GetInputShape(X2_INDEX);
    OP_TILING_CHECK(x1Shape == nullptr || x2Shape == nullptr, OP_LOGE(nodeName, "Get input shape failed."), return ge::GRAPH_FAILED);
    uint64_t m = x1Shape->GetStorageShape().GetDim(0);
    uint64_t k = x1Shape->GetStorageShape().GetDim(1);
    uint64_t n = x2Shape->GetStorageShape().GetDim(0);
    ge::Format x2Format = static_cast<ge::Format>(ge::GetPrimaryFormat(context->GetInputDesc(X2_INDEX)->GetStorageFormat()));
    bool isNz = (x2Format == ge::FORMAT_FRACTAL_NZ);
    if (isNz) {
        printf("nz mode\n");
        printf("x2 shape: %lu,%lu,%lu,%lu\n", x2Shape->GetStorageShape().GetDim(0), x2Shape->GetStorageShape().GetDim(1), x2Shape->GetStorageShape().GetDim(2), x2Shape->GetStorageShape().GetDim(3));
        n = x2Shape->GetStorageShape().GetDim(0) * x2Shape->GetStorageShape().GetDim(3);
    }
    printf("m,k,n=%lu,%lu,%lu\n", m, k, n);
    OP_LOGD(nodeName, "M=%lu K=%lu N=%lu isNz=%d", m, k, n, isNz);

    // 2. 设置 blockDim
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t aicNum = ascendcPlatform.GetCoreNumAic();
    uint32_t numBlocks = ascendcPlatform.CalcTschBlockDim(aicNum, aicNum, aicNum);
    context->SetBlockDim(numBlocks);
    context->SetScheduleMode(1); // batch mode

    // 3. HCCL 通信配置
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    const char *groupPtr = attrs->GetAttrPointer<char>(GROUP_INDEX);
    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(std::string(groupPtr), OP_TYPE_ALL_TO_ALL,
                                                 "AlltoAll=level0:fullmesh;level1:pairwise");
    mc2CcTilingConfig.SetCommEngine(AIV_TYPE);
    OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(tilingData->mc2InitTiling) != 0,
        OP_LOGE(nodeName, "mc2InitTiling failed"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(tilingData->mc2CcTiling) != 0,
        OP_LOGE(nodeName, "mc2CcTiling failed"), return ge::GRAPH_FAILED);

    // 4. Matmul tiling — M_total = per-rank M * rankSize (allgather后全量)
    int64_t rankNum = 2;
    const int64_t *rankSizePtr = attrs->GetAttrPointer<int64_t>(RANK_SIZE_INDEX);
    if (rankSizePtr != nullptr) {
        rankNum = *rankSizePtr;
    }
    // 使用 SWAT tiling engine 计算 matmul tiling 参数
    QuantMatmulTilingSwat<mm::DataType::DT_FLOAT8_E4M3FN, mm::DataType::DT_FLOAT8_E4M3FN> tilingEngine;
    bool transB = (x2Format == ge::FORMAT_ND); // ND→transB=true, NZ→transB=false
    tilingEngine.GetTilingData(m * rankNum, n, k, false, transB, tilingData->qbmmTilingData);
    OP_LOGD(nodeName, "Matmul tiling: baseM=%u baseN=%u baseK=%u stepK=%u",
        tilingData->qbmmTilingData.baseM, tilingData->qbmmTilingData.baseN,
        tilingData->qbmmTilingData.baseK, tilingData->qbmmTilingData.stepK);

    // 5. AllGather tiling
    tilingData->allGatherTiling.M = m;
    tilingData->allGatherTiling.Ka = k;
    tilingData->allGatherTiling.N = n;
    tilingData->allGatherTiling.rankSize = rankNum;
    tilingData->allGatherTiling.aivNum = aicNum;  // CV 1:1 (36)
    tilingData->allGatherTiling.aicNum = aicNum;
    tilingData->allGatherTiling.sendCoreNumPerRank = aicNum / rankNum;
    tilingData->allGatherTiling.mteTileK = k / 512;
    tilingData->allGatherTiling.mteKSplitNum = MTE_K_SPLIT_NUM;
    tilingData->allGatherTiling.mteMSplitNum = tilingData->allGatherTiling.sendCoreNumPerRank / MTE_K_SPLIT_NUM;
    tilingData->allGatherTiling.mteKSplitSize = k / MTE_K_SPLIT_NUM;
    tilingData->allGatherTiling.mteMSplitSize = m / tilingData->allGatherTiling.mteMSplitNum;
    tilingData->allGatherTiling.mteMSplitLargeBlockNum = m % tilingData->allGatherTiling.mteMSplitNum;
    tilingData->allGatherTiling.xNums = m * k;
    tilingData->allGatherTiling.xSize = m * k;
    tilingData->allGatherTiling.scaleSize = m * CeilDiv<uint64_t>(k, (uint64_t)64) * 2;
    tilingData->allGatherTiling.singleCoreM = tilingData->qbmmTilingData.baseM;

    OP_LOGD(nodeName, "AllGather tiling: sendCorePerRank=%u mteMSplitNum=%u mteKSplitNum=%u",
        tilingData->allGatherTiling.sendCoreNumPerRank,
        tilingData->allGatherTiling.mteMSplitNum,
        tilingData->allGatherTiling.mteKSplitNum);

    context->SetTilingKey(0);

    // 7. Workspace
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = BASE_WORKSPACE_SIZE + m * rankNum * n * sizeof(int32_t);

    return ge::GRAPH_SUCCESS;
}

struct AddRmsNormDynamicQuantAllGatherQbmmCompileInfo {};

ge::graphStatus TilingParseForAddRmsNormDynamicQuantAllGatherQbmm(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}
 
IMPL_OP_OPTILING(AddRmsNormDynamicQuantAllGatherQbmm)
    .Tiling(AddRmsNormDynamicQuantAllGatherQbmmTilingFunc)
    .TilingParse<AddRmsNormDynamicQuantAllGatherQbmmCompileInfo>(TilingParseForAddRmsNormDynamicQuantAllGatherQbmm);

} // namespace MC2Tiling
