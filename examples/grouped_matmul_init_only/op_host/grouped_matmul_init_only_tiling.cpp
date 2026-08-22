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
 * \file grouped_matmul_init_only_tiling.cpp
 * \brief
 *
 * init-only 头开销测量算子的 host tiling：体现 GroupedMatmul 类算子"重" host tiling
 * （平台信息、shape 解析、循环 G 次算 per-group 偏移、填充大 TCubeTiling 结构标量）。
 *
 * 注意（ODR 硬约束）：本文件所有 helper 函数一律 static 内部链接并加 GroupedMatmulInitOnly
 * 前缀命名，避免与 examples 下其它算子 tiling 的同名函数 multiple definition 链接冲突。
 */

#include "log/log.h"
#include "util/math_util.h"
#include "tiling_base/tiling_util.h"
#include "tiling_base/tiling_templates_registry.h"
#include "examples/grouped_matmul_init_only/op_kernel/grouped_matmul_init_only_tiling_data.h"
#include "examples/grouped_matmul_init_only/op_kernel/grouped_matmul_init_only_tiling_key.h"

namespace optiling {

using namespace Ops::Transformer::OpTiling;

const uint32_t GMM_INIT_ONLY_BLOCK_DIM = 8;
const uint32_t GMM_INIT_ONLY_WS_SYS_SIZE = 16U * 1024U * 1024U;
constexpr uint32_t GMM_INIT_ONLY_IDX0 = 0;
constexpr uint32_t GMM_INIT_ONLY_IDX1 = 1;
constexpr uint32_t GMM_INIT_ONLY_IDX2 = 2;

struct GroupedMatmulInitOnlyCompileInfo {
};

// 获取平台信息：coreNum / ubSize / L1 size（static + 前缀，避免 ODR 冲突）
static ge::graphStatus GroupedMatmulInitOnlyGetPlatformInfo(
    gert::TilingContext* context, uint64_t& ubSize, uint64_t& l1Size, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAic();
    if (coreNum == 0) {
        coreNum = ascendcPlatform.GetCoreNumAiv();
    }
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1Size);
    OP_CHECK_IF(l1Size == 0, OP_LOGE(context, "l1Size is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// 解析 shape：M,K,N,G（static + 前缀），并校验 fp16
static ge::graphStatus GroupedMatmulInitOnlyGetShapeAttrsInfo(
    gert::TilingContext* context, int64_t& mDim, int64_t& kDim, int64_t& nDim, int64_t& gDim)
{
    // x: [M, K]
    auto inputX = context->GetInputShape(GMM_INIT_ONLY_IDX0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputX);
    auto shapeX = EnsureNotScalar(inputX->GetStorageShape());
    // weight: [G, K, N]
    auto inputW = context->GetInputShape(GMM_INIT_ONLY_IDX1);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputW);
    auto shapeW = EnsureNotScalar(inputW->GetStorageShape());
    // group_list: [G]
    auto inputGl = context->GetInputShape(GMM_INIT_ONLY_IDX2);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputGl);
    auto shapeGl = EnsureNotScalar(inputGl->GetStorageShape());

    OP_CHECK_IF(
        shapeX.GetDimNum() < 2 || shapeW.GetDimNum() < 2,
        OP_LOGE(context, "GroupedMatmulInitOnly: x dim=%zu(<2) or weight dim=%zu(<2)",
            shapeX.GetDimNum(), shapeW.GetDimNum()),
        return ge::GRAPH_FAILED);

    mDim = shapeX.GetDim(GMM_INIT_ONLY_IDX0);
    kDim = shapeX.GetDim(shapeX.GetDimNum() - 1);
    nDim = shapeW.GetDim(shapeW.GetDimNum() - 1);
    // G：weight 首维（[G,K,N]）；若 group_list 有维度则以其首维为准
    gDim = shapeW.GetDim(GMM_INIT_ONLY_IDX0);
    if (shapeGl.GetDimNum() >= 1 && shapeGl.GetDim(GMM_INIT_ONLY_IDX0) > 0) {
        gDim = shapeGl.GetDim(GMM_INIT_ONLY_IDX0);
    }
    if (gDim <= 0) {
        gDim = 1;
    }
    if (gDim > GMM_INIT_ONLY_MAX_G) {
        gDim = GMM_INIT_ONLY_MAX_G;
    }

    // dtype 校验：仅支持 fp16
    auto xDesc = context->GetInputDesc(GMM_INIT_ONLY_IDX0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    if (xDesc->GetDataType() != ge::DT_FLOAT16) {
        OP_LOGE(context, "GroupedMatmulInitOnly: only float16 x is supported");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GroupedMatmulInitOnlyGetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = GMM_INIT_ONLY_WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

// 填充大 tilingData：模拟 TCubeTiling 结构标量 + 循环 G 次算 per-group m 偏移
static void GroupedMatmulInitOnlyFillTilingData(
    GroupedMatmulInitOnlyTilingData* tiling, int64_t mDim, int64_t kDim, int64_t nDim, int64_t gDim,
    uint64_t ubSize, uint64_t l1Size, int64_t coreNum)
{
    tiling->M = mDim;
    tiling->K = kDim;
    tiling->N = nDim;
    tiling->G = gDim;

    // 循环 G 次计算 per-group m 偏移（模拟 GMM host 侧对 group_list 的解析）
    int64_t perGroupM = (gDim > 0) ? (mDim / gDim) : mDim;
    int64_t remainder = (gDim > 0) ? (mDim % gDim) : 0;
    int64_t cumOffset = 0;
    for (int64_t i = 0; i < gDim && i < GMM_INIT_ONLY_MAX_G; i++) {
        int64_t gm = perGroupM + (i < remainder ? 1 : 0);
        tiling->groupList[i] = gm;           // 该 group 的 m 大小
        tiling->groupMOffset[i] = cumOffset; // 累积 m 偏移
        cumOffset += gm;
    }

    // 模拟 TCubeTiling 的 base/single/step 标量计算（体现 host tiling 计算量）
    int64_t baseM = 128;
    int64_t baseN = 256;
    int64_t baseK = 64;
    if (baseM > mDim && mDim > 0) baseM = mDim;
    if (baseN > nDim && nDim > 0) baseN = nDim;
    if (baseK > kDim && kDim > 0) baseK = kDim;

    int64_t singleCoreM = (coreNum > 0) ? ((mDim + coreNum - 1) / coreNum) : mDim;
    if (singleCoreM <= 0) singleCoreM = baseM;
    int64_t singleCoreN = nDim;
    int64_t singleCoreK = kDim;

    tiling->baseM = baseM;
    tiling->baseN = baseN;
    tiling->baseK = baseK;
    tiling->singleCoreM = singleCoreM;
    tiling->singleCoreN = singleCoreN;
    tiling->singleCoreK = singleCoreK;
    tiling->stepM = 1;
    tiling->stepN = 1;
    tiling->stepKa = (baseK > 0) ? ((kDim + baseK - 1) / baseK) : 1;
    tiling->stepKb = tiling->stepKa;
    tiling->depthA1 = 4;
    tiling->depthB1 = 4;
    tiling->iterateOrder = 0;
    tiling->usedCoreNum = coreNum;
    tiling->coreNum = coreNum;
    tiling->ubSize = static_cast<int64_t>(ubSize);
    tiling->l1Size = static_cast<int64_t>(l1Size);
    tiling->l0cSize = 256 * 1024;
    tiling->l0aSize = 64 * 1024;
    tiling->l0bSize = 64 * 1024;
    tiling->mTiles = (baseM > 0) ? ((mDim + baseM - 1) / baseM) : 1;
    tiling->nTiles = (baseN > 0) ? ((nDim + baseN - 1) / baseN) : 1;
    tiling->kTiles = (baseK > 0) ? ((kDim + baseK - 1) / baseK) : 1;
}

// tiling 分发入口
static ge::graphStatus GroupedMatmulInitOnlyTilingFunc(gert::TilingContext* context)
{
    // 1、平台信息
    uint64_t ubSize = 0;
    uint64_t l1Size = 0;
    int64_t coreNum = 0;
    OP_CHECK_IF(
        GroupedMatmulInitOnlyGetPlatformInfo(context, ubSize, l1Size, coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);

    // 2、shape / dtype
    int64_t mDim = 0;
    int64_t kDim = 0;
    int64_t nDim = 0;
    int64_t gDim = 0;
    OP_CHECK_IF(
        GroupedMatmulInitOnlyGetShapeAttrsInfo(context, mDim, kDim, nDim, gDim) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetShapeAttrsInfo error"), return ge::GRAPH_FAILED);

    // 3、workspace
    OP_CHECK_IF(
        GroupedMatmulInitOnlyGetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"), return ge::GRAPH_FAILED);

    // 4、填 tilingData
    GroupedMatmulInitOnlyTilingData* tiling = context->GetTilingData<GroupedMatmulInitOnlyTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(GroupedMatmulInitOnlyTilingData), 0, sizeof(GroupedMatmulInitOnlyTilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);
    GroupedMatmulInitOnlyFillTilingData(tiling, mDim, kDim, nDim, gDim, ubSize, l1Size, coreNum);

    // 5、blockDim + tilingKey(0)
    context->SetBlockDim(GMM_INIT_ONLY_BLOCK_DIM);
    uint64_t tilingKey = GET_TPL_TILING_KEY(GMM_INIT_ONLY_TPL_SCH_MODE_0);
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForGroupedMatmulInitOnly([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

// tiling 注册入口
IMPL_OP_OPTILING(GroupedMatmulInitOnly)
    .Tiling(GroupedMatmulInitOnlyTilingFunc)
    .TilingParse<GroupedMatmulInitOnlyCompileInfo>(TilingParseForGroupedMatmulInitOnly);
} // namespace optiling
