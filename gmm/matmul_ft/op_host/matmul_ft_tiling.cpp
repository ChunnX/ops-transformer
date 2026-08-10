/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

/*!
 * \file matmul_ft_tiling.cpp
 * \brief Host-side tiling for MatmulFt.
 */
#include <cmath>
#include <limits>

#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "../op_kernel/matmul_ft_tiling_data.h"
#include "../op_kernel/matmul_ft_tiling_key.h"

namespace optiling {
namespace {
constexpr size_t SYSTEM_WORKSPACE_SIZE = 16UL * 1024UL * 1024UL;
}

template <typename Policy>
MatmulFtComputeTilingData BuildComputeTiling(uint32_t m, uint32_t n, uint32_t k,
    float eMax, uint32_t reduceCores, uint32_t splitKs)
{
    const uint32_t splitNNum = (n + Policy::L1TileShape_N - 1) / Policy::L1TileShape_N;
    const uint32_t firstBlockN = splitNNum;
    const uint32_t remainBlockN = n;

    const uint32_t totalInputElements = m + n;
    const uint32_t totalOutputElements = (m + 7) / 8 + (n + 7) / 8;
    const uint32_t rowOutputElements = (m + 7) / 8;
    const uint32_t colOutputElements = (n + 7) / 8;
    const uint32_t xLen = m > n ? m : n;

    uint32_t splitReduceM = (splitNNum + reduceCores - 1) / reduceCores;
    splitReduceM = splitReduceM >= Policy::UBTileShapeforBRed_M ? Policy::UBTileShapeforBRed_M : splitReduceM;
    splitReduceM = splitReduceM < 2 ? 2 : splitReduceM;

    const uint32_t splitReduceNNum = (k + Policy::UBTileShapeforBRed_N - 1) / Policy::UBTileShapeforBRed_N;
    uint32_t splitReduceN = Policy::UBTileShapeforBRed_N;
    if (splitReduceNNum < 2) {
        splitReduceN = (k + 1) / 2;
    }

    const uint32_t nRemainSplit = n % Policy::L1TileShape_N;
    const float commonSize = static_cast<float>(k) * Policy::L1TileShape_N;
    const float remainSize = static_cast<float>(k) * nRemainSplit;

    const float commonStdFactor = std::sqrt(2.0f * std::log(commonSize));
    float remainStdFactor = commonStdFactor;
    float remainKnRatio = commonSize;
    float remainKnSqrtRatio = std::sqrt(commonSize);
    float remainKSqrtNRatio = std::sqrt(static_cast<float>(k)) * Policy::L1TileShape_N;
    if (nRemainSplit > 0) {
        remainStdFactor = std::sqrt(2.0f * std::log(remainSize));
        remainKnRatio = remainSize;
        remainKnSqrtRatio = std::sqrt(static_cast<float>(nRemainSplit));
        remainKSqrtNRatio = std::sqrt(static_cast<float>(k)) * nRemainSplit;
    }

    MatmulFtComputeTilingData tiling{};
    tiling.problemGemmShape = {m, n, k};
    tiling.problemGemmShapeFirst = {m, firstBlockN, k};
    tiling.problemGemmShapeRemain = {m, remainBlockN, k};
    tiling.problemShape = {m, n};
    tiling.problemShapeCol = {n, m};
    tiling.problemCompShape = {1, totalInputElements};
    tiling.problemSliceShape = {splitNNum, m};
    tiling.totalInputElements = totalInputElements;
    tiling.totalOutputElements = totalOutputElements;
    tiling.rowOutputElements = rowOutputElements;
    tiling.colOutputElements = colOutputElements;
    tiling.xLen = xLen;
    tiling.layoutThreLen = m;
    tiling.roundingAlpha = 1.0f;
    tiling.eMax = eMax * 6 * std::sqrt(static_cast<float>(k) / 1024.0f);
    tiling.stdEstARowRatio = 1.0f / std::sqrt(2.0f * std::log(static_cast<float>(k)));
    tiling.aRowScaleRatio = 1.0f / static_cast<float>(k);

    tiling.stdEstRatios[0] = 1.0f / commonStdFactor;
    tiling.stdEstRatios[1] = 1.0f / remainStdFactor;
    tiling.knRatios[0] = commonSize;
    tiling.knRatios[1] = remainKnRatio;
    tiling.knScaleRatios[0] = 1.0f / commonSize;
    tiling.knScaleRatios[1] = 1.0f / remainKnRatio;
    tiling.knSqrtRatios[0] = std::sqrt(commonSize);
    tiling.knSqrtRatios[1] = remainKnSqrtRatio;
    tiling.kSqrtNRatios[0] = std::sqrt(static_cast<float>(k)) * Policy::L1TileShape_N;
    tiling.kSqrtNRatios[1] = remainKSqrtNRatio;

    tiling.nScaleRatios[0] = 1.0f / Policy::L1TileShape_N;
    tiling.nRatios[0] = static_cast<float>(Policy::L1TileShape_N);
    tiling.nSqrtRatios[0] = std::sqrt(static_cast<float>(Policy::L1TileShape_N));
    tiling.nSquareRatios[0] = static_cast<float>(Policy::L1TileShape_N * Policy::L1TileShape_N);
    const uint32_t remainN = nRemainSplit > 0 ? nRemainSplit : Policy::L1TileShape_N;
    tiling.nScaleRatios[1] = 1.0f / remainN;
    tiling.nRatios[1] = static_cast<float>(remainN);
    tiling.nSqrtRatios[1] = std::sqrt(static_cast<float>(remainN));
    tiling.nSquareRatios[1] = static_cast<float>(remainN * remainN);

    tiling.splitNNum = splitNNum;
    tiling.splitReduceM = splitReduceM;
    tiling.splitReduceN = splitReduceN;
    tiling.splitKNum = splitKs;
    tiling.outputThre = 1;
    tiling.outputCE = 1;
    tiling.outputWorkspace = 0;
    return tiling;
}


static ge::graphStatus MatmulFtTilingFunc(gert::TilingContext* context)
{
    const gert::StorageShape* aShape = context->GetInputShape(0);
    const gert::StorageShape* bShape = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, aShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, bShape);

    const gert::Shape& aStorageShape = aShape->GetStorageShape();
    const gert::Shape& bStorageShape = bShape->GetStorageShape();
    OP_CHECK_IF(aStorageShape.GetDimNum() != 2 || bStorageShape.GetDimNum() != 2,
        OP_LOGE(context, "MatmulFt expects rank-2 A and B"), return ge::GRAPH_FAILED);

    const int64_t m = aStorageShape.GetDim(0);
    const int64_t k = aStorageShape.GetDim(1);
    const int64_t bK = bStorageShape.GetDim(0);
    const int64_t n = bStorageShape.GetDim(1);
    OP_CHECK_IF(m <= 0 || n <= 0 || k <= 0 || k != bK,
        OP_LOGE(context, "invalid MatmulFt problem shape"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(m > UINT32_MAX || n > UINT32_MAX || k > UINT32_MAX,
        OP_LOGE(context, "MatmulFt shape exceeds uint32 range"), return ge::GRAPH_FAILED);

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    uint32_t idx = 0;
    const float* eMaxAttr = attrs->GetAttrPointer<float>(idx++);
    const int64_t* reduceCoresAttr = attrs->GetAttrPointer<int64_t>(idx++);
    OP_CHECK_NULL_WITH_CONTEXT(context, eMaxAttr);
    OP_CHECK_NULL_WITH_CONTEXT(context, reduceCoresAttr);
    OP_CHECK_IF(*reduceCoresAttr <= 0,
        OP_LOGE(context, "reduce_cores must be greater than 0"), return ge::GRAPH_FAILED);

    const float eMax = *eMaxAttr;
    const uint32_t reduceCores = static_cast<uint32_t>(*reduceCoresAttr);
    const uint32_t splitKs = 1;
    const auto* aDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, aDesc);
    const ge::DataType aDtype = aDesc->GetDataType();

    auto platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfoPtr == nullptr,
        OP_LOGE(context, "platformInfoPtr is null"), return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    const uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    OP_CHECK_IF(aicNum == 0,
        OP_LOGE(context, "invalid AIC core count"), return ge::GRAPH_FAILED);

    MatmulFtTilingData* tiling = context->GetTilingData<MatmulFtTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    if (aDtype == ge::DT_FLOAT16) {
        tiling->compute = BuildComputeTiling<MatmulFtFp16TilingPolicy>(
            static_cast<uint32_t>(m), static_cast<uint32_t>(n), static_cast<uint32_t>(k),
            eMax, reduceCores, splitKs);
    } else if (aDtype == ge::DT_FLOAT) {
        tiling->compute = BuildComputeTiling<MatmulFtFp32TilingPolicy>(
            static_cast<uint32_t>(m), static_cast<uint32_t>(n), static_cast<uint32_t>(k),
            eMax, reduceCores, splitKs);
    } else {
        tiling->compute = BuildComputeTiling<MatmulFtBf16TilingPolicy>(
            static_cast<uint32_t>(m), static_cast<uint32_t>(n), static_cast<uint32_t>(k),
            eMax, reduceCores, splitKs);
    }

    size_t* workspaceSizes = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaceSizes);
    uint64_t userWorkspaceElements = static_cast<uint64_t>(tiling->compute.splitNNum) + 1;
    OP_CHECK_IF(tiling->compute.problemGemmShape.m >
            std::numeric_limits<uint64_t>::max() / userWorkspaceElements,
        OP_LOGE(context, "MatmulFt workspace size overflow"), return ge::GRAPH_FAILED);
    userWorkspaceElements *= tiling->compute.problemGemmShape.m;
    OP_CHECK_IF(tiling->compute.splitKNum >
            std::numeric_limits<uint64_t>::max() / userWorkspaceElements,
        OP_LOGE(context, "MatmulFt workspace size overflow"), return ge::GRAPH_FAILED);
    userWorkspaceElements *= tiling->compute.splitKNum;
    OP_CHECK_IF(userWorkspaceElements >
            (std::numeric_limits<size_t>::max() - SYSTEM_WORKSPACE_SIZE) / sizeof(float),
        OP_LOGE(context, "MatmulFt workspace size overflow"), return ge::GRAPH_FAILED);
    workspaceSizes[0] = SYSTEM_WORKSPACE_SIZE +
        static_cast<size_t>(userWorkspaceElements) * sizeof(float);

    context->SetBlockDim(aicNum);
    context->SetTilingKey(GET_TPL_TILING_KEY(MATMUL_FT_TPL_SCH_MODE_BF16));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MatmulFt).Tiling(MatmulFtTilingFunc);
} // namespace optiling
