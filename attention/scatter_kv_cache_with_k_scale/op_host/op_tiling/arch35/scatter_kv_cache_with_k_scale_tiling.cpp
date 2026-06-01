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
 * \file scatter_kv_cache_with_k_scale_tiling.cpp
 * \brief Tiling implementation for scatter_kv_cache_with_k_scale
 */

#include "platform/platform_ascendc.h"
#include "util/math_util.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "../../../op_kernel/arch35/scatter_kv_cache_with_k_scale_tiling_data.h"
#include "../../../op_kernel/arch35/scatter_kv_cache_with_k_scale_tiling_key.h"

namespace optiling {

using namespace Ops::Transformer::OpTiling;

constexpr uint32_t DCACHE_SIZE = 32 * 1024;
constexpr uint32_t STATIC_UB_ESTIMATE = 0;

struct ScatterKvCacheWithKScaleCompileInfo {};

// ==================== tensor info (shape + stride, align gather_elements) ====================

static void GetTensorInfo(gert::TilingContext *ctx, size_t idx, int64_t *shape, int64_t *stride, int64_t &dimNum)
{
    auto *shapeRef = ctx->GetInputShape(idx);

    bool isView = ctx->InputIsView(idx);
    if (isView) {
        auto *s = ctx->GetInputStride(idx);
        if (s != nullptr && s->GetDimNum() > 0) {
            auto viewShape = shapeRef->GetShape();
            dimNum = viewShape.GetDimNum();
            for (int64_t i = 0; i < dimNum; ++i) {
                shape[i] = viewShape.GetDim(i);
                stride[i] = s->GetStride(i);
            }
            return;
        }
    }

    auto storageShape = EnsureNotScalar(shapeRef->GetStorageShape());
    dimNum = storageShape.GetDimNum();
    int64_t st = 1;
    for (int32_t i = static_cast<int32_t>(dimNum) - 1; i >= 0; --i) {
        shape[i] = storageShape.GetDim(i);
        stride[i] = st;
        st *= storageShape.GetDim(i);
    }
}

// ==================== platform / workspace ====================

static ge::graphStatus GetPlatformInfo(gert::TilingContext *context, uint64_t &ubSize, int64_t &coreNum)
{
    fe::PlatFormInfos *platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext *context)
{
    int64_t userWorkspaceSize = 0;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = static_cast<size_t>(userWorkspaceSize + static_cast<int64_t>(sysWorkspaceSize));
    return ge::GRAPH_SUCCESS;
}

// ==================== dtype / shape validation ====================

static ge::graphStatus ValidateDtype(gert::TilingContext *context)
{
    auto keyDesc = context->GetInputDesc(0);
    auto valueDesc = context->GetInputDesc(1);
    auto kcDesc = context->GetInputDesc(2);
    auto vcDesc = context->GetInputDesc(3);
    auto smDesc = context->GetInputDesc(4);
    auto ksDesc = context->GetInputDesc(5);
    auto kscDesc = context->GetInputDesc(6);
    OP_CHECK_NULL_WITH_CONTEXT(context, keyDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, valueDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, kcDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, vcDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, smDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, ksDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, kscDesc);

    auto keyDtype = keyDesc->GetDataType();
    auto valueDtype = valueDesc->GetDataType();
    auto kcDtype = kcDesc->GetDataType();
    auto vcDtype = vcDesc->GetDataType();
    auto smDtype = smDesc->GetDataType();
    auto ksDtype = ksDesc->GetDataType();
    auto kscDtype = kscDesc->GetDataType();

    OP_CHECK_IF((keyDtype != ge::DT_FLOAT8_E5M2 && keyDtype != ge::DT_FLOAT8_E4M3FN),
                OP_LOGE(context, "key dtype must be FP8_E5M2 or FP8_E4M3FN, got %d", keyDtype),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((keyDtype != valueDtype || keyDtype != kcDtype || keyDtype != vcDtype),
                OP_LOGE(context, "key/value/key_cache/value_cache dtype must be same"), return ge::GRAPH_FAILED);
    OP_CHECK_IF((smDtype != ge::DT_INT32 && smDtype != ge::DT_INT64),
                OP_LOGE(context, "slot_mapping dtype must be DT_INT32 or DT_INT64, got %d", smDtype),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((ksDtype != ge::DT_FLOAT || kscDtype != ge::DT_FLOAT),
                OP_LOGE(context, "key_scale and key_scale_cache dtype must be DT_FLOAT"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ValidateShape(gert::TilingContext *context, int64_t numTokens, int64_t numHead,
                                     int64_t kHeadSize, int64_t vHeadSize, int64_t numBlocks, int64_t blockSize)
{
    auto sValue = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, sValue);
    auto valueShape = EnsureNotScalar(sValue->GetStorageShape());
    OP_CHECK_IF((valueShape.GetDim(1) != numHead), OP_LOGE(context, "value numHead mismatch"), return ge::GRAPH_FAILED);

    auto sKc = context->GetInputShape(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, sKc);
    auto kcShape = EnsureNotScalar(sKc->GetStorageShape());
    OP_CHECK_IF((kcShape.GetDimNum() != 4), OP_LOGE(context, "key_cache must be 4D"), return ge::GRAPH_FAILED);
    OP_CHECK_IF((kcShape.GetDim(1) != numHead), OP_LOGE(context, "key_cache numHead mismatch"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((kcShape.GetDim(3) != kHeadSize), OP_LOGE(context, "key_cache kHeadSize mismatch"),
                return ge::GRAPH_FAILED);

    auto sVc = context->GetInputShape(3);
    OP_CHECK_NULL_WITH_CONTEXT(context, sVc);
    auto vcShape = EnsureNotScalar(sVc->GetStorageShape());
    OP_CHECK_IF((vcShape.GetDimNum() != 4), OP_LOGE(context, "value_cache must be 4D"), return ge::GRAPH_FAILED);
    OP_CHECK_IF((vcShape.GetDim(1) != numHead), OP_LOGE(context, "value_cache numHead mismatch"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((vcShape.GetDim(3) != vHeadSize), OP_LOGE(context, "value_cache vHeadSize mismatch"),
                return ge::GRAPH_FAILED);

    auto sSm = context->GetInputShape(4);
    OP_CHECK_NULL_WITH_CONTEXT(context, sSm);
    auto smShape = EnsureNotScalar(sSm->GetStorageShape());
    OP_CHECK_IF((smShape.GetDim(0) != numTokens), OP_LOGE(context, "slot_mapping shape[0] != numTokens"),
                return ge::GRAPH_FAILED);

    auto sKs = context->GetInputShape(5);
    OP_CHECK_NULL_WITH_CONTEXT(context, sKs);
    auto ksShape = EnsureNotScalar(sKs->GetStorageShape());
    OP_CHECK_IF((ksShape.GetDimNum() != 2 || ksShape.GetDim(0) != numTokens || ksShape.GetDim(1) != numHead),
                OP_LOGE(context, "key_scale shape invalid"), return ge::GRAPH_FAILED);

    auto sKsc = context->GetInputShape(6);
    OP_CHECK_NULL_WITH_CONTEXT(context, sKsc);
    auto kscShape = EnsureNotScalar(sKsc->GetStorageShape());
    OP_CHECK_IF((kscShape.GetDimNum() != 4 || kscShape.GetDim(0) != numBlocks || kscShape.GetDim(1) != numHead ||
                 kscShape.GetDim(2) != blockSize || kscShape.GetDim(3) != 1),
                OP_LOGE(context, "key_scale_cache shape invalid"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// ==================== main tiling ====================

static ge::graphStatus ScatterKvCacheWithKScaleTilingFunc(gert::TilingContext *context)
{
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ValidateDtype(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "ValidateDtype error"),
                return ge::GRAPH_FAILED);

    int64_t keyShapeArr[3];
    int64_t keyStrideArr[3];
    int64_t keyDim;
    GetTensorInfo(context, 0, keyShapeArr, keyStrideArr, keyDim);
    OP_CHECK_IF((keyDim != 3), OP_LOGE(context, "key must be 3D"), return ge::GRAPH_FAILED);
    int64_t numTokens = keyShapeArr[0];
    int64_t numHead = keyShapeArr[1];
    int64_t kHeadSize = keyShapeArr[2];

    int64_t valueShapeArr[3];
    int64_t valueStrideArr[3];
    int64_t valueDim;
    GetTensorInfo(context, 1, valueShapeArr, valueStrideArr, valueDim);
    int64_t vHeadSize = valueShapeArr[2];

    int64_t kcShapeArr[4];
    int64_t kcStrideArr[4];
    int64_t kcDim;
    GetTensorInfo(context, 2, kcShapeArr, kcStrideArr, kcDim);
    OP_CHECK_IF((kcDim != 4), OP_LOGE(context, "key_cache must be 4D"), return ge::GRAPH_FAILED);
    int64_t numBlocks = kcShapeArr[0];
    int64_t blockSize = kcShapeArr[2];

    int64_t vcShapeArr[4];
    int64_t vcStrideArr[4];
    int64_t vcDim;
    GetTensorInfo(context, 3, vcShapeArr, vcStrideArr, vcDim);

    int64_t smShapeArr[1];
    int64_t smStrideArr[1];
    int64_t smDim;
    GetTensorInfo(context, 4, smShapeArr, smStrideArr, smDim);

    int64_t ksShapeArr[2];
    int64_t ksStrideArr[2];
    int64_t ksDim;
    GetTensorInfo(context, 5, ksShapeArr, ksStrideArr, ksDim);

    int64_t kscShapeArr[4];
    int64_t kscStrideArr[4];
    int64_t kscDim;
    GetTensorInfo(context, 6, kscShapeArr, kscStrideArr, kscDim);

    OP_CHECK_IF(ValidateShape(context, numTokens, numHead, kHeadSize, vHeadSize, numBlocks, blockSize) !=
                    ge::GRAPH_SUCCESS,
                OP_LOGE(context, "ValidateShape error"), return ge::GRAPH_FAILED);

    ScatterKvCacheWithKScaleTilingData *tiling = context->GetTilingData<ScatterKvCacheWithKScaleTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(ScatterKvCacheWithKScaleTilingData), 0,
                         sizeof(ScatterKvCacheWithKScaleTilingData)) != EOK,
                OP_LOGE(context, "memset tiling data error"), return ge::GRAPH_FAILED);

    tiling->numTokens = numTokens;
    tiling->numHead = numHead;
    tiling->kHeadSize = kHeadSize;
    tiling->vHeadSize = vHeadSize;
    tiling->numBlocks = numBlocks;
    tiling->blockSize = blockSize;
    tiling->maxSlot = numBlocks * blockSize;

    for (int i = 0; i < 3; ++i)
        tiling->keyStride[i] = keyStrideArr[i];
    for (int i = 0; i < 3; ++i)
        tiling->valueStride[i] = valueStrideArr[i];
    for (int i = 0; i < 4; ++i)
        tiling->keyCacheStride[i] = kcStrideArr[i];
    for (int i = 0; i < 4; ++i)
        tiling->valueCacheStride[i] = vcStrideArr[i];
    tiling->slotMappingStride[0] = smStrideArr[0];
    for (int i = 0; i < 2; ++i)
        tiling->keyScaleStride[i] = ksStrideArr[i];
    for (int i = 0; i < 4; ++i)
        tiling->keyScaleCacheStride[i] = kscStrideArr[i];

    int64_t needCoreNum = (numTokens < coreNum) ? numTokens : coreNum;
    needCoreNum = (needCoreNum < 1) ? 1 : needCoreNum;
    tiling->needCoreNum = needCoreNum;
    context->SetBlockDim(needCoreNum);

    OP_CHECK_IF((ubSize <= DCACHE_SIZE + STATIC_UB_ESTIMATE), OP_LOGE(context, "ubSize too small"),
                return ge::GRAPH_FAILED);
    auto res = context->SetLocalMemorySize(static_cast<uint32_t>(ubSize - DCACHE_SIZE - STATIC_UB_ESTIMATE));
    OP_CHECK_IF((res != ge::GRAPH_SUCCESS), OP_LOGE(context, "SetLocalMemorySize failed"), return ge::GRAPH_FAILED);

    uint64_t tilingKey;
    if (kHeadSize == vHeadSize) {
        tilingKey = GET_TPL_TILING_KEY(SCATTER_KV_CACHE_SCENE_SPECIALIZED);
    } else {
        tilingKey = GET_TPL_TILING_KEY(SCATTER_KV_CACHE_SCENE_GENERALIZED);
    }
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForScatterKvCacheWithKScale([[maybe_unused]] gert::TilingParseContext *context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ScatterKvCacheWithKScale)
    .Tiling(ScatterKvCacheWithKScaleTilingFunc)
    .TilingParse<ScatterKvCacheWithKScaleCompileInfo>(TilingParseForScatterKvCacheWithKScale);

} // namespace optiling
