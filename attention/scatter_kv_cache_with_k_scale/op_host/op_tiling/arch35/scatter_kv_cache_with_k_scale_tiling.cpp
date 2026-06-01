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
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "../../../op_kernel/arch35/scatter_kv_cache_with_k_scale_tiling_data.h"
#include "../../../op_kernel/arch35/scatter_kv_cache_with_k_scale_tiling_key.h"

namespace optiling {

using namespace Ops::Transformer::OpTiling;

// ==================== 常量定义 ====================

// 平台相关常量
constexpr uint32_t DCACHE_SIZE = 32 * 1024;
constexpr uint32_t STATIC_UB_ESTIMATE = 0;

// 维度数量常量
constexpr int64_t KEY_DIM_NUM = 3;
constexpr int64_t KEY_CACHE_DIM_NUM = 4;
constexpr int64_t VALUE_CACHE_DIM_NUM = 4;
constexpr int64_t SLOT_MAPPING_DIM_NUM = 1;
constexpr int64_t KEY_SCALE_DIM_NUM = 2;
constexpr int64_t KEY_SCALE_CACHE_DIM_NUM = 4;

// 输入tensor索引常量
constexpr size_t INPUT_KEY_IDX = 0;
constexpr size_t INPUT_VALUE_IDX = 1;
constexpr size_t INPUT_KEY_CACHE_IDX = 2;
constexpr size_t INPUT_VALUE_CACHE_IDX = 3;
constexpr size_t INPUT_SLOT_MAPPING_IDX = 4;
constexpr size_t INPUT_KEY_SCALE_IDX = 5;
constexpr size_t INPUT_KEY_SCALE_CACHE_IDX = 6;

// 输出tensor索引常量
constexpr size_t OUTPUT_KEY_CACHE_IDX = 0;
constexpr size_t OUTPUT_VALUE_CACHE_IDX = 1;
constexpr size_t OUTPUT_KEY_SCALE_CACHE_IDX = 2;

// shape维度索引常量
constexpr int64_t DIM_TOKEN = 0;
constexpr int64_t DIM_HEAD = 1;
constexpr int64_t DIM_HEAD_SIZE = 2;
constexpr int64_t DIM_BLOCK = 0;
constexpr int64_t DIM_BLOCK_SIZE = 2;
constexpr int64_t DIM_CACHE_HEAD = 1;
constexpr int64_t DIM_CACHE_HEAD_SIZE = 3;
constexpr int64_t DIM_SLOT_TOKEN = 0;
constexpr int64_t DIM_SCALE_TOKEN = 0;
constexpr int64_t DIM_SCALE_HEAD = 1;
constexpr int64_t DIM_SCALE_CACHE_BLOCK = 0;
constexpr int64_t DIM_SCALE_CACHE_HEAD = 1;
constexpr int64_t DIM_SCALE_CACHE_BLOCK_SIZE = 2;
constexpr int64_t DIM_SCALE_CACHE_SIZE = 3;

// 核心数相关常量
constexpr int64_t MIN_CORE_NUM = 1;

// 字节大小常量
constexpr size_t SIZEOF_32BIT = 4;
constexpr size_t SIZEOF_64BIT = 8;

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

// ==================== dtype validation ====================

static ge::graphStatus ValidateDtype(gert::TilingContext *context)
{
    auto keyDesc = context->GetInputDesc(INPUT_KEY_IDX);
    auto valueDesc = context->GetInputDesc(INPUT_VALUE_IDX);
    auto kcDesc = context->GetInputDesc(INPUT_KEY_CACHE_IDX);
    auto vcDesc = context->GetInputDesc(INPUT_VALUE_CACHE_IDX);
    auto smDesc = context->GetInputDesc(INPUT_SLOT_MAPPING_IDX);
    auto ksDesc = context->GetInputDesc(INPUT_KEY_SCALE_IDX);
    auto kscDesc = context->GetInputDesc(INPUT_KEY_SCALE_CACHE_IDX);
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

// ==================== shape validation ====================

static ge::graphStatus ValidateShape(gert::TilingContext *context, int64_t numTokens, int64_t numHead,
                                     int64_t kHeadSize, int64_t vHeadSize, int64_t numBlocks, int64_t blockSize)
{
    auto sValue = context->GetInputShape(INPUT_VALUE_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, sValue);
    auto valueShape = EnsureNotScalar(sValue->GetStorageShape());
    OP_CHECK_IF((valueShape.GetDim(DIM_HEAD) != numHead), OP_LOGE(context, "value numHead mismatch"),
                return ge::GRAPH_FAILED);

    auto sKc = context->GetInputShape(INPUT_KEY_CACHE_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, sKc);
    auto kcShape = EnsureNotScalar(sKc->GetStorageShape());
    OP_CHECK_IF((kcShape.GetDimNum() != KEY_CACHE_DIM_NUM), OP_LOGE(context, "key_cache must be 4D"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((kcShape.GetDim(DIM_CACHE_HEAD) != numHead), OP_LOGE(context, "key_cache numHead mismatch"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((kcShape.GetDim(DIM_CACHE_HEAD_SIZE) != kHeadSize), OP_LOGE(context, "key_cache kHeadSize mismatch"),
                return ge::GRAPH_FAILED);

    auto sVc = context->GetInputShape(INPUT_VALUE_CACHE_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, sVc);
    auto vcShape = EnsureNotScalar(sVc->GetStorageShape());
    OP_CHECK_IF((vcShape.GetDimNum() != VALUE_CACHE_DIM_NUM), OP_LOGE(context, "value_cache must be 4D"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((vcShape.GetDim(DIM_CACHE_HEAD) != numHead), OP_LOGE(context, "value_cache numHead mismatch"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((vcShape.GetDim(DIM_CACHE_HEAD_SIZE) != vHeadSize), OP_LOGE(context, "value_cache vHeadSize mismatch"),
                return ge::GRAPH_FAILED);

    auto sSm = context->GetInputShape(INPUT_SLOT_MAPPING_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, sSm);
    auto smShape = EnsureNotScalar(sSm->GetStorageShape());
    OP_CHECK_IF((smShape.GetDim(DIM_SLOT_TOKEN) != numTokens), OP_LOGE(context, "slot_mapping shape[0] != numTokens"),
                return ge::GRAPH_FAILED);

    auto sKs = context->GetInputShape(INPUT_KEY_SCALE_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, sKs);
    auto ksShape = EnsureNotScalar(sKs->GetStorageShape());
    OP_CHECK_IF((ksShape.GetDimNum() != KEY_SCALE_DIM_NUM || ksShape.GetDim(DIM_SCALE_TOKEN) != numTokens ||
                 ksShape.GetDim(DIM_SCALE_HEAD) != numHead),
                OP_LOGE(context, "key_scale shape invalid"), return ge::GRAPH_FAILED);

    auto sKsc = context->GetInputShape(INPUT_KEY_SCALE_CACHE_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, sKsc);
    auto kscShape = EnsureNotScalar(sKsc->GetStorageShape());
    OP_CHECK_IF(
        (kscShape.GetDimNum() != KEY_SCALE_CACHE_DIM_NUM || kscShape.GetDim(DIM_SCALE_CACHE_BLOCK) != numBlocks ||
         kscShape.GetDim(DIM_SCALE_CACHE_HEAD) != numHead || kscShape.GetDim(DIM_SCALE_CACHE_BLOCK_SIZE) != blockSize ||
         kscShape.GetDim(DIM_SCALE_CACHE_SIZE) != 1),
        OP_LOGE(context, "key_scale_cache shape invalid"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// ==================== stride copy helpers ====================

template <size_t N>
static void CopyStrides(int64_t *dst, const int64_t *src)
{
    for (size_t i = 0; i < N; ++i) {
        dst[i] = src[i];
    }
}

// ==================== extract tensor parameters ====================

struct TensorParams {
    int64_t numTokens;
    int64_t numHead;
    int64_t kHeadSize;
    int64_t vHeadSize;
    int64_t numBlocks;
    int64_t blockSize;
    int64_t maxSlot;
    int64_t keyStride[KEY_DIM_NUM];
    int64_t valueStride[KEY_DIM_NUM];
    int64_t keyCacheStride[KEY_CACHE_DIM_NUM];
    int64_t valueCacheStride[VALUE_CACHE_DIM_NUM];
    int64_t slotMappingStride[SLOT_MAPPING_DIM_NUM];
    int64_t keyScaleStride[KEY_SCALE_DIM_NUM];
    int64_t keyScaleCacheStride[KEY_SCALE_CACHE_DIM_NUM];
};

static ge::graphStatus ExtractTensorParams(gert::TilingContext *context, TensorParams &params)
{
    int64_t keyShapeArr[KEY_DIM_NUM];
    int64_t keyStrideArr[KEY_DIM_NUM];
    int64_t keyDim;
    GetTensorInfo(context, INPUT_KEY_IDX, keyShapeArr, keyStrideArr, keyDim);
    OP_CHECK_IF((keyDim != KEY_DIM_NUM), OP_LOGE(context, "key must be 3D"), return ge::GRAPH_FAILED);
    params.numTokens = keyShapeArr[DIM_TOKEN];
    params.numHead = keyShapeArr[DIM_HEAD];
    params.kHeadSize = keyShapeArr[DIM_HEAD_SIZE];
    CopyStrides<KEY_DIM_NUM>(params.keyStride, keyStrideArr);

    int64_t valueShapeArr[KEY_DIM_NUM];
    int64_t valueStrideArr[KEY_DIM_NUM];
    int64_t valueDim;
    GetTensorInfo(context, INPUT_VALUE_IDX, valueShapeArr, valueStrideArr, valueDim);
    params.vHeadSize = valueShapeArr[DIM_HEAD_SIZE];
    CopyStrides<KEY_DIM_NUM>(params.valueStride, valueStrideArr);

    int64_t kcShapeArr[KEY_CACHE_DIM_NUM];
    int64_t kcStrideArr[KEY_CACHE_DIM_NUM];
    int64_t kcDim;
    GetTensorInfo(context, INPUT_KEY_CACHE_IDX, kcShapeArr, kcStrideArr, kcDim);
    OP_CHECK_IF((kcDim != KEY_CACHE_DIM_NUM), OP_LOGE(context, "key_cache must be 4D"), return ge::GRAPH_FAILED);
    params.numBlocks = kcShapeArr[DIM_BLOCK];
    params.blockSize = kcShapeArr[DIM_BLOCK_SIZE];
    CopyStrides<KEY_CACHE_DIM_NUM>(params.keyCacheStride, kcStrideArr);

    int64_t vcShapeArr[VALUE_CACHE_DIM_NUM];
    int64_t vcStrideArr[VALUE_CACHE_DIM_NUM];
    int64_t vcDim;
    GetTensorInfo(context, INPUT_VALUE_CACHE_IDX, vcShapeArr, vcStrideArr, vcDim);
    CopyStrides<VALUE_CACHE_DIM_NUM>(params.valueCacheStride, vcStrideArr);

    int64_t smShapeArr[SLOT_MAPPING_DIM_NUM];
    int64_t smStrideArr[SLOT_MAPPING_DIM_NUM];
    int64_t smDim;
    GetTensorInfo(context, INPUT_SLOT_MAPPING_IDX, smShapeArr, smStrideArr, smDim);
    CopyStrides<SLOT_MAPPING_DIM_NUM>(params.slotMappingStride, smStrideArr);

    int64_t ksShapeArr[KEY_SCALE_DIM_NUM];
    int64_t ksStrideArr[KEY_SCALE_DIM_NUM];
    int64_t ksDim;
    GetTensorInfo(context, INPUT_KEY_SCALE_IDX, ksShapeArr, ksStrideArr, ksDim);
    CopyStrides<KEY_SCALE_DIM_NUM>(params.keyScaleStride, ksStrideArr);

    int64_t kscShapeArr[KEY_SCALE_CACHE_DIM_NUM];
    int64_t kscStrideArr[KEY_SCALE_CACHE_DIM_NUM];
    int64_t kscDim;
    GetTensorInfo(context, INPUT_KEY_SCALE_CACHE_IDX, kscShapeArr, kscStrideArr, kscDim);
    CopyStrides<KEY_SCALE_CACHE_DIM_NUM>(params.keyScaleCacheStride, kscStrideArr);

    params.maxSlot = params.numBlocks * params.blockSize;

    return ge::GRAPH_SUCCESS;
}

// ==================== fill tiling data ====================

static void FillTilingData(ScatterKvCacheWithKScaleTilingData *tiling, const TensorParams &params, int64_t coreNum)
{
    tiling->numTokens = params.numTokens;
    tiling->numHead = params.numHead;
    tiling->kHeadSize = params.kHeadSize;
    tiling->vHeadSize = params.vHeadSize;
    tiling->numBlocks = params.numBlocks;
    tiling->blockSize = params.blockSize;
    tiling->maxSlot = params.maxSlot;

    CopyStrides<KEY_DIM_NUM>(tiling->keyStride, params.keyStride);
    CopyStrides<KEY_DIM_NUM>(tiling->valueStride, params.valueStride);
    CopyStrides<KEY_CACHE_DIM_NUM>(tiling->keyCacheStride, params.keyCacheStride);
    CopyStrides<VALUE_CACHE_DIM_NUM>(tiling->valueCacheStride, params.valueCacheStride);
    CopyStrides<SLOT_MAPPING_DIM_NUM>(tiling->slotMappingStride, params.slotMappingStride);
    CopyStrides<KEY_SCALE_DIM_NUM>(tiling->keyScaleStride, params.keyScaleStride);
    CopyStrides<KEY_SCALE_CACHE_DIM_NUM>(tiling->keyScaleCacheStride, params.keyScaleCacheStride);

    int64_t needCoreNum = std::max(MIN_CORE_NUM, std::min(params.numTokens, coreNum));
    tiling->needCoreNum = needCoreNum;
}

// ==================== dump tiling info ====================

static void DumpTilingInfo(gert::TilingContext *context, const ScatterKvCacheWithKScaleTilingData *tiling)
{
    OP_LOGI(context, "==================== ScatterKvCacheWithKScale Tiling Info ====================");
    OP_LOGI(context, "Basic Parameters:");
    OP_LOGI(context, "  numTokens      : %lld", tiling->numTokens);
    OP_LOGI(context, "  numHead        : %lld", tiling->numHead);
    OP_LOGI(context, "  kHeadSize      : %lld", tiling->kHeadSize);
    OP_LOGI(context, "  vHeadSize      : %lld", tiling->vHeadSize);
    OP_LOGI(context, "  numBlocks      : %lld", tiling->numBlocks);
    OP_LOGI(context, "  blockSize      : %lld", tiling->blockSize);
    OP_LOGI(context, "  maxSlot        : %lld", tiling->maxSlot);
    OP_LOGI(context, "  needCoreNum    : %lld", tiling->needCoreNum);

    OP_LOGI(context, "Strides:");
    OP_LOGI(context, "  keyStride      : [%lld, %lld, %lld]", tiling->keyStride[0], tiling->keyStride[1],
            tiling->keyStride[2]);
    OP_LOGI(context, "  valueStride    : [%lld, %lld, %lld]", tiling->valueStride[0], tiling->valueStride[1],
            tiling->valueStride[2]);
    OP_LOGI(context, "  keyCacheStride : [%lld, %lld, %lld, %lld]", tiling->keyCacheStride[0],
            tiling->keyCacheStride[1], tiling->keyCacheStride[2], tiling->keyCacheStride[3]);
    OP_LOGI(context, "  valueCacheStride: [%lld, %lld, %lld, %lld]", tiling->valueCacheStride[0],
            tiling->valueCacheStride[1], tiling->valueCacheStride[2], tiling->valueCacheStride[3]);
    OP_LOGI(context, "  slotMappingStride: [%lld]", tiling->slotMappingStride[0]);
    OP_LOGI(context, "  keyScaleStride : [%lld, %lld]", tiling->keyScaleStride[0], tiling->keyScaleStride[1]);
    OP_LOGI(context, "  keyScaleCacheStride: [%lld, %lld, %lld, %lld]", tiling->keyScaleCacheStride[0],
            tiling->keyScaleCacheStride[1], tiling->keyScaleCacheStride[2], tiling->keyScaleCacheStride[3]);

    OP_LOGI(context, "Scene Mode: %s",
            (tiling->kHeadSize == tiling->vHeadSize) ? "SPECIALIZED (kHeadSize == vHeadSize)" :
                                                       "GENERALIZED (kHeadSize != vHeadSize)");
    OP_LOGI(context, "===============================================================================");
}

// ==================== set tiling key ====================

static void SetTilingKeyByScene(gert::TilingContext *context, int64_t kHeadSize, int64_t vHeadSize)
{
    uint64_t tilingKey;
    if (kHeadSize == vHeadSize) {
        tilingKey = GET_TPL_TILING_KEY(SCATTER_KV_CACHE_SCENE_SPECIALIZED);
    } else {
        tilingKey = GET_TPL_TILING_KEY(SCATTER_KV_CACHE_SCENE_GENERALIZED);
    }
    context->SetTilingKey(tilingKey);
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

    TensorParams params;
    OP_CHECK_IF(ExtractTensorParams(context, params) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "ExtractTensorParams error"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(ValidateShape(context, params.numTokens, params.numHead, params.kHeadSize, params.vHeadSize,
                              params.numBlocks, params.blockSize) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "ValidateShape error"), return ge::GRAPH_FAILED);

    ScatterKvCacheWithKScaleTilingData *tiling = context->GetTilingData<ScatterKvCacheWithKScaleTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(ScatterKvCacheWithKScaleTilingData), 0,
                         sizeof(ScatterKvCacheWithKScaleTilingData)) != EOK,
                OP_LOGE(context, "memset tiling data error"), return ge::GRAPH_FAILED);

    FillTilingData(tiling, params, coreNum);
    context->SetBlockDim(tiling->needCoreNum);

    DumpTilingInfo(context, tiling);

    OP_CHECK_IF((ubSize <= DCACHE_SIZE + STATIC_UB_ESTIMATE), OP_LOGE(context, "ubSize too small"),
                return ge::GRAPH_FAILED);
    auto res = context->SetLocalMemorySize(static_cast<uint32_t>(ubSize - DCACHE_SIZE - STATIC_UB_ESTIMATE));
    OP_CHECK_IF((res != ge::GRAPH_SUCCESS), OP_LOGE(context, "SetLocalMemorySize failed"), return ge::GRAPH_FAILED);

    SetTilingKeyByScene(context, params.kHeadSize, params.vHeadSize);
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