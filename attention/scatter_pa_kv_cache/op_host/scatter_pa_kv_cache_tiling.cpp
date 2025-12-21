/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file scatter_pa_kv_cache_tiling.cpp
 * \brief
 */

#include <algorithm>
#include <string>
#include "tiling/tiling_api.h"
#include "scatter_pa_kv_cache_tiling.h"
#include "log/log.h"
#include "util/math_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_info_def.h"
#include "op_common/op_host/util/platform_util.h"


namespace optiling {
#ifndef UINT_MAX
#define UINT_MAX (__INT_MAX__ * 2U + 1U)
#endif
constexpr uint64_t DIM_0 = 0;
constexpr uint64_t DIM_1 = 1;
constexpr uint64_t DIM_2 = 2;
constexpr uint64_t DIM_3 = 3;
constexpr uint64_t DIM_4 = 4;
constexpr uint64_t DIM_5 = 5;
constexpr uint64_t DIM_6 = 6;
constexpr uint64_t INPUT_KEY_INDEX = 0;
constexpr uint64_t INPUT_KEY_CACHE_INDEX = 1;
constexpr uint64_t INPUT_SLOT_MAPPING_INDEX = 2;
constexpr uint64_t INPUT_VALUE_INDEX = 3;
constexpr uint64_t INPUT_VALUE_CACHE_INDEX = 4;
constexpr uint64_t INPUT_COMPRESS_LENS = 5;
constexpr uint64_t INPUT_COMPRESS_SEQ_OFFSET = 6;
constexpr uint64_t INPUT_SEQ_LENS = 7;
constexpr uint64_t INPUT_CACHE_MODE_INDEX = 0;
constexpr uint64_t INPUT_SCATTER_MODE_INDEX = 1;
constexpr uint64_t INPUT_STRIDES_INDEX = 2;
constexpr uint64_t INPUT_OFFSET_INDEX = 3;
constexpr uint64_t TILING_ID_OBP_NZ = 1000;
constexpr uint64_t TILING_ID_DTYPE_NZ = 10;
constexpr uint64_t TILING_ID_MODE_NZ = 100;
constexpr uint64_t TILING_ID_FULL_NZ = 1;
constexpr uint64_t ALIGN = 32;
constexpr uint64_t RESERVED_BUFFER = 1024;
constexpr uint64_t INT32_DTYPE_SIZE = 4;
constexpr uint64_t INT64_DTYPE_SIZE = 8;
constexpr uint64_t NORM = 0;
constexpr uint64_t PA_NZ = 1;
constexpr uint64_t TASK_MULTIPLE = 2; // Compress_rope模式下KV分核，分核任务量翻倍
constexpr uint64_t TILING_PARA_SIZE = 64;
constexpr uint64_t TILING_ID_DTYPE = 100000000;
constexpr uint64_t TILING_ID_MODE = 10000000;
constexpr uint64_t TILING_ID_MLA = 1000000;
constexpr uint64_t TILING_ID_MLA_FULL = 2000000;
constexpr uint32_t TILING_ID_NCT = 200000;
constexpr uint64_t SMALL_SHAPE = 1000;
constexpr uint64_t UB_SLOT_MAPPING_SIZE = static_cast<uint64_t>(128) * 1024;

constexpr int64_t TEMPLATE_NORMAL = 0;
constexpr int64_t TEMPLATE_NZ = 1;
constexpr int64_t TEMPLATE_ALIBI = 2;
constexpr int64_t TEMPLATE_ROPE = 3;
constexpr int64_t TEMPLATE_SISO = 4;
constexpr int64_t TEMPLATE_OMNI = 5;
constexpr int64_t TEMPLATE_NCT = 6;
constexpr int64_t TEMPLATE_SISO_NCT = 7;

constexpr uint64_t ASCENDC_TOOLS_WORKSPACE = static_cast<uint64_t>(16) * 1024 * 1024;

bool ScatterPaKvCacheMembaseTiling::IsCapable()
{
    return true;
}

ge::graphStatus ScatterPaKvCacheMembaseTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterPaKvCacheMembaseTiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OP_LOGE(context_, "platformInfo is nullptr."), return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    params_.socVersion = ascendcPlatform.GetSocVersion();
    params_.usedCoreNum = static_cast<uint64_t>(ascendcPlatform.GetCoreNumAiv());
    if (params_.usedCoreNum == 0) {
        OP_LOGE(context_, "coreNum is 0");
        return ge::GRAPH_FAILED;
    }
    params_.sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    if (ubSize == static_cast<uint64_t>(0)) {
        OP_LOGE(context_, "ubSize is 0");
        return ge::GRAPH_FAILED;
    }
    params_.ubSize = static_cast<int64_t>(ubSize) - RESERVED_BUFFER;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterPaKvCacheMembaseTiling::GetInputDtype()
{
    auto inputKeyDesc = context_->GetInputDesc(INPUT_KEY_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputKeyDesc);
    ge::DataType inputKeyDtype = inputKeyDesc->GetDataType();
    auto inputKeyCacheDesc = context_->GetInputDesc(INPUT_KEY_CACHE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputKeyCacheDesc);
    ge::DataType inputKeyCacheDtype = inputKeyCacheDesc->GetDataType();
    if (inputKeyDtype != inputKeyCacheDtype) {
        OP_LOGE(context_, "key, key_cache dtype must be same.");
        return ge::GRAPH_FAILED;
    }
    params_.typeByteK = static_cast<int64_t>(GetSizeByDataType(inputKeyDtype));
    if (params_.typeByteK <= 0) {
        OP_LOGE(context_, "get key dtype bytes failed.");
        return ge::GRAPH_FAILED;
    }
    if (!(inputKeyDtype == ge::DT_FLOAT16 || inputKeyDtype == ge::DT_BF16 || inputKeyDtype == ge::DT_INT8)) {
        OP_LOGE(context_, "input key dtype not support.");
        return ge::GRAPH_FAILED;
    }

    if (params_.templateType != TEMPLATE_SISO && params_.templateType != TEMPLATE_SISO_NCT) {
        auto inputValueDesc = context_->GetInputDesc(INPUT_VALUE_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context_, inputValueDesc);
        ge::DataType inputValueDtype = inputValueDesc->GetDataType();
        auto inputValueCacheDesc = context_->GetInputDesc(INPUT_VALUE_CACHE_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context_, inputValueCacheDesc);
        ge::DataType inputValueCacheDtype = inputValueCacheDesc->GetDataType();
        if (inputValueDtype != inputValueCacheDtype) {
            OP_LOGE(context_, "value, value_cache dtype must be same.");
            return ge::GRAPH_FAILED;
        }
        params_.typeByteV = static_cast<int64_t>(GetSizeByDataType(inputValueDtype));
        if (params_.typeByteV <= 0) {
            OP_LOGE(context_, "get value dtype bytes failed.");
            return ge::GRAPH_FAILED;
        }
        if (!(inputValueDtype == ge::DT_FLOAT16 || inputValueDtype == ge::DT_BF16 || inputValueDtype == ge::DT_INT8)) {
            OP_LOGE(context_, "input value dtype not support.");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterPaKvCacheMembaseTiling::GetIndexDtype()
{
    auto inputSlotMappingDesc = context_->GetInputDesc(INPUT_SLOT_MAPPING_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputSlotMappingDesc);
    ge::DataType inputSlotMappingDtype = inputSlotMappingDesc->GetDataType();
    if (inputSlotMappingDtype == ge::DT_INT32) {
        params_.typeByteSlot = INT32_DTYPE_SIZE;
    } else if (inputSlotMappingDtype == ge::DT_INT64) {
        params_.typeByteSlot = INT64_DTYPE_SIZE;
    } else {
        OP_LOGE(context_, "slot_mapping dtype must be int32 or int64");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterPaKvCacheMembaseTiling::CheckInputDimNumUpdate()
{
    inputKeyShape_ = context_->GetInputShape(INPUT_KEY_INDEX)->GetOriginShape();
    inputKeyCacheInShape_ = context_->GetInputShape(INPUT_KEY_CACHE_INDEX)->GetOriginShape();
    slotMappingShape_ = context_->GetInputShape(INPUT_SLOT_MAPPING_INDEX)->GetOriginShape();

    size_t kDimNum = inputKeyShape_.GetDimNum();
    size_t kCacheDimNum = inputKeyCacheInShape_.GetDimNum();
    size_t slotDimNum = slotMappingShape_.GetDimNum();

    OP_CHECK_IF((kDimNum != static_cast<size_t>(DIM_3)), OP_LOGE(context_, "key should be is 3 dim."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF((kCacheDimNum != static_cast<size_t>(DIM_4)), OP_LOGE(context_, "key_cache should be is 4 dim."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF((slotDimNum != static_cast<size_t>(DIM_1)), OP_LOGE(context_, "slot_mapping should be is 1 dim."),
                return ge::GRAPH_FAILED);
    if (params_.templateType != TEMPLATE_SISO && params_.templateType != TEMPLATE_SISO_NCT) {
        inputValueShape_ = context_->GetInputShape(INPUT_VALUE_INDEX)->GetOriginShape();
        inputValueCacheInShape_ = context_->GetInputShape(INPUT_VALUE_CACHE_INDEX)->GetOriginShape();

        size_t vDimNum = inputValueShape_.GetDimNum();
        size_t vCacheDimNum = inputValueCacheInShape_.GetDimNum();

        OP_CHECK_IF((vDimNum != static_cast<size_t>(DIM_3)), OP_LOGE(context_, "value should be is 3 dim."),
                    return ge::GRAPH_FAILED);

        OP_CHECK_IF((vCacheDimNum != static_cast<size_t>(DIM_4)), OP_LOGE(context_, "value_cache should be is 4 dim."),
                    return ge::GRAPH_FAILED);

        bool isSame = (inputKeyShape_.GetDim(DIM_0) == inputValueShape_.GetDim(DIM_0) &&
                       inputKeyShape_.GetDim(DIM_1) == inputValueShape_.GetDim(DIM_1));
        OP_CHECK_IF((!isSame), OP_LOGE(context_, "dim0 and dim1 of key and value should be same."),
                    return ge::GRAPH_FAILED);

        if (params_.templateType != TEMPLATE_NZ) {
            isSame = (inputKeyCacheInShape_.GetDim(DIM_0) == inputValueCacheInShape_.GetDim(DIM_0) &&
                      inputKeyCacheInShape_.GetDim(DIM_1) == inputValueCacheInShape_.GetDim(DIM_1));
            OP_CHECK_IF((!isSame), OP_LOGE(context_, "dim0 and dim1 of key_cache and value_cache should be same."),
                        return ge::GRAPH_FAILED);
        } else {
            isSame = (inputKeyCacheInShape_.GetDim(DIM_0) == inputValueCacheInShape_.GetDim(DIM_0) &&
                      inputKeyCacheInShape_.GetDim(DIM_2) == inputValueCacheInShape_.GetDim(DIM_2));
            OP_CHECK_IF((!isSame), OP_LOGE(context_, "dim0 and dim2 of key_cache and value_cache should be same."),
                        return ge::GRAPH_FAILED);
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterPaKvCacheMembaseTiling::CheckInputDimNumCompress()
{
    compressLensShape_ = context_->GetInputShape(INPUT_COMPRESS_LENS)->GetOriginShape();
    seqLensShape_ = context_->GetInputShape(INPUT_SEQ_LENS)->GetOriginShape();

    size_t compressLensDimNum = compressLensShape_.GetDimNum();
    size_t seqLensDimNum = seqLensShape_.GetDimNum();

    OP_CHECK_IF((compressLensDimNum != static_cast<size_t>(DIM_1)),
                OP_LOGE(context_, "compressLens should be is 1 dim."), return ge::GRAPH_FAILED);
    OP_CHECK_IF((seqLensDimNum != static_cast<size_t>(DIM_1)), OP_LOGE(context_, "seqLensDimNum should be is 1 dim."),
                return ge::GRAPH_FAILED);

    if (params_.templateType == TEMPLATE_ROPE || params_.templateType == TEMPLATE_OMNI) {
        compressSeqOffsetShape_ = context_->GetInputShape(INPUT_SEQ_LENS)->GetOriginShape();
        size_t compressSeqOffsetDimNum = compressSeqOffsetShape_.GetDimNum();
        OP_CHECK_IF((compressSeqOffsetDimNum != static_cast<size_t>(DIM_1)),
                    OP_LOGE(context_, "compressSeqOffsetDimNum should be is 1 dim."), return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterPaKvCacheMembaseTiling::CheckInputDimNum()
{
    if (CheckInputDimNumUpdate() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (params_.templateType == TEMPLATE_ALIBI || params_.templateType == TEMPLATE_ROPE ||
        params_.templateType == TEMPLATE_OMNI) {
        return CheckInputDimNumCompress();
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterPaKvCacheMembaseTiling::CheckInputDimUpdate()
{
    uint64_t numTokens = static_cast<uint64_t>(inputKeyShape_.GetDim(DIM_0));
    uint64_t numHead = static_cast<uint64_t>(inputKeyShape_.GetDim(DIM_1));
    uint64_t kHeadSize = static_cast<uint64_t>(inputKeyShape_.GetDim(DIM_2));
    uint64_t numBlocks = static_cast<uint64_t>(inputKeyCacheInShape_.GetDim(DIM_0));
    uint64_t blockSize = 0;
    if (params_.templateType != TEMPLATE_NZ) {
        blockSize = static_cast<uint64_t>(inputKeyCacheInShape_.GetDim(DIM_1));
    } else {
        blockSize = static_cast<uint64_t>(inputKeyCacheInShape_.GetDim(DIM_2));
    }

    OP_CHECK_IF((numTokens == 0 || numTokens > UINT_MAX), OP_LOGE(context_, "numTokens is invalid."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((numHead == 0 || numHead > UINT_MAX), OP_LOGE(context_, "numHead is invalid."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((kHeadSize == 0 || kHeadSize > UINT_MAX), OP_LOGE(context_, "kHeadSize is invalid."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((blockSize == 0 || blockSize > UINT_MAX), OP_LOGE(context_, "blockSize is invalid."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((numBlocks * blockSize < numTokens),
                OP_LOGE(context_, "numBlocks * blockSize should larger than numTokens."), return ge::GRAPH_FAILED);

    params_.numTokens = numTokens;
    params_.numHead = numHead;
    params_.kHeadSize = kHeadSize;
    params_.blockSize = blockSize;

    if (params_.templateType != TEMPLATE_SISO && params_.templateType != TEMPLATE_SISO_NCT) {
        uint64_t vHeadSize = static_cast<uint64_t>(inputValueShape_.GetDim(DIM_2));
        OP_CHECK_IF((vHeadSize == 0 || vHeadSize > UINT_MAX), OP_LOGE(context_, "vHeadSize is invalid."),
                    return ge::GRAPH_FAILED);
        params_.vHeadSize = vHeadSize;
        if (params_.templateType == TEMPLATE_NZ) {
            uint64_t kCacheDim1 = static_cast<uint64_t>(inputKeyCacheInShape_.GetDim(DIM_1));
            uint64_t kCacheDim3 = static_cast<uint64_t>(inputKeyCacheInShape_.GetDim(DIM_3));
            uint64_t vCacheDim1 = static_cast<uint64_t>(inputValueCacheInShape_.GetDim(DIM_1));
            uint64_t vCacheDim3 = static_cast<uint64_t>(inputValueCacheInShape_.GetDim(DIM_3));

            uint64_t lastDimK = ALIGN / params_.typeByteK;
            uint64_t lastDimV = ALIGN / params_.typeByteV;
            OP_CHECK_IF((kCacheDim1 != numHead * kHeadSize / lastDimK),
                        OP_LOGE(context_, "dim1 of keyCache is invalid."), return ge::GRAPH_FAILED);
            OP_CHECK_IF((vCacheDim1 != numHead * vHeadSize / lastDimV),
                        OP_LOGE(context_, "dim1 of valueCache is invalid."), return ge::GRAPH_FAILED);
            OP_CHECK_IF((kCacheDim3 != lastDimK), OP_LOGE(context_, "dim3 of keyCache is invalid."),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF((vCacheDim3 != lastDimV), OP_LOGE(context_, "dim3 of valueCache is invalid."),
                        return ge::GRAPH_FAILED);
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterPaKvCacheMembaseTiling::CheckInputDimCompress()
{
    auto seqLens = context_->GetOptionalInputTensor(INPUT_SEQ_LENS);
    OP_CHECK_NULL_WITH_CONTEXT(context_, seqLens);

    uint32_t numBatchs = static_cast<uint32_t>(seqLens->GetStorageShape().GetDim(DIM_0));
    OP_CHECK_IF((numBatchs <= 0 || numBatchs > UINT32_MAX), OP_LOGE(context_, "numBatchs is invalid."),
                return ge::GRAPH_FAILED);
    uint32_t numTokens = params_.numHead * numBatchs;
    OP_CHECK_IF((numTokens <= 0 || numTokens > UINT32_MAX), OP_LOGE(context_, "numTokens is invalid."),
                return ge::GRAPH_FAILED);

    params_.numBatchs = numBatchs;
    params_.numTokens = numTokens;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterPaKvCacheMembaseTiling::CheckInputDim()
{
    if (CheckInputDimUpdate() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (params_.templateType == TEMPLATE_ALIBI || params_.templateType == TEMPLATE_ROPE ||
        params_.templateType == TEMPLATE_OMNI) {
        return CheckInputDimCompress();
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterPaKvCacheMembaseTiling::GetTemplateType()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    auto cacheMode = attrs->GetStr(INPUT_CACHE_MODE_INDEX);
    auto scatterMode = attrs->GetStr(INPUT_SCATTER_MODE_INDEX);
    if (strcmp(cacheMode, "PA_NZ") == 0) {
        // entering template nz
        params_.templateType = TEMPLATE_NZ;
    } else {
        if (strcmp(scatterMode, "Rope") == 0) {
            params_.templateType = TEMPLATE_ROPE;
        } else if (strcmp(scatterMode, "Alibi") == 0) {
            params_.templateType = TEMPLATE_ALIBI;
        } else if (strcmp(scatterMode, "Omni") == 0) {
            params_.templateType = TEMPLATE_OMNI;
        } else if (strcmp(scatterMode, "Nct") == 0) {
            params_.templateType = TEMPLATE_NCT;
        } else {
            params_.templateType = TEMPLATE_NORMAL;
        }
    }

    if (context_->GetInputTensor(INPUT_VALUE_CACHE_INDEX)->GetOriginShape().GetDimNum() == 0) {
        if (params_.templateType == TEMPLATE_NCT) {
            params_.templateType = TEMPLATE_SISO_NCT;
        } else {
            params_.templateType = TEMPLATE_SISO;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterPaKvCacheMembaseTiling::GetShapeAttrsInfo()
{
    if (GetTemplateType() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (GetInputDtype() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (GetIndexDtype() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (CheckInputDimNum() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckInputDim() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (params_.templateType == TEMPLATE_NCT || params_.templateType == TEMPLATE_SISO_NCT) {
        auto attrs = context_->GetAttrs();
        auto strides = attrs->GetListInt(INPUT_STRIDES_INDEX);
        auto offsets = attrs->GetListInt(INPUT_OFFSET_INDEX);
        params_.kStride = strides->GetData()[0];
        params_.vStride = strides->GetData()[1];
        params_.kOffset = offsets->GetData()[0];
        params_.vOffset = offsets->GetData()[1];
    }
    params_.cacheMode = PA_NZ;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterPaKvCacheMembaseTiling::DoNormOpTiling()
{
    params_.usedCoreNum = params_.numTokens < params_.usedCoreNum ? params_.numTokens : params_.usedCoreNum;
    uint64_t tilingKey = 0;
    bool isAlign = ((params_.numHead * params_.kHeadSize * params_.typeByteK) % ALIGN == 0 &&
                    (params_.numHead * params_.vHeadSize * params_.typeByteK) % ALIGN == 0);
    if (params_.numTokens < SMALL_SHAPE && isAlign) {
        if (params_.kHeadSize == params_.vHeadSize) {
            tilingKey = TILING_ID_DTYPE * params_.typeByteK + TILING_ID_MODE * params_.templateType;
        } else {
            tilingKey = TILING_ID_DTYPE * params_.typeByteK + TILING_ID_MODE * params_.templateType + TILING_ID_MLA;
        }
    } else {
        tilingKey = TILING_ID_DTYPE * params_.typeByteK + TILING_ID_MODE * params_.templateType + TILING_ID_MLA_FULL;
    }
    params_.tilingKey = tilingKey;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterPaKvCacheMembaseTiling::DoCompressAlibiOpTiling()
{
    uint32_t tilingKey = 0;
    params_.usedCoreNum = params_.numTokens < params_.usedCoreNum ? params_.numTokens : params_.usedCoreNum;

    tilingKey = TILING_ID_DTYPE * params_.typeByteK + TILING_ID_MODE * params_.templateType;
    params_.tilingKey = tilingKey;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterPaKvCacheMembaseTiling::DoCompressRopeAndOmniOpTiling()
{
    uint32_t tilingKey = 0;
    params_.usedCoreNum = params_.numTokens * TASK_MULTIPLE <= params_.usedCoreNum ? params_.numTokens * TASK_MULTIPLE :
                                                                                     params_.usedCoreNum;
    tilingKey = TILING_ID_DTYPE * params_.typeByteK + TILING_ID_MODE * params_.templateType;
    auto xDataType = context_->GetInputDesc(DIM_0)->GetDataType();
    if (xDataType == ge::DataType::DT_BF16) {
        tilingKey += TILING_ID_DTYPE;
    }
    params_.tilingKey = tilingKey;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterPaKvCacheMembaseTiling::DoNzOpTiling()
{
    bool isAlign =
        ((params_.kHeadSize * params_.typeByteK) % ALIGN == 0 && (params_.vHeadSize * params_.typeByteV) % ALIGN == 0);
    OP_CHECK_IF((!isAlign), OP_LOGE(context_, "kHeadSize and vHeadSize should be align to 32."),
                return ge::GRAPH_FAILED);
    bool canLoad = ((params_.numHead * params_.kHeadSize * params_.typeByteK +
                     params_.numHead * params_.vHeadSize * params_.typeByteV) <= params_.ubSize);

    params_.blockFactor = Ops::Base::CeilDiv<uint64_t>(params_.numTokens, params_.usedCoreNum);
    params_.usedCoreNum =
        std::min(Ops::Base::CeilDiv<uint64_t>(params_.numTokens, params_.blockFactor), params_.usedCoreNum);
    params_.tailBlockFactor = params_.numTokens - params_.blockFactor * (params_.usedCoreNum - 1);

    uint64_t tilingKey = 0;
    if (canLoad) {
        tilingKey = TILING_ID_OBP_NZ + TILING_ID_MODE_NZ * static_cast<uint64_t>(PA_NZ) +
                    TILING_ID_DTYPE_NZ * params_.typeByteSlot + TILING_ID_FULL_NZ;
    } else {
        tilingKey = TILING_ID_OBP_NZ + TILING_ID_MODE_NZ * static_cast<uint64_t>(PA_NZ) +
                    TILING_ID_DTYPE_NZ * params_.typeByteSlot;
    }

    params_.tilingKey = tilingKey;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterPaKvCacheMembaseTiling::DoSisoOpTiling()
{
    params_.usedCoreNum = params_.numTokens < params_.usedCoreNum ? params_.numTokens : params_.usedCoreNum;
    uint64_t tilingKey = 0;
    bool isAlign = ((params_.numHead * params_.kHeadSize * params_.typeByteK) % ALIGN == 0);
    if (params_.numTokens < SMALL_SHAPE && isAlign) {
        tilingKey = TILING_ID_DTYPE * params_.typeByteK + TILING_ID_MODE * params_.templateType;
    } else {
        tilingKey = TILING_ID_DTYPE * params_.typeByteK + TILING_ID_MODE * params_.templateType + TILING_ID_MLA_FULL;
    }
    params_.tilingKey = tilingKey;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterPaKvCacheMembaseTiling::DoOpTiling()
{
    if (params_.templateType == TEMPLATE_NORMAL || params_.templateType == TEMPLATE_NCT) {
        return DoNormOpTiling();
    } else if (params_.templateType == TEMPLATE_ROPE || params_.templateType == TEMPLATE_OMNI) {
        return DoCompressRopeAndOmniOpTiling();
    } else if (params_.templateType == TEMPLATE_ALIBI) {
        return DoCompressAlibiOpTiling();
    } else if (params_.templateType == TEMPLATE_NZ) {
        return DoNzOpTiling();
    } else if (params_.templateType == TEMPLATE_SISO || params_.templateType == TEMPLATE_SISO_NCT) {
        return DoSisoOpTiling();
    } else {
        OP_LOGE(context_, "templateType is invalid.");
    }
    return ge::GRAPH_FAILED;
}

uint64_t ScatterPaKvCacheMembaseTiling::GetTilingKey() const
{
    return params_.tilingKey;
}

ge::graphStatus ScatterPaKvCacheMembaseTiling::GetWorkspaceSize()
{
    workspaceSize_ = ASCENDC_TOOLS_WORKSPACE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterPaKvCacheMembaseTiling::PostTiling()
{
    auto workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = workspaceSize_;

    context_->SetBlockDim(params_.usedCoreNum);
    tilingData_.set_usedCoreNum(params_.usedCoreNum);
    tilingData_.set_blockFactor(params_.blockFactor);
    tilingData_.set_tailBlockFactor(params_.tailBlockFactor);
    tilingData_.set_numTokens(params_.numTokens);
    tilingData_.set_kHeadSize(params_.kHeadSize);
    tilingData_.set_vHeadSize(params_.vHeadSize);
    tilingData_.set_numHead(params_.numHead);
    tilingData_.set_blockSize(params_.blockSize);
    tilingData_.set_ubSize(params_.ubSize);
    tilingData_.set_batch(params_.numBatchs);
    tilingData_.set_kStride(params_.kStride);
    tilingData_.set_vStride(params_.vStride);
    tilingData_.set_kOffset(params_.kOffset);
    tilingData_.set_vOffset(params_.vOffset);

    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
void ScatterPaKvCacheMembaseTiling::DumpTilingInfo()
{
    std::ostringstream info;
    info << "totalCoreNum: " << params_.usedCoreNum << std::endl;
    info << "usedCoreNum: " << params_.usedCoreNum << std::endl;
    info << "blockFactor: " << params_.blockFactor << std::endl;
    info << "tailBlockFactor: " << params_.tailBlockFactor << std::endl;
    info << "numTokens: " << params_.numTokens << std::endl;
    info << "numHead: " << params_.numHead  << std::endl;
    info << "kHeadSize: " << params_.kHeadSize << std::endl;
    info << "vHeadSize: " << params_.vHeadSize << std::endl;
    info << "blockSize: " << params_.blockSize << std::endl;
    info << "typeByte: " << params_.typeByteSlot << std::endl;
    info << "kStride: " << params_.kStride << std::endl;
    info << "vStride: " << params_.vStride << std::endl;
    info << "kOffset: " << params_.kOffset << std::endl;
    info << "vOffset: " << params_.vOffset << std::endl;
    info << "tilingKey: " << params_.tilingKey << std::endl;
}
} // namespace optiling