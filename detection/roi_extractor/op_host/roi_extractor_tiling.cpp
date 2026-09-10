/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You should not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the LICENSE file in the root of the software repository for the full text of the License.
 */

/*!
 * \file roi_extractor_tiling.cpp
 * \brief Tiling: collect FPN level slots, validate shapes/dtypes/attrs, fill
 * the RoiExtractorTiling fields (codegen force-include) and set TilingKey
 * bit0 = (pool_mode == "max") - matching RoiExtractorTilingKey on the kernel
 * side (docs/MDE.md section 2.2).
 */
#include "roi_extractor_tiling.h"
#include "log/log.h"

using namespace ge;

namespace optiling {
namespace {
constexpr int64_t MAX_LEVELS = 5;
constexpr int64_t MAX_POOLED = 56;
constexpr int64_t MAX_SAMPLE_NUM = 16;
constexpr double MAX_ROI_SCALE_FACTOR = 16.0;
// Attr declaration order: finest_scale(0), roi_scale_factor(1), spatial_scale(2),
// pooled_height(3), pooled_width(4), sample_num(5), pool_mode(6), aligned(7).
constexpr uint32_t ATTR_FINEST_SCALE = 0;
constexpr uint32_t ATTR_ROI_SCALE_FACTOR = 1;
constexpr uint32_t ATTR_SPATIAL_SCALE = 2;
constexpr uint32_t ATTR_POOLED_HEIGHT = 3;
constexpr uint32_t ATTR_POOLED_WIDTH = 4;
constexpr uint32_t ATTR_SAMPLE_NUM = 5;
constexpr uint32_t ATTR_POOL_MODE = 6;
constexpr uint32_t ATTR_ALIGNED = 7;
// Input order: feats0..feats4 (0..4), rois (5), index (6).
constexpr uint32_t ROIS_INDEX = 5;
constexpr uint32_t INDEX_INDEX = 6;
} // namespace

ge::graphStatus RoiExtractorCompileInfo::ParamCheck(gert::TilingContext *context)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(context->GetNodeName(), "attrs is nullptr."), return ge::GRAPH_FAILED);

    const int64_t *finestScalePtr = attrs->GetAttrPointer<int64_t>(ATTR_FINEST_SCALE);
    int64_t finestScale = finestScalePtr == nullptr ? 56 : *finestScalePtr;
    OP_CHECK_IF(finestScale < 1, OP_LOGE(context->GetNodeName(), "finest_scale must be >= 1, but got %ld.", finestScale),
                return ge::GRAPH_FAILED);

    const float *roiScaleFactorPtr = attrs->GetAttrPointer<float>(ATTR_ROI_SCALE_FACTOR);
    float roiScaleFactor = roiScaleFactorPtr == nullptr ? 0.0f : *roiScaleFactorPtr;
    OP_CHECK_IF(roiScaleFactor < 0.0f || roiScaleFactor > MAX_ROI_SCALE_FACTOR,
                OP_LOGE(context->GetNodeName(), "roi_scale_factor must be in [0, %f], but got %f.", MAX_ROI_SCALE_FACTOR,
                        roiScaleFactor),
                return ge::GRAPH_FAILED);

    const int64_t *pooledHeightPtr = attrs->GetAttrPointer<int64_t>(ATTR_POOLED_HEIGHT);
    const int64_t *pooledWidthPtr = attrs->GetAttrPointer<int64_t>(ATTR_POOLED_WIDTH);
    int64_t pooledHeight = pooledHeightPtr == nullptr ? 7 : *pooledHeightPtr;
    int64_t pooledWidth = pooledWidthPtr == nullptr ? 7 : *pooledWidthPtr;
    OP_CHECK_IF(pooledHeight < 1 || pooledHeight > MAX_POOLED,
                OP_LOGE(context->GetNodeName(), "pooled_height must be in [1, %ld], but got %ld.", MAX_POOLED,
                        pooledHeight),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(pooledWidth < 1 || pooledWidth > MAX_POOLED,
                OP_LOGE(context->GetNodeName(), "pooled_width must be in [1, %ld], but got %ld.", MAX_POOLED,
                        pooledWidth),
                return ge::GRAPH_FAILED);

    const int64_t *sampleNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_SAMPLE_NUM);
    int64_t sampleNum = sampleNumPtr == nullptr ? 0 : *sampleNumPtr;
    OP_CHECK_IF(sampleNum < 0 || sampleNum > MAX_SAMPLE_NUM,
                OP_LOGE(context->GetNodeName(), "sample_num must be in [0, %ld], but got %ld.", MAX_SAMPLE_NUM,
                        sampleNum),
                return ge::GRAPH_FAILED);

    const char *poolModePtr = attrs->GetAttrPointer<char>(ATTR_POOL_MODE);
    std::string poolMode = poolModePtr == nullptr ? "avg" : std::string(poolModePtr);
    OP_CHECK_IF(poolMode != "avg" && poolMode != "max",
                OP_LOGE(context->GetNodeName(), "pool_mode only supports avg or max, but got %s.", poolMode.c_str()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus RoiExtractorTilingFunc(gert::TilingContext *context)
{
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OP_LOGE(context->GetNodeName(), "GetPlatformInfo is nullptr."),
                return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    OP_CHECK_IF(aivNum == 0 || aicNum == 0,
                OP_LOGE(context->GetNodeName(), "num of core obtained is 0, aicNum=%u, aivNum=%u.", aicNum, aivNum),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(context->GetWorkspaceSizes(1) == nullptr,
                OP_LOGE(context->GetNodeName(), "workSpaceSize got from ge is nullptr."),
                return ge::GRAPH_FAILED);

    uint32_t blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);
    context->SetBlockDim(blockDim);

    auto retCheck = RoiExtractorCompileInfo::ParamCheck(context);
    OP_CHECK_IF(retCheck != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "RoiExtractorCompileInfo::ParamCheck failed."),
                return ge::GRAPH_FAILED);
    auto attrs = context->GetAttrs();

    // ---- collect FPN level slots (contiguous prefix, 1..5) ----
    const gert::StorageShape *featsShapes[MAX_LEVELS] = {nullptr, nullptr, nullptr, nullptr, nullptr};
    const gert::CompileTimeTensorDesc *featsDescs[MAX_LEVELS] = {nullptr, nullptr, nullptr, nullptr, nullptr};
    int64_t numLevels = 0;
    bool seenEmptySlot = false;
    for (int64_t slot = 0; slot < MAX_LEVELS; slot++) {
        auto shape = context->GetOptionalInputShape(static_cast<uint32_t>(slot));
        // UT faker marks absent optional inputs as 0-dim shapes; real runtime passes nullptr.
        if (shape == nullptr || shape->GetStorageShape().GetDimNum() == 0) {
            seenEmptySlot = true;
            continue;
        }
        OP_CHECK_IF(seenEmptySlot,
                    OP_LOGE(context->GetNodeName(), "feats slots must be a contiguous prefix, feats%ld is set after an empty slot.",
                            slot),
                    return ge::GRAPH_FAILED);
        featsShapes[slot] = shape;
        featsDescs[slot] = context->GetOptionalInputDesc(static_cast<uint32_t>(slot));
        OP_CHECK_IF(featsDescs[slot] == nullptr,
                    OP_LOGE(context->GetNodeName(), "feats%ld desc is null.", slot), return ge::GRAPH_FAILED);
        numLevels = slot + 1;
    }
    OP_CHECK_IF(numLevels < 1, OP_LOGE(context->GetNodeName(), "at least one feats level must be provided."),
                return ge::GRAPH_FAILED);

    // ---- cross-level consistency: rank 4 (NCHW), N/C identical, dtype fp16/fp32 ----
    int64_t batchSize = featsShapes[0]->GetStorageShape().GetDim(0);
    int64_t channels = featsShapes[0]->GetStorageShape().GetDim(1);
    auto featsDType = featsDescs[0]->GetDataType();
    OP_CHECK_IF(featsDType != ge::DT_FLOAT16 && featsDType != ge::DT_FLOAT,
                OP_LOGE(context->GetNodeName(), "feats dtype must be float16 or float32, but got %d.",
                        static_cast<int32_t>(featsDType)),
                return ge::GRAPH_FAILED);
    for (int64_t slot = 0; slot < numLevels; slot++) {
        auto storageShape = featsShapes[slot]->GetStorageShape();
        OP_CHECK_IF(storageShape.GetDimNum() != 4,
                    OP_LOGE(context->GetNodeName(), "feats%ld must be 4D NCHW, but got %ldD.", slot,
                            storageShape.GetDimNum()),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(storageShape.GetDim(0) != batchSize || storageShape.GetDim(1) != channels,
                    OP_LOGE(context->GetNodeName(),
                            "feats%ld batch/channel dims must equal feats0 (%ld, %ld), but got (%ld, %ld).",
                            slot, batchSize, channels, storageShape.GetDim(0), storageShape.GetDim(1)),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(storageShape.GetDim(2) < 1 || storageShape.GetDim(3) < 1,
                    OP_LOGE(context->GetNodeName(), "feats%ld H/W must be positive.", slot),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(featsDescs[slot]->GetDataType() != featsDType,
                    OP_LOGE(context->GetNodeName(), "feats%ld dtype must equal feats0.", slot),
                    return ge::GRAPH_FAILED);
    }

    // ---- rois / index ----
    // GetInputShape is instance-indexed in some builds; use the IR-indexed optional lookup.
    auto roisShape = context->GetOptionalInputShape(ROIS_INDEX);
    OP_CHECK_IF(roisShape == nullptr, OP_LOGE(context->GetNodeName(), "rois shape is null."),
                return ge::GRAPH_FAILED);
    auto roisDesc = context->GetOptionalInputDesc(ROIS_INDEX);
    OP_CHECK_IF(roisDesc == nullptr, OP_LOGE(context->GetNodeName(), "rois desc is null."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(roisShape->GetStorageShape().GetDimNum() != 2,
                OP_LOGE(context->GetNodeName(), "rois must be 2D, but got %ldD.",
                        roisShape->GetStorageShape().GetDimNum()),
                return ge::GRAPH_FAILED);
    int64_t roisNum = roisShape->GetStorageShape().GetDim(0);
    int64_t roisLastDim = roisShape->GetStorageShape().GetDim(1);
    OP_CHECK_IF(roisLastDim != 4 && roisLastDim != 5,
                OP_LOGE(context->GetNodeName(), "rois last dim must be 4 or 5, but got %ld.", roisLastDim),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(roisDesc->GetDataType() != featsDType,
                OP_LOGE(context->GetNodeName(), "rois dtype must equal feats dtype."),
                return ge::GRAPH_FAILED);

    auto indexShape = context->GetOptionalInputShape(INDEX_INDEX);
    int64_t hasIndex = 0;
    if (roisLastDim == 4) {
        OP_CHECK_IF(indexShape == nullptr,
                    OP_LOGE(context->GetNodeName(), "index must be provided when rois last dim is 4."),
                    return ge::GRAPH_FAILED);
    }
    if (indexShape != nullptr) {
        auto indexDesc = context->GetOptionalInputDesc(INDEX_INDEX);
        OP_CHECK_IF(indexDesc == nullptr, OP_LOGE(context->GetNodeName(), "index desc is null."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(indexDesc->GetDataType() != ge::DT_INT32,
                    OP_LOGE(context->GetNodeName(), "index dtype must be int32, but got %d.",
                            static_cast<int32_t>(indexDesc->GetDataType())),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(indexShape->GetStorageShape().GetDimNum() != 1 ||
                    indexShape->GetStorageShape().GetDim(0) != roisNum,
                    OP_LOGE(context->GetNodeName(), "index must be 1D with length %ld (rois num).", roisNum),
                    return ge::GRAPH_FAILED);
        hasIndex = 1;
    }

    // ---- output ----
    auto outputShape = context->GetOutputShape(0);
    OP_CHECK_IF(outputShape == nullptr, OP_LOGE(context->GetNodeName(), "output shape is null."),
                return ge::GRAPH_FAILED);
    auto outputDesc = context->GetOutputDesc(0);
    OP_CHECK_IF(outputDesc == nullptr, OP_LOGE(context->GetNodeName(), "output desc is null."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(outputDesc->GetDataType() != featsDType,
                OP_LOGE(context->GetNodeName(), "output dtype must equal feats dtype."),
                return ge::GRAPH_FAILED);

    // ---- spatial_scale list length must equal L ----
    const gert::ContinuousVector *spatialScaleVec = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_SPATIAL_SCALE);
    int64_t spatialScaleLen = 0;
    const float *spatialScales = nullptr;
    if (spatialScaleVec != nullptr) {
        spatialScaleLen = static_cast<int64_t>(spatialScaleVec->GetSize());
        spatialScales = reinterpret_cast<const float *>(spatialScaleVec->GetData());
    }
    OP_CHECK_IF(spatialScaleLen != numLevels,
                OP_LOGE(context->GetNodeName(), "spatial_scale length must equal feats level count %ld, but got %ld.",
                        numLevels, spatialScaleLen),
                return ge::GRAPH_FAILED);
    for (int64_t slot = 0; slot < spatialScaleLen; slot++) {
        OP_CHECK_IF(spatialScales[slot] <= 0.0f || spatialScales[slot] > 1.0f,
                    OP_LOGE(context->GetNodeName(), "spatial_scale[%ld] must be in (0, 1], but got %f.", slot,
                            spatialScales[slot]),
                    return ge::GRAPH_FAILED);
    }

    // ---- fill tiling data (field-identical with the kernel dataclass) ----
    RoiExtractorTiling *tilingData = context->GetTilingData<RoiExtractorTiling>();
    OP_CHECK_IF(tilingData == nullptr, OP_LOGE(context->GetNodeName(), "GetTilingData failed."),
                return ge::GRAPH_FAILED);
    tilingData->numLevels = numLevels;
    tilingData->roisNum = roisNum;
    tilingData->batchSize = batchSize;
    tilingData->channels = channels;
    const int64_t *pooledHeightPtr = attrs->GetAttrPointer<int64_t>(ATTR_POOLED_HEIGHT);
    const int64_t *pooledWidthPtr = attrs->GetAttrPointer<int64_t>(ATTR_POOLED_WIDTH);
    const int64_t *sampleNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_SAMPLE_NUM);
    const int64_t *finestScalePtr = attrs->GetAttrPointer<int64_t>(ATTR_FINEST_SCALE);
    const float *roiScaleFactorPtr = attrs->GetAttrPointer<float>(ATTR_ROI_SCALE_FACTOR);
    const bool *alignedPtr = attrs->GetAttrPointer<bool>(ATTR_ALIGNED);
    tilingData->pooledHeight = pooledHeightPtr == nullptr ? 7 : *pooledHeightPtr;
    tilingData->pooledWidth = pooledWidthPtr == nullptr ? 7 : *pooledWidthPtr;
    tilingData->sampleNum = sampleNumPtr == nullptr ? 0 : *sampleNumPtr;
    tilingData->finestScale = finestScalePtr == nullptr ? 56 : *finestScalePtr;
    tilingData->roiScaleFactor = roiScaleFactorPtr == nullptr ? 0.0f : *roiScaleFactorPtr;
    tilingData->aligned = (alignedPtr == nullptr || *alignedPtr) ? 1 : 0;
    tilingData->hasIndex = hasIndex;
    tilingData->colOffset = hasIndex == 1 ? 0 : 1;

    float *scaleFields[MAX_LEVELS] = {&(tilingData->spatialScale0), &(tilingData->spatialScale1),
                                      &(tilingData->spatialScale2), &(tilingData->spatialScale3),
                                      &(tilingData->spatialScale4)};
    int64_t *hFields[MAX_LEVELS] = {&(tilingData->levelH0), &(tilingData->levelH1), &(tilingData->levelH2),
                                    &(tilingData->levelH3), &(tilingData->levelH4)};
    int64_t *wFields[MAX_LEVELS] = {&(tilingData->levelW0), &(tilingData->levelW1), &(tilingData->levelW2),
                                    &(tilingData->levelW3), &(tilingData->levelW4)};
    for (int64_t slot = 0; slot < MAX_LEVELS; slot++) {
        if (slot < numLevels) {
            *(scaleFields[slot]) = spatialScales == nullptr ? 1.0f : spatialScales[slot];
            *(hFields[slot]) = featsShapes[slot]->GetStorageShape().GetDim(2);
            *(wFields[slot]) = featsShapes[slot]->GetStorageShape().GetDim(3);
        } else {
            *(scaleFields[slot]) = 1.0f;
            *(hFields[slot]) = 1;
            *(wFields[slot]) = 1;
        }
    }

    // ---- TilingKey: bit0 = pool_mode is max (matches RoiExtractorTilingKey) ----
    const char *poolModePtr = attrs->GetAttrPointer<char>(ATTR_POOL_MODE);
    std::string poolMode = poolModePtr == nullptr ? "avg" : std::string(poolModePtr);
    uint64_t tilingKey = (poolMode == "max") ? 1ULL : 0ULL;
    context->SetTilingKey(tilingKey);

    // ---- workspace: system reserve only ----
    size_t workspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t *workSpaces = context->GetWorkspaceSizes(1);
    workSpaces[0] = workspaceSize;

    OP_LOGI(context->GetNodeName(),
            "RoiExtractor tiling completed: blockDim=%u, aivNum=%u, aicNum=%u, L=%ld, K=%ld, N=%ld, C=%ld, "
            "pooled=%ldx%ld, sampleNum=%ld, poolMode=%s, aligned=%ld, hasIndex=%ld, tilingKey=%llu, workspaceSize=%zu.",
            blockDim, aivNum, aicNum, numLevels, roisNum, batchSize, channels, tilingData->pooledHeight,
            tilingData->pooledWidth, tilingData->sampleNum, poolMode.c_str(), tilingData->aligned, hasIndex,
            tilingKey, workspaceSize);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForRoiExtractor(gert::TilingParseContext *context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(RoiExtractor)
    .Tiling(RoiExtractorTilingFunc)
    .TilingParse<RoiExtractorCompileInfo>(TilingParseForRoiExtractor);

} // namespace optiling
