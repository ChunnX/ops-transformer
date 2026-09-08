/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file roi_extractor_tiling.cpp
 * \brief RoiExtractor tiling: mirror the wrapper parameter checks, fill the
 *        TilingData (padded fixed-arity slots), pack the TilingKey bit and set
 *        blockDim = min(aivNum, R * ceil(C/64)) (>=1 for the R=0 empty launch).
 */

#include "roi_extractor_tiling.h"
#include "log/log.h"
#include "op_host/tiling_util.h"

#include <algorithm>
#include <string>

using namespace ge;

namespace optiling {

// Input indices (matches roi_extractor_def.cpp registration order)
constexpr uint32_t IDX_FEATURES_0 = 0;   // features_0 .. features_7: 0..7
constexpr uint32_t IDX_ROIS = 8;
constexpr uint32_t IDX_INDEX = 9;
constexpr uint32_t IDX_Y = 0;            // output

// Attr indices (SPEC §5 fixed order)
constexpr uint32_t ATTR_FINEST_SCALE = 0;
constexpr uint32_t ATTR_ROI_SCALE_FACTOR = 1;
constexpr uint32_t ATTR_SPATIAL_SCALE = 2;
constexpr uint32_t ATTR_POOLED_HEIGHT = 3;
constexpr uint32_t ATTR_POOLED_WIDTH = 4;
constexpr uint32_t ATTR_SAMPLE_NUM = 5;
constexpr uint32_t ATTR_POOL_MODE = 6;
constexpr uint32_t ATTR_ALIGNED = 7;

ge::graphStatus RoiExtractorCompileInfo::ParamCheck(gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    // ---- attrs ----
    auto finestScalePtr = attrs->GetAttrPointer<int64_t>(ATTR_FINEST_SCALE);
    OP_CHECK_NULL_WITH_CONTEXT(context, finestScalePtr);
    const int64_t finestScale = *finestScalePtr;
    OP_CHECK_IF(finestScale < 1 || finestScale > 512,
                OP_LOGE(nodeName, "finest_scale must be in [1, 512], but got %ld.", finestScale),
                return ge::GRAPH_PARAM_INVALID);

    auto pooledHPtr = attrs->GetAttrPointer<int64_t>(ATTR_POOLED_HEIGHT);
    OP_CHECK_NULL_WITH_CONTEXT(context, pooledHPtr);
    const int64_t pooledH = *pooledHPtr;
    OP_CHECK_IF(pooledH < 1 || pooledH > 1024,
                OP_LOGE(nodeName, "pooled_height must be in [1, 1024], but got %ld.", pooledH),
                return ge::GRAPH_PARAM_INVALID);

    auto pooledWPtr = attrs->GetAttrPointer<int64_t>(ATTR_POOLED_WIDTH);
    OP_CHECK_NULL_WITH_CONTEXT(context, pooledWPtr);
    const int64_t pooledW = *pooledWPtr;
    OP_CHECK_IF(pooledW < 1 || pooledW > 1024,
                OP_LOGE(nodeName, "pooled_width must be in [1, 1024], but got %ld.", pooledW),
                return ge::GRAPH_PARAM_INVALID);

    auto sampleNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_SAMPLE_NUM);
    OP_CHECK_NULL_WITH_CONTEXT(context, sampleNumPtr);
    const int64_t sampleNum = *sampleNumPtr;
    OP_CHECK_IF(sampleNum < 0 || sampleNum > 1024,
                OP_LOGE(nodeName, "sample_num must be in [0, 1024], but got %ld.", sampleNum),
                return ge::GRAPH_PARAM_INVALID);

    auto poolModePtr = attrs->GetAttrPointer<char>(ATTR_POOL_MODE);
    OP_CHECK_NULL_WITH_CONTEXT(context, poolModePtr);
    OP_CHECK_IF(std::string(poolModePtr) != "avg",
                OP_LOGE(nodeName, "pool_mode only supports \"avg\", but got %s.", poolModePtr),
                return ge::GRAPH_PARAM_INVALID);

    // roi_scale_factor is accepted but does not participate in computation (canndev parity).

    // ---- features levels: contiguous prefix features_0..features_{L-1} ----
    int64_t level = 0;
    for (uint32_t i = 0; i < ROI_EXTRACTOR_MAX_LEVEL; i++) {
        if (context->GetOptionalInputShape(IDX_FEATURES_0 + i) != nullptr) {
            OP_CHECK_IF(level != static_cast<int64_t>(i),
                        OP_LOGE(nodeName,
                                "features must be a contiguous prefix features_0..features_%ld, "
                                "but features_%u is provided while features_%ld is missing.",
                                level - 1, i, i - 1),
                        return ge::GRAPH_PARAM_INVALID);
            level = level + 1;
        }
    }
    OP_CHECK_IF(level < 1, OP_LOGE(nodeName, "at least one feature level (features_0) is required."),
                return ge::GRAPH_PARAM_INVALID);

    auto firstShape = context->GetInputShape(IDX_FEATURES_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, firstShape);
    OP_CHECK_IF(firstShape->GetStorageShape().GetDimNum() != 4,
                OP_LOGE(nodeName, "features must be 4D NCHW, but got %ldD.",
                        firstShape->GetStorageShape().GetDimNum()),
                return ge::GRAPH_PARAM_INVALID);
    const int64_t nDim = firstShape->GetStorageShape().GetDim(0);
    const int64_t cDim = firstShape->GetStorageShape().GetDim(1);
    OP_CHECK_IF(nDim < 1 || cDim < 1,
                OP_LOGE(nodeName, "features N and C must be positive, but got N=%ld, C=%ld.", nDim, cDim),
                return ge::GRAPH_PARAM_INVALID);

    auto firstDesc = context->GetInputDesc(IDX_FEATURES_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, firstDesc);
    const ge::DataType featsDType = firstDesc->GetDataType();
    OP_CHECK_IF(featsDType != ge::DT_FLOAT16 && featsDType != ge::DT_FLOAT,
                OP_LOGE(nodeName, "features dtype must be float16 or float32, but got %d.",
                        static_cast<int32_t>(featsDType)),
                return ge::GRAPH_PARAM_INVALID);

    for (int64_t i = 1; i < level; i++) {
        auto shape = context->GetInputShape(IDX_FEATURES_0 + i);
        OP_CHECK_NULL_WITH_CONTEXT(context, shape);
        OP_CHECK_IF(shape->GetStorageShape().GetDimNum() != 4,
                    OP_LOGE(nodeName, "features_%ld must be 4D NCHW.", i),
                    return ge::GRAPH_PARAM_INVALID);
        OP_CHECK_IF(shape->GetStorageShape().GetDim(0) != nDim || shape->GetStorageShape().GetDim(1) != cDim,
                    OP_LOGE(nodeName,
                            "all feature levels must share N and C, but level %ld got N=%ld, C=%ld "
                            "while level 0 got N=%ld, C=%ld.",
                            i, shape->GetStorageShape().GetDim(0), shape->GetStorageShape().GetDim(1),
                            nDim, cDim),
                    return ge::GRAPH_PARAM_INVALID);
        auto desc = context->GetInputDesc(IDX_FEATURES_0 + i);
        OP_CHECK_NULL_WITH_CONTEXT(context, desc);
        OP_CHECK_IF(desc->GetDataType() != featsDType,
                    OP_LOGE(nodeName, "all feature levels must share dtype, level %ld got %d vs %d.",
                            i, static_cast<int32_t>(desc->GetDataType()),
                            static_cast<int32_t>(featsDType)),
                    return ge::GRAPH_PARAM_INVALID);
        OP_CHECK_IF(shape->GetStorageShape().GetDim(2) < 1 || shape->GetStorageShape().GetDim(3) < 1,
                    OP_LOGE(nodeName, "features_%ld H/W must be positive.", i),
                    return ge::GRAPH_PARAM_INVALID);
    }

    // ---- rois [R, 5], dtype coupled with features ----
    auto roisShape = context->GetInputShape(IDX_ROIS);
    OP_CHECK_NULL_WITH_CONTEXT(context, roisShape);
    OP_CHECK_IF(roisShape->GetStorageShape().GetDimNum() != 2 || roisShape->GetStorageShape().GetDim(1) != 5,
                OP_LOGE(nodeName, "rois must be [R, 5], but got %ldD with last dim %ld.",
                        roisShape->GetStorageShape().GetDimNum(),
                        roisShape->GetStorageShape().GetDim(roisShape->GetStorageShape().GetDimNum() - 1)),
                return ge::GRAPH_PARAM_INVALID);
    const int64_t rDim = roisShape->GetStorageShape().GetDim(0);
    OP_CHECK_IF(rDim < 0, OP_LOGE(nodeName, "rois R must be non-negative, but got %ld.", rDim),
                return ge::GRAPH_PARAM_INVALID);
    auto roisDesc = context->GetInputDesc(IDX_ROIS);
    OP_CHECK_NULL_WITH_CONTEXT(context, roisDesc);
    OP_CHECK_IF(roisDesc->GetDataType() != featsDType,
                OP_LOGE(nodeName, "rois dtype must equal features dtype, but got %d vs %d.",
                        static_cast<int32_t>(roisDesc->GetDataType()), static_cast<int32_t>(featsDType)),
                return ge::GRAPH_PARAM_INVALID);

    // ---- optional index int32 [R] ----
    auto indexShape = context->GetOptionalInputShape(IDX_INDEX);
    if (indexShape != nullptr) {
        auto indexDesc = context->GetOptionalInputDesc(IDX_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context, indexDesc);
        OP_CHECK_IF(indexDesc->GetDataType() != ge::DT_INT32,
                    OP_LOGE(nodeName, "index dtype must be int32, but got %d.",
                            static_cast<int32_t>(indexDesc->GetDataType())),
                    return ge::GRAPH_PARAM_INVALID);
        OP_CHECK_IF(indexShape->GetStorageShape().GetDimNum() != 1 ||
                        indexShape->GetStorageShape().GetDim(0) != rDim,
                    OP_LOGE(nodeName, "index must be 1D with length R=%ld.", rDim),
                    return ge::GRAPH_PARAM_INVALID);
    }

    // ---- y [R, C, pooled_h, pooled_w], dtype coupled ----
    auto yShape = context->GetOutputShape(IDX_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    OP_CHECK_IF(yShape->GetStorageShape().GetDimNum() != 4 ||
                    yShape->GetStorageShape().GetDim(0) != rDim ||
                    yShape->GetStorageShape().GetDim(1) != cDim ||
                    yShape->GetStorageShape().GetDim(2) != pooledH ||
                    yShape->GetStorageShape().GetDim(3) != pooledW,
                OP_LOGE(nodeName,
                        "y must be [R, C, pooled_height, pooled_width] = [%ld, %ld, %ld, %ld].",
                        rDim, cDim, pooledH, pooledW),
                return ge::GRAPH_PARAM_INVALID);
    auto yDesc = context->GetOutputDesc(IDX_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, yDesc);
    OP_CHECK_IF(yDesc->GetDataType() != featsDType,
                OP_LOGE(nodeName, "y dtype must equal features dtype, but got %d vs %d.",
                        static_cast<int32_t>(yDesc->GetDataType()), static_cast<int32_t>(featsDType)),
                return ge::GRAPH_PARAM_INVALID);

    // ---- spatial_scale list length >= L ----
    auto scaleList = attrs->GetListFloat(ATTR_SPATIAL_SCALE);
    OP_CHECK_NULL_WITH_CONTEXT(context, scaleList);
    OP_CHECK_IF(static_cast<int64_t>(scaleList->GetSize()) < level,
                OP_LOGE(nodeName, "spatial_scale length must be >= level count %ld, but got %zu.",
                        level, scaleList->GetSize()),
                return ge::GRAPH_PARAM_INVALID);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus RoiExtractorTilingFunc(gert::TilingContext *context)
{
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OP_LOGE(context->GetNodeName(), "GetPlatformInfo is nullptr."),
                return ge::GRAPH_PARAM_INVALID);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(aivNum == 0,
                OP_LOGE(context->GetNodeName(), "num of AIV core obtained is 0."),
                return ge::GRAPH_PARAM_INVALID);

    OP_CHECK_IF(context->GetWorkspaceSizes(1) == nullptr,
                OP_LOGE(context->GetNodeName(), "workSpaceSize got from ge is nullptr."),
                return ge::GRAPH_PARAM_INVALID);

    auto ret = RoiExtractorCompileInfo::ParamCheck(context);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "RoiExtractor param check failed."),
                return ge::GRAPH_PARAM_INVALID);

    auto attrs = context->GetAttrs();
    const int64_t finestScale = *attrs->GetAttrPointer<int64_t>(ATTR_FINEST_SCALE);
    const int64_t pooledH = *attrs->GetAttrPointer<int64_t>(ATTR_POOLED_HEIGHT);
    const int64_t pooledW = *attrs->GetAttrPointer<int64_t>(ATTR_POOLED_WIDTH);
    const int64_t sampleNum = *attrs->GetAttrPointer<int64_t>(ATTR_SAMPLE_NUM);
    const bool aligned = *attrs->GetAttrPointer<bool>(ATTR_ALIGNED);
    auto scaleList = attrs->GetListFloat(ATTR_SPATIAL_SCALE);
    const float *scaleData = scaleList->GetData();

    // ---- shapes / dtype ----
    int64_t level = 0;
    for (uint32_t i = 0; i < ROI_EXTRACTOR_MAX_LEVEL; i++) {
        if (context->GetOptionalInputShape(IDX_FEATURES_0 + i) != nullptr) {
            level = level + 1;
        }
    }
    auto firstShape = context->GetInputShape(IDX_FEATURES_0);
    const int64_t nDim = firstShape->GetStorageShape().GetDim(0);
    const int64_t cDim = firstShape->GetStorageShape().GetDim(1);
    auto roisShape = context->GetInputShape(IDX_ROIS);
    const int64_t rDim = roisShape->GetStorageShape().GetDim(0);
    auto featsDesc = context->GetInputDesc(IDX_FEATURES_0);
    const bool isFp16 = featsDesc->GetDataType() == ge::DT_FLOAT16;
    const bool hasIndex = context->GetOptionalInputShape(IDX_INDEX) != nullptr;

    // ---- fill tiling data ----
    // RoiExtractorTiling comes from the pypto kernel (.py dataclass) via force-include.
    RoiExtractorTiling *tilingData = context->GetTilingData<RoiExtractorTiling>();
    tilingData->r = rDim;
    tilingData->n = nDim;
    tilingData->c = cDim;
    tilingData->l = level;
    tilingData->pooled_h = pooledH;
    tilingData->pooled_w = pooledW;
    tilingData->sample_num = sampleNum;
    tilingData->finest_scale = finestScale;
    tilingData->aligned = aligned ? 1 : 0;
    tilingData->is_fp16 = isFp16 ? 1 : 0;
    for (int64_t i = 0; i < ROI_EXTRACTOR_MAX_LEVEL; i++) {
        if (i < level) {
            auto shape = context->GetInputShape(IDX_FEATURES_0 + i);
            tilingData->feat_h[i] = shape->GetStorageShape().GetDim(2);
            tilingData->feat_w[i] = shape->GetStorageShape().GetDim(3);
        } else {
            // padded fixed-arity slots: absent levels carry h=w=1 (never dereferenced)
            tilingData->feat_h[i] = 1;
            tilingData->feat_w[i] = 1;
        }
        tilingData->spatial_scale[i] = (i < level) ? scaleData[i] : 0.0f;
    }
    for (int64_t i = 0; i <= ROI_EXTRACTOR_MAX_LEVEL; i++) {
        // B[i] = 2^i * finest_scale; fp16-exact fp32 values (finest_scale <= 512)
        tilingData->b_pow2[i] = static_cast<float>(static_cast<int64_t>(1) << i) * static_cast<float>(finestScale);
    }

    // ---- TilingKey: bit0 = HasIndex (values=[0,1] -> packed value == bit OR) ----
    context->SetTilingKey(static_cast<uint64_t>(hasIndex ? 1 : 0));

    // ---- block dim: min(aivNum, R * ceil(C / 64)); >=1 (R=0 empty launch) ----
    const int64_t nCb = (cDim + ROI_EXTRACTOR_CH_BLOCK - 1) / ROI_EXTRACTOR_CH_BLOCK;
    const int64_t totalTasks = rDim * nCb;
    uint32_t blockDim = static_cast<uint32_t>(totalTasks > 0 ? std::min(static_cast<int64_t>(aivNum), totalTasks) : 1);
    context->SetBlockDim(blockDim);

    // ---- workspace: system reserve only (no semantic workspace, DESIGN §5) ----
    size_t *workSpaces = context->GetWorkspaceSizes(1);
    workSpaces[0] = ascendcPlatform.GetLibApiWorkSpaceSize();

    OP_LOGI(context->GetNodeName(),
            "RoiExtractor tiling completed: blockDim=%u, aivNum=%u, L=%ld, R=%ld, N=%ld, C=%ld, "
            "pooled=%ldx%ld, sampleNum=%ld, finestScale=%ld, aligned=%d, isFp16=%d, hasIndex=%d.",
            blockDim, aivNum, level, rDim, nDim, cDim, pooledH, pooledW, sampleNum, finestScale,
            aligned ? 1 : 0, isFp16 ? 1 : 0, hasIndex ? 1 : 0);
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
