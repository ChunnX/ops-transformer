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
 * \file aclnn_roi_extractor.cpp
 * \brief aclnnRoiExtractor L2: unpack feats list (<=5 levels) into the
 * feats0..feats4 optional-input slots, validate, launch, view-copy to y.
 */
#include "aclnn_roi_extractor.h"
#include "roi_extractor.h"
#include "aclnn_kernels/contiguous.h"
#include "external/aclnn_kernels/aclnn_platform.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/shape_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;

namespace roi_extractor {
constexpr uint64_t MAX_LEVELS = 5;
constexpr int64_t MAX_POOLED = 56;
constexpr int64_t MAX_SAMPLE_NUM = 16;

static const std::initializer_list<op::DataType> ROI_EXTRACTOR_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT16, DataType::DT_FLOAT};

static inline bool CheckNotNull(const aclTensorList *feats, const aclTensor *rois, const aclTensor *index,
                                const aclFloatArray *spatialScales, char *poolMode, const aclTensor *y)
{
    OP_CHECK_NULL(feats, return false);
    for (uint64_t i = 0; i < feats->Size(); i++) {
        OP_CHECK_NULL((*feats)[i], return false);
    }
    OP_CHECK_NULL(rois, return false);
    OP_CHECK_NULL(spatialScales, return false);
    OP_CHECK_NULL(poolMode, return false);
    OP_CHECK_NULL(y, return false);
    return true;
}

static inline bool CheckParams(const aclTensorList *feats, const aclTensor *rois, const aclTensor *index,
                               int64_t finestScale, double roiScaleFactor, const aclFloatArray *spatialScales,
                               int64_t pooledHeight, int64_t pooledWidth, int64_t samplingRatio, char *poolMode,
                               bool aligned, const aclTensor *y)
{
    OP_CHECK(feats->Size() >= 1 && feats->Size() <= MAX_LEVELS,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Size of feats must be in [1, %llu], but got %llu.", MAX_LEVELS,
                     feats->Size()),
             return false);
    for (uint64_t i = 0; i < feats->Size(); i++) {
        const aclTensor *feat = (*feats)[i];
        OP_CHECK(feat->GetViewShape().GetDimNum() == 4,
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID, "feats[%llu] must be 4D NCHW, but got %ldD.", i,
                         feat->GetViewShape().GetDimNum()),
                 return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(feat, ROI_EXTRACTOR_DTYPE_SUPPORT_LIST, return false);
        OP_CHECK(feat->GetViewShape().GetDim(0) == (*feats)[0]->GetViewShape().GetDim(0) &&
                     feat->GetViewShape().GetDim(1) == (*feats)[0]->GetViewShape().GetDim(1),
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                         "feats[%llu] batch/channel dims must equal feats[0], but got (%ld, %ld) vs (%ld, %ld).", i,
                         feat->GetViewShape().GetDim(0), feat->GetViewShape().GetDim(1),
                         (*feats)[0]->GetViewShape().GetDim(0), (*feats)[0]->GetViewShape().GetDim(1)),
                 return false);
        OP_CHECK_DTYPE_NOT_SAME(feat, (*feats)[0], return false);
    }
    OP_CHECK(rois->GetViewShape().GetDimNum() == 2,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "rois must be 2D, but got %ldD.", rois->GetViewShape().GetDimNum()),
             return false);
    const int64_t roisLastDim = rois->GetViewShape().GetDim(1);
    OP_CHECK(roisLastDim == 4 || roisLastDim == 5,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "rois last dim must be 4 or 5, but got %ld.", roisLastDim),
             return false);
    OP_CHECK_DTYPE_NOT_SAME(rois, (*feats)[0], return false);
    if (roisLastDim == 4) {
        OP_CHECK(index != nullptr,
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID, "index must be provided when rois last dim is 4."), return false);
    }
    if (index != nullptr) {
        OP_CHECK(index->GetViewShape().GetDimNum() == 1 &&
                     index->GetViewShape().GetDim(0) == rois->GetViewShape().GetDim(0),
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID, "index must be 1D with length %ld.", rois->GetViewShape().GetDim(0)),
                 return false);
        OP_CHECK(index->GetDataType() == DataType::DT_INT32,
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID, "index dtype must be INT32."), return false);
    }
    uint64_t spatialScalesSize = 0;
    auto scalesSizeRet = aclGetFloatArraySize(spatialScales, &spatialScalesSize);
    OP_CHECK(scalesSizeRet == ACLNN_SUCCESS && spatialScalesSize == feats->Size(),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Size of spatialScales must equal feats level count %llu, but got %llu.",
                     feats->Size(), spatialScalesSize),
             return false);
    OP_CHECK(pooledHeight >= 1 && pooledHeight <= MAX_POOLED,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "pooledHeight must be in [1, %ld], but got %ld.", MAX_POOLED,
                     pooledHeight),
             return false);
    OP_CHECK(pooledWidth >= 1 && pooledWidth <= MAX_POOLED,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "pooledWidth must be in [1, %ld], but got %ld.", MAX_POOLED, pooledWidth),
             return false);
    OP_CHECK(samplingRatio >= 0 && samplingRatio <= MAX_SAMPLE_NUM,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "samplingRatio must be in [0, %ld], but got %ld.", MAX_SAMPLE_NUM,
                     samplingRatio),
             return false);
    std::string poolModeStr = std::string(poolMode);
    OP_CHECK(poolModeStr == "avg" || poolModeStr == "max",
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "poolMode only supports avg or max, but got %s.", poolModeStr.c_str()),
             return false);
    OP_CHECK(finestScale >= 1, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "finestScale must be >= 1, but got %ld.", finestScale),
             return false);
    OP_CHECK(roiScaleFactor >= 0.0 && roiScaleFactor <= 16.0,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "roiScaleFactor must be in [0, 16], but got %f.", roiScaleFactor),
             return false);
    OP_CHECK_DTYPE_NOT_SAME(y, (*feats)[0], return false);
    return true;
}
} // namespace roi_extractor

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnRoiExtractorGetWorkspaceSize(
    const aclTensorList *feats, const aclTensor *rois, const aclTensor *index, int64_t finestScale,
    double roiScaleFactor, const aclFloatArray *spatialScales, int64_t pooledHeight, int64_t pooledWidth,
    int64_t samplingRatio, char *poolMode, bool aligned, aclTensor *y, uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    L2_DFX_PHASE_1(aclnnRoiExtractor,
                   DFX_IN(feats, rois, index, finestScale, roiScaleFactor, spatialScales, pooledHeight, pooledWidth,
                          samplingRatio, poolMode, aligned),
                   DFX_OUT(y));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    if (!roi_extractor::CheckNotNull(feats, rois, index, spatialScales, poolMode, y)) {
        return ACLNN_ERR_INNER_NULLPTR;
    }
    auto ret = roi_extractor::CheckParams(feats, rois, index, finestScale, roiScaleFactor, spatialScales, pooledHeight,
                                          pooledWidth, samplingRatio, poolMode, aligned, y);
    CHECK_RET(ret, ACLNN_ERR_PARAM_INVALID);

    if ((*feats)[0]->IsEmpty() || rois->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // unpack the level list into the feats0..feats4 slots (nullptr padding)
    const aclTensor *featsSlots[roi_extractor::MAX_LEVELS] = {nullptr, nullptr, nullptr, nullptr, nullptr};
    for (uint64_t i = 0; i < feats->Size() && i < roi_extractor::MAX_LEVELS; i++) {
        featsSlots[i] = l0op::Contiguous((*feats)[i], uniqueExecutor.get());
        CHECK_RET(featsSlots[i] != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    auto roisContiguous = l0op::Contiguous(rois, uniqueExecutor.get());
    CHECK_RET(roisContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const aclTensor *indexContiguous = nullptr;
    if (index != nullptr) {
        indexContiguous = l0op::Contiguous(index, uniqueExecutor.get());
        CHECK_RET(indexContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    auto out = l0op::RoiExtractor(featsSlots[0], featsSlots[1], featsSlots[2], featsSlots[3], featsSlots[4],
                                  roisContiguous, indexContiguous, finestScale, roiScaleFactor, spatialScales,
                                  pooledHeight, pooledWidth, samplingRatio, poolMode, aligned, uniqueExecutor.get());
    CHECK_RET(out != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyResult = l0op::ViewCopy(out, y, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnRoiExtractor(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                              const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnRoiExtractor);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
