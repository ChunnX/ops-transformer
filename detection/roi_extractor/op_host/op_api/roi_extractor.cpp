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
 * \file roi_extractor.cpp
 * \brief Level0 RoiExtractor: allocate output and add to the AICore launch list.
 */
#include "roi_extractor.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(RoiExtractor);

const aclTensor *RoiExtractor(const aclTensor *feats0, const aclTensor *feats1, const aclTensor *feats2,
                              const aclTensor *feats3, const aclTensor *feats4, const aclTensor *rois,
                              const aclTensor *index, int64_t finestScale, double roiScaleFactor,
                              const aclFloatArray *spatialScales, int64_t pooledHeight, int64_t pooledWidth,
                              int64_t sampleNum, const char *poolMode, bool aligned, aclOpExecutor *executor)
{
    L0_DFX(RoiExtractor, feats0, feats1, feats2, feats3, feats4, rois, index, finestScale, roiScaleFactor,
           spatialScales, pooledHeight, pooledWidth, sampleNum, poolMode, aligned);

    const int64_t roisNum = rois->GetViewShape().GetDim(0);
    const int64_t channels = feats0->GetViewShape().GetDim(1);
    op::Shape outShape({roisNum, channels, pooledHeight, pooledWidth});
    auto out = executor->AllocTensor(outShape, feats0->GetDataType(), op::Format::FORMAT_ND);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(RoiExtractor,
                                           OP_INPUT(feats0, feats1, feats2, feats3, feats4, rois, index),
                                           OP_OUTPUT(out),
                                           OP_ATTR(finestScale, static_cast<float>(roiScaleFactor), spatialScales,
                                                   pooledHeight, pooledWidth, sampleNum, poolMode, aligned));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "RoiExtractor ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return nullptr;
    }
    return out;
}
} // namespace l0op
