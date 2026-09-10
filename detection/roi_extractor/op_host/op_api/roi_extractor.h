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
 * \file roi_extractor.h
 * \brief Level0 RoiExtractor launcher.
 */
#ifndef OP_API_INC_LEVEL0_ROI_EXTRACTOR_H_
#define OP_API_INC_LEVEL0_ROI_EXTRACTOR_H_

#include "opdev/op_executor.h"

namespace l0op {
const aclTensor *RoiExtractor(const aclTensor *feats0, const aclTensor *feats1, const aclTensor *feats2,
                              const aclTensor *feats3, const aclTensor *feats4, const aclTensor *rois,
                              const aclTensor *index, int64_t finestScale, double roiScaleFactor,
                              const aclFloatArray *spatialScales, int64_t pooledHeight, int64_t pooledWidth,
                              int64_t sampleNum, const char *poolMode, bool aligned, aclOpExecutor *executor);
}

#endif // OP_API_INC_LEVEL0_ROI_EXTRACTOR_H_
