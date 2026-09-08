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
 * \file roi_extractor_proto.h
 * \brief Graph-side prototype of RoiExtractor (mirror of op_host/roi_extractor_def.cpp).
 */

#ifndef ROI_EXTRACTOR_PROTO_H_
#define ROI_EXTRACTOR_PROTO_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

REG_OP(RoiExtractor)
    .OPTIONAL_INPUT(features_0, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(features_1, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(features_2, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(features_3, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(features_4, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(features_5, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(features_6, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(features_7, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(rois, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(index, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(finest_scale, Int, 56)
    .ATTR(roi_scale_factor, Float, 0.0)
    .ATTR(spatial_scale, ListFloat, {0.25, 0.125, 0.0625, 0.03125})
    .ATTR(pooled_height, Int, 7)
    .ATTR(pooled_width, Int, 7)
    .ATTR(sample_num, Int, 0)
    .ATTR(pool_mode, String, "avg")
    .ATTR(aligned, Bool, true)
    .OP_END_FACTORY_REG(RoiExtractor)

} // namespace ge

#endif // ROI_EXTRACTOR_PROTO_H_
