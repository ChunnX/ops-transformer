/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the LICENSE file in the root of the software repository for the full text of the License.
 */

/*!
 * \file stem_oam_prep_paged_kv_proto.h
 * \brief stem_oam_prep_paged_kv operator prototype definition
 */

#ifndef OPS_BUILT_IN_OP_PROTO_INC_STEM_OAM_PREP_PAGED_KV_PROTO_H_
#define OPS_BUILT_IN_OP_PROTO_INC_STEM_OAM_PREP_PAGED_KV_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {

REG_OP(StemOamPrepPagedKv)
    .INPUT(kCache, TensorType({DT_FLOAT8_E4M3FN}))
    .INPUT(vCache, TensorType({DT_FLOAT8_E4M3FN}))
    .INPUT(kvIndices, TensorType({DT_INT32}))
    .INPUT(kvSeqLens, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(kScaleCache, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(vScale, TensorType({DT_FLOAT}))
    .ATTR(lambdaMag, Float, 0.3)
    .ATTR(kvLayout, String, "BNBD")
    .ATTR(stemBlockSize, Int, 128)
    .ATTR(stemStride, Int, 16)
    .OUTPUT(kFlat, TensorType({DT_BF16}))
    .OUTPUT(vBias, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(StemOamPrepPagedKv)

} // namespace ge

#endif // OPS_BUILT_IN_OP_PROTO_INC_STEM_OAM_PREP_PAGED_KV_PROTO_H_
