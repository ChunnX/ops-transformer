/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file stem_oam_prep_paged_kv_proto.h
 * \brief StemOamPrepPagedKV OpProto definition for graph mode
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_STEM_OAM_PREP_PAGED_KV_PROTO_H_
#define OPS_BUILT_IN_OP_PROTO_INC_STEM_OAM_PREP_PAGED_KV_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {

REG_OP(StemOamPrepPagedKV)
    .INPUT(KCache, TensorType({DT_FLOAT8_E4M3FN}))
    .INPUT(VCache, TensorType({DT_FLOAT8_E4M3FN}))
    .INPUT(KvIndices, TensorType::IndexNumberType())
    .INPUT(KvSeqLens, TensorType::IndexNumberType())
    .INPUT(KScaleCache, TensorType({DT_FLOAT}))
    .INPUT(VScale, TensorType({DT_FLOAT}))
    .ATTR(LambdaMag, Float, 0.3)
    .ATTR(CacheLayout, Int, 0)
    .ATTR(KvBlockSize, Int, 64)
    .ATTR(StemBlockSize, Int, 128)
    .ATTR(StemStride, Int, 16)
    .OUTPUT(KFlat, TensorType({DT_BFLOAT16}))
    .OUTPUT(VBias, TensorType({DT_FLOAT}))
    .END();

} // namespace ge

#endif // OPS_BUILT_IN_OP_PROTO_INC_STEM_OAM_PREP_PAGED_KV_PROTO_H_