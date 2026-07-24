/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file compressor_grad_proto.h
 * \brief
 */
#ifndef COMPRESSOR_GRAD_PROTO_H
#define COMPRESSOR_GRAD_PROTO_H

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

    aclnnStatus aclnnCompressorGradGetWorkspaceSize(
    const aclTensor *x, const aclTensor *wkv, const aclTensor *wgate,
    const aclTensor *dCmpKv, const aclTensor *softmaxScore, const aclTensor *kv,
    const aclTensor *cu_seq_lens, const aclTensor *seq_used,
    const aclTensor *startPosOptional, const char *layoutX, const char *layoutDCmpKv,
    const char *layoutSoftmaxScore, const char *layoutKv,
    const char *layoutDX, int64_t coff, int64_t cmpRatio,
    const aclTensor *d_x, const aclTensor *d_wkv, const aclTensor *d_wgate,
    const aclTensor *d_ape, uint64_t *workspaceSize, aclOpExecutor **executor)

REG_OP(CompressprGrad)
    .INPUT(x, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(wkv, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(wgate, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(dCmpKv, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(softmaxScore, TensorType({DT_FLOAT}))
    .INPUT(kv, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(cuSeqLensOptional, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(cu_seq_lens_k, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(seq_used, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(start_pos, TensorType({DT_INT32}))
    .OUTPUT(d_x, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(d_wkv, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(d_wgate, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(d_ape, TensorType({DT_FLOAT}))
    .ATTR(layoutX, String, "BSH")
    .ATTR(layoutDCmpKv, String, "BSD")
    .ATTR(layoutSoftmaxScore, String, "BSCD")
    .ATTR(layoutKv, String, "BSCD")
    .ATTR(layoutDX, String, "BSH")
    .ATTR(coff, Int, 2)
    .ATTR(cmpRatio, Int, 2)
    .OP_END_FACTORY_REG(CompressorGrad)

} // namespace ge

#endif // COMPRESSOR_GRAD_PROTO_H
