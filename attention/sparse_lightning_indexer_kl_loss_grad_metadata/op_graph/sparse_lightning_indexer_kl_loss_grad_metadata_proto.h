/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sparse_lightning_indexer_kl_loss_grad_metadata_proto.h
 * \brief
 */
#ifndef SPARSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_METADATA_PROTO_H
#define SPARSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_METADATA_PROTO_H

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

REG_OP(SparseLightningIndexerKLLossGradMetadata)
    .OPTIONAL_INPUT(cu_seqlens_q, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(cu_seqlens_k, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(seqused_q, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(seqused_k, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(cmp_residual_k, TensorType({DT_INT32}))
    .OUTPUT(metadata, TensorType({DT_INT32}))
    .REQUIRED_ATTR(b_size, Int)
    .REQUIRED_ATTR(s1_size, Int)
    .REQUIRED_ATTR(s2_size, Int)
    .REQUIRED_ATTR(k_size, Int)
    .ATTR(mask_mode, Int, 3)
    .ATTR(cmp_ratio, Int, 1)
    .ATTR(layout_q, String, "BSND")
    .ATTR(layout_k, String, "BSND")
    .REQUIRED_ATTR(aic_core_num, Int)
    .OP_END_FACTORY_REG(SparseLightningIndexerKLLossGradMetadata)

} // namespace ge

#endif // SPARSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_METADATA_PROTO_H
