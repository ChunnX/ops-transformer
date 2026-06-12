/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <torch/extension.h>
#include <string>
#include <tuple>
#include "aclnn_common.h"

namespace op_api {

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_sparse_lightning_indexer_kl_loss_grad(
    const at::Tensor &q,
    const at::Tensor &k,
    const at::Tensor &w,
    const at::Tensor &sparse_indices,
    const at::Tensor &attn_softmax_l1_norm,
    const c10::optional<at::Tensor> &cu_seqlens_q,
    const c10::optional<at::Tensor> &cu_seqlens_k,
    const c10::optional<at::Tensor> &seqused_q,
    const c10::optional<at::Tensor> &seqused_k,
    const c10::optional<at::Tensor> &cmp_residual_k,
    const c10::optional<at::Tensor> &metadata,
    std::string layout_q,
    std::string layout_k,
    int64_t mask_mode,
    int64_t cmp_ratio)
{
    at::Tensor dq = at::empty_like(q);
    at::Tensor dk = at::empty_like(k);
    at::Tensor dw = at::empty_like(w);
    at::Tensor softmax_out = at::empty_like(attn_softmax_l1_norm);

    char *layout_q_ptr = const_cast<char *>(layout_q.c_str());
    char *layout_k_ptr = const_cast<char *>(layout_k.c_str());

    ACLNN_CMD(aclnnSparseLightningIndexerKLLossGrad,
        q, k, w, sparse_indices, attn_softmax_l1_norm, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k,
        cmp_residual_k, metadata, layout_q_ptr, layout_k_ptr, mask_mode, cmp_ratio,
        dq, dk, dw, softmax_out);

    return std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>(dq, dk, dw, softmax_out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def(
        "npu_sparse_lightning_indexer_kl_loss_grad",
        &npu_sparse_lightning_indexer_kl_loss_grad,
        "npu_sparse_lightning_indexer_kl_loss_grad");
}

}  // namespace op_api
