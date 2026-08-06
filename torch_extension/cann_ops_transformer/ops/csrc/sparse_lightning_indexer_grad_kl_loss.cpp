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
 * \file sparse_flash_attention_grad.cpp
 * \brief PTA wrapper of aclnnSparseFlashAttentionGrad (SFAG, MLA OSS Sink).
 */

#include <cctype>
#include <string>
#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {
constexpr int64_t SLI_grad_kl_loss_METADATA_SIZE = 64;

const int DIMENSION_1D = 1;
const int DIMENSION_3D = 3;
const int DIMENSION_4D = 4;

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> SparseLightningIndexerGradKlLoss(
    const at::Tensor &query,
    const at::Tensor &key,
    const at::Tensor &query_index,
    const at::Tensor &key_index,
    const at::Tensor &weights,
    const at::Tensor &sparse_indices,
    const at::Tensor &softmax_max,
    const at::Tensor &softmax_sum,
    double scale_value,
    const c10::optional<at::Tensor> &query_rope,
    const c10::optional<at::Tensor> &key_rope,
    c10::OptionalArrayRef<c10::SymInt> actual_seq_qlen,
    c10::OptionalArrayRef<c10::SymInt> actual_seq_klen,
    c10::optional<c10::string_view> layout,
    c10::optional<int64_t> sparse_mode,
    c10::optional<int64_t> pre_tokens,
    c10::optional<int64_t> next_tokens,
    const c10::optional<at::Tensor> &sinks)
{
    const at::Tensor &query_rope_const = query_rope.value_or(at::Tensor());
    const at::Tensor &key_rope_const = key_rope.value_or(at::Tensor());
    auto actual_seq_qlen_const = actual_seq_qlen.has_value() ? c10::asIntArrayRefUnchecked(actual_seq_qlen.value()) : at::IntArrayRef{};
    auto actual_seq_klen_const = actual_seq_klen.has_value() ? c10::asIntArrayRefUnchecked(actual_seq_klen.value()) : at::IntArrayRef{};
    const at::Tensor &sinks_const = sinks.value_or(at::Tensor());
    c10::string_view layout_str = layout.value_or("BSND");
    char *layout_ptr = const_cast<char *>(layout_str.data());
    int64_t sparse_mode_const = sparse_mode.value_or(3);
    int64_t pre_tokens_const = pre_tokens.value_or(9223372036854775807);
    int64_t next_tokens_const = next_tokens.value_or(9223372036854775807);
    bool deterministic_const = true;
    TORCH_CHECK(query.dim() == DIMENSION_3D || query.dim() == DIMENSION_4D,
        "The shapes of the input query should be 3 or 4 dimensional, but got ",
        query.dim(), "-dimensional", OPS_ERROR(ErrCode::PARAM));
    if (query_rope_const.defined()) {
        TORCH_CHECK(query_rope_const.dim() == DIMENSION_3D || query_rope_const.dim() == DIMENSION_4D,
            "The shapes of the input query_rope should be 3 or 4 dimensional, but got ",
            query_rope_const.dim(), "-dimensional", OPS_ERROR(ErrCode::PARAM));
    }
    TORCH_CHECK(key.dim() == DIMENSION_3D || key.dim() == DIMENSION_4D,
        "The shapes of the input key should be 3 or 4 dimensional, but got ", key.dim(),
        "-dimensional", OPS_ERROR(ErrCode::PARAM));
    if (key_rope_const.defined()) {
        TORCH_CHECK(key_rope_const.dim() == DIMENSION_3D || key_rope_const.dim() == DIMENSION_4D,
            "The shapes of the input key_rope should be 3 or 4 dimensional, but got ",
            key_rope_const.dim(), "-dimensional", OPS_ERROR(ErrCode::PARAM));
    }

    at::Tensor d_query_index = at::empty_like(query_index);
    at::Tensor d_key_index = at::empty_like(key_index);
    at::Tensor d_weights = at::empty_like(weights);
    at::Tensor loss = at::empty_like({1});

    if (sinks_const.defined()) {
        TORCH_CHECK(sinks_const.dim() == DIMENSION_1D,
            "The shapes of the input sinks should be 1 dimensional, but got ",
            sinks_const.dim(), "-dimensional", OPS_ERROR(ErrCode::PARAM));
        ACLNN_CMD(
            aclnnSparseLightningIndexerGradKLLossV2, query, key, query_index, key_index, weights,
            sparse_indices, softmax_max, softmax_sum, query_rope_const, key_rope_const, actual_seq_qlen_const,
            actual_seq_klen_const, sinks_const, scale_value, layout_ptr, sparse_mode_const, pre_tokens_const, next_tokens_const, deterministic_const,
            d_query_index, d_key_index, d_weights, loss);            
    } else {
        ACLNN_CMD(
            aclnnSparseLightningIndexerGradKLLoss, query, key, query_index, key_index, weights,
            sparse_indices, softmax_max, softmax_sum, query_rope_const, key_rope_const, actual_seq_qlen_const,
            actual_seq_klen_const, scale_value, layout_ptr, sparse_mode_const, pre_tokens_const, next_tokens_const, deterministic_const,
            d_query_index, d_key_index, d_weights, loss);
    }


    return std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>(d_query_index, d_key_index, d_weights, loss);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("sparse_lightning_indexer_grad_kl_loss", &SparseLightningIndexerGradKlLoss,
          "sparse_lightning_indexer_grad_kl_loss");
}

} // namespace op_api
