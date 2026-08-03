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
 * \file msa_index_score.cpp
 * \brief torch_extension 适配：aclnnMsaIndexScore C++ wrapper。
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {
namespace {
constexpr int64_t QUERY_DIM = 3;
constexpr int64_t KEY_DIM = 4;
constexpr int64_t BLOCK_TABLE_DIM = 2;
constexpr int64_t SEQ_LEN_Q_DIM = 1;
constexpr int64_t SEQ_LEN_K_DIM = 1;
constexpr int64_t START_LOC_DIM = 1;
constexpr int64_t SCALE_DIM = 3;
constexpr int64_t ATTEN_MASK_DIM = 2;
constexpr int64_t SCORE_STRIDE_ALIGN = 16;
constexpr int64_t SUPPORTED_BLOCK_SIZE = 128;
constexpr int64_t HEAD_DIM_ALIGN = 16;
constexpr int64_t MAX_HEAD_DIM = 128;
constexpr int64_t ATTEN_MASK_SIZE = 2048;
constexpr int64_t SPARSE_MODE_DEFAULT = 0;
constexpr int64_t SPARSE_MODE_RIGHT_DOWN = 3;
} // namespace

at::Tensor msa_index_score(const at::Tensor &query, const at::Tensor &key,
                           const c10::optional<at::Tensor> &block_table,
                           const c10::optional<at::Tensor> &scale,
                           const c10::optional<at::Tensor> &atten_mask,
                           const c10::optional<at::Tensor> &actual_seq_lengths_query,
                           const c10::optional<at::Tensor> &actual_seq_lengths_key,
                           const at::Tensor &start_loc, int64_t sparse_mode)
{
    TORCH_CHECK(query.defined() && query.numel() > 0, "Tensor query is empty.");
    TORCH_CHECK(key.defined() && key.numel() > 0, "Tensor key is empty.");
    TORCH_CHECK(start_loc.defined() && start_loc.numel() > 0, "Tensor start_loc is empty.");
    TORCH_CHECK(block_table.has_value() && block_table.value().defined() && block_table.value().numel() > 0,
                "PageAttention requires block_table.");
    TORCH_CHECK(actual_seq_lengths_query.has_value() && actual_seq_lengths_query.value().defined(),
                "TND query requires actual_seq_lengths_query.");
    TORCH_CHECK(actual_seq_lengths_key.has_value() && actual_seq_lengths_key.value().defined(),
                "PageAttention requires actual_seq_lengths_key.");

    const at::Tensor &bt = block_table.value();
    const at::Tensor &cu_q = actual_seq_lengths_query.value();
    const at::Tensor &kv_len = actual_seq_lengths_key.value();

    TORCH_CHECK(query.dim() == QUERY_DIM, "query must be 3D [T, Hq, D], but got dim=", query.dim());
    TORCH_CHECK(key.dim() == KEY_DIM, "P0 key must be 4D BBND [NP, P, Hkv, D], but got dim=", key.dim());
    TORCH_CHECK(bt.dim() == BLOCK_TABLE_DIM, "block_table must be 2D [B, MB], but got dim=", bt.dim());
    TORCH_CHECK(cu_q.dim() == SEQ_LEN_Q_DIM, "actual_seq_lengths_query must be 1D [B+1], but got dim=",
                cu_q.dim());
    TORCH_CHECK(kv_len.dim() == SEQ_LEN_K_DIM, "actual_seq_lengths_key must be 1D [B], but got dim=",
                kv_len.dim());
    TORCH_CHECK(start_loc.dim() == START_LOC_DIM, "start_loc must be 1D [B], but got dim=", start_loc.dim());

    TORCH_CHECK(sparse_mode == SPARSE_MODE_DEFAULT || sparse_mode == SPARSE_MODE_RIGHT_DOWN,
                "sparse_mode must be 0 or 3, but got ", sparse_mode);
    if (sparse_mode == SPARSE_MODE_RIGHT_DOWN) {
        TORCH_CHECK(atten_mask.has_value() && atten_mask.value().defined(),
                    "sparse_mode=3 requires atten_mask of shape [2048, 2048].");
        TORCH_CHECK(atten_mask.value().dim() == ATTEN_MASK_DIM &&
                        atten_mask.value().size(0) == ATTEN_MASK_SIZE &&
                        atten_mask.value().size(1) == ATTEN_MASK_SIZE,
                    "atten_mask must be [2048, 2048].");
        TORCH_CHECK(atten_mask.value().scalar_type() == at::kChar, "atten_mask dtype must be int8.");
    } else {
        TORCH_CHECK(!atten_mask.has_value() || !atten_mask.value().defined(),
                    "sparse_mode=0 must not pass atten_mask.");
    }

    TORCH_CHECK(query.scalar_type() == at::kHalf || query.scalar_type() == at::kBFloat16,
                "query dtype must be float16 or bfloat16, but got ", query.scalar_type());
    const bool is_quant = (key.scalar_type() == at::kChar);
    if (is_quant) {
        TORCH_CHECK(query.scalar_type() == at::kHalf, "int8 key currently requires float16 query.");
        TORCH_CHECK(scale.has_value() && scale.value().defined() && scale.value().numel() > 0,
                    "int8 key requires scale of shape [NP, N_kv, block_size].");
        TORCH_CHECK(scale.value().dim() == SCALE_DIM, "scale must be 3D [NP, N_kv, P].");
        TORCH_CHECK(scale.value().scalar_type() == at::kFloat, "scale dtype must be float32.");
    } else {
        TORCH_CHECK(key.scalar_type() == query.scalar_type(), "non-quant key dtype must match query.");
        TORCH_CHECK(!scale.has_value() || !scale.value().defined(), "non-quant path must not pass scale.");
    }

    TORCH_CHECK(bt.scalar_type() == at::kInt, "block_table dtype must be int32.");
    TORCH_CHECK(cu_q.scalar_type() == at::kInt, "actual_seq_lengths_query dtype must be int32.");
    TORCH_CHECK(kv_len.scalar_type() == at::kInt, "actual_seq_lengths_key dtype must be int32.");
    TORCH_CHECK(start_loc.scalar_type() == at::kInt, "start_loc dtype must be int32.");

    TORCH_CHECK(key.size(1) == SUPPORTED_BLOCK_SIZE, "block_size currently only supports ",
                SUPPORTED_BLOCK_SIZE);
    TORCH_CHECK(key.size(2) == 1, "num_kv_heads currently only supports 1.");
    TORCH_CHECK(query.size(2) == key.size(3), "query/key head_dim mismatch.");
    TORCH_CHECK(query.size(2) > 0 && (query.size(2) % HEAD_DIM_ALIGN) == 0, "head_dim align error.");
    TORCH_CHECK(query.size(2) <= MAX_HEAD_DIM, "head_dim too large.");
    TORCH_CHECK(cu_q.size(0) == kv_len.size(0) + 1, "cu_seq_q size must be batch+1.");
    TORCH_CHECK(start_loc.size(0) == kv_len.size(0), "start_loc size must equal batch.");
    TORCH_CHECK(bt.size(0) == kv_len.size(0), "block_table batch must equal batch.");

    at::Tensor score{nullptr};
    {
        auto local_device = c10::Device(query.device());
        const c10::OptionalDeviceGuard device_guard(local_device);
        const int64_t total_q = query.size(0);
        const int64_t num_q_heads = query.size(1);
        const int64_t max_blocks = bt.size(1);
        const int64_t score_stride =
            ((max_blocks + SCORE_STRIDE_ALIGN - 1) / SCORE_STRIDE_ALIGN) * SCORE_STRIDE_ALIGN;
        score = at::empty({num_q_heads, total_q, score_stride}, query.options().dtype(at::kFloat));
    }

    ACLNN_CMD(aclnnMsaIndexScore, query, key, block_table, scale, atten_mask, actual_seq_lengths_query,
              actual_seq_lengths_key, start_loc, sparse_mode, score);
    return score;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("msa_index_score", &msa_index_score, "msa_index_score");
}
} // namespace op_api
