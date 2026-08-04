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
constexpr int64_t KEY_CACHE_DIM = 4;
constexpr int64_t BLOCK_TABLE_DIM = 2;
constexpr int64_t SEQ_LEN_Q_DIM = 1;
constexpr int64_t SEQ_LEN_K_DIM = 1;
constexpr int64_t START_LOC_DIM = 1;
constexpr int64_t SCALE_DIM = 3;
constexpr int64_t SCORE_STRIDE_ALIGN = 16;
constexpr int64_t SUPPORTED_BLOCK_SIZE = 128;
constexpr int64_t HEAD_DIM_ALIGN = 16;
constexpr int64_t MAX_HEAD_DIM = 128;
} // namespace

at::Tensor msa_index_score(const at::Tensor &query, const at::Tensor &key_cache, const at::Tensor &block_table,
                           const at::Tensor &actual_seq_lengths_query, const at::Tensor &actual_seq_lengths_key,
                           const at::Tensor &start_loc, const c10::optional<at::Tensor> &scale, int64_t block_size,
                           bool fill_invalid_tail)
{
    TORCH_CHECK(query.defined() && query.numel() > 0, "Tensor query is empty.");
    TORCH_CHECK(key_cache.defined() && key_cache.numel() > 0, "Tensor key_cache is empty.");
    TORCH_CHECK(block_table.defined() && block_table.numel() > 0, "Tensor block_table is empty.");
    TORCH_CHECK(actual_seq_lengths_query.defined() && actual_seq_lengths_query.numel() > 0,
                "Tensor actual_seq_lengths_query is empty.");
    TORCH_CHECK(actual_seq_lengths_key.defined() && actual_seq_lengths_key.numel() > 0,
                "Tensor actual_seq_lengths_key is empty.");
    TORCH_CHECK(start_loc.defined() && start_loc.numel() > 0, "Tensor start_loc is empty.");

    TORCH_CHECK(query.dim() == QUERY_DIM, "query must be 3D [T, Hq, D], but got dim=", query.dim());
    TORCH_CHECK(key_cache.dim() == KEY_CACHE_DIM,
                "key_cache must be 4D [NP, P, Hkv, D], but got dim=", key_cache.dim());
    TORCH_CHECK(block_table.dim() == BLOCK_TABLE_DIM,
                "block_table must be 2D [B, MB], but got dim=", block_table.dim());
    TORCH_CHECK(actual_seq_lengths_query.dim() == SEQ_LEN_Q_DIM,
                "actual_seq_lengths_query must be 1D [B+1], but got dim=", actual_seq_lengths_query.dim());
    TORCH_CHECK(actual_seq_lengths_key.dim() == SEQ_LEN_K_DIM,
                "actual_seq_lengths_key must be 1D [B], but got dim=", actual_seq_lengths_key.dim());
    TORCH_CHECK(start_loc.dim() == START_LOC_DIM, "start_loc must be 1D [B], but got dim=", start_loc.dim());

    TORCH_CHECK(query.scalar_type() == at::kHalf || query.scalar_type() == at::kBFloat16,
                "query dtype must be float16 or bfloat16, but got ", query.scalar_type());
    const bool is_quant = (key_cache.scalar_type() == at::kChar);
    if (is_quant) {
        TORCH_CHECK(query.scalar_type() == at::kHalf,
                    "int8 key_cache currently requires float16 query, but got ", query.scalar_type());
        TORCH_CHECK(scale.has_value() && scale.value().defined() && scale.value().numel() > 0,
                    "int8 key_cache requires scale of shape [NP, N_kv, block_size].");
        TORCH_CHECK(scale.value().dim() == SCALE_DIM, "scale must be 3D [NP, N_kv, P], but got dim=",
                    scale.value().dim());
        TORCH_CHECK(scale.value().scalar_type() == at::kFloat, "scale dtype must be float32, but got ",
                    scale.value().scalar_type());
    } else {
        TORCH_CHECK(key_cache.scalar_type() == query.scalar_type(),
                    "non-quant key_cache dtype must match query dtype.");
        TORCH_CHECK(!scale.has_value() || !scale.value().defined(),
                    "non-quant path must not pass scale (score-side scale is not supported).");
    }

    TORCH_CHECK(block_table.scalar_type() == at::kInt, "block_table dtype must be int32, but got ",
                block_table.scalar_type());
    TORCH_CHECK(actual_seq_lengths_query.scalar_type() == at::kInt,
                "actual_seq_lengths_query dtype must be int32, but got ", actual_seq_lengths_query.scalar_type());
    TORCH_CHECK(actual_seq_lengths_key.scalar_type() == at::kInt,
                "actual_seq_lengths_key dtype must be int32, but got ", actual_seq_lengths_key.scalar_type());
    TORCH_CHECK(start_loc.scalar_type() == at::kInt, "start_loc dtype must be int32, but got ",
                start_loc.scalar_type());

    TORCH_CHECK(block_size == SUPPORTED_BLOCK_SIZE, "block_size currently only supports ", SUPPORTED_BLOCK_SIZE,
                ", but got ", block_size);
    TORCH_CHECK(key_cache.size(1) == block_size, "key_cache dim1 (block_size) must equal attr block_size (",
                block_size, "), but got ", key_cache.size(1));
    TORCH_CHECK(key_cache.size(2) == 1, "num_kv_heads currently only supports 1, but got ", key_cache.size(2));
    TORCH_CHECK(query.size(2) == key_cache.size(3), "query head_dim (", query.size(2),
                ") must equal key_cache head_dim (", key_cache.size(3), ").");
    TORCH_CHECK(query.size(2) > 0 && (query.size(2) % HEAD_DIM_ALIGN) == 0,
                "head_dim must be a positive multiple of ", HEAD_DIM_ALIGN, ", but got ", query.size(2));
    TORCH_CHECK(query.size(2) <= MAX_HEAD_DIM, "head_dim larger than ", MAX_HEAD_DIM, " is not supported, got ",
                query.size(2));
    TORCH_CHECK(actual_seq_lengths_query.size(0) == actual_seq_lengths_key.size(0) + 1,
                "actual_seq_lengths_query size must be batch+1, got ", actual_seq_lengths_query.size(0),
                " while actual_seq_lengths_key size is ", actual_seq_lengths_key.size(0));
    TORCH_CHECK(start_loc.size(0) == actual_seq_lengths_key.size(0),
                "start_loc size must equal batch (=actual_seq_lengths_key size), got ", start_loc.size(0), " vs ",
                actual_seq_lengths_key.size(0));
    TORCH_CHECK(block_table.size(0) == actual_seq_lengths_key.size(0),
                "block_table batch dim must equal batch, got ", block_table.size(0), " vs ",
                actual_seq_lengths_key.size(0));

    at::Tensor score{nullptr};
    {
        // DeviceGuard：必须在申请输出之前，作用域包住输出申请
        auto local_device = c10::Device(query.device());
        const c10::OptionalDeviceGuard device_guard(local_device);
        const int64_t total_q = query.size(0);
        const int64_t num_q_heads = query.size(1);
        const int64_t max_blocks = block_table.size(1);
        const int64_t score_stride =
            ((max_blocks + SCORE_STRIDE_ALIGN - 1) / SCORE_STRIDE_ALIGN) * SCORE_STRIDE_ALIGN;
        score = at::empty({num_q_heads, total_q, score_stride}, query.options().dtype(at::kFloat));
    }

    ACLNN_CMD(aclnnMsaIndexScore, query, key_cache, block_table, scale, actual_seq_lengths_query,
              actual_seq_lengths_key, start_loc, block_size, fill_invalid_tail, score);
    return score;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("msa_index_score", &msa_index_score, "msa_index_score");
}
} // namespace op_api
