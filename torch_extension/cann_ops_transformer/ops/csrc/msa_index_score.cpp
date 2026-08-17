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
 *        Atlas A2/A3；key 支持 PA BBND/BNBD 与 TND packed。不支持 Ascend 950 / FP8。
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {
namespace {
constexpr int64_t QUERY_DIM = 3;
constexpr int64_t KEY_TND_DIM = 3;
constexpr int64_t KEY_PA_DIM = 4;
constexpr int64_t BLOCK_TABLE_DIM = 2;
constexpr int64_t SEQ_LEN_Q_DIM = 1;
constexpr int64_t SEQ_LEN_K_DIM = 1;
constexpr int64_t START_LOC_DIM = 1;
constexpr int64_t SCALE_PA_DIM = 3;
constexpr int64_t SCALE_TND_DIM = 2;
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
                           const c10::optional<at::Tensor> &actual_seq_qlen,
                           const c10::optional<at::Tensor> &actual_seq_klen,
                           const at::Tensor &start_loc, int64_t sparse_mode,
                           int64_t init_blocks, int64_t local_blocks)
{
    TORCH_CHECK(query.defined() && query.numel() > 0, "Tensor query is empty.");
    TORCH_CHECK(key.defined() && key.numel() > 0, "Tensor key is empty.");
    TORCH_CHECK(start_loc.defined() && start_loc.numel() > 0, "Tensor start_loc is empty.");
    TORCH_CHECK(actual_seq_qlen.has_value() && actual_seq_qlen.value().defined(),
                "TND query requires actual_seq_qlen.");
    TORCH_CHECK(actual_seq_klen.has_value() && actual_seq_klen.value().defined(),
                "actual_seq_klen is required.");

    const at::Tensor &cu_q = actual_seq_qlen.value();
    const at::Tensor &kv_len = actual_seq_klen.value();
    const bool is_tnd = (key.dim() == KEY_TND_DIM);
    TORCH_CHECK(query.dim() == QUERY_DIM, "query must be 3D [T, Hq, D], but got dim=", query.dim());
    TORCH_CHECK(is_tnd || key.dim() == KEY_PA_DIM,
                "key must be TND [T2,N2,D] or PA 4D BBND/BNBD, but got dim=", key.dim());
    if (is_tnd) {
        TORCH_CHECK(!block_table.has_value() || !block_table.value().defined(),
                    "TND key must not pass block_table.");
    } else {
        TORCH_CHECK(block_table.has_value() && block_table.value().defined() &&
                        block_table.value().numel() > 0,
                    "PageAttention requires block_table.");
    }
    TORCH_CHECK(cu_q.dim() == SEQ_LEN_Q_DIM, "actual_seq_qlen must be 1D [B+1], but got dim=",
                cu_q.dim());
    TORCH_CHECK(kv_len.dim() == SEQ_LEN_K_DIM, "actual_seq_klen must be 1D, but got dim=", kv_len.dim());
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
                    "int8 key requires dequant scale.");
        TORCH_CHECK(scale.value().scalar_type() == at::kFloat, "scale dtype must be float32.");
        if (is_tnd) {
            TORCH_CHECK(scale.value().dim() == SCALE_TND_DIM ||
                            (scale.value().dim() == 1 && key.size(1) == 1),
                        "TND scale must be [T2, N2] or [T2].");
        } else {
            TORCH_CHECK(scale.value().dim() == SCALE_PA_DIM, "PA scale must be 3D [NP, N_kv, P].");
        }
    } else {
        TORCH_CHECK(key.scalar_type() == query.scalar_type(), "non-quant key dtype must match query.");
        TORCH_CHECK(!scale.has_value() || !scale.value().defined(), "non-quant path must not pass scale.");
    }

    TORCH_CHECK(cu_q.scalar_type() == at::kInt, "actual_seq_qlen dtype must be int32.");
    TORCH_CHECK(kv_len.scalar_type() == at::kInt, "actual_seq_klen dtype must be int32.");
    TORCH_CHECK(start_loc.scalar_type() == at::kInt, "start_loc dtype must be int32.");

    int64_t num_kv_heads = 1;
    int64_t key_head_dim = 0;
    int64_t max_blocks = 0;
    if (is_tnd) {
        num_kv_heads = key.size(1);
        key_head_dim = key.size(2);
        TORCH_CHECK(cu_q.size(0) == kv_len.size(0), "TND actual_seq_qlen/klen must both be [B+1].");
        TORCH_CHECK(start_loc.size(0) + 1 == kv_len.size(0), "start_loc size must equal batch.");
        TORCH_CHECK(init_blocks >= 0 && local_blocks >= 0, "init_blocks/local_blocks must be >= 0.");
        auto kv_host = kv_len.cpu();
        const int32_t *pref = kv_host.data_ptr<int32_t>();
        const int64_t npref = kv_host.size(0);
        for (int64_t i = 0; i + 1 < npref; ++i) {
            const int64_t kv = static_cast<int64_t>(pref[i + 1]) - static_cast<int64_t>(pref[i]);
            const int64_t blocks = (kv <= 0) ? 0 : ((kv + SUPPORTED_BLOCK_SIZE - 1) / SUPPORTED_BLOCK_SIZE);
            if (blocks > max_blocks) {
                max_blocks = blocks;
            }
        }
        TORCH_CHECK(max_blocks > 0, "TND maxBlocks must be positive.");
        TORCH_CHECK(init_blocks <= max_blocks && local_blocks <= max_blocks,
                    "init_blocks/local_blocks must be <= maxBlocks.");
    } else {
        const at::Tensor &bt = block_table.value();
        TORCH_CHECK(bt.dim() == BLOCK_TABLE_DIM, "block_table must be 2D [B, MB], but got dim=", bt.dim());
        TORCH_CHECK(bt.scalar_type() == at::kInt, "block_table dtype must be int32.");
        const bool is_bbnd = (key.size(1) == SUPPORTED_BLOCK_SIZE);
        const bool is_bnbd = (key.size(2) == SUPPORTED_BLOCK_SIZE);
        TORCH_CHECK(is_bbnd || is_bnbd, "PA key must be BBND[NP,P,N2,D] or BNBD[NP,N2,P,D] with P=",
                    SUPPORTED_BLOCK_SIZE);
        num_kv_heads = is_bbnd ? key.size(2) : key.size(1);
        key_head_dim = key.size(3);
        TORCH_CHECK(cu_q.size(0) == kv_len.size(0) + 1, "actual_seq_qlen size must be batch+1.");
        TORCH_CHECK(start_loc.size(0) == kv_len.size(0), "start_loc size must equal batch.");
        TORCH_CHECK(bt.size(0) == kv_len.size(0), "block_table batch must equal batch.");
        TORCH_CHECK(init_blocks >= 0 && local_blocks >= 0, "init_blocks/local_blocks must be >= 0.");
        TORCH_CHECK(init_blocks <= bt.size(1) && local_blocks <= bt.size(1),
                    "init_blocks/local_blocks must be <= maxBlocks.");
        max_blocks = bt.size(1);
    }
    TORCH_CHECK(num_kv_heads == 1, "num_kv_heads currently only supports 1.");
    TORCH_CHECK(query.size(2) == key_head_dim, "query/key head_dim mismatch.");
    TORCH_CHECK(query.size(2) > 0 && (query.size(2) % HEAD_DIM_ALIGN) == 0, "head_dim align error.");
    TORCH_CHECK(query.size(2) <= MAX_HEAD_DIM, "head_dim too large.");

    at::Tensor score{nullptr};
    {
        auto local_device = c10::Device(query.device());
        const c10::OptionalDeviceGuard device_guard(local_device);
        const int64_t total_q = query.size(0);
        const int64_t num_q_heads = query.size(1);
        const int64_t score_stride =
            ((max_blocks + SCORE_STRIDE_ALIGN - 1) / SCORE_STRIDE_ALIGN) * SCORE_STRIDE_ALIGN;
        score = at::empty({num_q_heads, total_q, score_stride}, query.options().dtype(at::kFloat));
    }

    ACLNN_CMD(aclnnMsaIndexScore, query, key, block_table, scale, atten_mask, actual_seq_qlen,
              actual_seq_klen, start_loc, sparse_mode, init_blocks, local_blocks, score);
    return score;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("msa_index_score", &msa_index_score, "msa_index_score");
}
} // namespace op_api
