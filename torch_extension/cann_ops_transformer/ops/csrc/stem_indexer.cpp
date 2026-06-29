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
 * \file stem_indexer.cpp
 * \brief
 */

#include <torch/extension.h>
#include <tuple>
#include "aclnn_common.h"

namespace op_api {
namespace {
constexpr int64_t DIM_ONE = 1;
constexpr int64_t DIM_THREE = 3;
constexpr int64_t DIM_FOUR = 4;
constexpr int64_t QFLAT_B_DIM = 0;
constexpr int64_t QFLAT_N1_DIM = 1;
constexpr int64_t QFLAT_QB_DIM = 2;
constexpr int64_t KFLAT_KB_DIM = 2;
constexpr int64_t METADATA_SIZE = 2048;

void CheckStemIndexerInputs(const at::Tensor &qflat, const at::Tensor &kflat, const at::Tensor &vbias,
                            const at::Tensor &q_seq_lens, const at::Tensor &kv_seq_lens,
                            const at::Tensor &num_prompt_tokens, const at::Tensor &metadata)
{
    TORCH_CHECK(qflat.dim() == DIM_FOUR,
                "qflat must be a 4D tensor with shape (B, N1, Qb, stem_stride * D), but got ", qflat.dim(), "D.");
    TORCH_CHECK(kflat.dim() == DIM_FOUR,
                "kflat must be a 4D tensor with shape (B, N2, Kb, stem_stride * D), but got ", kflat.dim(), "D.");
    TORCH_CHECK(vbias.dim() == DIM_THREE,
                "vbias must be a 3D tensor with shape (B, N2, Kb), but got ", vbias.dim(), "D.");

    TORCH_CHECK(qflat.scalar_type() == at::ScalarType::BFloat16,
                "qflat dtype must be bfloat16, but got ", qflat.scalar_type(), ".");
    TORCH_CHECK(kflat.scalar_type() == at::ScalarType::BFloat16,
                "kflat dtype must be bfloat16, but got ", kflat.scalar_type(), ".");
    TORCH_CHECK(vbias.scalar_type() == at::ScalarType::Float,
                "vbias dtype must be float32, but got ", vbias.scalar_type(), ".");

    TORCH_CHECK(kflat.size(2) == vbias.size(2),
                "The Kb dimensions of kflat and vbias must be equal.");
    TORCH_CHECK(qflat.size(3) == kflat.size(3),
                "The last dimensions of qflat and kflat must be equal.");
}
} // namespace

at::Tensor stem_indexer_metadata(const at::Tensor &q_seq_lens, const at::Tensor &kv_seq_lens,
                                     int64_t q_heads, int64_t kv_heads, bool causal, int64_t stem_block_size,
                                     int64_t dim_qkflat, int64_t window_size)
{
    at::Device output_device = at::Device(std::string("npu"));
    at::Tensor output = torch::empty({METADATA_SIZE}, torch::dtype(torch::kInt32).device(output_device));
    ACLNN_CMD(aclnnStemIndexerMetadata, q_seq_lens, kv_seq_lens, q_heads, kv_heads, causal, stem_block_size,
              dim_qkflat, window_size, output);
    return output;
}

std::tuple<at::Tensor, at::Tensor> stem_indexer(
    const at::Tensor &qflat, const at::Tensor &kflat, const at::Tensor &vbias,
    const at::Tensor &q_seq_lens, const at::Tensor &kv_seq_lens, const c10::optional<at::Tensor> &num_prompt_tokens,
    const c10::optional<at::Tensor> &metadata, bool causal, int64_t stem_block_size, int64_t stem_stride, double alpha,
    int64_t initial_blocks, int64_t window_size, double k_block_num_rate_medium, int64_t k_block_num_bias_medium,
    double k_block_num_rate_large, int64_t k_block_num_bias_large)
{   
    const at::Tensor &num_prompt_tokens_const = num_prompt_tokens.value_or(at::Tensor());
    const at::Tensor &metadata_const = metadata.value_or(at::Tensor());
    CheckStemIndexerInputs(qflat, kflat, vbias, q_seq_lens, kv_seq_lens, num_prompt_tokens_const, metadata_const);

    at::Tensor sparse_indices = at::empty(
        {qflat.size(QFLAT_B_DIM), qflat.size(QFLAT_N1_DIM), qflat.size(QFLAT_QB_DIM), kflat.size(KFLAT_KB_DIM)},
        qflat.options().dtype(at::kInt));
    at::Tensor sparse_seq_len = at::empty(
        {qflat.size(QFLAT_B_DIM), qflat.size(QFLAT_N1_DIM), qflat.size(QFLAT_QB_DIM)},
        qflat.options().dtype(at::kInt));

    ACLNN_CMD(aclnnStemIndexer, qflat, kflat, vbias, q_seq_lens, kv_seq_lens, num_prompt_tokens_const, metadata_const, causal,
              stem_block_size, stem_stride, alpha, initial_blocks, window_size, k_block_num_rate_medium,
              k_block_num_bias_medium, k_block_num_rate_large, k_block_num_bias_large, sparse_indices, sparse_seq_len);

    return std::tuple<at::Tensor, at::Tensor>(sparse_indices, sparse_seq_len);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("stem_indexer_metadata", &npu_stem_indexer_metadata, "stem_indexer_metadata");
    m.def("stem_indexer", &npu_stem_indexer, "stem_indexer");
}
} // namespace op_api
