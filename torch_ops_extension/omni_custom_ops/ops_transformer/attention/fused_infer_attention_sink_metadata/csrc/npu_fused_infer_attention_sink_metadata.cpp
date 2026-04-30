/* *
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <torch/library.h>
#include "../../../../csrc_base/ops_common.h"

namespace custom {
const int SIZE = 1024;
using namespace at_npu::native;

// 为NPU设备实现前向接口
at::Tensor npu_fused_infer_attention_sink_metadata_npu(
    int64_t num_heads_q, int64_t num_heads_kv, int64_t heads_dim_qk, int64_t heads_dim_v,
    const c10::optional<at::Tensor> &actual_seq_lengths,
    const c10::optional<at::Tensor> &actual_seq_lengths_kv,
    int64_t batch_size, int64_t sparse_mode, int64_t pre_tokens, int64_t next_tokens,
    c10::string_view input_layout, c10::string_view input_layout_kv, int64_t sink_num, int64_t k_sink_num,
    int64_t rope_head_dim, int64_t block_size, int64_t aic_core_num, int64_t aiv_core_num)
{
    // convert str
    std::string input_layout_str = std::string(input_layout);
    std::string input_layout_kv_str = std::string(input_layout_kv);

    // construct the output tensor
    std::vector<c10::SymInt> sym_shape = { c10::SymInt(SIZE) };
    auto options = at::TensorOptions().dtype(at::ScalarType::Int).device(at::kPrivateUse1);

    if (actual_seq_lengths.has_value() && actual_seq_lengths.value().defined()) {
        options = options.device(actual_seq_lengths.value().device());
    }
    at::Tensor meta_data = at::zeros_symint(sym_shape, options);

    char *input_layout_ptr = const_cast<char *>(input_layout_str.c_str());
    char *input_layout_kv_ptr = const_cast<char *>(input_layout_kv_str.c_str());
    EXEC_NPU_CMD_V1(aclnnAiInfraFusedInferAttentionSinkMetadata, actual_seq_lengths, actual_seq_lengths_kv, num_heads_q,
            num_heads_kv, heads_dim_qk, heads_dim_v, batch_size, sparse_mode, pre_tokens, next_tokens, input_layout_ptr,
            input_layout_kv_ptr, sink_num, k_sink_num, rope_head_dim, block_size, aic_core_num, aiv_core_num, meta_data);

    return meta_data;
}

// 为META设备实现前向接口
at::Tensor npu_fused_infer_attention_sink_metadata_meta(
    int64_t num_heads_q, int64_t num_heads_kv, int64_t heads_dim_qk, int64_t heads_dim_v,
    const c10::optional<at::Tensor> &actual_seq_lengths,
    const c10::optional<at::Tensor> &actual_seq_lengths_kv,
    int64_t batch_size, int64_t sparse_mode, int64_t pre_tokens, int64_t next_tokens,
    c10::string_view input_layout, c10::string_view input_layout_kv, int64_t sink_num, int64_t k_sink_num,
    int64_t rope_head_dim, int64_t block_size, int64_t aic_core_num, int64_t aiv_core_num)
{
    // construct the output tensor
    std::vector<c10::SymInt> sym_shape = { c10::SymInt(SIZE) };
    auto options = at::TensorOptions().dtype(at::ScalarType::Int).device(at::kPrivateUse1);

    if (actual_seq_lengths.has_value() && actual_seq_lengths.value().defined()) {
        options = options.device(actual_seq_lengths.value().device());
    }
    at::Tensor meta_data = at::zeros_symint(sym_shape, options);

    return meta_data;
}

}

// 为NPU设备注册前向实现
TORCH_LIBRARY_IMPL(custom, PrivateUse1, m) {
    m.impl("_npu_fused_infer_attention_sink_metadata", &custom::npu_fused_infer_attention_sink_metadata_npu);
}

// step5, 为META设备注册前向实现
TORCH_LIBRARY_IMPL(custom, Meta, m) {
    m.impl("_npu_fused_infer_attention_sink_metadata", &custom::npu_fused_infer_attention_sink_metadata_meta);
}

