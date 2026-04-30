/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <torch/library.h>
#include "../../../../csrc_base/ops_common.h"

using torch::autograd::Function;
using torch::autograd::AutogradContext;
using variable_list = std::vector<at::Tensor>;

namespace custom {
using namespace at_npu::native;

// npu tensor max size
const int SIZE = 8;
const static int64_t DIM_0 = 0;
const static int64_t DIM_1 = 1;
const static int64_t DIM_2 = 2;
const static int64_t DIM_3 = 3;
const static int64_t DIM_4 = 4;

// 工具函数，推导输出shape
    at::Tensor construct_sparse_flash_attention_output_tensor(
        const at::Tensor& query, const at::Tensor& value)
    {
        std::vector<c10::SymInt> output_size;
        if (query.dim() == DIM_3) {
            output_size = {query.size(DIM_0), query.size(DIM_1), query.size(DIM_2)};
        } else {
            output_size = {query.size(DIM_0), query.size(DIM_1), query.size(DIM_2), query.size(DIM_3)};
        }

        at::Tensor output = at::empty_symint(output_size, query.options().dtype(query.dtype()));

        return output;
    }


// step2, 为NPU设备实现前向接口
    std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_ai_infra_sparse_flash_attention_pioneer_npu(
        const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
        const at::Tensor &sparse_indices, double scale_value,
        const c10::optional<at::Tensor> &block_table,
        const c10::optional<at::Tensor> &actual_seq_lengths_query,
        const c10::optional<at::Tensor> &actual_seq_lengths_kv,
        const c10::optional<at::Tensor> &query_rope,
        const c10::optional<at::Tensor> &key_rope, 
        const c10::optional<at::Tensor> &key_sink,
        const c10::optional<at::Tensor> &value_sink,
        int64_t sparse_block_size,
        c10::string_view layout_query, c10::string_view layout_kv,
        int64_t sparse_mode,
        int64_t pre_tokens, int64_t next_tokens,
        int64_t attention_mode, bool return_softmax_lse)
    {
        const at::Tensor &query_rope_const = query_rope.value_or(at::Tensor());
        const at::Tensor &key_rope_const = key_rope.value_or(at::Tensor());
        const at::Tensor &ac_seq_qlen = actual_seq_lengths_query.value_or(at::Tensor());
        const at::Tensor &ac_seq_kvlen = actual_seq_lengths_kv.value_or(at::Tensor());
        const at::Tensor &key_sink_const = key_sink.value_or(at::Tensor());
        const at::Tensor &value_sink_const = value_sink.value_or(at::Tensor());
        const at::Tensor &block_table_const = block_table.value_or(at::Tensor());

        std::string layout_query_str = std::string(layout_query);
        std::string layout_kv_str = std::string(layout_kv);

        // construct the output tensor
        at::Tensor sparse_flash_attention_output = custom::construct_sparse_flash_attention_output_tensor(
            query, value);
        at::Tensor softmax_max;
        at::Tensor softmax_sum;
        std::vector<c10::SymInt> softmax_max_size;
        std::vector<c10::SymInt> softmax_sum_size;
        if (query.dim() == DIM_3) {
            softmax_max_size = {key.size(1), query.size(0), query.size(1) / key.size(1)};
            softmax_sum_size = {key.size(1), query.size(0), query.size(1) / key.size(1)};
        } else {
            softmax_max_size = {query.size(0), key.size(2), query.size(1), query.size(2) / key.size(2)};
            softmax_sum_size = {query.size(0), key.size(2), query.size(1), query.size(2) / key.size(2)};
        }
        softmax_max = at::empty_symint(softmax_max_size, query.options().dtype(at::kFloat));
        softmax_sum = at::empty_symint(softmax_sum_size, query.options().dtype(at::kFloat));
        // convert str
        char *layout_query_ptr = const_cast<char *>(layout_query_str.c_str());
        char *layout_kv_ptr = const_cast<char *>(layout_kv_str.c_str());

        EXEC_NPU_CMD_V1(aclnnAiInfraSparseFlashAttentionPioneer, query,
            key, value, sparse_indices, block_table_const, ac_seq_qlen, ac_seq_kvlen, 
            query_rope_const, key_rope_const, key_sink_const, value_sink_const,
            scale_value, sparse_block_size, layout_query_ptr, layout_kv_ptr, 
            sparse_mode, pre_tokens, next_tokens, attention_mode, return_softmax_lse,
            sparse_flash_attention_output, softmax_max, softmax_sum);
        return std::tuple<at::Tensor, at::Tensor, at::Tensor>(sparse_flash_attention_output, softmax_max, softmax_sum);
    }


    std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_ai_infra_sparse_flash_attention_pioneer_meta(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    const at::Tensor &sparse_indices, double scale_value,
    const c10::optional<at::Tensor> &block_table,
    const c10::optional<at::Tensor> &actual_seq_lengths_query,
    const c10::optional<at::Tensor> &actual_seq_lengths_kv,
    const c10::optional<at::Tensor> &query_rope,
    const c10::optional<at::Tensor> &key_rope, 
    const c10::optional<at::Tensor> &key_sink,
    const c10::optional<at::Tensor> &value_sink,
    int64_t sparse_block_size,
    c10::string_view layout_query, c10::string_view layout_kv,
    int64_t sparse_mode,
    int64_t pre_tokens, int64_t next_tokens,
    int64_t attention_mode, bool return_softmax_lse)
    {
        std::string layout_query_str = std::string(layout_query);
        at::Tensor output = construct_sparse_flash_attention_output_tensor(
        query, value);

        at::Tensor softmax_max;
        at::Tensor softmax_sum;
        std::vector<c10::SymInt> softmax_max_size;
        std::vector<c10::SymInt> softmax_sum_size;
        if (query.dim() == DIM_3) {
            softmax_max_size = {key.size(1), query.size(0), query.size(1) / key.size(1)};
            softmax_sum_size = {key.size(1), query.size(0), query.size(1) / key.size(1)};
        } else {
            softmax_max_size = {query.size(0), key.size(2), query.size(1), query.size(2) / key.size(2)};
            softmax_sum_size = {query.size(0), key.size(2), query.size(1), query.size(2) / key.size(2)};
        }
        softmax_max = at::empty_symint(softmax_max_size, query.options().dtype(at::kFloat));
        softmax_sum = at::empty_symint(softmax_sum_size, query.options().dtype(at::kFloat));

        if (!return_softmax_lse) {
                softmax_max = at::empty_symint({0}, query.options().dtype(at::kFloat));
                softmax_sum = at::empty_symint({0}, query.options().dtype(at::kFloat));
        }


        return std::tuple<at::Tensor, at::Tensor, at::Tensor>(output, softmax_max, softmax_sum);
    }

}


//为NPU设备注册前向实现
TORCH_LIBRARY_IMPL(custom, PrivateUse1, m) {
    m.impl("npu_ai_infra_sparse_flash_attention_pioneer", &custom::npu_ai_infra_sparse_flash_attention_pioneer_npu);
}


TORCH_LIBRARY_IMPL(custom, Meta, m) {
    m.impl("npu_ai_infra_sparse_flash_attention_pioneer", &custom::npu_ai_infra_sparse_flash_attention_pioneer_meta);
}
