/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <torch/library.h>
#include "ops_common.h"

namespace custom {
using namespace at_npu::native;

// npu tensor max size
const int SIZE = 8;
const int64_t DIM_THREE = 3;

// 工具函数，推导输出 attention_out / softmax_lse 的 shape 与 dtype
//   - attention_out: BF16，shape 跟随 query 布局（BSND/TND），head-dim 取自 value
//   - softmax_lse:   FLOAT32，return_softmax_lse 为真时 (B,N1,S1)/(N1,T)，否则空张量
std::tuple<at::Tensor, at::Tensor> construct_bsa_output_tensors(
    const at::Tensor& query, const at::Tensor& value, bool return_softmax_lse)
{
    for (auto i = 0; i < query.sizes().size(); i++) {
        TORCH_CHECK(query.size(i) > 0, "All values within query's shape should be greater "
            "than 0, but shape[", i, "] is ", query.size(i));
    }

    at::SmallVector<int64_t, SIZE> atten_out_size;
    at::SmallVector<int64_t, SIZE> softmax_lse_size;
    int64_t d_size = value.size(value.dim() - 1);
    if (query.dim() == DIM_THREE) {
        // TND -> (T, N1, D)
        int64_t t_size = query.size(0);
        int64_t n1_size = query.size(1);
        atten_out_size = {t_size, n1_size, d_size};
        softmax_lse_size = {n1_size, t_size};
    } else {
        // BSND -> (B, S1, N1, D)
        int64_t b_size = query.size(0);
        int64_t s1_size = query.size(1);
        int64_t n1_size = query.size(2);
        atten_out_size = {b_size, s1_size, n1_size, d_size};
        softmax_lse_size = {b_size, n1_size, s1_size};
    }
    if (!return_softmax_lse) {
        softmax_lse_size = {};
    }

    at::Tensor attention_out = at::empty(atten_out_size, query.options().dtype(at::kBFloat16));
    at::Tensor softmax_lse = at::empty(softmax_lse_size, query.options().dtype(at::kFloat));
    return std::tuple<at::Tensor, at::Tensor>(attention_out, softmax_lse);
}

// step2, 为NPU设备实现前向接口（函数形参顺序 = schema 顺序）
std::tuple<at::Tensor, at::Tensor> npu_quant_block_sparse_attention_npu(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    const at::Tensor &q_descale, const at::Tensor &k_descale, const at::Tensor &v_descale, const at::Tensor &p_scale,
    const at::Tensor &sparse_indices, const at::Tensor &sparse_seq_len, const at::Tensor &atten_mask,
    double softmax_scale, int64_t sparse_q_block_size, int64_t sparse_kv_block_size, int64_t paBlockStride,
    const c10::optional<at::Tensor> &cu_seqlens_q, const c10::optional<at::Tensor> &cu_seqlens_kv,
    const c10::optional<at::Tensor> &seqused_q, const c10::optional<at::Tensor> &seqused_kv,
    const c10::optional<at::Tensor> &block_table, const c10::optional<at::Tensor> &metadata,
    int64_t max_seqlen_q, int64_t max_seqlen_kv,
    c10::string_view layout_kv, c10::string_view layout_q, c10::string_view layout_sparse_indices,
    int64_t quant_mode, int64_t mask_mode, bool return_softmax_lse)
{
    TORCH_CHECK(query.numel() > 0, "Tensor query is empty.");
    TORCH_CHECK(paBlockStride > 0, "paBlockStride should be greater than 0, but got ", paBlockStride);
    TORCH_CHECK(key.stride(0) == paBlockStride,
        "paBlockStride should equal key.stride(0), but got paBlockStride=", paBlockStride,
        " and key.stride(0)=", key.stride(0));

    // construct the output tensors
    std::tuple<at::Tensor, at::Tensor> outputs =
        construct_bsa_output_tensors(query, value, return_softmax_lse);
    at::Tensor attention_out = std::get<0>(outputs);
    at::Tensor softmax_lse = std::get<1>(outputs);

    // convert str
    std::string layout_kv_str = std::string(layout_kv);
    std::string layout_q_str = std::string(layout_q);
    std::string layout_sparse_indices_str = std::string(layout_sparse_indices);
    char *layout_kv_ptr = const_cast<char *>(layout_kv_str.c_str());
    char *layout_q_ptr = const_cast<char *>(layout_q_str.c_str());
    char *layout_sparse_indices_ptr = const_cast<char *>(layout_sparse_indices_str.c_str());

    // EXEC_NPU_CMD_V1 实参顺序 = 算子 IR 声明顺序（输入 -> 属性 -> 输出），与 schema 形参顺序不同
    EXEC_NPU_CMD_V1(aclnnQuantBlockSparseAttention,
        query, key, value, q_descale, k_descale, v_descale, p_scale,
        cu_seqlens_q, cu_seqlens_kv, seqused_q, seqused_kv,
        sparse_indices, sparse_seq_len, block_table, atten_mask, metadata,
        max_seqlen_q, max_seqlen_kv, softmax_scale, sparse_q_block_size, sparse_kv_block_size, paBlockStride,
        layout_kv_ptr, layout_q_ptr, layout_sparse_indices_ptr, quant_mode, mask_mode, return_softmax_lse,
        attention_out, softmax_lse);

    return std::tuple<at::Tensor, at::Tensor>(attention_out, softmax_lse);
}

// step3, 为META设备实现前向接口
std::tuple<at::Tensor, at::Tensor> npu_quant_block_sparse_attention_meta(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    const at::Tensor &q_descale, const at::Tensor &k_descale, const at::Tensor &v_descale, const at::Tensor &p_scale,
    const at::Tensor &sparse_indices, const at::Tensor &sparse_seq_len, const at::Tensor &atten_mask,
    double softmax_scale, int64_t sparse_q_block_size, int64_t sparse_kv_block_size, int64_t paBlockStride,
    const c10::optional<at::Tensor> &cu_seqlens_q, const c10::optional<at::Tensor> &cu_seqlens_kv,
    const c10::optional<at::Tensor> &seqused_q, const c10::optional<at::Tensor> &seqused_kv,
    const c10::optional<at::Tensor> &block_table, const c10::optional<at::Tensor> &metadata,
    int64_t max_seqlen_q, int64_t max_seqlen_kv,
    c10::string_view layout_kv, c10::string_view layout_q, c10::string_view layout_sparse_indices,
    int64_t quant_mode, int64_t mask_mode, bool return_softmax_lse)
{
    (void)paBlockStride;
    return construct_bsa_output_tensors(query, value, return_softmax_lse);
}
}

// step4, 为NPU设备注册前向实现
TORCH_LIBRARY_IMPL(custom, PrivateUse1, m) {
    m.impl("npu_quant_block_sparse_attention", &custom::npu_quant_block_sparse_attention_npu);
}

// step5, 为META设备注册前向实现
TORCH_LIBRARY_IMPL(custom, Meta, m) {
    m.impl("npu_quant_block_sparse_attention", &custom::npu_quant_block_sparse_attention_meta);
}
