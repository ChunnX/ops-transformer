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
const static int FLASH_THRESHOLD = 512;
const static int64_t PFA_SPARSE_HIGH_PRECISION_NO_MASK = 10;
const static int64_t PFA_SPARSE_HIGH_PRECISION_BAND = 14;
const static int64_t DIM_0 = 0;
const static int64_t DIM_1 = 1;
const static int64_t DIM_2 = 2;
const static int64_t DIM_3 = 3;
const static int64_t DIM_4 = 4;
const static int64_t PA_BBH_DIMS = 3;
const static int64_t PA_BNBD_DIMS = 4;
const static int64_t PA_NZ_DIMS = 5;
const int SIZE = 8;
using namespace at_npu::native;

// 工具函数，推导输出shape
std::tuple<at::Tensor, at::Tensor> construct_fia_output_tensor_v2(
    const at::Tensor &query,
    const at::Tensor &value,
    std::string input_layout_str,
    const c10::optional<at::Tensor> &quant_scale_out,
    const c10::optional<at::Tensor> &block_table,
    int64_t num_query_heads,
    int64_t num_key_value_heads, // 增加kvhead用于计算BBH情况下的D
    bool return_softmax_lse,
    const c10::optional<at::Tensor> &query_rope)
{
    at::Tensor output;
    c10::SymInt batchSize = 1;
    c10::SymInt qsSize = 1;
    at::Tensor tmp_output;
    if (query.dim() == DIM_3) {
        tmp_output = at::empty_symint({query.sym_size(0), query.sym_size(1), query.sym_size(2)}, query.options().dtype(query.dtype()));
    } else {
        tmp_output = at::empty_symint({query.sym_size(0), query.sym_size(1), query.sym_size(2), query.sym_size(3)}, query.options().dtype(query.dtype()));
    }
    if (input_layout_str == "BNSD_BSND") {
        tmp_output = at::empty_symint({query.sym_size(DIM_0), query.sym_size(DIM_2), query.sym_size(DIM_1), query.sym_size(DIM_3)},
            query.options().dtype(query.dtype()));
        batchSize = query.sym_size(DIM_0);
        qsSize = query.sym_size(DIM_2);
    } else if (input_layout_str == "BNSD_NBSD") {
        tmp_output = at::empty_symint(
            {query.sym_size(DIM_1), query.sym_size(DIM_0), query.sym_size(DIM_2), query.sym_size(DIM_3)},
            query.options().dtype(query.dtype()));
        batchSize = query.sym_size(DIM_0);
        qsSize = query.sym_size(DIM_2);
    } else if (input_layout_str == "BSND_NBSD") {
        tmp_output = at::empty_symint(
            {query.sym_size(DIM_2), query.sym_size(DIM_0), query.sym_size(DIM_1), query.sym_size(DIM_3)},
            query.options().dtype(query.dtype()));
        batchSize = query.sym_size(DIM_0);
        qsSize = query.sym_size(DIM_1);
    } else if (input_layout_str == "BSH_NBSD") {
        TORCH_CHECK(num_query_heads > 0, "The num_query_heads should be greater than zero, but got ", num_query_heads);
        tmp_output = at::empty_symint(
            {num_query_heads, query.sym_size(DIM_0), query.sym_size(DIM_1), query.sym_size(DIM_2) / num_query_heads},
            query.options().dtype(query.dtype()));
        batchSize = query.sym_size(DIM_0);
        qsSize = query.sym_size(DIM_1);
    } else if (input_layout_str == "TND_NTD") {
        tmp_output = at::empty_symint(
            {query.sym_size(DIM_1), query.sym_size(DIM_0), query.sym_size(DIM_2)},
            query.options().dtype(query.dtype()));
    } else if (input_layout_str == "NSD") {
        batchSize = 1;
        qsSize = query.sym_size(DIM_1);
    } else if (input_layout_str == "BSH") {
        batchSize = query.sym_size(DIM_0);
        qsSize = query.sym_size(DIM_1);
    } else if (input_layout_str == "BSND") {
        batchSize = query.sym_size(DIM_0);
        qsSize = query.sym_size(DIM_1);
    } else if (input_layout_str == "BNSD") {
        batchSize = query.sym_size(DIM_0);
        qsSize = query.sym_size(DIM_2);
    } else if (input_layout_str == "TND") {
        int64_t kv_dim = value.dim();
        if (block_table.has_value()) { // IFA目前TND只支持PA场景，PFA目前TND只支持非PA场景
            if (kv_dim == PA_BBH_DIMS) { // BBH的情况下，D = H / N
                tmp_output = at::empty_symint(
                    {query.sym_size(DIM_0), query.sym_size(DIM_1), value.sym_size(DIM_2) / num_key_value_heads},
                    query.options().dtype(query.dtype()));
            } else if (kv_dim == PA_BNBD_DIMS) { // BNBD情况下取D
                tmp_output = at::empty_symint(
                    {query.sym_size(DIM_0), query.sym_size(DIM_1), value.sym_size(DIM_3)},
                    query.options().dtype(query.dtype()));
            } else if (kv_dim == PA_NZ_DIMS) { // blockNum, N, D / 16, blockSize, 16取DIM2*DIM4
                tmp_output = at::empty_symint(
                    {query.sym_size(DIM_0), query.sym_size(DIM_1), value.sym_size(DIM_2) * value.sym_size(DIM_4)},
                    query.options().dtype(query.dtype()));
            } else {
                tmp_output = at::empty_symint(
                    {query.sym_size(DIM_0), query.sym_size(DIM_1), value.sym_size(DIM_2)},
                    query.options().dtype(query.dtype()));
            }
        } else {
            tmp_output = at::empty_symint(
                {query.sym_size(DIM_0), query.sym_size(DIM_1), value.sym_size(DIM_2)},
                query.options().dtype(query.dtype()));
        }
    } else if (input_layout_str == "NTD_TND") {
        int64_t kv_dim = value.dim();
        if (kv_dim == 0) {
            kv_dim = query.dim();
        }
        if (block_table.has_value()) { // pa场景
            if (kv_dim == PA_BBH_DIMS) { // BBH的情况下，D = H / N
                tmp_output = at::empty_symint(
                    {query.sym_size(DIM_1), query.sym_size(DIM_0), value.sym_size(DIM_2) / num_key_value_heads},
                    query.options().dtype(query.dtype()));
            } else if (kv_dim == PA_BNBD_DIMS) { // BNBD情况下取D
                tmp_output = at::empty_symint(
                    {query.sym_size(DIM_1), query.sym_size(DIM_0), value.sym_size(DIM_3)},
                    query.options().dtype(query.dtype()));
            } else if (kv_dim == PA_NZ_DIMS) { // blockNum, N, D / 16, blockSize, 16取DIM2*DIM4
                tmp_output = at::empty_symint(
                    {query.sym_size(DIM_1), query.sym_size(DIM_0), value.sym_size(DIM_2) * value.sym_size(DIM_4)},
                    query.options().dtype(query.dtype()));
            } else {
                tmp_output = at::empty_symint(
                    {query.sym_size(DIM_1), query.sym_size(DIM_0), value.sym_size(DIM_2)},
                    query.options().dtype(query.dtype()));
            }
        } else {
            tmp_output = at::empty_symint(
                {query.sym_size(DIM_1), query.sym_size(DIM_0), value.sym_size(DIM_2)},
                query.options().dtype(query.dtype()));
        }
    }
    if (quant_scale_out.has_value()) {
        if (tmp_output.dim() == DIM_3) {
            output = at::empty_symint({tmp_output.sym_size(DIM_0), tmp_output.sym_size(DIM_1), tmp_output.sym_size(DIM_2)},
                query.options().dtype(c10::ScalarType::Char));
        } else {
            output = at::empty_symint( {tmp_output.sym_size(DIM_0), tmp_output.sym_size(DIM_1), tmp_output.sym_size(DIM_2), tmp_output.sym_size(DIM_3)},
                query.options().dtype(c10::ScalarType::Char));
        } 
    } else if (query.dtype() == at::kChar) {
        if (query_rope.has_value()) {
            const at::Tensor &query_rope_tensor = c10::value_or_else(query_rope, [] { return at::Tensor(); });
            if (tmp_output.dim() == DIM_3) {
                output = at::empty_symint({tmp_output.sym_size(DIM_0), tmp_output.sym_size(DIM_1), tmp_output.sym_size(DIM_2)},
                    query.options().dtype(query_rope_tensor.dtype()));
            } else {
                output = at::empty_symint( {tmp_output.sym_size(DIM_0), tmp_output.sym_size(DIM_1), tmp_output.sym_size(DIM_2), tmp_output.sym_size(DIM_3)},
                    query.options().dtype(query_rope_tensor.dtype()));
            }
        } else {
            if (tmp_output.dim() == DIM_3) {
                output = at::empty_symint({tmp_output.sym_size(DIM_0), tmp_output.sym_size(DIM_1), tmp_output.sym_size(DIM_2)},
                    query.options().dtype(c10::ScalarType::Half));
            } else {
                output = at::empty_symint( {tmp_output.sym_size(DIM_0), tmp_output.sym_size(DIM_1), tmp_output.sym_size(DIM_2), tmp_output.sym_size(DIM_3)},
                    query.options().dtype(c10::ScalarType::Half));
            }
        }
    } else {
        if (tmp_output.dim() == DIM_3) {
            output = at::empty_symint({tmp_output.sym_size(DIM_0), tmp_output.sym_size(DIM_1), tmp_output.sym_size(DIM_2)},
                tmp_output.options().dtype(tmp_output.dtype()));
        } else {
            output = at::empty_symint( {tmp_output.sym_size(DIM_0), tmp_output.sym_size(DIM_1), tmp_output.sym_size(DIM_2), tmp_output.sym_size(DIM_3)},
                tmp_output.options().dtype(tmp_output.dtype()));
        }
    }

    at::Tensor softmax_lse;
    auto options = at::TensorOptions().dtype(c10::ScalarType::Float).device(output.device());
    if (input_layout_str == "TND" || input_layout_str == "TND_NTD") {
        if (block_table.has_value()) { // IFA目前TND只支持PA场景，PFA目前TND只支持非PA场景
            if (query.sym_size(DIM_2) == 0) { // 增加softmax lse的情况下，可能存在空tensor的分支
                softmax_lse = at::empty_symint({query.sym_size(DIM_0), num_query_heads, 0},
                    options);
            } else {
                softmax_lse = at::empty_symint({query.sym_size(DIM_0), num_query_heads, 1},
                    options);
            }
        } else {
            softmax_lse = at::empty_symint({query.sym_size(DIM_0), query.sym_size(DIM_1), 1},
                options);
        }
    } else if (input_layout_str == "NTD_TND" || input_layout_str == "NTD") {
        if (block_table.has_value()) { // pa场景
            if (query.sym_size(DIM_2) == 0) { // 增加softmax lse的情况下，可能存在空tensor的分支
                softmax_lse = at::empty_symint({query.sym_size(DIM_1), query.sym_size(DIM_0), 0},
                    options);
            } else {
                softmax_lse = at::empty_symint({query.sym_size(DIM_1), query.sym_size(DIM_0), 1},
                    options);
            }
        } else {
            softmax_lse = at::empty_symint({query.sym_size(DIM_1), query.sym_size(DIM_0), 1},
                options);
        }
    } else {
        softmax_lse = at::empty_symint({batchSize, num_query_heads, qsSize, 1},
            options);
    }

    if (!return_softmax_lse) {
        softmax_lse = at::empty_symint({0}, options);
    }
    return std::tuple<at::Tensor, at::Tensor>(output, softmax_lse);
}

// 为NPU设备实现前向接口
std::tuple<at::Tensor, at::Tensor> npu_fused_infer_attention_sink_npu(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    const c10::optional<at::Tensor> &query_rope,
    const c10::optional<at::Tensor> &key_rope,
    const c10::optional<at::Tensor> &pse_shift,
    const c10::optional<at::Tensor> &atten_mask,
    const c10::optional<at::Tensor> &actual_seq_qlen,
    const c10::optional<at::Tensor> &actual_seq_kvlen,
    const c10::optional<at::Tensor> &block_table,
    const c10::optional<at::Tensor> &dequant_scale_query,
    const c10::optional<at::Tensor> &dequant_scale_key,
    const c10::optional<at::Tensor> &dequant_offset_key,
    const c10::optional<at::Tensor> &dequant_scale_value,
    const c10::optional<at::Tensor> &dequant_offset_value,
    const c10::optional<at::Tensor> &dequant_scale_key_rope,
    const c10::optional<at::Tensor> &key_sink,
    const c10::optional<at::Tensor> &key_rope_sink,
    const c10::optional<at::Tensor> &value_sink,
    const c10::optional<at::Tensor> &quant_scale_out,
    const c10::optional<at::Tensor> &quant_offset_out,
    const c10::optional<at::Tensor> &meta_data,
    int64_t num_query_heads, int64_t num_key_value_heads, double softmax_scale,
    int64_t pre_tokens, int64_t next_tokens, c10::string_view input_layout,
    int64_t sparse_mode, int64_t block_size,
    int64_t query_quant_mode, int64_t key_quant_mode, int64_t value_quant_mode,
    int64_t inner_precise, bool return_softmax_lse, int64_t sink_number,
    c10::optional<int64_t> query_dtype, c10::optional<int64_t> key_dtype, c10::optional<int64_t> value_dtype,
    c10::optional<int64_t> query_rope_dtype, c10::optional<int64_t> key_rope_dtype,
    c10::optional<int64_t> key_shared_prefix_dtype, c10::optional<int64_t> value_shared_prefix_dtype,
    c10::optional<int64_t> dequant_scale_query_dtype, c10::optional<int64_t> dequant_scale_key_dtype,
    c10::optional<int64_t> dequant_scale_value_dtype, c10::optional<int64_t> dequant_scale_key_rope_dtype)
{
    // convert str
    std::string input_layout_str = std::string(input_layout);

    // construct the output tensor
    std::tuple<at::Tensor, at::Tensor> fia_output = custom::construct_fia_output_tensor_v2(query, value, input_layout_str,
                                                                                           quant_scale_out, block_table, num_query_heads,
                                                                                           num_key_value_heads,
                                                                                           return_softmax_lse, query_rope);
    at::Tensor output = std::get<0>(fia_output);
    at::Tensor softmax_lse = std::get<1>(fia_output);

    char *input_layout_ptr = const_cast<char *>(input_layout_str.c_str());

    at::Tensor dequant_scale1;
    at::Tensor quant_scale1;
    at::Tensor dequant_scale2;
    at::Tensor antiquant_scale;
    at::Tensor antiquant_offset;
    at::Tensor query_padding_size;
    at::Tensor kv_padding_size;
    int64_t antiquant_mode = 0;

    at::TensorList valueTensors = value;
    at::TensorList keyTensors = key;
    EXEC_NPU_CMD_V1(aclnnAiInfraFusedInferAttentionSink, query, 
            keyTensors, valueTensors, pse_shift, atten_mask, actual_seq_qlen, actual_seq_kvlen, dequant_scale1, quant_scale1, dequant_scale2,
            quant_scale_out, quant_offset_out, antiquant_scale, antiquant_offset, block_table, query_padding_size, kv_padding_size,
            dequant_scale_key, dequant_offset_key, dequant_scale_value, dequant_offset_value,
            query_rope, key_rope, dequant_scale_key_rope, dequant_scale_query, meta_data, key_sink, key_rope_sink, value_sink,
            num_query_heads, softmax_scale, pre_tokens, next_tokens, input_layout_ptr, num_key_value_heads, sparse_mode, inner_precise,
            block_size, antiquant_mode, return_softmax_lse, key_quant_mode, value_quant_mode, query_quant_mode, sink_number, output, softmax_lse);

    return std::tuple<at::Tensor, at::Tensor>(output, softmax_lse);
}

// 为META设备实现前向接口
std::tuple<at::Tensor, at::Tensor> npu_fused_infer_attention_sink_meta(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    const c10::optional<at::Tensor> &query_rope,
    const c10::optional<at::Tensor> &key_rope,
    const c10::optional<at::Tensor> &pse_shift,
    const c10::optional<at::Tensor> &atten_mask,
    const c10::optional<at::Tensor> &actual_seq_qlen,
    const c10::optional<at::Tensor> &actual_seq_kvlen,
    const c10::optional<at::Tensor> &block_table,
    const c10::optional<at::Tensor> &dequant_scale_query,
    const c10::optional<at::Tensor> &dequant_scale_key,
    const c10::optional<at::Tensor> &dequant_offset_key,
    const c10::optional<at::Tensor> &dequant_scale_value,
    const c10::optional<at::Tensor> &dequant_offset_value,
    const c10::optional<at::Tensor> &dequant_scale_key_rope,
    const c10::optional<at::Tensor> &key_sink,
    const c10::optional<at::Tensor> &key_rope_sink,
    const c10::optional<at::Tensor> &value_sink,
    const c10::optional<at::Tensor> &quant_scale_out,
    const c10::optional<at::Tensor> &quant_offset_out,
    const c10::optional<at::Tensor> &meta_data,
    int64_t num_query_heads, int64_t num_key_value_heads, double softmax_scale,
    int64_t pre_tokens, int64_t next_tokens, c10::string_view input_layout,
    int64_t sparse_mode, int64_t block_size,
    int64_t query_quant_mode, int64_t key_quant_mode, int64_t value_quant_mode,
    int64_t inner_precise, bool return_softmax_lse, int64_t sink_number,
    c10::optional<int64_t> query_dtype, c10::optional<int64_t> key_dtype, c10::optional<int64_t> value_dtype,
    c10::optional<int64_t> query_rope_dtype, c10::optional<int64_t> key_rope_dtype,
    c10::optional<int64_t> key_shared_prefix_dtype, c10::optional<int64_t> value_shared_prefix_dtype,
    c10::optional<int64_t> dequant_scale_query_dtype, c10::optional<int64_t> dequant_scale_key_dtype,
    c10::optional<int64_t> dequant_scale_value_dtype, c10::optional<int64_t> dequant_scale_key_rope_dtype)
{
    // convert str
    std::string input_layout_str = std::string(input_layout);

    // construct the output tensor
    std::tuple<at::Tensor, at::Tensor> fia_output = custom::construct_fia_output_tensor_v2(query, value, input_layout_str,
                                                                                           quant_scale_out, block_table, num_query_heads,
                                                                                           num_key_value_heads,
                                                                                           return_softmax_lse, query_rope);
    at::Tensor output = std::get<0>(fia_output);
    at::Tensor softmax_lse = std::get<1>(fia_output);

    return std::tuple<at::Tensor, at::Tensor>(output, softmax_lse);
}


}

// 为NPU设备注册前向实现
TORCH_LIBRARY_IMPL(custom, PrivateUse1, m) {
    m.impl("npu_fused_infer_attention_sink", &custom::npu_fused_infer_attention_sink_npu);
}

// step5, 为META设备注册前向实现
TORCH_LIBRARY_IMPL(custom, Meta, m) {
    m.impl("npu_fused_infer_attention_sink", &custom::npu_fused_infer_attention_sink_meta);
}
