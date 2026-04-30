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
using namespace at_npu::native;

// 工具函数，推导输出shape
std::tuple<at::Tensor, at::Tensor> construct_sfa_output_tensor_v2(
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
    if (input_layout_str == "TND") {
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
    } else if (input_layout_str == "NTD_TND") {
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
std::tuple<at::Tensor, at::Tensor> npu_sparse_flash_attention_gqa_symint_npu(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    const at::Tensor &sparse_indices, double scale_value, int64_t sparse_block_size,
    c10::OptionalArrayRef<c10::SymInt> actual_seq_lengths_query,
    c10::OptionalArrayRef<c10::SymInt> actual_seq_lengths_kv,
    const c10::optional<at::Tensor> &block_table,
    int64_t num_query_heads, int64_t num_key_value_heads,
    c10::string_view input_layout,
    c10::string_view input_layout_kv,
    int64_t sparse_mode,
    int64_t block_size,
    bool return_softmax_lse)
{   

    // convert str
    std::string input_layout_str = std::string(input_layout);
    std::string input_layout_kv_str = std::string(input_layout_kv);
    
    c10::optional<at::Tensor> quant_scale_out;
    c10::optional<at::Tensor> query_rope;
    // construct the output tensor
    std::tuple<at::Tensor, at::Tensor> fia_output = custom::construct_sfa_output_tensor_v2(query, value, input_layout_str,
                                                                                           quant_scale_out, block_table, num_query_heads,
                                                                                           num_key_value_heads,
                                                                                           return_softmax_lse, query_rope);
    at::Tensor output = std::get<0>(fia_output);
    at::Tensor softmax_lse = std::get<1>(fia_output);

    char *input_layout_ptr = const_cast<char *>(input_layout_str.c_str());
    char *input_layout_kv_ptr = const_cast<char *>(input_layout_kv_str.c_str());
    
    c10::OptionalArrayRef<c10::SymInt> actual_shared_prefix_len;

    at::TensorList valueTensors = value;
    at::TensorList keyTensors = key;

    EXEC_NPU_CMD_V1(aclnnAiInfraSparseFlashAttentionGqa, query, keyTensors, valueTensors,
            sparse_indices, scale_value, sparse_block_size, 
            actual_seq_lengths_query, actual_seq_lengths_kv, 
            block_table, 
            num_query_heads, num_key_value_heads,
            input_layout_ptr, input_layout_kv_ptr, sparse_mode, block_size, return_softmax_lse, output, softmax_lse);

    // std::printf("=============== EXEC_NPU_CMD_V1(aclnnAiInfraSparseFlashAttentionGqa) end ======================\n");
    return std::tuple<at::Tensor, at::Tensor>(output, softmax_lse);
}

std::tuple<at::Tensor, at::Tensor> npu_sparse_flash_attention_gqa_out_symint_npu(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    const at::Tensor &sparse_indices, double scale_value, int64_t sparse_block_size,
    c10::OptionalArrayRef<c10::SymInt> actual_seq_lengths_query,
    c10::OptionalArrayRef<c10::SymInt> actual_seq_lengths_kv,
    const c10::optional<at::Tensor> &block_table,
    int64_t num_query_heads, int64_t num_key_value_heads,
    c10::string_view input_layout,
    c10::string_view input_layout_kv,
    int64_t sparse_mode,
    int64_t block_size,
    bool return_softmax_lse,
    const c10::optional<at::Tensor> &workspace,
    at::TensorList out)
{   
    at::Tensor output = out[0];
    at::Tensor softmax_lse = out[1];
    // convert str
    std::string input_layout_str = std::string(input_layout);
    
    c10::optional<at::Tensor> quant_scale_out;
    c10::optional<at::Tensor> query_rope;


    char *input_layout_ptr = const_cast<char *>(input_layout_str.c_str());

    std::string input_layout_kv_str = std::string(input_layout_kv);
    char *input_layout_kv_ptr = const_cast<char *>(input_layout_kv_str.c_str());
    

    c10::OptionalArrayRef<c10::SymInt> actual_shared_prefix_len;

    at::TensorList valueTensors = value;
    at::TensorList keyTensors = key;
    
    if (workspace.has_value()) {
        void* workspace_addr = const_cast<void *>(workspace.value().storage().data());
        uint64_t workspace_size = static_cast<uint64_t>(workspace.value().numel() * workspace.value().element_size());
        EXEC_UPDATE_NPU_CMD_V1(aclnnAiInfraSparseFlashAttentionGqa, workspace_addr, workspace_size, query, keyTensors, valueTensors,
                sparse_indices, scale_value, sparse_block_size, 
                actual_seq_lengths_query, actual_seq_lengths_kv, 
                block_table, 
                num_query_heads, num_key_value_heads,
                input_layout_ptr, input_layout_kv_ptr, sparse_mode, block_size, return_softmax_lse, output, softmax_lse);
        
    } else {
        EXEC_NPU_CMD_V1(aclnnAiInfraSparseFlashAttentionGqa, query, keyTensors, valueTensors,
                sparse_indices, scale_value, sparse_block_size, 
                actual_seq_lengths_query, actual_seq_lengths_kv, 
                block_table, 
                num_query_heads, num_key_value_heads,
                input_layout_ptr, input_layout_kv_ptr, sparse_mode, block_size, return_softmax_lse, output, softmax_lse);
    }

    return std::tuple<at::Tensor, at::Tensor>(output, softmax_lse);
}

// 为META设备实现前向接口
std::tuple<at::Tensor, at::Tensor> npu_sparse_flash_attention_gqa_tensor_meta(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    const at::Tensor &sparse_indices, double scale_value, int64_t sparse_block_size,
    c10::optional<at::Tensor> actual_seq_lengths_query,
    c10::optional<at::Tensor> actual_seq_lengths_kv,
    const c10::optional<at::Tensor> &block_table,
    int64_t num_query_heads, int64_t num_key_value_heads,
    c10::string_view input_layout,
    c10::string_view input_layout_kv,
    int64_t sparse_mode,
    int64_t block_size,
    bool return_softmax_lse)
{   

    // convert str
    std::string input_layout_str = std::string(input_layout);

    c10::optional<at::Tensor> quant_scale_out;
    c10::optional<at::Tensor> query_rope;

    // construct the output tensor
    std::tuple<at::Tensor, at::Tensor> fia_output = custom::construct_sfa_output_tensor_v2(query, value, input_layout_str,
                                                                                           quant_scale_out, block_table, num_query_heads,
                                                                                           num_key_value_heads,
                                                                                           return_softmax_lse, query_rope);
    at::Tensor output = std::get<0>(fia_output);
    at::Tensor softmax_lse = std::get<1>(fia_output);


    return std::tuple<at::Tensor, at::Tensor>(output, softmax_lse);
}

std::tuple<at::Tensor, at::Tensor> npu_sparse_flash_attention_gqa_symint_meta(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    const at::Tensor &sparse_indices, double scale_value, int64_t sparse_block_size,
    c10::OptionalArrayRef<c10::SymInt> actual_seq_lengths_query,
    c10::OptionalArrayRef<c10::SymInt> actual_seq_lengths_kv,
    const c10::optional<at::Tensor> &block_table,
    int64_t num_query_heads, int64_t num_key_value_heads,
    c10::string_view input_layout,
    c10::string_view input_layout_kv,
    int64_t sparse_mode,
    int64_t block_size,
    bool return_softmax_lse)
{   

    // convert str
    std::string input_layout_str = std::string(input_layout);

    c10::optional<at::Tensor> quant_scale_out;
    c10::optional<at::Tensor> query_rope;

    // construct the output tensor
    std::tuple<at::Tensor, at::Tensor> fia_output = custom::construct_sfa_output_tensor_v2(query, value, input_layout_str,
                                                                                           quant_scale_out, block_table, num_query_heads,
                                                                                           num_key_value_heads,
                                                                                           return_softmax_lse, query_rope);
    at::Tensor output = std::get<0>(fia_output);
    at::Tensor softmax_lse = std::get<1>(fia_output);


    return std::tuple<at::Tensor, at::Tensor>(output, softmax_lse);
}


std::tuple<at::Tensor, at::Tensor> npu_sparse_flash_attention_gqa_infer_output_symint_npu(
    const at::Tensor &query,const at::Tensor &value,
    const c10::optional<at::Tensor> &block_table,
    int64_t num_query_heads, int64_t num_key_value_heads,
    c10::string_view input_layout,
    bool return_softmax_lse)
{   

    // convert str
    std::string input_layout_str = std::string(input_layout);

    c10::optional<at::Tensor> quant_scale_out;
    c10::optional<at::Tensor> query_rope;

    // construct the output tensor
    std::tuple<at::Tensor, at::Tensor> fia_output = custom::construct_sfa_output_tensor_v2(query, value, input_layout_str,
                                                                                           quant_scale_out, block_table, num_query_heads,
                                                                                           num_key_value_heads,
                                                                                           return_softmax_lse, query_rope);
    at::Tensor output = std::get<0>(fia_output);
    at::Tensor softmax_lse = std::get<1>(fia_output);


    return std::tuple<at::Tensor, at::Tensor>(output, softmax_lse);
}


at::Tensor npu_sparse_flash_attention_gqa_get_max_workspace_symint_npu(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    const at::Tensor &sparse_indices, double scale_value, int64_t sparse_block_size,
    c10::OptionalArrayRef<c10::SymInt> actual_seq_lengths_query,
    c10::OptionalArrayRef<c10::SymInt> actual_seq_lengths_kv,
    const c10::optional<at::Tensor> &block_table,
    int64_t num_query_heads, int64_t num_key_value_heads,
    c10::string_view input_layout,
    c10::string_view input_layout_kv,
    int64_t sparse_mode,
    int64_t block_size,
    bool return_softmax_lse)
{   

    // convert str
    std::string input_layout_str = std::string(input_layout);
    
    c10::optional<at::Tensor> quant_scale_out;
    c10::optional<at::Tensor> query_rope;
    // construct the output tensor
    std::tuple<at::Tensor, at::Tensor> fia_output = custom::construct_sfa_output_tensor_v2(query, value, input_layout_str,
                                                                                           quant_scale_out, block_table, num_query_heads,
                                                                                           num_key_value_heads,
                                                                                           return_softmax_lse, query_rope);
    at::Tensor output = std::get<0>(fia_output);
    at::Tensor softmax_lse = std::get<1>(fia_output);

    char *input_layout_ptr = const_cast<char *>(input_layout_str.c_str());

    c10::OptionalArrayRef<c10::SymInt> actual_shared_prefix_len;

    at::TensorList valueTensors = value;
    at::TensorList keyTensors = key;
    std::string input_layout_kv_str = std::string(input_layout_kv);
    char *input_layout_kv_ptr = const_cast<char *>(input_layout_kv_str.c_str());

    uint64_t workspaceSize = EXEC_GET_MAX_WORKSPACE_CMD(aclnnAiInfraSparseFlashAttentionGqa, query, keyTensors, valueTensors,
            sparse_indices, scale_value, sparse_block_size, 
            actual_seq_lengths_query, actual_seq_lengths_kv, 
            block_table, 
            num_query_heads, num_key_value_heads,
            input_layout_ptr, input_layout_kv_ptr, sparse_mode, block_size, return_softmax_lse, output, softmax_lse);

    at::Tensor workspaceTensor = at::empty_symint({workspaceSize}, query.options().dtype(query.dtype()));
    return workspaceTensor;
}

}

// 为NPU设备注册前向实现
TORCH_LIBRARY_IMPL(custom, PrivateUse1, m) {
    m.impl("npu_sparse_flash_attention_gqa", &custom::npu_sparse_flash_attention_gqa_symint_npu);
    m.impl("npu_sparse_flash_attention_gqa.out", &custom::npu_sparse_flash_attention_gqa_out_symint_npu);
    m.impl("npu_sparse_flash_attention_gqa_get_max_workspace", &custom::npu_sparse_flash_attention_gqa_get_max_workspace_symint_npu);
    m.impl("npu_sparse_flash_attention_gqa_infer_output", &custom::npu_sparse_flash_attention_gqa_infer_output_symint_npu);
}

// step5, 为META设备注册前向实现
TORCH_LIBRARY_IMPL(custom, Meta, m) {
    m.impl("npu_sparse_flash_attention_gqa", &custom::npu_sparse_flash_attention_gqa_symint_meta);
    m.impl("npu_sparse_flash_attention_gqa.tensor", &custom::npu_sparse_flash_attention_gqa_tensor_meta);
}
