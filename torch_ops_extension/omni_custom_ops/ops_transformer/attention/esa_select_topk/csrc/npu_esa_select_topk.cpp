// npu_esa_select_topk.cpp
#include <iostream>
#include <torch/library.h>
#include "../../../../csrc_base/ops_common.h"

namespace custom {
using namespace at_npu::native;

// 为NPU设备实现前向接口
at::Tensor npu_esa_select_topk_symint_npu(
    const at::Tensor & query,
    const at::Tensor & key,
    int64_t blk_size,
    int64_t init_blk_num,
    int64_t local_blk_num,
    int64_t topk,
    c10::string_view input_layout,
    c10::OptionalArrayRef<c10::SymInt> actual_seq_q_len_optional,
    c10::OptionalArrayRef<c10::SymInt> actual_seq_k_len_optional,
    c10::OptionalArrayRef<c10::SymInt> actual_cmp_seq_k_len_optional,
    int64_t compress_blk_size)
{
    std::string input_layout_str = std::string(input_layout);
    
    // construct the output tensor
    at::Tensor topk_indices;
    if (input_layout_str == "TND") {
        topk_indices = at::empty_symint({query.sym_size(0), key.sym_size(2), init_blk_num + local_blk_num + topk + 1}, query.options().dtype(at::kInt));
    }
        // convert str
    char *input_layout_ptr = const_cast<char *>(input_layout_str.c_str());
    EXEC_NPU_CMD_V1(aclnnAiInfraEsaSelectTopk, query, key, actual_seq_q_len_optional, actual_seq_k_len_optional, actual_cmp_seq_k_len_optional,
                    blk_size, init_blk_num, local_blk_num, topk, input_layout_ptr, compress_blk_size, topk_indices);
    return topk_indices;
}

at::Tensor npu_esa_select_topk_out_symint_npu(
    const at::Tensor & query,
    const at::Tensor & key,
    int64_t blk_size,
    int64_t init_blk_num,
    int64_t local_blk_num,
    int64_t topk,
    c10::string_view input_layout,
    c10::OptionalArrayRef<c10::SymInt> actual_seq_q_len_optional,
    c10::OptionalArrayRef<c10::SymInt> actual_seq_k_len_optional,
    c10::OptionalArrayRef<c10::SymInt> actual_cmp_seq_k_len_optional,
    int64_t compress_blk_size,
    const c10::optional<at::Tensor> &workspace,
    at::Tensor & out)
{
    std::string input_layout_str = std::string(input_layout);
    char *input_layout_ptr = const_cast<char *>(input_layout_str.c_str());
    if (workspace.has_value()) {
        void* workspace_addr = const_cast<void *>(workspace.value().storage().data());
        uint64_t workspace_size = static_cast<uint64_t>(workspace.value().numel() * workspace.value().element_size());
        EXEC_UPDATE_NPU_CMD_V1(aclnnAiInfraEsaSelectTopk, workspace_addr, workspace_size, query, key, actual_seq_q_len_optional,
                               actual_seq_k_len_optional, actual_cmp_seq_k_len_optional,
                               blk_size, init_blk_num, local_blk_num, topk, input_layout_ptr, compress_blk_size, out);
    } else {
        EXEC_NPU_CMD_V1(aclnnAiInfraEsaSelectTopk, query, key, actual_seq_q_len_optional, actual_seq_k_len_optional, actual_cmp_seq_k_len_optional,
                        blk_size, init_blk_num, local_blk_num, topk, input_layout_ptr, compress_blk_size, out);
    }
    return out;
}

at::Tensor npu_esa_select_topk_get_max_workspace_symint_npu(
    const at::Tensor & query,
    const at::Tensor & key,
    int64_t blk_size,
    int64_t init_blk_num,
    int64_t local_blk_num,
    int64_t topk,
    c10::string_view input_layout,
    c10::OptionalArrayRef<c10::SymInt> actual_seq_q_len_optional,
    c10::OptionalArrayRef<c10::SymInt> actual_seq_k_len_optional,
    c10::OptionalArrayRef<c10::SymInt> actual_cmp_seq_k_len_optional,
    int64_t compress_blk_size)
{
    std::string input_layout_str = std::string(input_layout);
    
    // construct the output tensor
    at::Tensor topk_indices;
    if (input_layout_str == "TND") {
        topk_indices = at::empty_symint({query.sym_size(0), key.sym_size(2), init_blk_num + local_blk_num + topk + 1}, query.options().dtype(at::kInt));
    }
        // convert str
    char *input_layout_ptr = const_cast<char *>(input_layout_str.c_str());
    uint64_t workspaceSize = EXEC_GET_MAX_WORKSPACE_CMD(aclnnAiInfraEsaSelectTopk, query, key, actual_seq_q_len_optional,
                                                        actual_seq_k_len_optional, actual_cmp_seq_k_len_optional,
                                                        blk_size, init_blk_num, local_blk_num, topk, input_layout_ptr,
                                                        compress_blk_size, topk_indices);
    at::Tensor workspaceTensor = at::empty_symint({workspaceSize}, query.options().dtype(query.dtype()));
    return workspaceTensor;
}

// 为META设备实现前向接口
at::Tensor npu_esa_select_topk_meta(
    const at::Tensor & query,
    const at::Tensor & key,
    int64_t blk_size,
    int64_t init_blk_num,
    int64_t local_blk_num,
    int64_t topk,
    c10::string_view input_layout,
    c10::OptionalArrayRef<c10::SymInt> actual_seq_q_len_optional,
    c10::OptionalArrayRef<c10::SymInt> actual_seq_k_len_optional,
    c10::OptionalArrayRef<c10::SymInt> actual_cmp_seq_k_len_optional,
    int64_t compress_blk_size)
{
    std::string input_layout_str = std::string(input_layout);
    at::Tensor topk_indices;
    if (input_layout_str == "TND") {
        topk_indices = at::empty_symint({query.sym_size(0), key.sym_size(2), init_blk_num + local_blk_num + topk + 1}, query.options().dtype(at::kInt));
    }
    return topk_indices;
}

at::Tensor _npu_esa_select_topk_infer_output(
    const at::Tensor & query,
    const at::Tensor & key,
    int64_t init_blk_num,
    int64_t local_blk_num,
    int64_t topk,
    c10::string_view input_layout)
{
    std::string input_layout_str = std::string(input_layout);
    at::Tensor topk_indices;
    if (input_layout_str == "TND") {
        topk_indices = at::empty_symint({query.sym_size(0), key.sym_size(2), init_blk_num + local_blk_num + topk + 1}, query.options().dtype(at::kInt));
    }
    return topk_indices;
}
}
// step3, 为NPU设备注册前向实现
TORCH_LIBRARY_IMPL(custom, PrivateUse1, m) {
    m.impl("npu_esa_select_topk", &custom::npu_esa_select_topk_symint_npu);
    m.impl("npu_esa_select_topk.out", &custom::npu_esa_select_topk_out_symint_npu);
    m.impl("_npu_esa_select_topk_get_max_workspace", &custom::npu_esa_select_topk_get_max_workspace_symint_npu);
    m.impl("_npu_esa_select_topk_infer_output", &custom::_npu_esa_select_topk_infer_output);
}

// step4, 为META设备注册前向实现
TORCH_LIBRARY_IMPL(custom, Meta, m) {
    m.impl("npu_esa_select_topk", &custom::npu_esa_select_topk_meta);
}