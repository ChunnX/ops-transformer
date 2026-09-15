/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {
namespace {
constexpr int64_t HEAD_DIM = 512;
constexpr int64_t MAX_NUM_HEADS = 128;
constexpr int64_t RANK_THREE = 3;
constexpr int64_t RANK_FOUR = 4;
constexpr int64_t METADATA_ELEMENTS = 480;
constexpr uintptr_t METADATA_ALIGNMENT = 64;

// FlashMla requires ND descriptors for every layout, including rank 3/4/5.
// The common at::Tensor converter selects NCL/NCHW/NCDHW for these ranks.
// ADL makes these local converters available to ACLNN_CMD's ConvertTypes.
struct NdTensor {
    at::Tensor tensor;
};

aclTensor *ConvertType(const NdTensor &input)
{
    const auto &tensor = input.tensor;
    if (!tensor.defined()) {
        return nullptr;
    }
    TORCH_CHECK(IsOpInputBaseFormat(tensor), "FlashMla requires tensors with base ND storage");
    static const auto create_tensor = GET_OP_API_FUNC(aclCreateTensor);
    TORCH_CHECK(create_tensor != nullptr, "aclCreateTensor was not found");
    // A view can span more storage than its logical shape. Describe the full
    // allocation so validation accepts its strides before AutoContiguous runs.
    const int64_t storage_elements = tensor.storage().nbytes() / tensor.itemsize();
    auto result = create_tensor(tensor.sizes().data(), tensor.dim(), ConvertToAclDataType(tensor.scalar_type()),
                                tensor.strides().data(), tensor.storage_offset(), ACL_FORMAT_ND, &storage_elements, 1,
                                const_cast<void *>(tensor.storage().data()));
    TORCH_CHECK(result != nullptr, "Failed to create an ND tensor descriptor");
    return result;
}

struct NdTensorList {
    const std::vector<at::Tensor> &tensors;
};

aclTensorList *ConvertType(const NdTensorList &input)
{
    // The wrapper validates that each list contains exactly one tensor.
    static const auto create_list = GET_OP_API_FUNC(aclCreateTensorList);
    TORCH_CHECK(create_list != nullptr, "aclCreateTensorList was not found");
    const auto tensor = ConvertType(NdTensor{input.tensors[0]});
    const auto result = create_list(&tensor, 1);
    if (result == nullptr) {
        Release(tensor);
        TORCH_CHECK(false, "Failed to create an ND tensor list descriptor");
    }
    return result;
}

void check_device(const at::Tensor &tensor, const c10::Device &device, const char *name)
{
    TORCH_CHECK(tensor.defined() && tensor.device() == device, name, " must be on ", device);
}

void check_optional_device(const c10::optional<at::Tensor> &tensor, const c10::Device &device, const char *name)
{
    if (tensor.has_value()) {
        check_device(*tensor, device, name);
    }
}

void check_sequence_descriptor(const c10::optional<at::Tensor> &tensor, const char *name)
{
    if (tensor.has_value()) {
        TORCH_CHECK(tensor->scalar_type() == at::kLong && tensor->dim() == 1 && tensor->is_contiguous(), name,
                    " must be a contiguous one-dimensional INT64 tensor");
    }
}

void check_metadata_descriptor(const at::Tensor &metadata)
{
    // Preserve the metadata transport contract from the former internal API.
    // FIA parameter/value checks remain in Host tiling and shared value tiling.
    TORCH_CHECK(metadata.scalar_type() == at::kInt && metadata.dim() == 1 && metadata.numel() == METADATA_ELEMENTS &&
                    metadata.is_contiguous() && metadata.storage_offset() == 0 && metadata.data_ptr() != nullptr &&
                    reinterpret_cast<uintptr_t>(metadata.data_ptr()) % METADATA_ALIGNMENT == 0,
                "metadata must be a 64-byte aligned INT32[480] tensor with zero offset");
}

bool has_only_dim0_stride(const at::Tensor &tensor)
{
    // Same predicate as FIA's IsFirstAxisOnlyNonContiguous. Other layouts are
    // copied rather than rejected, preserving the original accepted inputs.
    if (tensor.is_contiguous()) {
        return true;
    }
    if (tensor.dim() <= 1 || (tensor.size(0) > 1 && tensor.stride(0) <= 0)) {
        return false;
    }
    int64_t expected_stride = 1;
    for (int64_t dim = tensor.dim() - 1; dim >= 1; --dim) {
        if (tensor.size(dim) != 1 && tensor.stride(dim) != expected_stride) {
            return false;
        }
        expected_stride *= tensor.size(dim);
    }
    return true;
}

at::Tensor normalize_cache(const at::Tensor &tensor, bool keep_dim0)
{
    return keep_dim0 && has_only_dim0_stride(tensor) ? tensor : tensor.contiguous();
}

std::pair<std::vector<int64_t>, std::vector<int64_t>> output_shapes(const at::Tensor &query, int64_t num_heads,
                                                                    const std::string &input_layout,
                                                                    bool softmax_lse_flag)
{
    TORCH_CHECK(num_heads > 0 && num_heads <= MAX_NUM_HEADS, "num_heads must be in [1, 128], got ", num_heads);
    const auto split = input_layout.find('_');
    const auto layout_q = input_layout.substr(0, split);
    const auto layout_out = split == std::string::npos ? layout_q : input_layout.substr(split + 1);
    TORCH_CHECK(layout_q == "BSH" || layout_q == "BSND" || layout_q == "BNSD" || layout_q == "TND",
                "Unsupported input_layout: ", input_layout);
    TORCH_CHECK(split == std::string::npos || (layout_q == "TND" ? layout_out == "NTD" : layout_out == "NBSD"),
                "Unsupported output layout suffix: ", input_layout);
    const auto expected_rank = (layout_q == "BSH" || layout_q == "TND") ? RANK_THREE : RANK_FOUR;
    TORCH_CHECK(query.dim() == expected_rank, "query must have rank ", expected_rank, " for ", input_layout);

    int64_t batch = 0;
    int64_t seq = 0;
    if (layout_q == "BSH") {
        batch = query.size(0);
        seq = query.size(1);
        TORCH_CHECK(query.size(2) == num_heads * HEAD_DIM, "query hidden size must be num_heads * 512");
    } else {
        TORCH_CHECK(query.size(-1) == HEAD_DIM, "query head dimension must be 512");
        const auto heads_axis = layout_q == "BSND" ? 2 : 1;
        TORCH_CHECK(query.size(heads_axis) == num_heads, "query head count must match num_heads");
        if (layout_q == "TND") {
            seq = query.size(0);
        } else {
            batch = query.size(0);
            seq = query.size(layout_q == "BSND" ? 1 : 2);
        }
    }
    std::vector<int64_t> output = query.sizes().vec();
    if (layout_out == "NBSD") {
        output = {num_heads, batch, seq, HEAD_DIM};
    } else if (layout_out == "NTD") {
        output = {num_heads, seq, HEAD_DIM};
    }
    std::vector<int64_t> lse = {0};
    if (softmax_lse_flag) {
        lse = layout_q == "TND" ? std::vector<int64_t>{seq, num_heads, 1} :
                                  std::vector<int64_t>{batch, num_heads, seq, 1};
    }
    return {output, lse};
}

} // namespace

std::tuple<at::Tensor, at::Tensor> flash_mla(
    const at::Tensor &query, const std::vector<at::Tensor> &key, const std::vector<at::Tensor> &value,
    const at::Tensor &query_rope, const at::Tensor &key_rope, const at::Tensor &metadata, int64_t num_heads,
    const c10::optional<at::Tensor> &atten_mask, const c10::optional<at::Tensor> &actual_seq_lengths_q,
    const c10::optional<at::Tensor> &actual_seq_lengths_kv, const c10::optional<at::Tensor> &block_table,
    const c10::optional<at::Tensor> &dequant_scale_query, const c10::optional<at::Tensor> &key_antiquant_scale,
    const c10::optional<at::Tensor> &value_antiquant_scale, double scale_value, int64_t pre_tokens, int64_t next_tokens,
    std::string input_layout, int64_t num_key_value_heads, int64_t sparse_mode, int64_t inner_precise,
    int64_t block_size, bool softmax_lse_flag, int64_t key_antiquant_mode, int64_t value_antiquant_mode,
    int64_t query_quant_mode)
{
    TORCH_CHECK(query.defined() && query.device().type() == c10::DeviceType::PrivateUse1,
                "query must be an NPU tensor");
    TORCH_CHECK(key.size() == 1 && value.size() == 1, "key and value must each contain exactly one tensor");
    const auto device = query.device();
    check_device(key[0], device, "key[0]");
    check_device(value[0], device, "value[0]");
    check_device(query_rope, device, "query_rope");
    check_device(key_rope, device, "key_rope");
    check_device(metadata, device, "metadata");
    check_optional_device(atten_mask, device, "atten_mask");
    check_optional_device(actual_seq_lengths_q, device, "actual_seq_lengths_q");
    check_optional_device(actual_seq_lengths_kv, device, "actual_seq_lengths_kv");
    check_optional_device(block_table, device, "block_table");
    check_optional_device(dequant_scale_query, device, "dequant_scale_query");
    check_optional_device(key_antiquant_scale, device, "key_antiquant_scale");
    check_optional_device(value_antiquant_scale, device, "value_antiquant_scale");
    const auto dtype = query.scalar_type();
    TORCH_CHECK(dtype == at::kHalf || dtype == at::kBFloat16 || dtype == at::kChar,
                "query dtype must be float16, bfloat16 or int8");
    const auto output_dtype = dtype == at::kChar ? at::kBFloat16 : dtype;
    const auto shapes = output_shapes(query, num_heads, input_layout, softmax_lse_flag);
    const c10::OptionalDeviceGuard device_guard(device);
    const auto attention_out = at::empty(shapes.first, query.options().dtype(output_dtype));
    const auto softmax_lse = at::empty(shapes.second, query.options().dtype(at::kFloat));
    // FIA's L2 entry returns before cache preprocessing and Host tiling when
    // attentionOut is empty. A generated single-op entry has no such L2 branch.
    if (attention_out.numel() == 0) {
        return {attention_out, softmax_lse};
    }
    check_sequence_descriptor(actual_seq_lengths_q, "seq_q");
    check_sequence_descriptor(actual_seq_lengths_kv, "seq_kv");
    check_metadata_descriptor(metadata);
    // Match the V4 arch22 MLA cache policy. IgnoreContiguous keeps the accepted
    // PA block stride; the remaining cache views are made contiguous in torch.
    const bool keep_dim0 = block_table.has_value() && block_table->numel() != 0 && dtype != at::kChar &&
                           GetOpApiFuncAddr("NnopbaseSupportTensorV2") != nullptr;
    const std::vector<at::Tensor> key_normalized{normalize_cache(key[0], keep_dim0)};
    const std::vector<at::Tensor> value_normalized{normalize_cache(value[0], keep_dim0)};
    const auto key_rope_normalized = normalize_cache(key_rope, keep_dim0);
    const NdTensor query_nd{query};
    const NdTensorList key_list{key_normalized};
    const NdTensorList value_list{value_normalized};
    const NdTensor query_rope_nd{query_rope};
    const NdTensor key_rope_nd{key_rope_normalized};
    const NdTensor seq_q_nd{actual_seq_lengths_q.value_or(at::Tensor())};
    const NdTensor seq_kv_nd{actual_seq_lengths_kv.value_or(at::Tensor())};
    const NdTensor block_table_nd{block_table.value_or(at::Tensor())};
    const NdTensor atten_mask_nd{atten_mask.value_or(at::Tensor())};
    const NdTensor dequant_scale_query_nd{dequant_scale_query.value_or(at::Tensor())};
    const NdTensor key_antiquant_scale_nd{key_antiquant_scale.value_or(at::Tensor())};
    const NdTensor value_antiquant_scale_nd{value_antiquant_scale.value_or(at::Tensor())};
    const NdTensor attention_out_nd{attention_out};
    const NdTensor softmax_lse_nd{softmax_lse};
    // metadata is also the plain Tensor used by ACLNN_CMD to select the device.
    ACLNN_CMD(aclnnInnerFlashMla, query_nd, key_list, value_list, atten_mask_nd, seq_q_nd, seq_kv_nd, block_table_nd,
              key_antiquant_scale_nd, value_antiquant_scale_nd, query_rope_nd, key_rope_nd, dequant_scale_query_nd,
              metadata, num_heads, scale_value, pre_tokens, next_tokens, input_layout, num_key_value_heads, sparse_mode,
              inner_precise, block_size, softmax_lse_flag, key_antiquant_mode, value_antiquant_mode, query_quant_mode,
              attention_out_nd, softmax_lse_nd);
    return {attention_out, softmax_lse};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("flash_mla", &flash_mla, "flash_mla");
}
} // namespace op_api
