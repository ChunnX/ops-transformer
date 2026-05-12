/**
 * -*- coding: utf-8 -*-
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Torch interface for `attention/block_sparse_attention` op. Mirrors the BSA
 * (blitz_sparse_attention) torch_interface structure. Per benchmark-mode
 * scoping the optional atten_mask is accepted at the python boundary for API
 * parity but ALWAYS passed as nullptr to the aclnn call — the block_sparse
 * kernel does not support fine-grained intra-block masking, so encoding the
 * sparsity at the block-level mask is the only path.
 */

#include <torch/extension.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include "acl/acl.h"
#include "aclnnop/aclnn_block_sparse_attention.h"

// Helper function to convert torch dtype to ACL dtype
aclDataType get_acl_dtype(const at::Tensor& tensor) {
    if (tensor.scalar_type() == at::kBFloat16) {
        return aclDataType::ACL_BF16;
    } else if (tensor.scalar_type() == at::kHalf) {
        return aclDataType::ACL_FLOAT16;
    } else if (tensor.scalar_type() == at::kFloat) {
        return aclDataType::ACL_FLOAT;
    } else if (tensor.scalar_type() == at::kInt) {
        return aclDataType::ACL_INT32;
    } else if (tensor.scalar_type() == at::kChar) {        // int8
        return aclDataType::ACL_INT8;
    } else if (tensor.scalar_type() == at::kByte) {        // uint8
        return aclDataType::ACL_UINT8;
    } else if (tensor.scalar_type() == at::kBool) {
        return aclDataType::ACL_BOOL;
    }
    return aclDataType::ACL_DT_UNDEFINED;
}

// Helper function to create ACL tensor from torch tensor
aclTensor* create_acl_tensor(const at::Tensor& tensor) {
    auto sizes = tensor.sizes();
    std::vector<int64_t> shape(sizes.begin(), sizes.end());

    // Calculate strides (row-major)
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    aclDataType dtype = get_acl_dtype(tensor);
    void* data_ptr = tensor.data_ptr();

    return aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0,
                           aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), data_ptr);
}

/**
 * @brief Python interface to `aclnnBlockSparseAttention`.
 *
 * @param query Tensor [B, N, S, D] (BNSD) or [T, N, D] (TND), fp16/bf16
 * @param key Tensor matching kv_input_layout
 * @param value Tensor matching kv_input_layout
 * @param block_sparse_mask Int8 tensor [B, N, q_blocks, kv_blocks] — required
 * @param block_shape List [block_x, block_y] e.g. [128, 128] — required
 * @param actual_seq_lengths List of Q seq lengths per batch (int64)
 * @param actual_seq_lengths_kv List of KV seq lengths per batch (int64)
 * @param num_heads Number of Q heads
 * @param num_key_value_heads Number of K/V heads (GQA)
 * @param q_input_layout "BNSD" or "TND"
 * @param kv_input_layout "BNSD" or "TND"
 * @param mask_type 0 = no mask (default)
 * @param scale_value Softmax scaling (typically 1/sqrt(D))
 * @param inner_precise 0 = fp32 softmax (high precision), 1 = fp16 softmax
 * @param block_size Paged-attention block size; pass 128 when unused
 * @param pre_tokens Causal window (use 2147483647 for "infinity" / unrestricted)
 * @param next_tokens Causal window (use 2147483647 for "infinity" / unrestricted)
 * @param softmax_lse_flag If true, return [B, N, S] float32 LSE; else empty {0}
 * @param atten_mask UNUSED — accepted for API parity but always passed as nullptr.
 *
 * @return tuple (attention_out, softmax_lse)
 */
std::tuple<at::Tensor, at::Tensor> block_sparse_attention(
    const at::Tensor &query,
    const at::Tensor &key,
    const at::Tensor &value,
    const at::Tensor &block_sparse_mask,
    std::vector<int64_t> block_shape,
    c10::optional<std::vector<int64_t>> actual_seq_lengths_opt = c10::nullopt,
    c10::optional<std::vector<int64_t>> actual_seq_lengths_kv_opt = c10::nullopt,
    c10::optional<int64_t> num_heads_opt = c10::nullopt,
    c10::optional<int64_t> num_key_value_heads_opt = c10::nullopt,
    c10::optional<std::string> q_input_layout_opt = c10::nullopt,
    c10::optional<std::string> kv_input_layout_opt = c10::nullopt,
    c10::optional<int64_t> mask_type_opt = c10::nullopt,
    c10::optional<double> scale_value_opt = c10::nullopt,
    c10::optional<int64_t> inner_precise_opt = c10::nullopt,
    c10::optional<int64_t> block_size_opt = c10::nullopt,
    c10::optional<int64_t> pre_tokens_opt = c10::nullopt,
    c10::optional<int64_t> next_tokens_opt = c10::nullopt,
    c10::optional<bool> softmax_lse_flag_opt = c10::nullopt,
    const c10::optional<at::Tensor> &atten_mask = c10::nullopt
) {
    // Defaults match the example test (test_aclnn_block_sparse_attention.cpp).
    int64_t num_heads = num_heads_opt.value_or(1);
    int64_t num_key_value_heads = num_key_value_heads_opt.value_or(num_heads);
    std::string q_input_layout = q_input_layout_opt.value_or("BNSD");
    std::string kv_input_layout = kv_input_layout_opt.value_or("BNSD");
    int64_t mask_type = mask_type_opt.value_or(0);
    double scale_value = scale_value_opt.value_or(1.0);
    int64_t inner_precise = inner_precise_opt.value_or(0);
    int64_t block_size = block_size_opt.value_or(128);
    // 2147483647 ("max int32") = effectively unrestricted window, per the
    // example test. README says preTokens/nextTokens "当前不支持" — meaning the
    // window restriction is not implemented; passing infinity matches the test.
    int64_t pre_tokens = pre_tokens_opt.value_or(2147483647);
    int64_t next_tokens = next_tokens_opt.value_or(2147483647);
    bool softmax_lse_flag = softmax_lse_flag_opt.value_or(false);

    // Validate inputs
    TORCH_CHECK(query.is_contiguous(), "query must be contiguous");
    TORCH_CHECK(key.is_contiguous(), "key must be contiguous");
    TORCH_CHECK(value.is_contiguous(), "value must be contiguous");
    TORCH_CHECK(block_sparse_mask.is_contiguous(), "block_sparse_mask must be contiguous");
    TORCH_CHECK(block_sparse_mask.scalar_type() == at::kChar,
                "block_sparse_mask must be int8 (torch.int8)");
    TORCH_CHECK(block_shape.size() == 2, "block_shape must be exactly 2 values [block_x, block_y]");
    TORCH_CHECK(q_input_layout == "BNSD" || q_input_layout == "TND",
                "q_input_layout must be 'BNSD' or 'TND'");
    TORCH_CHECK(kv_input_layout == "BNSD" || kv_input_layout == "TND",
                "kv_input_layout must be 'BNSD' or 'TND'");

    // Build aclTensors (atten_mask deliberately not forwarded — see header comment).
    aclTensor* query_tensor = create_acl_tensor(query);
    aclTensor* key_tensor = create_acl_tensor(key);
    aclTensor* value_tensor = create_acl_tensor(value);
    aclTensor* block_sparse_mask_tensor = create_acl_tensor(block_sparse_mask);

    // aclIntArrays (host data must outlive the aclnn call).
    aclIntArray* block_shape_array = aclCreateIntArray(block_shape.data(), block_shape.size());

    aclIntArray* actual_seq_lengths_array = nullptr;
    aclIntArray* actual_seq_lengths_kv_array = nullptr;
    if (actual_seq_lengths_opt.has_value()) {
        const auto& v = actual_seq_lengths_opt.value();
        actual_seq_lengths_array = aclCreateIntArray(v.data(), v.size());
    }
    if (actual_seq_lengths_kv_opt.has_value()) {
        const auto& v = actual_seq_lengths_kv_opt.value();
        actual_seq_lengths_kv_array = aclCreateIntArray(v.data(), v.size());
    }

    // Layout strings: null-terminated char buffers (16 bytes each, matching the example test).
    char q_layout_buf[16] = {0};
    char kv_layout_buf[16] = {0};
    std::strncpy(q_layout_buf, q_input_layout.c_str(), sizeof(q_layout_buf) - 1);
    std::strncpy(kv_layout_buf, kv_input_layout.c_str(), sizeof(kv_layout_buf) - 1);

    // Output attention_out (same dtype + shape as query).
    at::Tensor output = torch::empty_like(query);
    aclTensor* output_tensor = create_acl_tensor(output);

    // LSE shape for BNSD: [B, N, S, 1] float32. The infershape code
    // (block_sparse_attention_infershape.cpp:194-197) sets LSE_DIM_D=1, so the
    // tensor must be 4-D. Passing [B, N, S] (3-D) makes
    // aclnnBlockSparseAttentionGetWorkspaceSize fail with error 561103.
    at::Tensor lse_out;
    aclTensor* lse_tensor = nullptr;
    if (softmax_lse_flag) {
        TORCH_CHECK(q_input_layout == "BNSD",
                    "softmax_lse_flag=True only supported for BNSD Q layout in this interface");
        int64_t B = query.size(0);
        int64_t N = num_heads;
        int64_t S = query.size(2);
        lse_out = torch::empty({B, N, S, 1},
                               torch::dtype(torch::kFloat32).device(query.device()));
        lse_tensor = create_acl_tensor(lse_out);
    } else {
        lse_out = torch::empty({0},
                               torch::dtype(torch::kFloat32).device(query.device()));
    }

    // First stage: GetWorkspaceSize.
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    void* workspace_addr = nullptr;

    int ret = aclnnBlockSparseAttentionGetWorkspaceSize(
        query_tensor,                  // query
        key_tensor,                    // key
        value_tensor,                  // value
        block_sparse_mask_tensor,      // blockSparseMaskOptional (required for us)
        nullptr,                       // attenMaskOptional — DELIBERATELY null
        block_shape_array,             // blockShapeOptional
        actual_seq_lengths_array,      // actualSeqLengthsOptional
        actual_seq_lengths_kv_array,   // actualSeqLengthsKvOptional
        nullptr,                       // blockTableOptional (paged attn — unused)
        q_layout_buf,                  // qInputLayout
        kv_layout_buf,                 // kvInputLayout
        num_key_value_heads,           // numKeyValueHeads
        mask_type,                     // maskType
        scale_value,                   // scaleValue
        inner_precise,                 // innerPrecise
        block_size,                    // blockSize
        pre_tokens,                    // preTokens
        next_tokens,                   // nextTokens
        softmax_lse_flag ? 1 : 0,      // softmaxLseFlag
        output_tensor,                 // attentionOut (out)
        lse_tensor,                    // softmaxLseOptional (out, may be null)
        &workspace_size,               // workspaceSize (out)
        &executor                      // executor (out)
    );

    TORCH_CHECK(ret == ACL_SUCCESS,
                "aclnnBlockSparseAttentionGetWorkspaceSize failed with error: ", ret);

    // Allocate workspace.
    if (workspace_size > 0) {
        ret = aclrtMalloc(&workspace_addr, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
        TORCH_CHECK(ret == ACL_SUCCESS, "aclrtMalloc workspace failed with error: ", ret);
    }

    // Acquire the current torch_npu stream.
    int deviceId;
    aclrtGetDevice(&deviceId);
    auto npuStream = c10_npu::getCurrentNPUStream(deviceId);
    auto aclStream = npuStream.stream(true);

    // Second stage: run.
    ret = aclnnBlockSparseAttention(workspace_addr, workspace_size, executor, aclStream);
    TORCH_CHECK(ret == ACL_SUCCESS,
                "aclnnBlockSparseAttention execution failed with error: ", ret);

    // Synchronize before freeing the workspace.
    ret = aclrtSynchronizeStream(aclStream);
    TORCH_CHECK(ret == ACL_SUCCESS, "aclrtSynchronizeStream failed with error: ", ret);

    // Cleanup.
    if (workspace_addr) aclrtFree(workspace_addr);
    if (block_shape_array) aclDestroyIntArray(block_shape_array);
    if (actual_seq_lengths_array) aclDestroyIntArray(actual_seq_lengths_array);
    if (actual_seq_lengths_kv_array) aclDestroyIntArray(actual_seq_lengths_kv_array);
    if (lse_tensor) aclDestroyTensor(lse_tensor);

    return std::make_tuple(output, lse_out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("block_sparse_attention", &block_sparse_attention,
        R"DOC(
        Interface to the `aclnnBlockSparseAttention` kernel (attention/block_sparse_attention).
        Performs flash attention with block-level sparsity. The atten_mask
        parameter is accepted for API parity with blitz_sparse_attention but
        is NOT forwarded to the aclnn call (the kernel does not support
        intra-block fine-grained masking on CANN 8.5.0-beta.1).

        Args:
            query (torch.Tensor): Query tensor (fp16/bf16). BNSD: [B, N, S, D] or TND: [T, N, D].
            key (torch.Tensor): Key tensor matching kv_input_layout.
            value (torch.Tensor): Value tensor matching kv_input_layout.
            block_sparse_mask (torch.Tensor): int8 mask [B, N, q_blocks, kv_blocks].
                1 = block kept; 0 = block skipped.
            block_shape (list[int]): [block_x, block_y] — Q block size × KV block size. Typically [128, 128].
            actual_seq_lengths (list[int], optional): Actual Q seq lengths per batch.
            actual_seq_lengths_kv (list[int], optional): Actual KV seq lengths per batch.
            num_heads (int, optional): Number of Q heads. Default 1.
            num_key_value_heads (int, optional): Number of KV heads (GQA). Default = num_heads.
            q_input_layout (str, optional): "BNSD" (default) or "TND".
            kv_input_layout (str, optional): "BNSD" (default) or "TND".
            mask_type (int, optional): 0 = no mask. Default 0.
            scale_value (float, optional): Softmax scale (typically 1/sqrt(D)). Default 1.0.
            inner_precise (int, optional): 0 = fp32 softmax (high precision), 1 = fp16. Default 0.
            block_size (int, optional): Paged-attn block size. Default 128.
            pre_tokens (int, optional): Causal window — pass 2147483647 (default) for unrestricted.
            next_tokens (int, optional): Causal window — pass 2147483647 (default) for unrestricted.
            softmax_lse_flag (bool, optional): If True, return [B, N, S] float32 LSE. Default False.
            atten_mask (torch.Tensor, optional): IGNORED — present for API parity only.

        Returns:
            tuple(torch.Tensor, torch.Tensor):
                - attention_out: same dtype + shape as query
                - softmax_lse: [B, N, S] float32 when softmax_lse_flag=True, else empty {0}
        )DOC",
        py::arg("query"),
        py::arg("key"),
        py::arg("value"),
        py::arg("block_sparse_mask"),
        py::arg("block_shape"),
        py::arg("actual_seq_lengths") = c10::optional<std::vector<int64_t>>(),
        py::arg("actual_seq_lengths_kv") = c10::optional<std::vector<int64_t>>(),
        py::arg("num_heads") = c10::optional<int64_t>(),
        py::arg("num_key_value_heads") = c10::optional<int64_t>(),
        py::arg("q_input_layout") = c10::optional<std::string>(),
        py::arg("kv_input_layout") = c10::optional<std::string>(),
        py::arg("mask_type") = c10::optional<int64_t>(),
        py::arg("scale_value") = c10::optional<double>(),
        py::arg("inner_precise") = c10::optional<int64_t>(),
        py::arg("block_size") = c10::optional<int64_t>(),
        py::arg("pre_tokens") = c10::optional<int64_t>(),
        py::arg("next_tokens") = c10::optional<int64_t>(),
        py::arg("softmax_lse_flag") = c10::optional<bool>(),
        py::arg("atten_mask") = c10::optional<at::Tensor>()
    );
}
