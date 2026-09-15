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
constexpr int64_t METADATA_SIZE = 480;
constexpr uintptr_t METADATA_ALIGNMENT = 64;
constexpr int64_t INFER_DIMENSION = -1;

int64_t metadata_dimension(const c10::optional<int64_t> &value, const char *name)
{
    TORCH_CHECK(!value.has_value() || *value >= 0, name, " must be nonnegative or None");
    return value.value_or(INFER_DIMENSION);
}

void check_seq(const c10::optional<at::Tensor> &seq, const c10::Device &device, const char *name)
{
    if (seq.has_value()) {
        TORCH_CHECK(seq->device() == device, name, " must be on ", device, ", got ", seq->device());
        TORCH_CHECK(seq->scalar_type() == at::kLong && seq->dim() == 1 && seq->is_contiguous(), name,
                    " must be a contiguous one-dimensional int64 tensor");
    }
}
} // namespace

at::Tensor flash_mla_metadata(int64_t num_heads, const c10::optional<int64_t> &batch_size,
                              const c10::optional<int64_t> &q_seq_size, const c10::optional<int64_t> &kv_seq_size,
                              const c10::optional<at::Tensor> &actual_seq_lengths_q,
                              const c10::optional<at::Tensor> &actual_seq_lengths_kv, std::string input_layer,
                              int64_t block_size, int64_t sparse_mode, int64_t pre_tokens,
                              int64_t next_tokens, bool has_atten_mask, int64_t quant_mode,
                              const c10::optional<c10::Device> &device)
{
    TORCH_CHECK(quant_mode == 0 || quant_mode == 1,
                "quant_mode must be 0 (nonquant) or 1 (INT8 full quantization)");
    const int64_t batch = metadata_dimension(batch_size, "batch_size");
    const int64_t q_size = metadata_dimension(q_seq_size, "q_seq_size");
    const int64_t kv_size = metadata_dimension(kv_seq_size, "kv_seq_size");
    auto output_device = c10::Device(c10::DeviceType::PrivateUse1);
    if (device.has_value()) {
        output_device = *device;
    } else if (actual_seq_lengths_q.has_value()) {
        output_device = actual_seq_lengths_q->device();
    } else if (actual_seq_lengths_kv.has_value()) {
        output_device = actual_seq_lengths_kv->device();
    }
    TORCH_CHECK(output_device.type() == c10::DeviceType::PrivateUse1, "metadata output must be on an NPU");
    at::Tensor metadata_out;
    {
        const c10::OptionalDeviceGuard device_guard(output_device);
        metadata_out = at::empty({METADATA_SIZE}, at::TensorOptions().device(output_device).dtype(at::kInt));
    }
    check_seq(actual_seq_lengths_q, metadata_out.device(), "actual_seq_lengths_q");
    check_seq(actual_seq_lengths_kv, metadata_out.device(), "actual_seq_lengths_kv");
    TORCH_CHECK(metadata_out.storage_offset() == 0 &&
                    reinterpret_cast<uintptr_t>(metadata_out.data_ptr()) % METADATA_ALIGNMENT == 0,
                "metadata allocation must have zero storage offset and be 64-byte aligned");
    ACLNN_CMD(aclnnInnerFlashMlaMetadata, actual_seq_lengths_q, actual_seq_lengths_kv, num_heads, batch,
              q_size, kv_size, input_layer, block_size, sparse_mode, pre_tokens, next_tokens,
              has_atten_mask, quant_mode, metadata_out);
    return metadata_out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("flash_mla_metadata", &flash_mla_metadata, "flash_mla_metadata");
}
} // namespace op_api
