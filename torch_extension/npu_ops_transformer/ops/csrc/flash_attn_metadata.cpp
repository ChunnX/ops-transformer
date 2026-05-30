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
 * \file flash_attn_metadata.cpp
 * \brief
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {
using npu_utils = at_npu::native::NpuUtils;

at::Tensor npu_flash_attn_metadata(const c10::optional<at::Tensor> &cu_seqlens_q,
                                   const c10::optional<at::Tensor> &cu_seqlens_kv,
                                   const c10::optional<at::Tensor> &seqused_q,
                                   const c10::optional<at::Tensor> &seqused_kv, int64_t num_heads_q,
                                   int64_t num_heads_kv, int64_t head_dim, int64_t batch_size, int64_t max_seqlen_q,
                                   int64_t max_seqlen_kv, int64_t mask_mode, int64_t win_left, int64_t win_right,
                                   std::string layout_q, std::string layout_kv, std::string layout_out, const at::Tensor &output)
{
    ACLNN_CMD(aclnnFlashAttnMetadata, cu_seqlens_q, cu_seqlens_kv, seqused_q, seqused_kv, batch_size, max_seqlen_q,
              max_seqlen_kv, num_heads_q, num_heads_kv, head_dim, mask_mode, win_left, win_right, layout_q, layout_kv,
              layout_out, output);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("npu_flash_attn_metadata", &npu_flash_attn_metadata, "npu_flash_attn_metadata");
}
} // namespace op_api
