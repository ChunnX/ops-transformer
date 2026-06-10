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
#include <torch/library.h>
#include "ops.h"

namespace {

TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def("stem_oam_prep_varlen_q(Tensor q, Tensor qSeqLens, Tensor cuSeqLensQ, *, Tensor? qScale=None, Tensor? kScale=None, int stemBlockSize=128, int stemStride=16) -> Tensor");
}

TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("stem_oam_prep_varlen_q", TORCH_FN(ascend_kernel::stem_oam_prep_varlen_q_torch));
}

at::Tensor stem_oam_prep_varlen_q_meta(
    const at::Tensor& q,
    const at::Tensor& qSeqLens,
    const at::Tensor& cuSeqLensQ,
    const c10::optional<at::Tensor>& qScale,
    const c10::optional<at::Tensor>& kScale,
    int64_t stemBlockSize,
    int64_t stemStride)
{
    int64_t B = stemBlockSize;
    int64_t S = stemStride;
    int64_t qflat_dim = S * 128;
    int64_t batch = qSeqLens.size(0);
    int64_t H_q   = q.size(1);
    int64_t maxQLen = qSeqLens.max().item<int32_t>();
    int64_t maxQPadded = ((maxQLen + B - 1) / B) * B;
    int64_t maxQb = maxQPadded / B;
    return at::empty({batch, H_q, maxQb, qflat_dim}, q.options().dtype(at::kBFloat16));
}

TORCH_LIBRARY_IMPL(npu, Meta, m)
{
    m.impl("stem_oam_prep_varlen_q", &stem_oam_prep_varlen_q_meta);
}

}