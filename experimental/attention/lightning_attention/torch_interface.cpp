/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include "acl/acl.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/DeviceUtils.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#include "op_host/fwd_lo_cmm128.h"
#include "op_host/bwd_lo_cmm128.h"

namespace npu_ops_transformer_ext::LightningAttn {

std::tuple<at::Tensor, at::Tensor>
lightning_attention_fwd_meta(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& slope, bool isTraining)
{
    int64_t b = q.sizes()[0];
    int64_t n = q.sizes()[1];
    int64_t s = q.sizes()[2];
    int64_t d = q.sizes()[3];
    auto dtype = q.options().dtype();
    auto device = q.options().device();

    int64_t chunkSize = 32;
    int chunkNum = (s + chunkSize - 1) / chunkSize;

    auto out = at::empty({b, n, s, d}, at::TensorOptions().dtype(at::kFloat).device(device));
    auto kvState = at::empty({b, n, chunkNum, d, d}, at::TensorOptions().dtype(at::kFloat).device(device));
    return {kvState, out};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
lightning_attention_bwd_meta(const at::Tensor& grad, const at::Tensor& q,
    const at::Tensor& k, const at::Tensor& v, const at::Tensor& slope, const at::Tensor& kv) {

    auto dQ = at::empty_like(grad, at::dtype(at::kFloat));
    auto dK = at::empty_like(grad, at::dtype(at::kFloat));
    auto dV = at::empty_like(grad, at::dtype(at::kFloat));
    return {dQ, dK, dV};
}

TORCH_LIBRARY_IMPL(npu_ops_transformer_ext, PrivateUse1, m) {
    m.impl("lightning_attention_fwd", &::LightningAttn::Wrapper::Forward::lo_cmm128::run);
    m.impl("lightning_attention_bwd", &::LightningAttn::Wrapper::Backward::lo_cmm128::run);
}

TORCH_LIBRARY_IMPL(npu_ops_transformer_ext, Meta, m) {
    m.impl("lightning_attention_fwd", &lightning_attention_fwd_meta);
    m.impl("lightning_attention_bwd", &lightning_attention_bwd_meta);
}

} // npu_ops_transformer_ext::LightningAttn

