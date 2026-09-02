/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*! \file grouped_matmul_quant_torch.cpp
 *  \brief PyTorch registration for GroupedMatmulQuant.
 */

#include <ATen/ATen.h>
#include <torch/library.h>

// EXEC_NPU_CMD moved between torch_npu and op-plugin across releases. Support
// the layouts shipped by the currently supported torch_npu packages.
#if __has_include("third_party/op-plugin/op_plugin/utils/op_api_common.h")
#include "third_party/op-plugin/op_plugin/utils/op_api_common.h"
#elif __has_include("op_plugin/utils/op_api_common.h")
#include "op_plugin/utils/op_api_common.h"
#elif __has_include("torch_npu/csrc/aten/ops/op_api/op_api_common.h")
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"
#else
#error "Cannot find the torch_npu ACLNN op-api helper"
#endif

#include "grouped_matmul_quant_torch_adpt.h"

namespace npu_ops_transformer_ext {
namespace {

at::Tensor GroupedMatmulQuantMeta(const at::Tensor& x, const at::Tensor& quantizedWeight,
                                  const at::Tensor& weightScale, const at::Tensor& weightOffset,
                                  const c10::optional<at::Tensor>& groupList, int64_t scaleGroupSize)
{
    static_cast<void>(weightScale);
    static_cast<void>(weightOffset);
    static_cast<void>(groupList);
    static_cast<void>(scaleGroupSize);
    constexpr int64_t FRACTAL_SIZE = 16;
    const int64_t m = x.size(0);
    const int64_t n = quantizedWeight.size(2) * FRACTAL_SIZE;
    return at::empty({m, n}, x.options());
}

} // namespace

TORCH_LIBRARY_IMPL(npu_ops_transformer_ext, PrivateUse1, m)
{
    m.impl("grouped_matmul_quant", vllm_ascend::grouped_matmul_quant);
}

TORCH_LIBRARY_IMPL(npu_ops_transformer_ext, Meta, m)
{
    m.impl("grouped_matmul_quant", TORCH_FN(GroupedMatmulQuantMeta));
}

} // namespace npu_ops_transformer_ext
