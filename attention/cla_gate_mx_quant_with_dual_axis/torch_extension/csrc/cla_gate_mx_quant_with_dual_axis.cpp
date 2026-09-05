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
#include "aclnnop/aclnn_cla_gate_mx_quant_with_dual_axis.h"
#include "aclnn_common.h"

namespace cann_ops_nn {
namespace quant {
namespace {
constexpr int64_t EXPECTED_S = 256;
constexpr int64_t EXPECTED_B = 4;
constexpr int64_t EXPECTED_N = 96;
constexpr int64_t EXPECTED_D = 256;
constexpr int64_t SPLIT_BLOCK_SIZE = 64;
constexpr int64_t DST_TYPE_FP8_E5M2 = 35;
constexpr int64_t DST_TYPE_FP8_E4M3FN = 36;

static aclDataType GetAclDataTypeFromDstType(int64_t dst_type)
{
    switch (dst_type) {
        case DST_TYPE_FP8_E5M2:
            return ACL_FLOAT8_E5M2;
        case DST_TYPE_FP8_E4M3FN:
            return ACL_FLOAT8_E4M3FN;
        default:
            TORCH_CHECK(false, "dst_type should be 35(float8_e5m2) or 36(float8_e4m3fn), but got ", dst_type);
    }
}

static at::ScalarType GetScalarTypeFromDstType(int64_t dst_type)
{
    switch (dst_type) {
        case DST_TYPE_FP8_E5M2:
            return at::ScalarType::Float8_e5m2;
        case DST_TYPE_FP8_E4M3FN:
            return at::ScalarType::Float8_e4m3fn;
        default:
            TORCH_CHECK(false, "dst_type should be 35(float8_e5m2) or 36(float8_e4m3fn), but got ", dst_type);
    }
}

void CheckNpuTensor(const at::Tensor& tensor, const char* name)
{
    TORCH_CHECK(tensor.defined(), name, " must be defined");
    TORCH_CHECK(tensor.device().type() == at::kPrivateUse1, name, " must be on NPU device");
}
} // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
cla_gate_mx_quant_with_dual_axis(
    const at::Tensor& global_out,
    const at::Tensor& local_out,
    const at::Tensor& global_gate_logits,
    const at::Tensor& local_gate_logits,
    int64_t scale_alg,
    int64_t dst_type)
{
    CheckNpuTensor(global_out, "global_out");
    CheckNpuTensor(local_out, "local_out");
    CheckNpuTensor(global_gate_logits, "global_gate_logits");
    CheckNpuTensor(local_gate_logits, "local_gate_logits");

    TORCH_CHECK(global_out.device() == local_out.device(),
                "global_out and local_out must be on the same device");
    TORCH_CHECK(global_out.device() == global_gate_logits.device(),
                "global_out and global_gate_logits must be on the same device");
    TORCH_CHECK(global_out.device() == local_gate_logits.device(),
                "global_out and local_gate_logits must be on the same device");

    TORCH_CHECK(global_out.dim() == 4, "global_out must be 4D, but got ", global_out.dim(), "D");
    TORCH_CHECK(local_out.dim() == 4, "local_out must be 4D, but got ", local_out.dim(), "D");
    TORCH_CHECK(global_out.size(0) == EXPECTED_S && global_out.size(1) == EXPECTED_B &&
                    global_out.size(2) == EXPECTED_N && global_out.size(3) == EXPECTED_D,
                "global_out must be [256, 4, 96, 256]");
    TORCH_CHECK(local_out.sizes() == global_out.sizes(),
                "local_out shape must match global_out shape");

    const int64_t gate_dim = global_gate_logits.dim();
    TORCH_CHECK(gate_dim == 3 || gate_dim == 4,
                "global_gate_logits must be 3D or 4D, but got ", gate_dim, "D");
    TORCH_CHECK(local_gate_logits.dim() == gate_dim,
                "local_gate_logits rank must match global_gate_logits rank");
    TORCH_CHECK(local_gate_logits.sizes() == global_gate_logits.sizes(),
                "local_gate_logits shape must match global_gate_logits shape");

    const auto input_dtype = global_out.scalar_type();
    TORCH_CHECK(input_dtype == at::kHalf || input_dtype == at::kBFloat16,
                "global_out dtype must be float16 or bfloat16, but got ", input_dtype);
    TORCH_CHECK(local_out.scalar_type() == input_dtype,
                "local_out dtype must match global_out dtype");
    TORCH_CHECK(global_gate_logits.scalar_type() == input_dtype,
                "global_gate_logits dtype must match global_out dtype");
    TORCH_CHECK(local_gate_logits.scalar_type() == input_dtype,
                "local_gate_logits dtype must match global_out dtype");

    TORCH_CHECK(scale_alg == 0 || scale_alg == 1, "scale_alg must be 0 or 1, but got ", scale_alg);
    aclDataType data_acl_type = GetAclDataTypeFromDstType(dst_type);
    at::ScalarType data_scalar_type = GetScalarTypeFromDstType(dst_type);

    const int64_t m = EXPECTED_S * EXPECTED_B;
    const int64_t k = EXPECTED_N * EXPECTED_D;

    at::Tensor row_data = at::empty({m, k}, global_out.options().dtype(data_scalar_type));
    at::Tensor col_data = at::empty({m, k}, global_out.options().dtype(data_scalar_type));
    at::Tensor row_scale = at::empty({m, k / SPLIT_BLOCK_SIZE, 2}, global_out.options().dtype(at::kByte));
    at::Tensor col_scale = at::empty({m / SPLIT_BLOCK_SIZE, k, 2}, global_out.options().dtype(at::kByte));

    TensorWrapper row_data_wrapper = {row_data, data_acl_type};
    TensorWrapper col_data_wrapper = {col_data, data_acl_type};
    TensorWrapper row_scale_wrapper = {row_scale, ACL_FLOAT8_E8M0};
    TensorWrapper col_scale_wrapper = {col_scale, ACL_FLOAT8_E8M0};

    // 四输入版本 aclnn 封装。若实际 aclnn 头文件参数顺序不同，以头文件为准。
    ACLNN_CMD(aclnnClaGateMxQuantWithDualAxis,
              global_out, local_out, global_gate_logits, local_gate_logits,
              scale_alg, dst_type,
              row_data_wrapper, row_scale_wrapper, col_data_wrapper, col_scale_wrapper);

    return std::make_tuple(std::move(row_data), std::move(row_scale),
                           std::move(col_data), std::move(col_scale));
}

} // namespace quant
} // namespace cann_ops_nn

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("cla_gate_mx_quant_with_dual_axis",
          &cann_ops_nn::quant::cla_gate_mx_quant_with_dual_axis,
          "ClaGateMxQuantWithDualAxis operator on NPU");
}
