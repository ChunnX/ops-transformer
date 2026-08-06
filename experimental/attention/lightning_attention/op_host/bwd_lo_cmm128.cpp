/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bwd_lo_cmm128.h"

#include "common/lo_cmm128_tiling_data.h"
#include "tiling/bwd_lo_cmm128_tiling.h"

#include "op_kernel/lightning_attn_cmm128_bwd.h"

#include <ATen/Operators.h>
#include <acl/acl.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/core/npu/DeviceUtils.h>
#include <torch_npu/csrc/framework/OpCommand.h>

using namespace LightningAttn::Common;
using namespace LightningAttn::Tiling;

#define LOG_PRINT(message, ...) \
  do { \
    char buffer[512]; \
    snprintf(buffer, sizeof(buffer), message __VA_OPT__(,) __VA_ARGS__); \
    std::cout << buffer << "\n"; \
    throw std::invalid_argument(buffer); \
  } while (0)
namespace LightningAttn::Wrapper::Backward::lo_cmm128
{

constexpr uint8_t CUBE_SIZE = 16;
constexpr uint8_t TENSOR_DIMS = 4;

std::string arrayRefToString(const c10::ArrayRef<long>& array) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < array.size(); ++i) {
        if (i != 0) oss << ", ";
        oss << array[i];
    }
    oss << "]";
    return oss.str();
}

void verify_tensor(const at::Tensor& tensor, const std::string& name, bool check_shape) {
    if (!tensor.defined()) {
        LOG_PRINT("Error: %s must be a 4D tensor (batch, seq, heads, hidden)", name.c_str());
    }
    if (tensor.data_ptr() == nullptr) {
        LOG_PRINT("Error: No memory allocated for %s", name.c_str());
    }
    if (check_shape && tensor.dim() != TENSOR_DIMS) {
        LOG_PRINT("Error: %s must be %uD tensor, but got %uD", name.c_str(), TENSOR_DIMS, tensor.dim());
    }
    if (tensor.numel() == 0) {
        LOG_PRINT("Error: %s is an empty tensor", name.c_str());
    }
}



std::tuple<at::Tensor, at::Tensor, at::Tensor> run(const at::Tensor& gradResult,
                                                   const at::Tensor& query,
                                                   const at::Tensor& key,
                                                   const at::Tensor& value,
                                                   const at::Tensor& slopeRate,
                                                   const at::Tensor& kvState)
{
    verify_tensor(gradResult, "gradResult", true);
    verify_tensor(query, "query", true);
    verify_tensor(key, "key", true);
    verify_tensor(value, "value", true);
    verify_tensor(slopeRate, "slopeRate", false);

    auto dtype = query.scalar_type();
    auto device = query.options().device();

    auto b = query.size(0), n = query.size(1), s = query.size(2), d = query.size(3);

    if (d % CUBE_SIZE != 0) {
        LOG_PRINT("Error: dimension is not divisible by %u, please check it", CUBE_SIZE);
    }

    bool isTraining = kvState.numel() > 0 ? true : false;

    if (dtype != key.scalar_type() || dtype != value.scalar_type()) {
        LOG_PRINT("Error: data types of Q, K, V are different");
    }

    auto q_sizes = query.sizes();
    if (q_sizes != key.sizes()) {
        LOG_PRINT("Error: size of Q and K are different (%s)", arrayRefToString(q_sizes).c_str());
    }
    if (q_sizes != value.sizes()) {
        LOG_PRINT("Error: size of Q and V are different (%s)", arrayRefToString(q_sizes).c_str());
    }

    at::Tensor gradQueryOut = at::empty_like(gradResult, at::dtype(at::kFloat));
    at::Tensor gradKeyOut = at::empty_like(gradResult, at::dtype(at::kFloat));
    at::Tensor gradValueOut = at::empty_like(gradResult, at::dtype(at::kFloat));

    BwdLoCmm128Tiling bwdLoCmm128Tiling(b, s, n, d, isTraining);
    if (!bwdLoCmm128Tiling.DoTiling(sizeof(uint16_t)))
    {
        LOG_PRINT("Error: Tiling failed");
    }

    LoCmm128TilingData* tilingData = bwdLoCmm128Tiling.GetTilingData();
    uint32_t blockDim = tilingData->coreNum;

    auto wsSize = static_cast<int64_t>(bwdLoCmm128Tiling.GetWorkspaceSize());
    auto workspace = at::empty({wsSize}, at::TensorOptions().dtype(at::kByte).device(device));

    int devIdx = query.device().index();
    c10_npu::NPUStream stream = c10_npu::getCurrentNPUStream(devIdx);
    void* aclstream = stream.stream(false);

    auto kernel =
        [blockDim, aclstream, gradResult, query, key, value, slopeRate, kvState, gradQueryOut, gradKeyOut, gradValueOut, workspace, td=*tilingData]()
        -> int
    {
        la_cmm128_backward<<<blockDim, nullptr, aclstream>>>(
            reinterpret_cast<uint8_t*>(const_cast<void*>(gradResult.storage().data())),
            reinterpret_cast<uint8_t*>(const_cast<void*>(query.storage().data())),
            reinterpret_cast<uint8_t*>(const_cast<void*>(key.storage().data())),
            reinterpret_cast<uint8_t*>(const_cast<void*>(value.storage().data())),
            reinterpret_cast<uint8_t*>(const_cast<void*>(slopeRate.storage().data())),
            reinterpret_cast<uint8_t*>(const_cast<void*>(kvState.storage().data())),
            reinterpret_cast<uint8_t*>(const_cast<void*>(gradQueryOut.storage().data())),
            reinterpret_cast<uint8_t*>(const_cast<void*>(gradKeyOut.storage().data())),
            reinterpret_cast<uint8_t*>(const_cast<void*>(gradValueOut.storage().data())),
            reinterpret_cast<uint8_t*>(const_cast<void*>(workspace.storage().data())),
            td
        );
        return 0;
    };

    at_npu::native::OpCommand::RunOpApiV2("la_cmm128_backward", kernel);
    return std::make_tuple(gradQueryOut, gradKeyOut, gradValueOut);
}

} // namespace LightningAttn::Wrapper::Backward::lo_cmm128
