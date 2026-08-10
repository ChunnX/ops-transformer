// -----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------

/*!
 * \file moe_init_routing.cpp
 * \brief ACLNN Wrapper for aclnnMoeInitRoutingV2
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {

constexpr int64_t DROP_PAD_MODE_DROPLESS = 0;
constexpr int64_t DROP_PAD_MODE_DROP_PAD = 1;
constexpr int64_t EXPERT_TOKENS_FLAG_NONE = 0;
constexpr int64_t EXPERT_TOKENS_FLAG_CUMSUM = 1;
constexpr int64_t EXPERT_TOKENS_FLAG_COUNT = 2;

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> moe_init_routing(
    const at::Tensor &x, const at::Tensor &expertIdx, int64_t activeNum, int64_t expertCapacity,
    int64_t expertNum, int64_t dropPadMode, int64_t expertTokensCountOrCumsumFlag,
    bool expertTokensBeforeCapacityFlag)
{
    TORCH_CHECK(x.dim() == 2, "x should be 2D, but got ", x.dim(), "-D.");
    TORCH_CHECK(
        x.scalar_type() == at::kHalf || x.scalar_type() == at::kFloat || x.scalar_type() == at::kBFloat16,
        "x dtype should be float16, float32 or bfloat16, but got ", x.scalar_type(), ".");
    TORCH_CHECK(expertIdx.dim() == 2, "expert_idx should be 2D, but got ", expertIdx.dim(), "-D.");
    TORCH_CHECK(expertIdx.scalar_type() == at::kInt || expertIdx.scalar_type() == at::kLong,
                "expert_idx dtype should be int32 or int64, but got ", expertIdx.scalar_type(), ".");
    TORCH_CHECK(x.size(0) == expertIdx.size(0),
                "x and expert_idx should have the same first dim, but got ",
                x.size(0), " and ", expertIdx.size(0), ".");
    TORCH_CHECK(dropPadMode == DROP_PAD_MODE_DROPLESS || dropPadMode == DROP_PAD_MODE_DROP_PAD,
                "drop_pad_mode must be 0 or 1, but got ", dropPadMode, ".");
    TORCH_CHECK(activeNum >= 0, "active_num must be non-negative, but got ", activeNum, ".");
    TORCH_CHECK(expertTokensCountOrCumsumFlag == EXPERT_TOKENS_FLAG_NONE ||
                    expertTokensCountOrCumsumFlag == EXPERT_TOKENS_FLAG_CUMSUM ||
                    expertTokensCountOrCumsumFlag == EXPERT_TOKENS_FLAG_COUNT,
                "expert_tokens_count_or_cumsum_flag must be 0, 1 or 2, but got ",
                expertTokensCountOrCumsumFlag, ".");
    if (expertTokensCountOrCumsumFlag > 0) {
        TORCH_CHECK(expertNum > 0,
                    "expert_num must be positive when expert_tokens_count_or_cumsum_flag > 0, but got ",
                    expertNum, ".");
    }
    if (dropPadMode == DROP_PAD_MODE_DROP_PAD) {
        TORCH_CHECK(expertNum > 0, "expert_num must be positive when drop_pad_mode=1, but got ", expertNum, ".");
        TORCH_CHECK(expertCapacity > 0,
                    "expert_capacity must be positive when drop_pad_mode=1, but got ", expertCapacity, ".");
    }

    int64_t n = x.size(0);
    int64_t cols = x.size(1);
    int64_t k = expertIdx.size(1);
    int64_t outActiveNum = (activeNum > 0) ? std::min(n * k, activeNum) : n * k;
    int64_t expandedRowIdxNum = n * k;

    at::Tensor expandedXOut;
    at::Tensor expandedRowIdxOut;
    at::Tensor expertTokensCountOrCumsumOut;
    at::Tensor expertTokensBeforeCapacityOut;
    c10::optional<at::Tensor> tokensCountOpt;
    c10::optional<at::Tensor> tokensBeforeCapacityOpt;
    {
        auto localDevice = c10::Device(x.device());
        const c10::OptionalDeviceGuard deviceGuard(localDevice);

        if (dropPadMode == DROP_PAD_MODE_DROP_PAD) {
            expandedXOut = at::empty({expertNum, expertCapacity, cols}, x.options());
            if (expertTokensBeforeCapacityFlag) {
                expertTokensBeforeCapacityOut = at::empty({expertNum}, x.options().dtype(at::kInt));
                tokensBeforeCapacityOpt = expertTokensBeforeCapacityOut;
            } else {
                expertTokensBeforeCapacityOut = at::empty({0}, x.options().dtype(at::kInt));
            }
        } else {
            expandedXOut = at::empty({outActiveNum, cols}, x.options());
            if (expertTokensCountOrCumsumFlag > 0) {
                expertTokensCountOrCumsumOut = at::empty({expertNum}, x.options().dtype(at::kInt));
                tokensCountOpt = expertTokensCountOrCumsumOut;
            } else {
                expertTokensCountOrCumsumOut = at::empty({0}, x.options().dtype(at::kInt));
            }
        }
        expandedRowIdxOut = at::empty({expandedRowIdxNum}, x.options().dtype(at::kInt));
    }

    ACLNN_CMD(aclnnMoeInitRoutingV2, x, expertIdx, activeNum, expertCapacity, expertNum, dropPadMode,
              expertTokensCountOrCumsumFlag, expertTokensBeforeCapacityFlag, expandedXOut, expandedRowIdxOut,
              tokensCountOpt, tokensBeforeCapacityOpt);

    if (!expertTokensCountOrCumsumOut.defined()) {
        expertTokensCountOrCumsumOut = at::empty({0}, x.options().dtype(at::kInt));
    }
    if (!expertTokensBeforeCapacityOut.defined()) {
        expertTokensBeforeCapacityOut = at::empty({0}, x.options().dtype(at::kInt));
    }

    return std::make_tuple(std::move(expandedXOut), std::move(expandedRowIdxOut),
                           std::move(expertTokensCountOrCumsumOut), std::move(expertTokensBeforeCapacityOut));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("moe_init_routing", &moe_init_routing, "moe_init_routing");
}

} // namespace op_api
