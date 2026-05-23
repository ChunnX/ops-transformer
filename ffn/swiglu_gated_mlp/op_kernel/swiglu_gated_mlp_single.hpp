/**
 * Copyright (c) 2026 联通（广东）产业互联网有限公司.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file swiglu_gated_mlp_single.hpp
 * \brief
 */

#ifndef OP_KERNEL_SWIGLU_GATED_MLP_SINGLE_HPP_
#define OP_KERNEL_SWIGLU_GATED_MLP_SINGLE_HPP_

#include "kernel_operator.h"
#include "glu_tiling.hpp"
#include "swiglu_gated_mlp_impl.hpp"

using namespace AscendC;

namespace swiglu_gated_mlp_kernel {

template <class KernelImpl, typename InT, typename OutT>
class SwigluGatedMlpSingle {
public:
    __aicore__ SwigluGatedMlpSingle() = default;
    __aicore__ ~SwigluGatedMlpSingle() = default;

    __aicore__ void Init(
        GM_ADDR xGm,
        GM_ADDR gateUpWeightGm,
        GM_ADDR downWeightGm,
        GM_ADDR yGm,
        GM_ADDR workspaceGm,
        GM_ADDR tilingGm)
    {
        GET_TILING_DATA(localTiling, tilingGm);
        Init(xGm, gateUpWeightGm, downWeightGm, yGm, workspaceGm, localTiling);
    }

    __aicore__ void Init(
        GM_ADDR xGm,
        GM_ADDR gateUpWeightGm,
        GM_ADDR downWeightGm,
        GM_ADDR yGm,
        GM_ADDR workspaceGm,
        const SwigluGatedMlpTilingData &tilingData)
    {
        tilingData_ = tilingData;
        kernel_.Init(xGm, gateUpWeightGm, downWeightGm, yGm, workspaceGm, tilingData_);
    }

    __aicore__ void Process()
    {
        if (tilingData_.dtypeTag == WG_MLP_EMPTY_TILING_KEY) {
            return;
        }
        kernel_.Process();
    }

private:
    SwigluGatedMlpTilingData tilingData_ {};
    KernelImpl kernel_ {};
};

}  // namespace swiglu_gated_mlp_kernel

#endif  // OP_KERNEL_SWIGLU_GATED_MLP_SINGLE_HPP_
