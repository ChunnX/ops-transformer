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
 * \file glu_tiling_kernel.hpp
 * \brief
 */

#ifndef OP_KERNEL_GLU_TILING_KERNEL_HPP_
#define OP_KERNEL_GLU_TILING_KERNEL_HPP_

#include "glu_tiling.hpp"

using namespace AscendC;

namespace swiglu_gated_mlp_kernel {

struct GluKernelTilingCtx {
    __aicore__ GluKernelTilingCtx() = default;

    __aicore__ explicit GluKernelTilingCtx(const SwigluGatedMlpTilingData &tilingData)
        : info(tilingData)
    {
        Init();
    }

    __aicore__ void Reset(const SwigluGatedMlpTilingData &tilingData)
    {
        info.Parse(tilingData);
        Init();
    }

    __aicore__ void Init()
    {
        blockIdx = static_cast<uint32_t>(GetBlockIdx());
        blockNum = static_cast<uint32_t>(GetBlockNum());

        validCore = (blockIdx < info.usedCoreNum) && !info.IsEmpty();
        rowStart = info.GetRowStartForCore(blockIdx);
        rowsThisCore = info.GetRowsForCore(blockIdx);
        rowEnd = rowStart + static_cast<uint64_t>(rowsThisCore);
    }

    __aicore__ bool IsValidCore() const
    {
        return validCore && rowsThisCore > 0;
    }

    __aicore__ uint64_t GetXOffset() const
    {
        return rowStart * info.hiddenSize;
    }

    __aicore__ uint64_t GetYOffset() const
    {
        return rowStart * info.outSize;
    }

    GluTilingInfo info {};
    uint32_t blockIdx = 0;
    uint32_t blockNum = 0;
    bool validCore = false;
    uint64_t rowStart = 0;
    uint32_t rowsThisCore = 0;
    uint64_t rowEnd = 0;
};

}  // namespace swiglu_gated_mlp_kernel

#endif  // OP_KERNEL_GLU_TILING_KERNEL_HPP_
