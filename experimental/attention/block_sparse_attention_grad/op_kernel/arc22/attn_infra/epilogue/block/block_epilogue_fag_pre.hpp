/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file block_epliogue_fag_pre.h
 * \brief Block Epliogue Fag Pre Kernel Implementation
 */

#ifndef CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_FAG_PRE_HPP
#define CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_FAG_PRE_HPP

#include "../../../attn_infra/arch/bsag_resource.hpp"
#include "../../../attn_infra/epilogue/bsag_epilogue_dispatch_policy.hpp"
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"

using namespace AscendC;
namespace NpuArch::Epilogue::Block {

template <class OutputType_, class UpdateType_, class InputType_>
class BlockEpilogue<EpilogueAtlasA2FAGPre, OutputType_, UpdateType_, InputType_> {
public:
    using DispatchPolicy = EpilogueAtlasA2FAGPre;
    using ArchTag = typename DispatchPolicy::ArchTag;

    struct Params {
        GM_ADDR dqWrk;
        GM_ADDR dkWrk;
        GM_ADDR dvWrk;
        GM_ADDR tilingData;

        __aicore__ inline Params() {}

        __aicore__ inline Params(GM_ADDR dqWrk_, GM_ADDR dkWrk_, GM_ADDR dvWrk_, GM_ADDR tilingData_)
            : dqWrk(dqWrk_),
              dkWrk(dkWrk_),
              dvWrk(dvWrk_),
              tilingData(tilingData_)
        {}
    };

    GlobalTensor<float> gradWorkSpaceGm;
    uint64_t cBlockIdx;

    uint64_t gradPreBlockFactor = 0;
    uint64_t gradPreBlockTotal = 0;
    uint64_t gradPreTail = 0;
    uint64_t initGradSize = 0;
    uint64_t gradOffset = 0;

    uint64_t usedCoreNum = 0;
    __aicore__ inline BlockEpilogue(Params const &params)
    {
        // KERNEL_TYPE_MIX_AIC_1_2 exposes two contiguous AIV block ids for
        // every AIC.  Use the physical AIV id so all vector cores clear a
        // unique slice of the accumulation workspace.
        cBlockIdx = GetBlockIdx();
        __gm__ BlockSparseAttentionGradTilingData *tilingData =
            reinterpret_cast<__gm__ BlockSparseAttentionGradTilingData *>(params.tilingData);
        usedCoreNum = tilingData->usedVecCoreNum;

        // Clear all accumulation planes in-kernel.  Sparse rows with very
        // small user Q blocks are not guaranteed to execute an overwriting
        // dQ tile, so treating grouped dQ as fully overwritten leaves holes
        // containing uninitialized workspace data.
        const uint64_t gradSize = tilingData->dqSize + tilingData->dkvSize * 2;
        gradPreBlockFactor = (gradSize + usedCoreNum - 1) / usedCoreNum;
        gradPreBlockTotal = (gradSize + gradPreBlockFactor - 1) / gradPreBlockFactor;
        const uint64_t gradPreTailTmp = gradSize % gradPreBlockFactor;
        gradPreTail = gradPreTailTmp == 0 ? gradPreBlockFactor : gradPreTailTmp;

        gradWorkSpaceGm.SetGlobalBuffer((__gm__ float *)params.dqWrk);
        initGradSize = cBlockIdx == gradPreBlockTotal - 1 ? gradPreTail : gradPreBlockFactor;
        gradOffset = cBlockIdx * gradPreBlockFactor;
    }

    __aicore__ inline ~BlockEpilogue() {}

    template <int32_t CORE_TYPE = g_coreType>
    __aicore__ inline void operator()();

    template <>
    __aicore__ inline void operator()<AscendC::AIC>()
    {}

    template <>
    __aicore__ inline void operator()<AscendC::AIV>()
    {
        if (cBlockIdx >= usedCoreNum) {
            return;
        }

        // Match FlashAttentionScoreGradPre: AIVs initialize only the FP32
        // accumulation buffers, then the mixed-kernel entry executes a
        // SyncAll on both AIC and AIV before any accumulation starts.
        if (cBlockIdx < gradPreBlockTotal) {
            InitOutput<float>(gradWorkSpaceGm[gradOffset], initGradSize, 0);
        }
    }
};

} // namespace NpuArch::Epilogue::Block

#endif // CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_FAG_PRE_HPP
