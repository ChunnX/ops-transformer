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
 * \file combine_send.h
 * \brief Single-core combine-send wrappers. All multi-core primitives (GetBlockIdx,
 *        GetBlockNum, GetSubBlockIdx, blockIdx_, aivCoreIdx_, subBlockIdx_,
 *        blockAivNum_, blockNum_) are replaced by TileInfo passed explicitly.
 */

#ifndef MEGA_MOE_SINGLE_CORE_COMBINE_SEND_H
#define MEGA_MOE_SINGLE_CORE_COMBINE_SEND_H

#include "mega_moe_tile_info.h"
#include "mega_moe_base.h"
#include "mega_moe_workspace_info.h"
#include "mega_moe_combine_send.h"

namespace CombineSend {

template <typename ElementMMadOut2, typename BlockShape>
__aicore__ inline void CombineTokens(
    uint32_t mLoc, uint32_t nLoc, uint32_t n,
    LocalTensor<int32_t>& tripleTensor,
    LocalTensor<ElementMMadOut2>& l0cOutUbGMM2,
    BlockShape& actualBlockShape, const Params& params,
    const MegaMoeModules::TileInfo& tile)
{
    MegaMoeCombineImpl::CombineTokens<ElementMMadOut2>(
        mLoc, nLoc, n, tripleTensor, l0cOutUbGMM2, actualBlockShape, params);
}

template <typename ElementMMadOut2, typename BlockShape>
__aicore__ inline void CombineTokensLayered(
    uint32_t mLoc, uint32_t nLoc, uint32_t n,
    LocalTensor<int32_t>& tripleTensor,
    LocalTensor<ElementMMadOut2>& l0cOutUbGMM2,
    BlockShape& actualBlockShape, const Params& params,
    const MegaMoeModules::TileInfo& tile)
{
    MegaMoeCombineImpl::CombineTokensLayered<ElementMMadOut2>(
        mLoc, nLoc, n, tripleTensor, l0cOutUbGMM2, actualBlockShape, params);
}

// NOTE: MegaMoeCombineImpl::CombineSendTokenToRemote internally calls GetBlockIdx()
// and GetBlockNum() to partition rank responsibilities across cores. This violates
// the single-core model. For now this wrapper delegates directly; a future refactor
// must replace those internals with TileInfo-based rank partitioning.
template <typename ElementMMadOut2>
__aicore__ inline void CombineSendTokenToRemote(
    const Params& params, const MegaMoeModules::TileInfo& tile)
{
    MegaMoeCombineImpl::CombineSendTokenToRemote<ElementMMadOut2>(params);
}

template <uint8_t QuantMode, typename ElementMMadOut2>
__aicore__ inline void ProcessCombineWrapper(
    uint32_t tokenStart, uint32_t tokenCount, uint32_t n,
    uint32_t groupIdx, uint32_t rankId,
    GM_ADDR gmm2OutAddr, const Params& params,
    LocalTensor<int32_t>& tripleTensor,
    int64_t ubTensorSize, int64_t offset,
    uint32_t quantTokenSizeBytes,
    const MegaMoeModules::TileInfo& tile)
{
    MegaMoeCombineImpl::CombineTokenGroup<QuantMode, ElementMMadOut2>(
        tokenStart, tokenCount, n, groupIdx, rankId,
        gmm2OutAddr, params, tripleTensor,
        ubTensorSize, offset, quantTokenSizeBytes);
}

} // namespace CombineSend

#endif // MEGA_MOE_SINGLE_CORE_COMBINE_SEND_H
