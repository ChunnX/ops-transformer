/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_token_unpermute_with_routing_map_grad_prob_none_drop_pad_false.h
 * \brief
 */
#ifndef MOE_TOKEN_UNPERMUTE_WITH_ROUTING_MAP_GRAD_PROB_NONE_DROP_PAD_FALSE_H
#define MOE_TOKEN_UNPERMUTE_WITH_ROUTING_MAP_GRAD_PROB_NONE_DROP_PAD_FALSE_H

#include "moe_token_unpermute_with_routing_map_grad_base.h"

namespace MoeTokenUnpermuteWithRoutingMapGrad {
using namespace AscendC;

template <typename OriT, typename IdxT>
class MoeTokenUnpermuteWithRoutingMapGradProbNoneDropPadFalse
    : protected MoeTokenUnpermuteWithRoutingMapGradBase<OriT, IdxT>
{
public:
    __aicore__ inline MoeTokenUnpermuteWithRoutingMapGradProbNoneDropPadFalse(){};
    __aicore__ inline void Init(
        GM_ADDR unpermuted_tokens_grad, GM_ADDR outIndex, GM_ADDR permuteTokenId, GM_ADDR routing_map,
        GM_ADDR permuted_tokens, GM_ADDR probs, GM_ADDR permuted_tokens_grad, GM_ADDR probs_grad,
        const MoeTokenUnpermuteWithRoutingMapGradTilingData& tiling_data);
    __aicore__ inline void Process();

protected:
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> inOutque;
    LocalTensor<OriT> inOutLocal;

    DataCopyExtParams copyParams{1, 0, 0, 0, 0};
};

template <typename OriT, typename IdxT>
__aicore__ inline void MoeTokenUnpermuteWithRoutingMapGradProbNoneDropPadFalse<OriT, IdxT>::Init(
    GM_ADDR unpermuted_tokens_grad, GM_ADDR outIndex, GM_ADDR permuteTokenId, GM_ADDR routing_map,
    GM_ADDR permuted_tokens, GM_ADDR probs, GM_ADDR permuted_tokens_grad, GM_ADDR probs_grad,
    const MoeTokenUnpermuteWithRoutingMapGradTilingData& tiling_data)
{
    MoeTokenUnpermuteWithRoutingMapGradBase<OriT, IdxT>::Init(
        unpermuted_tokens_grad, outIndex, permuteTokenId, routing_map, permuted_tokens, probs, permuted_tokens_grad,
        probs_grad, tiling_data);
    this->pipe.InitBuffer(
        inOutque, DOUBLE_BUFFER, (this->inputReserveNum * this->hiddenSizeAlign) * this->inputTypeSize);
}

template <typename OriT, typename IdxT>
__aicore__ inline void MoeTokenUnpermuteWithRoutingMapGradProbNoneDropPadFalse<OriT, IdxT>::Process()
{
    int64_t outNumCurrentCore = this->coreIndex < this->formerCoreNum ? this->rowIdMapEachCore : this->rowIdMapTailCore;
    int64_t inputBlockNum = BLOCK_SIZE_32 / this->inputTypeSize;
    for (int64_t indicesLoopTime = 0; indicesLoopTime < outNumCurrentCore;
         indicesLoopTime++) { // 外循环是根据x的数量循环
        int64_t rowIdMapLoopOffset = this->rowIdMapStartOffset + indicesLoopTime;
        int64_t tokenId = rowIdMapLoopOffset / this->topK;
        int64_t permuteTokenId = this->sortedTwiceIndexGm.GetValue(rowIdMapLoopOffset);
        SToMTE2Sync();
        for (int64_t hiddenLoop = 0; hiddenLoop < this->hiddenSizeLoopTimes; hiddenLoop++) {
            uint32_t hiddenLoopNum =
                hiddenLoop == this->hiddenSizeLoopTimes - 1 ? this->hiddenSizeTail : this->hiddenSizeAlign;
            uint32_t hiddenLoopBlockLen = hiddenLoopNum * this->inputTypeSize;
            int64_t hiddenLoopOffset = hiddenLoop * this->hiddenSizeAlign;
            inOutLocal = inOutque.AllocTensor<OriT>();
            copyParams.blockLen = hiddenLoopBlockLen;
            int64_t unpermutedTokensGradOffset = tokenId * this->hiddenSize + hiddenLoopOffset;
            DataCopyPad(
                inOutLocal, this->unpermutedTokensGradGm[unpermutedTokensGradOffset], copyParams, this->inputPadParams);
            inOutque.EnQue<QuePosition::VECIN, QuePosition::VECOUT, OriT>(inOutLocal);
            inOutLocal = inOutque.DeQue<QuePosition::VECIN, QuePosition::VECOUT, OriT>();
            int64_t permutedTokensGradOffset = permuteTokenId * this->hiddenSize + hiddenLoopOffset;
            DataCopyPad(this->permutedTokensGradGm[permutedTokensGradOffset], inOutLocal, copyParams);
            inOutque.FreeTensor(inOutLocal);
        }
    }
}

} // namespace MoeTokenUnpermuteWithRoutingMapGrad
#endif // MOE_TOKEN_UNPERMUTE_WITH_ROUTING_MAP_GRAD_PROB_NONE_DROP_PAD_FALSE_H