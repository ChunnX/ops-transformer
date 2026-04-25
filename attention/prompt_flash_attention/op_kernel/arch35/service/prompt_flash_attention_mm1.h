/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file prompt_flash_attention_mm1.h
 * \brief
 */
#ifndef PROMPT_FLASH_ATTENTION_MM1_H
#define PROMPT_FLASH_ATTENTION_MM1_H

using namespace matmul;

template <typename PFAT, typename mmType>
class PromptFlashAttentionNormalMM1 {
public:
    using T = typename PFAT::inputType;
    using KV_T = typename PFAT::kvInputType;
    using computeType = typename PromptFlashAttentionTypeTraits<T, PFAT::calcMode>::softmaxType;
    using mmOutputType = typename PromptFlashAttentionTypeTraits<T,PFAT::calcMode>::mmOutputType;
    __aicore__ inline PromptFlashAttentionNormalMM1() {};
    __aicore__ inline void WaitIterateAll();
    // 全量化、非量化、PA场景
    __aicore__ inline void IterateAll(LocalTensor<mmOutputType>& mmResUb, GlobalTensor <T>& queryGm, GlobalTensor<KV_T>& keyGm,
        GlobalTensor<KV_T>& queryRopeGm, GlobalTensor<KV_T>& keyRopeGm, __gm__ uint8_t* blocktablePtr,
        const TaskParam& taskParam, const ConstParam& constParam);
    // 伪量化场景
    __aicore__ inline void IterateAll(LocalTensor<mmOutputType>& mmResUb,
        TSCM<QuePosition::VECIN, 1, 0x4>& queryScmQueue, TSCM<QuePosition::VECIN, 1, 0x4>& keyScmQueue,
        const TaskParam& taskParam, const ConstParam& constParam, TQue<QuePosition::VECOUT, 1>& antiquantOutputQueue);
public:
    mmType mm;
private:
    LocalTensor<T> queryScmTensor;
};

template <typename PFAT, typename mmType>
__aicore__ inline void PromptFlashAttentionNormalMM1<PFAT, mmType>::WaitIterateAll()
{
    mm.WaitIterateAll();
}

template <typename PFAT, typename mmType>
__aicore__ inline void PromptFlashAttentionNormalMM1<PFAT, mmType>::IterateAll(
    LocalTensor<mmOutputType>& mmResUb, GlobalTensor <T>& queryGm, GlobalTensor<KV_T>& keyGm,
    GlobalTensor<KV_T>& queryRopeGm, GlobalTensor<KV_T>& keyRopeGm, __gm__ uint8_t* blocktablePtr,
    const TaskParam& taskParam, const ConstParam& constParam)
{
    // Matmul接口使用限制，传输自定义结构体需要放在最前面,否则matmul的消息传输不完善
    uint32_t cubeSOuterSizeAlign2 = ((taskParam.cubeSOuterSize + 1U) >> 1 << 1); // singleM要求2对齐
    uint32_t cubeSOuterSizeAlign32 = (taskParam.cubeSOuterSize + 31) >> 5 << 5;
    if constexpr (PFAT::isSplitCoreByCube) {
        PFAMatmulPolicyData pData = {0};
        pData.reuseLeft = taskParam.isFirstInnerIter ? 0 : 1; // inner循环第一次不复用Q矩阵
        pData.leftBufIdx = PFAT::useDN ? (2 + taskParam.taskPingPong) : taskParam.splitPingPong; // 0,1:query pingpong buf idx
        pData.rightBufIdx = PFAT::useDN ? taskParam.splitPingPong : (2 + taskParam.taskPingPong); // 2,3:key pingpong buf idx
        pData.mOrNAdditionalSize = PFAT::useDN ? (cubeSOuterSizeAlign32 - taskParam.cubeSOuterSize) :
            (cubeSOuterSizeAlign2 - taskParam.cubeSOuterSize); // 配合自管理，只搬运实际长度
        mm.SetSelfDefineData(pData);
    }
    if constexpr (PFAT::useDN) {
        mm.SetOrgShape(constParam.bmm1TilingDataRectN,
            constParam.bmm1TilingDataRectM,
            constParam.bmm1TilingDataRectKb,
            constParam.bmm1TilingDataRectKa,
            PFAT::sOuter);
        // n和d size传原始大小，不是对齐之后的大小
        mm.SetTail(taskParam.singleProcessSInnerBmmTail, cubeSOuterSizeAlign32, constParam.qkHeadSize);
        mm.SetTensorA(keyGm[taskParam.tensorBOffset]);
    } else {
        mm.SetOrgShape(constParam.bmm1TilingDataRectM,
            constParam.bmm1TilingDataRectN,
            constParam.bmm1TilingDataRectKa,
            constParam.bmm1TilingDataRectKb,
            PFAT::sInner);
        // n和d size传原始大小，不是对齐之后的大小
        mm.SetTail(cubeSOuterSizeAlign2, taskParam.singleProcessSInnerBmmTail, constParam.qkHeadSize);
        mm.SetTensorA(queryGm[taskParam.tensorAOffset]);
    }
    

    if constexpr (PFAT::useDN) {
        mm.SetTensorB(queryGm[taskParam.tensorAOffset], true);
    } else {
        mm.SetTensorB(keyGm[taskParam.tensorBOffset], true);
    }

    mm.template IterateAll<false>(mmResUb, false, false, true);
}

template <typename PFAT, typename mmType>
__aicore__ inline void PromptFlashAttentionNormalMM1<PFAT, mmType>::IterateAll(
    LocalTensor<mmOutputType>& mmResUb, TSCM<QuePosition::VECIN, 1, 0x4>& queryScmQueue,
    TSCM<QuePosition::VECIN, 1, 0x4>& keyScmQueue, const TaskParam& taskParam, const ConstParam& constParam, TQue<QuePosition::VECOUT, 1>& antiquantOutputQueue) {

    queryScmTensor = queryScmQueue.DeQue<T>();
    LocalTensor<T> keyScmTensor = keyScmQueue.DeQue<T>();
    mm.SetOrgShape(constParam.bmm1TilingDataRectM,
                   constParam.bmm1TilingDataRectN,
                   constParam.bmm1TilingDataRectKa,
                   constParam.bmm1TilingDataRectKb,
                   taskParam.mm1SingleCoreN);

    mm.SetTail(taskParam.cubeSOuterSize, taskParam.mm1SingleCoreN, constParam.headSize);

    mm.SetTensorA(queryScmTensor);
    mm.SetTensorB(keyScmTensor, true);

    mm.template IterateAll<false>(mmResUb, false, false, true);
    queryScmQueue.FreeTensor(queryScmTensor);
    keyScmQueue.FreeTensor(keyScmTensor);
}

#endif  // PROMPT_FLASH_ATTENTION_MM1_H