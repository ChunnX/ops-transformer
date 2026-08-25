/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON‑INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_flash_attention_score_grad_s1s2_bn2gs1s2_post_rebase.h
 * \brief
 */

#ifndef QUANT_FLASH_ATTENTION_SCORE_GRAD_S1S2_BNGS1S2_POST_KERNEL_REGBASE_H_
#define QUANT_FLASH_ATTENTION_SCORE_GRAD_S1S2_BNGS1S2_POST_KERNEL_REGBASE_H_
#include "vector_api/vf_post_reduce_sink.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#define QUANT_FAG_POST_CLASS_TEMPLATE                                                                                  \
template <typename T1, typename T2, typename OUTDTYPE=T1, const uint8_t SPLIT_AXIS = 0, const bool IS_ROPE = false,   \
const uint8_t DETER_SPARSE_TYPE = 0, const bool IS_TND = 0, const bool IS_TND_SWIZZLE = 0>

#define QUANT_FAG_POST_FUNCTION_TEMPLATE                                                                               \
template <typename T1, typename T2, typename OUTDTYPE, const uint8_t SPLIT_AXIS, bool IS_ROPE,                          \
const uint8_t DETER_SPARSE_TYPE, const bool IS_TND, const bool IS_TND_SWIZZLE>

#define QUANT_FAG_POST_FUNCTION_PARAMS_TEMPLATE T1, T2, OUTDTYPE, SPLIT_AXIS, IS_ROPE, DETER_SPARSE_TYPE, IS_TND, IS_TND_SWIZZLE

QUANT_FAG_POST_CLASS_TEMPLATE
class QuantFlashAttentionScoreGradS1S2BNGS1S2PostRegbase {
public:
    __aicore__ inline QuantFlashAttentionScoreGradS1S2BNGS1S2PostRegbase(){};
    __aicore__ inline void Init(__gm__ uint8_t *dq, __gm__ uint8_t *dk, __gm__ uint8_t *dv, __gm__ uint8_t *dsink,
        __gm__ uint8_t *workspace, QuantFagTilingType ordTilingData, TPipe *pipe_in);
    __aicore__ inline void Process();
    __aicore__ inline void ProcessDqkv();
    __aicore__ inline void ProcessBNS2Deter();

    constexpr static uint32_t ADDR_ALIGN_SIZE = 512;
    uint32_t REGBASE_POST_BASE = 128 * 128;
    uint32_t VALUE_DIM = 128;
    uint32_t POST_S_BASE = 96;
    TPipe *pipe;
    QuantFagTilingType tilingData;
    TQue<QuePosition::VECIN, 1> inQueuePing;
    TQue<QuePosition::VECOUT, 1> outQueuePing;
    TQue<QuePosition::VECIN, 1> inQueuePong;
    TQue<QuePosition::VECOUT, 1> outQueuePong;
    GlobalTensor<OUTDTYPE> dqkv[3];
    GlobalTensor<float> dqkvWorkspace[3];
    GlobalTensor<float> deterGm[2];
    uint32_t vBlockIdx;
    uint64_t loop;
    uint64_t inputTotalSize;
    uint64_t qPostTailNum;
};

QUANT_FAG_POST_FUNCTION_TEMPLATE
__aicore__ inline void QuantFlashAttentionScoreGradS1S2BNGS1S2PostRegbase<QUANT_FAG_POST_FUNCTION_PARAMS_TEMPLATE>::Init(
    __gm__ uint8_t *dq, __gm__ uint8_t *dk, __gm__ uint8_t *dv, __gm__ uint8_t *dsink, __gm__ uint8_t *workspace,
    QuantFagTilingType ordTilingData, TPipe *pipe_in)
{
    vBlockIdx = GetBlockIdx();
    tilingData = ordTilingData;
    pipe = pipe_in;

    dqkv[0].SetGlobalBuffer((__gm__ OUTDTYPE *)dq);
    dqkv[1].SetGlobalBuffer((__gm__ OUTDTYPE *)dk);
    dqkv[2].SetGlobalBuffer((__gm__ OUTDTYPE *)dv);

    loop = tilingData->q_post_block_factor;
    inputTotalSize = tilingData->q_post_block_total;
    qPostTailNum = tilingData->q_post_tail_num;

    dqkvWorkspace[0].SetGlobalBuffer((__gm__ float *)workspace + tilingData->dq_work_space_offset / sizeof(T2));
    dqkvWorkspace[1].SetGlobalBuffer((__gm__ float *)workspace + tilingData->dk_work_space_offset / sizeof(T2));
    dqkvWorkspace[2].SetGlobalBuffer((__gm__ float *)workspace + tilingData->dv_work_space_offset / sizeof(T2));

    pipe->InitBuffer(inQueuePing, 1, REGBASE_POST_BASE * sizeof(T2));
    pipe->InitBuffer(outQueuePing, 1, REGBASE_POST_BASE * sizeof(OUTDTYPE));
    pipe->InitBuffer(inQueuePong, 1, REGBASE_POST_BASE * sizeof(T2));
    pipe->InitBuffer(outQueuePong, 1, REGBASE_POST_BASE * sizeof(OUTDTYPE));
}

QUANT_FAG_POST_FUNCTION_TEMPLATE
__aicore__ inline void QuantFlashAttentionScoreGradS1S2BNGS1S2PostRegbase<QUANT_FAG_POST_FUNCTION_PARAMS_TEMPLATE>::ProcessDqkv()
{
    for (int qkvIdx = 0; qkvIdx < 3; qkvIdx++){
        if (qkvIdx == 1) {
            loop = tilingData->k_post_block_factor;
            inputTotalSize = tilingData->k_post_block_total;
            qPostTailNum = tilingData->k_post_tail_num;
        } else if (qkvIdx == 2) {
            loop = tilingData->v_post_block_factor;
            inputTotalSize = tilingData->v_post_block_total;
            qPostTailNum = tilingData->v_post_tail_num;
        }
        uint64_t blockCore = loop * REGBASE_POST_BASE;
        uint64_t begin = vBlockIdx * blockCore;
        uint64_t end = begin + blockCore;

        if (end > inputTotalSize) {
            end = inputTotalSize;
        }

        uint64_t dqPingGmOffset = vBlockIdx * loop * POST_S_BASE * VALUE_DIM;
        for (uint64_t pingIdx = begin; pingIdx < end; pingIdx = pingIdx + (REGBASE_POST_BASE << 1)) {
            LocalTensor<float> vecInPing = inQueuePing.AllocTensor<float>();
            uint64_t pongIdx = pingIdx + REGBASE_POST_BASE;
            uint64_t pingSize = pongIdx < inputTotalSize ? REGBASE_POST_BASE : qPostTailNum;
            uint64_t dqPongGmOffset = dqPingGmOffset + VALUE_DIM*POST_S_BASE; //96*128
            DataCopy(vecInPing, dqkvWorkspace[qkvIdx][pingIdx], (pingSize + 7) >> 3 << 3);
            inQueuePing.EnQue(vecInPing);
            inQueuePing.DeQue<float>();
            if (qkvIdx < 2) {
                Muls(vecInPing, vecInPing, (float)tilingData->softmax_scale, pingSize);
            }
            LocalTensor<OUTDTYPE> vecOutPing = outQueuePing.AllocTensor<OUTDTYPE>();
            Cast(vecOutPing, vecInPing, RoundMode::CAST_ROUND, pingSize);
            uint64_t pongSize;
            LocalTensor<float> vecInPong;
            bool neeedPong = loop > 1 && pongIdx < end;
            if (neeedPong) {
                vecInPong = inQueuePong.AllocTensor<float>();
                pongSize = pongIdx + REGBASE_POST_BASE < inputTotalSize ? REGBASE_POST_BASE : qPostTailNum;
                DataCopy(vecInPong, dqkvWorkspace[qkvIdx][pongIdx], (pongSize + 7) >> 3 << 3);
            }
            outQueuePing.EnQue(vecOutPing);
            outQueuePing.DeQue<OUTDTYPE>();
            inQueuePing.FreeTensor(vecInPing);
            DataCopy(dqkv[qkvIdx][pingIdx], vecOutPing, (pingSize + 15) >> 4 << 4);
            outQueuePing.FreeTensor(vecOutPing);
            LocalTensor<OUTDTYPE> vecOutPong;
            if (neeedPong) {
                vecOutPong = outQueuePong.AllocTensor<OUTDTYPE>();
                inQueuePong.EnQue(vecInPong);
                inQueuePong.DeQue<float>();
                if (qkvIdx < 2) {
                    Muls(vecInPong, vecInPong, (float)tilingData->softmax_scale, pongSize);
                }
                Cast(vecOutPong, vecInPong, RoundMode::CAST_ROUND, pongSize);
                outQueuePong.EnQue(vecOutPong);
                outQueuePong.DeQue<OUTDTYPE>();
                inQueuePong.FreeTensor(vecInPong);
                DataCopy(dqkv[qkvIdx][pongIdx], vecOutPong, (pongSize + 15) >> 4 << 4);
                outQueuePong.FreeTensor(vecOutPong);
            }
        }

        // bn2s2 dk dv no need to do muls and cast
        if constexpr (SPLIT_AXIS == 5) {
            break;
        }
    }
}

QUANT_FAG_POST_FUNCTION_TEMPLATE
__aicore__ inline void QuantFlashAttentionScoreGradS1S2BNGS1S2PostRegbase<QUANT_FAG_POST_FUNCTION_PARAMS_TEMPLATE>::Process()
{
    if (g_coreType != AIV) {
        return;
    }
    if constexpr (SPLIT_AXIS != BN2 && !IsSameType<T1, float>::value) {
        ProcessDqkv();
    }
}

#endif // _QUANT_FLASH_ATTENTION_SCORE_GRAD_S1S2_BNGS1S2_POST_KERNEL_REBASE_H_