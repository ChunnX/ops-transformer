/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON‑INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_flash_attention_score_grad_presfmg_rebase.h
 * \brief
 */

#ifndef QUANT_FLASH_ATTENTION_SCORE_GRAD_PRESFMG_KERNEL_REGBASE_H_
#define QUANT_FLASH_ATTENTION_SCORE_GRAD_PRESFMG_KERNEL_REGBASE_H_
#include "math.h"
#include "kernel_basic_intf.h"
#include "vector_api/vf_anti_quant_softmax_grad_front_cast.h"
#include "vector_api/vf_softmax_grad_front_cast.h"
#include "quant_flash_attention_score_grad_common.h"

using namespace AscendC;

#define PRE_FUNCTION_TEMPLATE \
template <typename T1, typename T2, typename OUTDTYPE, DTemplateType dTemplateType, \
const uint8_t LAYOUT>

#define PRE_FUNCTION_ARGS_TEMPLATE \
T1, T2, OUTDTYPE, dTemplateType, LAYOUT

// T1 bf8 outdtypes bf16
template <typename T1, typename T2, typename OUTDTYPE, DTemplateType dTemplateType = DTemplateType::Aligned128,
const uint8_t LAYOUT = 0>
class QuantFlashAttentionScoreGradPresfmgRegbase {
public:
    __aicore__ inline QuantFlashAttentionScoreGradPresfmgRegbase(){};
    __aicore__ inline void
    Init(GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR dx, GM_ADDR y, GM_ADDR deqScaleDy, GM_ADDR actual_seq_qlen,
        GM_ADDR workspace,
        QuantFagTilingType ordTilingData,
        TPipe *pipe_in);
    __aicore__ inline void Process();
    __aicore__ inline void SyncALLCores();

    __aicore__ inline void CalTempDimAlign();
    __aicore__ inline void InitIndex(int64_t startIdx, int64_t &curS);
    __aicore__ inline void CopyInSfmg(int64_t leftNburst, int64_t &curS);
    __aicore__ inline void DoCopyIn(int64_t curS, int64_t curNBurst, int64_t dstOffset);
    __aicore__ inline void CalculateSoftmaxGrad(int64_t taskId, int64_t sfmgOutputOffset,
                                        int64_t curNBurst, int64_t deqScaleIdx = 0);
    __aicore__ inline void DoSoftmaxGradFp8();

    constexpr static uint32_t HEAD_DIM_ALIGN = static_cast<uint32_t>(dTemplateType);
    constexpr static uint32_t TEMP_DIM_ALIGN_320 = 320;
    constexpr static uint32_t TEMP_DIM_ALIGN_384 = 384;
    constexpr static uint32_t TEMP_DIM_ALIGN_448 = 448;
    constexpr static uint32_t TEMP_DIM_ALIGN_192 = 192;
    constexpr static uint32_t TEMP_DIM_ALIGN_128 = 128;
    constexpr static uint32_t TEMP_DIM_ALIGN_256 = 256;
    constexpr static int64_t BLOCK_BYTE_SIZE = 32;
    constexpr static uint32_t QUANT_BLOCK_SIZE = 512;

    TPipe *pipe;
    GlobalTensor<float> dqWorkSpaceGm, dkWorkSpaceGm, dvWorkSpaceGm, deqScaleDyGm, sfmgWorkspaceGm;
    GlobalTensor<T1> dxGm;
    GlobalTensor<OUTDTYPE> dqGm, dkGm, dvGm, yGm;

    QuantFagTilingType tilingData;

    uint32_t vBlockIdx;
    // query
    uint64_t ubBaseSize;
    uint64_t qPreBlockFactor;
    uint64_t qPreBlockTotal;
    uint64_t qPreBlockTail;
    uint64_t kPreBlockFactor;
    uint64_t kPreBlockTotal;
    uint64_t kPreBlockTail;
    uint64_t vPreBlockFactor;
    uint64_t vPreBlockTotal;
    uint64_t vPreBlockTail;

    uint64_t initdqSize;
    uint64_t dqOffset;
    uint64_t initdkSize;
    uint64_t dkOffset;
    uint64_t initdvSize;
    uint64_t dvOffset;

    int64_t b;
    int64_t n1;
    int64_t s1;
    int64_t s2;
    int64_t d;
    int64_t bIdx = 0;
    int64_t nIdx = 0;
    int64_t sIdx = 0;
    int64_t ns1d = 0;
    int64_t s1d = 0;
    int64_t nd = 0;
    int64_t bnd = 0;

    // fp8专用
    int64_t ns1o = 0;
    int64_t s1o = 0;
    int64_t soIdx = 0;

    int64_t dAlignToBlock;
    int64_t dAlignToBlockB16;
    int64_t tempDimAlign = 0;
    uint32_t layout;
    uint32_t usedCoreNum;
    GM_ADDR actual_seq_qlen_addr;

    LocalTensor<OUTDTYPE> input1Buf;
    LocalTensor<T1> input2Buf;
    LocalTensor<T2> outputBuf;

    TQue<QuePosition::VECIN, 1> input1Que[2];
    TQue<QuePosition::VECIN, 1> input2Que[2];
    TQue<QuePosition::VECOUT, 1> out1Que;
};

PRE_FUNCTION_TEMPLATE
__aicore__ inline void QuantFlashAttentionScoreGradPresfmgRegbase<PRE_FUNCTION_ARGS_TEMPLATE>::Init(
    GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR dx, GM_ADDR y, GM_ADDR deqScaleDy, GM_ADDR actual_seq_qlen,
    GM_ADDR workspace,
    QuantFagTilingType orgTilingData,
    TPipe *pipe_in)
{
    if (g_coreType == AIV) {
        vBlockIdx = GetBlockIdx();

        tilingData = orgTilingData;
        pipe = pipe_in;

        // tiling data
        qPreBlockFactor = tilingData->q_pre_block_factor;
        qPreBlockTotal = tilingData->q_pre_block_total;
        qPreBlockTail = tilingData->q_pre_block_tail;
        kPreBlockFactor = tilingData->k_pre_block_factor;
        kPreBlockTotal = tilingData->k_pre_block_total;
        kPreBlockTail = tilingData->k_pre_block_tail;
        vPreBlockFactor = tilingData->v_pre_block_factor;
        vPreBlockTotal = tilingData->v_pre_block_total;
        vPreBlockTail = tilingData->v_pre_block_tail;

        b = tilingData->b;
        n1 = tilingData->n2 * tilingData->g;
        s1 = tilingData->s1;
        d = tilingData->d;
        ns1d = n1 * s1 * d;
        s1d = s1 * d;
        nd = n1 * d;
        bnd = b * n1 * d;

        // fp8专用
        ns1o = n1 * Ceil(s1, QUANT_BLOCK_SIZE);
        s1o = Ceil(s1, QUANT_BLOCK_SIZE);

        usedCoreNum = tilingData->sfmg_used_core_num;
        layout = LAYOUT;

        int64_t blockNums = BLOCK_BYTE_SIZE / sizeof(T1);
        int64_t blockNumsB16 = BLOCK_BYTE_SIZE / sizeof(OUTDTYPE);
        dAlignToBlock = AlignTo(d, blockNums);
        dAlignToBlockB16 = AlignTo(d, blockNumsB16);
        actual_seq_qlen_addr = actual_seq_qlen;

        dxGm.SetGlobalBuffer((__gm__ T1 *)dx);
        dqGm.SetGlobalBuffer((__gm__ OUTDTYPE *)dq);
        dkGm.SetGlobalBuffer((__gm__ OUTDTYPE *)dk);
        dvGm.SetGlobalBuffer((__gm__ OUTDTYPE *)dv);
        yGm.SetGlobalBuffer((__gm__ OUTDTYPE *)y);
        deqScaleDyGm.SetGlobalBuffer((__gm__ float *)deqScaleDy);

        dqWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace +
                                      tilingData->dq_work_space_offset / sizeof(T2));
        dkWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace +
                                      tilingData->dk_work_space_offset / sizeof(T2));
        dvWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace +
                                      tilingData->dv_work_space_offset / sizeof(T2));
        sfmgWorkspaceGm.SetGlobalBuffer((__gm__ float *)workspace +
                                        tilingData->sfmg_work_space_offset / sizeof(T2));

        initdqSize = vBlockIdx == qPreBlockTotal - 1 ? qPreBlockTail : qPreBlockFactor;
        dqOffset = ((uint64_t)vBlockIdx) * qPreBlockFactor;
        initdkSize = vBlockIdx == kPreBlockTotal - 1 ? kPreBlockTail : kPreBlockFactor;
                dkOffset = ((uint64_t)vBlockIdx) * kPreBlockFactor;
        initdvSize = vBlockIdx == vPreBlockTotal - 1 ? vPreBlockTail : vPreBlockFactor;
        dvOffset = ((uint64_t)vBlockIdx) * vPreBlockFactor;

        pipe->InitBuffer(input1Que[0], 1, tilingData->sfmg_y_buffer_len);
        pipe->InitBuffer(input1Que[1], 1, tilingData->sfmg_y_buffer_len);
        pipe->InitBuffer(input2Que[0], 1, tilingData->sfmg_dy_buffer_len);
        pipe->InitBuffer(input2Que[1], 1, tilingData->sfmg_dy_buffer_len);
        pipe->InitBuffer(out1Que, 2, tilingData->sfmg_output_buffer_len);
    }
}

PRE_FUNCTION_TEMPLATE
__aicore__ inline void QuantFlashAttentionScoreGradPresfmgRegbase<PRE_FUNCTION_ARGS_TEMPLATE>::Process()
{
    // process
    if (g_coreType == AIV && vBlockIdx < usedCoreNum) {
        if (tilingData->s2_outer > 1) {
            InitOutput<float>(dqWorkSpaceGm[dqOffset], initdqSize, 0);
        }

        if constexpr (IsSameType<T1, hifloat8_t>::value) {
            // FP8量化方案中，采用不同的数据分块方案
            CalTempDimAlign();
            DoSoftmaxGradFp8();
        }
    }
}

PRE_FUNCTION_TEMPLATE
__aicore__ inline void QuantFlashAttentionScoreGradPresfmgRegbase<PRE_FUNCTION_ARGS_TEMPLATE>::CalTempDimAlign()
{
    tempDimAlign = HEAD_DIM_ALIGN;
}

PRE_FUNCTION_TEMPLATE
__aicore__ inline void
QuantFlashAttentionScoreGradPresfmgRegbase<PRE_FUNCTION_ARGS_TEMPLATE>::InitIndex(int64_t startIdx, int64_t &curS)
{
    bIdx = startIdx / ns1d;
    int64_t bTail = startIdx % ns1d;
    nIdx = bTail / s1d;
    int64_t nTail = bTail % s1d;
    sIdx = nTail / d;
    soIdx = sIdx / QUANT_BLOCK_SIZE;
}

PRE_FUNCTION_TEMPLATE
__aicore__ inline void
QuantFlashAttentionScoreGradPresfmgRegbase<PRE_FUNCTION_ARGS_TEMPLATE>::DoCopyIn(int64_t curS, int64_t curNBurst,
                                                                               int64_t dstOffset)
{
    int64_t srcOffset = 0;
    int64_t transposeStride = 0;
    int64_t transposeStrideB16 = 0;
    if (layout == BNGSD) {
        srcOffset = bIdx * ns1d + nIdx * s1d + sIdx * d;
        transposeStride = 0;
        transposeStrideB16 = 0;
    } else if (layout == BSNGD) {
        srcOffset = bIdx * ns1d + sIdx * (nd) + nIdx * d;
        transposeStride = (nd - d) * sizeof(T1);
        transposeStrideB16 = (nd - d) * sizeof(OUTDTYPE);
    }
    int64_t dstBlockStrideB16 = (tempDimAlign - dAlignToBlockB16) * sizeof(OUTDTYPE) / BLOCK_BYTE_SIZE;
    int64_t dstBlockStride = (tempDimAlign - dAlignToBlock) * sizeof(T1) / BLOCK_BYTE_SIZE;
    DataCopyPad(input1Buf[dstOffset], yGm[srcOffset], {static_cast<uint16_t>(curNBurst),
        static_cast<uint32_t>(d * sizeof(OUTDTYPE)), transposeStrideB16, dstBlockStrideB16, 0},
        {true, 0, static_cast<uint8_t>((dAlignToBlockB16 - d)), 0});
    DataCopyPad(input2Buf[dstOffset].template ReinterpretCast<uint8_t>(),
        dxGm[srcOffset].template ReinterpretCast<uint8_t>(),
        {static_cast<uint16_t>(curNBurst), static_cast<uint32_t>(d * sizeof(T1)),
        static_cast<uint32_t>(transposeStride), static_cast<uint32_t>(dstBlockStride), 0},
        {true, 0, static_cast<uint8_t>((dAlignToBlock - d)), 0});
}

PRE_FUNCTION_TEMPLATE
__aicore__ inline void
QuantFlashAttentionScoreGradPresfmgRegbase<PRE_FUNCTION_ARGS_TEMPLATE>::CopyInSfmg(int64_t leftNburst, int64_t &curS)
{
    int64_t dstOffset = 0;
    while (leftNburst > 0) {
        int64_t curNburst = 0;
        if (curS - sIdx < leftNburst) { // 需要借N或借B
            curNburst = curS - sIdx;
            DoCopyIn(curS, curNburst, dstOffset);
            leftNburst = leftNburst - curNburst;
            sIdx = 0;
            if (nIdx < n1 - 1) { // 需要借N
                nIdx += 1;
            } else {
                nIdx = 0;
                if (bIdx < b - 1) { // 需要借B
                    bIdx += 1;
                    curS = s1;
                } else { // 没有轴可以借了, end
                    leftNburst = 0;
                }
            }
        } else { // 当前s够用
            curNburst = leftNburst;
            DoCopyIn(curS, curNburst, dstOffset);
            sIdx = sIdx + leftNburst;
            leftNburst = 0;
        }
        dstOffset = dstOffset + curNburst * tempDimAlign;
    }
}

PRE_FUNCTION_TEMPLATE
__aicore__ inline void QuantFlashAttentionScoreGradPresfmgRegbase<PRE_FUNCTION_ARGS_TEMPLATE>::CalculateSoftmaxGrad(
    int64_t taskId, int64_t sfmgOutputOffset, int64_t curNBurst, int64_t deqScaleIdx)
{
    LocalTensor<OUTDTYPE> yInTensor = input1Que[taskId & 1].DeQue<OUTDTYPE>();
    LocalTensor<T1> dxInTensor = input2Que[taskId & 1].DeQue<T1>();
    auto output1Buf = out1Que.AllocTensor<T2>();

    if constexpr (IsSameType<T1, hifloat8_t>::value) {
        float deqScaleDxValue = deqScaleDyGm.GetValue(deqScaleIdx);
        AscendC::MyAntiQuantSoftmaxGradFrontCast<T1, T2, OUTDTYPE, HEAD_DIM_ALIGN>(output1Buf, yInTensor, dxInTensor,
            deqScaleDxValue, static_cast<uint32_t>(curNBurst),
            static_cast<uint32_t>(dAlignToBlockB16));
    }
    out1Que.EnQue(output1Buf);
    out1Que.DeQue<T2>();
    DataCopyPad(sfmgWorkspaceGm[sfmgOutputOffset], output1Buf,
        {1, static_cast<uint32_t>(curNBurst * sizeof(T2)), 0, 0, 0});
    out1Que.FreeTensor(output1Buf);
}

PRE_FUNCTION_TEMPLATE
__aicore__ inline void QuantFlashAttentionScoreGradPresfmgRegbase<PRE_FUNCTION_ARGS_TEMPLATE>::DoSoftmaxGradFp8()
{
    // process
    if (vBlockIdx < usedCoreNum) {
        int64_t normalAxisSize = tilingData->normal_axis_size;
        // FP8 Per‑Block 量化中保证 nBurst=128
        int64_t nBurst = tilingData->single_loop_nburst_num; // 一次普通loop处理多少个D
        int64_t curS = s1;
        int64_t taskId = 0;
        int64_t dataBlockIdx = (int)vBlockIdx - (int)usedCoreNum;
        int64_t numOf128InS1 = ((int)s1 + nBurst - 1) / nBurst; // s1按大小切分成128的block
        int64_t s1Residual = (int)s1 % nBurst == 0 ? nBurst : (int)s1 % nBurst; // 尾块大小

        while (true) {
            dataBlockIdx += usedCoreNum; // 用核数做偏移量
            // 下方的逻辑等价于，每次值到理论没有尾块的s1blockStart上，然后减去这之前可能有的所有尾块大小
            int64_t sfmgOutputOffset = dataBlockIdx * nBurst - (dataBlockIdx / numOf128InS1) * (nBurst - s1Residual);
            if (sfmgOutputOffset >= normalAxisSize) {
                break;
            }

            input1Buf = input1Que[taskId & 1].AllocTensor<OUTDTYPE>();
            input2Buf = input2Que[taskId & 1].AllocTensor<T1>();
            int64_t curNBurst = ((dataBlockIdx + 1) % numOf128InS1 == 0) ? s1Residual : nBurst;

            InitIndex(sfmgOutputOffset * d, curS);
            CopyInSfmg(curNBurst, curS);

            input1Que[taskId & 1].EnQue(input1Buf);
            input2Que[taskId & 1].EnQue(input2Buf);
            int64_t deqDyScaleIdx = bIdx * ns1o + nIdx * s1o + soIdx; // dequant GM默认排布BNSD
            CalculateSoftmaxGrad(taskId, sfmgOutputOffset, curNBurst, 0);

            input1Que[taskId & 1].FreeTensor(input1Buf);
            input2Que[taskId & 1].FreeTensor(input2Buf);
            taskId++;
        }
    }
}

PRE_FUNCTION_TEMPLATE
__aicore__ inline void QuantFlashAttentionScoreGradPresfmgRegbase<PRE_FUNCTION_ARGS_TEMPLATE>::SyncALLCores()
{
    SyncAll<false>();
}

#endif // _QUANT_FLASH_ATTENTION_SCORE_GRAD_PRE_SFMG_KERNEL_REBASE_H_