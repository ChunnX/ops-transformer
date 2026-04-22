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
 * \file attention_pioneer_backward_s1s2_bn2gs1s2_post_regbase.h
 * \brief
 */
#ifndef ATTENTION_PIONEER_BACKWARD_S1S2_BNGS1S2_POST_KERNEL_REGBASE_H_
#define ATTENTION_PIONEER_BACKWARD_S1S2_BNGS1S2_POST_KERNEL_REGBASE_H_
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#define FAG_POST_CLASS_TEMPLATE                                                                                             \
    template <typename T1, typename T2, typename OUTDTYPE=T1, const uint8_t SPLIT_AXIS = 0, const bool IS_ROPE = false, const uint8_t DETER_SPARSE_TYPE = 0, const bool IS_TND = 0, const bool IS_TND_SWIZZLE = 0> 
#define FAG_POST_FUNCTION_TEMPLATE                                                                                          \
    template <typename T1, typename T2, typename OUTDTYPE, const uint8_t SPLIT_AXIS, bool IS_ROPE, const uint8_t DETER_SPARSE_TYPE, const bool IS_TND, const bool IS_TND_SWIZZLE>
#define FAG_POST_FUNCTION_PARAMS_TEMPLATE T1, T2, OUTDTYPE, SPLIT_AXIS, IS_ROPE, DETER_SPARSE_TYPE, IS_TND, IS_TND_SWIZZLE

FAG_POST_CLASS_TEMPLATE 
class AttentionPioneerBackwardS1S2BNGS1S2PostRegbase {
public:
    __aicore__ inline AttentionPioneerBackwardS1S2BNGS1S2PostRegbase(){};
    __aicore__ inline void Init(__gm__ uint8_t *dq, __gm__ uint8_t *dk, __gm__ uint8_t *dkSink, __gm__ uint8_t *dv, __gm__ uint8_t *dvSink, __gm__ uint8_t *dqRope,
                                __gm__ uint8_t *dkRope,__gm__ uint8_t *workspace,
                                FagTilingType ordTilingData,
                                TPipe *pipe_in);
    __aicore__ inline void Process();
    __aicore__ inline void ProcessBNS2Deter();
    __aicore__ inline void ProcessSinkOutput();  // Sink reduce后的muls+cast处理

    constexpr static uint32_t ADDR_ALIGN_SIZE = 512;
    uint32_t REGBASE_POST_BASE = 128 * 128;
    uint32_t ROPE_DIM = 64;
    uint32_t VALUE_DIM = 128;
    uint32_t POST_S_BASE = 96;
    TPipe *pipe;
    FagTilingType tilingData;
    TQue<QuePosition::VECIN, 1> inQueuePing;
    TQue<QuePosition::VECOUT, 1> outQueuePing;
    TQue<QuePosition::VECIN, 1> inQueuePong;
    TQue<QuePosition::VECOUT, 1> outQueuePong;
    TQue<QuePosition::VECIN, 1> inQueuedKeySink;
    TQue<QuePosition::VECOUT, 1> outQueuedKeySink;
    TBuf<> accumdKeySinkBuf;
    TQue<QuePosition::VECIN, 1> inQueuedValueSink;
    TQue<QuePosition::VECOUT, 1> outQueuedValueSink;
    TBuf<> accumdValueSinkBuf;
    GlobalTensor<OUTDTYPE> dqkv[3];
    GlobalTensor<OUTDTYPE> dqkRope[2];
    GlobalTensor<float> dqkvWorkspace[3];
    GlobalTensor<float> deterGm[2];
    // Sink相关GlobalTensor
    GlobalTensor<OUTDTYPE> dkSinkGm;
    GlobalTensor<OUTDTYPE> dvSinkGm;
    GlobalTensor<float> dkSinkWorkSpaceGm;
    GlobalTensor<float> dvSinkWorkSpaceGm;
    uint32_t vBlockIdx;
    uint64_t loop;
    uint64_t inputTotalSize;
    uint64_t qPostTailNum;
};

FAG_POST_FUNCTION_TEMPLATE
__aicore__ inline void AttentionPioneerBackwardS1S2BNGS1S2PostRegbase<FAG_POST_FUNCTION_PARAMS_TEMPLATE>::Init(
    __gm__ uint8_t *dq, __gm__ uint8_t *dk, __gm__ uint8_t *dkSink, __gm__ uint8_t *dv, __gm__ uint8_t *dvSink, __gm__ uint8_t *dqRope,
    __gm__ uint8_t *dkRope, __gm__ uint8_t *workspace,
    FagTilingType ordTilingData, TPipe *pipe_in)
{
    vBlockIdx = GetBlockIdx();
    tilingData = ordTilingData;
    pipe = pipe_in;

    dqkv[0].SetGlobalBuffer((__gm__ OUTDTYPE *)dq);
    dqkv[1].SetGlobalBuffer((__gm__ OUTDTYPE *)dk);
    dqkv[2].SetGlobalBuffer((__gm__ OUTDTYPE *)dv);

    dqkRope[0].SetGlobalBuffer((__gm__ OUTDTYPE *)dqRope);
    dqkRope[1].SetGlobalBuffer((__gm__ OUTDTYPE *)dkRope);

    loop = tilingData->postTilingData.qPostBlockFactor;
    inputTotalSize = tilingData->postTilingData.qPostBlockTotal;
    qPostTailNum = tilingData->postTilingData.qPostTailNum;

    dqkvWorkspace[0].SetGlobalBuffer((__gm__ float *)workspace + tilingData->postTilingData.dqWorkSpaceOffset / sizeof(T2));
    dqkvWorkspace[1].SetGlobalBuffer((__gm__ float *)workspace + tilingData->postTilingData.dkWorkSpaceOffset / sizeof(T2));
    dqkvWorkspace[2].SetGlobalBuffer((__gm__ float *)workspace + tilingData->postTilingData.dvWorkSpaceOffset / sizeof(T2));

    // Sink相关GlobalTensor初始化
    dkSinkGm.SetGlobalBuffer((__gm__ OUTDTYPE *)dkSink);
    dvSinkGm.SetGlobalBuffer((__gm__ OUTDTYPE *)dvSink);
    dkSinkWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace + tilingData->postTilingData.dkSinkWorkSpaceOffset / sizeof(T2));
    dvSinkWorkSpaceGm.SetGlobalBuffer((__gm__ float *)workspace + tilingData->postTilingData.dvSinkWorkSpaceOffset / sizeof(T2));

    if constexpr (IS_ROPE) {
        REGBASE_POST_BASE = POST_S_BASE * (ROPE_DIM + VALUE_DIM);
    }

    pipe->InitBuffer(inQueuePing, 1, REGBASE_POST_BASE * sizeof(T2));
    pipe->InitBuffer(outQueuePing, 1, REGBASE_POST_BASE * sizeof(OUTDTYPE));
    pipe->InitBuffer(inQueuePong, 1, REGBASE_POST_BASE * sizeof(T2));
    pipe->InitBuffer(outQueuePong, 1, REGBASE_POST_BASE * sizeof(OUTDTYPE));

    constexpr uint32_t SINK_PROCESS_BLOCK_SIZE = 2048;
    pipe->InitBuffer(inQueuedKeySink, 1, SINK_PROCESS_BLOCK_SIZE * sizeof(T2));
    pipe->InitBuffer(outQueuedKeySink, 1, SINK_PROCESS_BLOCK_SIZE * sizeof(OUTDTYPE));
    pipe->InitBuffer(accumdKeySinkBuf, SINK_PROCESS_BLOCK_SIZE * sizeof(T2));
    pipe->InitBuffer(inQueuedValueSink, 1, SINK_PROCESS_BLOCK_SIZE * sizeof(T2));
    pipe->InitBuffer(outQueuedValueSink, 1, SINK_PROCESS_BLOCK_SIZE * sizeof(OUTDTYPE));
    pipe->InitBuffer(accumdValueSinkBuf, SINK_PROCESS_BLOCK_SIZE * sizeof(T2));

}

FAG_POST_FUNCTION_TEMPLATE
__aicore__ inline void AttentionPioneerBackwardS1S2BNGS1S2PostRegbase<FAG_POST_FUNCTION_PARAMS_TEMPLATE>::Process()
{
    if (g_coreType != AIV) {
        return;
    }
    for (int qkvIdx = 0; qkvIdx < 3; qkvIdx++) {
        if (qkvIdx == 1) {
            loop = tilingData->postTilingData.kPostBlockFactor;
            inputTotalSize = tilingData->postTilingData.kPostBlockTotal;
            qPostTailNum = tilingData->postTilingData.kPostTailNum;
        } else if (qkvIdx == 2) {
            loop = tilingData->postTilingData.vPostBlockFactor;
            inputTotalSize = tilingData->postTilingData.vPostBlockTotal;
            qPostTailNum = tilingData->postTilingData.vPostTailNum;
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
            uint64_t dqPongGmOffset = dqPingGmOffset + VALUE_DIM * POST_S_BASE; // 96*128
            uint64_t seqPingSize = pingSize / (ROPE_DIM + VALUE_DIM);
            DataCopy(vecInPing, dqkvWorkspace[qkvIdx][pingIdx], (pingSize + 7) >> 3 << 3);
            inQueuePing.EnQue(vecInPing);
            inQueuePing.DeQue<float>();
            if (qkvIdx < 2) {
                Muls(vecInPing, vecInPing, (float)tilingData->s1s2BNGS1S2BaseParams.scaleValue, pingSize);
            }
            LocalTensor<OUTDTYPE> vecOutPing = outQueuePing.AllocTensor<OUTDTYPE>();
            Cast(vecOutPing, vecInPing, RoundMode::CAST_ROUND, pingSize);
            uint64_t pongSize;
            uint64_t seqPongSize;
            LocalTensor<float> vecInPong;
            bool neeedPong = loop > 1 && pongIdx < end;
            if (neeedPong) {
                vecInPong = inQueuePong.AllocTensor<float>();
                pongSize = pongIdx + REGBASE_POST_BASE < inputTotalSize ? REGBASE_POST_BASE : qPostTailNum;
                seqPongSize = pongSize / (ROPE_DIM + VALUE_DIM);
                DataCopy(vecInPong, dqkvWorkspace[qkvIdx][pongIdx], (pongSize + 7) >> 3 << 3);
            }
            outQueuePing.EnQue(vecOutPing);
            outQueuePing.DeQue<OUTDTYPE>();
            inQueuePing.FreeTensor(vecInPing);
            if constexpr (IS_ROPE) {
                if (qkvIdx == 2) {
                    DataCopy(dqkv[qkvIdx][pingIdx], vecOutPing, pingSize);
                }  else {
                    // 96 * (128 + 64)基本块
                    DataCopyParams dataCopyParams;
                    dataCopyParams.blockCount = seqPingSize;
                    // 128 * sizeof(OUTDTYPE) / 32B
                    dataCopyParams.blockLen = 4 * sizeof(OUTDTYPE);
                    // 64 * sizeof(OUTDTYPE) / 32B
                    dataCopyParams.srcStride = 2 * sizeof(OUTDTYPE);
                    DataCopy(dqkv[qkvIdx][dqPingGmOffset], vecOutPing, dataCopyParams);

                    dataCopyParams.blockLen = 2 * sizeof(OUTDTYPE);
                    dataCopyParams.srcStride = 4 * sizeof(OUTDTYPE);
                    DataCopy(dqkRope[qkvIdx][dqPingGmOffset / 2], vecOutPing[128], dataCopyParams);
                }
                dqPingGmOffset = dqPingGmOffset + VALUE_DIM * POST_S_BASE;
            } else {
                DataCopy(dqkv[qkvIdx][pingIdx], vecOutPing, (pingSize + 15) >> 4 << 4);
            }
            outQueuePing.FreeTensor(vecOutPing);
            LocalTensor<OUTDTYPE> vecOutPong;
            if (neeedPong) {
                vecOutPong = outQueuePong.AllocTensor<OUTDTYPE>();
                inQueuePong.EnQue(vecInPong);
                inQueuePong.DeQue<float>();
                if (qkvIdx < 2) {
                    Muls(vecInPong, vecInPong, (float)tilingData->s1s2BNGS1S2BaseParams.scaleValue, pongSize);
                }
                Cast(vecOutPong, vecInPong, RoundMode::CAST_ROUND, pongSize);
                outQueuePong.EnQue(vecOutPong);
                outQueuePong.DeQue<OUTDTYPE>();
                inQueuePong.FreeTensor(vecInPong);
                if constexpr (IS_ROPE) {
                    if (qkvIdx == 2) {
                        DataCopy(dqkv[qkvIdx][pongIdx], vecOutPong, pongSize);
                    }  else {
                        // 96 * (128 + 64)基本块
                        DataCopyParams dataCopyParams;
                        dataCopyParams.blockCount = seqPongSize;
                        // 128 * sizeof(OUTDTYPE) / 32B
                        dataCopyParams.blockLen = 4 * sizeof(OUTDTYPE);
                        // 64 * sizeof(OUTDTYPE) / 32B
                        dataCopyParams.srcStride = 2 * sizeof(OUTDTYPE);
                        DataCopy(dqkv[qkvIdx][dqPongGmOffset], vecOutPong, dataCopyParams);
    
                        dataCopyParams.blockLen = 2 * sizeof(OUTDTYPE);
                        dataCopyParams.srcStride = 4 * sizeof(OUTDTYPE);
                        DataCopy(dqkRope[qkvIdx][dqPongGmOffset / 2], vecOutPong[128], dataCopyParams);
                    }
                    dqPingGmOffset = dqPingGmOffset + VALUE_DIM * POST_S_BASE;
                } else {
                    DataCopy(dqkv[qkvIdx][pongIdx], vecOutPong, (pongSize + 15) >> 4 << 4);
                }
                outQueuePong.FreeTensor(vecOutPong);
            }
        }

        // bn2s2 dk dv no need to do muls and cast
        if constexpr (SPLIT_AXIS == 5) {
            break;
        }
    }
    if (DETER_SPARSE_TYPE != NO_DETER && DETER_SPARSE_TYPE != DETER_OLD) {
        ProcessSinkOutput();
    }

}

FAG_POST_FUNCTION_TEMPLATE
__aicore__ inline void AttentionPioneerBackwardS1S2BNGS1S2PostRegbase<FAG_POST_FUNCTION_PARAMS_TEMPLATE>::ProcessSinkOutput()
{
    //  输入dk/dvsinkworkspacegm [B * 128, N2, D/Dv] 输出dkSinkGm/dvSinkGm[128, N2, D/Dv]
    // 对B轴进行reduce，之后的数据再进行muls和cast
    if (g_coreType != AIV) {
        return;
    }

    int64_t bSize = tilingData->s1s2BNGS1S2BaseParams.b;
    int64_t n2Size = tilingData->s1s2BNGS1S2BaseParams.n2;
    int64_t dSize = tilingData->s1s2BNGS1S2BaseParams.d;
    int64_t dSizeV = tilingData->s1s2BNGS1S2BaseParams.d1;
    int64_t sinkS2Size = tilingData->s1s2BNGS1S2SplitCoreParams.sinkS2Size;
    float scaleValue = tilingData->s1s2BNGS1S2BaseParams.scaleValue;

    // 使用实际的核数而不是硬编码的值
    int64_t sinkUsedCoreNum = GetBlockNum() * GetTaskRation();  // 获取实际的 block dim
    // printf("sinkUsedCoreNum = %d", sinkUsedCoreNum);
    constexpr int64_t SINK_PROCESS_BLOCK_SIZE = 2048;  // 单次处理的最大元素数

    int64_t totalDkSize = sinkS2Size * n2Size * dSize;
    int64_t dkPerCore = (totalDkSize + sinkUsedCoreNum - 1) / sinkUsedCoreNum;
    int64_t coreIdx = vBlockIdx;
    int64_t dkStart = coreIdx * dkPerCore;
    int64_t dkEnd = (dkStart + dkPerCore) > totalDkSize ? totalDkSize : (dkStart + dkPerCore);
    if (coreIdx >= sinkUsedCoreNum) {
        dkStart = totalDkSize;
        dkEnd = totalDkSize;
    }

    int64_t totalDvSize = sinkS2Size * n2Size * dSizeV;
    int64_t dvPerCore = (totalDvSize + sinkUsedCoreNum - 1) / sinkUsedCoreNum;
    int64_t dvStart = coreIdx * dvPerCore;
    int64_t dvEnd = (dvStart + dvPerCore) > totalDvSize ? totalDvSize : (dvStart + dvPerCore);
    if (coreIdx >= sinkUsedCoreNum) {
        dvStart = totalDvSize;
        dvEnd = totalDvSize;
    }
    // printf("sinkUsedCoreNum  = %ld\n", sinkUsedCoreNum);
    // printf("dkPerCore  = %ld\n", dkPerCore);
    // printf("dvPerCore  = %ld\n", dvPerCore);
    // printf("dkStart  = %ld\n", dkStart);
    // printf("dkEnd  = %ld\n", dkEnd);
    // printf("dvStart  = %ld\n", dvStart);
    // printf("dvEnd  = %ld\n", dvEnd);
    // uint32_t vBlockIdx = GetBlockIdx();
    // if(vBlockIdx == 0) {
    //     // DumpTensor(dkSinkWorkSpaceGm,1001,256 * 128);
    //     DumpTensor(dvSinkWorkSpaceGm, 1002, 128 * 128 * 3);
    //     // DumpTensor(dvSinkWorkSpaceGm[128*128], 1003, 100);
    // }
    // dk
    int64_t dkBatchStride = sinkS2Size * n2Size * dSize;
    for (int64_t offset = dkStart; offset < dkEnd; offset += SINK_PROCESS_BLOCK_SIZE) {
        int64_t curBlockSize = (offset + SINK_PROCESS_BLOCK_SIZE) > dkEnd ? (dkEnd - offset) : SINK_PROCESS_BLOCK_SIZE;
        int64_t alignedBlockSize = (curBlockSize + 7) >> 3 << 3;

        LocalTensor<float> accumTensor = accumdKeySinkBuf.AllocTensor<float>();
        // LocalTensor<float> srcTensor = inQueuedKeySink.AllocTensor<float>();
        Duplicate<float>(accumTensor, (float)0.0, alignedBlockSize);

        for (int64_t bIdx = 0; bIdx < bSize; bIdx++) {
            LocalTensor<float> srcTensor = inQueuedKeySink.AllocTensor<float>();
            int64_t srcOffset = bIdx * dkBatchStride + offset;
            DataCopy(srcTensor, dkSinkWorkSpaceGm[srcOffset], alignedBlockSize);
            inQueuedKeySink.EnQue(srcTensor);
            inQueuedKeySink.DeQue<float>();
            PipeBarrier<PIPE_V>();
            SetFlag<AscendC::HardEvent::MTE2_V>(0);
            WaitFlag<AscendC::HardEvent::MTE2_V>(0);
            Add(accumTensor, accumTensor, srcTensor, alignedBlockSize);
            inQueuedKeySink.FreeTensor(srcTensor);
        }

        // mul and cast
        PipeBarrier<PIPE_V>();
        Muls(accumTensor, accumTensor, scaleValue, curBlockSize);
        LocalTensor<OUTDTYPE> dstTensor = outQueuedKeySink.AllocTensor<OUTDTYPE>();
        PipeBarrier<PIPE_V>();
        Cast(dstTensor, accumTensor, RoundMode::CAST_ROUND, curBlockSize);
        outQueuedKeySink.EnQue(dstTensor);
        outQueuedKeySink.DeQue<OUTDTYPE>();
        // 16对齐输出
        int64_t alignedOutSize = (curBlockSize + 15) >> 4 << 4;
        DataCopy(dkSinkGm[offset], dstTensor, alignedOutSize);

        outQueuedKeySink.FreeTensor(dstTensor);
    }

    // dv 处理
    int64_t dvBatchStride = sinkS2Size * n2Size * dSizeV;
    for (int64_t offset = dvStart; offset < dvEnd; offset += SINK_PROCESS_BLOCK_SIZE) {
        int64_t curBlockSize = (offset + SINK_PROCESS_BLOCK_SIZE) > dvEnd ? (dvEnd - offset) : SINK_PROCESS_BLOCK_SIZE;
        int64_t alignedBlockSize = (curBlockSize + 7) >> 3 << 3;

        LocalTensor<float> accumTensor = accumdValueSinkBuf.Get<float>();
        // LocalTensor<float> srcTensor = inQueuedValueSink.AllocTensor<float>();
        Duplicate<float>(accumTensor, (float)0.0, alignedBlockSize);

        for (int64_t bIdx = 0; bIdx < bSize; bIdx++) {
            LocalTensor<float> srcTensor = inQueuedValueSink.AllocTensor<float>();
            int64_t srcOffset = bIdx * dvBatchStride + offset;
            DataCopy(srcTensor, dvSinkWorkSpaceGm[srcOffset], alignedBlockSize);
            inQueuedValueSink.EnQue(srcTensor);
            inQueuedValueSink.DeQue<float>();
            PipeBarrier<PIPE_V>();
            SetFlag<AscendC::HardEvent::MTE2_V>(0);
            WaitFlag<AscendC::HardEvent::MTE2_V>(0);

            Add(accumTensor, accumTensor, srcTensor, alignedBlockSize);
            inQueuedValueSink.FreeTensor(srcTensor);

        }
        LocalTensor<OUTDTYPE> dstTensor = outQueuedValueSink.AllocTensor<OUTDTYPE>();
        PipeBarrier<PIPE_V>();
        Cast(dstTensor, accumTensor, RoundMode::CAST_ROUND, curBlockSize);

        outQueuedValueSink.EnQue(dstTensor);
        outQueuedValueSink.DeQue<OUTDTYPE>();
        // 16对齐输出
        int64_t alignedOutSize = (curBlockSize + 15) >> 4 << 4;
        DataCopy(dvSinkGm[offset], dstTensor, alignedOutSize);

        // inQueuedValueSink.FreeTensor(srcTensor);
        outQueuedValueSink.FreeTensor(dstTensor);
    }
}

#endif // _ATTENTION_PIONEER_BACKWARD_S1S2_BNGS1S2_POST_KERNEL_REGBASE_H_