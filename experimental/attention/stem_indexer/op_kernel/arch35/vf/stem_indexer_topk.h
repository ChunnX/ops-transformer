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
 * \file stem_indexer_topk.h
 * \brief
 */
#ifndef stem_indexer_TOPK_H
#define stem_indexer_TOPK_H

#include "kernel_operator.h"
#include "vf_topk_gather.h"
#include "vf_topk_16_gather.h"

namespace SIKernel {
template<typename SCORE_T>
class SITopk {
public:
    __aicore__ inline uint32_t GetSharedTmpBufferSize()
    {
        // 2 * SICommon::Align(topK, (uint32_t)256): 两块hisIndexLocal
        // 5 * 256：histogramsLocal + idxLocal[0-3]
        // 64：nkValueLocal
        uint64_t bufferSize1 = (2 * SICommon::Align(topK, (uint32_t)256) + 5 * 256 + 64)  * sizeof(uint32_t);
        // SICommon::Align(topK, (uint32_t)256) + trunkLen：tmpIndexLocal
        uint64_t bufferSize2 = (SICommon::Align(topK, (uint32_t)256) + trunkLen)  * sizeof(uint32_t);
        return bufferSize1 + bufferSize2;
    }

    static constexpr uint32_t MAX_LOOP_M = 4;

    // reuseMm1ResLocal 内的复用偏移（indices 按 uint32_t 元素数，score 按 SCORE_T 元素数）
    // idx0~idx3: 4 * 4*256 = 4096
    // histograms: 4*256 = 1024
    // nkValue: 4*64 = 256
    // tmpIndex: 4*256 = 1024
    // 合计 6400，之后排布 indicesOut(1024) 和 scoreOut(1024)
    static constexpr uint32_t REUSE_INDICES_OUT_OFFSET = 6400;
    static constexpr uint32_t REUSE_SCORE_OUT_OFFSET_U32 = 7424;
    static constexpr uint32_t REUSE_SCORE_OUT_OFFSET =
        REUSE_SCORE_OUT_OFFSET_U32 * sizeof(uint32_t) / sizeof(SCORE_T);

    __aicore__ inline uint32_t GetReUseSharedTmpBufferSize()
    {
        // MAX_LOOP_M * 256：histogramsLocal
        // MAX_LOOP_M * 64：nkValueLocal
        // 其余(idx0-3/hisIndex)皆复用；topK 直接复用 topkNumLocal 切片，不单独申请
        uint64_t histograms = MAX_LOOP_M * 256;
        uint64_t nkValue = MAX_LOOP_M * 64;
        uint64_t tmpIndex = MAX_LOOP_M * SICommon::Align(topK, (uint32_t)256);
        return (histograms + nkValue + tmpIndex) * sizeof(uint32_t);
    }

    __aicore__ inline void Init(uint32_t topK, uint32_t trunkLen)
    {
        this->topK = topK;
        this->trunkLen = trunkLen;
    }

    __aicore__ inline void SetTopK(uint32_t topK)
    {
        this->topK = topK;
    }
    
    __aicore__ inline void InitBuffers(LocalTensor<uint32_t>& sharedTmpBuffer)
    {
        LocalTensor<uint32_t> hisIndexLocal1 = sharedTmpBuffer[0];
        LocalTensor<uint32_t> hisIndexLocal2 = hisIndexLocal1[SICommon::Align(topK, (uint32_t)256)];
        hisIndexLocal[0] = hisIndexLocal1;
        hisIndexLocal[1] = hisIndexLocal2;
        histogramsLocal = hisIndexLocal2[SICommon::Align(topK, (uint32_t)256)];
        idx0Local = histogramsLocal[MAX_LOOP_M * 256];
        idx1Local = idx0Local[MAX_LOOP_M * 256];
        idx2Local = idx1Local[MAX_LOOP_M * 256];
        idx3Local = idx2Local[MAX_LOOP_M * 256];
        nkValueLocal = idx3Local[MAX_LOOP_M * 256];
        tmpIndexLocal = nkValueLocal[MAX_LOOP_M * 64];
    }

    __aicore__ inline void InitBuffers(LocalTensor<uint32_t>& sharedTmpBuffer,
                                       LocalTensor<uint32_t>& indicesOutLocal)
    {
        // hisIndexLocal[0] 复用 indicesOutLocal；其余(hisIndex[1]/histograms/nk/tmpIndex)在 Batch4Rows 中复用
        hisIndexLocal[0] = indicesOutLocal;
    }
    // __aicore__ inline void operator()(const LocalTensor<uint32_t>& mrgValueLocal, const LocalTensor<uint32_t>& indicesOutLocal,
    //                                   const LocalTensor<uint32_t>& hisValueLocal, const LocalTensor<uint32_t>& reuseMm1ResLocal,
    //                                   const LocalTensor<uint32_t>& reuseGlobalIndexLocal, uint32_t s2SeqLen, uint32_t loopIdx, uint32_t s2LoopNum)
    // {
    //     // uint32_t时UB需要进行复用，此处将每8位的targetbin复用mm1res
    //     idx0Local = reuseMm1ResLocal;
    //     idx1Local = idx0Local[256];
    //     idx2Local = idx1Local[256];
    //     idx3Local = idx2Local[256];
    //     // hisIndexLocal[0]复用globalIndexLocal
    //     hisIndexLocal[0] = reuseGlobalIndexLocal;

    //     if (s2LoopNum == 1) {
    //         SITopkb32gather::SiTopKVF<false>(tmpIndexLocal, hisValueLocal, mrgValueLocal, histogramsLocal,
    //                                            idx0Local, idx1Local, idx2Local, idx3Local,
    //                                            nkValueLocal, topK, s2SeqLen);
    //         PipeBarrier<PIPE_V>();
    //         AscendC::DataCopy(indicesOutLocal, tmpIndexLocal, SICommon::Align(topK, (uint32_t)256));
    //     } else {
    //         if (loopIdx == 0) {
    //             SITopkb32gather::SiTopKVF<true>(tmpIndexLocal, hisValueLocal, mrgValueLocal, histogramsLocal,
    //                                               idx0Local, idx1Local, idx2Local, idx3Local,
    //                                               nkValueLocal, topK, s2SeqLen);
    //             PipeBarrier<PIPE_V>();
    //             AscendC::DataCopy(hisIndexLocal[1], tmpIndexLocal, SICommon::Align(topK, (uint32_t)256));
    //         } else {
    //             SITopkb32gather::SiTopKVF<true>(tmpIndexLocal, hisValueLocal, mrgValueLocal, histogramsLocal,
    //                                               idx0Local, idx1Local, idx2Local, idx3Local,
    //                                               nkValueLocal, topK, s2SeqLen);
    //             PipeBarrier<PIPE_V>();
    //             SITopkb32gather::SiTopKGatherVF(hisIndexLocal[1], hisValueLocal,
    //                                               mrgValueLocal, tmpIndexLocal, hisIndexLocal[0],
    //                                               topK, loopIdx * trunkLen - SICommon::Align(topK, (uint32_t)256),
    //                                               s2SeqLen);
    //         }
    //     }
    // }

    __aicore__ inline void Batch4Rows(
        const LocalTensor<SCORE_T>& mrgValueLocal,
        const LocalTensor<uint32_t>& indicesOutLocal,
        const LocalTensor<SCORE_T>& hisValueLocal,
        const LocalTensor<uint32_t>& reuseMm1ResLocal,
        const LocalTensor<uint32_t>& reuseGlobalIndexLocal,
        LocalTensor<uint32_t>& topkNumLocal,
        uint32_t mBaseIdx, uint32_t batchRowNum, uint32_t mrgRowStride,
        uint32_t inputOffset, uint32_t validLen, uint32_t loopIdx, uint32_t s2LoopNum)
    {
        if (batchRowNum == 0U || validLen == 0U) {
            return;
        }
        const uint32_t offset = mrgRowStride;
        const uint32_t topkAlign = SICommon::Align(topK, (uint32_t)256);
        uint32_t rowGlobalIdxOffset = mBaseIdx * topkAlign;
        // 所有临时 buffer 复用 reuseMm1ResLocal，按 idx0-3 → histograms → nk → tmpIndex 顺序排布
        idx0Local = reuseMm1ResLocal;
        idx1Local = idx0Local[MAX_LOOP_M * 256];
        idx2Local = idx1Local[MAX_LOOP_M * 256];
        idx3Local = idx2Local[MAX_LOOP_M * 256];
        histogramsLocal = idx3Local[MAX_LOOP_M * 256];
        nkValueLocal = histogramsLocal[MAX_LOOP_M * 256];
        tmpIndexLocal = nkValueLocal[MAX_LOOP_M * 64];
        // hisIndexLocal[0] 复用 globalIndexLocal，hisIndexLocal[1] 复用 indicesOutLocal
        hisIndexLocal[0] = reuseGlobalIndexLocal[rowGlobalIdxOffset];
        hisIndexLocal[1] = indicesOutLocal;

        LocalTensor<SCORE_T> inputValueLocal = mrgValueLocal[mBaseIdx * mrgRowStride + inputOffset];
        LocalTensor<uint32_t> topKLocalSlice = topkNumLocal[mBaseIdx];

        if (s2LoopNum == 1) {
            if constexpr (std::is_same_v<SCORE_T, uint16_t>) {
                LocalTensor<uint16_t> tmpIdxLocal16 = tmpIndexLocal.ReinterpretCast<uint16_t>();
                topkb16gather::LiTopKVF<false>(tmpIdxLocal16, hisValueLocal, inputValueLocal, histogramsLocal,
                                               idx0Local, idx1Local, idx2Local, idx3Local,
                                               nkValueLocal, topKLocalSlice, validLen, batchRowNum, offset);
            } else {
                SITopkb32gather::SiTopKVF<false>(tmpIndexLocal, hisValueLocal, inputValueLocal, histogramsLocal,
                                                   idx0Local, idx1Local, idx2Local, idx3Local,
                                                   nkValueLocal, topKLocalSlice, validLen, batchRowNum, offset);
            }
            PipeBarrier<PIPE_V>();
            for (uint32_t m = 0; m < batchRowNum; ++m) {
                uint32_t mInnerIdx = mBaseIdx + m;
                if constexpr (std::is_same_v<SCORE_T, uint16_t>) {
                    LocalTensor<uint16_t> tmpIdxLocal16 = tmpIndexLocal.ReinterpretCast<uint16_t>();
                    topkb16gather::LiTopKTmpIndexU16ToU32(hisIndexLocal[0][m * topkAlign],
                                                          tmpIdxLocal16[m * topkAlign],
                                                          topkNumLocal.GetValue(mInnerIdx));
                } else {
                    AscendC::DataCopy(hisIndexLocal[0][m * topkAlign], tmpIndexLocal[m * topkAlign], topkAlign);
                }
            }
        } else if (loopIdx == 0) {
            if constexpr (std::is_same_v<SCORE_T, uint16_t>) {
                LocalTensor<uint16_t> tmpIdxLocal16 = tmpIndexLocal.ReinterpretCast<uint16_t>();
                topkb16gather::LiTopKVF<true>(tmpIdxLocal16, hisValueLocal, inputValueLocal, histogramsLocal,
                                              idx0Local, idx1Local, idx2Local, idx3Local,
                                              nkValueLocal, topKLocalSlice, validLen, batchRowNum, offset);
            } else {
                SITopkb32gather::SiTopKVF<true>(tmpIndexLocal, hisValueLocal, inputValueLocal, histogramsLocal,
                                                  idx0Local, idx1Local, idx2Local, idx3Local,
                                                  nkValueLocal, topKLocalSlice, validLen, batchRowNum, offset);
            }
            PipeBarrier<PIPE_V>();
            for (uint32_t m = 0; m < batchRowNum; ++m) {
                uint32_t mInnerIdx = mBaseIdx + m;
                // 输出到 globalIndexLocal(hisIndexLocal[0])，给下一轮 gather 和 CopyOutFourRows 用
                if constexpr (std::is_same_v<SCORE_T, uint16_t>) {
                    LocalTensor<uint16_t> tmpIdxLocal16 = tmpIndexLocal.ReinterpretCast<uint16_t>();
                    topkb16gather::LiTopKTmpIndexU16ToU32(hisIndexLocal[0][m * topkAlign],
                                                          tmpIdxLocal16[m * topkAlign],
                                                          topkNumLocal.GetValue(mInnerIdx));
                } else {
                    AscendC::DataCopy(hisIndexLocal[0][m * topkAlign], tmpIndexLocal[m * topkAlign], topkAlign);
                }
                AscendC::DataCopy(mrgValueLocal[(mBaseIdx + m) * mrgRowStride], hisValueLocal[m * topkAlign], topkAlign);
            }
        } else {
            if constexpr (std::is_same_v<SCORE_T, uint16_t>) {
                LocalTensor<uint16_t> tmpIdxLocal16 = tmpIndexLocal.ReinterpretCast<uint16_t>();
                topkb16gather::LiTopKVF<true>(tmpIdxLocal16, hisValueLocal, inputValueLocal, histogramsLocal,
                                              idx0Local, idx1Local, idx2Local, idx3Local,
                                              nkValueLocal, topKLocalSlice, validLen, batchRowNum, offset);
            } else {
                SITopkb32gather::SiTopKVF<true>(tmpIndexLocal, hisValueLocal, inputValueLocal, histogramsLocal,
                                                  idx0Local, idx1Local, idx2Local, idx3Local,
                                                  nkValueLocal, topKLocalSlice, validLen, batchRowNum, offset);
            }
            PipeBarrier<PIPE_V>();
            // gather: 从 globalIndexLocal(hisIndexLocal[0]) 读上轮结果，输出到 indicesOutLocal(hisIndexLocal[1])
            for (uint32_t m = 0; m < batchRowNum; ++m) {
                uint32_t mInnerIdx = mBaseIdx + m;
                if constexpr (std::is_same_v<SCORE_T, uint16_t>) {
                    LocalTensor<uint16_t> tmpIdxLocal16 = tmpIndexLocal.ReinterpretCast<uint16_t>();
                    topkb16gather::LiTopKGatherVF(
                        hisIndexLocal[1][m * topkAlign],
                        hisValueLocal[m * topkAlign],
                        inputValueLocal[m * offset],
                        tmpIdxLocal16[m * topkAlign],
                        hisIndexLocal[0][m * topkAlign],
                        topkNumLocal.GetValue(mInnerIdx),
                        loopIdx * trunkLen - topkAlign,
                        validLen);
                } else {
                    SITopkb32gather::SiTopKGatherVF(
                        hisIndexLocal[1][m * topkAlign],
                        hisValueLocal[m * topkAlign],
                        mrgValueLocal[mInnerIdx * mrgRowStride + inputOffset],
                        tmpIndexLocal[m * topkAlign],
                        hisIndexLocal[0][m * topkAlign],
                        topkNumLocal.GetValue(mInnerIdx),
                        loopIdx * trunkLen - topkAlign,
                        validLen);
                }
                PipeBarrier<PIPE_V>();
                // 拷回 globalIndexLocal(hisIndexLocal[0])，给下一轮/CopyOutFourRows 用
                AscendC::DataCopy(hisIndexLocal[0][m * topkAlign], hisIndexLocal[1][m * topkAlign], topkAlign);
                AscendC::DataCopy(mrgValueLocal[mInnerIdx * mrgRowStride], hisValueLocal[m * topkAlign], topkAlign);
            }
        }
    }

private:
    LocalTensor<uint32_t> hisIndexLocal[2]; // 每trunkLen长度的s2选出的topK个索引
    LocalTensor<uint32_t> histogramsLocal;  // 直方图的临时Buf MAX_LOOP_M * 256 * 4B
    LocalTensor<uint32_t> idx0Local;        // 输入数据第1个8位Buf MAX_LOOP_M * 256 * 4B
    LocalTensor<uint32_t> idx1Local;        // 输入数据第2个8位Buf MAX_LOOP_M * 256 * 4B
    LocalTensor<uint32_t> idx2Local;        // 输入数据第3个8位Buf MAX_LOOP_M * 256 * 4B
    LocalTensor<uint32_t> idx3Local;        // 输入数据第4个8位Buf MAX_LOOP_M * 256 * 4B
    LocalTensor<uint32_t> nkValueLocal;     // next_k 暂存Buf MAX_LOOP_M * 64 * 4B
    LocalTensor<uint32_t> tmpIndexLocal;    // 每行tmpIdx MAX_LOOP_M * Align(topK,256) * 4B
    uint32_t topK = 512;
    uint32_t trunkLen = 8192;
};
}
#endif
