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

namespace SIKernel {
// Radix uses 8-bit buckets, with 256 buckets and 64 nkValue elements.
constexpr uint32_t HISTOGRAM_BIN_NUM = 256U;
constexpr uint32_t NK_VALUE_SIZE = 64U;

// Primary template placeholder; only the uint32_t sortable-key specialization is implemented.
template <typename T>
class SITopk {
public:
    __aicore__ inline void operator()(LocalTensor<uint32_t>& outputIdxLocal,
                                      LocalTensor<T>& inputLocal,
                                      uint32_t s2SeqLen)
    {
    }
};

template <>
class SITopk<uint32_t> {
public:
    __aicore__ inline uint32_t GetSharedTmpBufferSize()
    {
        // 2 * SICommon::Align(topK, HISTOGRAM_BIN_NUM): 两块hisIndexLocal
        // 5 * 256：histogramsLocal + idxLocal[0-3]
        // 64：nkValueLocal
        uint64_t bufferSize1 = (2 * SICommon::Align(topK, HISTOGRAM_BIN_NUM) +
                                5 * HISTOGRAM_BIN_NUM + NK_VALUE_SIZE) * sizeof(uint32_t);
        // SICommon::Align(topK, HISTOGRAM_BIN_NUM) + trunkLen：tmpIndexLocal
        uint64_t bufferSize2 = (SICommon::Align(topK, HISTOGRAM_BIN_NUM) + trunkLen) * sizeof(uint32_t);
        return bufferSize1 + bufferSize2;
    }

    __aicore__ inline uint32_t GetReUseSharedTmpBufferSize()
    {
        // 1 * 256：histogramsLocal
        // 64：nkValueLocal
        // 其余皆复用
        uint64_t bufferSize1 = (HISTOGRAM_BIN_NUM + NK_VALUE_SIZE) * sizeof(uint32_t);
        // SICommon::Align(topK, HISTOGRAM_BIN_NUM) + trunkLen：tmpIndexLocal
        uint64_t bufferSize2 = (SICommon::Align(topK, HISTOGRAM_BIN_NUM) + trunkLen) * sizeof(uint32_t);
        return bufferSize1 + bufferSize2;
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
        LocalTensor<uint32_t> hisIndexLocal2 = hisIndexLocal1[SICommon::Align(topK, HISTOGRAM_BIN_NUM)];
        hisIndexLocal[0] = hisIndexLocal1;
        hisIndexLocal[1] = hisIndexLocal2;
        histogramsLocal = hisIndexLocal2[SICommon::Align(topK, HISTOGRAM_BIN_NUM)];
        idx0Local = histogramsLocal[HISTOGRAM_BIN_NUM];
        idx1Local = idx0Local[HISTOGRAM_BIN_NUM];
        idx2Local = idx1Local[HISTOGRAM_BIN_NUM];
        idx3Local = idx2Local[HISTOGRAM_BIN_NUM];
        nkValueLocal = idx3Local[HISTOGRAM_BIN_NUM];
        tmpIndexLocal = nkValueLocal[NK_VALUE_SIZE];
    }

    __aicore__ inline void InitBuffers(LocalTensor<uint32_t>& sharedTmpBuffer,
                                       LocalTensor<uint32_t>& indicesOutLocal)
    {
        LocalTensor<uint32_t> hisIndexLocal2 = indicesOutLocal;
        hisIndexLocal[1] = hisIndexLocal2;
        histogramsLocal = sharedTmpBuffer[0];
        nkValueLocal = histogramsLocal[HISTOGRAM_BIN_NUM];
        tmpIndexLocal = nkValueLocal[NK_VALUE_SIZE];
    }

    __aicore__ inline void operator()(const LocalTensor<uint32_t>& mrgValueLocal,
                                      const LocalTensor<uint32_t>& indicesOutLocal,
                                      const LocalTensor<uint32_t>& hisValueLocal,
                                      const LocalTensor<uint32_t>& reuseMm1ResLocal,
                                      const LocalTensor<uint32_t>& reuseGlobalIndexLocal,
                                      uint32_t s2SeqLen, uint32_t loopIdx, uint32_t s2LoopNum)
    {
        // uint32_t时UB需要进行复用，此处将每8位的targetbin复用mm1res
        idx0Local = reuseMm1ResLocal;
        idx1Local = idx0Local[HISTOGRAM_BIN_NUM];
        idx2Local = idx1Local[HISTOGRAM_BIN_NUM];
        idx3Local = idx2Local[HISTOGRAM_BIN_NUM];
        // hisIndexLocal[0]复用globalIndexLocal
        hisIndexLocal[0] = reuseGlobalIndexLocal;

        if (s2LoopNum == 1) {
            SITopkb32gather::SiTopKVF<false>(tmpIndexLocal, hisValueLocal, mrgValueLocal, histogramsLocal,
                                             idx0Local, idx1Local, idx2Local, idx3Local,
                                             nkValueLocal, topK, s2SeqLen);
            PipeBarrier<PIPE_V>();
            AscendC::DataCopy(indicesOutLocal, tmpIndexLocal, SICommon::Align(topK, HISTOGRAM_BIN_NUM));
        } else {
            if (loopIdx == 0) {
                SITopkb32gather::SiTopKVF<true>(tmpIndexLocal, hisValueLocal, mrgValueLocal, histogramsLocal,
                                                idx0Local, idx1Local, idx2Local, idx3Local,
                                                nkValueLocal, topK, s2SeqLen);
                PipeBarrier<PIPE_V>();
                AscendC::DataCopy(hisIndexLocal[1], tmpIndexLocal, SICommon::Align(topK, HISTOGRAM_BIN_NUM));
            } else {
                SITopkb32gather::SiTopKVF<true>(tmpIndexLocal, hisValueLocal, mrgValueLocal, histogramsLocal,
                                                idx0Local, idx1Local, idx2Local, idx3Local,
                                                nkValueLocal, topK, s2SeqLen);
                PipeBarrier<PIPE_V>();
                SITopkb32gather::SiTopKGatherVF(hisIndexLocal[1], hisValueLocal,
                                                mrgValueLocal, tmpIndexLocal, hisIndexLocal[0],
                                                topK, loopIdx * trunkLen - SICommon::Align(topK, HISTOGRAM_BIN_NUM),
                                                s2SeqLen);
            }
        }
    }
private:
    LocalTensor<uint32_t> hisIndexLocal[2]; // 每trunkLen长度的s2选出的topK个索引
    LocalTensor<uint32_t> histogramsLocal;  // 直方图的临时Buf 256 * 4B
    LocalTensor<uint32_t> idx0Local;        // 输入数据第1个8位Buf 256 * 4B
    LocalTensor<uint32_t> idx1Local;        // 输入数据第2个8位Buf 256 * 4B
    LocalTensor<uint32_t> idx2Local;        // 输入数据第3个8位Buf 256 * 4B
    LocalTensor<uint32_t> idx3Local;        // 输入数据第4个8位Buf 256 * 4B
    LocalTensor<uint32_t> nkValueLocal;     // next_k 暂存Buf 64 * 4B
    LocalTensor<uint32_t> tmpIndexLocal;    // 每trunkLen + topK的临时index
    uint32_t topK = 512;
    uint32_t trunkLen = 8192;
};
}
#endif
