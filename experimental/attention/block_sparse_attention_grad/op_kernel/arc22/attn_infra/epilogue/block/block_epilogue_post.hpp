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
 * \file block_epliogue_post.h
 * \brief Block Epliogue Post Kernel Implementation
 */

#ifndef CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_POST_HPP
#define CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_POST_HPP

#include "../../../attn_infra/arch/bsag_resource.hpp"
#include "../../../attn_infra/epilogue/bsag_epilogue_dispatch_policy.hpp"
#include "kernel_operator.h"

using namespace AscendC;

namespace NpuArch::Epilogue::Block {

struct ShapeBnsd {
    uint64_t b;
    uint64_t n;
    uint64_t s;
    uint64_t d;
};

template <uint32_t INPUT_LAYOUT, typename OutputDtype_, typename UpdateType_, typename InputType_>
class BlockPost {
public:
    using DispatchPolicy = EpilogueAtlasA2FAGPre;
    using ArchTag = typename DispatchPolicy::ArchTag;

    struct Params {
        GM_ADDR dq;
        GM_ADDR dk;
        GM_ADDR dv;
        GM_ADDR dqWrk;
        GM_ADDR dkWrk;
        GM_ADDR dvWrk;
        GM_ADDR tilingData;
        GM_ADDR actualSeqQlen;
        GM_ADDR actualSeqKvlen;

        __aicore__ inline Params() {}

        __aicore__ inline Params(GM_ADDR dq_, GM_ADDR dk_, GM_ADDR dv_, GM_ADDR dqWrk_, GM_ADDR dkWrk_, GM_ADDR dvWrk_,
                                 GM_ADDR tilingData_, GM_ADDR actualSeqQlen_, GM_ADDR actualSeqKvlen_)
            : dq(dq_),
              dk(dk_),
              dv(dv_),
              dqWrk(dqWrk_),
              dkWrk(dkWrk_),
              dvWrk(dvWrk_),
              tilingData(tilingData_),
              actualSeqQlen(actualSeqQlen_),
              actualSeqKvlen(actualSeqKvlen_)
        {}
    };

    NpuArch::Arch::Resource<ArchTag> resource;
    constexpr static uint32_t BUFFER_NUM = 2;
    constexpr static uint64_t INPUT_NUM = 2;
    constexpr static uint64_t BNSD = 1;
    constexpr static uint64_t TND = 0;
    constexpr static uint32_t WORKSPACE_NUM_ALIGN = 256;
    constexpr static uint32_t BASE_BLOCK_BYTE = 32;
    constexpr static uint32_t POST_COEX_NODE = 3;
    constexpr static uint32_t FP16_BLOCK_NUMS = 16;
    constexpr static int32_t DQ = 1;
    constexpr static int32_t DK = -1;
    constexpr static int32_t DV = -2;

    GM_ADDR actualSeqQlen;
    GM_ADDR actualSeqKvlen;
    GlobalTensor<OutputDtype_> dqGm, dkGm, dvGm;
    GlobalTensor<float> dqWorkSpaceGm, dkWorkSpaceGm, dvWorkSpaceGm;
    LocalTensor<float> input[BUFFER_NUM];
    LocalTensor<OutputDtype_> output[BUFFER_NUM];

    uint64_t usedCoreNum = 0;
    uint64_t cBlockIdx = 0;
    float scaleValue = 0;

    uint64_t computeS1 = 0;
    uint64_t computeS2 = 0;
    uint64_t actualCol = 0;
    uint64_t curBatch1 = 0;
    uint64_t curBatch2 = 0; // k
    uint64_t curN1Idx = 0;  // q_n
    uint64_t curN2Idx = 0;  // k_n
    uint64_t curS1Idx = 0;  // q_s
    uint64_t curS2Idx = 0;  // v_s

    uint64_t curT1Idx = 0;
    uint64_t transpseQStride = 0;
    uint64_t transpseKvStride = 0;

    uint64_t dqOffset = 0;
    uint64_t dkvOffset = 0;

    uint64_t ubBaseSize = 0;
    uint64_t ubBasePreBufferSize = 0;
    uint64_t b = 0;
    uint64_t n1 = 0;
    uint64_t n2 = 0;
    uint64_t s1 = 0;
    uint64_t s2 = 0;
    uint64_t d = 0;

    __aicore__ inline BlockPost(Params const &params)
    {
        cBlockIdx = GetBlockIdx();
        __gm__ BlockSparseAttentionGradTilingData *tilingData =
            reinterpret_cast<__gm__ BlockSparseAttentionGradTilingData *>(params.tilingData);
        // KERNEL_TYPE_MIX_AIC_1_2 exposes two contiguous AIV block ids for
        // every AIC.  Give every AIV a disjoint output range so scale/cast is
        // parallelized without duplicate GM reads and writes.
        usedCoreNum = tilingData->usedVecCoreNum;
        if (cBlockIdx >= usedCoreNum) {
            return;
        }

        b = tilingData->batch;
        n1 = tilingData->numHeads;
        n2 = tilingData->kvHeads;
        s1 = tilingData->maxQSeqlen;
        s2 = tilingData->maxKvSeqlen;
        d = tilingData->headDim;
        scaleValue = tilingData->scaleValue;
        ubBaseSize = tilingData->postUbBaseSize / BUFFER_NUM * BUFFER_NUM / WORKSPACE_NUM_ALIGN * WORKSPACE_NUM_ALIGN;

        uint64_t qPostSize = tilingData->dqSize / d;
        uint64_t kvPostSize = tilingData->dkvSize / d;

        uint64_t qPostBlockTotal = qPostSize;                         // 把d前面合洲，总共的行数
        uint64_t qPostBlockEeachCore = qPostBlockTotal / usedCoreNum; // 每个核处理的行数
        uint64_t qPostBlockNumEeachCore = qPostBlockEeachCore * d;    // 每个核处理的元素数量
        uint64_t qPostTailNum = qPostBlockTotal % usedCoreNum;        // 剩余的行数，给尾核处理

        uint64_t kvPostBlockTotal = kvPostSize;                         // 把d前面合洲，总共的行数
        uint64_t kvPostBlockEeachCore = kvPostBlockTotal / usedCoreNum; // 每个核处理的行数
        uint64_t kvPostBlockNumEeachCore = kvPostBlockEeachCore * d;    // 每个核处理的元素数量
        uint64_t kvPostTailNum = kvPostBlockTotal % usedCoreNum;        // 剩余的行数，给尾核处理

        actualSeqQlen = params.actualSeqQlen;
        actualSeqKvlen = params.actualSeqKvlen;
        if constexpr (INPUT_LAYOUT == TND) {
            transpseQStride = (n1 * d - d) * sizeof(OutputDtype_);
            transpseKvStride = (n2 * d - d) * sizeof(OutputDtype_);
        } else if constexpr (INPUT_LAYOUT == BNSD) {
            transpseQStride = 0;
            transpseKvStride = 0;
        }

        computeS1 = cBlockIdx == usedCoreNum - 1 ? (qPostTailNum + qPostBlockEeachCore) :
                                                   qPostBlockEeachCore; // 当前核需要计算的dq的行数
        dqOffset = ((uint64_t)cBlockIdx) * qPostBlockNumEeachCore;      // 当前核dq的偏移元素个数
        computeS2 = cBlockIdx == usedCoreNum - 1 ? (kvPostBlockEeachCore + kvPostTailNum) :
                                                   kvPostBlockEeachCore; // 当前核需要计算的dkv的行数
        dkvOffset = ((uint64_t)cBlockIdx) * kvPostBlockNumEeachCore;     // 当前核dkv的偏移元素个数

        dqWorkSpaceGm.SetGlobalBuffer((__gm__ float *)params.dqWrk + dqOffset);
        dkWorkSpaceGm.SetGlobalBuffer((__gm__ float *)params.dkWrk + dkvOffset);
        dvWorkSpaceGm.SetGlobalBuffer((__gm__ float *)params.dvWrk + dkvOffset);

        if constexpr (INPUT_LAYOUT == TND) {
            dqGm.SetGlobalBuffer((__gm__ OutputDtype_ *)params.dq + dqOffset);
            dkGm.SetGlobalBuffer((__gm__ OutputDtype_ *)params.dk + dkvOffset);
            dvGm.SetGlobalBuffer((__gm__ OutputDtype_ *)params.dv + dkvOffset);
        } else if constexpr (INPUT_LAYOUT == BNSD) {
            dqGm.SetGlobalBuffer((__gm__ OutputDtype_ *)params.dq);
            dkGm.SetGlobalBuffer((__gm__ OutputDtype_ *)params.dk);
            dvGm.SetGlobalBuffer((__gm__ OutputDtype_ *)params.dv);
        }

        ubBasePreBufferSize = ubBaseSize / BUFFER_NUM / BASE_BLOCK_BYTE * BASE_BLOCK_BYTE; // double buffer
        for (uint64_t i = 0; i < BUFFER_NUM; i++) {
            input[i] = resource.ubBuf.template GetBufferByByte<float>(ubBasePreBufferSize * 3 * i);
            output[i] = resource.ubBuf.template GetBufferByByte<OutputDtype_>(ubBasePreBufferSize * 2 +
                                                                              ubBasePreBufferSize * 3 * i);
        }

        // 获取当前核的dq dk dv 对应的索引序号
        struct ShapeBnsd qShape {
            b, n1, s1, d
        };
        InitIndex(dqOffset, curS1Idx, actualSeqQlen, curBatch1, curN1Idx, curS1Idx, qShape);
        struct ShapeBnsd kvShape {
            b, n2, s2, d
        };
        InitIndex(dkvOffset, curS2Idx, actualSeqKvlen, curBatch2, curN2Idx, curS2Idx, kvShape);
    }

    // Decode an element offset into batch/head/sequence indices.
    // seqS contains per-batch lengths, not cumulative offsets; curS receives the selected length.
    __aicore__ inline void InitIndex(uint64_t startIdx, uint64_t &curS, GM_ADDR seqS, uint64_t &bIdx, uint64_t &nIdx,
                                     uint64_t &sIdx, struct ShapeBnsd shape)
    {
        if constexpr (INPUT_LAYOUT == TND) {
            uint64_t prefixSum = 0;
            for (uint64_t bDimIdx = 0; bDimIdx < shape.b; bDimIdx++) {
                uint64_t curBatchLen = ((__gm__ int64_t *)seqS)[bDimIdx];
                uint64_t curBatchElements = curBatchLen * shape.n * shape.d;

                if (startIdx < prefixSum + curBatchElements) {
                    bIdx = bDimIdx;
                    curS = curBatchLen;

                    uint64_t bTail = startIdx - prefixSum;
                    nIdx = bTail / (curS * shape.d);
                    uint64_t nTail = bTail % (curS * shape.d);
                    sIdx = nTail / shape.d;
                    break;
                }
                prefixSum += curBatchElements;
            }
        } else {
            bIdx = startIdx / (shape.n * shape.s * shape.d);
            uint64_t bTail = startIdx % (shape.n * shape.s * shape.d);
            nIdx = bTail / (shape.s * shape.d);
            uint64_t nTail = bTail % (shape.s * shape.d);
            sIdx = nTail / shape.d;
        }
    }

    __aicore__ inline ~BlockPost() {}

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
        Process();
    }

    __aicore__ inline void ProcessOut(uint64_t conputeS, int32_t qkvFlag)
    {
        uint64_t ubBaseSizeNum = ubBasePreBufferSize / sizeof(OutputDtype_); // 一块buffer处理的元素数量
        uint64_t singleLoopSCount = ubBaseSizeNum / d;
        uint64_t loopTimes = static_cast<uint64_t>(CeilDiv(conputeS, singleLoopSCount));
        uint64_t tailS = conputeS % singleLoopSCount;
        uint64_t curSIdx = (qkvFlag > 0) ? curS1Idx : curS2Idx;
        uint64_t curBatchIdx = (qkvFlag > 0) ? curBatch1 : curBatch2;
        GlobalTensor<float> inGm = dqWorkSpaceGm;
        if (qkvFlag == -1) {
            inGm = dkWorkSpaceGm;
        } else if (qkvFlag == -2) {
            inGm = dvWorkSpaceGm;
        }

        int64_t ping = 0;
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
        for (uint64_t i = 0; i < loopTimes; i++) {
            auto event_id = ping ? EVENT_ID0 : EVENT_ID1;
            uint64_t sCount = singleLoopSCount;
            uint64_t totalSCout = i * singleLoopSCount;
            uint64_t gmOffset = totalSCout * d;
            if (i == loopTimes - 1 && tailS != 0) {
                sCount = tailS;
            }

            wait_flag(PIPE_MTE3, PIPE_MTE2, event_id);

            DataCopy(input[ping], inGm[gmOffset], sCount * d); // d 为32b 对齐场景

            set_flag(PIPE_MTE2, PIPE_V, event_id);
            wait_flag(PIPE_MTE2, PIPE_V, event_id);

            if (qkvFlag != DV) {
                Muls(input[ping], input[ping], (float)scaleValue, sCount * d);
                AscendC::PipeBarrier<PIPE_V>();
            }

            AscendC::LocalTensor<float> srcLocal = input[ping];
            AscendC::LocalTensor<OutputDtype_> dstLocal = output[ping];
            Cast(dstLocal, srcLocal, AscendC::RoundMode::CAST_ROUND, sCount * d);

            set_flag(PIPE_V, PIPE_MTE3, event_id);
            wait_flag(PIPE_V, PIPE_MTE3, event_id);

            if constexpr (INPUT_LAYOUT == TND) {
                GlobalTensor<OutputDtype_> outGm = dqGm;
                if (qkvFlag == DK) {
                    outGm = dkGm;
                } else if (qkvFlag == DV) {
                    outGm = dvGm;
                }
                DataCopy(outGm[gmOffset], dstLocal, sCount * d);
            } else {
                // The accumulation workspace and the public output are both
                // flattened BNSD tensors.  Each AIV owns one contiguous row
                // range, so no B/N/S boundary walk or strided copy is needed.
                // Keeping the absolute per-core base explicit also lets one
                // large DMA span adjacent heads without changing layout.
                GlobalTensor<OutputDtype_> outGm = dqGm;
                uint64_t outOffset = dqOffset + gmOffset;
                if (qkvFlag == DK) {
                    outGm = dkGm;
                    outOffset = dkvOffset + gmOffset;
                } else if (qkvFlag == DV) {
                    outGm = dvGm;
                    outOffset = dkvOffset + gmOffset;
                }
                DataCopy(outGm[outOffset], dstLocal, sCount * d);
            }

            set_flag(PIPE_MTE3, PIPE_MTE2, event_id);

            if (BUFFER_NUM == 2) {
                ping = 1 - ping;
            }
        }
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    }

    __aicore__ inline void Process()
    {
        ProcessOut(computeS1, DQ);
        ProcessOut(computeS2, DK);
        // Only TND advances logical B/N/S indices while scattering DK.  BNSD
        // uses the contiguous workspace offset directly and has no index
        // state to restore before DV.
        if constexpr (INPUT_LAYOUT == TND) {
            curS2Idx = 0;
            curBatch2 = 0;
            curN2Idx = 0;
            struct ShapeBnsd kvShape {
                b, n2, s2, d
            };
            InitIndex(dkvOffset, curS2Idx, actualSeqKvlen, curBatch2, curN2Idx, curS2Idx, kvShape);
        }
        ProcessOut(computeS2, DV);
    }
};

} // namespace NpuArch::Epilogue::Block

#endif // CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_POST_HPP
