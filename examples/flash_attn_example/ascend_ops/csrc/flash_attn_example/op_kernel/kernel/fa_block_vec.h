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
 * \file fa_block_vec.h
 * \brief
 */
#ifndef FA_BLOCK_VEC_H
#define FA_BLOCK_VEC_H

#include "kernel_operator.h"

#include "adv_api/activation/softmax.h"
#include "../vector/vf/vf_softmax.h"
#include "../vector/vf/vf_rescale.h"
#include "../vector/vf/vf_combine.h"
#include "../fa_kernel_public.h"
#include "../vector/vector.h"
#include "../memcopy/memory_copy.h"

using namespace AscendC;
using namespace FaVectorApi;
using namespace AscendC::Impl::Detail;
using namespace fa_kernel::hardware;
using namespace fa_kernel::config;
using namespace fa_kernel::layout;
using namespace fa_kernel;
using namespace fa_kernel::util;
using namespace fa_base_vector;

namespace BaseApi {

template <typename INPUT_T, typename T, typename OUTPUT_T, LayOutTypeEnum layout = LayOutTypeEnum::None,
          LayOutTypeEnum outLayout = LayOutTypeEnum::None, S1TemplateType s1TemplateType = S1TemplateType::Aligned128,
          S2TemplateType s2TemplateType = S2TemplateType::Aligned128,
          DTemplateType dTemplateType = DTemplateType::Aligned128,
          DTemplateType dVTemplateType = DTemplateType::Aligned128, bool hasAttn = false, bool isCombine = false>
class FABlockVec {
public:
    /* =================编译期常量的基本块信息================= */
    static constexpr uint32_t mBaseSize = (uint32_t)s1TemplateType;
    static constexpr uint32_t s2BaseSize = (uint32_t)s2TemplateType;
    static constexpr uint32_t vec1HalfS1BaseSize = mBaseSize >> 1;
    static constexpr uint32_t vec1Srcstride = (mBaseSize >> 1) + 1;
    static constexpr uint32_t dTemplateAlign64 = Align64Func((uint16_t)dVTemplateType);

    static constexpr uint32_t DB = 2;
    static constexpr uint32_t PRELOAD_N = 2; // C1 C1 C2
    static constexpr bool HAS_MASK = hasAttn;
    static constexpr bool COMBINE = isCombine;

    static constexpr uint32_t initOutputEventId = 0U; // attnOut和lse，刷无效行会用到剩余ub，需要加同步

    static constexpr LAYOUT_Q MASK_LAYOUT = LAYOUT_Q::SG;

    static constexpr T BOOL_ATTN_MASK_SCALAR_VALUE = -1000000000000.0; // 用于mask为bool类型
    uint32_t negativeIntScalar = *((uint32_t *)&BOOL_ATTN_MASK_SCALAR_VALUE);

    using mm2ResPos = Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>;

    using attnMaskGmType = typename std::conditional<hasAttn, GlobalTensor<uint8_t>, int8_t>::type;

    using combineGmType = typename std::conditional<COMBINE, GlobalTensor<float>, int8_t>::type;

    using ConstInfoX = ConstInfo_t<FaKernelType::NO_QUANT>;

    using OUT_T = OUTPUT_T;

    // gm
    TPipe *tPipe = nullptr;
    GlobalTensor<OUTPUT_T> attentionOutGm;
    GlobalTensor<float> softmaxLseGm;

    GlobalTensor<uint64_t> actualSeqLengthsGmQ;

    attnMaskGmType attnMaskGmInt;

    combineGmType accumOutGm;
    combineGmType softmaxCombineSumGm;
    combineGmType softmaxCombineMaxGm;

    // ub
    TBuf<> commonTBuf; // common的复用空间
    TQue<QuePosition::VECOUT, 1> stage1OutQue[2];
    TQue<QuePosition::VECIN, 1> attnMaskInQue[2];
    TQue<QuePosition::VECIN, 1> pseInQue;
    TBuf<> stage2OutBuf;
    TEventID mte3ToVId[2]; // 存放MTE3_V的eventId, 2份表示可能存在pingpong
    TEventID vToMte3Id[2]; // 存放V_MTE3的eventId, 2份表示可能存在pingpong
    TBuf<> softmaxMaxBuf[PRELOAD_N + 1];
    TBuf<> softmaxSumBuf[PRELOAD_N + 1];
    TBuf<> softmaxExpBuf[PRELOAD_N + 1];
    TBuf<> mm2InBuf;
    /* 用来做Broadcast[S1,1]->[S2,8]的临时UB区域 */
    TQue<QuePosition::VECOUT, 1> maxBrdcst;
    TQue<QuePosition::VECOUT, 1> sumBrdcst;

    const ConstInfoX &constInfo;
    T negativeFloatScalar;

    // ==================== Functions ======================
    __aicore__ inline FABlockVec(ConstInfoX &constInfo) : constInfo(constInfo){};

    __aicore__ inline void InitVecBlock(TPipe *pipe, __gm__ uint8_t *actualSeqQlenAddr,
                                        __gm__ uint8_t *actualSeqKvlenAddr, __gm__ uint8_t *attnMask,
                                        __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace)
    {
        tPipe = pipe;
        uint32_t tmp1 = NEGATIVE_MIN_VALUE_FP32;
        this->negativeFloatScalar = *((T *)&tmp1);

        InitVecInput(actualSeqQlenAddr, actualSeqKvlenAddr, attnMask, attentionOut, workspace);
    }

    __aicore__ inline void InitVecInput(__gm__ uint8_t *actualSeqQlenAddr, __gm__ uint8_t *actualSeqKvlenAddr,
                                        __gm__ uint8_t *attnMask, __gm__ uint8_t *attentionOut,
                                        __gm__ uint8_t *workspace)
    {
        this->attentionOutGm.SetGlobalBuffer((__gm__ OUTPUT_T *)attentionOut);

        actualSeqLengthsGmQ.SetGlobalBuffer((__gm__ uint64_t *)actualSeqQlenAddr, constInfo.actualSeqLenSize);

        if constexpr (hasAttn) {
            attnMaskGmInt.SetGlobalBuffer((__gm__ uint8_t *)attnMask);
        }

        if constexpr (COMBINE) {
            accumOutGm.SetGlobalBuffer((__gm__ float *)workspace);
            softmaxCombineSumGm.SetGlobalBuffer((__gm__ float *)workspace + constInfo.accumOutSize);
            softmaxCombineMaxGm.SetGlobalBuffer((__gm__ float *)workspace + constInfo.accumOutSize +
                                                constInfo.logSumExpSize);
        }
    }

    __aicore__ inline void ProcessVec1(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
                                       Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf,
                                       RunInfoX &runInfo)
    {
        bmm1ResBuf.WaitCrossCore();
        if (unlikely(runInfo.actVecMSize == 0)) {
            bmm1ResBuf.SetCrossCore();
            outputBuf.SetCrossCore();
            return;
        }

        ProcessVec1Nd(outputBuf, bmm1ResBuf, runInfo);
    }

    __aicore__ inline void ClearOutput()
    {
        SetFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId); // 释放剩余ub
        InitOutputSingleCore();
        WaitFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
        SyncAll();
    }

    __aicore__ inline void InitOutputSingleCore()
    {
        int64_t tSize = constInfo.bSize * constInfo.s1Size;
        int64_t totalOutputSize = tSize * constInfo.n2Size * constInfo.gSize * constInfo.dSizeV;
        int64_t singleCoreSize =
            (totalOutputSize + (2 * constInfo.coreNum) - 1) / (2 * constInfo.coreNum); // 2 means c:v = 1:2
        int64_t tailSize = totalOutputSize - constInfo.aivIdx * singleCoreSize;
        int64_t singleInitOutputSize = tailSize < singleCoreSize ? tailSize : singleCoreSize;

        if (singleInitOutputSize > 0) {
            WaitFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
            matmul::InitOutput<OUT_T>(attentionOutGm[constInfo.aivIdx * singleCoreSize], singleInitOutputSize, 0);
            SetFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
        }
    }

    // =================================Private Functions=================================
    __aicore__ inline void SoftmaxDataCopyOut(RunInfoX &runInfo, LocalTensor<float> &sumUb, LocalTensor<float> &maxUb)
    {
        if constexpr (COMBINE) {
            if (runInfo.isS2SplitCore) {
                ComputeLogSumExpAndCopyToGm(runInfo, sumUb, maxUb);
            }
        }
    }

    __aicore__ inline void ProcessVec1Nd(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
                                         Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf,
                                         RunInfoX &runInfo)
    {
        LocalTensor<uint8_t> attnMaskUb;
        LocalTensor<uint8_t> attnMaskUbPre;
        if constexpr (hasAttn == true) {
            attnMaskUb = this->attnMaskInQue[runInfo.loop % DB].template AllocTensor<uint8_t>();
            AttnMaskCopyIn(attnMaskUb, 0, runInfo.actVecMSize, runInfo); // 全量拷贝
        }

        LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.mloop % (PRELOAD_N + 1)].template Get<float>();
        LocalTensor<float> maxUb = this->softmaxMaxBuf[runInfo.mloop % (PRELOAD_N + 1)].template Get<float>();
        LocalTensor<float> expUb = this->softmaxExpBuf[runInfo.loop % (PRELOAD_N + 1)].template Get<T>();
        LocalTensor<uint8_t> apiTmpBuffer;

        apiTmpBuffer = this->commonTBuf.template Get<uint8_t>();

        int64_t stage1Offset = 0;
        stage1Offset = runInfo.loop % DB;

        LocalTensor<T> mmRes = bmm1ResBuf.template GetTensor<T>();
        auto stage1CastTensor = this->stage1OutQue[stage1Offset].template AllocTensor<INPUT_T>();
        if (runInfo.isFirstS2Loop) {
            if (likely(runInfo.actSingleLoopS2Size == 128)) {
                ProcessVec1Vf<T, INPUT_T, false, mBaseSize, s2BaseSize, EQ_128, hasAttn>(
                    stage1CastTensor, nullptr, sumUb, maxUb, mmRes, expUb, sumUb, maxUb, attnMaskUb, apiTmpBuffer,
                    runInfo.actVecMSize, runInfo.actSingleLoopS2Size, static_cast<T>(constInfo.softmaxScale),
                    negativeFloatScalar);
            } else if (runInfo.actSingleLoopS2Size <= 64) {
                ProcessVec1Vf<T, INPUT_T, false, mBaseSize, s2BaseSize, GT_0_AND_LTE_64, hasAttn>(
                    stage1CastTensor, nullptr, sumUb, maxUb, mmRes, expUb, sumUb, maxUb, attnMaskUb, apiTmpBuffer,
                    runInfo.actVecMSize, runInfo.actSingleLoopS2Size, static_cast<T>(constInfo.softmaxScale),
                    negativeFloatScalar);
            } else if (runInfo.actSingleLoopS2Size < 128) {
                ProcessVec1Vf<T, INPUT_T, false, mBaseSize, s2BaseSize, GT_64_AND_LTE_128, hasAttn>(
                    stage1CastTensor, nullptr, sumUb, maxUb, mmRes, expUb, sumUb, maxUb, attnMaskUb, apiTmpBuffer,
                    runInfo.actVecMSize, runInfo.actSingleLoopS2Size, static_cast<T>(constInfo.softmaxScale),
                    negativeFloatScalar);
            }
        } else {
            if (likely(runInfo.actSingleLoopS2Size == 128)) {
                ProcessVec1Vf<T, INPUT_T, true, mBaseSize, s2BaseSize, EQ_128, hasAttn>(
                    stage1CastTensor, nullptr, sumUb, maxUb, mmRes, expUb, sumUb, maxUb, attnMaskUb, apiTmpBuffer,
                    runInfo.actVecMSize, runInfo.actSingleLoopS2Size, static_cast<T>(constInfo.softmaxScale),
                    negativeFloatScalar);
            } else if (runInfo.actSingleLoopS2Size <= 64) {
                ProcessVec1Vf<T, INPUT_T, true, mBaseSize, s2BaseSize, GT_0_AND_LTE_64, hasAttn>(
                    stage1CastTensor, nullptr, sumUb, maxUb, mmRes, expUb, sumUb, maxUb, attnMaskUb, apiTmpBuffer,
                    runInfo.actVecMSize, runInfo.actSingleLoopS2Size, static_cast<T>(constInfo.softmaxScale),
                    negativeFloatScalar);
            } else {
                ProcessVec1Vf<T, INPUT_T, true, mBaseSize, s2BaseSize, GT_64_AND_LTE_128, hasAttn>(
                    stage1CastTensor, nullptr, sumUb, maxUb, mmRes, expUb, sumUb, maxUb, attnMaskUb, apiTmpBuffer,
                    runInfo.actVecMSize, runInfo.actSingleLoopS2Size, static_cast<T>(constInfo.softmaxScale),
                    negativeFloatScalar);
            }
        }
        bmm1ResBuf.SetCrossCore();
        if constexpr (hasAttn) {
            this->attnMaskInQue[runInfo.loop % DB].template FreeTensor(attnMaskUb);
        }

        // ===================DataCopy to L1 ====================
        this->stage1OutQue[stage1Offset].template EnQue(stage1CastTensor);
        this->stage1OutQue[stage1Offset].template DeQue<INPUT_T>();
        LocalTensor<INPUT_T> mm2AL1Tensor;
        mm2AL1Tensor = outputBuf.GetTensor<INPUT_T>();

        if (likely(runInfo.actVecMSize != 0)) {
            DataCopy(mm2AL1Tensor[constInfo.subBlockIdx * (BLOCK_BYTES / sizeof(INPUT_T)) *
                                  (runInfo.actMSize - runInfo.actVecMSize)],
                     stage1CastTensor,
                     {s2BaseSize / 16, (uint16_t)runInfo.actVecMSize, (uint16_t)(vec1Srcstride - runInfo.actVecMSize),
                      (uint16_t)(mBaseSize - runInfo.actVecMSize)});
        }
        this->stage1OutQue[stage1Offset].template FreeTensor(stage1CastTensor);

        outputBuf.SetCrossCore();
        // ======================================================
        if (!runInfo.isFirstS2Loop) {
            UpdateExpSumAndExpMax<T>(sumUb, maxUb, expUb, sumUb, maxUb, apiTmpBuffer, runInfo.actVecMSize);
        }

        if (unlikely(runInfo.isLastS2Loop)) {
            SoftmaxDataCopyOut(runInfo, sumUb, maxUb);
        }
    }

    __aicore__ inline void ProcessVec2OnUb(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm2ResBuf,
                                           RunInfoX &runInfo)
    {
        if (unlikely(runInfo.actVecMSize == 0)) {
            bmm2ResBuf.SetCrossCore();
            return;
        }

        int64_t vec2CalcSize = runInfo.actVecMSize * dTemplateAlign64;

        LocalTensor<T> vec2ResUb = this->stage2OutBuf.template Get<T>();
        LocalTensor<T> mmRes = bmm2ResBuf.template GetTensor<T>();
        WaitFlag<HardEvent::MTE3_V>(mte3ToVId[0]);
        if (unlikely(runInfo.isFirstS2Loop)) {
            DataCopy(vec2ResUb, mmRes, vec2CalcSize);
        } else {
            LocalTensor<T> expUb = softmaxExpBuf[runInfo.loop % (PRELOAD_N + 1)].template Get<T>();
            LocalTensor<T> pScaleUb;

            float deSCalePreVValue = 1.0f;
            if (!runInfo.isLastS2Loop) {
                FlashUpdateNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, false>(
                    vec2ResUb, mmRes, vec2ResUb, expUb, pScaleUb, runInfo.actVecMSize, dTemplateAlign64, 1.0, 1.0);
            } else {
                LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.mloop % (PRELOAD_N + 1)].template Get<float>();
                FlashUpdateLastNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, false>(vec2ResUb, mmRes, vec2ResUb, expUb,
                                                                                  pScaleUb, sumUb, runInfo.actVecMSize,
                                                                                  dTemplateAlign64, 1.0, 1.0);
            }
        }
        bmm2ResBuf.SetCrossCore();
        if (runInfo.isLastS2Loop) {
            if (unlikely(runInfo.isFirstS2Loop)) {
                LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.mloop % (PRELOAD_N + 1)].template Get<float>();
                LastDivNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64>(vec2ResUb, vec2ResUb, sumUb, runInfo.actVecMSize,
                                                                   (uint16_t)dTemplateAlign64, 0.0F);
            }
            CopyOutAttentionOut(runInfo, vec2ResUb, 0, runInfo.actVecMSize);
        }
        SetFlag<HardEvent::MTE3_V>(mte3ToVId[0]);
    }

    __aicore__ inline void Bmm2ResCastAndCopyOut(RunInfoX &runInfo, LocalTensor<T> &vec2ResUb, uint32_t mStartVec,
                                                 uint32_t mDealSize)
    {
        LocalTensor<OUTPUT_T> attnOut;
        int64_t dSizeAligned64 = (int64_t)dVTemplateType;

        attnOut.SetAddr(vec2ResUb.address_);

        RowInvalid(vec2ResUb, mStartVec, mDealSize, runInfo, dSizeAligned64);
        Cast(attnOut, vec2ResUb, RoundMode::CAST_ROUND, mDealSize * dSizeAligned64);
        SetFlag<HardEvent::V_MTE3>(vToMte3Id[0]);
        WaitFlag<HardEvent::V_MTE3>(vToMte3Id[0]);

        Bmm2DataCopyOutTrans(runInfo, attnOut, mStartVec, mDealSize);
    }

    __aicore__ inline void CopyOutAttentionOut(RunInfoX &runInfo, LocalTensor<T> &vec2ResUb, uint32_t mStartVec,
                                               uint32_t mDealSize)
    {
        if constexpr (COMBINE) {
            if (runInfo.isS2SplitCore) {
                Bmm2ResForCombineCopyOut(runInfo, vec2ResUb, mStartVec, mDealSize);
            } else {
                Bmm2ResCastAndCopyOut(runInfo, vec2ResUb, mStartVec, mDealSize);
            }
        } else {
            Bmm2ResCastAndCopyOut(runInfo, vec2ResUb, mStartVec, mDealSize);
        }
    }

    __aicore__ inline bool CalcBlockNeedRowInvalid(RunInfoX &runInfo, int64_t s1FirstValidToken,
                                                   int64_t s1LastValidToken)
    {
        int32_t vecMStartIdx = runInfo.gS1Idx + runInfo.vecMbaseIdx;
        int32_t vecMEndIdx = vecMStartIdx + runInfo.actVecMSize - 1;
        int32_t s1StartTdx, s1EndTdx;
        bool ret = false;
        if constexpr (layout == LayOutTypeEnum::LAYOUT_BSH) {
            // S1G layout
            s1StartTdx = vecMStartIdx / constInfo.gSize;
            s1EndTdx = vecMEndIdx / constInfo.gSize;
            ret = (s1StartTdx < s1FirstValidToken) || (s1EndTdx > s1LastValidToken);
        }
        return ret;
    }

    template <typename VEC2_RES_T>
    __aicore__ inline void RowInvalid(LocalTensor<VEC2_RES_T> &vec2ResUb, int64_t mStartVec, int64_t mDealSize,
                                      RunInfoX &runInfo, int64_t dSizeAligned64)
    {
        if constexpr (hasAttn) {
            int64_t s1FirstValidToken = Min(Max(-runInfo.nextTokensLeftUp, 0), runInfo.actS1Size);
            int64_t s1LastValidToken = Min(Max(runInfo.preTokensLeftUp + runInfo.actS2Size, 0), runInfo.actS1Size);
            s1LastValidToken = Max(s1LastValidToken - 1, 0);
            bool hasValidRow = (s1FirstValidToken > 0) || (s1LastValidToken < runInfo.actS1Size);
            bool batchNeedRowInvalid = constInfo.isRowInvalidOpen || // 手动开启行无效
                                       ((constInfo.sparseMode != SparseMode::LEFT_UP_CAUSAL) &&
                                        hasValidRow); // sparse = 0 or 3 or 4，preToekens or nextTokens负数
            if (!batchNeedRowInvalid) {
                return;
            }

            bool blockNeedRowInvalid = CalcBlockNeedRowInvalid(runInfo, s1FirstValidToken, s1LastValidToken);
            blockNeedRowInvalid = blockNeedRowInvalid || constInfo.isRowInvalidOpen;

            if (blockNeedRowInvalid) {
                LocalTensor<float> maxTensor =
                    softmaxMaxBuf[runInfo.mloop % (PRELOAD_N + 1)].template Get<float>()[mStartVec];
                RowInvalidUpdateVF<float>(vec2ResUb, maxTensor, mDealSize, constInfo.dSizeV,
                                          static_cast<uint32_t>(dSizeAligned64));
            }
        }
    }

    __aicore__ inline void Bmm2DataCopyOutTrans(const RunInfoX &info, LocalTensor<OUTPUT_T> &attnOutUb,
                                                uint32_t vecMIdx, uint32_t dealRowCount)
    {
        FaUbTensor<OUTPUT_T> ubTensor{
            .tensor = attnOutUb, .rowCount = dealRowCount, .colCount = (uint32_t)(dTemplateAlign64)};
        GmCoord gmCoord{.bIdx = info.bIdx,
                        .n2Idx = info.n2Idx,
                        .gS1Idx = info.gS1Idx + info.vecMbaseIdx + vecMIdx,
                        .dIdx = 0,
                        .gS1DealSize = dealRowCount,
                        .dDealSize = (uint32_t)constInfo.dSizeV};
        CopyAttentionOut(ubTensor, gmCoord);
    }

    __aicore__ inline void CopyAttentionOut(FaUbTensor<OUTPUT_T> &ubTensor, GmCoord &gmCoord)
    {
        constexpr GmFormat OUT_FORMAT = GmFormat::BSNGD;
        FaGmTensor<OUTPUT_T, OUT_FORMAT> outGmTensor;
        outGmTensor.gmTensor = attentionOutGm;
        InitOffset(outGmTensor.offsetInfo, constInfo.bSize, constInfo.n2Size, constInfo.gSize, constInfo.s1Size,
                   constInfo.dSizeV);
        CopyAttnOutUbToGm<OUTPUT_T, OUT_FORMAT>(outGmTensor, ubTensor, gmCoord);
    }

    __aicore__ inline void BroadCastAndCopyOut(const RunInfoX &runInfo, LocalTensor<float> &sumUb,
                                               LocalTensor<float> &maxUb, int64_t gmOffset, int64_t calculateSize)
    {
        // Copy sum to gm
        LocalTensor<float> sumOutTensor = sumBrdcst.template AllocTensor<float>();
        FaVectorApi::BroadcastMaxSum(sumOutTensor, sumUb, runInfo.actVecMSize);
        sumBrdcst.template EnQue(sumOutTensor);
        sumBrdcst.template DeQue<float>();
        DataCopy(softmaxCombineSumGm[gmOffset], sumOutTensor, calculateSize);
        sumBrdcst.template FreeTensor(sumOutTensor);

        // Copy max to gm
        LocalTensor<float> maxOutTensor = maxBrdcst.template AllocTensor<float>();
        FaVectorApi::BroadcastMaxSum(maxOutTensor, maxUb, runInfo.actVecMSize);
        maxBrdcst.template EnQue(maxOutTensor);
        maxBrdcst.template DeQue<float>();
        DataCopy(softmaxCombineMaxGm[gmOffset], maxOutTensor, calculateSize);
        maxBrdcst.template FreeTensor(maxOutTensor);
    }

    __aicore__ inline void ComputeLogSumExpAndCopyToGm(const RunInfoX &runInfo, LocalTensor<float> &sumUb,
                                                       LocalTensor<float> &maxUb)
    {
        if (unlikely(runInfo.actVecMSize == 0)) {
            return;
        }
        int64_t calculateSize = runInfo.actVecMSize * FLOAT_BASE_SIZE;
        // 是否要改成halfMRealSize
        int64_t gmOffset = runInfo.faTmpOutWsPos * mBaseSize * FLOAT_BASE_SIZE + runInfo.vecMbaseIdx * FLOAT_BASE_SIZE;
        // combineS2Idx?nBufferStartM?
        // Copy sum to gm
        BroadCastAndCopyOut(runInfo, sumUb, maxUb, gmOffset, calculateSize);
    }

    __aicore__ inline void Bmm2ResForCombineCopyOut(const RunInfoX &runInfo, LocalTensor<T> &vec2ResUb,
                                                    uint32_t mStartVec, uint32_t mDealSize)
    {
        int64_t dSizeAligned64 = (int64_t)dVTemplateType;
        SetFlag<HardEvent::V_MTE3>(vToMte3Id[runInfo.loop % DB]);
        WaitFlag<HardEvent::V_MTE3>(vToMte3Id[runInfo.loop % DB]);
        uint64_t gmOffset =
            runInfo.faTmpOutWsPos * mBaseSize * constInfo.dSizeV + (runInfo.vecMbaseIdx + mStartVec) * constInfo.dSizeV;

        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = mDealSize;
        dataCopyParams.blockLen = constInfo.dSizeV * sizeof(T);
        dataCopyParams.srcStride = (dSizeAligned64 - constInfo.dSizeV) / (FA_BYTE_BLOCK / sizeof(T));
        dataCopyParams.dstStride = 0;

        DataCopyPad(accumOutGm[gmOffset], vec2ResUb, dataCopyParams);
    }

    __aicore__ inline void ProcessVec2(mm2ResPos &bmm2ResBuf, RunInfoX &runInfo)
    {
        bmm2ResBuf.WaitCrossCore();
        ProcessVec2OnUb(bmm2ResBuf, runInfo);
    }

    __aicore__ inline void SoftmaxInitBuffer()
    {
        tPipe->InitBuffer(softmaxSumBuf[0], 256); // [64, 1]
        tPipe->InitBuffer(softmaxSumBuf[1], 256); // [64, 1]
        tPipe->InitBuffer(softmaxSumBuf[2], 256); // [64, 1]

        tPipe->InitBuffer(softmaxMaxBuf[0], 256); // [64, 1]
        tPipe->InitBuffer(softmaxMaxBuf[1], 256); // [64, 1]
        tPipe->InitBuffer(softmaxMaxBuf[2], 256); // [64, 1]

        tPipe->InitBuffer(softmaxExpBuf[0], 256); // [64, 1]
        tPipe->InitBuffer(softmaxExpBuf[1], 256); // [64, 1]
        tPipe->InitBuffer(softmaxExpBuf[2], 256); // [64, 1]

        tPipe->InitBuffer(maxBrdcst, 1, 2048); // [64, 8]
        tPipe->InitBuffer(sumBrdcst, 1, 2048); // [64, 8]
    }

    __aicore__ inline void InitBuffers()
    {
        uint32_t mm1ResultSize = mBaseSize / CV_RATIO * s2BaseSize * sizeof(T);
        uint32_t mm2ResultSize = mBaseSize / CV_RATIO * dTemplateAlign64 * sizeof(T);

        SoftmaxInitBuffer();
        if constexpr (hasAttn) {
            tPipe->InitBuffer(attnMaskInQue[0], 1, 8192);
            tPipe->InitBuffer(attnMaskInQue[1], 1, 8192);
        }

        tPipe->InitBuffer(commonTBuf, 512);
        tPipe->InitBuffer(stage2OutBuf, 64 * dTemplateAlign64 * sizeof(T));

        tPipe->InitBuffer(stage1OutQue[0], 1, 16896);
        tPipe->InitBuffer(stage1OutQue[1], 1, 16896);
    }

    __aicore__ inline void AllocEventID()
    {
        mte3ToVId[0] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
        mte3ToVId[1] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
        vToMte3Id[0] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
        vToMte3Id[1] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
        SetFlag<HardEvent::MTE3_V>(mte3ToVId[0]);
        SetFlag<HardEvent::MTE3_V>(mte3ToVId[1]);
    }

    __aicore__ inline void FreeEventID()
    {
        WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToVId[0]);
        WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToVId[1]);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(mte3ToVId[0]);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(mte3ToVId[1]);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(vToMte3Id[0]);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(vToMte3Id[1]);
    }

    __aicore__ inline void AttnMaskCopyIn(LocalTensor<uint8_t> &attnMaskUb, uint32_t vecMIdx, uint32_t mDealSize,
                                          RunInfoX &runInfo)
    {
        MaskInfo maskInfo;
        maskInfo.gs1StartIdx = runInfo.gS1Idx + runInfo.vecMbaseIdx + vecMIdx;
        maskInfo.gs1dealNum = mDealSize;
        maskInfo.s1Size = runInfo.actS1Size;
        maskInfo.gSize = constInfo.gSize;
        maskInfo.s2StartIdx = runInfo.s2Idx;
        maskInfo.s2dealNum = runInfo.actSingleLoopS2Size;
        maskInfo.s2Size = runInfo.actS2Size;
        maskInfo.nBaseSize = s2BaseSize;
        maskInfo.preToken = constInfo.preTokens;
        maskInfo.nextToken = constInfo.nextTokens;
        maskInfo.sparseMode = static_cast<SparseMode>(constInfo.sparseMode);
        maskInfo.batchIdx = (constInfo.attnMaskBatch == 1) ? 0 : runInfo.bIdx;
        maskInfo.attnMaskBatchStride = constInfo.attnMaskS1Size * constInfo.attnMaskS2Size;
        maskInfo.attnMaskS1Stride = constInfo.attnMaskS2Size;
        maskInfo.attnMaskDstStride = (s2BaseSize - Align(maskInfo.s2dealNum, 32U)) / 32;
        maskInfo.maskValue = negativeIntScalar;
        maskInfo.layout = MASK_LAYOUT;
        maskInfo.attnMaskType = MASK_BOOL; // compatible with int8/uint8

        bool IsSkipMask = IsSkipAttentionmask(maskInfo);

        if (!IsSkipMask) {
            AttentionmaskCopyIn<uint8_t, MASK_LAYOUT, true, s2BaseSize>(attnMaskUb, attnMaskGmInt, maskInfo);
        } else {
            Duplicate(attnMaskUb, static_cast<uint8_t>(0U), maskInfo.gs1dealNum * s2BaseSize);
        }
    }

    __aicore__ inline void DealZeroActSeqLen(uint32_t bN2Cur)
    {
        uint32_t n2Idx = bN2Cur % constInfo.n2Size;
        uint32_t bIdx = bN2Cur / constInfo.n2Size;
        BsngdOffsetInfo offsetInfo;
        InitOffset(offsetInfo, constInfo.bSize, constInfo.n2Size, constInfo.gSize, constInfo.s1Size, constInfo.dSizeV);
        DealActSeqLenIsZero<OUTPUT_T>(bIdx, n2Idx, offsetInfo, attentionOutGm);
    }
};

template <typename INPUT_T, typename T, typename OUTPUT_T, LayOutTypeEnum layout = LayOutTypeEnum::None,
          LayOutTypeEnum outLayout = LayOutTypeEnum::None, S1TemplateType s1TemplateType = S1TemplateType::Aligned128,
          S2TemplateType s2TemplateType = S2TemplateType::Aligned128,
          DTemplateType dTemplateType = DTemplateType::Aligned128,
          DTemplateType dVTemplateType = DTemplateType::Aligned128, bool hasAttn = false, bool isCombine = false>
class FABlockVecDummy {
public:
    static constexpr bool HAS_MASK = hasAttn;
    static constexpr bool COMBINE = isCombine;
    using OUT_T = OUTPUT_T;
    using ConstInfoX = ConstInfo_t<FaKernelType::NO_QUANT>;
    __aicore__ inline FABlockVecDummy(ConstInfoX &constInfo){};
};

} // namespace BaseApi
#endif // FA_BLOCK_VEC_H
