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
 * \file hc_pre_m_k_split_core.h
 * \brief
 */

#include "kernel_operator.h"
#include "hc_pre_base_arch35.h"
#include "hc_pre_cube_compute_arch35.h"

namespace HcPreNs {
using namespace AscendC;
static constexpr uint64_t L1_ALLOC_SIZE = 512 * 1024;
static constexpr uint64_t L1_BUF_NUM = 2;
static constexpr uint64_t L1_BUF_OFFSET = 128 * 256;
constexpr static int SYNC_MODE4 = 4;
template <HardEvent event>
__aicore__ inline void SetWaitFlag(HardEvent evt)
{
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(evt));
    SetFlag<event>(eventId);
    WaitFlag<event>(eventId);
}

template <typename T>
class HcPreMSplitCorePart1 {
public:
    __aicore__ inline HcPreMSplitCorePart1()
    {}

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR hcFn, GM_ADDR hcScale, GM_ADDR hcBase,
        GM_ADDR y, GM_ADDR post, GM_ADDR combFrag, const HcPreTilingData* tilingDataPtr, TPipe* pipePtr)
    {
        pipe = pipePtr;
        tilingData = tilingDataPtr;
        xGm.SetGlobalBuffer((__gm__ T*)x);
        hcFnGm.SetGlobalBuffer((__gm__ float*)hcFn);
        yGm.SetGlobalBuffer((__gm__ T*)y);

        hcScaleGm.SetGlobalBuffer((__gm__ float*)hcScale);
        hcBaseGm.SetGlobalBuffer((__gm__ float*)hcBase);
        postGm.SetGlobalBuffer((__gm__ float*)post);
        combFragGm.SetGlobalBuffer((__gm__ float*)combFrag);

        TBuf<TPosition::A1> l1Buffer;
        pipe->InitBuffer(l1Buffer, L1_ALLOC_SIZE);
        xL1_ = l1Buffer.Get<float>();
        wL1_ = l1Buffer.Get<float>()[L1_BUF_NUM * L1_BUF_OFFSET];
        
        pipe->InitBufPool(tbufPool0, tilingData->bufferPool0Size);
        tbufPool0.InitBuffer(mmXBuf, CeilDiv(tilingData->mL1Size, 2) * RoundUp<float>(tilingData->hcMix) * sizeof(float));
        mmXLocal = mmXBuf.Get<float>();

        if ASCEND_IS_AIC {
            mmService_.Init();
            CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIC_AIV_FLAG);
            CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIC_AIV_FLAG + FLAG_ID_MAX);
            CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIC_AIV_FLAG);
            CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIC_AIV_FLAG + FLAG_ID_MAX);
        } else {
            tbufPool0.InitBuffer(rmsNormBuf, RoundUp<float>(CeilDiv(tilingData->mL1Size, 2)) * sizeof(float));
            tbufPool0.InitBufPool(tbufPool1, tilingData->bufferPool1Size);

            tbufPool0.InitBuffer(hcBaseBuf0, tilingData->hcMultAlign * sizeof(float));
            tbufPool0.InitBuffer(hcBaseBuf1, tilingData->hcMultAlign * sizeof(float));
            tbufPool0.InitBuffer(hcBaseBuf2, tilingData->hcMult * tilingData->hcMultAlign * sizeof(float));

            hcBase0Local = hcBaseBuf0.Get<float>();
            hcBase1Local = hcBaseBuf1.Get<float>();
            hcBase2Local = hcBaseBuf2.Get<float>();
        }
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIV {
          CopyIn(hcBaseGm, hcBase0Local, 1, tilingData->hcMult);
          CopyIn(hcBaseGm[tilingData->hcMult], hcBase1Local, 1, tilingData->hcMult);
          CopyIn(hcBaseGm[tilingData->hcMult * 2], hcBase2Local, tilingData->hcMult, tilingData->hcMult);
          event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
          SetFlag<HardEvent::MTE2_V>(eventId);
          WaitFlag<HardEvent::MTE2_V>(eventId);
        }

        int64_t curBlockIdx = GetBlockIdx();
        int64_t totalBlockNum = GetBlockNum();

        uint64_t mBlkDimIdx = curBlockIdx % tilingData->cubeBlockDimM;
        uint64_t kBlkDimIdx = curBlockIdx % tilingData->cubeBlockDimK;

        // todo 移到tiling计算    
        uint64_t mCnt = CeilDiv(tilingData->bs, tilingData->mL1Size);
        uint64_t singleCoreMaxRound = CeilDiv(mCnt, tilingData->cubeBlockDimM);
        uint64_t mainCoreCount = mCnt % tilingData->cubeBlockDimM;
        uint64_t singleCoreRound = (mainCoreCount == 0 || curBlockIdx < mainCoreCount) ? singleCoreMaxRound : singleCoreMaxRound - 1;
        uint64_t mGmOffset = 0;

        int64_t xGmBaseOffset = 0;
        if ASCEND_IS_AIV {
            int64_t aivCurBlockIdx = GetBlockIdx();
            if (aivCurBlockIdx % 2 == 0) {
                xGmBaseOffset = aivCurBlockIdx / 2 * singleCoreMaxRound * tilingData->mL1Size * tilingData->hcMult * tilingData->d;
            } else {
                xGmBaseOffset = (aivCurBlockIdx - 1) / 2 * singleCoreMaxRound * tilingData->mL1Size * tilingData->hcMult * tilingData->d + 
                                 (tilingData->mL1Size / 2) * tilingData->hcMult * tilingData->d;
            }
        }

        // todo 移到tiling计算        
        uint64_t cvLoopKSize = tilingData->kL1Size / tilingData->kUbSize;  

        // m轴切分 按照0 0 1 1..分核
        for (uint64_t roundIdx = 0; roundIdx < singleCoreRound; mGmOffset += tilingData->mL1Size, ++roundIdx)
        {
            uint64_t mL1RealSize = AscendC::Std::min(tilingData->bs - mGmOffset, (uint64_t)tilingData->mL1Size);
            uint64_t kGmStartOffset = 0;
            uint64_t kGmEndOffset = tilingData->multCoreSplitKSize;

            for (int64_t kGmOffset = kGmStartOffset; kGmOffset < kGmEndOffset; kGmOffset += tilingData->kL1Size) {
                if ASCEND_IS_AIC {
                    bool isFirstKL1 = kGmOffset == kGmStartOffset;
                    bool isLastKL1 = (kGmOffset + tilingData->kL1Size) >= kGmEndOffset;
                    uint64_t kL1RealSize = AscendC::Std::min(kGmEndOffset - kGmOffset, (uint64_t)tilingData->kL1Size);
                    mmService_.CopyInB1Nd2Nz(tilingData->multCoreSplitKSize, kL1RealSize, 
                                             tilingData->hcMix, hcFnGm[kGmOffset], 
                                             wL1_[mmService_.GetBL1BufferId() * L1_BUF_OFFSET]);
                    CrossCoreWaitFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIV_AIC_FLAG + FLAG_ID_MAX);
                    CrossCoreWaitFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIV_AIC_FLAG);
                    uint64_t mL1AlignSize = Align(mL1RealSize, AscendC::BLOCK_CUBE);
                    uint64_t nL1AlignSize = Align((uint64_t)tilingData->hcMix, AscendC::BLOCK_CUBE);
                    mmService_.Process(tilingData->bs, tilingData->hcMix, mL1RealSize, (256 / AscendC::Std::max(mL1AlignSize, nL1AlignSize)) * 32, 
                                       isFirstKL1, isLastKL1, xL1_[aL1BufferID_ * L1_BUF_OFFSET], wL1_[mmService_.GetBL1BufferId() * L1_BUF_OFFSET]);
                    if (isLastKL1) {
                        mmService_.CopyOut(mmGm);
                        // CrossCoreSetFlag<SYNC_MODE4, PIPE_FIX>(SYNC_AIC_AIV_PRE_POST_FLAG);
                        // CrossCoreSetFlag<SYNC_MODE4, PIPE_FIX>(SYNC_AIC_AIV_PRE_POST_FLAG + FLAG_ID_MAX);
                    }
                    // xL1_[(cvLoopIdx_ & 1) * L1_BUF_OFFSET]
                    CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIC_AIV_FLAG); // 写出ub搬出，cv流水同步比较复杂，暂不讨论
                    CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE1>(SYNC_AIC_AIV_FLAG + FLAG_ID_MAX);
                } else {
                    CrossCoreWaitFlag<SYNC_MODE4, PIPE_MTE3>(SYNC_AIC_AIV_FLAG);
                    int64_t rowFactor = mL1RealSize / 2;
                    int64_t tailRowFactor = mL1RealSize - rowFactor;
                    int64_t curRowFactor = rowFactor;
                    int64_t mL1SizeAlign = CeilAlign(mL1RealSize, C0_SIZE);
                    if (curBlockIdx % 2 == 1) {
                        curRowFactor = tailRowFactor;
                    }
                    // 后续xNd2NzLocal开DB时，通过bufferIdx & 1取不同的buffer地址，for循环中累加bufferIdx
                    // 当前实现不对xNd2NzLocal开DB, 后续进行修改
                    int64_t bufferIdx = 0;
                    SetFlag<HardEvent::MTE3_V>(static_cast<event_t>(bufferIdx & 1));
                    for (int64_t cvLoopIdx = 0; cvLoopIdx < cvLoopKSize; cvLoopIdx++) {
                        uint64_t kRealSize = kGmOffset + tilingData->kUbSize >= kGmEndOffset ? kGmEndOffset - kGmOffset : tilingData->kUbSize;

                        xLocal = xQue.template AllocTensor<T>();
                        CopyIn(xGm[xGmBaseOffset + mGmOffset * tilingData->hcMult * tilingData->d + kGmOffset + cvLoopIdx * tilingData->kUbSize],
                               xLocal, curRowFactor, tilingData->kUbSize, tilingData->hcMult * tilingData->d - tilingData->kUbSize);
                        xQue.template EnQue(xLocal);
                        xLocal = xQue.template DeQue<T>();
                        VFProcessCast(xCastLocal, xLocal, curRowFactor, tilingData->kUbSize);
                        xQue.template FreeTensor(xLocal);
                        
                        WaitFlag<HardEvent::MTE3_V>(static_cast<event_t>(bufferIdx & 1));
                        VFTransND2NZ(xNd2NzLocal, xCastLocal, curRowFactor, tilingData->kUbSize);
                        SetFlag<HardEvent::V_MTE3>(static_cast<event_t>(bufferIdx & 1));
                        WaitFlag<HardEvent::V_MTE3>(static_cast<event_t>(bufferIdx & 1));

                        if (curBlockIdx % 2 == 0) {
                            DataCopyParams dataCopyXParams;
                            dataCopyXParams.blockCount = CeilDiv(tilingData->kUbSize, C0_SIZE);
                            dataCopyXParams.blockLen = curRowFactor * C0_SIZE * sizeof(float) / BLOCK_SIZE;
                            dataCopyXParams.srcStride = CeilAlign(curRowFactor, C0_SIZE) - curRowFactor;
                            dataCopyXParams.dstStride = CeilAlign(mL1RealSize, C0_SIZE) - curRowFactor;
                            CopyToL1(xNd2NzLocal, xL1_[(aL1BufferID_ * L1_BUF_OFFSET) + cvLoopIdx * tilingData->kUbSize * mL1SizeAlign], dataCopyXParams);
                        } else {
                            DataCopyParams dataCopyXParams;
                            dataCopyXParams.blockCount = CeilDiv(tilingData->kUbSize, C0_SIZE);
                            dataCopyXParams.blockLen = curRowFactor * C0_SIZE * sizeof(float) / BLOCK_SIZE;
                            dataCopyXParams.srcStride = CeilAlign(curRowFactor, C0_SIZE) -  curRowFactor;
                            dataCopyXParams.dstStride = CeilAlign(mL1RealSize, C0_SIZE) - curRowFactor;
                            CopyToL1(xNd2NzLocal, xL1_[(aL1BufferID_ * L1_BUF_OFFSET) + rowFactor * (BLOCK_SIZE / sizeof(float)) + cvLoopIdx * tilingData->kUbSize * mL1SizeAlign], dataCopyXParams);
                        }
                        SetFlag<HardEvent::MTE3_V>(static_cast<event_t>(bufferIdx & 1));

                        float coeff = 1 / static_cast<float>(tilingData->hcMult * tilingData->d);
                        SetWaitFlag<HardEvent::S_V>(HardEvent::S_V);

                        // rmsNormLocal结果需要累加，完成后rmsNormLocal shape为m,1
                        if (kGmOffset == kGmStartOffset && cvLoopIdx == 0) {
                            VFProcessInvRmsPart1<false>(rmsNormLocal, xCastLocal, coeff, curRowFactor, tilingData->kUbSize);
                        } else {
                            VFProcessInvRmsPart1<true>(rmsNormLocal, xCastLocal, coeff, curRowFactor, tilingData->kUbSize);
                        }
                    }
                    WaitFlag<HardEvent::MTE3_V>(static_cast<event_t>(bufferIdx & 1));
                    CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE3>(SYNC_AIV_AIC_FLAG);
                }
                aL1BufferID_ ^= 1;
            }
        }
        SyncAll<false>();
    }

private:
    TPipe* pipe;
    const HcPreTilingData* tilingData;
    // (M, K) * (N, K)

    GlobalTensor<T> xGm;
    GlobalTensor<float> hcFnGm;
    GlobalTensor<float> workspaceGm;
    GlobalTensor<float> mmGm;
    GlobalTensor<float> yGm;
    GlobalTensor<float> invRmsGm;

    TQue<QuePosition::VECIN, 1> xQue;
    TQue<QuePosition::A1, 1> mmXQue;
    TQue<QuePosition::VECOUT, 1> rmsNormQue;
    TBuf<QuePosition::VECCALC> castBuf;
    TBuf<QuePosition::VECCALC> nd2NzBuf;

    LocalTensor<T> xLocal;
    LocalTensor<float> mmXLocal;
    LocalTensor<float> rmsNormLocal;
    LocalTensor<float> xCastLocal;
    LocalTensor<float> xNd2NzLocal;

    int64_t kBlockFactor = 32; // K方向的分核数 后续从tilingData中获取
    int64_t kFactor = 256;

    HcPreCubeCompute mmService_;
    LocalTensor<float> xL1_;
    LocalTensor<float> wL1_;
    static constexpr uint64_t SYNC_AIV_AIC_FLAG = 8;
    static constexpr uint64_t SYNC_AIC_AIV_FLAG = 9;
    static constexpr uint64_t SYNC_AIV_AIC_PRE_POST_FLAG = 10;
    static constexpr uint64_t SYNC_AIC_AIV_PRE_POST_FLAG = 11;
    static constexpr uint64_t FLAG_ID_MAX = 16;
    uint64_t cvLoopIdx_ = 0;
    uint8_t aL1BufferID_ = 0;
};


template <typename T>
class HcPreMKSplitCorePart2 {
public:
    __aicore__ inline HcPreMKSplitCorePart2()
    {}

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR hcScale, GM_ADDR hcBase, GM_ADDR y, GM_ADDR post,
        GM_ADDR combFrag, GM_ADDR workspace, const HcPreTilingData* tilingDataPtr, TPipe* pipePtr)
    {
        pipe = pipePtr;
        tilingData = tilingDataPtr;

        xGm.SetGlobalBuffer((__gm__ T*)x);
        hcScaleGm.SetGlobalBuffer((__gm__ float*)hcScale);
        hcBaseGm.SetGlobalBuffer((__gm__ float*)hcBase);
        yGm.SetGlobalBuffer((__gm__ T*)y);
        postGm.SetGlobalBuffer((__gm__ float*)post);
        combFragGm.SetGlobalBuffer((__gm__ float*)combFrag);
        mmGm.SetGlobalBuffer((__gm__ float*)workspace);
        rmsGm.SetGlobalBuffer((__gm__ float*)workspace + tilingData->kBlockFactor * tilingData->bs * tilingData->hcMix);


        // InQue
        pipe->InitBuffer(
            xQue, 2, tilingData->rowFactor * tilingData->hcMult * RoundUp<T>(tilingData->dFactor) * sizeof(T));
        int64_t rmsAndmmQueSize = tilingData->kBlockFactor * RoundUp<float>(tilingData->rowFactor) * sizeof(float) + 
                                  tilingData->kBlockFactor * tilingData->rowFactor * RoundUp<float>(tilingData->hcMix) * sizeof(float);
        pipe->InitBuffer(rmsAndmmQue, 2, rmsAndmmQueSize);

        // OutQue
        pipe->InitBuffer(
            yQue, 2, tilingData->rowFactor * RoundUp<T>(tilingData->dFactor) * sizeof(T));
        pipe->InitBuffer(postQue, 2, tilingData->rowFactor * tilingData->hcMultAlign * sizeof(float));
        pipe->InitBuffer(
            combFragQue, 2, tilingData->rowFactor * tilingData->hcMult * tilingData->hcMultAlign * sizeof(float));

        // TBuf
        pipe->InitBuffer(hcBaseBuf0, tilingData->hcMultAlign * sizeof(float));
        pipe->InitBuffer(hcBaseBuf1, tilingData->hcMultAlign * sizeof(float));
        pipe->InitBuffer(hcBaseBuf2, tilingData->hcMult * tilingData->hcMultAlign * sizeof(float));
        pipe->InitBuffer(mixesBuf, tilingData->rowFactor * RoundUp<float>(tilingData->hcMix) * sizeof(float));

        hcBase0Local = hcBaseBuf0.Get<float>();
        hcBase1Local = hcBaseBuf1.Get<float>();
        hcBase2Local = hcBaseBuf2.Get<float>();
        mixesLocal = mixesBuf.Get<float>();
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIV {
            int64_t stage1UsedCoreNum = tilingData->cubeBlockDimK;
            int64_t curBlockIdx = GetBlockIdx();
            int64_t stage2UsedCoreNum = tilingData->secondUsedCoreNum;
            if (curBlockIdx >= stage2UsedCoreNum) {
                return;
            }
            int64_t rowOuterLoop =
                (curBlockIdx == stage2UsedCoreNum - 1) ? tilingData->rowLoopOfTailBlock : tilingData->rowLoopOfFormerBlock;
            int64_t tailRowFactor = (curBlockIdx == stage2UsedCoreNum - 1) ? tilingData->tailRowFactorOfTailBlock :
                                                                        tilingData->tailRowFactorOfFormerBlock;

            CopyIn(hcBaseGm, hcBase0Local, 1, tilingData->hcMult);
            CopyIn(hcBaseGm[tilingData->hcMult], hcBase1Local, 1, tilingData->hcMult);
            CopyIn(hcBaseGm[tilingData->hcMult * 2], hcBase2Local, tilingData->hcMult, tilingData->hcMult);
            event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eventId);
            WaitFlag<HardEvent::MTE2_V>(eventId);
            int64_t mmGmBaseOffset = curBlockIdx * tilingData->rowOfFormerBlock * tilingData->hcMix;
            int64_t rmsGmBaseOffset = curBlockIdx * tilingData->rowOfFormerBlock;
            int64_t xGmBaseOffset = curBlockIdx * tilingData->rowOfFormerBlock * tilingData->hcMult * tilingData->d;
            int64_t mmLocalSize = stage1UsedCoreNum * tilingData->rowFactor * RoundUp<float>(tilingData->hcMix);
            for (int64_t rowOuterIdx = 0; rowOuterIdx < rowOuterLoop; rowOuterIdx++) {
                int64_t curRowFactor = (rowOuterIdx == rowOuterLoop - 1) ? tailRowFactor : tilingData->rowFactor;
                rmsAndmmLocal = rmsAndmmQue.AllocTensor<float>();
                CopyInWithLoopMode(
                    mmGm[mmGmBaseOffset + rowOuterIdx * tilingData->rowFactor * tilingData->hcMix], rmsAndmmLocal, tilingData->kBlockFactor, curRowFactor, tilingData->hcMix, tilingData->bs * tilingData->hcMix);
                CopyIn(
                    rmsGm[rmsGmBaseOffset + rowOuterIdx * tilingData->rowFactor],
                    rmsAndmmLocal[mmLocalSize], tilingData->kBlockFactor, curRowFactor, tilingData->bs - curRowFactor);

                rmsAndmmQue.EnQue(rmsAndmmLocal);
                rmsAndmmLocal = rmsAndmmQue.DeQue<float>();
                
                VFProcessInvRmsPart3WithGroupReduce(mixesLocal, rmsAndmmLocal, rmsAndmmLocal[mmLocalSize], tilingData->normEps, tilingData->kBlockFactor, curRowFactor, tilingData->hcMix);
                
                VFProcessPre(
                    mixesLocal, mixesLocal, hcBase0Local, hcScaleGm.GetValue(0), tilingData->hcEps,
                    curRowFactor, tilingData->hcMult, tilingData->hcMix);
                for (int64_t dLoopIdx = 0; dLoopIdx < tilingData->dLoop; dLoopIdx++) {
                    int64_t curDFactor =
                        (dLoopIdx == tilingData->dLoop - 1) ? tilingData->tailDFactor : tilingData->dFactor;
                    xLocal = xQue.template AllocTensor<T>();
                    CopyIn(
                        xGm[xGmBaseOffset + rowOuterIdx * tilingData->rowFactor * tilingData->hcMult * tilingData->d +
                            dLoopIdx * tilingData->dFactor],
                        xLocal, tilingData->rowFactor * tilingData->hcMult, curDFactor, tilingData->d - curDFactor);
                    xQue.template EnQue(xLocal);
                    xLocal = xQue.template DeQue<T>();
                    
                    yLocal = yQue.template AllocTensor<T>();
                    VFProcessY(yLocal, mixesLocal, xLocal, curRowFactor, tilingData->hcMult, curDFactor, tilingData->hcMix);
                    xQue.template FreeTensor(xLocal);
                    yQue.template EnQue(yLocal);
                    yLocal = yQue.template DeQue<T>();
                    CopyOut(yLocal, yGm[curBlockIdx * tilingData->rowOfFormerBlock * tilingData->d + rowOuterIdx * tilingData->rowFactor * tilingData->d + dLoopIdx * tilingData->dFactor], curRowFactor, curDFactor, tilingData->d - curDFactor);
                    yQue.template FreeTensor(yLocal);
                }

                // post
                postLocal = postQue.AllocTensor<float>();
                VFProcessPost(
                    postLocal, mixesLocal[tilingData->hcMult], hcBase1Local,
                    hcScaleGm.GetValue(1), tilingData->hcEps, curRowFactor, tilingData->hcMult, tilingData->hcMix);

                postQue.EnQue(postLocal);
                postLocal = postQue.DeQue<float>();
                CopyOut(postLocal, postGm[curBlockIdx * tilingData->rowOfFormerBlock * tilingData->hcMult + rowOuterIdx * tilingData->rowFactor * tilingData->hcMult], curRowFactor, tilingData->hcMult);
                postQue.FreeTensor(postLocal);

                // combFrag
                combFragLocal = combFragQue.AllocTensor<float>();
                VFProcessCombFragRLessVLUseFourUnfold(
                    combFragLocal, mixesLocal[tilingData->hcMult * 2], hcBase2Local, hcScaleGm.GetValue(2), tilingData->hcEps,
                    tilingData->iterTimes - 1, curRowFactor, tilingData->hcMult, tilingData->hcMult, tilingData->hcMix);
                rmsAndmmQue.FreeTensor(rmsAndmmLocal);

                combFragQue.EnQue(combFragLocal);
                combFragLocal = combFragQue.DeQue<float>();
                CopyOut(combFragLocal, combFragGm[curBlockIdx * tilingData->rowOfFormerBlock * tilingData->hcMult * tilingData->hcMult + rowOuterIdx * tilingData->rowFactor * tilingData->hcMult * tilingData->hcMult], curRowFactor * tilingData->hcMult, tilingData->hcMult);
                combFragQue.FreeTensor(combFragLocal);
            }
        }
    }

private:
    TPipe* pipe;
    const HcPreTilingData* tilingData;
    GlobalTensor<float> hcScaleGm;
    GlobalTensor<float> hcBaseGm;
    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;
    GlobalTensor<float> postGm;
    GlobalTensor<float> combFragGm;

    GlobalTensor<float> mmGm;
    GlobalTensor<float> rmsGm;

    TQue<QuePosition::VECIN, 1> rmsAndmmQue;
    TQue<QuePosition::VECIN, 1> xQue;

    TQue<QuePosition::VECOUT, 1> yQue;
    TQue<QuePosition::VECOUT, 1> postQue;
    TQue<QuePosition::VECOUT, 1> combFragQue;

    TBuf<QuePosition::VECCALC> mixesBuf;
    TBuf<QuePosition::VECCALC> hcBaseBuf0;
    TBuf<QuePosition::VECCALC> hcBaseBuf1;
    TBuf<QuePosition::VECCALC> hcBaseBuf2;

    LocalTensor<float> mixesLocal;
    LocalTensor<float> rmsAndmmLocal;
    LocalTensor<T> xLocal;
    LocalTensor<T> yLocal;
    LocalTensor<float> postLocal;
    LocalTensor<float> combFragLocal;
    LocalTensor<float> hcBase0Local;
    LocalTensor<float> hcBase1Local;
    LocalTensor<float> hcBase2Local;
};



} // namespace HCPreSinkhorn

#endif