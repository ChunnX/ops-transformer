/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file chunk_gated_delta_rule_stage2_arch35.h
 * \brief Stage2: state recurrence with FP32 intermediate computation.
 *
 * New flow (per chunk):
 *   1. AIV: load state → cast FP32 → output BF16 curStateBf16 (for AIC) + decayed state → stateOut
 *   2. AIC: W@S (FP32 C) → wsVPrime  [no accum]
 *   3. AIV: U(BF16) - W@S(FP32) → cast BF16 → vInner GM  [for stage3]
 *   4. AIC: Q'@S → out; (U-WS)^T @ kg (FP32 C) → wsStateUpdate  [no accum, no atomicAdd]
 *   5. AIV: S_decayed + (U-WS)@K → cast stateType → stateOut GM
 */
#ifndef CHUNK_GATED_DELTA_RULE_STAGE2_ARCH35_H
#define CHUNK_GATED_DELTA_RULE_STAGE2_ARCH35_H

#include "kernel_tiling/kernel_tiling.h"
#include "chunk_gated_delta_rule_utils.h"
#include "../chunk_gated_delta_rule_tiling_data.h"

namespace ChunkGatedDeltaRule {
using namespace AscendC;
using namespace matmul;

using aT2 = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t, true>;
using bT2 = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t, true>;
template <typename cType>
using StageTwoMTT = matmul::MatmulImpl<aT2, bT2, MatmulType<TPosition::GM, CubeFormat::ND, cType>>;
using StageTwoMT = StageTwoMTT<bfloat16_t>;
using StageTwoMTFp32C = StageTwoMTT<float>;

template <typename stateType>
struct StageTwoParams {
    GlobalTensor<bfloat16_t> qPrime;         // (Nv, Sp, Dk)
    GlobalTensor<bfloat16_t> vInner;         // (Nv, Sp, Dv) BF16 — U, then overwritten with U-WS
    GlobalTensor<float> gCum;                // (Nv, Sp)
    GlobalTensor<bfloat16_t> kCumdecay;      // (Nv, Sp, Dk)
    GlobalTensor<bfloat16_t> curStateBf16;   // (Nv, Dv, Dk) BF16 state for AIC matmul
    GlobalTensor<stateType> stateIn;         // (Nv, Dv, Dk) state input
    GlobalTensor<stateType> stateOut;        // (Nv, Dv, Dk) state output
    GlobalTensor<bfloat16_t> kg;
    GlobalTensor<bfloat16_t> out;
    GM_ADDR wsVPrime;          // FP32 W@S workspace: aiCoreNum * chunkSize * Dv * sizeof(float)
    GM_ADDR wsStateUpdate;     // FP32 (U-WS)@K workspace: aiCoreNum * Dv * Dk * sizeof(float)
    StageTwoMT *mm1Bf16;
    StageTwoMTFp32C *mm1Fp32;
    TPipe *pipe;
    ChunkGroup *cg;
    int64_t Nv;
    int64_t Nk;
    int64_t Dv;
    int64_t Dk;
    int64_t stateStride1;
    bool useInitialState;
    bool isFirstGroup;
};

template <typename stateType = bfloat16_t, bool gOptional = false>
class Stage2 {
public:
    static constexpr bool kIsFp32 = std::is_same_v<stateType, float>;

    __aicore__ inline void Init(StageTwoParams<stateType> *initParams, int32_t coreNum)
    {
        sTP_ = initParams;
        pipe_ = sTP_->pipe;
        chunkSize_ = sTP_->cg->chunkSize;
        seqLength_ = sTP_->cg->length;
        Sp_ = (seqLength_ + chunkSize_ - 1) / chunkSize_ * chunkSize_;
        chunkNum_ = Sp_ / chunkSize_;
        coreNum_ = coreNum;
        Nv_ = sTP_->Nv;
        Nk_ = sTP_->Nk;
        Dv_ = sTP_->Dv;
        Dk_ = sTP_->Dk;
        curDk_ = Ceil(Dk_, BLOCK_SIZE / sizeof(bfloat16_t)) * (BLOCK_SIZE / sizeof(bfloat16_t));
        curDv_ = Ceil(Dv_, BLOCK_SIZE / sizeof(bfloat16_t)) * (BLOCK_SIZE / sizeof(bfloat16_t));
        curChunkSize_ = chunkSize_;
        useInitialState_ = sTP_->useInitialState;
        stateStride1_ = (useInitialState_) ? sTP_->stateStride1 : Dv_ * Dk_;
        isFirstGroup_ = sTP_->isFirstGroup;
        coreId_ = GetBlockIdx();
        if ASCEND_IS_AIV {
            coreId_ /= AIC_AIV_1_1;
            InitLocalBuffers();
            InitWsPointers();
        }
    }

    __aicore__ inline void InitLocalBuffers()
    {
        // UB budget: 192 KB per AIV.
        //
        // Phase 2 (U-WS): tmp1=32KB(fp32 U) + tmp2=32KB(fp32 W@S)
        // Phase 3 BF16:   tmp1=64KB(fp32 S) + tmp2=64KB(fp32 (U-WS)@K)
        // Phase 3 FP32:   tmp2=64KB(fp32 (U-WS)@K), S is in inQueue_
        //
        // FP32 state: inQueue_=64KB  outQueue_=32KB  tmpBuff_=96KB  => 192KB
        // BF16 state: inQueue_=32KB  outQueue_=32KB  tmpBuff_=128KB => 192KB

        uint64_t stateElems = static_cast<uint64_t>(Dv_) * curDk_;
        uint64_t deltaElems = static_cast<uint64_t>(chunkSize_) * curDv_;
        uint64_t maxElems = Std::max(stateElems, deltaElems);

        // inQueue_: load state (FP32 or BF16) and U (BF16)
        uint64_t inSize = kIsFp32 ? stateElems * sizeof(float)
                                  : Std::max(stateElems * sizeof(bfloat16_t), deltaElems * sizeof(bfloat16_t));
        pipe_->InitBuffer(inQueue_, BUFFER_NUM_ONE, inSize);

        // outQueue_: BF16 outputs (curStateBf16, vInner)
        uint64_t outSize = Std::max(stateElems * sizeof(bfloat16_t), deltaElems * sizeof(bfloat16_t));
        pipe_->InitBuffer(outQueue_, BUFFER_NUM_ONE, outSize);

        // tmpBuff_: two FP32 buffers
        uint64_t tmp1Elems = kIsFp32 ? deltaElems : maxElems;        // FP32 state skips tmp1 in Phase 3
        uint64_t tmp2Elems = maxElems;
        uint64_t tmpSize = tmp1Elems * sizeof(float) + tmp2Elems * sizeof(float) + sizeof(float);
        pipe_->InitBuffer(tmpBuff_, tmpSize);

        uint32_t off = 0;
        tmpBuffer1_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(tmp1Elems), off);
        off += Ceil(tmp1Elems * sizeof(float), BLOCK_SIZE) * BLOCK_SIZE;
        tmpBuffer2_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(tmp2Elems), off);
        off += Ceil(tmp2Elems * sizeof(float), BLOCK_SIZE) * BLOCK_SIZE;
        lastGCum_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(NUM_ONE), off);
    }

    __aicore__ inline void InitWsPointers()
    {
        wsVPrimeGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(sTP_->wsVPrime + (uint64_t)coreId_ * chunkSize_ * Dv_ * sizeof(float)));
        wsStateUpdateGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(sTP_->wsStateUpdate + (uint64_t)coreId_ * Dv_ * Dk_ * sizeof(float)));
    }

    __aicore__ inline void Process()
    {
        int64_t nvPerCore = (Nv_ + coreNum_ - 1) / coreNum_;
        int64_t nvStart = coreId_ * nvPerCore;
        int64_t nvEnd = nvStart + nvPerCore;
        nvEnd = nvEnd > Nv_ ? Nv_ : nvEnd;
        int64_t lastChunkSize = seqLength_ % chunkSize_ == 0 ? chunkSize_ : seqLength_ % chunkSize_;
        for (int64_t nvId = nvStart; nvId < nvEnd; nvId++) {
            curChunkSize_ = chunkSize_;
            for (int64_t cId = 0; cId < chunkNum_; cId++) {
                int64_t length = cId * chunkSize_;
                if (cId == chunkNum_ - 1) {
                    curChunkSize_ = lastChunkSize;
                }
                if ASCEND_IS_AIV {
                    ProcessAiv(nvId, cId, length);
                }
                if ASCEND_IS_AIC {
                    ProcessAic(nvId, cId, length);
                }
            }
        }
    }

    // ================== AIV ==================

    __aicore__ inline void ProcessAiv(int64_t nvId, int64_t cId, int64_t length)
    {
        if (GetSubBlockIdx() != 0) {
            return;
        }

        // Phase 1: State preparation
        GlobalTensor<stateType> stateGm = (isFirstGroup_ && cId == 0)
            ? sTP_->stateIn[nvId * stateStride1_]
            : sTP_->stateOut[nvId * Dv_ * Dk_];
        CalStateDecay(nvId, stateGm, sTP_->gCum[nvId * Sp_ + length]);
        CrossCoreSetFlag<0x4, PIPE_MTE3>(0x2); // notify AIC: state ready

        // Phase 2: U - W@S
        CrossCoreWaitFlag<0x4, PIPE_MTE3>(0x6); // wait for AIC CalVPrime
        CalDeltaVInner(nvId, length);

        // Phase 3: S + (U-WS)@K
        CrossCoreWaitFlag<0x4, PIPE_MTE3>(0x8); // wait for AIC CalStateNew
        CalStateUpdate(nvId);
    }

    __aicore__ inline void CalStateDecay(int64_t nvId, GlobalTensor<stateType> stateGm, GlobalTensor<float> gCum)
    {
        // Load state into inQueue_
        CopyInState(stateGm);
        auto stateIn = inQueue_.DeQue<stateType>();
        SetFlag<HardEvent::MTE2_V>(MTE2_V_EVENT);
        WaitFlag<HardEvent::MTE2_V>(MTE2_V_EVENT);

        // Cast to FP32 in tmpBuffer1_
        Cast(tmpBuffer1_, stateIn, RoundMode::CAST_NONE, Dv_ * curDk_);
        PipeBarrier<PIPE_V>();
        inQueue_.FreeTensor(stateIn);

        // Output non-decayed BF16 state for AIC matmul
        auto bf16State = outQueue_.AllocTensor<bfloat16_t>();
        Cast(bf16State, tmpBuffer1_, RoundMode::CAST_RINT, Dv_ * curDk_);
        SetFlag<HardEvent::V_MTE3>(V_MTE3_EVENT);
        WaitFlag<HardEvent::V_MTE3>(V_MTE3_EVENT);
        outQueue_.EnQue(bf16State);
        bf16State = outQueue_.DeQue<bfloat16_t>();
        DataCopyExtParams bf16Params{static_cast<uint16_t>(Dv_),
                                     static_cast<uint32_t>(Dk_ * sizeof(bfloat16_t)), 0, 0, 0};
        DataCopyPad(sTP_->curStateBf16[nvId * Dv_ * Dk_], bf16State, bf16Params);
        outQueue_.FreeTensor(bf16State);

        // Apply gCumExp decay to FP32 state in tmpBuffer1_
        if constexpr (gOptional) {
            DataCacheCleanAndInvalid<float, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(gCum[curChunkSize_ - 1]);
            float gVal = gCum.GetValue(curChunkSize_ - 1);
            lastGCum_.SetValue(0, gVal);
            SetFlag<HardEvent::S_V>(S_V_EVENT);
            WaitFlag<HardEvent::S_V>(S_V_EVENT);
            Exp<float, 0, true>(lastGCum_, lastGCum_, 1);
        } else {
            lastGCum_.SetValue(0, 1.0f);
        }
        float decayFactor = lastGCum_.GetValue(0);
        SetFlag<HardEvent::S_V>(S_V_EVENT);
        WaitFlag<HardEvent::S_V>(S_V_EVENT);
        Muls(tmpBuffer1_, tmpBuffer1_, decayFactor, Dv_ * curDk_);
        PipeBarrier<PIPE_V>();

        // Output decayed state to stateOut
        CopyOutStateDecayed(nvId);
    }

    __aicore__ inline void CopyOutStateDecayed(int64_t nvId)
    {
        if constexpr (kIsFp32) {
            // FP32: manual V→MTE3 sync, DataCopyPad directly from tmpBuffer1_
            SetFlag<HardEvent::V_MTE3>(V_MTE3_EVENT);
            WaitFlag<HardEvent::V_MTE3>(V_MTE3_EVENT);
            DataCopyExtParams params{static_cast<uint16_t>(Dv_),
                                     static_cast<uint32_t>(Dk_ * sizeof(float)), 0, 0, 0};
            DataCopyPad(sTP_->stateOut[nvId * Dv_ * Dk_], tmpBuffer1_, params);
        } else {
            // BF16: cast to BF16 via outQueue_
            auto stateOut = outQueue_.AllocTensor<bfloat16_t>();
            Cast(stateOut, tmpBuffer1_, RoundMode::CAST_RINT, Dv_ * curDk_);
            SetFlag<HardEvent::V_MTE3>(V_MTE3_EVENT);
            WaitFlag<HardEvent::V_MTE3>(V_MTE3_EVENT);
            outQueue_.EnQue(stateOut);
            stateOut = outQueue_.DeQue<bfloat16_t>();
            DataCopyExtParams params{static_cast<uint16_t>(Dv_),
                                     static_cast<uint32_t>(Dk_ * sizeof(bfloat16_t)), 0, 0, 0};
            DataCopyPad(sTP_->stateOut[nvId * Dv_ * Dk_], stateOut, params);
            outQueue_.FreeTensor(stateOut);
        }
    }

    __aicore__ inline void CalDeltaVInner(int64_t nvId, int64_t length)
    {
        uint64_t vInnerOffset = nvId * Sp_ * Dv_ + length * Dv_;

        // Load U (BF16) into inQueue_
        LocalTensor<bfloat16_t> uLocal = inQueue_.AllocTensor<bfloat16_t>();
        int padding = Ceil(Dv_, BLOCK_SIZE / sizeof(bfloat16_t)) * (BLOCK_SIZE / sizeof(bfloat16_t)) - Dv_;
        DataCopyExtParams uParams{static_cast<uint16_t>(curChunkSize_),
                                  static_cast<uint32_t>(Dv_ * sizeof(bfloat16_t)), 0, 0, 0};
        DataCopyPadExtParams<bfloat16_t> uPadParams{true, 0, static_cast<uint8_t>(padding), 0};
        DataCopyPad(uLocal, sTP_->vInner[vInnerOffset], uParams, uPadParams);
        inQueue_.EnQue(uLocal);
        uLocal = inQueue_.DeQue<bfloat16_t>();
        SetFlag<HardEvent::MTE2_V>(MTE2_V_EVENT);
        WaitFlag<HardEvent::MTE2_V>(MTE2_V_EVENT);

        // Cast U to FP32 in tmpBuffer1_
        Cast(tmpBuffer1_, uLocal, RoundMode::CAST_NONE, curChunkSize_ * curDv_);
        PipeBarrier<PIPE_V>();
        inQueue_.FreeTensor(uLocal);

        // Load W@S (FP32) into tmpBuffer2_
        DataCopyExtParams vpParams{static_cast<uint16_t>(curChunkSize_),
                                   static_cast<uint32_t>(Dv_ * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> vpPadParams{false, 0, 0, 0};
        DataCopyPad(tmpBuffer2_, wsVPrimeGm_, vpParams, vpPadParams);
        SetFlag<HardEvent::MTE2_V>(MTE2_V_EVENT);
        WaitFlag<HardEvent::MTE2_V>(MTE2_V_EVENT);

        // U - W@S
        Sub(tmpBuffer1_, tmpBuffer1_, tmpBuffer2_, curChunkSize_ * curDv_);
        PipeBarrier<PIPE_V>();

        // Cast to BF16, output to vInner GM
        auto deltaBf16 = outQueue_.AllocTensor<bfloat16_t>();
        Cast(deltaBf16, tmpBuffer1_, RoundMode::CAST_RINT, curChunkSize_ * curDv_);
        SetFlag<HardEvent::V_MTE3>(V_MTE3_EVENT);
        WaitFlag<HardEvent::V_MTE3>(V_MTE3_EVENT);
        outQueue_.EnQue(deltaBf16);
        deltaBf16 = outQueue_.DeQue<bfloat16_t>();
        DataCopyExtParams outParams{static_cast<uint16_t>(curChunkSize_),
                                    static_cast<uint32_t>(Dv_ * sizeof(bfloat16_t)), 0, 0, 0};
        DataCopyPad(sTP_->vInner[vInnerOffset], deltaBf16, outParams);
        outQueue_.FreeTensor(deltaBf16);

        CrossCoreSetFlag<0x4, PIPE_MTE3>(0x7); // notify AIC: vInner ready
    }

    __aicore__ inline void CalStateUpdate(int64_t nvId)
    {
        // Load decayed S from stateOut
        GlobalTensor<stateType> sGm = sTP_->stateOut[nvId * Dv_ * Dk_];
        CopyInState(sGm);
        auto sLocal = inQueue_.DeQue<stateType>();
        SetFlag<HardEvent::MTE2_V>(MTE2_V_EVENT);
        WaitFlag<HardEvent::MTE2_V>(MTE2_V_EVENT);

        if constexpr (kIsFp32) {
            // S is already FP32 in sLocal — add (U-WS)@K directly
            // Load (U-WS)@K into tmpBuffer2_
            LoadStateUpdateWs();

            // S + (U-WS)@K  (result in sLocal)
            Add(sLocal, sLocal, tmpBuffer2_, Dv_ * curDk_);
            PipeBarrier<PIPE_V>();

            // Output: manual V→MTE3 sync (no outQueue_ for FP32)
            SetFlag<HardEvent::V_MTE3>(V_MTE3_EVENT);
            WaitFlag<HardEvent::V_MTE3>(V_MTE3_EVENT);
            DataCopyExtParams params{static_cast<uint16_t>(Dv_),
                                     static_cast<uint32_t>(Dk_ * sizeof(float)), 0, 0, 0};
            DataCopyPad(sTP_->stateOut[nvId * Dv_ * Dk_], sLocal, params);
            inQueue_.FreeTensor(sLocal);
        } else {
            // Cast S (BF16) to FP32 in tmpBuffer1_
            Cast(tmpBuffer1_, sLocal, RoundMode::CAST_NONE, Dv_ * curDk_);
            PipeBarrier<PIPE_V>();
            inQueue_.FreeTensor(sLocal);

            // Load (U-WS)@K into tmpBuffer2_
            LoadStateUpdateWs();

            // S + (U-WS)@K
            Add(tmpBuffer1_, tmpBuffer1_, tmpBuffer2_, Dv_ * curDk_);
            PipeBarrier<PIPE_V>();

            // Cast to BF16, output
            auto stateOut = outQueue_.AllocTensor<bfloat16_t>();
            Cast(stateOut, tmpBuffer1_, RoundMode::CAST_RINT, Dv_ * curDk_);
            SetFlag<HardEvent::V_MTE3>(V_MTE3_EVENT);
            WaitFlag<HardEvent::V_MTE3>(V_MTE3_EVENT);
            outQueue_.EnQue(stateOut);
            stateOut = outQueue_.DeQue<bfloat16_t>();
            DataCopyExtParams params{static_cast<uint16_t>(Dv_),
                                     static_cast<uint32_t>(Dk_ * sizeof(bfloat16_t)), 0, 0, 0};
            DataCopyPad(sTP_->stateOut[nvId * Dv_ * Dk_], stateOut, params);
            outQueue_.FreeTensor(stateOut);
        }
    }

    __aicore__ inline void LoadStateUpdateWs()
    {
        DataCopyExtParams params{static_cast<uint16_t>(Dv_),
                                 static_cast<uint32_t>(Dk_ * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
        DataCopyPad(tmpBuffer2_, wsStateUpdateGm_, params, padParams);
        SetFlag<HardEvent::MTE2_V>(MTE2_V_EVENT);
        WaitFlag<HardEvent::MTE2_V>(MTE2_V_EVENT);
    }

    // ================== AIC ==================

    __aicore__ inline void ProcessAic(int64_t nvId, int64_t cId, int64_t length)
    {
        uint64_t mm_offset0 = nvId * Sp_ * Dk_ + length * Dk_;
        uint64_t mm_offset1 = nvId * Sp_ * Dv_ + length * Dv_;
        auto stateForAic = sTP_->curStateBf16[nvId * Dv_ * Dk_];

        // Wait for AIV to prepare state
        CrossCoreWaitFlag<0x4, PIPE_MTE1>(0x2);

        // W@S → FP32 workspace (no accum)
        GlobalTensor<float> wsVPrimeGm;
        wsVPrimeGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(sTP_->wsVPrime + (uint64_t)coreId_ * chunkSize_ * Dv_ * sizeof(float)));
        CalVPrime(sTP_->kCumdecay[mm_offset0], stateForAic, wsVPrimeGm);
        CrossCoreSetFlag<0x4, PIPE_MTE1>(0x6); // notify AIV: W@S done

        // Q'@S → out (does not depend on vInner)
        CalAttnInter(sTP_->qPrime[mm_offset0], stateForAic,
                     sTP_->out[nvId * Dv_ + cId * chunkSize_ * Nv_ * Dv_]);

        // Wait for AIV to compute U-WS and write to vInner
        CrossCoreWaitFlag<0x4, PIPE_MTE1>(0x7);

        // (U-WS)^T @ kg → FP32 workspace (no accum, no atomicAdd)
        GlobalTensor<float> wsStateUpdateGm;
        wsStateUpdateGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(sTP_->wsStateUpdate + (uint64_t)coreId_ * Dv_ * Dk_ * sizeof(float)));
        CalStateNew(sTP_->vInner[mm_offset1], sTP_->kg[mm_offset0], wsStateUpdateGm);
        CrossCoreSetFlag<0x4, PIPE_MTE1>(0x8); // notify AIV: state matmul done
    }

    // ================== Matmul helpers ==================

    template <typename MMType, typename AT, typename BT, typename CT>
    __aicore__ inline void RunMatmul(MMType *mm, GlobalTensor<AT> a, bool transA, GlobalTensor<BT> b, bool transB,
                                     GlobalTensor<CT> c, int64_t M, int64_t N, int64_t K, int32_t accum = 0)
    {
        mm->SetOrgShape(M, N, K);
        mm->SetSingleShape(M, N, K);
        mm->SetTensorA(a, transA);
        mm->SetTensorB(b, transB);
        mm->IterateAll(c, accum);
        mm->End();
    }

    template <typename MMType, typename AT, typename BT, typename CT>
    __aicore__ inline void RunMatmul(MMType *mm, GlobalTensor<AT> a, bool transA, GlobalTensor<BT> b, bool transB,
                                     GlobalTensor<CT> c, int64_t M, int64_t N, int64_t K, int64_t Ka, int64_t Kb,
                                     int32_t accum = 0)
    {
        mm->SetOrgShape(M, N, K, Ka, Kb);
        mm->SetSingleShape(M, N, K);
        mm->SetTensorA(a, transA);
        mm->SetTensorB(b, transB);
        mm->IterateAll(c, accum);
        mm->End();
    }

    __aicore__ inline void CalAttnInter(GlobalTensor<bfloat16_t> qPrime, GlobalTensor<bfloat16_t> state,
                                        GlobalTensor<bfloat16_t> out)
    {
        RunMatmul(sTP_->mm1Bf16, qPrime, false, state, true, out, curChunkSize_, Dv_, Dk_, Dk_, Nv_ * Dv_);
    }

    __aicore__ inline void CalVPrime(GlobalTensor<bfloat16_t> kCumdecay, GlobalTensor<bfloat16_t> state,
                                     GlobalTensor<float> vPrime)
    {
        RunMatmul(sTP_->mm1Fp32, kCumdecay, false, state, true, vPrime, curChunkSize_, Dv_, Dk_);
    }

    __aicore__ inline void CalStateNew(GlobalTensor<bfloat16_t> vInner, GlobalTensor<bfloat16_t> kg,
                                       GlobalTensor<float> stateUpdate)
    {
        RunMatmul(sTP_->mm1Fp32, vInner, true, kg, false, stateUpdate, Dv_, Dk_, curChunkSize_);
    }

    // ================== Copy helpers ==================

    __aicore__ inline void CopyInState(GlobalTensor<stateType> tmpGM)
    {
        LocalTensor<stateType> inLocal = inQueue_.AllocTensor<stateType>();
        int padding;
        uint32_t dstStride = 0;
        if constexpr (kIsFp32) {
            int32_t floatAligned = Ceil(Dk_, BLOCK_FLOAT_NUM) * BLOCK_FLOAT_NUM;
            padding = floatAligned - Dk_;
            if (floatAligned < curDk_) {
                uint32_t dataBlocks = static_cast<uint32_t>(floatAligned * sizeof(float) / BLOCK_SIZE);
                uint32_t rowBlocks = static_cast<uint32_t>(curDk_ * sizeof(float) / BLOCK_SIZE);
                dstStride = rowBlocks - dataBlocks;
                Duplicate<stateType>(inLocal, static_cast<stateType>(0), Dv_ * curDk_);
                SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT);
                WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT);
            }
        } else {
            padding = Ceil(Dk_, BLOCK_SIZE / sizeof(stateType)) * (BLOCK_SIZE / sizeof(stateType)) - Dk_;
        }
        DataCopyExtParams inParams{static_cast<uint16_t>(Dv_), static_cast<uint32_t>(Dk_ * sizeof(stateType)),
                                   static_cast<uint32_t>(0), dstStride, 0};
        DataCopyPadExtParams<stateType> copyPadParams{true, 0, static_cast<uint8_t>(padding), 0};
        DataCopyPad(inLocal, tmpGM, inParams, copyPadParams);
        inQueue_.EnQue(inLocal);
    }

private:
    StageTwoParams<stateType> *sTP_;
    TPipe *pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM_ONE> inQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM_ONE> outQueue_;
    TBuf<TPosition::VECCALC> tmpBuff_;
    LocalTensor<float> tmpBuffer1_;
    LocalTensor<float> tmpBuffer2_;
    LocalTensor<float> lastGCum_;
    GlobalTensor<float> wsVPrimeGm_;
    GlobalTensor<float> wsStateUpdateGm_;
    int64_t Nk_;
    int64_t Nv_;
    int64_t Dk_;
    int64_t Dv_;
    int64_t curDk_;
    int64_t curDv_;
    int64_t seqLength_;
    int32_t chunkSize_;
    int32_t curChunkSize_;
    int64_t Sp_;
    int32_t chunkNum_;
    int32_t coreNum_;
    int32_t coreId_;
    int64_t stateStride1_;
    bool useInitialState_;
    bool isFirstGroup_;
};
} // namespace ChunkGatedDeltaRule
#endif // CHUNK_GATED_DELTA_RULE_STAGE2_ARCH35_H
