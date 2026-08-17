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
 * \file chunk_gated_delta_rule_stage1.h
 * \brief
 */
#ifndef CHUNK_GATED_DELTA_RULE_STAGE1_H
#define CHUNK_GATED_DELTA_RULE_STAGE1_H

#include "kernel_tiling/kernel_tiling.h"
#include "chunk_gated_delta_rule_utils.h"
#include "../chunk_gated_delta_rule_tiling_data.h"
#include "vf/chunk_gated_delta_rule_stage1_vf.h"

namespace ChunkGatedDeltaRule {
using namespace AscendC;
using namespace matmul;

using aT1 = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t>;
using bT1 = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t>;
template <typename cType>
using StageOneMTT = matmul::MatmulImpl<aT1, bT1, MatmulType<TPosition::GM, CubeFormat::ND, cType>>;

constexpr uint64_t UB_REST_BYTES = 140 * 1024; // 140KB
constexpr uint64_t INVERSE_SHAPE = 32;         // 对角块边长
constexpr uint64_t INVERSE_COUNT = 5;          // 求逆所需空间
constexpr uint32_t ALIGN_SIZE = 16;
constexpr uint32_t MAX_PARALLEL_NUM = 6;
constexpr uint32_t HALF_TWO = 2;

// Matmul 形状参数结构体
struct MatmulShapeParams {
    uint64_t m;  // 原始 M 维度
    uint64_t n;  // 原始 N 维度
    uint64_t k;  // 原始 K 维度
    uint64_t sm; // 单次计算 M 维度
    uint64_t sn; // 单次计算 N 维度
    uint64_t sk; // 单次计算 K 维度
    uint64_t ka = 0; // R2: A 矩阵行 stride (0=contiguous, use k)
    uint64_t kb = 0; // R2: B 矩阵行 stride (0=contiguous, use k)
};

template <typename vInnerType>
struct GDRStageOneInitParams {
    // input
    GlobalTensor<bfloat16_t> query; // (T, Nk, Dk)
    GlobalTensor<bfloat16_t> key;   // (T, Nk, Dk)
    GlobalTensor<bfloat16_t> value; // (T, Nv, Dv)
    GlobalTensor<bfloat16_t> beta;  // (T, Nv)
    GlobalTensor<float> g;          // (T, Nv)
    // ouput
    GlobalTensor<float> gCumExp;        // (Nv, cg_len)
    GlobalTensor<bfloat16_t> kCumdecay; // (Nv, cg_len, Dk)
    GlobalTensor<vInnerType> vInner;    // (Nv, cg_len, Dv)
    GlobalTensor<bfloat16_t> qPrime;    // (Nv, cg_len, Dk)
    GlobalTensor<bfloat16_t> kG;        // (Nv, cg_len, Dk)
    GlobalTensor<bfloat16_t> qK;        // (Nv, cg_len, C)
    // other
    GM_ADDR ws;
    GlobalTensor<float> stageOneMask; // (Nv, cg_len, C)
    ChunkGroup cg;
};

template <bool kStateIsFp32 = false, bool gOptional = false>
class Stage1 {
public:
    using vInnerType = std::conditional_t<kStateIsFp32, float, bfloat16_t>;
    using MMBf16 = StageOneMTT<bfloat16_t>;
    using MMVInner = StageOneMTT<vInnerType>;
    __aicore__ inline Stage1(MMBf16 &mmBf16, MMVInner &mmVInner) : mmBf16_(mmBf16), mmVInner_(mmVInner)
    {
    }
    __aicore__ inline void SetGlobalTensors(const GDRStageOneInitParams<vInnerType> &initParams)
    {
        queryGm_ = initParams.query;
        keyGm_ = initParams.key;
        valueGm_ = initParams.value;
        betaGm_ = initParams.beta;

        outGCumExpGm_ = initParams.gCumExp;
        outKCumdecayGm_ = initParams.kCumdecay;
        outVInnerGm_ = initParams.vInner;
        outQPrimeGm_ = initParams.qPrime;
        outKgBaseGm_ = initParams.kG;
        outQkGm_ = initParams.qK;
        stageOneMask_ = initParams.stageOneMask;

        if constexpr (gOptional) {
            gGm_ = initParams.g;
        }
        // ccOffset_
        uint64_t workSpaceOffset = 0;
        uint64_t curWS = coreIdx_ * paraNum_ * ccOffset_ * sizeof(bfloat16_t);
        kkWsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(initParams.ws + workSpaceOffset + curWS));
        workSpaceOffset += coreNum_ * paraNum_ * ccOffset_ * sizeof(bfloat16_t);

        attnWsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(initParams.ws + workSpaceOffset + curWS));
        workSpaceOffset += coreNum_ * paraNum_ * ccOffset_ * sizeof(bfloat16_t);

        // R5: attnWsNegGm_ stores A_beta_gexp for matmul4 (k_cumdecay)
        attnWsNegGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(initParams.ws + workSpaceOffset + curWS));
    }

    __aicore__ inline void InitLocalBuffers()
    {
        maxLen_ = AscendC::Std::max(AscendC::Std::max(dvAligned_ / HALF_TWO, dkAligned_ / HALF_TWO), chunkSize_);
        pipe_->InitBuffer(inQueue_, BUFFER_NUM_ONE, chunkSize_ * maxLen_ * sizeof(float));
        pipe_->InitBuffer(outQueue_, BUFFER_NUM_ONE, chunkSize_ * maxLen_ * sizeof(float));
        if constexpr (gOptional) {
            pipe_->InitBuffer(gOutQueue_, BUFFER_NUM_ONE, chunkSize_ * sizeof(float));
        }

        pipe_->InitBuffer(tmpBuff_, UB_REST_BYTES);
        uint32_t buffOffset = 0;
        betaUbBfloat16_ = tmpBuff_.GetWithOffset<bfloat16_t>(static_cast<uint32_t>(halfChunkSize_), buffOffset);
        buffOffset += halfChunkSize_ * sizeof(bfloat16_t);

        gCumUbFloat_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(chunkSize_), buffOffset);
        buffOffset += chunkSize_ * sizeof(float);

        gBUbFloat_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(halfChunkSize_), buffOffset);
        buffOffset += halfChunkSize_ * sizeof(float);

        gEndBroadUbFloat_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(halfChunkSize_), buffOffset);
        buffOffset += halfChunkSize_ * sizeof(float);

        betaUbFloat_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(halfChunkSize_ * paraNum_), buffOffset);
        buffOffset += halfChunkSize_ * sizeof(float) * paraNum_;

        uint64_t tmpBufferLen = chunkSize_ * maxLen_ * paraNum_;
        gBroadUbFloat_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(tmpBufferLen), buffOffset);
        gammaUbFloat_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(tmpBufferLen), buffOffset);
        valueUbFloat_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(tmpBufferLen), buffOffset);
        qUbFloat_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(tmpBufferLen), buffOffset);
        buffOffset += tmpBufferLen * sizeof(float);

        tmpBufferLen = chunkSize_ * maxLen_;
        gTransBroadUbFloat_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(tmpBufferLen), buffOffset);
        attnUbFloat_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(tmpBufferLen), buffOffset);
        gCumExpBroadUbFloat_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(tmpBufferLen), buffOffset);
        buffOffset += tmpBufferLen * sizeof(float);

        tmpBufferLen = halfChunkSize_ * dkAligned_;
        qUbFloatCon_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(tmpBufferLen), buffOffset);
        kUbFloatCon_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(tmpBufferLen), buffOffset);
        buffOffset += tmpBufferLen * sizeof(float);

        inputGatherBuffer_ = tmpBuff_.GetWithOffset<uint32_t>(static_cast<uint32_t>(chunkSize_), buffOffset);
        buffOffset += chunkSize_ * sizeof(uint32_t);

        gCumSumUbFloat_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(chunkSize_ * paraNum_), buffOffset);
        buffOffset += chunkSize_ * sizeof(float) * paraNum_;

        gCumExpUbFloat_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(chunkSize_ * paraNum_), buffOffset);
        buffOffset += chunkSize_ * sizeof(float) * paraNum_;

        tmpBufferLen = halfChunkSize_ * halfChunkSize_ * INVERSE_COUNT;
        inverseBuffer_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(tmpBufferLen), buffOffset);
        qPrimeUbFloat_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(tmpBufferLen), buffOffset);
        vBetaUbFloat_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(tmpBufferLen), buffOffset);
        gBKUbFloat_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(tmpBufferLen), buffOffset);
        kgUbFloat_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(tmpBufferLen), buffOffset);
        kkLocal_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(tmpBufferLen), buffOffset);
        buffOffset += tmpBufferLen * sizeof(float);

        inverseRes_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(chunkSize_ * halfChunkSize_), buffOffset);
    }

    __aicore__ inline void InitGatherBuffer()
    {
        for (uint32_t i = 0; i < chunkSize_; ++i) {
            inputGatherBuffer_.SetValue(i, i * BLOCK_SIZE);
        }
        int32_t eventID = static_cast<int32_t>(pipe_->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventID);
        WaitFlag<HardEvent::S_V>(eventID);
    }

    __aicore__ inline void Init(const GDRStageOneInitParams<vInnerType> &initParams, TPipe *pipe,
                                const ChunkGatedDeltaRuleTilingData *tilingData)
    {
        pipe_ = pipe;
        tiling_ = tilingData;
        nk_ = tiling_->nk;
        nv_ = tiling_->nv;
        chunkSize_ = tiling_->chunkSize;
        dk_ = tiling_->dk;
        dv_ = tiling_->dv;
        paraNum_ = tiling_->stageOneParaNum;
        dkAligned_ = (dk_ + ALIGN_SIZE - 1) / ALIGN_SIZE * ALIGN_SIZE;
        dvAligned_ = (dv_ + ALIGN_SIZE - 1) / ALIGN_SIZE * ALIGN_SIZE;
        scale_ = tiling_->scale;
        coreNum_ = tiling_->aiCoreNum;
        cg_ = initParams.cg;
        vRowStride_ = nv_ * dv_;
        numChunk_ = (cg_.length + chunkSize_ - 1) / chunkSize_;
        subBlockIdx_ = GetSubBlockIdx();
        halfChunkSize_ = chunkSize_ / TASK_RATIO;
        subOffset_ = subBlockIdx_ * halfChunkSize_;
        coreIdx_ = GetBlockIdx();
        ccOffset_ = chunkSize_ * chunkSize_;
        ckOffset_ = chunkSize_ * dk_;
        cvOffset_ = chunkSize_ * dv_;
        qkStride_ = static_cast<uint64_t>(nk_) * dk_; // R2: Q/K row stride in strided GM
        if ASCEND_IS_AIV {
            coreIdx_ /= TASK_RATIO;
            InitLocalBuffers();
            InitGatherBuffer();
        }
        SetGlobalTensors(initParams);
    }

    __aicore__ inline void Process()
    {
        uint32_t totalChunk = nv_ * numChunk_;
        uint32_t tailChunkNum = totalChunk / coreNum_;  // tail核处理的块数
        uint32_t formerChunkNum = tailChunkNum + 1;     // former核处理的块数
        uint32_t formerCoreNum = totalChunk % coreNum_; // former核数量
        uint32_t start;
        uint32_t end;
        if (coreIdx_ < formerCoreNum) {
            start = coreIdx_ * formerChunkNum;
            end = start + formerChunkNum;
        } else {
            start = formerCoreNum * formerChunkNum + (coreIdx_ - formerCoreNum) * tailChunkNum;
            end = start + tailChunkNum;
        }

        for (uint32_t taskId = start; taskId < end; taskId += paraNum_) {
            uint32_t curParaNum = paraNum_ < end - taskId ? paraNum_ : end - taskId;
            // 获取每个chunk有效长度
            for (uint32_t i = 0; i < curParaNum; ++i) {
                uint32_t curTaskId = taskId + i;
                uint64_t curNId = curTaskId % nv_;
                uint64_t curCgId = curTaskId / nv_;
                SetChunkOffset(i, curNId, curCgId);
            }
            ProcessParaChunk(curParaNum);
        }
    }

private:
    // ----------------------------------------------------------
    // SetChunkOffset
    //   curNId  : head 编号 (Nv 维度)
    //   curCgId : CG 内的 chunk 编号 (0 ~ CG_CHUNKS-1)
    // ----------------------------------------------------------
    __aicore__ inline void SetChunkOffset(uint64_t id, uint64_t curNId, uint64_t curCgId)
    {
        validLenBatch_[id] = chunkSize_;
        // 尾chunk处理
        if (curCgId == numChunk_ - 1 && cg_.length % chunkSize_ != 0) {
            validLenBatch_[id] = cg_.length % chunkSize_;
        }
        if (validLenBatch_[id] < halfChunkSize_) {
            subValidLenBatch_[id] = (subBlockIdx_ == 0) ? validLenBatch_[id] : 0;
        } else {
            subValidLenBatch_[id] = (subBlockIdx_ == 0) ? halfChunkSize_ : validLenBatch_[id] - halfChunkSize_;
        }
        // offset
        uint64_t cgLenPad = (cg_.length + chunkSize_ - 1) / chunkSize_ * chunkSize_;
        chunkRowBase_[id] = curNId * cgLenPad + curCgId * chunkSize_;

        chunkStartRowBatch_[id] = cg_.startPos + curCgId * chunkSize_;
        nIdBatch_[id] = curNId;
        bgOffsetBatch_[id] = chunkStartRowBatch_[id] * nv_ + curNId;
    }

    __aicore__ inline void ProcessParaChunk(int32_t curParaNum)
    {
        if ASCEND_IS_AIC {
            ParaChunkAIC(curParaNum);
        }
        if ASCEND_IS_AIV {
            ParaChunkAIV(curParaNum);
        }
    }

    __aicore__ inline void ParaChunkAIC(int32_t curParaNum)
    {
        // R2: KKT = K @ K^T, read directly from strided GM (no contiguous copy, no sync0 needed)
        for (uint32_t i = 0; i < curParaNum; ++i) {
            uint64_t qk_base = chunkStartRowBatch_[i] * nk_ * dk_ + nIdBatch_[i] * nk_ / nv_ * dk_;
            AICProcess(mmBf16_, keyGm_[qk_base], keyGm_[qk_base], kkWsGm_[i * ccOffset_],
                       {chunkSize_, chunkSize_, dk_, chunkSize_, chunkSize_, dk_, qkStride_, qkStride_}, true);
            AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(0x8); // 同步1
        }

        // R2: QKT = Q @ K^T, read directly from strided GM
        for (uint32_t i = 0; i < curParaNum; ++i) {
            uint64_t qk_base = chunkStartRowBatch_[i] * nk_ * dk_ + nIdBatch_[i] * nk_ / nv_ * dk_;
            AICProcess(mmBf16_, queryGm_[qk_base], keyGm_[qk_base],
                       outQkGm_[chunkRowBase_[i] * chunkSize_],
                       {validLenBatch_[i], validLenBatch_[i], dk_, validLenBatch_[i], validLenBatch_[i], dk_,
                        qkStride_, qkStride_},
                       true);
        }
        AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(0xA); // 同步5

        // 求逆左下角矩阵
        for (uint32_t i = 0; i < curParaNum; ++i) {
            AscendC::CrossCoreWaitFlag(0x7); // 同步2
            AttnInverseMMCompute(i * ccOffset_);
            AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(0xB); // R4: 通知 AIV 可做列缩放
        }

        // R4+R5: k_cumdecay = -A_beta_gexp @ K (读 attnWsNegGm_, K 从 strided GM)
        for (uint32_t i = 0; i < curParaNum; ++i) {
            AscendC::CrossCoreWaitFlag(0x6); // 同步3
            uint64_t qk_base = chunkStartRowBatch_[i] * nk_ * dk_ + nIdBatch_[i] * nk_ / nv_ * dk_;
            AICProcess(mmBf16_, attnWsNegGm_[i * ccOffset_], keyGm_[qk_base],
                       outKCumdecayGm_[chunkRowBase_[i] * dk_],
                       {chunkSize_, dk_, chunkSize_, chunkSize_, dk_, chunkSize_, 0, qkStride_});
        }

        // R4: vInner = A_beta @ V (读 attnWs_, V 从 strided GM)
        for (uint32_t i = 0; i < curParaNum; ++i) {
            AscendC::CrossCoreWaitFlag(0x5); // 同步4
            uint64_t vOffset = chunkStartRowBatch_[i] * vRowStride_ + nIdBatch_[i] * dv_;
            AICProcess(mmVInner_, attnWsGm_[i * ccOffset_], valueGm_[vOffset],
                       outVInnerGm_[chunkRowBase_[i] * dv_],
                       {chunkSize_, dv_, chunkSize_, chunkSize_, dv_, chunkSize_, 0, vRowStride_});
        }
    }

    __aicore__ inline void ParaChunkAIV(int32_t curParaNum)
    {
        // R2: QKPreProcess removed, AIC reads Q/K directly from strided GM
        if constexpr (gOptional) {
            for (uint32_t i = 0; i < curParaNum; ++i) {
                // g_cum_exp = g.cumsum(dim=-1).exp()
                GCumExpCompute(gGm_[bgOffsetBatch_[i]], outGCumExpGm_[chunkRowBase_[i]],
                               gCumSumUbFloat_[i * chunkSize_], gCumExpUbFloat_[i * chunkSize_], validLenBatch_[i]);
                // attn_1 = (g_cum_exp[:None] / g_cum_exp[None,:]) * mask
                uint64_t gUbOffset = i * chunkSize_ * maxLen_;
                GammaCompute(gBroadUbFloat_[gUbOffset], gammaUbFloat_[gUbOffset], gCumSumUbFloat_[i * chunkSize_]);
            }
        }

        for (uint32_t i = 0; i < curParaNum; ++i) {
            uint64_t betaUbOffset = i * halfChunkSize_;
            BetaCopyInWithStride(betaGm_[bgOffsetBatch_[i]], betaUbFloat_[betaUbOffset], subValidLenBatch_[i]);
        }

        for (uint32_t i = 0; i < curParaNum; ++i) {
            AscendC::CrossCoreWaitFlag(0x8); // 同步1
            // attn_1 = kkt * attn_1
            KKBetaCompute(kkWsGm_[i * ccOffset_], betaUbFloat_[i * halfChunkSize_]);
            // attn_1对角块求逆，对角块shape为INVERSE_SHAPE=32
            InverseCompute(attnWsGm_[i * ccOffset_], gammaUbFloat_[i * chunkSize_ * maxLen_]);
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(0x7); // 同步2
        }

        // R4: 列缩放 β → A_beta → 写 attnWs (供 matmul5 v_inner)
        for (uint32_t i = 0; i < curParaNum; ++i) {
            AscendC::CrossCoreWaitFlag(0xB); // ← AIC: AttnInverseMMCompute 完成
            ColumnScaleBeta(attnWsGm_[i * ccOffset_], betaUbFloat_[i * halfChunkSize_]);
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(0x5); // 同步4: A_beta 就绪
        }

        // R5: 列缩放 g_exp → A_beta_gexp → 写 attnWsNegGm_ (供 matmul4 k_cumdecay)
        for (uint32_t i = 0; i < curParaNum; ++i) {
            if constexpr (gOptional) {
                ColumnScaleGExp(attnWsNegGm_[i * ccOffset_], gCumExpUbFloat_[i * chunkSize_]);
            } else {
                // !gOptional: g_exp = 1, A_beta_gexp = A_beta, copy attnWs → attnWsNegGm_
                ColumnScaleGExp(attnWsNegGm_[i * ccOffset_], gCumExpUbFloat_[i * chunkSize_]);
            }
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(0x6); // 同步3: A_beta_gexp 就绪
        }

        // q_prime = query * scale_ * g_cum_exp[:, None]  # (C, Dk)
        for (uint32_t i = 0; i < curParaNum; ++i) {
            // R2: read Q from strided GM instead of queryConGm_
            uint64_t qk_base = chunkStartRowBatch_[i] * nk_ * dk_ + nIdBatch_[i] * nk_ / nv_ * dk_;
            QPrimeCompute(outQPrimeGm_[chunkRowBase_[i] * dk_], gCumSumUbFloat_[i * chunkSize_],
                          gCumExpUbFloat_[i * chunkSize_], queryGm_[qk_base + subOffset_ * qkStride_],
                          subValidLenBatch_[i], qkStride_);
        }
        AscendC::CrossCoreWaitFlag(0xA); // 同步5
    }
    __aicore__ inline void GCumExpCompute(const GlobalTensor<float> src, const GlobalTensor<float> dst,
                                          LocalTensor<float> gCumSumUbFloat, LocalTensor<float> gCumExpUbFloat,
                                          uint32_t validLen)
    {
        // Copy g
        GCopyInWithStride(src, validLen);
        // CumSum计算
        uint32_t outer = 1;
        uint32_t inner = chunkSize_;
        CumSumInfo cumSumInfo{outer, inner};
        CumSum<float>(gCumSumUbFloat, gCumUbFloat_, gCumUbFloat_, cumSumInfo);
        PipeBarrier<PIPE_V>();
        // Exp计算
        Exp<float, 0, true>(gCumExpUbFloat, gCumSumUbFloat, chunkSize_);
        PipeBarrier<PIPE_V>();

        if (subBlockIdx_ == 0) {
            auto tmpOut = gOutQueue_.AllocTensor<float>();
            DataCopy(tmpOut, gCumSumUbFloat, chunkSize_);
            gOutQueue_.EnQue(tmpOut);
            tmpOut = gOutQueue_.DeQue<float>();
            DataCopyExtParams params{static_cast<uint16_t>(1), static_cast<uint32_t>(validLen * sizeof(float)), 0, 0,
                                     0};
            DataCopyPad(dst, tmpOut, params);
            gOutQueue_.FreeTensor(tmpOut);
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void GammaCompute(const LocalTensor<float> gBroadUbFloat, LocalTensor<float> gammaUbFloat,
                                        LocalTensor<float> gCumSumUbFloat)
    {
        // BroadCast
        uint32_t divShape[2] = {chunkSize_, chunkSize_};
        uint32_t gShape[2] = {chunkSize_, 1};
        uint32_t gTransShape[2] = {1, chunkSize_};
        Broadcast<float, BROADCAST_AXIS, 1>(gBroadUbFloat, gCumSumUbFloat, divShape, gShape);
        Broadcast<float, BROADCAST_AXIS, 0>(gTransBroadUbFloat_, gCumSumUbFloat, divShape, gTransShape);
        PipeBarrier<PIPE_V>();
        // div
        Sub(gammaUbFloat, gBroadUbFloat, gTransBroadUbFloat_, ccOffset_);
        PipeBarrier<PIPE_V>();
        // mask  1
        DataCopyInFp32(ccOffset_, stageOneMask_[GetBlockIdx() * ccOffset_]);
        auto maskLocal = inQueue_.DeQue<float>();
        Mul(gammaUbFloat, gammaUbFloat, maskLocal, ccOffset_);
        PipeBarrier<PIPE_V>();
        // exp
        Exp<float, 0, true>(gammaUbFloat, gammaUbFloat, ccOffset_);
        PipeBarrier<PIPE_V>();
        // mask  2
        Mul(gammaUbFloat, gammaUbFloat, maskLocal, ccOffset_);
        PipeBarrier<PIPE_V>();

        inQueue_.FreeTensor(maskLocal);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void KKBetaCompute(const GlobalTensor<bfloat16_t> src, LocalTensor<float> betaUbFloat)
    {
        // copy value
        uint32_t kkLength = chunkSize_ * halfChunkSize_;
        uint64_t kkBeginOffset = subOffset_ * chunkSize_;
        DataCopyInBf16(kkLength, src[kkBeginOffset]);
        auto bf16Local = inQueue_.DeQue<bfloat16_t>();
        Cast(kkLocal_, bf16Local, RoundMode::CAST_NONE, kkLength);
        PipeBarrier<PIPE_V>();
        inQueue_.FreeTensor(bf16Local);

        uint32_t betaShape[2] = {halfChunkSize_, 1};
        uint32_t kkShape[2] = {halfChunkSize_, chunkSize_};
        Broadcast<float, BROADCAST_AXIS, 1>(attnUbFloat_, betaUbFloat, kkShape, betaShape);
        PipeBarrier<PIPE_V>();
        Mul(attnUbFloat_, kkLocal_, attnUbFloat_, chunkSize_ * halfChunkSize_);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void InverseCompute(const GlobalTensor<bfloat16_t> dst, LocalTensor<float> gammaUbFloat)
    {
        uint64_t curVecLen = chunkSize_ * halfChunkSize_;
        if constexpr (gOptional) {
            Mul(attnUbFloat_, attnUbFloat_, gammaUbFloat[subOffset_ * chunkSize_], curVecLen);
        } else {
            DataCopyInFp32(curVecLen, stageOneMask_[GetBlockIdx() * ccOffset_ + subOffset_ * chunkSize_]);
            auto maskLocal = inQueue_.DeQue<float>();
            Mul(attnUbFloat_, attnUbFloat_, maskLocal, curVecLen);
            inQueue_.FreeTensor(maskLocal);
        }
        PipeBarrier<PIPE_V>();

        Muls(inverseRes_, attnUbFloat_, static_cast<float>(-1.0), curVecLen);
        Duplicate(inverseBuffer_, static_cast<float>(0.0), INVERSE_SHAPE * chunkSize_);
        PipeBarrier<PIPE_V>();
        int32_t eventID = static_cast<int32_t>(pipe_->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventID);
        WaitFlag<HardEvent::V_S>(eventID);
        for (uint32_t j = 0; j < INVERSE_SHAPE; ++j) {
            inverseBuffer_.SetValue(j * chunkSize_ + j, static_cast<float>(1.0));
        }
        eventID = static_cast<int32_t>(pipe_->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventID);
        WaitFlag<HardEvent::S_V>(eventID);

        InverseAIVVF<INVERSE_SHAPE>(attnUbFloat_, inverseRes_, inverseBuffer_, subOffset_, chunkSize_);
        PipeBarrier<PIPE_V>();
        auto tmp = outQueue_.AllocTensor<bfloat16_t>();
        Cast(tmp, inverseRes_, RoundMode::CAST_RINT, curVecLen);
        outQueue_.EnQue(tmp);
        DataCopyOutBf16(halfChunkSize_, chunkSize_, chunkSize_, dst[subOffset_ * chunkSize_]);
    }

    // R4: A_beta = A @ diag(beta) — column scaling (第 j 列乘 beta[j])
    //   Read A (BF16 H×C) from attnWs GM → Cast FP32 → col-scale beta → Cast BF16 → write back attnWs
    __aicore__ inline void ColumnScaleBeta(const GlobalTensor<bfloat16_t> aWsGm, LocalTensor<float> betaUbFloat)
    {
        uint64_t curVecLen = chunkSize_ * halfChunkSize_;
        DataCopyInBf16(curVecLen, aWsGm[subOffset_ * chunkSize_]);
        auto bf16Local = inQueue_.DeQue<bfloat16_t>();
        Cast(inverseRes_, bf16Local, RoundMode::CAST_NONE, curVecLen);
        PipeBarrier<PIPE_V>();
        inQueue_.FreeTensor(bf16Local);
        // Column scale: A_beta[i,j] = A[i,j] * beta[j]
        uint32_t betaShape[2] = {halfChunkSize_, 1};
        uint32_t aShape[2] = {halfChunkSize_, chunkSize_};
        Broadcast<float, BROADCAST_AXIS, 1>(attnUbFloat_, betaUbFloat, aShape, betaShape);
        PipeBarrier<PIPE_V>();
        Mul(inverseRes_, inverseRes_, attnUbFloat_, curVecLen);
        PipeBarrier<PIPE_V>();
        auto tmp = outQueue_.AllocTensor<bfloat16_t>();
        Cast(tmp, inverseRes_, RoundMode::CAST_RINT, curVecLen);
        outQueue_.EnQue(tmp);
        DataCopyOutBf16(halfChunkSize_, chunkSize_, chunkSize_, aWsGm[subOffset_ * chunkSize_]);
    }

    // R5: A_beta_gexp = A_beta @ diag(g_exp) — column scaling (第 j 列乘 g_exp[j])
    //   A_beta still in inverseRes_ (UB), col-scale g_exp, negate, → Cast BF16 → write attnWsNegGm_
    __aicore__ inline void ColumnScaleGExp(const GlobalTensor<bfloat16_t> negWsGm, LocalTensor<float> gCumExpUbFloat)
    {
        uint64_t curVecLen = chunkSize_ * halfChunkSize_;
        if constexpr (gOptional) {
            uint32_t gExpShape[2] = {halfChunkSize_, 1};
            uint32_t aShape[2] = {halfChunkSize_, chunkSize_};
            Broadcast<float, BROADCAST_AXIS, 1>(attnUbFloat_, gCumExpUbFloat[subOffset_], aShape, gExpShape);
            PipeBarrier<PIPE_V>();
            Mul(inverseRes_, inverseRes_, attnUbFloat_, curVecLen);
            PipeBarrier<PIPE_V>();
        }
        // Negate: k_cumdecay = -A_beta_gexp @ K (kernel uses negative + accum=1)
        Muls(inverseRes_, inverseRes_, static_cast<float>(-1.0), curVecLen);
        PipeBarrier<PIPE_V>();
        auto tmp = outQueue_.AllocTensor<bfloat16_t>();
        Cast(tmp, inverseRes_, RoundMode::CAST_RINT, curVecLen);
        outQueue_.EnQue(tmp);
        DataCopyOutBf16(halfChunkSize_, chunkSize_, chunkSize_, negWsGm[subOffset_ * chunkSize_]);
    }

    // R2: q_prime = query * scale * g_cum_exp[:, None], read Q from strided GM
    __aicore__ inline void QPrimeCompute(const GlobalTensor<bfloat16_t> outQPrimeGm, LocalTensor<float> gCumSumUbFloat,
                                         LocalTensor<float> gCumExpUbFloat,
                                         const GlobalTensor<bfloat16_t> queryStridedGm, uint32_t subValidRows,
                                         uint64_t srcRowStride)
    {
        // R2: data copy in qUbFloatCon_ from strided GM
        DataCopyInBf16WithStride(subValidRows, dk_, queryStridedGm, srcRowStride);
        auto bf16Query = inQueue_.DeQue<bfloat16_t>();
        Cast(qUbFloatCon_, bf16Query, RoundMode::CAST_NONE, subValidRows * dkAligned_);
        PipeBarrier<PIPE_V>();
        inQueue_.FreeTensor(bf16Query);
        if (subValidRows < halfChunkSize_) {
            Duplicate(qUbFloatCon_[subValidRows * dkAligned_], static_cast<float>(0.0f),
                      (halfChunkSize_ - subValidRows) * dkAligned_);
            PipeBarrier<PIPE_V>();
        }
        // query * scale
        if constexpr (gOptional) {
            Muls(qUbFloat_, qUbFloatCon_, scale_, halfChunkSize_ * dkAligned_);
            PipeBarrier<PIPE_V>();
            uint32_t gCumExpShape[2] = {halfChunkSize_, 1};
            uint32_t qShape[2] = {halfChunkSize_, dkAligned_};
            Broadcast<float, BROADCAST_AXIS, 1>(gCumExpBroadUbFloat_, gCumExpUbFloat[subOffset_], qShape, gCumExpShape);
            PipeBarrier<PIPE_V>();
            // query * scale * exp(g_cum)[:, None]       # (C, Dk)
            Mul(qPrimeUbFloat_, qUbFloat_, gCumExpBroadUbFloat_, halfChunkSize_ * dkAligned_);
        } else {
            Muls(qPrimeUbFloat_, qUbFloatCon_, scale_, halfChunkSize_ * dkAligned_);
        }
        PipeBarrier<PIPE_V>();
        uint64_t qgBeginOffset = subOffset_ * dk_;
        qPrimeLocal_ = outQueue_.AllocTensor<bfloat16_t>();
        Cast(qPrimeLocal_, qPrimeUbFloat_, RoundMode::CAST_RINT, halfChunkSize_ * dkAligned_);
        outQueue_.EnQue<bfloat16_t>(qPrimeLocal_);
        DataCopyOutBf16(halfChunkSize_, dk_, dkAligned_, outQPrimeGm[qgBeginOffset]); // stage1 out
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void DataCopyInFp32(uint64_t len, GlobalTensor<float> y)
    {
        DataCopyPadExtParams<float> padParams;
        DataCopyExtParams kkParams{static_cast<uint16_t>(1), static_cast<uint32_t>(len * sizeof(float)), 0, 0, 0};
        fp32InLocal_ = inQueue_.AllocTensor<float>();
        DataCopyPad(fp32InLocal_, y, kkParams, padParams);
        inQueue_.EnQue<float>(fp32InLocal_);
    }

    __aicore__ inline void DataCopyInBf16(uint64_t len, GlobalTensor<bfloat16_t> y)
    {
        DataCopyPadExtParams<bfloat16_t> padParams;
        DataCopyExtParams kkParams{static_cast<uint16_t>(1), static_cast<uint32_t>(len * sizeof(bfloat16_t)), 0, 0, 0};
        bf16InLocal_ = inQueue_.AllocTensor<bfloat16_t>();
        DataCopyPad(bf16InLocal_, y, kkParams, padParams);
        inQueue_.EnQue<bfloat16_t>(bf16InLocal_);
    }

    __aicore__ inline void BetaCopyInWithStride(const GlobalTensor<bfloat16_t> src, LocalTensor<float> betaUbFloat,
                                                uint32_t subValidRows)
    {
        uint64_t betaBeginOffset = subOffset_ * nv_;
        DataCopyInBf16WithStride(subValidRows, 1, src[betaBeginOffset], nv_);
        betaLocal_ = inQueue_.DeQue<bfloat16_t>();
        if (subValidRows < halfChunkSize_) {
            Duplicate(betaUbBfloat16_, bfloat16_t(0.0f), halfChunkSize_);
            PipeBarrier<PIPE_V>();
        }
        Gather(betaUbBfloat16_, betaLocal_, inputGatherBuffer_, static_cast<uint32_t>(0), subValidRows);
        PipeBarrier<PIPE_V>();

        Cast(betaUbFloat, betaUbBfloat16_, AscendC::RoundMode::CAST_NONE, halfChunkSize_);
        PipeBarrier<PIPE_V>();
        inQueue_.FreeTensor(betaLocal_);
    }

    __aicore__ inline void GCopyInWithStride(const GlobalTensor<float> src, uint32_t validLen)
    {
        DataCopyInFp32WithStride(validLen, 1, src, nv_);
        gLocal_ = inQueue_.DeQue<float>();
        if (validLen < chunkSize_) {
            Duplicate(gCumUbFloat_, 0.0f, chunkSize_);
            PipeBarrier<PIPE_V>();
        }
        Gather(gCumUbFloat_, gLocal_, inputGatherBuffer_, static_cast<uint32_t>(0), validLen);
        PipeBarrier<PIPE_V>();

        inQueue_.FreeTensor(gLocal_);
    }

    __aicore__ inline void DataCopyInFp32WithStride(uint64_t rows, // 要搬的行数
                                                    uint64_t cols, // 每行的元素数
                                                    const GlobalTensor<float> src,
                                                    uint64_t srcRowStride, // GM上相邻行的间距(元素数)
                                                    uint32_t dstRowStride = 0)
    {
        DataCopyPadExtParams<float> padParams = {false, static_cast<uint8_t>(0), static_cast<uint8_t>(0),
                                                 static_cast<float>(0)};
        uint32_t srcGap = (srcRowStride - cols) * sizeof(float);
        DataCopyExtParams params{static_cast<uint16_t>(rows), static_cast<uint32_t>(cols * sizeof(float)),
                                 static_cast<uint32_t>(srcGap), static_cast<uint32_t>(dstRowStride), 0};
        fp32InLocal_ = inQueue_.AllocTensor<float>();
        DataCopyPad(fp32InLocal_, src, params, padParams);
        inQueue_.EnQue<float>(fp32InLocal_);
    }

    __aicore__ inline void DataCopyInBf16WithStride(uint64_t rows, // 要搬的行数
                                                    uint64_t cols, // 每行的元素数
                                                    GlobalTensor<bfloat16_t> src,
                                                    uint64_t srcRowStride) // GM上相邻行的间距(元素数)
    {
        DataCopyPadExtParams<bfloat16_t> padParams = {false, static_cast<uint8_t>(0), static_cast<uint8_t>(0),
                                                      static_cast<float>(0)};
        uint32_t srcGap = (srcRowStride - cols) * sizeof(bfloat16_t);
        DataCopyExtParams params{static_cast<uint16_t>(rows), static_cast<uint32_t>(cols * sizeof(bfloat16_t)),
                                 static_cast<uint32_t>(srcGap), 0, 0};
        bf16InLocal_ = inQueue_.AllocTensor<bfloat16_t>();
        DataCopyPad(bf16InLocal_, src, params, padParams);
        inQueue_.EnQue<bfloat16_t>(bf16InLocal_);
    }

    __aicore__ inline void DataCopyOutBf16(uint32_t rows, uint32_t cols, uint32_t colsAligned,
                                           GlobalTensor<bfloat16_t> y)
    {
        auto tmpLocal = outQueue_.DeQue<bfloat16_t>();
        uint32_t srcStride = (colsAligned - cols) * sizeof(bfloat16_t) / BLOCK_SIZE;
        DataCopyExtParams yGMParams{static_cast<uint16_t>(rows), static_cast<uint32_t>(cols * sizeof(bfloat16_t)),
                                    static_cast<uint32_t>(srcStride), 0, 0};
        DataCopyPad(y, tmpLocal, yGMParams);
        outQueue_.FreeTensor(tmpLocal);
    }

    __aicore__ inline void AttnInverseMMCompute(uint64_t offset)
    {
        uint64_t leftDown = offset + chunkSize_ * INVERSE_SHAPE;
        uint64_t rightDown = leftDown + INVERSE_SHAPE;
        // 右矩阵左下角 @ 右矩阵左上角 -> 右矩阵左下角
        AICProcess(mmBf16_, attnWsGm_[leftDown], attnWsGm_[offset], attnWsGm_[leftDown],
                   {chunkSize_, chunkSize_, chunkSize_, INVERSE_SHAPE, INVERSE_SHAPE, INVERSE_SHAPE});
        int32_t eventID = static_cast<int32_t>(pipe_->FetchEventID(HardEvent::FIX_MTE2));
        SetFlag<HardEvent::FIX_MTE2>(eventID);
        WaitFlag<HardEvent::FIX_MTE2>(eventID);
        // 右矩阵右下角 @ 右矩阵左下角 -> 右矩阵左下角
        AICProcess(mmBf16_, attnWsGm_[rightDown], attnWsGm_[leftDown], attnWsGm_[leftDown],
                   {chunkSize_, chunkSize_, chunkSize_, INVERSE_SHAPE, INVERSE_SHAPE, INVERSE_SHAPE});
        eventID = static_cast<int32_t>(pipe_->FetchEventID(HardEvent::FIX_MTE2));
        SetFlag<HardEvent::FIX_MTE2>(eventID);
        WaitFlag<HardEvent::FIX_MTE2>(eventID);
    }

    template <typename MMType, typename CType>
    __aicore__ inline void AICProcess(MMType &mmRef, GlobalTensor<bfloat16_t> x, GlobalTensor<bfloat16_t> y,
                                      GlobalTensor<CType> z, const MatmulShapeParams &shape, bool transB = false)
    {
        if (shape.ka > 0) {
            // R2: strided GM matmul, orgKc=n (matches 3-arg behavior)
            mmRef.SetOrgShape(shape.m, shape.n, shape.ka, shape.kb, shape.n);
        } else {
            mmRef.SetOrgShape(shape.m, shape.n, shape.k);
        }
        mmRef.SetSingleShape(shape.sm, shape.sn, shape.sk);
        mmRef.SetTensorA(x);
        mmRef.SetTensorB(y, transB);
        mmRef.IterateAll(z);
        mmRef.End();
    }

    TPipe *pipe_;
    MMBf16 &mmBf16_;
    MMVInner &mmVInner_;
    const ChunkGatedDeltaRuleTilingData *tiling_;
    ChunkGroup cg_;
    uint32_t nk_;
    uint32_t nv_;
    uint32_t dk_;
    uint32_t dv_;
    uint32_t dkAligned_;
    uint32_t dvAligned_;
    uint32_t numChunk_;
    uint64_t vRowStride_;
    uint32_t halfChunkSize_;
    uint32_t subBlockIdx_;
    uint32_t subOffset_;
    uint32_t coreIdx_;
    uint32_t chunkSize_;
    uint32_t maxLen_;
    uint32_t coreNum_;
    float scale_;
    uint32_t paraNum_;
    uint64_t ccOffset_;
    uint64_t ckOffset_;
    uint64_t cvOffset_;
    uint64_t qkStride_; // R2: Q/K row stride in strided GM = Nk*Dk
    uint32_t validLenBatch_[MAX_PARALLEL_NUM];
    uint32_t subValidLenBatch_[MAX_PARALLEL_NUM];
    uint32_t chunkRowBase_[MAX_PARALLEL_NUM];
    uint64_t chunkStartRowBatch_[MAX_PARALLEL_NUM];
    uint64_t nIdBatch_[MAX_PARALLEL_NUM];
    uint64_t bgOffsetBatch_[MAX_PARALLEL_NUM];

    // chunk GM pointers
    GlobalTensor<bfloat16_t> queryGm_;
    GlobalTensor<bfloat16_t> keyGm_;
    GlobalTensor<bfloat16_t> valueGm_;
    GlobalTensor<bfloat16_t> betaGm_;
    GlobalTensor<float> gGm_;
    GlobalTensor<float> outGCumExpGm_;
    GlobalTensor<bfloat16_t> outKCumdecayGm_;
    GlobalTensor<vInnerType> outVInnerGm_;
    GlobalTensor<bfloat16_t> outQPrimeGm_;
    GlobalTensor<bfloat16_t> outKgBaseGm_;
    GlobalTensor<bfloat16_t> outKgGm_;
    GlobalTensor<bfloat16_t> outQkGm_;
    GlobalTensor<bfloat16_t> kkWsGm_;
    GlobalTensor<bfloat16_t> attnWsGm_;
    GlobalTensor<bfloat16_t> attnWsNegGm_; // R5: A_beta_gexp (negated) for matmul4
    GlobalTensor<float> stageOneMask_;

    // UB queues
    TQue<QuePosition::VECIN, 1> inQueue_;
    TQue<QuePosition::VECOUT, 1> outQueue_;
    TQue<QuePosition::VECOUT, 1> gOutQueue_;

    TBuf<TPosition::VECCALC> tmpBuff_;

    // UB tensors
    LocalTensor<bfloat16_t> betaUbBfloat16_;
    LocalTensor<float> betaUbFloat_;
    LocalTensor<float> valueUbFloat_;
    LocalTensor<float> attnUbFloat_;
    LocalTensor<float> inverseBuffer_;
    LocalTensor<float> inverseRes_;
    LocalTensor<float> gCumUbFloat_;
    LocalTensor<float> gCumSumUbFloat_;
    LocalTensor<float> gCumExpUbFloat_;
    LocalTensor<float> gCumExpBroadUbFloat_;
    LocalTensor<float> gBUbFloat_;
    LocalTensor<float> qUbFloat_;
    LocalTensor<float> gBroadUbFloat_;
    LocalTensor<float> gTransBroadUbFloat_;
    LocalTensor<float> gEndBroadUbFloat_;
    LocalTensor<float> gammaUbFloat_;
    LocalTensor<float> qUbFloatCon_;
    LocalTensor<float> kUbFloatCon_;
    LocalTensor<float> gBKUbFloat_;
    LocalTensor<float> kgUbFloat_;
    LocalTensor<float> vBetaUbFloat_;
    LocalTensor<float> qPrimeUbFloat_;
    LocalTensor<uint32_t> inputGatherBuffer_;

    LocalTensor<bfloat16_t> betaLocal_;
    LocalTensor<bfloat16_t> valueLocal_;
    LocalTensor<bfloat16_t> qPrimeLocal_;
    LocalTensor<bfloat16_t> vBetaLocal_;
    LocalTensor<float> kkLocal_;
    LocalTensor<float> gLocal_;
    LocalTensor<bfloat16_t> gBKLocal_;
    LocalTensor<bfloat16_t> kgLocal_;

    LocalTensor<bfloat16_t> bf16InLocal_;
    LocalTensor<float> fp32InLocal_;
};
} // namespace ChunkGatedDeltaRule
#endif // CHUNK_GATED_DELTA_RULE_STAGE1_H
