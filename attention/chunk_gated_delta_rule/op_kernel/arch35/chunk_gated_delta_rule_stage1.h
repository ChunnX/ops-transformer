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
 * \file chunk_gated_delta_rule_stage1.h
 * \brief arch35 Stage1: 重构后计算流程
 *   - R2: 移除 QKPreprocess，KKT/QKT 直接从 strided GM 读取 (SetOrgShape 指定 Ka/Kb)
 *   - R4: KKT 不含 beta，beta 改为对求逆结果 A 做行缩放
 *   - R5: k_beta_inv = (beta * A * (-gamma)) @ K，AIC matmul 直接用原始 K
 *   - R3/R6: 不再输出 gCum 到 GM (Stage2/Stage3 自行重算 cumsumg)
 */
#ifndef CHUNK_GATED_DELTA_RULE_STAGE1_H
#define CHUNK_GATED_DELTA_RULE_STAGE1_H

#include "kernel_tiling/kernel_tiling.h"
#include "chunk_gated_delta_rule_utils.h"
#include "../chunk_gated_delta_rule_tiling_data.h"

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

struct MatmulShapeParams {
    uint64_t m;
    uint64_t n;
    uint64_t k;
    uint64_t sm;
    uint64_t sn;
    uint64_t sk;
    uint64_t ka;
    uint64_t kb;
};

template <typename vInnerType>
struct GDRStageOneInitParams {
    GlobalTensor<bfloat16_t> query;     // (T, Nk, Dk)
    GlobalTensor<bfloat16_t> key;       // (T, Nk, Dk)
    GlobalTensor<bfloat16_t> value;     // (T, Nv, Dv)
    GlobalTensor<bfloat16_t> beta;      // (T, Nv)
    GlobalTensor<float> g;              // (T, Nv)
    GlobalTensor<float> gCumExp;        // 保留字段，重构后不再写入 GM (R6)
    GlobalTensor<bfloat16_t> kCumdecay; // (Nv, cg_len, Dk) -> kBetaInv (R3/R5)
    GlobalTensor<vInnerType> vInner;    // (Nv, cg_len, Dv)
    GlobalTensor<bfloat16_t> qPrime;    // (Nv, cg_len, Dk)
    GlobalTensor<bfloat16_t> kG;        // (Nv, cg_len, Dk)
    GlobalTensor<bfloat16_t> qK;        // (Nv, cg_len, C)
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
    __aicore__ inline Stage1(MMBf16 &mmBf16, MMVInner &mmVInner)
        : mmBf16_(mmBf16),
          mmVInner_(mmVInner)
    {
    }
    __aicore__ inline void SetGlobalTensors(const GDRStageOneInitParams<vInnerType> &initParams)
    {
        queryGm_ = initParams.query;
        keyGm_ = initParams.key;
        valueGm_ = initParams.value;
        betaGm_ = initParams.beta;
        outKCumdecayGm_ = initParams.kCumdecay;
        outVInnerGm_ = initParams.vInner;
        outQPrimeGm_ = initParams.qPrime;
        outKgBaseGm_ = initParams.kG;
        outQkGm_ = initParams.qK;
        stageOneMask_ = initParams.stageOneMask;
        if constexpr (gOptional) {
            gGm_ = initParams.g;
        }
        // R2: 不再分配 queryConGm_ / keyConGm_
        // R4: 不再分配 vBetaWsGm_
        // R5: 不再分配 gBKWsGm_ (kBetaInv 的左矩阵直接用 attnWs)
        uint64_t workSpaceOffset = 0;
        uint64_t curWS = coreIdx_ * paraNum_ * ckOffset_ * sizeof(bfloat16_t);
        // kkWsGm_ 仍需保留: AIV 侧 KKBetaCompute 读取 KKT 结果做后续向量计算
        kkWsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(initParams.ws + workSpaceOffset + curWS));
        workSpaceOffset += coreNum_ * paraNum_ * ckOffset_ * sizeof(bfloat16_t);

        // attnWsGm_: 求逆结果 A，以及 A_beta、A_beta_neg_gamma 中间结果
        curWS = coreIdx_ * paraNum_ * ccOffset_ * sizeof(bfloat16_t);
        attnWsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(initParams.ws + workSpaceOffset + curWS));
    }

    __aicore__ inline void InitLocalBuffers()
    {
        maxLen_ = AscendC::Std::max(AscendC::Std::max(dvAligned_ / HALF_TWO, dkAligned_ / HALF_TWO), chunkSize_);
        pipe_->InitBuffer(inQueue_, BUFFER_NUM_ONE, chunkSize_ * maxLen_ * sizeof(float));
        pipe_->InitBuffer(outQueue_, BUFFER_NUM_ONE, chunkSize_ * maxLen_ * sizeof(float));

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

        inverseGatherBuffer_ = tmpBuff_.GetWithOffset<uint32_t>(static_cast<uint32_t>(INVERSE_SHAPE), buffOffset);
        buffOffset += INVERSE_SHAPE * sizeof(uint32_t);

        inverseRes_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(chunkSize_ * halfChunkSize_), buffOffset);
    }

    __aicore__ inline void InitGatherBuffer()
    {
        for (uint32_t i = 0; i < chunkSize_; ++i) {
            inputGatherBuffer_.SetValue(i, i * BLOCK_SIZE);
        }
        for (uint32_t i = 0; i < INVERSE_SHAPE; ++i) {
            inverseGatherBuffer_.SetValue<uint32_t>(i, (i * chunkSize_) * sizeof(float));
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
        qkStride_ = static_cast<uint64_t>(nk_) * dk_;
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
        uint32_t tailChunkNum = totalChunk / coreNum_;
        uint32_t formerChunkNum = tailChunkNum + 1;
        uint32_t formerCoreNum = totalChunk % coreNum_;
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
    __aicore__ inline void SetChunkOffset(uint64_t id, uint64_t curNId, uint64_t curCgId)
    {
        validLenBatch_[id] = chunkSize_;
        if (curCgId == numChunk_ - 1 && cg_.length % chunkSize_ != 0) {
            validLenBatch_[id] = cg_.length % chunkSize_;
        }
        if (validLenBatch_[id] < halfChunkSize_) {
            subValidLenBatch_[id] = (subBlockIdx_ == 0) ? validLenBatch_[id] : 0;
        } else {
            subValidLenBatch_[id] = (subBlockIdx_ == 0) ? halfChunkSize_ : validLenBatch_[id] - halfChunkSize_;
        }
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
        // R2: KKT = K @ K^T，直接从 strided GM 读取 (不再经过 queryConGm_/keyConGm_)
        for (uint32_t i = 0; i < curParaNum; ++i) {
            AscendC::CrossCoreWaitFlag(0x9); // 同步0: AIV 已计算 gamma 等
            uint64_t qk_base = chunkStartRowBatch_[i] * nk_ * dk_ + nIdBatch_[i] * nk_ / nv_ * dk_;
            AICProcess(mmBf16_, keyGm_[qk_base], keyGm_[qk_base], kkWsGm_[i * ccOffset_],
                       {chunkSize_, chunkSize_, dk_, chunkSize_, chunkSize_, dk_, qkStride_, qkStride_}, true);
            AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(0x8); // 同步1
        }

        // R2: QKT = Q @ K^T，直接从 strided GM 读取
        for (uint32_t i = 0; i < curParaNum; ++i) {
            uint64_t qk_base = chunkStartRowBatch_[i] * nk_ * dk_ + nIdBatch_[i] * nk_ / nv_ * dk_;
            AICProcess(mmBf16_, queryGm_[qk_base], keyGm_[qk_base],
                       outQkGm_[chunkRowBase_[i] * chunkSize_],
                       {validLenBatch_[i], validLenBatch_[i], dk_, validLenBatch_[i], validLenBatch_[i], dk_,
                        qkStride_, qkStride_},
                       true);
        }
        AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(0xA); // 同步5

        // 求逆左下角矩阵辅助 matmul
        for (uint32_t i = 0; i < curParaNum; ++i) {
            AscendC::CrossCoreWaitFlag(0x7); // 同步2
            AttnInverseMMCompute(i * ccOffset_);
        }

        // R3/R5: kBetaInv = A_beta_neg_gamma @ K，直接从 strided GM 读 K
        for (uint32_t i = 0; i < curParaNum; ++i) {
            AscendC::CrossCoreWaitFlag(0x6); // 同步3
            uint64_t qk_base = chunkStartRowBatch_[i] * nk_ * dk_ + nIdBatch_[i] * nk_ / nv_ * dk_;
            AICProcess(mmBf16_, attnWsGm_[i * ccOffset_], keyGm_[qk_base],
                       outKCumdecayGm_[chunkRowBase_[i] * dk_],
                       {chunkSize_, dk_, chunkSize_, chunkSize_, dk_, chunkSize_, 0, qkStride_});
        }

        // R4: vInner = A_beta @ V，直接从 strided GM 读 V (A_beta 存在 attnWs 中)
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
        // 计算 g_cum / g_cum_exp / gamma (AIV 内部使用，R6 不再写出到 GM)
        if constexpr (gOptional) {
            for (uint32_t i = 0; i < curParaNum; ++i) {
                GCumExpCompute(gGm_[bgOffsetBatch_[i]], gCumSumUbFloat_[i * chunkSize_],
                               gCumExpUbFloat_[i * chunkSize_], validLenBatch_[i]);
                uint64_t gUbOffset = i * chunkSize_ * maxLen_;
                GammaCompute(gBroadUbFloat_[gUbOffset], gammaUbFloat_[gUbOffset], gCumSumUbFloat_[i * chunkSize_]);
            }
        }

        // beta 搬入
        for (uint32_t i = 0; i < curParaNum; ++i) {
            uint64_t betaUbOffset = i * halfChunkSize_;
            BetaCopyInWithStride(betaGm_[bgOffsetBatch_[i]], betaUbFloat_[betaUbOffset], subValidLenBatch_[i]);
        }

        for (uint32_t i = 0; i < curParaNum; ++i) {
            AscendC::CrossCoreWaitFlag(0x8); // 同步1: KKT 已就绪
            // R4: KKBetaCompute 不再乘 beta，attn = kk * gamma (gamma 在 gOptional 时为 decay_mask)
            KKBetaCompute(kkWsGm_[i * ccOffset_]);
            // 求逆: A = (I + kk*gamma)^{-1}，结果存于 attnWs
            InverseCompute(attnWsGm_[i * ccOffset_], gammaUbFloat_[i * chunkSize_ * maxLen_],
                           betaUbFloat_[i * halfChunkSize_]);
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(0x7); // 同步2
        }

        for (uint32_t i = 0; i < curParaNum; ++i) {
            // R5: A_beta *= (-gamma)，结果写回 attnWs 供 AIC matmul 用
            NegGammaMultiply(attnWsGm_[i * ccOffset_], gammaUbFloat_[i * chunkSize_ * maxLen_],
                             betaUbFloat_[i * halfChunkSize_]);
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(0x6); // 同步3
        }

        for (uint32_t i = 0; i < curParaNum; ++i) {
            // kg = k * exp(g_cum[-1] - g_cum)，仍需输出 (Stage2 state 更新用)
            outKgGm_ = outKgBaseGm_[chunkRowBase_[i] * dk_];
            KgCompute(outKgGm_, gCumSumUbFloat_[i * chunkSize_], chunkStartRowBatch_[i], nIdBatch_[i]);
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(0x5); // 同步4: A_beta @ V 可启动
        }

        for (uint32_t i = 0; i < curParaNum; ++i) {
            // q_prime = query * scale * g_cum_exp[:, None]
            uint64_t qk_base = chunkStartRowBatch_[i] * nk_ * dk_ + nIdBatch_[i] * nk_ / nv_ * dk_;
            QPrimeCompute(outQPrimeGm_[chunkRowBase_[i] * dk_], gCumSumUbFloat_[i * chunkSize_],
                          gCumExpUbFloat_[i * chunkSize_], queryGm_[qk_base], subValidLenBatch_[i], qkStride_);
        }
        AscendC::CrossCoreWaitFlag(0xA); // 同步5
    }

    // R6: g_cum / g_cum_exp 仅在 AIV 内部计算，不再 DataCopyPad 到 GM
    __aicore__ inline void GCumExpCompute(const GlobalTensor<float> src, LocalTensor<float> gCumSumUbFloat,
                                          LocalTensor<float> gCumExpUbFloat, uint32_t validLen)
    {
        GCopyInWithStride(src, validLen);
        uint32_t outer = 1;
        uint32_t inner = chunkSize_;
        CumSumInfo cumSumInfo{outer, inner};
        CumSum<float>(gCumSumUbFloat, gCumUbFloat_, gCumUbFloat_, cumSumInfo);
        PipeBarrier<PIPE_V>();
        Exp<float, 0, true>(gCumExpUbFloat, gCumSumUbFloat, chunkSize_);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void GammaCompute(const LocalTensor<float> gBroadUbFloat, LocalTensor<float> gammaUbFloat,
                                        LocalTensor<float> gCumSumUbFloat)
    {
        uint32_t divShape[2] = {chunkSize_, chunkSize_};
        uint32_t gShape[2] = {chunkSize_, 1};
        uint32_t gTransShape[2] = {1, chunkSize_};
        Broadcast<float, BROADCAST_AXIS, 1>(gBroadUbFloat, gCumSumUbFloat, divShape, gShape);
        Broadcast<float, BROADCAST_AXIS, 0>(gTransBroadUbFloat_, gCumSumUbFloat, divShape, gTransShape);
        PipeBarrier<PIPE_V>();
        Sub(gammaUbFloat, gBroadUbFloat, gTransBroadUbFloat_, ccOffset_);
        PipeBarrier<PIPE_V>();
        DataCopyInFp32(ccOffset_, stageOneMask_[GetBlockIdx() * ccOffset_]);
        auto maskLocal = inQueue_.DeQue<float>();
        Mul(gammaUbFloat, gammaUbFloat, maskLocal, ccOffset_);
        PipeBarrier<PIPE_V>();
        Exp<float, 0, true>(gammaUbFloat, gammaUbFloat, ccOffset_);
        PipeBarrier<PIPE_V>();
        Mul(gammaUbFloat, gammaUbFloat, maskLocal, ccOffset_);
        PipeBarrier<PIPE_V>();
        inQueue_.FreeTensor(maskLocal);
        PipeBarrier<PIPE_V>();
    }

    // R4: KKBetaCompute 不再乘 beta，attn = kk * gamma (gOptional) 或 attn = kk * mask (!gOptional)
    __aicore__ inline void KKBetaCompute(const GlobalTensor<bfloat16_t> src)
    {
        uint32_t kkLength = chunkSize_ * halfChunkSize_;
        uint64_t kkBeginOffset = subOffset_ * chunkSize_;
        DataCopyInBf16(kkLength, src[kkBeginOffset]);
        auto bf16Local = inQueue_.DeQue<bfloat16_t>();
        Cast(kkLocal_, bf16Local, RoundMode::CAST_NONE, kkLength);
        PipeBarrier<PIPE_V>();
        inQueue_.FreeTensor(bf16Local);
        // 不再 broadcast beta 并相乘
    }

    // R4/R5: 求逆后施加 beta 行缩放，得到 A_beta (存入 inverseRes_)；-gamma 在 NegGammaMultiply 中施加
    __aicore__ inline void InverseCompute(const GlobalTensor<bfloat16_t> dst, LocalTensor<float> gammaUbFloat,
                                          LocalTensor<float> betaUbFloat)
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
        PipeBarrier<PIPE_V>();
        int32_t eventID = static_cast<int32_t>(pipe_->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventID);
        WaitFlag<HardEvent::V_S>(eventID);

        InverseAIV(subOffset_, INVERSE_SHAPE);
        // inverseRes_ 此时是 A = (I + kk*gamma)^{-1}

        // R4: A_beta = beta[:,None] * A (行缩放)，结果存入 inverseRes_
        uint32_t betaShape[2] = {halfChunkSize_, 1};
        uint32_t aShape[2] = {halfChunkSize_, chunkSize_};
        Broadcast<float, BROADCAST_AXIS, 1>(attnUbFloat_, betaUbFloat, aShape, betaShape);
        PipeBarrier<PIPE_V>();
        Mul(inverseRes_, inverseRes_, attnUbFloat_, halfChunkSize_ * chunkSize_);
        PipeBarrier<PIPE_V>();
    }

    // R5: A_beta *= (-gamma)，结果 Cast 到 BF16 写入 attnWs (供 AIC matmul 用)
    __aicore__ inline void NegGammaMultiply(const GlobalTensor<bfloat16_t> dst, LocalTensor<float> gammaUbFloat,
                                            LocalTensor<float> betaUbFloat)
    {
        uint64_t curVecLen = chunkSize_ * halfChunkSize_;
        // A_beta 已在 inverseRes_ 中
        if constexpr (gOptional) {
            // A_beta_neg_gamma = A_beta * (-gamma) = A_beta * (-1) * gamma
            Mul(attnUbFloat_, inverseRes_, gammaUbFloat[subOffset_ * chunkSize_], curVecLen);
            PipeBarrier<PIPE_V>();
            Muls(attnUbFloat_, attnUbFloat_, static_cast<float>(-1.0), curVecLen);
        } else {
            // !gOptional: gamma 即 mask，A_beta * (-mask)
            DataCopyInFp32(curVecLen, stageOneMask_[GetBlockIdx() * ccOffset_ + subOffset_ * chunkSize_]);
            auto maskLocal = inQueue_.DeQue<float>();
            Mul(attnUbFloat_, inverseRes_, maskLocal, curVecLen);
            PipeBarrier<PIPE_V>();
            Muls(attnUbFloat_, attnUbFloat_, static_cast<float>(-1.0), curVecLen);
            inQueue_.FreeTensor(maskLocal);
        }
        PipeBarrier<PIPE_V>();
        auto tmp = outQueue_.AllocTensor<bfloat16_t>();
        Cast(tmp, attnUbFloat_, RoundMode::CAST_RINT, curVecLen);
        outQueue_.EnQue(tmp);
        DataCopyOutBf16(halfChunkSize_, chunkSize_, chunkSize_, dst[subOffset_ * chunkSize_]);
    }

    __aicore__ inline void InverseAIV(uint64_t offset, uint32_t inverseVecLen)
    {
        uint64_t inverseBufferOffset = 0;
        auto row = inverseBuffer_[inverseBufferOffset];
        inverseBufferOffset += inverseVecLen * inverseVecLen;
        auto col = inverseBuffer_[inverseBufferOffset];
        inverseBufferOffset += inverseVecLen * inverseVecLen + inverseVecLen;
        auto yLocal = inverseBuffer_[inverseBufferOffset];
        inverseBufferOffset += inverseVecLen * inverseVecLen;
        auto eiMatrix = inverseBuffer_[inverseBufferOffset];

        Duplicate(yLocal, static_cast<float>(0.0), inverseVecLen * inverseVecLen);
        Duplicate(eiMatrix, static_cast<float>(0.0), inverseVecLen * inverseVecLen);
        int32_t vsEventID = static_cast<int32_t>(pipe_->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(vsEventID);
        WaitFlag<HardEvent::V_S>(vsEventID);
        for (uint32_t j = 0; j < inverseVecLen; ++j) {
            eiMatrix.SetValue(j * inverseVecLen + j, static_cast<float>(1.0));
        }
        inverseRes_.SetValue(offset, static_cast<float>(1.0));

        uint32_t srcShape[2] = {1, inverseVecLen};
        int32_t eventID = static_cast<int32_t>(pipe_->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventID);
        WaitFlag<HardEvent::S_V>(eventID);
        for (int i = 1; i < inverseVecLen; ++i) {
            uint32_t curI = i - 1;
            uint32_t validRows = inverseVecLen - i;
            Gather(col, attnUbFloat_[offset + i * chunkSize_ + curI], inverseGatherBuffer_, (uint32_t)0, validRows);
            PipeBarrier<PIPE_V>();
            uint32_t dstShape[2] = {validRows, inverseVecLen};
            uint32_t colSrcShape[2] = {validRows, 1};
            Broadcast<float, BROADCAST_AXIS, 1>(col[inverseVecLen], col, dstShape, colSrcShape);
            Broadcast<float, BROADCAST_AXIS, 0>(row, inverseRes_[offset + curI * chunkSize_], dstShape, srcShape);
            PipeBarrier<PIPE_V>();
            MulAddDst(yLocal[i * inverseVecLen], col[inverseVecLen], row, inverseVecLen * validRows);
            PipeBarrier<PIPE_V>();
            Sub(inverseRes_[offset + i * chunkSize_], eiMatrix[i * inverseVecLen],
                yLocal[i * inverseVecLen], inverseVecLen);
            PipeBarrier<PIPE_V>();
        }
    }

    // R5: kg = k * exp(g_cum[-1] - g_cum)，从 strided GM 读 K
    __aicore__ inline void KgCompute(const GlobalTensor<bfloat16_t> outKgGm, LocalTensor<float> gCumSumUbFloat,
                                     uint64_t chunkStartRow, uint64_t nId)
    {
        if constexpr (gOptional) {
            uint32_t gEndShape[2] = {1, 1};
            uint32_t gBroadShape[2] = {halfChunkSize_, 1};
            Broadcast<float, BROADCAST_AXIS, 0>(gEndBroadUbFloat_, gCumSumUbFloat[chunkSize_ - 1], gBroadShape,
                                                gEndShape);
            PipeBarrier<PIPE_V>();
            Sub(gEndBroadUbFloat_, gEndBroadUbFloat_, gCumSumUbFloat[subOffset_], halfChunkSize_);
            PipeBarrier<PIPE_V>();
            Exp<float, 0, true>(gEndBroadUbFloat_, gEndBroadUbFloat_, halfChunkSize_);
            PipeBarrier<PIPE_V>();
            // 从 strided GM 读 K
            uint64_t qk_base = chunkStartRow * nk_ * dk_ + nId * nk_ / nv_ * dk_;
            DataCopyInBf16WithStride(halfChunkSize_, dk_, keyGm_[qk_base + subOffset_ * qkStride_], qkStride_);
            auto bf16Key = inQueue_.DeQue<bfloat16_t>();
            Cast(kUbFloatCon_, bf16Key, RoundMode::CAST_NONE, halfChunkSize_ * dkAligned_);
            PipeBarrier<PIPE_V>();
            inQueue_.FreeTensor(bf16Key);
            uint32_t kShape[2] = {halfChunkSize_, dkAligned_};
            Broadcast<float, BROADCAST_AXIS, 1>(kgUbFloat_, gEndBroadUbFloat_, kShape, gBroadShape);
            PipeBarrier<PIPE_V>();
            Mul(kgUbFloat_, kgUbFloat_, kUbFloatCon_, halfChunkSize_ * dkAligned_);
            PipeBarrier<PIPE_V>();
            uint64_t kgBeginOffset = subOffset_ * dk_;
            kgLocal_ = outQueue_.AllocTensor<bfloat16_t>();
            Cast(kgLocal_, kgUbFloat_, RoundMode::CAST_RINT, halfChunkSize_ * dkAligned_);
            outQueue_.EnQue<bfloat16_t>(kgLocal_);
            DataCopyOutBf16(halfChunkSize_, dk_, dkAligned_, outKgGm[kgBeginOffset]);
        }
    }

    // R4: q_prime = query * scale * g_cum_exp[:, None]，从 strided GM 读 Q
    __aicore__ inline void QPrimeCompute(const GlobalTensor<bfloat16_t> outQPrimeGm, LocalTensor<float> gCumSumUbFloat,
                                         LocalTensor<float> gCumExpUbFloat, const GlobalTensor<bfloat16_t> queryStridedGm,
                                         uint32_t subValidRows, uint64_t srcRowStride)
    {
        DataCopyInBf16WithStride(subValidRows, dk_, queryStridedGm[subOffset_ * srcRowStride], srcRowStride);
        auto bf16Query = inQueue_.DeQue<bfloat16_t>();
        Cast(qUbFloatCon_, bf16Query, RoundMode::CAST_NONE, subValidRows * dkAligned_);
        PipeBarrier<PIPE_V>();
        inQueue_.FreeTensor(bf16Query);
        if (subValidRows < halfChunkSize_) {
            Duplicate(qUbFloatCon_[subValidRows * dkAligned_], static_cast<float>(0.0f),
                      (halfChunkSize_ - subValidRows) * dkAligned_);
            PipeBarrier<PIPE_V>();
        }
        if constexpr (gOptional) {
            Muls(qUbFloat_, qUbFloatCon_, scale_, halfChunkSize_ * dkAligned_);
            PipeBarrier<PIPE_V>();
            uint32_t gCumExpShape[2] = {halfChunkSize_, 1};
            uint32_t qShape[2] = {halfChunkSize_, dkAligned_};
            Broadcast<float, BROADCAST_AXIS, 1>(gCumExpBroadUbFloat_, gCumExpUbFloat[subOffset_], qShape, gCumExpShape);
            PipeBarrier<PIPE_V>();
            Mul(qPrimeUbFloat_, qUbFloat_, gCumExpBroadUbFloat_, halfChunkSize_ * dkAligned_);
        } else {
            Muls(qPrimeUbFloat_, qUbFloatCon_, scale_, halfChunkSize_ * dkAligned_);
        }
        PipeBarrier<PIPE_V>();
        uint64_t qgBeginOffset = subOffset_ * dk_;
        qPrimeLocal_ = outQueue_.AllocTensor<bfloat16_t>();
        Cast(qPrimeLocal_, qPrimeUbFloat_, RoundMode::CAST_RINT, halfChunkSize_ * dkAligned_);
        outQueue_.EnQue(qPrimeLocal_);
        DataCopyOutBf16(halfChunkSize_, dk_, dkAligned_, outQPrimeGm[qgBeginOffset]);
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

    __aicore__ inline void DataCopyInFp32WithStride(uint64_t rows, uint64_t cols, const GlobalTensor<float> src,
                                                    uint64_t srcRowStride, uint32_t dstRowStride = 0)
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

    __aicore__ inline void DataCopyInBf16WithStride(uint64_t rows, uint64_t cols, GlobalTensor<bfloat16_t> src,
                                                    uint64_t srcRowStride)
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
        AICProcess(mmBf16_, attnWsGm_[leftDown], attnWsGm_[offset], attnWsGm_[leftDown],
                   {chunkSize_, chunkSize_, chunkSize_, INVERSE_SHAPE, INVERSE_SHAPE, INVERSE_SHAPE, 0, 0});
        int32_t eventID = static_cast<int32_t>(pipe_->FetchEventID(HardEvent::FIX_MTE2));
        SetFlag<HardEvent::FIX_MTE2>(eventID);
        WaitFlag<HardEvent::FIX_MTE2>(eventID);
        AICProcess(mmBf16_, attnWsGm_[rightDown], attnWsGm_[leftDown], attnWsGm_[leftDown],
                   {chunkSize_, chunkSize_, chunkSize_, INVERSE_SHAPE, INVERSE_SHAPE, INVERSE_SHAPE, 0, 0});
        eventID = static_cast<int32_t>(pipe_->FetchEventID(HardEvent::FIX_MTE2));
        SetFlag<HardEvent::FIX_MTE2>(eventID);
        WaitFlag<HardEvent::FIX_MTE2>(eventID);
    }

    template <typename MMType, typename CType>
    __aicore__ inline void AICProcess(MMType &mmRef, GlobalTensor<bfloat16_t> x, GlobalTensor<bfloat16_t> y,
                                      GlobalTensor<CType> z, const MatmulShapeParams &shape, bool transB = false)
    {
        mmRef.SetOrgShape(shape.m, shape.n, shape.k, shape.ka, shape.kb);
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
    uint64_t qkStride_;
    uint32_t validLenBatch_[MAX_PARALLEL_NUM];
    uint32_t subValidLenBatch_[MAX_PARALLEL_NUM];
    uint32_t chunkRowBase_[MAX_PARALLEL_NUM];
    uint64_t chunkStartRowBatch_[MAX_PARALLEL_NUM];
    uint64_t nIdBatch_[MAX_PARALLEL_NUM];
    uint64_t bgOffsetBatch_[MAX_PARALLEL_NUM];

    GlobalTensor<bfloat16_t> queryGm_;
    GlobalTensor<bfloat16_t> keyGm_;
    GlobalTensor<bfloat16_t> valueGm_;
    GlobalTensor<bfloat16_t> betaGm_;
    GlobalTensor<float> gGm_;
    GlobalTensor<bfloat16_t> outKCumdecayGm_;
    GlobalTensor<vInnerType> outVInnerGm_;
    GlobalTensor<bfloat16_t> outQPrimeGm_;
    GlobalTensor<bfloat16_t> outKgBaseGm_;
    GlobalTensor<bfloat16_t> outKgGm_;
    GlobalTensor<bfloat16_t> outQkGm_;
    GlobalTensor<bfloat16_t> kkWsGm_;
    GlobalTensor<bfloat16_t> attnWsGm_;
    GlobalTensor<float> stageOneMask_;

    TQue<QuePosition::VECIN, 1> inQueue_;
    TQue<QuePosition::VECOUT, 1> outQueue_;

    TBuf<TPosition::VECCALC> tmpBuff_;

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
    LocalTensor<uint32_t> inverseGatherBuffer_;
    LocalTensor<float> qUbFloatCon_;
    LocalTensor<float> kUbFloatCon_;
    LocalTensor<float> gBKUbFloat_;
    LocalTensor<float> kgUbFloat_;
    LocalTensor<float> vBetaUbFloat_;
    LocalTensor<float> qPrimeUbFloat_;
    LocalTensor<uint32_t> inputGatherBuffer_;

    LocalTensor<bfloat16_t> betaLocal_;
    LocalTensor<bfloat16_t> qPrimeLocal_;
    LocalTensor<bfloat16_t> kgLocal_;
    LocalTensor<float> kkLocal_;
    LocalTensor<float> gLocal_;

    LocalTensor<bfloat16_t> bf16InLocal_;
    LocalTensor<float> fp32InLocal_;
};
} // namespace ChunkGatedDeltaRule
#endif // CHUNK_GATED_DELTA_RULE_STAGE1_H
