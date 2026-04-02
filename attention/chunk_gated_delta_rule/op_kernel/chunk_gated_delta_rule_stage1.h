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
#include "chunk_gated_delta_rule_tiling_data.h"

namespace ChunkGatedDeltaRule {
using namespace AscendC;
using namespace matmul;

using aT1 = MatmulType<TPosition::GM, CubeFormat::ND, float>;
using bT1 = MatmulType<TPosition::GM, CubeFormat::ND, float>;
using cT1 = MatmulType<TPosition::GM, CubeFormat::ND, float>;
using StageOneMT = matmul::MatmulImpl<aT1, bT1, cT1>;

constexpr uint64_t UB_REST_BYTES = 151 * 1024;  // 140KB
constexpr uint64_t INVERSE_SHAPE = 32;          // 对角块边长
constexpr uint64_t INVERSE_COUNT = 5;           // 求逆所需空间
constexpr uint32_t ALIGN_SIZE = 16;
constexpr uint32_t MAX_PARALLEL_NUM = 6;

// Matmul 形状参数结构体
struct MatmulShapeParams {
    uint64_t m;    // 原始 M 维度
    uint64_t n;    // 原始 N 维度
    uint64_t k;    // 原始 K 维度
    uint64_t sm;   // 单次计算 M 维度
    uint64_t sn;   // 单次计算 N 维度
    uint64_t sk;   // 单次计算 K 维度
};

struct GDRStageOneInitParams {
    // input
    GlobalTensor<bfloat16_t> query;     // (T, Nk, Dk)
    GlobalTensor<bfloat16_t> key;       // (T, Nk, Dk)
    GlobalTensor<bfloat16_t> value;     // (T, Nv, Dv)
    GlobalTensor<bfloat16_t> beta;      // (T, Nv)
    GlobalTensor<float> g;              // (T, Nv)
    // ouput
    GlobalTensor<float> gCumExp;        // (Nv, cg_len)
    GlobalTensor<float> kCumdecay;      // (Nv, cg_len, Dk)
    GlobalTensor<float> vInner;         // (Nv, cg_len, Dv)
    GlobalTensor<float> qPrime;         // (Nv, cg_len, Dk)
    GlobalTensor<float> kG;             // (Nv, cg_len, Dk)
    GlobalTensor<float> qK;             // (Nv, cg_len, C)
    // other
    GM_ADDR ws;
    GlobalTensor<float> stageOneMask;   // (Nv, cg_len, C)
    ChunkGroup cg;
    bool gOptional;
};

class Stage1 {
public:
    __aicore__ inline Stage1(StageOneMT &mmFp32) : mmFp32(mmFp32) {}
    __aicore__ inline void SetGlobalTensors(const GDRStageOneInitParams &initParams)
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

        if (gOptional_) {
            gGm_ = initParams.g;
        }

        uint64_t workSpaceOffset = 0;
        gBKWsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(initParams.ws + workSpaceOffset +
                                                                  coreIdx_ * paraNum_ * ckOffset_ * sizeof(float)));

        workSpaceOffset += coreNum_ * paraNum_ * ckOffset_ * sizeof(float);
        kkWsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(initParams.ws + workSpaceOffset +
                                                                 coreIdx_ * paraNum_ * ccOffset_ * sizeof(float)));

        workSpaceOffset += coreNum_ * paraNum_ * ccOffset_ * sizeof(float);
        vBetaWsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(initParams.ws + workSpaceOffset +
                                                                    coreIdx_ * paraNum_ * cvOffset_ * sizeof(float)));

        workSpaceOffset += coreNum_ * paraNum_ * cvOffset_ * sizeof(float);
        attnWsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(initParams.ws + workSpaceOffset +
                                                                   coreIdx_ * paraNum_ * ccOffset_ * sizeof(float)));

        workSpaceOffset += coreNum_ * paraNum_ * ccOffset_ * sizeof(float);
        uint64_t queryContinousOffset = workSpaceOffset + coreIdx_ * paraNum_ * ckOffset_ * sizeof(float);
        queryContinousGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(initParams.ws + queryContinousOffset));

        workSpaceOffset += coreNum_ * paraNum_ * ckOffset_ * sizeof(float);
        uint64_t keyContinousOffset = workSpaceOffset + coreIdx_ * paraNum_ * ckOffset_ * sizeof(float);
        keyContinousGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(initParams.ws + keyContinousOffset));
    }

    __aicore__ inline void InitLocalBuffers()
    {
        maxLen_ = AscendC::Std::max(AscendC::Std::max(dvAligned_ / 2, dkAligned_ / 2), chunkSize_);
        pipe_->InitBuffer(fp32InQueue_, BUFFER_NUM_ONE, chunkSize_ * maxLen_ * sizeof(float));
        pipe_->InitBuffer(fp32OutQueue_, BUFFER_NUM_ONE, chunkSize_ * maxLen_ * sizeof(float));
        if (gOptional_) {
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

        gBroadUbFloat_ = tmpBuff_.GetWithOffset<float>(
            static_cast<uint32_t>(chunkSize_ * maxLen_ * paraNum_), buffOffset);
        gammaUbFloat_ = gBroadUbFloat_;
        valueUbFloat_ = gBroadUbFloat_;
        qUbFloat_ = gBroadUbFloat_;
        buffOffset += chunkSize_ * maxLen_ * sizeof(float) * paraNum_;
        
        gTransBroadUbFloat_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(chunkSize_ * maxLen_), buffOffset);
        attnUbFloat_ = gTransBroadUbFloat_;
        gCumExpBroadUbFloat_ = gTransBroadUbFloat_;
        buffOffset += chunkSize_ * maxLen_ * sizeof(float);

        qUbFloatCon_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(halfChunkSize_ * dkAligned_), buffOffset);
        buffOffset += halfChunkSize_ * dkAligned_ * sizeof(float);

        kUbFloatCon_ = tmpBuff_.GetWithOffset<float>(
            static_cast<uint32_t>(halfChunkSize_ * dkAligned_), buffOffset);
        buffOffset += halfChunkSize_ * dkAligned_ * sizeof(float);

        inverseUbFloat_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(halfChunkSize_ * halfChunkSize_ *
                                                        INVERSE_COUNT), buffOffset);
        buffOffset += halfChunkSize_ * halfChunkSize_ * INVERSE_COUNT * sizeof(float);

        colBuffer_ = tmpBuff_.GetWithOffset<uint32_t>(static_cast<uint32_t>(INVERSE_SHAPE), buffOffset);
        buffOffset += INVERSE_SHAPE * sizeof(uint32_t);

        gatherOffsetFp32_ = tmpBuff_.GetWithOffset<uint32_t>(static_cast<uint32_t>(chunkSize_), buffOffset);
        buffOffset += chunkSize_ * sizeof(uint32_t);

        gatherOffsetBf16_ = tmpBuff_.GetWithOffset<uint32_t>(static_cast<uint32_t>(halfChunkSize_), buffOffset);
        buffOffset += halfChunkSize_ * sizeof(uint32_t);

        gCumExpUbFloat_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(chunkSize_ * paraNum_), buffOffset);
        buffOffset += chunkSize_ * sizeof(float) * paraNum_;
    }

    __aicore__ inline void InitGatherBuffer()
    {
        for (uint32_t i = 0; i < chunkSize_; ++i) {
            gatherOffsetFp32_.SetValue(i, i * BLOCK_SIZE);
        }
        for (uint32_t i = 0; i < halfChunkSize_; ++i) {
            gatherOffsetBf16_.SetValue(i, i * BLOCK_SIZE);
        }
        for (uint32_t i = 0; i < INVERSE_SHAPE; ++i) {
            colBuffer_.SetValue<uint32_t>(i, (i * chunkSize_) * sizeof(float));
        }
        int32_t eventID = static_cast<int32_t>(pipe_->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventID);
        WaitFlag<HardEvent::S_V>(eventID);
    }

    __aicore__ inline void Init(const GDRStageOneInitParams &initParams, TPipe *pipe,
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
        gOptional_ = initParams.gOptional;
        vRowStride_ = nv_ * dv_;
        numChunk_ = (cg_.length + chunkSize_ - 1) / chunkSize_;
        subBlockIdx_ = GetSubBlockIdx();
        halfChunkSize_ = chunkSize_ / TASK_RATIO;
        subOffset_ = subBlockIdx_ * halfChunkSize_;
        coreIdx_ = GetBlockIdx();
        ccOffset_ = chunkSize_ * chunkSize_;
        ckOffset_ = chunkSize_ * dk_;
        cvOffset_ = chunkSize_ * dv_;
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
        uint32_t tailChunkNum = totalChunk / coreNum_;   // tail核处理的块数
        uint32_t formerChunkNum = tailChunkNum + 1;      // former核处理的块数
        uint32_t formerCoreNum = totalChunk % coreNum_;  // former核数量
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
        AscendC::CrossCoreWaitFlag(0x9); // 同步0
        // key @ key.transpose(-1,-2)
        for (uint32_t i = 0; i < curParaNum; ++i) {
            AICProcess(keyContinousGm_[i * ckOffset_], keyContinousGm_[i * ckOffset_], kkWsGm_[i * ccOffset_],
                       {chunkSize_, chunkSize_, dk_, chunkSize_, chunkSize_, dk_}, true);
        }
        AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(0x8); // 同步1

        // query @ key.transpose(-1,-2)   stage1 out
        for (uint32_t i = 0; i < curParaNum; ++i) {
            AICProcess(queryContinousGm_[i * ckOffset_], keyContinousGm_[i * ckOffset_], 
                       outQkGm_[chunkRowBase_[i] * chunkSize_],
                       {validLenBatch_[i], validLenBatch_[i], dk_, validLenBatch_[i], validLenBatch_[i], dk_},
                       true);
        }
        AscendC::CrossCoreWaitFlag(0x7); // 同步2

        // 求逆左下角矩阵
        for (uint32_t i = 0; i < curParaNum; ++i) {
            AttnInverseMMCompute(i * ccOffset_);
        }
        AscendC::CrossCoreWaitFlag(0x6); // 同步3

        // attn @ k_cumdecay
        for (uint32_t i = 0; i < curParaNum; ++i) {
            AICProcess(attnWsGm_[i * ccOffset_], gBKWsGm_[i * ckOffset_], outKCumdecayGm_[chunkRowBase_[i] * dk_],
                       {chunkSize_, dk_, chunkSize_, chunkSize_, dk_, chunkSize_});
        }
        AscendC::CrossCoreWaitFlag(0x5); // 同步4

        // attn @ v_beta    stage1 out
        for (uint32_t i = 0; i < curParaNum; ++i) {
            AICProcess(attnWsGm_[i * ccOffset_], vBetaWsGm_[i * cvOffset_], outVInnerGm_[chunkRowBase_[i] * dv_],
                       {chunkSize_, dv_, chunkSize_, chunkSize_, dv_, chunkSize_});
        }
    }

    __aicore__ inline void ParaChunkAIV(int32_t curParaNum)
    {
        // 获取连续QK
        for (uint32_t i = 0; i < curParaNum; ++i) {
            uint64_t subRow = chunkStartRowBatch_[i] + subOffset_;
            uint64_t qk_base = subRow * nk_ * dk_ + nIdBatch_[i] * nk_ / nv_ * dk_;
            uint64_t wsOffset_ = i * ckOffset_ + subOffset_ * dk_;
            outKgGm_ = outKgBaseGm_[chunkRowBase_[i] * dk_];
            QKPreProcess(queryGm_[qk_base], queryContinousGm_[wsOffset_], outKgGm_, subValidLenBatch_[i]);
            QKPreProcess(keyGm_[qk_base], keyContinousGm_[wsOffset_], outKgGm_, subValidLenBatch_[i], true);
        }
        AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(0x9); // 同步0
        if (gOptional_) {
            for (uint32_t i = 0; i < curParaNum; ++i) {
                // g_cum_exp = g.cumsum(dim=-1).exp()
                GCumExpCompute(gGm_[bgOffsetBatch_[i]], outGCumExpGm_[chunkRowBase_[i]],
                               gCumExpUbFloat_[i * chunkSize_], validLenBatch_[i]);
                // attn_1 = (g_cum_exp[:None] / g_cum_exp[None,:]) * mask
                uint64_t gUbOffset = i * chunkSize_ * maxLen_;
                GammaCompute(gBroadUbFloat_[gUbOffset], gammaUbFloat_[gUbOffset], gCumExpUbFloat_[i * chunkSize_]);
            }
        }
        
        for (uint32_t i = 0; i < curParaNum; ++i) {
            uint64_t betaUbOffset = i * halfChunkSize_;
            BetaCopyInWithStride(betaGm_[bgOffsetBatch_[i]], betaUbFloat_[betaUbOffset], subValidLenBatch_[i]);
        }
        AscendC::CrossCoreWaitFlag(0x8); // 同步1

        for (uint32_t i = 0; i < curParaNum; ++i) {
            uint64_t betaUbOffset = i * halfChunkSize_;
            // attn_1 = kkt * attn_1
            KKBetaCompute(kkWsGm_[i * ccOffset_], betaUbFloat_[betaUbOffset]);
            // attn_1对角块求逆，对角块shape为INVERSE_SHAPE=32
            uint64_t gammaUbOffset = i * chunkSize_ * maxLen_;
            InverseCompute(attnWsGm_[i * ccOffset_], gammaUbFloat_[gammaUbOffset]);
        }
        AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(0x7); // 同步2

        for (uint32_t i = 0; i < curParaNum; ++i) {
            // kg = key * (g_cum_exp[-1, None] / g_cum_exp)[..., None]
            // k_cumdecay = -1.0 * k * beta * g_cum_exp
            outKgGm_ = outKgBaseGm_[chunkRowBase_[i] * dk_];
            uint64_t betaUbOffset = i * halfChunkSize_;
            uint64_t kUbOffset = i * halfChunkSize_ * dkAligned_;
            uint64_t wsOffset_ = i * ckOffset_ + subOffset_ * dk_;
            GBKCompute(gBKWsGm_[i * ckOffset_], outKgGm_, betaUbFloat_[betaUbOffset],
                       kUbFloatCon_[kUbOffset], gCumExpUbFloat_[i * chunkSize_], keyContinousGm_[wsOffset_]);
        }
        AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(0x6); // 同步3

        for (uint32_t i = 0; i < curParaNum; ++i) {
            // v_beta = value * beta.unsqueeze(-1)  # (C, Dv)
            uint64_t betaUbOffset = i * halfChunkSize_;
            uint64_t vOffset = chunkStartRowBatch_[i] * vRowStride_ + nIdBatch_[i] * dv_;
            uint64_t valueUbOffset = i * chunkSize_ * maxLen_;
            VBetaCompute(valueGm_[vOffset], vBetaWsGm_[i * cvOffset_], betaUbFloat_[betaUbOffset],
                         valueUbFloat_[valueUbOffset], subValidLenBatch_[i]);
        }
        AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(0x5); // 同步4

        for (uint32_t i = 0; i < curParaNum; ++i) {
            // q_prime = query * scale_ * g_cum_exp[:, None]  # (C, Dk)
            uint64_t wsOffset_ = i * ckOffset_ + subOffset_ * dk_;
            uint64_t qUbOffset = i * halfChunkSize_ * dkAligned_;
            QPrimeCompute(outQPrimeGm_[chunkRowBase_[i] * dk_], qUbFloatCon_[qUbOffset],
                          gCumExpUbFloat_[i * chunkSize_], queryContinousGm_[wsOffset_]);
        }
    }
    __aicore__ inline void QKPreProcess(const GlobalTensor<bfloat16_t>& srcGm, const GlobalTensor<float>& dstGm,
                                        const GlobalTensor<float>& outKgGm, uint32_t subValidRows, bool kgFlag = false)
    {
        // copyIn
        DataCopyInBf16WithStride(subValidRows, dk_, srcGm, nk_ * dk_);
        // copyOut
        auto tmpTensor = fp32OutQueue_.AllocTensor<float>();
        // compute
        LocalTensor<bfloat16_t> bf16Tensor = fp32InQueue_.DeQue<bfloat16_t>();
        Cast(tmpTensor, bf16Tensor, AscendC::RoundMode::CAST_NONE, subValidRows * dkAligned_);
        PipeBarrier<PIPE_V>();
        if (subValidRows < halfChunkSize_) {
            Duplicate(tmpTensor[subValidRows * dkAligned_], static_cast<float>(0.0f),
                      (halfChunkSize_ - subValidRows) * dkAligned_);
            PipeBarrier<PIPE_V>();
        }
        fp32OutQueue_.EnQue(tmpTensor);
        fp32InQueue_.FreeTensor(bf16Tensor);
        tmpTensor = fp32OutQueue_.DeQue<float>();

        uint32_t srcStride = (dkAligned_ - dk_) * sizeof(float) / BLOCK_SIZE;
        DataCopyExtParams outParams{static_cast<uint16_t>(halfChunkSize_),
                                    static_cast<uint32_t>(dk_ * sizeof(float)), srcStride, 0, 0};
        DataCopyPad(dstGm, tmpTensor, outParams);
        if (!gOptional_ && kgFlag) {
            DataCopyPad(outKgGm[subOffset_ * dk_], tmpTensor, outParams);
        }
        fp32OutQueue_.FreeTensor(tmpTensor);
    }

    __aicore__ inline void GCumExpCompute(const GlobalTensor<float> src, const GlobalTensor<float> dst,
                                          LocalTensor<float> gCumExpUbFloat, uint32_t validLen)
    {
        // Copy g
        GCopyInWithStride(src, validLen);
        // CumSum计算
        uint32_t outer = 1;
        uint32_t inner = chunkSize_;
        CumSumInfo cumSumInfo{outer, inner};
        CumSum<float>(gCumUbFloat_, gCumUbFloat_, gCumUbFloat_, cumSumInfo);
        PipeBarrier<PIPE_V>();
        // Exp计算
        Exp<float, 0, true>(gCumExpUbFloat, gCumUbFloat_, chunkSize_);
        PipeBarrier<PIPE_V>();
        if (subBlockIdx_ == 0) {
            auto tmpOut = gOutQueue_.AllocTensor<float>();
            DataCopy(tmpOut, gCumExpUbFloat, chunkSize_);
            gOutQueue_.EnQue<float>(tmpOut);
            tmpOut = gOutQueue_.DeQue<float>();
            DataCopyExtParams params{static_cast<uint16_t>(1),
                                    static_cast<uint32_t>(validLen * sizeof(float)), 0, 0, 0};
            DataCopyPad(dst, tmpOut, params);
            gOutQueue_.FreeTensor(tmpOut);
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void GammaCompute(const LocalTensor<float> gBroadUbFloat,
                                        LocalTensor<float> gammaUbFloat, LocalTensor<float> gCumExpUbFloat)
    {
        // BroadCast
        uint32_t divShape[2] = {chunkSize_, chunkSize_};
        uint32_t gShape[2] = {chunkSize_, 1};
        uint32_t gTransShape[2] = {1, chunkSize_};
        Broadcast<float, BROADCAST_AXIS, 1>(gBroadUbFloat, gCumExpUbFloat, divShape, gShape);
        Broadcast<float, BROADCAST_AXIS, 0>(gTransBroadUbFloat_, gCumExpUbFloat, divShape, gTransShape);
        PipeBarrier<PIPE_V>();
        // div
        Div(gammaUbFloat, gBroadUbFloat, gTransBroadUbFloat_, ccOffset_);
        PipeBarrier<PIPE_V>();
        // mask
        DataCopyInFp32(ccOffset_, stageOneMask_[GetBlockIdx() * ccOffset_]);
        kkLocal_ = fp32InQueue_.DeQue<float>();
        Mul(gammaUbFloat, gammaUbFloat, kkLocal_, ccOffset_);
        fp32InQueue_.FreeTensor(kkLocal_);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void KKBetaCompute(const GlobalTensor<float> src, LocalTensor<float> betaUbFloat)
    {
        // copy value
        uint32_t kkLength = chunkSize_ * halfChunkSize_;
        uint64_t kkBeginOffset = subOffset_ * chunkSize_;
        DataCopyInFp32(kkLength, src[kkBeginOffset]);
        kkLocal_ = fp32InQueue_.DeQue<float>();

        uint32_t betaShape[2] = {halfChunkSize_, 1};
        uint32_t kkShape[2] = {halfChunkSize_, chunkSize_};
        Broadcast<float, BROADCAST_AXIS, 1>(attnUbFloat_, betaUbFloat, kkShape, betaShape);
        PipeBarrier<PIPE_V>();
        Mul(attnUbFloat_, kkLocal_, attnUbFloat_, chunkSize_ * halfChunkSize_);
        PipeBarrier<PIPE_V>();
        fp32InQueue_.FreeTensor(kkLocal_);
    }

    __aicore__ inline void InverseCompute(const GlobalTensor<float> src, LocalTensor<float> gammaUbFloat)
    {
        uint64_t curVecLen = chunkSize_ * halfChunkSize_;
        if (gOptional_) {
            Mul(attnUbFloat_, attnUbFloat_, gammaUbFloat[subOffset_ * chunkSize_], curVecLen);
        } else {
            DataCopyInFp32(curVecLen, stageOneMask_[GetBlockIdx() * ccOffset_ + subOffset_ * chunkSize_]);
            kkLocal_ = fp32InQueue_.DeQue<float>();
            Mul(attnUbFloat_, attnUbFloat_, kkLocal_, curVecLen);
            fp32InQueue_.FreeTensor(kkLocal_);
        }
        PipeBarrier<PIPE_V>();

        inverseLocal_ = fp32OutQueue_.AllocTensor<float>();
        Muls(inverseLocal_, attnUbFloat_, static_cast<float>(-1.0), curVecLen);
        PipeBarrier<PIPE_V>();

        InverseAIV(subOffset_, INVERSE_SHAPE);
        fp32OutQueue_.EnQue(inverseLocal_);
        DataCopyOutFp32(halfChunkSize_, chunkSize_, chunkSize_, src[subOffset_ * chunkSize_]);
    }

    __aicore__ inline void InverseAIV(uint64_t offset, uint32_t inverseVecLen)
    {
        PipeBarrier<PIPE_V>();
        uint64_t inverseBufferOffset = 0;
        auto row = inverseUbFloat_[inverseBufferOffset];
        inverseBufferOffset += inverseVecLen * inverseVecLen;
        auto col = inverseUbFloat_[inverseBufferOffset];
        inverseBufferOffset += inverseVecLen * inverseVecLen + inverseVecLen;
        auto yLocal = inverseUbFloat_[inverseBufferOffset];
        inverseBufferOffset += inverseVecLen * inverseVecLen;
        auto ei = inverseUbFloat_[inverseBufferOffset];

        Duplicate(ei, static_cast<float>(0.0), inverseVecLen);
        Duplicate(yLocal, static_cast<float>(0.0), inverseVecLen * inverseVecLen); // yLocal清零
        inverseLocal_.SetValue(offset, static_cast<float>(1.0));
        
        uint32_t srcShape[2] = {1, inverseVecLen};
        int32_t eventID = static_cast<int32_t>(pipe_->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventID);
        WaitFlag<HardEvent::S_V>(eventID);
        for (int i = 1; i < inverseVecLen; ++i) {
            uint32_t curI = i - 1;
            uint32_t validRows = inverseVecLen - i;
            Gather(col, attnUbFloat_[offset + i * chunkSize_ + curI], colBuffer_, (uint32_t)0, validRows);
            PipeBarrier<PIPE_V>();
            uint32_t dstShape[2] = {validRows, inverseVecLen};
            uint32_t colSrcShape[2] = {validRows, 1};
            Broadcast<float, BROADCAST_AXIS, 1>(col[inverseVecLen], col, dstShape, colSrcShape);
            Broadcast<float, BROADCAST_AXIS, 0>(row, inverseLocal_[offset + curI * chunkSize_], dstShape, srcShape);
            PipeBarrier<PIPE_V>();
            MulAddDst(yLocal[i * inverseVecLen], col[inverseVecLen], row, inverseVecLen * validRows);
            PipeBarrier<PIPE_V>();
            ei.SetValue(i - 1, static_cast<float>(0.0));
            ei.SetValue(i, static_cast<float>(1.0));
            eventID = static_cast<int32_t>(pipe_->FetchEventID(HardEvent::S_V));
            SetFlag<HardEvent::S_V>(eventID);
            WaitFlag<HardEvent::S_V>(eventID);
            // xi = (I - SUM) / Lii = I - SUM
            Sub(inverseLocal_[offset + i * chunkSize_], ei, yLocal[i * inverseVecLen], inverseVecLen);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void GBKCompute(const GlobalTensor<float> gBKWsGm, const GlobalTensor<float> outKgGm,
                                      LocalTensor<float> betaUbFloat, LocalTensor<float> kUbFloatCon,
                                      LocalTensor<float> gCumExpUbFloat, const GlobalTensor<float> keyContinousGm)
    {
        if (gOptional_) {
            // tmp = -1.0 * beta * g_cum_exp
            Mul(gBUbFloat_, betaUbFloat, gCumExpUbFloat[subOffset_], halfChunkSize_);
            PipeBarrier<PIPE_V>();
            Muls(gBUbFloat_, gBUbFloat_, static_cast<float>(-1), halfChunkSize_);
        } else {
            Muls(gBUbFloat_, betaUbFloat, static_cast<float>(-1), halfChunkSize_);
        }
        PipeBarrier<PIPE_V>();
        // k_cumdecay = k * tmp =  -1.0 * k * beta * g_cum_exp
        uint32_t betaShape[2] = {halfChunkSize_, 1};
        uint32_t kShape[2] = {halfChunkSize_, dkAligned_};
        gBKLocal_ = fp32OutQueue_.AllocTensor<float>();
        Broadcast<float, BROADCAST_AXIS, 1>(gBKLocal_, gBUbFloat_, kShape, betaShape);
        PipeBarrier<PIPE_V>();
        // data copy in kUbFloatCon
        uint32_t dstStride = 0;
        if (dk_ % BLOCK_BF16_NUM != 0) {
            dstStride = (dk_ % BLOCK_BF16_NUM) > BLOCK_FLOAT_NUM ? 0 : 1;
        }
        DataCopyInFp32WithStride(halfChunkSize_, dk_, keyContinousGm, dk_, dstStride); // GM上相邻行的间距(元素数)
        kUbFloatCon = fp32InQueue_.DeQue<float>();
        // compute
        Mul(gBKLocal_, gBKLocal_, kUbFloatCon, halfChunkSize_ * dkAligned_);
        fp32OutQueue_.EnQue<float>(gBKLocal_);
        uint64_t gBKBeginOffset = subOffset_ * dk_;
        DataCopyOutFp32(halfChunkSize_, dk_, dkAligned_, gBKWsGm[gBKBeginOffset]);
        PipeBarrier<PIPE_V>();
        if (gOptional_) {
            // kg = k * (g_cum_exp[-1, None] / g_cum_exp)[..., None]
            uint32_t gEndShape[2] = {1, 1};
            uint32_t gBroadShape[2] = {halfChunkSize_, 1};
            Broadcast<float, BROADCAST_AXIS, 0>(gEndBroadUbFloat_, gCumExpUbFloat[chunkSize_ - 1],
                                                gBroadShape, gEndShape);
            PipeBarrier<PIPE_V>();
            Div(gEndBroadUbFloat_, gEndBroadUbFloat_, gCumExpUbFloat[subOffset_], halfChunkSize_);
            PipeBarrier<PIPE_V>();
            kgLocal_ = fp32OutQueue_.AllocTensor<float>();
            Broadcast<float, BROADCAST_AXIS, 1>(kgLocal_, gEndBroadUbFloat_, kShape, gBroadShape);
            PipeBarrier<PIPE_V>();
            Mul(kgLocal_, kgLocal_, kUbFloatCon, halfChunkSize_ * dkAligned_);
            PipeBarrier<PIPE_V>();
            fp32OutQueue_.EnQue<float>(kgLocal_);
            uint64_t kgBeginOffset = subOffset_ * dk_;
            DataCopyOutFp32(halfChunkSize_, dk_, dkAligned_, outKgGm[kgBeginOffset]);  // stage1 out
        }
        fp32InQueue_.FreeTensor(kUbFloatCon);
    }

    __aicore__ inline void VBetaCompute(const GlobalTensor<bfloat16_t> valueGm, const GlobalTensor<float> vBetaWsGm,
                                        LocalTensor<float> betaUbFloat, LocalTensor<float> valueUbFloat,
                                        uint32_t subValidRows)
    {
        uint64_t vBeginOffset = subOffset_ * vRowStride_;
        DataCopyInBf16WithStride(subValidRows, dv_, valueGm[vBeginOffset], vRowStride_);
        valueLocal_ = fp32InQueue_.DeQue<bfloat16_t>();
        vBetaLocal_ = fp32OutQueue_.AllocTensor<float>();
        Cast(valueUbFloat, valueLocal_, AscendC::RoundMode::CAST_NONE, subValidRows * dvAligned_);
        PipeBarrier<PIPE_V>();
        fp32InQueue_.FreeTensor(valueLocal_);
        if (subValidRows < halfChunkSize_) {
            Duplicate(valueUbFloat[subValidRows * dvAligned_], static_cast<float>(0.0f),
                      (halfChunkSize_ - subValidRows) * dvAligned_);
            PipeBarrier<PIPE_V>();
        }
        uint32_t betaShape[2] = {halfChunkSize_, 1};
        uint32_t vShape[2] = {halfChunkSize_, dvAligned_};
        Broadcast<float, BROADCAST_AXIS, 1>(vBetaLocal_, betaUbFloat, vShape, betaShape);
        PipeBarrier<PIPE_V>();
        Mul(vBetaLocal_, valueUbFloat, vBetaLocal_, halfChunkSize_ * dvAligned_);
        PipeBarrier<PIPE_V>();
        fp32OutQueue_.EnQue<float>(vBetaLocal_);
        DataCopyOutFp32(halfChunkSize_, dv_, dvAligned_, vBetaWsGm[subOffset_ * dv_]);
    }

    __aicore__ inline void QPrimeCompute(const GlobalTensor<float> outQPrimeGm, LocalTensor<float> qUbFloatCon,
                                         LocalTensor<float> gCumExpUbFloat, const GlobalTensor<float> queryContinousGm)
    {
        qPrimeLocal_ = fp32OutQueue_.AllocTensor<float>();
        // data copy in qUbFloatCon
        uint32_t dstStride = 0;
        if (dk_ % BLOCK_BF16_NUM != 0) {
            dstStride = (dk_ % BLOCK_BF16_NUM) > BLOCK_FLOAT_NUM ? 0 : 1;
        }
        DataCopyInFp32WithStride(halfChunkSize_, dk_, queryContinousGm, dk_, dstStride); // GM上相邻行的间距(元素数)
        qUbFloatCon = fp32InQueue_.DeQue<float>();
        // query * scale
        if (gOptional_) {
            Muls(qUbFloat_, qUbFloatCon, scale_, halfChunkSize_ * dkAligned_);
            PipeBarrier<PIPE_V>();
            uint32_t gCumExpShape[2] = {halfChunkSize_, 1};
            uint32_t qShape[2] = {halfChunkSize_, dkAligned_};
            Broadcast<float, BROADCAST_AXIS, 1>(gCumExpBroadUbFloat_, gCumExpUbFloat[subOffset_],
                                                qShape, gCumExpShape);
            PipeBarrier<PIPE_V>();
            // query * scale * g_cum_exp[:, None]       # (C, Dk)
            Mul(qPrimeLocal_, qUbFloat_, gCumExpBroadUbFloat_, halfChunkSize_ * dkAligned_);
        } else {
            Muls(qPrimeLocal_, qUbFloatCon, scale_, halfChunkSize_ * dkAligned_);
        }
        PipeBarrier<PIPE_V>();
        fp32OutQueue_.EnQue<float>(qPrimeLocal_);
        fp32InQueue_.FreeTensor(qUbFloatCon);
        uint64_t qgBeginOffset = subOffset_ * dk_;
        DataCopyOutFp32(halfChunkSize_, dk_, dkAligned_, outQPrimeGm[qgBeginOffset]);  // stage1 out
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void DataCopyInFp32(uint64_t len, GlobalTensor<float> y)
    {
        DataCopyPadExtParams<float> padParams;
        DataCopyExtParams kkParams{static_cast<uint16_t>(1), static_cast<uint32_t>(len * sizeof(float)), 0, 0, 0};
        fp32InLocal_ = fp32InQueue_.AllocTensor<float>();
        DataCopyPad(fp32InLocal_, y, kkParams, padParams);
        fp32InQueue_.EnQue<float>(fp32InLocal_);
    }

    __aicore__ inline void BetaCopyInWithStride(const GlobalTensor<bfloat16_t> src, LocalTensor<float> betaUbFloat,
                                                uint32_t subValidRows)
    {
        uint64_t betaBeginOffset = subOffset_ * nv_;
        DataCopyInBf16WithStride(subValidRows, 1, src[betaBeginOffset], nv_);
        betaLocal_ = fp32InQueue_.DeQue<bfloat16_t>();
        if (subValidRows < halfChunkSize_) {
            Duplicate(betaUbBfloat16_, bfloat16_t(0.0f), halfChunkSize_);
            PipeBarrier<PIPE_V>();
        }
        Gather(betaUbBfloat16_, betaLocal_, gatherOffsetBf16_, static_cast<uint32_t>(0), subValidRows);
        PipeBarrier<PIPE_V>();

        Cast(betaUbFloat, betaUbBfloat16_, AscendC::RoundMode::CAST_NONE, halfChunkSize_);
        PipeBarrier<PIPE_V>();
        fp32InQueue_.FreeTensor(betaLocal_);
    }

    __aicore__ inline void GCopyInWithStride(const GlobalTensor<float> src, uint32_t validLen)
    {
        DataCopyInFp32WithStride(validLen, 1, src, nv_);
        gLocal_ = fp32InQueue_.DeQue<float>();
        if (validLen < chunkSize_) {
            Duplicate(gCumUbFloat_, 0.0f, chunkSize_);
            PipeBarrier<PIPE_V>();
        }
        Gather(gCumUbFloat_, gLocal_, gatherOffsetFp32_, static_cast<uint32_t>(0), validLen);
        PipeBarrier<PIPE_V>();

        fp32InQueue_.FreeTensor(gLocal_);
    }

    __aicore__ inline void DataCopyInFp32WithStride(uint64_t rows,  // 要搬的行数
                                                    uint64_t cols,  // 每行的元素数
                                                    const GlobalTensor<float> src,
                                                    uint64_t srcRowStride,  // GM上相邻行的间距(元素数)
                                                    uint32_t dstRowStride = 0)
    {
        DataCopyPadExtParams<float> padParams = {false, static_cast<uint8_t>(0), static_cast<uint8_t>(0),
                                                      static_cast<float>(0)};
        uint32_t srcGap = (srcRowStride - cols) * sizeof(float);
        DataCopyExtParams params{static_cast<uint16_t>(rows),
                                 static_cast<uint32_t>(cols * sizeof(float)),
                                 static_cast<uint32_t>(srcGap), static_cast<uint32_t>(dstRowStride), 0};
        fp32InLocal_ = fp32InQueue_.AllocTensor<float>();
        DataCopyPad(fp32InLocal_, src, params, padParams);
        fp32InQueue_.EnQue<float>(fp32InLocal_);
    }

    __aicore__ inline void DataCopyInBf16WithStride(uint64_t rows,  // 要搬的行数
                                                    uint64_t cols,  // 每行的元素数
                                                    GlobalTensor<bfloat16_t> src,
                                                    uint64_t srcRowStride) // GM上相邻行的间距(元素数)
    {
        DataCopyPadExtParams<bfloat16_t> padParams = {false, static_cast<uint8_t>(0), static_cast<uint8_t>(0),
                                                      static_cast<float>(0)};
        uint32_t srcGap = (srcRowStride - cols) * sizeof(bfloat16_t);
        DataCopyExtParams params{static_cast<uint16_t>(rows),
                                 static_cast<uint32_t>(cols * sizeof(bfloat16_t)),
                                 static_cast<uint32_t>(srcGap), 0, 0};
        bf16InLocal_ = fp32InQueue_.AllocTensor<bfloat16_t>();
        DataCopyPad(bf16InLocal_, src, params, padParams);
        fp32InQueue_.EnQue<bfloat16_t>(bf16InLocal_);
    }

    __aicore__ inline void DataCopyOutFp32(uint32_t rows, uint32_t cols,
                                                uint32_t colsAligned, GlobalTensor<float> y)
    {
        fp32OutLocal_ = fp32OutQueue_.DeQue<float>();
        uint32_t srcStride = (colsAligned - cols) * sizeof(float) / BLOCK_SIZE;
        DataCopyExtParams yGMParams{static_cast<uint16_t>(rows),
                                    static_cast<uint32_t>(cols * sizeof(float)),
                                    static_cast<uint32_t>(srcStride), 0, 0};
        DataCopyPad(y, fp32OutLocal_, yGMParams);
        fp32OutQueue_.FreeTensor(fp32OutLocal_);
    }

    __aicore__ inline void AttnInverseMMCompute(uint64_t offset)
    {
        uint64_t leftDown = offset + chunkSize_ * INVERSE_SHAPE;
        uint64_t rightDown = leftDown + INVERSE_SHAPE;
        // 右矩阵左下角 @ 右矩阵左上角 -> 右矩阵左下角
        InverseAICProcess(attnWsGm_[leftDown], attnWsGm_[offset], attnWsGm_[leftDown]);
        int32_t eventID = static_cast<int32_t>(pipe_->FetchEventID(HardEvent::FIX_MTE2));
        SetFlag<HardEvent::FIX_MTE2>(eventID);
        WaitFlag<HardEvent::FIX_MTE2>(eventID);
        // 右矩阵右下角 @ 右矩阵左下角 -> 右矩阵左下角
        InverseAICProcess(attnWsGm_[rightDown], attnWsGm_[leftDown], attnWsGm_[leftDown]);
        eventID = static_cast<int32_t>(pipe_->FetchEventID(HardEvent::FIX_MTE2));
        SetFlag<HardEvent::FIX_MTE2>(eventID);
        WaitFlag<HardEvent::FIX_MTE2>(eventID);
    }

    __aicore__ inline void AICProcess(GlobalTensor<float> x, GlobalTensor<float> y, GlobalTensor<float> z,
                                      const MatmulShapeParams &shape, bool transB = false)
    {
        mmFp32.SetOrgShape(shape.m, shape.n, shape.k);
        mmFp32.SetSingleShape(shape.sm, shape.sn, shape.sk);
        mmFp32.SetTensorA(x);
        mmFp32.SetTensorB(y, transB);
        mmFp32.IterateAll(z);
        mmFp32.End();
    }

    __aicore__ inline void InverseAICProcess(GlobalTensor<float> x, GlobalTensor<float> y, GlobalTensor<float> z)
    {
        mmFp32.SetOrgShape(chunkSize_, chunkSize_, chunkSize_);
        mmFp32.SetSingleShape(INVERSE_SHAPE, INVERSE_SHAPE, INVERSE_SHAPE);
        mmFp32.SetTensorA(x);
        mmFp32.SetTensorB(y);
        mmFp32.IterateAll(z);
        mmFp32.End();
    }
    
    TPipe *pipe_;
    StageOneMT &mmFp32;
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
    bool gOptional_;
    uint32_t paraNum_;
    uint64_t ccOffset_;
    uint64_t ckOffset_;
    uint64_t cvOffset_;
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
    GlobalTensor<float> outKCumdecayGm_;
    GlobalTensor<float> outVInnerGm_;
    GlobalTensor<float> outQPrimeGm_;
    GlobalTensor<float> outKgBaseGm_;
    GlobalTensor<float> outKgGm_;
    GlobalTensor<float> outQkGm_;
    GlobalTensor<float> vBetaWsGm_;
    GlobalTensor<float> kkWsGm_;
    GlobalTensor<float> attnWsGm_;
    GlobalTensor<float> gBKWsGm_;
    GlobalTensor<float> queryContinousGm_;
    GlobalTensor<float> keyContinousGm_;
    GlobalTensor<float> stageOneMask_;

    // UB queues
    TQue<QuePosition::VECIN, 1> fp32InQueue_;
    TQue<QuePosition::VECOUT, 1> fp32OutQueue_;
    TQue<QuePosition::VECOUT, 1> gOutQueue_;

    TBuf<TPosition::VECCALC> tmpBuff_;

    // UB tensors
    LocalTensor<bfloat16_t> betaUbBfloat16_;
    LocalTensor<float> betaUbFloat_;
    LocalTensor<float> valueUbFloat_;
    LocalTensor<float> attnUbFloat_;
    LocalTensor<float> inverseUbFloat_;
    LocalTensor<float> inverseLocal_;
    LocalTensor<float> gCumUbFloat_;
    LocalTensor<float> gCumExpUbFloat_;
    LocalTensor<float> gCumExpBroadUbFloat_;
    LocalTensor<float> gBUbFloat_;
    LocalTensor<float> qUbFloat_;
    LocalTensor<float> gBroadUbFloat_;
    LocalTensor<float> gTransBroadUbFloat_;
    LocalTensor<float> gEndBroadUbFloat_;
    LocalTensor<float> gammaUbFloat_;
    LocalTensor<uint32_t> colBuffer_;
    LocalTensor<float> qUbFloatCon_;
    LocalTensor<float> kUbFloatCon_;
    LocalTensor<uint32_t> gatherOffsetFp32_;
    LocalTensor<uint32_t> gatherOffsetBf16_;

    LocalTensor<bfloat16_t> betaLocal_;
    LocalTensor<bfloat16_t> valueLocal_;
    LocalTensor<float> qPrimeLocal_;
    LocalTensor<float> vBetaLocal_;
    LocalTensor<float> kkLocal_;
    LocalTensor<float> gLocal_;
    LocalTensor<float> gBKLocal_;
    LocalTensor<float> kgLocal_;
    
    LocalTensor<bfloat16_t> bf16InLocal_;
    LocalTensor<float> fp32InLocal_;
    LocalTensor<float> fp32OutLocal_;
};
} // namespace ChunkGatedDeltaRule
#endif // CHUNK_GATED_DELTA_RULE_STAGE1_H