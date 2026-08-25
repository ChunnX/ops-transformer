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
 * \file chunk_gated_delta_rule_stage1_arch35.h
 * \brief
 */
#ifndef CHUNK_GATED_DELTA_RULE_STAGE1_ARCH35_H
#define CHUNK_GATED_DELTA_RULE_STAGE1_ARCH35_H

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
    // Stage1 与状态 dtype 无关：Q/K/V/beta 和各 Cube 中间量均为 BF16，门控累计量为 FP32。
    // 两种状态分支都输出 BF16 Z0，供 Stage2 的 C1/V2 和 Stage3 共同使用。
    using vInnerType = bfloat16_t;
    using MMBf16 = StageOneMTT<bfloat16_t>;
    using MMVInner = StageOneMTT<vInnerType>;
    __aicore__ inline Stage1(MMBf16 &mmBf16, MMVInner &mmVInner) : mmBf16_(mmBf16), mmVInner_(mmVInner)
    {
    }

    /**
     * 绑定 Stage1 的输入、输出和临时 workspace。
     * 输入：Q/K[T,Nk,Dk]、V[T,Nv,Dv]、beta[T,Nv]、可选 g[T,Nv]。
     * 输出：G、K_decay、Z0、Q'、K 和 QK^T；临时区按 CK/CC/CV 三类矩阵分区并按核隔离。
     */
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
        // ckOffset_
        uint64_t workSpaceOffset = 0;
        uint64_t curWS = coreIdx_ * paraNum_ * ckOffset_ * sizeof(bfloat16_t);
        gBKWsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(initParams.ws + workSpaceOffset + curWS));
        workSpaceOffset += coreNum_ * paraNum_ * ckOffset_ * sizeof(bfloat16_t);

        queryConGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(initParams.ws + workSpaceOffset + curWS));
        workSpaceOffset += coreNum_ * paraNum_ * ckOffset_ * sizeof(bfloat16_t);

        keyConGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(initParams.ws + workSpaceOffset + curWS));
        workSpaceOffset += coreNum_ * paraNum_ * ckOffset_ * sizeof(bfloat16_t);

        // ccOffset_
        curWS = coreIdx_ * paraNum_ * ccOffset_ * sizeof(bfloat16_t);
        kkWsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(initParams.ws + workSpaceOffset + curWS));
        workSpaceOffset += coreNum_ * paraNum_ * ccOffset_ * sizeof(bfloat16_t);

        attnWsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(initParams.ws + workSpaceOffset + curWS));
        workSpaceOffset += coreNum_ * paraNum_ * ccOffset_ * sizeof(bfloat16_t);

        // cvOffset_
        curWS = coreIdx_ * paraNum_ * cvOffset_ * sizeof(bfloat16_t);
        vBetaWsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(initParams.ws + workSpaceOffset + curWS));
    }

    /**
     * 初始化 AIV 的队列和 UB 复用区。
     * UB 同时容纳门控、beta、矩阵逆、Q'/K_decay/Z0 等临时 FP32 计算视图；同生命周期张量复用地址。
     */
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

    /**
     * 构造跨行搬运的 gather 索引。
     * inputGatherBuffer 用于从带 stride 的 beta/g 中抽取连续元素；inverseGatherBuffer 用于抽取三角矩阵列。
     */
    __aicore__ inline void InitGatherBuffer()
    {
        for (uint32_t i = 0; i < chunkSize_; ++i) {
            inputGatherBuffer_.SetValue(i, i * BLOCK_SIZE);
        }
        int32_t eventID = static_cast<int32_t>(pipe_->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventID);
        WaitFlag<HardEvent::S_V>(eventID);
    }

    /**
     * 解析 tiling 和当前 chunk group，计算 C*C、C*Dk、C*Dv 偏移并初始化 AIC/AIV 资源。
     * 输入 initParams 为本组 GM 张量，tilingData 给出 C、Dk、Dv、并行数和核数；无数值输出。
     */
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
        if ASCEND_IS_AIV {
            coreIdx_ /= TASK_RATIO;
            InitLocalBuffers();
            InitGatherBuffer();
        }
        SetGlobalTensors(initParams);
    }

    /**
     * 将任务 (value_head, chunk) 均匀分配到逻辑 AI Core，并按 stageOneParaNum 批量执行。
     * 每个任务最终产生当前 chunk 的 G、K_decay、Z0、Q'、K 和 QK^T。
     */
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
    /**
     * 计算一个任务的有效长度和 GM 偏移。
     * 输入：id 为批内并行槽，curNId 为 value head，curCgId 为组内 chunk；
     * 输出：validLen/subValidLen、head-major workspace 起点和 TND 输入起点。
     */
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

    /**
     * 调度一批 chunk：AIC 执行矩阵乘，AIV 执行搬运、门控和三角逆；二者通过 cross-core flag 流水同步。
     */
    __aicore__ inline void ProcessParaChunk(int32_t curParaNum)
    {
        if ASCEND_IS_AIC {
            ParaChunkAIC(curParaNum);
        }
        if ASCEND_IS_AIV {
            ParaChunkAIV(curParaNum);
        }
    }

    /**
     * Stage1 的 Cube 部分。
     * 依次计算 KK^T、QK^T、分块三角逆拼接、K_decay=A@(-beta*exp(G)*K)、Z0=A@(beta*V)。
     * 输入均为 BF16 GM；输出 QK^T、K_decay、Z0 为 BF16 GM。
     */
    __aicore__ inline void ParaChunkAIC(int32_t curParaNum)
    {
        // key @ key.transpose(-1,-2)
        for (uint32_t i = 0; i < curParaNum; ++i) {
            AscendC::CrossCoreWaitFlag(0x9); // 同步0
            AICProcess(mmBf16_, keyConGm_[i * ckOffset_], keyConGm_[i * ckOffset_], kkWsGm_[i * ccOffset_],
                       {chunkSize_, chunkSize_, dk_, chunkSize_, chunkSize_, dk_}, true);
            AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(0x8); // 同步1
        }

        // query @ key.transpose(-1,-2)   stage1 out
        for (uint32_t i = 0; i < curParaNum; ++i) {
            AICProcess(mmBf16_, queryConGm_[i * ckOffset_], keyConGm_[i * ckOffset_],
                       outQkGm_[chunkRowBase_[i] * chunkSize_],
                       {validLenBatch_[i], validLenBatch_[i], dk_, validLenBatch_[i], validLenBatch_[i], dk_}, true);
        }
        AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(0xA); // 同步5

        // 求逆左下角矩阵
        for (uint32_t i = 0; i < curParaNum; ++i) {
            AscendC::CrossCoreWaitFlag(0x7); // 同步2
            AttnInverseMMCompute(i * ccOffset_);
        }

        // attn @ k_cumdecay
        for (uint32_t i = 0; i < curParaNum; ++i) {
            AscendC::CrossCoreWaitFlag(0x6); // 同步3
            AICProcess(mmBf16_, attnWsGm_[i * ccOffset_], gBKWsGm_[i * ckOffset_],
                       outKCumdecayGm_[chunkRowBase_[i] * dk_],
                       {chunkSize_, dk_, chunkSize_, chunkSize_, dk_, chunkSize_});
        }

        // attn @ v_beta    stage1 out
        for (uint32_t i = 0; i < curParaNum; ++i) {
            AscendC::CrossCoreWaitFlag(0x5); // 同步4
            AICProcess(mmVInner_, attnWsGm_[i * ccOffset_], vBetaWsGm_[i * cvOffset_],
                       outVInnerGm_[chunkRowBase_[i] * dv_],
                       {chunkSize_, dv_, chunkSize_, chunkSize_, dv_, chunkSize_});
        }
    }

    /**
     * Stage1 的 Vector 部分。
     * 将 TND Q/K/V/beta/g 整理成连续 chunk，计算 G、门控三角矩阵、beta 加权项、Q' 和 K，并驱动 AIC 流水。
     * 公式：G_i=sum_{r<=i}g_r；Q'_i=scale*exp(G_i)*Q_i；无 g 时令 G_i=0。
     */
    __aicore__ inline void ParaChunkAIV(int32_t curParaNum)
    {
        // 获取连续QK
        for (uint32_t i = 0; i < curParaNum; ++i) {
            uint64_t subRow = chunkStartRowBatch_[i] + subOffset_;
            uint64_t qk_base = subRow * nk_ * dk_ + nIdBatch_[i] * nk_ / nv_ * dk_;
            uint64_t wsOffset_ = i * ckOffset_ + subOffset_ * dk_;
            outKgGm_ = outKgBaseGm_[chunkRowBase_[i] * dk_];
            QKPreProcess(queryGm_[qk_base], queryConGm_[wsOffset_], outKgGm_, subValidLenBatch_[i]);
            QKPreProcess(keyGm_[qk_base], keyConGm_[wsOffset_], outKgGm_, subValidLenBatch_[i], true);
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(0x9); // 同步0
        }
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

        for (uint32_t i = 0; i < curParaNum; ++i) {
            // kg = key * (g_cum_exp[-1, None] / g_cum_exp)[..., None]
            // k_cumdecay = -1.0 * k * beta * g_cum_exp
            outKgGm_ = outKgBaseGm_[chunkRowBase_[i] * dk_];
            uint64_t wsOffset_ = i * ckOffset_ + subOffset_ * dk_;
            GBKCompute(gBKWsGm_[i * ckOffset_], outKgGm_, betaUbFloat_[i * halfChunkSize_],
                       gCumSumUbFloat_[i * chunkSize_], gCumExpUbFloat_[i * chunkSize_], keyConGm_[wsOffset_]);
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(0x6); // 同步3
        }

        for (uint32_t i = 0; i < curParaNum; ++i) {
            // v_beta = value * beta.unsqueeze(-1)  # (C, Dv)
            uint64_t betaUbOffset = i * halfChunkSize_;
            uint64_t vOffset = chunkStartRowBatch_[i] * vRowStride_ + nIdBatch_[i] * dv_;
            uint64_t valueUbOffset = i * chunkSize_ * maxLen_;
            VBetaCompute(valueGm_[vOffset], vBetaWsGm_[i * cvOffset_], betaUbFloat_[betaUbOffset],
                         valueUbFloat_[valueUbOffset], subValidLenBatch_[i]);
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(0x5); // 同步4
        }

        for (uint32_t i = 0; i < curParaNum; ++i) {
            // q_prime = query * scale_ * g_cum_exp[:, None]  # (C, Dk)
            uint64_t wsOffset_ = i * ckOffset_ + subOffset_ * dk_;
            QPrimeCompute(outQPrimeGm_[chunkRowBase_[i] * dk_], gCumSumUbFloat_[i * chunkSize_],
                          gCumExpUbFloat_[i * chunkSize_], queryConGm_[wsOffset_]);
        }
        AscendC::CrossCoreWaitFlag(0xA); // 同步5
    }
    /**
     * 将当前 AIV 负责的半个 Q/K chunk 从带 head stride 的 TND GM 搬成连续 [C/2,Dk] BF16。
     * 尾行补零；无 g 分支处理 K 时还把原始 K 写入 kg，供 Stage2 C3 使用。
     */
    __aicore__ inline void QKPreProcess(const GlobalTensor<bfloat16_t> &srcGm, const GlobalTensor<bfloat16_t> &dstGm,
                                        const GlobalTensor<bfloat16_t> &outKgGm, uint32_t subValidRows,
                                        bool kgFlag = false)
    {
        // copyOut
        auto tmpTensor = outQueue_.AllocTensor<bfloat16_t>();
        if (subValidRows > 0) {
            // copyIn
            DataCopyInBf16WithStride(subValidRows, dk_, srcGm, nk_ * dk_);
            LocalTensor<bfloat16_t> bf16Tensor = inQueue_.DeQue<bfloat16_t>();
            DataCopy(tmpTensor, bf16Tensor, subValidRows * dkAligned_);
            inQueue_.FreeTensor(bf16Tensor);
        }

        if (subValidRows < halfChunkSize_) {
            Duplicate(tmpTensor[subValidRows * dkAligned_], static_cast<bfloat16_t>(0.0f),
                      (halfChunkSize_ - subValidRows) * dkAligned_);
            PipeBarrier<PIPE_V>();
        }
        outQueue_.EnQue(tmpTensor);
        tmpTensor = outQueue_.DeQue<bfloat16_t>();

        uint32_t srcStride = (dkAligned_ - dk_) * sizeof(bfloat16_t) / BLOCK_SIZE;
        DataCopyExtParams outParams{static_cast<uint16_t>(halfChunkSize_),
                                    static_cast<uint32_t>(dk_ * sizeof(bfloat16_t)), srcStride, 0, 0};
        DataCopyPad(dstGm, tmpTensor, outParams);
        if constexpr (!gOptional) {
            if (kgFlag) {
                DataCopyPad(outKgGm[subOffset_ * dk_], tmpTensor, outParams);
            }
        }
        outQueue_.FreeTensor(tmpTensor);
    }

    /**
     * 计算门控累计量。
     * 输入：src 为按 Nv 跨行存放的 g[validLen] FP32；输出：gCumSumUbFloat=G、gCumExpUbFloat=exp(G)，
     * 其中 G_i=sum_{r=0}^{i}g_r。AIV0 另将有效 G 写入 dst[Nv,Sp]，供 Stage2/3 使用。
     */
    __aicore__ inline void GCumExpCompute(const GlobalTensor<float> src, const GlobalTensor<float> dst,
                                          LocalTensor<float> gCumSumUbFloat, LocalTensor<float> gCumExpUbFloat,
                                          uint32_t validLen)
    {
        // Copy g
        GCopyInWithStride(src, validLen);
        // C 固定为 64，正好对应一个 FP32 vector register；在 RegBase 中一次完成 G 和 exp(G)。
        CumSumExpVf(reinterpret_cast<__ubuf__ float *>(gCumUbFloat_.GetPhyAddr()),
                    reinterpret_cast<__ubuf__ float *>(gCumSumUbFloat.GetPhyAddr()),
                    reinterpret_cast<__ubuf__ float *>(gCumExpUbFloat.GetPhyAddr()));
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

    /**
     * MicroAPI RegBase 实现 64 元素 inclusive scan 和指数。
     * 输入 gAddr[64] FP32（尾部已补零）；输出 gCumAddr[64]=cumsum(g)，
     * gCumExpAddr[64]=exp(cumsum(g))。前缀和使用 stride=1/2/4/8/16/32 的 Hillis-Steele 扫描。
     */
    static __simd_vf__ inline void CumSumExpVf(__ubuf__ float *gAddr, __ubuf__ float *gCumAddr,
                                               __ubuf__ float *gCumExpAddr)
    {
        using namespace AscendC::MicroAPI;

        RegTensor<float> gCumReg;
        RegTensor<float> shiftedReg;
        RegTensor<float> gCumExpReg;
        RegTensor<int32_t> laneIndexReg;
        RegTensor<int32_t> gatherIndexReg;
        MaskReg allMask = CreateMask<int32_t, MaskPattern::ALL>();
        MaskReg addMask;

        LoadAlign<float, LoadDist::DIST_NORM>(gCumReg, gAddr);
        Arange(laneIndexReg, static_cast<int32_t>(0));

        // 负 gather 索引钳制为 0，并用 addMask 保留每层不参与累加的前导 lane。
        for (int32_t stride = 1; stride < static_cast<int32_t>(64); stride <<= 1) {
            Adds<int32_t>(gatherIndexReg, laneIndexReg, -stride, allMask);
            Maxs<int32_t>(gatherIndexReg, gatherIndexReg, static_cast<int32_t>(0), allMask);
            Gather<float, uint32_t>(shiftedReg, gCumReg,
                                    reinterpret_cast<RegTensor<uint32_t> &>(gatherIndexReg));
            Compares<int32_t, CMPMODE::GE>(addMask, laneIndexReg, stride, allMask);
            Add<float, MaskMergeMode::MERGING>(gCumReg, gCumReg, shiftedReg, addMask);
        }

        static constexpr ExpSpecificMode expMode = {
            MaskMergeMode::ZEROING, AscendC::ExpAlgo::PRECISION_1ULP_FTZ_FALSE};
        Exp<float, &expMode>(gCumExpReg, gCumReg, allMask);

        StoreAlign<float, StoreDist::DIST_NORM>(gCumAddr, gCumReg, allMask);
        StoreAlign<float, StoreDist::DIST_NORM>(gCumExpAddr, gCumExpReg, allMask);
    }

    /**
     * 构造严格下三角门控矩阵 gamma。
     * 输入 G[C] FP32 和严格下三角 mask；输出 gamma[i,j]=exp(G_i-G_j)*1(j<i)，shape [C,C] FP32。
     */
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

    /**
     * 将 Cube 输出的 KK^T 半块搬入 UB 并施加行 beta。
     * 输入 src=KK^T[C,C] BF16、beta[C/2] FP32；输出 attnUbFloat[i,j]=beta_i*(K_i dot K_j)。
     */
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

    /**
     * 计算单位下三角矩阵的逆并写出 BF16。
     * 先令 L[i,j]=beta_i*(K_i dot K_j)*exp(G_i-G_j)*1(j<i)，再求 A=(I+L)^(-1)；
     * 输入 gamma 为门控 mask，输出 dst=A[C,C] BF16，两个 AIV 分别处理 32 行。
     */
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

    /**
     * 用前代递推求 32x32 单位下三角块的逆。
     * 对第 i 行计算 A_i=I_i-sum_{j<i}L_{i,j}A_j；输入位于 attnUbFloat，结果写入 inverseRes。
     */
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
            // xi = (I - SUM) / Lii = I - SUM
            Sub(inverseRes_[offset + i * chunkSize_], eiMatrix[i * inverseVecLen],
                yLocal[i * inverseVecLen], inverseVecLen);
            PipeBarrier<PIPE_V>();
        }
    }

    /**
     * 生成 K_decay 的 Cube 右矩阵并输出 C3 使用的 K。
     * 输入 K[C/2,Dk]、beta[C/2]、exp(G)[C]；输出
     * gBKWs=-beta_i*exp(G_i)*K_i（无 g 时 exp(G_i)=1），kg=K。二者均为 BF16。
     */
    __aicore__ inline void GBKCompute(const GlobalTensor<bfloat16_t> gBKWsGm, const GlobalTensor<bfloat16_t> outKgGm,
                                      LocalTensor<float> betaUbFloat, LocalTensor<float> gCumSumUbFloat,
                                      LocalTensor<float> gCumExpUbFloat, const GlobalTensor<bfloat16_t> keyContinousGm)
    {
        if constexpr (gOptional) {
            // tmp = -1.0 * beta * exp(g_cum)
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
        Broadcast<float, BROADCAST_AXIS, 1>(gBKUbFloat_, gBUbFloat_, kShape, betaShape);
        PipeBarrier<PIPE_V>();
        // data copy in kUbFloatCon_
        DataCopyInBf16WithStride(halfChunkSize_, dk_, keyContinousGm, dk_);
        auto bf16Key = inQueue_.DeQue<bfloat16_t>();
        Cast(kUbFloatCon_, bf16Key, RoundMode::CAST_NONE, halfChunkSize_ * dkAligned_);
        PipeBarrier<PIPE_V>();
        inQueue_.FreeTensor(bf16Key);
        // compute
        Mul(gBKUbFloat_, gBKUbFloat_, kUbFloatCon_, halfChunkSize_ * dkAligned_);
        PipeBarrier<PIPE_V>();
        gBKLocal_ = outQueue_.AllocTensor<bfloat16_t>();
        Cast(gBKLocal_, gBKUbFloat_, RoundMode::CAST_RINT, halfChunkSize_ * dkAligned_);
        outQueue_.EnQue<bfloat16_t>(gBKLocal_);
        uint64_t gBKBeginOffset = subOffset_ * dk_;
        DataCopyOutBf16(halfChunkSize_, dk_, dkAligned_, gBKWsGm[gBKBeginOffset]);
        PipeBarrier<PIPE_V>();
        if constexpr (gOptional) {
            // V2 已对 Z0 施加 exp(G_last-G_i)，因此 kg 必须保持原始 K，避免 C3 重复乘门控。
            uint64_t kgBeginOffset = subOffset_ * dk_;
            kgLocal_ = outQueue_.AllocTensor<bfloat16_t>();
            Cast(kgLocal_, kUbFloatCon_, RoundMode::CAST_RINT, halfChunkSize_ * dkAligned_);
            outQueue_.EnQue<bfloat16_t>(kgLocal_);
            DataCopyOutBf16(halfChunkSize_, dk_, dkAligned_, outKgGm[kgBeginOffset]); // stage1 out
        }
    }

    /**
     * 计算 beta 加权的 value。
     * 输入 V[C/2,Dv] BF16、beta[C/2] FP32；输出 vBeta[i,:]=beta_i*V_i，转为 BF16 供 A@vBeta。
     */
    __aicore__ inline void VBetaCompute(const GlobalTensor<bfloat16_t> valueGm,
                                        const GlobalTensor<bfloat16_t> vBetaWsGm, LocalTensor<float> betaUbFloat,
                                        LocalTensor<float> valueUbFloat, uint32_t subValidRows)
    {
        uint64_t vBeginOffset = subOffset_ * vRowStride_;
        DataCopyInBf16WithStride(subValidRows, dv_, valueGm[vBeginOffset], vRowStride_);
        valueLocal_ = inQueue_.DeQue<bfloat16_t>();
        Cast(valueUbFloat, valueLocal_, AscendC::RoundMode::CAST_NONE, subValidRows * dvAligned_);
        PipeBarrier<PIPE_V>();
        inQueue_.FreeTensor(valueLocal_);
        if (subValidRows < halfChunkSize_) {
            Duplicate(valueUbFloat[subValidRows * dvAligned_], static_cast<float>(0.0f),
                      (halfChunkSize_ - subValidRows) * dvAligned_);
            PipeBarrier<PIPE_V>();
        }
        uint32_t betaShape[2] = {halfChunkSize_, 1};
        uint32_t vShape[2] = {halfChunkSize_, dvAligned_};
        Broadcast<float, BROADCAST_AXIS, 1>(vBetaUbFloat_, betaUbFloat, vShape, betaShape);
        PipeBarrier<PIPE_V>();
        Mul(vBetaUbFloat_, valueUbFloat, vBetaUbFloat_, halfChunkSize_ * dvAligned_);
        PipeBarrier<PIPE_V>();
        vBetaLocal_ = outQueue_.AllocTensor<bfloat16_t>();
        Cast(vBetaLocal_, vBetaUbFloat_, RoundMode::CAST_RINT, halfChunkSize_ * dvAligned_);
        outQueue_.EnQue<bfloat16_t>(vBetaLocal_);
        DataCopyOutBf16(halfChunkSize_, dv_, dvAligned_, vBetaWsGm[subOffset_ * dv_]);
    }

    /**
     * 计算 Stage2 C2 使用的 Q'。
     * 输入 Q[C/2,Dk] BF16 和 exp(G)[C] FP32；输出 Q'_i=scale*exp(G_i)*Q_i BF16；无 g 时 exp(G_i)=1。
     */
    __aicore__ inline void QPrimeCompute(const GlobalTensor<bfloat16_t> outQPrimeGm, LocalTensor<float> gCumSumUbFloat,
                                         LocalTensor<float> gCumExpUbFloat,
                                         const GlobalTensor<bfloat16_t> queryContinousGm)
    {
        // data copy in qUbFloatCon_
        DataCopyInBf16WithStride(halfChunkSize_, dk_, queryContinousGm, dk_);
        auto bf16Query = inQueue_.DeQue<bfloat16_t>();
        Cast(qUbFloatCon_, bf16Query, RoundMode::CAST_NONE, halfChunkSize_ * dkAligned_);
        PipeBarrier<PIPE_V>();
        inQueue_.FreeTensor(bf16Query);
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

    /** 将连续 FP32 GM 的 len 个元素搬入 inQueue，输出为队列中的 FP32 LocalTensor。 */
    __aicore__ inline void DataCopyInFp32(uint64_t len, GlobalTensor<float> y)
    {
        DataCopyPadExtParams<float> padParams;
        DataCopyExtParams kkParams{static_cast<uint16_t>(1), static_cast<uint32_t>(len * sizeof(float)), 0, 0, 0};
        fp32InLocal_ = inQueue_.AllocTensor<float>();
        DataCopyPad(fp32InLocal_, y, kkParams, padParams);
        inQueue_.EnQue<float>(fp32InLocal_);
    }

    /** 将连续 BF16 GM 的 len 个元素搬入 inQueue，输出为队列中的 BF16 LocalTensor。 */
    __aicore__ inline void DataCopyInBf16(uint64_t len, GlobalTensor<bfloat16_t> y)
    {
        DataCopyPadExtParams<bfloat16_t> padParams;
        DataCopyExtParams kkParams{static_cast<uint16_t>(1), static_cast<uint32_t>(len * sizeof(bfloat16_t)), 0, 0, 0};
        bf16InLocal_ = inQueue_.AllocTensor<bfloat16_t>();
        DataCopyPad(bf16InLocal_, y, kkParams, padParams);
        inQueue_.EnQue<bfloat16_t>(bf16InLocal_);
    }

    /**
     * 从 beta[T,Nv] 按 Nv stride 抽取当前半 chunk，尾部补零并转为 FP32。
     * 输入 src BF16；输出 betaUbFloat[C/2] FP32。
     */
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

    /** 从 g[T,Nv] 按 Nv stride 抽取有效 chunk 到 gCumUbFloat[C] FP32，尾部补零。 */
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

    /** 按元素 stride 将 FP32 GM 二维切片搬入队列；输入 rows/cols/srcRowStride，输出连续或指定行距的 UB。 */
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

    /** 按元素 stride 将 BF16 GM 二维切片搬入队列，并对每行做 32B 对齐填充。 */
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

    /** 将 UB 中 [rows,colsAligned] BF16 的每行前 cols 个元素紧凑写回 GM。 */
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

    /**
     * 用 2x2 分块公式补齐 64x64 下三角逆矩阵的左下块。
     * 输入两个已由 AIV 求逆的 32x32 对角块；输出 A21=-A22*L21*A11，写回 attnWs 左下块。
     */
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

    /**
     * Stage1 通用 Cube 封装：计算 z=x@y（transB=true 时为 x@y^T）。
     * 输入 shape 同时给出原始 MNK 和单次 MNK；输出 z 的 dtype 由 CType/Matmul 模板决定。
     */
    template <typename MMType, typename CType>
    __aicore__ inline void AICProcess(MMType &mmRef, GlobalTensor<bfloat16_t> x, GlobalTensor<bfloat16_t> y,
                                      GlobalTensor<CType> z, const MatmulShapeParams &shape, bool transB = false)
    {
        mmRef.SetOrgShape(shape.m, shape.n, shape.k);
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
    GlobalTensor<bfloat16_t> vBetaWsGm_;
    GlobalTensor<bfloat16_t> kkWsGm_;
    GlobalTensor<bfloat16_t> attnWsGm_;
    GlobalTensor<bfloat16_t> gBKWsGm_;
    GlobalTensor<bfloat16_t> queryConGm_;
    GlobalTensor<bfloat16_t> keyConGm_;
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
#endif // CHUNK_GATED_DELTA_RULE_STAGE1_ARCH35_H
