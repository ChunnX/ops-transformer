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
 * \file chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_halflocal.h
 * \brief
 */
#ifndef CHUNK_GATED_DELTA_RULE_NEUMANN_STAGE1_C64_1616_FP16_HALFLOCAL_H
#define CHUNK_GATED_DELTA_RULE_NEUMANN_STAGE1_C64_1616_FP16_HALFLOCAL_H

#include "kernel_tiling/kernel_tiling.h"
#include "../chunk_gated_delta_rule_neumann_utils.h"
#include "../chunk_gated_delta_rule_neumann_tiling_data.h"

namespace ChunkGatedDeltaRuleNeumann {
using namespace AscendC;
using namespace matmul;

using aT1 = MatmulType<TPosition::GM, CubeFormat::ND, float>;
using bT1 = MatmulType<TPosition::GM, CubeFormat::ND, float>;
using cT1 = MatmulType<TPosition::GM, CubeFormat::ND, float>;
using StageOneMT = matmul::MatmulImpl<aT1, bT1, cT1>;
using aFp16T1 = MatmulType<TPosition::GM, CubeFormat::ND, half>;
using bFp16T1 = MatmulType<TPosition::GM, CubeFormat::ND, half>;
using cFp16T1 = MatmulType<TPosition::GM, CubeFormat::ND, half>;
using cFp16OutT1 = MatmulType<TPosition::GM, CubeFormat::ND, float>;
using aFp16L1T1 = MatmulType<TPosition::TSCM, CubeFormat::NZ, half>;
using bFp16L1T1 = MatmulType<TPosition::TSCM, CubeFormat::NZ, half>;
using cFp16L1T1 = MatmulType<TPosition::TSCM, CubeFormat::NZ, half>;
using StageOneFp16MT = matmul::MatmulImpl<aFp16T1, bFp16T1, cFp16T1>;
using StageOneFp16OutMT = matmul::MatmulImpl<aFp16T1, bFp16T1, cFp16OutT1>;
using StageOneFp16L1MT = matmul::MatmulImpl<aFp16L1T1, bFp16L1T1, cFp16L1T1>;
using StageOneFp16L1GmOutMT = matmul::MatmulImpl<aFp16L1T1, bFp16T1, cFp16OutT1>;
using aLocalT1 = MatmulType<TPosition::VECCALC, CubeFormat::ND, float>;
using bLocalT1 = MatmulType<TPosition::VECCALC, CubeFormat::ND, float>;
using cLocalT1 = MatmulType<TPosition::VECCALC, CubeFormat::ND, float>;
using StageOneGmGmLocalMT = matmul::MatmulImpl<aT1, bT1, cLocalT1>;
using StageOneLocalMT = matmul::MatmulImpl<aLocalT1, bLocalT1, cLocalT1>;
using StageOneGmLocalGmMT = matmul::MatmulImpl<aT1, bLocalT1, cT1>;

constexpr uint64_t UB_REST_BYTES = 100 * 1024;  // 100KB
constexpr uint64_t INVERSE_SHAPE = 16;          // 16x16 Neumann basic block
constexpr uint64_t C64_BLOCK32 = 32;
constexpr uint64_t AIC_RAW_L1_BYTES = 16 * 1024;
constexpr uint64_t AIC_RAW_L0_AB_BYTES = 32 * 1024;
constexpr uint64_t AIC_RAW_L0_C_BYTES = 64 * 1024;
constexpr uint32_t ALIGN_SIZE = 16;
constexpr uint32_t RAW_FINAL_N_TILE = 16;
constexpr uint32_t RAW_FLOAT_C0 = BLOCK_SIZE / sizeof(float);
constexpr uint32_t RAW_FRAC_ELEMS = BLOCK_CUBE * RAW_FLOAT_C0;
constexpr uint32_t RAW_FRACTAL_BYTES = 512;
constexpr uint32_t RAW_CACHELINE_FLOATS = 16;
static constexpr IsResetLoad3dConfig RAW_LOAD3DV2_CONFIG = {true, true};
static constexpr IsResetLoad3dConfig RAW_LOAD3DV2_FMATRIX_CONFIG = {false, false};
constexpr uint32_t RAW_MMAD_DEBUG_STEP = 0;      // 0: full path, 1: GM->L1 only, 2: L1->L0 only, 3: Mmad only, 4: no-op, 5: AIC direct GM overwrite, 6: const L0A/L0B MMAD writeback
constexpr uint8_t RAW_MMAD_SINGLE_UNIT_FLAG = 0;
constexpr uint8_t RAW_MMAD_NON_FINAL_UNIT_FLAG = 2;
constexpr uint8_t RAW_MMAD_FINAL_UNIT_FLAG = 3;
constexpr bool RAW_MMAD_USE_MATMULIMPL_FALLBACK = false;
constexpr bool RAW_MMAD_DEBUG_OVERWRITE_PRODUCT = false;
constexpr bool RAW_MMAD_DEBUG_WRITE_QK = false;
constexpr bool RAW_MMAD_DEBUG_MATMUL_READBACK = false;
constexpr bool RAW_MMAD_DEBUG_TMP_READBACK = false;
constexpr bool RAW_MMAD_DEBUG_READBACK_TO_QK = false;
constexpr bool RAW_MMAD_DEBUG_SELF_READBACK = false;
constexpr bool RAW_MMAD_DEBUG_FINAL_USE_KK = false;
constexpr bool RAW_MMAD_DEBUG_QK_USE_KK = false;
constexpr bool RAW_MMAD_DEBUG_STAGE1_OUT_USE_KK = false;
constexpr bool RAW_MMAD_DEBUG_OVERWRITE_INV_WITH_KK = false;
constexpr bool RAW_MMAD_DEBUG_ZERO_QK = false;
constexpr bool RAW_MMAD_DEBUG_BASE0_FALLBACK = false;
constexpr bool RAW_MMAD_DEBUG_BASE16_FALLBACK = false;
constexpr bool RAW_MMAD_DEBUG_SCALAR_ADD_PRODUCT = false;
constexpr bool RAW_MMAD_USE_MATMULIMPL_POWER_FALLBACK = true;
constexpr bool RAW_MMAD_USE_MATMULIMPL_COMPOSE_FALLBACK = true;
constexpr bool RAW_MMAD_USE_MATMULIMPL_FINAL_FALLBACK = true;
constexpr bool RAW_MMAD_USE_MATMULIMPL_KCUM_FINAL_FALLBACK = true;
constexpr bool RAW_MMAD_USE_MATMULIMPL_VINNER_FINAL_FALLBACK = true;
constexpr bool RAW_MMAD_INVADD_USE_POWER_PLUS_IDENTITY = true;
constexpr uint32_t RAW_MMAD_INVADD_POST_STORE_SYNC_MODE = 2;  // 0: full DCCI, 1: FIX->M/MTE2, 2: FIX->MTE2 only
constexpr bool RAW_MMAD_DEBUG_OVERWRITE_RAW_FINAL_WITH_MATMULIMPL = false;
constexpr bool RAW_MMAD_DEBUG_SKIP_RAW_FINAL = false;
constexpr bool RAW_MMAD_DEBUG_RAW_FINAL_TO_TMP_THEN_FALLBACK = false;
constexpr bool RAW_MMAD_DEBUG_RAW_FINAL_TO_INV_THEN_FALLBACK = false;
constexpr bool RAW_MMAD_DEBUG_RAW_FINAL_TO_ATTN_THEN_FALLBACK = false;
constexpr bool RAW_MMAD_DEBUG_RAW_FINAL_TO_SAFE_SCRATCH_THEN_FALLBACK = false;
constexpr uint32_t RAW_MMAD_DEBUG_RAW_FINAL_TILE_LIMIT = 0;  // 0 means no debug limit.
constexpr bool RAW_MMAD_NEEDS_AIC_LOCAL_BUFFERS = !RAW_MMAD_USE_MATMULIMPL_POWER_FALLBACK ||
                                                 !RAW_MMAD_USE_MATMULIMPL_COMPOSE_FALLBACK ||
                                                 !(RAW_MMAD_USE_MATMULIMPL_FINAL_FALLBACK ||
                                                   (RAW_MMAD_USE_MATMULIMPL_KCUM_FINAL_FALLBACK &&
                                                    RAW_MMAD_USE_MATMULIMPL_VINNER_FINAL_FALLBACK)) ||
                                                 !RAW_MMAD_USE_MATMULIMPL_FALLBACK;

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
    __aicore__ inline Stage1(StageOneMT &mmFp32, StageOneMT &mmFp32C16, StageOneMT &mmFp32C32,
                             StageOneMT &mmFp32C64, StageOneFp16OutMT &, StageOneFp16MT &, StageOneFp16MT &,
                             StageOneFp16L1GmOutMT &, StageOneFp16L1MT &, StageOneFp16L1MT &,
                             StageOneGmGmLocalMT &, StageOneGmGmLocalMT &, StageOneLocalMT &, StageOneLocalMT &,
                             StageOneGmLocalGmMT &, StageOneGmLocalGmMT &)
        : Stage1(mmFp32, mmFp32C16, mmFp32C32, mmFp32C64) {}
    __aicore__ inline Stage1(StageOneMT &mmFp32, StageOneMT &mmFp32C16, StageOneMT &mmFp32C32,
                             StageOneMT &mmFp32C64)
        : mmFp32(mmFp32), mmFp32C16(mmFp32C16), mmFp32C32(mmFp32C32), mmFp32C64(mmFp32C64) {}
    __aicore__ inline Stage1(StageOneMT &mmFp32, StageOneMT &mmFp32C16, StageOneMT &mmFp32C32,
                             StageOneMT &mmFp32C64,
                             StageOneFp16OutMT &, StageOneFp16MT &, StageOneFp16MT &,
                             StageOneFp16L1GmOutMT &, StageOneFp16L1MT &, StageOneFp16L1MT &)
        : Stage1(mmFp32, mmFp32C16, mmFp32C32, mmFp32C64) {}
    __aicore__ inline void SetGlobalTensors(const GDRStageOneInitParams &initParams)
    {
        queryBaseGm_ = initParams.query;
        keyBaseGm_ = initParams.key;
        valueBaseGm_ = initParams.value;
        betaBaseGm_ = initParams.beta;

        outGCumExpBaseGm_ = initParams.gCumExp;
        outKCumdecayBaseGm_ = initParams.kCumdecay;
        outVInnerBaseGm_ = initParams.vInner;
        outQPrimeBaseGm_ = initParams.qPrime;
        outKgBaseGm_ = initParams.kG;
        outQkBaseGm_ = initParams.qK;
        stageOneMask_ = initParams.stageOneMask;

        if (gOptional_) {
            gBaseGm_ = initParams.g;
        }

        uint64_t workSpaceOffset = 0;
        gBKWsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(initParams.ws + workSpaceOffset +
                                                                  coreIdx_ * chunkSize_ * dk_ * sizeof(float)));

        workSpaceOffset += coreNum_ * chunkSize_ * dk_ * sizeof(float);
        kkWsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(initParams.ws + workSpaceOffset +
                                                                 coreIdx_ * chunkSize_ * chunkSize_ * sizeof(float)));

        workSpaceOffset += coreNum_ * chunkSize_ * chunkSize_ * sizeof(float);
        vBetaWsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(initParams.ws + workSpaceOffset +
                                                                    coreIdx_ * chunkSize_ * dv_ * sizeof(float)));

        workSpaceOffset += coreNum_ * chunkSize_ * dv_ * sizeof(float);
        attnWsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(initParams.ws + workSpaceOffset +
                                                                   coreIdx_ * chunkSize_ * chunkSize_ * sizeof(float)));

        workSpaceOffset += coreNum_ * chunkSize_ * chunkSize_ * sizeof(float);
        queryContinousGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(initParams.ws + workSpaceOffset +
                                                                          coreIdx_ * chunkSize_ * dk_ * sizeof(float)));

        workSpaceOffset += coreNum_ * chunkSize_ * dk_ * sizeof(float);
        keyContinousGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(initParams.ws + workSpaceOffset +
                                                                          coreIdx_ * chunkSize_ * dk_ * sizeof(float)));

        workSpaceOffset += coreNum_ * chunkSize_ * dk_ * sizeof(float);
        L_powWsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(initParams.ws + workSpaceOffset +
                                                                   coreIdx_ * chunkSize_ * chunkSize_ * sizeof(float)));
        workSpaceOffset += coreNum_ * chunkSize_ * chunkSize_ * sizeof(float);
        tmpWsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(initParams.ws + workSpaceOffset +
                                                                 coreIdx_ * chunkSize_ * chunkSize_ * sizeof(float)));
        workSpaceOffset += coreNum_ * chunkSize_ * chunkSize_ * sizeof(float);
        INVWsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(initParams.ws + workSpaceOffset +
                                                                 coreIdx_ * chunkSize_ * chunkSize_ * sizeof(float)));

        workSpaceOffset += coreNum_ * chunkSize_ * chunkSize_ * sizeof(float);
        uint32_t rhsMax = dk_ > dv_ ? dk_ : dv_;
        rawFinalScratchGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(
            initParams.ws + workSpaceOffset + coreIdx_ * chunkSize_ * rhsMax * sizeof(float)));
    }

    __aicore__ inline void InitLocalBuffers()
    {
        uint32_t maxLen = AscendC::Std::max(AscendC::Std::max(dvAligned_ / 2, dkAligned_ / 2), chunkSize_);
        pipe_->InitBuffer(fp32InQueue_, BUFFER_NUM_ONE, chunkSize_ * maxLen * sizeof(float));
        pipe_->InitBuffer(fp32OutQueue_, BUFFER_NUM_ONE, chunkSize_ * maxLen * sizeof(float));
        if (gOptional_) {
            pipe_->InitBuffer(gOutQueue_, BUFFER_NUM_ONE, chunkSize_ * sizeof(float));
        }

        pipe_->InitBuffer(tmpBuff_, UB_REST_BYTES);
        uint32_t buffOffset = 0;
        betaUbBfloat16_ = tmpBuff_.GetWithOffset<bfloat16_t>(static_cast<uint32_t>(halfChunkSize_), buffOffset);
        buffOffset += halfChunkSize_ * sizeof(bfloat16_t);
        buffOffset = (buffOffset + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;

        gCumUbFloat_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(chunkSize_), buffOffset);
        buffOffset += chunkSize_ * sizeof(float);

        gBUbFloat_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(halfChunkSize_), buffOffset);
        buffOffset += halfChunkSize_ * sizeof(float);

        gEndBroadUbFloat_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(halfChunkSize_), buffOffset);
        buffOffset += halfChunkSize_ * sizeof(float);

        betaUbFloat_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(halfChunkSize_), buffOffset);
        buffOffset += halfChunkSize_ * sizeof(float);

        gBroadUbFloat_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(chunkSize_ * maxLen), buffOffset);
        gammaUbFloat_ = gBroadUbFloat_;
        kUbFloat_ = gBroadUbFloat_;
        valueUbFloat_ = gBroadUbFloat_;
        qUbFloat_ = gBroadUbFloat_;
        buffOffset += chunkSize_ * maxLen  * sizeof(float);

        gTransBroadUbFloat_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(chunkSize_ * maxLen), buffOffset);
        attnUbFloat_ = gTransBroadUbFloat_;
        gCumExpBroadUbFloat_ = gTransBroadUbFloat_;
        buffOffset += chunkSize_ * maxLen * sizeof(float);

        qUbFloatCon_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(halfChunkSize_ * dkAligned_), buffOffset);
        buffOffset += halfChunkSize_ * dkAligned_ * sizeof(float);

        kUbFloatCon_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(halfChunkSize_ * dkAligned_), buffOffset);
        buffOffset += halfChunkSize_ * dkAligned_ * sizeof(float);

        gatherOffsetFp32_ = tmpBuff_.GetWithOffset<uint32_t>(static_cast<uint32_t>(chunkSize_), buffOffset);
        buffOffset += chunkSize_ * sizeof(uint32_t);

        gatherOffsetBf16_ = tmpBuff_.GetWithOffset<uint32_t>(static_cast<uint32_t>(halfChunkSize_), buffOffset);
    }

    __aicore__ inline void InitAicLocalBuffers()
    {
        pipe_->InitBuffer(aicL1ABuff_, AIC_RAW_L1_BYTES);
        pipe_->InitBuffer(aicL1BBuff_, AIC_RAW_L1_BYTES);
        aicL1A_ = aicL1ABuff_.Get<float>();
        aicL1B_ = aicL1BBuff_.Get<float>();

        pipe_->InitBuffer(aicL0ABuff_, AIC_RAW_L0_AB_BYTES);
        pipe_->InitBuffer(aicL0BBuff_, AIC_RAW_L0_AB_BYTES);
        pipe_->InitBuffer(aicL0CBuff_, AIC_RAW_L0_C_BYTES);
        aicL0A_ = aicL0ABuff_.Get<float>();
        aicL0B_ = aicL0BBuff_.Get<float>();
        aicL0C_ = aicL0CBuff_.Get<float>();
    }

    __aicore__ inline void EnsureAicLocalBuffers()
    {
        if constexpr (RAW_MMAD_NEEDS_AIC_LOCAL_BUFFERS) {
            if (!aicLocalBuffersReady_) {
                InitAicLocalBuffers();
                aicLocalBuffersReady_ = true;
            }
        }
    }

    __aicore__ inline void InitGatherBuffer()
    {
        for (uint32_t i = 0; i < chunkSize_; ++i) {
            gatherOffsetFp32_.SetValue(i, i * BLOCK_SIZE);
        }
        for (uint32_t i = 0; i < halfChunkSize_; ++i) {
            gatherOffsetBf16_.SetValue(i, i * BLOCK_SIZE);
        }
        int32_t eventID = static_cast<int32_t>(pipe_->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventID);
        WaitFlag<HardEvent::S_V>(eventID);
    }

    __aicore__ inline void Init(const GDRStageOneInitParams &initParams, TPipe *pipe,
                                const ChunkGatedDeltaRuleNeumannTilingData *tilingData)
    {
        pipe_ = pipe;
        tiling_ = tilingData;
        nk_ = tiling_->nk;
        nv_ = tiling_->nv;
        chunkSize_ = tiling_->chunkSize;
        dk_ = tiling_->dk;
        dv_ = tiling_->dv;
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
        validLen_ = chunkSize_;
        subValidRows_ = halfChunkSize_;
        subOffset_ = subBlockIdx_ * halfChunkSize_;
        coreIdx_ = GetBlockIdx();
        aicLocalBuffersReady_ = false;
        if ASCEND_IS_AIV{
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

        for (int32_t taskId = start; taskId < end; ++taskId) {
            validLen_ = chunkSize_;
            uint64_t nId   = taskId % nv_;
            uint64_t cgId = taskId / nv_;
            // 尾chunk处理
            if (cgId == numChunk_ - 1 && cg_.length % chunkSize_ != 0) {
                validLen_ = cg_.length % chunkSize_;
            }
            if (validLen_ < halfChunkSize_) {
                subValidRows_ = (subBlockIdx_ == 0) ? validLen_ : 0;
            } else {
                subValidRows_ = (subBlockIdx_ == 0) ? halfChunkSize_ : validLen_ - halfChunkSize_;
            }
            // chunk在全局T上的起始行 = chunkGroup起始行 + chunk内偏移
            uint64_t chunkStartRow = cg_.startPos + cgId * chunkSize_;
            SetChunkTensors(nId, cgId, chunkStartRow);
            ProcessOneChunk();
        }
    }

private:
    // ----------------------------------------------------------
    // SetChunkTensors
    //   nId       : head 编号 (Nv 维度)
    //   localChunkId : CG 内的 chunk 编号 (0 ~ CG_CHUNKS-1)
    //   chunkStartRow   : 当前 chunk 在全局 T 上的起始行
    // ----------------------------------------------------------
   __aicore__ inline void SetChunkTensors(uint64_t nId, uint64_t localChunkId, uint64_t chunkStartRow)
    {
        uint64_t kid = nId * nk_ / nv_;
        uint64_t subRow = chunkStartRow + subOffset_;
        uint64_t qk_base = subRow * nk_ * dk_ + kid * dk_;
        queryGm_ = queryBaseGm_[qk_base];
        keyGm_   = keyBaseGm_[qk_base];

        uint64_t vOffset = chunkStartRow * vRowStride_ + nId * dv_;
        valueGm_ = valueBaseGm_[vOffset];

        uint64_t bgOffset = chunkStartRow * nv_ + nId;
        betaGm_ = betaBaseGm_[bgOffset];
        if (gOptional_) {
            gGm_ = gBaseGm_[bgOffset];
        }

        uint64_t cgLenPad = (cg_.length + chunkSize_ - 1) / chunkSize_ * chunkSize_;
        uint64_t chunkRowBase = nId * cgLenPad + localChunkId * chunkSize_;

        outGCumExpGm_ = outGCumExpBaseGm_[chunkRowBase];
        outKCumdecayGm_ = outKCumdecayBaseGm_[chunkRowBase * dk_];
        outQPrimeGm_ = outQPrimeBaseGm_[chunkRowBase * dk_];
        outKgGm_ = outKgBaseGm_[chunkRowBase * dk_];
        outVInnerGm_ = outVInnerBaseGm_[chunkRowBase * dv_];
        outQkGm_ = outQkBaseGm_[chunkRowBase * chunkSize_];
    }

    __aicore__ inline void ProcessOneChunk()
    {
        if ASCEND_IS_AIC {
            SetAtomicNone();
            RestoreRawMmadState();
            AscendC::CrossCoreWaitFlag(0x9);  // 同步0
            // key @ key.transpose(-1,-2)
            AICProcess(keyContinousGm_, keyContinousGm_, kkWsGm_,
                       {chunkSize_, chunkSize_, dk_, chunkSize_, chunkSize_, dk_}, true);
            AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(0x8);  // 同步1：通知 AIV kkWsGm_ 就绪
            ClearQkMatrix();
            AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(0xd);  // qkt compact buffer cleared
            AscendC::CrossCoreWaitFlag(0x7);  // 同步2：等待 AIV 写出 L_powWsGm_ / INVWsGm_

            // AIV generates the initial lower-triangular L and I-L in GM; AIC does all inverse math.
            C64InverseBy16x16Neumann();
            RestoreRawMmadState();

            AscendC::CrossCoreWaitFlag(0x6);  // 同步3：等待 AIV GBKCompute
            if constexpr (RAW_MMAD_DEBUG_STEP == 5) {
                DebugClearInvMatrix();
            }
            if constexpr (RAW_MMAD_DEBUG_OVERWRITE_INV_WITH_KK) {
                AICProcess(kkWsGm_, kkWsGm_, INVWsGm_,
                           {chunkSize_, chunkSize_, chunkSize_, chunkSize_, chunkSize_, chunkSize_});
                SyncFixToMte2();
            }
            // INV @ k_cumdecay
            if constexpr (RAW_MMAD_DEBUG_FINAL_USE_KK) {
                FinalMmad64ByN<RAW_MMAD_USE_MATMULIMPL_KCUM_FINAL_FALLBACK>(kkWsGm_, gBKWsGm_, outKCumdecayGm_,
                                                                            dk_);
            } else {
                FinalMmad64ByN<RAW_MMAD_USE_MATMULIMPL_KCUM_FINAL_FALLBACK>(INVWsGm_, gBKWsGm_, outKCumdecayGm_,
                                                                            dk_);
            }
            AscendC::CrossCoreWaitFlag(0x5);  // 同步4：等待 AIV VBetaCompute
            if constexpr (RAW_MMAD_DEBUG_STEP == 5) {
                DebugClearInvMatrix();
            }
            // INV @ v_beta    stage1 out
            if constexpr (RAW_MMAD_DEBUG_FINAL_USE_KK) {
                FinalMmad64ByN<RAW_MMAD_USE_MATMULIMPL_VINNER_FINAL_FALLBACK>(kkWsGm_, vBetaWsGm_, outVInnerGm_,
                                                                              dv_);
            } else {
                FinalMmad64ByN<RAW_MMAD_USE_MATMULIMPL_VINNER_FINAL_FALLBACK>(INVWsGm_, vBetaWsGm_, outVInnerGm_,
                                                                              dv_);
            }
            if constexpr (RAW_MMAD_DEBUG_STAGE1_OUT_USE_KK) {
                AICProcess(kkWsGm_, kkWsGm_, outKCumdecayGm_,
                           {chunkSize_, dk_, chunkSize_, chunkSize_, dk_, chunkSize_});
                AICProcess(kkWsGm_, kkWsGm_, outVInnerGm_,
                           {chunkSize_, dv_, chunkSize_, chunkSize_, dv_, chunkSize_});
            }
            RestoreRawMmadState();
            AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(0xc);  // 同步5：通知 AIV 本 chunk 的共享 workspace 已读完
        }
        if ASCEND_IS_AIV {
            // 获取连续QK
            QKPreProcess();
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(0x9);  // 同步0
            AscendC::CrossCoreWaitFlag(0xd);  // 等待 AIC 清理 qkt compact buffer
            TailQKComputeAiv();
            if (gOptional_) {
                // g_cum_exp = g.cumsum(dim=-1).exp()
                GCumExpCompute();
                // attn_1 = (g_cum_exp[:None] / g_cum_exp[None,:]) * mask
                GammaCompute();
            }
            BetaCopyInWithStride();
            AscendC::CrossCoreWaitFlag(0x8);  // 同步1：等待 AIC kkWsGm_ 就绪
            // L = (key @ key.T) * beta * decay * lower-triangular-mask
            KKBetaCompute();
            // Write L_powWsGm_ = L and INVWsGm_ = I - L. Inverse iterations stay on AIC.
            InverseCompute();
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(0x7);  // 同步2：通知 AIC 初始矩阵已就绪

            GBKCompute();
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(0x6);  // 同步3
            // v_beta = value * beta.unsqueeze(-1)  # (C, Dv)
            VBetaCompute();
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(0x5);  // 同步4
            // q_prime = query * scale_ * g_cum_exp[:, None]       # (C, Dk)
            QPrimeCompute();
            AscendC::CrossCoreWaitFlag(0xc);  // 同步5：等待 AIC 读完共享 workspace，避免下一 chunk 覆盖
        }
    }

    __aicore__ inline void QKPreProcessCompute(const GlobalTensor<bfloat16_t>& srcGm, const GlobalTensor<float>& dstGm,
                                                LocalTensor<float>& dstBuffer, bool kgFlag = false)
    {
        // copyIn
        DataCopyInBf16WithStride(subValidRows_, dk_, srcGm, nk_ * dk_);
        // compute
        LocalTensor<bfloat16_t> bf16Tensor = fp32InQueue_.DeQue<bfloat16_t>();
        Cast(dstBuffer, bf16Tensor, AscendC::RoundMode::CAST_NONE, subValidRows_ * dkAligned_);
        PipeBarrier<PIPE_V>();
        fp32InQueue_.FreeTensor(bf16Tensor);

        if (subValidRows_ < halfChunkSize_) {
            Duplicate(dstBuffer[subValidRows_ * dkAligned_], static_cast<float>(0.0f),
                      (halfChunkSize_ - subValidRows_) * dkAligned_);
            PipeBarrier<PIPE_V>();
        }

        // copyOut
        auto tmpTensor = fp32OutQueue_.AllocTensor<float>();
        DataCopy(tmpTensor, dstBuffer, halfChunkSize_ * dkAligned_);
        fp32OutQueue_.EnQue(tmpTensor);
        tmpTensor = fp32OutQueue_.DeQue<float>();

        uint32_t srcStride = (dkAligned_ - dk_) * sizeof(float) / BLOCK_SIZE;
        DataCopyExtParams outParams{static_cast<uint16_t>(halfChunkSize_),
                                    static_cast<uint32_t>(dk_ * sizeof(float)), srcStride, 0, 0};
        DataCopyPad(dstGm, tmpTensor, outParams);
        if (!gOptional_ && kgFlag) {
            DataCopyPad(outKgGm_[subOffset_ * dk_], tmpTensor, outParams);
        }
        SyncMte3ToMte2();
        fp32OutQueue_.FreeTensor(tmpTensor);
    }

    __aicore__ inline void QKPreProcess()
    {
        uint64_t outOffset = subOffset_ * dk_;
        QKPreProcessCompute(queryGm_, queryContinousGm_[outOffset], qUbFloatCon_);
        QKPreProcessCompute(keyGm_, keyContinousGm_[outOffset], kUbFloatCon_, true);
    }

    __aicore__ inline void TailQKComputeAiv()
    {
        if (subBlockIdx_ == 0) {
            uint32_t localRows = AscendC::Std::min(validLen_, halfChunkSize_);
            for (uint32_t row = 0; row < localRows; ++row) {
                for (uint32_t col = 0; col < validLen_; ++col) {
                    float sum = 0.0f;
                    if (col < halfChunkSize_) {
                        for (uint32_t k = 0; k < dk_; ++k) {
                            sum += qUbFloatCon_.GetValue(row * dkAligned_ + k) *
                                   kUbFloatCon_.GetValue(col * dkAligned_ + k);
                        }
                    } else {
                        for (uint32_t k = 0; k < dk_; ++k) {
                            sum += qUbFloatCon_.GetValue(row * dkAligned_ + k) *
                                   keyContinousGm_.GetValue(col * dk_ + k);
                        }
                    }
                    outQkGm_.SetValue(row * validLen_ + col, sum);
                }
            }
        } else {
            for (uint32_t row = halfChunkSize_; row < validLen_; ++row) {
                for (uint32_t col = 0; col < validLen_; ++col) {
                    float sum = 0.0f;
                    uint32_t localRow = row - halfChunkSize_;
                    if (col < halfChunkSize_) {
                        for (uint32_t k = 0; k < dk_; ++k) {
                            sum += qUbFloatCon_.GetValue(localRow * dkAligned_ + k) *
                                   keyContinousGm_.GetValue(col * dk_ + k);
                        }
                    } else {
                        uint32_t localCol = col - halfChunkSize_;
                        for (uint32_t k = 0; k < dk_; ++k) {
                            sum += qUbFloatCon_.GetValue(localRow * dkAligned_ + k) *
                                   kUbFloatCon_.GetValue(localCol * dkAligned_ + k);
                        }
                    }
                    outQkGm_.SetValue(row * validLen_ + col, sum);
                }
            }
        }
        SyncSToMte2();
        FlushMatrixCacheLines(outQkGm_, validLen_, validLen_, validLen_);
    }

    __aicore__ inline void GCumExpCompute()
    {
        // Copy g
        GCopyInWithStride();
        // CumSum计算
        uint32_t outer = 1;
        uint32_t inner = chunkSize_;
        CumSumInfo cumSumInfo{outer, inner};
        CumSum<float>(gCumUbFloat_, gCumUbFloat_, gCumUbFloat_, cumSumInfo);
        PipeBarrier<PIPE_V>();
        // Exp计算
        gCumExpUbFloat_ = gOutQueue_.AllocTensor<float>();
        Exp<float, 0, true>(gCumExpUbFloat_, gCumUbFloat_, chunkSize_);
        gOutQueue_.EnQue<float>(gCumExpUbFloat_);
        DataCopyOutG(validLen_);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void GammaCompute()
    {
        // BroadCast
        uint32_t divShape[2] = {chunkSize_, chunkSize_};
        uint32_t gShape[2] = {chunkSize_, 1};
        uint32_t gTransShape[2] = {1, chunkSize_};
        Broadcast<float, BROADCAST_AXIS, 1>(gBroadUbFloat_, gCumExpUbFloat_, divShape, gShape);
        Broadcast<float, BROADCAST_AXIS, 0>(gTransBroadUbFloat_, gCumExpUbFloat_, divShape, gTransShape);
        PipeBarrier<PIPE_V>();
        // div
        Div(gammaUbFloat_, gBroadUbFloat_, gTransBroadUbFloat_, chunkSize_ * chunkSize_);
        PipeBarrier<PIPE_V>();
        // mask
        DataCopyInFp32(chunkSize_ * chunkSize_, stageOneMask_[GetBlockIdx() * chunkSize_ * chunkSize_]);
        kkLocal_ = fp32InQueue_.DeQue<float>();
        Mul(gammaUbFloat_, gammaUbFloat_, kkLocal_, chunkSize_ * chunkSize_);
        fp32InQueue_.FreeTensor(kkLocal_);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void KKBetaCompute()
    {
        // copy value
        uint32_t kkLength = chunkSize_ * halfChunkSize_;
        uint64_t kkBeginOffset = subOffset_ * chunkSize_;
        DataCopyInFp32(kkLength, kkWsGm_[kkBeginOffset]);
        kkLocal_ = fp32InQueue_.DeQue<float>();

        uint32_t betaShape[2] = {halfChunkSize_, 1};
        uint32_t kkShape[2] = {halfChunkSize_, chunkSize_};
        Broadcast<float, BROADCAST_AXIS, 1>(attnUbFloat_, betaUbFloat_, kkShape, betaShape);
        PipeBarrier<PIPE_V>();
        Mul(attnUbFloat_, kkLocal_, attnUbFloat_, chunkSize_ * halfChunkSize_);
        PipeBarrier<PIPE_V>();
        fp32InQueue_.FreeTensor(kkLocal_);
    }

    __aicore__ inline void InverseCompute()
    {
        uint64_t curVecLen = chunkSize_ * halfChunkSize_;

        if (gOptional_) {
            Mul(attnUbFloat_, attnUbFloat_, gammaUbFloat_[subOffset_ * chunkSize_], curVecLen);
        } else {
            DataCopyInFp32(curVecLen, stageOneMask_[subOffset_ * chunkSize_]);
            kkLocal_ = fp32InQueue_.DeQue<float>();
            Mul(attnUbFloat_, attnUbFloat_, kkLocal_, curVecLen);
            fp32InQueue_.FreeTensor(kkLocal_);
        }
        PipeBarrier<PIPE_V>();

        // 写 L 到 L_powWsGm_（attnUbFloat_ 此时是 L 的对应半块）
        inverseLocal_ = fp32OutQueue_.AllocTensor<float>();
        DataCopy(inverseLocal_, attnUbFloat_, curVecLen);
        PipeBarrier<PIPE_V>();
        fp32OutQueue_.EnQue(inverseLocal_);
        DataCopyOutFp32(halfChunkSize_, chunkSize_, chunkSize_, L_powWsGm_[subOffset_ * chunkSize_]);

        // 构造 I - L，仅写 INVWsGm_。attnWsGm_ 会在 Round1 被 L@L 覆盖，不需要初始化。
        inverseLocal_ = fp32OutQueue_.AllocTensor<float>();
        Muls(inverseLocal_, attnUbFloat_, static_cast<float>(-1.0), curVecLen);
        PipeBarrier<PIPE_V>();
        for (uint32_t i = 0; i < halfChunkSize_; ++i) {
            inverseLocal_.SetValue(i * chunkSize_ + subOffset_ + i, static_cast<float>(1.0));
        }
        PipeBarrier<PIPE_V>();
        fp32OutQueue_.EnQue(inverseLocal_);
        DataCopyOutFp32(halfChunkSize_, chunkSize_, chunkSize_, INVWsGm_[subOffset_ * chunkSize_]);
    }

    __aicore__ inline void GBKCompute()
    {
        if (gOptional_) {
            // tmp = -1.0 * beta * g_cum_exp
            Mul(gBUbFloat_, betaUbFloat_, gCumExpUbFloat_[subOffset_], halfChunkSize_);
            PipeBarrier<PIPE_V>();
            Muls(gBUbFloat_, gBUbFloat_, static_cast<float>(-1), halfChunkSize_);
        } else {
            Muls(gBUbFloat_, betaUbFloat_, static_cast<float>(-1), halfChunkSize_);
        }
        PipeBarrier<PIPE_V>();
        // k_cumdecay = k * tmp =  -1.0 * k * beta * g_cum_exp
        uint32_t betaShape[2] = {halfChunkSize_, 1};
        uint32_t kShape[2] = {halfChunkSize_, dkAligned_};
        gBKLocal_ = fp32OutQueue_.AllocTensor<float>();
        Broadcast<float, BROADCAST_AXIS, 1>(gBKLocal_, gBUbFloat_, kShape, betaShape);
        PipeBarrier<PIPE_V>();
        Mul(gBKLocal_, gBKLocal_, kUbFloatCon_, halfChunkSize_ * dkAligned_);
        fp32OutQueue_.EnQue<float>(gBKLocal_);
        uint64_t gBKBeginOffset = subOffset_ * dk_;
        DataCopyOutFp32(halfChunkSize_, dk_, dkAligned_, gBKWsGm_[gBKBeginOffset]);
        PipeBarrier<PIPE_V>();
        if (gOptional_) {
            // kg = k * (g_cum_exp[-1, None] / g_cum_exp)[..., None]
            uint32_t gEndShape[2] = {1, 1};
            uint32_t gBroadShape[2] = {halfChunkSize_, 1};
            Broadcast<float, BROADCAST_AXIS, 0>(gEndBroadUbFloat_, gCumExpUbFloat_[chunkSize_ - 1],
                                                gBroadShape, gEndShape);
            PipeBarrier<PIPE_V>();
            Div(gEndBroadUbFloat_, gEndBroadUbFloat_, gCumExpUbFloat_[subOffset_], halfChunkSize_);
            PipeBarrier<PIPE_V>();
            kgLocal_ = fp32OutQueue_.AllocTensor<float>();
            Broadcast<float, BROADCAST_AXIS, 1>(kgLocal_, gEndBroadUbFloat_, kShape, gBroadShape);
            PipeBarrier<PIPE_V>();
            Mul(kgLocal_, kgLocal_, kUbFloatCon_, halfChunkSize_ * dkAligned_);
            PipeBarrier<PIPE_V>();
            fp32OutQueue_.EnQue<float>(kgLocal_);
            uint64_t kgBeginOffset = subOffset_ * dk_;
            DataCopyOutFp32(halfChunkSize_, dk_, dkAligned_, outKgGm_[kgBeginOffset]);  // stage1 out
        }
    }

    __aicore__ inline void VBetaCompute()
    {
        uint64_t vBeginOffset = subOffset_ * vRowStride_;
        DataCopyInBf16WithStride(subValidRows_, dv_, valueGm_[vBeginOffset], vRowStride_);
        valueLocal_ = fp32InQueue_.DeQue<bfloat16_t>();
        vBetaLocal_ = fp32OutQueue_.AllocTensor<float>();
        Cast(valueUbFloat_, valueLocal_, AscendC::RoundMode::CAST_NONE, subValidRows_ * dvAligned_);
        PipeBarrier<PIPE_V>();
        fp32InQueue_.FreeTensor(valueLocal_);
        if (subValidRows_ < halfChunkSize_) {
            Duplicate(valueUbFloat_[subValidRows_ * dvAligned_], static_cast<float>(0.0f),
                      (halfChunkSize_ - subValidRows_) * dvAligned_);
            PipeBarrier<PIPE_V>();
        }
        uint32_t betaShape[2] = {halfChunkSize_, 1};
        uint32_t vShape[2] = {halfChunkSize_, dvAligned_};
        Broadcast<float, BROADCAST_AXIS, 1>(vBetaLocal_, betaUbFloat_, vShape, betaShape);
        PipeBarrier<PIPE_V>();
        Mul(vBetaLocal_, valueUbFloat_, vBetaLocal_, halfChunkSize_ * dvAligned_);
        PipeBarrier<PIPE_V>();
        fp32OutQueue_.EnQue<float>(vBetaLocal_);
        DataCopyOutFp32(halfChunkSize_, dv_, dvAligned_, vBetaWsGm_[subOffset_ * dv_]);
    }

    __aicore__ inline void QPrimeCompute()
    {
        qPrimeLocal_ = fp32OutQueue_.AllocTensor<float>();
        // query * scale
        if (gOptional_) {
            Muls(qUbFloat_, qUbFloatCon_, scale_, halfChunkSize_ * dkAligned_);
            PipeBarrier<PIPE_V>();
            uint32_t gCumExpShape[2] = {halfChunkSize_, 1};
            uint32_t qShape[2] = {halfChunkSize_, dkAligned_};
            Broadcast<float, BROADCAST_AXIS, 1>(gCumExpBroadUbFloat_, gCumExpUbFloat_[subOffset_],
                                                qShape, gCumExpShape);
            PipeBarrier<PIPE_V>();
            // query * scale * g_cum_exp[:, None]       # (C, Dk)
            Mul(qPrimeLocal_, qUbFloat_, gCumExpBroadUbFloat_, halfChunkSize_ * dkAligned_);
            gOutQueue_.FreeTensor(gCumExpUbFloat_);
        } else {
            Muls(qPrimeLocal_, qUbFloatCon_, scale_, halfChunkSize_ * dkAligned_);
        }
        PipeBarrier<PIPE_V>();
        fp32OutQueue_.EnQue<float>(qPrimeLocal_);
        uint64_t qgBeginOffset = subOffset_ * dk_;
        DataCopyOutFp32(halfChunkSize_, dk_, dkAligned_, outQPrimeGm_[qgBeginOffset]);  // stage1 out
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void DataCopyInFp32(uint64_t len, GlobalTensor<float> y)
    {
        DataCopyPadExtParams<float> padParams = {false, static_cast<uint8_t>(0), static_cast<uint8_t>(0),
                                                 static_cast<float>(0)};
        DataCopyExtParams kkParams{static_cast<uint16_t>(1), static_cast<uint32_t>(len * sizeof(float)), 0, 0, 0};
        fp32InLocal_ = fp32InQueue_.AllocTensor<float>();
        DataCopyPad(fp32InLocal_, y, kkParams, padParams);
        fp32InQueue_.EnQue<float>(fp32InLocal_);
    }

    __aicore__ inline void BetaCopyInWithStride()
    {
        uint64_t betaBeginOffset = subOffset_ * nv_;
        DataCopyInBf16WithStride(subValidRows_, 1, betaGm_[betaBeginOffset], nv_);
        betaLocal_ = fp32InQueue_.DeQue<bfloat16_t>();
        if (subValidRows_ < halfChunkSize_) {
            Duplicate(betaUbBfloat16_, bfloat16_t(0.0f), halfChunkSize_);
            PipeBarrier<PIPE_V>();
        }
        Gather(betaUbBfloat16_, betaLocal_, gatherOffsetBf16_, static_cast<uint32_t>(0), subValidRows_);
        PipeBarrier<PIPE_V>();

        Cast(betaUbFloat_, betaUbBfloat16_, AscendC::RoundMode::CAST_NONE, halfChunkSize_);
        PipeBarrier<PIPE_V>();
        fp32InQueue_.FreeTensor(betaLocal_);
    }

    __aicore__ inline void GCopyInWithStride()
    {
        DataCopyInFp32WithStride(validLen_, 1, gGm_, nv_);
        gLocal_ = fp32InQueue_.DeQue<float>();
        if (validLen_ < chunkSize_) {
            Duplicate(gCumUbFloat_, 0.0f, chunkSize_);
            PipeBarrier<PIPE_V>();
        }
        Gather(gCumUbFloat_, gLocal_, gatherOffsetFp32_, static_cast<uint32_t>(0), validLen_);
        PipeBarrier<PIPE_V>();

        fp32InQueue_.FreeTensor(gLocal_);
    }

    __aicore__ inline void DataCopyInFp32WithStride(uint64_t rows,  // 要搬的行数
                                                    uint64_t cols,  // 每行的元素数
                                                    const GlobalTensor<float> src,
                                                    uint64_t srcRowStride) // GM上相邻行的间距(元素数)
    {
        DataCopyPadExtParams<float> padParams = {false, static_cast<uint8_t>(0), static_cast<uint8_t>(0),
                                                 static_cast<float>(0)};
        uint32_t srcGap = (srcRowStride - cols) * sizeof(float);
        DataCopyExtParams params{static_cast<uint16_t>(rows),
                                 static_cast<uint32_t>(cols * sizeof(float)),
                                 static_cast<uint32_t>(srcGap), 0, 0};
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
        SyncMte3ToMte2();
        fp32OutQueue_.FreeTensor(fp32OutLocal_);
    }

    __aicore__ inline void DataCopyOutG(uint64_t length)
    {
        gCumExpUbFloat_ = gOutQueue_.DeQue<float>();
        if (subBlockIdx_ == 0) {
            DataCopyExtParams params{static_cast<uint16_t>(1),
                                     static_cast<uint32_t>(length * sizeof(float)),
                                     0, 0, 0};
            DataCopyPad(outGCumExpGm_, gCumExpUbFloat_, params);  // stage1 out
        }
    }

    __aicore__ inline void AICProcess(GlobalTensor<float> x, GlobalTensor<float> y, GlobalTensor<float> z,
                                      const MatmulShapeParams &shape, bool transB = false,
                                      bool enableAtomicAdd = false)
    {
        mmFp32.SetOrgShape(shape.m, shape.n, shape.k);
        mmFp32.SetSingleShape(shape.sm, shape.sn, shape.sk);
        mmFp32.SetTensorA(x);
        mmFp32.SetTensorB(y, transB);
        mmFp32.IterateAll(z, enableAtomicAdd ? 1 : 0);
        mmFp32.End();
    }

    __aicore__ inline uint64_t MatrixOffset(uint64_t row, uint64_t col)
    {
        return row * chunkSize_ + col;
    }

    __aicore__ inline uint32_t AlignUp(uint32_t value, uint32_t align)
    {
        return (value + align - 1) / align * align;
    }

    __aicore__ inline void CopyGmMatrixToL1A(GlobalTensor<float> gm, LocalTensor<float> l1,
                                             uint32_t rows, uint32_t cols, uint32_t stride)
    {
        Nd2NzParams copyParams;
        copyParams.ndNum = 1;
        copyParams.nValue = rows;
        copyParams.dValue = cols;
        copyParams.srcNdMatrixStride = 0;
        copyParams.srcDValue = stride;
        copyParams.dstNzC0Stride = AlignUp(rows, BLOCK_CUBE);
        copyParams.dstNzNStride = 1;
        copyParams.dstNzMatrixStride = 0;
        DataCopy(l1, gm, copyParams);
    }

    __aicore__ inline void CopyGmMatrixToL1B(GlobalTensor<float> gm, LocalTensor<float> l1,
                                             uint32_t rows, uint32_t cols, uint32_t stride)
    {
        Nd2NzParams copyParams;
        copyParams.ndNum = rows / BLOCK_CUBE;
        copyParams.nValue = BLOCK_CUBE;
        copyParams.dValue = cols;
        copyParams.srcNdMatrixStride = BLOCK_CUBE * stride;
        copyParams.srcDValue = stride;
        copyParams.dstNzC0Stride = BLOCK_CUBE;
        copyParams.dstNzNStride = 1;
        copyParams.dstNzMatrixStride = AlignUp(cols, RAW_FLOAT_C0) * BLOCK_CUBE;
        DataCopy(l1, gm, copyParams);
    }

    __aicore__ inline void CopyL1MatrixToL0A(LocalTensor<float> l1, uint32_t mSize, uint32_t kSize)
    {
        LoadData3DParamsV2<float> loadParams;
        loadParams.l1H = mSize / BLOCK_CUBE;
        loadParams.l1W = BLOCK_CUBE;
        loadParams.padList[0] = 0;
        loadParams.padList[1] = 0;
        loadParams.padList[2] = 0;
        loadParams.padList[3] = 255;
        loadParams.mExtension = mSize;
        loadParams.kExtension = kSize;
        loadParams.mStartPt = 0;
        loadParams.kStartPt = 0;
        loadParams.strideW = 1;
        loadParams.strideH = 1;
        loadParams.filterW = 1;
        loadParams.filterSizeW = false;
        loadParams.filterH = 1;
        loadParams.filterSizeH = false;
        loadParams.dilationFilterW = 1;
        loadParams.dilationFilterH = 1;
        loadParams.enTranspose = 0;
        loadParams.fMatrixCtrl = 0;
        loadParams.channelSize = kSize;
        LoadData<float, RAW_LOAD3DV2_CONFIG>(aicL0A_, l1, loadParams);
    }

    __aicore__ inline void CopyL1MatrixToL0B(LocalTensor<float> l1, uint32_t kSize, uint32_t nSize)
    {
        LoadData3DParamsV2<float> loadParams;
        loadParams.l1H = kSize / BLOCK_CUBE;
        loadParams.l1W = BLOCK_CUBE;
        loadParams.padList[0] = 0;
        loadParams.padList[1] = 0;
        loadParams.padList[2] = 0;
        loadParams.padList[3] = 255;
        loadParams.mExtension = kSize;
        loadParams.kExtension = nSize;
        loadParams.mStartPt = 0;
        loadParams.kStartPt = 0;
        loadParams.strideW = 1;
        loadParams.strideH = 1;
        loadParams.filterW = 1;
        loadParams.filterSizeW = false;
        loadParams.filterH = 1;
        loadParams.filterSizeH = false;
        loadParams.dilationFilterW = 1;
        loadParams.dilationFilterH = 1;
        loadParams.enTranspose = 1;
        loadParams.fMatrixCtrl = 0;
        loadParams.channelSize = nSize;
        LoadData<float, RAW_LOAD3DV2_CONFIG>(aicL0B_, l1, loadParams);
    }

    __aicore__ inline void CopyL0CMatrixToGm(GlobalTensor<float> gm, uint32_t mSize, uint32_t nSize,
                                             uint32_t dstStride, uint8_t unitFlag = RAW_MMAD_SINGLE_UNIT_FLAG)
    {
        FixpipeParamsV220 fixParams;
        fixParams.mSize = mSize;
        fixParams.nSize = nSize;
        fixParams.srcStride = AlignUp(mSize, BLOCK_CUBE);
        fixParams.dstStride = dstStride;
        fixParams.ndNum = 1;
        fixParams.srcNdStride = 0;
        fixParams.dstNdStride = 0;
        fixParams.reluEn = false;
        fixParams.quantPre = QuantMode_t::NoQuant;
        fixParams.unitFlag = unitFlag;
        Fixpipe<float, float, CFG_ROW_MAJOR>(gm, aicL0C_, fixParams);
    }

    __aicore__ inline void CopyL0CMatrixToGm(GlobalTensor<float> gm, uint32_t blockSize)
    {
        CopyL0CMatrixToGm(gm, blockSize, blockSize, chunkSize_);
    }

    __aicore__ inline void AtomicAddL0CMatrixToGm(GlobalTensor<float> gm, uint32_t mSize, uint32_t nSize,
                                                  uint32_t dstStride, uint8_t unitFlag = RAW_MMAD_SINGLE_UNIT_FLAG)
    {
        SetAtomicAdd<float>();
        CopyL0CMatrixToGm(gm, mSize, nSize, dstStride, unitFlag);
        SyncFixToM();
        SetAtomicNone();
    }

    __aicore__ inline void AtomicAddL0CMatrixToGm(GlobalTensor<float> gm, uint32_t blockSize)
    {
        AtomicAddL0CMatrixToGm(gm, blockSize, blockSize, chunkSize_);
    }

    __aicore__ inline MmadParams MakeRawMmadParams(uint32_t mSize, uint32_t nSize, uint32_t kSize,
                                                   bool initC = true, uint8_t unitFlag = RAW_MMAD_SINGLE_UNIT_FLAG)
    {
        MmadParams params(static_cast<uint16_t>(mSize), static_cast<uint16_t>(nSize),
                          static_cast<uint16_t>(kSize), unitFlag, false, initC);
        params.kDirectionAlign = false;
#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 3510) || (__NPU_ARCH__ == 5102))
        params.disableGemv = true;
#endif
        return params;
    }

    __aicore__ inline MmadParams MakeRawMmadParams(uint32_t blockSize, bool initC = true)
    {
        return MakeRawMmadParams(blockSize, blockSize, blockSize, initC);
    }

    __aicore__ inline InitConstValueParams<float> MakeRawL0FillParams(uint32_t blockSize, float value)
    {
        uint32_t alignedElems = AlignUp(blockSize, BLOCK_CUBE) * AlignUp(blockSize, RAW_FLOAT_C0);
        return InitConstValueParams<float>(
            1, static_cast<uint16_t>(alignedElems * sizeof(float) / RAW_FRACTAL_BYTES), 0, value);
    }

    __aicore__ inline void WriteIdentityMatrixToGm(GlobalTensor<float> gm, uint32_t blockSize)
    {
        for (uint32_t row = 0; row < blockSize; ++row) {
            for (uint32_t col = 0; col < blockSize; ++col) {
                gm.SetValue(row * chunkSize_ + col, row == col ? static_cast<float>(1.0f) : static_cast<float>(0.0f));
            }
        }
        SyncSToMte2();
        FlushBlockCacheLines(gm, blockSize);
    }

    __aicore__ inline void RawMmadInvAdd16(uint64_t invOffset, GlobalTensor<float> powerGm)
    {
        EnsureAicLocalBuffers();
        if constexpr (RAW_MMAD_DEBUG_STEP == 4) {
            return;
        }
        if constexpr (RAW_MMAD_DEBUG_STEP == 5) {
            for (uint32_t row = 0; row < INVERSE_SHAPE; ++row) {
                for (uint32_t col = 0; col < INVERSE_SHAPE; ++col) {
                    INVWsGm_.SetValue(invOffset + row * chunkSize_ + col, static_cast<float>(0.0f));
                }
            }
            SyncSToMte2();
            FlushInvBlockCacheLines(invOffset, INVERSE_SHAPE);
            return;
        }
        if constexpr (RAW_MMAD_DEBUG_STEP == 6) {
            if constexpr (!RAW_MMAD_DEBUG_MATMUL_READBACK) {
                (void)powerGm;
            }
            SyncMToMte1();
            InitConstValueParams<float> fillParams = MakeRawL0FillParams(INVERSE_SHAPE, static_cast<float>(1.0f));
            InitConstValue(aicL0A_, fillParams);
            InitConstValue(aicL0B_, fillParams);
            SyncMte1ToM();
            PipeBarrier<PIPE_M>();
            Mmad(aicL0C_, aicL0A_, aicL0B_, MakeRawMmadParams(INVERSE_SHAPE));
            PipeBarrier<PIPE_M>();
            SyncMToFix();
            if constexpr (RAW_MMAD_DEBUG_WRITE_QK) {
                CopyL0CMatrixToGm(outQkGm_, INVERSE_SHAPE);
            } else if constexpr (RAW_MMAD_DEBUG_TMP_READBACK) {
                CopyL0CMatrixToGm(tmpWsGm_[invOffset], INVERSE_SHAPE);
            } else {
                CopyL0CMatrixToGm(INVWsGm_[invOffset], INVERSE_SHAPE);
            }
            SyncFixToM();
            SyncFixToMte2();
            SyncFixToS();
            if constexpr (RAW_MMAD_DEBUG_WRITE_QK) {
                FlushBlockCacheLines(outQkGm_, INVERSE_SHAPE);
            } else {
                if constexpr (RAW_MMAD_DEBUG_TMP_READBACK) {
                    FlushBlockCacheLines(tmpWsGm_[invOffset], INVERSE_SHAPE);
                } else {
                    FlushInvBlockCacheLines(invOffset, INVERSE_SHAPE);
                }
            }
            if constexpr (RAW_MMAD_DEBUG_MATMUL_READBACK) {
                MatmulShapeParams shape = {chunkSize_, chunkSize_, chunkSize_,
                                           INVERSE_SHAPE, INVERSE_SHAPE, INVERSE_SHAPE};
                if constexpr (RAW_MMAD_DEBUG_WRITE_QK) {
                    AICProcessWith(outQkGm_, powerGm,
                                   RAW_MMAD_DEBUG_READBACK_TO_QK ? outQkGm_ : INVWsGm_[invOffset], shape,
                                   mmFp32C16);
                } else if constexpr (RAW_MMAD_DEBUG_TMP_READBACK) {
                    if constexpr (RAW_MMAD_DEBUG_SELF_READBACK) {
                        AICProcessWith(tmpWsGm_[invOffset], tmpWsGm_[invOffset],
                                       RAW_MMAD_DEBUG_READBACK_TO_QK ? outQkGm_ : INVWsGm_[invOffset], shape,
                                       mmFp32C16);
                    } else {
                        AICProcessWith(tmpWsGm_[invOffset], powerGm,
                                       RAW_MMAD_DEBUG_READBACK_TO_QK ? outQkGm_ : INVWsGm_[invOffset], shape,
                                       mmFp32C16);
                    }
                } else {
                    AICProcessWith(INVWsGm_[invOffset], powerGm,
                                   RAW_MMAD_DEBUG_READBACK_TO_QK ? outQkGm_ : INVWsGm_[invOffset], shape,
                                   mmFp32C16);
                }
                SyncFixToMte2();
            }
            return;
        }
        if constexpr (RAW_MMAD_USE_MATMULIMPL_FALLBACK) {
            MatmulShapeParams shape = {chunkSize_, chunkSize_, chunkSize_,
                                       INVERSE_SHAPE, INVERSE_SHAPE, INVERSE_SHAPE};
            AICProcess(INVWsGm_[invOffset], powerGm, INVWsGm_[invOffset], shape, false,
                       !RAW_MMAD_DEBUG_OVERWRITE_PRODUCT);
            SyncFixToMte2();
            return;
        }

        PrepareRawMmadState();
        SyncFixToM();
        SyncMToMte1();
        CopyGmMatrixToL1A(INVWsGm_[invOffset], aicL1A_, INVERSE_SHAPE, INVERSE_SHAPE, chunkSize_);
        CopyGmMatrixToL1B(tmpWsGm_[invOffset], aicL1B_, INVERSE_SHAPE, INVERSE_SHAPE, chunkSize_);
        SyncMte2ToMte1();
        if constexpr (RAW_MMAD_DEBUG_STEP == 1) {
            return;
        }

        CopyL1MatrixToL0A(aicL1A_, INVERSE_SHAPE, INVERSE_SHAPE);
        CopyL1MatrixToL0B(aicL1B_, INVERSE_SHAPE, INVERSE_SHAPE);
        SyncMte1ToM();
        SyncFixToM();
        if constexpr (RAW_MMAD_DEBUG_STEP == 2) {
            return;
        }

        PipeBarrier<PIPE_M>();
        SetHF32Mode(HF32Mode::DISABLE);
        Mmad(aicL0C_, aicL0A_, aicL0B_,
             MakeRawMmadParams(INVERSE_SHAPE, INVERSE_SHAPE, INVERSE_SHAPE,
                               true, RAW_MMAD_NON_FINAL_UNIT_FLAG));
        PipeBarrier<PIPE_M>();

        SyncMToMte1();
        CopyGmMatrixToL1A(INVWsGm_[invOffset], aicL1A_, INVERSE_SHAPE, INVERSE_SHAPE, chunkSize_);
        CopyGmMatrixToL1B(powerGm, aicL1B_, INVERSE_SHAPE, INVERSE_SHAPE, chunkSize_);
        SyncMte2ToMte1();
        CopyL1MatrixToL0A(aicL1A_, INVERSE_SHAPE, INVERSE_SHAPE);
        CopyL1MatrixToL0B(aicL1B_, INVERSE_SHAPE, INVERSE_SHAPE);
        SyncMte1ToM();

        PipeBarrier<PIPE_M>();
        Mmad(aicL0C_, aicL0A_, aicL0B_,
             MakeRawMmadParams(INVERSE_SHAPE, INVERSE_SHAPE, INVERSE_SHAPE,
                               false, RAW_MMAD_FINAL_UNIT_FLAG));
        PipeBarrier<PIPE_M>();

        SyncMToFix();
        if constexpr (RAW_MMAD_DEBUG_STEP == 3) {
            return;
        }
        if constexpr (RAW_MMAD_DEBUG_SCALAR_ADD_PRODUCT) {
            CopyL0CMatrixToGm(tmpWsGm_[invOffset], INVERSE_SHAPE);
            SyncFixToM();
            SyncFixToMte2();
            SyncFixToS();
            FlushBlockCacheLines(tmpWsGm_[invOffset], INVERSE_SHAPE);
            ScalarAddTmpToInv16(invOffset);
            return;
        } else if constexpr (RAW_MMAD_DEBUG_OVERWRITE_PRODUCT) {
            CopyL0CMatrixToGm(INVWsGm_[invOffset], INVERSE_SHAPE);
        } else {
            CopyL0CMatrixToGm(INVWsGm_[invOffset], INVERSE_SHAPE, INVERSE_SHAPE,
                              chunkSize_, RAW_MMAD_FINAL_UNIT_FLAG);
        }
        SyncRawInvAddStore(invOffset);
    }

    __aicore__ inline void RawMmadInvMul16(uint64_t invOffset, GlobalTensor<float> powerPlusIdentityGm)
    {
        EnsureAicLocalBuffers();
        PrepareRawMmadState();
        SyncFixToM();
        SyncMToMte1();

        CopyGmMatrixToL1A(INVWsGm_[invOffset], aicL1A_, INVERSE_SHAPE, INVERSE_SHAPE, chunkSize_);
        CopyGmMatrixToL1B(powerPlusIdentityGm, aicL1B_, INVERSE_SHAPE, INVERSE_SHAPE, chunkSize_);
        SyncMte2ToMte1();

        CopyL1MatrixToL0A(aicL1A_, INVERSE_SHAPE, INVERSE_SHAPE);
        CopyL1MatrixToL0B(aicL1B_, INVERSE_SHAPE, INVERSE_SHAPE);
        SyncMte1ToM();
        SyncFixToM();

        PipeBarrier<PIPE_M>();
        SetHF32Mode(HF32Mode::DISABLE);
        Mmad(aicL0C_, aicL0A_, aicL0B_, MakeRawMmadParams(INVERSE_SHAPE));
        PipeBarrier<PIPE_M>();

        SyncMToFix();
        CopyL0CMatrixToGm(INVWsGm_[invOffset], INVERSE_SHAPE);
        SyncRawInvAddStore(invOffset);
    }

    __aicore__ inline void RawMmadMatrix(GlobalTensor<float> x, GlobalTensor<float> y, GlobalTensor<float> z,
                                         uint32_t mSize, uint32_t nSize, uint32_t kSize,
                                         uint32_t xStride, uint32_t yStride, uint32_t zStride,
                                         bool enableAtomicAdd = false)
    {
        EnsureAicLocalBuffers();
        PrepareRawMmadState();
        SyncFixToM();
        SyncMToMte1();

        CopyGmMatrixToL1A(x, aicL1A_, mSize, kSize, xStride);
        CopyGmMatrixToL1B(y, aicL1B_, kSize, nSize, yStride);
        SyncMte2ToMte1();

        CopyL1MatrixToL0A(aicL1A_, mSize, kSize);
        CopyL1MatrixToL0B(aicL1B_, kSize, nSize);
        SyncMte1ToM();
        SyncFixToM();

        PipeBarrier<PIPE_M>();
        SetHF32Mode(HF32Mode::DISABLE);
        Mmad(aicL0C_, aicL0A_, aicL0B_, MakeRawMmadParams(mSize, nSize, kSize));
        PipeBarrier<PIPE_M>();

        SyncMToFix();
        if (enableAtomicAdd) {
            AtomicAddL0CMatrixToGm(z, mSize, nSize, zStride);
        } else {
            CopyL0CMatrixToGm(z, mSize, nSize, zStride);
        }
        SyncFixToM();
        SyncFixToMte2();
        SyncFixToS();
        FlushMatrixCacheLines(z, mSize, nSize, zStride);
    }

    __aicore__ inline void RawMmadAccumulateL0C(GlobalTensor<float> x, GlobalTensor<float> y,
                                                uint32_t mSize, uint32_t nSize, uint32_t kSize,
                                                uint32_t xStride, uint32_t yStride, bool initC, bool isLastK,
                                                bool enableUnitFlag = true)
    {
        EnsureAicLocalBuffers();
        PrepareRawMmadState();
        SyncFixToM();
        SyncMToMte1();

        CopyGmMatrixToL1A(x, aicL1A_, mSize, kSize, xStride);
        CopyGmMatrixToL1B(y, aicL1B_, kSize, nSize, yStride);
        SyncMte2ToMte1();

        CopyL1MatrixToL0A(aicL1A_, mSize, kSize);
        CopyL1MatrixToL0B(aicL1B_, kSize, nSize);
        SyncMte1ToM();

        PipeBarrier<PIPE_M>();
        SetHF32Mode(HF32Mode::DISABLE);
        uint8_t unitFlag = RAW_MMAD_SINGLE_UNIT_FLAG;
        if (enableUnitFlag) {
            unitFlag = isLastK ? RAW_MMAD_FINAL_UNIT_FLAG : RAW_MMAD_NON_FINAL_UNIT_FLAG;
        }
        Mmad(aicL0C_, aicL0A_, aicL0B_,
             MakeRawMmadParams(mSize, nSize, kSize, initC, unitFlag));
        PipeBarrier<PIPE_M>();
    }

    __aicore__ inline void StoreAccumulatedL0C(GlobalTensor<float> z, uint32_t mSize, uint32_t nSize,
                                               uint32_t zStride, bool enableUnitFlag = true)
    {
        SyncMToFix();
        uint8_t unitFlag = enableUnitFlag ? RAW_MMAD_FINAL_UNIT_FLAG : RAW_MMAD_SINGLE_UNIT_FLAG;
        CopyL0CMatrixToGm(z, mSize, nSize, zStride, unitFlag);
        SyncFixToM();
        SyncFixToMte2();
        SyncFixToS();
        FlushMatrixCacheLines(z, mSize, nSize, zStride);
    }

    __aicore__ inline void RawMmadSquare(GlobalTensor<float> x, GlobalTensor<float> y, GlobalTensor<float> z,
                                         uint32_t blockSize, bool enableAtomicAdd = false)
    {
        RawMmadMatrix(x, y, z, blockSize, blockSize, blockSize,
                      chunkSize_, chunkSize_, chunkSize_, enableAtomicAdd);
    }

    __aicore__ inline void RawMmadBlocked32(GlobalTensor<float> x, GlobalTensor<float> y, GlobalTensor<float> z)
    {
        for (uint32_t row = 0; row < C64_BLOCK32; row += INVERSE_SHAPE) {
            for (uint32_t col = 0; col < C64_BLOCK32; col += INVERSE_SHAPE) {
                bool initC = true;
                for (uint32_t kBase = 0; kBase < C64_BLOCK32; kBase += INVERSE_SHAPE) {
                    bool isLastK = (kBase + INVERSE_SHAPE >= C64_BLOCK32);
                    RawMmadAccumulateL0C(x[MatrixOffset(row, kBase)], y[MatrixOffset(kBase, col)],
                                         INVERSE_SHAPE, INVERSE_SHAPE, INVERSE_SHAPE,
                                         chunkSize_, chunkSize_, initC, isLastK);
                    initC = false;
                }
                StoreAccumulatedL0C(z[MatrixOffset(row, col)], INVERSE_SHAPE, INVERSE_SHAPE, chunkSize_);
            }
        }
    }

    __aicore__ inline void RawMmad64ByN(GlobalTensor<float> x, GlobalTensor<float> y, GlobalTensor<float> z,
                                        uint32_t nSize)
    {
        if (nSize <= chunkSize_) {
            for (uint32_t row = 0; row < chunkSize_; row += INVERSE_SHAPE) {
                bool initC = true;
                for (uint32_t kBase = 0; kBase < chunkSize_; kBase += INVERSE_SHAPE) {
                    bool isLastK = (kBase + INVERSE_SHAPE >= chunkSize_);
                    RawMmadAccumulateL0C(x[MatrixOffset(row, kBase)], y[kBase * nSize],
                                         INVERSE_SHAPE, nSize, INVERSE_SHAPE,
                                         chunkSize_, nSize, initC, isLastK, false);
                    initC = false;
                }
                StoreAccumulatedL0C(z[row * nSize], INVERSE_SHAPE, nSize, nSize, false);
            }
            return;
        }
        uint32_t tileCount = 0;
        for (uint32_t row = 0; row < chunkSize_; row += INVERSE_SHAPE) {
            for (uint32_t col = 0; col < nSize; col += RAW_FINAL_N_TILE) {
                uint32_t nTile = AscendC::Std::min(static_cast<uint32_t>(RAW_FINAL_N_TILE), nSize - col);
                bool initC = true;
                for (uint32_t kBase = 0; kBase < chunkSize_; kBase += INVERSE_SHAPE) {
                    bool isLastK = (kBase + INVERSE_SHAPE >= chunkSize_);
                    RawMmadAccumulateL0C(x[MatrixOffset(row, kBase)], y[kBase * nSize + col],
                                         INVERSE_SHAPE, nTile, INVERSE_SHAPE,
                                         chunkSize_, nSize, initC, isLastK, false);
                    initC = false;
                }
                StoreAccumulatedL0C(z[row * nSize + col], INVERSE_SHAPE, nTile, nSize, false);
                if constexpr (RAW_MMAD_DEBUG_RAW_FINAL_TILE_LIMIT != 0) {
                    ++tileCount;
                    if (tileCount >= RAW_MMAD_DEBUG_RAW_FINAL_TILE_LIMIT) {
                        return;
                    }
                }
            }
        }
    }

    __aicore__ inline void ClearMatrixGm(GlobalTensor<float> gm, uint32_t rows, uint32_t cols, uint32_t stride)
    {
        for (uint32_t row = 0; row < rows; ++row) {
            for (uint32_t col = 0; col < cols; ++col) {
                gm.SetValue(row * stride + col, static_cast<float>(0.0f));
            }
        }
        SyncSToMte2();
        FlushMatrixCacheLines(gm, rows, cols, stride);
    }

    template <bool USE_EXTRA_FALLBACK>
    __aicore__ inline void FinalMmad64ByN(GlobalTensor<float> x, GlobalTensor<float> y, GlobalTensor<float> z,
                                          uint32_t nSize)
    {
        if constexpr (RAW_MMAD_USE_MATMULIMPL_FINAL_FALLBACK || USE_EXTRA_FALLBACK) {
            AICProcessWith(x, y, z, {chunkSize_, nSize, chunkSize_, chunkSize_, nSize, chunkSize_}, mmFp32C64);
            SyncFixToMte2();
        } else {
            if constexpr (!RAW_MMAD_DEBUG_SKIP_RAW_FINAL) {
                if constexpr (RAW_MMAD_DEBUG_RAW_FINAL_TO_SAFE_SCRATCH_THEN_FALLBACK) {
                    RawMmad64ByN(x, y, rawFinalScratchGm_, nSize);
                } else if constexpr (RAW_MMAD_DEBUG_RAW_FINAL_TO_ATTN_THEN_FALLBACK) {
                    RawMmad64ByN(x, y, attnWsGm_, nSize);
                } else if constexpr (RAW_MMAD_DEBUG_RAW_FINAL_TO_TMP_THEN_FALLBACK) {
                    if constexpr (RAW_MMAD_DEBUG_RAW_FINAL_TO_INV_THEN_FALLBACK) {
                        RawMmad64ByN(x, y, INVWsGm_, nSize);
                    } else {
                        RawMmad64ByN(x, y, tmpWsGm_, nSize);
                    }
                } else {
                    RawMmad64ByN(x, y, z, nSize);
                }
            }
            if constexpr (RAW_MMAD_DEBUG_RAW_FINAL_TO_SAFE_SCRATCH_THEN_FALLBACK ||
                          RAW_MMAD_DEBUG_RAW_FINAL_TO_ATTN_THEN_FALLBACK ||
                          RAW_MMAD_DEBUG_RAW_FINAL_TO_TMP_THEN_FALLBACK ||
                          RAW_MMAD_DEBUG_OVERWRITE_RAW_FINAL_WITH_MATMULIMPL) {
                RestoreRawMmadState();
                AICProcessWith(x, y, z, {chunkSize_, nSize, chunkSize_, chunkSize_, nSize, chunkSize_}, mmFp32C64);
                SyncFixToMte2();
            }
            RestoreRawMmadState();
        }
    }

    __aicore__ inline void ScalarAddTmpToInv16(uint64_t invOffset)
    {
        for (uint32_t row = 0; row < INVERSE_SHAPE; ++row) {
            for (uint32_t col = 0; col < INVERSE_SHAPE; ++col) {
                uint64_t offset = invOffset + row * chunkSize_ + col;
                float value = INVWsGm_.GetValue(offset) + tmpWsGm_.GetValue(offset);
                INVWsGm_.SetValue(offset, value);
            }
        }
        SyncSToMte2();
        FlushInvBlockCacheLines(invOffset, INVERSE_SHAPE);
    }

    __aicore__ inline void FlushInvBlockCacheLines(uint64_t invOffset, uint32_t blockSize)
    {
        FlushBlockCacheLines(INVWsGm_[invOffset], blockSize);
    }

    __aicore__ inline void AddIdentityToPowerBlock(GlobalTensor<float> powerGm)
    {
        for (uint32_t row = 0; row < INVERSE_SHAPE; ++row) {
            powerGm.SetValue(row * chunkSize_ + row, static_cast<float>(1.0f));
        }
        SyncSToMte2();
        FlushBlockCacheLines(powerGm, INVERSE_SHAPE);
    }

    __aicore__ inline void SyncRawInvAddStore(uint64_t invOffset)
    {
        if constexpr (RAW_MMAD_INVADD_POST_STORE_SYNC_MODE == 0) {
            SyncFixToM();
            SyncFixToMte2();
            SyncFixToS();
            FlushInvBlockCacheLines(invOffset, INVERSE_SHAPE);
        } else if constexpr (RAW_MMAD_INVADD_POST_STORE_SYNC_MODE == 1) {
            SyncFixToM();
            SyncFixToMte2();
        } else {
            (void)invOffset;
            SyncFixToMte2();
        }
    }

    __aicore__ inline void FlushBlockCacheLines(GlobalTensor<float> gm, uint32_t blockSize)
    {
        FlushMatrixCacheLines(gm, blockSize, blockSize, chunkSize_);
    }

    __aicore__ inline void FlushMatrixCacheLines(GlobalTensor<float> gm, uint32_t rows, uint32_t cols,
                                                 uint32_t stride)
    {
        for (uint32_t row = 0; row < rows; ++row) {
            for (uint32_t col = 0; col < cols; col += RAW_CACHELINE_FLOATS) {
                Barrier();
                DataCacheCleanAndInvalid<float, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
                    gm[row * stride + col]);
            }
        }
        Barrier();
    }

    __aicore__ inline void DebugClearInvMatrix()
    {
        for (uint32_t row = 0; row < chunkSize_; ++row) {
            for (uint32_t col = 0; col < chunkSize_; ++col) {
                INVWsGm_.SetValue(row * chunkSize_ + col, static_cast<float>(0.0f));
            }
        }
        SyncSToMte2();
        FlushBlockCacheLines(INVWsGm_, chunkSize_);
    }

    __aicore__ inline void ClearQkMatrix()
    {
        for (uint32_t row = 0; row < validLen_; ++row) {
            for (uint32_t col = 0; col < validLen_; ++col) {
                outQkGm_.SetValue(row * validLen_ + col, static_cast<float>(0.0f));
            }
        }
        SyncSToMte2();
        FlushMatrixCacheLines(outQkGm_, validLen_, validLen_, validLen_);
    }

    __aicore__ inline void AICBlockProcess(GlobalTensor<float> x, GlobalTensor<float> y, GlobalTensor<float> z,
                                           uint64_t blockSize, bool enableAtomicAdd = false)
    {
        MatmulShapeParams shape = {chunkSize_, chunkSize_, chunkSize_, blockSize, blockSize, blockSize};
        StageOneMT &mm = (blockSize == INVERSE_SHAPE) ? mmFp32C16 : mmFp32C32;
        AICProcessWith(x, y, z, shape, mm, false, enableAtomicAdd);
    }

    __aicore__ inline void AICProcessWith(GlobalTensor<float> x, GlobalTensor<float> y, GlobalTensor<float> z,
                                          const MatmulShapeParams &shape, StageOneMT &mm,
                                          bool transB = false, bool enableAtomicAdd = false)
    {
        mm.SetOrgShape(shape.m, shape.n, shape.k);
        mm.SetSingleShape(shape.sm, shape.sn, shape.sk);
        mm.SetTensorA(x);
        mm.SetTensorB(y, transB);
        mm.IterateAll(z, enableAtomicAdd ? 1 : 0);
        mm.End();
    }

    __aicore__ inline void Neumann16InverseBlock(uint64_t base)
    {
        uint64_t offset = MatrixOffset(base, base);
        bool useFallback = RAW_MMAD_USE_MATMULIMPL_FALLBACK ||
                           (RAW_MMAD_DEBUG_BASE0_FALLBACK && base == 0) ||
                           (RAW_MMAD_DEBUG_BASE16_FALLBACK && base == INVERSE_SHAPE);
        if constexpr (RAW_MMAD_INVADD_USE_POWER_PLUS_IDENTITY) {
            if (!useFallback) {
                // Compute L2/L4/L8 before mutating each power block in-place to I + power.
                PowerMmadSquare(L_powWsGm_[offset], L_powWsGm_[offset], attnWsGm_[offset], INVERSE_SHAPE);
                PowerMmadSquare(attnWsGm_[offset], attnWsGm_[offset], L_powWsGm_[offset], INVERSE_SHAPE);
                PowerMmadSquare(L_powWsGm_[offset], L_powWsGm_[offset], tmpWsGm_[offset], INVERSE_SHAPE);

                AddIdentityToPowerBlock(attnWsGm_[offset]);
                RawMmadInvMul16(offset, attnWsGm_[offset]);
                AddIdentityToPowerBlock(L_powWsGm_[offset]);
                RawMmadInvMul16(offset, L_powWsGm_[offset]);
                AddIdentityToPowerBlock(tmpWsGm_[offset]);
                RawMmadInvMul16(offset, tmpWsGm_[offset]);
                SyncFixToMte2();
                return;
            }
        }
        if (!useFallback) {
            WriteIdentityMatrixToGm(tmpWsGm_[offset], INVERSE_SHAPE);
        }

        // Round 1: L² = L @ L, INV += INV @ L²
        PowerMmadSquare(L_powWsGm_[offset], L_powWsGm_[offset], attnWsGm_[offset], INVERSE_SHAPE);
        InvAdd16(offset, attnWsGm_[offset], useFallback);

        // Round 2: L⁴ = L² @ L², INV += INV @ L⁴
        PowerMmadSquare(attnWsGm_[offset], attnWsGm_[offset], L_powWsGm_[offset], INVERSE_SHAPE);
        InvAdd16(offset, L_powWsGm_[offset], useFallback);

        // Round 3: L⁸ = L⁴ @ L⁴, INV += INV @ L⁸
        PowerMmadSquare(L_powWsGm_[offset], L_powWsGm_[offset], attnWsGm_[offset], INVERSE_SHAPE);
        InvAdd16(offset, attnWsGm_[offset], useFallback);
        SyncFixToMte2();
    }

    __aicore__ inline void InvAdd16(uint64_t invOffset, GlobalTensor<float> powerGm, bool useFallback)
    {
        if (useFallback) {
            MatmulShapeParams shape = {chunkSize_, chunkSize_, chunkSize_,
                                       INVERSE_SHAPE, INVERSE_SHAPE, INVERSE_SHAPE};
            AICProcessWith(INVWsGm_[invOffset], powerGm, INVWsGm_[invOffset], shape, mmFp32C16, false,
                           !RAW_MMAD_DEBUG_OVERWRITE_PRODUCT);
            SyncFixToMte2();
            return;
        }
        RawMmadInvAdd16(invOffset, powerGm);
    }

    __aicore__ inline void PowerMmadSquare(GlobalTensor<float> x, GlobalTensor<float> y, GlobalTensor<float> z,
                                           uint32_t blockSize)
    {
        if constexpr (RAW_MMAD_USE_MATMULIMPL_POWER_FALLBACK) {
            AICBlockProcess(x, y, z, blockSize);
            SyncFixToMte2();
        } else {
            RawMmadSquare(x, y, z, blockSize);
        }
    }

    __aicore__ inline void ComposeMmadSquare(GlobalTensor<float> x, GlobalTensor<float> y, GlobalTensor<float> z,
                                             uint32_t blockSize)
    {
        if constexpr (RAW_MMAD_USE_MATMULIMPL_COMPOSE_FALLBACK) {
            AICBlockProcess(x, y, z, blockSize);
            SyncFixToMte2();
        } else if (blockSize == C64_BLOCK32) {
            RawMmadBlocked32(x, y, z);
        } else {
            RawMmadSquare(x, y, z, blockSize);
        }
    }

    __aicore__ inline void SyncFixToMte2()
    {
        int32_t eventID = static_cast<int32_t>(pipe_->FetchEventID(HardEvent::FIX_MTE2));
        SetFlag<HardEvent::FIX_MTE2>(eventID);
        WaitFlag<HardEvent::FIX_MTE2>(eventID);
    }

    __aicore__ inline void SyncFixToMte1()
    {
        int32_t eventID = static_cast<int32_t>(pipe_->FetchEventID(HardEvent::FIX_MTE1));
        SetFlag<HardEvent::FIX_MTE1>(eventID);
        WaitFlag<HardEvent::FIX_MTE1>(eventID);
    }

    __aicore__ inline void SyncVToMte3()
    {
        int32_t eventID = static_cast<int32_t>(pipe_->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventID);
        WaitFlag<HardEvent::V_MTE3>(eventID);
    }

    __aicore__ inline void SyncMte3ToMte2()
    {
        int32_t eventID = static_cast<int32_t>(pipe_->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventID);
        WaitFlag<HardEvent::MTE3_MTE2>(eventID);
    }

    __aicore__ inline void Compose32From16(uint64_t base)
    {
        uint64_t top = base;
        uint64_t bottom = base + INVERSE_SHAPE;
        uint64_t topDiag = MatrixOffset(top, top);
        uint64_t bottomDiag = MatrixOffset(bottom, bottom);
        uint64_t lowerLeft = MatrixOffset(bottom, top);

        // For X=[A,0;C,D], X^-1=[A^-1,0;-D^-1*C*A^-1,D^-1].
        // INV lower-left still stores -C, so lower = D^-1 * (-C) * A^-1.
        ComposeMmadSquare(INVWsGm_[lowerLeft], INVWsGm_[topDiag], tmpWsGm_[lowerLeft], INVERSE_SHAPE);
        ComposeMmadSquare(INVWsGm_[bottomDiag], tmpWsGm_[lowerLeft], INVWsGm_[lowerLeft], INVERSE_SHAPE);
    }

    __aicore__ inline void Compose64From32()
    {
        uint64_t topDiag = MatrixOffset(0, 0);
        uint64_t bottomDiag = MatrixOffset(C64_BLOCK32, C64_BLOCK32);
        uint64_t lowerLeft = MatrixOffset(C64_BLOCK32, 0);

        ComposeMmadSquare(INVWsGm_[lowerLeft], INVWsGm_[topDiag], tmpWsGm_[lowerLeft], C64_BLOCK32);
        ComposeMmadSquare(INVWsGm_[bottomDiag], tmpWsGm_[lowerLeft], INVWsGm_[lowerLeft], C64_BLOCK32);
    }

    __aicore__ inline void C64InverseBy16x16Neumann()
    {
        if (validLen_ > 0) {
            Neumann16InverseBlock(0);
        }
        if (validLen_ > INVERSE_SHAPE) {
            Neumann16InverseBlock(INVERSE_SHAPE);
            Compose32From16(0);
        }
        if (validLen_ > C64_BLOCK32) {
            Neumann16InverseBlock(C64_BLOCK32);
        }
        if (validLen_ > C64_BLOCK32 + INVERSE_SHAPE) {
            Neumann16InverseBlock(C64_BLOCK32 + INVERSE_SHAPE);
            Compose32From16(C64_BLOCK32);
        }

        if (validLen_ > C64_BLOCK32) {
            Compose64From32();
        }
    }

    __aicore__ inline void SyncMte2ToMte1()
    {
        int32_t eventID = static_cast<int32_t>(pipe_->FetchEventID(HardEvent::MTE2_MTE1));
        SetFlag<HardEvent::MTE2_MTE1>(eventID);
        WaitFlag<HardEvent::MTE2_MTE1>(eventID);
    }

    __aicore__ inline void SyncMte1ToM()
    {
        int32_t eventID = static_cast<int32_t>(pipe_->FetchEventID(HardEvent::MTE1_M));
        SetFlag<HardEvent::MTE1_M>(eventID);
        WaitFlag<HardEvent::MTE1_M>(eventID);
    }

    __aicore__ inline void SyncMToMte1()
    {
        int32_t eventID = static_cast<int32_t>(pipe_->FetchEventID(HardEvent::M_MTE1));
        SetFlag<HardEvent::M_MTE1>(eventID);
        WaitFlag<HardEvent::M_MTE1>(eventID);
    }

    __aicore__ inline void SyncMToFix()
    {
        int32_t eventID = static_cast<int32_t>(pipe_->FetchEventID(HardEvent::M_FIX));
        SetFlag<HardEvent::M_FIX>(eventID);
        WaitFlag<HardEvent::M_FIX>(eventID);
    }

    __aicore__ inline void SyncSToMte2()
    {
        int32_t eventID = static_cast<int32_t>(pipe_->FetchEventID(HardEvent::S_MTE2));
        SetFlag<HardEvent::S_MTE2>(eventID);
        WaitFlag<HardEvent::S_MTE2>(eventID);
    }

    __aicore__ inline void SyncFixToM()
    {
        int32_t eventID = static_cast<int32_t>(pipe_->FetchEventID(HardEvent::FIX_M));
        SetFlag<HardEvent::FIX_M>(eventID);
        WaitFlag<HardEvent::FIX_M>(eventID);
    }

    __aicore__ inline void SyncFixToS()
    {
        int32_t eventID = static_cast<int32_t>(pipe_->FetchEventID(HardEvent::FIX_S));
        SetFlag<HardEvent::FIX_S>(eventID);
        WaitFlag<HardEvent::FIX_S>(eventID);
    }

    __aicore__ inline void RestoreRawMmadState()
    {
        SetAtomicNone();
        SetNdParaImpl(0x1);
        SetHF32Mode(HF32Mode::DISABLE);
        SetHF32TransMode(HF32TransMode::NEAREST_EVEN);
        SetMMLayoutTransform(false);
    }

    __aicore__ inline void PrepareRawMmadState()
    {
        SetLoadDataPaddingValue<uint64_t>(0);
        SetNdParaImpl(0x1);
        SetHF32Mode(HF32Mode::DISABLE);
        SetHF32TransMode(HF32TransMode::NEAREST_EVEN);
        SetMMLayoutTransform(false);
    }

    TPipe *pipe_;
    StageOneMT &mmFp32;
    StageOneMT &mmFp32C16;
    StageOneMT &mmFp32C32;
    StageOneMT &mmFp32C64;
    const ChunkGatedDeltaRuleNeumannTilingData *tiling_;
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
    uint32_t validLen_;
    uint32_t subValidRows_;
    uint32_t coreNum_;
    float scale_;
    bool gOptional_;
    bool aicLocalBuffersReady_;

    // base GM pointers
    GlobalTensor<bfloat16_t> queryBaseGm_;
    GlobalTensor<bfloat16_t> keyBaseGm_;
    GlobalTensor<bfloat16_t> valueBaseGm_;
    GlobalTensor<bfloat16_t> betaBaseGm_;
    GlobalTensor<float> gBaseGm_;
    GlobalTensor<float> outGCumExpBaseGm_, outVInnerBaseGm_, outKgBaseGm_, outQkBaseGm_;
    GlobalTensor<float> outKCumdecayBaseGm_, outQPrimeBaseGm_;

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
    GlobalTensor<float> outKgGm_;
    GlobalTensor<float> outQkGm_;

    GlobalTensor<float> vBetaWsGm_;
    GlobalTensor<float> kkWsGm_;
    GlobalTensor<float> attnWsGm_;
    GlobalTensor<float> gBKWsGm_;
    GlobalTensor<float> queryContinousGm_;
    GlobalTensor<float> keyContinousGm_;
    GlobalTensor<float> querytmpGm_;
    GlobalTensor<float> stageOneMask_;
    GlobalTensor<float> L_powWsGm_;  // Neumann: L², L⁴, L⁸
    GlobalTensor<float> tmpWsGm_;    // Neumann: INV × L_pow intermediate
    GlobalTensor<float> INVWsGm_;    // Neumann: inverse matrix (I+L)^-1
    GlobalTensor<float> rawFinalScratchGm_;

    // UB queues
    TQue<QuePosition::VECIN, 1> fp32InQueue_;
    TQue<QuePosition::VECOUT, 1> fp32OutQueue_;
    TQue<QuePosition::VECOUT, 1> gOutQueue_;

    TBuf<TPosition::VECCALC> tmpBuff_;
    TBuf<TPosition::A1> aicL1ABuff_;
    TBuf<TPosition::B1> aicL1BBuff_;
    TBuf<TPosition::A2> aicL0ABuff_;
    TBuf<TPosition::B2> aicL0BBuff_;
    TBuf<TPosition::CO1> aicL0CBuff_;

    // UB tensors
    LocalTensor<bfloat16_t> betaUbBfloat16_;
    LocalTensor<float> betaUbFloat_;
    LocalTensor<float> valueUbFloat_;
    LocalTensor<float> attnUbFloat_;
    LocalTensor<float> inverseLocal_;
    LocalTensor<float> aicL1A_;
    LocalTensor<float> aicL1B_;
    LocalTensor<float> aicL0A_;
    LocalTensor<float> aicL0B_;
    LocalTensor<float> aicL0C_;
    LocalTensor<float> gCumUbFloat_;
    LocalTensor<float> gCumExpUbFloat_;
    LocalTensor<float> gCumExpBroadUbFloat_;
    LocalTensor<float> gBUbFloat_;
    LocalTensor<float> kUbFloat_;
    LocalTensor<float> qUbFloat_;
    LocalTensor<float> gBroadUbFloat_;
    LocalTensor<float> gTransBroadUbFloat_;
    LocalTensor<float> gEndBroadUbFloat_;
    LocalTensor<float> gammaUbFloat_;
    LocalTensor<float> qUbFloatCon_;
    LocalTensor<float> kUbFloatCon_;
    LocalTensor<uint32_t> gatherOffsetFp32_;
    LocalTensor<uint32_t> gatherOffsetBf16_;

    LocalTensor<bfloat16_t> betaLocal_;
    LocalTensor<bfloat16_t> valueLocal_;
    LocalTensor<bfloat16_t> kLocal_;
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
} // namespace ChunkGatedDeltaRuleNeumann
#endif // CHUNK_GATED_DELTA_RULE_NEUMANN_STAGE1_C64_1616_FP16_HALFLOCAL_H
