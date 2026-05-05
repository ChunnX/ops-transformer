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
 * \file chunk_gated_delta_rule_neumann_stage1_C64_3232_FP16_fulllocal.h
 * \brief
 */
#ifndef CHUNK_GATED_DELTA_RULE_NEUMANN_STAGE1_C64_3232_FP16_FULLLOCAL_H
#define CHUNK_GATED_DELTA_RULE_NEUMANN_STAGE1_C64_3232_FP16_FULLLOCAL_H

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
constexpr uint64_t INVERSE_SHAPE = 32;          // 32x32 Neumann basic block
constexpr uint64_t INVERSE_TAIL16_SHAPE = 16;   // tail shortcut for <=16 valid rows
constexpr uint64_t AIC_RAW_L1_WORK_BYTES = 16 * 1024;
constexpr uint64_t AIC_RAW_L1_A_BYTES = AIC_RAW_L1_WORK_BYTES;
constexpr uint64_t AIC_RAW_L1_B_BYTES = 16 * 1024;
constexpr uint64_t AIC_LOCAL_MM_WS_BYTES = 48 * 1024;
constexpr uint64_t AIC_LOW_L1_MATRIX_ELEMS = 64 * 64;
constexpr uint64_t AIC_LOW_L1_TILE_ELEMS = INVERSE_SHAPE * INVERSE_SHAPE;
constexpr uint64_t AIC_LOW_L1_BYTES = (AIC_LOW_L1_MATRIX_ELEMS * 3 + AIC_LOW_L1_TILE_ELEMS) * sizeof(half);
constexpr uint64_t AIC_RAW_L0_AB_BYTES = 32 * 1024;
constexpr uint64_t AIC_RAW_L0_AB_HALF_PINGPONG_BYTES = AIC_RAW_L0_AB_BYTES / 2;
constexpr uint64_t AIC_RAW_L0_C_BYTES = 64 * 1024;
constexpr uint32_t ALIGN_SIZE = 16;
constexpr uint32_t RAW_FLOAT_C0 = BLOCK_SIZE / sizeof(float);
constexpr uint32_t RAW_FRACTAL_BYTES = 512;
constexpr uint32_t RawCeilDivConstexpr(uint32_t value, uint32_t divisor)
{
    return (value + divisor - 1) / divisor;
}
constexpr uint32_t RawAlignUpConstexpr(uint32_t value, uint32_t align)
{
    return RawCeilDivConstexpr(value, align) * align;
}
constexpr uint32_t RAW_L0C_TILE_FLOATS =
    RawAlignUpConstexpr(static_cast<uint32_t>(INVERSE_TAIL16_SHAPE), BLOCK_CUBE) *
    RawAlignUpConstexpr(static_cast<uint32_t>(INVERSE_TAIL16_SHAPE), RAW_FLOAT_C0);
constexpr uint32_t FLOAT_CACHELINE_ELEMS = BLOCK_SIZE / sizeof(float);
static constexpr IsResetLoad3dConfig RAW_LOAD3DV2_CONFIG = {true, true};
constexpr uint8_t RAW_MMAD_NON_FINAL_UNIT_FLAG = 2;
constexpr uint8_t RAW_MMAD_FINAL_UNIT_FLAG = 3;
constexpr bool RAW_MMAD_CMATRIX_SOURCE_L0C = false;
constexpr bool RAW_MMAD_INIT_C = true;
constexpr bool RAW_MMAD_ACCUMULATE_C = false;
// Experimental L0C->half TSCM->L0 path. Keep disabled by default until the
// NZ/ZN L1->L0 layout is verified end-to-end.
constexpr bool FP16_L1_RESIDENT_INVERSE = false;
constexpr bool FP16_L1_RESIDENT_SCALAR_PROBE = false;
constexpr bool FP16_L1_ROUNDTRIP_THEN_RAW_PROBE = false;
constexpr bool FP16_L1_ROUNDTRIP_MMAD_WRITEBACK_PROBE = false;

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
                             StageOneMT &mmFp32C64, StageOneFp16OutMT &mmFp16Out,
                             StageOneFp16MT &mmFp16C16, StageOneFp16MT &mmFp16C32,
                             StageOneFp16L1GmOutMT &mmFp16L1Out, StageOneFp16L1MT &mmFp16L1C16,
                             StageOneFp16L1MT &mmFp16L1C32, StageOneGmGmLocalMT &, StageOneGmGmLocalMT &,
                             StageOneLocalMT &, StageOneLocalMT &, StageOneGmLocalGmMT &, StageOneGmLocalGmMT &)
        : Stage1(mmFp32, mmFp32C16, mmFp32C32, mmFp32C64, mmFp16Out, mmFp16C16, mmFp16C32,
                 mmFp16L1Out, mmFp16L1C16, mmFp16L1C32) {}
    __aicore__ inline Stage1(StageOneMT &mmFp32, StageOneMT &mmFp32C16, StageOneMT &mmFp32C32)
        : mmFp32(mmFp32), mmFp32C16(mmFp32C16), mmFp32C32(mmFp32C32),
          mmFp16L1C16_(nullptr), mmFp16L1C32_(nullptr) {}
    __aicore__ inline Stage1(StageOneMT &mmFp32, StageOneMT &mmFp32C16, StageOneMT &mmFp32C32,
                             StageOneMT &, StageOneFp16OutMT &, StageOneFp16MT &, StageOneFp16MT &,
                             StageOneFp16L1GmOutMT &, StageOneFp16L1MT &mmFp16L1C16,
                             StageOneFp16L1MT &mmFp16L1C32)
        : mmFp32(mmFp32), mmFp32C16(mmFp32C16), mmFp32C32(mmFp32C32),
          mmFp16L1C16_(&mmFp16L1C16), mmFp16L1C32_(&mmFp16L1C32) {}
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

    __aicore__ inline void InitAicLocalBuffers()
    {
        if constexpr (FP16_L1_RESIDENT_INVERSE) {
            mmFp16L1C16_->Init(&tiling_->matmulTilingFp16C16L1, pipe_);
            mmFp16L1C32_->Init(&tiling_->matmulTilingFp16C32L1, pipe_);
            pipe_->InitBuffer(tmpBuff_, UB_REST_BYTES);
            aicLocalMmWs_ = tmpBuff_.GetWithOffset<uint8_t>(
                static_cast<uint32_t>(AIC_LOCAL_MM_WS_BYTES), 0);
            aicLocalMmWs_.SetSize(AIC_LOCAL_MM_WS_BYTES);
            mmFp16L1C16_->SetLocalWorkspace(aicLocalMmWs_);
            mmFp16L1C32_->SetLocalWorkspace(aicLocalMmWs_);
        }
        pipe_->InitBuffer(aicL1ABuff_, AIC_RAW_L1_A_BYTES);
        pipe_->InitBuffer(aicL1BBuff_, AIC_RAW_L1_B_BYTES);
        pipe_->InitBuffer(aicHalfL1Buff_, AIC_LOW_L1_BYTES);
        aicL1A_ = aicL1ABuff_.GetWithOffset<float>(
            static_cast<uint32_t>(AIC_RAW_L1_WORK_BYTES / sizeof(float)), 0);
        aicL1B_ = aicL1BBuff_.Get<float>();

        uint32_t halfL1Offset = 0;
        aicHalfL1Inv_ = aicHalfL1Buff_.GetWithOffset<half>(
            static_cast<uint32_t>(AIC_LOW_L1_MATRIX_ELEMS), halfL1Offset);
        halfL1Offset += static_cast<uint32_t>(AIC_LOW_L1_MATRIX_ELEMS * sizeof(half));
        aicHalfL1Power_ = aicHalfL1Buff_.GetWithOffset<half>(
            static_cast<uint32_t>(AIC_LOW_L1_MATRIX_ELEMS), halfL1Offset);
        halfL1Offset += static_cast<uint32_t>(AIC_LOW_L1_MATRIX_ELEMS * sizeof(half));
        aicHalfL1Scratch_ = aicHalfL1Buff_.GetWithOffset<half>(
            static_cast<uint32_t>(AIC_LOW_L1_TILE_ELEMS), halfL1Offset);
        halfL1Offset += static_cast<uint32_t>(AIC_LOW_L1_TILE_ELEMS * sizeof(half));
        aicHalfL1TileInv_ = aicHalfL1Buff_.GetWithOffset<half>(
            static_cast<uint32_t>(AIC_LOW_L1_TILE_ELEMS), halfL1Offset);
        halfL1Offset += static_cast<uint32_t>(AIC_LOW_L1_TILE_ELEMS * sizeof(half));
        aicHalfL1TilePower_ = aicHalfL1Buff_.GetWithOffset<half>(
            static_cast<uint32_t>(AIC_LOW_L1_TILE_ELEMS), halfL1Offset);
        halfL1Offset += static_cast<uint32_t>(AIC_LOW_L1_TILE_ELEMS * sizeof(half));
        aicHalfL1Identity_ = aicHalfL1Buff_.GetWithOffset<half>(
            static_cast<uint32_t>(AIC_LOW_L1_TILE_ELEMS), halfL1Offset);

        pipe_->InitBuffer(aicL0ABuff_, AIC_RAW_L0_AB_BYTES);
        pipe_->InitBuffer(aicL0BBuff_, AIC_RAW_L0_AB_BYTES);
        pipe_->InitBuffer(aicL0CBuff_, AIC_RAW_L0_C_BYTES);
        aicL0A_ = aicL0ABuff_.Get<float>();
        aicL0B_ = aicL0BBuff_.Get<float>();
        aicL0C_ = aicL0CBuff_.Get<float>();
        aicHalfL0A_ = aicL0ABuff_.GetWithOffset<half>(
            static_cast<uint32_t>(AIC_RAW_L0_AB_HALF_PINGPONG_BYTES / sizeof(half)), 0);
        aicHalfL0B_ = aicL0BBuff_.GetWithOffset<half>(
            static_cast<uint32_t>(AIC_RAW_L0_AB_HALF_PINGPONG_BYTES / sizeof(half)), 0);
        aicHalfL0AAlt_ = aicL0ABuff_.GetWithOffset<half>(
            static_cast<uint32_t>(AIC_RAW_L0_AB_HALF_PINGPONG_BYTES / sizeof(half)),
            static_cast<uint32_t>(AIC_RAW_L0_AB_HALF_PINGPONG_BYTES));
        aicHalfL0BAlt_ = aicL0BBuff_.GetWithOffset<half>(
            static_cast<uint32_t>(AIC_RAW_L0_AB_HALF_PINGPONG_BYTES / sizeof(half)),
            static_cast<uint32_t>(AIC_RAW_L0_AB_HALF_PINGPONG_BYTES));
    }

    __aicore__ inline void InitRawMmadLocalBuffers()
    {
        // Raw MMAD needs exclusive A2/B2/CO1 space. MatmulImpl also allocates these
        // pools, so reset the AIC pipe before taking raw local buffers.
        pipe_->Reset();
        InitAicLocalBuffers();
        aicLocalBuffersReady_ = true;
    }

    __aicore__ inline void RestoreStage1MatmulLocalBuffers()
    {
        pipe_->Reset();
        aicLocalBuffersReady_ = false;
        mmFp32.Init(&tiling_->matmulTilingFp32, pipe_);
        mmFp32C16.Init(&tiling_->matmulTilingFp32C16, pipe_);
        mmFp32C32.Init(&tiling_->matmulTilingFp32C32, pipe_);
        if constexpr (FP16_L1_RESIDENT_INVERSE) {
            mmFp16L1C16_->Init(&tiling_->matmulTilingFp16C16L1, pipe_);
            mmFp16L1C32_->Init(&tiling_->matmulTilingFp16C32L1, pipe_);
        }
    }

    __aicore__ inline void EnsureAicLocalBuffers()
    {
        if (!aicLocalBuffersReady_) {
            InitRawMmadLocalBuffers();
        }
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
            AscendC::CrossCoreWaitFlag(0x9);  // 同步0
            // key @ key.transpose(-1,-2)
            AICProcess(keyContinousGm_, keyContinousGm_, kkWsGm_,
                       {chunkSize_, chunkSize_, dk_, chunkSize_, chunkSize_, dk_}, true);
            AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(0x8);  // 同步1：通知 AIV kkWsGm_ 就绪
            // query @ key.transpose(-1,-2)   stage1 out
            AICProcess(queryContinousGm_, keyContinousGm_, outQkGm_,
                       {validLen_, validLen_, dk_, validLen_, validLen_, dk_}, true);
            AscendC::CrossCoreWaitFlag(0x7);  // 同步2：等待 AIV 写出 L_powWsGm_ / INVWsGm_

            // AIV generates lower-triangular L in GM; AIC keeps INV accumulation in MMAD L0C.
            C64InverseBy32x32Neumann();

            AscendC::CrossCoreWaitFlag(0x6);  // 同步3：等待 AIV GBKCompute
            // INV @ k_cumdecay
            AICProcess(INVWsGm_, gBKWsGm_, outKCumdecayGm_,
                       {chunkSize_, dk_, chunkSize_, chunkSize_, dk_, chunkSize_});
            AscendC::CrossCoreWaitFlag(0x5);  // 同步4：等待 AIV VBetaCompute
            // INV @ v_beta    stage1 out
            AICProcess(INVWsGm_, vBetaWsGm_, outVInnerGm_,
                       {chunkSize_, dv_, chunkSize_, chunkSize_, dv_, chunkSize_});
            AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(0xc);  // 同步5：通知 AIV 本 chunk 的共享 workspace 已读完
        }
        if ASCEND_IS_AIV {
            // 获取连续QK
            QKPreProcess();
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(0x9);  // 同步0
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

    __aicore__ inline void QKPreProcessCompute(const GlobalTensor<bfloat16_t>& srcGm,
                                                const GlobalTensor<float>& dstGm,
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
        fp32OutQueue_.FreeTensor(tmpTensor);
    }

    __aicore__ inline void QKPreProcess()
    {
        uint64_t outOffset = subOffset_ * dk_;
        QKPreProcessCompute(queryGm_, queryContinousGm_[outOffset], qUbFloatCon_);
        QKPreProcessCompute(keyGm_, keyContinousGm_[outOffset], kUbFloatCon_, true);
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

    __aicore__ inline uint64_t MatrixNzOffset(uint64_t row, uint64_t col)
    {
        return (col / BLOCK_CUBE) * chunkSize_ * BLOCK_CUBE + row * BLOCK_CUBE + (col % BLOCK_CUBE);
    }

    __aicore__ inline uint64_t LocalFloatNzOffset(uint64_t row, uint64_t col, uint64_t rows)
    {
        uint64_t alignedRows = AlignUp(static_cast<uint32_t>(rows), BLOCK_CUBE);
        return (col / RAW_FLOAT_C0) * alignedRows * RAW_FLOAT_C0 + row * RAW_FLOAT_C0 + (col % RAW_FLOAT_C0);
    }

    __aicore__ inline uint64_t CompactMatrixNzOffset(uint64_t row, uint64_t col, uint64_t blockSize)
    {
        return (col / BLOCK_CUBE) * blockSize * BLOCK_CUBE + row * BLOCK_CUBE + (col % BLOCK_CUBE);
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

    __aicore__ inline void CopyGmFloatMatrixToHalfL1(GlobalTensor<float> gm, LocalTensor<half> l1,
                                                    uint32_t rows, uint32_t cols, uint32_t stride,
                                                    uint32_t rowBase = 0, uint32_t colBase = 0)
    {
        for (uint32_t row = 0; row < rows; ++row) {
            for (uint32_t col = 0; col < cols; ++col) {
                float value = gm.GetValue(row * stride + col);
                l1.SetValue(MatrixNzOffset(rowBase + row, colBase + col), static_cast<half>(value));
            }
        }
        SyncMToMte1();
    }

    __aicore__ inline void CopyHalfL1MatrixToGmFloat(LocalTensor<half> l1, GlobalTensor<float> gm,
                                                    uint32_t rows, uint32_t cols, uint32_t stride,
                                                    uint32_t rowBase = 0, uint32_t colBase = 0)
    {
        SyncMte1ToM();
        for (uint32_t row = 0; row < rows; ++row) {
            for (uint32_t col = 0; col < cols; ++col) {
                float value = static_cast<float>(l1.GetValue(MatrixNzOffset(rowBase + row, colBase + col)));
                gm.SetValue(row * stride + col, value);
            }
        }
        SyncScalarToMte2();
        FlushFloatMatrixCacheLines(gm, rows, cols, stride);
    }

    __aicore__ inline void CopyHalfL1MatrixToGmFloatByMmad(LocalTensor<half> l1, GlobalTensor<float> gm,
                                                           uint32_t blockSize, uint32_t stride)
    {
        RawMmadHalfL1AccumulateTile(l1, aicHalfL1Identity_, aicL0C_,
                                   blockSize, RAW_MMAD_INIT_C, RAW_MMAD_FINAL_UNIT_FLAG);
        SyncMToFix();
        CopyL0CMatrixToGm(gm, aicL0C_, blockSize, blockSize, stride, RAW_MMAD_FINAL_UNIT_FLAG);
        SyncFixToM();
        SyncFixToMte2();
        SyncFixToS();
        FlushFloatMatrixCacheLines(gm, blockSize, blockSize, stride);
    }

    __aicore__ inline void CopyHalfL1FullBlockToCompact(LocalTensor<half> full, LocalTensor<half> compact,
                                                       uint32_t rowBase, uint32_t colBase, uint32_t blockSize)
    {
        SyncMte1ToM();
        for (uint32_t row = 0; row < blockSize; ++row) {
            for (uint32_t col = 0; col < blockSize; ++col) {
                compact.SetValue(CompactMatrixNzOffset(row, col, blockSize),
                                 full.GetValue(MatrixNzOffset(rowBase + row, colBase + col)));
            }
        }
        SyncMToMte1();
    }

    __aicore__ inline void CopyHalfL1CompactToFullBlock(LocalTensor<half> compact, LocalTensor<half> full,
                                                       uint32_t rowBase, uint32_t colBase, uint32_t blockSize)
    {
        SyncFixToM();
        for (uint32_t row = 0; row < blockSize; ++row) {
            for (uint32_t col = 0; col < blockSize; ++col) {
                full.SetValue(MatrixNzOffset(rowBase + row, colBase + col),
                              compact.GetValue(CompactMatrixNzOffset(row, col, blockSize)));
            }
        }
        SyncMToMte1();
    }

    __aicore__ inline void InitHalfL1Identity()
    {
        SyncMte1ToM();
        for (uint32_t row = 0; row < INVERSE_SHAPE; ++row) {
            for (uint32_t col = 0; col < INVERSE_SHAPE; ++col) {
                aicHalfL1Identity_.SetValue(CompactMatrixNzOffset(row, col, INVERSE_SHAPE),
                                           row == col ? half(1.0f) : half(0.0f));
            }
        }
        SyncMToMte1();
    }

    __aicore__ inline void CopyHalfL1MatrixToL0A(LocalTensor<half> l1, LocalTensor<half> l0,
                                                uint32_t blockSize)
    {
        LoadData2DParams loadParams;
        loadParams.startIndex = 0;
        loadParams.repeatTimes =
            static_cast<uint16_t>(blockSize * blockSize * sizeof(half) / RAW_FRACTAL_BYTES);
        loadParams.srcStride = 1;
        loadParams.sid = 0;
        loadParams.dstGap = 0;
        loadParams.ifTranspose = false;
        loadParams.addrMode = 0;
        LoadData(l0, l1, loadParams);
    }

    __aicore__ inline void CopyHalfL1MatrixToL0B(LocalTensor<half> l1, LocalTensor<half> l0,
                                                uint32_t blockSize)
    {
        LoadData2DParams loadParams;
        loadParams.startIndex = 0;
        loadParams.repeatTimes =
            static_cast<uint16_t>(blockSize * blockSize * sizeof(half) / RAW_FRACTAL_BYTES);
        loadParams.srcStride = 1;
        loadParams.sid = 0;
        loadParams.dstGap = 0;
        loadParams.ifTranspose = true;
        loadParams.addrMode = 0;
        LoadData(l0, l1, loadParams);
    }

    __aicore__ inline void CopyL0CMatrixToHalfL1(LocalTensor<half> l1, LocalTensor<float> l0c,
                                                uint32_t mSize, uint32_t nSize, uint8_t unitFlag)
    {
        FixpipeParamsV220 fixParams;
        fixParams.nSize = nSize;
        fixParams.mSize = mSize;
        fixParams.srcStride = AlignUp(mSize, BLOCK_CUBE);
        fixParams.dstStride = AlignUp(mSize, BLOCK_CUBE) * BLOCK_CUBE * sizeof(half) / BLOCK_SIZE;
        fixParams.ndNum = 1;
        fixParams.srcNdStride = 0;
        fixParams.dstNdStride = 0;
        fixParams.reluEn = false;
        fixParams.quantPre = QuantMode_t::F322F16;
        fixParams.unitFlag = unitFlag;
        fixParams.isChannelSplit = false;
        Fixpipe<half, float, CFG_NZ>(l1, l0c, fixParams);
    }

    __aicore__ inline void CopyL0CMatrixToGm(GlobalTensor<float> gm, LocalTensor<float> l0c,
                                             uint32_t mSize, uint32_t nSize, uint32_t dstStride,
                                             uint8_t unitFlag)
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
        Fixpipe<float, float, CFG_ROW_MAJOR>(gm, l0c, fixParams);
    }

    __aicore__ inline void CopyL0CMatrixToGm(GlobalTensor<float> gm, uint32_t mSize, uint32_t nSize,
                                             uint32_t dstStride, uint8_t unitFlag)
    {
        CopyL0CMatrixToGm(gm, aicL0C_, mSize, nSize, dstStride, unitFlag);
    }

    __aicore__ inline MmadParams MakeRawMmadParams(uint32_t mSize, uint32_t nSize, uint32_t kSize,
                                                   bool initCMatrix, uint8_t unitFlag)
    {
        // MmadParams order: (..., unitFlag, cmatrixSource, cmatrixInitVal).
        // cmatrixSource=false keeps the C accumulator in L0C; cmatrixInitVal=false
        // accumulates into the same CMatrix instead of doing scalar or Fixpipe add.
        MmadParams params(static_cast<uint16_t>(mSize), static_cast<uint16_t>(nSize),
                          static_cast<uint16_t>(kSize), unitFlag, RAW_MMAD_CMATRIX_SOURCE_L0C, initCMatrix);
        params.kDirectionAlign = false;
#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 3510) || (__NPU_ARCH__ == 5102))
        params.disableGemv = true;
#endif
        return params;
    }

    __aicore__ inline void WriteRawInvAddIdentityTiles(uint64_t invOffset, uint32_t blockSize)
    {
        constexpr uint32_t rawTile = INVERSE_TAIL16_SHAPE;
        for (uint32_t tileBase = 0; tileBase < blockSize; tileBase += rawTile) {
            GlobalTensor<float> identityGm = tmpWsGm_[invOffset + tileBase * chunkSize_ + tileBase];
            for (uint32_t row = 0; row < rawTile; ++row) {
                for (uint32_t col = 0; col < rawTile; ++col) {
                    identityGm.SetValue(row * chunkSize_ + col,
                                        row == col ? static_cast<float>(1.0f) : static_cast<float>(0.0f));
                }
            }
            SyncScalarToMte2();
            FlushFloatMatrixCacheLines(identityGm, rawTile, rawTile, chunkSize_);
        }
    }

    __aicore__ inline void WriteZeroUpperTriangleTiles(GlobalTensor<float> gm, uint32_t blockSize)
    {
        constexpr uint32_t rawTile = INVERSE_TAIL16_SHAPE;
        for (uint32_t rowBase = 0; rowBase < blockSize; rowBase += rawTile) {
            for (uint32_t colBase = rowBase + rawTile; colBase < blockSize; colBase += rawTile) {
                GlobalTensor<float> upperGm = gm[rowBase * chunkSize_ + colBase];
                for (uint32_t row = 0; row < rawTile; ++row) {
                    for (uint32_t col = 0; col < rawTile; ++col) {
                        upperGm.SetValue(row * chunkSize_ + col, static_cast<float>(0.0f));
                    }
                }
                SyncScalarToMte2();
                FlushFloatMatrixCacheLines(upperGm, rawTile, rawTile, chunkSize_);
            }
        }
    }

    __aicore__ inline LocalTensor<float> RawL0CTile(uint32_t tileIdx)
    {
        return aicL0C_[tileIdx * RAW_L0C_TILE_FLOATS];
    }

    __aicore__ inline void RawMmadAccumulateTile(GlobalTensor<float> lhsGm, GlobalTensor<float> rhsGm,
                                                 LocalTensor<float> cTile,
                                                 uint32_t mSize, uint32_t nSize, uint32_t kSize,
                                                 bool initCMatrix, uint8_t unitFlag)
    {
        SyncMToMte1();
        CopyGmMatrixToL1A(lhsGm, aicL1A_, mSize, kSize, chunkSize_);
        CopyGmMatrixToL1B(rhsGm, aicL1B_, kSize, nSize, chunkSize_);
        SyncMte2ToMte1();
        CopyL1MatrixToL0A(aicL1A_, mSize, kSize);
        CopyL1MatrixToL0B(aicL1B_, kSize, nSize);
        SyncMte1ToM();

        PipeBarrier<PIPE_M>();
        SetHF32Mode(HF32Mode::DISABLE);
        Mmad(cTile, aicL0A_, aicL0B_, MakeRawMmadParams(mSize, nSize, kSize, initCMatrix, unitFlag));
        PipeBarrier<PIPE_M>();
    }

    __aicore__ inline void RawMmadHalfL1AccumulateTile(LocalTensor<half> lhsL1,
                                                      LocalTensor<half> rhsL1,
                                                      LocalTensor<float> cTile,
                                                      uint32_t blockSize,
                                                      bool initCMatrix,
                                                      uint8_t unitFlag,
                                                      uint32_t l0BufferIdx = 0)
    {
        LocalTensor<half> l0A = (l0BufferIdx == 0) ? aicHalfL0A_ : aicHalfL0AAlt_;
        LocalTensor<half> l0B = (l0BufferIdx == 0) ? aicHalfL0B_ : aicHalfL0BAlt_;
        SyncMToMte1();
        CopyHalfL1MatrixToL0A(lhsL1, l0A, blockSize);
        CopyHalfL1MatrixToL0B(rhsL1, l0B, blockSize);
        SyncMte1ToM();

        PipeBarrier<PIPE_M>();
        SetHF32Mode(HF32Mode::DISABLE);
        Mmad(cTile, l0A, l0B,
             MakeRawMmadParams(blockSize, blockSize, blockSize, initCMatrix, unitFlag));
        PipeBarrier<PIPE_M>();
    }

    __aicore__ inline void HalfL1PowerBlock(LocalTensor<half> lhs, LocalTensor<half> rhs,
                                           LocalTensor<half> dst, uint32_t blockSize)
    {
        RawMmadHalfL1AccumulateTile(lhs, rhs, aicL0C_, blockSize, RAW_MMAD_INIT_C, RAW_MMAD_FINAL_UNIT_FLAG);
        SyncMToFix();
        CopyL0CMatrixToHalfL1(dst, aicL0C_, blockSize, blockSize, RAW_MMAD_FINAL_UNIT_FLAG);
        SyncFixToMte1();
    }

    __aicore__ inline void HalfL1InvAddBlock(LocalTensor<half> inv, LocalTensor<half> power,
                                            uint32_t blockSize)
    {
        RawMmadHalfL1AccumulateTile(inv, aicHalfL1Identity_, aicL0C_,
                                   blockSize, RAW_MMAD_INIT_C, RAW_MMAD_NON_FINAL_UNIT_FLAG, 0);
        RawMmadHalfL1AccumulateTile(inv, power, aicL0C_,
                                   blockSize, RAW_MMAD_ACCUMULATE_C, RAW_MMAD_FINAL_UNIT_FLAG, 1);
        SyncMToFix();
        CopyL0CMatrixToHalfL1(inv, aicL0C_, blockSize, blockSize, RAW_MMAD_FINAL_UNIT_FLAG);
        SyncFixToMte1();
    }

    __aicore__ inline void ScalarHalfL1MatmulBlock(LocalTensor<half> x, LocalTensor<half> y,
                                                   LocalTensor<half> z, uint32_t blockSize)
    {
        SyncMte1ToM();
        for (uint32_t row = 0; row < blockSize; ++row) {
            for (uint32_t col = 0; col < blockSize; ++col) {
                float acc = 0.0f;
                for (uint32_t k = 0; k < blockSize; ++k) {
                    acc += static_cast<float>(x.GetValue(CompactMatrixNzOffset(row, k, blockSize))) *
                           static_cast<float>(y.GetValue(CompactMatrixNzOffset(k, col, blockSize)));
                }
                z.SetValue(CompactMatrixNzOffset(row, col, blockSize), static_cast<half>(acc));
            }
        }
        SyncMToMte1();
    }

    __aicore__ inline void ScalarHalfL1AddBlock(LocalTensor<half> dst, LocalTensor<half> src,
                                                uint32_t blockSize)
    {
        SyncMte1ToM();
        for (uint32_t row = 0; row < blockSize; ++row) {
            for (uint32_t col = 0; col < blockSize; ++col) {
                uint64_t offset = CompactMatrixNzOffset(row, col, blockSize);
                float value = static_cast<float>(dst.GetValue(offset)) + static_cast<float>(src.GetValue(offset));
                dst.SetValue(offset, static_cast<half>(value));
            }
        }
        SyncMToMte1();
    }

    __aicore__ inline void ScalarHalfL1InvAddBlock(LocalTensor<half> inv, LocalTensor<half> power,
                                                   LocalTensor<half> scratch, uint32_t blockSize)
    {
        ScalarHalfL1MatmulBlock(inv, power, scratch, blockSize);
        ScalarHalfL1AddBlock(inv, scratch, blockSize);
    }

    __aicore__ inline void HalfL1NeumannBlock(uint32_t base, uint32_t blockSize)
    {
        CopyHalfL1FullBlockToCompact(aicHalfL1Inv_, aicHalfL1TileInv_, base, base, blockSize);
        CopyHalfL1FullBlockToCompact(aicHalfL1Power_, aicHalfL1TilePower_, base, base, blockSize);
        LocalTensor<half> inv = aicHalfL1TileInv_;
        LocalTensor<half> power = aicHalfL1TilePower_;
        LocalTensor<half> scratch = aicHalfL1Scratch_;

        if constexpr (FP16_L1_RESIDENT_SCALAR_PROBE) {
            ScalarHalfL1MatmulBlock(power, power, scratch, blockSize);
            ScalarHalfL1InvAddBlock(inv, scratch, power, blockSize);

            ScalarHalfL1MatmulBlock(scratch, scratch, power, blockSize);
            ScalarHalfL1InvAddBlock(inv, power, scratch, blockSize);

            ScalarHalfL1MatmulBlock(power, power, scratch, blockSize);
            ScalarHalfL1InvAddBlock(inv, scratch, power, blockSize);

            if (blockSize > INVERSE_TAIL16_SHAPE) {
                ScalarHalfL1MatmulBlock(scratch, scratch, power, blockSize);
                ScalarHalfL1InvAddBlock(inv, power, scratch, blockSize);
            }
            CopyHalfL1CompactToFullBlock(inv, aicHalfL1Inv_, base, base, blockSize);
            return;
        }

        HalfL1PowerBlock(power, power, scratch, blockSize);
        HalfL1InvAddBlock(inv, scratch, blockSize);

        HalfL1PowerBlock(scratch, scratch, power, blockSize);
        HalfL1InvAddBlock(inv, power, blockSize);

        HalfL1PowerBlock(power, power, scratch, blockSize);
        HalfL1InvAddBlock(inv, scratch, blockSize);

        if (blockSize > INVERSE_TAIL16_SHAPE) {
            HalfL1PowerBlock(scratch, scratch, power, blockSize);
            HalfL1InvAddBlock(inv, power, blockSize);
        }
        CopyHalfL1CompactToFullBlock(inv, aicHalfL1Inv_, base, base, blockSize);
    }

    __aicore__ inline void RawMmadInvAddOutputTile(uint64_t invOffset, GlobalTensor<float> powerGm,
                                                   LocalTensor<float> cTile, uint32_t rowBase, uint32_t colBase,
                                                   uint32_t blockSize)
    {
        constexpr uint32_t rawTile = INVERSE_TAIL16_SHAPE;

        // Implement updated_INV = old_INV + old_INV @ power in CMatrix:
        // 1) old_INV @ I seeds the output tile into L0C.
        // 2) old_INV @ power uses cmatrixInitVal=false to accumulate into L0C.
        RawMmadAccumulateTile(INVWsGm_[invOffset + rowBase * chunkSize_ + colBase],
                              tmpWsGm_[invOffset + colBase * chunkSize_ + colBase],
                              cTile, rawTile, rawTile, rawTile, RAW_MMAD_INIT_C, RAW_MMAD_NON_FINAL_UNIT_FLAG);
        for (uint32_t kOffset = 0; kOffset < blockSize; kOffset += rawTile) {
            uint8_t unitFlag =
                (kOffset + rawTile >= blockSize) ? RAW_MMAD_FINAL_UNIT_FLAG : RAW_MMAD_NON_FINAL_UNIT_FLAG;
            RawMmadAccumulateTile(INVWsGm_[invOffset + rowBase * chunkSize_ + kOffset],
                                  powerGm[kOffset * chunkSize_ + colBase],
                                  cTile, rawTile, rawTile, rawTile, RAW_MMAD_ACCUMULATE_C, unitFlag);
        }
    }

    __aicore__ inline void StoreRawMmadInvAddRow(uint64_t invOffset, uint32_t rowBase, uint32_t blockSize)
    {
        constexpr uint32_t rawTile = INVERSE_TAIL16_SHAPE;
        SyncMToFix();
        for (uint32_t colBase = 0; colBase < blockSize; colBase += rawTile) {
            uint32_t tileIdx = colBase / rawTile;
            CopyL0CMatrixToGm(INVWsGm_[invOffset + rowBase * chunkSize_ + colBase], RawL0CTile(tileIdx),
                              rawTile, rawTile, chunkSize_, RAW_MMAD_FINAL_UNIT_FLAG);
        }
        SyncFixToM();
        SyncFixToMte2();
        SyncFixToS();
        FlushFloatMatrixCacheLines(INVWsGm_[invOffset + rowBase * chunkSize_], rawTile, blockSize, chunkSize_);
    }

    __aicore__ inline void StoreRawMmadRow(GlobalTensor<float> dstGm, uint32_t rowBase, uint32_t blockSize)
    {
        constexpr uint32_t rawTile = INVERSE_TAIL16_SHAPE;
        SyncMToFix();
        for (uint32_t colBase = 0; colBase < blockSize; colBase += rawTile) {
            uint32_t tileIdx = colBase / rawTile;
            CopyL0CMatrixToGm(dstGm[rowBase * chunkSize_ + colBase], RawL0CTile(tileIdx),
                              rawTile, rawTile, chunkSize_, RAW_MMAD_FINAL_UNIT_FLAG);
        }
        SyncFixToM();
        SyncFixToMte2();
        SyncFixToS();
        FlushFloatMatrixCacheLines(dstGm[rowBase * chunkSize_], rawTile, blockSize, chunkSize_);
    }

    __aicore__ inline void StoreRawMmadLowerTriangularRow(GlobalTensor<float> dstGm, uint32_t rowBase)
    {
        constexpr uint32_t rawTile = INVERSE_TAIL16_SHAPE;
        SyncMToFix();
        for (uint32_t colBase = 0; colBase <= rowBase; colBase += rawTile) {
            uint32_t tileIdx = colBase / rawTile;
            CopyL0CMatrixToGm(dstGm[rowBase * chunkSize_ + colBase], RawL0CTile(tileIdx),
                              rawTile, rawTile, chunkSize_, RAW_MMAD_FINAL_UNIT_FLAG);
        }
        SyncFixToM();
        SyncFixToMte2();
        SyncFixToS();
        FlushFloatMatrixCacheLines(dstGm[rowBase * chunkSize_], rawTile, rowBase + rawTile, chunkSize_);
    }

    __aicore__ inline void RawMmadBlockOutputTile(GlobalTensor<float> lhsGm, GlobalTensor<float> rhsGm,
                                                  LocalTensor<float> cTile, uint32_t rowBase, uint32_t colBase,
                                                  uint32_t blockSize)
    {
        constexpr uint32_t rawTile = INVERSE_TAIL16_SHAPE;
        for (uint32_t kOffset = 0; kOffset < blockSize; kOffset += rawTile) {
            bool initC = (kOffset == 0);
            uint8_t unitFlag =
                (kOffset + rawTile >= blockSize) ? RAW_MMAD_FINAL_UNIT_FLAG : RAW_MMAD_NON_FINAL_UNIT_FLAG;
            RawMmadAccumulateTile(lhsGm[rowBase * chunkSize_ + kOffset],
                                  rhsGm[kOffset * chunkSize_ + colBase],
                                  cTile, rawTile, rawTile, rawTile, initC, unitFlag);
        }
    }

    __aicore__ inline void RawMmadLowerTriangularBlockOutputTile(GlobalTensor<float> lhsGm, GlobalTensor<float> rhsGm,
                                                                 LocalTensor<float> cTile, uint32_t rowBase,
                                                                 uint32_t colBase)
    {
        constexpr uint32_t rawTile = INVERSE_TAIL16_SHAPE;
        for (uint32_t kOffset = colBase; kOffset <= rowBase; kOffset += rawTile) {
            bool initC = (kOffset == colBase);
            uint8_t unitFlag = (kOffset == rowBase) ? RAW_MMAD_FINAL_UNIT_FLAG : RAW_MMAD_NON_FINAL_UNIT_FLAG;
            RawMmadAccumulateTile(lhsGm[rowBase * chunkSize_ + kOffset],
                                  rhsGm[kOffset * chunkSize_ + colBase],
                                  cTile, rawTile, rawTile, rawTile, initC, unitFlag);
        }
    }

    __aicore__ inline void RawMmadBlock(GlobalTensor<float> lhsGm, GlobalTensor<float> rhsGm,
                                        GlobalTensor<float> dstGm, uint32_t blockSize)
    {
        constexpr uint32_t rawTile = INVERSE_TAIL16_SHAPE;
        for (uint32_t rowBase = 0; rowBase < blockSize; rowBase += rawTile) {
            for (uint32_t colBase = 0; colBase < blockSize; colBase += rawTile) {
                uint32_t tileIdx = colBase / rawTile;
                RawMmadBlockOutputTile(lhsGm, rhsGm, RawL0CTile(tileIdx), rowBase, colBase, blockSize);
            }
            StoreRawMmadRow(dstGm, rowBase, blockSize);
        }
    }

    __aicore__ inline void RawMmadLowerTriangularBlock(GlobalTensor<float> lhsGm, GlobalTensor<float> rhsGm,
                                                       GlobalTensor<float> dstGm, uint32_t blockSize)
    {
        constexpr uint32_t rawTile = INVERSE_TAIL16_SHAPE;
        // Power matrices must stay strictly lower triangular. Full-tile power without
        // clearing upper tiles caused 1e-2 class output drift in precision probes.
        for (uint32_t rowBase = 0; rowBase < blockSize; rowBase += rawTile) {
            for (uint32_t colBase = 0; colBase <= rowBase; colBase += rawTile) {
                uint32_t tileIdx = colBase / rawTile;
                RawMmadLowerTriangularBlockOutputTile(lhsGm, rhsGm, RawL0CTile(tileIdx), rowBase, colBase);
            }
            StoreRawMmadLowerTriangularRow(dstGm, rowBase);
        }
        WriteZeroUpperTriangleTiles(dstGm, blockSize);
    }

    __aicore__ inline void RawMmadInvAddBlock(uint64_t invOffset, GlobalTensor<float> powerGm, uint32_t blockSize)
    {
        constexpr uint32_t rawTile = INVERSE_TAIL16_SHAPE;
        for (uint32_t rowBase = 0; rowBase < blockSize; rowBase += rawTile) {
            for (uint32_t colBase = 0; colBase < blockSize; colBase += rawTile) {
                uint32_t tileIdx = colBase / rawTile;
                RawMmadInvAddOutputTile(invOffset, powerGm, RawL0CTile(tileIdx), rowBase, colBase, blockSize);
            }
            StoreRawMmadInvAddRow(invOffset, rowBase, blockSize);
        }
    }

    __aicore__ inline void SyncScalarToMte2()
    {
        int32_t eventID = static_cast<int32_t>(pipe_->FetchEventID(HardEvent::S_MTE2));
        SetFlag<HardEvent::S_MTE2>(eventID);
        WaitFlag<HardEvent::S_MTE2>(eventID);
    }

    __aicore__ inline void FlushFloatMatrixCacheLines(GlobalTensor<float> gm, uint32_t rows, uint32_t cols,
                                                      uint32_t stride)
    {
        for (uint32_t row = 0; row < rows; ++row) {
            for (uint32_t col = 0; col < cols; col += FLOAT_CACHELINE_ELEMS) {
                Barrier();
                DataCacheCleanAndInvalid<float, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
                    gm[row * stride + col]);
            }
        }
        Barrier();
    }

    __aicore__ inline void AICBlockProcess(GlobalTensor<float> x, GlobalTensor<float> y, GlobalTensor<float> z,
                                           uint64_t blockSize, bool enableAtomicAdd = false)
    {
        MatmulShapeParams shape = {chunkSize_, chunkSize_, chunkSize_, blockSize, blockSize, blockSize};
        AICProcess(x, y, z, shape, false, enableAtomicAdd);
    }

    __aicore__ inline void AICBlockProcessFp16L1(LocalTensor<half> x, LocalTensor<half> y, LocalTensor<half> z,
                                                 uint32_t blockSize, bool enableAtomicAdd = false)
    {
        StageOneFp16L1MT &mm = (blockSize == INVERSE_TAIL16_SHAPE) ? *mmFp16L1C16_ : *mmFp16L1C32_;
        mm.SetOrgShape(blockSize, blockSize, blockSize, blockSize, blockSize);
        mm.SetSingleShape(blockSize, blockSize, blockSize);
        mm.SetTensorA(x);
        mm.SetTensorB(y);
        mm.IterateAll(z, enableAtomicAdd ? 1 : 0);
        mm.End();
    }

    __aicore__ inline void AddHalfL1Matrix(LocalTensor<half> dst, LocalTensor<half> src, uint32_t blockSize)
    {
        SyncFixToM();
        for (uint32_t row = 0; row < blockSize; ++row) {
            for (uint32_t col = 0; col < blockSize; ++col) {
                uint64_t offset = CompactMatrixNzOffset(row, col, blockSize);
                float value = static_cast<float>(dst.GetValue(offset)) + static_cast<float>(src.GetValue(offset));
                dst.SetValue(offset, static_cast<half>(value));
            }
        }
        SyncMToMte1();
    }

    __aicore__ inline void MulHalfL1Block(LocalTensor<half> lhs, LocalTensor<half> rhs,
                                          LocalTensor<half> dst, uint32_t blockSize)
    {
        AICBlockProcessFp16L1(lhs, rhs, dst, blockSize);
        SyncFixToMte1();
    }

    __aicore__ inline void Neumann16InverseBlock(uint64_t base)
    {
        if constexpr (FP16_L1_RESIDENT_INVERSE) {
            if constexpr (FP16_L1_ROUNDTRIP_THEN_RAW_PROBE) {
                if constexpr (FP16_L1_ROUNDTRIP_MMAD_WRITEBACK_PROBE) {
                    CopyHalfL1MatrixToGmFloatByMmad(aicHalfL1Inv_[MatrixNzOffset(base, base)],
                                                    INVWsGm_[MatrixOffset(base, base)],
                                                    INVERSE_TAIL16_SHAPE, chunkSize_);
                } else {
                CopyHalfL1MatrixToGmFloat(aicHalfL1Inv_, INVWsGm_[MatrixOffset(base, base)],
                                         INVERSE_TAIL16_SHAPE, INVERSE_TAIL16_SHAPE, chunkSize_, base, base);
                }
            } else {
                HalfL1NeumannBlock(static_cast<uint32_t>(base), INVERSE_TAIL16_SHAPE);
                CopyHalfL1MatrixToGmFloat(aicHalfL1Inv_, INVWsGm_[MatrixOffset(base, base)],
                                         INVERSE_TAIL16_SHAPE, INVERSE_TAIL16_SHAPE, chunkSize_, base, base);
                return;
            }
        }
        uint64_t offset = MatrixOffset(base, base);
        WriteRawInvAddIdentityTiles(offset, INVERSE_TAIL16_SHAPE);

        RawMmadLowerTriangularBlock(L_powWsGm_[offset], L_powWsGm_[offset], attnWsGm_[offset],
                                    INVERSE_TAIL16_SHAPE);
        RawMmadInvAddBlock(offset, attnWsGm_[offset], INVERSE_TAIL16_SHAPE);

        RawMmadLowerTriangularBlock(attnWsGm_[offset], attnWsGm_[offset], L_powWsGm_[offset],
                                    INVERSE_TAIL16_SHAPE);
        RawMmadInvAddBlock(offset, L_powWsGm_[offset], INVERSE_TAIL16_SHAPE);

        RawMmadLowerTriangularBlock(L_powWsGm_[offset], L_powWsGm_[offset], attnWsGm_[offset],
                                    INVERSE_TAIL16_SHAPE);
        RawMmadInvAddBlock(offset, attnWsGm_[offset], INVERSE_TAIL16_SHAPE);
    }

    __aicore__ inline void Neumann32InverseBlock(uint64_t base)
    {
        if constexpr (FP16_L1_RESIDENT_INVERSE) {
            HalfL1NeumannBlock(static_cast<uint32_t>(base), INVERSE_SHAPE);
            CopyHalfL1MatrixToGmFloat(aicHalfL1Inv_, INVWsGm_[MatrixOffset(base, base)],
                                     INVERSE_SHAPE, INVERSE_SHAPE, chunkSize_, base, base);
            return;
        }
        uint64_t offset = MatrixOffset(base, base);
        WriteRawInvAddIdentityTiles(offset, INVERSE_SHAPE);

        // Round 1: L² = L @ L, INV += INV @ L²
        RawMmadLowerTriangularBlock(L_powWsGm_[offset], L_powWsGm_[offset], attnWsGm_[offset], INVERSE_SHAPE);
        RawMmadInvAddBlock(offset, attnWsGm_[offset], INVERSE_SHAPE);

        // Round 2: L⁴ = L² @ L², INV += INV @ L⁴
        RawMmadLowerTriangularBlock(attnWsGm_[offset], attnWsGm_[offset], L_powWsGm_[offset], INVERSE_SHAPE);
        RawMmadInvAddBlock(offset, L_powWsGm_[offset], INVERSE_SHAPE);

        // Round 3: L⁸ = L⁴ @ L⁴, INV += INV @ L⁸
        RawMmadLowerTriangularBlock(L_powWsGm_[offset], L_powWsGm_[offset], attnWsGm_[offset], INVERSE_SHAPE);
        RawMmadInvAddBlock(offset, attnWsGm_[offset], INVERSE_SHAPE);

        // Round 4: L¹⁶ = L⁸ @ L⁸, INV += INV @ L¹⁶
        RawMmadLowerTriangularBlock(attnWsGm_[offset], attnWsGm_[offset], L_powWsGm_[offset], INVERSE_SHAPE);
        RawMmadInvAddBlock(offset, L_powWsGm_[offset], INVERSE_SHAPE);
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

    __aicore__ inline void SyncMte2ToMte1()
    {
        int32_t eventID = static_cast<int32_t>(pipe_->FetchEventID(HardEvent::MTE2_MTE1));
        SetFlag<HardEvent::MTE2_MTE1>(eventID);
        WaitFlag<HardEvent::MTE2_MTE1>(eventID);
    }

    __aicore__ inline void SyncMte2ToM()
    {
        int32_t eventID = static_cast<int32_t>(pipe_->FetchEventID(HardEvent::MTE2_M));
        SetFlag<HardEvent::MTE2_M>(eventID);
        WaitFlag<HardEvent::MTE2_M>(eventID);
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

    __aicore__ inline void Compose64From32()
    {
        if constexpr (FP16_L1_RESIDENT_INVERSE) {
            RawMmadBlock(INVWsGm_[MatrixOffset(INVERSE_SHAPE, 0)], INVWsGm_[MatrixOffset(0, 0)],
                         tmpWsGm_[MatrixOffset(INVERSE_SHAPE, 0)], INVERSE_SHAPE);
            RawMmadBlock(INVWsGm_[MatrixOffset(INVERSE_SHAPE, INVERSE_SHAPE)],
                         tmpWsGm_[MatrixOffset(INVERSE_SHAPE, 0)],
                         INVWsGm_[MatrixOffset(INVERSE_SHAPE, 0)], INVERSE_SHAPE);
            CopyGmFloatMatrixToHalfL1(INVWsGm_[MatrixOffset(INVERSE_SHAPE, 0)], aicHalfL1Inv_,
                                     INVERSE_SHAPE, INVERSE_SHAPE, chunkSize_, INVERSE_SHAPE, 0);
            return;
        }
        uint64_t topDiag = MatrixOffset(0, 0);
        uint64_t bottomDiag = MatrixOffset(INVERSE_SHAPE, INVERSE_SHAPE);
        uint64_t lowerLeft = MatrixOffset(INVERSE_SHAPE, 0);

        RawMmadBlock(INVWsGm_[lowerLeft], INVWsGm_[topDiag], tmpWsGm_[lowerLeft], INVERSE_SHAPE);
        RawMmadBlock(INVWsGm_[bottomDiag], tmpWsGm_[lowerLeft], INVWsGm_[lowerLeft], INVERSE_SHAPE);
    }

    __aicore__ inline void Compose32From16(uint64_t base)
    {
        if constexpr (FP16_L1_RESIDENT_INVERSE) {
            uint32_t top = static_cast<uint32_t>(base);
            uint32_t bottom = top + INVERSE_TAIL16_SHAPE;
            RawMmadBlock(INVWsGm_[MatrixOffset(bottom, top)], INVWsGm_[MatrixOffset(top, top)],
                         tmpWsGm_[MatrixOffset(bottom, top)], INVERSE_TAIL16_SHAPE);
            RawMmadBlock(INVWsGm_[MatrixOffset(bottom, bottom)], tmpWsGm_[MatrixOffset(bottom, top)],
                         INVWsGm_[MatrixOffset(bottom, top)], INVERSE_TAIL16_SHAPE);
            CopyGmFloatMatrixToHalfL1(INVWsGm_[MatrixOffset(bottom, top)], aicHalfL1Inv_,
                                     INVERSE_TAIL16_SHAPE, INVERSE_TAIL16_SHAPE,
                                     chunkSize_, bottom, top);
            return;
        }
        uint64_t top = base;
        uint64_t bottom = base + INVERSE_TAIL16_SHAPE;
        uint64_t topDiag = MatrixOffset(top, top);
        uint64_t bottomDiag = MatrixOffset(bottom, bottom);
        uint64_t lowerLeft = MatrixOffset(bottom, top);

        RawMmadBlock(INVWsGm_[lowerLeft], INVWsGm_[topDiag], tmpWsGm_[lowerLeft], INVERSE_TAIL16_SHAPE);
        RawMmadBlock(INVWsGm_[bottomDiag], tmpWsGm_[lowerLeft], INVWsGm_[lowerLeft], INVERSE_TAIL16_SHAPE);
    }

    __aicore__ inline void Inverse32By16Compose(uint64_t base)
    {
        Neumann16InverseBlock(base);
        if (validLen_ > base + INVERSE_TAIL16_SHAPE) {
            Neumann16InverseBlock(base + INVERSE_TAIL16_SHAPE);
            Compose32From16(base);
        }
    }

    __aicore__ inline void C64InverseBy32x32Neumann()
    {
        EnsureAicLocalBuffers();
        PrepareRawMmadState();
        SyncFixToM();
        if constexpr (FP16_L1_RESIDENT_INVERSE) {
            InitHalfL1Identity();
            CopyGmFloatMatrixToHalfL1(L_powWsGm_, aicHalfL1Power_, chunkSize_, chunkSize_, chunkSize_);
            CopyGmFloatMatrixToHalfL1(INVWsGm_, aicHalfL1Inv_, chunkSize_, chunkSize_, chunkSize_);
        }

        if (validLen_ <= INVERSE_TAIL16_SHAPE) {
            Neumann16InverseBlock(0);
            RestoreRawMmadState();
            RestoreStage1MatmulLocalBuffers();
            return;
        }
        Inverse32By16Compose(0);
        if (validLen_ > INVERSE_SHAPE) {
            Inverse32By16Compose(INVERSE_SHAPE);
            Compose64From32();
        }

        RestoreRawMmadState();
        RestoreStage1MatmulLocalBuffers();
    }

    TPipe *pipe_;
    StageOneMT &mmFp32;
    StageOneMT &mmFp32C16;
    StageOneMT &mmFp32C32;
    StageOneFp16L1MT *mmFp16L1C16_;
    StageOneFp16L1MT *mmFp16L1C32_;
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

    // UB queues
    TQue<QuePosition::VECIN, 1> fp32InQueue_;
    TQue<QuePosition::VECOUT, 1> fp32OutQueue_;
    TQue<QuePosition::VECOUT, 1> gOutQueue_;

    TBuf<TPosition::VECCALC> tmpBuff_;
    TBuf<TPosition::A1> aicL1ABuff_;
    TBuf<TPosition::B1> aicL1BBuff_;
    TBuf<TPosition::TSCM> aicHalfL1Buff_;
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
    LocalTensor<uint8_t> aicLocalMmWs_;
    LocalTensor<half> aicHalfL1Inv_;
    LocalTensor<half> aicHalfL1Power_;
    LocalTensor<half> aicHalfL1Scratch_;
    LocalTensor<half> aicHalfL1TileInv_;
    LocalTensor<half> aicHalfL1TilePower_;
    LocalTensor<half> aicHalfL1Identity_;
    LocalTensor<half> aicHalfL0A_;
    LocalTensor<half> aicHalfL0B_;
    LocalTensor<half> aicHalfL0AAlt_;
    LocalTensor<half> aicHalfL0BAlt_;
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
#endif // CHUNK_GATED_DELTA_RULE_NEUMANN_STAGE1_C64_3232_FP16_FULLLOCAL_H
