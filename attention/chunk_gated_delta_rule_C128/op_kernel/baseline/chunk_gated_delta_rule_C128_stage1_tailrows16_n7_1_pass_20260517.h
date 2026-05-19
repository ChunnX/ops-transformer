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
 * \file chunk_gated_delta_rule_C128_stage1.h
 * \brief
 */
#ifndef CHUNK_GATED_DELTA_RULE_C128_STAGE1_H
#define CHUNK_GATED_DELTA_RULE_C128_STAGE1_H

#include "kernel_tiling/kernel_tiling.h"
#include "chunk_gated_delta_rule_C128_utils.h"
#include "chunk_gated_delta_rule_C128_tiling_data.h"

namespace ChunkGatedDeltaRuleC128 {
using namespace AscendC;
using namespace matmul;

using aT1 = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t>;
using bT1 = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t>;
using cT1 = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t>;
using StageOneMT = matmul::MatmulImpl<aT1, bT1, cT1>;

constexpr uint64_t UB_REST_BYTES = 120 * 1024; // Stage1 keeps one [c/2,c] slice plus [c/2,c/2] inverse scratch.
constexpr uint64_t INVERSE_SHAPE = 16;         // 16x16 Neumann basic block inside one C64 inverse.
constexpr uint64_t C64_BLOCK32 = 32;
constexpr uint64_t C64_PACKED_SHAPE = 64;
constexpr uint32_t C64_PACKED_SIZE_U32 = 64;
constexpr uint32_t C64_FRACTAL_U32 = 16;
constexpr uint32_t C64_PACKED_BLOCK_COUNT = C64_PACKED_SIZE_U32 / C64_FRACTAL_U32;
constexpr uint64_t AIC_RAW_L1_WORK_BYTES = 16 * 1024;
constexpr uint64_t AIC_RAW_L1_A_BYTES = AIC_RAW_L1_WORK_BYTES;
constexpr uint64_t AIC_RAW_L1_B_BYTES = 16 * 1024;
constexpr uint64_t AIC_LOW_L1_TILE_ELEMS = C64_PACKED_SHAPE * C64_PACKED_SHAPE;
constexpr uint64_t C128_RESIDENT_TILE_COUNT = 3;
constexpr uint64_t C128_RESIDENT_PARA_SLOTS = 1;
constexpr uint64_t AIC_LOW_L1_BYTES = AIC_LOW_L1_TILE_ELEMS * 13 * sizeof(half);
constexpr uint32_t C64_L0C_TILE_ELEMS = C64_PACKED_SIZE_U32 * C64_PACKED_SIZE_U32;
constexpr uint64_t AIC_RAW_L0_AB_BYTES = 32 * 1024;
constexpr uint64_t AIC_RAW_L0_AB_HALF_PINGPONG_BYTES = AIC_RAW_L0_AB_BYTES / 2;
constexpr uint64_t AIC_RAW_L0_C_BYTES = 64 * 1024;
constexpr uint32_t ALIGN_SIZE = 16;
constexpr uint32_t MAX_PARALLEL_NUM = 6;
constexpr uint32_t HALF_TWO = 2;
constexpr uint32_t RAW_FLOAT_C0 = BLOCK_SIZE / sizeof(float);
constexpr uint32_t RAW_FRACTAL_BYTES = 512;
constexpr uint32_t RAW_HALF_FRACTAL_ELEMS = RAW_FRACTAL_BYTES / sizeof(half);
static constexpr IsResetLoad3dConfig RAW_LOAD3DV2_CONFIG = {true, true};
constexpr uint8_t RAW_MMAD_NON_FINAL_UNIT_FLAG = 2;
constexpr uint8_t RAW_MMAD_FINAL_UNIT_FLAG = 3;
constexpr bool RAW_MMAD_CMATRIX_SOURCE_L0C = false;
constexpr bool RAW_MMAD_INIT_C = true;
constexpr bool RAW_MMAD_ACCUMULATE_C = false;
constexpr uint32_t RAW_L0AB_DB_FREE_EVENT0 = EVENT_ID3;
constexpr uint32_t RAW_L0AB_DB_FREE_EVENT1 = EVENT_ID4;

constexpr uint32_t C64_DIAG_BASE0 = 0;
constexpr uint32_t C64_DIAG_BASE1 = C64_FRACTAL_U32;
constexpr uint32_t C64_DIAG_BASE2 = C64_BLOCK32;
constexpr uint32_t C64_DIAG_BASE3 = C64_BLOCK32 + C64_FRACTAL_U32;
constexpr uint32_t C64_DIAG_NZ_OFFSET0 = 0;
constexpr uint32_t C64_DIAG_NZ_OFFSET1 = 1 * (C64_PACKED_BLOCK_COUNT + 1) * RAW_HALF_FRACTAL_ELEMS;
constexpr uint32_t C64_DIAG_NZ_OFFSET2 = 2 * (C64_PACKED_BLOCK_COUNT + 1) * RAW_HALF_FRACTAL_ELEMS;
constexpr uint32_t C64_DIAG_NZ_OFFSET3 = 3 * (C64_PACKED_BLOCK_COUNT + 1) * RAW_HALF_FRACTAL_ELEMS;
constexpr uint32_t C64_DIAG_NZ_STRIDE_FRACTALS = C64_PACKED_BLOCK_COUNT + 1;
constexpr uint32_t C64_DIAG_IDENTITY_COPY_BLOCKS = RAW_FRACTAL_BYTES / BLOCK_SIZE;
constexpr uint32_t C64_DIAG_IDENTITY_DST_GAP_BLOCKS =
    (C64_DIAG_NZ_OFFSET1 - C64_DIAG_NZ_OFFSET0 - RAW_HALF_FRACTAL_ELEMS) * sizeof(half) / BLOCK_SIZE;
constexpr uint32_t C64_DIAG_L0_LOAD_DST_GAP_FRACTALS = C64_DIAG_NZ_STRIDE_FRACTALS - 1;
constexpr uint32_t C64_ODD_EVEN_PAIR_L0_LOAD_DST_GAP_FRACTALS =
    (C64_DIAG_NZ_OFFSET2 - C64_DIAG_NZ_OFFSET0) / RAW_HALF_FRACTAL_ELEMS - 1;
constexpr uint32_t C64_L0_ROW1_COL0_OFFSET = C64_PACKED_BLOCK_COUNT * RAW_HALF_FRACTAL_ELEMS;
constexpr uint32_t C64_FULL_L0_BLOCK_DST_GAP_FRACTALS = C64_PACKED_BLOCK_COUNT - 1;
constexpr uint32_t C64_FIXPIPE_HALF_DST_STRIDE = C64_PACKED_SIZE_U32 * C64_FRACTAL_U32 * sizeof(half) / BLOCK_SIZE;
constexpr uint32_t C64_FIXPIPE_BF16_DST_STRIDE =
    C64_PACKED_SIZE_U32 * C64_FRACTAL_U32 * sizeof(bfloat16_t) / BLOCK_SIZE;
constexpr uint32_t C32_PACKED_SIZE_U32 = 32;
constexpr uint32_t C32_PACKED_BLOCK_COUNT = C32_PACKED_SIZE_U32 / C64_FRACTAL_U32;
constexpr uint32_t C32_DIAG_NZ_OFFSET0 = 0;
constexpr uint32_t C32_DIAG_NZ_OFFSET1 = 1 * (C32_PACKED_BLOCK_COUNT + 1) * RAW_HALF_FRACTAL_ELEMS;
constexpr uint32_t C32_DIAG_NZ_STRIDE_FRACTALS = C32_PACKED_BLOCK_COUNT + 1;
constexpr uint32_t C32_DIAG_L0_LOAD_DST_GAP_FRACTALS = C32_DIAG_NZ_STRIDE_FRACTALS - 1;
constexpr int16_t FP16_ONE_BITS = 15360;

#define CGDR_C128_I16_NZ_ROW0 FP16_ONE_BITS, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
#define CGDR_C128_I16_NZ_ROW1 0, FP16_ONE_BITS, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
#define CGDR_C128_I16_NZ_ROW2 0, 0, FP16_ONE_BITS, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
#define CGDR_C128_I16_NZ_ROW3 0, 0, 0, FP16_ONE_BITS, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
#define CGDR_C128_I16_NZ_ROW4 0, 0, 0, 0, FP16_ONE_BITS, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
#define CGDR_C128_I16_NZ_ROW5 0, 0, 0, 0, 0, FP16_ONE_BITS, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
#define CGDR_C128_I16_NZ_ROW6 0, 0, 0, 0, 0, 0, FP16_ONE_BITS, 0, 0, 0, 0, 0, 0, 0, 0, 0
#define CGDR_C128_I16_NZ_ROW7 0, 0, 0, 0, 0, 0, 0, FP16_ONE_BITS, 0, 0, 0, 0, 0, 0, 0, 0
#define CGDR_C128_I16_NZ_ROW8 0, 0, 0, 0, 0, 0, 0, 0, FP16_ONE_BITS, 0, 0, 0, 0, 0, 0, 0
#define CGDR_C128_I16_NZ_ROW9 0, 0, 0, 0, 0, 0, 0, 0, 0, FP16_ONE_BITS, 0, 0, 0, 0, 0, 0
#define CGDR_C128_I16_NZ_ROW10 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, FP16_ONE_BITS, 0, 0, 0, 0, 0
#define CGDR_C128_I16_NZ_ROW11 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, FP16_ONE_BITS, 0, 0, 0, 0
#define CGDR_C128_I16_NZ_ROW12 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, FP16_ONE_BITS, 0, 0, 0
#define CGDR_C128_I16_NZ_ROW13 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, FP16_ONE_BITS, 0, 0
#define CGDR_C128_I16_NZ_ROW14 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, FP16_ONE_BITS, 0
#define CGDR_C128_I16_NZ_ROW15 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, FP16_ONE_BITS
#define CGDR_C128_I16_NZ_FRACTAL                                                                                       \
    CGDR_C128_I16_NZ_ROW0, CGDR_C128_I16_NZ_ROW1, CGDR_C128_I16_NZ_ROW2, CGDR_C128_I16_NZ_ROW3, CGDR_C128_I16_NZ_ROW4, \
        CGDR_C128_I16_NZ_ROW5, CGDR_C128_I16_NZ_ROW6, CGDR_C128_I16_NZ_ROW7, CGDR_C128_I16_NZ_ROW8,                    \
        CGDR_C128_I16_NZ_ROW9, CGDR_C128_I16_NZ_ROW10, CGDR_C128_I16_NZ_ROW11, CGDR_C128_I16_NZ_ROW12,                 \
        CGDR_C128_I16_NZ_ROW13, CGDR_C128_I16_NZ_ROW14, CGDR_C128_I16_NZ_ROW15

static constexpr volatile __gm__ int16_t CGDR_C128_IDENTITY16_NZ4[RAW_HALF_FRACTAL_ELEMS * C64_PACKED_BLOCK_COUNT] = {
    CGDR_C128_I16_NZ_FRACTAL, CGDR_C128_I16_NZ_FRACTAL, CGDR_C128_I16_NZ_FRACTAL, CGDR_C128_I16_NZ_FRACTAL};

#undef CGDR_C128_I16_NZ_ROW0
#undef CGDR_C128_I16_NZ_ROW1
#undef CGDR_C128_I16_NZ_ROW2
#undef CGDR_C128_I16_NZ_ROW3
#undef CGDR_C128_I16_NZ_ROW4
#undef CGDR_C128_I16_NZ_ROW5
#undef CGDR_C128_I16_NZ_ROW6
#undef CGDR_C128_I16_NZ_ROW7
#undef CGDR_C128_I16_NZ_ROW8
#undef CGDR_C128_I16_NZ_ROW9
#undef CGDR_C128_I16_NZ_ROW10
#undef CGDR_C128_I16_NZ_ROW11
#undef CGDR_C128_I16_NZ_ROW12
#undef CGDR_C128_I16_NZ_ROW13
#undef CGDR_C128_I16_NZ_ROW14
#undef CGDR_C128_I16_NZ_ROW15
#undef CGDR_C128_I16_NZ_FRACTAL

// Matmul 形状参数结构体
struct MatmulShapeParams {
    uint64_t m;  // 原始 M 维度
    uint64_t n;  // 原始 N 维度
    uint64_t k;  // 原始 K 维度
    uint64_t sm; // 单次计算 M 维度
    uint64_t sn; // 单次计算 N 维度
    uint64_t sk; // 单次计算 K 维度
};

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
    GlobalTensor<bfloat16_t> vInner;    // (Nv, cg_len, Dv)
    GlobalTensor<bfloat16_t> qPrime;    // (Nv, cg_len, Dk)
    GlobalTensor<bfloat16_t> kG;        // (Nv, cg_len, Dk)
    GlobalTensor<bfloat16_t> qK;        // (Nv, cg_len, C)
    // other
    GM_ADDR ws;
    GlobalTensor<float> stageOneMask; // (Nv, cg_len, C)
    ChunkGroup cg;
    bool gOptional;
};

class Stage1 {
public:
    __aicore__ inline Stage1(StageOneMT &mm) : mm(mm)
    {
    }
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
        workSpaceOffset += coreNum_ * paraNum_ * cvOffset_ * sizeof(bfloat16_t);

        curWS = coreIdx_ * paraNum_ * ckOffset_ * sizeof(half);
        gBKHalfWsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(initParams.ws + workSpaceOffset + curWS));
        workSpaceOffset += coreNum_ * paraNum_ * ckOffset_ * sizeof(half);

        curWS = coreIdx_ * paraNum_ * cvOffset_ * sizeof(half);
        vBetaHalfWsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(initParams.ws + workSpaceOffset + curWS));
        workSpaceOffset += coreNum_ * paraNum_ * cvOffset_ * sizeof(half);

        curWS = coreIdx_ * paraNum_ * ccOffset_ * sizeof(half);
        L_powHalfWsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(initParams.ws + workSpaceOffset + curWS));
        workSpaceOffset += coreNum_ * paraNum_ * ccOffset_ * sizeof(half);

        INVInitHalfWsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(initParams.ws + workSpaceOffset + curWS));
        workSpaceOffset += coreNum_ * paraNum_ * ccOffset_ * sizeof(half);
    }

    __aicore__ inline void InitLocalBuffers()
    {
        maxCols_ = AscendC::Std::max(AscendC::Std::max(dvAligned_, dkAligned_), chunkSize_);
        subBlockElementNum_ = halfChunkSize_ * maxCols_;
        pipe_->InitBuffer(inQueue_, BUFFER_NUM_ONE, subBlockElementNum_ * sizeof(float));
        pipe_->InitBuffer(outQueue_, BUFFER_NUM_ONE, subBlockElementNum_ * sizeof(bfloat16_t));
        if (gOptional_) {
            pipe_->InitBuffer(gOutQueue_, BUFFER_NUM_ONE, chunkSize_ * sizeof(float));
        }

        pipe_->InitBuffer(tmpBuff_, UB_REST_BYTES);
        uint32_t buffOffset = 0;
        betaUbBf16_ = tmpBuff_.GetWithOffset<bfloat16_t>(static_cast<uint32_t>(halfChunkSize_), buffOffset);
        buffOffset += halfChunkSize_ * sizeof(bfloat16_t);

        gCumUbFloat_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(chunkSize_), buffOffset);
        buffOffset += chunkSize_ * sizeof(float);

        gBUbFloat_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(halfChunkSize_), buffOffset);
        buffOffset += halfChunkSize_ * sizeof(float);

        gEndBroadUbFloat_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(halfChunkSize_), buffOffset);
        buffOffset += halfChunkSize_ * sizeof(float);

        betaUbFloat_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(halfChunkSize_ * paraNum_), buffOffset);
        buffOffset += halfChunkSize_ * sizeof(float) * paraNum_;

        gBroadUbFloat_ =
            tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(subBlockElementNum_ * paraNum_), buffOffset);
        gammaUbFloat_ = gBroadUbFloat_;
        valueUbFloat_ = gBroadUbFloat_;
        qUbFloat_ = gBroadUbFloat_;
        buffOffset += subBlockElementNum_ * sizeof(float) * paraNum_;

        gTransBroadUbFloat_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(subBlockElementNum_), buffOffset);
        attnUbFloat_ = gTransBroadUbFloat_;
        gCumExpBroadUbFloat_ = gTransBroadUbFloat_;
        buffOffset += subBlockElementNum_ * sizeof(float);

        qUbFloatCon_ = gTransBroadUbFloat_;
        kUbFloatCon_ = qUbFloatCon_;

        inputGatherBuffer_ = tmpBuff_.GetWithOffset<uint32_t>(static_cast<uint32_t>(chunkSize_), buffOffset);
        buffOffset += chunkSize_ * sizeof(uint32_t);

        gCumSumUbFloat_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(chunkSize_ * paraNum_), buffOffset);
        buffOffset += chunkSize_ * sizeof(float) * paraNum_;

        gCumExpUbFloat_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(chunkSize_ * paraNum_), buffOffset);
        buffOffset += chunkSize_ * sizeof(float) * paraNum_;

        uint64_t inverseWorkElementNum = AscendC::Std::max(
            subBlockElementNum_, static_cast<uint64_t>(halfChunkSize_ * halfChunkSize_ * 3 + halfChunkSize_ * 2));
        inverseBuffer_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(inverseWorkElementNum), buffOffset);
        qPrimeUbFloat_ = inverseBuffer_;
        vBetaUbFloat_ = inverseBuffer_;
        gBKUbFloat_ = inverseBuffer_;
        kgUbFloat_ = inverseBuffer_;
        kkLocal_ = inverseBuffer_;
        buffOffset += inverseWorkElementNum * sizeof(float);

        inverseGatherBuffer_ = tmpBuff_.GetWithOffset<uint32_t>(static_cast<uint32_t>(halfChunkSize_), buffOffset);
        buffOffset += halfChunkSize_ * sizeof(uint32_t);

        inverseRes_ = gBroadUbFloat_;
    }

    __aicore__ inline void InitGatherBuffer()
    {
        for (uint32_t i = 0; i < chunkSize_; ++i) {
            inputGatherBuffer_.SetValue(i, i * BLOCK_SIZE);
        }
        for (uint32_t i = 0; i < halfChunkSize_; ++i) {
            inverseGatherBuffer_.SetValue<uint32_t>(i, (i * chunkSize_) * sizeof(float));
        }
        int32_t eventID = static_cast<int32_t>(pipe_->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventID);
        WaitFlag<HardEvent::S_V>(eventID);
    }

    __aicore__ inline void Init(const GDRStageOneInitParams &initParams, TPipe *pipe,
                                const ChunkGatedDeltaRuleC128TilingData *tilingData)
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
        aicLocalBuffersReady_ = false;
        packed64IdentityPrepared_ = false;
        preparedHalfIdentityBlockSize_ = 0;
        halfL1ZeroPreparedSize_ = 0;
        rawL0AbDbEventsReady_ = false;
        rawL0AbDbCursor_ = 0;
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
        AscendC::CrossCoreWaitFlag(0x9); // 同步0
        // key @ key.transpose(-1,-2)
        for (uint32_t i = 0; i < curParaNum; ++i) {
            AICProcess(keyConGm_[i * ckOffset_], keyConGm_[i * ckOffset_], kkWsGm_[i * ccOffset_],
                       {chunkSize_, chunkSize_, dk_, chunkSize_, chunkSize_, dk_}, true);
        }
        AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(0x8); // 同步1

        // query @ key.transpose(-1,-2)   stage1 out
        for (uint32_t i = 0; i < curParaNum; ++i) {
            AICProcess(queryConGm_[i * ckOffset_], keyConGm_[i * ckOffset_], outQkGm_[chunkRowBase_[i] * chunkSize_],
                       {validLenBatch_[i], validLenBatch_[i], dk_, validLenBatch_[i], validLenBatch_[i], dk_}, true);
        }
        AscendC::CrossCoreWaitFlag(0x7); // 同步2

        // 求逆左下角矩阵
        for (uint32_t i = 0; i < curParaNum; ++i) {
            if (validLenBatch_[i] >= chunkSize_) {
                C64InverseBy16x16Neumann(i, 0);
                C64InverseBy16x16Neumann(i, C64_PACKED_SIZE_U32);
                ComposeC128LowerLeftFp16(i);
            }
        }
        for (uint32_t i = 0; i < curParaNum; ++i) {
            if (validLenBatch_[i] > C64_PACKED_SIZE_U32 && validLenBatch_[i] < chunkSize_) {
                ComposeTailC128LowerLeftGm(i);
            }
        }
        AscendC::CrossCoreWaitFlag(0x6); // 同步3

        bool hasTail = ParaHasTail(curParaNum);

        // attn @ k_cumdecay
        for (uint32_t i = 0; i < curParaNum; ++i) {
            if (!hasTail || validLenBatch_[i] >= chunkSize_) {
                FusedRhsC128AICProcess(i, gBKWsGm_[i * ckOffset_], outKCumdecayGm_[chunkRowBase_[i] * dk_], dk_);
            }
        }
        AscendC::CrossCoreWaitFlag(0x5); // 同步4

        // attn @ v_beta    stage1 out
        for (uint32_t i = 0; i < curParaNum; ++i) {
            if (!hasTail || validLenBatch_[i] >= chunkSize_) {
                FusedRhsC128AICProcess(i, vBetaWsGm_[i * cvOffset_], outVInnerGm_[chunkRowBase_[i] * dv_], dv_);
            }
        }
        if (hasTail) {
            TailRhsC128AICProcess(curParaNum);
            RestoreStage1MatmulLocalBuffers();
        } else {
            RestoreStage1MatmulLocalBuffers();
        }
        AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(0xc); // 同步5：共享 workspace 已读完
    }

    __aicore__ inline bool ParaHasTail(int32_t curParaNum)
    {
        for (uint32_t i = 0; i < static_cast<uint32_t>(curParaNum); ++i) {
            if (validLenBatch_[i] < chunkSize_) {
                return true;
            }
        }
        return false;
    }

    __aicore__ inline void TailRhsC128AICProcess(int32_t curParaNum)
    {
        for (uint32_t i = 0; i < static_cast<uint32_t>(curParaNum); ++i) {
            if (validLenBatch_[i] >= chunkSize_) {
                continue;
            }
            TailRhsC128AICProcess(i, gBKWsGm_[i * ckOffset_], outKCumdecayGm_[chunkRowBase_[i] * dk_], dk_);
            TailRhsC128AICProcess(i, vBetaWsGm_[i * cvOffset_], outVInnerGm_[chunkRowBase_[i] * dv_], dv_);
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
            QKPreProcess(queryGm_[qk_base], queryConGm_[wsOffset_], outKgGm_, subValidLenBatch_[i]);
            QKPreProcess(keyGm_[qk_base], keyConGm_[wsOffset_], outKgGm_, subValidLenBatch_[i], true);
        }
        AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(0x9); // 同步0
        if (gOptional_) {
            for (uint32_t i = 0; i < curParaNum; ++i) {
                // g_cum_exp = g.cumsum(dim=-1).exp()
                GCumExpCompute(gGm_[bgOffsetBatch_[i]], outGCumExpGm_[chunkRowBase_[i]],
                               gCumSumUbFloat_[i * chunkSize_], gCumExpUbFloat_[i * chunkSize_], validLenBatch_[i]);
                // attn_1 = (g_cum_exp[:None] / g_cum_exp[None,:]) * mask
                uint64_t gUbOffset = i * subBlockElementNum_;
                GammaCompute(gBroadUbFloat_[gUbOffset], gammaUbFloat_[gUbOffset], gCumSumUbFloat_[i * chunkSize_]);
            }
        }

        for (uint32_t i = 0; i < curParaNum; ++i) {
            uint64_t betaUbOffset = i * halfChunkSize_;
            BetaCopyInWithStride(betaGm_[bgOffsetBatch_[i]], betaUbFloat_[betaUbOffset], subValidLenBatch_[i]);
        }
        AscendC::CrossCoreWaitFlag(0x8); // 同步1

        for (uint32_t i = 0; i < curParaNum; ++i) {
            // attn_1 = kkt * attn_1
            KKBetaCompute(kkWsGm_[i * ccOffset_], betaUbFloat_[i * halfChunkSize_]);
            // AIV prepares FP16 L and initial I-L for the two 64x64 diagonal inverses.
            InverseCompute(attnWsGm_[i * ccOffset_], L_powHalfWsGm_[i * ccOffset_], INVInitHalfWsGm_[i * ccOffset_],
                           gammaUbFloat_[i * subBlockElementNum_], validLenBatch_[i]);
        }
        AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(0x7); // 同步2

        for (uint32_t i = 0; i < curParaNum; ++i) {
            // kg = key * (g_cum_exp[-1, None] / g_cum_exp)[..., None]
            // k_cumdecay = -1.0 * k * beta * g_cum_exp
            outKgGm_ = outKgBaseGm_[chunkRowBase_[i] * dk_];
            uint64_t wsOffset_ = i * ckOffset_ + subOffset_ * dk_;
            GBKCompute(gBKWsGm_[i * ckOffset_], gBKHalfWsGm_[i * ckOffset_], outKgGm_,
                       betaUbFloat_[i * halfChunkSize_],
                       gCumSumUbFloat_[i * chunkSize_], gCumExpUbFloat_[i * chunkSize_], keyConGm_[wsOffset_]);
        }
        AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(0x6); // 同步3

        for (uint32_t i = 0; i < curParaNum; ++i) {
            // v_beta = value * beta.unsqueeze(-1)  # (C, Dv)
            uint64_t betaUbOffset = i * halfChunkSize_;
            uint64_t vOffset = chunkStartRowBatch_[i] * vRowStride_ + nIdBatch_[i] * dv_;
            uint64_t valueUbOffset = i * subBlockElementNum_;
            VBetaCompute(valueGm_[vOffset], vBetaWsGm_[i * cvOffset_], vBetaHalfWsGm_[i * cvOffset_],
                         betaUbFloat_[betaUbOffset],
                         valueUbFloat_[valueUbOffset], subValidLenBatch_[i]);
        }
        AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(0x5); // 同步4

        for (uint32_t i = 0; i < curParaNum; ++i) {
            // q_prime = query * scale_ * g_cum_exp[:, None]  # (C, Dk)
            uint64_t wsOffset_ = i * ckOffset_ + subOffset_ * dk_;
            QPrimeCompute(outQPrimeGm_[chunkRowBase_[i] * dk_], gCumSumUbFloat_[i * chunkSize_],
                          gCumExpUbFloat_[i * chunkSize_], queryConGm_[wsOffset_]);
        }
        AscendC::CrossCoreWaitFlag(0xc); // 同步5：避免下一批 parallel chunk 覆盖共享 workspace
    }
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
        if (!gOptional_ && kgFlag) {
            DataCopyPad(outKgGm[subOffset_ * dk_], tmpTensor, outParams);
        }
        outQueue_.FreeTensor(tmpTensor);
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
        uint64_t curVecLen = chunkSize_ * halfChunkSize_;
        uint32_t divShape[2] = {halfChunkSize_, chunkSize_};
        uint32_t gShape[2] = {halfChunkSize_, 1};
        uint32_t gTransShape[2] = {1, chunkSize_};
        Broadcast<float, BROADCAST_AXIS, 1>(gBroadUbFloat, gCumSumUbFloat[subOffset_], divShape, gShape);
        Broadcast<float, BROADCAST_AXIS, 0>(gTransBroadUbFloat_, gCumSumUbFloat, divShape, gTransShape);
        PipeBarrier<PIPE_V>();
        // div
        Sub(gammaUbFloat, gBroadUbFloat, gTransBroadUbFloat_, curVecLen);
        PipeBarrier<PIPE_V>();
        // mask  1
        DataCopyInFp32(curVecLen, stageOneMask_[GetBlockIdx() * ccOffset_ + subOffset_ * chunkSize_]);
        auto maskLocal = inQueue_.DeQue<float>();
        Mul(gammaUbFloat, gammaUbFloat, maskLocal, curVecLen);
        PipeBarrier<PIPE_V>();
        // exp
        Exp<float, 0, true>(gammaUbFloat, gammaUbFloat, curVecLen);
        PipeBarrier<PIPE_V>();
        // mask  2
        Mul(gammaUbFloat, gammaUbFloat, maskLocal, curVecLen);
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

    __aicore__ inline void InverseCompute(const GlobalTensor<bfloat16_t> dst, const GlobalTensor<half> lPowGm,
                                          const GlobalTensor<half> invGm, LocalTensor<float> gammaUbFloat,
                                          uint32_t validLen)
    {
        uint64_t curVecLen = chunkSize_ * halfChunkSize_;
        if (gOptional_) {
            Mul(attnUbFloat_, attnUbFloat_, gammaUbFloat, curVecLen);
        } else {
            DataCopyInFp32(curVecLen, stageOneMask_[GetBlockIdx() * ccOffset_ + subOffset_ * chunkSize_]);
            auto maskLocal = inQueue_.DeQue<float>();
            Mul(attnUbFloat_, attnUbFloat_, maskLocal, curVecLen);
            inQueue_.FreeTensor(maskLocal);
        }
        PipeBarrier<PIPE_V>();

        auto lPowLocal = outQueue_.AllocTensor<half>();
        Cast(lPowLocal, attnUbFloat_, RoundMode::CAST_NONE, curVecLen);
        PipeBarrier<PIPE_V>();
        outQueue_.EnQue(lPowLocal);
        DataCopyOutHalfRect(halfChunkSize_, halfChunkSize_, chunkSize_, chunkSize_, subOffset_,
                            lPowGm[subOffset_ * chunkSize_ + subOffset_]);

        BuildInitialOrExactInverse(validLen);

        auto invLocal = outQueue_.AllocTensor<half>();
        Cast(invLocal, inverseRes_, RoundMode::CAST_NONE, curVecLen);
        PipeBarrier<PIPE_V>();
        outQueue_.EnQue(invLocal);
        uint32_t invInitCols = (subBlockIdx_ == 0) ? halfChunkSize_ : chunkSize_;
        DataCopyOutHalfRect(halfChunkSize_, invInitCols, chunkSize_, chunkSize_, 0, invGm[subOffset_ * chunkSize_]);

        auto tmp = outQueue_.AllocTensor<bfloat16_t>();
        Cast(tmp, inverseRes_, RoundMode::CAST_RINT, curVecLen);
        outQueue_.EnQue(tmp);
        DataCopyOutBf16(halfChunkSize_, chunkSize_, chunkSize_, dst[subOffset_ * chunkSize_]);
    }

    __aicore__ inline void BuildInitialOrExactInverse(uint32_t validLen)
    {
        uint64_t curVecLen = chunkSize_ * halfChunkSize_;
        Muls(inverseRes_, attnUbFloat_, static_cast<float>(-1.0), curVecLen);
        PipeBarrier<PIPE_V>();
        if (validLen < chunkSize_) {
            InverseAIV(subOffset_, halfChunkSize_);
            return;
        }
        for (uint32_t row = 0; row < halfChunkSize_; ++row) {
            inverseRes_.SetValue(row * chunkSize_ + subOffset_ + row, static_cast<float>(1.0f));
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void InverseAIV(uint64_t offset, uint32_t inverseVecLen)
    {
        PipeBarrier<PIPE_V>();
        uint64_t inverseBufferOffset = 0;
        auto row = inverseBuffer_[inverseBufferOffset];
        inverseBufferOffset += inverseVecLen * inverseVecLen;
        auto col = inverseBuffer_[inverseBufferOffset];
        inverseBufferOffset += inverseVecLen * inverseVecLen + inverseVecLen;
        auto yLocal = inverseBuffer_[inverseBufferOffset];
        inverseBufferOffset += inverseVecLen * inverseVecLen;
        auto ei = inverseBuffer_[inverseBufferOffset];

        Duplicate(ei, static_cast<float>(0.0), inverseVecLen);
        Duplicate(yLocal, static_cast<float>(0.0), inverseVecLen * inverseVecLen);
        inverseRes_.SetValue(offset, static_cast<float>(1.0));

        uint32_t srcShape[2] = {1, inverseVecLen};
        int32_t eventID = static_cast<int32_t>(pipe_->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventID);
        WaitFlag<HardEvent::S_V>(eventID);
        for (int i = 1; i < inverseVecLen; ++i) {
            uint32_t curI = i - 1;
            uint32_t validRows = inverseVecLen - i;
            Gather(col, attnUbFloat_[offset + i * chunkSize_ + curI], inverseGatherBuffer_, static_cast<uint32_t>(0),
                   validRows);
            PipeBarrier<PIPE_V>();
            uint32_t dstShape[2] = {validRows, inverseVecLen};
            uint32_t colSrcShape[2] = {validRows, 1};
            Broadcast<float, BROADCAST_AXIS, 1>(col[inverseVecLen], col, dstShape, colSrcShape);
            Broadcast<float, BROADCAST_AXIS, 0>(row, inverseRes_[offset + curI * chunkSize_], dstShape, srcShape);
            PipeBarrier<PIPE_V>();
            MulAddDst(yLocal[i * inverseVecLen], col[inverseVecLen], row, inverseVecLen * validRows);
            PipeBarrier<PIPE_V>();
            ei.SetValue(i - 1, static_cast<float>(0.0));
            ei.SetValue(i, static_cast<float>(1.0));
            eventID = static_cast<int32_t>(pipe_->FetchEventID(HardEvent::S_V));
            SetFlag<HardEvent::S_V>(eventID);
            WaitFlag<HardEvent::S_V>(eventID);
            Sub(inverseRes_[offset + i * chunkSize_], ei, yLocal[i * inverseVecLen], inverseVecLen);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void GBKCompute(const GlobalTensor<bfloat16_t> gBKWsGm,
                                      const GlobalTensor<half> gBKHalfWsGm,
                                      const GlobalTensor<bfloat16_t> outKgGm, LocalTensor<float> betaUbFloat,
                                      LocalTensor<float> gCumSumUbFloat, LocalTensor<float> gCumExpUbFloat,
                                      const GlobalTensor<bfloat16_t> keyContinousGm)
    {
        if (gOptional_) {
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
        auto gBKHalfLocal = outQueue_.AllocTensor<half>();
        Cast(gBKHalfLocal, gBKUbFloat_, RoundMode::CAST_NONE, halfChunkSize_ * dkAligned_);
        outQueue_.EnQue<half>(gBKHalfLocal);
        DataCopyOutHalf(halfChunkSize_, dk_, dkAligned_, gBKHalfWsGm[gBKBeginOffset]);
        PipeBarrier<PIPE_V>();
        if (gOptional_) {
            // kg = k * (g_cum_exp[-1, None] / g_cum_exp)[..., None]
            uint32_t gEndShape[2] = {1, 1};
            uint32_t gBroadShape[2] = {halfChunkSize_, 1};
            Broadcast<float, BROADCAST_AXIS, 0>(gEndBroadUbFloat_, gCumSumUbFloat[chunkSize_ - 1], gBroadShape,
                                                gEndShape);
            PipeBarrier<PIPE_V>();
            Sub(gEndBroadUbFloat_, gEndBroadUbFloat_, gCumSumUbFloat[subOffset_], halfChunkSize_);
            PipeBarrier<PIPE_V>();
            Exp<float, 0, true>(gEndBroadUbFloat_, gEndBroadUbFloat_, halfChunkSize_);
            PipeBarrier<PIPE_V>();
            Broadcast<float, BROADCAST_AXIS, 1>(kgUbFloat_, gEndBroadUbFloat_, kShape, gBroadShape);
            PipeBarrier<PIPE_V>();
            Mul(kgUbFloat_, kgUbFloat_, kUbFloatCon_, halfChunkSize_ * dkAligned_);
            PipeBarrier<PIPE_V>();
            uint64_t kgBeginOffset = subOffset_ * dk_;
            kgLocal_ = outQueue_.AllocTensor<bfloat16_t>();
            Cast(kgLocal_, kgUbFloat_, RoundMode::CAST_RINT, halfChunkSize_ * dkAligned_);
            outQueue_.EnQue<bfloat16_t>(kgLocal_);
            DataCopyOutBf16(halfChunkSize_, dk_, dkAligned_, outKgGm[kgBeginOffset]); // stage1 out
        }
    }

    __aicore__ inline void VBetaCompute(const GlobalTensor<bfloat16_t> valueGm,
                                        const GlobalTensor<bfloat16_t> vBetaWsGm,
                                        const GlobalTensor<half> vBetaHalfWsGm,
                                        LocalTensor<float> betaUbFloat, LocalTensor<float> valueUbFloat,
                                        uint32_t subValidRows)
    {
        if (subValidRows == 0) {
            Duplicate(vBetaUbFloat_, static_cast<float>(0.0f), halfChunkSize_ * dvAligned_);
            PipeBarrier<PIPE_V>();
            vBetaLocal_ = outQueue_.AllocTensor<bfloat16_t>();
            Cast(vBetaLocal_, vBetaUbFloat_, RoundMode::CAST_RINT, halfChunkSize_ * dvAligned_);
            outQueue_.EnQue<bfloat16_t>(vBetaLocal_);
            DataCopyOutBf16(halfChunkSize_, dv_, dvAligned_, vBetaWsGm[subOffset_ * dv_]);
            auto vBetaHalfLocal = outQueue_.AllocTensor<half>();
            Cast(vBetaHalfLocal, vBetaUbFloat_, RoundMode::CAST_NONE, halfChunkSize_ * dvAligned_);
            outQueue_.EnQue<half>(vBetaHalfLocal);
            DataCopyOutHalf(halfChunkSize_, dv_, dvAligned_, vBetaHalfWsGm[subOffset_ * dv_]);
            return;
        }
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
        auto vBetaHalfLocal = outQueue_.AllocTensor<half>();
        Cast(vBetaHalfLocal, vBetaUbFloat_, RoundMode::CAST_NONE, halfChunkSize_ * dvAligned_);
        outQueue_.EnQue<half>(vBetaHalfLocal);
        DataCopyOutHalf(halfChunkSize_, dv_, dvAligned_, vBetaHalfWsGm[subOffset_ * dv_]);
    }

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
        if (gOptional_) {
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
        if (subValidRows == 0) {
            Duplicate(betaUbBf16_, bfloat16_t(0.0f), halfChunkSize_);
            PipeBarrier<PIPE_V>();
            Cast(betaUbFloat, betaUbBf16_, AscendC::RoundMode::CAST_NONE, halfChunkSize_);
            PipeBarrier<PIPE_V>();
            return;
        }
        uint64_t betaBeginOffset = subOffset_ * nv_;
        DataCopyInBf16WithStride(subValidRows, 1, src[betaBeginOffset], nv_);
        betaLocal_ = inQueue_.DeQue<bfloat16_t>();
        if (subValidRows < halfChunkSize_) {
            Duplicate(betaUbBf16_, bfloat16_t(0.0f), halfChunkSize_);
            PipeBarrier<PIPE_V>();
        }
        Gather(betaUbBf16_, betaLocal_, inputGatherBuffer_, static_cast<uint32_t>(0), subValidRows);
        PipeBarrier<PIPE_V>();

        Cast(betaUbFloat, betaUbBf16_, AscendC::RoundMode::CAST_NONE, halfChunkSize_);
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

    __aicore__ inline void DataCopyOutHalf(uint32_t rows, uint32_t cols, uint32_t colsAligned, GlobalTensor<half> y)
    {
        auto outLocal = outQueue_.DeQue<half>();
        uint32_t srcStride = (colsAligned - cols) * sizeof(half) / BLOCK_SIZE;
        DataCopyExtParams yGMParams{static_cast<uint16_t>(rows), static_cast<uint32_t>(cols * sizeof(half)),
                                    static_cast<uint32_t>(srcStride), 0, 0};
        DataCopyPad(y, outLocal, yGMParams);
        outQueue_.FreeTensor(outLocal);
    }

    __aicore__ inline void DataCopyOutBf16(uint32_t rows, uint32_t cols, uint32_t colsAligned,
                                           GlobalTensor<bfloat16_t> y)
    {
        auto outLocal = outQueue_.DeQue<bfloat16_t>();
        uint32_t srcStride = (colsAligned - cols) * sizeof(bfloat16_t) / BLOCK_SIZE;
        DataCopyExtParams yGMParams{static_cast<uint16_t>(rows), static_cast<uint32_t>(cols * sizeof(bfloat16_t)),
                                    static_cast<uint32_t>(srcStride), 0, 0};
        DataCopyPad(y, outLocal, yGMParams);
        outQueue_.FreeTensor(outLocal);
    }

    __aicore__ inline void DataCopyOutHalfRect(uint32_t rows, uint32_t cols, uint32_t srcColsAligned,
                                               uint32_t dstColsStride, uint32_t srcColOffset, GlobalTensor<half> y)
    {
        auto outLocal = outQueue_.DeQue<half>();
        uint32_t srcStride = (srcColsAligned - cols) * sizeof(half) / BLOCK_SIZE;
        uint32_t dstStride = (dstColsStride - cols) * sizeof(half);
        DataCopyExtParams yGMParams{static_cast<uint16_t>(rows), static_cast<uint32_t>(cols * sizeof(half)),
                                    static_cast<uint32_t>(srcStride), static_cast<uint32_t>(dstStride), 0};
        DataCopyPad(y, outLocal[srcColOffset], yGMParams);
        outQueue_.FreeTensor(outLocal);
    }

    __aicore__ inline uint64_t CompactMatrixNzOffset(uint64_t row, uint64_t col, uint64_t blockSize)
    {
        return (col / BLOCK_CUBE) * blockSize * BLOCK_CUBE + row * BLOCK_CUBE + (col % BLOCK_CUBE);
    }

    __aicore__ inline uint64_t L0MatrixBlockOffset(uint64_t row, uint64_t col, uint64_t blockSize)
    {
        return ((row / BLOCK_CUBE) * (blockSize / BLOCK_CUBE) + (col / BLOCK_CUBE)) * RAW_HALF_FRACTAL_ELEMS;
    }

    __aicore__ inline uint64_t MatrixOffset(uint64_t row, uint64_t col)
    {
        return row * chunkSize_ + col;
    }

    __aicore__ inline uint32_t AlignUp(uint32_t value, uint32_t align)
    {
        return (value + align - 1) / align * align;
    }

    __aicore__ inline void InitAicLocalBuffers()
    {
        pipe_->InitBuffer(aicL1ABuff_, AIC_RAW_L1_A_BYTES);
        pipe_->InitBuffer(aicL1BBuff_, AIC_RAW_L1_B_BYTES);
        pipe_->InitBuffer(aicHalfL1Buff_, AIC_LOW_L1_BYTES);
        aicL1A_ = aicL1ABuff_.GetWithOffset<float>(static_cast<uint32_t>(AIC_RAW_L1_WORK_BYTES / sizeof(float)), 0);
        aicL1B_ = aicL1BBuff_.Get<float>();

        uint32_t halfL1Offset = 0;
        aicHalfL1Scratch_ =
            aicHalfL1Buff_.GetWithOffset<half>(static_cast<uint32_t>(AIC_LOW_L1_TILE_ELEMS), halfL1Offset);
        halfL1Offset += static_cast<uint32_t>(AIC_LOW_L1_TILE_ELEMS * sizeof(half));
        aicHalfL1TileInv_ =
            aicHalfL1Buff_.GetWithOffset<half>(static_cast<uint32_t>(AIC_LOW_L1_TILE_ELEMS), halfL1Offset);
        halfL1Offset += static_cast<uint32_t>(AIC_LOW_L1_TILE_ELEMS * sizeof(half));
        aicHalfL1TilePower_ =
            aicHalfL1Buff_.GetWithOffset<half>(static_cast<uint32_t>(AIC_LOW_L1_TILE_ELEMS), halfL1Offset);
        halfL1Offset += static_cast<uint32_t>(AIC_LOW_L1_TILE_ELEMS * sizeof(half));
        aicHalfL1Identity_ =
            aicHalfL1Buff_.GetWithOffset<half>(static_cast<uint32_t>(AIC_LOW_L1_TILE_ELEMS), halfL1Offset);
        halfL1Offset += static_cast<uint32_t>(AIC_LOW_L1_TILE_ELEMS * sizeof(half));
        aicHalfL1Zero_ = aicHalfL1Buff_.GetWithOffset<half>(static_cast<uint32_t>(AIC_LOW_L1_TILE_ELEMS), halfL1Offset);
        halfL1Offset += static_cast<uint32_t>(AIC_LOW_L1_TILE_ELEMS * sizeof(half));
        aicHalfL1PackedInv64_ =
            aicHalfL1Buff_.GetWithOffset<half>(static_cast<uint32_t>(AIC_LOW_L1_TILE_ELEMS), halfL1Offset);
        halfL1Offset += static_cast<uint32_t>(AIC_LOW_L1_TILE_ELEMS * sizeof(half));
        aicHalfL1ResidentInv_ = aicHalfL1Buff_.GetWithOffset<half>(
            static_cast<uint32_t>(AIC_LOW_L1_TILE_ELEMS * C128_RESIDENT_TILE_COUNT * C128_RESIDENT_PARA_SLOTS),
            halfL1Offset);
        halfL1Offset += static_cast<uint32_t>(AIC_LOW_L1_TILE_ELEMS * C128_RESIDENT_TILE_COUNT *
                                              C128_RESIDENT_PARA_SLOTS * sizeof(half));
        aicHalfL1Identity64_ =
            aicHalfL1Buff_.GetWithOffset<half>(static_cast<uint32_t>(AIC_LOW_L1_TILE_ELEMS), halfL1Offset);
        halfL1Offset += static_cast<uint32_t>(AIC_LOW_L1_TILE_ELEMS * sizeof(half));
        aicBf16L1ResidentInv_ = aicHalfL1Buff_.GetWithOffset<bfloat16_t>(
            static_cast<uint32_t>(AIC_LOW_L1_TILE_ELEMS * C128_RESIDENT_TILE_COUNT * C128_RESIDENT_PARA_SLOTS),
            halfL1Offset);

        pipe_->InitBuffer(aicL0ABuff_, AIC_RAW_L0_AB_BYTES);
        pipe_->InitBuffer(aicL0BBuff_, AIC_RAW_L0_AB_BYTES);
        pipe_->InitBuffer(aicL0CBuff_, AIC_RAW_L0_C_BYTES);
        aicL0A_ = aicL0ABuff_.Get<float>();
        aicL0B_ = aicL0BBuff_.Get<float>();
        aicL0C_ = aicL0CBuff_.Get<float>();
        aicHalfL0A_ =
            aicL0ABuff_.GetWithOffset<half>(static_cast<uint32_t>(AIC_RAW_L0_AB_HALF_PINGPONG_BYTES / sizeof(half)), 0);
        aicHalfL0B_ =
            aicL0BBuff_.GetWithOffset<half>(static_cast<uint32_t>(AIC_RAW_L0_AB_HALF_PINGPONG_BYTES / sizeof(half)), 0);
        aicHalfL0AAlt_ =
            aicL0ABuff_.GetWithOffset<half>(static_cast<uint32_t>(AIC_RAW_L0_AB_HALF_PINGPONG_BYTES / sizeof(half)),
                                            static_cast<uint32_t>(AIC_RAW_L0_AB_HALF_PINGPONG_BYTES));
        aicHalfL0BAlt_ =
            aicL0BBuff_.GetWithOffset<half>(static_cast<uint32_t>(AIC_RAW_L0_AB_HALF_PINGPONG_BYTES / sizeof(half)),
                                            static_cast<uint32_t>(AIC_RAW_L0_AB_HALF_PINGPONG_BYTES));
    }

    __aicore__ inline void InitRawMmadLocalBuffers()
    {
        ReleaseRawL0AbDbEvents();
        pipe_->Reset();
        InitAicLocalBuffers();
        aicLocalBuffersReady_ = true;
        packed64IdentityPrepared_ = false;
        preparedHalfIdentityBlockSize_ = 0;
        halfL1ZeroPreparedSize_ = 0;
        rawL0AbDbCursor_ = 0;
        PrepareRawMmadAuxiliaryTiles();
    }

    __aicore__ inline void RestoreStage1MatmulLocalBuffers()
    {
        ReleaseRawL0AbDbEvents();
        pipe_->Reset();
        aicLocalBuffersReady_ = false;
        packed64IdentityPrepared_ = false;
        preparedHalfIdentityBlockSize_ = 0;
        halfL1ZeroPreparedSize_ = 0;
        rawL0AbDbCursor_ = 0;
        mm.Init(&tiling_->matmulTilingFp32, pipe_);
    }

    __aicore__ inline void EnsureAicLocalBuffers()
    {
        if (!aicLocalBuffersReady_) {
            InitRawMmadLocalBuffers();
        }
    }

    __aicore__ inline MmadParams MakeRawMmadParams(uint32_t mSize, uint32_t nSize, uint32_t kSize, bool initCMatrix,
                                                   uint8_t unitFlag)
    {
        MmadParams params(static_cast<uint16_t>(mSize), static_cast<uint16_t>(nSize), static_cast<uint16_t>(kSize),
                          unitFlag, RAW_MMAD_CMATRIX_SOURCE_L0C, initCMatrix);
        params.kDirectionAlign = false;
#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 3510) || (__NPU_ARCH__ == 5102))
        params.disableGemv = true;
#endif
        return params;
    }

    __aicore__ inline void CopyL0CMatrixToHalfL1(LocalTensor<half> l1, LocalTensor<float> l0c, uint32_t mSize,
                                                 uint32_t nSize, uint8_t unitFlag)
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

    __aicore__ inline void CopyL0CMatrixToPackedHalfL1(LocalTensor<half> l1, LocalTensor<float> l0c, uint32_t dstRow,
                                                       uint32_t dstCol, uint32_t mSize, uint32_t nSize,
                                                       uint32_t dstPackedSize, uint8_t unitFlag)
    {
        FixpipeParamsV220 fixParams;
        fixParams.nSize = nSize;
        fixParams.mSize = mSize;
        fixParams.srcStride = AlignUp(mSize, BLOCK_CUBE);
        fixParams.dstStride = AlignUp(dstPackedSize, BLOCK_CUBE) * BLOCK_CUBE * sizeof(half) / BLOCK_SIZE;
        fixParams.ndNum = 1;
        fixParams.srcNdStride = 0;
        fixParams.dstNdStride = 0;
        fixParams.reluEn = false;
        fixParams.quantPre = QuantMode_t::F322F16;
        fixParams.unitFlag = unitFlag;
        fixParams.isChannelSplit = false;
        Fixpipe<half, float, CFG_NZ>(l1[CompactMatrixNzOffset(dstRow, dstCol, dstPackedSize)], l0c, fixParams);
    }

    __aicore__ inline void FixpipePacked64FullToHalfL1(LocalTensor<half> l1, LocalTensor<float> l0c, uint8_t unitFlag)
    {
        FixpipeParamsV220 fixParams;
        fixParams.nSize = C64_PACKED_SIZE_U32;
        fixParams.mSize = C64_PACKED_SIZE_U32;
        fixParams.srcStride = C64_PACKED_SIZE_U32;
        fixParams.dstStride = C64_FIXPIPE_HALF_DST_STRIDE;
        fixParams.ndNum = 1;
        fixParams.srcNdStride = 0;
        fixParams.dstNdStride = 0;
        fixParams.reluEn = false;
        fixParams.quantPre = QuantMode_t::F322F16;
        fixParams.unitFlag = unitFlag;
        fixParams.isChannelSplit = false;
        Fixpipe<half, float, CFG_NZ>(l1[C64_DIAG_NZ_OFFSET0], l0c, fixParams);
    }

    __aicore__ inline void FixpipePacked64FullToBf16L1(LocalTensor<bfloat16_t> l1, LocalTensor<float> l0c,
                                                       uint8_t unitFlag)
    {
        FixpipeParamsV220 fixParams;
        fixParams.nSize = C64_PACKED_SIZE_U32;
        fixParams.mSize = C64_PACKED_SIZE_U32;
        fixParams.srcStride = C64_PACKED_SIZE_U32;
        fixParams.dstStride = C64_FIXPIPE_BF16_DST_STRIDE;
        fixParams.ndNum = 1;
        fixParams.srcNdStride = 0;
        fixParams.dstNdStride = 0;
        fixParams.reluEn = false;
        fixParams.quantPre = QuantMode_t::F322BF16;
        fixParams.unitFlag = unitFlag;
        fixParams.isChannelSplit = false;
        Fixpipe<bfloat16_t, float, CFG_NZ>(l1[C64_DIAG_NZ_OFFSET0], l0c, fixParams);
    }

    __aicore__ inline void CopyL0CMatrixToGmBf16(GlobalTensor<bfloat16_t> gm, LocalTensor<float> l0c, uint32_t mSize,
                                                 uint32_t nSize, uint32_t dstStride, uint8_t unitFlag)
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
        fixParams.quantPre = QuantMode_t::F322BF16;
        fixParams.unitFlag = unitFlag;
        Fixpipe<bfloat16_t, float, CFG_ROW_MAJOR>(gm, l0c, fixParams);
    }

    __aicore__ inline void LoadGmHalfPacked64DiagonalBlocksToPackedL1NoWait(GlobalTensor<half> gm, LocalTensor<half> l1,
                                                                            uint32_t srcStride)
    {
        Nd2NzParams copyParams;
        copyParams.ndNum = C64_PACKED_BLOCK_COUNT;
        copyParams.nValue = INVERSE_SHAPE;
        copyParams.dValue = INVERSE_SHAPE;
        copyParams.srcNdMatrixStride = MatrixOffset(INVERSE_SHAPE, INVERSE_SHAPE);
        copyParams.srcDValue = srcStride;
        copyParams.dstNzC0Stride = C64_PACKED_SIZE_U32;
        copyParams.dstNzNStride = 1;
        copyParams.dstNzMatrixStride = C64_DIAG_NZ_OFFSET1 - C64_DIAG_NZ_OFFSET0;
        DataCopy(l1[C64_DIAG_NZ_OFFSET0], gm, copyParams);
    }

    __aicore__ inline void LoadGmHalfPacked64FullToPackedL1NoWait(GlobalTensor<half> gm, LocalTensor<half> l1,
                                                                  uint32_t srcStride)
    {
        Nd2NzParams copyParams;
        copyParams.ndNum = 1;
        copyParams.nValue = C64_PACKED_SIZE_U32;
        copyParams.dValue = C64_PACKED_SIZE_U32;
        copyParams.srcNdMatrixStride = 0;
        copyParams.srcDValue = srcStride;
        copyParams.dstNzC0Stride = C64_PACKED_SIZE_U32;
        copyParams.dstNzNStride = 1;
        copyParams.dstNzMatrixStride = 0;
        DataCopy(l1, gm, copyParams);
    }

    __aicore__ inline void LoadGmHalfBlockToPackedHalfL1WithStrideNoWait(GlobalTensor<half> gm, LocalTensor<half> l1,
                                                                         uint32_t rowBase, uint32_t colBase,
                                                                         uint32_t rows, uint32_t cols,
                                                                         uint32_t srcStride, uint32_t packedRows)
    {
        Nd2NzParams copyParams;
        copyParams.ndNum = 1;
        copyParams.nValue = rows;
        copyParams.dValue = cols;
        copyParams.srcNdMatrixStride = 0;
        copyParams.srcDValue = srcStride;
        copyParams.dstNzC0Stride = packedRows;
        copyParams.dstNzNStride = 1;
        copyParams.dstNzMatrixStride = 0;
        DataCopy(l1[CompactMatrixNzOffset(rowBase, colBase, packedRows)], gm, copyParams);
    }

    __aicore__ inline void LoadGmBf16BlockToPackedBf16L1WithStrideNoWait(GlobalTensor<bfloat16_t> gm,
                                                                         LocalTensor<bfloat16_t> l1,
                                                                         uint32_t rowBase, uint32_t colBase,
                                                                         uint32_t rows, uint32_t cols,
                                                                         uint32_t srcStride, uint32_t packedRows)
    {
        Nd2NzParams copyParams;
        copyParams.ndNum = 1;
        copyParams.nValue = rows;
        copyParams.dValue = cols;
        copyParams.srcNdMatrixStride = 0;
        copyParams.srcDValue = srcStride;
        copyParams.dstNzC0Stride = packedRows;
        copyParams.dstNzNStride = 1;
        copyParams.dstNzMatrixStride = 0;
        DataCopy(l1[CompactMatrixNzOffset(rowBase, colBase, packedRows)], gm, copyParams);
    }

    __aicore__ inline void LoadGmBf16Packed64FullToPackedL1NoWait(GlobalTensor<bfloat16_t> gm,
                                                                  LocalTensor<bfloat16_t> l1,
                                                                  uint32_t srcStride)
    {
        LoadGmBf16BlockToPackedBf16L1WithStrideNoWait(gm, l1, 0, 0, C64_PACKED_SIZE_U32, C64_PACKED_SIZE_U32,
                                                      srcStride, C64_PACKED_SIZE_U32);
    }

    __aicore__ inline LocalTensor<bfloat16_t> GetRhsPacked64TopStageByIdx(uint32_t stageIdx)
    {
        return (stageIdx == 0) ? aicHalfL1TilePower_.template ReinterpretCast<bfloat16_t>()
                               : aicHalfL1Identity_.template ReinterpretCast<bfloat16_t>();
    }

    __aicore__ inline LocalTensor<bfloat16_t> GetRhsPacked64BottomStageByIdx(uint32_t stageIdx)
    {
        return (stageIdx == 0) ? aicHalfL1Scratch_.template ReinterpretCast<bfloat16_t>()
                               : aicHalfL1Identity64_.template ReinterpretCast<bfloat16_t>();
    }

    __aicore__ inline void PrepareRhsPacked64Bf16L1NoWait(GlobalTensor<bfloat16_t> rhsGm,
                                                           LocalTensor<bfloat16_t> rhsL1,
                                                           uint32_t tileCols, uint32_t rhsStride)
    {
        if (tileCols < C64_PACKED_SIZE_U32) {
            ZeroHalfL1(rhsL1.template ReinterpretCast<half>(), C64_PACKED_SIZE_U32 * C64_PACKED_SIZE_U32);
        }
        LoadGmBf16BlockToPackedBf16L1WithStrideNoWait(rhsGm, rhsL1, 0, 0, C64_PACKED_SIZE_U32, tileCols, rhsStride,
                                                      C64_PACKED_SIZE_U32);
    }

    __aicore__ inline void PrepareRhsPacked64Bf16L1(GlobalTensor<bfloat16_t> rhsGm, LocalTensor<bfloat16_t> rhsL1,
                                                    uint32_t tileCols, uint32_t rhsStride)
    {
        PrepareRhsPacked64Bf16L1NoWait(rhsGm, rhsL1, tileCols, rhsStride);
        SyncMte2ToMte1();
    }

    __aicore__ inline void ZeroHalfL1(LocalTensor<half> l1, uint32_t elems)
    {
        ZeroHalfL1NoSync(l1, elems);
        SyncMToMte1();
    }

    __aicore__ inline void ZeroHalfL1NoSync(LocalTensor<half> l1, uint32_t elems)
    {
        uint32_t bytes = elems * sizeof(half);
        InitConstValueParams<half> zeroParams(1, static_cast<uint16_t>(AlignUp(bytes, BLOCK_SIZE) / BLOCK_SIZE), 0,
                                              half(0.0f));
        InitConstValue(l1, zeroParams);
    }

    __aicore__ inline void ZeroPacked64UpperTriangularBlocksSingleSync(LocalTensor<half> l1)
    {
        ZeroHalfL1NoSync(l1[C64_DIAG_NZ_OFFSET1 - RAW_HALF_FRACTAL_ELEMS], RAW_HALF_FRACTAL_ELEMS);
        ZeroHalfL1NoSync(l1[C64_DIAG_NZ_OFFSET2 - 2 * RAW_HALF_FRACTAL_ELEMS], 2 * RAW_HALF_FRACTAL_ELEMS);
        ZeroHalfL1NoSync(l1[C64_DIAG_NZ_OFFSET3 - 3 * RAW_HALF_FRACTAL_ELEMS], 3 * RAW_HALF_FRACTAL_ELEMS);
        SyncMToMte1();
    }

    __aicore__ inline void EnsureHalfL1Zero(uint32_t elems)
    {
        if (halfL1ZeroPreparedSize_ >= elems) {
            return;
        }
        ZeroHalfL1(aicHalfL1Zero_, elems);
        halfL1ZeroPreparedSize_ = elems;
    }

    __aicore__ inline void EnsurePacked64Identity()
    {
        if (packed64IdentityPrepared_) {
            return;
        }

        ZeroHalfL1(aicHalfL1Identity64_, C64_PACKED_SIZE_U32 * C64_PACKED_SIZE_U32);
        GlobalTensor<half> identity16Nz4Gm;
        identity16Nz4Gm.SetGlobalBuffer((__gm__ half *)(CGDR_C128_IDENTITY16_NZ4));
#if defined(ASCENDC_OOM) && ASCENDC_OOM == 1
        AscendC::OOMCheckAddrRange((__gm__ uint8_t *)(CGDR_C128_IDENTITY16_NZ4),
                                   RAW_FRACTAL_BYTES * C64_PACKED_BLOCK_COUNT);
#endif
        DataCopyParams copyParams;
        copyParams.blockCount = C64_PACKED_BLOCK_COUNT;
        copyParams.blockLen = C64_DIAG_IDENTITY_COPY_BLOCKS;
        copyParams.srcStride = 0;
        copyParams.dstStride = C64_DIAG_IDENTITY_DST_GAP_BLOCKS;
        DataCopy(aicHalfL1Identity64_[C64_DIAG_NZ_OFFSET0], identity16Nz4Gm, copyParams);
        SyncMte2ToMte1();
        packed64IdentityPrepared_ = true;
    }

    __aicore__ inline void PrepareRawMmadAuxiliaryTiles()
    {
        EnsureHalfL1Zero(C64_PACKED_SIZE_U32 * C64_PACKED_SIZE_U32);
        EnsurePacked64Identity();
    }

    __aicore__ inline void CopyHalfL1MatrixToL0A(LocalTensor<half> l1, LocalTensor<half> l0, uint32_t blockSize)
    {
        uint32_t blockCount = AlignUp(blockSize, BLOCK_CUBE) / BLOCK_CUBE;
        LoadData2DParams loadParams;
        loadParams.startIndex = 0;
        loadParams.repeatTimes = static_cast<uint16_t>(blockCount);
        loadParams.srcStride = static_cast<uint16_t>(blockCount);
        loadParams.sid = 0;
        loadParams.dstGap = 0;
        loadParams.ifTranspose = false;
        loadParams.addrMode = 0;
        for (uint32_t rowBlock = 0; rowBlock < blockCount; ++rowBlock) {
            LoadData(l0[rowBlock * blockCount * RAW_HALF_FRACTAL_ELEMS], l1[rowBlock * RAW_HALF_FRACTAL_ELEMS],
                     loadParams);
        }
    }

    __aicore__ inline void CopyHalfL1MatrixToL0B(LocalTensor<half> l1, LocalTensor<half> l0, uint32_t blockSize)
    {
        uint32_t blockCount = AlignUp(blockSize, BLOCK_CUBE) / BLOCK_CUBE;
        LoadData2DParams loadParams;
        loadParams.startIndex = 0;
        loadParams.repeatTimes = static_cast<uint16_t>(blockCount);
        loadParams.srcStride = static_cast<uint16_t>(blockCount);
        loadParams.sid = 0;
        loadParams.dstGap = 0;
        loadParams.ifTranspose = true;
        loadParams.addrMode = 0;
        for (uint32_t kBlock = 0; kBlock < blockCount; ++kBlock) {
            LoadData(l0[kBlock * blockCount * RAW_HALF_FRACTAL_ELEMS], l1[kBlock * RAW_HALF_FRACTAL_ELEMS], loadParams);
        }
    }

    __aicore__ inline void CopyBf16L1MatrixToL0A(LocalTensor<bfloat16_t> l1, LocalTensor<bfloat16_t> l0,
                                                 uint32_t blockSize)
    {
        uint32_t blockCount = AlignUp(blockSize, BLOCK_CUBE) / BLOCK_CUBE;
        LoadData2DParams loadParams;
        loadParams.startIndex = 0;
        loadParams.repeatTimes = static_cast<uint16_t>(blockCount);
        loadParams.srcStride = static_cast<uint16_t>(blockCount);
        loadParams.sid = 0;
        loadParams.dstGap = 0;
        loadParams.ifTranspose = false;
        loadParams.addrMode = 0;
        for (uint32_t rowBlock = 0; rowBlock < blockCount; ++rowBlock) {
            LoadData(l0[rowBlock * blockCount * RAW_HALF_FRACTAL_ELEMS], l1[rowBlock * RAW_HALF_FRACTAL_ELEMS],
                     loadParams);
        }
    }

    __aicore__ inline void CopyPackedBf16L1RectToL0(LocalTensor<bfloat16_t> l1, LocalTensor<bfloat16_t> l0,
                                                    uint32_t srcRow, uint32_t srcCol, uint32_t rows, uint32_t cols,
                                                    uint32_t srcPackedSize, bool transpose)
    {
        uint32_t rowBlockCount = AlignUp(rows, BLOCK_CUBE) / BLOCK_CUBE;
        uint32_t colBlockCount = AlignUp(cols, BLOCK_CUBE) / BLOCK_CUBE;
        uint32_t srcBlockStride = AlignUp(srcPackedSize, BLOCK_CUBE) / BLOCK_CUBE;
        LoadData2DParams loadParams;
        loadParams.startIndex = 0;
        loadParams.repeatTimes = static_cast<uint16_t>(colBlockCount);
        loadParams.srcStride = static_cast<uint16_t>(srcBlockStride);
        loadParams.sid = 0;
        loadParams.dstGap = 0;
        loadParams.ifTranspose = transpose;
        loadParams.addrMode = 0;
        for (uint32_t rowBlock = 0; rowBlock < rowBlockCount; ++rowBlock) {
            uint32_t row = rowBlock * BLOCK_CUBE;
            uint32_t srcOffset =
                static_cast<uint32_t>(CompactMatrixNzOffset(srcRow + row, srcCol, srcPackedSize));
            uint32_t dstOffset = rowBlock * colBlockCount * RAW_HALF_FRACTAL_ELEMS;
            LoadData(l0[dstOffset], l1[srcOffset], loadParams);
        }
    }

    __aicore__ inline void LoadPacked16PairToL0OffsetWithGap(LocalTensor<half> l1, LocalTensor<half> l0,
                                                             uint32_t srcRow0, uint32_t srcCol0, uint32_t srcRow1,
                                                             uint32_t srcCol1, uint32_t srcPackedSize,
                                                             uint32_t dstOffset, uint32_t dstGap, bool transpose)
    {
        uint64_t srcOffset0 = CompactMatrixNzOffset(srcRow0, srcCol0, srcPackedSize);
        uint64_t srcOffset1 = CompactMatrixNzOffset(srcRow1, srcCol1, srcPackedSize);
        LoadData2DParams loadParams;
        loadParams.startIndex = 0;
        loadParams.repeatTimes = C32_PACKED_BLOCK_COUNT;
        loadParams.srcStride = static_cast<uint16_t>((srcOffset1 - srcOffset0) / RAW_HALF_FRACTAL_ELEMS);
        loadParams.sid = 0;
        loadParams.dstGap = static_cast<uint16_t>(dstGap);
        loadParams.ifTranspose = transpose;
        loadParams.addrMode = 0;
        LoadData(l0[dstOffset], l1[srcOffset0], loadParams);
    }

    __aicore__ inline void LoadOddLowerBlocksToFull64L0A(LocalTensor<half> l0)
    {
        LoadPacked16PairToL0OffsetWithGap(aicHalfL1Scratch_, l0, INVERSE_SHAPE, 0, C64_BLOCK32 + INVERSE_SHAPE,
                                          C64_BLOCK32, C64_PACKED_SIZE_U32, C64_L0_ROW1_COL0_OFFSET,
                                          C64_ODD_EVEN_PAIR_L0_LOAD_DST_GAP_FRACTALS, false);
    }

    __aicore__ inline void LoadEvenDiagonalInvBlocksToFull64L0B(LocalTensor<half> l0)
    {
        LoadPacked16PairToL0OffsetWithGap(aicHalfL1PackedInv64_, l0, 0, 0, C64_BLOCK32, C64_BLOCK32, C64_PACKED_SIZE_U32,
                                          C64_DIAG_NZ_OFFSET0, C64_ODD_EVEN_PAIR_L0_LOAD_DST_GAP_FRACTALS, true);
    }

    __aicore__ inline void LoadOddDiagonalInvBlocksToFull64L0A(LocalTensor<half> l0)
    {
        LoadPacked16PairToL0OffsetWithGap(aicHalfL1PackedInv64_, l0, INVERSE_SHAPE, INVERSE_SHAPE,
                                          C64_BLOCK32 + INVERSE_SHAPE, C64_BLOCK32 + INVERSE_SHAPE, C64_PACKED_SIZE_U32,
                                          C64_DIAG_NZ_OFFSET1, C64_ODD_EVEN_PAIR_L0_LOAD_DST_GAP_FRACTALS, false);
    }

    __aicore__ inline void LoadOddLowerTmpBlocksToFull64L0B(LocalTensor<half> tmp64, LocalTensor<half> l0)
    {
        LoadPacked16PairToL0OffsetWithGap(tmp64, l0, INVERSE_SHAPE, 0, C64_BLOCK32 + INVERSE_SHAPE, C64_BLOCK32,
                                          C64_PACKED_SIZE_U32, C64_L0_ROW1_COL0_OFFSET,
                                          C64_ODD_EVEN_PAIR_L0_LOAD_DST_GAP_FRACTALS, true);
    }

    template <uint32_t srcRow, uint32_t srcCol, uint32_t srcPackedSize, bool transpose>
    __aicore__ inline void LoadPacked32BlockToL0Fixed(LocalTensor<half> l1, LocalTensor<half> l0)
    {
        constexpr uint32_t srcBlockStride = srcPackedSize / C64_FRACTAL_U32;
        constexpr uint32_t srcOffset0 = (srcCol / C64_FRACTAL_U32) * srcPackedSize * C64_FRACTAL_U32 +
                                        srcRow * C64_FRACTAL_U32 + (srcCol % C64_FRACTAL_U32);
        constexpr uint32_t srcOffset1 = (srcCol / C64_FRACTAL_U32) * srcPackedSize * C64_FRACTAL_U32 +
                                        (srcRow + C64_FRACTAL_U32) * C64_FRACTAL_U32 + (srcCol % C64_FRACTAL_U32);
        LoadData2DParams loadParams;
        loadParams.startIndex = 0;
        loadParams.repeatTimes = C32_PACKED_BLOCK_COUNT;
        loadParams.srcStride = static_cast<uint16_t>(srcBlockStride);
        loadParams.sid = 0;
        loadParams.dstGap = 0;
        loadParams.ifTranspose = transpose;
        loadParams.addrMode = 0;
        LoadData(l0[0], l1[srcOffset0], loadParams);
        LoadData(l0[C32_PACKED_BLOCK_COUNT * RAW_HALF_FRACTAL_ELEMS], l1[srcOffset1], loadParams);
    }

    __aicore__ inline void LoadPacked32BlockToFull64L0(LocalTensor<half> l1, LocalTensor<half> l0, uint32_t srcRow,
                                                       uint32_t srcCol, uint32_t dstRow, uint32_t dstCol,
                                                       uint32_t srcPackedSize, bool transpose)
    {
        uint32_t srcBlockStride = AlignUp(srcPackedSize, BLOCK_CUBE) / BLOCK_CUBE;
        LoadData2DParams loadParams;
        loadParams.startIndex = 0;
        loadParams.repeatTimes = C32_PACKED_BLOCK_COUNT;
        loadParams.srcStride = static_cast<uint16_t>(srcBlockStride);
        loadParams.sid = 0;
        loadParams.dstGap = 0;
        loadParams.ifTranspose = transpose;
        loadParams.addrMode = 0;
        for (uint32_t rowBlock = 0; rowBlock < C32_PACKED_BLOCK_COUNT; ++rowBlock) {
            uint32_t rowOffset = rowBlock * C64_FRACTAL_U32;
            uint32_t srcOffset =
                static_cast<uint32_t>(CompactMatrixNzOffset(srcRow + rowOffset, srcCol, srcPackedSize));
            uint32_t dstOffset =
                static_cast<uint32_t>(L0MatrixBlockOffset(dstRow + rowOffset, dstCol, C64_PACKED_SIZE_U32));
            LoadData(l0[dstOffset], l1[srcOffset], loadParams);
        }
    }

    template <bool transpose>
    __aicore__ inline void LoadPacked64DiagOnlyToL0(LocalTensor<half> l1, LocalTensor<half> l0)
    {
        LoadData2DParams loadParams;
        loadParams.startIndex = 0;
        loadParams.repeatTimes = C64_PACKED_BLOCK_COUNT;
        loadParams.srcStride = C64_DIAG_NZ_STRIDE_FRACTALS;
        loadParams.sid = 0;
        loadParams.dstGap = C64_DIAG_L0_LOAD_DST_GAP_FRACTALS;
        loadParams.ifTranspose = transpose;
        loadParams.addrMode = 0;
        LoadData(l0[C64_DIAG_NZ_OFFSET0], l1[C64_DIAG_NZ_OFFSET0], loadParams);
    }

    template <bool transpose>
    __aicore__ inline void LoadPacked64DiagOnlyToL0PairedLoads(LocalTensor<half> l1, LocalTensor<half> l0)
    {
        LoadData2DParams loadParams;
        loadParams.startIndex = 0;
        loadParams.repeatTimes = 2;
        loadParams.srcStride = C64_DIAG_NZ_STRIDE_FRACTALS;
        loadParams.sid = 0;
        loadParams.dstGap = C64_DIAG_L0_LOAD_DST_GAP_FRACTALS;
        loadParams.ifTranspose = transpose;
        loadParams.addrMode = 0;
        LoadData(l0[C64_DIAG_NZ_OFFSET0], l1[C64_DIAG_NZ_OFFSET0], loadParams);
        LoadData(l0[C64_DIAG_NZ_OFFSET2], l1[C64_DIAG_NZ_OFFSET2], loadParams);
    }

    __aicore__ inline void LoadPacked64DiagOnlyToL0A(LocalTensor<half> l1, LocalTensor<half> l0)
    {
        LoadPacked64DiagOnlyToL0PairedLoads<false>(l1, l0);
    }

    __aicore__ inline void LoadPacked64DiagOnlyToL0B(LocalTensor<half> l1, LocalTensor<half> l0)
    {
        LoadPacked64DiagOnlyToL0<true>(l1, l0);
    }

    __aicore__ inline LocalTensor<half> GetPacked64L0AByDbIdx(uint32_t l0BufferIdx)
    {
        return (l0BufferIdx == 0) ? aicHalfL0A_ : aicHalfL0AAlt_;
    }

    __aicore__ inline LocalTensor<half> GetPacked64L0BByDbIdx(uint32_t l0BufferIdx)
    {
        return (l0BufferIdx == 0) ? aicHalfL0B_ : aicHalfL0BAlt_;
    }

    __aicore__ inline LocalTensor<bfloat16_t> GetPacked64Bf16L0AByDbIdx(uint32_t l0BufferIdx)
    {
        return GetPacked64L0AByDbIdx(l0BufferIdx).template ReinterpretCast<bfloat16_t>();
    }

    __aicore__ inline LocalTensor<bfloat16_t> GetPacked64Bf16L0BByDbIdx(uint32_t l0BufferIdx)
    {
        return GetPacked64L0BByDbIdx(l0BufferIdx).template ReinterpretCast<bfloat16_t>();
    }

    __aicore__ inline event_t RawL0AbFreeEvent(uint32_t l0BufferIdx)
    {
        return rawL0AbFreeEvents_[l0BufferIdx & 1U];
    }

    __aicore__ inline void InitRawL0AbDbEvents()
    {
        if (rawL0AbDbEventsReady_) {
            return;
        }
        rawL0AbFreeEvents_[0] = static_cast<event_t>(RAW_L0AB_DB_FREE_EVENT0);
        rawL0AbFreeEvents_[1] = static_cast<event_t>(RAW_L0AB_DB_FREE_EVENT1);
        SetFlag<HardEvent::M_MTE1>(rawL0AbFreeEvents_[0]);
        SetFlag<HardEvent::M_MTE1>(rawL0AbFreeEvents_[1]);
        rawL0AbDbEventsReady_ = true;
    }

    __aicore__ inline void ReleaseRawL0AbDbEvents()
    {
        if (!rawL0AbDbEventsReady_) {
            return;
        }
        WaitFlag<HardEvent::M_MTE1>(rawL0AbFreeEvents_[0]);
        WaitFlag<HardEvent::M_MTE1>(rawL0AbFreeEvents_[1]);
        rawL0AbDbEventsReady_ = false;
        rawL0AbDbCursor_ = 0;
    }

    __aicore__ inline uint32_t NextRawL0AbDbIdx()
    {
        uint32_t l0BufferIdx = rawL0AbDbCursor_;
        rawL0AbDbCursor_ ^= 1U;
        return l0BufferIdx;
    }

    __aicore__ inline void WaitRawL0AbDbFree(uint32_t l0BufferIdx)
    {
        InitRawL0AbDbEvents();
        WaitFlag<HardEvent::M_MTE1>(RawL0AbFreeEvent(l0BufferIdx));
    }

    __aicore__ inline void ReleaseRawL0AbDb(uint32_t l0BufferIdx)
    {
        SetFlag<HardEvent::M_MTE1>(RawL0AbFreeEvent(l0BufferIdx));
    }

    __aicore__ inline void MarkRawL0AbLoadReady(uint32_t l0BufferIdx)
    {
        SetFlag<HardEvent::MTE1_M>(RawL0AbFreeEvent(l0BufferIdx));
    }

    __aicore__ inline void WaitRawL0AbLoadReady(uint32_t l0BufferIdx)
    {
        WaitFlag<HardEvent::MTE1_M>(RawL0AbFreeEvent(l0BufferIdx));
    }

    __aicore__ inline void LoadPacked64DiagOnlyOperandsToL0(LocalTensor<half> lhsL1, LocalTensor<half> rhsL1,
                                                            uint32_t l0BufferIdx, bool clearL0 = true)
    {
        LocalTensor<half> l0A = GetPacked64L0AByDbIdx(l0BufferIdx);
        LocalTensor<half> l0B = GetPacked64L0BByDbIdx(l0BufferIdx);
        if (clearL0) {
            CopyHalfL1MatrixToL0A(aicHalfL1Zero_, l0A, C64_PACKED_SIZE_U32);
            CopyHalfL1MatrixToL0B(aicHalfL1Zero_, l0B, C64_PACKED_SIZE_U32);
        }
        LoadPacked64DiagOnlyToL0A(lhsL1, l0A);
        LoadPacked64DiagOnlyToL0B(rhsL1, l0B);
    }

    __aicore__ inline void IssuePacked64DiagOnlyOperandsToL0Db(LocalTensor<half> lhsL1, LocalTensor<half> rhsL1,
                                                               uint32_t l0BufferIdx, bool clearL0 = true)
    {
        WaitRawL0AbDbFree(l0BufferIdx);
        LoadPacked64DiagOnlyOperandsToL0(lhsL1, rhsL1, l0BufferIdx, clearL0);
        MarkRawL0AbLoadReady(l0BufferIdx);
    }

    __aicore__ inline void BeginMmadRawL0AbDb(LocalTensor<float> cTile, uint32_t l0BufferIdx, uint32_t mSize,
                                              uint32_t nSize, uint32_t kSize, bool initCMatrix, uint8_t unitFlag)
    {
        WaitRawL0AbLoadReady(l0BufferIdx);
        PipeBarrier<PIPE_M>();
        SetHF32Mode(HF32Mode::DISABLE);
        Mmad(cTile, GetPacked64L0AByDbIdx(l0BufferIdx), GetPacked64L0BByDbIdx(l0BufferIdx),
             MakeRawMmadParams(mSize, nSize, kSize, initCMatrix, unitFlag));
    }

    __aicore__ inline void BeginMmadRawL0AbDbBf16(LocalTensor<float> cTile, uint32_t l0BufferIdx, uint32_t mSize,
                                                  uint32_t nSize, uint32_t kSize, bool initCMatrix, uint8_t unitFlag)
    {
        WaitRawL0AbLoadReady(l0BufferIdx);
        PipeBarrier<PIPE_M>();
        SetHF32Mode(HF32Mode::DISABLE);
        Mmad(cTile, GetPacked64Bf16L0AByDbIdx(l0BufferIdx), GetPacked64Bf16L0BByDbIdx(l0BufferIdx),
             MakeRawMmadParams(mSize, nSize, kSize, initCMatrix, unitFlag));
    }

    __aicore__ inline void FinishMmadRawL0AbDb(uint32_t l0BufferIdx)
    {
        PipeBarrier<PIPE_M>();
        ReleaseRawL0AbDb(l0BufferIdx);
    }

    __aicore__ inline void BeginMmadRawL0AbDbPacked64(LocalTensor<float> cTile, uint32_t l0BufferIdx, bool initCMatrix,
                                                      uint8_t unitFlag)
    {
        BeginMmadRawL0AbDb(cTile, l0BufferIdx, C64_PACKED_SIZE_U32, C64_PACKED_SIZE_U32, C64_PACKED_SIZE_U32,
                           initCMatrix, unitFlag);
    }

    __aicore__ inline void MmadRawL0AbDbPacked64(LocalTensor<float> cTile, uint32_t l0BufferIdx, bool initCMatrix,
                                                 uint8_t unitFlag)
    {
        BeginMmadRawL0AbDbPacked64(cTile, l0BufferIdx, initCMatrix, unitFlag);
        FinishMmadRawL0AbDb(l0BufferIdx);
    }

    __aicore__ inline void MmadRawL0AbDbPacked32(LocalTensor<float> cTile, uint32_t l0BufferIdx, bool initCMatrix,
                                                 uint8_t unitFlag)
    {
        BeginMmadRawL0AbDb(cTile, l0BufferIdx, C32_PACKED_SIZE_U32, C32_PACKED_SIZE_U32, C32_PACKED_SIZE_U32,
                           initCMatrix, unitFlag);
        FinishMmadRawL0AbDb(l0BufferIdx);
    }

    __aicore__ inline void ZeroPacked64DiagOnlyL0Operands()
    {
        SyncMToMte1();
        CopyHalfL1MatrixToL0A(aicHalfL1Zero_, aicHalfL0A_, C64_PACKED_SIZE_U32);
        CopyHalfL1MatrixToL0B(aicHalfL1Zero_, aicHalfL0B_, C64_PACKED_SIZE_U32);
        CopyHalfL1MatrixToL0A(aicHalfL1Zero_, aicHalfL0AAlt_, C64_PACKED_SIZE_U32);
        CopyHalfL1MatrixToL0B(aicHalfL1Zero_, aicHalfL0BAlt_, C64_PACKED_SIZE_U32);
        SyncMte1ToM();
    }

    __aicore__ inline void RawMmadPacked64DiagOnly(LocalTensor<half> lhsL1, LocalTensor<half> rhsL1,
                                                   LocalTensor<float> cTile, bool initCMatrix, uint8_t unitFlag,
                                                   uint32_t l0BufferIdx = 0, bool clearL0 = true)
    {
        IssuePacked64DiagOnlyOperandsToL0Db(lhsL1, rhsL1, l0BufferIdx & 1U, clearL0);
        MmadRawL0AbDbPacked64(cTile, l0BufferIdx & 1U, initCMatrix, unitFlag);
    }

    __aicore__ inline void RawMmadPacked64DiagOnlyDbPair(LocalTensor<half> lhsL10, LocalTensor<half> rhsL10,
                                                         LocalTensor<half> lhsL11, LocalTensor<half> rhsL11,
                                                         LocalTensor<float> cTile, bool initCMatrix0, uint8_t unitFlag0,
                                                         bool initCMatrix1, uint8_t unitFlag1, bool clearL0 = true)
    {
        uint32_t l0BufferIdx0 = NextRawL0AbDbIdx();
        uint32_t l0BufferIdx1 = NextRawL0AbDbIdx();
        IssuePacked64DiagOnlyOperandsToL0Db(lhsL10, rhsL10, l0BufferIdx0, clearL0);
        IssuePacked64DiagOnlyOperandsToL0Db(lhsL11, rhsL11, l0BufferIdx1, clearL0);
        BeginMmadRawL0AbDbPacked64(cTile, l0BufferIdx0, initCMatrix0, unitFlag0);
        FinishMmadRawL0AbDb(l0BufferIdx0);
        BeginMmadRawL0AbDbPacked64(cTile, l0BufferIdx1, initCMatrix1, unitFlag1);
        FinishMmadRawL0AbDb(l0BufferIdx1);
    }

    __aicore__ inline void LoadPacked64FullOperandsToL0(LocalTensor<half> lhsL1, LocalTensor<half> rhsL1,
                                                        uint32_t l0BufferIdx)
    {
        LocalTensor<half> l0A = GetPacked64L0AByDbIdx(l0BufferIdx);
        LocalTensor<half> l0B = GetPacked64L0BByDbIdx(l0BufferIdx);
        CopyHalfL1MatrixToL0A(lhsL1, l0A, C64_PACKED_SIZE_U32);
        CopyHalfL1MatrixToL0B(rhsL1, l0B, C64_PACKED_SIZE_U32);
    }

    __aicore__ inline void LoadRhs64OperandsToL0(LocalTensor<half> lhsL1, LocalTensor<half> rhsL1, uint32_t tileCols,
                                                 uint32_t l0BufferIdx)
    {
        LocalTensor<half> l0A = GetPacked64L0AByDbIdx(l0BufferIdx);
        LocalTensor<half> l0B = GetPacked64L0BByDbIdx(l0BufferIdx);
        CopyHalfL1MatrixToL0A(lhsL1, l0A, C64_PACKED_SIZE_U32);
        if (tileCols < C64_PACKED_SIZE_U32) {
            CopyHalfL1MatrixToL0B(aicHalfL1Zero_, l0B, C64_PACKED_SIZE_U32);
        }
        CopyHalfL1MatrixToL0B(rhsL1, l0B, C64_PACKED_SIZE_U32);
    }

    __aicore__ inline void LoadRhs64Bf16OperandsToL0(LocalTensor<bfloat16_t> lhsL1, LocalTensor<bfloat16_t> rhsL1,
                                                     uint32_t lhsRows, uint32_t tileCols, uint32_t l0BufferIdx)
    {
        LocalTensor<bfloat16_t> l0A = GetPacked64Bf16L0AByDbIdx(l0BufferIdx);
        LocalTensor<bfloat16_t> l0B = GetPacked64Bf16L0BByDbIdx(l0BufferIdx);
        CopyPackedBf16L1RectToL0(lhsL1, l0A, 0, 0, lhsRows, C64_PACKED_SIZE_U32, C64_PACKED_SIZE_U32, false);
        CopyPackedBf16L1RectToL0(rhsL1, l0B, 0, 0, C64_PACKED_SIZE_U32, tileCols, C64_PACKED_SIZE_U32, true);
    }

    __aicore__ inline void IssueRhsPacked64Bf16RectToL0Db(LocalTensor<bfloat16_t> lhsL1,
                                                          LocalTensor<bfloat16_t> rhsL1, uint32_t tileCols,
                                                          uint32_t l0BufferIdx,
                                                          uint32_t lhsRows = C64_PACKED_SIZE_U32)
    {
        WaitRawL0AbDbFree(l0BufferIdx);
        LoadRhs64Bf16OperandsToL0(lhsL1, rhsL1, lhsRows, tileCols, l0BufferIdx);
        MarkRawL0AbLoadReady(l0BufferIdx);
    }

    __aicore__ inline void RawMmadRhs64Bf16(LocalTensor<bfloat16_t> lhsL1, LocalTensor<bfloat16_t> rhsL1,
                                           LocalTensor<float> cTile, uint32_t tileCols, bool initCMatrix,
                                           uint8_t unitFlag, uint32_t l0BufferIdx)
    {
        IssueRhsPacked64Bf16RectToL0Db(lhsL1, rhsL1, tileCols, l0BufferIdx);
        BeginMmadRawL0AbDbBf16(cTile, l0BufferIdx, C64_PACKED_SIZE_U32, tileCols, C64_PACKED_SIZE_U32, initCMatrix,
                               unitFlag);
        FinishMmadRawL0AbDb(l0BufferIdx);
    }

    __aicore__ inline void IssuePacked64FullOperandsToL0Db(LocalTensor<half> lhsL1, LocalTensor<half> rhsL1,
                                                           uint32_t l0BufferIdx)
    {
        WaitRawL0AbDbFree(l0BufferIdx);
        LoadPacked64FullOperandsToL0(lhsL1, rhsL1, l0BufferIdx);
        MarkRawL0AbLoadReady(l0BufferIdx);
    }

    __aicore__ inline void RawMmadPacked64Full(LocalTensor<half> lhsL1, LocalTensor<half> rhsL1,
                                               LocalTensor<float> cTile, bool initCMatrix, uint8_t unitFlag,
                                               uint32_t l0BufferIdx)
    {
        IssuePacked64FullOperandsToL0Db(lhsL1, rhsL1, l0BufferIdx & 1U);
        MmadRawL0AbDbPacked64(cTile, l0BufferIdx & 1U, initCMatrix, unitFlag);
    }

    __aicore__ inline void HalfL1PowerPacked64DiagOnlyOnC(LocalTensor<half> lhs, LocalTensor<half> rhs,
                                                          LocalTensor<half> dst, LocalTensor<float> cTile,
                                                          bool clearL0 = true)
    {
        RawMmadPacked64DiagOnly(lhs, rhs, cTile, RAW_MMAD_INIT_C, RAW_MMAD_FINAL_UNIT_FLAG, 0, clearL0);
        SyncMToFix();
        FixpipePacked64FullToHalfL1(dst, cTile, RAW_MMAD_FINAL_UNIT_FLAG);
        SyncFixToMte1();
    }

    __aicore__ inline void LoadOddLowerEvenInvFull64OperandsToL0(uint32_t l0BufferIdx)
    {
        LocalTensor<half> l0A = GetPacked64L0AByDbIdx(l0BufferIdx);
        LocalTensor<half> l0B = GetPacked64L0BByDbIdx(l0BufferIdx);
        CopyHalfL1MatrixToL0A(aicHalfL1Zero_, l0A, C64_PACKED_SIZE_U32);
        CopyHalfL1MatrixToL0B(aicHalfL1Zero_, l0B, C64_PACKED_SIZE_U32);
        LoadOddLowerBlocksToFull64L0A(l0A);
        LoadEvenDiagonalInvBlocksToFull64L0B(l0B);
    }

    __aicore__ inline void LoadOddInvOddTmpFull64OperandsToL0(LocalTensor<half> tmp64, uint32_t l0BufferIdx)
    {
        LocalTensor<half> l0A = GetPacked64L0AByDbIdx(l0BufferIdx);
        LocalTensor<half> l0B = GetPacked64L0BByDbIdx(l0BufferIdx);
        CopyHalfL1MatrixToL0A(aicHalfL1Zero_, l0A, C64_PACKED_SIZE_U32);
        CopyHalfL1MatrixToL0B(aicHalfL1Zero_, l0B, C64_PACKED_SIZE_U32);
        LoadOddDiagonalInvBlocksToFull64L0A(l0A);
        LoadOddLowerTmpBlocksToFull64L0B(tmp64, l0B);
    }

    __aicore__ inline void IssueOddLowerEvenInvFull64(uint32_t l0BufferIdx)
    {
        WaitRawL0AbDbFree(l0BufferIdx);
        LoadOddLowerEvenInvFull64OperandsToL0(l0BufferIdx);
        MarkRawL0AbLoadReady(l0BufferIdx);
    }

    __aicore__ inline void PrepareOddInvOddTmpFull64L0A(uint32_t l0BufferIdx)
    {
        WaitRawL0AbDbFree(l0BufferIdx);
        LocalTensor<half> l0A = GetPacked64L0AByDbIdx(l0BufferIdx);
        LocalTensor<half> l0B = GetPacked64L0BByDbIdx(l0BufferIdx);
        CopyHalfL1MatrixToL0A(aicHalfL1Zero_, l0A, C64_PACKED_SIZE_U32);
        CopyHalfL1MatrixToL0B(aicHalfL1Zero_, l0B, C64_PACKED_SIZE_U32);
        LoadOddDiagonalInvBlocksToFull64L0A(l0A);
    }

    __aicore__ inline void CompleteOddInvOddTmpFull64L0B(LocalTensor<half> tmp64, uint32_t l0BufferIdx)
    {
        LoadOddLowerTmpBlocksToFull64L0B(tmp64, GetPacked64L0BByDbIdx(l0BufferIdx));
        MarkRawL0AbLoadReady(l0BufferIdx);
    }

    __aicore__ inline void LoadOdd32LowerToFull64L0A(LocalTensor<half> l0)
    {
        LoadPacked32BlockToFull64L0(aicHalfL1Scratch_, l0, C64_BLOCK32, 0, C64_BLOCK32, 0, C64_PACKED_SIZE_U32, false);
    }

    __aicore__ inline void LoadEven32InvToFull64L0B(LocalTensor<half> l0)
    {
        LoadPacked32BlockToFull64L0(aicHalfL1PackedInv64_, l0, 0, 0, 0, 0, C64_PACKED_SIZE_U32, true);
    }

    __aicore__ inline void LoadOdd32InvToFull64L0A(LocalTensor<half> l0)
    {
        LoadPacked32BlockToFull64L0(aicHalfL1PackedInv64_, l0, C64_BLOCK32, C64_BLOCK32, C64_BLOCK32, C64_BLOCK32,
                                    C64_PACKED_SIZE_U32, false);
    }

    __aicore__ inline void LoadOdd32TmpToFull64L0B(LocalTensor<half> tmp64, LocalTensor<half> l0)
    {
        LoadPacked32BlockToFull64L0(tmp64, l0, C64_BLOCK32, 0, C64_BLOCK32, 0, C64_PACKED_SIZE_U32, true);
    }

    __aicore__ inline void LoadOdd32LowerEven32InvFull64OperandsToL0(uint32_t l0BufferIdx)
    {
        LocalTensor<half> l0A = GetPacked64L0AByDbIdx(l0BufferIdx);
        LocalTensor<half> l0B = GetPacked64L0BByDbIdx(l0BufferIdx);
        CopyHalfL1MatrixToL0A(aicHalfL1Zero_, l0A, C64_PACKED_SIZE_U32);
        CopyHalfL1MatrixToL0B(aicHalfL1Zero_, l0B, C64_PACKED_SIZE_U32);
        LoadOdd32LowerToFull64L0A(l0A);
        LoadEven32InvToFull64L0B(l0B);
    }

    __aicore__ inline void LoadOdd32InvOdd32TmpFull64OperandsToL0(LocalTensor<half> tmp64, uint32_t l0BufferIdx)
    {
        LocalTensor<half> l0A = GetPacked64L0AByDbIdx(l0BufferIdx);
        LocalTensor<half> l0B = GetPacked64L0BByDbIdx(l0BufferIdx);
        CopyHalfL1MatrixToL0A(aicHalfL1Zero_, l0A, C64_PACKED_SIZE_U32);
        CopyHalfL1MatrixToL0B(aicHalfL1Zero_, l0B, C64_PACKED_SIZE_U32);
        LoadOdd32InvToFull64L0A(l0A);
        LoadOdd32TmpToFull64L0B(tmp64, l0B);
    }

    __aicore__ inline void IssueOdd32LowerEven32InvFull64(uint32_t l0BufferIdx)
    {
        WaitRawL0AbDbFree(l0BufferIdx);
        LoadOdd32LowerEven32InvFull64OperandsToL0(l0BufferIdx);
        MarkRawL0AbLoadReady(l0BufferIdx);
    }

    __aicore__ inline void PrepareOdd32InvOdd32TmpFull64L0A(uint32_t l0BufferIdx)
    {
        WaitRawL0AbDbFree(l0BufferIdx);
        LocalTensor<half> l0A = GetPacked64L0AByDbIdx(l0BufferIdx);
        LocalTensor<half> l0B = GetPacked64L0BByDbIdx(l0BufferIdx);
        CopyHalfL1MatrixToL0A(aicHalfL1Zero_, l0A, C64_PACKED_SIZE_U32);
        CopyHalfL1MatrixToL0B(aicHalfL1Zero_, l0B, C64_PACKED_SIZE_U32);
        LoadOdd32InvToFull64L0A(l0A);
    }

    __aicore__ inline void CompleteOdd32InvOdd32TmpFull64L0B(LocalTensor<half> tmp64, uint32_t l0BufferIdx)
    {
        LoadOdd32TmpToFull64L0B(tmp64, GetPacked64L0BByDbIdx(l0BufferIdx));
        MarkRawL0AbLoadReady(l0BufferIdx);
    }

    __aicore__ inline void InvertPacked64DiagonalNeumannToResult()
    {
        ZeroPacked64DiagOnlyL0Operands();
        LocalTensor<float> cInv = aicL0C_;
        LocalTensor<float> cPower = aicL0C_[C64_L0C_TILE_ELEMS];
        LocalTensor<half> diagScratch64 = aicHalfL1TileInv_;
        LocalTensor<half> invInit64 = aicHalfL1Scratch_;

        HalfL1PowerPacked64DiagOnlyOnC(aicHalfL1TilePower_, aicHalfL1TilePower_, diagScratch64, cPower, false);
        RawMmadPacked64DiagOnlyDbPair(invInit64, aicHalfL1Identity64_, invInit64, diagScratch64, cInv, RAW_MMAD_INIT_C,
                                      RAW_MMAD_NON_FINAL_UNIT_FLAG, RAW_MMAD_ACCUMULATE_C, RAW_MMAD_FINAL_UNIT_FLAG,
                                      false);
        SyncMToFix();
        FixpipePacked64FullToHalfL1(aicHalfL1PackedInv64_, cInv, RAW_MMAD_FINAL_UNIT_FLAG);
        SyncFixToMte1();

        HalfL1PowerPacked64DiagOnlyOnC(diagScratch64, diagScratch64, aicHalfL1TilePower_, cPower, false);
        RawMmadPacked64DiagOnly(aicHalfL1PackedInv64_, aicHalfL1TilePower_, cInv, RAW_MMAD_ACCUMULATE_C,
                                RAW_MMAD_FINAL_UNIT_FLAG, 1, false);
        SyncMToFix();
        FixpipePacked64FullToHalfL1(aicHalfL1PackedInv64_, cInv, RAW_MMAD_FINAL_UNIT_FLAG);
        SyncFixToMte1();

        HalfL1PowerPacked64DiagOnlyOnC(aicHalfL1TilePower_, aicHalfL1TilePower_, diagScratch64, cPower, false);
        RawMmadPacked64DiagOnly(aicHalfL1PackedInv64_, diagScratch64, cInv, RAW_MMAD_ACCUMULATE_C,
                                RAW_MMAD_FINAL_UNIT_FLAG, 1, false);
        SyncMToFix();
        FixpipePacked64FullToHalfL1(aicHalfL1PackedInv64_, cInv, RAW_MMAD_FINAL_UNIT_FLAG);
        SyncFixToMte1();
    }

    __aicore__ inline void LoadPacked64DiagonalInputsToL1(uint32_t paraIdx, uint32_t diagBase)
    {
        uint64_t offset = paraIdx * ccOffset_ + MatrixOffset(diagBase, diagBase);
        LoadGmHalfPacked64DiagonalBlocksToPackedL1NoWait(L_powHalfWsGm_[offset], aicHalfL1TilePower_, chunkSize_);
        LoadGmHalfPacked64FullToPackedL1NoWait(INVInitHalfWsGm_[offset], aicHalfL1Scratch_, chunkSize_);
        SyncMte2ToMte1();
    }

    __aicore__ inline void ComposeOddEvenLower16To32AccumFull64(uint32_t paraIdx, uint32_t diagBase)
    {
        LocalTensor<float> cInv = aicL0C_;
        LocalTensor<float> cTmp = aicL0C_[C64_L0C_TILE_ELEMS];
        LocalTensor<half> tmp64 = aicHalfL1TileInv_;
        uint32_t firstL0BufferIdx = NextRawL0AbDbIdx();
        uint32_t secondL0BufferIdx = NextRawL0AbDbIdx();

        IssueOddLowerEvenInvFull64(firstL0BufferIdx);
        BeginMmadRawL0AbDbPacked64(cTmp, firstL0BufferIdx, RAW_MMAD_INIT_C, RAW_MMAD_FINAL_UNIT_FLAG);
        FinishMmadRawL0AbDb(firstL0BufferIdx);
        SyncMToFix();
        FixpipePacked64FullToHalfL1(tmp64, cTmp, RAW_MMAD_FINAL_UNIT_FLAG);
        SyncFixToMte1();
        PrepareOddInvOddTmpFull64L0A(secondL0BufferIdx);
        CompleteOddInvOddTmpFull64L0B(tmp64, secondL0BufferIdx);
        BeginMmadRawL0AbDbPacked64(cInv, secondL0BufferIdx, RAW_MMAD_ACCUMULATE_C, RAW_MMAD_FINAL_UNIT_FLAG);
        FinishMmadRawL0AbDb(secondL0BufferIdx);
        SyncMToFix();
        FixpipePacked64FullToHalfL1(aicHalfL1PackedInv64_, cInv, RAW_MMAD_FINAL_UNIT_FLAG);
        SyncFixToM();
        SyncFixToMte1();
    }

    __aicore__ inline void ComposeOddEvenLower32To64AccumFull64(uint32_t paraIdx, uint32_t diagBase)
    {
        LocalTensor<float> cInv = aicL0C_;
        LocalTensor<float> cTmp = aicL0C_[C64_L0C_TILE_ELEMS];
        LocalTensor<half> tmp64 = aicHalfL1TileInv_;
        uint32_t residentSlot = paraIdx * C128_RESIDENT_TILE_COUNT + diagBase / C64_PACKED_SIZE_U32;
        LocalTensor<half> resident = aicHalfL1ResidentInv_[static_cast<uint32_t>(residentSlot * AIC_LOW_L1_TILE_ELEMS)];
        LocalTensor<bfloat16_t> bf16Resident =
            aicBf16L1ResidentInv_[static_cast<uint32_t>(residentSlot * AIC_LOW_L1_TILE_ELEMS)];
        uint32_t firstL0BufferIdx = NextRawL0AbDbIdx();
        uint32_t secondL0BufferIdx = NextRawL0AbDbIdx();

        IssueOdd32LowerEven32InvFull64(firstL0BufferIdx);
        PrepareOdd32InvOdd32TmpFull64L0A(secondL0BufferIdx);
        BeginMmadRawL0AbDbPacked64(cTmp, firstL0BufferIdx, RAW_MMAD_INIT_C, RAW_MMAD_FINAL_UNIT_FLAG);
        FinishMmadRawL0AbDb(firstL0BufferIdx);
        SyncMToFix();
        FixpipePacked64FullToHalfL1(tmp64, cTmp, RAW_MMAD_FINAL_UNIT_FLAG);
        SyncFixToMte1();
        CompleteOdd32InvOdd32TmpFull64L0B(tmp64, secondL0BufferIdx);
        BeginMmadRawL0AbDbPacked64(cInv, secondL0BufferIdx, RAW_MMAD_ACCUMULATE_C, RAW_MMAD_FINAL_UNIT_FLAG);
        FinishMmadRawL0AbDb(secondL0BufferIdx);
        SyncMToFix();
        FixpipePacked64FullToHalfL1(resident, cInv, RAW_MMAD_NON_FINAL_UNIT_FLAG);
        SyncFixToMte1();
        FixpipePacked64FullToBf16L1(bf16Resident, cInv, RAW_MMAD_FINAL_UNIT_FLAG);
        SyncFixToMte1();
        ZeroPacked64UpperTriangularBlocksSingleSync(resident);
        ZeroPacked64UpperTriangularBlocksSingleSync(bf16Resident.template ReinterpretCast<half>());
        SyncFixToM();
        SyncFixToMte1();
    }

    __aicore__ inline void ComposeC128LowerLeftFp16(uint32_t paraIdx)
    {
        if (validLenBatch_[paraIdx] < chunkSize_) {
            return;
        }
        EnsureAicLocalBuffers();
        PrepareRawMmadState();

        uint64_t base = paraIdx * ccOffset_;
        uint32_t residentBase = paraIdx * C128_RESIDENT_TILE_COUNT;
        LocalTensor<half> topInv = aicHalfL1ResidentInv_[static_cast<uint32_t>(residentBase * AIC_LOW_L1_TILE_ELEMS)];
        LocalTensor<half> bottomInv =
            aicHalfL1ResidentInv_[static_cast<uint32_t>((residentBase + 1) * AIC_LOW_L1_TILE_ELEMS)];
        LocalTensor<half> lowerLeftNeg = aicHalfL1Scratch_;
        LocalTensor<half> tmp64 = aicHalfL1TileInv_;
        LocalTensor<float> cTmp = aicL0C_;
        LocalTensor<float> cOut = aicL0C_[C64_L0C_TILE_ELEMS];

        // INVInitHalfWsGm_ stores I - L, so the C128 lower-left tile is -L21.
        LoadGmHalfPacked64FullToPackedL1NoWait(INVInitHalfWsGm_[base + MatrixOffset(C64_PACKED_SIZE_U32, 0)],
                                               lowerLeftNeg, chunkSize_);
        SyncMte2ToMte1();
        RawMmadPacked64Full(lowerLeftNeg, topInv, cTmp, RAW_MMAD_INIT_C, RAW_MMAD_FINAL_UNIT_FLAG, NextRawL0AbDbIdx());
        SyncMToFix();
        FixpipePacked64FullToHalfL1(tmp64, cTmp, RAW_MMAD_FINAL_UNIT_FLAG);
        SyncFixToM();
        SyncFixToMte1();

        RawMmadPacked64Full(bottomInv, tmp64, cOut, RAW_MMAD_INIT_C, RAW_MMAD_FINAL_UNIT_FLAG, NextRawL0AbDbIdx());
        SyncMToFix();
        LocalTensor<bfloat16_t> lowerLeftBf16Resident =
            aicBf16L1ResidentInv_[static_cast<uint32_t>((residentBase + 2) * AIC_LOW_L1_TILE_ELEMS)];
        FixpipePacked64FullToBf16L1(lowerLeftBf16Resident, cOut, RAW_MMAD_FINAL_UNIT_FLAG);
        SyncFixToM();
        SyncFixToMte1();
    }

    __aicore__ inline void ComposeTailC128LowerLeftGm(uint32_t paraIdx)
    {
        EnsureAicLocalBuffers();
        PrepareRawMmadState();

        uint64_t base = paraIdx * ccOffset_;
        LocalTensor<half> lowerLeftNeg = aicHalfL1Scratch_;
        LocalTensor<half> topInv = aicHalfL1TilePower_;
        LocalTensor<half> bottomInv = aicHalfL1PackedInv64_;
        LocalTensor<half> tmp64 = aicHalfL1TileInv_;
        LocalTensor<float> cTmp = aicL0C_;
        LocalTensor<float> cOut = aicL0C_[C64_L0C_TILE_ELEMS];

        LoadGmHalfPacked64FullToPackedL1NoWait(INVInitHalfWsGm_[base + MatrixOffset(C64_PACKED_SIZE_U32, 0)],
                                               lowerLeftNeg, chunkSize_);
        LoadGmHalfPacked64FullToPackedL1NoWait(INVInitHalfWsGm_[base], topInv, chunkSize_);
        LoadGmHalfPacked64FullToPackedL1NoWait(
            INVInitHalfWsGm_[base + MatrixOffset(C64_PACKED_SIZE_U32, C64_PACKED_SIZE_U32)], bottomInv, chunkSize_);
        SyncMte2ToMte1();

        RawMmadPacked64Full(lowerLeftNeg, topInv, cTmp, RAW_MMAD_INIT_C, RAW_MMAD_FINAL_UNIT_FLAG, NextRawL0AbDbIdx());
        SyncMToFix();
        FixpipePacked64FullToHalfL1(tmp64, cTmp, RAW_MMAD_FINAL_UNIT_FLAG);
        SyncFixToM();
        SyncFixToMte1();

        RawMmadPacked64Full(bottomInv, tmp64, cOut, RAW_MMAD_INIT_C, RAW_MMAD_FINAL_UNIT_FLAG, NextRawL0AbDbIdx());
        SyncMToFix();
        CopyL0CMatrixToGmBf16(attnWsGm_[base + MatrixOffset(C64_PACKED_SIZE_U32, 0)], cOut, C64_PACKED_SIZE_U32,
                              C64_PACKED_SIZE_U32, chunkSize_, RAW_MMAD_FINAL_UNIT_FLAG);
        SyncFixToM();
        SyncFixToMte2();
        SyncFixToS();
    }

    __aicore__ inline uint32_t C64ValidLen(uint32_t paraIdx, uint32_t diagBase)
    {
        if (validLenBatch_[paraIdx] <= diagBase) {
            return 0;
        }
        uint32_t remain = validLenBatch_[paraIdx] - diagBase;
        return remain >= C64_PACKED_SIZE_U32 ? C64_PACKED_SIZE_U32 : remain;
    }

    __aicore__ inline void ComputeRhsPacked64Bf16Tiles(LocalTensor<bfloat16_t> topInv,
                                                       LocalTensor<bfloat16_t> bottomInv,
                                                       LocalTensor<bfloat16_t> lowerLeftInv,
                                                       GlobalTensor<bfloat16_t> rhsGm,
                                                       GlobalTensor<bfloat16_t> outGm, uint32_t rhsCols,
                                                       bool hasBottom,
                                                       uint32_t topRows = C64_PACKED_SIZE_U32,
                                                       uint32_t bottomRows = C64_PACKED_SIZE_U32)
    {
        LocalTensor<float> cTop = aicL0C_;
        LocalTensor<float> cBottom = aicL0C_[C64_L0C_TILE_ELEMS];
        uint32_t stageIdx = 0;
        uint32_t colBase = 0;
        uint32_t tileCols = rhsCols > C64_PACKED_SIZE_U32 ? C64_PACKED_SIZE_U32 : rhsCols;
        LocalTensor<bfloat16_t> rhsTop = GetRhsPacked64TopStageByIdx(stageIdx);
        LocalTensor<bfloat16_t> rhsBottom = GetRhsPacked64BottomStageByIdx(stageIdx);

        PrepareRhsPacked64Bf16L1NoWait(rhsGm[colBase], rhsTop, tileCols, rhsCols);
        if (hasBottom) {
            PrepareRhsPacked64Bf16L1NoWait(rhsGm[C64_PACKED_SIZE_U32 * rhsCols + colBase], rhsBottom, tileCols,
                                          rhsCols);
        }
        SyncMte2ToMte1();

        while (true) {
            uint32_t nextColBase = colBase + C64_PACKED_SIZE_U32;
            bool hasNextTile = nextColBase < rhsCols;
            uint32_t nextTileCols = 0;
            uint32_t nextStageIdx = stageIdx ^ 1U;

            if (hasNextTile) {
                nextTileCols = rhsCols - nextColBase;
                nextTileCols = nextTileCols > C64_PACKED_SIZE_U32 ? C64_PACKED_SIZE_U32 : nextTileCols;
                LocalTensor<bfloat16_t> nextRhsTop = GetRhsPacked64TopStageByIdx(nextStageIdx);
                LocalTensor<bfloat16_t> nextRhsBottom = GetRhsPacked64BottomStageByIdx(nextStageIdx);
                PrepareRhsPacked64Bf16L1NoWait(rhsGm[nextColBase], nextRhsTop, nextTileCols, rhsCols);
                if (hasBottom) {
                    PrepareRhsPacked64Bf16L1NoWait(rhsGm[C64_PACKED_SIZE_U32 * rhsCols + nextColBase], nextRhsBottom,
                                                   nextTileCols, rhsCols);
                }
            }

            uint32_t topL0BufferIdx = NextRawL0AbDbIdx();
            uint32_t bottomL0BufferIdx = NextRawL0AbDbIdx();
            IssueRhsPacked64Bf16RectToL0Db(topInv, rhsTop, tileCols, topL0BufferIdx, topRows);
            if (hasBottom) {
                IssueRhsPacked64Bf16RectToL0Db(bottomInv, rhsBottom, tileCols, bottomL0BufferIdx, bottomRows);
            }

            BeginMmadRawL0AbDbBf16(cTop, topL0BufferIdx, topRows, tileCols, C64_PACKED_SIZE_U32,
                                   RAW_MMAD_INIT_C, RAW_MMAD_FINAL_UNIT_FLAG);
            FinishMmadRawL0AbDb(topL0BufferIdx);
            SyncMToFix();
            CopyL0CMatrixToGmBf16(outGm[colBase], cTop, topRows, tileCols, rhsCols,
                                  RAW_MMAD_FINAL_UNIT_FLAG);
            SyncFixToM();
            SyncFixToMte2();

            if (hasBottom) {
                BeginMmadRawL0AbDbBf16(cBottom, bottomL0BufferIdx, bottomRows, tileCols, C64_PACKED_SIZE_U32,
                                       RAW_MMAD_INIT_C, RAW_MMAD_NON_FINAL_UNIT_FLAG);
                IssueRhsPacked64Bf16RectToL0Db(lowerLeftInv, rhsTop, tileCols, topL0BufferIdx, bottomRows);
                FinishMmadRawL0AbDb(bottomL0BufferIdx);
                BeginMmadRawL0AbDbBf16(cBottom, topL0BufferIdx, bottomRows, tileCols, C64_PACKED_SIZE_U32,
                                       RAW_MMAD_ACCUMULATE_C, RAW_MMAD_FINAL_UNIT_FLAG);
                FinishMmadRawL0AbDb(topL0BufferIdx);
                SyncMToFix();
                CopyL0CMatrixToGmBf16(outGm[C64_PACKED_SIZE_U32 * rhsCols + colBase], cBottom, bottomRows,
                                      tileCols, rhsCols, RAW_MMAD_FINAL_UNIT_FLAG);
                SyncFixToM();
                SyncFixToMte2();
            }

            if (!hasNextTile) {
                break;
            }
            SyncMte2ToMte1();
            colBase = nextColBase;
            tileCols = nextTileCols;
            stageIdx = nextStageIdx;
            rhsTop = GetRhsPacked64TopStageByIdx(stageIdx);
            rhsBottom = GetRhsPacked64BottomStageByIdx(stageIdx);
        }
        SyncFixToS();
    }

    __aicore__ inline void FusedRhsC128AICProcess(uint32_t paraIdx, GlobalTensor<bfloat16_t> rhsGm,
                                                  GlobalTensor<bfloat16_t> outGm, uint32_t rhsCols)
    {
        if (validLenBatch_[paraIdx] < chunkSize_) {
            return;
        }
        EnsureAicLocalBuffers();
        PrepareRawMmadState();

        uint32_t residentBase = paraIdx * C128_RESIDENT_TILE_COUNT;
        LocalTensor<bfloat16_t> topInv =
            aicBf16L1ResidentInv_[static_cast<uint32_t>(residentBase * AIC_LOW_L1_TILE_ELEMS)];
        LocalTensor<bfloat16_t> bottomInv =
            aicBf16L1ResidentInv_[static_cast<uint32_t>((residentBase + 1) * AIC_LOW_L1_TILE_ELEMS)];
        LocalTensor<bfloat16_t> lowerLeftInv =
            aicBf16L1ResidentInv_[static_cast<uint32_t>((residentBase + 2) * AIC_LOW_L1_TILE_ELEMS)];
        ComputeRhsPacked64Bf16Tiles(topInv, bottomInv, lowerLeftInv, rhsGm, outGm, rhsCols, true);
    }

    __aicore__ inline void TailRhsC128AICProcess(uint32_t paraIdx, GlobalTensor<bfloat16_t> rhsGm,
                                                 GlobalTensor<bfloat16_t> outGm, uint32_t rhsCols)
    {
        uint32_t tailLen = validLenBatch_[paraIdx];
        if (tailLen == 0 || tailLen >= chunkSize_) {
            return;
        }
        EnsureAicLocalBuffers();
        PrepareRawMmadState();

        uint64_t base = paraIdx * ccOffset_;
        LocalTensor<bfloat16_t> topInv = aicBf16L1ResidentInv_;
        LocalTensor<bfloat16_t> bottomInv =
            aicBf16L1ResidentInv_[static_cast<uint32_t>(AIC_LOW_L1_TILE_ELEMS)];
        LocalTensor<bfloat16_t> lowerLeftInv =
            aicBf16L1ResidentInv_[static_cast<uint32_t>(2 * AIC_LOW_L1_TILE_ELEMS)];

        LoadGmBf16Packed64FullToPackedL1NoWait(attnWsGm_[base], topInv, chunkSize_);
        if (tailLen > C64_PACKED_SIZE_U32) {
            LoadGmBf16Packed64FullToPackedL1NoWait(attnWsGm_[base + MatrixOffset(C64_PACKED_SIZE_U32,
                                                                                 C64_PACKED_SIZE_U32)],
                                                   bottomInv, chunkSize_);
            LoadGmBf16Packed64FullToPackedL1NoWait(attnWsGm_[base + MatrixOffset(C64_PACKED_SIZE_U32, 0)],
                                                   lowerLeftInv, chunkSize_);
        }
        SyncMte2ToMte1();

        bool hasBottom = tailLen > C64_PACKED_SIZE_U32;
        uint32_t topRows = hasBottom ? C64_PACKED_SIZE_U32 : AlignUp(tailLen, BLOCK_CUBE);
        uint32_t bottomRows = hasBottom ? AlignUp(tailLen - C64_PACKED_SIZE_U32, BLOCK_CUBE) : 0;
        ComputeRhsPacked64Bf16Tiles(topInv, bottomInv, lowerLeftInv, rhsGm, outGm, rhsCols,
                                    hasBottom, topRows, bottomRows);
    }

    __aicore__ inline void C64InverseBy16x16Neumann(uint32_t paraIdx, uint32_t diagBase)
    {
        if (C64ValidLen(paraIdx, diagBase) < C64_PACKED_SIZE_U32) {
            return;
        }
        EnsureAicLocalBuffers();
        PrepareRawMmadState();
        LoadPacked64DiagonalInputsToL1(paraIdx, diagBase);
        InvertPacked64DiagonalNeumannToResult();
        ComposeOddEvenLower16To32AccumFull64(paraIdx, diagBase);
        ComposeOddEvenLower32To64AccumFull64(paraIdx, diagBase);
    }

    __aicore__ inline void SyncScalarToMte2()
    {
        int32_t eventID = static_cast<int32_t>(pipe_->FetchEventID(HardEvent::S_MTE2));
        SetFlag<HardEvent::S_MTE2>(eventID);
        WaitFlag<HardEvent::S_MTE2>(eventID);
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
    }

    __aicore__ inline void PrepareRawMmadState()
    {
        SetLoadDataPaddingValue<uint64_t>(0);
        SetNdParaImpl(0x1);
        SetHF32Mode(HF32Mode::DISABLE);
        SetHF32TransMode(HF32TransMode::NEAREST_EVEN);
        SetMMLayoutTransform(false);
    }

    __aicore__ inline void AICProcess(GlobalTensor<bfloat16_t> x, GlobalTensor<bfloat16_t> y,
                                      GlobalTensor<bfloat16_t> z, const MatmulShapeParams &shape, bool transB = false)
    {
        mm.SetOrgShape(shape.m, shape.n, shape.k);
        mm.SetSingleShape(shape.sm, shape.sn, shape.sk);
        mm.SetTensorA(x);
        mm.SetTensorB(y, transB);
        mm.IterateAll(z);
        mm.End();
    }

    TPipe *pipe_;
    StageOneMT &mm;
    const ChunkGatedDeltaRuleC128TilingData *tiling_;
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
    uint32_t maxCols_;
    uint64_t subBlockElementNum_;
    uint32_t coreNum_;
    float scale_;
    bool gOptional_;
    bool aicLocalBuffersReady_ = false;
    bool packed64IdentityPrepared_ = false;
    uint32_t preparedHalfIdentityBlockSize_ = 0;
    uint32_t halfL1ZeroPreparedSize_ = 0;
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
    GlobalTensor<bfloat16_t> outVInnerGm_;
    GlobalTensor<bfloat16_t> outQPrimeGm_;
    GlobalTensor<bfloat16_t> outKgBaseGm_;
    GlobalTensor<bfloat16_t> outKgGm_;
    GlobalTensor<bfloat16_t> outQkGm_;
    GlobalTensor<bfloat16_t> vBetaWsGm_;
    GlobalTensor<half> vBetaHalfWsGm_;
    GlobalTensor<bfloat16_t> kkWsGm_;
    GlobalTensor<bfloat16_t> attnWsGm_;
    GlobalTensor<bfloat16_t> gBKWsGm_;
    GlobalTensor<half> gBKHalfWsGm_;
    GlobalTensor<bfloat16_t> queryConGm_;
    GlobalTensor<bfloat16_t> keyConGm_;
    GlobalTensor<float> stageOneMask_;
    GlobalTensor<half> L_powHalfWsGm_;
    GlobalTensor<half> INVInitHalfWsGm_;

    // UB queues
    TQue<QuePosition::VECIN, 1> inQueue_;
    TQue<QuePosition::VECOUT, 1> outQueue_;
    TQue<QuePosition::VECOUT, 1> gOutQueue_;

    TBuf<TPosition::VECCALC> tmpBuff_;
    TBuf<TPosition::A1> aicL1ABuff_;
    TBuf<TPosition::B1> aicL1BBuff_;
    TBuf<TPosition::TSCM> aicHalfL1Buff_;
    TBuf<TPosition::A2> aicL0ABuff_;
    TBuf<TPosition::B2> aicL0BBuff_;
    TBuf<TPosition::CO1> aicL0CBuff_;

    // UB tensors
    LocalTensor<bfloat16_t> betaUbBf16_;
    LocalTensor<float> betaUbFloat_;
    LocalTensor<float> valueUbFloat_;
    LocalTensor<float> attnUbFloat_;
    LocalTensor<float> inverseBuffer_;
    LocalTensor<float> inverseRes_;
    LocalTensor<float> aicL1A_;
    LocalTensor<float> aicL1B_;
    LocalTensor<float> aicL0A_;
    LocalTensor<float> aicL0B_;
    LocalTensor<float> aicL0C_;
    LocalTensor<half> aicHalfL1Scratch_;
    LocalTensor<half> aicHalfL1TileInv_;
    LocalTensor<half> aicHalfL1TilePower_;
    LocalTensor<half> aicHalfL1Identity_;
    LocalTensor<half> aicHalfL1Identity64_;
    LocalTensor<half> aicHalfL1Zero_;
    LocalTensor<half> aicHalfL1PackedInv64_;
    LocalTensor<half> aicHalfL1ResidentInv_;
    LocalTensor<bfloat16_t> aicBf16L1ResidentInv_;
    LocalTensor<half> aicHalfL0A_;
    LocalTensor<half> aicHalfL0B_;
    LocalTensor<half> aicHalfL0AAlt_;
    LocalTensor<half> aicHalfL0BAlt_;
    event_t rawL0AbFreeEvents_[2] = {static_cast<event_t>(0), static_cast<event_t>(0)};
    bool rawL0AbDbEventsReady_ = false;
    uint32_t rawL0AbDbCursor_ = 0;
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
    LocalTensor<uint32_t> inverseGatherBuffer_;

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
} // namespace ChunkGatedDeltaRuleC128
#endif // CHUNK_GATED_DELTA_RULE_C128_STAGE1_H
