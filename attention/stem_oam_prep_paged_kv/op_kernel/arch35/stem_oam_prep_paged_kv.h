/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file stem_oam_prep_paged_kv.h
 * \brief StemOamPrepPagedKV kernel implementation (arch35, split_mode=2)
 *
 * Algorithm (split_mode=2 — stem_block level flat distribution + kv_block gather):
 *   Phase 1: K Processing — per stem_block: de-page K rows → Cast FP8→FP32 → ×kScaleCache
 *            → group sum with anti-diagonal flip → Cast FP32→BF16 → write kFlat
 *   Phase 2: V Phase A — local sumLog/sumLogSq accumulation → write gm_stats
 *   SyncAll #1
 *   Phase 3: Aggregate μ/σ — 组头 core reads gm_stats, computes μ/inv_std → write gm_shared
 *   SyncAll #2
 *   Phase 4: V Phase C — read μ/inv_std → recompute logVals → Normalize → ReLU → ×λ
 *            → BlockAvg → write vBias
 */
#ifndef STEM_OAM_PREP_PAGED_KV_H
#define STEM_OAM_PREP_PAGED_KV_H

#include "kernel_operator.h"
#include "stem_oam_prep_paged_kv_tiling_data.h"

namespace StemOamPrepPagedKV {
using namespace AscendC;

constexpr uint32_t UB_ALIGN = 32;
constexpr float ZERO_FLOAT = 0.0f;
constexpr float EPSILON_FLOAT = 1e-6f;

class KernelStemPrepKV {
public:
    __aicore__ inline KernelStemPrepKV(TPipe *pipe, const StemPrepKVTilingData *__restrict tilingData)
        : pipe_(pipe), tilingData_(*tilingData)
    {
    }

    __aicore__ inline void Init(GM_ADDR kCache, GM_ADDR vCache, GM_ADDR kvIndices, GM_ADDR kvSeqLens,
                                GM_ADDR kScaleCache, GM_ADDR vScale, GM_ADDR kFlat, GM_ADDR vBias, GM_ADDR workspace);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitBuffers();
    __aicore__ inline void SetWsToZero();

    __aicore__ inline void ProcessKBlock(uint32_t batchIdx, uint32_t headIdx, uint32_t kb, uint32_t kvLen);
    __aicore__ inline void ProcessVPhaseA(uint32_t batchIdx, uint32_t headIdx, uint32_t kb, uint32_t kvLen,
                                          float &localSumLog, float &localSumLogSq);
    __aicore__ inline void AggregateMuSigma();
    __aicore__ inline void ProcessVPhaseC(uint32_t batchIdx, uint32_t headIdx, uint32_t kb, uint32_t kvLen, float mu,
                                          float invStd);

    __aicore__ inline uint64_t CalcKVCacheOffset(uint32_t physBlock, uint32_t rowInBlock, uint32_t headIdx);
    __aicore__ inline uint64_t CalcKScaleCacheOffset(uint32_t physBlock, uint32_t rowInBlock, uint32_t headIdx);
    __aicore__ inline uint32_t GetPhysicalBlock(uint32_t batchIdx, uint32_t rowIdx);
    __aicore__ inline uint32_t GetRowInKVBlock(uint32_t rowIdx);

    __aicore__ inline void DepageRowKV(GlobalTensor<fp8_e4m3fn_t> cacheGm, uint32_t physBlock, uint32_t rowInBlock,
                                       uint32_t headIdx);
    __aicore__ inline float ReadKScaleCache(uint32_t rowIdx, uint32_t headIdx);

    TPipe *pipe_;
    StemPrepKVTilingData tilingData_;

    GlobalTensor<fp8_e4m3fn_t> kCacheGm_;
    GlobalTensor<fp8_e4m3fn_t> vCacheGm_;
    GlobalTensor<int32_t> kvIndicesGm_;
    GlobalTensor<int32_t> kvSeqLensGm_;
    GlobalTensor<float> kScaleCacheGm_;
    GlobalTensor<float> vScaleGm_;
    GlobalTensor<bfloat16_t> kFlatGm_;
    GlobalTensor<float> vBiasGm_;
    GlobalTensor<float> gmStats_;
    GlobalTensor<float> gmShared_;

    TBuf<TPosition::VECCALC> ubMainBuf_;

    LocalTensor<fp8_e4m3fn_t> rawRowBuf_;
    LocalTensor<float> denseRowBuf_;
    LocalTensor<float> groupSumBuf_;
    LocalTensor<bfloat16_t> kflatBuf_;
    LocalTensor<uint8_t> reduceTmpBuf_;
    LocalTensor<float> vComputeBuf_;
    LocalTensor<float> normsBuf_;
    LocalTensor<float> vNormDownBuf_;
    LocalTensor<float> vProcSmallBuf_;
    LocalTensor<int32_t> kvIndicesBuf_;
    LocalTensor<float> logValsBuf_;
    LocalTensor<float> normalizedBuf_;

    uint32_t coreId_;
    float vScaleVal_;
    uint32_t flatStart_;
    uint32_t flatEnd_;
};

__aicore__ inline void KernelStemPrepKV::Init(GM_ADDR kCache, GM_ADDR vCache, GM_ADDR kvIndices, GM_ADDR kvSeqLens,
                                              GM_ADDR kScaleCache, GM_ADDR vScale, GM_ADDR kFlat, GM_ADDR vBias,
                                              GM_ADDR workspace)
{
    coreId_ = GetBlockIdx();

    kCacheGm_.SetGlobalBuffer((__gm__ fp8_e4m3fn_t *)kCache);
    vCacheGm_.SetGlobalBuffer((__gm__ fp8_e4m3fn_t *)vCache);
    kvIndicesGm_.SetGlobalBuffer((__gm__ int32_t *)kvIndices);
    kvSeqLensGm_.SetGlobalBuffer((__gm__ int32_t *)kvSeqLens);
    kScaleCacheGm_.SetGlobalBuffer((__gm__ float *)kScaleCache);
    vScaleGm_.SetGlobalBuffer((__gm__ float *)vScale);
    kFlatGm_.SetGlobalBuffer((__gm__ bfloat16_t *)kFlat);
    vBiasGm_.SetGlobalBuffer((__gm__ float *)vBias);

    uint32_t total_bh = tilingData_.batch_size * tilingData_.num_kv_heads;
    uint64_t statsBytes = static_cast<uint64_t>(tilingData_.total_stem_blocks) * 2 * sizeof(float);
    uint64_t sharedBytes = static_cast<uint64_t>(total_bh) * 2 * sizeof(float);
    gmStats_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace));
    gmShared_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace + statsBytes));

    InitBuffers();
    SetWsToZero();
}

__aicore__ inline void KernelStemPrepKV::InitBuffers()
{
    uint32_t totalUbBytes = 128 + 512 + 512 + 32 + 256 + 256 + 64 + 32 + 64 + 64 + MAX_KFLAT_DIM * sizeof(float) +
                            MAX_KFLAT_DIM * sizeof(bfloat16_t) + REDUCE_TMP_SIZE * sizeof(float);
    pipe_->InitBuffer(ubMainBuf_, totalUbBytes);

    uint32_t offset = 0;
    rawRowBuf_ = ubMainBuf_.GetWithOffset<fp8_e4m3fn_t>(DIM_QK, offset);
    offset += DIM_QK * sizeof(fp8_e4m3fn_t);

    denseRowBuf_ = ubMainBuf_.GetWithOffset<float>(DIM_QK, offset);
    offset += DIM_QK * sizeof(float);

    vComputeBuf_ = ubMainBuf_.GetWithOffset<float>(DIM_QK, offset);
    offset += DIM_QK * sizeof(float);

    vProcSmallBuf_ = ubMainBuf_.GetWithOffset<float>(8, offset);
    offset += 8 * sizeof(float);

    kvIndicesBuf_ = ubMainBuf_.GetWithOffset<int32_t>(KV_INDICES_BATCH_SIZE, offset);
    offset += KV_INDICES_BATCH_SIZE * sizeof(int32_t);

    normsBuf_ = ubMainBuf_.GetWithOffset<float>(MAX_STEM_STRIDE, offset);
    offset += MAX_STEM_STRIDE * sizeof(float);

    vNormDownBuf_ = ubMainBuf_.GetWithOffset<float>(MAX_R_VAL, offset);
    offset += MAX_R_VAL * sizeof(float);

    logValsBuf_ = ubMainBuf_.GetWithOffset<float>(MAX_R_VAL, offset);
    offset += MAX_R_VAL * sizeof(float);

    normalizedBuf_ = ubMainBuf_.GetWithOffset<float>(MAX_R_VAL, offset);
    offset += MAX_R_VAL * sizeof(float);

    groupSumBuf_ = ubMainBuf_.GetWithOffset<float>(MAX_KFLAT_DIM, offset);
    offset += MAX_KFLAT_DIM * sizeof(float);

    kflatBuf_ = ubMainBuf_.GetWithOffset<bfloat16_t>(MAX_KFLAT_DIM, offset);
    offset += MAX_KFLAT_DIM * sizeof(bfloat16_t);

    reduceTmpBuf_ = ubMainBuf_.GetWithOffset<uint8_t>(REDUCE_TMP_SIZE * sizeof(float), offset);
}

__aicore__ inline void KernelStemPrepKV::SetWsToZero()
{
    if (coreId_ >= tilingData_.used_core_num) {
        return;
    }

    Duplicate<float>(vProcSmallBuf_, ZERO_FLOAT, 8);
    DataCopyExtParams params{1, static_cast<uint32_t>(8 * sizeof(float)), 0, 0, 0};
    DataCopyPad(gmStats_[coreId_ * 2], vProcSmallBuf_, params);

    uint32_t total_bh = tilingData_.batch_size * tilingData_.num_kv_heads;
    Duplicate<float>(vProcSmallBuf_, ZERO_FLOAT, 8);
    DataCopyExtParams sharedParams{1, static_cast<uint32_t>(total_bh * 2 * sizeof(float)), 0, 0, 0};

#ifndef __CCE_KT_TEST__
    AscendC::SyncAll();
#endif
}

__aicore__ inline uint64_t KernelStemPrepKV::CalcKVCacheOffset(uint32_t physBlock, uint32_t rowInBlock,
                                                               uint32_t headIdx)
{
    uint64_t D = tilingData_.dim_qk;
    uint64_t kvBS = tilingData_.kv_block_size;
    uint64_t H = tilingData_.num_kv_heads;

    if (tilingData_.kv_layout == 0) {
        return physBlock * kvBS * H * D + rowInBlock * H * D + headIdx * D;
    } else {
        return physBlock * H * kvBS * D + headIdx * kvBS * D + rowInBlock * D;
    }
}

__aicore__ inline uint64_t KernelStemPrepKV::CalcKScaleCacheOffset(uint32_t physBlock, uint32_t rowInBlock,
                                                                   uint32_t headIdx)
{
    uint64_t kvBS = tilingData_.kv_block_size;
    uint64_t H = tilingData_.num_kv_heads;

    if (tilingData_.kv_layout == 0) {
        return physBlock * kvBS * H + rowInBlock * H + headIdx;
    } else {
        return physBlock * H * kvBS + headIdx * kvBS + rowInBlock;
    }
}

__aicore__ inline uint32_t KernelStemPrepKV::GetPhysicalBlock(uint32_t batchIdx, uint32_t rowIdx)
{
    uint32_t logicalKVBlock = rowIdx / tilingData_.kv_block_size;
    uint32_t maxKVBlocks = tilingData_.max_kv_blocks_size;
    return static_cast<uint32_t>(
        kvIndicesGm_.GetValue(static_cast<int64_t>(batchIdx) * static_cast<int64_t>(maxKVBlocks) + logicalKVBlock));
}

__aicore__ inline uint32_t KernelStemPrepKV::GetRowInKVBlock(uint32_t rowIdx)
{
    return rowIdx % tilingData_.kv_block_size;
}

__aicore__ inline void KernelStemPrepKV::DepageRowKV(GlobalTensor<fp8_e4m3fn_t> cacheGm, uint32_t physBlock,
                                                     uint32_t rowInBlock, uint32_t headIdx)
{
    uint64_t offset = CalcKVCacheOffset(physBlock, rowInBlock, headIdx);
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(DIM_QK * sizeof(fp8_e4m3fn_t)), 0, 0, 0};
    DataCopyPadExtParams<fp8_e4m3fn_t> padParams{false, 0, 0, 0};
    DataCopyPad<fp8_e4m3fn_t, PaddingMode::Normal>(rawRowBuf_, cacheGm[offset], copyParams, padParams);
}

__aicore__ inline float KernelStemPrepKV::ReadKScaleCache(uint32_t rowIdx, uint32_t headIdx)
{
    uint32_t logicalKVBlock = rowIdx / tilingData_.kv_block_size;
    uint32_t maxKVBlocks = tilingData_.max_kv_blocks_size;
    uint32_t physBlock =
        static_cast<uint32_t>(kvIndicesGm_.GetValue(static_cast<int64_t>(rowIdx / tilingData_.kv_block_size)));

    int32_t blockId =
        kvIndicesGm_.GetValue(static_cast<int64_t>(0) * static_cast<int64_t>(maxKVBlocks) + logicalKVBlock);
    physBlock = static_cast<uint32_t>(blockId);

    uint32_t rowInBlock = GetRowInKVBlock(rowIdx);
    uint64_t kscaleOffset = CalcKScaleCacheOffset(physBlock, rowInBlock, headIdx);
    return kScaleCacheGm_.GetValue(static_cast<int64_t>(kscaleOffset));
}

__aicore__ inline void KernelStemPrepKV::ProcessKBlock(uint32_t batchIdx, uint32_t headIdx, uint32_t kb, uint32_t kvLen)
{
    uint32_t B = tilingData_.stem_block_size;
    uint32_t S = tilingData_.stem_stride;
    uint32_t R = tilingData_.R_val;
    uint32_t D = tilingData_.dim_qk;
    uint32_t kflatDim = tilingData_.kflat_dim;

    Duplicate<float>(groupSumBuf_, ZERO_FLOAT, kflatDim);

    for (uint32_t r = 0; r < R; r++) {
        for (uint32_t g = 0; g < S; g++) {
            uint32_t rowIdx = kb * B + r * S + g;

            if (rowIdx >= kvLen) {
                continue;
            }

            uint32_t physBlock = GetPhysicalBlock(batchIdx, rowIdx);
            uint32_t rowInBlock = GetRowInKVBlock(rowIdx);

            DepageRowKV(kCacheGm_, physBlock, rowInBlock, headIdx);

            event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

            Cast<float, fp8_e4m3fn_t>(denseRowBuf_, rawRowBuf_, RoundMode::CAST_NONE, D);

            PipeBarrier<PIPE_V>();

            float kscaleVal = ReadKScaleCache(rowIdx, headIdx);
            Muls<float>(vComputeBuf_, denseRowBuf_, kscaleVal, D);

            PipeBarrier<PIPE_V>();

            uint32_t flipOffset = (S - 1 - g) * D;
            Add<float>(groupSumBuf_[flipOffset], groupSumBuf_[flipOffset], vComputeBuf_, D);

            PipeBarrier<PIPE_V>();
        }
    }

    Cast<bfloat16_t, float>(kflatBuf_, groupSumBuf_, RoundMode::CAST_RINT, kflatDim);

    PipeBarrier<PIPE_V>();

    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);

    uint64_t kFlatOffset = static_cast<uint64_t>(batchIdx) * tilingData_.num_kv_heads * tilingData_.max_Kb * kflatDim +
                           static_cast<uint64_t>(headIdx) * tilingData_.max_Kb * kflatDim +
                           static_cast<uint64_t>(kb) * kflatDim;
    DataCopyExtParams outParams{1, static_cast<uint32_t>(kflatDim * sizeof(bfloat16_t)), 0, 0, 0};
    DataCopyPad<bfloat16_t, PaddingMode::Normal>(kFlatGm_[kFlatOffset], kflatBuf_, outParams);

    event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
}

__aicore__ inline void KernelStemPrepKV::ProcessVPhaseA(uint32_t batchIdx, uint32_t headIdx, uint32_t kb,
                                                        uint32_t kvLen, float &localSumLog, float &localSumLogSq)
{
    uint32_t B = tilingData_.stem_block_size;
    uint32_t S = tilingData_.stem_stride;
    uint32_t R = tilingData_.R_val;
    uint32_t D = tilingData_.dim_qk;

    for (uint32_t idxOff = 0; idxOff < R; idxOff++) {
        float maxNorm = ZERO_FLOAT;

        for (uint32_t s = 0; s < S; s++) {
            uint32_t rowIdx = kb * B + idxOff * S + s;

            if (rowIdx >= kvLen) {
                continue;
            }

            uint32_t physBlock = GetPhysicalBlock(batchIdx, rowIdx);
            uint32_t rowInBlock = GetRowInKVBlock(rowIdx);

            DepageRowKV(vCacheGm_, physBlock, rowInBlock, headIdx);

            event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

            Cast<float, fp8_e4m3fn_t>(denseRowBuf_, rawRowBuf_, RoundMode::CAST_NONE, D);
            PipeBarrier<PIPE_V>();

            Muls<float>(vComputeBuf_, denseRowBuf_, vScaleVal_, D);
            PipeBarrier<PIPE_V>();

            Mul<float>(vComputeBuf_, vComputeBuf_, vComputeBuf_, D);
            PipeBarrier<PIPE_V>();

            uint32_t srcShape[2] = {1, D};
            LocalTensor<float> normDst = vProcSmallBuf_;
            AscendC::ReduceSum<float, Pattern::Reduce::AR, true>(normDst, vComputeBuf_, srcShape, true);
            PipeBarrier<PIPE_V>();

            Sqrt<float>(normDst, normDst, 1);
            PipeBarrier<PIPE_V>();

            float normVal = normDst.GetValue(0);
            if (normVal > maxNorm) {
                maxNorm = normVal;
            }
        }

        if (maxNorm > ZERO_FLOAT) {
            float logVal;
            LocalTensor<float> logDst = vProcSmallBuf_;
            Adds<float>(logDst, vProcSmallBuf_, EPSILON_FLOAT, 1);
            PipeBarrier<PIPE_V>();
            Log<float>(logDst, logDst, 1);
            PipeBarrier<PIPE_V>();

            vNormDownBuf_.SetValue(idxOff, maxNorm);
            LocalTensor<float> epsBuf = vProcSmallBuf_;
            Adds<float>(epsBuf, vNormDownBuf_[idxOff], EPSILON_FLOAT, 1);
            PipeBarrier<PIPE_V>();
            Log<float>(logValsBuf_[idxOff], epsBuf, 1);
            PipeBarrier<PIPE_V>();

            logVal = logValsBuf_.GetValue(idxOff);
            localSumLog += logVal;
            localSumLogSq += logVal * logVal;
        } else {
            float logVal = log(EPSILON_FLOAT);
            logValsBuf_.SetValue(idxOff, logVal);
            localSumLog += logVal;
            localSumLogSq += logVal * logVal;
        }
    }
}

__aicore__ inline void KernelStemPrepKV::AggregateMuSigma()
{
    if (coreId_ >= tilingData_.used_core_num) {
        return;
    }

    uint32_t maxKb = tilingData_.max_Kb;
    uint32_t numKVHeads = tilingData_.num_kv_heads;
    uint32_t batchSize = tilingData_.batch_size;

    for (uint32_t flatIdx = flatStart_; flatIdx < flatEnd_; flatIdx++) {
        uint32_t stemBlockIdx = flatIdx % maxKb;

        if (stemBlockIdx != 0) {
            continue;
        }

        uint32_t curBatchIdx = flatIdx / (numKVHeads * maxKb);
        uint32_t curHeadIdx = (flatIdx % (numKVHeads * maxKb)) / maxKb;

        int32_t kvLen = kvSeqLensGm_.GetValue(static_cast<int64_t>(curBatchIdx));
        uint32_t numKb = (static_cast<uint32_t>(kvLen) + tilingData_.stem_block_size - 1) / tilingData_.stem_block_size;
        if (kvLen == 0) {
            numKb = 0;
        }
        uint32_t kDownLen = numKb * tilingData_.R_val;

        float globalSumLog = ZERO_FLOAT;
        float globalSumLogSq = ZERO_FLOAT;

        for (uint32_t kb = 0; kb < numKb; kb++) {
            uint32_t targetFlat = curBatchIdx * numKVHeads * maxKb + curHeadIdx * maxKb + kb;
            uint32_t statsOffset = targetFlat * 2;
            float partialSumLog = gmStats_.GetValue(static_cast<int64_t>(statsOffset));
            float partialSumLogSq = gmStats_.GetValue(static_cast<int64_t>(statsOffset + 1));
            globalSumLog += partialSumLog;
            globalSumLogSq += partialSumLogSq;
        }

        float mu = ZERO_FLOAT;
        float invStd = ZERO_FLOAT;

        if (kDownLen > 0) {
            mu = globalSumLog / static_cast<float>(kDownLen);
            float variance = (globalSumLogSq - globalSumLog * globalSumLog / static_cast<float>(kDownLen)) /
                             static_cast<float>(kDownLen > 1 ? kDownLen - 1 : 1);
            if (variance < ZERO_FLOAT) {
                variance = ZERO_FLOAT;
            }
            LocalTensor<float> sqrtBuf = vProcSmallBuf_;
            Sqrt<float>(sqrtBuf, sqrtBuf, 1);
            PipeBarrier<PIPE_V>();

            float sigma = sqrtf(variance);
            invStd = 1.0f / (sigma + EPSILON_FLOAT);
        }

        uint32_t bhIdx = curBatchIdx * numKVHeads + curHeadIdx;
        gmShared_.SetValue(static_cast<int64_t>(bhIdx * 2), mu);
        gmShared_.SetValue(static_cast<int64_t>(bhIdx * 2 + 1), invStd);
    }
}

__aicore__ inline void KernelStemPrepKV::ProcessVPhaseC(uint32_t batchIdx, uint32_t headIdx, uint32_t kb,
                                                        uint32_t kvLen, float mu, float invStd)
{
    uint32_t B = tilingData_.stem_block_size;
    uint32_t S = tilingData_.stem_stride;
    uint32_t R = tilingData_.R_val;
    uint32_t D = tilingData_.dim_qk;
    float LambdaMag = tilingData_.lambda_mag;

    for (uint32_t idxOff = 0; idxOff < R; idxOff++) {
        float maxNorm = ZERO_FLOAT;

        for (uint32_t s = 0; s < S; s++) {
            uint32_t rowIdx = kb * B + idxOff * S + s;

            if (rowIdx >= kvLen) {
                continue;
            }

            uint32_t physBlock = GetPhysicalBlock(batchIdx, rowIdx);
            uint32_t rowInBlock = GetRowInKVBlock(rowIdx);

            DepageRowKV(vCacheGm_, physBlock, rowInBlock, headIdx);

            event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

            Cast<float, fp8_e4m3fn_t>(denseRowBuf_, rawRowBuf_, RoundMode::CAST_NONE, D);
            PipeBarrier<PIPE_V>();

            Muls<float>(vComputeBuf_, denseRowBuf_, vScaleVal_, D);
            PipeBarrier<PIPE_V>();

            Mul<float>(vComputeBuf_, vComputeBuf_, vComputeBuf_, D);
            PipeBarrier<PIPE_V>();

            uint32_t srcShape[2] = {1, D};
            LocalTensor<float> normDst = vProcSmallBuf_;
            AscendC::ReduceSum<float, Pattern::Reduce::AR, true>(normDst, vComputeBuf_, srcShape, true);
            PipeBarrier<PIPE_V>();

            Sqrt<float>(normDst, normDst, 1);
            PipeBarrier<PIPE_V>();

            float normVal = normDst.GetValue(0);
            if (normVal > maxNorm) {
                maxNorm = normVal;
            }
        }

        if (maxNorm > ZERO_FLOAT) {
            LocalTensor<float> epsBuf = vProcSmallBuf_;
            vNormDownBuf_.SetValue(idxOff, maxNorm);
            Adds<float>(epsBuf, vNormDownBuf_[idxOff], EPSILON_FLOAT, 1);
            PipeBarrier<PIPE_V>();
            Log<float>(logValsBuf_[idxOff], epsBuf, 1);
            PipeBarrier<PIPE_V>();
        } else {
            logValsBuf_.SetValue(idxOff, log(EPSILON_FLOAT));
        }
    }

    Subs<float>(normalizedBuf_, logValsBuf_, mu, R);
    PipeBarrier<PIPE_V>();

    Muls<float>(normalizedBuf_, normalizedBuf_, invStd, R);
    PipeBarrier<PIPE_V>();

    LocalTensor<float> reluBuf = normalizedBuf_;
    Relu<float>(reluBuf, normalizedBuf_, R);
    PipeBarrier<PIPE_V>();

    Muls<float>(normalizedBuf_, reluBuf, lambdaMag, R);
    PipeBarrier<PIPE_V>();

    float vBiasVal = ZERO_FLOAT;
    for (uint32_t i = 0; i < R; i++) {
        vBiasVal += normalizedBuf_.GetValue(i);
    }
    vBiasVal /= static_cast<float>(R);

    uint64_t vBiasOffset = static_cast<uint64_t>(batchIdx) * tilingData_.num_kv_heads * tilingData_.vbias_stride +
                           static_cast<uint64_t>(headIdx) * tilingData_.vbias_stride + static_cast<uint64_t>(kb);
    ((__gm__ float *)vBiasGm_.GetGlobalBuffer())[vBiasOffset] = vBiasVal;
}

__aicore__ inline void KernelStemPrepKV::Process()
{
    if (coreId_ >= tilingData_.used_core_num) {
        return;
    }

    uint32_t stemBlocksPerCore = tilingData_.stem_blocks_per_core;
    uint32_t totalStemBlocks = tilingData_.total_stem_blocks;
    uint32_t maxKb = tilingData_.max_Kb;
    uint32_t numKVHeads = tilingData_.num_kv_heads;
    uint32_t batchSize = tilingData_.batch_size;

    flatStart_ = coreId_ * stemBlocksPerCore;
    flatEnd_ = flatStart_ + stemBlocksPerCore;
    if (flatEnd_ > totalStemBlocks) {
        flatEnd_ = totalStemBlocks;
    }

    float localSumLog = ZERO_FLOAT;
    float localSumLogSq = ZERO_FLOAT;

    for (uint32_t flatIdx = flatStart_; flatIdx < flatEnd_; flatIdx++) {
        uint32_t curBatchIdx = flatIdx / (numKVHeads * maxKb);
        uint32_t curHeadIdx = (flatIdx % (numKVHeads * maxKb)) / maxKb;
        uint32_t curKb = flatIdx % maxKb;

        int32_t kvLen = kvSeqLensGm_.GetValue(static_cast<int64_t>(curBatchIdx));
        uint32_t numKb = (static_cast<uint32_t>(kvLen) + tilingData_.stem_block_size - 1) / tilingData_.stem_block_size;
        if (kvLen <= 0) {
            numKb = 0;
        }

        vScaleVal_ = vScaleGm_.GetValue(static_cast<int64_t>(curHeadIdx));

        if (curKb >= numKb) {
            uint32_t kflatDim = tilingData_.kflat_dim;
            uint64_t kFlatOffset = static_cast<uint64_t>(curBatchIdx) * numKVHeads * maxKb * kflatDim +
                                   static_cast<uint64_t>(curHeadIdx) * maxKb * kflatDim +
                                   static_cast<uint64_t>(curKb) * kflatDim;
            Duplicate<bfloat16_t>(kflatBuf_, static_cast<bfloat16_t>(0), kflatDim);
            DataCopyExtParams outParams{1, static_cast<uint32_t>(kflatDim * sizeof(bfloat16_t)), 0, 0, 0};
            DataCopyPad<bfloat16_t, PaddingMode::Normal>(kFlatGm_[kFlatOffset], kflatBuf_, outParams);
            continue;
        }

        ProcessKBlock(curBatchIdx, curHeadIdx, curKb, static_cast<uint32_t>(kvLen));

        ProcessVPhaseA(curBatchIdx, curHeadIdx, curKb, static_cast<uint32_t>(kvLen), localSumLog, localSumLogSq);
    }

    gmStats_.SetValue(static_cast<int64_t>(coreId_ * 2), localSumLog);
    gmStats_.SetValue(static_cast<int64_t>(coreId_ * 2 + 1), localSumLogSq);

#ifndef __CCE_KT_TEST__
    AscendC::SyncAll();
#endif

    AggregateMuSigma();

#ifndef __CCE_KT_TEST__
    AscendC::SyncAll();
#endif

    for (uint32_t flatIdx = flatStart_; flatIdx < flatEnd_; flatIdx++) {
        uint32_t curBatchIdx = flatIdx / (numKVHeads * maxKb);
        uint32_t curHeadIdx = (flatIdx % (numKVHeads * maxKb)) / maxKb;
        uint32_t curKb = flatIdx % maxKb;

        int32_t kvLen = kvSeqLensGm_.GetValue(static_cast<int64_t>(curBatchIdx));
        uint32_t numKb = (static_cast<uint32_t>(kvLen) + tilingData_.stem_block_size - 1) / tilingData_.stem_block_size;
        if (kvLen <= 0) {
            numKb = 0;
        }

        if (curKb >= numKb) {
            continue;
        }

        uint32_t bhIdx = curBatchIdx * numKVHeads + curHeadIdx;
        float mu = gmShared_.GetValue(static_cast<int64_t>(bhIdx * 2));
        float invStd = gmShared_.GetValue(static_cast<int64_t>(bhIdx * 2 + 1));

        vScaleVal_ = vScaleGm_.GetValue(static_cast<int64_t>(curHeadIdx));

        ProcessVPhaseC(curBatchIdx, curHeadIdx, curKb, static_cast<uint32_t>(kvLen), mu, invStd);
    }
}

} // namespace StemOamPrepPagedKV
#endif // STEM_OAM_PREP_PAGED_KV_H
