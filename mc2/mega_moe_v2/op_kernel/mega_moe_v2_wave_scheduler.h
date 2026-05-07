/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* DeepGEMM-style wave scheduler for V2: persistent (expert, mBlock, nBlock, phase) iterator
   shared by AIC + AIV. Mirrors deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh. */

#ifndef MEGA_MOE_V2_WAVE_SCHEDULER_H
#define MEGA_MOE_V2_WAVE_SCHEDULER_H

#include "kernel_operator.h"
#include "mega_moe_v2_base.h"

using namespace AscendC;

namespace MegaMoeV2Impl {

enum class BlockPhase : uint8_t {
    Linear1 = 0,   // GMM1 (hidden -> intermediate*2) + SwiGLU + MX-quant
    Linear2 = 1,   // GMM2 (intermediate -> hidden) + combine
};

struct WaveBlock {
    uint32_t   waveIdx;
    uint32_t   expertIdx;     // global expert id within current rank
    uint32_t   mBlockIdx;
    uint32_t   nBlockIdx;
    uint32_t   numMBlocks;    // for current expert/phase
    uint32_t   numNBlocks;    // for current expert/phase
    uint32_t   poolBlockOffset; // accumulated (m_block_global) over experts in wave (DeepGEMM concept)
    BlockPhase phase;
    bool       valid;
};

// Stateful iterator over current-wave blocks for a given phase.
// Replaces template-callback ForEachBlock — AscendC lambdas resolve to __host__ callables,
// which can't dispatch from __aicore__ context.
struct WaveBlockIter {
    BlockPhase phase;
    uint32_t numNBlocks;
    uint32_t expertsInWave;
    uint32_t totalBlocks;
    uint32_t step;
    uint32_t increment;     // BeginIter: totalSms_; BeginIterSerial: 1
};

class WaveScheduler {
public:
    __aicore__ inline void Init(const MegaMoeV2TilingData *tiling, GM_ADDR cumsumGmAddr)
    {
        tiling_ = tiling;
        kExpertsPerWave_ = tiling->kExpertsPerWave;
        kNumWaves_ = tiling->kNumWaves;
        expertPerRank_ = tiling->expertPerRank;
        worldSize_ = tiling->epWorldSize;
        smIdx_ = GetBlockIdx() / GetTaskRation();
        totalSms_ = GetBlockNum();

        // L1_TILE_M / L1_TILE_N for GMM1 / GMM2 (must match impl.h)
        l1TileM_ = 128;
        l1TileN_ = 256;
        intermediate_ = tiling->n / SWIGLU_N_HALF;
        hidden_ = tiling->k;

        cumsumGm_.SetGlobalBuffer((__gm__ int32_t*)cumsumGmAddr);
        // tokenPerExpert (per-rank totals) read via cumsum: last row stores totals after CrossRankSync.
        // expert_total[e] = cumsumGm_[(worldSize-1)*expertPerRank + e]  (consistent with v1 mega_moe.h:206)
    }

    // returns expert m (token count after dispatch) for given local expertIdx
    __aicore__ inline uint32_t GetExpertTokenCount(uint32_t expertIdx) const
    {
        if (expertIdx >= expertPerRank_) {
            return 0u;
        }
        int64_t lastRow = static_cast<int64_t>(worldSize_ - 1) * expertPerRank_;
        int32_t cum = cumsumGm_.GetValue(lastRow + expertIdx);
        int32_t prev = (expertIdx == 0u) ? 0
                        : cumsumGm_.GetValue(lastRow + expertIdx - 1);
        int32_t diff = cum - prev;
        return diff > 0 ? static_cast<uint32_t>(diff) : 0u;
    }

    // pool block offset: cumulative m-block count across experts already started in this wave.
    __aicore__ inline uint32_t GetPoolBlockOffset(uint32_t expertOffsetInWave) const
    {
        uint32_t off = 0;
        uint32_t waveBaseExpert = currentWaveBase_;
        for (uint32_t e = 0; e < expertOffsetInWave; ++e) {
            uint32_t m = GetExpertTokenCount(waveBaseExpert + e);
            off += Ops::Base::CeilDiv(m, l1TileM_);
        }
        return off;
    }

    // get number of m-blocks for given expert
    __aicore__ inline uint32_t GetNumMBlocks(uint32_t expertIdx) const
    {
        uint32_t m = GetExpertTokenCount(expertIdx);
        return Ops::Base::CeilDiv(m, l1TileM_);
    }

    // get number of n-blocks for given phase
    __aicore__ inline uint32_t GetNumNBlocks(BlockPhase phase) const
    {
        uint32_t n = (phase == BlockPhase::Linear1) ? intermediate_ : hidden_;
        return Ops::Base::CeilDiv(n, l1TileN_);
    }

    // advance wave: Linear1 fully drains -> Linear2 starts -> next wave.
    // Caller drives outer loop; this returns false when all waves exhausted.
    __aicore__ inline bool BeginWave(uint32_t waveIdx)
    {
        if (waveIdx >= kNumWaves_) {
            return false;
        }
        currentWaveIdx_ = waveIdx;
        currentWaveBase_ = waveIdx * kExpertsPerWave_;
        return true;
    }

    // Persistent block iteration over current wave + phase.
    // Each (sm_idx + step*total_sms) maps to one (expert_in_wave, mBlock, nBlock).
    // Constraint (DeepGEMM scheduler:148-181): all m_blocks of an expert must complete Linear1
    // before any m_block of that expert starts Linear2 -- enforced by caller running phase=Linear1
    // for entire wave first, then phase=Linear2.
    //
    // Caller pattern:
    //   WaveBlockIter it; scheduler_.BeginIter(phase, it);
    //   WaveBlock wb;
    //   while (scheduler_.NextBlock(it, wb)) { ... }
    __aicore__ inline void BeginIter(BlockPhase phase, WaveBlockIter& it) const
    {
        it.phase = phase;
        it.numNBlocks = GetNumNBlocks(phase);
        it.expertsInWave =
            (currentWaveBase_ + kExpertsPerWave_ <= expertPerRank_)
                ? kExpertsPerWave_
                : (expertPerRank_ - currentWaveBase_);
        it.totalBlocks = 0;
        for (uint32_t e = 0; e < it.expertsInWave; ++e) {
            it.totalBlocks += GetNumMBlocks(currentWaveBase_ + e) * it.numNBlocks;
        }
        it.step = smIdx_;
        it.increment = totalSms_;
    }

    // Serial single-core iteration — used by V2 debug baseline. One AIC + one vec0 (pair 0)
    // walks the entire wave in lockstep, slot=0 throughout. Eliminates cross-pair sync.
    __aicore__ inline void BeginIterSerial(BlockPhase phase, WaveBlockIter& it) const
    {
        BeginIter(phase, it);
        it.step = 0;
        it.increment = 1;
    }

    __aicore__ inline bool NextBlock(WaveBlockIter& it, WaveBlock& wb) const
    {
        if (it.step >= it.totalBlocks) {
            return false;
        }
        uint32_t remaining = it.step;
        uint32_t eOffset = 0;
        uint32_t mBlock = 0;
        uint32_t nBlock = 0;
        uint32_t poolBase = 0;
        for (uint32_t e = 0; e < it.expertsInWave; ++e) {
            uint32_t mBlocks = GetNumMBlocks(currentWaveBase_ + e);
            uint32_t expertBlocks = mBlocks * it.numNBlocks;
            if (remaining < expertBlocks) {
                eOffset = e;
                mBlock = remaining / it.numNBlocks;
                nBlock = remaining % it.numNBlocks;
                break;
            }
            remaining -= expertBlocks;
            poolBase += mBlocks;
        }
        wb.waveIdx = currentWaveIdx_;
        wb.expertIdx = currentWaveBase_ + eOffset;
        wb.mBlockIdx = mBlock;
        wb.nBlockIdx = nBlock;
        wb.numMBlocks = GetNumMBlocks(wb.expertIdx);
        wb.numNBlocks = it.numNBlocks;
        wb.poolBlockOffset = poolBase + mBlock;
        wb.phase = it.phase;
        wb.valid = true;
        it.step += it.increment;
        return true;
    }

    __aicore__ inline uint32_t GetCurrentWaveBaseExpert() const { return currentWaveBase_; }
    __aicore__ inline uint32_t GetNumWaves() const { return kNumWaves_; }
    __aicore__ inline uint32_t GetExpertsPerWave() const { return kExpertsPerWave_; }
    __aicore__ inline uint32_t GetL1TileM() const { return l1TileM_; }
    __aicore__ inline uint32_t GetL1TileN() const { return l1TileN_; }

    // Map (waveIdx, expertOffsetInWave, mBlockIdx) -> chunk slot in ring buffer (mod chunkDepth).
    // Used by AIC busy-wait on dispatchArrival[chunk_id] and l2ArrivalMask[m_block_global].
    __aicore__ inline uint32_t GetChunkId(uint32_t waveIdx, uint32_t poolBlockOffset) const
    {
        uint32_t numChunksPerWave = tiling_->numChunksPerWave;
        uint32_t mBlocksPerChunk =
            Ops::Base::CeilDiv(tiling_->chunkM, l1TileM_);   // typically 1 (chunkM == L1_TILE_M)
        uint32_t chunkInWave = poolBlockOffset / (mBlocksPerChunk == 0u ? 1u : mBlocksPerChunk);
        return waveIdx * numChunksPerWave + chunkInWave;
    }

    // Global linear m-block id (for l2ArrivalMask indexing) across all waves.
    __aicore__ inline uint32_t GetGlobalMBlockId(uint32_t waveIdx, uint32_t poolBlockOffset) const
    {
        uint32_t numChunksPerWave = tiling_->numChunksPerWave;
        return waveIdx * numChunksPerWave + poolBlockOffset;
    }

private:
    const MegaMoeV2TilingData *tiling_{nullptr};
    GlobalTensor<int32_t> cumsumGm_;

    uint32_t kExpertsPerWave_{0};
    uint32_t kNumWaves_{0};
    uint32_t expertPerRank_{0};
    uint32_t worldSize_{0};
    uint32_t smIdx_{0};
    uint32_t totalSms_{0};
    uint32_t l1TileM_{128};
    uint32_t l1TileN_{256};
    uint32_t intermediate_{0};
    uint32_t hidden_{0};

    uint32_t currentWaveIdx_{0};
    uint32_t currentWaveBase_{0};
};

} // namespace MegaMoeV2Impl
#endif // MEGA_MOE_V2_WAVE_SCHEDULER_H
