/**
 * Copyright (c) 2026 fk2323
 * Licensed under CANN Open Software License Agreement Version 2.0.
 * See LICENSE in the root of the repository.
 */

#define fmm016_streaming_top_k_blocked fmm016_streaming_top_k_blocked_v2
#include "kernel_operator.h"
#include "fmm016_router_tiling.h"

namespace {
constexpr uint32_t kSourceTile = 1024;
constexpr uint32_t kMaxTopK = 16;
constexpr uint32_t kMaxIndexHeads = 4;
constexpr uint32_t kMaxIndexDim = 16;
constexpr uint32_t kReduceWorkspaceFloats = 512;
constexpr float kNegativeInfinity = -1.0e30F;
#ifdef FMM016_ROUTER_TILED
constexpr uint32_t kTargetTile = 8;
#endif

class StreamingTopKBlockedKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR qIndex, GM_ADDR kIndexTranspose, GM_ADDR indexWeight,
        GM_ADDR selectedIndex, GM_ADDR selectedValid,
        const Fmm016StreamingTopKBlockedTilingData &tiling,
        AscendC::TPipe *pipe)
    {
        t_ = tiling;
        q_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(qIndex),
            t_.totalRows * t_.indexHeads * t_.indexDim);
        kt_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(kIndexTranspose),
            t_.bhCount * t_.indexDim * t_.paddedLeaves);
        weight_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(indexWeight),
            t_.totalRows * t_.indexHeads);
        outIndex_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(selectedIndex),
            t_.totalRows * t_.topK);
        outValid_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(selectedValid),
            t_.totalRows * t_.topK);
        pipe->InitBuffer(kQueue_, 1, kSourceTile * sizeof(float));
#ifdef FMM016_ROUTER_TILED
        pipe->InitBuffer(dotBuffer_, kTargetTile * kSourceTile * sizeof(float));
        pipe->InitBuffer(scoreBuffer_, kTargetTile * kSourceTile * sizeof(float));
        pipe->InitBuffer(tempBuffer_, kSourceTile * sizeof(float));
#else
        pipe->InitBuffer(dotBuffer_, kSourceTile * sizeof(float));
        pipe->InitBuffer(scoreBuffer_, kSourceTile * sizeof(float));
#endif
        pipe->InitBuffer(reduceBuffer_, kReduceWorkspaceFloats * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        const uint32_t blocks = AscendC::GetBlockNum();
        const uint32_t groups = (t_.totalRows + 31U) / 32U;
        for (uint32_t group = AscendC::GetBlockIdx(); group < groups; group += blocks) {
            const uint32_t begin = group * 32U;
            uint32_t end = begin + 32U;
            if (end > t_.totalRows) end = t_.totalRows;
#ifdef FMM016_ROUTER_TILED
            // A shared K tile is valid only within one (batch, head) plane.
            // Fall back to the established scalar-row path for the rare
            // 32-row scheduling group that straddles a BH boundary.
            if (begin / t_.leaves != (end - 1U) / t_.leaves) {
                for (uint32_t row = begin; row < end; ++row) ProcessRow(row);
            } else {
                uint32_t row = begin;
                while (row < end) {
                    uint32_t rowEnd = row + kTargetTile;
                    if (rowEnd > end) rowEnd = end;
                    ProcessRows(row, rowEnd);
                    row = rowEnd;
                }
            }
#else
            for (uint32_t row = begin; row < end; ++row) ProcessRow(row);
#endif
        }
    }

private:
#ifdef FMM016_ROUTER_TILED
    __aicore__ inline void ProcessRows(uint32_t rowBegin, uint32_t rowEnd)
    {
        const uint32_t activeRows = rowEnd - rowBegin;
        float qValue[kTargetTile][kMaxIndexHeads][kMaxIndexDim];
        float weightValue[kTargetTile][kMaxIndexHeads];
        float bestScore[kTargetTile][kMaxTopK];
        int32_t bestId[kTargetTile][kMaxTopK];
        uint32_t bhValue[kTargetTile];
        uint32_t targetValue[kTargetTile];

        // Q-index and head weights are reused for every source tile.  Cache
        // them once in scalar storage instead of issuing synchronous GM
        // GetValue operations inside the O(leaves / 1024) scan.
        for (uint32_t lane = 0; lane < activeRows; ++lane) {
            const uint32_t row = rowBegin + lane;
            const uint32_t bh = row / t_.leaves;
            bhValue[lane] = bh;
            targetValue[lane] = row - bh * t_.leaves;
            for (uint32_t head = 0; head < t_.indexHeads; ++head) {
                weightValue[lane][head] =
                    weight_.GetValue(row * t_.indexHeads + head);
                for (uint32_t dim = 0; dim < t_.indexDim; ++dim) {
                    const uint32_t qOffset =
                        (row * t_.indexHeads + head) * t_.indexDim + dim;
                    qValue[lane][head][dim] = q_.GetValue(qOffset);
                }
            }
            for (uint32_t slot = 0; slot < kMaxTopK; ++slot) {
                bestScore[lane][slot] = kNegativeInfinity;
                bestId[lane][slot] = 0;
            }
        }

        auto temporary = tempBuffer_.Get<float>();
        auto reduce = reduceBuffer_.Get<float>();
        for (uint32_t sourceBase = 0; sourceBase < t_.paddedLeaves;
             sourceBase += kSourceTile) {
            for (uint32_t lane = 0; lane < activeRows; ++lane) {
                auto score = scoreBuffer_.GetWithOffset<float>(
                    kSourceTile, lane * kSourceTile * sizeof(float));
                AscendC::Duplicate(score, 0.0F, kSourceTile);
            }
            AscendC::PipeBarrier<PIPE_V>();

            for (uint32_t head = 0; head < t_.indexHeads; ++head) {
                for (uint32_t lane = 0; lane < activeRows; ++lane) {
                    auto dot = dotBuffer_.GetWithOffset<float>(
                        kSourceTile, lane * kSourceTile * sizeof(float));
                    AscendC::Duplicate(dot, 0.0F, kSourceTile);
                }
                AscendC::PipeBarrier<PIPE_V>();

                for (uint32_t dim = 0; dim < t_.indexDim; ++dim) {
                    auto kLocal = kQueue_.AllocTensor<float>();
                    const uint32_t kOffset =
                        (bhValue[0] * t_.indexDim + dim) *
                        t_.paddedLeaves + sourceBase;
                    AscendC::DataCopy(kLocal, kt_[kOffset], kSourceTile);
                    kQueue_.EnQue(kLocal);
                    kLocal = kQueue_.DeQue<float>();
                    for (uint32_t lane = 0; lane < activeRows; ++lane) {
                        auto dot = dotBuffer_.GetWithOffset<float>(
                            kSourceTile, lane * kSourceTile * sizeof(float));
                        AscendC::Muls(temporary, kLocal,
                            qValue[lane][head][dim], kSourceTile);
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::Add(dot, dot, temporary, kSourceTile);
                        AscendC::PipeBarrier<PIPE_V>();
                    }
                    kQueue_.FreeTensor(kLocal);
                }

                for (uint32_t lane = 0; lane < activeRows; ++lane) {
                    auto dot = dotBuffer_.GetWithOffset<float>(
                        kSourceTile, lane * kSourceTile * sizeof(float));
                    auto score = scoreBuffer_.GetWithOffset<float>(
                        kSourceTile, lane * kSourceTile * sizeof(float));
                    AscendC::Muls(dot, dot, t_.invSqrtIndexDim, kSourceTile);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Relu(dot, dot, kSourceTile);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Muls(dot, dot, weightValue[lane][head], kSourceTile);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Add(score, score, dot, kSourceTile);
                    AscendC::PipeBarrier<PIPE_V>();
                }
            }

            AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            uint32_t sourceEnd = sourceBase + kSourceTile;
            if (sourceEnd > t_.leaves) sourceEnd = t_.leaves;
            const uint32_t sourceCount = sourceEnd - sourceBase;

            for (uint32_t lane = 0; lane < activeRows; ++lane) {
                auto score = scoreBuffer_.GetWithOffset<float>(
                    kSourceTile, lane * kSourceTile * sizeof(float));
                int32_t nearBegin = static_cast<int32_t>(targetValue[lane]) - 2;
                int32_t nearEnd = static_cast<int32_t>(targetValue[lane]) + 2;
                if (nearBegin < static_cast<int32_t>(sourceBase)) {
                    nearBegin = static_cast<int32_t>(sourceBase);
                }
                if (nearEnd >= static_cast<int32_t>(sourceEnd)) {
                    nearEnd = static_cast<int32_t>(sourceEnd) - 1;
                }
                for (int32_t source = nearBegin; source <= nearEnd; ++source) {
                    score.SetValue(static_cast<uint32_t>(source) - sourceBase,
                        kNegativeInfinity);
                }
            }
            AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);

            uint32_t tileCandidates = t_.topK;
            if (tileCandidates > sourceCount) tileCandidates = sourceCount;
            for (uint32_t lane = 0; lane < activeRows; ++lane) {
                auto score = scoreBuffer_.GetWithOffset<float>(
                    kSourceTile, lane * kSourceTile * sizeof(float));
                for (uint32_t tileSlot = 0; tileSlot < tileCandidates; ++tileSlot) {
                    AscendC::ReduceMax(reduce, score, reduce[8], sourceCount, true);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
                    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
                    float candidate = reduce.GetValue(0);
                    float encodedIndex = reduce.GetValue(1);
                    const uint32_t localIndex =
                        *reinterpret_cast<uint32_t *>(&encodedIndex);
                    int32_t candidateId =
                        static_cast<int32_t>(sourceBase + localIndex);
                    for (uint32_t slot = 0; slot < t_.topK; ++slot) {
                        if (candidate > bestScore[lane][slot]) {
                            const float displaced = bestScore[lane][slot];
                            const int32_t displacedId = bestId[lane][slot];
                            bestScore[lane][slot] = candidate;
                            bestId[lane][slot] = candidateId;
                            candidate = displaced;
                            candidateId = displacedId;
                        }
                    }
                    score.SetValue(localIndex, kNegativeInfinity);
                    AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
                    AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
                }
            }
        }

        for (uint32_t lane = 0; lane < activeRows; ++lane) {
            const uint32_t outputBase = (rowBegin + lane) * t_.topK;
            for (uint32_t slot = 0; slot < t_.topK; ++slot) {
                outIndex_.SetValue(outputBase + slot, bestId[lane][slot]);
                outValid_.SetValue(outputBase + slot,
                    static_cast<uint8_t>(bestScore[lane][slot] > -1.0e20F));
            }
        }
    }
#endif

    __aicore__ inline void ProcessRow(uint32_t row)
    {
        const uint32_t bh = row / t_.leaves;
        const uint32_t target = row - bh * t_.leaves;
        float bestScore[kMaxTopK];
        int32_t bestId[kMaxTopK];
        for (uint32_t slot = 0; slot < kMaxTopK; ++slot) {
            bestScore[slot] = kNegativeInfinity;
            bestId[slot] = 0;
        }
        auto dot = dotBuffer_.Get<float>();
        auto score = scoreBuffer_.Get<float>();
        auto reduce = reduceBuffer_.Get<float>();
        for (uint32_t sourceBase = 0; sourceBase < t_.paddedLeaves;
             sourceBase += kSourceTile) {
            AscendC::Duplicate(score, 0.0F, kSourceTile);
            AscendC::PipeBarrier<PIPE_V>();
            for (uint32_t head = 0; head < t_.indexHeads; ++head) {
                AscendC::Duplicate(dot, 0.0F, kSourceTile);
                AscendC::PipeBarrier<PIPE_V>();
                for (uint32_t dim = 0; dim < t_.indexDim; ++dim) {
                    auto kLocal = kQueue_.AllocTensor<float>();
                    const uint32_t kOffset =
                        (bh * t_.indexDim + dim) * t_.paddedLeaves + sourceBase;
                    AscendC::DataCopy(kLocal, kt_[kOffset], kSourceTile);
                    kQueue_.EnQue(kLocal);
                    kLocal = kQueue_.DeQue<float>();
                    const uint32_t qOffset =
                        (row * t_.indexHeads + head) * t_.indexDim + dim;
                    AscendC::Muls(kLocal, kLocal, q_.GetValue(qOffset), kSourceTile);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Add(dot, dot, kLocal, kSourceTile);
                    AscendC::PipeBarrier<PIPE_V>();
                    kQueue_.FreeTensor(kLocal);
                }
                AscendC::Muls(dot, dot, t_.invSqrtIndexDim, kSourceTile);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Relu(dot, dot, kSourceTile);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Muls(dot, dot,
                    weight_.GetValue(row * t_.indexHeads + head), kSourceTile);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Add(score, score, dot, kSourceTile);
                AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            uint32_t sourceEnd = sourceBase + kSourceTile;
            if (sourceEnd > t_.leaves) sourceEnd = t_.leaves;
            const uint32_t sourceCount = sourceEnd - sourceBase;

            // Mask the at-most-five near leaves in UB, then reduce only the
            // real (unpadded) part of this source tile.  The former code read
            // every score through scalar GetValue; extracting only topK tile
            // candidates keeps the exact global insertion semantics while
            // moving the O(leaves) search onto the vector reduction unit.
            int32_t nearBegin = static_cast<int32_t>(target) - 2;
            int32_t nearEnd = static_cast<int32_t>(target) + 2;
            if (nearBegin < static_cast<int32_t>(sourceBase)) {
                nearBegin = static_cast<int32_t>(sourceBase);
            }
            if (nearEnd >= static_cast<int32_t>(sourceEnd)) {
                nearEnd = static_cast<int32_t>(sourceEnd) - 1;
            }
            for (int32_t source = nearBegin; source <= nearEnd; ++source) {
                score.SetValue(static_cast<uint32_t>(source) - sourceBase,
                    kNegativeInfinity);
            }
            AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);

            uint32_t tileCandidates = t_.topK;
            if (tileCandidates > sourceCount) tileCandidates = sourceCount;
            for (uint32_t tileSlot = 0; tileSlot < tileCandidates; ++tileSlot) {
                AscendC::ReduceMax(reduce, score, reduce[8], sourceCount, true);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
                float candidate = reduce.GetValue(0);
                float encodedIndex = reduce.GetValue(1);
                const uint32_t localIndex =
                    *reinterpret_cast<uint32_t *>(&encodedIndex);
                int32_t candidateId =
                    static_cast<int32_t>(sourceBase + localIndex);
                for (uint32_t slot = 0; slot < t_.topK; ++slot) {
                    if (candidate > bestScore[slot]) {
                        const float displaced = bestScore[slot];
                        const int32_t displacedId = bestId[slot];
                        bestScore[slot] = candidate;
                        bestId[slot] = candidateId;
                        candidate = displaced;
                        candidateId = displacedId;
                    }
                }
                score.SetValue(localIndex, kNegativeInfinity);
                AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
            }
        }
        const uint32_t outputBase = row * t_.topK;
        for (uint32_t slot = 0; slot < t_.topK; ++slot) {
            outIndex_.SetValue(outputBase + slot, bestId[slot]);
            outValid_.SetValue(outputBase + slot,
                static_cast<uint8_t>(bestScore[slot] > -1.0e20F));
        }
    }

    Fmm016StreamingTopKBlockedTilingData t_ {};
    AscendC::GlobalTensor<float> q_, kt_, weight_;
    AscendC::GlobalTensor<int32_t> outIndex_;
    AscendC::GlobalTensor<uint8_t> outValid_;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> kQueue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> dotBuffer_, scoreBuffer_,
        reduceBuffer_, tempBuffer_;
};
}

extern "C" __global__ __aicore__ void fmm016_streaming_top_k_blocked(
    GM_ADDR qIndex, GM_ADDR kIndexTranspose, GM_ADDR indexWeight,
    GM_ADDR selectedIndex, GM_ADDR selectedValid,
    GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(Fmm016StreamingTopKBlockedTilingData);
    GET_TILING_DATA(t, tiling);
    AscendC::TPipe pipe;
    StreamingTopKBlockedKernel kernel;
    kernel.Init(qIndex, kIndexTranspose, indexWeight,
        selectedIndex, selectedValid, t, &pipe);
    kernel.Process();
}
