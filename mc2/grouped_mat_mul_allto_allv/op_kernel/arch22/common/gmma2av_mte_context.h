/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Single owner of runtime state; stages share this context without duplicating UB or cursors.
#ifndef COMMON_GMMA2AV_MTE_CONTEXT_H
#define COMMON_GMMA2AV_MTE_CONTEXT_H

#include "kernel_operator.h"
#include "../../../../common/op_kernel/moe_distribute_base.h"
#include "gmma2av_mte_constants.h"
#include "gmma2av_mte_timer.h"

namespace MC2KernelTemplate {

template <typename T, typename TimerType = GmmA2avMteNullTimer>
class GmmA2avMteOp {
public:
    using Timer = TimerType;
    __aicore__ inline GmmA2avMteOp() {}

    __aicore__ inline void Init(const GroupedMatMulAlltoAllvMteTaskTilingInfo *taskTilingInfo, GM_ADDR sendBuffer,
                                GM_ADDR recvBuffer, GM_ADDR cumsumBuffer, AscendC::TPipe *pipe,
                                const GmmA2avCoCTiling *cocTiling = nullptr, uint64_t commBufferSize = 0UL,
                                uint32_t isA3 = 0U, GM_ADDR timerBuffer = 0UL, uint32_t timerEnabled = 0U,
                                int64_t kernelEntryCycle = 0);

    __aicore__ inline void BeginExpertPipeline();

    // Stage path: copy one expert's locally owned M tiles from the GMM
    // workspace into this rank's window, waiting for each local compute
    // wave before reading its rows. This path never waits on a peer, pulls payload, or
    // joins any rendezvous that requires the Comm subblock.
    __aicore__ inline void StageExpertPayload(uint32_t startExpertIdx, uint32_t endExpertIdx);

    // Publish the Stage→Comm completion progress for this producer core.
    // Called only for experts whose M tiles this core actually staged
    // (scheduler-gated on HasExpertWork). Release ordering: CopyGmToGm
    // completes the payload MTE3 writes with a DDR barrier before returning,
    // and StoreSyncFlag ends with its own DDR barrier, so the payload is
    // globally visible before the progress value is.
    __aicore__ inline void PublishStageProgress(uint32_t expertIdx);

    // Cross-rank protocol for one expert, executed by the Comm subblocks
    // only. Order per expert: wait every local producer's stage progress,
    // publish ExpertReady to all owned destinations, then wait owned
    // sources and pull. Every wait has an independent publisher:
    // ExpertReady publication depends only on the publisher rank's local
    // stage progress, so worker progress skew cannot form a publish/wait
    // cycle. No all-AIV rendezvous exists in this path.
    __aicore__ inline void CommExpertChunk(uint32_t startExpertIdx, uint32_t endExpertIdx);

    __aicore__ inline void PublishChunkReady(uint32_t startExpertIdx, uint32_t endExpertIdx);

    __aicore__ inline void WaitChunkReady(uint32_t startExpertIdx, uint32_t endExpertIdx);

    __aicore__ inline void PullExpertChunk(uint32_t startExpertIdx, uint32_t endExpertIdx);

    __aicore__ inline void Wait(uint32_t startExpertIdx);

    // Publish this round's completion tag after all local payload pulls.
    // The tag persists until the next round reaches its completion phase.
    __aicore__ inline void End();

public:
    // AIV0 consumes local AIC wave progress; AIV1 runs the peer protocol
    // independently. Neither role participates in per-wave cross-core flags.
    __aicore__ inline bool IsStageAiv() const;

    __aicore__ inline bool IsCommAiv() const;

    // Paired AIC ownership, shared by Stage and Comm expert cursors.
    __aicore__ inline bool HasExpertWork(uint32_t expertIdx) const;

private:
    __aicore__ inline GM_ADDR GetWindow(uint32_t rank) const;

    __aicore__ inline __gm__ int32_t *GetCountTable(uint32_t rank) const;

    __aicore__ inline __gm__ T *GetOutputData(uint32_t rank) const;

    __aicore__ inline void PublishLocalCountTable();

    // Stage one expert's rows from the GMM output workspace into this rank's
    // window. Ownership follows the AIC-side M-row striped scheduling: the
    // AIC paired with logical core c computed exactly the tiles mIdx with
    // (mIdx - startCoreIdx_e) mod aicCoreNum == c, so this AIV copies only
    // those tiles. Both sides replay the same cross-expert rotation that
    // advances by ceil(rows_e / L1_TILE_M), keeping tile ownership identical
    // without any tiling-table extension.
    __aicore__ inline void PublishLocalChunk(uint32_t startExpertIdx, uint32_t endExpertIdx);

    __aicore__ inline void BuildDestinationOffsets();

    __aicore__ inline void BuildPeerSourceOffsets();

    // Local (non-CCL) stage-progress protocol. Slot c is written ONLY by
    // the Stage subblock of logical core c through a 32-byte MTE3
    // transaction that ends with a DDR barrier; readers use 32-byte MTE2
    // transactions with the L2 bypass hint (see WaitStageProgress). Both
    // widths are naturally aligned inside one 64-byte cache line, and the
    // producer's DDR barrier after the payload copy plus the reader's DDR
    // barrier after observing the value provide the release/acquire
    // ordering.
    __aicore__ inline __gm__ int64_t *GetStageProgressSlot(uint32_t producerCore) const;

    __aicore__ inline __gm__ int64_t *GetComputeProgressSlot(uint32_t producerCore) const;

    __aicore__ inline void WaitComputeProgress(int64_t sequence);

    __aicore__ inline void WaitStageProgress(uint32_t producerCore, int64_t boundary);

    __aicore__ inline void WaitLocalProgress(__gm__ int64_t *slot, int64_t boundary, bool bypassL2 = true);

    // Wait until every producer that owns M tiles of the expert range has
    // staged them, replaying the same cross-expert rotation the AIC
    // scheduler and PublishLocalChunk use. Cores without tiles for an
    // expert never write progress for it and are never waited on. The
    // rotation advance below keeps this instance's HasExpertWork answers
    // identical to the Stage subblock's.
    __aicore__ inline void WaitLocalStageProgress(uint32_t startExpertIdx, uint32_t endExpertIdx);

    __aicore__ inline void PullPeerExpertFromSource(uint32_t srcRank, uint32_t expertIdx);

    __aicore__ inline AscendC::LocalTensor<int32_t> GetControlTensor();

    __aicore__ inline AscendC::LocalTensor<int64_t> GetSyncTensor();

    __aicore__ inline AscendC::LocalTensor<uint32_t> GetDestinationOffsetTensor();

    __aicore__ inline AscendC::LocalTensor<uint32_t> GetSourceOffsetTensor();

    __aicore__ static inline bool SafeAdd(uint64_t lhs, uint64_t rhs, uint64_t &result);

    __aicore__ static inline bool SafeMulU32(uint64_t lhs, uint64_t rhs, uint64_t &result);

    __aicore__ inline void LoadInt32Row(__gm__ int32_t *source, const AscendC::LocalTensor<int32_t> &local,
                                        uint32_t elementCount);

    __aicore__ inline void StoreInt32Row(__gm__ int32_t *destination, const AscendC::LocalTensor<int32_t> &local,
                                         uint32_t elementCount);

    __aicore__ inline void LocalAivSync();

    // Single-writer rendezvous (mirrors matmul_allto_all's CrossRankSyncV1):
    // each rank publishes ONE state slot in its OWN window and every peer
    // polls that slot in the publisher's window. Because a window's slots
    // are written only by their owner, the Begin-of-round ResetSyncRegion can
    // never race a peer's publish, and no published value is ever overwritten
    // while a peer still polls for it.
    __aicore__ inline void PublishAndWaitRankPhase(uint64_t phaseBase, int64_t flag);

    __aicore__ inline uint64_t GetBufferIndex() const;

    __aicore__ inline __gm__ int64_t *GetFixedSyncSlot(uint32_t windowRank, uint64_t phaseBase, uint64_t index) const;

    __aicore__ inline __gm__ int64_t *GetExpertReadySlot(uint32_t windowRank, uint32_t sourceRank,
                                                         uint32_t expertIdx) const;

    // sendCnt/recvCnt come from the same tiling computation on both ends, so
    // a skipped publish slot is exactly the slot the peer skips waiting on:
    // publish is needed only when this rank actually sends expert e to D, and
    // waiting is needed only when this rank actually receives expert e from
    // S. Expert 0 is always kept unmasked on both sides: its flag is the
    // release/acquire handshake for this rank's count table, which
    // BuildPeerSourceOffsets loads even when no expert-0 rows are exchanged.
    // The self slot is retained for the worker-local publish/acquire path.
    __aicore__ inline bool IsExpertSlotActive(const int32_t *countTable, uint32_t peerRank, uint32_t expertIdx,
                                              uint32_t expertNum) const;

    // Initialize the whole fixed region only on the first invocation;
    // subsequent calls reset expert-ready slots and retain rank phase tags.
    __aicore__ inline void ResetSyncRegion();

    // One batched MTE3 transaction publishes every expert slot of one chunk
    // for one destination rank, replacing one 32-byte store plus DDR barrier
    // per active expert. Inactive slots are written as zero; they are never
    // polled this invocation (the peer's wait mask is identical), and the
    // next invocation's publisher overwrites them before any waiter observes.
    __aicore__ inline bool PublishExpertReadyRange(uint32_t dstRank, uint32_t startExpertIdx, uint32_t endExpertIdx,
                                                   int64_t flag);

    __aicore__ inline void PublishExpertReadyPerFlag(uint32_t dstRank, uint32_t startExpertIdx, uint32_t endExpertIdx,
                                                     int64_t flag);

    // One batched MTE2 transaction polls every expert slot of one chunk from
    // one source rank. A single DDR barrier after all active slots hold the
    // expected value replaces the per-flag barrier, so a ready chunk costs
    // one MTE round trip instead of one per active (expert, source) pair.
    __aicore__ inline bool WaitExpertReadyRange(uint32_t srcRank, uint32_t startExpertIdx, uint32_t endExpertIdx,
                                                int64_t expected);

    __aicore__ inline void WaitExpertReadyPerFlag(uint32_t srcRank, uint32_t startExpertIdx, uint32_t endExpertIdx,
                                                  int64_t expected);

    __aicore__ inline void StoreSyncFlag(__gm__ int64_t *destination, int64_t flag);

    __aicore__ inline bool IsPrimaryAiv() const;

    __aicore__ inline void WaitSyncFlag(__gm__ int64_t *source, int64_t expected, bool bypassL2);

    __aicore__ inline uint32_t AlignControlElements(uint32_t elementCount) const;

    __aicore__ inline void CopyGmToGm(__gm__ T *source, __gm__ T *destination, uint64_t elementCount);

    __aicore__ inline void BeginGmToGmCopy();

    __aicore__ inline void CopyGmToGmSegment(__gm__ T *source, __gm__ T *destination, uint64_t elementCount,
                                             uint32_t &moveIdx);

    __aicore__ inline void EndGmToGmCopy();

private:
    TimerType timer_;
    const GroupedMatMulAlltoAllvMteTaskTilingInfo *taskTilingInfo_{nullptr};
    __gm__ HcclOpResParam *a3WinContext_{nullptr};
    __gm__ AscendC::HcclCombineOpParam *a2WinContext_{nullptr};
    __gm__ T *sendBuffer_{nullptr};
    __gm__ T *recvBuffer_{nullptr};
    // Base of the local stage-progress region (one 64-byte slot per AIC
    // logical core, appended after the cumsum region in user workspace).
    // Never overlaps the cross-card sync mailbox in the CCL window.
    __gm__ int64_t *stageProgressBase_{nullptr};
    __gm__ int64_t *computeProgressBase_{nullptr};
    int64_t computeWaveSequence_{0};
    uint64_t winSize_{0UL};
    uint64_t localExpertBaseRows_{0UL};
    // AIV-side replay of the AIC scheduler's cross-expert first-core
    // rotation; both sides advance it by ceil(rows_e / L1_TILE_M).
    uint32_t expertStartCoreIdx_{0U};
    uint32_t rank_{0U};
    uint32_t rankSize_{0U};
    bool isA3_{false};
    uint32_t coreIdx_{0U};
    uint32_t subBlockIdx_{0U};
    uint32_t logicalCoreIdx_{0U};
    uint32_t rankWorkerCount_{0U};
    uint32_t ubMoveElements_{GmmA2avMteDetail::DEFAULT_UB_MOVE_ELEMENTS};
    AscendC::TBuf<AscendC::TPosition::VECCALC> ubBuffer_;
};


} // namespace MC2KernelTemplate

#endif // COMMON_GMMA2AV_MTE_CONTEXT_H
