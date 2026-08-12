/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ALLTO_ALLV_GROUPED_MAT_MUL_SHMEM_H
#define ALLTO_ALLV_GROUPED_MAT_MUL_SHMEM_H

#include "shmem.h"
#include "allto_allv_grouped_mat_mul_shmem_schedule.h"
#include "allto_allv_grouped_mat_mul_shmem_tiling.h"

namespace AlltoAllvGmmShmem {

using namespace AscendC;

using Element = bfloat16_t;
constexpr uint32_t BF16_BYTES = sizeof(Element);
constexpr uint32_t WORKSPACE_ROW_ALIGNMENT_BYTES = 512;
constexpr uint32_t AIV_BARRIER_FLAG = 8;
constexpr uint32_t WORKSPACE_STAGE_COUNT = 2;
constexpr uint32_t LOCAL_COPY_UB_STAGES = 2;
constexpr uint32_t UDMA_SYNC_ID = 0;

__aicore__ inline uint32_t MinU32(uint32_t lhs, uint32_t rhs)
{
    return lhs < rhs ? lhs : rhs;
}

class AivShmemCommunicator {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR symmetric,
                                const AlltoAllvGroupedMatMulShmemTilingData *tiling, TPipe *pipe,
                                uint32_t rank, uint32_t peerCount)
    {
        x_ = reinterpret_cast<__gm__ Element *>(x);
        symmetric_ = reinterpret_cast<__gm__ Element *>(symmetric);
        tiling_ = tiling;
        rank_ = rank;
        peerCount_ = peerCount;
        alignedK_ = static_cast<uint32_t>(AlignUpU64(tiling_->k, WORKSPACE_ROW_ALIGNMENT_BYTES / BF16_BYTES));
        commRows_ = tiling_->commInterval * tiling_->m0;
        copyTileElements_ = tiling_->commTileM * tiling_->k0;
        pipe->InitBuffer(copyBuffer_, LOCAL_COPY_UB_STAGES * copyTileElements_ * BF16_BYTES);
        for (uint32_t stage = 0; stage < LOCAL_COPY_UB_STAGES; ++stage) {
            copyEventIds_[stage] = stage;
            SetFlag<HardEvent::MTE3_MTE2>(copyEventIds_[stage]);
        }
    }

    __aicore__ inline void Process()
    {
        auto syncMmadFinish = reinterpret_cast<__gm__ int32_t *>(
            reinterpret_cast<__gm__ uint8_t *>(symmetric_) + tiling_->syncMmadOffset);
        auto syncCommFinish = reinterpret_cast<__gm__ int32_t *>(
            reinterpret_cast<__gm__ uint8_t *>(symmetric_) + tiling_->syncCommOffset);
        const bool isRootCore = GetBlockIdx() == 0;
        if (isRootCore) {
            aclshmemx_signal_op(syncMmadFinish, 0, ACLSHMEM_SIGNAL_SET, rank_);
            aclshmemx_signal_op(syncCommFinish, 0, ACLSHMEM_SIGNAL_SET, rank_);
        }
        aclshmemx_barrier_all_vec();

        uint32_t receiveAccum = 0;
        for (uint32_t round = 0; round < tiling_->commLoops; ++round) {
            const uint32_t stage = round % WORKSPACE_STAGE_COUNT;
            if (round >= WORKSPACE_STAGE_COUNT) {
                CrossCoreWaitFlag(stage);
                AivBarrier();
            }

            for (uint32_t srcPeer = 0; srcPeer < peerCount_; ++srcPeer) {
                const uint32_t recvRows = GetActualRoundRows(tiling_->recvCounts, srcPeer, round);
                receiveAccum += GetCompletedCommBlocks(recvRows, tiling_->commBlockM);
            }
            if (isRootCore) {
                aclshmemx_signal_op(syncMmadFinish, round + 1, ACLSHMEM_SIGNAL_SET, rank_);
            }

            const uint32_t aicoreIdx = GetBlockIdx() / GetSubBlockNum();
            const uint32_t subcoreIdx = GetSubBlockIdx();
            if (aicoreIdx < peerCount_) {
                const uint32_t dstPeer = aicoreIdx;
                const bool isLocal = dstPeer == rank_;
                const bool ownsPeer = (isLocal && subcoreIdx == 0) || (!isLocal && subcoreIdx == 1);
                const uint32_t sendRows = GetActualRoundRows(tiling_->sendCounts, dstPeer, round);
                if (ownsPeer && sendRows != 0) {
                    auto remoteMmadFinish = static_cast<__gm__ int32_t *>(shmem_ptr(syncMmadFinish, dstPeer));
                    aclshmem_signal_wait_until(remoteMmadFinish, ACLSHMEM_CMP_EQ, round + 1);
                    const uint64_t srcRow = GetPeerInputOffset(tiling_->sendCounts, dstPeer,
                                                              tiling_->localExpertNum) +
                                            static_cast<uint64_t>(round) * commRows_;
                    const uint64_t dstOffset = GetStagePeerOffset(stage, rank_, peerCount_, commRows_, tiling_->k,
                                                                  BF16_BYTES);
                    if (isLocal) {
                        CopyLocal2D(symmetric_ + dstOffset, alignedK_, x_ + srcRow * tiling_->k, tiling_->k,
                                    sendRows, tiling_->k);
                    } else {
                        PutRemote2D(symmetric_ + dstOffset, alignedK_, x_ + srcRow * tiling_->k, tiling_->k,
                                    sendRows, tiling_->k, dstPeer);
                    }
                    aclshmem_fence();
                    aclshmemx_signal_op(syncCommFinish,
                                        static_cast<int32_t>(GetCompletedCommBlocks(sendRows, tiling_->commBlockM)),
                                        ACLSHMEM_SIGNAL_ADD, dstPeer);
                }
            }

            if (isRootCore) {
                aclshmem_signal_wait_until(syncCommFinish, ACLSHMEM_CMP_EQ, receiveAccum);
            }
            AivBarrier();
            CrossCoreSetFlag<0x2, PIPE_MTE3>(stage);
        }
        for (uint32_t stage = 0; stage < LOCAL_COPY_UB_STAGES; ++stage) {
            WaitFlag<HardEvent::MTE3_MTE2>(copyEventIds_[stage]);
        }
    }

private:
    __aicore__ inline uint32_t GetActualRoundRows(const uint32_t *counts, uint32_t peer, uint32_t round) const
    {
        const uint32_t total = GetPeerTokenCount(counts, peer, tiling_->localExpertNum);
        const uint64_t begin = static_cast<uint64_t>(round) * commRows_;
        return static_cast<uint32_t>(MinU32(commRows_, static_cast<uint32_t>(ClipSubU64(total, begin))));
    }

    __aicore__ inline void AivBarrier() const
    {
        CrossCoreSetFlag<0x0, PIPE_MTE3>(AIV_BARRIER_FLAG);
        CrossCoreWaitFlag(AIV_BARRIER_FLAG);
    }

    __aicore__ inline void CopyLocal2D(__gm__ Element *dst, uint32_t dstStride, __gm__ Element *src,
                                       uint32_t srcStride, uint32_t rows, uint32_t columns)
    {
        for (uint32_t row = 0; row < rows; row += tiling_->commTileM) {
            const uint32_t tileRows = MinU32(tiling_->commTileM, rows - row);
            for (uint32_t column = 0; column < columns; column += tiling_->k0) {
                const uint32_t tileColumns = MinU32(tiling_->k0, columns - column);
                const uint32_t ubColumns = static_cast<uint32_t>(AlignUpU64(tileColumns, 32U / BF16_BYTES));
                const uint32_t stage = copyBufferIndex_;
                WaitFlag<HardEvent::MTE3_MTE2>(copyEventIds_[stage]);
                LocalTensor<Element> local = copyBuffer_.GetWithOffset<Element>(
                    copyTileElements_, stage * copyTileElements_ * BF16_BYTES);
                GlobalTensor<Element> srcTensor;
                GlobalTensor<Element> dstTensor;
                srcTensor.SetGlobalBuffer(src + static_cast<uint64_t>(row) * srcStride + column);
                dstTensor.SetGlobalBuffer(dst + static_cast<uint64_t>(row) * dstStride + column);

                DataCopyExtParams inParams{static_cast<uint16_t>(tileRows), tileColumns * BF16_BYTES,
                                           (srcStride - tileColumns) * BF16_BYTES,
                                           (ubColumns - tileColumns) * BF16_BYTES / 32U, 0U};
                DataCopyPadExtParams<Element> padParams{ubColumns != tileColumns, 0U,
                                                        static_cast<uint8_t>(ubColumns - tileColumns), 0};
                DataCopyPad(local, srcTensor, inParams, padParams);
                SetFlag<HardEvent::MTE2_MTE3>(copyEventIds_[stage]);
                WaitFlag<HardEvent::MTE2_MTE3>(copyEventIds_[stage]);
                DataCopyExtParams outParams{static_cast<uint16_t>(tileRows), tileColumns * BF16_BYTES,
                                            (ubColumns - tileColumns) * BF16_BYTES / 32U,
                                            (dstStride - tileColumns) * BF16_BYTES, 0U};
                DataCopyPad(dstTensor, local, outParams);
                SetFlag<HardEvent::MTE3_MTE2>(copyEventIds_[stage]);
                copyBufferIndex_ = (stage + 1U) % LOCAL_COPY_UB_STAGES;
            }
        }
    }

    __aicore__ inline void PutRemote2D(__gm__ Element *dst, uint32_t dstStride, __gm__ Element *src,
                                       uint32_t srcStride, uint32_t rows, uint32_t columns, uint32_t peer)
    {
        const uint64_t bytesPerRow = static_cast<uint64_t>(columns) * BF16_BYTES;
        if (rows == 1 || (srcStride == columns && dstStride == columns)) {
            SubmitUdma(dst, src, static_cast<uint32_t>(rows * bytesPerRow), peer);
        } else {
            for (uint32_t row = 0; row < rows; ++row) {
                SubmitUdma(dst + static_cast<uint64_t>(row) * dstStride,
                           src + static_cast<uint64_t>(row) * srcStride,
                           static_cast<uint32_t>(bytesPerRow), peer);
            }
        }
        aclshmemx_udma_quiet(peer);
    }

    __aicore__ inline void SubmitUdma(__gm__ Element *dst, __gm__ Element *src, uint32_t bytes, uint32_t peer)
    {
        aclshmemx_udma_put_nbi<uint8_t, PIPE_S>(reinterpret_cast<__gm__ uint8_t *>(dst),
                                                reinterpret_cast<__gm__ uint8_t *>(src),
                                                static_cast<__ubuf__ uint8_t *>(nullptr), bytes, peer, UDMA_SYNC_ID);
    }

    __gm__ Element *x_{nullptr};
    __gm__ Element *symmetric_{nullptr};
    const AlltoAllvGroupedMatMulShmemTilingData *tiling_{nullptr};
    uint32_t rank_{0};
    uint32_t peerCount_{0};
    uint32_t alignedK_{0};
    uint32_t commRows_{0};
    uint32_t copyTileElements_{0};
    uint32_t copyBufferIndex_{0};
    uint32_t copyEventIds_[LOCAL_COPY_UB_STAGES]{0};
    TBuf<TPosition::VECCALC> copyBuffer_;
};

template <class Matmul>
__aicore__ inline void RunAicGroupedMatmul(Matmul &mm, GM_ADDR weight, GM_ADDR output, GM_ADDR symmetric,
                                           const AlltoAllvGroupedMatMulShmemTilingData *tiling, uint32_t peerCount)
{
    GlobalTensor<Element> symmetricTensor;
    GlobalTensor<Element> weightTensor;
    GlobalTensor<Element> outputTensor;
    symmetricTensor.SetGlobalBuffer(reinterpret_cast<__gm__ Element *>(symmetric));
    weightTensor.SetGlobalBuffer(reinterpret_cast<__gm__ Element *>(weight));
    outputTensor.SetGlobalBuffer(reinterpret_cast<__gm__ Element *>(output));

    const uint32_t coreIdx = GetBlockIdx();
    const uint32_t coreCount = GetBlockNum();
    const uint32_t commRows = tiling->commInterval * tiling->m0;
    const uint32_t alignedK = static_cast<uint32_t>(
        AlignUpU64(tiling->k, WORKSPACE_ROW_ALIGNMENT_BYTES / BF16_BYTES));

    uint64_t taskBase = 0;
    for (uint32_t round = 0; round < tiling->commLoops; ++round) {
        const uint32_t stage = round % WORKSPACE_STAGE_COUNT;
        CrossCoreWaitFlag(stage);
        for (uint32_t expert = 0; expert < tiling->localExpertNum; ++expert) {
            for (uint32_t srcPeer = 0; srcPeer < peerCount; ++srcPeer) {
                const uint32_t count = tiling->recvCounts[srcPeer * tiling->localExpertNum + expert];
                const uint64_t expertBegin = GetExpertBeginInPeer(tiling->recvCounts, srcPeer, expert,
                                                                  tiling->localExpertNum);
                const RoundExpertSlice slice = GetRoundExpertSlice(
                    expertBegin, count, static_cast<uint64_t>(round) * commRows, commRows);
                const uint32_t mBlocks = static_cast<uint32_t>(CeilDivU64(slice.rows, tiling->m0));
                const uint32_t nBlocks = static_cast<uint32_t>(CeilDivU64(tiling->n, tiling->n0));
                const uint64_t taskCount = static_cast<uint64_t>(mBlocks) * nBlocks;
                const uint32_t baseRemainder = static_cast<uint32_t>(taskBase % coreCount);
                const uint32_t firstTask = coreIdx >= baseRemainder ? coreIdx - baseRemainder
                                                                    : coreCount + coreIdx - baseRemainder;
                for (uint64_t task = firstTask; task < taskCount; task += coreCount) {
                    const uint32_t mBlock = static_cast<uint32_t>(task / nBlocks);
                    const uint32_t nBlock = static_cast<uint32_t>(task % nBlocks);
                    const uint32_t offsetM = mBlock * tiling->m0;
                    const uint32_t offsetN = nBlock * tiling->n0;
                    const uint32_t actualM = MinU32(tiling->m0, slice.rows - offsetM);
                    const uint32_t actualN = MinU32(tiling->n0, tiling->n - offsetN);
                    const uint64_t stagePeerOffset = GetStagePeerOffset(stage, srcPeer, peerCount, commRows,
                                                                        tiling->k, BF16_BYTES);
                    const uint64_t aOffset = stagePeerOffset +
                                             static_cast<uint64_t>(slice.stageRow + offsetM) * alignedK;
                    const uint64_t bOffset = static_cast<uint64_t>(expert) * tiling->k * tiling->n + offsetN;
                    const uint64_t outputBase = GetExpertOutputOffset(tiling->recvCounts, expert, srcPeer, peerCount,
                                                                      tiling->localExpertNum);
                    const uint64_t cOffset = (outputBase + slice.expertRow + offsetM) * tiling->n + offsetN;

                    mm.SetOrgShape(commRows, tiling->n, alignedK, tiling->k, tiling->n);
                    mm.SetSingleShape(actualM, actualN, tiling->k);
                    mm.SetTensorA(symmetricTensor[aOffset], false);
                    mm.SetTensorB(weightTensor[bOffset], false);
                    mm.template IterateAll<false>(outputTensor[cOffset], false);
                }
                taskBase += taskCount;
            }
        }
        if (round + WORKSPACE_STAGE_COUNT < tiling->commLoops) {
            CrossCoreSetFlag<0x2, PIPE_FIX>(stage);
        }
    }
    mm.End();
}

} // namespace AlltoAllvGmmShmem

#endif // ALLTO_ALLV_GROUPED_MAT_MUL_SHMEM_H
