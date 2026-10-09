/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file block_sparse_attention_grad_kernel_wide.h
 * \brief Block Sparse Attention Grad Kernel Wide
 */

#ifndef BSAG_KERNEL_WIDE_FRAGMENT
#define BSAG_KERNEL_WIDE_FRAGMENT

// A work item stays within one batch/head; compact per-Q metadata bounds AIV stack use.
struct WideQInfo {
    uint64_t qOffset;
    uint32_t curQSeqIdx;
    uint32_t curQBlockIdx;
    uint32_t curCalQSize;
};

struct SegmentInfo {
    TaskInfo tiles[GROUPED_TILES];
    uint32_t tileCount;
    uint32_t uniqueKvCount;
    uint32_t nextKvStart;
};

// A wide packet stores coordinates instead of duplicating TaskInfo for
// every active edge.  The compact representation serves K-major Cube1 /
// dK,dV traversal and Q-major dQ/vector traversal.
struct WideKvDesc {
    uint32_t kvBasicIdx;
    uint16_t edgeBegin;
    uint16_t edgeCount;
};

// Q-major runs index active edges only. Byte fields fit the cap-8 packet in one scalar word.
struct WideQDesc {
    uint8_t qLocal;
    uint8_t edgeBegin;
    uint8_t edgeCount;
    uint8_t reserved;
};

struct WidePacketInfo {
    // low 8 bits: qLocal; high 16 bits: kvLocal.  The workspace slot
    // is the K-major edge index itself.
    uint32_t edgeCode[WIDE_MAX_EDGES];
    WideKvDesc kv[WIDE_MAX_KV_COLUMNS];
    uint16_t qOrder[WIDE_MAX_EDGES];
    WideQDesc activeQ[WIDE_MAX_EDGES];
    uint32_t edgeCount;
    uint32_t uniqueKvCount;
    uint32_t activeQCount;
    uint32_t nextKvStart;
    uint32_t nextQStart;
    uint64_t workspaceBase;
};

// Reuse original mask bytes across packets of one work item; no bit packing.
// Four-byte windows bound AIV stack use when scanning the same KV columns.
struct WideMaskScanState {
    uint32_t maskWindows[WIDE_MAX_Q_TASKS];
    WideQMask loadedMaskWindows;
    uint32_t maskWindowStart;
    uint32_t maskWindowEnd;
};

__aicore__ inline void ResetWideMaskScanState(WideMaskScanState &state) const
{
    // maskWindows is initialized lazily and is read only when the
    // corresponding loaded bit is set.
    state.loadedMaskWindows = 0;
    state.maskWindowStart = ~0U;
    state.maskWindowEnd = 0;
}

struct MaskLayoutInfo {
    uint32_t kvBlockCount;
    uint64_t headStride;
    uint64_t batchStride;
};

__aicore__ inline uint32_t WideEdgeQ(uint32_t edgeCode) const
{
    return edgeCode & 0xffU;
}

__aicore__ inline uint32_t WideEdgeKv(uint32_t edgeCode) const
{
    return (edgeCode >> 8) & 0xffffU;
}

__aicore__ inline void BuildWidePacket(const Params &params, const TaskInfo &workInfo,
                                       const __gm__ BlockSparseAttentionGradTilingData *tilingData,
                                       const uint32_t *qLimits, const uint64_t *maskRowOffsets,
                                       const MaskLayoutInfo &maskLayout, uint32_t qCount, uint32_t maxKvLimit,
                                       uint32_t kvStart, uint32_t qStart, uint64_t workspaceBase,
                                       WideMaskScanState &maskState, WidePacketInfo &packet) const
{
    packet.edgeCount = 0;
    packet.uniqueKvCount = 0;
    packet.activeQCount = 0;
    packet.nextKvStart = kvStart;
    packet.nextQStart = qStart;
    packet.workspaceBase = workspaceBase;
    const uint32_t requestedEdgeCapacity =
        tilingData->widePacketEdgeCapacity < WIDE_MAX_EDGES ? tilingData->widePacketEdgeCapacity : WIDE_MAX_EDGES;
    // Advance the cursor even for zero capacity to avoid an infinite packet loop.
    const uint32_t edgeCapacity = requestedEdgeCapacity == 0 ? 1 : requestedEdgeCapacity;

    uint32_t kvBasicIdx = kvStart;
    uint32_t qScanStart = qStart;
    if (qScanStart >= qCount && kvBasicIdx < maxKvLimit) {
        ++kvBasicIdx;
        qScanStart = 0;
        packet.nextKvStart = kvBasicIdx;
        packet.nextQStart = 0;
    }
    // Cache one stable four-column byte window per Q row.  The
    // packet remains KV-major, so column grouping and K/V reuse are
    // unchanged; inner-loop mask tests become scalar shifts instead
    // of one strided GM byte transaction per (KV,Q) candidate.
    constexpr uint32_t MASK_WINDOW_BYTES = sizeof(uint32_t);
    while (kvBasicIdx < maxKvLimit && packet.edgeCount < edgeCapacity && packet.uniqueKvCount < WIDE_MAX_KV_COLUMNS) {
        // Use per-head counts on both AIC and AIV to keep packet traversal identical.
        if (PerBlockPrefix(workInfo.curBatchIdx, workInfo.curHeadIdx, kvBasicIdx, 1, tilingData->basicKVBlockSize,
                           tilingData) == 0) {
            ++kvBasicIdx;
            qScanStart = 0;
            packet.nextKvStart = kvBasicIdx;
            packet.nextQStart = 0;
            continue;
        }
        if (kvBasicIdx < maskState.maskWindowStart || kvBasicIdx >= maskState.maskWindowEnd) {
            maskState.maskWindowStart = kvBasicIdx & ~(MASK_WINDOW_BYTES - 1U);
            maskState.maskWindowEnd = maskState.maskWindowStart + MASK_WINDOW_BYTES;
            if (maskState.maskWindowEnd > maxKvLimit) {
                maskState.maskWindowEnd = maxKvLimit;
            }
            maskState.loadedMaskWindows = 0;
        }
        const uint32_t kvLocal = packet.uniqueKvCount;
        const uint32_t columnBegin = packet.edgeCount;
        uint32_t qLocal = qScanStart;
        const uint32_t maskShift = (kvBasicIdx - maskState.maskWindowStart) * 8U;
        for (; qLocal < qCount; ++qLocal) {
            if (kvBasicIdx >= qLimits[qLocal]) {
                continue;
            }
            const WideQMask qWindowBit = static_cast<WideQMask>(1) << qLocal;
            if ((maskState.loadedMaskWindows & qWindowBit) == 0) {
                const uint32_t qWindowCandidateEnd = maskState.maskWindowStart + MASK_WINDOW_BYTES;
                const uint32_t qWindowEnd =
                    qWindowCandidateEnd < qLimits[qLocal] ? qWindowCandidateEnd : qLimits[qLocal];
                maskState.maskWindows[qLocal] = LoadMaskByteWindow(
                    params, maskRowOffsets[qLocal] + maskState.maskWindowStart, qWindowEnd - maskState.maskWindowStart,
                    maskRowOffsets[qLocal] + maskLayout.kvBlockCount);
                maskState.loadedMaskWindows |= qWindowBit;
            }
            if (((maskState.maskWindows[qLocal] >> maskShift) & 0xffU) == 0) {
                continue;
            }
            if (packet.edgeCount == edgeCapacity) {
                break;
            }
            packet.edgeCode[packet.edgeCount] = qLocal | (kvLocal << 8);
            ++packet.edgeCount;
        }

        if (packet.edgeCount != columnBegin) {
            WideKvDesc &kv = packet.kv[packet.uniqueKvCount++];
            kv.kvBasicIdx = kvBasicIdx;
            kv.edgeBegin = static_cast<uint16_t>(columnBegin);
            kv.edgeCount = static_cast<uint16_t>(packet.edgeCount - columnBegin);
        }

        if (qLocal < qCount) {
            // The physical packet is full in the middle of a dense
            // KV column.  Resume at the first unconsumed Q edge in the
            // next packet.  dK/dV already use atomic final stores, so
            // the partial-column reductions remain exact while common
            // sparse columns are still reduced into one store.
            packet.nextKvStart = kvBasicIdx;
            packet.nextQStart = qLocal;
            break;
        }
        ++kvBasicIdx;
        qScanStart = 0;
        packet.nextKvStart = kvBasicIdx;
        packet.nextQStart = 0;
    }

    // Stable-sort packet-local edges by Q while preserving KV order within each Q.
    for (uint32_t edge = 0; edge < packet.edgeCount; ++edge) {
        packet.qOrder[edge] = static_cast<uint16_t>(edge);
    }
    for (uint32_t ordered = 1; ordered < packet.edgeCount; ++ordered) {
        const uint16_t key = packet.qOrder[ordered];
        const uint32_t keyQ = WideEdgeQ(packet.edgeCode[key]);
        uint32_t insert = ordered;
        while (insert > 0) {
            const uint16_t previous = packet.qOrder[insert - 1];
            if (WideEdgeQ(packet.edgeCode[previous]) <= keyQ) {
                break;
            }
            packet.qOrder[insert] = previous;
            --insert;
        }
        packet.qOrder[insert] = key;
    }
    for (uint32_t ordered = 0; ordered < packet.edgeCount; ++ordered) {
        const uint32_t qLocal = WideEdgeQ(packet.edgeCode[packet.qOrder[ordered]]);
        if (packet.activeQCount == 0 || packet.activeQ[packet.activeQCount - 1].qLocal != qLocal) {
            WideQDesc &desc = packet.activeQ[packet.activeQCount++];
            desc.qLocal = static_cast<uint8_t>(qLocal);
            desc.edgeBegin = static_cast<uint8_t>(ordered);
            desc.edgeCount = 1;
            desc.reserved = 0;
        } else {
            ++packet.activeQ[packet.activeQCount - 1].edgeCount;
        }
    }
}

__aicore__ inline void RunWideCube1(const WidePacketInfo &packet, const TaskInfo &workInfo, const WideQInfo *qInfos,
                                    const __gm__ BlockSparseAttentionGradTilingData *tilingData,
                                    AscendC::GlobalTensor<ElementInput> gQ, AscendC::GlobalTensor<ElementInput> gK,
                                    AscendC::GlobalTensor<ElementInput> gV, AscendC::GlobalTensor<ElementA1> gDout,
                                    AscendC::GlobalTensor<float> gS, AscendC::GlobalTensor<float> gDp,
                                    BlockMmadBSAG1 &blockMmad1, uint32_t &wideCubeFlag, BlockMmadBSAG3 &blockMmad3)
{
    const uint32_t headDim = tilingData->headDim;
    // Drain the previous phase before reusing Cube1's L1/L0 regions.
    AscendC::PipeBarrier<PIPE_ALL>();
    for (uint32_t kvLocal = 0; kvLocal < packet.uniqueKvCount; ++kvLocal) {
        const WideKvDesc &kv = packet.kv[kvLocal];
        TaskInfo firstTile = MakeWideTile(workInfo, qInfos, packet, kv.edgeBegin, tilingData);
        LayoutB1 layoutB(headDim, firstTile.curCalKVSize, headDim);
        GemmCoord preloadShape{firstTile.curCalQSize, firstTile.curCalKVSize, headDim};
        // Each cache slot waits for its final MTE1 read before overwrite.
        blockMmad1.PreloadB(gK[firstTile.kvOffset], layoutB, preloadShape, 0, tilingData->wideFastPipeline != 0);
        blockMmad1.PreloadB(gV[firstTile.kvOffset], layoutB, preloadShape, 1, tilingData->wideFastPipeline != 0);
        const uint32_t edgeEnd = kv.edgeBegin + kv.edgeCount;
        for (uint32_t edge = kv.edgeBegin; edge < edgeEnd; ++edge) {
            // Complete the preceding MMAD before reusing its operands.
            AscendC::PipeBarrier<PIPE_M>();
            TaskInfo tile = MakeWideTile(workInfo, qInfos, packet, edge, tilingData);
            LayoutA1 layoutA(tile.curCalQSize, headDim, headDim);
            GemmCoord shape{tile.curCalQSize, tile.curCalKVSize, headDim};
            const bool cacheQ = tilingData->wideQOperandCache != 0;
            auto localQ = blockMmad3.GetCachedB(cacheQ ? tile.groupQIdx + GROUPED_Q_BLOCKS : 0);
            auto localDo = blockMmad3.GetCachedB(cacheQ ? tile.groupQIdx : 0);
            blockMmad1.WithCachedBNz(0, gQ[tile.qOffset], gS[tile.sOffset], layoutA, layoutB, shape, wideCubeFlag,
                                     cacheQ ? &localQ : nullptr, edge + 1 == edgeEnd);
            blockMmad1.WithCachedBNz(1, gDout[tile.qOffset], gDp[tile.sOffset], layoutA, layoutB, shape, wideCubeFlag,
                                     cacheQ ? &localDo : nullptr, edge + 1 == edgeEnd);
        }
    }
}

__aicore__ inline void WaitWideFlag() const
{
    // MTE1_MTE2 event 7 is the explicit wide L1-region reuse token.
    // Events 0/1 belong to Cube1, 2/3 to the native-zN dQ/P/dS
    // streams, and 4/5 guard the paired dOut/Q cache.  Event 6 stays
    // seeded so all SetFlag() tokens are drained exactly once.
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID5);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID6);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID7);
}

// Reduce every active K column of a packet into one dQ Fixpipe store.
// qSeenMask spans all packets of the work item: the first non-empty
// packet is a direct write (the Q task has a unique core owner), and
// later packets atomically add their disjoint K-range contribution.
__aicore__ inline void RunWideDq(const WidePacketInfo &packet, const TaskInfo &workInfo, const WideQInfo *qInfos,
                                 uint32_t qCount, const __gm__ BlockSparseAttentionGradTilingData *tilingData,
                                 AscendC::GlobalTensor<ElementInput> gK, AscendC::GlobalTensor<ElementInput> gDs,
                                 AscendC::GlobalTensor<float> gDq, BlockMmadBSAG2 &blockMmad2, WideQMask &qSeenMask)
{
    const uint32_t headDim = tilingData->headDim;

    // A wide packet is K-major and contains at most eight distinct KV
    // columns.  Cache each K tile once in the shared high-L1 region;
    // the Q-major reduction below addresses it through kvLocal.
    for (uint32_t kvLocal = 0; kvLocal < packet.uniqueKvCount; ++kvLocal) {
        const uint32_t edge = packet.kv[kvLocal].edgeBegin;
        TaskInfo tile = MakeWideTile(workInfo, qInfos, packet, edge, tilingData);
        LayoutB2 layoutB(tile.curCalKVSize, headDim, headDim);
        GemmCoord shape{tile.curCalQSize, headDim, tile.curCalKVSize};
        blockMmad2.PreloadFullB(gK[tile.kvOffset], layoutB, shape, kvLocal);
    }
    blockMmad2.FinishFullPreload();

    for (uint32_t activeQ = 0; activeQ < packet.activeQCount; ++activeQ) {
        // Per-slot MTE/M/FIX events guard dQ operand and accumulator reuse.
        const WideQDesc &qDesc = packet.activeQ[activeQ];
        const uint32_t qLocal = qDesc.qLocal;
        const uint32_t qBegin = qDesc.edgeBegin;
        const uint32_t edgeCount = qDesc.edgeCount;
        TaskInfo lastTile;
        const uint32_t initial = edgeCount < 2 ? edgeCount : 2;
        for (uint32_t i = 0; i < initial; ++i) {
            const uint32_t edge = packet.qOrder[qBegin + i];
            TaskInfo tile = MakeWideTile(workInfo, qInfos, packet, edge, tilingData);
            GemmCoord shape{tile.curCalQSize, headDim, tile.curCalKVSize};
            blockMmad2.PreloadAccumANz(gDs[tile.sOffset], shape, i);
        }
        for (uint32_t i = 0; i < edgeCount; ++i) {
            // Complete the preceding MMAD before reusing its operands.
            AscendC::PipeBarrier<PIPE_M>();
            const uint32_t slot = i & 1U;
            const uint32_t edge = packet.qOrder[qBegin + i];
            const uint32_t kvLocal = WideEdgeKv(packet.edgeCode[edge]);
            TaskInfo tile = MakeWideTile(workInfo, qInfos, packet, edge, tilingData);
            LayoutA2 layoutA(tile.curCalQSize, tile.curCalKVSize);
            LayoutB2 layoutB(tile.curCalKVSize, headDim);
            GemmCoord shape{tile.curCalQSize, headDim, tile.curCalKVSize};
            blockMmad2.AccumulatePreloadedA(slot, kvLocal, layoutA, layoutB, shape, i == 0, i + 1 == edgeCount,
                                            qLocal & 1U);
            lastTile = tile;
            if (i + 2 < edgeCount) {
                const uint32_t nextEdge = packet.qOrder[qBegin + i + 2];
                TaskInfo nextTile = MakeWideTile(workInfo, qInfos, packet, nextEdge, tilingData);
                GemmCoord nextShape{nextTile.curCalQSize, headDim, nextTile.curCalKVSize};
                blockMmad2.PreloadAccumANz(gDs[nextTile.sOffset], nextShape, slot);
            }
        }
        LayoutC2 layoutC(lastTile.curCalQSize, headDim, headDim);
        GemmCoord outShape{lastTile.curCalQSize, headDim, lastTile.curCalKVSize};
        const WideQMask qBit = static_cast<WideQMask>(1) << qLocal;
        blockMmad2.FlushAccumulator(gDq[lastTile.qOffset], layoutC, outShape, (qSeenMask & qBit) != 0, qLocal & 1U);
        qSeenMask |= qBit;
    }
}

// dK/dV may receive contributions from several Q work items and
// cores.  Traverse each KV column once and use the eight persistent-B
// slots as four (dO,Q) pairs.  P and dS stream through independent A
// slots and L0C accumulators, retaining one atomic Fixpipe store per
// KV column and matrix while halving cache fences and descriptor work.
__aicore__ inline void RunWideDkvPair(const WidePacketInfo &packet, const TaskInfo &workInfo, const WideQInfo *qInfos,
                                      const __gm__ BlockSparseAttentionGradTilingData *tilingData,
                                      AscendC::GlobalTensor<ElementInput> gQ, AscendC::GlobalTensor<ElementA1> gDout,
                                      AscendC::GlobalTensor<ElementInput> gP, AscendC::GlobalTensor<ElementInput> gDs,
                                      AscendC::GlobalTensor<float> gDk, AscendC::GlobalTensor<float> gDv,
                                      BlockMmadBSAG2 &blockMmad2, BlockMmadBSAG3 &blockMmad3)
{
    const uint32_t headDim = tilingData->headDim;
    if (tilingData->wideQOperandCache != 0) {
        // Retire dQ's L1 reads before reusing the P/dS slots. Q/dO
        // remain resident; dKV does not overwrite the K cache.
        // Shared M_MTE1 and FIX_M bank events below protect L0A/B
        // and L0C, allowing dQ Fixpipe to overlap operand movement.
        SyncExtendedL1Reuse();
    } else {
        // The refill path aliases additional L1 regions.
        AscendC::PipeBarrier<PIPE_ALL>();
    }
    for (uint32_t kvLocal = 0; kvLocal < packet.uniqueKvCount; ++kvLocal) {
        const WideKvDesc &kv = packet.kv[kvLocal];
        const uint32_t edgeEnd = kv.edgeBegin + kv.edgeCount;
        uint32_t issued = 0;
        uint32_t cursor = kv.edgeBegin;
        TaskInfo lastTile;
        while (cursor < edgeEnd) {
            // Complete the preceding MMAD batch before refilling the operand cache.
            AscendC::PipeBarrier<PIPE_M>();
            const uint32_t batchCount = edgeEnd - cursor < GROUPED_Q_BLOCKS ? edgeEnd - cursor : GROUPED_Q_BLOCKS;
            // Events 4/5 were published by the final dV/dK L1 reads
            // of the preceding batch (or seeded at kernel entry).
            // Consume them before replacing its eight B-cache slots.
            if (tilingData->wideQOperandCache == 0) {
                blockMmad3.AcquireFullPreloadReuse();
                for (uint32_t i = 0; i < batchCount; ++i) {
                    TaskInfo tile = MakeWideTile(workInfo, qInfos, packet, cursor + i, tilingData);
                    LayoutB3 layoutB(tile.curCalQSize, headDim, headDim);
                    GemmCoord shape{tile.curCalKVSize, headDim, tile.curCalQSize};
                    blockMmad3.PreloadFullB(gDout[tile.qOffset], layoutB, shape, i);
                    blockMmad3.PreloadFullB(gQ[tile.qOffset], layoutB, shape, i + GROUPED_Q_BLOCKS);
                }
                blockMmad3.FinishFullPreload();
            }

            TaskInfo firstTile = MakeWideTile(workInfo, qInfos, packet, cursor, tilingData);
            GemmCoord firstSourceShape{firstTile.curCalQSize, headDim, firstTile.curCalKVSize};
            // P and dS are zN[Q,KV].  Stream them through the compact
            // low-L1 slots, then transpose each 16x16 fractal only on
            // L1->L0A for P^T*dO and dS^T*Q.
            blockMmad2.PreloadAccumANz(gP[firstTile.sOffset], firstSourceShape, 0);
            blockMmad2.PreloadAccumANz(gDs[firstTile.sOffset], firstSourceShape, 1);
            for (uint32_t i = 0; i < batchCount; ++i) {
                TaskInfo tile = MakeWideTile(workInfo, qInfos, packet, cursor + i, tilingData);
                LayoutB3 layoutB(tile.curCalQSize, headDim, headDim);
                GemmCoord shape{tile.curCalKVSize, headDim, tile.curCalQSize};
                const bool initC = issued == 0;
                const bool finalC = issued + 1 == kv.edgeCount;

                blockMmad2.FinishAccumPreloadForExternal(0);
                blockMmad3.AccumulateTransposeFromExternalZNL1(
                    blockMmad2.GetAccumL1A(0), PINGPONG_OFFSET_2,
                    tilingData->wideQOperandCache != 0 ? tile.groupQIdx : i, layoutB, shape, 0, initC, finalC,
                    tilingData->wideQOperandCache == 0 && i + 1 == batchCount);
                if (i + 1 < batchCount) {
                    TaskInfo nextTile = MakeWideTile(workInfo, qInfos, packet, cursor + i + 1, tilingData);
                    GemmCoord nextSourceShape{nextTile.curCalQSize, headDim, nextTile.curCalKVSize};
                    blockMmad2.PreloadAccumANz(gP[nextTile.sOffset], nextSourceShape, 0);
                }

                blockMmad2.FinishAccumPreloadForExternal(1);
                blockMmad3.AccumulateTransposeFromExternalZNL1(
                    blockMmad2.GetAccumL1A(1), PINGPONG_OFFSET_2 + 1,
                    (tilingData->wideQOperandCache != 0 ? tile.groupQIdx : i) + GROUPED_Q_BLOCKS, layoutB, shape, 1,
                    initC, finalC, tilingData->wideQOperandCache == 0 && i + 1 == batchCount);
                ++issued;
                lastTile = tile;
                if (i + 1 < batchCount) {
                    TaskInfo nextTile = MakeWideTile(workInfo, qInfos, packet, cursor + i + 1, tilingData);
                    GemmCoord nextSourceShape{nextTile.curCalQSize, headDim, nextTile.curCalKVSize};
                    blockMmad2.PreloadAccumANz(gDs[nextTile.sOffset], nextSourceShape, 1);
                }
            }
            cursor += batchCount;
        }
        LayoutC3 layoutC(lastTile.curCalKVSize, headDim, headDim);
        GemmCoord outShape{lastTile.curCalKVSize, headDim, lastTile.curCalQSize};
        blockMmad3.FlushAccumulator(gDv[lastTile.kvOffset], layoutC, outShape, true, 0);
        blockMmad3.FlushAccumulator(gDk[lastTile.kvOffset], layoutC, outShape, true, 1);
    }
}

__aicore__ inline void RunWideGrad(const WidePacketInfo &packet, const TaskInfo &workInfo, const WideQInfo *qInfos,
                                   uint32_t qCount, const __gm__ BlockSparseAttentionGradTilingData *tilingData,
                                   AscendC::GlobalTensor<ElementInput> gK, AscendC::GlobalTensor<ElementInput> gQ,
                                   AscendC::GlobalTensor<ElementA1> gDout, AscendC::GlobalTensor<ElementInput> gP,
                                   AscendC::GlobalTensor<ElementInput> gDs, AscendC::GlobalTensor<float> gDq,
                                   AscendC::GlobalTensor<float> gDk, AscendC::GlobalTensor<float> gDv,
                                   BlockMmadBSAG2 &blockMmad2, BlockMmadBSAG3 &blockMmad3, WideQMask &qSeenMask)
{
    // Cube1 and dQ alias the compact low-L1 region; the high eight-B
    // cache may still be feeding the preceding packet's dKV pair.
    // Complete dQ first because both BlockMmad objects alias the same
    // physical L0 banks, then reuse low L1 as native-zN P/dS streams
    // while dOut/Q remain resident in the high eight-slot cache.
    SyncExtendedL1Reuse();
    if (tilingData->wideQOperandCache == 0) {
        blockMmad3.AcquireFullPreloadReuse();
    }
    RunWideDq(packet, workInfo, qInfos, qCount, tilingData, gK, gDs, gDq, blockMmad2, qSeenMask);
    if (tilingData->wideQOperandCache == 0) {
        blockMmad3.ReleaseFullPreloadReuse();
    }
    RunWideDkvPair(packet, workInfo, qInfos, tilingData, gQ, gDout, gP, gDs, gDk, gDv, blockMmad2, blockMmad3);
}

__aicore__ inline uint32_t BuildWideWorkItem(AscendC::GlobalTensor<int64_t> gActualQseqlen,
                                             AscendC::GlobalTensor<int64_t> gActualKvseqlen,
                                             const __gm__ BlockSparseAttentionGradTilingData *tilingData,
                                             uint32_t groupSize, const TaskInfo &current, WideQInfo *qInfos,
                                             TaskInfo &nextWork, bool needNext)
{
    const uint32_t runtimeCapacity =
        tilingData->wideQTasksPerWorkItem < WIDE_MAX_Q_TASKS ? tilingData->wideQTasksPerWorkItem : WIDE_MAX_Q_TASKS;
    TaskInfo cursor = current;
    uint32_t qCount = 0;
    while (qCount < runtimeCapacity) {
        WideQInfo &qInfo = qInfos[qCount++];
        qInfo.qOffset = cursor.qOffset;
        qInfo.curQSeqIdx = cursor.curQSeqIdx;
        qInfo.curQBlockIdx = cursor.curQBlcokIdx;
        qInfo.curCalQSize = PerBlockPrefix(cursor.curBatchIdx, cursor.curHeadIdx, cursor.curQBlcokIdx, 0,
                                           cursor.curCalQSize, tilingData, cursor.curQSeqIdx % tilingData->blockShapeX);
        if (qCount == runtimeCapacity || cursor.curQSeqIdx + cursor.curCalQSize == cursor.qSeqlen) {
            break;
        }
        TaskInfo candidate;
        updateNextGroupedTaskInfo(gActualQseqlen, gActualKvseqlen, tilingData->numHeads, tilingData->kvHeads, groupSize,
                                  tilingData->headDim, tilingData->blockShapeX, tilingData->basicQBlockSize,
                                  tilingData->inputLayout, cursor, candidate);
        cursor = candidate;
    }
    if (needNext) {
        if (tilingData->wideTaskStride != 0) {
            // BNSD batches have fixed physical sequence lengths.
            // Map a cyclic work-item index back to the original
            // user-block fragments without crossing their boundaries.
            const uint32_t blockX = tilingData->blockShapeX;
            const uint32_t fragments = (blockX + 127) / 128;
            const uint32_t fullBlocks = current.qSeqlen / blockX;
            const uint32_t qTasks = fullBlocks * fragments + (current.qSeqlen - fullBlocks * blockX + 127) / 128;
            const uint32_t workItems = (qTasks + runtimeCapacity - 1) / runtimeCapacity;
            const uint32_t qTask = (current.curQSeqIdx / blockX) * fragments + (current.curQSeqIdx % blockX) / 128;
            const uint32_t workIndex = (current.curBatchIdx * tilingData->numHeads + current.curHeadIdx) * workItems +
                                       qTask / runtimeCapacity + tilingData->wideTaskStride;
            const uint32_t headIndex = workIndex / workItems;
            const uint32_t nextQTask = (workIndex % workItems) * runtimeCapacity;
            nextWork = current;
            nextWork.curBatchIdx = headIndex / tilingData->numHeads;
            nextWork.curHeadIdx = headIndex % tilingData->numHeads;
            nextWork.curQSeqIdx = (nextQTask / fragments) * blockX + (nextQTask % fragments) * 128;
            nextWork.qOffset =
                (static_cast<uint64_t>(headIndex) * tilingData->maxQSeqlen + nextWork.curQSeqIdx) * tilingData->headDim;
            nextWork.kvOffset =
                (static_cast<uint64_t>(nextWork.curBatchIdx) * tilingData->kvHeads + nextWork.curHeadIdx / groupSize) *
                tilingData->maxKvSeqlen * tilingData->headDim;
            UpdateTaskInfoCalQSize(blockX, tilingData->basicQBlockSize, nextWork);
        } else {
            updateNextGroupedTaskInfo(gActualQseqlen, gActualKvseqlen, tilingData->numHeads, tilingData->kvHeads,
                                      groupSize, tilingData->headDim, tilingData->blockShapeX,
                                      tilingData->basicQBlockSize, tilingData->inputLayout, cursor, nextWork);
        }
    }
    return qCount;
}

__aicore__ inline void ProcessWideAic(const Params &params, __gm__ BlockSparseAttentionGradTilingData *tilingData)
{
    const uint32_t coreIdx = AscendC::GetBlockIdx();
    const uint32_t groupSize = tilingData->numHeads / tilingData->kvHeads;
    const uint32_t workLength = tilingData->taskNumPerCore + (tilingData->tailTaskNum > coreIdx ? 1 : 0);

    AscendC::GlobalTensor<ElementInput> gQ;
    AscendC::GlobalTensor<ElementInput> gK;
    AscendC::GlobalTensor<ElementInput> gV;
    AscendC::GlobalTensor<ElementA1> gDout;
    AscendC::GlobalTensor<int64_t> gActualQseqlen;
    AscendC::GlobalTensor<int64_t> gActualKvseqlen;
    AscendC::GlobalTensor<float> gS;
    AscendC::GlobalTensor<float> gDp;
    AscendC::GlobalTensor<ElementInput> gP;
    AscendC::GlobalTensor<ElementInput> gDs;
    AscendC::GlobalTensor<float> gDq;
    AscendC::GlobalTensor<float> gDk;
    AscendC::GlobalTensor<float> gDv;
    gQ.SetGlobalBuffer((__gm__ ElementInput *)params.q);
    gK.SetGlobalBuffer((__gm__ ElementInput *)params.k);
    gV.SetGlobalBuffer((__gm__ ElementInput *)params.v);
    gDout.SetGlobalBuffer((__gm__ ElementA1 *)params.dout);
    gActualQseqlen.SetGlobalBuffer((__gm__ int64_t *)params.actualQseqlen);
    gActualKvseqlen.SetGlobalBuffer((__gm__ int64_t *)params.actualKvseqlen);
    gS.SetGlobalBuffer((__gm__ float *)params.workspace);
    gP.SetGlobalBuffer((__gm__ ElementInput *)(params.workspace + tilingData->wideRawPlaneBytes));
    gDp.SetGlobalBuffer((__gm__ float *)(params.workspace + tilingData->sOutSize));
    gDs.SetGlobalBuffer(
        (__gm__ ElementInput *)(params.workspace + tilingData->sOutSize + tilingData->wideRawPlaneBytes));
    gDq.SetGlobalBuffer((__gm__ float *)(params.workspace + tilingData->sOutSize + tilingData->dPOutSize));
    gDk.SetGlobalBuffer(
        (__gm__ float *)(params.workspace + tilingData->sOutSize + tilingData->dPOutSize + tilingData->dQOutSize));
    gDv.SetGlobalBuffer((__gm__ float *)(params.workspace + tilingData->sOutSize + tilingData->dPOutSize +
                                         tilingData->dQOutSize + tilingData->dKOutSize));

    TaskInfo current;
    initTaskInfo(gActualQseqlen, gActualKvseqlen, tilingData, tilingData->numHeads, tilingData->kvHeads, groupSize,
                 tilingData->headDim, tilingData->maxQSeqlen, tilingData->maxKvSeqlen, tilingData->blockShapeX,
                 tilingData->basicQBlockSize, tilingData->inputLayout, coreIdx, current);

    // The resident path partitions L1 into P/dS streams [0,64) KiB,
    // eight K slots [64,320) KiB, and Q/dO pairs [320,512) KiB.
    // Q/dO survive Cube1 and both gradient phases of the whole work
    // item. The legacy path retains its four-Q refill batches.
    BlockMmadBSAG1 blockMmad1(resource);
    BlockMmadBSAG2 blockMmad2(resource, 0, PINGPONG_OFFSET_2, true, false, true, false,
                              tilingData->wideQOperandCache != 0 ? 1 : 0);
    BlockMmadBSAG3 blockMmad3(resource, L1_SIZE_OFFSET * 7 / 2, PINGPONG_OFFSET_4, true, false, true, false,
                              tilingData->wideQOperandCache != 0 ? 2 : 0);
    blockMmad1.SetFastCachedInputs(tilingData->wideFastPipeline != 0);
    blockMmad3.SetFastCachedInputs(tilingData->wideFastPipeline != 0);
    const MaskLayoutInfo maskLayout = GetMaskLayout(tilingData);
    uint32_t mmapFlag = 0;
    uint32_t processedWorks = 0;
    uint32_t dbgPackets = 0;

    AscendC::SyncAll<false>();
    SetFlag();
    // Match FA's ND/NZ load semantics and make every padded lane
    // deterministic for short Q/KV tails.
    AscendC::SetLoadDataPaddingValue<uint64_t>(0);
    AscendC::SetNdParaImpl(0x1);
    while (processedWorks < workLength) {
        WideQInfo qInfos[WIDE_MAX_Q_TASKS];
        TaskInfo nextWork;
        const uint32_t qCount = BuildWideWorkItem(gActualQseqlen, gActualKvseqlen, tilingData, groupSize, current,
                                                  qInfos, nextWork, processedWorks + 1 < workLength);
        if (tilingData->wideQOperandCache != 0) {
            blockMmad3.AcquireFullPreloadReuse();
            for (uint32_t qLocal = 0; qLocal < qCount; ++qLocal) {
                const auto &q = qInfos[qLocal];
                if (q.curCalQSize == 0) {
                    continue;
                }
                const LayoutB3 layout(q.curCalQSize, tilingData->headDim, tilingData->headDim);
                const GemmCoord shape{128, tilingData->headDim, q.curCalQSize};
                blockMmad3.PreloadFullB(gDout[q.qOffset], layout, shape, qLocal);
                blockMmad3.PreloadFullB(gQ[q.qOffset], layout, shape, qLocal + GROUPED_Q_BLOCKS);
            }
            blockMmad3.FinishFullPreload();
        }
        uint32_t qLimits[WIDE_MAX_Q_TASKS] = {};
        uint64_t maskRowOffsets[WIDE_MAX_Q_TASKS] = {};
        uint32_t maxKvLimit = 0;
        const uint32_t kvBlockCount = (current.kvSeqlen + tilingData->blockShapeY - 1) / tilingData->blockShapeY;
        for (uint32_t qLocal = 0; qLocal < qCount; ++qLocal) {
            const TaskInfo qTask = MakeWideQTask(current, qInfos, qLocal);
            qLimits[qLocal] = GetKvBlockLimit(params, qTask, kvBlockCount, tilingData);
            maskRowOffsets[qLocal] = GetMaskRowOffset(qTask, maskLayout);
            if (qLimits[qLocal] > maxKvLimit) {
                maxKvLimit = qLimits[qLocal];
            }
        }

        WideMaskScanState maskScanState;
        ResetWideMaskScanState(maskScanState);
        WidePacketInfo packets[2];
        uint32_t stage = 0;
        uint32_t previousStage = 0;
        bool havePrevious = false;
        WideQMask qSeenMask = 0;
        uint32_t kvStart = 0;
        uint32_t qStart = 0;
        while (kvStart < maxKvLimit) {
            WidePacketInfo &packet = packets[stage];
            const uint64_t workspaceBase = static_cast<uint64_t>(coreIdx) * tilingData->wideWorkspaceCoreElements +
                                           static_cast<uint64_t>(stage) * tilingData->wideWorkspaceStageElements;
            BuildWidePacket(params, current, tilingData, qLimits, maskRowOffsets, maskLayout, qCount, maxKvLimit,
                            kvStart, qStart, workspaceBase, maskScanState, packet);
            kvStart = packet.nextKvStart;
            qStart = packet.nextQStart;
            if (packet.edgeCount == 0) {
                continue;
            }
            ++dbgPackets;

            // Gradient P/dS reads and the following Cube1 operand
            // loads share PIPE_MTE2 and retire in program order.  The
            // first Cube1 MTE2->MTE1 completion therefore also proves
            // that the old low-precision stage has been consumed
            // before raw-ready lets AIV overwrite it.  A separate
            // whole-pipeline MTE2->FIX fence would only serialize the
            // independent gradient-read and raw-write GM planes.
            SyncExtendedL1Reuse();
            RunWideCube1(packet, current, qInfos, tilingData, gQ, gK, gV, gDout, gS, gDp, blockMmad1, mmapFlag,
                         blockMmad3);
            const uint32_t cubeToVec = stage == 0 ? WIDE_CUBE2VEC_STAGE0 : WIDE_CUBE2VEC_STAGE1;
            AscendC::CrossCoreSetFlag<2, PIPE_FIX>(cubeToVec);

            if (havePrevious) {
                const uint32_t vecToCube = previousStage == 0 ? WIDE_VEC2CUBE_STAGE0 : WIDE_VEC2CUBE_STAGE1;
                AscendC::WaitEvent(vecToCube);
                RunWideGrad(packets[previousStage], current, qInfos, qCount, tilingData, gK, gQ, gDout, gP, gDs, gDq,
                            gDk, gDv, blockMmad2, blockMmad3, qSeenMask);
            }
            previousStage = stage;
            stage = 1 - stage;
            havePrevious = true;
        }
        if (havePrevious) {
            const uint32_t vecToCube = previousStage == 0 ? WIDE_VEC2CUBE_STAGE0 : WIDE_VEC2CUBE_STAGE1;
            AscendC::WaitEvent(vecToCube);
            RunWideGrad(packets[previousStage], current, qInfos, qCount, tilingData, gK, gQ, gDout, gP, gDs, gDq, gDk,
                        gDv, blockMmad2, blockMmad3, qSeenMask);
            // Publish a real reuse token for the final high-L1 reads.
            SyncExtendedL1Reuse();
        }
        if (tilingData->wideQOperandCache != 0) {
            blockMmad3.ReleaseFullPreloadReuse();
        }
        ++processedWorks;
        if (processedWorks < workLength) {
            current = nextWork;
        }
    }
    WaitWideFlag();
    AscendC::CrossCoreSetFlag<2, PIPE_FIX>(CUBE2POST);
}

__aicore__ inline void FillWideSfmParams(SfmParams &tileParams, const Params &params, const TaskInfo &tile,
                                         const __gm__ BlockSparseAttentionGradTilingData *tilingData, GM_ADDR sftmgGm,
                                         uint32_t subBlockIdx, uint32_t metadataSlot) const
{
    const uint64_t firstHalfRows = (tile.curCalQSize + 1) / 2;
    const uint64_t executeRow = subBlockIdx == 0 ? firstHalfRows : tile.curCalQSize - firstHalfRows;
    const uint64_t curVecCoreS1Idx = tile.curQSeqIdx + subBlockIdx * firstHalfRows;
    // Raw S/dP are standard FP32 NZ tiles.  Keep the tile base here;
    // the vector epilogue applies rawRowOffset * C0 internally.
    const uint64_t rawByteOffset = tile.sOffset * sizeof(float);
    // P/dS remain in native zN, so both vector subcores address the
    // tile base and scatter their own row range using rawRowOffset.
    const uint64_t lowpByteOffset = tilingData->wideRawPlaneBytes + tile.sOffset * sizeof(ElementInput);
    tileParams = SfmParams(
        params.workspace + rawByteOffset, params.softmaxLse, params.workspace + tilingData->sOutSize + rawByteOffset,
        params.actualQseqlen, params.actualKvseqlen, sftmgGm, params.workspace + lowpByteOffset,
        params.workspace + tilingData->sOutSize + lowpByteOffset, params.tiling, executeRow, tile.curCalKVSize,
        executeRow * tile.curCalKVSize, tile.curBatchIdx, tile.curHeadIdx, curVecCoreS1Idx, metadataSlot,
        tile.kvValidSize, tile.curCalQSize, subBlockIdx * firstHalfRows, true);
}

__aicore__ inline void ProcessWideAiv(const Params &params, __gm__ BlockSparseAttentionGradTilingData *tilingData)
{
    const uint32_t vecCoreIdx = AscendC::GetBlockIdx();
    const uint32_t coreIdx = vecCoreIdx / 2;
    const uint32_t subBlockIdx = vecCoreIdx % 2;
    const uint32_t groupSize = tilingData->numHeads / tilingData->kvHeads;
    const uint32_t workLength = tilingData->taskNumPerCore + (tilingData->tailTaskNum > coreIdx ? 1 : 0);

    AscendC::GlobalTensor<int64_t> gActualQseqlen;
    AscendC::GlobalTensor<int64_t> gActualKvseqlen;
    gActualQseqlen.SetGlobalBuffer((__gm__ int64_t *)params.actualQseqlen);
    gActualKvseqlen.SetGlobalBuffer((__gm__ int64_t *)params.actualKvseqlen);
    TaskInfo current;
    initTaskInfo(gActualQseqlen, gActualKvseqlen, tilingData, tilingData->numHeads, tilingData->kvHeads, groupSize,
                 tilingData->headDim, tilingData->maxQSeqlen, tilingData->maxKvSeqlen, tilingData->blockShapeX,
                 tilingData->basicQBlockSize, tilingData->inputLayout, coreIdx, current);

    GM_ADDR sftmgGm = params.workspace + tilingData->sOutSize + tilingData->dPOutSize + tilingData->dQOutSize +
                      tilingData->dKOutSize + tilingData->dVOutSize;
    SfmgParams sfmgParams(params.dout, params.out, params.actualQseqlen, sftmgGm, params.tiling);
    EpilogueFAGSfmg vecSftmg(sfmgParams);
    EpilogueFAGOp softmaxOp;
    softmaxOp.BeginGroupedPipeline(params.softmaxLse, params.actualQseqlen, params.actualKvseqlen, sftmgGm,
                                   params.tiling);
    const MaskLayoutInfo maskLayout = GetMaskLayout(tilingData);
    uint32_t processedWorks = 0;
    uint32_t dbgPackets = 0;

    while (processedWorks < workLength) {
        WideQInfo qInfos[WIDE_MAX_Q_TASKS];
        TaskInfo nextWork;
        const uint32_t qCount = BuildWideWorkItem(gActualQseqlen, gActualKvseqlen, tilingData, groupSize, current,
                                                  qInfos, nextWork, processedWorks + 1 < workLength);

        // D = rowsum(dO * O) is Q-only.  Produce it once per Q task,
        // then order all following metadata MTE2 reads after the GM
        // writes from both paired AIVs.
        for (uint32_t qLocal = 0; qLocal < qCount; ++qLocal) {
            const uint32_t firstHalfRows = (qInfos[qLocal].curCalQSize + 1) / 2;
            const uint32_t subRows = subBlockIdx == 0 ? firstHalfRows : qInfos[qLocal].curCalQSize - firstHalfRows;
            if (subRows != 0) {
                vecSftmg(qInfos[qLocal].qOffset / tilingData->headDim + subBlockIdx * firstHalfRows, subRows);
            }
        }
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);

        uint32_t qLimits[WIDE_MAX_Q_TASKS] = {};
        uint64_t maskRowOffsets[WIDE_MAX_Q_TASKS] = {};
        uint32_t maxKvLimit = 0;
        const uint32_t kvBlockCount = (current.kvSeqlen + tilingData->blockShapeY - 1) / tilingData->blockShapeY;
        for (uint32_t qLocal = 0; qLocal < qCount; ++qLocal) {
            const TaskInfo qTask = MakeWideQTask(current, qInfos, qLocal);
            qLimits[qLocal] = GetKvBlockLimit(params, qTask, kvBlockCount, tilingData);
            maskRowOffsets[qLocal] = GetMaskRowOffset(qTask, maskLayout);
            if (qLimits[qLocal] > maxKvLimit) {
                maxKvLimit = qLimits[qLocal];
            }
        }

        WideMaskScanState maskScanState;
        ResetWideMaskScanState(maskScanState);
        WidePacketInfo packet;
        uint32_t stage = 0;
        uint32_t kvStart = 0;
        uint32_t qStart = 0;
        softmaxOp.BeginWideRawStageReuse();
        while (kvStart < maxKvLimit) {
            const uint64_t workspaceBase = static_cast<uint64_t>(coreIdx) * tilingData->wideWorkspaceCoreElements +
                                           static_cast<uint64_t>(stage) * tilingData->wideWorkspaceStageElements;
            BuildWidePacket(params, current, tilingData, qLimits, maskRowOffsets, maskLayout, qCount, maxKvLimit,
                            kvStart, qStart, workspaceBase, maskScanState, packet);
            kvStart = packet.nextKvStart;
            qStart = packet.nextQStart;
            if (packet.edgeCount == 0) {
                continue;
            }
            ++dbgPackets;
            const uint32_t cubeToVec = stage == 0 ? WIDE_CUBE2VEC_STAGE0 : WIDE_CUBE2VEC_STAGE1;
            AscendC::WaitEvent(cubeToVec);

            // One cross-core token covers the complete packet. Flatten Q-major
            // runs into one two-stage pipeline to overlap prefetch across Q boundaries.
            uint16_t validEdges[WIDE_MAX_EDGES];
            uint8_t validMetadataSlots[WIDE_MAX_EDGES];
            uint32_t validEdgeCount = 0;
            uint32_t activeQCount = 0;
            for (uint32_t activeQ = 0; activeQ < packet.activeQCount; ++activeQ) {
                const WideQDesc &qDesc = packet.activeQ[activeQ];
                const uint32_t qLocal = qDesc.qLocal;
                const uint32_t qBegin = qDesc.edgeBegin;
                const uint32_t qEnd = qBegin + qDesc.edgeCount;
                // A one-row Q tail is owned entirely by AIV0.  AIV1
                // still publishes the packet-level cross-core token,
                // but must not issue a zero-block DMA.
                const uint32_t firstHalfRows = (qInfos[qLocal].curCalQSize + 1) / 2;
                const uint32_t executeRows =
                    subBlockIdx == 0 ? firstHalfRows : qInfos[qLocal].curCalQSize - firstHalfRows;
                if (executeRows == 0) {
                    continue;
                }
                const uint8_t metadataSlot = static_cast<uint8_t>(activeQCount & 3U);
                ++activeQCount;
                for (uint32_t qEdge = qBegin; qEdge < qEnd; ++qEdge) {
                    validEdges[validEdgeCount] = packet.qOrder[qEdge];
                    validMetadataSlots[validEdgeCount] = metadataSlot;
                    ++validEdgeCount;
                }
            }

            if (validEdgeCount != 0) {
                // Rotate four metadata slots by active Q, independently
                // of the two raw input stages.  All edges of one Q can
                // then reuse one LSE/D copy.  Before a slot wraps, at
                // least three intervening Q runs have already forced
                // both raw stages through their V->MTE2 reuse fences,
                // so every previous vector consumer has retired.
                softmaxOp.ResetGroupedQMetadata();
                SfmParams tileParams[2];
                uint32_t inputStage = 0;
                const uint32_t firstEdge = validEdges[0];
                const uint32_t firstMetadataSlot = validMetadataSlots[0];
                TaskInfo firstTile = MakeWideTile(current, qInfos, packet, firstEdge, tilingData);
                FillWideSfmParams(tileParams[0], params, firstTile, tilingData, sftmgGm, subBlockIdx,
                                  firstMetadataSlot);
                softmaxOp.InvalidateGroupedQMetadataSlot(firstMetadataSlot);
                softmaxOp.PrefetchGrouped(tileParams[0], 0);

                for (uint32_t edgeIdx = 0; edgeIdx < validEdgeCount; ++edgeIdx) {
                    if (edgeIdx + 1 < validEdgeCount) {
                        const uint32_t nextStage = 1 - inputStage;
                        const uint32_t nextEdge = validEdges[edgeIdx + 1];
                        const uint32_t currentEdge = validEdges[edgeIdx];
                        const uint32_t nextQ = WideEdgeQ(packet.edgeCode[nextEdge]);
                        const uint32_t currentQ = WideEdgeQ(packet.edgeCode[currentEdge]);
                        const uint32_t nextMetadataSlot = validMetadataSlots[edgeIdx + 1];
                        TaskInfo nextTile = MakeWideTile(current, qInfos, packet, nextEdge, tilingData);
                        FillWideSfmParams(tileParams[nextStage], params, nextTile, tilingData, sftmgGm, subBlockIdx,
                                          nextMetadataSlot);
                        if (nextQ != currentQ) {
                            softmaxOp.InvalidateGroupedQMetadataSlot(nextMetadataSlot);
                        }
                        softmaxOp.PrefetchGrouped(tileParams[nextStage], nextStage);
                    }
                    softmaxOp.ComputeGrouped(tileParams[inputStage], inputStage);
                    inputStage = 1 - inputStage;
                }
            }
            const uint32_t vecToCube = stage == 0 ? WIDE_VEC2CUBE_STAGE0 : WIDE_VEC2CUBE_STAGE1;
            AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(vecToCube);
            stage = 1 - stage;
        }
        softmaxOp.EndWideRawStageReuse();
        ++processedWorks;
        if (processedWorks < workLength) {
            current = nextWork;
        }
    }
    softmaxOp.EndGroupedPipeline();
}

#endif // BSAG_KERNEL_WIDE_FRAGMENT
