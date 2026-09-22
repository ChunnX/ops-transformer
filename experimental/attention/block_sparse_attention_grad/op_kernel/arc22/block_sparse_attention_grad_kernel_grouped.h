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
 * \file block_sparse_attention_grad_kernel_grouped.h
 * \brief Block Sparse Attention Grad Kernel Grouped
 */

#ifndef BSAG_KERNEL_GROUPED_FRAGMENT
#define BSAG_KERNEL_GROUPED_FRAGMENT

// The grouped pipeline traverses all Q blocks of one batch/head
// before advancing the head. BNSD is contiguous in that order;
// TND uses the same logical order with an N*D physical stride.
__aicore__ inline void updateNextGroupedTaskInfo(AscendC::GlobalTensor<int64_t> gActualQseqlen,
                                                 AscendC::GlobalTensor<int64_t> gActualKvseqlen, uint32_t numHeads,
                                                 uint32_t kvHeads, uint32_t groupSize, uint32_t headDim,
                                                 uint32_t blockShapeX, uint32_t basicQBlockSize, uint32_t inputLayout,
                                                 const TaskInfo &taskInfo, TaskInfo &nextTask)
{
    if (inputLayout == 1) {
        updateNextTaskInfo(gActualQseqlen, gActualKvseqlen, numHeads, kvHeads, groupSize, headDim, blockShapeX,
                           basicQBlockSize, inputLayout, taskInfo, nextTask);
        return;
    }

    nextTask = taskInfo;
    const uint64_t qStride = static_cast<uint64_t>(numHeads) * headDim;
    const uint64_t kvStride = static_cast<uint64_t>(kvHeads) * headDim;
    nextTask.qOffset = taskInfo.qOffset + taskInfo.curCalQSize * qStride;
    if (taskInfo.curQSeqIdx + taskInfo.curCalQSize == taskInfo.qSeqlen) {
        nextTask.curQSeqIdx = 0;
        if (taskInfo.curHeadIdx == numHeads - 1) {
            nextTask.curBatchIdx = taskInfo.curBatchIdx + 1;
            nextTask.curHeadIdx = 0;
            nextTask.qOffset -= static_cast<uint64_t>(taskInfo.curHeadIdx) * headDim;
            nextTask.kvOffset +=
                static_cast<uint64_t>(taskInfo.kvSeqlen) * kvStride - static_cast<uint64_t>(kvHeads - 1) * headDim;
            nextTask.qSeqlen =
                static_cast<uint32_t>(static_cast<int64_t>(gActualQseqlen.GetValue(nextTask.curBatchIdx)));
            nextTask.kvSeqlen =
                static_cast<uint32_t>(static_cast<int64_t>(gActualKvseqlen.GetValue(nextTask.curBatchIdx)));
        } else {
            nextTask.curHeadIdx = taskInfo.curHeadIdx + 1;
            nextTask.qOffset = taskInfo.qOffset - static_cast<uint64_t>(taskInfo.curQSeqIdx) * qStride + headDim;
            if (nextTask.curHeadIdx % groupSize == 0) {
                nextTask.kvOffset = taskInfo.kvOffset + headDim;
            }
        }
    } else {
        nextTask.curQSeqIdx = taskInfo.curQSeqIdx + taskInfo.curCalQSize;
    }
    UpdateTaskInfoCalQSize(blockShapeX, basicQBlockSize, nextTask);
}

__aicore__ inline uint32_t GetPackedFragmentCapacity(const __gm__ BlockSparseAttentionGradTilingData *tilingData) const
{
    // Bound one Q's dK/dV FP32 Fixpipe burst to roughly 64 KiB.  The
    // result is rounded to fragment pairs because two sub-128 KV
    // fragments share one 128-token Cube tile.  This scales with the
    // actual fragment/output shape and does not specialize a block.
    constexpr uint32_t FIXPIPE_BURST_BUDGET = 64 * 1024;
    const uint64_t fragmentBytes =
        static_cast<uint64_t>(tilingData->basicKVBlockSize) * tilingData->headDim * sizeof(float);
    uint32_t capacity = fragmentBytes == 0 ? 2 : static_cast<uint32_t>(FIXPIPE_BURST_BUDGET / fragmentBytes);
    if (capacity < 2) {
        capacity = 2;
    }
    if (capacity > GROUPED_PACKED_FRAGMENTS_PER_Q) {
        capacity = GROUPED_PACKED_FRAGMENTS_PER_Q;
    }
    return capacity & ~1U;
}

__aicore__ inline void ClearGroupedPadding(const Params &params,
                                           const __gm__ BlockSparseAttentionGradTilingData *tilingData) const
{
    if (tilingData->basicKVBlockSize >= 128 || tilingData->blockShapeX < 16) {
        return;
    }
    const uint32_t vecCoreIdx = AscendC::GetBlockIdx();
    const uint32_t coreIdx = vecCoreIdx / 2;
    const uint32_t stage = vecCoreIdx % 2;
    const uint64_t groupedFp32Bytes =
        static_cast<uint64_t>(tilingData->usedVecCoreNum / 2) * GROUPED_WORKSPACE_CORE_SIZE * sizeof(float);
    const uint64_t paddingOffset = static_cast<uint64_t>(coreIdx) * GROUPED_WORKSPACE_CORE_SIZE +
                                   static_cast<uint64_t>(stage) * GROUPED_WORKSPACE_STAGE_SIZE +
                                   static_cast<uint64_t>(GROUPED_TILES - 1) * GROUPED_TILE_ELEMENTS;
    AscendC::GlobalTensor<ElementInput> gPaddingP;
    AscendC::GlobalTensor<ElementInput> gPaddingDs;
    gPaddingP.SetGlobalBuffer((__gm__ ElementInput *)(params.workspace + groupedFp32Bytes));
    gPaddingDs.SetGlobalBuffer((__gm__ ElementInput *)(params.workspace + tilingData->sOutSize + groupedFp32Bytes));
    AscendC::InitOutput<ElementInput>(gPaddingP[paddingOffset], GROUPED_TILE_ELEMENTS, 0);
    AscendC::InitOutput<ElementInput>(gPaddingDs[paddingOffset], GROUPED_TILE_ELEMENTS, 0);
}

__aicore__ inline void BuildGroupedSegment(const Params &params,
                                           const __gm__ BlockSparseAttentionGradTilingData *tilingData,
                                           const TaskInfo *qInfos, const uint32_t *qLimits, uint32_t qCount,
                                           uint32_t kvStart, uint32_t kvCount, uint32_t *packedKvStarts,
                                           const uint64_t *maskRowOffsets, uint64_t workspaceBase,
                                           SegmentInfo &segment) const
{
    segment.tileCount = 0;
    segment.uniqueKvCount = 0;
    segment.nextKvStart = kvStart + kvCount;

    // When a sparse KV block occupies less than one 128-token Cube
    // tile, collect sparse fragments independently for every Q tile
    // and concatenate as many as fit in the hardware tile.  This is
    // capacity based: it applies to any supported fragment size and
    // never keys a path on one user block_shape value.  Keeping the
    // two physical offsets in TaskInfo lets Cube consume one dense
    // compute tile while dK/dV are scattered back to their sources.
    if (qCount >= 1 && tilingData->basicKVBlockSize < 128 && tilingData->blockShapeX >= 16) {
        uint32_t selected[GROUPED_Q_BLOCKS][GROUPED_PACKED_FRAGMENTS_PER_Q] = {};
        uint32_t selectedCount[GROUPED_Q_BLOCKS] = {};
        bool firstSegment[GROUPED_Q_BLOCKS] = {};
        const uint32_t fragmentCapacity = GetPackedFragmentCapacity(tilingData);
        for (uint32_t qLocal = 0; qLocal < qCount; ++qLocal) {
            uint32_t kvBasicIdx = packedKvStarts[qLocal];
            firstSegment[qLocal] = kvBasicIdx == 0;
            CollectEnabledMaskBlocks(params, maskRowOffsets[qLocal], qLimits[qLocal], fragmentCapacity, kvBasicIdx,
                                     selected[qLocal], selectedCount[qLocal]);
            packedKvStarts[qLocal] = kvBasicIdx;
        }
        const uint64_t kvStride = tilingData->inputLayout == 0 ?
                                      static_cast<uint64_t>(tilingData->kvHeads) * tilingData->headDim :
                                      tilingData->headDim;
        for (uint32_t qLocal = 0; qLocal < qCount; ++qLocal) {
            const TaskInfo &qInfo = qInfos[qLocal];
            const uint32_t kvTileCount =
                (qInfo.kvSeqlen + tilingData->basicKVBlockSize - 1) / tilingData->basicKVBlockSize;
            uint32_t fragment = 0;
            uint32_t packedLocal = 0;
            while (fragment < selectedCount[qLocal]) {
                const uint32_t firstKv = selected[qLocal][fragment++];
                const uint32_t firstSize = firstKv + 1 < kvTileCount ?
                                               tilingData->basicKVBlockSize :
                                               qInfo.kvSeqlen - firstKv * tilingData->basicKVBlockSize;
                TaskInfo &tile = segment.tiles[segment.tileCount];
                tile = qInfo;
                tile.kvBasicIdx = firstKv;
                tile.kvBasicIdxSecond = 0;
                tile.kvFirstSize = firstSize;
                tile.kvSecondSize = 0;
                tile.kvPartCount = 1;
                tile.kvOffset =
                    qInfo.kvOffset + static_cast<uint64_t>(firstKv) * tilingData->basicKVBlockSize * kvStride;
                tile.kvSecondOffset = 0;
                if (fragment < selectedCount[qLocal]) {
                    const uint32_t secondKv = selected[qLocal][fragment];
                    const uint32_t secondSize = secondKv + 1 < kvTileCount ?
                                                    tilingData->basicKVBlockSize :
                                                    qInfo.kvSeqlen - secondKv * tilingData->basicKVBlockSize;
                    if (firstSize + secondSize <= 128) {
                        ++fragment;
                        tile.kvBasicIdxSecond = secondKv;
                        tile.kvSecondSize = secondSize;
                        tile.kvPartCount = 2;
                        tile.kvSecondOffset =
                            qInfo.kvOffset + static_cast<uint64_t>(secondKv) * tilingData->basicKVBlockSize * kvStride;
                    }
                }
                tile.kvValidSize = tile.kvFirstSize + tile.kvSecondSize;
                tile.curCalKVSize = 128;
                tile.sOffset = workspaceBase + static_cast<uint64_t>(segment.tileCount) * GROUPED_TILE_ELEMENTS;
                tile.groupQIdx = qLocal;
                tile.groupKvIdx = packedLocal++;
                tile.firstQSegment = firstSegment[qLocal];
                ++segment.tileCount;
            }
            if (packedLocal > segment.uniqueKvCount) {
                segment.uniqueKvCount = packedLocal;
            }
        }
        return;
    }

    // Preserve a four-Q group while filling the packet by active-edge
    // capacity.  A KV column is indivisible: either all enabled Q
    // edges in that column enter this packet, or the whole column is
    // deferred to the next packet.  This keeps dK/dV at one reduction
    // and one Fixpipe flush per physical KV column while allowing a
    // sparse packet to contain as many as GROUPED_TILES unique KVs.
    // The mask remains the original Q-major uint8 tensor.
    if (qCount >= 1 && tilingData->basicKVBlockSize == 128) {
        uint32_t kvBasicIdx = kvStart;
        uint32_t selectedKvCount = 0;
        uint32_t selectedEdgeCount = 0;
        uint32_t selectedKvIndices[GROUPED_PACKET_KV_BLOCKS] = {};
        bool selectedEnabled[GROUPED_PACKET_KV_BLOCKS][GROUPED_Q_BLOCKS] = {};
        const uint64_t kvStride = tilingData->inputLayout == 0 ?
                                      static_cast<uint64_t>(tilingData->kvHeads) * tilingData->headDim :
                                      tilingData->headDim;
        uint32_t maxKvLimit = 0;
        for (uint32_t qLocal = 0; qLocal < qCount; ++qLocal) {
            if (qLimits[qLocal] > maxKvLimit) {
                maxKvLimit = qLimits[qLocal];
            }
        }
        while (kvBasicIdx < maxKvLimit && selectedKvCount < GROUPED_KV_BLOCKS && selectedEdgeCount < GROUPED_TILES) {
            bool columnEnabled[GROUPED_Q_BLOCKS] = {};
            uint32_t columnEdgeCount = 0;
            for (uint32_t qLocal = 0; qLocal < qCount; ++qLocal) {
                if (kvBasicIdx >= qLimits[qLocal]) {
                    continue;
                }
                const bool enabled = ComputeBlockEnabled(params, maskRowOffsets[qLocal] + kvBasicIdx);
                columnEnabled[qLocal] = enabled;
                columnEdgeCount += enabled ? 1U : 0U;
            }
            if (columnEdgeCount == 0) {
                ++kvBasicIdx;
                continue;
            }
            if (selectedEdgeCount + columnEdgeCount > GROUPED_TILES) {
                break;
            }
            selectedKvIndices[selectedKvCount] = kvBasicIdx;
            for (uint32_t qLocal = 0; qLocal < qCount; ++qLocal) {
                selectedEnabled[selectedKvCount][qLocal] = columnEnabled[qLocal];
            }
            ++selectedKvCount;
            selectedEdgeCount += columnEdgeCount;
            ++kvBasicIdx;
        }
        segment.nextKvStart = kvBasicIdx;
        segment.uniqueKvCount = selectedKvCount;
        // The AIV completion protocol publishes one event per Q run,
        // so materialize selected columns in Q-major order.
        for (uint32_t qLocal = 0; qLocal < qCount; ++qLocal) {
            const TaskInfo &qInfo = qInfos[qLocal];
            for (uint32_t kvLocal = 0; kvLocal < selectedKvCount; ++kvLocal) {
                const uint32_t selectedKv = selectedKvIndices[kvLocal];
                if (!selectedEnabled[kvLocal][qLocal]) {
                    continue;
                }
                const uint32_t kvTileCount =
                    (qInfo.kvSeqlen + tilingData->basicKVBlockSize - 1) / tilingData->basicKVBlockSize;
                TaskInfo &tile = segment.tiles[segment.tileCount];
                tile = qInfo;
                tile.curCalKVSize = selectedKv + 1 < kvTileCount ?
                                        tilingData->basicKVBlockSize :
                                        qInfo.kvSeqlen - selectedKv * tilingData->basicKVBlockSize;
                tile.kvOffset =
                    qInfo.kvOffset + static_cast<uint64_t>(selectedKv) * tilingData->basicKVBlockSize * kvStride;
                tile.sOffset = workspaceBase + static_cast<uint64_t>(segment.tileCount) * GROUPED_TILE_ELEMENTS;
                tile.groupQIdx = qLocal;
                tile.groupKvIdx = kvLocal;
                tile.kvBasicIdx = selectedKv;
                tile.kvBasicIdxSecond = 0;
                tile.kvFirstSize = tile.curCalKVSize;
                tile.kvSecondSize = 0;
                tile.kvSecondOffset = 0;
                tile.kvPartCount = 1;
                tile.kvValidSize = tile.curCalKVSize;
                tile.firstQSegment = kvStart == 0;
                ++segment.tileCount;
            }
        }
        return;
    }

    // Use the four-column schedule for sub-128 basic KV
    // paths that do not use fragment packing.
    if (qCount >= 1) {
        uint32_t kvBasicIdx = kvStart;
        uint32_t selectedKvCount = 0;
        uint32_t selectedKvIndices[GROUPED_KV_BLOCKS] = {};
        bool selectedEnabled[GROUPED_KV_BLOCKS][GROUPED_Q_BLOCKS] = {};
        const uint64_t kvStride = tilingData->inputLayout == 0 ?
                                      static_cast<uint64_t>(tilingData->kvHeads) * tilingData->headDim :
                                      tilingData->headDim;
        while (kvBasicIdx < qLimits[0] && selectedKvCount < GROUPED_KV_BLOCKS) {
            bool anyEnabled = false;
            for (uint32_t qLocal = 0; qLocal < qCount; ++qLocal) {
                if (kvBasicIdx >= qLimits[qLocal]) {
                    continue;
                }
                const bool enabled = ComputeBlockEnabled(params, maskRowOffsets[qLocal] + kvBasicIdx);
                selectedEnabled[selectedKvCount][qLocal] = enabled;
                anyEnabled |= enabled;
            }
            if (anyEnabled) {
                selectedKvIndices[selectedKvCount] = kvBasicIdx;
                ++selectedKvCount;
            }
            ++kvBasicIdx;
        }
        segment.nextKvStart = kvBasicIdx;
        segment.uniqueKvCount = selectedKvCount;
        for (uint32_t qLocal = 0; qLocal < qCount; ++qLocal) {
            const TaskInfo &qInfo = qInfos[qLocal];
            for (uint32_t kvLocal = 0; kvLocal < selectedKvCount; ++kvLocal) {
                const uint32_t selectedKv = selectedKvIndices[kvLocal];
                if (!selectedEnabled[kvLocal][qLocal]) {
                    continue;
                }
                const uint32_t kvTileCount =
                    (qInfo.kvSeqlen + tilingData->basicKVBlockSize - 1) / tilingData->basicKVBlockSize;
                TaskInfo &tile = segment.tiles[segment.tileCount];
                tile = qInfo;
                tile.curCalKVSize = selectedKv + 1 < kvTileCount ?
                                        tilingData->basicKVBlockSize :
                                        qInfo.kvSeqlen - selectedKv * tilingData->basicKVBlockSize;
                tile.kvOffset =
                    qInfo.kvOffset + static_cast<uint64_t>(selectedKv) * tilingData->basicKVBlockSize * kvStride;
                tile.sOffset = workspaceBase + static_cast<uint64_t>(segment.tileCount) * GROUPED_TILE_ELEMENTS;
                tile.groupQIdx = qLocal;
                tile.groupKvIdx = kvLocal;
                tile.kvBasicIdx = selectedKv;
                tile.kvBasicIdxSecond = 0;
                tile.kvFirstSize = tile.curCalKVSize;
                tile.kvSecondSize = 0;
                tile.kvSecondOffset = 0;
                tile.kvPartCount = 1;
                tile.kvValidSize = tile.curCalKVSize;
                tile.firstQSegment = kvStart == 0;
                ++segment.tileCount;
            }
        }
        return;
    }
}

__aicore__ inline void RunGroupedCube1CachedB(
    SegmentInfo &segment, const __gm__ BlockSparseAttentionGradTilingData *tilingData,
    AscendC::GlobalTensor<ElementInput> gQ, AscendC::GlobalTensor<ElementInput> gK,
    AscendC::GlobalTensor<ElementInput> gV, AscendC::GlobalTensor<ElementA1> gDout, AscendC::GlobalTensor<float> gS,
    AscendC::GlobalTensor<float> gDp, BlockMmadBSAG1 &blockMmad1, uint32_t &mmadFlag)
{
    const uint32_t headDim = tilingData->headDim;
    const uint64_t qStride =
        (tilingData->inputLayout == 0 ? static_cast<uint64_t>(tilingData->numHeads) * headDim : headDim);
    const uint64_t kvStride =
        (tilingData->inputLayout == 0 ? static_cast<uint64_t>(tilingData->kvHeads) * headDim : headDim);
    for (uint32_t kvLocal = 0; kvLocal < segment.uniqueKvCount; ++kvLocal) {
        int32_t first = -1;
        for (uint32_t t = 0; t < segment.tileCount; ++t) {
            if (segment.tiles[t].groupKvIdx == kvLocal) {
                first = static_cast<int32_t>(t);
                break;
            }
        }
        if (first < 0) {
            continue;
        }
        TaskInfo &firstTile = segment.tiles[first];
        LayoutB1 layoutB(headDim, firstTile.curCalKVSize, kvStride);
        GemmCoord preloadShape{firstTile.curCalQSize, firstTile.curCalKVSize, headDim};
        const uint64_t packedKvOffset = GroupedKvHeadOffset(firstTile, tilingData);
        blockMmad1.PreloadB(gK[packedKvOffset], layoutB, preloadShape, 0);
        blockMmad1.PreloadB(gV[packedKvOffset], layoutB, preloadShape, 1);
        for (uint32_t t = 0; t < segment.tileCount; ++t) {
            TaskInfo &tile = segment.tiles[t];
            if (tile.groupKvIdx != kvLocal) {
                continue;
            }
            LayoutA1 layoutA(tile.curCalQSize, headDim, qStride);
            LayoutC1 layoutC(tile.curCalQSize, tile.curCalKVSize);
            GemmCoord shape{tile.curCalQSize, tile.curCalKVSize, headDim};
            const uint64_t packedQOffset = GroupedQHeadOffset(tile, tilingData);
            blockMmad1.WithCachedB(0, gQ[packedQOffset], gS[tile.sOffset], layoutA, layoutB, layoutC, shape, mmadFlag);
            blockMmad1.WithCachedB(1, gDout[packedQOffset], gDp[tile.sOffset], layoutA, layoutB, layoutC, shape,
                                   mmadFlag);
        }
    }
}

__aicore__ inline void RunGroupedCube1Packed(
    SegmentInfo &segment, const __gm__ BlockSparseAttentionGradTilingData *tilingData,
    AscendC::GlobalTensor<ElementInput> gQ, AscendC::GlobalTensor<ElementInput> gK,
    AscendC::GlobalTensor<ElementInput> gV, AscendC::GlobalTensor<ElementA1> gDout, AscendC::GlobalTensor<float> gS,
    AscendC::GlobalTensor<float> gDp, AscendC::GlobalTensor<ElementInput> gZero, BlockMmadBSAG1 &blockMmad1,
    uint32_t &mmadFlag)
{
    const uint32_t headDim = tilingData->headDim;
    const uint64_t qStride =
        tilingData->inputLayout == 0 ? static_cast<uint64_t>(tilingData->numHeads) * headDim : headDim;
    const uint64_t kvStride =
        tilingData->inputLayout == 0 ? static_cast<uint64_t>(tilingData->kvHeads) * headDim : headDim;
    const uint64_t zeroOffset =
        segment.tiles[0].sOffset + static_cast<uint64_t>(GROUPED_TILES - 1) * GROUPED_TILE_ELEMENTS;
    for (uint32_t t = 0; t < segment.tileCount; ++t) {
        TaskInfo &tile = segment.tiles[t];
        LayoutB1 firstLayout(headDim, tile.kvFirstSize, kvStride);
        LayoutB1 secondLayout(headDim, tile.kvSecondSize, kvStride);
        LayoutB1 paddingLayout(headDim, tile.curCalKVSize - tile.kvValidSize, headDim);
        GemmCoord shape{tile.curCalQSize, tile.curCalKVSize, headDim};
        blockMmad1.PreloadGatheredBAlongN(gK[tile.kvOffset], firstLayout, gK[tile.kvSecondOffset], secondLayout,
                                          gZero[zeroOffset], paddingLayout, tile.kvFirstSize, tile.kvSecondSize, shape,
                                          0);
        blockMmad1.PreloadGatheredBAlongN(gV[tile.kvOffset], firstLayout, gV[tile.kvSecondOffset], secondLayout,
                                          gZero[zeroOffset], paddingLayout, tile.kvFirstSize, tile.kvSecondSize, shape,
                                          1);
        LayoutA1 layoutA(tile.curCalQSize, headDim, qStride);
        LayoutB1 combinedLayout(headDim, tile.curCalKVSize);
        LayoutC1 layoutC(tile.curCalQSize, tile.curCalKVSize);
        const uint64_t qOffset = GroupedQHeadOffset(tile, tilingData);
        blockMmad1.WithCachedB(0, gQ[qOffset], gS[tile.sOffset], layoutA, combinedLayout, layoutC, shape, mmadFlag);
        blockMmad1.WithCachedB(1, gDout[qOffset], gDp[tile.sOffset], layoutA, combinedLayout, layoutC, shape, mmadFlag);
    }
}

__aicore__ inline void RunGroupedPackedGrad(
    SegmentInfo &segment, const __gm__ BlockSparseAttentionGradTilingData *tilingData,
    AscendC::GlobalTensor<ElementInput> gK, AscendC::GlobalTensor<ElementInput> gQ,
    AscendC::GlobalTensor<ElementA1> gDout, AscendC::GlobalTensor<ElementInput> gP,
    AscendC::GlobalTensor<ElementInput> gDs, AscendC::GlobalTensor<float> gDq, AscendC::GlobalTensor<float> gDk,
    AscendC::GlobalTensor<float> gDv, BlockMmadBSAG2 &blockMmad2, BlockMmadBSAG3 &blockMmad3, uint32_t &mmadFlag,
    uint64_t *cachedQOffsets, uint32_t &cachedQMask, bool finalSegment)
{
    const uint32_t headDim = tilingData->headDim;
    const uint64_t kvInputStride =
        tilingData->inputLayout == 0 ? static_cast<uint64_t>(tilingData->kvHeads) * headDim : headDim;
    const uint64_t qInputStride =
        tilingData->inputLayout == 0 ? static_cast<uint64_t>(tilingData->numHeads) * headDim : headDim;
    const uint64_t qOutputStride = qInputStride;
    const uint64_t kvOutputStride = kvInputStride;
    const uint64_t zeroOffset =
        segment.tiles[0].sOffset + static_cast<uint64_t>(GROUPED_TILES - 1) * GROUPED_TILE_ELEMENTS;

    // Q and dOut are invariant for all gathered KV tiles of the same
    // Q task.  Populate their persistent L1 slots before dQ so dK
    // can consume the same dS L1 tile immediately after dQ.
    SyncExtendedL1Reuse();
    bool cacheUpdated = false;
    for (uint32_t qLocal = 0; qLocal < GROUPED_Q_BLOCKS; ++qLocal) {
        int32_t first = -1;
        for (uint32_t t = 0; t < segment.tileCount; ++t) {
            if (segment.tiles[t].groupQIdx == qLocal) {
                first = static_cast<int32_t>(t);
                break;
            }
        }
        if (first < 0) {
            continue;
        }
        TaskInfo &tile = segment.tiles[first];
        LayoutB3 layoutB(tile.curCalQSize, headDim, qInputStride);
        GemmCoord shape{tile.curCalKVSize, headDim, tile.curCalQSize};
        if (((cachedQMask >> qLocal) & 1U) == 0 || cachedQOffsets[qLocal] != tile.qOffset) {
            const uint64_t qOffset = GroupedQHeadOffset(tile, tilingData);
            blockMmad3.PreloadFullB(gDout[qOffset], layoutB, shape, qLocal);
            blockMmad3.PreloadFullB(gQ[qOffset], layoutB, shape, qLocal + GROUPED_Q_BLOCKS);
            cachedQOffsets[qLocal] = tile.qOffset;
            cachedQMask |= 1U << qLocal;
            cacheUpdated = true;
        }
    }
    if (cacheUpdated) {
        blockMmad3.FinishFullPreload();
    }

    // dQ: each gathered K tile is one full reduction operand.  Keep
    // the at-most-two tiles for one Q in L1 and accumulate in L0C.
    // After each dQ MMAD, reuse its zN dS tile for dK by transposing
    // only the L1->L0A transfer into the other L0C bank.
    for (uint32_t qLocal = 0; qLocal < GROUPED_Q_BLOCKS; ++qLocal) {
        uint32_t tileIndices[GROUPED_PACKED_FRAGMENTS_PER_Q];
        uint32_t matches = 0;
        bool firstKvSegment = false;
        for (uint32_t t = 0; t < segment.tileCount; ++t) {
            if (segment.tiles[t].groupQIdx == qLocal) {
                tileIndices[matches++] = t;
                firstKvSegment |= segment.tiles[t].firstQSegment;
            }
        }
        if (matches == 0) {
            continue;
        }
        AscendC::WaitEvent(VEC2CUBE);
        for (uint32_t i = 0; i < matches; ++i) {
            TaskInfo &tile = segment.tiles[tileIndices[i]];
            LayoutB2 firstLayout(tile.kvFirstSize, headDim, kvInputStride);
            LayoutB2 secondLayout(tile.kvSecondSize, headDim, kvInputStride);
            LayoutB2 paddingLayout(tile.curCalKVSize - tile.kvValidSize, headDim, headDim);
            GemmCoord shape{tile.curCalQSize, headDim, tile.curCalKVSize};
            blockMmad2.PreloadGatheredBAlongK(gK[tile.kvOffset], firstLayout, gK[tile.kvSecondOffset], secondLayout,
                                              gDs[zeroOffset], paddingLayout, tile.kvFirstSize, tile.kvSecondSize,
                                              shape, i);
        }
        const uint32_t accumulatorSlot = qLocal & 1U;
        for (uint32_t i = 0; i < matches; ++i) {
            TaskInfo &tile = segment.tiles[tileIndices[i]];
            LayoutA2 layoutA(tile.curCalQSize, tile.curCalKVSize);
            LayoutB2 layoutB(tile.curCalKVSize, headDim);
            GemmCoord shape{tile.curCalQSize, headDim, tile.curCalKVSize};
            blockMmad2.PreloadAccumA(gDs[tile.sOffset], layoutA, shape, i & 1U);
            blockMmad2.AccumulatePreloadedA(i & 1U, i, layoutA, layoutB, shape, i == 0, i + 1 == matches,
                                            accumulatorSlot);

            const bool contiguousOutput =
                tile.kvPartCount == 2 &&
                tile.kvSecondOffset == tile.kvOffset + static_cast<uint64_t>(tile.kvFirstSize) * kvOutputStride;
            const uint32_t dkAccumulatorSlot = 1U - accumulatorSlot;
            LayoutB3 dkLayoutB(tile.curCalQSize, headDim);
            LayoutC3 dkLayoutC(tile.kvValidSize, headDim, kvOutputStride);
            GemmCoord dkShape{tile.kvValidSize, headDim, tile.curCalQSize};
            blockMmad3.AccumulateTransposeFromExternalZNL1(blockMmad2.GetAccumL1A(i & 1U), (i & 1U) + PINGPONG_OFFSET_2,
                                                           tile.groupQIdx + GROUPED_Q_BLOCKS, dkLayoutB, dkShape,
                                                           dkAccumulatorSlot);
            if (tile.kvPartCount == 1 || contiguousOutput) {
                blockMmad3.FlushAccumulator(gDk[tile.kvOffset], dkLayoutC, dkShape, true, dkAccumulatorSlot);
            } else {
                LayoutC3 firstLayout(tile.kvFirstSize, headDim, kvOutputStride);
                LayoutC3 secondLayout(tile.kvSecondSize, headDim, kvOutputStride);
                blockMmad3.FlushAccumulatorSplitRows(gDk[tile.kvOffset], firstLayout, gDk[tile.kvSecondOffset],
                                                     secondLayout, dkShape, tile.kvFirstSize, true, dkAccumulatorSlot);
            }
        }
        TaskInfo &last = segment.tiles[tileIndices[matches - 1]];
        LayoutC2 layoutC(last.curCalQSize, headDim, qOutputStride);
        GemmCoord outShape{last.curCalQSize, headDim, last.curCalKVSize};
        blockMmad2.FlushAccumulator(gDq[last.qOffset], layoutC, outShape, !firstKvSegment, accumulatorSlot);
    }
    if (finalSegment) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
    }

    // dK was fused with dQ above.  The remaining P^T*dOut pass
    // computes dV from its persistent dOut cache.
    for (uint32_t pass = 0; pass < 1; ++pass) {
        for (uint32_t t = 0; t < segment.tileCount; ++t) {
            TaskInfo &tile = segment.tiles[t];
            // All gathered fragments are contiguous in the logical
            // compute tile even when their GM destinations differ.
            // Accumulate them with one MMAD, then either issue one
            // contiguous store or scatter two row ranges directly
            // from L0C.  This is capacity driven and does not depend
            // on a particular user block_shape value.
            const bool contiguousOutput =
                tile.kvPartCount == 2 &&
                tile.kvSecondOffset == tile.kvOffset + static_cast<uint64_t>(tile.kvFirstSize) * kvOutputStride;
            const uint32_t slot = t & 1U;
            LayoutA3 layoutA(tile.kvValidSize, tile.curCalQSize, tile.curCalKVSize);
            LayoutB3 layoutB(tile.curCalQSize, headDim);
            LayoutC3 layoutC(tile.kvValidSize, headDim, kvOutputStride);
            GemmCoord shape{tile.kvValidSize, headDim, tile.curCalQSize};
            AscendC::GlobalTensor<ElementInput> source = gP[tile.sOffset];
            blockMmad3.PreloadAccumA(source, layoutA, shape, slot);
            blockMmad3.AccumulatePreloadedA(slot, tile.groupQIdx, layoutA, layoutB, shape, true, true, slot);
            AscendC::GlobalTensor<float> firstOutput = gDv[tile.kvOffset];
            if (tile.kvPartCount == 1 || contiguousOutput) {
                blockMmad3.FlushAccumulator(firstOutput, layoutC, shape, true, slot);
            } else {
                AscendC::GlobalTensor<float> secondOutput = gDv[tile.kvSecondOffset];
                LayoutC3 firstLayout(tile.kvFirstSize, headDim, kvOutputStride);
                LayoutC3 secondLayout(tile.kvSecondSize, headDim, kvOutputStride);
                blockMmad3.FlushAccumulatorSplitRows(firstOutput, firstLayout, secondOutput, secondLayout, shape,
                                                     tile.kvFirstSize, true, slot);
            }
        }
        if (finalSegment) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(pass == 0 ? EVENT_ID6 : EVENT_ID7);
        }
    }
    if (finalSegment) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID7);
    }
}

__aicore__ inline void RunGroupedGrad(SegmentInfo &segment, const __gm__ BlockSparseAttentionGradTilingData *tilingData,
                                      AscendC::GlobalTensor<ElementInput> gK, AscendC::GlobalTensor<ElementInput> gQ,
                                      AscendC::GlobalTensor<ElementA1> gDout, AscendC::GlobalTensor<ElementInput> gP,
                                      AscendC::GlobalTensor<ElementInput> gDs, AscendC::GlobalTensor<float> gDq,
                                      AscendC::GlobalTensor<float> gDk, AscendC::GlobalTensor<float> gDv,
                                      BlockMmadBSAG2 &blockMmad2, BlockMmadBSAG3 &blockMmad3, uint32_t &mmadFlag,
                                      bool exclusiveKvOwner, uint64_t &writtenKvBlocks, uint64_t *cachedQOffsets,
                                      uint32_t &cachedQMask, bool finalSegment)
{
    const uint32_t headDim = tilingData->headDim;
    const uint64_t qInputStride =
        (tilingData->inputLayout == 0 ? static_cast<uint64_t>(tilingData->numHeads) * headDim : headDim);
    const uint64_t kvInputStride =
        (tilingData->inputLayout == 0 ? static_cast<uint64_t>(tilingData->kvHeads) * headDim : headDim);
    const uint64_t qOutputStride =
        tilingData->inputLayout == 0 ? static_cast<uint64_t>(tilingData->numHeads) * headDim : headDim;
    const uint64_t kvOutputStride =
        tilingData->inputLayout == 0 ? static_cast<uint64_t>(tilingData->kvHeads) * headDim : headDim;
    // The legacy sub-128 path keeps at most four K tiles in extended
    // B slots. A normal 128-basic-tile packet can contain up to 16
    // unique KVs, so wider packets refill the same four slots in
    // batches and retain two Q accumulators in the two L0C banks.
    SyncExtendedL1Reuse();
    // Preserve the four-slot CachedB path when the packet fits it
    // (notably a dense 4Q x 4KV packet).  Only wider sparse packets
    // need the streaming path.
    const bool streamDqK = tilingData->basicKVBlockSize == 128 && segment.uniqueKvCount > GROUPED_KV_BLOCKS;
    if (!streamDqK) {
        for (uint32_t kvLocal = 0; kvLocal < segment.uniqueKvCount; ++kvLocal) {
            int32_t first = -1;
            for (uint32_t t = 0; t < segment.tileCount; ++t) {
                if (segment.tiles[t].groupKvIdx == kvLocal) {
                    first = static_cast<int32_t>(t);
                    break;
                }
            }
            if (first < 0) {
                continue;
            }
            TaskInfo &tile = segment.tiles[first];
            LayoutB2 layoutB(tile.curCalKVSize, headDim, kvInputStride);
            GemmCoord shape{tile.curCalQSize, headDim, tile.curCalKVSize};
            blockMmad2.PreloadB(gK[GroupedKvHeadOffset(tile, tilingData)], layoutB, shape, kvLocal);
        }
    }

    if (streamDqK) {
        // The first K batch reuses the phase-boundary fence above.
        // Later batches establish a real MTE1->MTE2 dependency before
        // overwriting the four compact B-cache slots. Pairing Q rows
        // keeps both L0C banks live and lets one K load serve either
        // row when both sparse edges use the same KV column.
        bool firstStreamKBatch = true;
        for (uint32_t qStart = 0; qStart < GROUPED_Q_BLOCKS; qStart += 2) {
            int32_t tileByKv[2][GROUPED_PACKET_KV_BLOCKS];
            uint32_t total[2] = {};
            uint32_t issued[2] = {};
            uint32_t lastTileIndex[2] = {};
            bool firstKvSegment[2] = {};
            bool eventConsumed[2] = {};
            for (uint32_t qPair = 0; qPair < 2; ++qPair) {
                for (uint32_t kvLocal = 0; kvLocal < GROUPED_PACKET_KV_BLOCKS; ++kvLocal) {
                    tileByKv[qPair][kvLocal] = -1;
                }
            }
            for (uint32_t t = 0; t < segment.tileCount; ++t) {
                TaskInfo &tile = segment.tiles[t];
                if (tile.groupQIdx < qStart || tile.groupQIdx >= qStart + 2) {
                    continue;
                }
                const uint32_t qPair = tile.groupQIdx - qStart;
                tileByKv[qPair][tile.groupKvIdx] = static_cast<int32_t>(t);
                ++total[qPair];
                lastTileIndex[qPair] = t;
                firstKvSegment[qPair] |= tile.firstQSegment;
            }
            if (total[0] == 0 && total[1] == 0) {
                continue;
            }

            uint32_t kvCursor = 0;
            while (kvCursor < segment.uniqueKvCount) {
                uint32_t batchKv[GROUPED_KV_BLOCKS];
                uint32_t batchFirstTile[GROUPED_KV_BLOCKS];
                uint32_t batchCount = 0;
                while (kvCursor < segment.uniqueKvCount && batchCount < GROUPED_KV_BLOCKS) {
                    const uint32_t kvLocal = kvCursor++;
                    const int32_t first = tileByKv[0][kvLocal] >= 0 ? tileByKv[0][kvLocal] : tileByKv[1][kvLocal];
                    if (first < 0) {
                        continue;
                    }
                    batchKv[batchCount] = kvLocal;
                    batchFirstTile[batchCount] = static_cast<uint32_t>(first);
                    ++batchCount;
                }
                if (batchCount == 0) {
                    continue;
                }
                if (!firstStreamKBatch) {
                    SyncExtendedL1Reuse();
                }
                firstStreamKBatch = false;
                for (uint32_t cacheSlot = 0; cacheSlot < batchCount; ++cacheSlot) {
                    TaskInfo &tile = segment.tiles[batchFirstTile[cacheSlot]];
                    LayoutB2 layoutB(tile.curCalKVSize, headDim, kvInputStride);
                    GemmCoord shape{tile.curCalQSize, headDim, tile.curCalKVSize};
                    blockMmad2.PreloadFullB(gK[GroupedKvHeadOffset(tile, tilingData)], layoutB, shape, cacheSlot);
                }
                // One MTE2->MTE1 fence covers all independent K
                // copies in this batch.
                blockMmad2.FinishFullPreload();

                for (uint32_t qPair = 0; qPair < 2; ++qPair) {
                    uint32_t batchTiles[GROUPED_KV_BLOCKS];
                    uint32_t batchCacheSlots[GROUPED_KV_BLOCKS];
                    uint32_t matches = 0;
                    for (uint32_t cacheSlot = 0; cacheSlot < batchCount; ++cacheSlot) {
                        const int32_t tileIndex = tileByKv[qPair][batchKv[cacheSlot]];
                        if (tileIndex < 0) {
                            continue;
                        }
                        batchTiles[matches] = static_cast<uint32_t>(tileIndex);
                        batchCacheSlots[matches] = cacheSlot;
                        ++matches;
                    }
                    if (matches == 0) {
                        continue;
                    }
                    // K is independent of the vector result and was
                    // prefetched above. Cross-core completions are an
                    // ordered token stream rather than tagged by Q,
                    // so consume every preceding live Q token before
                    // reading this Q's first dS tile.
                    if (!eventConsumed[qPair]) {
                        for (uint32_t readyPair = 0; readyPair <= qPair; ++readyPair) {
                            if (total[readyPair] != 0 && !eventConsumed[readyPair]) {
                                AscendC::WaitEvent(VEC2CUBE);
                                eventConsumed[readyPair] = true;
                            }
                        }
                    }
                    const uint32_t initialPreloads = matches < 2 ? matches : 2;
                    for (uint32_t i = 0; i < initialPreloads; ++i) {
                        TaskInfo &tile = segment.tiles[batchTiles[i]];
                        LayoutA2 layoutA(tile.curCalQSize, tile.curCalKVSize);
                        GemmCoord shape{tile.curCalQSize, headDim, tile.curCalKVSize};
                        blockMmad2.PreloadAccumA(gDs[tile.sOffset], layoutA, shape, i);
                    }
                    for (uint32_t i = 0; i < matches; ++i) {
                        const uint32_t slot = i & 1U;
                        TaskInfo &tile = segment.tiles[batchTiles[i]];
                        LayoutA2 layoutA(tile.curCalQSize, tile.curCalKVSize);
                        LayoutB2 layoutB(tile.curCalKVSize, headDim);
                        GemmCoord shape{tile.curCalQSize, headDim, tile.curCalKVSize};
                        const bool initC = issued[qPair] == 0;
                        const bool finalC = issued[qPair] + 1 == total[qPair];
                        blockMmad2.AccumulatePreloadedA(slot, batchCacheSlots[i], layoutA, layoutB, shape, initC,
                                                        finalC, qPair);
                        ++issued[qPair];
                        if (i + 2 < matches) {
                            TaskInfo &nextTile = segment.tiles[batchTiles[i + 2]];
                            LayoutA2 nextLayoutA(nextTile.curCalQSize, nextTile.curCalKVSize);
                            GemmCoord nextShape{nextTile.curCalQSize, headDim, nextTile.curCalKVSize};
                            blockMmad2.PreloadAccumA(gDs[nextTile.sOffset], nextLayoutA, nextShape, slot);
                        }
                    }
                }
            }

            for (uint32_t qPair = 0; qPair < 2; ++qPair) {
                if (total[qPair] == 0) {
                    continue;
                }
                TaskInfo &tile = segment.tiles[lastTileIndex[qPair]];
                LayoutC2 layoutC(tile.curCalQSize, headDim, qOutputStride);
                GemmCoord outShape{tile.curCalQSize, headDim, tile.curCalKVSize};
                blockMmad2.FlushAccumulator(gDq[tile.qOffset], layoutC, outShape, !firstKvSegment[qPair], qPair);
            }
        }
        // Protect the compact region before blockMmad1 reuses it in
        // the next Cube1 segment. High-L1 dK/dV work can overlap the
        // resulting dependency.
        if (!firstStreamKBatch) {
            SyncExtendedL1Reuse();
        }
    } else {
        for (uint32_t qLocal = 0; qLocal < GROUPED_Q_BLOCKS; ++qLocal) {
            uint32_t tileIndices[GROUPED_TILES];
            uint32_t matches = 0;
            bool firstKvSegment = false;
            for (uint32_t t = 0; t < segment.tileCount; ++t) {
                if (segment.tiles[t].groupQIdx == qLocal) {
                    tileIndices[matches++] = t;
                    firstKvSegment |= segment.tiles[t].firstQSegment;
                }
            }
            if (matches == 0) {
                continue;
            }
            AscendC::WaitEvent(VEC2CUBE);
            const uint32_t accumulatorSlot = qLocal & 1U;
            const uint32_t initialPreloads = matches < 2 ? matches : 2;
            for (uint32_t i = 0; i < initialPreloads; ++i) {
                TaskInfo &tile = segment.tiles[tileIndices[i]];
                LayoutA2 layoutA(tile.curCalQSize, tile.curCalKVSize);
                GemmCoord shape{tile.curCalQSize, headDim, tile.curCalKVSize};
                blockMmad2.PreloadAccumA(gDs[tile.sOffset], layoutA, shape, i);
            }
            for (uint32_t i = 0; i < matches; ++i) {
                const uint32_t slot = i & 1U;
                TaskInfo &tile = segment.tiles[tileIndices[i]];
                LayoutA2 layoutA(tile.curCalQSize, tile.curCalKVSize);
                LayoutB2 layoutB(tile.curCalKVSize, headDim);
                GemmCoord shape{tile.curCalQSize, headDim, tile.curCalKVSize};
                blockMmad2.AccumulatePreloadedA(slot, tile.groupKvIdx, layoutA, layoutB, shape, i == 0,
                                                i + 1 == matches, accumulatorSlot);
                if (i + 2 < matches) {
                    TaskInfo &nextTile = segment.tiles[tileIndices[i + 2]];
                    LayoutA2 nextLayoutA(nextTile.curCalQSize, nextTile.curCalKVSize);
                    GemmCoord nextShape{nextTile.curCalQSize, headDim, nextTile.curCalKVSize};
                    blockMmad2.PreloadAccumA(gDs[nextTile.sOffset], nextLayoutA, nextShape, slot);
                }
            }
            TaskInfo &tile = segment.tiles[tileIndices[matches - 1]];
            LayoutC2 layoutC(tile.curCalQSize, headDim, qOutputStride);
            GemmCoord outShape{tile.curCalQSize, headDim, tile.curCalKVSize};
            blockMmad2.FlushAccumulator(gDq[tile.qOffset], layoutC, outShape, !firstKvSegment, accumulatorSlot);
        }
    }
    if (finalSegment) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
    }

    // dOut and Q are shared by all dV/dK work in this gradient
    // segment. Their eight L1 slots do not overlap dQ, so reload a
    // pair only when Cube1 invalidated it or the Q offset changed.
    bool cacheUpdated = false;
    for (uint32_t qLocal = 0; qLocal < GROUPED_Q_BLOCKS; ++qLocal) {
        int32_t first = -1;
        for (uint32_t t = 0; t < segment.tileCount; ++t) {
            if (segment.tiles[t].groupQIdx == qLocal) {
                first = static_cast<int32_t>(t);
                break;
            }
        }
        if (first < 0) {
            continue;
        }
        TaskInfo &tile = segment.tiles[first];
        LayoutB3 layoutB(tile.curCalQSize, headDim, qInputStride);
        GemmCoord shape{tile.curCalKVSize, headDim, tile.curCalQSize};
        if (((cachedQMask >> qLocal) & 1U) == 0 || cachedQOffsets[qLocal] != tile.qOffset) {
            const uint64_t packedQOffset = GroupedQHeadOffset(tile, tilingData);
            blockMmad3.PreloadFullB(gDout[packedQOffset], layoutB, shape, qLocal);
            blockMmad3.PreloadFullB(gQ[packedQOffset], layoutB, shape, qLocal + GROUPED_Q_BLOCKS);
            cachedQOffsets[qLocal] = tile.qOffset;
            cachedQMask |= 1U << qLocal;
            cacheUpdated = true;
        }
    }
    if (cacheUpdated) {
        blockMmad3.FinishFullPreload();
    }
    bool firstKvOutput[GROUPED_PACKET_KV_BLOCKS] = {};
    for (uint32_t kvLocal = 0; kvLocal < segment.uniqueKvCount; ++kvLocal) {
        uint32_t tileIndices[GROUPED_Q_BLOCKS];
        uint32_t matches = 0;
        uint32_t kvBasicIdx = 0;
        for (uint32_t t = 0; t < segment.tileCount; ++t) {
            if (segment.tiles[t].groupKvIdx == kvLocal) {
                tileIndices[matches++] = t;
                kvBasicIdx = segment.tiles[t].kvBasicIdx;
            }
        }
        if (matches == 0) {
            continue;
        }
        const uint32_t accumulatorSlot = kvLocal & 1U;
        firstKvOutput[kvLocal] = exclusiveKvOwner && kvBasicIdx < 64 && ((writtenKvBlocks >> kvBasicIdx) & 1ULL) == 0;
        const uint32_t initialPreloads = matches < 2 ? matches : 2;
        for (uint32_t i = 0; i < initialPreloads; ++i) {
            TaskInfo &tile = segment.tiles[tileIndices[i]];
            LayoutA3 layoutA(tile.curCalKVSize, tile.curCalQSize);
            GemmCoord shape{tile.curCalKVSize, headDim, tile.curCalQSize};
            blockMmad3.PreloadAccumA(gP[tile.sOffset], layoutA, shape, i);
        }
        for (uint32_t i = 0; i < matches; ++i) {
            const uint32_t slot = i & 1U;
            TaskInfo &tile = segment.tiles[tileIndices[i]];
            LayoutA3 layoutA(tile.curCalKVSize, tile.curCalQSize);
            LayoutB3 layoutB(tile.curCalQSize, headDim);
            GemmCoord shape{tile.curCalKVSize, headDim, tile.curCalQSize};
            blockMmad3.AccumulatePreloadedA(slot, tile.groupQIdx, layoutA, layoutB, shape, i == 0, i + 1 == matches,
                                            accumulatorSlot);
            if (i + 2 < matches) {
                TaskInfo &nextTile = segment.tiles[tileIndices[i + 2]];
                LayoutA3 nextLayoutA(nextTile.curCalKVSize, nextTile.curCalQSize);
                GemmCoord nextShape{nextTile.curCalKVSize, headDim, nextTile.curCalQSize};
                blockMmad3.PreloadAccumA(gP[nextTile.sOffset], nextLayoutA, nextShape, slot);
            }
        }
        TaskInfo &lastTile = segment.tiles[tileIndices[matches - 1]];
        LayoutC3 layoutC(lastTile.curCalKVSize, headDim, kvOutputStride);
        GemmCoord outShape{lastTile.curCalKVSize, headDim, lastTile.curCalQSize};
        blockMmad3.FlushAccumulator(gDv[lastTile.kvOffset], layoutC, outShape, !firstKvOutput[kvLocal],
                                    accumulatorSlot);
    }

    if (finalSegment) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID6);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID7);
    }
    for (uint32_t kvLocal = 0; kvLocal < segment.uniqueKvCount; ++kvLocal) {
        uint32_t tileIndices[GROUPED_Q_BLOCKS];
        uint32_t matches = 0;
        for (uint32_t t = 0; t < segment.tileCount; ++t) {
            if (segment.tiles[t].groupKvIdx == kvLocal) {
                tileIndices[matches++] = t;
            }
        }
        if (matches == 0) {
            continue;
        }
        const uint32_t accumulatorSlot = kvLocal & 1U;

        const uint32_t initialPreloads = matches < 2 ? matches : 2;
        for (uint32_t i = 0; i < initialPreloads; ++i) {
            TaskInfo &tile = segment.tiles[tileIndices[i]];
            LayoutA3 layoutA(tile.curCalKVSize, tile.curCalQSize);
            GemmCoord shape{tile.curCalKVSize, headDim, tile.curCalQSize};
            blockMmad3.PreloadAccumA(gDs[tile.sOffset], layoutA, shape, i);
        }
        for (uint32_t i = 0; i < matches; ++i) {
            const uint32_t slot = i & 1U;
            TaskInfo &tile = segment.tiles[tileIndices[i]];
            LayoutA3 layoutA(tile.curCalKVSize, tile.curCalQSize);
            LayoutB3 layoutB(tile.curCalQSize, headDim);
            GemmCoord shape{tile.curCalKVSize, headDim, tile.curCalQSize};
            blockMmad3.AccumulatePreloadedA(slot, tile.groupQIdx + GROUPED_Q_BLOCKS, layoutA, layoutB, shape, i == 0,
                                            i + 1 == matches, accumulatorSlot);
            if (i + 2 < matches) {
                TaskInfo &nextTile = segment.tiles[tileIndices[i + 2]];
                LayoutA3 nextLayoutA(nextTile.curCalKVSize, nextTile.curCalQSize);
                GemmCoord nextShape{nextTile.curCalKVSize, headDim, nextTile.curCalQSize};
                blockMmad3.PreloadAccumA(gDs[nextTile.sOffset], nextLayoutA, nextShape, slot);
            }
        }
        TaskInfo &lastTile = segment.tiles[tileIndices[matches - 1]];
        LayoutC3 layoutC(lastTile.curCalKVSize, headDim, kvOutputStride);
        GemmCoord outShape{lastTile.curCalKVSize, headDim, lastTile.curCalQSize};
        blockMmad3.FlushAccumulator(gDk[lastTile.kvOffset], layoutC, outShape, !firstKvOutput[kvLocal],
                                    accumulatorSlot);
        if (firstKvOutput[kvLocal]) {
            writtenKvBlocks |= 1ULL << lastTile.kvBasicIdx;
        }
    }
}

__aicore__ inline void RunGroupedGradDispatch(
    SegmentInfo &segment, const __gm__ BlockSparseAttentionGradTilingData *tilingData,
    AscendC::GlobalTensor<ElementInput> gK, AscendC::GlobalTensor<ElementInput> gQ,
    AscendC::GlobalTensor<ElementA1> gDout, AscendC::GlobalTensor<ElementInput> gP,
    AscendC::GlobalTensor<ElementInput> gDs, AscendC::GlobalTensor<float> gDq, AscendC::GlobalTensor<float> gDk,
    AscendC::GlobalTensor<float> gDv, BlockMmadBSAG2 &blockMmad2, BlockMmadBSAG3 &blockMmad3, uint32_t &mmadFlag,
    bool exclusiveKvOwner, uint64_t &writtenKvBlocks, uint64_t *cachedQOffsets, uint32_t &cachedQMask,
    bool finalSegment)
{
    if (tilingData->basicKVBlockSize < 128 && tilingData->blockShapeX >= 16) {
        RunGroupedPackedGrad(segment, tilingData, gK, gQ, gDout, gP, gDs, gDq, gDk, gDv, blockMmad2, blockMmad3,
                             mmadFlag, cachedQOffsets, cachedQMask, finalSegment);
    } else {
        RunGroupedGrad(segment, tilingData, gK, gQ, gDout, gP, gDs, gDq, gDk, gDv, blockMmad2, blockMmad3, mmadFlag,
                       exclusiveKvOwner, writtenKvBlocks, cachedQOffsets, cachedQMask, finalSegment);
    }
}

__aicore__ inline void ProcessGroupedAic(const Params &params, __gm__ BlockSparseAttentionGradTilingData *tilingData)
{
    const uint32_t coreIdx = AscendC::GetBlockIdx();
    const uint32_t groupSize = tilingData->numHeads / tilingData->kvHeads;
    const uint32_t taskLength = tilingData->taskNumPerCore + (tilingData->tailTaskNum > coreIdx ? 1 : 0);
    const uint64_t groupedFp32Bytes =
        static_cast<uint64_t>(tilingData->usedVecCoreNum / 2) * GROUPED_WORKSPACE_CORE_SIZE * sizeof(float);

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
    gDout.SetGlobalBuffer((__gm__ ElementInput *)params.dout);
    gActualQseqlen.SetGlobalBuffer((__gm__ int64_t *)params.actualQseqlen);
    gActualKvseqlen.SetGlobalBuffer((__gm__ int64_t *)params.actualKvseqlen);
    gS.SetGlobalBuffer((__gm__ float *)params.workspace);
    gP.SetGlobalBuffer((__gm__ ElementInput *)(params.workspace + WORKSPACE_P16_OFFSET));
    gDp.SetGlobalBuffer((__gm__ float *)(params.workspace + tilingData->sOutSize));
    gDs.SetGlobalBuffer((__gm__ ElementInput *)(params.workspace + tilingData->sOutSize + WORKSPACE_P16_OFFSET));
    gDq.SetGlobalBuffer((__gm__ float *)(params.workspace + tilingData->sOutSize + tilingData->dPOutSize));
    gDk.SetGlobalBuffer(
        (__gm__ float *)(params.workspace + tilingData->sOutSize + tilingData->dPOutSize + tilingData->dQOutSize));
    gDv.SetGlobalBuffer((__gm__ float *)(params.workspace + tilingData->sOutSize + tilingData->dPOutSize +
                                         tilingData->dQOutSize + tilingData->dKOutSize));

    TaskInfo current;
    initTaskInfo(gActualQseqlen, gActualKvseqlen, tilingData, tilingData->numHeads, tilingData->kvHeads, groupSize,
                 tilingData->headDim, tilingData->maxQSeqlen, tilingData->maxKvSeqlen, tilingData->blockShapeX,
                 tilingData->basicQBlockSize, tilingData->inputLayout, coreIdx, current);

    // The active Cube1 CachedB/Packed schedules use the two A
    // ping-pong slots and B cache slots 0/1 in the compact region below
    // 192 KiB.  This leaves blockMmad3's persistent dOut/Q slots
    // [192 KiB, 448 KiB) untouched across all KV segments of the
    // current Q group.
    BlockMmadBSAG1 blockMmad1(resource);
    BlockMmadBSAG2 blockMmad2(resource, 0, PINGPONG_OFFSET_2, true);
    BlockMmadBSAG3 blockMmad3(resource, L1_SIZE_OFFSET * 7 / 2, PINGPONG_OFFSET_4, true, false, true);
    gP.SetGlobalBuffer((__gm__ ElementInput *)(params.workspace + groupedFp32Bytes));
    gDs.SetGlobalBuffer((__gm__ ElementInput *)(params.workspace + tilingData->sOutSize + groupedFp32Bytes));
    uint32_t mmapFlag = 0;
    SegmentInfo segments[2];
    uint32_t stage = 0;
    uint32_t previousStage = 0;
    bool havePrevious = false;
    bool previousExclusiveKvOwner = false;
    uint32_t processedTasks = 0;
    uint64_t writtenKvBlocks = 0;
    uint64_t cachedQOffsets[GROUPED_Q_BLOCKS] = {};
    uint32_t cachedQMask = 0;
    const uint32_t tasksPerFullQBlock =
        (tilingData->blockShapeX + tilingData->basicQBlockSize - 1) / tilingData->basicQBlockSize;
    const uint32_t fullQBlocks = tilingData->maxQSeqlen / tilingData->blockShapeX;
    const uint32_t qRemainder = tilingData->maxQSeqlen - fullQBlocks * tilingData->blockShapeX;
    const uint32_t qBlocksPerHead =
        fullQBlocks * tasksPerFullQBlock + (qRemainder + tilingData->basicQBlockSize - 1) / tilingData->basicQBlockSize;
    const MaskLayoutInfo maskLayout = GetMaskLayout(tilingData);
    bool exclusiveKvOwner = false;
    AscendC::SyncAll<false>();
    SetFlag();
    while (processedTasks < taskLength) {
        const bool startsWholeHead = current.curQSeqIdx == 0;
        if (startsWholeHead) {
            if (processedTasks != 0 && havePrevious) {
                RunGroupedGradDispatch(segments[previousStage], tilingData, gK, gQ, gDout, gP, gDs, gDq, gDk, gDv,
                                       blockMmad2, blockMmad3, mmapFlag, previousExclusiveKvOwner, writtenKvBlocks,
                                       cachedQOffsets, cachedQMask, false);
                havePrevious = false;
            }
            writtenKvBlocks = 0;
            cachedQMask = 0;
        }
        if (startsWholeHead) {
            exclusiveKvOwner =
                tilingData->numHeads == tilingData->kvHeads && taskLength - processedTasks >= qBlocksPerHead;
        }
        TaskInfo qInfos[GROUPED_Q_BLOCKS];
        uint32_t qLimits[GROUPED_Q_BLOCKS];
        qInfos[0] = current;
        uint32_t qCount = 1;
        TaskInfo nextGroup;
        bool nextReady = false;
        while (qCount < GROUPED_ACTIVE_Q_BLOCKS && processedTasks + qCount < taskLength) {
            TaskInfo candidate;
            updateNextGroupedTaskInfo(gActualQseqlen, gActualKvseqlen, tilingData->numHeads, tilingData->kvHeads,
                                      groupSize, tilingData->headDim, tilingData->blockShapeX,
                                      tilingData->basicQBlockSize, tilingData->inputLayout, qInfos[qCount - 1],
                                      candidate);
            if (candidate.curBatchIdx != current.curBatchIdx || candidate.curHeadIdx != current.curHeadIdx) {
                nextGroup = candidate;
                nextReady = true;
                break;
            }
            qInfos[qCount++] = candidate;
        }
        if (processedTasks + qCount < taskLength && !nextReady) {
            updateNextGroupedTaskInfo(gActualQseqlen, gActualKvseqlen, tilingData->numHeads, tilingData->kvHeads,
                                      groupSize, tilingData->headDim, tilingData->blockShapeX,
                                      tilingData->basicQBlockSize, tilingData->inputLayout, qInfos[qCount - 1],
                                      nextGroup);
            nextReady = true;
        }

        uint32_t maxKvLimit = 0;
        const uint32_t kvBlockCount = (qInfos[0].kvSeqlen + tilingData->blockShapeY - 1) / tilingData->blockShapeY;
        for (uint32_t qLocal = 0; qLocal < qCount; ++qLocal) {
            qLimits[qLocal] = GetKvBlockLimit(params, qInfos[qLocal], kvBlockCount, tilingData);
            if (qLimits[qLocal] > maxKvLimit) {
                maxKvLimit = qLimits[qLocal];
            }
        }
        uint64_t maskRowOffsets[GROUPED_Q_BLOCKS] = {};
        for (uint32_t qLocal = 0; qLocal < qCount; ++qLocal) {
            maskRowOffsets[qLocal] = GetMaskRowOffset(qInfos[qLocal], maskLayout);
        }
        const bool packedSegments = tilingData->basicKVBlockSize < 128 && tilingData->blockShapeX >= 16;
        uint32_t packedKvStarts[GROUPED_Q_BLOCKS] = {};
        uint32_t kvStart = 0;
        while (true) {
            bool traversalDone = kvStart >= maxKvLimit;
            if (packedSegments) {
                traversalDone = true;
                for (uint32_t qLocal = 0; qLocal < qCount; ++qLocal) {
                    traversalDone &= packedKvStarts[qLocal] >= qLimits[qLocal];
                }
            }
            if (traversalDone) {
                break;
            }
            const uint32_t kvCount =
                maxKvLimit - kvStart < GROUPED_KV_BLOCKS ? maxKvLimit - kvStart : GROUPED_KV_BLOCKS;
            SegmentInfo &segment = segments[stage];
            const uint64_t workspaceBase = static_cast<uint64_t>(coreIdx) * GROUPED_WORKSPACE_CORE_SIZE +
                                           static_cast<uint64_t>(stage) * GROUPED_WORKSPACE_STAGE_SIZE;
            BuildGroupedSegment(params, tilingData, qInfos, qLimits, qCount, kvStart, kvCount, packedKvStarts,
                                maskRowOffsets, workspaceBase, segment);
            if (!packedSegments) {
                kvStart = segment.nextKvStart;
            }
            if (segment.tileCount == 0) {
                continue;
            }
            if (tilingData->basicKVBlockSize < 128 && tilingData->blockShapeX >= 16) {
                RunGroupedCube1Packed(segment, tilingData, gQ, gK, gV, gDout, gS, gDp, gP, blockMmad1, mmapFlag);
            } else if (tilingData->headDim == 128) {
                // Use the stable L1 CachedB pipeline.  Keeping K/V in
                // L0B across unrelated gradient phases aliases event
                // lifetimes under sustained execution.  Keep the raw
                // QK^T and dO*V^T results in FP32, matching FAG: only
                // P and dS are rounded to the input dtype after the
                // vector softmax/softmax-grad arithmetic is complete.
                RunGroupedCube1CachedB(segment, tilingData, gQ, gK, gV, gDout, gS, gDp, blockMmad1, mmapFlag);
            } else {
                RunGroupedCube1CachedB(segment, tilingData, gQ, gK, gV, gDout, gS, gDp, blockMmad1, mmapFlag);
            }
            // Compact Cube1 does not alias blockMmad3's persistent
            // dOut/Q slots.  Keep cached rows across KV segments;
            // cachedQOffsets invalidates individual slots when the
            // Q group changes, while the whole-head boundary above
            // resets the mask after draining the pending segment.
            AscendC::CrossCoreSetFlag<2, PIPE_FIX>(CUBE2VEC);
            if (havePrevious) {
                RunGroupedGradDispatch(segments[previousStage], tilingData, gK, gQ, gDout, gP, gDs, gDq, gDk, gDv,
                                       blockMmad2, blockMmad3, mmapFlag, previousExclusiveKvOwner, writtenKvBlocks,
                                       cachedQOffsets, cachedQMask, false);
            }
            previousStage = stage;
            previousExclusiveKvOwner = exclusiveKvOwner;
            stage = 1 - stage;
            havePrevious = true;
        }
        processedTasks += qCount;
        if (processedTasks < taskLength && nextReady) {
            current = nextGroup;
        }
    }
    if (havePrevious) {
        DrainFinalCube1Flags();
        RunGroupedGradDispatch(segments[previousStage], tilingData, gK, gQ, gDout, gP, gDs, gDq, gDk, gDv, blockMmad2,
                               blockMmad3, mmapFlag, previousExclusiveKvOwner, writtenKvBlocks, cachedQOffsets,
                               cachedQMask, true);
    }
    WaitGroupedFlag();
    AscendC::CrossCoreSetFlag<2, PIPE_FIX>(CUBE2POST);
}

__aicore__ inline void ProcessGroupedAiv(const Params &params, __gm__ BlockSparseAttentionGradTilingData *tilingData)
{
    const uint32_t vecCoreIdx = AscendC::GetBlockIdx();
    const uint32_t coreIdx = vecCoreIdx / 2;
    const uint32_t subBlockIdx = vecCoreIdx % 2;
    const uint32_t groupSize = tilingData->numHeads / tilingData->kvHeads;
    const uint32_t taskLength = tilingData->taskNumPerCore + (tilingData->tailTaskNum > coreIdx ? 1 : 0);
    const uint64_t groupedFp32Bytes =
        static_cast<uint64_t>(tilingData->usedVecCoreNum / 2) * GROUPED_WORKSPACE_CORE_SIZE * sizeof(float);

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
    uint32_t stage = 0;
    uint32_t processedTasks = 0;

    while (processedTasks < taskLength) {
        TaskInfo qInfos[GROUPED_Q_BLOCKS];
        uint32_t qLimits[GROUPED_Q_BLOCKS];
        qInfos[0] = current;
        uint32_t qCount = 1;
        TaskInfo nextGroup;
        bool nextReady = false;
        while (qCount < GROUPED_ACTIVE_Q_BLOCKS && processedTasks + qCount < taskLength) {
            TaskInfo candidate;
            updateNextGroupedTaskInfo(gActualQseqlen, gActualKvseqlen, tilingData->numHeads, tilingData->kvHeads,
                                      groupSize, tilingData->headDim, tilingData->blockShapeX,
                                      tilingData->basicQBlockSize, tilingData->inputLayout, qInfos[qCount - 1],
                                      candidate);
            if (candidate.curBatchIdx != current.curBatchIdx || candidate.curHeadIdx != current.curHeadIdx) {
                nextGroup = candidate;
                nextReady = true;
                break;
            }
            qInfos[qCount++] = candidate;
        }
        if (processedTasks + qCount < taskLength && !nextReady) {
            updateNextGroupedTaskInfo(gActualQseqlen, gActualKvseqlen, tilingData->numHeads, tilingData->kvHeads,
                                      groupSize, tilingData->headDim, tilingData->blockShapeX,
                                      tilingData->basicQBlockSize, tilingData->inputLayout, qInfos[qCount - 1],
                                      nextGroup);
            nextReady = true;
        }

        for (uint32_t qLocal = 0; qLocal < qCount; ++qLocal) {
            const uint32_t firstHalfRows = (qInfos[qLocal].curCalQSize + 1) / 2;
            const uint32_t subRows = subBlockIdx == 0 ? firstHalfRows : qInfos[qLocal].curCalQSize - firstHalfRows;
            // D uses flattened (token, head) rows in [T, numHeads, 8].
            // Scale AIV1's half-row offset by numHeads to address the same tokens.
            if (tilingData->inputLayout == 0) {
                vecSftmg(
                    qInfos[qLocal].qOffset / tilingData->headDim + subBlockIdx * firstHalfRows * tilingData->numHeads,
                    subRows);
            } else {
                vecSftmg(qInfos[qLocal].qOffset / tilingData->headDim + subBlockIdx * firstHalfRows, subRows);
            }
        }
        // vecSftmg queues D's UB->GM copy. Order the first prefetch read
        // after those MTE3 writes to prevent a read-after-write hazard.
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);

        uint32_t maxKvLimit = 0;
        const uint32_t kvBlockCount = (qInfos[0].kvSeqlen + tilingData->blockShapeY - 1) / tilingData->blockShapeY;
        for (uint32_t qLocal = 0; qLocal < qCount; ++qLocal) {
            qLimits[qLocal] = GetKvBlockLimit(params, qInfos[qLocal], kvBlockCount, tilingData);
            if (qLimits[qLocal] > maxKvLimit) {
                maxKvLimit = qLimits[qLocal];
            }
        }
        uint64_t maskRowOffsets[GROUPED_Q_BLOCKS] = {};
        for (uint32_t qLocal = 0; qLocal < qCount; ++qLocal) {
            maskRowOffsets[qLocal] = GetMaskRowOffset(qInfos[qLocal], maskLayout);
        }
        // D and LSE depend only on q, so retain all four metadata
        // slots across the group's KV segments.  The epilogue owns
        // non-overlapping UB storage for the full group.
        softmaxOp.ResetGroupedQMetadata();
        const bool packedSegments = tilingData->basicKVBlockSize < 128 && tilingData->blockShapeX >= 16;
        uint32_t packedKvStarts[GROUPED_Q_BLOCKS] = {};
        uint32_t kvStart = 0;
        while (true) {
            bool traversalDone = kvStart >= maxKvLimit;
            if (packedSegments) {
                traversalDone = true;
                for (uint32_t qLocal = 0; qLocal < qCount; ++qLocal) {
                    traversalDone &= packedKvStarts[qLocal] >= qLimits[qLocal];
                }
            }
            if (traversalDone) {
                break;
            }
            const uint32_t kvCount =
                maxKvLimit - kvStart < GROUPED_KV_BLOCKS ? maxKvLimit - kvStart : GROUPED_KV_BLOCKS;
            SegmentInfo segment;
            const uint64_t workspaceBase = static_cast<uint64_t>(coreIdx) * GROUPED_WORKSPACE_CORE_SIZE +
                                           static_cast<uint64_t>(stage) * GROUPED_WORKSPACE_STAGE_SIZE;
            BuildGroupedSegment(params, tilingData, qInfos, qLimits, qCount, kvStart, kvCount, packedKvStarts,
                                maskRowOffsets, workspaceBase, segment);
            if (!packedSegments) {
                kvStart = segment.nextKvStart;
            }
            if (segment.tileCount == 0) {
                continue;
            }
            AscendC::WaitEvent(CUBE2VEC);
            // Tiles are q-major inside one segment.  Four metadata
            // slots retain D/LSE for the complete q group.
            SfmParams tileParams[GROUPED_TILES];
            for (uint32_t t = 0; t < segment.tileCount; ++t) {
                TaskInfo &tile = segment.tiles[t];
                const uint64_t firstHalfRows = (tile.curCalQSize + 1) / 2;
                const uint64_t executeRow = subBlockIdx == 0 ? firstHalfRows : tile.curCalQSize - firstHalfRows;
                const uint64_t coreOffset = subBlockIdx * firstHalfRows * tile.curCalKVSize;
                const uint64_t curVecCoreS1Idx = tile.curQSeqIdx + subBlockIdx * firstHalfRows;
                const uint64_t vector16Soffset = groupedFp32Bytes + (tile.sOffset + coreOffset) * sizeof(ElementInput);
                const uint64_t vectorInputOffset = (tile.sOffset + coreOffset) * sizeof(float);
                tileParams[t] =
                    SfmParams(params.workspace + vectorInputOffset, params.softmaxLse,
                              params.workspace + tilingData->sOutSize + vectorInputOffset, params.actualQseqlen,
                              params.actualKvseqlen, sftmgGm, params.workspace + vector16Soffset,
                              params.workspace + tilingData->sOutSize + vector16Soffset, params.tiling, executeRow,
                              tile.curCalKVSize, executeRow * tile.curCalKVSize, tile.curBatchIdx, tile.curHeadIdx,
                              curVecCoreS1Idx, tile.groupQIdx, tile.kvValidSize);
            }
            uint32_t inputStage = 0;
            softmaxOp.PrefetchGrouped(tileParams[0], inputStage);
            for (uint32_t t = 0; t < segment.tileCount; ++t) {
                if (t + 1 < segment.tileCount) {
                    softmaxOp.PrefetchGrouped(tileParams[t + 1], 1 - inputStage);
                }
                softmaxOp.ComputeGrouped(tileParams[t], inputStage);
                if (t + 1 == segment.tileCount || segment.tiles[t + 1].groupQIdx != segment.tiles[t].groupQIdx) {
                    AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(VEC2CUBE);
                }
                inputStage = 1 - inputStage;
            }
            stage = 1 - stage;
        }
        processedTasks += qCount;
        if (processedTasks < taskLength && nextReady) {
            current = nextGroup;
        }
    }
    softmaxOp.EndGroupedPipeline();
}

#endif // BSAG_KERNEL_GROUPED_FRAGMENT
