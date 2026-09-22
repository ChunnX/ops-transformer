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
 * \file block_sparse_attention_grad_kernel.h
 * \brief Block Sparse Attention Grad Kernel Implementation
 */

#ifndef BLOCK_SPARSE_ATTENTION_GRAD_KERNEL_H
#define BLOCK_SPARSE_ATTENTION_GRAD_KERNEL_H

#include "attn_infra/bsag_base_defs.hpp"
#include "attn_infra/arch/bsag_arch.hpp"
#include "attn_infra/arch/bsag_cross_core_sync.hpp"
#include "attn_infra/arch/bsag_resource.hpp"
#include "attn_infra/layout/bsag_layout.hpp"

#include "attn_infra/gemm/block/bsag_block_mmad.hpp"
#include "attn_infra/gemm/bsag_gemm_dispatch_policy.hpp"
#include "attn_infra/gemm/bsag_gemm_type.hpp"
#include "attn_infra/epilogue/block/bsag_block_epilogue.hpp"
#include "attn_infra/epilogue/bsag_epilogue_dispatch_policy.hpp"
#include "attn_infra/epilogue/block/block_epilogue_fag_pre.hpp"
#include "attn_infra/epilogue/block/block_epilogue_post.hpp"
#include "attn_infra/epilogue/block/block_epilogue_softmaxgrad.hpp"
#include "attn_infra/epilogue/block/block_epilogue_simply_softmax.hpp"

using namespace NpuArch;

namespace BSA {

#include "block_sparse_attention_grad_kernel_common.h"

template <class BlockMmadBSAG1_, class BlockMmadBSAG2_, class BlockMmadBSAG3_, class EpilogueFAGPre_,
          class EpilogueFAGSfmg_, class EpilogueFAGOp_, class EpilogueFAGPost_, uint32_t INPUT_LAYOUT>
class BlockSparseAttentionGradKernel {
public:
    using BlockMmadBSAG1 = BlockMmadBSAG1_;
    using BlockMmadBSAG2 = BlockMmadBSAG2_;
    using BlockMmadBSAG3 = BlockMmadBSAG3_;
    using EpilogueFAGPre = EpilogueFAGPre_;
    using EpilogueFAGSfmg = EpilogueFAGSfmg_;
    using EpilogueFAGOp = EpilogueFAGOp_;
    using EpilogueFAGPost = EpilogueFAGPost_;
    using PreParams = typename EpilogueFAGPre::Params;
    using PostParams = typename EpilogueFAGPost::Params;
    using SfmgParams = typename EpilogueFAGSfmg_::Params;
    using SfmParams = typename EpilogueFAGOp_::Params;
    using ArchTag = typename BlockMmadBSAG1_::ArchTag;
    using ElementInput = typename BlockMmadBSAG1::ElementA;

    using ElementA1 = typename BlockMmadBSAG1::ElementA;
    using LayoutA1 = typename BlockMmadBSAG1::LayoutA;
    using LayoutB1 = typename BlockMmadBSAG1::LayoutB;
    using LayoutC1 = typename BlockMmadBSAG1::LayoutC;

    using LayoutA2 = typename BlockMmadBSAG2::LayoutA;
    using LayoutB2 = typename BlockMmadBSAG2::LayoutB;
    using LayoutC2 = typename BlockMmadBSAG2::LayoutC;

    using LayoutA3 = typename BlockMmadBSAG3::LayoutA;
    using LayoutB3 = typename BlockMmadBSAG3::LayoutB;
    using LayoutC3 = typename BlockMmadBSAG3::LayoutC;

    struct Params {
        GM_ADDR dout;
        GM_ADDR q;
        GM_ADDR k;
        GM_ADDR v;
        GM_ADDR out;
        GM_ADDR softmaxLse;
        GM_ADDR blockSparseMask;
        GM_ADDR blockShape;
        GM_ADDR attentionMask;
        GM_ADDR actualQseqlen;
        GM_ADDR actualKvseqlen;
        GM_ADDR dq;
        GM_ADDR dk;
        GM_ADDR dv;
        GM_ADDR workspace;
        GM_ADDR tiling;

        __aicore__ inline Params() {}

        __aicore__ inline Params(GM_ADDR dout_, GM_ADDR q_, GM_ADDR k_, GM_ADDR v_, GM_ADDR out_, GM_ADDR softmaxLse_,
                                 GM_ADDR blockSparseMask_, GM_ADDR blockShape_, GM_ADDR attentionMask_,
                                 GM_ADDR actualQseqlen_, GM_ADDR actualKvseqlen_, GM_ADDR dq_, GM_ADDR dk_, GM_ADDR dv_,
                                 GM_ADDR workspace_, GM_ADDR tiling_data_)
            : dout(dout_),
              q(q_),
              k(k_),
              v(v_),
              out(out_),
              softmaxLse(softmaxLse_),
              blockSparseMask(blockSparseMask_),
              blockShape(blockShape_),
              attentionMask(attentionMask_),
              actualQseqlen(actualQseqlen_),
              actualKvseqlen(actualKvseqlen_),
              dq(dq_),
              dk(dk_),
              dv(dv_),
              workspace(workspace_),
              tiling(tiling_data_)
        {}
    };

    struct TaskInfo {
        uint32_t curBatchIdx;
        uint32_t curHeadIdx;
        uint32_t curQSeqIdx;
        uint32_t curQBlcokIdx;
        uint32_t curCalQSize;
        uint32_t curCalKVSize;
        uint32_t qSeqlen;
        uint32_t kvSeqlen;
        uint64_t qOffset;  // Q, dout, dq
        uint64_t kvOffset; // K, V, dk, dv
        uint64_t sOffset;  // workspace : S, P, dp, ds
        uint32_t groupQIdx;
        uint32_t groupKvIdx;
        uint32_t kvBasicIdx;
        // Explicit padding preserves the TaskInfo layout across AIC schedules.
        uint32_t kvBasicIdxSecond;
        uint32_t kvFirstSize;
        uint32_t kvSecondSize;
        uint32_t kvValidSize;
        uint64_t kvSecondOffset;
        uint32_t kvPartCount;
        bool firstQSegment;
    };

// Schedule fragments are included inside the kernel class.
#include "block_sparse_attention_grad_kernel_wide.h"
#include "block_sparse_attention_grad_kernel_grouped.h"

    // Per-head counts have shape [B, Nq, maxMaskBlocks, 2]: Q rows, then KV columns.
    __aicore__ inline uint32_t PerBlockPrefix(uint32_t batch, uint32_t head, uint32_t block, uint32_t component,
                                              uint32_t physical,
                                              const __gm__ BlockSparseAttentionGradTilingData *tilingData,
                                              uint32_t prefixOffset = 0) const
    {
        if (tilingData->hasPerBlockMask == 0 || validCounts == nullptr)
            return physical;
        const uint64_t stride = static_cast<uint64_t>(tilingData->maxMaskBlocks) * 2;
        const uint64_t offset = (static_cast<uint64_t>(batch) * tilingData->numHeads + head) * stride +
                                static_cast<uint64_t>(block) * 2 + component;
        // Counts are relative to the user block, not the current compute fragment.
        const int64_t remaining = static_cast<int64_t>(validCounts[offset]) - static_cast<int64_t>(prefixOffset);
        return remaining <= 0 ?
                   0 :
                   (static_cast<uint32_t>(remaining) < physical ? static_cast<uint32_t>(remaining) : physical);
    }

    __aicore__ inline uint32_t GetKvBlockLimit(const Params &params, const TaskInfo &info, uint32_t kvBlockCount,
                                               const __gm__ BlockSparseAttentionGradTilingData *tilingData) const
    {
        return info.curCalQSize == 0 ? 0 : kvBlockCount;
    }

    // Grouped tile offsets in the original (non-packed) GM inputs.
    __aicore__ inline uint64_t GroupedQHeadOffset(const TaskInfo &info,
                                                  const __gm__ BlockSparseAttentionGradTilingData *tilingData) const
    {
        return info.qOffset;
    }

    __aicore__ inline uint64_t GroupedKvHeadOffset(const TaskInfo &info,
                                                   const __gm__ BlockSparseAttentionGradTilingData *tilingData) const
    {
        return info.kvOffset;
    }

    __aicore__ inline bool ComputeBlockEnabled(const Params &params, uint64_t maskOffset) const
    {
        return ((__gm__ uint8_t *)params.blockSparseMask)[maskOffset] != 0;
    }

    // Load up to eight consecutive entries from the original byte mask.
    // This is only a wider scalar GM transaction: the bytes retain their
    // original representation and are never converted into a persistent
    // bit mask. Typed loads are issued only when naturally aligned and
    // wholly contained in the same physical mask row; unaligned prefixes
    // and row tails fall back to smaller accesses.
    __aicore__ inline uint64_t LoadMaskByteWindow(const Params &params, uint64_t absoluteOffset, uint32_t byteCount,
                                                  uint64_t physicalRowEnd) const
    {
        if (byteCount == 0) {
            return 0;
        }
        __gm__ uint8_t *mask = reinterpret_cast<__gm__ uint8_t *>(params.blockSparseMask);
        uint64_t values = 0;
        uint32_t copiedBytes = 0;
        while (copiedBytes < byteCount) {
            const uint64_t offset = absoluteOffset + copiedBytes;
            const uint32_t remaining = byteCount - copiedBytes;
            if ((offset & 7ULL) == 0 && remaining >= sizeof(uint64_t) && offset + sizeof(uint64_t) <= physicalRowEnd) {
                const uint64_t word = *reinterpret_cast<__gm__ uint64_t *>(mask + offset);
                values |= word << (copiedBytes * 8U);
                copiedBytes += sizeof(uint64_t);
            } else if ((offset & 3ULL) == 0 && remaining >= sizeof(uint32_t) &&
                       offset + sizeof(uint32_t) <= physicalRowEnd) {
                const uint32_t word = *reinterpret_cast<__gm__ uint32_t *>(mask + offset);
                values |= static_cast<uint64_t>(word) << (copiedBytes * 8U);
                copiedBytes += sizeof(uint32_t);
            } else {
                values |= static_cast<uint64_t>(mask[offset]) << (copiedBytes * 8U);
                ++copiedBytes;
            }
        }
        return values;
    }

    __aicore__ inline MaskLayoutInfo GetMaskLayout(const __gm__ BlockSparseAttentionGradTilingData *tilingData) const
    {
        const uint32_t qBlockCount = (tilingData->maxQSeqlen + tilingData->blockShapeX - 1) / tilingData->blockShapeX;
        const uint32_t kvBlockCount = (tilingData->maxKvSeqlen + tilingData->blockShapeY - 1) / tilingData->blockShapeY;
        MaskLayoutInfo layout;
        layout.kvBlockCount = kvBlockCount;
        layout.headStride = static_cast<uint64_t>(qBlockCount) * kvBlockCount;
        layout.batchStride = static_cast<uint64_t>(tilingData->numHeads) * layout.headStride;
        return layout;
    }

    __aicore__ inline uint64_t GetMaskRowOffset(const TaskInfo &qInfo, const MaskLayoutInfo &layout) const
    {
        return static_cast<uint64_t>(qInfo.curBatchIdx) * layout.batchStride +
               static_cast<uint64_t>(qInfo.curHeadIdx) * layout.headStride +
               static_cast<uint64_t>(qInfo.curQBlcokIdx) * layout.kvBlockCount;
    }

    // Scan the original byte mask directly.  Four adjacent uint8 values
    // are fetched together only as a scalar load transaction; no packed
    // or persistent bit mask is produced.  The zero-word fast path is
    // especially useful for highly sparse masks.
    __aicore__ inline void CollectEnabledMaskBlocks(const Params &params, uint64_t rowOffset, uint32_t limit,
                                                    uint32_t capacity, uint32_t &cursor, uint32_t *selected,
                                                    uint32_t &selectedCount) const
    {
        __gm__ uint8_t *mask = reinterpret_cast<__gm__ uint8_t *>(params.blockSparseMask);
        while (cursor < limit && selectedCount < capacity && ((rowOffset + cursor) & 3U) != 0) {
            if (mask[rowOffset + cursor] != 0) {
                selected[selectedCount++] = cursor;
            }
            ++cursor;
        }
        while (cursor + 4 <= limit && selectedCount < capacity) {
            const uint64_t offset = rowOffset + cursor;
            const uint32_t values = *reinterpret_cast<__gm__ uint32_t *>(mask + offset);
            if (values == 0) {
                cursor += 4;
                continue;
            }
#pragma unroll
            for (uint32_t lane = 0; lane < 4; ++lane) {
                if (((values >> (lane * 8)) & 0xffU) != 0) {
                    selected[selectedCount++] = cursor;
                }
                ++cursor;
                if (selectedCount == capacity) {
                    return;
                }
            }
        }
        while (cursor < limit && selectedCount < capacity) {
            if (mask[rowOffset + cursor] != 0) {
                selected[selectedCount++] = cursor;
            }
            ++cursor;
        }
    }

    __aicore__ inline void SetFlag()
    {
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID5);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID6);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID7);
    }

    __aicore__ inline void WaitFlag()
    {
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
        // Retire every token seeded by SetFlag(), including unused legacy slots.
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

    __aicore__ inline void DrainFinalCube1Flags() const
    {
        // No Cube1 operation follows the final grouped segment.  Retire
        // its L1 ping-pong tokens while the final gradient phase can
        // still hide the scalar event latency.  Both L0C tokens are left
        // for the gradient accumulator ping-pong to consume.
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
    }

    __aicore__ inline void WaitGroupedFlag() const
    {
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID5);
    }

    __aicore__ inline void UpdateTaskInfoCalQSize(uint32_t blockShapeX, uint32_t basicQBlockSize, TaskInfo &taskInfo)
    {
        taskInfo.curQBlcokIdx = taskInfo.curQSeqIdx / blockShapeX;
        uint32_t curQBlcokSeqIdx = taskInfo.curQBlcokIdx * blockShapeX;
        if (curQBlcokSeqIdx + blockShapeX <= taskInfo.qSeqlen) {
            // 没有跨Batch
            if (taskInfo.curQSeqIdx + basicQBlockSize <= curQBlcokSeqIdx + blockShapeX) {
                // 没有跨BasicBlock，处理一个完整的basicBlock
                taskInfo.curCalQSize = basicQBlockSize;
            } else {
                // 处理不完一整个basicBlock，被稀疏Block阻断了，处理长度为当前起始seq到稀疏块末尾
                taskInfo.curCalQSize = curQBlcokSeqIdx + blockShapeX - taskInfo.curQSeqIdx;
            }
        } else {
            // 跨Batch了，只能处理当前Batch内的一部分
            if (taskInfo.curQSeqIdx + basicQBlockSize <= taskInfo.qSeqlen) {
                // 没有跨BasicBlock，处理一个完整的basicBlock
                taskInfo.curCalQSize = basicQBlockSize;
            } else {
                // 处理不完一整个basicBlock，被Batch阻断了，处理长度为当前起始seq到Batch末尾
                taskInfo.curCalQSize = taskInfo.qSeqlen - taskInfo.curQSeqIdx;
            }
        }
    }

    __aicore__ inline void initTaskInfo(AscendC::GlobalTensor<int64_t> gActualQseqlen,
                                        AscendC::GlobalTensor<int64_t> gActualKvseqlen,
                                        __gm__ BlockSparseAttentionGradTilingData *tilingData, uint32_t numHeads,
                                        uint32_t kvHeads, uint32_t groupSize, uint32_t headDim, uint32_t maxQSeqlen,
                                        uint32_t maxKvSeqlen, uint32_t blockShapeX, uint32_t basicQBlockSize,
                                        uint32_t inputLayout, uint32_t coreIdx, TaskInfo &taskInfo)
    {
        // BNSD:curBatch * numHeads * maxQSeqlen + curQSeqOffset; TND:cusum(gActualQseqlen[0:curBatch-1]) +
        // curQSeqOffset
        uint32_t preQSeqLengths = tilingData->preQSeqLengths[coreIdx];
        // BNSD:curBatch * kvHeads * maxKvSeqlen; TND:cusum(gActualKvseqlen[0:curBatch-1])
        uint32_t preKVSeqLengths = tilingData->preKVSeqLengths[coreIdx];
        taskInfo.curBatchIdx = tilingData->beginBatch[coreIdx];
        taskInfo.curHeadIdx = tilingData->beginHead[coreIdx];
        taskInfo.curQSeqIdx = tilingData->beginQSeqOffset[coreIdx];
        if (inputLayout == 0) {
            taskInfo.qOffset = preQSeqLengths * numHeads * headDim + taskInfo.curHeadIdx * headDim;
            taskInfo.kvOffset = preKVSeqLengths * kvHeads * headDim + taskInfo.curHeadIdx / groupSize * headDim;
            taskInfo.qSeqlen =
                static_cast<uint32_t>(static_cast<int64_t>(gActualQseqlen.GetValue(taskInfo.curBatchIdx)));
            taskInfo.kvSeqlen =
                static_cast<uint32_t>(static_cast<int64_t>(gActualKvseqlen.GetValue(taskInfo.curBatchIdx)));
        } else {
            taskInfo.qOffset = preQSeqLengths * headDim;
            taskInfo.kvOffset = preKVSeqLengths * headDim;
            taskInfo.qSeqlen = maxQSeqlen;
            taskInfo.kvSeqlen = maxKvSeqlen;
        }
        UpdateTaskInfoCalQSize(blockShapeX, basicQBlockSize, taskInfo);
    }

    __aicore__ inline void updateNextTaskInfo(AscendC::GlobalTensor<int64_t> gActualQseqlen,
                                              AscendC::GlobalTensor<int64_t> gActualKvseqlen, uint32_t numHeads,
                                              uint32_t kvHeads, uint32_t groupSize, uint32_t headDim,
                                              uint32_t blockShapeX, uint32_t basicQBlockSize, uint32_t inputLayout,
                                              const TaskInfo &taskInfo, TaskInfo &nextTask)
    {
        nextTask = taskInfo;
        if (inputLayout == 0) { // TND, Traverse N-axis first
            if (taskInfo.curHeadIdx == numHeads - 1) {
                nextTask.qOffset = taskInfo.qOffset + (taskInfo.curCalQSize - 1) * numHeads * headDim + headDim;
                if (taskInfo.curQSeqIdx + taskInfo.curCalQSize == taskInfo.qSeqlen) {
                    nextTask.curBatchIdx = taskInfo.curBatchIdx + 1;
                    nextTask.curHeadIdx = 0;
                    nextTask.curQSeqIdx = 0;
                    nextTask.qSeqlen =
                        static_cast<uint32_t>(static_cast<int64_t>(gActualQseqlen.GetValue(nextTask.curBatchIdx)));
                    nextTask.kvSeqlen =
                        static_cast<uint32_t>(static_cast<int64_t>(gActualKvseqlen.GetValue(nextTask.curBatchIdx)));
                    nextTask.kvOffset = taskInfo.kvOffset + (taskInfo.kvSeqlen - 1) * kvHeads * headDim + headDim;
                } else { // batch 不变
                    nextTask.curHeadIdx = 0;
                    nextTask.curQSeqIdx = taskInfo.curQSeqIdx + taskInfo.curCalQSize;
                    nextTask.kvOffset = taskInfo.kvOffset - kvHeads * headDim + headDim;
                }
            } else { // batch/QSeqIdx 不变
                nextTask.qOffset = taskInfo.qOffset + headDim;
                nextTask.curHeadIdx = taskInfo.curHeadIdx + 1;
                if (nextTask.curHeadIdx % groupSize == 0) {
                    nextTask.kvOffset = taskInfo.kvOffset + headDim;
                }
            }
        } else { // BNSD, Traverse S-axis first
            nextTask.qOffset = taskInfo.qOffset + taskInfo.curCalQSize * headDim;
            if (taskInfo.curQSeqIdx + taskInfo.curCalQSize == taskInfo.qSeqlen) {
                nextTask.curQSeqIdx = 0;
                if (taskInfo.curHeadIdx == numHeads - 1) {
                    nextTask.curBatchIdx = taskInfo.curBatchIdx + 1;
                    nextTask.curHeadIdx = 0;
                    nextTask.kvOffset = taskInfo.kvOffset + taskInfo.kvSeqlen * headDim;
                } else {
                    nextTask.curHeadIdx = taskInfo.curHeadIdx + 1;
                    if (nextTask.curHeadIdx % groupSize == 0) {
                        nextTask.kvOffset = taskInfo.kvOffset + taskInfo.kvSeqlen * headDim;
                    }
                }
            } else {
                nextTask.curQSeqIdx = taskInfo.curQSeqIdx + taskInfo.curCalQSize;
            }
        }
        UpdateTaskInfoCalQSize(blockShapeX, basicQBlockSize, nextTask);
    }

    // Extended L1 slots are intentionally reused across the dQ, dV and
    // dK phases.  Drain the seeded event, establish a real MTE1->MTE2
    // dependency after all preceding L1 reads, then restore the token
    // consumed later by PreloadB(slot 3).
    __aicore__ inline void SyncExtendedL1Reuse() const
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID7);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID7);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID7);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID7);
    }

    __aicore__ inline TaskInfo MakeWideQTask(const TaskInfo &workInfo, const WideQInfo *qInfos, uint32_t qLocal) const
    {
        TaskInfo task = workInfo;
        task.qOffset = qInfos[qLocal].qOffset;
        task.curQSeqIdx = qInfos[qLocal].curQSeqIdx;
        task.curQBlcokIdx = qInfos[qLocal].curQBlockIdx;
        task.curCalQSize = qInfos[qLocal].curCalQSize;
        return task;
    }

    __aicore__ inline TaskInfo MakeWideTile(const TaskInfo &workInfo, const WideQInfo *qInfos,
                                            const WidePacketInfo &packet, uint32_t edgeIndex,
                                            const __gm__ BlockSparseAttentionGradTilingData *tilingData) const
    {
        const uint32_t edgeCode = packet.edgeCode[edgeIndex];
        const uint32_t qLocal = WideEdgeQ(edgeCode);
        const uint32_t kvLocal = WideEdgeKv(edgeCode);
        const uint32_t kvBasicIdx = packet.kv[kvLocal].kvBasicIdx;
        TaskInfo tile = MakeWideQTask(workInfo, qInfos, qLocal);
        const uint32_t kvTileCount = (tile.kvSeqlen + tilingData->basicKVBlockSize - 1) / tilingData->basicKVBlockSize;
        tile.curCalKVSize = kvBasicIdx + 1 < kvTileCount ? tilingData->basicKVBlockSize :
                                                           tile.kvSeqlen - kvBasicIdx * tilingData->basicKVBlockSize;
        tile.curCalKVSize =
            PerBlockPrefix(tile.curBatchIdx, tile.curHeadIdx, kvBasicIdx, 1, tile.curCalKVSize, tilingData);
        tile.kvOffset += static_cast<uint64_t>(kvBasicIdx) * tilingData->basicKVBlockSize * tilingData->headDim;
        tile.sOffset = packet.workspaceBase + static_cast<uint64_t>(edgeIndex) * GROUPED_TILE_ELEMENTS;
        tile.groupQIdx = qLocal;
        tile.groupKvIdx = kvLocal;
        tile.kvBasicIdx = kvBasicIdx;
        tile.kvBasicIdxSecond = 0;
        tile.kvFirstSize = tile.curCalKVSize;
        tile.kvSecondSize = 0;
        tile.kvValidSize = tile.curCalKVSize;
        tile.kvSecondOffset = 0;
        tile.kvPartCount = 1;
        tile.firstQSegment = false;
        return tile;
    }

    __aicore__ inline BlockSparseAttentionGradKernel() {}

    template <int32_t CORE_TYPE = g_coreType>
    __aicore__ inline void operator()(Params const &params);

    template <>
    __aicore__ inline void operator()<AscendC::AIC>(Params const &params)
    {
        auto *dispatchTiling = reinterpret_cast<__gm__ BlockSparseAttentionGradTilingData *>(params.tiling);
        validCounts =
            dispatchTiling->hasPerBlockMask ? reinterpret_cast<__gm__ int32_t *>(params.attentionMask) : nullptr;
        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t coreNum = AscendC::GetBlockNum();

        __gm__ BlockSparseAttentionGradTilingData *tilingData =
            reinterpret_cast<__gm__ BlockSparseAttentionGradTilingData *>(params.tiling);
        uint32_t batch = tilingData->batch;
        uint32_t numHeads = tilingData->numHeads;
        uint32_t kvHeads = tilingData->kvHeads;
        uint32_t groupSize = numHeads / kvHeads;
        uint32_t headDim = tilingData->headDim;
        uint32_t maxQSeqlen = tilingData->maxQSeqlen;
        uint32_t maxKvSeqlen = tilingData->maxKvSeqlen;
        uint32_t inputLayout = tilingData->inputLayout;
        uint32_t blockShapeX = tilingData->blockShapeX;
        uint32_t blockShapeY = tilingData->blockShapeY;

        uint64_t sOutSize = tilingData->sOutSize;
        uint64_t dPOutSize = tilingData->dPOutSize;
        uint64_t dQOutSize = tilingData->dQOutSize;
        uint64_t dKOutSize = tilingData->dKOutSize;
        uint64_t dVOutSize = tilingData->dVOutSize;

        uint32_t basicQBlockSize = tilingData->basicQBlockSize;
        uint32_t basicKVBlockSize = tilingData->basicKVBlockSize;
        uint32_t taskNumPerCore = tilingData->taskNumPerCore;
        uint32_t tailTaskNum = tilingData->tailTaskNum;
        uint32_t taskLength = tailTaskNum > coreIdx ? taskNumPerCore + 1 : taskNumPerCore;

        if (groupSize == 0 || blockShapeX == 0 || blockShapeY == 0) {
            return;
        }
        // Host tiling selects the schedule for both AIC and AIV.
        switch (tilingData->scheduleMode) {
            case 2: // wide
                ProcessWideAic(params, tilingData);
                return;
            case 1: // grouped
                ProcessGroupedAic(params, tilingData);
                return;
            default: // legacy
                break;
        }
        // Initialize global tensors
        AscendC::GlobalTensor<ElementA1> gDout;
        gDout.SetGlobalBuffer((__gm__ ElementInput *)params.dout);
        AscendC::GlobalTensor<ElementInput> gQ;
        gQ.SetGlobalBuffer((__gm__ ElementInput *)params.q);
        AscendC::GlobalTensor<ElementInput> gK;
        gK.SetGlobalBuffer((__gm__ ElementInput *)params.k);
        AscendC::GlobalTensor<ElementInput> gV;
        gV.SetGlobalBuffer((__gm__ ElementInput *)params.v);
        AscendC::GlobalTensor<int64_t> gActualQseqlen;
        gActualQseqlen.SetGlobalBuffer((__gm__ int64_t *)params.actualQseqlen);
        AscendC::GlobalTensor<int64_t> gActualKvseqlen;
        gActualKvseqlen.SetGlobalBuffer((__gm__ int64_t *)params.actualKvseqlen);

        AscendC::GlobalTensor<float> gS;
        gS.SetGlobalBuffer((__gm__ float *)params.workspace);
        AscendC::GlobalTensor<ElementInput> gP;
        gP.SetGlobalBuffer((__gm__ ElementInput *)(params.workspace + WORKSPACE_P16_OFFSET));

        AscendC::GlobalTensor<float> gDp;
        gDp.SetGlobalBuffer((__gm__ float *)(params.workspace + sOutSize));
        AscendC::GlobalTensor<ElementInput> gDs;
        gDs.SetGlobalBuffer((__gm__ ElementInput *)(params.workspace + sOutSize + WORKSPACE_P16_OFFSET));
        AscendC::GlobalTensor<float> gDq;
        gDq.SetGlobalBuffer((__gm__ float *)(params.workspace + sOutSize + dPOutSize));
        AscendC::GlobalTensor<float> gDk;
        gDk.SetGlobalBuffer((__gm__ float *)(params.workspace + sOutSize + dPOutSize + dQOutSize));
        AscendC::GlobalTensor<float> gDv;
        gDv.SetGlobalBuffer((__gm__ float *)(params.workspace + sOutSize + dPOutSize + dQOutSize + dKOutSize));
#if 0 // Deterministic reduction is disabled.
            uint8_t deterministic = tilingData->deterministic;
            uint64_t gradSize = tilingData->gradSize;
            uint64_t dkvElementSize = tilingData->dkvSize;
            // Deterministic path: per-core-per-Q-head workspace slots
            uint64_t detDkOffset = sOutSize + dPOutSize + dQOutSize + dKOutSize + dVOutSize + gradSize;
            uint64_t detDvOffset = detDkOffset + tilingData->detDkWorkspaceSize;
            uint32_t groupSizeDet = tilingData->groupSizeForDet;
            uint64_t perCoreDetSize = (uint64_t)groupSizeDet * dkvElementSize * sizeof(float);
            uint64_t detDkBaseCore = detDkOffset + (uint64_t)coreIdx * perCoreDetSize;
            uint64_t detDvBaseCore = detDvOffset + (uint64_t)coreIdx * perCoreDetSize;
#endif

        TaskInfo taskInfo[2];
        TaskInfo preTaskInfo;
        initTaskInfo(gActualQseqlen, gActualKvseqlen, tilingData, numHeads, kvHeads, groupSize, headDim, maxQSeqlen,
                     maxKvSeqlen, blockShapeX, basicQBlockSize, inputLayout, coreIdx, taskInfo[0]);
        uint32_t maskQBlockNum = (maxQSeqlen + blockShapeX - 1) / blockShapeX;
        uint32_t maskKvBlockNum = (maxKvSeqlen + blockShapeY - 1) / blockShapeY;
        uint32_t batchBlocks = numHeads * maskQBlockNum * maskKvBlockNum;
        uint32_t headBlocks = maskQBlockNum * maskKvBlockNum;

        uint64_t actualStrideQ = headDim;
        uint64_t actualStrideKV = headDim;
        if (inputLayout == 0) {
            actualStrideQ = numHeads * headDim;
            actualStrideKV = kvHeads * headDim;
        }

        BlockMmadBSAG1 blockMmad1(resource);
        BlockMmadBSAG2 blockMmad2(resource, L1_SIZE_OFFSET, PINGPONG_OFFSET_2, true);
        BlockMmadBSAG3 blockMmad3(resource, L1_SIZE_OFFSET * 2, PINGPONG_OFFSET_4, true);
        uint32_t count = 0;
        uint32_t pingpongFlag = 0;
        uint64_t gSOffset = coreIdx * WORKSPACE_BLOCK_SIZE_DB;
        uint32_t mmadFlag = 0;

        AscendC::SyncAll<false>();

        SetFlag();
        uint32_t vec2CubeFlag = 0;
        for (uint32_t i = 0; i < taskLength; i++) {
            TaskInfo curInfo = taskInfo[i % 2];
            curInfo.curCalQSize = PerBlockPrefix(curInfo.curBatchIdx, curInfo.curHeadIdx, curInfo.curQBlcokIdx, 0,
                                                 curInfo.curCalQSize, tilingData, curInfo.curQSeqIdx % blockShapeX);
            LayoutA1 cachedLayoutA1;
            LayoutB3 cachedLayoutB3;
            if (inputLayout == 0) {
                cachedLayoutA1 = LayoutA1(curInfo.curCalQSize, headDim, numHeads * headDim);
                cachedLayoutB3 = LayoutB3(curInfo.curCalQSize, headDim, numHeads * headDim);
            } else {
                cachedLayoutA1 = LayoutA1(curInfo.curCalQSize, headDim);
                cachedLayoutB3 = LayoutB3(curInfo.curCalQSize, headDim);
            }
            if (curInfo.curCalQSize != 0) {
                const GemmCoord cachedShape{curInfo.curCalQSize, headDim, headDim};
                blockMmad1.PreloadA(gQ[curInfo.qOffset], cachedLayoutA1, cachedShape, 0);
                blockMmad1.PreloadA(gDout[curInfo.qOffset], cachedLayoutA1, cachedShape, 1);
            }
            uint64_t kvBlockOffset = 0;
            uint64_t beginKVOffset = curInfo.kvOffset;
            uint32_t curKvBlockNum = (curInfo.kvSeqlen + blockShapeY - 1) / blockShapeY;
            const uint32_t kvBlockLimit = GetKvBlockLimit(params, curInfo, curKvBlockNum, tilingData);
            for (uint32_t idx = 0; idx < kvBlockLimit; idx++) {
                // BlcokSpaseMask shape : [batch, numhead, CeilDiv(maxQSeqlen, blockShapeX), CeilDiv(maxKvSeqlen,
                // blockShapeY)]
                uint64_t maskOffset = curInfo.curBatchIdx * batchBlocks + curInfo.curHeadIdx * headBlocks +
                                      curInfo.curQBlcokIdx * maskKvBlockNum + idx;
                if (ComputeBlockEnabled(params, maskOffset)) {
                    uint64_t kvBlockBasicOffset = 0;
                    uint32_t kvBlockSize =
                        (idx != curKvBlockNum - 1) ? blockShapeY : curInfo.kvSeqlen - blockShapeY * idx;
                    // AIC and AIV must apply identical count clipping and zero-block skips
                    // to keep their cross-core handshake counts equal.
                    kvBlockSize =
                        PerBlockPrefix(curInfo.curBatchIdx, curInfo.curHeadIdx, idx, 1, kvBlockSize, tilingData);
                    if (kvBlockSize == 0) {
                        kvBlockOffset += blockShapeY;
                        continue;
                    }
                    vec2CubeFlag++;
                    uint32_t kvLoop = (kvBlockSize + basicKVBlockSize - 1) / basicKVBlockSize;
                    for (uint32_t loop = 0; loop < kvLoop; loop++) {
                        curInfo.curCalKVSize =
                            (loop != kvLoop - 1) ? basicKVBlockSize : kvBlockSize - basicKVBlockSize * loop;
                        if (inputLayout == 0) {
                            curInfo.kvOffset = beginKVOffset + (kvBlockOffset + kvBlockBasicOffset) * kvHeads * headDim;
                        } else {
                            curInfo.kvOffset = beginKVOffset + (kvBlockOffset + kvBlockBasicOffset) * headDim;
                        }
                        curInfo.sOffset = gSOffset + WORKSPACE_BLOCK_SIZE * pingpongFlag;
                        LayoutA1 layoutA1;
                        LayoutB1 layoutB1;
                        LayoutC1 layoutC1(curInfo.curCalQSize, curInfo.curCalKVSize);
                        if (inputLayout == 0) {
                            layoutA1 = LayoutA1(curInfo.curCalQSize, headDim, numHeads * headDim);
                            layoutB1 = LayoutB1(headDim, curInfo.curCalKVSize, kvHeads * headDim);
                        } else {
                            layoutA1 = LayoutA1(curInfo.curCalQSize, headDim);
                            layoutB1 = LayoutB1(headDim, curInfo.curCalKVSize);
                        }
                        GemmCoord actualShape1{curInfo.curCalQSize, curInfo.curCalKVSize, headDim};
                        blockMmad1.WithCachedA(0, gK[curInfo.kvOffset], gS[curInfo.sOffset], layoutA1, layoutB1,
                                               layoutC1, actualShape1, mmadFlag);

                        blockMmad1.WithCachedA(1, gV[curInfo.kvOffset], gDp[curInfo.sOffset], layoutA1, layoutB1,
                                               layoutC1, actualShape1, mmadFlag);
                        AscendC::CrossCoreSetFlag<2, PIPE_FIX>(CUBE2VEC);
                        if (count > 0) {
                            AscendC::WaitEvent(VEC2CUBE);

                            LayoutA2 layoutA2(preTaskInfo.curCalQSize, preTaskInfo.curCalKVSize);
                            LayoutB2 layoutB2;
                            LayoutC2 layoutC2;

                            LayoutA3 layoutA3(preTaskInfo.curCalKVSize, preTaskInfo.curCalQSize);
                            LayoutB3 layoutB3;
                            LayoutC3 layoutC3;
                            if (inputLayout == 0) {
                                layoutB2 = LayoutB2(preTaskInfo.curCalKVSize, headDim, kvHeads * headDim);
                                layoutC2 = LayoutC2(preTaskInfo.curCalQSize, headDim, numHeads * headDim);
                                layoutB3 = LayoutB3(preTaskInfo.curCalQSize, headDim, numHeads * headDim);
                                layoutC3 = LayoutC3(preTaskInfo.curCalKVSize, headDim, kvHeads * headDim);
                            } else {
                                layoutB2 = LayoutB2(preTaskInfo.curCalKVSize, headDim);
                                layoutC2 = LayoutC2(preTaskInfo.curCalQSize, headDim);
                                layoutB3 = LayoutB3(preTaskInfo.curCalQSize, headDim);
                                layoutC3 = LayoutC3(preTaskInfo.curCalKVSize, headDim);
                            }
                            GemmCoord actualShape2{preTaskInfo.curCalQSize, headDim, preTaskInfo.curCalKVSize};
                            GemmCoord actualShape3{preTaskInfo.curCalKVSize, headDim, preTaskInfo.curCalQSize};

                            blockMmad2(gDs[preTaskInfo.sOffset], gK[preTaskInfo.kvOffset], gDq[preTaskInfo.qOffset],
                                       layoutA2, layoutB2, layoutC2, actualShape2, mmadFlag);

#if 0 // Deterministic reduction is disabled.
                                if (deterministic)
                                    gDv.SetGlobalBuffer((__gm__ float *)(params.workspace + detDvBaseCore +
                                                                         (uint64_t)(preTaskInfo.curHeadIdx % groupSizeDet) *
                                                                             dkvElementSize * sizeof(float)));
#endif
                            blockMmad3.WithCachedB(0, gP[preTaskInfo.sOffset], gDv[preTaskInfo.kvOffset], layoutA3,
                                                   layoutB3, layoutC3, actualShape3, mmadFlag);

#if 0 // Deterministic reduction is disabled.
                                if (deterministic)
                                    gDk.SetGlobalBuffer((__gm__ float *)(params.workspace + detDkBaseCore +
                                                                         (uint64_t)(preTaskInfo.curHeadIdx % groupSizeDet) *
                                                                             dkvElementSize * sizeof(float)));
#endif
                            blockMmad3.WithCachedB(1, gDs[preTaskInfo.sOffset], gDk[preTaskInfo.kvOffset], layoutA3,
                                                   layoutB3, layoutC3, actualShape3, mmadFlag);
                        }
                        if (count == 0 || preTaskInfo.qOffset != curInfo.qOffset) {
                            const GemmCoord cachedShapeB{headDim, headDim, curInfo.curCalQSize};
                            blockMmad3.PreloadB(gDout[curInfo.qOffset], cachedLayoutB3, cachedShapeB, 0);
                            blockMmad3.PreloadB(gQ[curInfo.qOffset], cachedLayoutB3, cachedShapeB, 1);
                        }
                        preTaskInfo = curInfo;
                        preTaskInfo.sOffset = curInfo.sOffset * 2; // float32偏移转成bf16/half偏移
                        pingpongFlag = 1 - pingpongFlag;
                        count++;
                        kvBlockBasicOffset += basicKVBlockSize;
                    }
                }
                kvBlockOffset += blockShapeY;
            }
            if (i != taskLength - 1) {
                updateNextTaskInfo(gActualQseqlen, gActualKvseqlen, numHeads, kvHeads, groupSize, headDim, blockShapeX,
                                   basicQBlockSize, inputLayout, taskInfo[i % 2], taskInfo[(i + 1) % 2]);
            }
        }
        if (vec2CubeFlag != 0) {
            AscendC::WaitEvent(VEC2CUBE);
            LayoutA2 layoutA2(preTaskInfo.curCalQSize, preTaskInfo.curCalKVSize);
            LayoutB2 layoutB2;
            LayoutC2 layoutC2;

            LayoutA3 layoutA3(preTaskInfo.curCalKVSize, preTaskInfo.curCalQSize);
            LayoutB3 layoutB3;
            LayoutC3 layoutC3;
            if (inputLayout == 0) {
                layoutB2 = LayoutB2(preTaskInfo.curCalKVSize, headDim, kvHeads * headDim);
                layoutC2 = LayoutC2(preTaskInfo.curCalQSize, headDim, numHeads * headDim);
                layoutB3 = LayoutB3(preTaskInfo.curCalQSize, headDim, numHeads * headDim);
                layoutC3 = LayoutC3(preTaskInfo.curCalKVSize, headDim, kvHeads * headDim);
            } else {
                layoutB2 = LayoutB2(preTaskInfo.curCalKVSize, headDim);
                layoutC2 = LayoutC2(preTaskInfo.curCalQSize, headDim);
                layoutB3 = LayoutB3(preTaskInfo.curCalQSize, headDim);
                layoutC3 = LayoutC3(preTaskInfo.curCalKVSize, headDim);
            }
            GemmCoord actualShape2{preTaskInfo.curCalQSize, headDim, preTaskInfo.curCalKVSize};
            GemmCoord actualShape3{preTaskInfo.curCalKVSize, headDim, preTaskInfo.curCalQSize};

            blockMmad2(gDs[preTaskInfo.sOffset], gK[preTaskInfo.kvOffset], gDq[preTaskInfo.qOffset], layoutA2, layoutB2,
                       layoutC2, actualShape2, mmadFlag);
#if 0 // Deterministic reduction is disabled.
                if (deterministic)
                    gDv.SetGlobalBuffer((__gm__ float *)(params.workspace + detDvBaseCore +
                                                         (uint64_t)(preTaskInfo.curHeadIdx % groupSizeDet) *
                                                             dkvElementSize * sizeof(float)));
#endif
            blockMmad3.WithCachedB(0, gP[preTaskInfo.sOffset], gDv[preTaskInfo.kvOffset], layoutA3, layoutB3, layoutC3,
                                   actualShape3, mmadFlag);
#if 0 // Deterministic reduction is disabled.
                if (deterministic)
                    gDk.SetGlobalBuffer((__gm__ float *)(params.workspace + detDkBaseCore +
                                                         (uint64_t)(preTaskInfo.curHeadIdx % groupSizeDet) *
                                                             dkvElementSize * sizeof(float)));
#endif
            blockMmad3.WithCachedB(1, gDs[preTaskInfo.sOffset], gDk[preTaskInfo.kvOffset], layoutA3, layoutB3, layoutC3,
                                   actualShape3, mmadFlag);
        }
        WaitFlag();
        AscendC::CrossCoreSetFlag<2, PIPE_FIX>(CUBE2POST);
    }

    template <>
    __aicore__ inline void operator()<AscendC::AIV>(Params const &params)
    {
        // Use the same per-head counts as AIC to preserve packet and handshake ordering.
        validCounts = reinterpret_cast<__gm__ BlockSparseAttentionGradTilingData *>(params.tiling)->hasPerBlockMask ?
                          reinterpret_cast<__gm__ int32_t *>(params.attentionMask) :
                          nullptr;
        __gm__ BlockSparseAttentionGradTilingData *tilingData =
            reinterpret_cast<__gm__ BlockSparseAttentionGradTilingData *>(params.tiling);
        // InitOutput obtains its temporary UB stack and event IDs from
        // the active TPipe.  Keep this pre-stage pipe alive through the
        // mixed AIC/AIV barrier, exactly as FlashAttentionScoreGradPre.
        AscendC::TPipe pipePre;
        // AIVs clear disjoint FP32 gradient-workspace slices in VecPre.
        // Every AIC path reaches the matching SyncAll below before it can
        // start accumulation.
        VecPre(params);
        // Packed-fragment padding uses grouped-only workspace offsets.
        // Applying them to another schedule would write outside its workspace.
        if (tilingData->scheduleMode == 1) {
            ClearGroupedPadding(params, tilingData);
        }
        AscendC::SyncAll<false>();
        pipePre.Destroy();

        switch (tilingData->scheduleMode) {
            case 2: // wide
                ProcessWideAiv(params, tilingData);
                break;
            case 1: // grouped
                ProcessGroupedAiv(params, tilingData);
                break;
            default: // legacy
                VecOp(params);
                break;
        }

        // Each AIC publishes completion to its two paired AIVs.  The AIV
        // barrier turns those pair-local notifications into a global
        // gradient-workspace completion barrier before output finalize.
        AscendC::WaitEvent(CUBE2POST);
        AscendC::SyncAll<true>();
#if 0 // Deterministic reduction is disabled.
            if (tilingData->deterministic) {
                VecDeterReduce(params);
            }
#endif
        VecPost(params);
    }

    __aicore__ inline void VecOp(Params const &params)
    {
        uint32_t vecCoreIdx = AscendC::GetBlockIdx(); // vecore 核数idx
        uint32_t coreIdx = vecCoreIdx / 2;            // cube 核数idx
        uint32_t subBlockIdx = vecCoreIdx % 2;

        __gm__ BlockSparseAttentionGradTilingData *tilingData =
            reinterpret_cast<__gm__ BlockSparseAttentionGradTilingData *>(params.tiling);
        uint32_t batch = tilingData->batch;
        uint32_t numHeads = tilingData->numHeads;
        uint32_t kvHeads = tilingData->kvHeads;
        uint32_t groupSize = numHeads / kvHeads;
        uint32_t inputLayout = tilingData->inputLayout;
        uint32_t blockShapeX = tilingData->blockShapeX;
        uint32_t blockShapeY = tilingData->blockShapeY;
        uint32_t headDim = tilingData->headDim;
        uint32_t maxQSeqlen = tilingData->maxQSeqlen;
        uint32_t maxKvSeqlen = tilingData->maxKvSeqlen;

        uint32_t basicQBlockSize = tilingData->basicQBlockSize;
        uint32_t basicKVBlockSize = tilingData->basicKVBlockSize;
        uint32_t taskNumPerCore = tilingData->taskNumPerCore;
        uint32_t tailTaskNum = tilingData->tailTaskNum;
        uint32_t taskLengthVec = tailTaskNum > coreIdx ? taskNumPerCore + 1 : taskNumPerCore;

        uint64_t sOutSize = tilingData->sOutSize;
        uint64_t dPOutSize = tilingData->dPOutSize;
        uint64_t dQOutSize = tilingData->dQOutSize;
        uint64_t dKOutSize = tilingData->dKOutSize;
        uint64_t dVOutSize = tilingData->dVOutSize;

        // Initialize global tensors
        AscendC::GlobalTensor<int64_t> gActualQseqlen;
        gActualQseqlen.SetGlobalBuffer((__gm__ int64_t *)params.actualQseqlen);
        AscendC::GlobalTensor<int64_t> gActualKvseqlen;
        gActualKvseqlen.SetGlobalBuffer((__gm__ int64_t *)params.actualKvseqlen);

        TaskInfo taskInfoVec[2]; // 索引以及shape信息
        TaskInfo preTaskInfo;
        initTaskInfo(gActualQseqlen, gActualKvseqlen, tilingData, numHeads, kvHeads, groupSize, headDim, maxQSeqlen,
                     maxKvSeqlen, blockShapeX, basicQBlockSize, inputLayout, coreIdx, taskInfoVec[0]);
        uint32_t maskQBlockNum = (maxQSeqlen + blockShapeX - 1) / blockShapeX;
        uint32_t maskKvBlockNum = (maxKvSeqlen + blockShapeY - 1) / blockShapeY;
        uint32_t batchBlocks = numHeads * maskQBlockNum * maskKvBlockNum;
        uint32_t headBlocks = maskQBlockNum * maskKvBlockNum;

        uint32_t pingpongFlag = 0;
        uint64_t gSOffset = coreIdx * WORKSPACE_BLOCK_SIZE_DB;

        GM_ADDR sftmgGm = params.workspace + sOutSize + dPOutSize + dQOutSize + dKOutSize + dVOutSize;
        SfmgParams SfmgParams(params.dout, params.out, params.actualQseqlen, sftmgGm, params.tiling);
        EpilogueFAGSfmg vecSftmg(SfmgParams);
        EpilogueFAGOp sStmOp;

        for (uint32_t i = 0; i < taskLengthVec; i++) {
            TaskInfo curInfo = taskInfoVec[i % 2];
            // Match AIC's Q clipping so both sides emit the same block handshakes.
            curInfo.curCalQSize = PerBlockPrefix(curInfo.curBatchIdx, curInfo.curHeadIdx, curInfo.curQBlcokIdx, 0,
                                                 curInfo.curCalQSize, tilingData, curInfo.curQSeqIdx % blockShapeX);
            uint64_t beginKVOffset = curInfo.kvOffset;

            // Split the row-wise SoftmaxGradFront between the two AIVs.
            const uint32_t firstHalfRows = (curInfo.curCalQSize + 1) / 2;
            const uint32_t subRows = subBlockIdx == 0 ? firstHalfRows : curInfo.curCalQSize - firstHalfRows;
            if (inputLayout == 0) {
                vecSftmg(curInfo.qOffset / headDim + subBlockIdx * firstHalfRows * numHeads, subRows);
            } else {
                vecSftmg(curInfo.qOffset / headDim + subBlockIdx * firstHalfRows, subRows);
            }

            uint32_t curKvBlockNum = (curInfo.kvSeqlen + blockShapeY - 1) / blockShapeY;
            const uint32_t kvBlockLimit = GetKvBlockLimit(params, curInfo, curKvBlockNum, tilingData);
            for (uint32_t idx = 0; idx < kvBlockLimit; idx++) {
                // BlcokSpaseMask shape : [batch, numhead, CeilDiv(maxQSeqlen, blockShapeX), CeilDiv(maxKvSeqlen,
                // blockShapeY)]
                uint64_t maskOffset = curInfo.curBatchIdx * batchBlocks + curInfo.curHeadIdx * headBlocks +
                                      curInfo.curQBlcokIdx * maskKvBlockNum + idx;
                if (ComputeBlockEnabled(params, maskOffset)) {
                    uint32_t kvBlockSize =
                        (idx != curKvBlockNum - 1) ? blockShapeY : curInfo.kvSeqlen - blockShapeY * idx;
                    // Match AIC's KV clipping and zero-count skips.
                    kvBlockSize =
                        PerBlockPrefix(curInfo.curBatchIdx, curInfo.curHeadIdx, idx, 1, kvBlockSize, tilingData);
                    if (kvBlockSize == 0) {
                        continue;
                    }
                    uint32_t kvLoop = (kvBlockSize + basicKVBlockSize - 1) / basicKVBlockSize;
                    for (uint32_t loop = 0; loop < kvLoop; loop++) {
                        curInfo.curCalKVSize =
                            (loop != kvLoop - 1) ? basicKVBlockSize : kvBlockSize - basicKVBlockSize * loop;
                        curInfo.sOffset = gSOffset + WORKSPACE_BLOCK_SIZE * pingpongFlag;

                        uint64_t actualRow = curInfo.curCalQSize;
                        uint64_t actualCol = curInfo.curCalKVSize;
                        uint64_t processNums = curInfo.curCalQSize * curInfo.curCalKVSize;
                        uint64_t curCoreBatch = curInfo.curBatchIdx;
                        uint64_t curCoreN1Idx = curInfo.curHeadIdx; // 当前q的N1的idx
                        uint64_t curCoreS1Idx = curInfo.curQSeqIdx; // 当前q的s1的idx
                        uint64_t curT1Idx = 0;
                        uint64_t sOutSize = tilingData->sOutSize;
                        uint64_t dPOutSize = tilingData->dPOutSize;
                        uint64_t dQOutSize = tilingData->dQOutSize;
                        uint64_t dKOutSize = tilingData->dKOutSize;
                        uint64_t dVOutSize = tilingData->dVOutSize;

                        const uint64_t firstHalfRows = (actualRow + 1) / 2;
                        uint64_t executeRow = subBlockIdx == 0 ? firstHalfRows : actualRow - firstHalfRows;
                        uint64_t coreOffset = subBlockIdx * firstHalfRows * actualCol;
                        uint64_t curVecCoreS1Idx = curCoreS1Idx + subBlockIdx * firstHalfRows;
                        uint64_t vector16Soffset =
                            (curInfo.sOffset * 2 + WORKSPACE_P16_OFFSET_ELEMENT + coreOffset) * sizeof(ElementInput);
                        uint64_t vector32Soffset = (curInfo.sOffset + coreOffset) * sizeof(float);

                        GM_ADDR s = params.workspace + vector32Soffset;
                        GM_ADDR softmaxLse = params.softmaxLse;
                        GM_ADDR dp = params.workspace + sOutSize + vector32Soffset;
                        GM_ADDR actualSeqQlen = params.actualQseqlen;
                        GM_ADDR actualSeqKvlen = params.actualKvseqlen;
                        GM_ADDR sftmgGm = params.workspace + sOutSize + dPOutSize + dQOutSize + dKOutSize + dVOutSize;
                        GM_ADDR pWorkspace = params.workspace + vector16Soffset;             // 连续
                        GM_ADDR dsWorkspace = params.workspace + sOutSize + vector16Soffset; // 连续
                        GM_ADDR tiling = params.tiling;

                        AscendC::WaitEvent(CUBE2VEC);

                        SfmParams sfmParams(s, softmaxLse, dp, actualSeqQlen, actualSeqKvlen, sftmgGm, pWorkspace,
                                            dsWorkspace, tiling, executeRow, actualCol, executeRow * actualCol,
                                            curCoreBatch, curCoreN1Idx, curVecCoreS1Idx, curT1Idx);
                        sStmOp(sfmParams);
                        // Mode 2 aggregates the completion events from the
                        // two AIV sub-cores paired with this AIC.  Do not
                        // use a global AIV barrier here: cores may own a
                        // different number of tasks.
                        AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(VEC2CUBE);

                        preTaskInfo = curInfo;
                        pingpongFlag = 1 - pingpongFlag;
                        preTaskInfo.sOffset = curInfo.sOffset * 2; // float32偏移转成bf16/half偏移
                    }
                }
            }

            if (i != taskLengthVec - 1) {
                updateNextTaskInfo(gActualQseqlen, gActualKvseqlen, numHeads, kvHeads, groupSize, headDim, blockShapeX,
                                   basicQBlockSize, inputLayout, taskInfoVec[i % 2], taskInfoVec[(i + 1) % 2]);
            }
        }
    }

    __aicore__ inline void VecPost(Params const &params)
    {
        __gm__ BlockSparseAttentionGradTilingData *tilingData =
            reinterpret_cast<__gm__ BlockSparseAttentionGradTilingData *>(params.tiling);

        uint64_t sOutSize = tilingData->sOutSize;
        uint64_t dPOutSize = tilingData->dPOutSize;
        uint64_t dQOutSize = tilingData->dQOutSize;
        uint64_t dKOutSize = tilingData->dKOutSize;
        uint64_t dVOutSize = tilingData->dVOutSize;

        GM_ADDR gDqGm = params.workspace + sOutSize + dPOutSize;
        GM_ADDR gDkGm = params.workspace + sOutSize + dPOutSize + dQOutSize;
        GM_ADDR gDvGm = params.workspace + sOutSize + dPOutSize + dQOutSize + dKOutSize;
        GM_ADDR actualSeqQlen = params.actualQseqlen;
        GM_ADDR actualSeqKvlen = params.actualKvseqlen;

        PostParams postParams(params.dq, params.dk, params.dv, gDqGm, gDkGm, gDvGm, params.tiling, actualSeqQlen,
                              actualSeqKvlen);
        EpilogueFAGPost vecPost(postParams);
        vecPost();
    }

    __aicore__ inline void VecPre(Params const &params)
    {
        __gm__ BlockSparseAttentionGradTilingData *tilingData =
            reinterpret_cast<__gm__ BlockSparseAttentionGradTilingData *>(params.tiling);

        uint64_t sOutSize = tilingData->sOutSize;
        uint64_t dPOutSize = tilingData->dPOutSize;
        uint64_t dQOutSize = tilingData->dQOutSize;
        uint64_t dKOutSize = tilingData->dKOutSize;
        uint64_t dVOutSize = tilingData->dVOutSize;

        GM_ADDR gDqWrkGm = params.workspace + sOutSize + dPOutSize;
        GM_ADDR gDkWrkGm = params.workspace + sOutSize + dPOutSize + dQOutSize;
        GM_ADDR gDvWrkGm = params.workspace + sOutSize + dPOutSize + dQOutSize + dKOutSize;

        PreParams preParms(gDqWrkGm, gDkWrkGm, gDvWrkGm, params.tiling);
        EpilogueFAGPre vecPre(preParms);
        vecPre();
#if 0 // Deterministic reduction is disabled.
            if (tilingData->deterministic) {
                ZeroDetWorkspace(params);
            }
#endif
    }

#if 0  // Deterministic reduction is disabled.
        __aicore__ inline void ZeroDetWorkspace(Params const &params)
        {
            __gm__ BlockSparseAttentionGradTilingData *tilingData =
                reinterpret_cast<__gm__ BlockSparseAttentionGradTilingData *>(params.tiling);
            uint32_t vecCoreIdx = AscendC::GetBlockIdx();
            uint32_t usedCoreNum = tilingData->usedVecCoreNum;
            if (vecCoreIdx >= usedCoreNum) {
                return;
            }

            uint32_t aicNum = tilingData->aicNumForDet;
            uint32_t groupSizeDet = tilingData->groupSizeForDet;
            uint64_t dkvSize = tilingData->dkvSize;
            uint64_t sOutSize = tilingData->sOutSize;
            uint64_t dPOutSize = tilingData->dPOutSize;
            uint64_t dQOutSize = tilingData->dQOutSize;
            uint64_t dKOutSize = tilingData->dKOutSize;
            uint64_t dVOutSize = tilingData->dVOutSize;
            uint64_t gradSize = tilingData->gradSize;

            uint64_t detDkOffset = sOutSize + dPOutSize + dQOutSize + dKOutSize + dVOutSize + gradSize;
            uint64_t detDvOffset = detDkOffset + tilingData->detDkWorkspaceSize;
            uint64_t detTotalElements = (uint64_t)aicNum * groupSizeDet * dkvSize;

            // Partition among AIV cores
            uint64_t blockFactor = (detTotalElements + usedCoreNum - 1) / usedCoreNum;
            uint64_t totalBlocks = (detTotalElements + blockFactor - 1) / blockFactor;
            uint64_t tailNumTmp = detTotalElements % blockFactor;
            uint64_t tailNum = (tailNumTmp == 0) ? blockFactor : tailNumTmp;

            if (vecCoreIdx >= totalBlocks) {
                return;
            }
            uint64_t initSize = (vecCoreIdx == totalBlocks - 1) ? tailNum : blockFactor;
            uint64_t offset = vecCoreIdx * blockFactor;

            uint64_t ubSizeAvail = tilingData->ubSize;
            uint64_t maxDataCount = ubSizeAvail / sizeof(float);
            maxDataCount = maxDataCount / 8 * 8;

            LocalTensor<float> zeroTensor = resource.ubBuf.template GetBufferByByte<float>(0);
            Duplicate(zeroTensor, (float)0.0, maxDataCount);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);

            // Zero dk workspace
            AscendC::GlobalTensor<float> detDkGm;
            detDkGm.SetGlobalBuffer((__gm__ float *)(params.workspace + detDkOffset));
            uint64_t cOutElement = initSize;
            uint64_t totalLoop = cOutElement / maxDataCount;
            uint64_t remainOutNum = cOutElement % maxDataCount;
            for (uint64_t i = 0; i < totalLoop; i++) {
                AscendC::DataCopy(detDkGm[offset + i * maxDataCount], zeroTensor, maxDataCount);
            }
            if (remainOutNum > 0) {
                if (remainOutNum * sizeof(float) % 32 == 0) {
                    AscendC::DataCopy(detDkGm[offset + totalLoop * maxDataCount], zeroTensor, remainOutNum);
                } else {
                    AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(remainOutNum * sizeof(float)), 0, 0, 0};
                    AscendC::DataCopyPad(detDkGm[offset + totalLoop * maxDataCount], zeroTensor, copyParams);
                }
            }

            // Zero dv workspace
            AscendC::GlobalTensor<float> detDvGm;
            detDvGm.SetGlobalBuffer((__gm__ float *)(params.workspace + detDvOffset));
            for (uint64_t i = 0; i < totalLoop; i++) {
                AscendC::DataCopy(detDvGm[offset + i * maxDataCount], zeroTensor, maxDataCount);
            }
            if (remainOutNum > 0) {
                if (remainOutNum * sizeof(float) % 32 == 0) {
                    AscendC::DataCopy(detDvGm[offset + totalLoop * maxDataCount], zeroTensor, remainOutNum);
                } else {
                    AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(remainOutNum * sizeof(float)), 0, 0, 0};
                    AscendC::DataCopyPad(detDvGm[offset + totalLoop * maxDataCount], zeroTensor, copyParams);
                }
            }

            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        }

        __aicore__ inline void VecDeterReduceKernel(AscendC::GlobalTensor<float> &detGm,
                                                    AscendC::GlobalTensor<float> &sharedGm, LocalTensor<float> &accumUb,
                                                    LocalTensor<float> &tempUb, uint32_t aicNum, uint64_t dkvSize,
                                                    uint64_t offset, uint64_t totalElements, uint64_t maxDataCount)
        {
            uint64_t totalLoop = totalElements / maxDataCount;
            uint64_t remainNum = totalElements % maxDataCount;

            for (uint64_t loop = 0; loop < totalLoop; loop++) {
                uint64_t curOffset = offset + loop * maxDataCount;

                // Read core 0's data to accumulator
                AscendC::DataCopy(accumUb, detGm[curOffset], maxDataCount);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

                // Add cores 1 to aicNum-1
                for (uint32_t aicIdx = 1; aicIdx < aicNum; aicIdx++) {
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
                    AscendC::DataCopy(tempUb, detGm[(uint64_t)aicIdx * dkvSize + curOffset], maxDataCount);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                    AscendC::Add(accumUb, accumUb, tempUb, maxDataCount);
                    AscendC::PipeBarrier<PIPE_V>();
                }

                // Write to shared workspace
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                AscendC::DataCopy(sharedGm[curOffset], accumUb, maxDataCount);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            }

            // Handle remainder
            if (remainNum > 0) {
                uint64_t curOffset = offset + totalLoop * maxDataCount;
                uint64_t remainAlign = (remainNum + 7) / 8 * 8;

                if (remainNum * sizeof(float) % 32 == 0) {
                    AscendC::DataCopy(accumUb, detGm[curOffset], remainNum);
                } else {
                    AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(remainNum * sizeof(float)), 0, 0, 0};
                    AscendC::DataCopyPadExtParams<float> padParams{true, 0, 0, 0};
                    AscendC::DataCopyPad(accumUb, detGm[curOffset], copyParams, padParams);
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

                for (uint32_t aicIdx = 1; aicIdx < aicNum; aicIdx++) {
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
                    if (remainNum * sizeof(float) % 32 == 0) {
                        AscendC::DataCopy(tempUb, detGm[(uint64_t)aicIdx * dkvSize + curOffset], remainNum);
                    } else {
                        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(remainNum * sizeof(float)), 0, 0, 0};
                        AscendC::DataCopyPadExtParams<float> padParams{true, 0, 0, 0};
                        AscendC::DataCopyPad(tempUb, detGm[(uint64_t)aicIdx * dkvSize + curOffset], copyParams, padParams);
                    }
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                    AscendC::Add(accumUb, accumUb, tempUb, remainAlign);
                    AscendC::PipeBarrier<PIPE_V>();
                }

                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                if (remainNum * sizeof(float) % 32 == 0) {
                    AscendC::DataCopy(sharedGm[curOffset], accumUb, remainNum);
                } else {
                    AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(remainNum * sizeof(float)), 0, 0, 0};
                    AscendC::DataCopyPad(sharedGm[curOffset], accumUb, copyParams);
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            }
        }

        __aicore__ inline void VecDeterReduce(Params const &params)
        {
            __gm__ BlockSparseAttentionGradTilingData *tilingData =
                reinterpret_cast<__gm__ BlockSparseAttentionGradTilingData *>(params.tiling);
            uint32_t vecCoreIdx = AscendC::GetBlockIdx();
            uint32_t usedCoreNum = tilingData->usedVecCoreNum;
            if (vecCoreIdx >= usedCoreNum) {
                return;
            }

            uint32_t aicNum = tilingData->aicNumForDet;
            uint32_t groupSizeDet = tilingData->groupSizeForDet;
            uint32_t totalSlots = aicNum * groupSizeDet;
            uint64_t dkvSize = tilingData->dkvSize;
            uint64_t headDim = tilingData->headDim;
            uint64_t sOutSize = tilingData->sOutSize;
            uint64_t dPOutSize = tilingData->dPOutSize;
            uint64_t dQOutSize = tilingData->dQOutSize;
            uint64_t dKOutSize = tilingData->dKOutSize;
            uint64_t dVOutSize = tilingData->dVOutSize;
            uint64_t gradSize = tilingData->gradSize;

            uint64_t detDkOffset = sOutSize + dPOutSize + dQOutSize + dKOutSize + dVOutSize + gradSize;
            uint64_t detDvOffset = detDkOffset + tilingData->detDkWorkspaceSize;

            // Shared workspace (reduction target, VecPost reads from here)
            AscendC::GlobalTensor<float> sharedDkGm;
            sharedDkGm.SetGlobalBuffer((__gm__ float *)(params.workspace + sOutSize + dPOutSize + dQOutSize));
            AscendC::GlobalTensor<float> sharedDvGm;
            sharedDvGm.SetGlobalBuffer((__gm__ float *)(params.workspace + sOutSize + dPOutSize + dQOutSize + dKOutSize));

            // Per-core workspace
            AscendC::GlobalTensor<float> detDkGm;
            detDkGm.SetGlobalBuffer((__gm__ float *)(params.workspace + detDkOffset));
            AscendC::GlobalTensor<float> detDvGm;
            detDvGm.SetGlobalBuffer((__gm__ float *)(params.workspace + detDvOffset));

            // Partition dk/dv data among AIV cores (same as VecPost)
            uint64_t kvPostSize = dkvSize / headDim;
            uint64_t kvPostBlockEachCore = kvPostSize / usedCoreNum;
            uint64_t kvPostBlockNumEachCore = kvPostBlockEachCore * headDim;
            uint64_t kvPostTailNum = kvPostSize % usedCoreNum;

            uint64_t computeS2 =
                (vecCoreIdx == usedCoreNum - 1) ? (kvPostBlockEachCore + kvPostTailNum) : kvPostBlockEachCore;
            uint64_t dkvOffset = vecCoreIdx * kvPostBlockNumEachCore;
            uint64_t totalElements = computeS2 * headDim;

            // UB allocation
            uint64_t ubSizeAvail = tilingData->ubSize;
            uint64_t maxDataCount = ubSizeAvail / sizeof(float) / 2;
            maxDataCount = maxDataCount / 8 * 8;

            LocalTensor<float> accumUb = resource.ubBuf.template GetBufferByByte<float>(0);
            LocalTensor<float> tempUb = resource.ubBuf.template GetBufferByByte<float>(maxDataCount * sizeof(float));

            // Reduce dk
            VecDeterReduceKernel(detDkGm, sharedDkGm, accumUb, tempUb, totalSlots, dkvSize, dkvOffset, totalElements,
                                 maxDataCount);
            // Reduce dv
            VecDeterReduceKernel(detDvGm, sharedDvGm, accumUb, tempUb, totalSlots, dkvSize, dkvOffset, totalElements,
                                 maxDataCount);
        }
#endif // Disabled deterministic reduction.

private:
    __gm__ int32_t *validCounts = nullptr;
    NpuArch::Arch::Resource<ArchTag> resource;
};

} // namespace BSA

#endif // BLOCK_SPARSE_ATTENTION_GRAD_KERNEL_H
