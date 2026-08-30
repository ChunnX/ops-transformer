/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mc2_queue_snapshot.h
 * \brief Kernel-side URMA queue snapshot engine for AIC/AIV error DFX.
 *
 * 每个 AIV 通信 block 在 Commit 前后快照队列状态到 GM workspace：
 *   ① CaptureCounters  — SQ/CQ head/tail 计数器
 *   ② CaptureSubmittedWqe — 回读刚提交的 WQE 原始字节 (__gm__ 读 sqVa)
 *   ③ CaptureLatestCqe   — 回读最新 CQE 原始字节 (__gm__ 读 scqVa)
 *
 * 异常时 host 侧 Mc2DumpTilingData 按 workspace 段自动 dump (type=13)，
 * 无需 host 侧任何改动。
 */

#ifndef MC2_QUEUE_SNAPSHOT_H
#define MC2_QUEUE_SNAPSHOT_H

#include "mc2_tiling_struct.h"

#if defined(__CCE_AICORE__)
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "../utils/mc2_hcomm_dfx_types.h"

namespace Mc2Kernel {

#if MC2_DFX_ENABLE
class Mc2QueueSnapshot {
public:
    __aicore__ inline Mc2QueueSnapshot() = default;

    __aicore__ inline void Init(GM_ADDR workspaceGM, Mc2Tiling::WorkspaceLayoutInfo *wsLayout)
    {
        slotBase_ = nullptr;
        if (wsLayout == nullptr) {
            return;
        }
        uint64_t libApiSize = 0;
        for (uint32_t i = 0; i < wsLayout->segCount && i < Mc2Tiling::MAX_WORKSPACE_SEGMENTS; i++) {
            if (wsLayout->segments[i].type == Mc2Tiling::WS_SEG_LIB_API) {
                libApiSize = wsLayout->segments[i].size;
            }
            if (wsLayout->segments[i].type == Mc2Tiling::WS_SEG_QUEUE_SNAPSHOT) {
                uint64_t userOffset = wsLayout->segments[i].offset - libApiSize;
                slotBase_ = reinterpret_cast<__gm__ Mc2Tiling::QueueSnapshotSlot *>(
                    static_cast<uint64_t>(workspaceGM) + userOffset);
                break;
            }
        }
    }

    __aicore__ inline void CaptureCounters(uint64_t channelHandle, uint32_t peer, uint32_t turn)
    {
        if (slotBase_ == nullptr || channelHandle == 0 ||
            peer >= Mc2Tiling::QUEUE_SNAPSHOT_MAX_PEERS) {
            return;
        }
        __gm__ Mc2Tiling::QueueSnapshotSlot *slot = &slotBase_[peer];
        __gm__ AscendC::ChannelEntity *entity =
            reinterpret_cast<__gm__ AscendC::ChannelEntity *>(channelHandle);

        slot->magic = Mc2Tiling::QUEUE_SNAPSHOT_MAGIC;
        slot->turn = turn;
        slot->peer = peer;
        slot->sqHead = entity->sqHead;
        slot->sqTail = entity->sqTail;
        slot->cqHead = entity->cqHead;
        slot->cqTail = entity->cqTail;

        uint64_t sqCtxAddr = reinterpret_cast<uint64_t>(entity->sqContextAddr);
        if (entity->sqNum > 0 && sqCtxAddr != 0) {
            __gm__ AscendC::SqContext *sqCtx =
                reinterpret_cast<__gm__ AscendC::SqContext *>(sqCtxAddr);
            if (sqCtx->type == AscendC::SQ_CONTEXT_TYPE_UB_JFS) {
                slot->sqDepth = sqCtx->contextInfo.ubJfs.sqDepth;
                slot->wqeSize = sqCtx->contextInfo.ubJfs.wqeSize;
                slot->sqLiveHead = ReadU32(sqCtx->contextInfo.ubJfs.headAddr);
                slot->sqLiveTail = ReadU32(sqCtx->contextInfo.ubJfs.tailAddr);
            }
        }

        uint64_t cqCtxAddr = reinterpret_cast<uint64_t>(entity->cqContextAddr);
        if (entity->cqNum > 0 && cqCtxAddr != 0) {
            __gm__ AscendC::CqContext *cqCtx =
                reinterpret_cast<__gm__ AscendC::CqContext *>(cqCtxAddr);
            if (cqCtx->type == AscendC::CQ_CONTEXT_TYPE_UB_JFC) {
                slot->cqDepth = cqCtx->contextInfo.ubJfc.cqDepth;
                slot->cqeSize = cqCtx->contextInfo.ubJfc.cqeSize;
                slot->cqLiveHead = ReadU32(cqCtx->contextInfo.ubJfc.headAddr);
                slot->cqLiveTail = ReadU32(cqCtx->contextInfo.ubJfc.tailAddr);
            }
        }
        FlushSlot(peer);
    }

    __aicore__ inline void CaptureSubmittedWqe(uint64_t channelHandle, uint32_t peer, uint32_t turn)
    {
        if (slotBase_ == nullptr || channelHandle == 0 ||
            peer >= Mc2Tiling::QUEUE_SNAPSHOT_MAX_PEERS) {
            return;
        }
        __gm__ Mc2Tiling::QueueSnapshotSlot *slot = &slotBase_[peer];
        if (slot->wqeSize == 0 || slot->sqDepth == 0) {
            return;
        }
        __gm__ AscendC::ChannelEntity *entity =
            reinterpret_cast<__gm__ AscendC::ChannelEntity *>(channelHandle);
        uint64_t sqCtxAddr = reinterpret_cast<uint64_t>(entity->sqContextAddr);
        if (entity->sqNum == 0 || sqCtxAddr == 0) {
            return;
        }
        __gm__ AscendC::SqContext *sqCtx =
            reinterpret_cast<__gm__ AscendC::SqContext *>(sqCtxAddr);
        if (sqCtx->type != AscendC::SQ_CONTEXT_TYPE_UB_JFS) {
            return;
        }
        uint64_t sqVa = sqCtx->contextInfo.ubJfs.sqVa;
        if (sqVa == 0) {
            return;
        }

        uint32_t tail = slot->sqLiveTail;
        uint32_t readBytes = slot->wqeSize;
        if (readBytes <= Mc2Tiling::QUEUE_SNAPSHOT_WQE_BYTES / 2U) {
            readBytes = readBytes * 2U;
        }
        readBytes = std::min<uint32_t>(readBytes, Mc2Tiling::QUEUE_SNAPSHOT_WQE_BYTES);

        uint32_t wqeIdx = (tail > 0) ? (tail - 1U) : (slot->sqDepth - 1U);
        uint64_t wqeAddr = sqVa + static_cast<uint64_t>(wqeIdx) * slot->wqeSize;

        __gm__ uint8_t *dst = reinterpret_cast<__gm__ uint8_t *>(slot->wqeLatest);
        __gm__ uint8_t *src = reinterpret_cast<__gm__ uint8_t *>(wqeAddr);
        CopyGmToGm(dst, src, readBytes);

        uint32_t ringPos = slot->ringPos;
        __gm__ uint8_t *ringDst = reinterpret_cast<__gm__ uint8_t *>(slot->wqeRing[ringPos]);
        CopyGmToGm(ringDst, src, readBytes);
        slot->wqeRingTurn[ringPos] = turn;
        slot->ringPos = (ringPos + 1U) % Mc2Tiling::QUEUE_SNAPSHOT_RING_SIZE;

        FlushSlot(peer);
    }

    __aicore__ inline void CaptureLatestCqe(uint64_t channelHandle, uint32_t peer, uint32_t turn)
    {
        if (slotBase_ == nullptr || channelHandle == 0 ||
            peer >= Mc2Tiling::QUEUE_SNAPSHOT_MAX_PEERS) {
            return;
        }
        __gm__ Mc2Tiling::QueueSnapshotSlot *slot = &slotBase_[peer];
        if (slot->cqeSize == 0 || slot->cqDepth == 0) {
            return;
        }
        __gm__ AscendC::ChannelEntity *entity =
            reinterpret_cast<__gm__ AscendC::ChannelEntity *>(channelHandle);
        uint64_t cqCtxAddr = reinterpret_cast<uint64_t>(entity->cqContextAddr);
        if (entity->cqNum == 0 || cqCtxAddr == 0) {
            return;
        }
        __gm__ AscendC::CqContext *cqCtx =
            reinterpret_cast<__gm__ AscendC::CqContext *>(cqCtxAddr);
        if (cqCtx->type != AscendC::CQ_CONTEXT_TYPE_UB_JFC) {
            return;
        }
        uint64_t scqVa = cqCtx->contextInfo.ubJfc.scqVa;
        if (scqVa == 0) {
            return;
        }

        uint32_t tail = slot->cqLiveTail;
        uint32_t readBytes = std::min<uint32_t>(slot->cqeSize, Mc2Tiling::QUEUE_SNAPSHOT_WQE_BYTES);
        uint32_t cqeIdx = (tail > 0) ? (tail - 1U) : (slot->cqDepth - 1U);
        uint64_t cqeAddr = scqVa + static_cast<uint64_t>(cqeIdx) * slot->cqeSize;

        uint32_t cqeWord0 = *reinterpret_cast<__gm__ uint32_t *>(cqeAddr);
        slot->cqeStatus = (cqeWord0 >> 24) & 0xFF;
        slot->cqeSubstatus = (cqeWord0 >> 16) & 0xFF;

        __gm__ uint8_t *dst = reinterpret_cast<__gm__ uint8_t *>(slot->cqeLatest);
        __gm__ uint8_t *src = reinterpret_cast<__gm__ uint8_t *>(cqeAddr);
        CopyGmToGm(dst, src, readBytes);

        (void)turn;
        FlushSlot(peer);
    }

private:
    __aicore__ inline uint32_t ReadU32(uint64_t addr)
    {
        if (addr == 0) {
            return 0;
        }
        return *reinterpret_cast<__gm__ uint32_t *>(addr);
    }

    __aicore__ inline void CopyGmToGm(__gm__ uint8_t *dst, __gm__ uint8_t *src, uint32_t size)
    {
        uint32_t words = size / sizeof(uint64_t);
        __gm__ uint64_t *srcW = reinterpret_cast<__gm__ uint64_t *>(src);
        __gm__ uint64_t *dstW = reinterpret_cast<__gm__ uint64_t *>(dst);
        for (uint32_t i = 0; i < words; i++) {
            dstW[i] = srcW[i];
        }
        uint32_t remain = size % sizeof(uint64_t);
        if (remain > 0) {
            __gm__ uint8_t *srcT = src + words * sizeof(uint64_t);
            __gm__ uint8_t *dstT = dst + words * sizeof(uint64_t);
            for (uint32_t i = 0; i < remain; i++) {
                dstT[i] = srcT[i];
            }
        }
    }

    __aicore__ inline void FlushSlot(uint32_t peer)
    {
        AscendC::GlobalTensor<uint8_t> cacheLine;
        cacheLine.SetGlobalBuffer(
            reinterpret_cast<GM_ADDR>(&slotBase_[peer]),
            Mc2Tiling::QUEUE_SNAPSHOT_SLOT_SIZE);
        DataCacheCleanAndInvalid<uint8_t, AscendC::CacheLine::ENTIRE_DATA_CACHE,
                                  AscendC::DcciDst::CACHELINE_OUT>(cacheLine);
    }

    __gm__ Mc2Tiling::QueueSnapshotSlot *slotBase_ = nullptr;
};
#else  // !MC2_DFX_ENABLE
class Mc2QueueSnapshot {
public:
    __aicore__ inline Mc2QueueSnapshot() = default;
    __aicore__ inline void Init(GM_ADDR, Mc2Tiling::WorkspaceLayoutInfo *) {}
    __aicore__ inline void CaptureCounters(uint64_t, uint32_t, uint32_t) {}
    __aicore__ inline void CaptureSubmittedWqe(uint64_t, uint32_t, uint32_t) {}
    __aicore__ inline void CaptureLatestCqe(uint64_t, uint32_t, uint32_t) {}
};
#endif // MC2_DFX_ENABLE

} // namespace Mc2Kernel
#endif // __CCE_AICORE__
#endif // MC2_QUEUE_SNAPSHOT_H
