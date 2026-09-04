/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATCCOS_COMM_TILE_REMOTE_COPY_UDMA_2D_HPP
#define CATCCOS_COMM_TILE_REMOTE_COPY_UDMA_2D_HPP

#include "catccos/catccos.hpp"
#include "catccos/detail/remote_copy_type.hpp"

#include "catlass/gemm/gemm_type.hpp"
#include "catlass/matrix_coord.hpp"

#include "shmem.h"

namespace Catccos::Comm::Tile {

/// Checked equivalent of release/v1.6.0's public quiet, which discards the CQ
/// status. These tiles submit only to QP0. Use SHMEM's metadata accessors so
/// the direct/relay table follows the pinned release's UDMA metadata layout.
CATLASS_DEVICE
uint32_t UdmaQuietWithStatus(uint32_t peerIdx)
{
    __gm__ aclshmemi_aiv_udma_info_t *udmaInfo =
        aclshmemi_udma_qp_info_fetch();
    __gm__ aclshmemi_udma_qp_table_t *table =
        aclshmemi_udma_active_table(udmaInfo);
    uint32_t qpNum = udmaInfo->qp_num;
    if constexpr (ACLSHMEM_RELAY_SUPPORTED) {
        uint32_t rankCount =
            static_cast<uint32_t>(aclshmemi_get_total_pe());
        uint32_t firstError = 0;
        for (uint32_t relayPeer = 0; relayPeer < rankCount; ++relayPeer) {
            // The diagonal slot is never initialized or used by SHMEM.
            if (relayPeer == peerIdx) {
                continue;
            }
            uint32_t slot = peerIdx * rankCount + relayPeer;
            __gm__ aclshmemi_udma_wq_ctx_t *qpContext =
                (__gm__ aclshmemi_udma_wq_ctx_t *)(
                    table->sq_ptr +
                    (slot * qpNum) * sizeof(aclshmemi_udma_wq_ctx_t));
            uint32_t wqeCount = qpContext->wqe_cnt;
            if (wqeCount == 0) {
                continue;
            }
            uint32_t status =
                aclshmemi_udma_poll_cq(slot, 0, wqeCount);
            if (firstError == 0 && status != 0) {
                firstError = status;
            }
        }
        return firstError;
    } else {
        __gm__ aclshmemi_udma_wq_ctx_t *qpContext =
            (__gm__ aclshmemi_udma_wq_ctx_t *)(
                table->sq_ptr +
                (peerIdx * qpNum) * sizeof(aclshmemi_udma_wq_ctx_t));
        return aclshmemi_udma_poll_cq(peerIdx, 0, qpContext->wqe_cnt);
    }
}

/// Layout-aware two-dimensional UDMA copy.
///
/// UDMA exposes a byte-contiguous transfer primitive. A single request is used
/// when both tensors are tightly packed. Otherwise one request is submitted per
/// row so that different source and destination leading dimensions are honored.
template <class ArchTag_, class SrcType_, class DstType_, detail::CopyDirect CopyDirect_>
class TileRemoteCopyUdma2D {
public:
    using ArchTag = ArchTag_;
    using ElementSrc = typename SrcType_::Element;
    using LayoutSrc = typename SrcType_::Layout;
    using ElementDst = typename DstType_::Element;
    using LayoutDst = typename DstType_::Layout;

    static constexpr detail::CopyDirect RemoteCopyDirect = CopyDirect_;
    static constexpr uint32_t UDMA_WQE_SCRATCH_BYTES = 256;
    // This PIPE_S variant may enqueue many row requests. The owning CommBlock
    // drains them once after the current peer/block loop instead of polling the
    // same cumulative CQ tail after every small tile.
    static constexpr bool DEFER_QUIET = true;
    static_assert(sizeof(ElementSrc) == sizeof(ElementDst),
        "UDMA copies bytes and therefore requires equal source and destination element sizes");

    CATLASS_DEVICE
    TileRemoteCopyUdma2D() = default;

    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<ElementDst> const &dstTensor, LayoutDst const &dstLayout,
        AscendC::GlobalTensor<ElementSrc> const &srcTensor, LayoutSrc const &srcLayout,
        Catlass::MatrixCoord const &copyShape,
        AscendC::LocalTensor<ElementSrc> const &tmpUb,
        uint32_t copyEventId,
        uint32_t peerIdx)
    {
        (void)tmpUb;
        (void)copyEventId;

        uint32_t rows = copyShape.row();
        uint32_t columns = copyShape.column();
        if (rows == 0 || columns == 0) {
            return;
        }
        uint64_t bytesPerRow = static_cast<uint64_t>(columns) * sizeof(ElementSrc);
        uint64_t srcStride = srcLayout.stride(0);
        uint64_t dstStride = dstLayout.stride(0);

        if (rows == 1 || (srcStride == columns && dstStride == columns)) {
            Submit(dstTensor, srcTensor, rows * bytesPerRow, peerIdx);
        } else {
            for (uint32_t row = 0; row < rows; ++row) {
                auto dstRow = dstTensor[row * dstStride];
                auto srcRow = srcTensor[row * srcStride];
                Submit(dstRow, srcRow, bytesPerRow, peerIdx);
            }
        }
    }

    /// Submit the optimized contiguous PUT and publish a 64-bit generation in
    /// the same WRITE_WITH_NOTIFY WQE.  The fused AllToAllV scheduler passes a
    /// complete peer slice whose source and destination layouts are both
    /// tightly packed, so one WQE is sufficient and the notification is a hard
    /// data-before-ready edge.
    CATLASS_DEVICE
    void PutContiguousWithSignal(
        AscendC::GlobalTensor<ElementDst> const &dstTensor,
        LayoutDst const &dstLayout,
        AscendC::GlobalTensor<ElementSrc> const &srcTensor,
        LayoutSrc const &srcLayout,
        Catlass::MatrixCoord const &copyShape,
        __gm__ uint64_t *signalAddress,
        uint64_t signalValue,
        uint32_t peerIdx)
    {
        static_assert(CopyDirect_ == detail::CopyDirect::Put,
            "UDMA write-with-notify is valid only for PUT tiles");
        uint32_t rows = copyShape.row();
        uint32_t columns = copyShape.column();
        if (rows == 0 || columns == 0) {
            return;
        }
        (void)dstLayout;
        (void)srcLayout;

        // A final-row notification would be unsafe for a strided transfer:
        // v1.6 UDMA WQEs deliberately carry no inter-submission ordering.
        // The only caller is the peer-serial AllToAllV schedule, which passes
        // one tightly packed [rows, K] peer slice. Do not silently skip a
        // transfer here: that would also omit its notification and deadlock
        // the receiver. The caller owns this contiguous-layout precondition.
        uint64_t messageLen =
            static_cast<uint64_t>(rows) * columns * sizeof(ElementSrc);
        SubmitWithSignal(
            dstTensor, srcTensor, messageLen,
            signalAddress, signalValue, peerIdx);
    }

    CATLASS_DEVICE
    void Finalize(uint32_t peerIdx)
    {
        aclshmemx_udma_quiet(peerIdx);
    }

    /// Preserve CQ timeout/error status for completion-sensitive collectives.
    /// They must not publish an acknowledgement when this returns nonzero.
    CATLASS_DEVICE
    uint32_t FinalizeWithStatus(uint32_t peerIdx)
    {
        return UdmaQuietWithStatus(peerIdx);
    }

private:
    static constexpr uint32_t SYNC_ID = 0;

    CATLASS_DEVICE
    static void Submit(
        AscendC::GlobalTensor<ElementDst> const &dstTensor,
        AscendC::GlobalTensor<ElementSrc> const &srcTensor,
        uint64_t messageLen,
        uint32_t peerIdx)
    {
        // Const GM tensor handles expose const-qualified pointers, while the
        // UDMA C API declares both GM operands as mutable byte pointers.
        auto dstAddr = reinterpret_cast<__gm__ uint8_t *>(
            const_cast<__gm__ ElementDst *>(dstTensor.GetPhyAddr()));
        auto srcAddr = reinterpret_cast<__gm__ uint8_t *>(
            const_cast<__gm__ ElementSrc *>(srcTensor.GetPhyAddr()));
        // PIPE_S publishes the WQE with scalar stores and does not consume UB
        // scratch or an MTE event. This avoids colliding with the fused kernel's
        // own TPipe, UB allocation, and MTE3 event IDs.
        auto unusedScratch = static_cast<__ubuf__ uint8_t *>(nullptr);
        auto elementCount = static_cast<uint32_t>(messageLen);

        if constexpr (CopyDirect_ == detail::CopyDirect::Put) {
            aclshmemx_udma_put_nbi<uint8_t, PIPE_S>(
                dstAddr,
                srcAddr,
                unusedScratch,
                elementCount, peerIdx, SYNC_ID);
        } else {
            aclshmemx_udma_get_nbi<uint8_t, PIPE_S>(
                dstAddr,
                srcAddr,
                unusedScratch,
                elementCount, peerIdx, SYNC_ID);
        }
    }

    CATLASS_DEVICE
    static void SubmitWithSignal(
        AscendC::GlobalTensor<ElementDst> const &dstTensor,
        AscendC::GlobalTensor<ElementSrc> const &srcTensor,
        uint64_t messageLen,
        __gm__ uint64_t *signalAddress,
        uint64_t signalValue,
        uint32_t peerIdx)
    {
        auto dstAddr = reinterpret_cast<__gm__ uint8_t *>(
            const_cast<__gm__ ElementDst *>(dstTensor.GetPhyAddr()));
        auto srcAddr = reinterpret_cast<__gm__ uint8_t *>(
            const_cast<__gm__ ElementSrc *>(srcTensor.GetPhyAddr()));
        auto unusedScratch = static_cast<__ubuf__ uint8_t *>(nullptr);
        aclshmemx_udma_put_signal_nbi<uint8_t, PIPE_S>(
            dstAddr, srcAddr, static_cast<uint32_t>(messageLen),
            signalAddress, signalValue, peerIdx,
            unusedScratch, SYNC_ID);
    }
};

/// Layout-aware two-dimensional UDMA copy whose local endpoint may be ordinary
/// device GM. The caller supplies a conservative UB scratch region used to publish a
/// UDMA WQE; the scratch carries control metadata rather than tensor payload.
///
/// This variant deliberately does not create a TPipe internally. Fused kernels
/// already own their pipeline resources, so the communication block provides a
/// persistent scratch tensor from Catlass::Arch::Resource instead.
template <class ArchTag_, class SrcType_, class DstType_, detail::CopyDirect CopyDirect_,
    uint32_t SyncId_ = 2>
class TileRemoteCopyUdma2DWithScratch {
public:
    using ArchTag = ArchTag_;
    using ElementSrc = typename SrcType_::Element;
    using LayoutSrc = typename SrcType_::Layout;
    using ElementDst = typename DstType_::Element;
    using LayoutDst = typename DstType_::Layout;

    static constexpr detail::CopyDirect RemoteCopyDirect = CopyDirect_;
    // Reserve the same conservative size as the existing generic UDMA tile.
    // Current WQE formats need no more than this amount.
    static constexpr uint32_t UDMA_WQE_SCRATCH_BYTES = 256;
    // Match the PIPE_S tile's completion contract: the owning CommBlock
    // submits all WQEs for one peer and performs one checked CQ drain at the
    // end of the block loop.  Keeping the completion point identical makes
    // PIPE_S versus PIPE_MTE3 a single-variable diagnostic comparison.
    static constexpr bool DEFER_QUIET = true;
    static_assert(sizeof(ElementSrc) == sizeof(ElementDst),
        "UDMA copies bytes and therefore requires equal source and destination element sizes");

    CATLASS_DEVICE
    TileRemoteCopyUdma2DWithScratch() = default;

    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<ElementDst> const &dstTensor, LayoutDst const &dstLayout,
        AscendC::GlobalTensor<ElementSrc> const &srcTensor, LayoutSrc const &srcLayout,
        Catlass::MatrixCoord const &copyShape,
        AscendC::LocalTensor<ElementSrc> const &tmpUb,
        uint32_t copyEventId,
        uint32_t peerIdx)
    {
        (void)copyEventId;

        uint32_t rows = copyShape.row();
        uint32_t columns = copyShape.column();
        if (rows == 0 || columns == 0) {
            return;
        }

        if (!scratchInitialized_) {
            InitializeScratch(tmpUb);
            scratchInitialized_ = true;
        }

        uint64_t bytesPerRow = static_cast<uint64_t>(columns) * sizeof(ElementSrc);
        uint64_t srcStride = srcLayout.stride(0);
        uint64_t dstStride = dstLayout.stride(0);

        if (rows == 1 || (srcStride == columns && dstStride == columns)) {
            Submit(dstTensor, srcTensor, tmpUb, rows * bytesPerRow, peerIdx);
        } else {
            for (uint32_t row = 0; row < rows; ++row) {
                auto dstRow = dstTensor[row * dstStride];
                auto srcRow = srcTensor[row * srcStride];
                Submit(dstRow, srcRow, tmpUb, bytesPerRow, peerIdx);
            }
        }
    }

    CATLASS_DEVICE
    void Finalize(uint32_t peerIdx)
    {
        aclshmemx_udma_quiet(peerIdx);
    }

    /// Match PIPE_S's checked completion for QP0, including relay slots.
    CATLASS_DEVICE
    uint32_t FinalizeWithStatus(uint32_t peerIdx)
    {
        return UdmaQuietWithStatus(peerIdx);
    }

private:
    // The fused communication block reserves MTE event IDs 0 and 1 for its
    // two local-copy UB stages. Keep PIPE_MTE3 UDMA publication separate.
    static constexpr uint32_t SYNC_ID = SyncId_;

    CATLASS_DEVICE
    static void Submit(
        AscendC::GlobalTensor<ElementDst> const &dstTensor,
        AscendC::GlobalTensor<ElementSrc> const &srcTensor,
        AscendC::LocalTensor<ElementSrc> const &tmpUb,
        uint64_t messageLen,
        uint32_t peerIdx)
    {
        auto dstAddr = reinterpret_cast<__gm__ uint8_t *>(
            const_cast<__gm__ ElementDst *>(dstTensor.GetPhyAddr()));
        auto srcAddr = reinterpret_cast<__gm__ uint8_t *>(
            const_cast<__gm__ ElementSrc *>(srcTensor.GetPhyAddr()));
        auto scratchAddr = (__ubuf__ uint8_t *)(tmpUb.GetPhyAddr());

        if constexpr (CopyDirect_ == detail::CopyDirect::Put) {
            aclshmemx_udma_put_nbi(
                dstAddr, srcAddr, scratchAddr,
                static_cast<uint32_t>(messageLen), peerIdx, SYNC_ID);
        } else {
            aclshmemx_udma_get_nbi(
                dstAddr, srcAddr, scratchAddr,
                static_cast<uint32_t>(messageLen), peerIdx, SYNC_ID);
        }
    }

    CATLASS_DEVICE
    static void InitializeScratch(AscendC::LocalTensor<ElementSrc> const &tmpUb)
    {
        auto scratch = (__ubuf__ uint64_t *)(tmpUb.GetPhyAddr());
        for (uint32_t i = 0; i < UDMA_WQE_SCRATCH_BYTES / sizeof(uint64_t); ++i) {
            scratch[i] = 0U;
        }
    }

    bool scratchInitialized_{false};
};

} // namespace Catccos::Comm::Tile

#endif // CATCCOS_COMM_TILE_REMOTE_COPY_UDMA_2D_HPP
