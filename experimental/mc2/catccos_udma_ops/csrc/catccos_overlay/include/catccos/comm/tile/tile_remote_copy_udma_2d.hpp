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
        aclshmemx_udma_quiet(peerIdx);
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
};

} // namespace Catccos::Comm::Tile

#endif // CATCCOS_COMM_TILE_REMOTE_COPY_UDMA_2D_HPP
