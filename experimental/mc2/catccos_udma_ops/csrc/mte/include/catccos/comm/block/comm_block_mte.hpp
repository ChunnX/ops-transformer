// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.
#pragma once

#include "catlass/arch/resource.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/matrix_coord.hpp"
#include "catccos/comm/mte_transport_params.hpp"
#include "shmem.h"

namespace Catccos::Comm {
enum class MteCopyDirection { Get, Put, Local };

namespace Block {

// The logical communication shape belongs to the operator schedule. Physical
// transfers use bounded, aligned chunks and a private UB ring on each AIV.
template <class Element, MteCopyDirection Direction>
class MteCommBlock {
public:
    using ElementSrc = Element;
    using ElementDst = Element;
    using LayoutSrc = Catlass::layout::RowMajor;
    using LayoutDst = Catlass::layout::RowMajor;
    using MatrixCoord = Catlass::MatrixCoord;
    using Config = OpsTransformer::CatccosMte::MteCommConfig;
    struct TileRemoteCopy {
        struct Params {
            MatrixCoord tileShape;
            CATLASS_HOST_DEVICE Params() = default;
            CATLASS_HOST_DEVICE Params(MatrixCoord shape) : tileShape(shape) {}
        };
    };
    struct Params {
        MatrixCoord blockShape;
        typename TileRemoteCopy::Params tileParams;
        Config transport;
        CATLASS_HOST_DEVICE Params() = default;
        CATLASS_HOST_DEVICE Params(MatrixCoord block, typename TileRemoteCopy::Params tile,
                                  Config config = {})
            : blockShape(block), tileParams(tile), transport(config) {}
        CATLASS_DEVICE MatrixCoord BlockShape() const { return blockShape; }
        CATLASS_DEVICE MatrixCoord TileShape() const { return tileParams.tileShape; }
    };

    template <class ArchTag>
    CATLASS_DEVICE MteCommBlock(Catlass::Arch::Resource<ArchTag> &resource, Params const &params)
        : config_(params.transport)
    {
        using namespace OpsTransformer::CatccosMte;
        static_assert(kMtePhysicalUbBytes - kMteSystemReservedBytes <= ArchTag::UB_SIZE,
            "The MTE host budget exceeds the architecture's allocatable UB");
        for (uint32_t i = 0; i < config_.commUbStages; ++i) {
            buffers_[i] = resource.ubBuf.template GetBufferByByte<Element>(
                kMteSignalScratchBytes + i * config_.commChunkBytes);
        }
    }

    CATLASS_DEVICE void SetWorker(uint32_t lane, uint32_t workers)
    {
        lane_ = lane;
        workers_ = workers;
    }
    CATLASS_DEVICE void InitBlockLoop()
    {
        slot_ = 0;
        for (uint32_t i = 0; i < config_.commUbStages; ++i) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(i);
        }
    }
    CATLASS_DEVICE void FinalizeBlockLoop()
    {
        for (uint32_t i = 0; i < config_.commUbStages; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(i);
        }
        slot_ = 0;
    }

    CATLASS_DEVICE void operator()(
        AscendC::GlobalTensor<Element> const &src, LayoutSrc const &srcLayout,
        AscendC::GlobalTensor<Element> const &dst, LayoutDst const &dstLayout,
        MatrixCoord const &shape, uint32_t peer = 0)
    {
        if (shape.row() == 0 || shape.column() == 0) {
            return;
        }
        auto srcPtr = const_cast<__gm__ Element *>(src.GetPhyAddr());
        auto dstPtr = const_cast<__gm__ Element *>(dst.GetPhyAddr());
        if constexpr (Direction == MteCopyDirection::Get) {
            srcPtr = reinterpret_cast<__gm__ Element *>(aclshmem_ptr(
                reinterpret_cast<__gm__ void *>(srcPtr), peer));
        } else if constexpr (Direction == MteCopyDirection::Put) {
            dstPtr = reinterpret_cast<__gm__ Element *>(aclshmem_ptr(
                reinterpret_cast<__gm__ void *>(dstPtr), peer));
        }
        AscendC::GlobalTensor<Element> source;
        AscendC::GlobalTensor<Element> target;
        source.SetGlobalBuffer(srcPtr);
        target.SetGlobalBuffer(dstPtr);
        const uint64_t columns = shape.column();
        const uint64_t elementsPerChunk = config_.commChunkBytes / sizeof(Element);
        if (srcLayout.stride(0) == columns && dstLayout.stride(0) == columns) {
            const uint64_t elements = uint64_t{shape.row()} * columns;
            const uint64_t chunks = (elements + elementsPerChunk - 1) / elementsPerChunk;
            for (uint64_t chunk = lane_; chunk < chunks; chunk += workers_) {
                const uint64_t offset = chunk * elementsPerChunk;
                const uint32_t count = static_cast<uint32_t>(
                    elements - offset < elementsPerChunk ? elements - offset : elementsPerChunk);
                Copy(source[offset], target[offset], count);
            }
        } else {
            const uint64_t chunksPerRow = (columns + elementsPerChunk - 1) / elementsPerChunk;
            const uint64_t chunks = uint64_t{shape.row()} * chunksPerRow;
            for (uint64_t chunk = lane_; chunk < chunks; chunk += workers_) {
                const uint64_t row = chunk / chunksPerRow;
                const uint64_t column = (chunk % chunksPerRow) * elementsPerChunk;
                const uint32_t count = static_cast<uint32_t>(
                    columns - column < elementsPerChunk ? columns - column : elementsPerChunk);
                Copy(source[row * srcLayout.stride(0) + column],
                     target[row * dstLayout.stride(0) + column], count);
            }
        }
    }

private:
    CATLASS_DEVICE void Copy(AscendC::GlobalTensor<Element> const &src,
                            AscendC::GlobalTensor<Element> const &dst, uint32_t count)
    {
        const uint32_t event = slot_;
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event);
        const uint32_t bytes = count * sizeof(Element);
        AscendC::DataCopyExtParams copyParams{1, bytes, 0, 0, 0};
        aclshmemi_copy_gm2ub(buffers_[slot_], src, copyParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(event);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(event);
        aclshmemi_copy_ub2gm(dst, buffers_[slot_], copyParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(event);
        slot_ = slot_ + 1 == config_.commUbStages ? 0 : slot_ + 1;
    }

    Config config_;
    AscendC::LocalTensor<Element> buffers_[OpsTransformer::CatccosMte::kMteMaxUbStages];
    uint32_t slot_{0};
    uint32_t lane_{0};
    uint32_t workers_{1};
};

} // namespace Block
} // namespace Catccos::Comm
