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
 * \file fragment_tensor.h
 * \brief FragmentTensor：多个 GM fragment 的统一抽象，支持 Slice / Copy / Scatter。
 */

#pragma once

#include "tensor_api/tensor.h"
#include "blaze/gemm/tile/pad_mx_kl1.h"
#include "fragment_tiling_data.h"

namespace Apace {
namespace Basic {

using namespace Apace::Utils;

using FragmentIdx = uint32_t;
using FragmentCount = uint32_t;

constexpr uint32_t MAX_FRAGMENT_COUNT = 32;

template<uint32_t Dims, uint32_t MaxFragments, typename LayoutFactory, typename ElementType>
class FragmentTensor;

template<uint32_t Dims, uint32_t MaxFragments, typename LayoutFactory, typename ElementType>
class FragmentSlice {
public:
    using FragTensor = FragmentTensor<Dims, MaxFragments, LayoutFactory, ElementType>;

    __aicore__ inline FragmentSlice() = default;

    __aicore__ inline FragmentSlice(
        const FragTensor& frag,
        const uint64_t offsets[Dims],
        const uint64_t sizes[Dims])
        : frag_(&frag)
    {
        for (uint32_t d = 0; d < Dims; ++d) {
            offsets_[d] = offsets[d];
            sizes_[d] = sizes[d];
        }
    }

    __aicore__ inline const FragTensor& GetFragmentTensor() const { return *frag_; }
    __aicore__ inline uint64_t GetOffset(uint32_t d) const { return offsets_[d]; }
    __aicore__ inline uint64_t GetSize(uint32_t d) const { return sizes_[d]; }

    template<typename CoordType, typename ShapeType>
    __aicore__ inline auto Slice(const CoordType& coord, const ShapeType& shape) const
    {
        uint64_t newOffsets[Dims]{};
        uint64_t newSizes[Dims]{};
        if constexpr (Dims == 2) {
            newOffsets[0] = offsets_[0] + AscendC::Te::Get<0>(coord);  newSizes[0] = AscendC::Te::Get<0>(shape);
            newOffsets[1] = offsets_[1] + AscendC::Te::Get<1>(coord);  newSizes[1] = AscendC::Te::Get<1>(shape);
        } else if constexpr (Dims == 3) {
            newOffsets[0] = offsets_[0] + AscendC::Te::Get<0>(coord);  newSizes[0] = AscendC::Te::Get<0>(shape);
            newOffsets[1] = offsets_[1] + AscendC::Te::Get<1>(coord);  newSizes[1] = AscendC::Te::Get<1>(shape);
            newOffsets[2] = offsets_[2] + AscendC::Te::Get<2>(coord);  newSizes[2] = AscendC::Te::Get<2>(shape);
        } else if constexpr (Dims == 4) {
            newOffsets[0] = offsets_[0] + AscendC::Te::Get<0>(coord);  newSizes[0] = AscendC::Te::Get<0>(shape);
            newOffsets[1] = offsets_[1] + AscendC::Te::Get<1>(coord);  newSizes[1] = AscendC::Te::Get<1>(shape);
            newOffsets[2] = offsets_[2] + AscendC::Te::Get<2>(coord);  newSizes[2] = AscendC::Te::Get<2>(shape);
            newOffsets[3] = offsets_[3] + AscendC::Te::Get<3>(coord);  newSizes[3] = AscendC::Te::Get<3>(shape);
        }
        return FragmentSlice(*frag_, newOffsets, newSizes);
    }

private:
    const FragTensor* frag_;
    uint64_t offsets_[Dims];
    uint64_t sizes_[Dims];
};

template<typename Engine, typename DestTensor,
         uint32_t Dims, uint32_t MaxF, typename GmLayoutF, typename ElemT>
__aicore__ inline void FragmentSliceCopy(
    Engine engine,
    DestTensor& dest,
    const FragmentSlice<Dims, MaxF, GmLayoutF, ElemT>& src)
{
    auto& fragTensor = src.GetFragmentTensor();
    const auto& fragTiling = fragTensor.GetFragTiling();
    const auto AssembleAxis = fragTiling.AssembleAxis;
    const auto nonAssembleAxis = 1 - AssembleAxis;
    auto composition = fragTensor.ComputeFragmentComposition(
        src.GetOffset(AssembleAxis), src.GetSize(AssembleAxis));

    for (uint32_t fi = 0; fi < composition.totalFragmentCnt; ++fi) {
        auto info = fragTensor.GetFragmentInfo(fi, composition);
        auto fragment = fragTensor.GetFragment(info.fragmentIdx);

        if (AssembleAxis == 0) {
            auto gmTensor = fragment.Slice(
                AscendC::Te::MakeCoord(info.fragmentStartOffset, src.GetOffset(nonAssembleAxis)),
                AscendC::Te::MakeShape(info.usableSize, src.GetSize(nonAssembleAxis)));
            auto l1Tensor = dest.Slice(
                AscendC::Te::MakeCoord(info.offsetInSubRegion, 0),
                AscendC::Te::MakeShape(info.usableSize, src.GetSize(nonAssembleAxis)));
            AscendC::Te::Copy(engine, l1Tensor, gmTensor);
        } else {
            auto gmTensor = fragment.Slice(
                AscendC::Te::MakeCoord(src.GetOffset(nonAssembleAxis), info.fragmentStartOffset),
                AscendC::Te::MakeShape(src.GetSize(nonAssembleAxis), info.usableSize));
            auto l1Tensor = dest.Slice(
                AscendC::Te::MakeCoord(0, info.offsetInSubRegion),
                AscendC::Te::MakeShape(src.GetSize(nonAssembleAxis), info.usableSize));
            AscendC::Te::Copy(engine, l1Tensor, gmTensor);
        }
    }
}

template<typename Engine, uint32_t Dims, uint32_t MaxF, typename LayoutF, typename ElemT, typename SrcTensor>
__aicore__ inline void FragmentSliceScatter(
    Engine engine,
    const FragmentSlice<Dims, MaxF, LayoutF, ElemT>& dest,
    SrcTensor& src)
{
    auto& fragTensor = dest.GetFragmentTensor();
    const auto& fragTiling = fragTensor.GetFragTiling();
    const auto AssembleAxis = fragTiling.AssembleAxis;
    const auto nonAssembleAxis = 1 - AssembleAxis;
    const uint64_t realFragmentSize = fragTensor.GetRealFragmentSize();

    auto composition = fragTensor.ComputeFragmentComposition(
        dest.GetOffset(AssembleAxis), dest.GetSize(AssembleAxis));

    for (uint32_t fi = 0; fi < composition.totalFragmentCnt; ++fi) {
        auto info = fragTensor.GetFragmentInfo(fi, composition);
        auto fragment = fragTensor.GetFragment(info.fragmentIdx);

        if (info.fragmentStartOffset >= realFragmentSize) { continue; }
        uint64_t remain = realFragmentSize - info.fragmentStartOffset;
        int64_t copyRows = static_cast<int64_t>(
            info.usableSize < remain ? info.usableSize : remain);

        if (AssembleAxis == 0) {
            auto cBlock = fragment.Slice(
                AscendC::Te::MakeCoord(info.fragmentStartOffset, dest.GetOffset(nonAssembleAxis)),
                AscendC::Te::MakeShape(copyRows, dest.GetSize(nonAssembleAxis)));
            auto l0cSlice = src.Slice(
                AscendC::Te::MakeCoord(static_cast<int64_t>(info.offsetInSubRegion), 0),
                AscendC::Te::MakeShape(copyRows, dest.GetSize(nonAssembleAxis)));
            AscendC::Te::Copy(engine, cBlock, l0cSlice);
        } else {
            auto cBlock = fragment.Slice(
                AscendC::Te::MakeCoord(dest.GetOffset(nonAssembleAxis), info.fragmentStartOffset),
                AscendC::Te::MakeShape(dest.GetSize(nonAssembleAxis), copyRows));
            auto l0cSlice = src.Slice(
                AscendC::Te::MakeCoord(0, static_cast<int64_t>(info.offsetInSubRegion)),
                AscendC::Te::MakeShape(dest.GetSize(nonAssembleAxis), copyRows));
            AscendC::Te::Copy(engine, cBlock, l0cSlice);
        }
    }
}

/*!
 * \brief FragmentTensor类（所有Fragment大小相同）
 */
template<uint32_t Dims, uint32_t MaxFragments = MAX_FRAGMENT_COUNT,
         typename LayoutFactory = void, typename ElementType = uint8_t>
class FragmentTensor {
public:
    __aicore__ inline FragmentTensor() = default;

    __aicore__ inline FragmentTensor(
        const FragmentTilingData<Dims>& fragTiling,
        GM_ADDR* addrList)
        : fragTiling_(fragTiling)
    {
        for (uint32_t i = 0; i < fragTiling.fragmentCnt; ++i) {
            addrList_[i] = addrList[i];
        }
    }

    __aicore__ inline auto GetFragment(uint32_t idx) const
    {
        auto fragmentAddr = reinterpret_cast<__gm__ ElementType*>(addrList_[idx]);

        uint64_t fragmentSizes[Dims];
        fragmentSizes[fragTiling_.AssembleAxis] = fragTiling_.fragmentSize;

        for (uint32_t i = 0; i < Dims; ++i) {
            if (i != fragTiling_.AssembleAxis) {
                fragmentSizes[i] = fragTiling_.assembledShape[i];
            }
        }

        auto memPtr = AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(fragmentAddr);

        if constexpr (Dims == 2) {
            return AscendC::Te::MakeTensor(memPtr, LayoutFactory{}(fragmentSizes[0], fragmentSizes[1]));
        } else if constexpr (Dims == 3) {
            return AscendC::Te::MakeTensor(memPtr,
                LayoutFactory{}(fragmentSizes[0], fragmentSizes[1], fragmentSizes[2]));
        } else if constexpr (Dims == 4) {
            return AscendC::Te::MakeTensor(memPtr,
                LayoutFactory{}(fragmentSizes[0], fragmentSizes[1], fragmentSizes[2], fragmentSizes[3]));
        }
    }

    __aicore__ inline uint64_t GetFragmentSize() const
    {
        return fragTiling_.fragmentSize;
    }

    __aicore__ inline uint64_t GetRealFragmentSize() const
    {
        return fragTiling_.realFragmentSize;
    }

    __aicore__ inline uint32_t GetFragmentCnt() const
    {
        return fragTiling_.fragmentCnt;
    }

    __aicore__ inline uint32_t GetSplitAxis() const
    {
        return fragTiling_.AssembleAxis;
    }

    __aicore__ inline GM_ADDR GetFragmentAddr(uint32_t idx) const
    {
        return addrList_[idx];
    }

    __aicore__ inline void UpdateAddrList(GM_ADDR* addrList)
    {
        for (uint32_t i = 0; i < fragTiling_.fragmentCnt; ++i) {
            addrList_[i] = addrList[i];
        }
    }

    __aicore__ inline const FragmentTilingData<Dims>& GetFragTiling() const { return fragTiling_; }

    template<typename CoordType, typename ShapeType>
    __aicore__ inline auto Slice(const CoordType& coord, const ShapeType& shape) const
    {
        uint64_t offsets[Dims]{};
        uint64_t sizes[Dims]{};

        if constexpr (Dims == 2) {
            offsets[0] = AscendC::Te::Get<0>(coord);  sizes[0] = AscendC::Te::Get<0>(shape);
            offsets[1] = AscendC::Te::Get<1>(coord);  sizes[1] = AscendC::Te::Get<1>(shape);
        } else if constexpr (Dims == 3) {
            offsets[0] = AscendC::Te::Get<0>(coord);  sizes[0] = AscendC::Te::Get<0>(shape);
            offsets[1] = AscendC::Te::Get<1>(coord);  sizes[1] = AscendC::Te::Get<1>(shape);
            offsets[2] = AscendC::Te::Get<2>(coord);  sizes[2] = AscendC::Te::Get<2>(shape);
        } else if constexpr (Dims == 4) {
            offsets[0] = AscendC::Te::Get<0>(coord);  sizes[0] = AscendC::Te::Get<0>(shape);
            offsets[1] = AscendC::Te::Get<1>(coord);  sizes[1] = AscendC::Te::Get<1>(shape);
            offsets[2] = AscendC::Te::Get<2>(coord);  sizes[2] = AscendC::Te::Get<2>(shape);
            offsets[3] = AscendC::Te::Get<3>(coord);  sizes[3] = AscendC::Te::Get<3>(shape);
        }

        return FragmentSlice<Dims, MaxFragments, LayoutFactory, ElementType>(*this, offsets, sizes);
    }

    struct FragmentComposition {
        uint32_t startFragmentIdx;
        uint32_t endFragmentIdx;
        uint32_t totalFragmentCnt;

        uint64_t headSize;
        uint64_t bodySize;
        uint32_t bodyCnt;
        uint64_t tailSize;

        uint64_t startRelativeCoord;
    };

    __aicore__ inline FragmentComposition ComputeFragmentComposition(
        uint64_t startCoord,
        uint64_t length) const
    {
        FragmentComposition comp;

        comp.startFragmentIdx = startCoord / fragTiling_.fragmentSize;
        comp.startRelativeCoord = startCoord % fragTiling_.fragmentSize;

        uint64_t endCoord = startCoord + length;
        comp.endFragmentIdx = endCoord / fragTiling_.fragmentSize;

        comp.totalFragmentCnt = comp.endFragmentIdx - comp.startFragmentIdx + 1;

        if (comp.totalFragmentCnt == 1) {
            comp.headSize = length;
            comp.bodySize = fragTiling_.fragmentSize;
            comp.bodyCnt = 0;
            comp.tailSize = length;
        } else {
            comp.headSize = fragTiling_.fragmentSize - comp.startRelativeCoord;
            comp.bodySize = fragTiling_.fragmentSize;
            comp.tailSize = endCoord % fragTiling_.fragmentSize;

            if (comp.tailSize == 0) {
                comp.tailSize = fragTiling_.fragmentSize;
                comp.endFragmentIdx--;
                comp.totalFragmentCnt--;
            }

            comp.bodyCnt = comp.totalFragmentCnt - 2;
        }

        return comp;
    }

    struct FragmentInfo {
        uint32_t fragmentIdx;
        GM_ADDR addr;
        uint64_t usableSize;
        uint64_t offsetInSubRegion;
        uint64_t fragmentStartOffset;
    };

    __aicore__ inline FragmentInfo GetFragmentInfo(
        uint32_t localIdx,
        const FragmentComposition& comp) const
    {
        FragmentInfo info;

        info.fragmentIdx = comp.startFragmentIdx + localIdx;
        info.addr = addrList_[info.fragmentIdx];

        if (localIdx == 0) {
            info.usableSize = comp.headSize;
            info.offsetInSubRegion = 0;
            info.fragmentStartOffset = comp.startRelativeCoord;
        } else if (localIdx < comp.totalFragmentCnt - 1) {
            info.usableSize = comp.bodySize;
            info.offsetInSubRegion = comp.headSize + (localIdx - 1) * comp.bodySize;
            info.fragmentStartOffset = 0;
        } else {
            info.usableSize = comp.tailSize;
            info.offsetInSubRegion = comp.headSize + comp.bodyCnt * comp.bodySize;
            info.fragmentStartOffset = 0;
        }

        return info;
    }

private:
    FragmentTilingData<Dims> fragTiling_;
    GM_ADDR addrList_[MaxFragments];
};

// L1 tensor K 轴补零，复用 Blaze PadZero 底层 API。
template <typename TensorL1>
__aicore__ inline void PadMxKAL1Zero(TensorL1& tensorL1, uint64_t kAxis)
{
    using type = typename TensorL1::elementType;
    auto layoutL1 = tensorL1.Layout();
    auto kAxisL1Align = AscendC::Std::get<0>(AscendC::Std::get<1>(layoutL1.Shape())) *
                        AscendC::Std::get<1>(AscendC::Std::get<1>(layoutL1.Shape()));

    if constexpr (AscendC::Te::IsSatisfiedPtnFormatV<TensorL1, AscendC::Te::NZLayoutPtn>) {
        if constexpr (Blaze::Gemm::Tile::PadMxKL1Base::IsMxFp4<type>()) { return; }
        if (kAxisL1Align - kAxis < AscendC::Te::C0_SIZE<type>) { return; }
        auto mAlign = AscendC::Std::get<0>(AscendC::Std::get<0>(layoutL1.Shape())) *
                      AscendC::Std::get<1>(AscendC::Std::get<0>(layoutL1.Shape()));
        auto kAxisND2NZAlign = AscendC::Std::ceil_align(kAxis, AscendC::Te::C0_SIZE<type>);
        auto sliceTensor = tensorL1.Slice(
            AscendC::Te::MakeCoord(0, kAxisND2NZAlign),
            AscendC::Te::MakeShape(mAlign, kAxisL1Align - kAxisND2NZAlign));
        Blaze::Gemm::Tile::PadMxKL1Base::PadZero(sliceTensor, 1, mAlign, 0);
    } else if constexpr (AscendC::Te::IsSatisfiedPtnFormatV<TensorL1, AscendC::Te::ZNLayoutPtn>) {
        if (kAxis == kAxisL1Align) { return; }
        auto m1 = AscendC::Std::get<1>(AscendC::Std::get<0>(layoutL1.Shape()));
        auto m0 = AscendC::Std::get<0>(AscendC::Std::get<0>(layoutL1.Shape()));
        auto dstRowStride = AscendC::Std::get<1>(AscendC::Std::get<0>(layoutL1.Stride()));
        auto dstGap = (dstRowStride / AscendC::Te::C0_ELEMENT<type>) - kAxisL1Align + kAxis;
        auto sliceTensor = tensorL1.Slice(
            AscendC::Te::MakeCoord(0, kAxis), AscendC::Te::MakeShape(m1 * m0, kAxisL1Align - kAxis));
        Blaze::Gemm::Tile::PadMxKL1Base::PadZero(sliceTensor, m1, kAxisL1Align - kAxis, dstGap);
    }
}

} // namespace Basic
} // namespace Apace
