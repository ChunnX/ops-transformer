/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMV_TILE_VEC_COPY_UB_TO_GM_PADDING_HPP
#define CATLASS_GEMV_TILE_VEC_COPY_UB_TO_GM_PADDING_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/gemm/gemm_type.hpp"

namespace FTSelf::Gemv::Tile {

using namespace Catlass;

template <
    class ArchTag,
    class GmType,
    FTSelf::Gemv::helper::VEC_PADDING_TYPE PaddingType,
    bool is_atoadd = false
>
struct VecCopyUBToGmPadding
{
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy UB to gm, can not find the specialization.");
};


template <class Element>
struct VecCopyUBToGmPadding<Arch::AtlasA2,
    Catlass::Gemm::GemmType<Element, layout::VectorLayout>,
    FTSelf::Gemv::helper::VEC_PADDING_TYPE::ALIGNED,
    false>
{
    using LayoutSrc = layout::VectorLayout;
    using LayoutDst = layout::VectorLayout;
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);

    // Mehtods

    CATLASS_DEVICE
    VecCopyUBToGmPadding() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<Element> dstTensor,
        AscendC::LocalTensor<Element> srcTensor,
        layout::VectorLayout const &layoutDst,
        layout::VectorLayout const &layoutSrc
    )
    {
        // AscendC::DataCopyExtParams params;
        // params.blockCount = 1;
        // params.blockLen = layoutDst.shape(0) * sizeof(Element);
        // params.srcStride = 0;
        // params.dstStride = 0;
        // params.rsv = 0;

        /*
        // 支持连续
        template <typename T>
        __aicore__ inline void DataCopy(const GlobalTensor <T>& dstGlobal,
            const LocalTensor <T>& srcLocal, const uint32_t calCount)

        // 支持连续和不连续
        template <typename T>
        __aicore__ inline void DataCopy(const GlobalTensor <T>& dstGlobal,
            const LocalTensor <T>& srcLocal,
            const DataCopyParams& repeatParams)
        */

        /*
        表7 DataCopyParams结构体参数定义
        参数名称                                含义
        blockCount              指定该指令包含的连续传输数据块个数，
                                取值范围：blockCount∈[1, 4095]。

        blockLen                指定该指令每个连续传输数据块长度，单位为datablock(32B)。取值范围：blockLen∈[1, 65535]。
                                特别的，当dstLocal位于C2PIPE2GM时，单位为128B；
                                当dstLocal位于C2时，表示源操作数的连续传输数据块长度，单位为64B。

        srcStride               源操作数，相邻连续数据块的间隔（前面一个数据块的尾与后面数据块的头的间隔），单位为datablock(32B)。
                                数据类型为uint16_t，srcStride不要超出该数据类型的取值范围。

        dstStride               目的操作数，相邻连续数据块间的间隔（前面一个数据块的尾与后面数据块的头的间隔），单位为datablock(32B)。
                                数据类型为uint16_t，dstStride不要超出该数据类型的取值范围。
                                特别的，当dstLocal位于C2PIPE2GM时，单位为128B；当dstLocal位于C2时，单位为64B。
        */

        AscendC::DataCopyParams datacopyparams;
        datacopyparams.blockCount = 1;
        // layoutDst.shape(0)
        datacopyparams.blockLen = CeilDiv(layoutDst.shape(0), ELE_NUM_PER_C0);
        datacopyparams.srcStride = 0;
        datacopyparams.dstStride = 0;

        AscendC::DataCopy(
            dstTensor,
            srcTensor,
            datacopyparams);

        // AscendC::DataCopyPad(
        //     dstTensor,
        //     srcTensor,
        //     params
        // );
    }
};

template <class Element>
struct VecCopyUBToGmPadding<Arch::AtlasA2,
    Catlass::Gemm::GemmType<Element, layout::VectorLayout>,
    FTSelf::Gemv::helper::VEC_PADDING_TYPE::PADDING,
    false>
{
    using LayoutSrc = layout::VectorLayout;
    using LayoutDst = layout::VectorLayout;
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);

    // Mehtods

    CATLASS_DEVICE
    VecCopyUBToGmPadding() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<Element> dstTensor,
        AscendC::LocalTensor<Element> srcTensor,
        layout::VectorLayout const &layoutDst,
        layout::VectorLayout const &layoutSrc
    )
    {
        AscendC::DataCopyExtParams params;
        params.blockCount = 1;
        params.blockLen = layoutDst.shape(0) * sizeof(Element);
        params.srcStride = 0;
        params.dstStride = 0;
        params.rsv = 0;
        /*
        template <typename T>
        __aicore__ inline void DataCopyPad(const GlobalTensor<T> &dstGlobal,
            const LocalTensor<T> &srcLocal,
            const DataCopyExtParams &dataCopyParams)
        */
        AscendC::DataCopyPad(
            dstTensor,
            srcTensor,
            params
        );
    }
};

template <class Element>
struct VecCopyUBToGmPadding<Arch::AtlasA2, Catlass::Gemm::GemmType<Element, layout::VectorLayout>,FTSelf::Gemv::helper::VEC_PADDING_TYPE::PADDING,true>
{
    using LayoutSrc = layout::VectorLayout;
    using LayoutDst = layout::VectorLayout;
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);

    // Mehtods

    CATLASS_DEVICE
    VecCopyUBToGmPadding() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<Element> dstTensor,
        AscendC::LocalTensor<Element> srcTensor,
        layout::VectorLayout const &layoutDst,
        layout::VectorLayout const &layoutSrc
    )
    {
        AscendC::SetAtomicAdd<Element>();
        AscendC::DataCopyExtParams params;
        params.blockCount = 1;
        params.blockLen = layoutDst.shape(0) * sizeof(Element);
        params.srcStride = 0;
        params.dstStride = 0;
        params.rsv = 0;
        AscendC::DataCopyPad(
            dstTensor,
            srcTensor,
            params
        );
        AscendC::SetAtomicNone();
    }
};

} // namespace FTSelf::Gemv::Tile

#endif // CATLASS_GEMV_TILE_VEC_COPY_UB_TO_GM_HPP
