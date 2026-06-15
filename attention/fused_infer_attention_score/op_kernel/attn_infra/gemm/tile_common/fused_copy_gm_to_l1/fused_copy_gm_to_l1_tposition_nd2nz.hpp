/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSED_COPY_GM_TO_L1_TPOSITION_ND2NZ_HPP
#define FUSED_COPY_GM_TO_L1_TPOSITION_ND2NZ_HPP

#include "../../../../attn_infra/fused_base_defs.hpp"
#include "../../../../attn_infra/arch/fused_arch.hpp"
#include "../../../../attn_infra/layout/fused_layout.hpp"
#include "../../../../attn_infra/gemm/fused_gemm_type.hpp"
#include "../../../../attn_infra/gemm/tile_common/fused_tile_copy_tla.hpp"
#include "fused_copy_gm_to_l1_interval_datacopy.hpp"

namespace NpuArch::Gemm::Tile {

/// new add gemm
template <class ArchTag, class Element>
struct CopyGmToL1<
    ArchTag,
    Gemm::GemmType<Element, layout::RowMajor>,
    Gemm::GemmType<Element, layout::zN, AscendC::TPosition::A1>> {
    using LayoutDst = layout::zN;
    using LayoutSrc = layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_C0_zN = BYTE_PER_C0 / sizeof(Element);

    // Mehtods

    __aicore__ inline
    CopyGmToL1() {};

    __aicore__ inline
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        AscendC::Nd2NzParams intriParams;

        intriParams.ndNum = 1;
        intriParams.dValue = layoutSrc.shape(1);
        intriParams.srcNdMatrixStride = 0;
        intriParams.dstNzC0Stride = layoutDst.stride(3) / ELE_NUM_PER_C0_zN;
        intriParams.dstNzMatrixStride = 0;

        if (layoutSrc.stride(0) < STRIDE_LIMIT) {
            intriParams.nValue = layoutSrc.shape(0);
            intriParams.srcDValue = layoutSrc.stride(0);
            intriParams.dstNzNStride = layoutDst.stride(0) / ELE_NUM_PER_C0_zN;
            AscendC::DataCopy(dstTensor, srcTensor, intriParams);
        } else {
            intriParams.nValue = 1;
            intriParams.srcDValue = 0;
            intriParams.dstNzNStride = 0;
            for (uint32_t i = 0; i < layoutSrc.shape(0); i++) {
                AscendC::DataCopy(dstTensor[i * ELE_NUM_PER_C0_zN], srcTensor[i * layoutSrc.stride(0)], intriParams);
            }
        }
    }
};

template <class ArchTag, class Element>
struct CopyGmToL1<
    ArchTag,
    Gemm::GemmType<Element, layout::RowMajor>,
    Gemm::GemmType<Element, layout::zZ, AscendC::TPosition::B1>> {
    using LayoutDst = layout::zZ;
    using LayoutSrc = layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_C0_zZ = BYTE_PER_C0 / sizeof(Element);

    // Mehtods

    __aicore__ inline
    CopyGmToL1() {};

    __aicore__ inline
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        AscendC::Nd2NzParams intriParams;
        uint32_t srcNdStride = C0_NUM_PER_FRACTAL * layoutSrc.stride(0);
        uint32_t ndNum = layoutSrc.shape(0) / C0_NUM_PER_FRACTAL;
        uint32_t remains = layoutSrc.shape(0) % C0_NUM_PER_FRACTAL;
        if (srcNdStride < STRIDE_LIMIT) {
            if (ndNum) {
                intriParams.ndNum = ndNum;
                intriParams.nValue = C0_NUM_PER_FRACTAL;
                intriParams.dValue = layoutSrc.shape(1);
                intriParams.srcNdMatrixStride = srcNdStride;
                intriParams.srcDValue = layoutSrc.stride(0);

                intriParams.dstNzC0Stride = layoutDst.stride(3) / ELE_NUM_PER_C0_zZ;
                intriParams.dstNzNStride = layoutDst.stride(0) / ELE_NUM_PER_C0_zZ;

                intriParams.dstNzMatrixStride = layoutDst.stride(1);

                AscendC::DataCopy(dstTensor, srcTensor, intriParams);
            }

            if (remains) {
                AscendC::Nd2NzParams tailParams;
                tailParams.ndNum = 1;
                tailParams.nValue = remains;
                tailParams.dValue = layoutSrc.shape(1);
                tailParams.srcNdMatrixStride = srcNdStride;
                tailParams.srcDValue = layoutSrc.stride(0);

                tailParams.dstNzC0Stride = layoutDst.stride(3) / ELE_NUM_PER_C0_zZ;
                tailParams.dstNzNStride = layoutDst.stride(0) / ELE_NUM_PER_C0_zZ;
                tailParams.dstNzMatrixStride = 0;  //`

                AscendC::DataCopy(dstTensor[ndNum * layoutDst.stride(1)], srcTensor[ndNum * srcNdStride], tailParams);
            }
        } else if (layoutSrc.stride(0) < STRIDE_LIMIT) {
            for (uint32_t i = 0; i < ndNum; i++) {
                AscendC::Nd2NzParams intriParams;
                intriParams.ndNum = 1;
                intriParams.nValue = C0_NUM_PER_FRACTAL;
                intriParams.dValue = layoutSrc.shape(1);
                intriParams.srcNdMatrixStride = 0;
                intriParams.srcDValue = layoutSrc.stride(0);

                intriParams.dstNzC0Stride = layoutDst.stride(3) / ELE_NUM_PER_C0_zZ;
                intriParams.dstNzNStride = layoutDst.stride(0) / ELE_NUM_PER_C0_zZ;
                intriParams.dstNzMatrixStride = 0;

                AscendC::DataCopy(dstTensor[i * layoutDst.stride(1)], srcTensor[i * srcNdStride], intriParams);
            }
            if (remains) {
                AscendC::Nd2NzParams tailParams;
                tailParams.ndNum = 1;
                tailParams.nValue = remains;
                tailParams.dValue = layoutSrc.shape(1);
                tailParams.srcNdMatrixStride = 0;
                tailParams.srcDValue = layoutSrc.stride(0);

                tailParams.dstNzC0Stride = layoutDst.stride(3) / ELE_NUM_PER_C0_zZ;
                tailParams.dstNzNStride = layoutDst.stride(0) / ELE_NUM_PER_C0_zZ;
                tailParams.dstNzMatrixStride = 0;

                AscendC::DataCopy(dstTensor[ndNum * layoutDst.stride(1)], srcTensor[ndNum * srcNdStride], tailParams);
            }
        } else {
            for (uint32_t i = 0; i < layoutSrc.shape(0); i++) {
                uint32_t idxR0 = i / C0_NUM_PER_FRACTAL;
                uint32_t idxInR0 = i % C0_NUM_PER_FRACTAL;

                AscendC::Nd2NzParams intriParams;
                intriParams.ndNum = 1;
                intriParams.nValue = 1;
                intriParams.dValue = layoutSrc.shape(1);
                intriParams.srcNdMatrixStride = 0;
                intriParams.srcDValue = 0;

                intriParams.dstNzC0Stride = layoutDst.stride(3) / ELE_NUM_PER_C0_zZ;
                intriParams.dstNzNStride = 0;
                intriParams.dstNzMatrixStride = 0;

                uint32_t offsetDst = i * idxR0 * layoutDst.stride(1) + idxInR0 * ELE_NUM_PER_C0_zZ;
                uint32_t offsetSrc = i * layoutSrc.stride(0);
                AscendC::DataCopy(dstTensor[offsetDst], srcTensor[offsetSrc], intriParams);
            }
        }
    }
};

template <class ArchTag, class Element>
struct CopyGmToL1<
    ArchTag,
    Gemm::GemmType<Element, layout::ColumnMajor>,
    Gemm::GemmType<Element, layout::nN, AscendC::TPosition::A1>> {
    using LayoutDst = layout::nN;
    using LayoutSrc = layout::ColumnMajor;

    static constexpr uint32_t ELE_NUM_PER_C0_nN = BYTE_PER_C0 / sizeof(Element);

    // Mehtods

    __aicore__ inline
    CopyGmToL1() {};

    __aicore__ inline
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        AscendC::Nd2NzParams intriParams;
        uint32_t srcNdStride = C0_NUM_PER_FRACTAL * layoutSrc.stride(1);
        uint32_t ndNum = layoutSrc.shape(1) / C0_NUM_PER_FRACTAL;
        uint32_t remains = layoutSrc.shape(1) % C0_NUM_PER_FRACTAL;
        if (srcNdStride < STRIDE_LIMIT) {
            if (ndNum) {
                intriParams.ndNum = ndNum;
                intriParams.nValue = C0_NUM_PER_FRACTAL;
                intriParams.dValue = layoutSrc.shape(0);
                intriParams.srcNdMatrixStride = srcNdStride;
                intriParams.srcDValue = layoutSrc.stride(1);

                intriParams.dstNzC0Stride = layoutDst.stride(1) / ELE_NUM_PER_C0_nN;
                intriParams.dstNzNStride = layoutDst.stride(2) / ELE_NUM_PER_C0_nN;

                intriParams.dstNzMatrixStride = layoutDst.stride(3);

                AscendC::DataCopy(dstTensor, srcTensor, intriParams);
            }

            if (remains) {
                AscendC::Nd2NzParams tailParams;
                tailParams.ndNum = 1;
                tailParams.nValue = remains;
                tailParams.dValue = layoutSrc.shape(0);
                tailParams.srcNdMatrixStride = srcNdStride;
                tailParams.srcDValue = layoutSrc.stride(1);

                tailParams.dstNzC0Stride = layoutDst.stride(1) / ELE_NUM_PER_C0_nN;
                tailParams.dstNzNStride = layoutDst.stride(2) / ELE_NUM_PER_C0_nN;
                tailParams.dstNzMatrixStride = 0;

                AscendC::DataCopy(dstTensor[ndNum * layoutDst.stride(3)], srcTensor[ndNum * srcNdStride], tailParams);
            }
        } else if (layoutSrc.stride(1) < STRIDE_LIMIT) {
            for (uint32_t i = 0; i < ndNum; i++) {
                AscendC::Nd2NzParams intriParams;
                intriParams.ndNum = 1;
                intriParams.nValue = C0_NUM_PER_FRACTAL;
                intriParams.dValue = layoutSrc.shape(0);
                intriParams.srcNdMatrixStride = 0;
                intriParams.srcDValue = layoutSrc.stride(1);

                intriParams.dstNzC0Stride = layoutDst.stride(1) / ELE_NUM_PER_C0_nN;
                intriParams.dstNzNStride = layoutDst.stride(2) / ELE_NUM_PER_C0_nN;
                intriParams.dstNzMatrixStride = 0;

                AscendC::DataCopy(dstTensor[i * layoutDst.stride(3)], srcTensor[i * srcNdStride], intriParams);
            }
            if (remains) {
                AscendC::Nd2NzParams tailParams;
                tailParams.ndNum = 1;
                tailParams.nValue = remains;
                tailParams.dValue = layoutSrc.shape(0);
                tailParams.srcNdMatrixStride = 0;
                tailParams.srcDValue = layoutSrc.stride(1);

                tailParams.dstNzC0Stride = layoutDst.stride(1) / ELE_NUM_PER_C0_nN;
                tailParams.dstNzNStride = layoutDst.stride(2) / ELE_NUM_PER_C0_nN;
                tailParams.dstNzMatrixStride = 0;

                AscendC::DataCopy(dstTensor[ndNum * layoutDst.stride(3)], srcTensor[ndNum * srcNdStride], tailParams);
            }
        } else {
            for (uint32_t i = 0; i < layoutSrc.shape(1); i++) {
                uint32_t idxR0 = i / C0_NUM_PER_FRACTAL;
                uint32_t idxInR0 = i % C0_NUM_PER_FRACTAL;

                AscendC::Nd2NzParams intriParams;
                intriParams.ndNum = 1;
                intriParams.nValue = 1;
                intriParams.dValue = layoutSrc.shape(0);
                intriParams.srcNdMatrixStride = 0;
                intriParams.srcDValue = 0;

                intriParams.dstNzC0Stride = layoutDst.stride(1) / ELE_NUM_PER_C0_nN;
                intriParams.dstNzNStride = 0;
                intriParams.dstNzMatrixStride = 0;

                uint32_t offsetDst = i * idxR0 * layoutDst.stride(3) + idxInR0 * ELE_NUM_PER_C0_nN;
                uint32_t offsetSrc = i * layoutSrc.stride(1);
                AscendC::DataCopy(dstTensor[offsetDst], srcTensor[offsetSrc], intriParams);
            }
        }
    }
};

template <class ArchTag, class Element>
struct CopyGmToL1<
    ArchTag,
    Gemm::GemmType<Element, layout::ColumnMajor>,
    Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::B1>> {
    using LayoutDst = layout::nZ;
    using LayoutSrc = layout::ColumnMajor;

    static constexpr uint32_t ELE_NUM_PER_C0_nZ = BYTE_PER_C0 / sizeof(Element);

    // Mehtods

    __aicore__ inline
    CopyGmToL1() {};

    __aicore__ inline
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        AscendC::Nd2NzParams intriParams;

        intriParams.ndNum = 1;
        intriParams.dValue = layoutSrc.shape(0);
        intriParams.srcNdMatrixStride = 0;
        intriParams.dstNzC0Stride = layoutDst.stride(1) / ELE_NUM_PER_C0_nZ;
        intriParams.dstNzMatrixStride = 0;

        if (layoutSrc.stride(1) < STRIDE_LIMIT) {
            intriParams.nValue = layoutSrc.shape(1);
            intriParams.srcDValue = layoutSrc.stride(1);
            intriParams.dstNzNStride = layoutDst.stride(2) / ELE_NUM_PER_C0_nZ;
            AscendC::DataCopy(dstTensor, srcTensor, intriParams);
        } else {
            intriParams.nValue = 1;
            intriParams.srcDValue = 0;
            intriParams.dstNzNStride = 0;
            for (uint32_t i = 0; i < layoutSrc.shape(1); i++) {
                AscendC::DataCopy(dstTensor[i * ELE_NUM_PER_C0_nZ], srcTensor[i * layoutSrc.stride(1)], intriParams);
            }
        }
    }
};

} // namespace NpuArch::Gemm::Tile

#endif // FUSED_COPY_GM_TO_L1_TPOSITION_ND2NZ_HPP
