/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSED_COPY_GM_TO_L1_INTERVAL_DATACOPY_HPP
#define FUSED_COPY_GM_TO_L1_INTERVAL_DATACOPY_HPP

#include "../../../../attn_infra/fused_base_defs.hpp"
#include "../../../../attn_infra/arch/fused_arch.hpp"
#include "../../../../attn_infra/layout/fused_layout.hpp"
#include "../../../../attn_infra/gemm/fused_gemm_type.hpp"
#include "../../../../attn_infra/gemm/tile_common/fused_tile_copy_tla.hpp"

namespace NpuArch::Gemm::Tile {

template <
    class ArchTag,
    /// GemmType for matrix operand
    class GmType,
    class L1Type = void
>
struct CopyGmToL1 {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy gm to l1, can not find the specialization.");
};

template <
    class ArchTag,
    /// GemmType for matrix operand
    class GmType,
    class L1Type = void
>
struct CopyGmToL1IntervalDataCopy {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy gm to l1, can not find the specialization.");
};

////////////////////////////////////////
/// Using the standard strided DataCopy interface to implement nd2nz
/// transfer may achieve higher data transfer efficiency when the data block shape is short and wide
/// Partial specialization for AtlasA2, half, RowMajor in and zN out.
template<>
struct CopyGmToL1IntervalDataCopy<Arch::AtlasA2, Gemm::GemmType<half, layout::RowMajor>> {
    using LayoutDst = layout::zN;
    using LayoutSrc = layout::RowMajor;
    using Element = half;

    static constexpr uint32_t ELE_NUM_PER_C0_zN = BYTE_PER_C0 / sizeof(Element);

    // Mehtods

    __aicore__ inline
    CopyGmToL1IntervalDataCopy() {};

    __aicore__ inline
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        for (int i = 0; i < layoutSrc.shape(0); ++i) {
            AscendC::DataCopyParams dataCopyParams(
                NpuArch::Detail::Alignment::CeilDiv(layoutSrc.shape(1), layoutDst.shape(2)),
                layoutDst.shape(2) / ELE_NUM_PER_C0_zN,
                0,
                (layoutDst.stride(3) - layoutDst.shape(2)) / ELE_NUM_PER_C0_zN
            );
            AscendC::DataCopy(dstTensor[i * layoutDst.shape(2)], srcTensor[i * layoutSrc.stride(0)], dataCopyParams);
        }
    }
};

/// Partial specialization for AtlasA2, half, PaddingRowMajor in and zN out.
/// Using the standard strided DataCopy interface to implement nd2nz
/// transfer may achieve higher data transfer efficiency when the data block shape is short and wide
template<>
struct CopyGmToL1IntervalDataCopy<Arch::AtlasA2, Gemm::GemmType<half, layout::PaddingRowMajor>> {
    using LayoutDst = layout::zN;
    using LayoutSrc = layout::PaddingRowMajor;
    using Element = half;

    static constexpr uint32_t ELE_NUM_PER_C0_zN = BYTE_PER_C0 / sizeof(Element);

    // Mehtods

    __aicore__ inline
    CopyGmToL1IntervalDataCopy() {};

    __aicore__ inline
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        for (int i = 0; i < layoutSrc.orgShape(0); ++i) {
            AscendC::DataCopyParams dataCopyParams(
                NpuArch::Detail::Alignment::CeilDiv(layoutSrc.orgShape(1), layoutDst.shape(2)),
                layoutDst.shape(2) / ELE_NUM_PER_C0_zN,
                0,
                (layoutDst.stride(3) - layoutDst.shape(2)) / ELE_NUM_PER_C0_zN
            );
            AscendC::DataCopy(dstTensor[i * layoutDst.shape(2)], srcTensor[i * layoutSrc.stride(0)], dataCopyParams);
        }
    }
};

/// Partial specialization for AtlasA2, half, ColumnMajor in and zN out.
/// Using the standard strided DataCopy interface to implement nd2nz
/// transfer may achieve higher data transfer efficiency when the data block shape is tall and narrow
template<>
struct CopyGmToL1IntervalDataCopy<Arch::AtlasA2, Gemm::GemmType<half, layout::ColumnMajor>> {
    using LayoutDst = layout::nZ;
    using LayoutSrc = layout::ColumnMajor;
    using Element = half;

    static constexpr uint32_t ELE_NUM_PER_C0_nZ = BYTE_PER_C0 / sizeof(Element);

    // Mehtods

    __aicore__ inline
    CopyGmToL1IntervalDataCopy() {};

    __aicore__ inline
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        for (int i = 0; i < layoutSrc.shape(1); ++i) {
            AscendC::DataCopyParams dataCopyParams(
                NpuArch::Detail::Alignment::CeilDiv(layoutSrc.shape(0), layoutDst.shape(0)),
                layoutDst.shape(0) / ELE_NUM_PER_C0_nZ,
                0,
                (layoutDst.stride(1) - layoutDst.shape(0)) / ELE_NUM_PER_C0_nZ
            );
            AscendC::DataCopy(dstTensor[i * layoutDst.shape(0)], srcTensor[i * layoutSrc.stride(1)], dataCopyParams);
        }
    }
};

/// Partial specialization for AtlasA2, half, PaddingColumnMajor in and zN out.
/// Using the standard strided DataCopy interface to implement nd2nz
/// transfer may achieve higher data transfer efficiency when the data block shape is tall and narrow
template<>
struct CopyGmToL1IntervalDataCopy<Arch::AtlasA2, Gemm::GemmType<half, layout::PaddingColumnMajor>> {
    using LayoutDst = layout::nZ;
    using LayoutSrc = layout::PaddingColumnMajor;
    using Element = half;

    static constexpr uint32_t ELE_NUM_PER_C0_nZ_PAD = BYTE_PER_C0 / sizeof(Element);

    // Mehtods

    __aicore__ inline
    CopyGmToL1IntervalDataCopy() {};

    __aicore__ inline
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        for (int i = 0; i < layoutSrc.orgShape(1); ++i) {
            AscendC::DataCopyParams dataCopyParams(
                NpuArch::Detail::Alignment::CeilDiv(layoutSrc.orgShape(0), layoutDst.shape(0)),
                layoutDst.shape(0) / ELE_NUM_PER_C0_nZ_PAD,
                0,
                (layoutDst.stride(1) - layoutDst.shape(0)) / ELE_NUM_PER_C0_nZ_PAD
            );
            AscendC::DataCopy(dstTensor[i * layoutDst.shape(0)], srcTensor[i * layoutSrc.stride(2)], dataCopyParams);
        }
    }
};

} // namespace NpuArch::Gemm::Tile

#endif // FUSED_COPY_GM_TO_L1_INTERVAL_DATACOPY_HPP
