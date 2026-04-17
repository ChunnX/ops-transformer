/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_TILE_COPY_GM_TO_L1_A5_HPP
#define CATLASS_GEMM_TILE_COPY_GM_TO_L1_A5_HPP

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/numeric_size.hpp"
#include "catlass/gemm/tile/tile_copy_tla.hpp"
#include "tla/tensor.hpp"

namespace Catlass::Gemm::Tile {

/// Partial specialization for CopyGmToL1, Ascend950, RowMajor in and zN out.
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<
    Arch::Ascend950,
    tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
    std::enable_if_t<tla::detail::isRowMajor<LayoutSrc>::value && tla::detail::iszN<ElementDst, LayoutDst>::value>> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<ElementSrc>::value;

    // Mehtods

    CATLASS_DEVICE
    TileCopyTla() {};

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(
        TensorDst const &dstTensor,
        TensorSrc const &srcTensor,
        uint32_t ndNum = 1,
        uint32_t srcNdMatrixStride = 0,
        uint32_t dstNzMatrixStride = 0
    )
    {
        static_assert(
            tla::detail::isRowMajor<typename TensorSrc::Layout>::value
                && tla::detail::iszN<typename TensorDst::Element, typename TensorDst::Layout>::value
                && TensorSrc::position == AscendC::TPosition::GM && TensorDst::position == AscendC::TPosition::A1,
            "The input parameters do not match. TensorSrc must be GM and RowMajor, while TensorDst must be L1 and zN"
        );

        const uint32_t nValue = tla::get<0>(srcTensor.shape());
        const uint32_t dValue = tla::get<1>(srcTensor.shape());
        const uint32_t srcDValue = tla::get<0>(srcTensor.stride());
        const uint32_t dstInnerStrideRow = tla::get<0, 0>(dstTensor.stride());
        const uint32_t dstOuterStrideCol = tla::get<1, 1>(dstTensor.stride());

        AscendC::Nd2NzParams intriParams;

        intriParams.ndNum = ndNum;
        intriParams.nValue = nValue;
        intriParams.dValue = dValue;
        // There is an issue with AscenC in the data migration of nd2nz of type float4: dValue needs to be passed
        // with the length of int8, and it is uncertain whether this will be fixed in subsequent versions.
        if constexpr (AscendC::Std::is_one_of_v<typename TensorSrc::Element, float4_e2m1x2_t, float4_e1m2x2_t>) {
            intriParams.dValue = CeilDiv(intriParams.dValue, 2);
        }
        intriParams.srcNdMatrixStride = srcNdMatrixStride;
        intriParams.srcDValue = srcDValue;
        if constexpr (AscendC::Std::is_one_of_v<typename TensorSrc::Element, float4_e2m1x2_t, float4_e1m2x2_t>) {
            intriParams.srcDValue = CeilDiv(intriParams.srcDValue, 2);
        }
        intriParams.dstNzC0Stride = dstOuterStrideCol / ELE_NUM_PER_C0;
        intriParams.dstNzNStride = dstInnerStrideRow / ELE_NUM_PER_C0;
        intriParams.dstNzMatrixStride = dstNzMatrixStride;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        AscendC::DataCopy(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], intriParams);
    }
};

/// Partial specialization for CopyGmToL1, Ascend950, zN in and zN out.
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<
    Arch::Ascend950,
    tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
    std::enable_if_t<tla::detail::iszN<ElementSrc, LayoutSrc>::value && tla::detail::iszN<ElementDst, LayoutDst>::value>> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<ElementSrc>::value;

    // Mehtods

    CATLASS_DEVICE
    TileCopyTla() {};

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(
        TensorDst const &dstTensor,
        TensorSrc const &srcTensor
    )
    {
        static_assert(
            tla::detail::iszN<typename TensorSrc::Element, typename TensorSrc::Layout>::value
                && tla::detail::iszN<typename TensorDst::Element, typename TensorDst::Layout>::value
                && TensorSrc::position == AscendC::TPosition::GM && TensorDst::position == AscendC::TPosition::A1,
            "The input parameters do not match. TensorSrc must be GM and zN, while TensorDst must be L1 and zN"
        );

        const uint32_t blockCount = tla::get<1, 1>(srcTensor.shape());
        const uint32_t blockLen = tla::get<0, 0>(srcTensor.shape()) * tla::get<0, 1>(srcTensor.shape());

        AscendC::DataCopyParams repeatParams;

        repeatParams.blockCount = blockCount;
        repeatParams.blockLen = blockLen;
        repeatParams.srcStride = tla::get<1, 1>(srcTensor.stride()) / ELE_NUM_PER_C0 - blockLen;
        repeatParams.dstStride = tla::get<1, 1>(dstTensor.stride()) / ELE_NUM_PER_C0 - blockLen;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        AscendC::DataCopy(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], repeatParams);
    }
};

/// Partial specialization for CopyGmToL1, Ascend950, ColumnMajor in and nZ out.
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<
    Arch::Ascend950,
    tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
    std::enable_if_t<tla::detail::isColumnMajor<LayoutSrc>::value && tla::detail::isnZ<ElementDst, LayoutDst>::value>> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<ElementSrc>::value;

    // Mehtods

    CATLASS_DEVICE
    TileCopyTla() {};

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(
        TensorDst const &dstTensor,
        TensorSrc const &srcTensor,
        uint32_t ndNum = 1,
        uint32_t srcNdMatrixStride = 0,
        uint32_t dstNzMatrixStride = 0
    )
    {
        static_assert(
            tla::detail::isColumnMajor<typename TensorSrc::Layout>::value
                && tla::detail::isnZ<typename TensorDst::Element, typename TensorDst::Layout>::value
                && TensorSrc::position == AscendC::TPosition::GM && TensorDst::position == AscendC::TPosition::A1,
            "The input parameters do not match. TensorSrc must be GM and ColumnMajor, "
            "while TensorDst must be L1 and nZ"
        );

        const uint32_t nValue = tla::get<1>(srcTensor.shape());
        const uint32_t dValue = tla::get<0>(srcTensor.shape());
        const uint32_t srcDValue = tla::get<1>(srcTensor.stride());
        const uint32_t dstInnerStrideCol = tla::get<1, 0>(dstTensor.stride());
        const uint32_t dstOuterStrideRow = tla::get<0, 1>(dstTensor.stride());

        AscendC::Nd2NzParams intriParams;

        intriParams.ndNum = ndNum;
        intriParams.nValue = nValue;
        intriParams.dValue = dValue;
        // There is an issue with AscenC in the data migration of nd2nz of type float4: dValue needs to be passed
        // with the length of int8, and it is uncertain whether this will be fixed in subsequent versions.
        if constexpr (AscendC::Std::is_one_of_v<typename TensorSrc::Element, float4_e2m1x2_t, float4_e1m2x2_t>) {
            intriParams.dValue = CeilDiv(intriParams.dValue, 2);
        }
        intriParams.srcNdMatrixStride = srcNdMatrixStride;
        intriParams.srcDValue = srcDValue;
        if constexpr (AscendC::Std::is_one_of_v<typename TensorSrc::Element, float4_e2m1x2_t, float4_e1m2x2_t>) {
            intriParams.srcDValue = CeilDiv(intriParams.srcDValue, 2);
        }
        intriParams.dstNzC0Stride = dstOuterStrideRow / ELE_NUM_PER_C0;
        intriParams.dstNzNStride = dstInnerStrideCol / ELE_NUM_PER_C0;
        intriParams.dstNzMatrixStride = dstNzMatrixStride;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        AscendC::DataCopy(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], intriParams);
    }
};

/// Partial specialization for CopyGmToL1, Ascend950, nZ in and nZ out.
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<
    Arch::Ascend950,
    tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
    std::enable_if_t<tla::detail::isnZ<ElementSrc, LayoutSrc>::value && tla::detail::isnZ<ElementDst, LayoutDst>::value>> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<ElementSrc>::value;

    // Mehtods

    CATLASS_DEVICE
    TileCopyTla() {};

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(
        TensorDst const &dstTensor,
        TensorSrc const &srcTensor
    )
    {
        static_assert(
            tla::detail::isnZ<typename TensorSrc::Element, typename TensorSrc::Layout>::value
                && tla::detail::isnZ<typename TensorDst::Element, typename TensorDst::Layout>::value
                && TensorSrc::position == AscendC::TPosition::GM && TensorDst::position == AscendC::TPosition::A1,
            "The input parameters do not match. TensorSrc must be GM and nZ, "
            "while TensorDst must be L1 and nZ"
        );

        const uint32_t blockCount = tla::get<0, 1>(srcTensor.shape());
        const uint32_t blockLen = tla::get<1, 0>(srcTensor.shape()) * tla::get<1, 1>(srcTensor.shape());

        AscendC::DataCopyParams repeatParams;

        repeatParams.blockCount = blockCount;
        repeatParams.blockLen = blockLen;
        repeatParams.srcStride = tla::get<0, 1>(srcTensor.stride()) / ELE_NUM_PER_C0 - blockLen;
        repeatParams.dstStride = tla::get<0, 1>(dstTensor.stride()) / ELE_NUM_PER_C0 - blockLen;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        AscendC::DataCopy(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], repeatParams);
    }
};

/// Partial specialization for CopyGmToL1, Ascend950, VectorLayout in and VectorLayout out.
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<
    Arch::Ascend950,
    tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
    std::enable_if_t<tla::detail::isVector<LayoutSrc>::value && tla::detail::isVector<LayoutDst>::value>> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<ElementSrc>::value;

    // Mehtods

    CATLASS_DEVICE
    TileCopyTla() {};

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor)
    {
        static_assert(
            tla::detail::isVector<typename TensorSrc::Layout>::value
                && tla::detail::isVector<typename TensorDst::Layout>::value
                && TensorSrc::position == AscendC::TPosition::GM && TensorDst::position == AscendC::TPosition::A1,
            "The input parameters do not match. TensorSrc must be GM and Vector, "
            "while TensorDst must be L1 and Vector"
        );

        AscendC::DataCopyParams intriParams;
        intriParams.blockCount = 1;
        intriParams.blockLen = CeilDiv(tla::get<0>(srcTensor.shape()), ELE_NUM_PER_C0);
        intriParams.srcStride = 0;
        intriParams.dstStride = 0;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        AscendC::DataCopy(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], intriParams);
    }
};

/// Partial specialization for CopyGmToL1, Ascend950, fp8_e8m0_t, MxScaleA RowMajor in and zZ out.
template <class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<
    Arch::Ascend950,
    tla::Tensor<AscendC::GlobalTensor<float8_e8m0_t>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
    tla::Tensor<AscendC::LocalTensor<float8_e8m0_t>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
    std::enable_if_t<
        tla::detail::isMxScaleANoTrans<float8_e8m0_t, LayoutSrc>::value &&
        tla::detail::isMxScalezZ<float8_e8m0_t, LayoutDst>::value>> {

    CATLASS_DEVICE
    TileCopyTla() {};

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor)
    {
        static_assert(
            std::is_same_v<typename TensorSrc::Element, float8_e8m0_t> &&
            std::is_same_v<typename TensorDst::Element, float8_e8m0_t> &&
            tla::detail::isMxScaleANoTrans<typename TensorSrc::Element, typename TensorSrc::Layout>::value &&
            tla::detail::isMxScalezZ<typename TensorDst::Element, typename TensorDst::Layout>::value &&
            TensorSrc::position == AscendC::TPosition::GM && TensorDst::position == AscendC::TPosition::A1,
            "The input parameters do not match. TensorSrc must be fp8_e8m0_t, GM and RowMajor, while TensorDst must be "
            "fp8_e8m0_t, L1 and zZ"
        );

        const uint32_t rows = tla::get<0>(srcTensor.shape());
        const uint32_t cols = tla::get<1, 0>(srcTensor.shape()) * tla::get<1, 1>(srcTensor.shape());
        const uint32_t srcDValue = tla::get<0>(srcTensor.stride());
        const uint32_t dstOuterStrideRow = tla::get<0, 1>(dstTensor.stride());

        AscendC::Dn2NzParams intriParams;

        intriParams.dnNum = 1;
        intriParams.nValue = CeilDiv<MX_SCALE_COPY_GROUP_NUM>(cols);
        intriParams.dValue = rows;
        intriParams.srcDnMatrixStride = 0;
        intriParams.srcDValue = CeilDiv<MX_SCALE_COPY_GROUP_NUM>(srcDValue);
        intriParams.dstNzC0Stride = dstOuterStrideRow / BYTE_PER_C0;
        intriParams.dstNzNStride = 1;
        intriParams.dstNzMatrixStride = 0;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        auto srcHalf = srcTensor.data()[srcOffset].template ReinterpretCast<half>();
        auto dstHalf = dstTensor.data()[dstOffset].template ReinterpretCast<half>();

        AscendC::DataCopy(dstHalf, srcHalf, intriParams);
    }
};

/// Partial specialization for CopyGmToL1, Ascend950, fp8_e8m0_t, MxScaleA ColumnMajor in and zZ out.
template <class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<
    Arch::Ascend950,
    tla::Tensor<AscendC::GlobalTensor<float8_e8m0_t>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
    tla::Tensor<AscendC::LocalTensor<float8_e8m0_t>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
    std::enable_if_t<
        tla::detail::isMxScaleATrans<float8_e8m0_t, LayoutSrc>::value &&
        tla::detail::isMxScalezZ<float8_e8m0_t, LayoutDst>::value>> {

    CATLASS_DEVICE
    TileCopyTla() {};

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor)
    {
        static_assert(
            std::is_same_v<typename TensorSrc::Element, float8_e8m0_t> &&
            std::is_same_v<typename TensorDst::Element, float8_e8m0_t> &&
            tla::detail::isMxScaleATrans<typename TensorSrc::Element, typename TensorSrc::Layout>::value &&
            tla::detail::isMxScalezZ<typename TensorDst::Element, typename TensorDst::Layout>::value &&
            TensorSrc::position == AscendC::TPosition::GM && TensorDst::position == AscendC::TPosition::A1,
            "The input parameters do not match. TensorSrc must be fp8_e8m0_t, GM and ColumnMajor, while TensorDst must "
            "be fp8_e8m0_t, L1 and zZ"
        );

        const uint32_t rows = tla::get<0>(srcTensor.shape());
        const uint32_t cols = tla::get<1, 0>(srcTensor.shape()) * tla::get<1, 1>(srcTensor.shape());
        const uint32_t srcDValue = tla::get<1, 1>(srcTensor.stride());
        const uint32_t dstOuterStrideRow = tla::get<0, 1>(dstTensor.stride());

        AscendC::Nd2NzParams intriParams;

        intriParams.ndNum = 1;
        intriParams.nValue = CeilDiv<MX_SCALE_COPY_GROUP_NUM>(cols);
        intriParams.dValue = rows;
        intriParams.srcNdMatrixStride = 0;
        intriParams.srcDValue = CeilDiv<MX_SCALE_COPY_GROUP_NUM>(srcDValue);
        intriParams.dstNzC0Stride = dstOuterStrideRow / BYTE_PER_C0;
        intriParams.dstNzNStride = 1;
        intriParams.dstNzMatrixStride = 0;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        auto srcHalf = srcTensor.data()[srcOffset].template ReinterpretCast<half>();
        auto dstHalf = dstTensor.data()[dstOffset].template ReinterpretCast<half>();

        AscendC::DataCopy(dstHalf, srcHalf, intriParams);
    }
};

/// Partial specialization for CopyGmToL1, Ascend950, fp8_e8m0_t, B RowMajor in and nN out.
template <class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<
    Arch::Ascend950,
    tla::Tensor<AscendC::GlobalTensor<float8_e8m0_t>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
    tla::Tensor<AscendC::LocalTensor<float8_e8m0_t>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
    std::enable_if_t<
        tla::detail::isMxScaleBNoTrans<float8_e8m0_t, LayoutSrc>::value &&
        tla::detail::isMxScalenN<float8_e8m0_t, LayoutDst>::value>> {

    CATLASS_DEVICE
    TileCopyTla() {};

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor)
    {
        static_assert(
            std::is_same_v<typename TensorSrc::Element, float8_e8m0_t> &&
            std::is_same_v<typename TensorDst::Element, float8_e8m0_t> &&
            tla::detail::isMxScaleBNoTrans<typename TensorSrc::Element, typename TensorSrc::Layout>::value &&
            tla::detail::isMxScalenN<typename TensorDst::Element, typename TensorDst::Layout>::value &&
            TensorSrc::position == AscendC::TPosition::GM && TensorDst::position == AscendC::TPosition::A1,
            "The input parameters do not match. TensorSrc must be fp8_e8m0_t, GM and RowMajor, while TensorDst must be "
            "fp8_e8m0_t, L1 and nN"
        );

        const uint32_t rows = tla::get<0, 0>(srcTensor.shape()) * tla::get<0, 1>(srcTensor.shape());
        const uint32_t cols = tla::get<1>(srcTensor.shape());
        const uint32_t srcDValue = tla::get<0, 1>(srcTensor.stride());
        const uint32_t dstOuterStrideCol = tla::get<1, 1>(dstTensor.stride());

        AscendC::Nd2NzParams intriParams;

        intriParams.ndNum = 1;
        intriParams.nValue = CeilDiv<MX_SCALE_COPY_GROUP_NUM>(rows);
        intriParams.dValue = cols;
        intriParams.srcNdMatrixStride = 0;
        intriParams.srcDValue = CeilDiv<MX_SCALE_COPY_GROUP_NUM>(srcDValue);
        intriParams.dstNzC0Stride = dstOuterStrideCol / BYTE_PER_C0;
        intriParams.dstNzNStride = 1;
        intriParams.dstNzMatrixStride = 0;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        auto srcHalf = srcTensor.data()[srcOffset].template ReinterpretCast<half>();
        auto dstHalf = dstTensor.data()[dstOffset].template ReinterpretCast<half>();

        AscendC::DataCopy(dstHalf, srcHalf, intriParams);
    }
};

/// Partial specialization for CopyGmToL1, Ascend950, fp8_e8m0_t, B ColumnMajor in and nN out.
template <class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyTla<
    Arch::Ascend950,
    tla::Tensor<AscendC::GlobalTensor<float8_e8m0_t>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
    tla::Tensor<AscendC::LocalTensor<float8_e8m0_t>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
    std::enable_if_t<
        tla::detail::isMxScaleBTrans<float8_e8m0_t, LayoutSrc>::value &&
        tla::detail::isMxScalenN<float8_e8m0_t, LayoutDst>::value>> {

    CATLASS_DEVICE
    TileCopyTla() {};

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor)
    {
        static_assert(
            std::is_same_v<typename TensorSrc::Element, float8_e8m0_t> &&
            std::is_same_v<typename TensorDst::Element, float8_e8m0_t> &&
            tla::detail::isMxScaleBTrans<typename TensorSrc::Element, typename TensorSrc::Layout>::value &&
            tla::detail::isMxScalenN<typename TensorDst::Element, typename TensorDst::Layout>::value &&
            TensorSrc::position == AscendC::TPosition::GM && TensorDst::position == AscendC::TPosition::A1,
            "The input parameters do not match. TensorSrc must be fp8_e8m0_t, GM and ColumnMajor, while TensorDst must "
            "be fp8_e8m0_t, L1 and nN"
        );

        const uint32_t rows = tla::get<0, 0>(srcTensor.shape()) * tla::get<0, 1>(srcTensor.shape());
        const uint32_t cols = tla::get<1>(srcTensor.shape());
        const uint32_t srcDValue = tla::get<1>(srcTensor.stride());
        const uint32_t dstOuterStrideCol = tla::get<1, 1>(dstTensor.stride());

        AscendC::Dn2NzParams intriParams;

        intriParams.dnNum = 1;
        intriParams.nValue = CeilDiv<MX_SCALE_COPY_GROUP_NUM>(rows);
        intriParams.dValue = cols;
        intriParams.srcDnMatrixStride = 0;
        intriParams.srcDValue = CeilDiv<MX_SCALE_COPY_GROUP_NUM>(srcDValue);
        intriParams.dstNzC0Stride = dstOuterStrideCol / BYTE_PER_C0;
        intriParams.dstNzNStride = 1;
        intriParams.dstNzMatrixStride = 0;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        auto srcHalf = srcTensor.data()[srcOffset].template ReinterpretCast<half>();
        auto dstHalf = dstTensor.data()[dstOffset].template ReinterpretCast<half>();

        AscendC::DataCopy(dstHalf, srcHalf, intriParams);
    }
};

///////////////////////////////////////////TileCopyFAQTla//////////////////////////////////////////////////////
/// used for FA LoadQ, multi matrix datacopy ND->NZ.
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyFAQTla<
    Arch::Ascend950,
    tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A1>>{
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementSrc);

    // Methods

    CATLASS_DEVICE
    TileCopyFAQTla() {};

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor)
    {
        static_assert(
                tla::detail::iszN<typename TensorDst::Element, typename TensorDst::Layout>::value
                && TensorSrc::position == AscendC::TPosition::GM && TensorDst::position == AscendC::TPosition::A1,
            "The input parameters do not match. TensorSrc must be GM, while TensorDst must be L1 and zN"
        );

        const uint32_t nValue = tla::get<1>(srcTensor.shape());
        const uint32_t dValue = tla::get<2>(srcTensor.shape());
        const uint32_t srcDValue = tla::get<1>(srcTensor.stride());
        const uint32_t ndNum = tla::get<0>(srcTensor.shape());
        const uint32_t srcNdMatrixStride = tla::get<0>(srcTensor.stride());
        const uint32_t dstNzNStride = tla::get<0>(srcTensor.shape());
        const uint32_t dstNzMatrixStride = ELE_NUM_PER_C0;
        const uint32_t dstNzC0Stride =  tla::get<0, 0>(dstTensor.shape()) * tla::get<0, 1>(dstTensor.shape());
        const uint32_t dstInnerStrideRow = tla::get<0, 0>(dstTensor.stride());
        const uint32_t dstOuterStrideCol = tla::get<1, 1>(dstTensor.stride());

        AscendC::Nd2NzParams intriParams;

        intriParams.nValue = nValue;
        intriParams.dValue = dValue;
        intriParams.srcDValue = srcDValue;
        intriParams.dstNzNStride = dstNzNStride;
        intriParams.dstNzC0Stride = dstNzC0Stride;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        if (srcNdMatrixStride < STRIDE_LIMIT) {
            intriParams.ndNum = ndNum;
            intriParams.srcNdMatrixStride = srcNdMatrixStride;
            intriParams.dstNzMatrixStride = dstNzMatrixStride;
            AscendC::DataCopy(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], intriParams);
        } else {
            intriParams.ndNum = 1;
            intriParams.srcNdMatrixStride = 0;
            intriParams.dstNzMatrixStride = 0;
            for (uint32_t i = 0; i < nValue; i++) {
                AscendC::DataCopy(dstTensor.data()[dstOffset + i * ELE_NUM_PER_C0],
                    srcTensor.data()[srcOffset + i * srcDValue],
                    intriParams);
            }
        }
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////TileCopyFAGTla//////////////////////////////////////////////////////
/// FAG: RowMajor->zN with mUp/mDown split (cube2/cube3 A).
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyFAGTla<Arch::Ascend950,
    tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
    std::enable_if_t<tla::detail::isRowMajor<LayoutSrc>::value &&
                     tla::detail::iszN<ElementDst, LayoutDst>::value>> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<ElementSrc>::value;

    CATLASS_DEVICE
    TileCopyFAGTla() {};

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE
    void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor)
    {
        int32_t mUp = (tla::get<0>(srcTensor.shape()) + 1) / 2;
        int32_t mDown = tla::get<0>(srcTensor.shape()) - mUp;
        uint32_t mAlign = RoundUp<C0_NUM_PER_FRACTAL>(tla::get<0>(srcTensor.shape()));

        AscendC::Nd2NzParams intriParams;
        intriParams.ndNum = 1;
        intriParams.nValue = mUp;
        intriParams.dValue = tla::get<1>(srcTensor.shape());
        intriParams.srcDValue = tla::get<0>(srcTensor.stride());
        intriParams.dstNzNStride = 1;
        intriParams.dstNzC0Stride = mAlign;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        AscendC::DataCopy(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], intriParams);
        if (mDown) {
            intriParams.nValue = mDown;
            AscendC::DataCopy(dstTensor.data()[dstOffset + mUp * 16], srcTensor.data()[srcOffset + mUp * 128 * 2], intriParams);
        }
    }
};

/// FAG: ColumnMajor->nZ with mUp/mDown split (cube3 B).
template <class ElementSrc, class ElementDst, class LayoutSrc, class LayoutDst, class CoordSrc, class CoordDst>
struct TileCopyFAGTla<Arch::Ascend950,
    tla::Tensor<AscendC::GlobalTensor<ElementSrc>, LayoutSrc, CoordSrc, AscendC::TPosition::GM>,
    tla::Tensor<AscendC::LocalTensor<ElementDst>, LayoutDst, CoordDst, AscendC::TPosition::A1>,
    std::enable_if_t<tla::detail::isColumnMajor<LayoutSrc>::value &&
                     tla::detail::isnZ<ElementDst, LayoutDst>::value>> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<ElementSrc>::value;

    CATLASS_DEVICE
    TileCopyFAGTla() {};

    template <class TensorDst, class TensorSrc>
    CATLASS_DEVICE
    void operator()(TensorDst const &dstTensor, TensorSrc const &srcTensor)
    {
        int32_t mUp = (tla::get<1>(srcTensor.shape()) + 1) / 2;
        int32_t mDown = tla::get<1>(srcTensor.shape()) - mUp;
        uint32_t mAlign = RoundUp<C0_NUM_PER_FRACTAL>(tla::get<1>(srcTensor.shape()));

        AscendC::Nd2NzParams intriParams;
        intriParams.ndNum = 1;
        intriParams.nValue = mUp;
        intriParams.dValue = tla::get<0>(srcTensor.shape());
        intriParams.srcDValue = tla::get<1>(srcTensor.stride());
        intriParams.dstNzNStride = 1;
        intriParams.dstNzC0Stride = mAlign;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        AscendC::DataCopy(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], intriParams);
        if (mDown) {
            intriParams.nValue = mDown;
            AscendC::DataCopy(dstTensor.data()[dstOffset + mUp * 16], srcTensor.data()[srcOffset + mUp * 128 * 2], intriParams);
        }
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace Catlass::Gemm::Tile

#endif // CATLASS_GEMM_TILE_COPY_GM_TO_L1_A5_HPP
