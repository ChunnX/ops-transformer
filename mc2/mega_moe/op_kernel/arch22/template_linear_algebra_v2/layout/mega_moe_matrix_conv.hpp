/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_MATRIX_CONV_HPP
#define MEGA_MOE_MATRIX_CONV_HPP

#include "../../template_linear_algebra_v2/mega_moe_catlass.hpp"
#include "../../template_linear_algebra_v2/tla_v2_coord.hpp"
#include "../../template_linear_algebra_v2/tla_v2_numeric_size.hpp"
#include "../../template_linear_algebra_v2/detail/tla_v2_alignment.hpp"
#include "../../template_linear_algebra_v2/mega_moe_matrix_coord.hpp"
#include "../../template_linear_algebra_v2/tla_v2_conv_coord.hpp"

namespace Catlass::layout {

struct NDC1HWC0 {
public:
    /// Index type used for coordinates
    using Index = uint32_t;

    /// Long index type used for offsets
    using LongIndex = int64_t;

    /// Logical rank of orgshape
    /// (N,D,C1,H,W,C0)
    static constexpr int ORG_SHAPE_RANK = 6;

    static constexpr int RANK = 5;
    using OrgShape = Coord<ORG_SHAPE_RANK, Index>;
    /// Logical coordinate
    using Shape = Coord<RANK, Index>;

    /// Stride vector
    using Stride = Coord<RANK, LongIndex>;

public:
    // Methods

    /// Constructor
    CATLASS_HOST_DEVICE constexpr NDC1HWC0(Index batch = 0, Index D = 0, Index C1 = 0, Index H = 0, Index W = 0,
                                           Index C0 = 0,

                                           Index rowsInFractal = 0, /// Number of rows inside the fractal
                                           Index rowsByFractal = 0, /// number of rows by the fractal
                                           Index colsInFractal = 0, /// number of cols inside the fractal
                                           Index colsByFractal = 0, /// number of cols by the fractal

                                           LongIndex strideC0 = 0, /// number of elements between adjacent C0 cols
                                           LongIndex strideHW = 0, /// number of elements between adjacent W rows
                                           LongIndex StrideC1 = 0, /// number of elements between adjacent C1 cols
                                           LongIndex StrideD = 0,  /// number of elements between adjacent D batchCols
                                           LongIndex StrideN = 0   /// number of elements between adjacent batch
                                           )
        : orgShape_(MakeCoord(batch, D, C1, H, W, C0)),
          shape_(MakeCoord(batch, rowsInFractal, rowsByFractal, colsInFractal, colsByFractal)),
          stride_(MakeCoord(strideC0, strideHW, StrideC1, StrideD, StrideN))
    {}

    /// Ctor
    CATLASS_HOST_DEVICE constexpr NDC1HWC0(OrgShape orgshape, Shape shape, Stride stride)
        : orgShape_(orgshape),
          shape_(shape),
          stride_(stride)
    {}

    CATLASS_HOST_DEVICE constexpr static NDC1HWC0 MakeLayout(Index Batch, Index D, Index C1, Index H, Index W, Index C0)
    {
        return NDC1HWC0(Batch, D, C1, H, W, C0,

                        W, H, C0, D * C1,

                        1,                  /// StrideC0
                        C0,                 /// StrideHW
                        H * W * C0,         /// StrideC1
                        H * W * C0 * C1,    /// StrideD
                        H * W * C0 * C1 * D /// StrideN
        );
    }

    // CATLASS_HOST_DEVICE
    /// Returns the offset of a coordinate in linear memory.
    CATLASS_HOST_DEVICE
    LongIndex GetOffset(Conv3d6HdCoord const &coord) const
    {
        return LongIndex(coord.n()) * stride_[4] + LongIndex(coord.d()) * stride_[3] +
               LongIndex(coord.c1()) * stride_[2] + LongIndex(coord.hw()) * stride_[1];
    }

    /// Returns the layout of a tile.
    CATLASS_HOST_DEVICE
    NDC1HWC0 GetTileLayout(OrgShape const &tileOriShape) const
    {
        Shape tileShape =
            MakeCoord(tileOriShape[0], tileOriShape[4], tileOriShape[3], shape(3), tileOriShape[1] * tileOriShape[2]);

        Stride tileStride =
            MakeCoord(stride(0), stride(1), (LongIndex)(tileOriShape[3] * tileOriShape[4] * shape(3)),
                      (LongIndex)(tileOriShape[2] * tileOriShape[3] * tileOriShape[4] * shape(3)),
                      (LongIndex)(tileOriShape[1] * tileOriShape[2] * tileOriShape[3] * tileOriShape[4] * shape(3)));
        return NDC1HWC0(tileOriShape, tileShape, tileStride);
    }

    /// Returns the origin shape of the layout
    CATLASS_HOST_DEVICE
    typename OrgShape::Index orgShape(int idx) const
    {
        return orgShape_[idx];
    }

    /// Returns the origin shape of the layout
    CATLASS_HOST_DEVICE
    typename OrgShape::Index &orgShape(int idx)
    {
        return orgShape_[idx];
    }

    /// Returns the shape of the layout
    CATLASS_HOST_DEVICE
    Shape shape() const
    {
        return shape_;
    }

    /// Returns the shape of the layout
    CATLASS_HOST_DEVICE
    Shape &shape()
    {
        return shape_;
    }

    /// Returns the shape of the layout
    CATLASS_HOST_DEVICE
    typename Shape::Index shape(int idx) const
    {
        return shape_[idx];
    }

    /// Returns the shape of the layout
    CATLASS_HOST_DEVICE
    typename Shape::Index &shape(int idx)
    {
        return shape_[idx];
    }

    /// Returns the stride of the layout
    CATLASS_HOST_DEVICE
    Stride stride() const
    {
        return stride_;
    }

    /// Returns the stride of the layout
    CATLASS_HOST_DEVICE
    Stride &stride()
    {
        return stride_;
    }

    /// Returns the stride of the layout
    CATLASS_HOST_DEVICE
    typename Stride::Index stride(int idx) const
    {
        return stride_[idx];
    }

    /// Returns the stride of the layout
    CATLASS_HOST_DEVICE
    typename Stride::Index &stride(int idx)
    {
        return stride_[idx];
    }

private:
    /// Origin Shape data member
    OrgShape orgShape_;

    /// Shape data member
    Shape shape_;

    /// Stride data member
    Stride stride_;
};

struct KDC1KHKWN1N0C0 {
public:
    static constexpr int RANK = 4;

    /// Index type used for coordinates
    using Index = uint32_t;

    /// Long index type used for offsets
    using LongIndex = int64_t;

    /// Logical rank of orgshape
    static constexpr int ORG_SHAPE_RANK = 4;

    /// Logical coordinate
    using OrgShape = Coord<ORG_SHAPE_RANK, Index>;

    /// Logical coordinate
    using Shape = Coord<RANK, Index>;

    /// Stride vector
    using Stride = Coord<RANK, LongIndex>;

public:
    // Methods

    /// Constructor
    CATLASS_HOST_DEVICE constexpr KDC1KHKWN1N0C0(
        Index KdC1KhKw = 0, /// Merging Kd,Kh,Kw,C1 axes of KDC1KHKWN1N0C0
        Index N1 = 0,       /// Cout = N1*N0
        Index N0 = 0, Index C0 = 0,

        Index rowsInFractal = 0, /// Number of rows inside the fractal
        Index rowsByFractal = 0, /// number of rows by the fractal
        Index colsInFractal = 0, /// number of cols inside the fractal
        Index colsByFractal = 0, /// number of cols by the fractal

        LongIndex strideC0 = 0,    /// number of elements between adjacent rows inside the fractal
        LongIndex StrideDC1HW = 0, /// number of elements between adjacent fractal rows
        LongIndex strideN0 = 0,    /// number of elements between adjacent cols inside the fractal
        LongIndex strideN1 = 0     /// number of elements between adjacent fractal cols
        )
        : orgShape_(MakeCoord(KdC1KhKw, N1, N0, C0)),
          shape_(MakeCoord(rowsInFractal, rowsByFractal, colsInFractal, colsByFractal)),
          stride_(MakeCoord(strideC0, strideN0, strideN1, StrideDC1HW))
    {}

    /// Ctor
    CATLASS_HOST_DEVICE constexpr KDC1KHKWN1N0C0(OrgShape orgShape, Shape shape, Stride stride)
        : orgShape_(orgShape),
          shape_(shape),
          stride_(stride)
    {}

    /// Make the layout of a coordinate (Kd*C1*Kh*Kw,N1,N0,C0)
    CATLASS_HOST_DEVICE constexpr static KDC1KHKWN1N0C0 MakeLayout(Index KdC1KhKw, Index N1, Index N0, Index C0)
    {
        return KDC1KHKWN1N0C0(KdC1KhKw, N1, N0, C0,

                              C0, KdC1KhKw, N0, N1,

                              1,            /// StrideC0
                              C0 * N0 * N1, /// StrideDC1HW
                              C0,           /// StrideN0
                              C0 * N0       /// StrideN1
        );
    }

    /// Returns the offset of a coordinate in linear memory.
    /// Assumes coordinate has convention (KdC1KhKw_idx, N1_idx)
    CATLASS_HOST_DEVICE
    LongIndex GetOffset(Conv3dFracZ3dCoord const &coord) const
    {
        return LongIndex(coord.kdc1khkw()) * stride_[3] + LongIndex(coord.n1()) * stride_[1];
    }

    /// Returns the layout of a tile.
    CATLASS_HOST_DEVICE
    KDC1KHKWN1N0C0 GetTileLayout(OrgShape const &tileOriShape) const
    {
        Shape tileShape = MakeCoord(shape(0),        /// C0
                                    tileOriShape[0], /// Kd*C1*Kh*Kw
                                    shape(2),        /// N0
                                    tileOriShape[1]  /// N1
        );
        Stride tileStride = MakeCoord(stride(0),                                     /// TileStrideC0
                                      stride(2) * tileOriShape[1] * tileOriShape[2], /// TileStrideDC1HW
                                      (LongIndex)shape(0),                           /// TileStrideN0
                                      stride(2) * tileOriShape[2]                    /// TileStrideN1
        );
        return KDC1KHKWN1N0C0(tileOriShape, tileShape, tileStride);
    }

    /// Returns the origin shape of the layout
    CATLASS_HOST_DEVICE
    typename OrgShape::Index orgShape(int idx) const
    {
        return orgShape_[idx];
    }

    /// Returns the origin shape of the layout
    CATLASS_HOST_DEVICE
    typename OrgShape::Index &orgShape(int idx)
    {
        return orgShape_[idx];
    }

    /// Returns the shape of the layout
    CATLASS_HOST_DEVICE
    Shape shape() const
    {
        return shape_;
    }

    /// Returns the shape of the layout
    CATLASS_HOST_DEVICE
    Shape &shape()
    {
        return shape_;
    }

    /// Returns the shape of the layout
    CATLASS_HOST_DEVICE
    typename Shape::Index shape(int idx) const
    {
        return shape_[idx];
    }

    /// Returns the shape of the layout
    CATLASS_HOST_DEVICE
    typename Shape::Index &shape(int idx)
    {
        return shape_[idx];
    }

    /// Returns the stride of the layout
    CATLASS_HOST_DEVICE
    Stride stride() const
    {
        return stride_;
    }

    /// Returns the stride of the layout
    CATLASS_HOST_DEVICE
    Stride &stride()
    {
        return stride_;
    }

    /// Returns the stride of the layout
    CATLASS_HOST_DEVICE
    typename Stride::Index stride(int idx) const
    {
        return stride_[idx];
    }

    /// Returns the stride of the layout
    CATLASS_HOST_DEVICE
    typename Stride::Index &stride(int idx)
    {
        return stride_[idx];
    }

private:
    /// Origin Shape data member
    OrgShape orgShape_;

    /// Shape data member
    Shape shape_;

    /// Stride data member
    Stride stride_;
};

} // namespace Catlass::layout

#endif // MEGA_MOE_MATRIX_CONV_HPP
