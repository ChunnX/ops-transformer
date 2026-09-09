/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_MATRIX_DENSE_HPP
#define MEGA_MOE_MATRIX_DENSE_HPP

#include "../../template_linear_algebra_v2/mega_moe_catlass.hpp"
#include "../../template_linear_algebra_v2/tla_v2_coord.hpp"
#include "../../template_linear_algebra_v2/tla_v2_numeric_size.hpp"
#include "../../template_linear_algebra_v2/detail/tla_v2_alignment.hpp"
#include "../../template_linear_algebra_v2/mega_moe_matrix_coord.hpp"
#include "../../template_linear_algebra_v2/tla_v2_conv_coord.hpp"

namespace Catlass::layout {

/// Mapping function for row-major matrices
struct RowMajor {
public:
    /// Logical rank of tensor
    static constexpr int RANK = 2;

    /// Index type used for coordinates
    using Index = uint32_t;

    /// Long index type used for offsets
    using LongIndex = int64_t;

    /// Logical coordinate
    using Shape = Coord<RANK, Index>;

    /// Stride vector
    using Stride = Coord<RANK, LongIndex>;

public:
    /// Constructor
    CATLASS_HOST_DEVICE
    RowMajor(Index rows = 0, Index cols = 0)
        : shape_(MakeCoord(rows, cols)),
          stride_(MakeCoord(LongIndex(cols), LongIndex(1)))
    {}

    /// Constructor
    CATLASS_HOST_DEVICE
    RowMajor(Index rows, Index cols, LongIndex ldm)
        : shape_(MakeCoord(rows, cols)),
          stride_(MakeCoord(ldm, LongIndex(1)))
    {}

    /// Ctor
    CATLASS_HOST_DEVICE
    RowMajor(Shape shape, Stride stride)
        : shape_(shape),
          stride_(stride)
    {}

    template <class Element>
    CATLASS_HOST_DEVICE static RowMajor MakeLayout(Index rows, Index cols)
    {
        return RowMajor(rows, cols);
    }

    template <class Element>
    CATLASS_HOST_DEVICE static RowMajor MakeLayoutInUb(MatrixCoord const &shape)
    {
        constexpr uint32_t ELE_NUM_PER_BLK = BytesToBits(BYTE_PER_BLK) / SizeOfBits<Element>::value;
        return RowMajor(shape.row(), shape.column(), RoundUp<ELE_NUM_PER_BLK>(shape.column()));
    }

    /// Returns the offset of a coordinate in linear memory.
    /// Assumes coordinate has convention (row, column)
    CATLASS_HOST_DEVICE
    LongIndex GetOffset(MatrixCoord const &coord) const
    {
        return LongIndex(coord.row()) * stride_[0] + LongIndex(coord.column());
    }

    /// Returns the layout of a tile.
    CATLASS_HOST_DEVICE
    RowMajor GetTileLayout(MatrixCoord const &tileShape) const
    {
        return RowMajor(tileShape, stride());
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

    /// Returns the length of the layout
    CATLASS_HOST_DEVICE
    LongIndex Capacity() const
    {
        return static_cast<LongIndex>(shape_[0]) * stride_[0];
    }

protected:
    //
    // Data members
    //

    /// Shape data member
    Shape shape_;

    /// Stride data member
    Stride stride_;
};

/// Mapping function for col-major matrices
struct ColumnMajor {
public:
    /// Logical rank of tensor
    static constexpr int RANK = 2;

    /// Index type used for coordinates
    using Index = uint32_t;

    /// Long index type used for offsets
    using LongIndex = int64_t;

    /// Logical coordinate
    using Shape = Coord<RANK, Index>;

    /// Stride vector
    using Stride = Coord<RANK, LongIndex>;

public:
    // Methods

    /// Constructor
    CATLASS_HOST_DEVICE
    ColumnMajor(Index rows = 0, Index cols = 0)
        : shape_(MakeCoord(rows, cols)),
          stride_(MakeCoord(LongIndex(1), LongIndex(rows)))
    {}

    /// Constructor
    CATLASS_HOST_DEVICE
    ColumnMajor(Index rows, Index cols, LongIndex ldm)
        : shape_(MakeCoord(rows, cols)),
          stride_(MakeCoord(LongIndex(1), ldm))
    {}

    /// Ctor
    CATLASS_HOST_DEVICE
    ColumnMajor(Shape shape, Stride stride)
        : shape_(shape),
          stride_(stride)
    {}

    template <class Element>
    CATLASS_HOST_DEVICE static ColumnMajor MakeLayout(Index rows, Index cols)
    {
        return ColumnMajor(rows, cols);
    }

    /// Returns the offset of a coordinate in linear memory.
    /// Assumes coordinate has convention (row, column)
    CATLASS_HOST_DEVICE
    LongIndex GetOffset(MatrixCoord const &coord) const
    {
        return LongIndex(coord.row()) + LongIndex(coord.column()) * stride_[1];
    }

    /// Returns the layout of a tile.
    CATLASS_HOST_DEVICE
    ColumnMajor GetTileLayout(MatrixCoord const &tileShape) const
    {
        return ColumnMajor(tileShape, stride());
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

    /// Returns the length of the layout
    CATLASS_HOST_DEVICE
    LongIndex Capacity() const
    {
        return static_cast<LongIndex>(shape_[1]) * stride_[1];
    }

protected:
    //
    // Data members
    //

    /// Shape data member
    Shape shape_;

    /// Stride data member
    Stride stride_;
};

} // namespace Catlass::layout

#endif // MEGA_MOE_MATRIX_DENSE_HPP
