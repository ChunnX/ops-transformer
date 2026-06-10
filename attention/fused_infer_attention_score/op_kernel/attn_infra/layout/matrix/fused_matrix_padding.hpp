/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSED_MATRIX_PADDING_HPP
#define FUSED_MATRIX_PADDING_HPP

#include "../../../attn_infra/fused_base_defs.hpp"
#include "../../../attn_infra/fused_coord.hpp"
#include "../../../attn_infra/detail/fused_alignment.hpp"
#include "../../../attn_infra/fused_matrix_coord.hpp"

namespace NpuArch::layout
{

struct PaddingRowMajor {
public:
    static constexpr int RANK = 4;

    static constexpr int ORG_SHAPE_RANK = 2;

    using Index = uint32_t;

    using LongIndex = int64_t;

    using OrgShape = Coord<ORG_SHAPE_RANK, Index>;

    using Shape = Coord<RANK, Index>;

    using Stride = Coord<RANK, LongIndex>;

public:
    HOST_DEVICE
    PaddingRowMajor(Index orgRows = 0, Index orgCols = 0, Index blockRows = 0, Index blockCols = 0) :
        orgShape_(MakeCoord(orgRows, orgCols)),
        shape_(MakeCoord(blockRows, NpuArch::Detail::Alignment::CeilDiv(orgRows, blockRows),
        blockCols, NpuArch::Detail::Alignment::CeilDiv(orgCols, blockCols))),
        stride_(MakeCoord((LongIndex)blockCols,
        (LongIndex)blockRows * (LongIndex)NpuArch::Detail::Alignment::RoundUp(orgCols, blockCols),
        (LongIndex)1, (LongIndex)blockRows * (LongIndex)blockCols)) {}

    HOST_DEVICE
    LongIndex GetOffset(MatrixCoord const &coord) const
    {
        LongIndex blockRows = (LongIndex)shape_[0];
        LongIndex blockCols = (LongIndex)shape_[2];
        return (LongIndex)coord.row() / blockRows * stride_[1]
            + (LongIndex)coord.column() / blockCols * stride_[3]
            + (LongIndex)coord.row() % blockRows * stride_[0]
            + (LongIndex)coord.column() % blockCols;
    }

    HOST_DEVICE
    PaddingRowMajor GetTileLayout(MatrixCoord const &tileShape) const
    {
        return PaddingRowMajor(tileShape.row(), tileShape.column(), shape_[0], shape_[2]);
    }

    HOST_DEVICE
    typename OrgShape::Index orgShape(int idx) const
    {
        return orgShape_[idx];
    }

    HOST_DEVICE
    typename OrgShape::Index &orgShape(int idx)
    {
        return orgShape_[idx];
    }

    HOST_DEVICE
    Shape shape() const
    {
        return shape_;
    }

    HOST_DEVICE
    Shape &shape()
    {
        return shape_;
    }

    HOST_DEVICE
    typename Shape::Index shape(int idx) const
    {
        return shape_[idx];
    }

    HOST_DEVICE
    typename Shape::Index &shape(int idx)
    {
        return shape_[idx];
    }

    HOST_DEVICE
    Stride stride() const
    {
        return stride_;
    }

    HOST_DEVICE
    Stride &stride()
    {
        return stride_;
    }

    HOST_DEVICE
    typename Stride::Index stride(int idx) const
    {
        return stride_[idx];
    }

    HOST_DEVICE
    typename Stride::Index &stride(int idx)
    {
        return stride_[idx];
    }

private:
    OrgShape orgShape_;

    Shape shape_;

    Stride stride_;
};

struct PaddingColumnMajor {
public:
    static constexpr int RANK = 4;

    static constexpr int ORG_SHAPE_RANK = 2;

    using Index = uint32_t;

    using LongIndex = int64_t;

    using OrgShape = Coord<ORG_SHAPE_RANK, Index>;

    using Shape = Coord<RANK, Index>;

    using Stride = Coord<RANK, LongIndex>;

public:
    HOST_DEVICE
    PaddingColumnMajor(Index orgRows = 0, Index orgCols = 0, Index blockRows = 0, Index blockCols = 0) :
        orgShape_(MakeCoord(orgRows, orgCols)),
        shape_(MakeCoord(blockRows, NpuArch::Detail::Alignment::CeilDiv(orgRows, blockRows),
        blockCols, NpuArch::Detail::Alignment::CeilDiv(orgCols, blockCols))),
        stride_(MakeCoord((LongIndex)1, (LongIndex)blockRows * (LongIndex)blockCols, (LongIndex)blockRows,
        (LongIndex)NpuArch::Detail::Alignment::RoundUp(orgRows, blockRows) * (LongIndex)blockCols)) {}

    HOST_DEVICE
    LongIndex GetOffset(MatrixCoord const &coord) const
    {
        LongIndex blockRows = (LongIndex)shape_[0];
        LongIndex blockCols = (LongIndex)shape_[2];
        return (LongIndex)coord.row() / blockRows * stride_[1]
            + (LongIndex)coord.column() / blockCols * stride_[3]
            + (LongIndex)coord.row() % blockRows
            + (LongIndex)coord.column() % blockCols * stride_[2];
    }

    HOST_DEVICE
    PaddingColumnMajor GetTileLayout(MatrixCoord const &tileShape) const
    {
        return PaddingColumnMajor(tileShape.row(), tileShape.column(), shape_[0], shape_[2]);
    }

    HOST_DEVICE
    typename OrgShape::Index orgShape(int idx) const
    {
        return orgShape_[idx];
    }

    HOST_DEVICE
    typename OrgShape::Index &orgShape(int idx)
    {
        return orgShape_[idx];
    }

    HOST_DEVICE
    Shape shape() const
    {
        return shape_;
    }

    HOST_DEVICE
    Shape &shape()
    {
        return shape_;
    }

    HOST_DEVICE
    typename Shape::Index shape(int idx) const
    {
        return shape_[idx];
    }

    HOST_DEVICE
    typename Shape::Index &shape(int idx)
    {
        return shape_[idx];
    }

    HOST_DEVICE
    Stride stride() const
    {
        return stride_;
    }

    HOST_DEVICE
    Stride &stride()
    {
        return stride_;
    }

    HOST_DEVICE
    typename Stride::Index stride(int idx) const
    {
        return stride_[idx];
    }

    HOST_DEVICE
    typename Stride::Index &stride(int idx)
    {
        return stride_[idx];
    }

private:
    OrgShape orgShape_;

    Shape shape_;

    Stride stride_;
};

}  // namespace NpuArch::layout

#endif  // FUSED_MATRIX_PADDING_HPP