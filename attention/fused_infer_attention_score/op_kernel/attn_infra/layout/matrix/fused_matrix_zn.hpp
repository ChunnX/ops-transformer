/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSED_MATRIX_ZN_HPP
#define FUSED_MATRIX_ZN_HPP

#include "../../../attn_infra/fused_base_defs.hpp"
#include "../../../attn_infra/fused_coord.hpp"
#include "../../../attn_infra/detail/fused_alignment.hpp"
#include "../../../attn_infra/fused_matrix_coord.hpp"

namespace NpuArch::layout
{

struct zN {
public:
    static constexpr int RANK = 4;

    using Index = uint32_t;

    using LongIndex = int64_t;

    static constexpr int ORG_SHAPE_RANK = 2;

    using OrgShape = Coord<ORG_SHAPE_RANK, Index>;

    using Shape = Coord<RANK, Index>;

    using Stride = Coord<RANK, LongIndex>;

public:
    HOST_DEVICE constexpr
    zN(Index orgRows = 0,
       Index orgCols = 0,
       Index rowsInFractal = 0,
       Index rowsByFractal = 0,
       Index colsInFractal = 0,
       Index colsByFractal = 0,
       LongIndex strideRowsInFractal = 0,
       LongIndex strideRowsByFractal = 0,
       LongIndex strideColsInFractal = 0,
       LongIndex strideColsByFractal = 0)
        : orgShape_(MakeCoord(orgRows, orgCols)),
          shape_(MakeCoord(rowsInFractal, rowsByFractal, colsInFractal, colsByFractal)),
          stride_(MakeCoord(strideRowsInFractal, strideRowsByFractal, strideColsInFractal, strideColsByFractal)) {}

    HOST_DEVICE constexpr
    zN(OrgShape orgShape, Shape shape, Stride stride) : orgShape_(orgShape), shape_(shape), stride_(stride) {}

    template <class Element>
    HOST_DEVICE constexpr
    static zN MakeLayout(Index orgRows, Index orgCols)
    {
        constexpr uint32_t ELE_NUM_PER_C0 = static_cast<uint32_t>(BYTE_PER_C0) / static_cast<uint32_t>(sizeof(Element));
        constexpr uint32_t ELE_NUM_PER_FRACTAL = BYTE_PER_FRACTAL / sizeof(Element);
        Index rowsRound = NpuArch::Detail::Alignment::RoundUp<C0_NUM_PER_FRACTAL>(orgRows);
        Index colsRound = NpuArch::Detail::Alignment::RoundUp<ELE_NUM_PER_C0>(orgCols);
        return zN(orgRows,
                  orgCols,
                  C0_NUM_PER_FRACTAL,
                  rowsRound / C0_NUM_PER_FRACTAL,
                  ELE_NUM_PER_C0,
                  colsRound / ELE_NUM_PER_C0,
                  ELE_NUM_PER_C0,
                  ELE_NUM_PER_FRACTAL,
                  1,
                  rowsRound * ELE_NUM_PER_C0);
    }

    HOST_DEVICE
    static zN MakeLayoutInL0C(MatrixCoord const &shape)
    {
        return zN(shape.row(),
                  shape.column(),
                  C0_NUM_PER_FRACTAL,
                  NpuArch::Detail::Alignment::CeilDiv<C0_NUM_PER_FRACTAL>(shape.row()),
                  C0_NUM_PER_FRACTAL,
                  NpuArch::Detail::Alignment::CeilDiv<C0_NUM_PER_FRACTAL>(shape.column()),
                  C0_NUM_PER_FRACTAL,
                  C0_NUM_PER_FRACTAL * C0_NUM_PER_FRACTAL,
                  1,
                  NpuArch::Detail::Alignment::RoundUp<C0_NUM_PER_FRACTAL>(shape.row()) * C0_NUM_PER_FRACTAL);
    }

    HOST_DEVICE
    LongIndex GetOffset(MatrixCoord const &coord) const
    {
        return LongIndex(coord.row()) / shape_[0] * stride_[1] + LongIndex(coord.column()) / shape_[2] * stride_[3] +
            (LongIndex(coord.row()) % shape_[0]) * stride_[0] + (LongIndex(coord.column()) % shape_[2]) * stride_[2];
    }

    HOST_DEVICE
    zN GetTileLayout(MatrixCoord const &tileOriShape) const
    {
        auto tileShape = MakeCoord(
            shape(0), NpuArch::Detail::Alignment::CeilDiv(tileOriShape.row(), shape(0)),
            shape(2), NpuArch::Detail::Alignment::CeilDiv(tileOriShape.column(), shape(2))
        );
        return zN(tileOriShape, tileShape, stride());
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

#endif  // FUSED_MATRIX_ZN_HPP