/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TLA_V2_CONV2D_PARAMS_HPP
#define TLA_V2_CONV2D_PARAMS_HPP

#include "../template_linear_algebra_v2/tla_v2_coord.hpp"
#include "tla_v2_conv2d_coord.hpp"

namespace Catlass {

class Conv2dFilterParams {
public:
    typedef uint8_t ShortIndex;
    typedef uint32_t Index;
    static constexpr uint32_t C0 = 16;
    using Ks = Coord<2, ShortIndex>;
    using Pads = Coord<4, ShortIndex>;
    using Strides = Coord<2, ShortIndex>;
    using Dilations = Coord<2, ShortIndex>;

private:
    Ks ks;
    Pads pads;
    Strides strides;
    Dilations dilations;

public:
    Conv2dFilterParams(ShortIndex kh = 0, ShortIndex kw = 0, ShortIndex padLeft = 0, ShortIndex padRight = 0,
                       ShortIndex padTop = 0, ShortIndex padBottom = 0, ShortIndex strideH = 0, ShortIndex strideW = 0,
                       ShortIndex dilationH = 0, ShortIndex dilationW = 0)
        : ks(MakeCoord(kh, kw)),
          pads(MakeCoord(padLeft, padRight, padTop, padBottom)),
          strides(MakeCoord(strideH, strideW)),
          dilations(MakeCoord(dilationH, dilationW))
    {}

    CATLASS_HOST_DEVICE
    ShortIndex const &kh() const
    {
        return this->ks[0];
    }
    CATLASS_HOST_DEVICE
    ShortIndex &kh()
    {
        return this->ks[0];
    }

    CATLASS_HOST_DEVICE
    ShortIndex const &kw() const
    {
        return this->ks[1];
    }
    CATLASS_HOST_DEVICE
    ShortIndex &kw()
    {
        return this->ks[1];
    }

    CATLASS_HOST_DEVICE
    ShortIndex const &padLeft() const
    {
        return this->pads[0];
    }
    CATLASS_HOST_DEVICE
    ShortIndex const &padLeft()
    {
        return this->pads[0];
    }

    CATLASS_HOST_DEVICE
    ShortIndex const &padRight() const
    {
        return this->pads[1];
    }
    CATLASS_HOST_DEVICE
    ShortIndex const &padRight()
    {
        return this->pads[1];
    }

    CATLASS_HOST_DEVICE
    ShortIndex const &padTop() const
    {
        return this->pads[2];
    }
    CATLASS_HOST_DEVICE
    ShortIndex const &padTop()
    {
        return this->pads[2];
    }

    CATLASS_HOST_DEVICE
    ShortIndex const &padBottom() const
    {
        return this->pads[3];
    }
    CATLASS_HOST_DEVICE
    ShortIndex const &padBottom()
    {
        return this->pads[3];
    }

    CATLASS_HOST_DEVICE
    ShortIndex const &strideH() const
    {
        return this->strides[0];
    }
    CATLASS_HOST_DEVICE
    ShortIndex const &strideH()
    {
        return this->strides[0];
    }

    CATLASS_HOST_DEVICE
    ShortIndex const &strideW() const
    {
        return this->strides[1];
    }
    CATLASS_HOST_DEVICE
    ShortIndex const &strideW()
    {
        return this->strides[1];
    }

    CATLASS_HOST_DEVICE
    ShortIndex const &dilationH() const
    {
        return this->dilations[0];
    }
    CATLASS_HOST_DEVICE
    ShortIndex const &dilationH()
    {
        return this->dilations[0];
    }

    CATLASS_HOST_DEVICE
    ShortIndex const &dilationW() const
    {
        return this->dilations[1];
    }
    CATLASS_HOST_DEVICE
    ShortIndex const &dilationW()
    {
        return this->dilations[1];
    }
};

struct Conv2dParams {
public:
    typedef uint8_t ShortIndex;
    typedef uint32_t Index;
    static constexpr uint32_t C0 = 16;
    using Pads = Coord<4, ShortIndex>;
    using Strides = Coord<2, ShortIndex>;
    using Dilations = Coord<2, ShortIndex>;

private:
    // Batch, Hi, Wi, Cin, Cout, Kh, Kw
    Conv2dFmapCoord fmapShape;     // {Batch, Cin1, Hi, Wi, C0}
    Conv2dFilterCoord filterShape; // {Cin1, Kh, Kw, Cout, C0}
    Conv2dFmapCoord outputShape;   // {Batch, Cout1, Ho, Wo, C0}
    Conv2dFilterParams configs;    // {Ks, Pads, Strides, Dilations}
    Conv2dCoord postIm2colShape;   // {Batch, Ho, Wo, Cout, Cin1}
public:
    /// Default ctor
    CATLASS_HOST_DEVICE
    Conv2dParams() {}

    CATLASS_HOST_DEVICE
    Conv2dParams(Index batch, Index hi, Index wi, Index cin, Index cout, ShortIndex kh, ShortIndex kw,
                 ShortIndex padLeft, ShortIndex padRight, ShortIndex padTop, ShortIndex padBottom, ShortIndex strideH,
                 ShortIndex strideW, ShortIndex dilationH, ShortIndex dilationW)
        : fmapShape(MakeCoord(batch, CeilDiv(cin, C0), hi, wi, C0)),
          filterShape(MakeCoord(CeilDiv(cin, C0), (Index)kh, (Index)kw, cout, C0)),
          configs(kh, kw, padLeft, padRight, padTop, padBottom, strideH, strideW, dilationH, dilationW)
    {
        Index cout1 = CeilDiv(cout, C0);
        Index ho = (hi + padTop + padBottom - dilationH * (kh - 1) - 1) / strideH + 1;
        Index wo = (wi + padLeft + padRight - dilationW * (kw - 1) - 1) / strideW + 1;
        outputShape = MakeCoord(batch, cout1, ho, wo, C0);
        postIm2colShape = MakeCoord(batch, ho, wo, cout, filterShape.cin1());
    }

    CATLASS_HOST_DEVICE
    static Conv2dParams MakeConv2dParams(const Index *dataSizes,        // [Batch, Hi, Wi, Cin, Cout]
                                         const ShortIndex *filterSizes, // [Kh, Kw]
                                         const ShortIndex *pads,        // [padLeft, padRight, padTop, padBottom]
                                         const ShortIndex *strides,     // [strideH, strideW]
                                         const ShortIndex *dilations)   // [dilationH, dilationW]
    {
        return Conv2dParams(dataSizes[0], dataSizes[1], dataSizes[2], dataSizes[3], dataSizes[4], filterSizes[0],
                            filterSizes[1], pads[0], pads[1], pads[2], pads[3], strides[0], strides[1], strides[2],
                            strides[3]);
    }

    CATLASS_HOST_DEVICE
    Conv2dFilterParams const &getFilterParams() const
    {
        return this->configs;
    }

    CATLASS_HOST_DEVICE
    Conv2dFmapCoord const &getOutputShape() const
    {
        return this->outputShape;
    }

    CATLASS_HOST_DEVICE
    Conv2dCoord const &getPostIm2colShape() const
    {
        return this->postIm2colShape;
    }

    CATLASS_HOST_DEVICE
    Index const &batch() const
    {
        return this->fmapShape.batch();
    }
    CATLASS_HOST_DEVICE
    Index &batch()
    {
        return this->fmapShape.batch();
    }

    CATLASS_HOST_DEVICE
    Index const &hi() const
    {
        return this->fmapShape.h();
    }
    CATLASS_HOST_DEVICE
    Index &hi()
    {
        return this->fmapShape.h();
    }

    CATLASS_HOST_DEVICE
    Index const &wi() const
    {
        return this->fmapShape.w();
    }
    CATLASS_HOST_DEVICE
    Index &wi()
    {
        return this->fmapShape.w();
    }

    CATLASS_HOST_DEVICE
    Index cin() const
    {
        return this->fmapShape.c1() * C0;
    }

    CATLASS_HOST_DEVICE
    Index const &cin1() const
    {
        return this->fmapShape.c1();
    }
    CATLASS_HOST_DEVICE
    Index &cin1()
    {
        return this->fmapShape.c1();
    }

    CATLASS_HOST_DEVICE
    Index const &cout() const
    {
        return this->filterShape.cout();
    }
    CATLASS_HOST_DEVICE
    Index &cout()
    {
        return this->filterShape.cout();
    }

    CATLASS_HOST_DEVICE
    Index const &cout1() const
    {
        return this->outputShape.c1();
    }
    CATLASS_HOST_DEVICE
    Index &cout1()
    {
        return this->outputShape.c1();
    }

    CATLASS_HOST_DEVICE
    Index coutRound() const
    {
        return (this->filterShape.cout() + C0 - 1) / C0 * C0;
    }

    CATLASS_HOST_DEVICE
    Index const &ho() const
    {
        return this->outputShape.h();
    }
    CATLASS_HOST_DEVICE
    Index &ho()
    {
        return this->outputShape.h();
    }

    CATLASS_HOST_DEVICE
    Index const &wo() const
    {
        return this->outputShape.w();
    }
    CATLASS_HOST_DEVICE
    Index &wo()
    {
        return this->outputShape.w();
    }

    CATLASS_HOST_DEVICE
    Index howo() const
    {
        return this->outputShape.h() * this->outputShape.w();
    }

    CATLASS_HOST_DEVICE
    Index howoRound() const
    {
        return (this->howo() + C0 - 1) / C0 * C0;
    }

    CATLASS_HOST_DEVICE
    ShortIndex const &kh() const
    {
        return this->configs.kh();
    }
    CATLASS_HOST_DEVICE
    ShortIndex &kh()
    {
        return this->configs.kh();
    }

    CATLASS_HOST_DEVICE
    ShortIndex const &kw() const
    {
        return this->configs.kw();
    }
    CATLASS_HOST_DEVICE
    ShortIndex &kw()
    {
        return this->configs.kw();
    }

    CATLASS_HOST_DEVICE
    ShortIndex const &padTop() const
    {
        return this->configs.padTop();
    }
    CATLASS_HOST_DEVICE
    ShortIndex const &padTop()
    {
        return this->configs.padTop();
    }

    CATLASS_HOST_DEVICE
    ShortIndex const &padBottom() const
    {
        return this->configs.padBottom();
    }
    CATLASS_HOST_DEVICE
    ShortIndex const &padBottom()
    {
        return this->configs.padBottom();
    }

    CATLASS_HOST_DEVICE
    ShortIndex const &padLeft() const
    {
        return this->configs.padLeft();
    }
    CATLASS_HOST_DEVICE
    ShortIndex const &padLeft()
    {
        return this->configs.padLeft();
    }

    CATLASS_HOST_DEVICE
    ShortIndex const &padRight() const
    {
        return this->configs.padRight();
    }
    CATLASS_HOST_DEVICE
    ShortIndex const &padRight()
    {
        return this->configs.padRight();
    }

    CATLASS_HOST_DEVICE
    ShortIndex const &strideH() const
    {
        return this->configs.strideH();
    }
    CATLASS_HOST_DEVICE
    ShortIndex const &strideH()
    {
        return this->configs.strideH();
    }

    CATLASS_HOST_DEVICE
    ShortIndex const &strideW() const
    {
        return this->configs.strideW();
    }
    CATLASS_HOST_DEVICE
    ShortIndex const &strideW()
    {
        return this->configs.strideW();
    }

    CATLASS_HOST_DEVICE
    ShortIndex const &dilationH() const
    {
        return this->configs.dilationH();
    }
    CATLASS_HOST_DEVICE
    ShortIndex const &dilationH()
    {
        return this->configs.dilationH();
    }

    CATLASS_HOST_DEVICE
    ShortIndex const &dilationW() const
    {
        return this->configs.dilationW();
    }
    CATLASS_HOST_DEVICE
    ShortIndex const &dilationW()
    {
        return this->configs.dilationW();
    }
};

} // namespace Catlass

#endif // TLA_V2_CONV2D_PARAMS_HPP
