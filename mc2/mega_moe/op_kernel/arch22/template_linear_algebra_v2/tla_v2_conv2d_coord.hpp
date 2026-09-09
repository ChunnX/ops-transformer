/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TLA_V2_CONV2D_COORD_HPP
#define TLA_V2_CONV2D_COORD_HPP

#include "../template_linear_algebra_v2/tla_v2_coord.hpp"

namespace Catlass {

/////////////////// Shapes and Coords for Conv2d ///////////////////

template <uint32_t Ho_ = 1, uint32_t Wo_ = 1,
          uint32_t Cin1_ = 1>
struct Conv2dFmapL1Shape { // (Ho, Wo, Cin1)
    static constexpr uint32_t Ho = Ho_;
    static constexpr uint32_t Wo = Wo_;
    static constexpr uint32_t Cin1 = Cin1_;

    /// Returns a Coord object
    CATLASS_HOST_DEVICE
    static Coord<3> ToCoord()
    {
        return MakeCoord(Ho, Wo, Cin1);
    }
};

template <uint32_t Cout_ = 16,
          uint32_t Cin1_ = 1>
struct Conv2dFilterL1Shape { // (Cout, Cin1)
    static constexpr uint32_t Cout = Cout_;
    static constexpr uint32_t Cin1 = Cin1_;

    /// Returns a Coord object
    CATLASS_HOST_DEVICE
    static Coord<2> ToCoord()
    {
        return MakeCoord(Cout, Cin1);
    }
};

template <uint32_t M_ = 16, uint32_t N_ = 16, uint32_t K_ = 16>
struct Conv2dL0Shape {
    static constexpr uint32_t M = M_;
    static constexpr uint32_t N = N_;
    static constexpr uint32_t K = K_;

    /// Returns a Coord object
    CATLASS_HOST_DEVICE
    static Coord<3> ToCoord()
    {
        return MakeCoord(M, N, K);
    }
};

struct Conv2dFmapCoord : public Coord<5, uint32_t> { // (Batch, C1, H, W, C0)
public:
    /// Integer-valued index
    using Index = uint32_t;

    /// Base type is a Coord of rank=5
    using Base = Coord<5, Index>;

    /// LongIndex type
    using LongIndex = typename Base::LongIndex;

    /// dimensions
    static constexpr uint32_t BATCH_INDEX = 0;
    static constexpr uint32_t C1_INDEX = 1;
    static constexpr uint32_t H_INDEX = 2;
    static constexpr uint32_t W_INDEX = 3;
    static constexpr uint32_t C0_INDEX = 4;

    /// Default ctor
    CATLASS_HOST_DEVICE
    Conv2dFmapCoord() {}

    /// Constructs from Coord<5>
    CATLASS_HOST_DEVICE
    Conv2dFmapCoord(Coord<5, Index> const &coord)
        : Base(coord)
    {}

    /// Helper to construct from C1, H, W, C0
    CATLASS_HOST_DEVICE
    Conv2dFmapCoord(Index batch, Index c1, Index h, Index w, Index c0)
        : Base(MakeCoord(batch, c1, h, w, c0))
    {}

    CATLASS_HOST_DEVICE
    Conv2dFmapCoord(LongIndex batch, LongIndex c1, LongIndex h, LongIndex w, LongIndex c0)
        : Base(MakeCoord(Index(batch), Index(c1), Index(h), Index(w), Index(c0)))
    {}

    CATLASS_HOST_DEVICE
    Index const &c1() const
    {
        return this->At(C1_INDEX);
    }
    CATLASS_HOST_DEVICE
    Index &c1()
    {
        return this->At(C1_INDEX);
    }

    CATLASS_HOST_DEVICE
    Index const &batch() const
    {
        return this->At(BATCH_INDEX);
    }
    CATLASS_HOST_DEVICE
    Index &batch()
    {
        return this->At(BATCH_INDEX);
    }

    CATLASS_HOST_DEVICE
    Index const &h() const
    {
        return this->At(H_INDEX);
    }
    CATLASS_HOST_DEVICE
    Index &h()
    {
        return this->At(H_INDEX);
    }

    CATLASS_HOST_DEVICE
    Index const &w() const
    {
        return this->At(W_INDEX);
    }
    CATLASS_HOST_DEVICE
    Index &w()
    {
        return this->At(W_INDEX);
    }

    CATLASS_HOST_DEVICE
    Index const &c0() const
    {
        return this->At(C0_INDEX);
    }
    CATLASS_HOST_DEVICE
    Index &c0()
    {
        return this->At(C0_INDEX);
    }

    /// Element-wise addition
    CATLASS_HOST_DEVICE
    Conv2dFmapCoord operator+(Base const &b) const
    {
        return Conv2dFmapCoord(Base::operator+(b));
    }

    /// In-place addition
    CATLASS_HOST_DEVICE
    Conv2dFmapCoord &operator+=(Base const &b)
    {
        Base::operator+=(b);
        return *this;
    }
};

struct Conv2dFilterCoord : public Coord<5, uint32_t> { // (Cin1, Kh, Kw, Cout, C0)
public:
    /// Integer-valued index
    using Index = uint32_t;

    /// Base type is a Coord of rank=5
    using Base = Coord<5, Index>;

    /// LongIndex type
    using LongIndex = typename Base::LongIndex;

    /// dimensions
    static constexpr uint32_t CIN1_INDEX = 0;
    static constexpr uint32_t KH_INDEX = 1;
    static constexpr uint32_t KW_INDEX = 2;
    static constexpr uint32_t COUT_INDEX = 3;
    static constexpr uint32_t C0_INDEX = 4;

    /// Default ctor
    CATLASS_HOST_DEVICE
    Conv2dFilterCoord() {}

    /// Constructs from Coord<5>
    CATLASS_HOST_DEVICE
    Conv2dFilterCoord(Coord<5, Index> const &coord)
        : Base(coord)
    {}

    /// Helper to construct from Cin1, Kh, Kw, Cout, C0
    CATLASS_HOST_DEVICE
    Conv2dFilterCoord(Index cin1, Index kh, Index kw, Index cout, Index c0)
        : Base(MakeCoord(cin1, kh, kw, cout, c0))
    {}

    CATLASS_HOST_DEVICE
    Conv2dFilterCoord(LongIndex cin1, LongIndex kh, LongIndex kw, LongIndex cout, LongIndex c0)
        : Base(MakeCoord(Index(cin1), Index(kh), Index(kw), Index(cout), Index(c0)))
    {}

    CATLASS_HOST_DEVICE
    Index const &cin1() const
    {
        return this->At(CIN1_INDEX);
    }
    CATLASS_HOST_DEVICE
    Index &cin1()
    {
        return this->At(CIN1_INDEX);
    }

    CATLASS_HOST_DEVICE
    Index const &kh() const
    {
        return this->At(KH_INDEX);
    }
    CATLASS_HOST_DEVICE
    Index &kh()
    {
        return this->At(KH_INDEX);
    }

    CATLASS_HOST_DEVICE
    Index const &kw() const
    {
        return this->At(KW_INDEX);
    }
    CATLASS_HOST_DEVICE
    Index &kw()
    {
        return this->At(KW_INDEX);
    }

    CATLASS_HOST_DEVICE
    Index const &cout() const
    {
        return this->At(COUT_INDEX);
    }
    CATLASS_HOST_DEVICE
    Index &cout()
    {
        return this->At(COUT_INDEX);
    }

    CATLASS_HOST_DEVICE
    Index const &c0() const
    {
        return this->At(C0_INDEX);
    }
    CATLASS_HOST_DEVICE
    Index &c0()
    {
        return this->At(C0_INDEX);
    }

    /// Element-wise addition
    CATLASS_HOST_DEVICE
    Conv2dFilterCoord operator+(Base const &b) const
    {
        return Conv2dFilterCoord(Base::operator+(b));
    }

    /// In-place addition
    CATLASS_HOST_DEVICE
    Conv2dFilterCoord &operator+=(Base const &b)
    {
        Base::operator+=(b);
        return *this;
    }
};

struct Conv2dHoWoCoCoord : public Coord<3, uint32_t> { // (Ho, Wo, Cout)
public:
    /// Integer-valued index
    using Index = uint32_t;

    /// Base type is a Coord of rank=3
    using Base = Coord<3, Index>;

    /// LongIndex type
    using LongIndex = typename Base::LongIndex;

    /// dimensions
    static constexpr uint32_t HO_INDEX = 0;
    static constexpr uint32_t WO_INDEX = 1;
    static constexpr uint32_t COUT_INDEX = 2;

    /// Default ctor
    CATLASS_HOST_DEVICE
    Conv2dHoWoCoCoord() {}

    /// Constructs from Coord<3>
    CATLASS_HOST_DEVICE
    Conv2dHoWoCoCoord(Coord<3, Index> const &coord)
        : Base(coord)
    {}

    /// Helper to construct from Ho, Wo, Cout
    CATLASS_HOST_DEVICE
    Conv2dHoWoCoCoord(Index ho, Index wo, Index cout)
        : Base(MakeCoord(ho, wo, cout))
    {}

    CATLASS_HOST_DEVICE
    Conv2dHoWoCoCoord(LongIndex ho, LongIndex wo, LongIndex cout)
        : Base(MakeCoord(Index(ho), Index(wo), Index(cout)))
    {}

    CATLASS_HOST_DEVICE
    Index const &ho() const
    {
        return this->At(HO_INDEX);
    }

    CATLASS_HOST_DEVICE
    Index &ho()
    {
        return this->At(HO_INDEX);
    }

    CATLASS_HOST_DEVICE
    Index const &wo() const
    {
        return this->At(WO_INDEX);
    }

    CATLASS_HOST_DEVICE
    Index &wo()
    {
        return this->At(WO_INDEX);
    }

    CATLASS_HOST_DEVICE
    Index howo() const
    {
        return this->At(HO_INDEX) * this->At(WO_INDEX);
    }

    CATLASS_HOST_DEVICE
    Index const &cout() const
    {
        return this->At(COUT_INDEX);
    }

    CATLASS_HOST_DEVICE
    Index &cout()
    {
        return this->At(COUT_INDEX);
    }

    /// Element-wise addition
    CATLASS_HOST_DEVICE
    Conv2dHoWoCoCoord operator+(Base const &b) const
    {
        return Conv2dHoWoCoCoord(Base::operator+(b));
    }

    /// In-place addition
    CATLASS_HOST_DEVICE
    Conv2dHoWoCoCoord &operator+=(Base const &b)
    {
        Base::operator+=(b);
        return *this;
    }
};

struct Conv2dCoord : public Coord<5, uint32_t> { // (Batch, H, W, Cout, Cin1)
public:
    /// Integer-valued index
    using Index = uint32_t;

    /// Base type is a Coord of rank=5
    using Base = Coord<5, Index>;

    /// LongIndex type
    using LongIndex = typename Base::LongIndex;

    /// dimensions
    static constexpr uint32_t BATCH_INDEX = 0;
    static constexpr uint32_t H_INDEX = 1;
    static constexpr uint32_t W_INDEX = 2;
    static constexpr uint32_t COUT_INDEX = 3;
    static constexpr uint32_t CIN1_INDEX = 4;

    /// Default ctor
    CATLASS_HOST_DEVICE
    Conv2dCoord() {}

    /// Constructs from Coord<5>
    CATLASS_HOST_DEVICE
    Conv2dCoord(Coord<5, Index> const &coord)
        : Base(coord)
    {}

    /// Helper to construct from Batch, H, W, Cout, Cin1
    CATLASS_HOST_DEVICE
    Conv2dCoord(Index batch, Index h, Index w, Index cout, Index cin1)
        : Base(MakeCoord(batch, h, w, cout, cin1))
    {}

    CATLASS_HOST_DEVICE
    Conv2dCoord(LongIndex batch, LongIndex h, LongIndex w, LongIndex cout, LongIndex cin1)
        : Base(MakeCoord(Index(batch), Index(h), Index(w), Index(cout), Index(cin1)))
    {}

    CATLASS_HOST_DEVICE
    Index const &batch() const
    {
        return this->At(BATCH_INDEX);
    }
    CATLASS_HOST_DEVICE
    Index &batch()
    {
        return this->At(BATCH_INDEX);
    }

    CATLASS_HOST_DEVICE
    Index const &h() const
    {
        return this->At(H_INDEX);
    }
    CATLASS_HOST_DEVICE
    Index &h()
    {
        return this->At(H_INDEX);
    }

    CATLASS_HOST_DEVICE
    Index const &w() const
    {
        return this->At(W_INDEX);
    }
    CATLASS_HOST_DEVICE
    Index &w()
    {
        return this->At(W_INDEX);
    }

    CATLASS_HOST_DEVICE
    Index const &cout() const
    {
        return this->At(COUT_INDEX);
    }
    CATLASS_HOST_DEVICE
    Index &cout()
    {
        return this->At(COUT_INDEX);
    }

    CATLASS_HOST_DEVICE
    Index const &cin1() const
    {
        return this->At(CIN1_INDEX);
    }
    CATLASS_HOST_DEVICE
    Index &cin1()
    {
        return this->At(CIN1_INDEX);
    }

    /// Element-wise addition
    CATLASS_HOST_DEVICE
    Conv2dCoord operator+(Base const &b) const
    {
        return Conv2dCoord(Base::operator+(b));
    }

    /// In-place addition
    CATLASS_HOST_DEVICE
    Conv2dCoord &operator+=(Base const &b)
    {
        Base::operator+=(b);
        return *this;
    }

    CATLASS_HOST_DEVICE
    auto GetHoWoCoCoord() const
    {
        return this->GetCoordByAxis<H_INDEX, W_INDEX, COUT_INDEX>();
    }
};

} // namespace Catlass

#endif // TLA_V2_CONV2D_COORD_HPP
