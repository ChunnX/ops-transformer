#include "include/tensor_api/tensor.h"
#include "basic_api/kernel_basic_intf.h"

#pragma once

namespace Blaze {
namespace Gemm {
namespace Block {

template<uint32_t SwizzleOffset = 1, uint32_t SwizzleDirection = 0>
class BlockSchedulerSwizzle{
    public:
    using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t>;
    using BlockShape = AscendC::Te::Shape<int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Te::Coord<int64_t, int64_t, int64_t>;
    using TileShape = AscendC::Te::Shape<int64_t, int64_t>;

    struct Params {
        TileShape tileShape;
    };

    __aicore__ inline BlockSchedulerSwizzle(const ProblemShape& shape, const Params& params) : problemShape_(shape), tileShape_(params.tileShape)
    {
        if constexpr (SwizzleDirection == 0) {
            // m first
            loopFirst_ = AscendC::Std::ceil_division(get<0>(problemShape_), get<0>(tileShape_));
            loopSecond_ = AscendC::Std::ceil_division(get<1>(problemShape_), get<1>(tileShape_));
        } else if constexpr (SwizzleDirection == 1) {
            // n first
            loopSecond_ = AscendC::Std::ceil_division(get<0>(problemShape_), get<0>(tileShape_));
            loopFirst_ = AscendC::Std::ceil_division(get<1>(problemShape_), get<1>(tileShape_));
        }
    }

    __aicore__ inline int64_t GetTileNum() {
        return loopFirst_ * loopSecond_;
    }

    __aicore__ inline BlockShape GetBlockShape(const BlockCoord& blockCoord)
    {
        return {min(get<0>(tileShape_), get<0>(problemShape_) - get<0>(blockCoord)),
                min(get<1>(tileShape_), get<1>(problemShape_) - get<1>(blockCoord)),
                get<2>(problemShape_)};
    }

    __aicore__ inline BlockCoord GetBlockCoord(int tileIdx)
    {
        int64_t blockSpan = SwizzleOffset * loopSecond_;
        int64_t blockIdx = tileIdx / blockSpan;
        int64_t inBlockIdx = tileIdx % blockSpan;

        int64_t firstValid = Min(loopFirst_ - blockIdx * SwizzleOffset, static_cast<int64_t>(SwizzleOffset));
        int64_t firstIdx = blockIdx * SwizzleOffset + inBlockIdx % firstValid;
        int64_t secondIdx = inBlockIdx / firstValid;
        if (blockIdx & 1) {
            secondIdx = loopSecond_ - secondIdx - 1;
        }

        if constexpr (SwizzleDirection == 0) {
            return {firstIdx * get<0>(tileShape_), secondIdx * get<1>(tileShape_), 0};
        } else {
            return {secondIdx * get<0>(tileShape_), firstIdx * get<1>(tileShape_), 0};
        }
    }

    private:
    ProblemShape problemShape_;
    TileShape tileShape_;
    int64_t loopFirst_;
    int64_t loopSecond_;
};

}
}
}