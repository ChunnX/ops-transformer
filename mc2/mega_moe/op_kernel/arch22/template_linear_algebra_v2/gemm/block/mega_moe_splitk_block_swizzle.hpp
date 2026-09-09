/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_MEGA_MOE_GEMM_BLOCK_SPLITK_SWIZZLE_HPP
#define CATLASS_MEGA_MOE_GEMM_BLOCK_SPLITK_SWIZZLE_HPP

#include "../../../template_linear_algebra_v2/mega_moe_catlass.hpp"
#include "../../../template_linear_algebra_v2/detail/tla_v2_alignment.hpp"
#include "../../../template_linear_algebra_v2/mega_moe_gemm_coord.hpp"
#include "../../../template_linear_algebra_v2/mega_moe_matrix_coord.hpp"

namespace Catlass::Gemm::Block {
template <uint32_t SwizzleOffset = 1, uint32_t SwizzleDirection = 0>
struct SplitkGemmIdentityBlockSwizzle {
    /// Data members

    GemmCoord problemShape;
    GemmCoord tileShape;
    GemmCoord loopsMNK;
    uint32_t splitkFactor = 1; // splite k dim into virtual cores

    /// Methods

    CATLASS_DEVICE
    SplitkGemmIdentityBlockSwizzle() {}

    CATLASS_DEVICE
    SplitkGemmIdentityBlockSwizzle(GemmCoord const &problemShape_, GemmCoord const &tileShape_,
                                   uint32_t splitkFactor_ = 1)
        : problemShape(problemShape_),
          tileShape(tileShape_),
          splitkFactor(splitkFactor_)
    {
        loopsMNK = CeilDiv(problemShape, tileShape);
    }

    CATLASS_DEVICE
    uint32_t GetKIdxBySplitkSliceIdx(uint32_t splitkSliceIdx) const
    {
        if (splitkSliceIdx < loopsMNK.k() % splitkFactor) {
            return (loopsMNK.k() / splitkFactor + 1) * splitkSliceIdx;
        } else {
            return splitkSliceIdx * (loopsMNK.k() / splitkFactor) + loopsMNK.k() % splitkFactor;
        }
    }

    CATLASS_DEVICE
    uint32_t GetSplitkSliceIdx(uint32_t taskIdx) const
    {
        uint32_t mnLoops = loopsMNK.m() * loopsMNK.n();
        return taskIdx % GetCoreLoops() / mnLoops;
    }

    CATLASS_DEVICE
    uint32_t GetCoreLoops() const
    {
        return loopsMNK.m() * loopsMNK.n() * splitkFactor;
    }

    CATLASS_DEVICE
    uint32_t GetBatchIdx(uint32_t taskIdx)
    {
        return taskIdx / GetCoreLoops();
    }

    CATLASS_DEVICE
    GemmCoord GetBlockCoord(uint32_t taskIdx)
    {
        uint32_t splitkSliceIdx = GetSplitkSliceIdx(taskIdx);
        uint32_t kIdx = GetKIdxBySplitkSliceIdx(splitkSliceIdx);

        uint32_t innerIdx = taskIdx % (loopsMNK.m() * loopsMNK.n());
        if constexpr (SwizzleDirection == 0) { // Zn
            uint32_t tileBlockLoop = CeilDiv(loopsMNK.m(), SwizzleOffset);
            uint32_t tileBlockIdx = innerIdx / (SwizzleOffset * loopsMNK.n());
            uint32_t inTileBlockIdx = innerIdx % (SwizzleOffset * loopsMNK.n());

            uint32_t nRow = SwizzleOffset;
            if (tileBlockIdx == tileBlockLoop - 1) {
                nRow = loopsMNK.m() - SwizzleOffset * tileBlockIdx;
            }
            uint32_t mIdx = tileBlockIdx * SwizzleOffset + inTileBlockIdx % nRow;
            uint32_t nIdx = inTileBlockIdx / nRow;
            if (tileBlockIdx % 2 == 1) {
                nIdx = loopsMNK.n() - nIdx - 1;
            }
            return GemmCoord{mIdx, nIdx, kIdx};
        } else if constexpr (SwizzleDirection == 1) { // Nz
            uint32_t tileBlockLoop = CeilDiv(loopsMNK.n(), SwizzleOffset);
            uint32_t tileBlockIdx = innerIdx / (SwizzleOffset * loopsMNK.m());
            uint32_t inTileBlockIdx = innerIdx % (SwizzleOffset * loopsMNK.m());

            uint32_t nCol = SwizzleOffset;
            if (tileBlockIdx == tileBlockLoop - 1) {
                nCol = loopsMNK.n() - SwizzleOffset * tileBlockIdx;
            }
            uint32_t mIdx = inTileBlockIdx / nCol;
            uint32_t nIdx = tileBlockIdx * SwizzleOffset + inTileBlockIdx % nCol;
            if (tileBlockIdx % 2 == 1) {
                mIdx = loopsMNK.m() - mIdx - 1;
            }
            return GemmCoord{mIdx, nIdx, kIdx};
        }
    }

    CATLASS_DEVICE
    GemmCoord GetActualBlockShape(GemmCoord blockCoord, uint32_t splitkSliceIdx)
    {
        uint32_t splitkSliceLen;
        if (splitkSliceIdx < loopsMNK.k() % splitkFactor) {
            splitkSliceLen = (loopsMNK.k() / splitkFactor + 1) * tileShape.k();
        } else {
            splitkSliceLen = (loopsMNK.k() / splitkFactor) * tileShape.k();
        }
        uint32_t mActual = (blockCoord.m() == (loopsMNK.m() - 1)) ?
                               (problemShape.m() - blockCoord.m() * tileShape.m()) :
                               tileShape.m();
        uint32_t nActual = (blockCoord.n() == (loopsMNK.n() - 1)) ?
                               (problemShape.n() - blockCoord.n() * tileShape.n()) :
                               tileShape.n();
        uint32_t kActual = (splitkSliceIdx == (splitkFactor - 1)) ?
                               (problemShape.k() - blockCoord.k() * tileShape.k()) :
                               splitkSliceLen;
        return GemmCoord{mActual, nActual, kActual};
    }
};

/// Block swizzleing function for single core splitk matmul
template <uint32_t SwizzleOffset = 1, uint32_t SwizzleDirection = 0>
struct SingleCoreSplitkGemmIdentityBlockSwizzle {
    /// Data members
    static_assert(SwizzleDirection == 0 || SwizzleDirection == 1, "SwizzleDirection only support 0/1");

    GemmCoord problemShape;
    GemmCoord tileMNK;
    GemmCoord loopsMNK;
    uint32_t mnLoopRangeStart;
    uint32_t mnLoopRangeLen;

    /// Methods
    CATLASS_DEVICE
    SingleCoreSplitkGemmIdentityBlockSwizzle() {}

    /// Methods
    CATLASS_DEVICE
    SingleCoreSplitkGemmIdentityBlockSwizzle(GemmCoord const &problemShape_, GemmCoord const &l1TileShape)
        : problemShape(problemShape_),
          tileMNK(l1TileShape)
    {
        loopsMNK = CeilDiv(problemShape, tileMNK);
        GetMNLoopRange(AscendC::GetBlockNum(), AscendC::GetBlockIdx());
    }

    CATLASS_DEVICE
    uint32_t GetSingleCoreLoops() const
    {
        return loopsMNK.k() * mnLoopRangeLen;
    }

    CATLASS_DEVICE
    void GetMNLoopRange(uint32_t coreNum, uint32_t coreIdx)
    {
        uint32_t mnLoops = loopsMNK.m() * loopsMNK.n();
        uint32_t bPerCore = CeilDiv(mnLoops, coreNum);
        // the first tmpNum cores process bPerCore blocks
        // and the last (coreNum - tmpNum) cores process (bPerCore - 1) blocks
        // total mnLoops = bPerCore * tmpNum + (bPerCore - 1) * (coreNum - tmpNum)
        uint32_t tmpNum = mnLoops - coreNum * (bPerCore - 1);

        mnLoopRangeStart =
            (coreIdx < tmpNum) ? (coreIdx * bPerCore) : (tmpNum * bPerCore + (coreIdx - tmpNum) * (bPerCore - 1));
        mnLoopRangeLen = (coreIdx < tmpNum) ? bPerCore : (bPerCore - 1);
    }

    CATLASS_DEVICE
    bool IsAtomicAdd(uint32_t loopIdx)
    {
        uint32_t kIdx = (loopIdx / mnLoopRangeLen);
        return kIdx != 0;
    }

    CATLASS_DEVICE
    GemmCoord GetBlockCoord(uint32_t loopIdx)
    {
        uint32_t kIdx = (loopIdx / mnLoopRangeLen);
        // mnLoopIdx
        uint32_t innerIdx = mnLoopRangeStart + loopIdx % mnLoopRangeLen;
        if constexpr (SwizzleDirection == 0) { // Zn
            uint32_t tileBlockLoop = CeilDiv(loopsMNK.m(), SwizzleOffset);
            uint32_t tileBlockIdx = innerIdx / (SwizzleOffset * loopsMNK.n());
            uint32_t inTileBlockIdx = innerIdx % (SwizzleOffset * loopsMNK.n());
            uint32_t tailOffset = loopsMNK.m() - (tileBlockLoop - 1) * SwizzleOffset;
            uint32_t nRow = SwizzleOffset;

            if (tileBlockLoop >= 2 && tileBlockIdx >= tileBlockLoop - 2 && tailOffset <= SwizzleOffset / 2) {
                tileBlockLoop = (loopsMNK.m() - tailOffset) / SwizzleOffset;
                tileBlockIdx = tileBlockLoop - 1;
                nRow = SwizzleOffset + tailOffset;
                inTileBlockIdx = innerIdx - SwizzleOffset * loopsMNK.n() * tileBlockIdx;
            } else if (tileBlockIdx == tileBlockLoop - 1) {
                nRow = loopsMNK.m() - SwizzleOffset * tileBlockIdx;
            }

            uint32_t nIdx = inTileBlockIdx / nRow;
            if (tileBlockIdx % 2 == 1) {
                nIdx = loopsMNK.n() - nIdx - 1;
            }

            uint32_t mIdx = tileBlockIdx * SwizzleOffset + inTileBlockIdx % nRow;
            if (nIdx % 2 == 1) {
                mIdx = tileBlockIdx * SwizzleOffset + (nRow - inTileBlockIdx % nRow - 1);
            }
            return GemmCoord{mIdx, nIdx, kIdx};
        } else if constexpr (SwizzleDirection == 1) { // Nz
            uint32_t tileBlockLoop = CeilDiv(loopsMNK.n(), SwizzleOffset);
            uint32_t tileBlockIdx = innerIdx / (SwizzleOffset * loopsMNK.m());
            uint32_t inTileBlockIdx = innerIdx % (SwizzleOffset * loopsMNK.m());
            uint32_t tailOffset = loopsMNK.n() - (tileBlockLoop - 1) * SwizzleOffset;
            uint32_t nCol = SwizzleOffset;

            if (tileBlockLoop >= 2 && tileBlockIdx >= tileBlockLoop - 2 && tailOffset < SwizzleOffset / 2) {
                tileBlockLoop = (loopsMNK.n() - tailOffset) / SwizzleOffset;
                tileBlockIdx = tileBlockLoop - 1;
                nCol = SwizzleOffset + tailOffset;
                inTileBlockIdx = innerIdx - SwizzleOffset * loopsMNK.m() * tileBlockIdx;
            } else if (tileBlockIdx == tileBlockLoop - 1) {
                nCol = loopsMNK.n() - SwizzleOffset * tileBlockIdx;
            }

            uint32_t mIdx = inTileBlockIdx / nCol;
            if (tileBlockIdx % 2 == 1) {
                mIdx = loopsMNK.m() - mIdx - 1;
            }

            uint32_t nIdx = tileBlockIdx * SwizzleOffset + inTileBlockIdx % nCol;
            if (mIdx % 2 == 1) {
                nIdx = tileBlockIdx * SwizzleOffset + (nCol - inTileBlockIdx % nCol - 1);
            }
            return GemmCoord{mIdx, nIdx, kIdx};
        }
    }

    CATLASS_DEVICE
    GemmCoord GetActualBlockShape(GemmCoord &blockCoord)
    {
        uint32_t mActual =
            (blockCoord.m() == (loopsMNK.m() - 1)) ? (problemShape.m() - blockCoord.m() * tileMNK.m()) : tileMNK.m();
        uint32_t nActual =
            (blockCoord.n() == (loopsMNK.n() - 1)) ? (problemShape.n() - blockCoord.n() * tileMNK.n()) : tileMNK.n();
        uint32_t kActual = (blockCoord.k() == (loopsMNK.k() - 1)) ?
                               (problemShape.k() - (loopsMNK.k() - 1) * tileMNK.k()) :
                               tileMNK.k();
        return GemmCoord{mActual, nActual, kActual};
    }
};

struct DynamicSingleCoreSplitkGemmIdentityBlockSwizzle : public SingleCoreSplitkGemmIdentityBlockSwizzle<> {
    uint32_t swizzleOffset;
    uint32_t swizzleDirection;

    CATLASS_DEVICE
    DynamicSingleCoreSplitkGemmIdentityBlockSwizzle(GemmCoord const &problemShape_, GemmCoord const &l1TileShape,
                                                    uint32_t swizzleOffset_, uint32_t swizzleDirection_)
        : swizzleOffset(swizzleOffset_),
          swizzleDirection(swizzleDirection_),
          SingleCoreSplitkGemmIdentityBlockSwizzle<>(problemShape_, l1TileShape)
    {}

    CATLASS_DEVICE
    GemmCoord GetBlockCoord(uint32_t loopIdx)
    {
        uint32_t kIdx = (loopIdx / mnLoopRangeLen);
        // mnLoopIdx
        uint32_t innerIdx = mnLoopRangeStart + loopIdx % mnLoopRangeLen;
        if (swizzleDirection == 0) { // Zn
            uint32_t tileBlockLoop = CeilDiv(loopsMNK.m(), swizzleOffset);
            uint32_t tileBlockIdx = innerIdx / (swizzleOffset * loopsMNK.n());
            uint32_t inTileBlockIdx = innerIdx % (swizzleOffset * loopsMNK.n());
            uint32_t tailOffset = loopsMNK.m() - (tileBlockLoop - 1) * swizzleOffset;
            uint32_t nRow = swizzleOffset;

            if (tileBlockLoop >= 2 && tileBlockIdx >= tileBlockLoop - 2 && tailOffset <= swizzleOffset / 2) {
                tileBlockLoop = (loopsMNK.m() - tailOffset) / swizzleOffset;
                tileBlockIdx = tileBlockLoop - 1;
                nRow = swizzleOffset + tailOffset;
                inTileBlockIdx = innerIdx - swizzleOffset * loopsMNK.n() * tileBlockIdx;
            } else if (tileBlockIdx == tileBlockLoop - 1) {
                nRow = loopsMNK.m() - swizzleOffset * tileBlockIdx;
            }

            uint32_t nIdx = inTileBlockIdx / nRow;
            if (tileBlockIdx % 2 == 1) {
                nIdx = loopsMNK.n() - nIdx - 1;
            }

            uint32_t mIdx = tileBlockIdx * swizzleOffset + inTileBlockIdx % nRow;
            if (nIdx % 2 == 1) {
                mIdx = tileBlockIdx * swizzleOffset + (nRow - inTileBlockIdx % nRow - 1);
            }
            return GemmCoord{mIdx, nIdx, kIdx};
        } else { // Nz
            uint32_t tileBlockLoop = CeilDiv(loopsMNK.n(), swizzleOffset);
            uint32_t tileBlockIdx = innerIdx / (swizzleOffset * loopsMNK.m());
            uint32_t inTileBlockIdx = innerIdx % (swizzleOffset * loopsMNK.m());
            uint32_t tailOffset = loopsMNK.n() - (tileBlockLoop - 1) * swizzleOffset;
            uint32_t nCol = swizzleOffset;

            if (tileBlockLoop >= 2 && tileBlockIdx >= tileBlockLoop - 2 && tailOffset < swizzleOffset / 2) {
                tileBlockLoop = (loopsMNK.n() - tailOffset) / swizzleOffset;
                tileBlockIdx = tileBlockLoop - 1;
                nCol = swizzleOffset + tailOffset;
                inTileBlockIdx = innerIdx - swizzleOffset * loopsMNK.m() * tileBlockIdx;
            } else if (tileBlockIdx == tileBlockLoop - 1) {
                nCol = loopsMNK.n() - swizzleOffset * tileBlockIdx;
            }

            uint32_t mIdx = inTileBlockIdx / nCol;
            if (tileBlockIdx % 2 == 1) {
                mIdx = loopsMNK.m() - mIdx - 1;
            }

            uint32_t nIdx = tileBlockIdx * swizzleOffset + inTileBlockIdx % nCol;
            if (mIdx % 2 == 1) {
                nIdx = tileBlockIdx * swizzleOffset + (nCol - inTileBlockIdx % nCol - 1);
            }
            return GemmCoord{mIdx, nIdx, kIdx};
        }
    }
};

/// Block swizzling function for Gemms
struct DynamicSplitkGemmIdentityBlockSwizzle : public SplitkGemmIdentityBlockSwizzle<> {
    uint32_t swizzleOffset{1};
    uint32_t swizzleDirection{0};

    CATLASS_DEVICE
    DynamicSplitkGemmIdentityBlockSwizzle(GemmCoord const &problemShape_, GemmCoord const &tileMNK_,
                                          uint32_t splitkFactor_, uint32_t swizzleOffset_, uint32_t swizzleDirection_)
        : swizzleOffset(swizzleOffset_),
          swizzleDirection(swizzleDirection_),
          SplitkGemmIdentityBlockSwizzle<>(problemShape_, tileMNK_, splitkFactor_)
    {}

    CATLASS_DEVICE
    DynamicSplitkGemmIdentityBlockSwizzle(GemmCoord const &problemShape_, GemmCoord const &tileMNK_,
                                          uint32_t splitkFactor_)
        : SplitkGemmIdentityBlockSwizzle<>(problemShape_, tileMNK_, splitkFactor_)
    {}

    CATLASS_DEVICE
    DynamicSplitkGemmIdentityBlockSwizzle() {}

    CATLASS_DEVICE
    void SetSwizzleParams(uint32_t swizzleOffset_, uint32_t swizzleDirection_)
    {
        swizzleOffset = swizzleOffset_;
        swizzleDirection = swizzleDirection_;
    }

    CATLASS_DEVICE
    GemmCoord GetBlockCoord(uint32_t taskIdx)
    {
        uint32_t splitkSliceIdx = GetSplitkSliceIdx(taskIdx);
        uint32_t kIdx = GetKIdxBySplitkSliceIdx(splitkSliceIdx);

        uint32_t innerIdx = taskIdx % (loopsMNK.m() * loopsMNK.n());
        if (swizzleDirection == 0) { // Zn
            uint32_t tileBlockLoop = CeilDiv(loopsMNK.m(), swizzleOffset);
            uint32_t tileBlockIdx = innerIdx / (swizzleOffset * loopsMNK.n());
            uint32_t inTileBlockIdx = innerIdx % (swizzleOffset * loopsMNK.n());

            uint32_t nRow = swizzleOffset;
            if (tileBlockIdx == tileBlockLoop - 1) {
                nRow = loopsMNK.m() - swizzleOffset * tileBlockIdx;
            }
            uint32_t mIdx = tileBlockIdx * swizzleOffset + inTileBlockIdx % nRow;
            uint32_t nIdx = inTileBlockIdx / nRow;
            if (tileBlockIdx % 2 == 1) {
                nIdx = loopsMNK.n() - nIdx - 1;
            }
            return GemmCoord{mIdx, nIdx, kIdx};
        } else { // Nz
            uint32_t tileBlockLoop = CeilDiv(loopsMNK.n(), swizzleOffset);
            uint32_t tileBlockIdx = innerIdx / (swizzleOffset * loopsMNK.m());
            uint32_t inTileBlockIdx = innerIdx % (swizzleOffset * loopsMNK.m());

            uint32_t nCol = swizzleOffset;
            if (tileBlockIdx == tileBlockLoop - 1) {
                nCol = loopsMNK.n() - swizzleOffset * tileBlockIdx;
            }
            uint32_t mIdx = inTileBlockIdx / nCol;
            uint32_t nIdx = tileBlockIdx * swizzleOffset + inTileBlockIdx % nCol;
            if (tileBlockIdx % 2 == 1) {
                mIdx = loopsMNK.m() - mIdx - 1;
            }
            return GemmCoord{mIdx, nIdx, kIdx};
        }
    }
};

/// Block swizzling function for streamk Gemms
} // namespace Catlass::Gemm::Block

#endif // CATLASS_MEGA_MOE_GEMM_BLOCK_SPLITK_SWIZZLE_HPP
