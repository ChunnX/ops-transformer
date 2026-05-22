/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GMM_EPILOGUE_TILE_TILE_SWIZZLE_HPP
#define GMM_EPILOGUE_TILE_TILE_SWIZZLE_HPP

#include "../../../gmm_infra/base_defs.hpp"
#include "../../../gmm_infra/detail/gmm_alignment.hpp"
#include "../../../gmm_infra/matrix_gmm_coord.hpp"

namespace Catlass::Epilogue::Tile {

struct EpilogueIdentityTileSwizzle {
    MatrixCoord blockShape;
    MatrixCoord tileShape;
    MatrixCoord loopsMN;

    GMM_DEVICE
    EpilogueIdentityTileSwizzle() = default;

    GMM_DEVICE
    EpilogueIdentityTileSwizzle(MatrixCoord const &blockShape, MatrixCoord const &tileShape) :
        blockShape(blockShape),
        tileShape(tileShape)
    {
        loopsMN = CeilDiv(blockShape, tileShape);
    }

    GMM_DEVICE
    uint32_t GetLoops() const
    {
        return loopsMN.row() * loopsMN.column();
    }

    GMM_DEVICE
    MatrixCoord GetTileCoord(uint32_t loopIdx) const
    {
        return MatrixCoord{ loopIdx / loopsMN.column(), loopIdx % loopsMN.column() };
    }

    GMM_DEVICE
    MatrixCoord GetActualTileShape(MatrixCoord const &tileCoord) const
    {
        return MatrixCoord::Min(tileShape, blockShape - tileCoord * tileShape);
    }
};

struct EpilogueHorizontalTileSwizzle {
    MatrixCoord blockShape;
    MatrixCoord tileShape;
    MatrixCoord loopsMN;

    GMM_DEVICE
    EpilogueHorizontalTileSwizzle() = default;

    GMM_DEVICE
    EpilogueHorizontalTileSwizzle(MatrixCoord const &blockShape, MatrixCoord const &tileShape) :
        blockShape(blockShape),
        tileShape(tileShape)
    {
        loopsMN = CeilDiv(blockShape, tileShape);
    }

    GMM_DEVICE
    uint32_t GetLoops() const
    {
        return loopsMN.row() * loopsMN.column();
    }

    GMM_DEVICE
    MatrixCoord GetTileCoord(uint32_t loopIdx) const
    {
        return MatrixCoord{ loopIdx % loopsMN.row(), loopIdx / loopsMN.row() };
    }

    GMM_DEVICE
    MatrixCoord GetActualTileShape(MatrixCoord const &tileCoord) const
    {
        return MatrixCoord::Min(tileShape, blockShape - tileCoord * tileShape);
    }
};

}

#endif  // GMM_EPILOGUE_TILE_TILE_SWIZZLE_HPP
