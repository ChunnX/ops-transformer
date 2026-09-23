/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_PER_TOKEN_V2_ROUTE_UTILS_HPP
#define MEGA_MOE_PER_TOKEN_V2_ROUTE_UTILS_HPP

#include "catlass/catlass.hpp"
#include "layout3d.hpp"

namespace Catlass::Epilogue::Block::detail {

template <class Params>
CATLASS_DEVICE int32_t GetLenRankInExpert(Params const &params, AscendC::GlobalTensor<int32_t> const &tokenPerExpert,
                                          Layout3D const &tokenPerExpertLayout, int32_t dstEpIdx, int32_t groupIdx)
{
    if (params.useUbRouteCache) {
        int32_t stride = params.ubRouteStride > 0 ? params.ubRouteStride : params.expertPerRank;
        return params.ubTokenPerExpert.GetValue(dstEpIdx * stride + groupIdx);
    }
    return tokenPerExpert(tokenPerExpertLayout(dstEpIdx, params.rank, groupIdx));
}

template <class Params>
CATLASS_DEVICE int32_t GetDstExpertOffset(Params const &params, AscendC::GlobalTensor<int32_t> const &preSumBeforeRank,
                                          int32_t dstEpIdx, int32_t groupIdx)
{
    if (params.useUbRouteCache) {
        return params.ubPreSumBeforeRank.GetValue(dstEpIdx * params.expertPerRank + groupIdx);
    }
    return preSumBeforeRank(dstEpIdx * params.expertPerRank + groupIdx);
}

enum class RouteAction : uint8_t {
    STOP,
    SKIP,
    COPY
};

struct PerTokenV2RouteRange {
    RouteAction action;
    int32_t stData;
    uint32_t lenData;
    uint32_t dstOffsetInExpert;
};

// Pure scalar range calculation. The caller still controls the destination
// tensor and performs its copy at the same point in the tile pipeline.
CATLASS_DEVICE PerTokenV2RouteRange GetRouteRange(int32_t lenRankInExpert, int32_t stTile, int32_t edTile,
                                                  int32_t &preSumRankInExpert)
{
    int32_t stRankInExpert = preSumRankInExpert;
    int32_t edRankInExpert = stRankInExpert + lenRankInExpert;
    preSumRankInExpert += lenRankInExpert;
    if (stRankInExpert >= edTile) {
        return {RouteAction::STOP, 0, 0, 0};
    } else if (edRankInExpert <= stTile) {
        return {RouteAction::SKIP, 0, 0, 0};
    }
    int32_t stData = max(stRankInExpert, stTile);
    int32_t edData = min(edRankInExpert, edTile);
    uint32_t lenData = edData - stData;
    if (lenData <= 0) {
        return {RouteAction::SKIP, 0, 0, 0};
    }

    uint32_t dstOffsetInExpert = 0;
    if (stTile > stRankInExpert) {
        dstOffsetInExpert = stTile - stRankInExpert;
    }
    return {RouteAction::COPY, stData, lenData, dstOffsetInExpert};
}

} // namespace Catlass::Epilogue::Block::detail

#endif // MEGA_MOE_PER_TOKEN_V2_ROUTE_UTILS_HPP
