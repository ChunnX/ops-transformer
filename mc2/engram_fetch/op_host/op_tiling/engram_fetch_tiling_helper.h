/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * EngramFetch tiling helper
 */

#ifndef MC2_ENGRAM_FETCH_TILING_HELPER_H
#define MC2_ENGRAM_FETCH_TILING_HELPER_H

#include <cstdint>

namespace optiling {

struct EngramFetchConfig {
    uint32_t handleIndex = 0;
    uint32_t indicesIndex = 1;
    uint32_t outputIndex = 0;
};

} // namespace optiling

#endif // MC2_ENGRAM_FETCH_TILING_HELPER_H