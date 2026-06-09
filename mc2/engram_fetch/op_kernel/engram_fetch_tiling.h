/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * EngramFetch tiling data definition
 */

#ifndef OP_HOST_ENGRAM_FETCH_TILING_H
#define OP_HOST_ENGRAM_FETCH_TILING_H

namespace op {
struct EngramFetchTilingData {
    int64_t numTokens;
    int64_t hidden;
    int64_t placeholder; // Reserved for future use
};

struct EngramFetchWaitTilingData {
    int64_t placeholder; // Reserved for future use
};
} // namespace op

#endif // OP_HOST_ENGRAM_FETCH_TILING_H