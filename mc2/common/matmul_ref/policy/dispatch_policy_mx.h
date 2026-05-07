/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * MX量化专用 dispatch policy 标签
 */

#pragma once

#include <cstdint>

template <uint64_t FULL_LOAD_MODE_, uint64_t STAGES_>
struct QuantMatmulMxMultiBlockWithSwat {
    static constexpr uint64_t fullLoadMode = FULL_LOAD_MODE_;
    static constexpr uint64_t stages = STAGES_;
};

template <uint64_t FULL_LOAD_MODE_>
struct QuantMatmulMxSwatScheduler {
    static constexpr uint64_t fullLoadMode = FULL_LOAD_MODE_;
};
