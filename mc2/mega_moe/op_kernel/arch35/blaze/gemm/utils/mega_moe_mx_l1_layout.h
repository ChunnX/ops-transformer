/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#pragma once

namespace Blaze {
namespace Gemm {

// Shared by the AIC consumer and AIV0 producer. Slots stay fixed across all
// persistent GMM calls, including tails; only the tensor's valid shape changes.
struct MegaMoeMxL1Layout {
    static constexpr uint64_t KB = 1024;
    static constexpr uint64_t A_STAGES = 2;
    static constexpr uint64_t B_STAGES = 4;
    static_assert(A_STAGES == 2 || A_STAGES == 4, "A stages must be 2 or 4");
    static_assert(B_STAGES == 2 || B_STAGES == 4, "B stages must be 2 or 4");
    static_assert(B_STAGES != 4 || A_STAGES == 2, "B4 requires A2");
    static constexpr uint64_t A_SLOT_BYTES = B_STAGES == 4 ? 64 * KB : 256 * KB / A_STAGES;
    static constexpr uint64_t B_SLOT_BYTES = 64 * KB;
    static constexpr uint64_t SCALE_SLOT_BYTES = 32 * KB;
    static_assert(B_STAGES != 4 ||
                  A_STAGES * A_SLOT_BYTES + B_STAGES * B_SLOT_BYTES + 4 * SCALE_SLOT_BYTES == 512 * KB,
                  "B4 L1 allocation must fit 512 KiB");

    // B4 map, in KiB:
    // [0,64) B0, [64,128) B2, [128,192) A0, [192,224) scA0, [224,256) scB0
    // [256,320) B1, [320,384) B3, [384,448) A1, [448,480) scA1, [480,512) scB1
    __aicore__ static inline uint64_t AOffset(uint64_t slot)
    {
        if constexpr (B_STAGES == 4) {
            return 128 * KB + slot * 256 * KB;
        }
        if constexpr (A_STAGES == 4) {
            slot = ((slot & 1U) << 1U) | (slot >> 1U);
        }
        return 128 * KB + slot * A_SLOT_BYTES;
    }

    __aicore__ static inline uint64_t BOffset(uint64_t slot)
    {
        if constexpr (B_STAGES == 4) {
            return (slot & 1U) * 256 * KB + (slot >> 1U) * B_SLOT_BYTES;
        }
        return slot * 384 * KB;
    }

    __aicore__ static inline uint64_t ScaleAOffset(uint64_t slot)
    {
        return B_STAGES == 4 ? 192 * KB + slot * 256 * KB : 64 * KB + slot * 384 * KB;
    }

    __aicore__ static inline uint64_t ScaleBOffset(uint64_t slot)
    {
        return B_STAGES == 4 ? 224 * KB + slot * 256 * KB : 96 * KB + slot * 384 * KB;
    }
};

} // namespace Gemm
} // namespace Blaze
