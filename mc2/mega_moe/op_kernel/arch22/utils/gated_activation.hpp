/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_GATED_ACTIVATION_HPP
#define MEGA_MOE_GATED_ACTIVATION_HPP

#include "../template_linear_algebra_v2/catlass.hpp"

namespace Catlass::Epilogue {

// All gated activations used by MegaMoe follow this interface. A new
// activation only needs to implement Compute; data movement, tiling and
// quantization remain shared by the epilogues.
struct SwigluActivation {
    // 3072 is the largest common tile that leaves at least 32 KiB of
    // temporary UB for every BF16, INT8 and W4A8 double-buffered path.
    static constexpr uint32_t PREFERRED_TILE_LENGTH = 3072;
    static constexpr uint32_t MIN_SHARED_TMP_BYTES = 32 * 1024;

    CATLASS_DEVICE
    static void Compute(AscendC::LocalTensor<float> const &output, AscendC::LocalTensor<float> const &gate,
                        AscendC::LocalTensor<float> const &up, AscendC::LocalTensor<uint8_t> const &sharedTmpBuffer,
                        uint32_t dataLength, float clampLimit)
    {
        if (clampLimit > 0.0f) {
            // Preserve MegaMoe's existing clamp semantics: the gate branch is
            // upper-bounded, while the up branch is symmetrically clamped.
            AscendC::ClampMax(gate, gate, sharedTmpBuffer, clampLimit, dataLength);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::ClampMax(up, up, sharedTmpBuffer, clampLimit, dataLength);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::ClampMin(up, up, sharedTmpBuffer, -clampLimit, dataLength);
            AscendC::PipeBarrier<PIPE_V>();
        }

        // SiLU(gate) * up = gate / (1 + exp(-gate)) * up.
        AscendC::Muls(output, gate, -1.0f, dataLength);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(output, output, dataLength);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(output, output, 1.0f, dataLength);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Div(output, gate, output, dataLength);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(output, output, up, dataLength);
        AscendC::PipeBarrier<PIPE_V>();
    }
};

} // namespace Catlass::Epilogue

#endif // MEGA_MOE_GATED_ACTIVATION_HPP
