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

enum class GatedActivationMode : uint32_t {
    SWIGLU = 0,
    SWIGLU_GPT_OSS = 1,
    SITU = 2,
};

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

// GPT-OSS uses an interleaved gate/up projection and a modified SwiGLU:
// gate * sigmoid(alpha * gate) * (up + 1). Alpha is intentionally a
// kernel-side default so the public MegaMoe interface remains unchanged.
struct GptOssSwigluActivation {
    static constexpr float DEFAULT_ALPHA = 1.702f;

    CATLASS_DEVICE
    static void Compute(AscendC::LocalTensor<float> const &output, AscendC::LocalTensor<float> const &gate,
                        AscendC::LocalTensor<float> const &up, AscendC::LocalTensor<uint8_t> const &sharedTmpBuffer,
                        uint32_t dataLength, float clampLimit, float alpha = DEFAULT_ALPHA)
    {
        if (clampLimit > 0.0f) {
            AscendC::ClampMax(gate, gate, sharedTmpBuffer, clampLimit, dataLength);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::ClampMax(up, up, sharedTmpBuffer, clampLimit, dataLength);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::ClampMin(up, up, sharedTmpBuffer, -clampLimit, dataLength);
            AscendC::PipeBarrier<PIPE_V>();
        }

        AscendC::Adds(up, up, 1.0f, dataLength);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(output, gate, -alpha, dataLength);
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

// Kimi K3 SiTU:
//     beta * tanh(gate / beta) * sigmoid(gate) * up
// When linearBeta is enabled, up becomes:
//     linearBeta * tanh(up / linearBeta)
// gate/up use the regular contiguous half split. beta and linearBeta are
// independent operator attributes; activation_clamp is not used by SiTU.
struct SituActivation {
    static constexpr float DEFAULT_BETA = 1.0f;
    static constexpr float LINEAR_BETA_DISABLED = 3.402823466e+38F;
    static constexpr uint32_t UB_BLOCK_BYTES = 32;
    static constexpr float POLY_RANGE_SQUARE = 0.25f;
    static_assert(2 * SwigluActivation::PREFERRED_TILE_LENGTH * sizeof(float) +
                          SwigluActivation::PREFERRED_TILE_LENGTH <=
                      SwigluActivation::MIN_SHARED_TMP_BYTES,
                  "SiTU small-input compensation exceeds the shared temporary buffer");

    // The default AscendC Tanh path evaluates an exponential subtraction.
    // For small |x| this loses significant digits, which become visible after
    // multiplying by a large beta. Compensate |x| <= 0.5 with the degree-11
    // Taylor polynomial and keep the intrinsic result outside that interval.
    CATLASS_DEVICE
    static void ComputeScaledTanh(AscendC::LocalTensor<float> const &output,
                                  AscendC::LocalTensor<float> const &input,
                                  AscendC::LocalTensor<uint8_t> const &sharedTmpBuffer, uint32_t dataLength,
                                  float scale)
    {
        AscendC::Muls(input, input, 1.0f / scale, dataLength);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Tanh(output, input, sharedTmpBuffer, dataLength);
        AscendC::PipeBarrier<PIPE_V>();

        uint32_t alignedDataLength =
            (dataLength * sizeof(float) + UB_BLOCK_BYTES - 1) / UB_BLOCK_BYTES * UB_BLOCK_BYTES / sizeof(float);
        auto square = sharedTmpBuffer.template ReinterpretCast<float>();
        auto polynomial = square[alignedDataLength];
        auto smallInputMask = sharedTmpBuffer[2 * alignedDataLength * sizeof(float)];

        AscendC::Mul(square, input, input, dataLength);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Duplicate(polynomial, -1382.0f / 155925.0f, dataLength);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(polynomial, polynomial, square, dataLength);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(polynomial, polynomial, 62.0f / 2835.0f, dataLength);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(polynomial, polynomial, square, dataLength);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(polynomial, polynomial, -17.0f / 315.0f, dataLength);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(polynomial, polynomial, square, dataLength);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(polynomial, polynomial, 2.0f / 15.0f, dataLength);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(polynomial, polynomial, square, dataLength);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(polynomial, polynomial, -1.0f / 3.0f, dataLength);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(polynomial, polynomial, square, dataLength);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(polynomial, polynomial, 1.0f, dataLength);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(polynomial, polynomial, input, dataLength);
        AscendC::PipeBarrier<PIPE_V>();

        // Apply only the small-input correction so the intrinsic Tanh keeps
        // its saturation behavior for larger values.
        AscendC::Sub(polynomial, polynomial, output, dataLength);
        AscendC::CompareScalar(smallInputMask, square, POLY_RANGE_SQUARE, AscendC::CMPMODE::LE, dataLength);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Select(polynomial, smallInputMask, polynomial, 0.0f,
                        AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, dataLength);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(output, output, polynomial, dataLength);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(output, output, scale, dataLength);
        AscendC::PipeBarrier<PIPE_V>();
    }

    CATLASS_DEVICE
    static void Compute(AscendC::LocalTensor<float> const &output, AscendC::LocalTensor<float> const &gate,
                        AscendC::LocalTensor<float> const &up, AscendC::LocalTensor<uint8_t> const &sharedTmpBuffer,
                        uint32_t dataLength, float linearBeta = LINEAR_BETA_DISABLED,
                        float beta = DEFAULT_BETA)
    {
        bool enableLinearBeta = linearBeta > 0.0f && linearBeta < LINEAR_BETA_DISABLED;
        if (enableLinearBeta) {
            // output = linearBeta * tanh(up / linearBeta). The original up
            // tile can then be reused for the sigmoid denominator.
            ComputeScaledTanh(output, up, sharedTmpBuffer, dataLength, linearBeta);

            AscendC::Muls(up, gate, -1.0f, dataLength);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Exp(up, up, dataLength);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds(up, up, 1.0f, dataLength);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Div(output, output, up, dataLength);
            AscendC::PipeBarrier<PIPE_V>();
        } else {
            // output = sigmoid(gate) * up.
            AscendC::Muls(output, gate, -1.0f, dataLength);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Exp(output, output, dataLength);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds(output, output, 1.0f, dataLength);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Div(output, up, output, dataLength);
            AscendC::PipeBarrier<PIPE_V>();
        }

        // Tanh requires non-overlapping dst/src/tmp tensors. Reusing up as
        // dst satisfies that constraint without allocating another UB tile.
        if (beta != DEFAULT_BETA) {
            ComputeScaledTanh(up, gate, sharedTmpBuffer, dataLength, beta);
        } else {
            AscendC::Tanh(up, gate, sharedTmpBuffer, dataLength);
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::Mul(output, output, up, dataLength);
        AscendC::PipeBarrier<PIPE_V>();
    }
};

// Runtime dispatcher shared by all A2/A3 epilogues. New gated activations can
// be added here without duplicating data movement, tiling or quantization.
struct GatedActivation {
    static constexpr uint32_t PREFERRED_TILE_LENGTH = SwigluActivation::PREFERRED_TILE_LENGTH;
    static constexpr uint32_t MIN_SHARED_TMP_BYTES = SwigluActivation::MIN_SHARED_TMP_BYTES;

    CATLASS_DEVICE
    static bool IsInterleaved(uint32_t activationCode)
    {
        return activationCode == static_cast<uint32_t>(GatedActivationMode::SWIGLU_GPT_OSS);
    }

    // Split [gate0, up0, gate1, up1, ...] into two packed branches. GatherMask
    // patterns 1 and 2 select even and odd indices respectively. uint16_t is
    // used to preserve both FP16 and BF16 bit patterns.
    template <class Element>
    CATLASS_DEVICE static void SplitInterleaved(AscendC::LocalTensor<Element> const &packedInput,
                                                AscendC::LocalTensor<uint8_t> const &sharedTmpBuffer,
                                                uint32_t branchStride, uint32_t dataLength)
    {
        auto src = packedInput.template ReinterpretCast<uint16_t>();
        auto split = sharedTmpBuffer.template ReinterpretCast<uint16_t>();
        uint64_t evenCount = 0;
        uint64_t oddCount = 0;
        AscendC::GatherMask(split, src, static_cast<uint8_t>(1), true, dataLength * 2,
                            {1, 1, 8, 0}, evenCount);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::GatherMask(split[branchStride], src, static_cast<uint8_t>(2), true, dataLength * 2,
                            {1, 1, 8, 0}, oddCount);
        AscendC::PipeBarrier<PIPE_V>();
    }

    template <class Element>
    CATLASS_DEVICE static void PrepareBranches(AscendC::LocalTensor<float> const &gate,
                                               AscendC::LocalTensor<float> const &up,
                                               AscendC::LocalTensor<Element> const &packedInput,
                                               AscendC::LocalTensor<uint8_t> const &sharedTmpBuffer,
                                               uint32_t branchStride, uint32_t dataLength,
                                               uint32_t activationCode)
    {
        if (IsInterleaved(activationCode)) {
            SplitInterleaved(packedInput, sharedTmpBuffer, branchStride, dataLength);
            auto split = sharedTmpBuffer.template ReinterpretCast<Element>();
            AscendC::Cast(gate, split, AscendC::RoundMode::CAST_NONE, dataLength);
            AscendC::Cast(up, split[branchStride], AscendC::RoundMode::CAST_NONE, dataLength);
        } else {
            AscendC::Cast(gate, packedInput, AscendC::RoundMode::CAST_NONE, dataLength);
            AscendC::Cast(up, packedInput[branchStride], AscendC::RoundMode::CAST_NONE, dataLength);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    CATLASS_DEVICE
    static void SplitInterleavedFp32(AscendC::LocalTensor<float> const &packedInput,
                                     AscendC::LocalTensor<uint8_t> const &sharedTmpBuffer, uint32_t branchStride,
                                     uint32_t dataLength)
    {
        auto split = sharedTmpBuffer.template ReinterpretCast<float>();
        uint64_t evenCount = 0;
        uint64_t oddCount = 0;
        AscendC::GatherMask(split, packedInput, static_cast<uint8_t>(1), true, dataLength * 2,
                            {1, 1, 8, 0}, evenCount);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::GatherMask(split[branchStride], packedInput, static_cast<uint8_t>(2), true, dataLength * 2,
                            {1, 1, 8, 0}, oddCount);
        AscendC::PipeBarrier<PIPE_V>();
    }

    CATLASS_DEVICE
    static void Compute(AscendC::LocalTensor<float> const &output, AscendC::LocalTensor<float> const &gate,
                        AscendC::LocalTensor<float> const &up, AscendC::LocalTensor<uint8_t> const &sharedTmpBuffer,
                        uint32_t dataLength, float clampLimit, uint32_t activationCode,
                        float beta = SituActivation::DEFAULT_BETA,
                        float linearBeta = SituActivation::LINEAR_BETA_DISABLED)
    {
        if (activationCode == static_cast<uint32_t>(GatedActivationMode::SWIGLU_GPT_OSS)) {
            GptOssSwigluActivation::Compute(output, gate, up, sharedTmpBuffer, dataLength, clampLimit);
            return;
        }
        if (activationCode == static_cast<uint32_t>(GatedActivationMode::SITU)) {
            SituActivation::Compute(output, gate, up, sharedTmpBuffer, dataLength, linearBeta, beta);
            return;
        }
        SwigluActivation::Compute(output, gate, up, sharedTmpBuffer, dataLength, clampLimit);
    }
};

} // namespace Catlass::Epilogue

#endif // MEGA_MOE_GATED_ACTIVATION_HPP
