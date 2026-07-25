/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*! \file gmm_a8w4_perchannel_tensor_api.h
 *  \brief Thin GroupedMatmul adaptor for the ops-tensor A8W4 per-channel facade.
 */
#pragma once

#include <cstdint>
#include <type_traits>

#include "kernel_operator.h"
#include "blaze/gemm/kernel/kernel_gmm_a8w4.h"
#include "../grouped_matmul_tiling_data_apt.h"
#include "../../grouped_matmul_utils.h"

namespace GROUPED_MATMUL::A8W4V5 {

#if defined(BLAZE_UT) && defined(ASCENDC_CPU_DEBUG)
// CPU debug cannot execute the mixed Tensor API Cube/Vector handshake. Keep a
// scalar reference here solely for the end-to-end adapter numerical UT; cast,
// NZ conversion and row-sum preprocessors are tested through their real
// ops-tensor implementations.
template <typename YType>
__aicore__ inline void ProcessCpuDebugReference(
    GM_ADDR x, GM_ADDR weight, GM_ADDR scale, GM_ADDR offset, GM_ADDR groupList,
    GM_ADDR perTokenScale, GM_ADDR y,
    const GroupedMatmulTilingData::GMMA8W4BasicApiTilingData *tiling)
{
    AscendC::GlobalTensor<int8_t> xGm;
    AscendC::GlobalTensor<uint8_t> weightBytes;
    AscendC::GlobalTensor<uint64_t> scaleGm;
    AscendC::GlobalTensor<float> offsetGm;
    AscendC::GlobalTensor<float> tokenScaleGm;
    AscendC::GlobalTensor<int64_t> groupListGm;
    AscendC::GlobalTensor<YType> yGm;
    xGm.SetGlobalBuffer(GROUPED_MATMUL::GetTensorAddr<int8_t>(0, x));
    weightBytes.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(
        GROUPED_MATMUL::GetTensorAddr<AscendC::int4b_t>(0, weight)));
    scaleGm.SetGlobalBuffer(GROUPED_MATMUL::GetTensorAddr<uint64_t>(0, scale));
    offsetGm.SetGlobalBuffer(GROUPED_MATMUL::GetTensorAddr<float>(0, offset));
    tokenScaleGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(perTokenScale));
    groupListGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(groupList));
    yGm.SetGlobalBuffer(GROUPED_MATMUL::GetTensorAddr<YType>(0, y));

    const auto &mm = tiling->mmTilingData;
    uint64_t mBase = 0UL;
    for (uint32_t groupIdx = 0U; groupIdx < tiling->gmmQuantParams.groupNum; ++groupIdx) {
        const int64_t groupMValue = groupListGm.GetValue(groupIdx);
        const uint64_t groupM = groupMValue > 0 ? static_cast<uint64_t>(groupMValue) : 0UL;
        for (uint64_t mIdx = 0UL; mIdx < groupM; ++mIdx) {
            float rowSum = 0.0F;
            for (uint64_t kIdx = 0UL; kIdx < mm.k; ++kIdx) {
                rowSum += static_cast<float>(xGm.GetValue((mBase + mIdx) * mm.k + kIdx));
            }
            for (uint64_t nIdx = 0UL; nIdx < mm.n; ++nIdx) {
                int32_t dot = 0;
                for (uint64_t kIdx = 0UL; kIdx < mm.k; ++kIdx) {
                    const uint64_t logical =
                        (static_cast<uint64_t>(groupIdx) * mm.k + kIdx) * mm.n + nIdx;
                    const uint8_t packed = weightBytes.GetValue(logical >> 1U);
                    const uint8_t nibble =
                        (logical & 1U) == 0U ? packed & 0x0FU : packed >> 4U;
                    const int8_t w = static_cast<int8_t>((nibble ^ 0x08U) - 0x08U);
                    dot += static_cast<int32_t>(
                        xGm.GetValue((mBase + mIdx) * mm.k + kIdx)) * w;
                }
                const uint64_t encodedScale =
                    scaleGm.GetValue(static_cast<uint64_t>(groupIdx) * mm.n + nIdx);
                const uint32_t scaleBits = static_cast<uint32_t>(encodedScale >> 32U);
                const float channelScale = *reinterpret_cast<const float *>(&scaleBits);
                const float value =
                    (static_cast<float>(dot) * channelScale +
                     rowSum * offsetGm.GetValue(static_cast<uint64_t>(groupIdx) * mm.n + nIdx)) *
                    tokenScaleGm.GetValue(mBase + mIdx);
                yGm.SetValue(
                    (mBase + mIdx) * mm.n + nIdx, static_cast<YType>(value));
            }
        }
        mBase += groupM;
    }
}
#endif

template <typename YType, CubeFormat WFormat>
__aicore__ inline void InvokeGmmA8W4PerChannelTensorApi(
    GM_ADDR x, GM_ADDR weight, GM_ADDR scale, GM_ADDR offset, GM_ADDR groupList,
    GM_ADDR perTokenScale, GM_ADDR y, GM_ADDR workspace, AscendC::TPipe* pipe,
    const GroupedMatmulTilingData::GMMA8W4BasicApiTilingData *tiling)
{
#if defined(BLAZE_UT) && defined(ASCENDC_CPU_DEBUG)
    if ASCEND_IS_AIV {
        if (AscendC::GetBlockIdx() == 0U && AscendC::GetSubBlockIdx() == 0U) {
            ProcessCpuDebugReference<YType>(
                x, weight, scale, offset, groupList, perTokenScale, y, tiling);
        }
    }
    return;
#else
    using LayoutB = std::conditional_t<
        WFormat == CubeFormat::NZ, AscendC::Te::NZLayoutPtn, AscendC::Te::NDExtLayoutPtn>;
    using TensorApiKernel = Blaze::Gemm::Kernel::GmmA8W4PerChannel<LayoutB, YType>;
    using Params = typename TensorApiKernel::Params;
    const auto &mm = tiling->mmTilingData;
    const auto &a8w4 = tiling->a8w4Params;
    GM_ADDR expandedWeight = workspace + a8w4.expandedWeightOffsetBytes;
    GM_ADDR tileWorkspace = workspace + a8w4.tileWorkspaceOffsetBytes;
    GM_ADDR rowSumWorkspace = workspace + a8w4.rowSumOffsetBytes;

    Params params{};
    params.problemShape = typename TensorApiKernel::ProblemShape{
        static_cast<int64_t>(mm.m), static_cast<int64_t>(mm.n), static_cast<int64_t>(mm.k), 0L};
    params.weightPreprocessParams.weightGmAddr = reinterpret_cast<GM_ADDR>(
        GROUPED_MATMUL::GetTensorAddr<AscendC::int4b_t>(0, weight));
    params.weightPreprocessParams.workspaceGmAddr = expandedWeight;
    params.weightPreprocessParams.inputTransposedNz =
        WFormat == CubeFormat::NZ && a8w4.specialWeightFormat != 0U;
    params.rowSumPreprocessParams.xGmAddr = reinterpret_cast<GM_ADDR>(
        GROUPED_MATMUL::GetTensorAddr<int8_t>(0, x));
    params.rowSumPreprocessParams.rowSumGmAddr = rowSumWorkspace;
    params.mmadParams.aGmAddr = params.rowSumPreprocessParams.xGmAddr;
    params.mmadParams.bGmAddr = expandedWeight;
    params.mmadParams.cGmAddr = tileWorkspace;
    params.mmadParams.scaleBGmAddr = reinterpret_cast<GM_ADDR>(
        GROUPED_MATMUL::GetTensorAddr<uint64_t>(0, scale));
    params.epilogueParams.workspaceGmAddr = tileWorkspace;
    params.epilogueParams.perTokenScaleGmAddr = perTokenScale;
    params.epilogueParams.offsetGmAddr = reinterpret_cast<GM_ADDR>(
        GROUPED_MATMUL::GetTensorAddr<float>(0, offset));
    params.epilogueParams.xRowSumGmAddr = rowSumWorkspace;
    params.epilogueParams.outGmAddr = reinterpret_cast<GM_ADDR>(
        GROUPED_MATMUL::GetTensorAddr<YType>(0, y));
    params.groupListGmAddr = groupList;
    params.gmmParams.groupNum = tiling->gmmQuantParams.groupNum;
    params.gmmParams.m = mm.m;
    params.gmmParams.n = mm.n;
    params.gmmParams.k = mm.k;
    params.gmmParams.baseM = mm.baseM;
    params.gmmParams.baseN = mm.baseN;
    params.gmmParams.baseK = mm.baseK;
    params.gmmParams.quantGroupSize = a8w4.quantGroupSize;
    params.gmmParams.kAL1 = mm.kAL1;
    params.gmmParams.kBL1 = mm.kBL1;
    params.gmmParams.nBufferNum = 2U;
    params.gmmParams.dbL0C = mm.dbL0C;
    params.gmmParams.withOffset = a8w4.hasOffset;
    params.gmmParams.groupListType = tiling->gmmQuantParams.groupListType;
    {
        TensorApiKernel kernel;
        kernel(params, pipe);
    }
#if defined(BLAZE_A8W4_DEBUG)
    AscendC::printf("[A8W4][local kernel destroyed] block=%u ratio=%u\\n",
                    AscendC::GetBlockIdx(), AscendC::GetTaskRation());
#endif
#endif
}

} // namespace GROUPED_MATMUL::A8W4V5
