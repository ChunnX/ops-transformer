/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

#include "gtest/gtest.h"
#include "blaze_kernel_stub.h"
#include "kernel_ut_runner.h"
#include "tikicpulib.h"

#include "../../../../op_kernel/arch35/a8w4_v5/gmm_a8w4_perchannel_tensor_api.h"

namespace {

using TilingData = GroupedMatmulTilingData::GMMA8W4BasicApiTilingData;

template <typename YType, CubeFormat WFormat>
__global__ __aicore__ void GmmA8W4TensorApiUtKernel(GM_ADDR x, GM_ADDR weight, GM_ADDR scale, GM_ADDR offset,
                                                     GM_ADDR groupList, GM_ADDR perTokenScale, GM_ADDR y,
                                                     GM_ADDR workspace, GM_ADDR tiling)
{
    auto *tilingData = reinterpret_cast<const TilingData *>(tiling);
    AscendC::TPipe pipe;
    GROUPED_MATMUL::A8W4V5::InvokeGmmA8W4TensorApi<YType, WFormat>(
        x, weight, scale, offset, groupList, perTokenScale, y, workspace, &pipe, tilingData);
}

template <CubeFormat WFormat>
__global__ __aicore__ void GmmA8W4PreprocessUtKernel(GM_ADDR weight, GM_ADDR expandedWeight, GM_ADDR tiling)
{
    using LayoutB = std::conditional_t<WFormat == CubeFormat::NZ,
                                       AscendC::Te::NZLayoutPtn,
                                       AscendC::Te::NDExtLayoutPtn>;
    auto *tilingData = reinterpret_cast<const TilingData *>(tiling);
    typename Blaze::Gemm::Kernel::GmmA8W4WeightPreprocess<LayoutB>::Params params{};
    params.weightGmAddr = reinterpret_cast<GM_ADDR>(
        GROUPED_MATMUL::GetTensorAddr<AscendC::int4b_t>(0, weight));
    params.workspaceGmAddr = expandedWeight;
    params.groupNum = tilingData->gmmQuantParams.groupNum;
    params.n = tilingData->mmTilingData.n;
    params.k = tilingData->mmTilingData.k;
    params.inputTransposedNz =
        WFormat == CubeFormat::NZ && tilingData->a8w4Params.specialWeightFormat != 0U;
    AscendC::TPipe pipe;
    Blaze::Gemm::Kernel::GmmA8W4WeightPreprocess<LayoutB> preprocess;
    preprocess(params, &pipe);
}

__global__ __aicore__ void GmmA8W4RowSumPreprocessUtKernel(
    GM_ADDR x, GM_ADDR rowSum, uint64_t m, uint64_t k)
{
    AscendC::TPipe pipe;
    Blaze::Gemm::Kernel::GmmA8W4RowSumPreprocess::Params params{};
    params.xGmAddr = x;
    params.rowSumGmAddr = rowSum;
    params.m = m;
    params.k = k;
    params.enabled = true;
    Blaze::Gemm::Kernel::GmmA8W4RowSumPreprocess preprocess;
    preprocess(params, &pipe);
}

class GmBuffer {
public:
    explicit GmBuffer(size_t bytes) : ptr_(reinterpret_cast<uint8_t *>(AscendC::GmAlloc(bytes)))
    {
        if (ptr_ != nullptr) {
            std::fill_n(ptr_, bytes, 0U);
        }
    }

    ~GmBuffer()
    {
        if (ptr_ != nullptr) {
            AscendC::GmFree(ptr_);
        }
    }

    GmBuffer(const GmBuffer &) = delete;
    GmBuffer &operator=(const GmBuffer &) = delete;

    GM_ADDR Get() const
    {
        return reinterpret_cast<GM_ADDR>(ptr_);
    }

    template <typename T>
    T *As() const
    {
        return reinterpret_cast<T *>(ptr_);
    }

private:
    uint8_t *ptr_{nullptr};
};

void MakeTensorList(GmBuffer &list, GM_ADDR tensor)
{
    auto *data = list.As<uint64_t>();
    data[0] = sizeof(uint64_t);
    data[1] = reinterpret_cast<uint64_t>(tensor);
}

void SetInt4(GmBuffer &buffer, uint64_t logicalIndex, int8_t value)
{
    auto *bytes = buffer.As<uint8_t>();
    const uint8_t nibble = static_cast<uint8_t>(value) & 0x0FU;
    const uint64_t byteIndex = logicalIndex >> 1U;
    if ((logicalIndex & 1U) == 0U) {
        bytes[byteIndex] = static_cast<uint8_t>((bytes[byteIndex] & 0xF0U) | nibble);
    } else {
        bytes[byteIndex] = static_cast<uint8_t>((bytes[byteIndex] & 0x0FU) | (nibble << 4U));
    }
}

int8_t PatternValue(uint32_t k, uint32_t n, uint32_t expert = 0U)
{
    constexpr int8_t VALUES[] = {-8, -1, 0, 7};
    return VALUES[(k + n + expert) % 4U];
}

TEST(GroupedMatmulA8W4V5Arch35Kernel, FullCastNdSignedBoundariesAndExperts)
{
    constexpr uint32_t E = 2U;
    constexpr uint32_t K = 3U;
    constexpr uint32_t N = 8U;
    constexpr uint64_t EXPERT_STRIDE = static_cast<uint64_t>(K) * N;

    GmBuffer weightData(E * EXPERT_STRIDE / 2U);
    GmBuffer weightList(2U * sizeof(uint64_t));
    GmBuffer expandedWeight(E * EXPERT_STRIDE);
    GmBuffer tilingBuffer(sizeof(TilingData));
    MakeTensorList(weightList, weightData.Get());
    for (uint32_t e = 0U; e < E; ++e) {
        for (uint32_t k = 0U; k < K; ++k) {
            for (uint32_t n = 0U; n < N; ++n) {
                SetInt4(weightData, (static_cast<uint64_t>(e) * K + k) * N + n, PatternValue(k, n, e));
            }
        }
    }

    auto *tiling = tilingBuffer.As<TilingData>();
    tiling->gmmQuantParams.groupNum = E;
    tiling->a8w4Params.expandedWeightStrideBytes = EXPERT_STRIDE;
    tiling->mmTilingData.k = K;
    tiling->mmTilingData.n = N;

    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    auto kernel = GmmA8W4PreprocessUtKernel<CubeFormat::ND>;
    ASSERT_TRUE(KERNEL_RUN_KF(kernel, 1U, weightList.Get(), expandedWeight.Get(), tilingBuffer.Get()));
    for (uint32_t e = 0U; e < E; ++e) {
        for (uint32_t k = 0U; k < K; ++k) {
            for (uint32_t n = 0U; n < N; ++n) {
                const uint64_t index = (static_cast<uint64_t>(e) * K + k) * N + n;
                EXPECT_EQ(expandedWeight.As<int8_t>()[index], PatternValue(k, n, e)) << "at " << index;
            }
        }
    }
}

void RunNzCastAndCheck(bool specialWeightFormat)
{
    constexpr uint32_t E = 1U;
    constexpr uint32_t K = 17U;
    constexpr uint32_t N = 40U;
    constexpr uint32_t INPUT_K_NZ = 32U;
    constexpr uint32_t INPUT_N_NZ = 64U;
    constexpr uint32_t SPECIAL_INPUT_K_NZ = 64U;
    constexpr uint32_t SPECIAL_INPUT_N_NZ = 48U;
    constexpr uint32_t OUTPUT_K_NZ = 32U;
    constexpr uint32_t OUTPUT_N_NZ = 64U;
    constexpr uint64_t OUTPUT_STRIDE = static_cast<uint64_t>(OUTPUT_K_NZ) * OUTPUT_N_NZ;
    const uint64_t inputElements = specialWeightFormat ?
        static_cast<uint64_t>(SPECIAL_INPUT_K_NZ) * SPECIAL_INPUT_N_NZ :
        static_cast<uint64_t>(INPUT_K_NZ) * INPUT_N_NZ;

    GmBuffer weightData(inputElements / 2U);
    GmBuffer weightList(2U * sizeof(uint64_t));
    GmBuffer expandedWeight(OUTPUT_STRIDE);
    GmBuffer tilingBuffer(sizeof(TilingData));
    MakeTensorList(weightList, weightData.Get());
    for (uint32_t k = 0U; k < K; ++k) {
        for (uint32_t n = 0U; n < N; ++n) {
            const uint64_t inputIndex = specialWeightFormat ?
                static_cast<uint64_t>(n) * SPECIAL_INPUT_K_NZ + k :
                static_cast<uint64_t>(k) * INPUT_N_NZ + n;
            SetInt4(weightData, inputIndex, PatternValue(k, n));
        }
    }

    auto *tiling = tilingBuffer.As<TilingData>();
    tiling->gmmQuantParams.groupNum = E;
    tiling->a8w4Params.specialWeightFormat = static_cast<uint8_t>(specialWeightFormat);
    tiling->a8w4Params.expandedWeightStrideBytes = OUTPUT_STRIDE;
    tiling->mmTilingData.k = K;
    tiling->mmTilingData.n = N;

    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    auto kernel = GmmA8W4PreprocessUtKernel<CubeFormat::NZ>;
    ASSERT_TRUE(KERNEL_RUN_KF(kernel, 1U, weightList.Get(), expandedWeight.Get(), tilingBuffer.Get()));
    for (uint32_t n = 0U; n < OUTPUT_N_NZ; ++n) {
        for (uint32_t k = 0U; k < OUTPUT_K_NZ; ++k) {
            const uint64_t outputIndex = static_cast<uint64_t>(n / 32U) * OUTPUT_K_NZ * 32U +
                                         static_cast<uint64_t>(k) * 32U + n % 32U;
            const int8_t expected = k < K && n < N ? PatternValue(k, n) : 0;
            EXPECT_EQ(expandedWeight.As<int8_t>()[outputIndex], expected)
                << "special=" << specialWeightFormat << ", k=" << k << ", n=" << n;
        }
    }
}

TEST(GroupedMatmulA8W4V5Arch35Kernel, FullCastNzC064ToC032AndPadding)
{
    RunNzCastAndCheck(false);
}

TEST(GroupedMatmulA8W4V5Arch35Kernel, FullCastSpecialEnkToStandardNz)
{
    RunNzCastAndCheck(true);
}

TEST(GroupedMatmulA8W4V5Arch35Kernel, RowSumFp32Handles4096ChunkAndKTail)
{
    constexpr uint32_t M = 5U;
    constexpr uint32_t K = 4099U;
    GmBuffer xData(static_cast<size_t>(M) * K);
    GmBuffer rowSumData(static_cast<size_t>(M) * sizeof(float));
    std::array<float, M> expected{};
    for (uint32_t m = 0U; m < M; ++m) {
        for (uint32_t k = 0U; k < K; ++k) {
            const int8_t value = static_cast<int8_t>((m * 3U + k) % 15U) - 7;
            xData.As<int8_t>()[static_cast<size_t>(m) * K + k] = value;
            expected[m] += static_cast<float>(value);
        }
    }

    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    ASSERT_TRUE(KERNEL_RUN_KF(GmmA8W4RowSumPreprocessUtKernel, 1U,
                              xData.Get(), rowSumData.Get(), M, K));
    for (uint32_t m = 0U; m < M; ++m) {
        EXPECT_FLOAT_EQ(rowSumData.As<float>()[m], expected[m]) << "at row " << m;
    }
}

TEST(GroupedMatmulA8W4V5Arch35Kernel, BaseM256EpilogueFitsA5UbPerVectorSubblock)
{
    using Epilogue = Blaze::Epilogue::Block::BlockEpilogueGmmA8W4<half, half>;
    constexpr uint64_t A5_UB_BYTES = 256UL * 1024UL;
    constexpr uint64_t EXPECTED_UB_BYTES = 130UL * 1024UL;
    EXPECT_EQ(Epilogue::GetUbSizeBytes(256, 256), EXPECTED_UB_BYTES);
    EXPECT_LT(Epilogue::GetUbSizeBytes(256, 256), A5_UB_BYTES);

    const int64_t firstHalf = (255 + 1) / 2;
    const int64_t secondHalf = 255 - firstHalf;
    EXPECT_EQ(firstHalf, 128);
    EXPECT_EQ(secondHalf, 127);
    EXPECT_EQ(firstHalf + secondHalf, 255);
}

TEST(GroupedMatmulA8W4V5Arch35Kernel, FixedTileL1FootprintMatchesPhysicalNzLayout)
{
    using Resource = Blaze::Gemm::A8W4FixedTileResource;
    EXPECT_EQ(Resource::GetAL1Bytes(16UL, 256UL), 4096UL);
    EXPECT_EQ(Resource::GetBL1Bytes(16UL, 256UL), 8192UL);
    EXPECT_EQ(Resource::GetScaleL1Bytes(16UL), 128UL);
    EXPECT_EQ(Resource::GetL1StageBytes(16UL, 16UL, 256UL, 256UL), 12416UL);
    EXPECT_EQ(Resource::GetL1StageBytes(256UL, 256UL, 256UL, 256UL), 130UL * 1024UL);
}

TEST(GroupedMatmulA8W4V5Arch35Kernel, PerChannelNdApiNumericalReferenceWithKTail)
{
    constexpr uint32_t M = 16U;
    constexpr uint32_t N = 32U;
    constexpr uint32_t K = 65U;
    constexpr uint32_t E = 1U;
    constexpr uint32_t CORE_NUM = 1U;
    constexpr size_t WORKSPACE_SIZE = 16UL * 1024UL * 1024UL;

    GmBuffer xData(static_cast<size_t>(M) * K);
    GmBuffer weightData(static_cast<size_t>(E) * K * N / 2U);
    GmBuffer scaleData(static_cast<size_t>(E) * N * sizeof(uint64_t));
    GmBuffer offsetData(static_cast<size_t>(E) * N * sizeof(float));
    GmBuffer groupListData(E * sizeof(int64_t));
    GmBuffer perTokenScaleData(M * sizeof(float));
    GmBuffer yData(static_cast<size_t>(M) * N * sizeof(half));
    GmBuffer workspace(WORKSPACE_SIZE);
    GmBuffer tilingBuffer(sizeof(TilingData));
    GmBuffer xList(2U * sizeof(uint64_t));
    GmBuffer weightList(2U * sizeof(uint64_t));
    GmBuffer scaleList(2U * sizeof(uint64_t));
    GmBuffer offsetList(2U * sizeof(uint64_t));
    GmBuffer yList(2U * sizeof(uint64_t));

    ASSERT_NE(xData.Get(), nullptr);
    ASSERT_NE(workspace.Get(), nullptr);
    std::fill_n(xData.As<int8_t>(), static_cast<size_t>(M) * K, static_cast<int8_t>(1));
    std::fill_n(weightData.As<uint8_t>(), static_cast<size_t>(E) * K * N / 2U, 0x11U);
    std::fill_n(scaleData.As<uint64_t>(), static_cast<size_t>(E) * N, 0x3F80000000000000ULL);
    std::fill_n(offsetData.As<float>(), static_cast<size_t>(E) * N, 0.5F);
    std::fill_n(perTokenScaleData.As<float>(), M, 1.0F);
    groupListData.As<int64_t>()[0] = M;

    MakeTensorList(xList, xData.Get());
    MakeTensorList(weightList, weightData.Get());
    MakeTensorList(scaleList, scaleData.Get());
    MakeTensorList(offsetList, offsetData.Get());
    MakeTensorList(yList, yData.Get());

    auto *tiling = tilingBuffer.As<TilingData>();
    tiling->gmmQuantParams.groupNum = E;
    tiling->gmmQuantParams.singleX = 1U;
    tiling->gmmQuantParams.singleW = 1U;
    tiling->gmmQuantParams.singleY = 1U;
    tiling->gmmQuantParams.groupType = 0;
    tiling->gmmQuantParams.groupListType = 1U;
    tiling->a8w4Params.coreNum = CORE_NUM;
    tiling->a8w4Params.dequantMode = 1U;
    tiling->a8w4Params.hasOffset = 1U;
    tiling->mmTilingData.m = M;
    tiling->mmTilingData.n = N;
    tiling->mmTilingData.k = K;
    tiling->mmTilingData.baseM = M;
    tiling->mmTilingData.baseN = N;
    tiling->mmTilingData.baseK = 64U;
    tiling->mmTilingData.kAL1 = K;
    tiling->mmTilingData.kBL1 = K;
    tiling->mmTilingData.dbL0C = 1U;

    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    auto kernel = GmmA8W4TensorApiUtKernel<half, CubeFormat::ND>;
    ASSERT_TRUE(KERNEL_RUN_KF(kernel, CORE_NUM, xList.Get(), weightList.Get(), scaleList.Get(), offsetList.Get(),
                              groupListData.Get(), perTokenScaleData.Get(), yList.Get(), workspace.Get(),
                              tilingBuffer.Get()));

    constexpr float expected = static_cast<float>(K) * 1.5F;
    for (uint32_t i = 0U; i < M * N; ++i) {
        EXPECT_NEAR(static_cast<float>(yData.As<half>()[i]), expected, 0.5F) << "at output index " << i;
    }
}

TEST(GroupedMatmulA8W4V5Arch35Kernel, PerGroupNdApiNumericalReference)
{
    constexpr uint32_t M = 16U;
    constexpr uint32_t N = 32U;
    constexpr uint32_t K = 64U;
    constexpr uint32_t E = 1U;
    constexpr uint32_t QUANT_GROUP_SIZE = 32U;
    constexpr uint32_t QUANT_GROUP_NUM = K / QUANT_GROUP_SIZE;
    constexpr size_t WORKSPACE_SIZE = 16UL * 1024UL * 1024UL;

    GmBuffer xData(static_cast<size_t>(M) * K);
    GmBuffer weightData(static_cast<size_t>(E) * K * N / 2U);
    GmBuffer scaleData(static_cast<size_t>(E) * QUANT_GROUP_NUM * N * sizeof(uint64_t));
    GmBuffer offsetData(sizeof(float));
    GmBuffer groupListData(E * sizeof(int64_t));
    GmBuffer perTokenScaleData(M * sizeof(float));
    GmBuffer yData(static_cast<size_t>(M) * N * sizeof(half));
    GmBuffer workspace(WORKSPACE_SIZE);
    GmBuffer tilingBuffer(sizeof(TilingData));
    GmBuffer xList(2U * sizeof(uint64_t));
    GmBuffer weightList(2U * sizeof(uint64_t));
    GmBuffer scaleList(2U * sizeof(uint64_t));
    GmBuffer offsetList(2U * sizeof(uint64_t));
    GmBuffer yList(2U * sizeof(uint64_t));

    std::fill_n(xData.As<int8_t>(), static_cast<size_t>(M) * K, static_cast<int8_t>(1));
    std::fill_n(weightData.As<uint8_t>(), static_cast<size_t>(E) * K * N / 2U, 0x11U);
    std::fill_n(scaleData.As<uint64_t>(), N, 0x3F80000000000000ULL);
    std::fill_n(scaleData.As<uint64_t>() + N, N, 0x4000000000000000ULL);
    std::fill_n(perTokenScaleData.As<float>(), M, 1.0F);
    groupListData.As<int64_t>()[0] = M;

    MakeTensorList(xList, xData.Get());
    MakeTensorList(weightList, weightData.Get());
    MakeTensorList(scaleList, scaleData.Get());
    MakeTensorList(offsetList, offsetData.Get());
    MakeTensorList(yList, yData.Get());

    auto *tiling = tilingBuffer.As<TilingData>();
    tiling->gmmQuantParams.groupNum = E;
    tiling->gmmQuantParams.singleX = 1U;
    tiling->gmmQuantParams.singleW = 1U;
    tiling->gmmQuantParams.singleY = 1U;
    tiling->gmmQuantParams.groupListType = 1U;
    tiling->a8w4Params.dequantMode = 0U;
    tiling->a8w4Params.quantGroupSize = QUANT_GROUP_SIZE;
    tiling->a8w4Params.quantGroupNum = QUANT_GROUP_NUM;
    tiling->mmTilingData.m = M;
    tiling->mmTilingData.n = N;
    tiling->mmTilingData.k = K;
    tiling->mmTilingData.baseM = M;
    tiling->mmTilingData.baseN = N;
    tiling->mmTilingData.baseK = QUANT_GROUP_SIZE;
    tiling->mmTilingData.kAL1 = QUANT_GROUP_SIZE;
    tiling->mmTilingData.kBL1 = QUANT_GROUP_SIZE;
    tiling->mmTilingData.dbL0C = 1U;

    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    auto kernel = GmmA8W4TensorApiUtKernel<half, CubeFormat::ND>;
    ASSERT_TRUE(KERNEL_RUN_KF(kernel, 1U, xList.Get(), weightList.Get(), scaleList.Get(), offsetList.Get(),
                              groupListData.Get(), perTokenScaleData.Get(), yList.Get(), workspace.Get(),
                              tilingBuffer.Get()));

    constexpr float expected = QUANT_GROUP_SIZE * 1.0F + QUANT_GROUP_SIZE * 2.0F;
    for (uint32_t i = 0U; i < M * N; ++i) {
        EXPECT_NEAR(static_cast<float>(yData.As<half>()[i]), expected, 0.5F) << "at output index " << i;
    }
}
} // namespace
