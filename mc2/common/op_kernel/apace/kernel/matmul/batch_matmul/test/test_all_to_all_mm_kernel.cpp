/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_all_to_all_mm_kernel.cpp
 * \brief AllToAllMmKernel（非量化 all-to-all MatMul，GM 累加版）Kernel UT 测试。
 *
 * 测试原理：
 * 1. 预填"通信 buffer"（remoteAGM = 全部 rank 的 A 数据），CommPolicy 使用 no-op stub，
 *    模拟通信已完成，聚焦验证 kernel 的矩阵乘计算、GM 累加（覆盖写 + 原子累加）与 bias 语义。
 * 2. 三种调度模式各自验证：
 *    - 模式1 (localMatmul=0, REMOTE)：单次 launch，遍历所有 rank，首次覆盖写 + 其余原子累加
 *    - 模式2 (localMatmul=1, DEFERRED_SYNC)：单次 launch，self 先写 + 其余原子累加
 *    - 模式3 (localMatmul=2)：LOCAL launch（覆盖写 + bias）后再 REMOTE launch（跳过 self 原子累加）
 * 3. 参考实现：C[m][n] = bias[n] + Σ_{r=0}^{rankSize-1} Σ_{k=0}^{K-1} A[r*oriM+m][k] * B[r*K+k][n]
 * 4. 输出回读 GM buffer 与参考实现逐元素比对（fp16 输出，容忍量化误差）。
 */

#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "blaze_kernel_stub.h"
#include "kernel_ut_runner.h"
#include "tikicpulib.h"
#include "kernel_operator.h"

#include "all_to_all_mm_kernel.h"

namespace {

// 通信等待策略 stub：UT 中通信 buffer 已预填，无需真实 wait
struct NoopCommWaitPolicy {
    __aicore__ inline void WaitTile(uint32_t tileIdx) {}
};

// 测试用 tiling 数据（host ↔ device 共享布局，字段顺序即序列化契约）
struct AllToAllMmTestTiling {
    uint32_t m; // 输出 M = originalM
    uint32_t n;
    uint32_t k; // 单 rank 的 K 段长度
    uint32_t baseM;
    uint32_t baseN;
    uint32_t baseK;
    uint32_t mTailTile;
    uint32_t nTailTile;
    uint32_t mBaseTailSplitCnt;
    uint32_t nBaseTailSplitCnt;
    uint32_t mTailMain;
    uint32_t nTailMain;
    uint32_t stepK;
    uint32_t nBufferNum;
    uint32_t dbL0C;
    uint32_t rankId;
    uint32_t rankSize;
    uint32_t originalM;
    uint32_t localMatmul; // 0: remote, 1: DEFERRED_SYNC 语义, 2: GM 累加
    uint32_t matmulMode;  // 0: REMOTE, 1: LOCAL, 2: DEFERRED_SYNC
    uint32_t headTileSize;
};

template <typename AType, typename BType, typename CType, typename BiasType, bool TransA, bool TransB>
__global__ __aicore__ void all_to_all_mm_kernel_entry(GM_ADDR remoteAGM, GM_ADDR localAGM, GM_ADDR bGM,
                                                      GM_ADDR biasGM, GM_ADDR cGM, GM_ADDR zeroBiasGM,
                                                      GM_ADDR tilingGM)
{
    AllToAllMmTestTiling tiling;
    memcpy(&tiling, tilingGM, sizeof(AllToAllMmTestTiling));

    using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    using LayoutA = std::conditional_t<TransA, AscendC::Te::DNExtLayoutPtn, AscendC::Te::NDExtLayoutPtn>;
    using LayoutB = std::conditional_t<TransB, AscendC::Te::DNExtLayoutPtn, AscendC::Te::NDExtLayoutPtn>;
    using LayoutC = AscendC::Te::NDExtLayoutPtn;
    using LayoutBias = AscendC::Te::NDExtLayoutPtn;

    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<ProblemShape,
                                                                                Blaze::Gemm::NONE_FULL_LOAD_MODE,
                                                                                LayoutA, LayoutB, AType>;
    using DispatchPolicy = Blaze::Gemm::MatmulMultiBlockBasic<Blaze::Gemm::NONE_FULL_LOAD_MODE, 0,
                                                              Blaze::Gemm::KernelMmadMultiBlockBasic, 0>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, AType, LayoutA, BType, LayoutB, CType, LayoutC,
                                                    BiasType, LayoutBias>;
    using KernelImpl =
        Blaze::Gemm::Kernel::AllToAllMmKernel<ProblemShape, BlockMmad, BlockScheduler, NoopCommWaitPolicy>;

    typename KernelImpl::Params params;
    params.problemShape = ProblemShape{static_cast<int64_t>(tiling.m), static_cast<int64_t>(tiling.n),
                                       static_cast<int64_t>(tiling.k), 1L};
    params.mmadParams.aGmAddr = remoteAGM;
    params.mmadParams.bGmAddr = bGM;
    params.mmadParams.cGmAddr = cGM;
    params.mmadParams.biasGmAddr = biasGM;
    params.mmadParams.mL1 = tiling.baseM;
    params.mmadParams.nL1 = tiling.baseN;
    params.mmadParams.kL1 = tiling.stepK * tiling.baseK;
    params.mmadParams.mL0 = tiling.baseM;
    params.mmadParams.nL0 = tiling.baseN;
    params.mmadParams.kL0 = tiling.baseK;
    params.mmadParams.l1Stages = tiling.nBufferNum;
    params.mmadParams.l0cStages = (tiling.dbL0C > 1) ? 2 : 1;
    params.schParams = {static_cast<int64_t>(tiling.baseM), static_cast<int64_t>(tiling.baseN),
                        static_cast<int64_t>(tiling.mTailTile), static_cast<int64_t>(tiling.nTailTile),
                        static_cast<int64_t>(tiling.mBaseTailSplitCnt),
                        static_cast<int64_t>(tiling.nBaseTailSplitCnt), static_cast<int64_t>(tiling.mTailMain),
                        static_cast<int64_t>(tiling.nTailMain)};
    params.mmTiling = {tiling.baseM, tiling.baseN, tiling.baseK, tiling.dbL0C};
    params.localParams = {tiling.rankId,
                          tiling.rankSize,
                          tiling.originalM,
                          localAGM,
                          zeroBiasGM,
                          tiling.localMatmul,
                          static_cast<typename KernelImpl::MatmulMode>(tiling.matmulMode),
                          tiling.headTileSize};

    KernelImpl kernel;
    kernel(params);
}

// 确定性数据填充：A/B ∈ [-0.5, 0.5)，bias ∈ [-1.0, 1.0)
inline void FillData(std::vector<float> &data)
{
    for (size_t i = 0; i < data.size(); ++i) {
        uint32_t x = static_cast<uint32_t>(i * 2654435761u + 97u);
        data[i] = (static_cast<float>((x >> 8) & 0xff) / 256.0f) - 0.5f;
    }
}

// CPU 参考实现
std::vector<float> ReferenceC(const std::vector<float> &a, const std::vector<float> &b,
                              const std::vector<float> &bias, uint32_t rankSize, uint32_t oriM, uint32_t k,
                              uint32_t n)
{
    std::vector<float> ref(static_cast<size_t>(oriM) * n, 0.0f);
    for (uint32_t i = 0; i < oriM; ++i) {
        for (uint32_t j = 0; j < n; ++j) {
            double acc = bias[j];
            for (uint32_t r = 0; r < rankSize; ++r) {
                for (uint32_t kk = 0; kk < k; ++kk) {
                    acc += static_cast<double>(a[(r * oriM + i) * k + kk]) *
                           static_cast<double>(b[(r * k + kk) * n + j]);
                }
            }
            ref[i * n + j] = static_cast<float>(acc);
        }
    }
    return ref;
}

void CheckResult(const std::vector<float> &ref, const half *c, uint32_t oriM, uint32_t n)
{
    double maxDiff = 0.0;
    double maxRef = 0.0;
    for (uint32_t i = 0; i < oriM * n; ++i) {
        float got = static_cast<float>(c[i]);
        double diff = std::fabs(static_cast<double>(got) - ref[i]);
        maxDiff = std::max(maxDiff, diff);
        maxRef = std::max(maxRef, std::fabs(static_cast<double>(ref[i])));
    }
    const double tolerance = 0.05 + 0.01 * maxRef;
    EXPECT_LE(maxDiff, tolerance) << "maxDiff=" << maxDiff << " maxRef=" << maxRef << " tolerance=" << tolerance;
}

} // namespace

class AllToAllMmKernelTest : public testing::Test {
public:
    void SetUp() override
    {
        remoteAGM = nullptr;
        localAGM = nullptr;
        bGM = nullptr;
        biasGM = nullptr;
        cGM = nullptr;
        zeroBiasGM = nullptr;
        tilingGM = nullptr;
    }

    void TearDown() override
    {
        if (remoteAGM)
            AscendC::GmFree((void *)remoteAGM);
        if (localAGM)
            AscendC::GmFree((void *)localAGM);
        if (bGM)
            AscendC::GmFree((void *)bGM);
        if (biasGM)
            AscendC::GmFree((void *)biasGM);
        if (cGM)
            AscendC::GmFree((void *)cGM);
        if (zeroBiasGM)
            AscendC::GmFree((void *)zeroBiasGM);
        if (tilingGM)
            AscendC::GmFree((void *)tilingGM);
    }

    GM_ADDR remoteAGM;
    GM_ADDR localAGM;
    GM_ADDR bGM;
    GM_ADDR biasGM;
    GM_ADDR cGM;
    GM_ADDR zeroBiasGM;
    GM_ADDR tilingGM;
};

// 运行一次 kernel launch（mode1/mode2 单次，mode3 由调用方先后 launch LOCAL 与 REMOTE）
void RunOneLaunch(GM_ADDR remoteAGM, GM_ADDR localAGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR cGM, GM_ADDR zeroBiasGM,
                  GM_ADDR tilingGM, uint32_t blockNum)
{
    auto kernelFunc = all_to_all_mm_kernel_entry<half, half, half, float, false, false>;
    EXPECT_TRUE(KERNEL_RUN_KF(kernelFunc, blockNum, remoteAGM, localAGM, bGM, biasGM, cGM, zeroBiasGM, tilingGM))
        << "Kernel execution failed";
}

void FillTiling(AllToAllMmTestTiling &tiling, uint32_t oriM, uint32_t n, uint32_t k, uint32_t rankSize,
                uint32_t rankId, uint32_t localMatmul, uint32_t matmulMode)
{
    tiling.m = oriM;
    tiling.n = n;
    tiling.k = k;
    tiling.baseM = 16;
    tiling.baseN = 16;
    tiling.baseK = 16;
    tiling.mTailTile = 1;
    tiling.nTailTile = 1;
    tiling.mBaseTailSplitCnt = 1;
    tiling.nBaseTailSplitCnt = 1;
    tiling.mTailMain = 0;
    tiling.nTailMain = 0;
    tiling.stepK = 1;
    tiling.nBufferNum = 2;
    tiling.dbL0C = 1;
    tiling.rankId = rankId;
    tiling.rankSize = rankSize;
    tiling.originalM = oriM;
    tiling.localMatmul = localMatmul;
    tiling.matmulMode = matmulMode;
    tiling.headTileSize = oriM;
}

// 通用场景验证：rankSize 个 rank 的 all-to-all MatMul
// twoPhase=true 时执行模式3（LOCAL launch + REMOTE launch），否则单次 launch（mode 由 localMatmul/matmulMode 决定）
void RunAndVerify(AllToAllMmKernelTest &test, uint32_t rankSize, uint32_t oriM, uint32_t k, uint32_t n,
                  uint32_t blockNum, uint32_t localMatmul, uint32_t matmulMode, bool twoPhase)
{
    const uint32_t rankId = 0; // 单进程 UT 固定 rank 0，self 数据取 A[0:oriM]

    size_t remoteASize = static_cast<size_t>(rankSize) * oriM * k * sizeof(half);
    size_t localASize = static_cast<size_t>(oriM) * k * sizeof(half);
    size_t bSize = static_cast<size_t>(rankSize) * k * n * sizeof(half);
    size_t biasSize = static_cast<size_t>(n) * sizeof(float);
    size_t cSize = static_cast<size_t>(oriM) * n * sizeof(half);
    size_t zeroBiasSize = 16 * sizeof(float);
    size_t tilingSize = sizeof(AllToAllMmTestTiling);

    test.remoteAGM = (GM_ADDR)AscendC::GmAlloc(remoteASize);
    test.localAGM = (GM_ADDR)AscendC::GmAlloc(localASize);
    test.bGM = (GM_ADDR)AscendC::GmAlloc(bSize);
    test.biasGM = (GM_ADDR)AscendC::GmAlloc(biasSize);
    test.cGM = (GM_ADDR)AscendC::GmAlloc(cSize);
    test.zeroBiasGM = (GM_ADDR)AscendC::GmAlloc(zeroBiasSize);
    test.tilingGM = (GM_ADDR)AscendC::GmAlloc(tilingSize);

    ASSERT_NE(test.remoteAGM, nullptr);
    ASSERT_NE(test.localAGM, nullptr);
    ASSERT_NE(test.bGM, nullptr);
    ASSERT_NE(test.biasGM, nullptr);
    ASSERT_NE(test.cGM, nullptr);
    ASSERT_NE(test.zeroBiasGM, nullptr);
    ASSERT_NE(test.tilingGM, nullptr);

    // 生成 host 侧数据并写入 GM
    std::vector<float> aHost(static_cast<size_t>(rankSize) * oriM * k);
    std::vector<float> bHost(static_cast<size_t>(rankSize) * k * n);
    std::vector<float> biasHost(n);
    FillData(aHost);
    FillData(bHost);
    FillData(biasHost);
    std::vector<half> aHalf(aHost.begin(), aHost.end());
    std::vector<half> bHalf(bHost.begin(), bHost.end());

    std::memcpy(test.remoteAGM, aHalf.data(), remoteASize); // 通信 buffer = 全部 rank 的 A
    std::memcpy(test.localAGM, aHalf.data() + static_cast<size_t>(rankId) * oriM * k, localASize);
    std::memcpy(test.bGM, bHalf.data(), bSize);
    // bias 为 float 类型（BiasType=float），直接以 float 写入
    std::memcpy(test.biasGM, biasHost.data(), biasSize);
    std::memset(test.zeroBiasGM, 0, zeroBiasSize); // zero-bias workspace 必须清零
    std::memset(test.cGM, 0, cSize);

    AllToAllMmTestTiling tiling;
    FillTiling(tiling, oriM, n, k, rankSize, rankId, localMatmul, matmulMode);
    std::memcpy(test.tilingGM, &tiling, tilingSize);

    if (twoPhase) {
        // 模式3：LOCAL launch（覆盖写 + bias） → REMOTE launch（跳过 self，原子累加）
        tiling.matmulMode = 1; // MatmulMode::LOCAL
        std::memcpy(test.tilingGM, &tiling, tilingSize);
        RunOneLaunch(test.remoteAGM, test.localAGM, test.bGM, test.biasGM, test.cGM, test.zeroBiasGM, test.tilingGM,
                     blockNum);
        tiling.matmulMode = 0; // MatmulMode::REMOTE
        std::memcpy(test.tilingGM, &tiling, tilingSize);
        RunOneLaunch(test.remoteAGM, test.localAGM, test.bGM, test.biasGM, test.cGM, test.zeroBiasGM, test.tilingGM,
                     blockNum);
    } else {
        RunOneLaunch(test.remoteAGM, test.localAGM, test.bGM, test.biasGM, test.cGM, test.zeroBiasGM, test.tilingGM,
                     blockNum);
    }

    // 参考实现并比对
    std::vector<float> ref = ReferenceC(aHost, bHost, biasHost, rankSize, oriM, k, n);
    const half *c = reinterpret_cast<const half *>(test.cGM);
    CheckResult(ref, c, oriM, n);
}

TEST_F(AllToAllMmKernelTest, Test_FP16_Mode1_Remote_MultiRank)
{
    RunAndVerify(*this, /*rankSize=*/2, /*oriM=*/16, /*k=*/16, /*n=*/16, /*blockNum=*/1,
                 /*localMatmul=*/0, /*matmulMode=*/0, /*twoPhase=*/false);
}

TEST_F(AllToAllMmKernelTest, Test_FP16_Mode2_DeferredSync_MultiRank)
{
    RunAndVerify(*this, /*rankSize=*/2, /*oriM=*/16, /*k=*/16, /*n=*/16, /*blockNum=*/1,
                 /*localMatmul=*/1, /*matmulMode=*/2, /*twoPhase=*/false);
}

TEST_F(AllToAllMmKernelTest, Test_FP16_Mode3_LocalRemote_MultiRank)
{
    RunAndVerify(*this, /*rankSize=*/2, /*oriM=*/16, /*k=*/16, /*n=*/16, /*blockNum=*/1,
                 /*localMatmul=*/2, /*matmulMode=*/0, /*twoPhase=*/true);
}

TEST_F(AllToAllMmKernelTest, Test_FP16_Mode1_Remote_MultiCore)
{
    // 2×2 = 4 个输出块，4 核并行
    RunAndVerify(*this, /*rankSize=*/2, /*oriM=*/32, /*k=*/16, /*n=*/32, /*blockNum=*/4,
                 /*localMatmul=*/0, /*matmulMode=*/0, /*twoPhase=*/false);
}

TEST_F(AllToAllMmKernelTest, Test_FP16_Mode1_Remote_SingleRank)
{
    // rankSize=1 退化场景：C = A·B (+bias)
    RunAndVerify(*this, /*rankSize=*/1, /*oriM=*/16, /*k=*/16, /*n=*/16, /*blockNum=*/1,
                 /*localMatmul=*/0, /*matmulMode=*/0, /*twoPhase=*/false);
}

TEST_F(AllToAllMmKernelTest, Test_FP16_Mode2_DeferredSync_SingleRank)
{
    RunAndVerify(*this, /*rankSize=*/1, /*oriM=*/16, /*k=*/16, /*n=*/16, /*blockNum=*/1,
                 /*localMatmul=*/1, /*matmulMode=*/2, /*twoPhase=*/false);
}
