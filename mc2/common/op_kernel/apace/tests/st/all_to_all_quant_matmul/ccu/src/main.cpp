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
 * \file main.cpp
 * \brief AllToAll MX Quant Matmul（CCU 版）ST 入口——本文件为单翻译单元编译源（-xasc）
 *
 * 用例参数：
 *   ./apace_a2a_qmm_ccu_st                          # precision 模式，默认参数
 *   ./apace_a2a_qmm_ccu_st precision [M] [tileM] [K] [N]
 *   ./apace_a2a_qmm_ccu_st perf [M] [tileM] [K] [N] [loop]
 *   默认: M=512, tileM=128, K=256, N=256, loop=10
 *
 * 数据语义（K-并行 GEMM）：x1=[M,K] FP8 E4M3FN（token 轴均分 R 块做 A2A），
 * workspace=[recvData|recvScale]，x2=[R*K,N] 完整权重不通信，
 * Y=[M/R,N] BF16 = Σ_r recvData[r块,:] @ x2[r*K段,:]（L0C 跨源累加，self 项直读 x1）。
 * 行级溯源数据（E4M3FN 2 的幂闭式，全部精确可表示）：位级断言零舍入噪声，
 * 可分类源 rank 混淆 / 目标路由混淆 / 行轮错位 / 字节损坏。
 * 退出码：precision 所有 rank 校验 PASS → 0，否则非 0。
 */

#include <functional>
#include <cstdlib>
#include "platform/platform_ascendc.h"

// 单翻译单元：CCU kernel + AIC kernel + host 编排（见 .asc 文件头说明）
#include "apace/kernel/fusions/all_to_all_quant_matmul/all_to_all_ccu_kernel_impl.asc"

// 运行模式
enum class RunMode {
    PRECISION,
    PERF
};

// 单 rank 线程的输入/输出参数（main 构造，Sample 消费）
struct ThreadContext {
    HcclRootInfo *rootInfo = nullptr;
    uint32_t rankId = 0;
    uint32_t rankSize = 0;
    RunMode mode{RunMode::PRECISION};
    uint64_t M{512};
    uint64_t tileM{128};
    uint64_t K{256};
    uint64_t N{256};
    uint32_t loopCount{10};
    int ret{0};
};

// FP8 E4M3FN / E8M0 常量编码（2 的幂，精确可表示；E8M0 127 = 1.0，dequant 恒等）
static constexpr uint8_t FP8_E4M3FN_POW2[4] = {0x38, 0x40, 0x48, 0x50};
static constexpr uint8_t FP8_E8M0_SCALE_1_0 = 127;

static inline uint8_t ENC4(uint64_t v)
{
    return FP8_E4M3FN_POW2[v % 4];
}
static inline uint8_t ENC2(uint64_t v)
{
    return (v % 2 == 0) ? FP8_E4M3FN_POW2[0] : FP8_E4M3FN_POW2[1];
}

// 期望 Y 值的 BF16 位型：Y[m] = K × Σ_{r=0}^{R-1} 2^((2i+r+m)%4 + r%2)
// （各加数均为 2 的幂，和 ≤ 256，×K ≤ 2^16——float 与 BF16 均精确表示）
static uint16_t ExpectYBf16Bits(uint32_t myRank, uint64_t m, uint64_t rankSize, uint64_t K)
{
    float sum = 0.0f;
    for (uint64_t r = 0; r < rankSize; r++) {
        const uint64_t e = (2 * myRank + r + m) % 4 + (r % 2);
        sum += static_cast<float>(1ULL << e);
    }
    const float expect = static_cast<float>(K) * sum;
    const uint32_t bits = *reinterpret_cast<const uint32_t *>(&expect);
    return static_cast<uint16_t>(bits >> 16);
}

static void PrintUsage(const char *prog)
{
    std::cout << "Usage:\n"
              << "  " << prog << "                          # precision, defaults\n"
              << "  " << prog << " precision [M] [tileM] [K] [N]\n"
              << "  " << prog << " perf [M] [tileM] [K] [N] [loop]\n"
              << "Defaults: M=512, tileM=128, K=256, N=256, loop=10"
              << std::endl;
}

// 测试向量生成器（A + scaleA）：rank s 的 x1 第 j 块第 m 行 = ENC4((2j+s+m)%4)
// （j=目标路由，s=源 rank，m=块内行号；scaleA 全 1.0）
static int GenerateInputData(uint32_t myRank, uint64_t rankSize, uint64_t blockRows,
                             uint64_t K, uint64_t scaleKLen,
                             void *x1, uint64_t sendSize,
                             void *x1Scale, uint64_t sendScaleSize)
{
    void *hostBuf = nullptr;
    ACLCHECK(aclrtMallocHost(&hostBuf, sendSize));
    uint8_t *tmpHost = static_cast<uint8_t *>(hostBuf);
    for (uint64_t j = 0; j < rankSize; j++) {
        for (uint64_t m = 0; m < blockRows; m++) {
            const uint8_t fillBits = ENC4(2 * j + myRank + m);
            uint8_t *row = tmpHost + (j * blockRows + m) * K;
            for (uint64_t k = 0; k < K; k++) {
                row[k] = fillBits;
            }
        }
    }
    aclError aclRet = aclrtMemcpy(x1, sendSize, hostBuf, sendSize, ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        (void)aclrtFreeHost(hostBuf);
        return static_cast<int>(aclRet);
    }
    ACLCHECK(aclrtFreeHost(hostBuf));

    void *hostScaleBuf = nullptr;
    ACLCHECK(aclrtMallocHost(&hostScaleBuf, sendScaleSize));
    memset(static_cast<uint8_t *>(hostScaleBuf), FP8_E8M0_SCALE_1_0, sendScaleSize);
    aclRet = aclrtMemcpy(x1Scale, sendScaleSize, hostScaleBuf, sendScaleSize, ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        (void)aclrtFreeHost(hostScaleBuf);
        return static_cast<int>(aclRet);
    }
    ACLCHECK(aclrtFreeHost(hostScaleBuf));
    return 0;
}

// 权重向量生成器（B + scaleB，不通信）：B=[R*K,N] DN 列主序（元素 (k,n) 位于 k+n*R*K），
// 第 r 段全值 ENC2(r)；scaleB 全 127（值全同，任意布局读同值）
static int GenerateWeightData(uint64_t rankSize, uint64_t K, uint64_t N, uint64_t scaleKLen,
                              void *x2, uint64_t weightSize,
                              void *x2Scale, uint64_t weightScaleSize)
{
    void *hostWeight = nullptr;
    ACLCHECK(aclrtMallocHost(&hostWeight, weightSize));
    uint8_t *tmpWeight = static_cast<uint8_t *>(hostWeight);
    for (uint64_t n = 0; n < N; n++) {
        for (uint64_t r = 0; r < rankSize; r++) {
            memset(tmpWeight + n * rankSize * K + r * K, ENC2(r), K);
        }
    }
    aclError aclRet = aclrtMemcpy(x2, weightSize, hostWeight, weightSize, ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        (void)aclrtFreeHost(hostWeight);
        return static_cast<int>(aclRet);
    }
    ACLCHECK(aclrtFreeHost(hostWeight));

    void *hostScale = nullptr;
    ACLCHECK(aclrtMallocHost(&hostScale, weightScaleSize));
    memset(static_cast<uint8_t *>(hostScale), FP8_E8M0_SCALE_1_0, weightScaleSize);
    aclRet = aclrtMemcpy(x2Scale, weightScaleSize, hostScale, weightScaleSize, ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        (void)aclrtFreeHost(hostScale);
        return static_cast<int>(aclRet);
    }
    ACLCHECK(aclrtFreeHost(hostScale));
    return 0;
}

// 调用一次完整通算融合（Custom → 双流 launch → sync）。
// CCU 资源经 a2aCtx 跨迭代复用：首帧在 Prepare 内分配，循环外统一 Release
static HcclResult InvokeAllToAllMatmul(HcclComm comm, aclrtStream streamAicore, aclrtStream streamCcu,
                                       AllToAllMatmulContext &a2aCtx,
                                       void *x1, void *x1Scale,
                                       void *x2, void *x2Scale, void *output, void *workspace,
                                       uint64_t sendCount, uint64_t scaleCount, uint64_t workspaceSize,
                                       ccuAllToAllMatmulTilingData &tilingData)
{
    HCCLCHECK(AllToAllMatmulCustom(x1, x1Scale, x2, x2Scale, output, workspace,
                                   sendCount, scaleCount, workspaceSize,
                                   HCCL_DATA_TYPE_INT8,
                                   comm, streamAicore, streamCcu, tilingData, a2aCtx));

    aclError aclRet = aclrtSynchronizeStream(streamAicore);
    if (aclRet != ACL_SUCCESS) { return HCCL_E_INTERNAL; }
    aclRet = aclrtSynchronizeStream(streamCcu);
    if (aclRet != ACL_SUCCESS) { return HCCL_E_INTERNAL; }

    return HCCL_SUCCESS;
}

// precision 模式校验：读回 output（BF16）→ 逐行逐元素位型断言 → 首错定位 → 落地 bin。
// 返回 0 = PASS；非 0 = FAIL（供 Sample 透传为进程退出码）
static int VerifyAndDumpOutput(uint32_t myRank, uint64_t rankSize, uint64_t blockRows,
                               uint64_t N, uint64_t K,
                               void *output, uint64_t outputSize)
{
    void *resultBuff = nullptr;
    ACLCHECK(aclrtMallocHost(&resultBuff, outputSize));
    aclError aclRet = aclrtMemcpy(resultBuff, outputSize, output, outputSize, ACL_MEMCPY_DEVICE_TO_HOST);
    if (aclRet != ACL_SUCCESS) {
        (void)aclrtFreeHost(resultBuff);
        return static_cast<int>(aclRet);
    }
    const uint16_t *outHost = static_cast<const uint16_t *>(resultBuff);

    const uint64_t totalElems = blockRows * N;
    uint64_t errCount = 0;
    bool hasFirstErr = false;
    uint64_t firstErrRow = 0;
    uint64_t firstErrCol = 0;
    uint16_t firstErrExpect = 0;
    uint16_t firstErrActual = 0;
    for (uint64_t m = 0; m < blockRows; m++) {
        const uint16_t expectBits = ExpectYBf16Bits(myRank, m, rankSize, K);
        const uint16_t *row = outHost + m * N;
        for (uint64_t n = 0; n < N; n++) {
            if (row[n] != expectBits) {
                errCount++;
                if (!hasFirstErr) {
                    hasFirstErr = true;
                    firstErrRow = m;
                    firstErrCol = n;
                    firstErrExpect = expectBits;
                    firstErrActual = row[n];
                }
            }
        }
    }

    if (errCount == 0) {
        std::cout << "rankId: " << myRank << ", A2A+MatMul PASS: all " << totalElems
                  << " output elems verified (K=" << K << ")" << std::endl;
    } else {
        std::cout << "rankId: " << myRank << ", A2A+MatMul FAIL: " << errCount
                  << " / " << totalElems << " elems mismatch" << std::endl;
        std::cout << "  first error at Y row " << firstErrRow << ", col " << firstErrCol
                  << ": expect 0x" << std::hex << firstErrExpect
                  << ", actual 0x" << firstErrActual << std::dec << std::endl;
    }

    std::string outputDir = "output/" + std::to_string(myRank);
    if (system(("mkdir -p " + outputDir).c_str()) != 0) {
        std::cout << "rankId: " << myRank << ", warn: mkdir " << outputDir << " failed" << std::endl;
    }
    std::string outFile = outputDir + "/npu_out.bin";
    std::ofstream ofs(outFile, std::ios::binary);
    ofs.write(static_cast<const char *>(resultBuff), outputSize);
    ofs.close();
    std::cout << "rankId: " << myRank << ", output written to " << outFile
              << " (" << outputSize << " bytes)" << std::endl;

    ACLCHECK(aclrtFreeHost(resultBuff));
    return (errCount == 0) ? 0 : 1;
}

// 单 rank 的执行入口（每个 rank 一个线程调用本函数）：
// 设备/通信组/双流初始化 → 7 buffer 分配与填充 → 构造 tiling →
// 循环执行 InvokeAllToAllMatmul（precision 1 次；perf loop 次）→ precision 校验 → 释放
int Sample(ThreadContext &ctx)
{
    if (ctx.rankSize > CCU_MAX_RANK_SIZE) {
        std::cout << "rankSize " << ctx.rankSize << " exceeds CCU_MAX_RANK_SIZE "
                  << CCU_MAX_RANK_SIZE << std::endl;
        ctx.ret = HCCL_E_PARA;
        return ctx.ret;
    }
    const uint32_t myRank = ctx.rankId;
    const uint32_t rankSize = ctx.rankSize;
    const uint64_t M = ctx.M;
    const uint64_t tileM = ctx.tileM;
    const uint64_t K = ctx.K;
    const uint64_t N = ctx.N;
    constexpr uint64_t MXFP_DIVISOR = 64;
    constexpr uint64_t MXFP_MULTI = 2;
    const uint64_t scaleKLen = K / MXFP_DIVISOR * MXFP_MULTI;

    const uint64_t blockRows = M / rankSize;
    const uint64_t dataCommTurn = blockRows / tileM;
    const uint64_t sendCount = M * K;
    const uint64_t scaleCount = M * scaleKLen;
    const uint64_t weightCount = static_cast<uint64_t>(rankSize) * K * N;
    const uint64_t weightScaleCount = static_cast<uint64_t>(rankSize) * scaleKLen * N;
    const uint64_t outputCount = blockRows * N;
    const uint64_t sendSize = sendCount * 1;
    const uint64_t sendScaleSize = scaleCount * 1;
    const uint64_t weightSize = weightCount * 1;
    const uint64_t weightScaleSize = weightScaleCount * 1;
    const uint64_t outputSize = outputCount * 2;

    ACLCHECK(aclrtSetDevice(static_cast<int32_t>(ctx.rankId)));
    HcclComm hcclComm;
    HCCLCHECK(HcclCommInitRootInfo(ctx.rankSize, ctx.rootInfo, ctx.rankId, &hcclComm));
    aclrtStream streamAicore;
    ACLCHECK(aclrtCreateStream(&streamAicore));
    aclrtStream streamCcu;
    ACLCHECK(aclrtCreateStream(&streamCcu));

    ctx.ret = [&]() -> int {
        void *x1 = nullptr;
        void *x1Scale = nullptr;
        void *x2 = nullptr;
        void *x2Scale = nullptr;
        void *output = nullptr;
        void *workspace = nullptr;
        const uint64_t workspaceSize = sendSize + sendScaleSize;
        ACLCHECK(aclrtMalloc(&x1, sendSize, ACL_MEM_MALLOC_HUGE_ONLY));
        ACLCHECK(aclrtMalloc(&x1Scale, sendScaleSize, ACL_MEM_MALLOC_HUGE_ONLY));
        ACLCHECK(aclrtMalloc(&x2, weightSize, ACL_MEM_MALLOC_HUGE_ONLY));
        ACLCHECK(aclrtMalloc(&x2Scale, weightScaleSize, ACL_MEM_MALLOC_HUGE_ONLY));
        ACLCHECK(aclrtMalloc(&output, outputSize, ACL_MEM_MALLOC_HUGE_ONLY));
        ACLCHECK(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_ONLY));
        auto freeAll = [&]() {
            (void)aclrtFree(workspace);
            (void)aclrtFree(output);
            (void)aclrtFree(x2Scale);
            (void)aclrtFree(x2);
            (void)aclrtFree(x1Scale);
            (void)aclrtFree(x1);
        };

        const int genRet = GenerateInputData(myRank, rankSize, blockRows, K, scaleKLen,
                                             x1, sendSize, x1Scale, sendScaleSize);
        if (genRet != 0) {
            freeAll();
            return genRet;
        }
        const int weightRet = GenerateWeightData(rankSize, K, N, scaleKLen,
                                                 x2, weightSize, x2Scale, weightScaleSize);
        if (weightRet != 0) {
            freeAll();
            return weightRet;
        }

        // 构造统一 tiling：matmul 手填（m=blockRows, n=N, k=K；baseM/baseN/baseK 为
        // 目标 case 调优值），comm 仅 tileM（其余由 Validate/Prepare 填）
        ccuAllToAllMatmulTilingData tilingData{};
        tilingData.comm.tileM = static_cast<uint32_t>(tileM);
        tilingData.matmul.m = static_cast<uint32_t>(blockRows);
        tilingData.matmul.n = static_cast<uint32_t>(N);
        tilingData.matmul.k = static_cast<uint32_t>(K);
        tilingData.matmul.baseM = 256;
        tilingData.matmul.baseN = 256;
        tilingData.matmul.baseK = 128;
        tilingData.matmul.stepK = 1;
        tilingData.matmul.scaleKL1 = 8192;
        tilingData.matmul.nBufferNum = 2;
        tilingData.matmul.dbL0c = 1;
        tilingData.matmul.mTailTile = 1;
        tilingData.matmul.nTailTile = 1;
        tilingData.matmul.mBaseTailSplitCnt = 1;
        tilingData.matmul.nBaseTailSplitCnt = 1;
        tilingData.matmul.mTailMain = 0;
        tilingData.matmul.nTailMain = 0;
        {
            auto *platform = platform_ascendc::PlatformAscendCManager::GetInstance();
            if (platform != nullptr) {
                uint32_t aicNum = platform->GetCoreNumAic();
                tilingData.matmul.usedCoreNum = (aicNum > 0) ? aicNum : 8;
            } else {
                tilingData.matmul.usedCoreNum = 8;
            }
        }

        std::cout << "rankId: " << myRank << ", mode: "
                  << (ctx.mode == RunMode::PERF ? "perf" : "precision")
                  << ", A2A+MXFP8 MatMul: rankSize=" << rankSize
                  << ", M=" << M << ", blockRows=" << blockRows
                  << ", K=" << K << ", N=" << N
                  << ", tileM=" << tileM << ", dataCommTurn=" << dataCommTurn
                  << " (+" << 1 << " scale round)"
                  << ", usedCoreNum=" << tilingData.matmul.usedCoreNum
                  << ", x1=" << (sendSize / 1024) << "KB"
                  << ", workspace=" << ((sendSize + sendScaleSize) / 1024) << "KB";
        if (ctx.mode == RunMode::PERF) {
            std::cout << ", loop=" << ctx.loopCount;
        }
        std::cout << std::endl;

        AllToAllMatmulContext a2aCtx;
        const uint32_t totalRuns = (ctx.mode == RunMode::PERF) ? ctx.loopCount : 1;
        int runRet = 0;
        for (uint32_t iter = 0; iter < totalRuns; iter++) {
            runRet = static_cast<int>(InvokeAllToAllMatmul(hcclComm, streamAicore, streamCcu, a2aCtx,
                                                           x1, x1Scale, x2, x2Scale, output, workspace,
                                                           sendCount, scaleCount,
                                                           sendSize + sendScaleSize, tilingData));
            if (runRet != HCCL_SUCCESS) {
                break;
            }
            if (ctx.mode == RunMode::PERF) {
                std::cout << "rankId: " << myRank << ", perf iter " << (iter + 1) << "/" << totalRuns
                          << " done" << std::endl;
            }
        }
        // 兜底同步后统一释放 CCU 资源（实例销毁前双流必须已静默）
        (void)aclrtSynchronizeStream(streamAicore);
        (void)aclrtSynchronizeStream(streamCcu);
        const int releaseRet = static_cast<int>(ReleaseAllToAllMatmulContext(a2aCtx));
        if (runRet == 0) {
            runRet = releaseRet;
        }
        if (runRet != 0) {
            freeAll();
            return runRet;
        }

        int verifyRet = 0;
        if (ctx.mode == RunMode::PRECISION) {
            std::this_thread::sleep_for(std::chrono::seconds(ctx.rankId));
            verifyRet = VerifyAndDumpOutput(myRank, rankSize, blockRows, N, K, output, outputSize);
        }
        freeAll();
        return verifyRet;
    }();

    HCCLCHECK(HcclCommDestroy(hcclComm));
    ACLCHECK(aclrtDestroyStream(streamCcu));
    ACLCHECK(aclrtDestroyStream(streamAicore));
    ACLCHECK(aclrtResetDevice(ctx.rankId));
    return ctx.ret;
}

// 程序入口：解析参数 → aclInit + 查询 device 数量（rankSize）→ rootRank 获取
// HcclRootInfo → 每 device 一线程执行 Sample → join 聚合（任一失败 → 退出码非 0）
int main(int argc, char *argv[])
{
    ThreadContext defaults;
    RunMode mode = RunMode::PRECISION;
    uint64_t M = defaults.M;
    uint64_t tileM = defaults.tileM;
    uint64_t K = defaults.K;
    uint64_t N = defaults.N;
    uint32_t loopCount = defaults.loopCount;

    if (argc >= 2) {
        const std::string modeStr = argv[1];
        if (modeStr == "perf") {
            mode = RunMode::PERF;
        } else if (modeStr == "precision") {
            mode = RunMode::PRECISION;
        } else {
            std::cout << "ERROR: unknown mode '" << modeStr << "'" << std::endl;
            PrintUsage(argv[0]);
            return 1;
        }
        try {
            if (argc >= 3) M = std::stoull(argv[2]);
            if (argc >= 4) tileM = std::stoull(argv[3]);
            if (argc >= 5) K = std::stoull(argv[4]);
            if (argc >= 6) N = std::stoull(argv[5]);
            if (argc >= 7 && mode == RunMode::PERF) loopCount = static_cast<uint32_t>(std::stoul(argv[6]));
        } catch (const std::exception &e) {
            std::cout << "ERROR: invalid numeric argument (" << e.what() << ")" << std::endl;
            PrintUsage(argv[0]);
            return 1;
        }
    }

    if (K == 0 || K % 64 != 0) {
        std::cout << "ERROR: K=" << K << " must be positive and divisible by 64 (MXFP8)" << std::endl;
        return 1;
    }
    if (tileM == 0) {
        std::cout << "ERROR: tileM=" << tileM << " must be positive" << std::endl;
        return 1;
    }

    std::cout << "Run mode: " << (mode == RunMode::PERF ? "perf" : "precision")
              << ", M=" << M << ", tileM=" << tileM << ", K=" << K << ", N=" << N;
    if (mode == RunMode::PERF) {
        std::cout << ", loop=" << loopCount;
    }
    std::cout << std::endl;

    ACLCHECK(aclInit(NULL));
    uint32_t rankSize;
    ACLCHECK(aclrtGetDeviceCount(&rankSize));
    std::cout << "Found " << rankSize << " NPU device(s) available" << std::endl;

    if (rankSize == 0 || M % rankSize != 0) {
        std::cout << "ERROR: M=" << M << " must be divisible by rankSize=" << rankSize << std::endl;
        return 1;
    }
    const uint64_t blockRows = M / rankSize;
    if (blockRows % tileM != 0) {
        std::cout << "ERROR: blockRows=" << blockRows << " (=M/rankSize) must be divisible by tileM="
                  << tileM << std::endl;
        return 1;
    }
    if (blockRows / tileM > MAX_COMMTURN) {
        std::cout << "ERROR: dataCommTurn=" << (blockRows / tileM) << " exceeds MAX_COMMTURN="
                  << MAX_COMMTURN << std::endl;
        return 1;
    }

    const char *schedMode = getenv("HCCL_OP_EXPANSION_MODE");
    if (schedMode == nullptr || std::string(schedMode) != "CCU_SCHED") {
        std::cout << "WARNING: HCCL_OP_EXPANSION_MODE=" << (schedMode ? schedMode : "(unset)")
                  << " (expected CCU_SCHED), CCU operations may fail!" << std::endl;
    } else {
        std::cout << "HCCL_OP_EXPANSION_MODE=" << schedMode << " OK" << std::endl;
    }

    int32_t rootRank = 0;
    ACLCHECK(aclrtSetDevice(rootRank));
    void *rootInfoBuf = nullptr;
    ACLCHECK(aclrtMallocHost(&rootInfoBuf, sizeof(HcclRootInfo)));
    HcclRootInfo *rootInfo = (HcclRootInfo *)rootInfoBuf;
    HCCLCHECK(HcclGetRootInfo(rootInfo));

    std::vector<std::thread> threads(rankSize);
    std::vector<ThreadContext> args(rankSize);
    for (uint32_t i = 0; i < rankSize; i++) {
        args[i].rootInfo = rootInfo;
        args[i].rankId = i;
        args[i].rankSize = rankSize;
        args[i].mode = mode;
        args[i].M = M;
        args[i].tileM = tileM;
        args[i].K = K;
        args[i].N = N;
        args[i].loopCount = loopCount;
        threads[i] = std::thread(Sample, std::ref(args[i]));
    }
    for (uint32_t i = 0; i < rankSize; i++) {
        threads[i].join();
    }

    ACLCHECK(aclrtFreeHost(rootInfoBuf));
    ACLCHECK(aclFinalize());

    int finalRet = 0;
    for (uint32_t i = 0; i < rankSize; i++) {
        if (args[i].ret != 0) {
            std::cout << "rankId: " << i << " failed with ret=" << args[i].ret << std::endl;
            finalRet = (finalRet == 0) ? 1 : finalRet;
        }
    }
    if (finalRet == 0) {
        std::cout << "=== ALL RANKS PASS ===" << std::endl;
    } else {
        std::cout << "=== SOME RANKS FAILED ===" << std::endl;
    }
    return finalRet;
}
