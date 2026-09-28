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
 * 主干流程（调用算子的全部所需）：
 *   main（参数解析 + aclInit + 每 device 一线程）→ RunAllToAllMatmul（设备/通信组/双流/显存
 *   + tiling 构造）→ InvokeAllToAllMatmul（AllToAllMatmulCustom + 双流 sync）→
 *   ReleaseAllToAllMatmulContext（循环外一次）。
 *   测试向量生成 / 期望值 / 校验落盘等辅助函数在 test_data_utils.h。
 * 退出码：precision 所有 rank 校验 PASS → 0，否则非 0。
 */

#include <functional>
#include <cstdlib>
#include "platform/platform_ascendc.h"

// 单翻译单元：CCU kernel + AIC kernel + host 编排（见 .asc 文件头说明）
#include "apace/kernel/fusions/all_to_all_quant_matmul/all_to_all_ccu_kernel_impl.asc"

// 测试辅助（数据生成 / 校验 / usage）——须在上方锚点链之后引入
#include "test_data_utils.h"

// 运行模式
enum class RunMode {
    PRECISION,
    PERF
};

// 单 rank 线程的输入/输出参数（main 构造，RunAllToAllMatmul 消费）
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

// 命令行入参解析 + 静态合法性检查：argc<2 时全部取 ctx 成员默认值；
// 模式词非法 / 数字解析失败 / K 非 64 倍数 / tileM 为 0 → 打印 usage 返回 false。
// rankSize 相关的整除检查依赖设备查询，留在 main 内
static bool ParseCommandLine(int argc, char *argv[], ThreadContext &ctx)
{
    if (argc >= 2) {
        const std::string modeStr = argv[1];
        if (modeStr == "perf") {
            ctx.mode = RunMode::PERF;
        } else if (modeStr == "precision") {
            ctx.mode = RunMode::PRECISION;
        } else {
            std::cout << "ERROR: unknown mode '" << modeStr << "'" << std::endl;
            PrintUsage(argv[0]);
            return false;
        }
        try {
            if (argc >= 3) ctx.M = std::stoull(argv[2]);
            if (argc >= 4) ctx.tileM = std::stoull(argv[3]);
            if (argc >= 5) ctx.K = std::stoull(argv[4]);
            if (argc >= 6) ctx.N = std::stoull(argv[5]);
            if (argc >= 7 && ctx.mode == RunMode::PERF) {
                ctx.loopCount = static_cast<uint32_t>(std::stoul(argv[6]));
            }
        } catch (const std::exception &e) {
            std::cout << "ERROR: invalid numeric argument (" << e.what() << ")" << std::endl;
            PrintUsage(argv[0]);
            return false;
        }
    }

    if (ctx.K == 0 || ctx.K % 64 != 0) {
        std::cout << "ERROR: K=" << ctx.K << " must be positive and divisible by 64 (MXFP8)" << std::endl;
        return false;
    }
    if (ctx.tileM == 0) {
        std::cout << "ERROR: tileM=" << ctx.tileM << " must be positive" << std::endl;
        return false;
    }
    return true;
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

// 单 rank 的执行入口（每个 rank 一个线程调用本函数）：
// 设备/通信组/双流初始化 → 7 buffer 分配与填充 → 构造 tiling →
// 循环执行 InvokeAllToAllMatmul（precision 1 次；perf loop 次）→ precision 校验 → 释放
int RunAllToAllMatmul(ThreadContext &ctx)
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

    // 立即调用 lambda：执行体内的 return（如 ACLCHECK）只带错误码退出本执行体并
    // 经返回值写入 ctx.ret，下方通信组/流/设备释放仍必然执行（替代 goto cleanup）
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
                  << " (scale 随轮 0 搭载)"
                  << ", usedCoreNum=" << tilingData.matmul.usedCoreNum
                  << ", x1=" << (sendSize / 1024) << "KB"
                  << ", workspace=" << ((sendSize + sendScaleSize) / 1024) << "KB";
        if (ctx.mode == RunMode::PERF) {
            std::cout << ", loop=" << ctx.loopCount;
        }
        std::cout << std::endl;

        AllToAllMatmulContext a2aCtx;
        const uint32_t totalRuns = (ctx.mode == RunMode::PERF) ? ctx.loopCount : 1;
        // event 计时旁证（对齐 quant_matmul_all_to_all 家族口径）：串行迭代下 event 间隔
        // = 端到端单次调用耗时（含 launch/同步间隙），与 profiler Task Duration 互补，
        // 两者之差即串行帧的 host 间隙税
        aclrtEvent iterStartEvent = nullptr;
        aclrtEvent iterEndEvent = nullptr;
        float eventElapsedTotalMs = 0.0F;
        // 重型算子 buffer（吸收 host launch gap：迭代间在 streamAicore 上插入大块
        // memset，设备忙 ~430us 期间各 rank host 线程完成下一帧提交，消除 host
        // bound 导致的 rank 间起跑偏斜——参考 mc2_test 框架的 Exp 重型算子方法；
        // msprof 解析按 kernel name 过滤，memset 自动排除在统计外）
        void *heavyBuf = nullptr;
        constexpr uint64_t HEAVY_BUF_SIZE = 128UL * 1024 * 1024; // 128MB
        constexpr uint32_t HEAVY_MEMSET_COUNT = 4;               // ~4×107us ≈ 430us
        if (ctx.mode == RunMode::PERF) {
            ACLCHECK(aclrtCreateEvent(&iterStartEvent));
            ACLCHECK(aclrtCreateEvent(&iterEndEvent));
            ACLCHECK(aclrtMalloc(&heavyBuf, HEAVY_BUF_SIZE, ACL_MEM_MALLOC_HUGE_ONLY));
        }
        int runRet = 0;
        for (uint32_t iter = 0; iter < totalRuns; iter++) {
            if (ctx.mode == RunMode::PERF) {
                ACLCHECK(aclrtRecordEvent(iterStartEvent, streamAicore));
            }
            runRet = static_cast<int>(InvokeAllToAllMatmul(hcclComm, streamAicore, streamCcu, a2aCtx,
                                                           x1, x1Scale, x2, x2Scale, output, workspace,
                                                           sendCount, scaleCount,
                                                           sendSize + sendScaleSize, tilingData));
            if (runRet != HCCL_SUCCESS) {
                break;
            }
            if (ctx.mode == RunMode::PERF) {
                // Invoke 内部已 sync streamAicore，end event 立即完成；elapsed 为端到端单次耗时
                float iterMs = 0.0F;
                ACLCHECK(aclrtRecordEvent(iterEndEvent, streamAicore));
                ACLCHECK(aclrtSynchronizeEvent(iterEndEvent));
                ACLCHECK(aclrtEventElapsedTime(&iterMs, iterStartEvent, iterEndEvent));
                if (iter > 0) {
                    // 首帧含 Prepare（通道创建 ~1.2s），不计入稳态均值
                    eventElapsedTotalMs += iterMs;
                }
                std::cout << "rankId: " << myRank << ", perf iter " << (iter + 1) << "/" << totalRuns
                          << " event=" << (iterMs * 1000.0F) << "us" << std::endl;
                // 插入重型算子（非阻塞：设备忙期间 host 完成下一帧提交；尾帧不插）
                if (iter < totalRuns - 1) {
                    for (uint32_t j = 0; j < HEAVY_MEMSET_COUNT; j++) {
                        ACLCHECK(aclrtMemsetAsync(heavyBuf, HEAVY_BUF_SIZE, 0,
                                                  HEAVY_BUF_SIZE, streamAicore));
                    }
                }
            }
        }
        if (ctx.mode == RunMode::PERF) {
            if (heavyBuf != nullptr) {
                (void)aclrtFree(heavyBuf);
            }
            const uint32_t steadyRuns = (totalRuns > 1) ? (totalRuns - 1) : 1;
            std::cout << "rankId: " << myRank << ", event timing steady avg="
                      << (eventElapsedTotalMs * 1000.0F / steadyRuns) << "us/iter over " << steadyRuns
                      << " iters (end-to-end incl launch+sync gap, first frame excluded)" << std::endl;
            ACLCHECK(aclrtDestroyEvent(iterStartEvent));
            ACLCHECK(aclrtDestroyEvent(iterEndEvent));
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
// HcclRootInfo → 每 device 一线程执行 RunAllToAllMatmul → join 聚合（任一失败 → 退出码非 0）
int main(int argc, char *argv[])
{
    ThreadContext cliParams; // 成员默认值 = 默认 case，CLI 覆盖
    if (!ParseCommandLine(argc, argv, cliParams)) {
        return 1;
    }

    std::cout << "Run mode: " << (cliParams.mode == RunMode::PERF ? "perf" : "precision")
              << ", M=" << cliParams.M << ", tileM=" << cliParams.tileM
              << ", K=" << cliParams.K << ", N=" << cliParams.N;
    if (cliParams.mode == RunMode::PERF) {
        std::cout << ", loop=" << cliParams.loopCount;
    }
    std::cout << std::endl;

    ACLCHECK(aclInit(NULL));
    uint32_t rankSize;
    ACLCHECK(aclrtGetDeviceCount(&rankSize));
    std::cout << "Found " << rankSize << " NPU device(s) available" << std::endl;

    if (rankSize == 0 || cliParams.M % rankSize != 0) {
        std::cout << "ERROR: M=" << cliParams.M << " must be divisible by rankSize=" << rankSize << std::endl;
        return 1;
    }
    const uint64_t blockRows = cliParams.M / rankSize;
    if (blockRows % cliParams.tileM != 0) {
        std::cout << "ERROR: blockRows=" << blockRows << " (=M/rankSize) must be divisible by tileM="
                  << cliParams.tileM << std::endl;
        return 1;
    }
    if (blockRows / cliParams.tileM > MAX_COMMTURN) {
        std::cout << "ERROR: dataCommTurn=" << (blockRows / cliParams.tileM) << " exceeds MAX_COMMTURN="
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
    std::vector<ThreadContext> args(rankSize, cliParams);
    for (uint32_t i = 0; i < rankSize; i++) {
        args[i].rootInfo = rootInfo;
        args[i].rankId = i;
        args[i].rankSize = rankSize;
        threads[i] = std::thread(RunAllToAllMatmul, std::ref(args[i]));
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
