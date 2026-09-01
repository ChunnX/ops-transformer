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
 * \brief quantMatmulReduceScatter ST 主程序
 *
 * 流程: fork 多 rank → 每进程绑定 NPU → HCCL 建链 → Tiling → Kernel 执行 → 精度比对
 *
 * 与 all_to_all_quant_matmul 的关键区别：
 *   - K 不切分（每卡持有完整 K），m 按 rankSize 切分
 *   - 通信数据为 C (bfloat16)，非 A (fp8)
 *   - 输出 per rank: [m/rankSize, n]
 */

#include <cmath>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <limits.h>
#include <unistd.h>
#include <memory>
#include <random>
#include <string>
#include <vector>
#include <sys/wait.h>

#include "kernel_basic_intf.h"
#include "acl/acl.h"
#include "hccl/hccl.h"
#include "hccl/hccl_comm.h"
#include "hccl/hccl_res.h"
#include "hccl/hccl_types.h"
#include "hccl/hccl_rank_graph.h"
#include <cstdlib>
#include "apace_st_utils.h"
#include "apace/utils/comm_channel_builder.h"
#include "../../utils/root_info_exchanger.h"
#include "apace/tiling/quant_matmul_tiling_swat.h"
#include "apace_qmrs_kernel_launcher.h"
#include "apace/core/aiv_comm/collective_comm_context.h"

#define HCCL_CHECK(status) \
    do { \
        HcclResult ret = (status); \
        if (ret != HCCL_SUCCESS) { \
            std::cerr << __FILE__ << ":" << __LINE__ << " HcclResult:" << ret << std::endl; \
        } \
    } while (0)

inline uint64_t CeilDiv(uint32_t a, uint32_t b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

void printUsage(const std::string &programName)
{
    std::cerr << "Usage: " << programName << " m k n rankNum [mode] [headMSize]" << std::endl;
    std::cerr << "Args: " << std::endl;
    std::cerr << "  m: row of matrix A (must be divisible by rankNum)" << std::endl;
    std::cerr << "  k: col of matrix A (full K, NOT split across ranks)" << std::endl;
    std::cerr << "  n: col of matrix B" << std::endl;
    std::cerr << "  rankNum: number of ranks" << std::endl;
    std::cerr << "  mode: optional, 'precision' (default) | 'perf'" << std::endl;
    std::cerr << "  headMSize: optional, long block M size (default 512)" << std::endl;
    std::cerr << "Example: " << programName << " 2048 3584 4096 4" << std::endl;
}

void parseArguments(int argc, char *argv[], int &m, int &k, int &n, int &rankNum, std::string &mode, int &headMSize)
{
    if (argc >= 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
        printUsage(argv[0]);
        exit(1);
    }
    if (argc < 5) {
        throw std::invalid_argument("ERROR: Lacks Arguments");
    }
    try {
        m = std::stoi(argv[1]);
        k = std::stoi(argv[2]);
        n = std::stoi(argv[3]);
        rankNum = std::stoi(argv[4]);
    } catch (const std::invalid_argument &) {
        throw std::invalid_argument("ERROR: m k n rankNum must be Integer");
    }

    mode = (argc >= 6) ? std::string(argv[5]) : std::string("precision");
    if (mode != "precision" && mode != "perf") {
        throw std::invalid_argument("ERROR: mode must be 'precision' or 'perf'");
    }

    headMSize = (argc >= 7) ? std::stoi(argv[6]) : 512;
    if (headMSize <= 0) {
        throw std::invalid_argument("ERROR: headMSize must be positive");
    }

    if (m <= 0 || k <= 0 || n <= 0 || rankNum <= 0) {
        throw std::invalid_argument("ERROR: m k n rankNum must be positive");
    }

    // m 必须被 rankNum 整除（M 轴按 rankSize 切分用于 AlltoAll）
    if (m % rankNum != 0) {
        throw std::invalid_argument("ERROR: m must be divisible by rankNum");
    }

    // Scale 对齐约束：CeilDiv(k, 64) 必须为偶数
    if (CeilDiv(k, 64) % 2 != 0) {
        throw std::invalid_argument("ERROR: CeilDiv(k, 64) must be an even number");
    }
}

int runQuantMatmulReduceScatter(int rankNum, int rankId, int m, int k, int n, const std::string &mode, int headMSizeArg)
{
    const char *ipport = "tcp://127.0.0.1:8998";
    INFO_LOG("rankNum=%d, rankId=%d, ipport=%s, mode=%s, headMSize=%d", rankNum, rankId, ipport, mode.c_str(),
             headMSizeArg);

    // 与 all_to_all_quant_matmul 不同：K 不切分，每卡持有完整 K
    // m 按 rankSize 切分用于 AlltoAll 通信
    uint32_t chunkM = static_cast<uint32_t>(m) / static_cast<uint32_t>(rankNum);

    QuantMatmulReduceScatterTilingData tilingData;

    // MatMul tiling: 使用完整 m, n, k（无 TP 切分）
    QuantMatmulTilingSwat<mm::DataType::DT_FLOAT8_E4M3FN, mm::DataType::DT_FLOAT8_E4M3FN, mm::BiasDataType::DT_FLOAT> tilingEngine;
    tilingEngine.GetTilingData(m, n, k, false, true, tilingData.mmTilingData);

    // 通信 tiling: PUT 模式多 tile (headMSize 由参数传入，默认 512)
    // 切分轴 = M, 非切分轴 = N
    // chunkM = m / rankSize, 按 headMSize 切分为 headTileCnt 个头块 + 可选 1 个尾块
    // PUT 模式: AIC 写 workspace, AIV 从 workspace 推送到其他 rank win 区
    // win 区布局 = C_i 连续 [m,n] = rank-major chunks (与 workspace 一致)
    uint32_t headMSize = static_cast<uint32_t>(headMSizeArg);
    if (headMSize > chunkM || headMSize == 0) {
        headMSize = chunkM;
    }
    headMSize = (headMSize / 16) * 16;
    if (headMSize == 0) {
        headMSize = 16;
    }

    uint32_t headTileCnt = chunkM / headMSize;
    uint32_t tailMSize = chunkM % headMSize;
    // tailMSize 必须 16 对齐 (Blaze matmul 最小行粒度)
    if (tailMSize > 0 && tailMSize % 16 != 0) {
        throw std::invalid_argument("ERROR: tailMSize=" + std::to_string(tailMSize) +
                                    " not 16-aligned. "
                                    "chunkM=" +
                                    std::to_string(chunkM) + " headMSize=" + std::to_string(headMSize) +
                                    ". Use a headMSize that divides chunkM into 16-aligned segments.");
    }
    uint32_t tailTileCnt = (tailMSize > 0) ? 1 : 0;
    uint32_t tileCnt = headTileCnt + tailTileCnt;

    tilingData.commTilingData.splitAxisTileCnt = static_cast<uint64_t>(headTileCnt);
    tilingData.commTilingData.splitAxisTileSize = static_cast<uint64_t>(headMSize);
    tilingData.commTilingData.splitAxisTailSize = static_cast<uint64_t>(tailMSize);
    tilingData.commTilingData.splitAxisTailCnt = static_cast<uint64_t>(tailTileCnt);
    tilingData.commTilingData.nonSplitAxisSize = static_cast<uint64_t>(n);

    // 子问题 tiling: m=headMSize, 用于 T>1 per-tile head tile
    if (tileCnt > 1) {
        QuantMatmulTilingSwat<mm::DataType::DT_FLOAT8_E4M3FN, mm::DataType::DT_FLOAT8_E4M3FN, mm::BiasDataType::DT_FLOAT> subTilingEngine;
        subTilingEngine.EnableBaseMHalving(true); // 块数<2×核数时 baseM 减半, 提升负载均衡(同 allto_all_matmul_v2)
        subTilingEngine.GetTilingData(headMSize, n, k, false, true, tilingData.subMmTilingData);
        // tail tile tiling: m=rankSize*tailMSize (tailMSize 已保证 16 对齐)
        if (tailTileCnt > 0 && tailMSize > 0) {
            QuantMatmulTilingSwat<mm::DataType::DT_FLOAT8_E4M3FN, mm::DataType::DT_FLOAT8_E4M3FN, mm::BiasDataType::DT_FLOAT> tailTilingEngine;
            uint32_t tailLogicalM = static_cast<uint32_t>(rankNum) * tailMSize;
            tailTilingEngine.GetTilingData(tailLogicalM, n, k, false, true, tilingData.tailMmTilingData);
        } else {
            tilingData.tailMmTilingData = {};
        }
    } else {
        tilingData.subMmTilingData = {};
        tilingData.tailMmTilingData = {};
    }

    if (rankId == 0) {
        INFO_LOG("chunkM=%u, headMSize=%u, headTileCnt=%u, tailMSize=%u, tailTileCnt=%u, tileCnt=%u (PUT multi-tile), "
                 "rankSize=%d",
                 chunkM, headMSize, headTileCnt, tailMSize, tailTileCnt, tileCnt, rankNum);

        INFO_LOG("[Tiling] mmTilingData: m=%u n=%u k=%u baseM=%u baseN=%u baseK=%u stepK=%u usedCoreNum=%u dbL0c=%u "
                 "nBufferNum=%u scaleKL1=%u mTailTile=%u nTailTile=%u mBaseTailSplitCnt=%u nBaseTailSplitCnt=%u "
                 "mTailMain=%u nTailMain=%u",
                 tilingData.mmTilingData.m, tilingData.mmTilingData.n, tilingData.mmTilingData.k,
                 tilingData.mmTilingData.baseM, tilingData.mmTilingData.baseN, tilingData.mmTilingData.baseK,
                 tilingData.mmTilingData.stepK, tilingData.mmTilingData.usedCoreNum, tilingData.mmTilingData.dbL0c,
                 tilingData.mmTilingData.nBufferNum, tilingData.mmTilingData.scaleKL1,
                 tilingData.mmTilingData.mTailTile, tilingData.mmTilingData.nTailTile,
                 tilingData.mmTilingData.mBaseTailSplitCnt, tilingData.mmTilingData.nBaseTailSplitCnt,
                 tilingData.mmTilingData.mTailMain, tilingData.mmTilingData.nTailMain);

        if (tileCnt > 1) {
            INFO_LOG("[Tiling] subMmTilingData: m=%u n=%u k=%u baseM=%u baseN=%u baseK=%u stepK=%u dbL0c=%u "
                     "nBufferNum=%u scaleKL1=%u mTailTile=%u nTailTile=%u mBaseTailSplitCnt=%u nBaseTailSplitCnt=%u "
                     "mTailMain=%u nTailMain=%u",
                     tilingData.subMmTilingData.m, tilingData.subMmTilingData.n, tilingData.subMmTilingData.k,
                     tilingData.subMmTilingData.baseM, tilingData.subMmTilingData.baseN,
                     tilingData.subMmTilingData.baseK, tilingData.subMmTilingData.stepK,
                     tilingData.subMmTilingData.dbL0c, tilingData.subMmTilingData.nBufferNum,
                     tilingData.subMmTilingData.scaleKL1, tilingData.subMmTilingData.mTailTile,
                     tilingData.subMmTilingData.nTailTile, tilingData.subMmTilingData.mBaseTailSplitCnt,
                     tilingData.subMmTilingData.nBaseTailSplitCnt, tilingData.subMmTilingData.mTailMain,
                     tilingData.subMmTilingData.nTailMain);

            if (tailTileCnt > 0 && tailMSize > 0) {
                INFO_LOG("[Tiling] tailMmTilingData: m=%u n=%u k=%u baseM=%u baseN=%u baseK=%u stepK=%u dbL0c=%u "
                         "nBufferNum=%u scaleKL1=%u mTailTile=%u nTailTile=%u mBaseTailSplitCnt=%u "
                         "nBaseTailSplitCnt=%u mTailMain=%u nTailMain=%u",
                         tilingData.tailMmTilingData.m, tilingData.tailMmTilingData.n, tilingData.tailMmTilingData.k,
                         tilingData.tailMmTilingData.baseM, tilingData.tailMmTilingData.baseN,
                         tilingData.tailMmTilingData.baseK, tilingData.tailMmTilingData.stepK,
                         tilingData.tailMmTilingData.dbL0c, tilingData.tailMmTilingData.nBufferNum,
                         tilingData.tailMmTilingData.scaleKL1, tilingData.tailMmTilingData.mTailTile,
                         tilingData.tailMmTilingData.nTailTile, tilingData.tailMmTilingData.mBaseTailSplitCnt,
                         tilingData.tailMmTilingData.nBaseTailSplitCnt, tilingData.tailMmTilingData.mTailMain,
                         tilingData.tailMmTilingData.nTailMain);
            }
        }
    }

    INFO_LOG("[Rank %d] Step 1: aclInit", rankId);
    ACL_CHECK(aclInit(nullptr));
    int32_t deviceId = rankId;
    ACL_CHECK(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;
    ACL_CHECK(aclrtCreateStream(&stream));
    INFO_LOG("[Rank %d] Step 2: device set, stream created", rankId);

    RootInfoExchanger exchanger(static_cast<uint32_t>(rankId), static_cast<uint32_t>(rankNum), ipport);
    HcclRootInfo rootInfo;
    INFO_LOG("[Rank %d] Step 3: exchanging rootInfo", rankId);
    if (!exchanger.Exchange(rootInfo)) {
        ERROR_LOG("rank %d exchange rootInfo failed", rankId);
        return -1;
    }
    INFO_LOG("[Rank %d] Step 4: rootInfo exchanged", rankId);

    HcclCommConfig config;
    HcclCommConfigInit(&config);
    config.hcclWorldRankID = static_cast<uint32_t>(rankId);
    HcclComm comm = nullptr;
    INFO_LOG("[Rank %d] Step 5: HcclCommInitRootInfoConfig", rankId);
    HCCL_CHECK(HcclCommInitRootInfoConfig(static_cast<uint32_t>(rankNum), &rootInfo, static_cast<uint32_t>(rankId),
                                          &config, &comm));
    if (comm == nullptr) {
        ERROR_LOG("rank %d HcclCommInitRootInfoConfig failed", rankId);
        return -1;
    }
    INFO_LOG("[Rank %d] Step 6: HCCL comm initialized", rankId);

    CommChannelBuilder builder(comm);
    CommContext hostCtx = {};
    const char *ctxTag = "quant_matmul_reduce_scatter";

    INFO_LOG("[Rank %d] Step 7: CreateDeviceContext", rankId);
    CommContext *devContext = reinterpret_cast<CommContext *>(
        builder.CreateDeviceContext(&hostCtx, sizeof(CommContext), ctxTag, &hostCtx.udmaCtx, &hostCtx.ubmemCtx));
    if (devContext == nullptr) {
        ERROR_LOG("rank %d CreateDeviceContext failed", rankId);
        return -1;
    }
    INFO_LOG("[Rank %d] Step 8: device context created", rankId);

    exchanger.Barrier();
    exchanger.Close();
    INFO_LOG("[Rank %d] Step 9: exchanger barrier + close", rankId);

    // 输入数据：A [m, k] fp8, ScaleA [m, ceil(k/64)*2] fp8_e8m0
    // B [n, k] fp8 (TransB=true, 列主序), ScaleB [n, ceil(k/64)*2] fp8_e8m0
    std::vector<uint8_t> hostA(m * k, 0);
    std::vector<uint8_t> hostB(n * k, 0); // TransB=true: B 存储为 [n, k] C-order
    std::vector<uint8_t> hostScaleA(m * CeilDiv(k, 64) * 2, 0);
    std::vector<uint8_t> hostScaleB(n * CeilDiv(k, 64) * 2, 0);
    // 输出：[chunkM, n] bfloat16 per rank
    std::vector<uint16_t> hostOutput(chunkM * n, 0);

    auto sizeA = static_cast<size_t>(1) * hostA.size() * sizeof(uint8_t);
    auto sizeB = static_cast<size_t>(1) * hostB.size() * sizeof(uint8_t);
    auto sizeScaleA = static_cast<size_t>(1) * hostScaleA.size() * sizeof(uint8_t);
    auto sizeScaleB = static_cast<size_t>(1) * hostScaleB.size() * sizeof(uint8_t);
    auto sizeOutput = static_cast<size_t>(1) * hostOutput.size() * sizeof(uint16_t);

    char exePath[PATH_MAX];
    ssize_t len = readlink("/proc/self/exe", exePath, sizeof(exePath) - 1);
    std::string baseDir = ".";
    if (len > 0) {
        exePath[len] = '\0';
        baseDir = exePath;
        size_t lastSlash = baseDir.find_last_of('/');
        if (lastSlash != std::string::npos) {
            baseDir.resize(lastSlash);
        }
    }
    std::string inputDir = baseDir + "/input/" + std::to_string(rankId);
    std::string outputDir = baseDir + "/output/" + std::to_string(rankId);
    ReadFile(inputDir + "/input_a.bin", hostA.data(), sizeA);
    ReadFile(inputDir + "/input_b.bin", hostB.data(), sizeB);
    ReadFile(inputDir + "/input_scaleA.bin", hostScaleA.data(), sizeScaleA);
    ReadFile(inputDir + "/input_scaleB.bin", hostScaleB.data(), sizeScaleB);

    GM_ADDR deviceA = nullptr;
    GM_ADDR deviceB = nullptr;
    GM_ADDR deviceScaleA = nullptr;
    GM_ADDR deviceScaleB = nullptr;
    GM_ADDR deviceOutput = nullptr;
    GM_ADDR deviceWorkspace = nullptr;

    ACL_CHECK(aclrtMalloc((void **)&deviceA, sizeA, ACL_MEM_MALLOC_HUGE_ONLY));
    ACL_CHECK(aclrtMalloc((void **)&deviceB, sizeB, ACL_MEM_MALLOC_HUGE_ONLY));
    ACL_CHECK(aclrtMalloc((void **)&deviceScaleA, sizeScaleA, ACL_MEM_MALLOC_HUGE_ONLY));
    ACL_CHECK(aclrtMalloc((void **)&deviceScaleB, sizeScaleB, ACL_MEM_MALLOC_HUGE_ONLY));
    // cGm: 最终输出 [chunkM, n] bfloat16
    size_t sizeC = static_cast<size_t>(chunkM) * n * sizeof(uint16_t);
    ACL_CHECK(aclrtMalloc((void **)&deviceOutput, sizeC, ACL_MEM_MALLOC_HUGE_ONLY));
    // workspace: AIC 写入 C_i 的缓冲区, 大小 = m * n * sizeof(bfloat16)
    // PUT 模式下 localAddr = workspace, AIV 从此处推送到其他 rank win 区
    size_t sizeWorkspace = static_cast<size_t>(m) * n * sizeof(uint16_t);
    ACL_CHECK(aclrtMalloc((void **)&deviceWorkspace, sizeWorkspace, ACL_MEM_MALLOC_HUGE_ONLY));
    INFO_LOG("[Rank %d] workspace size=%zu bytes (m=%d, n=%d)", rankId, sizeWorkspace, m, n);

    ACL_CHECK(aclrtMemcpy(deviceA, sizeA, hostA.data(), sizeA, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(deviceB, sizeB, hostB.data(), sizeB, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(deviceScaleA, sizeScaleA, hostScaleA.data(), sizeScaleA, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(deviceScaleB, sizeScaleB, hostScaleB.data(), sizeScaleB, ACL_MEMCPY_HOST_TO_DEVICE));
    INFO_LOG("[Rank %d] Step 10: data loaded to device", rankId);

    INFO_LOG("[Rank %d] Step 11: launching kernel (usedCoreNum=%u, PUT mode)", rankId,
             tilingData.mmTilingData.usedCoreNum);
    QuantMatmulReduceScatterKernelE4M3E4M3<<<tilingData.mmTilingData.usedCoreNum, nullptr, stream>>>(
        devContext, deviceA, deviceScaleA, deviceB, deviceScaleB, deviceWorkspace, deviceOutput, tilingData);
    INFO_LOG("[Rank %d] Step 12: kernel launched, syncing...", rankId);

    ACL_CHECK(aclrtSynchronizeStream(stream));
    INFO_LOG("[Rank %d] Step 13: stream synced", rankId);

    // 只拷回 chunkM * n 的输出（reduceSum 后的 per-rank 结果），precision/perf 模式共用
    ACL_CHECK(aclrtMemcpy(hostOutput.data(), sizeOutput, deviceOutput, sizeOutput, ACL_MEMCPY_DEVICE_TO_HOST));

    WriteFile(outputDir + "/npu_out.bin", hostOutput.data(), sizeOutput);

    constexpr int32_t BENCHMARK_ITERATIONS = 10;
    int32_t iterations = (mode == "perf") ? BENCHMARK_ITERATIONS : 1;
    if (iterations > 1) {
        for (int32_t i = 1; i < iterations; ++i) {
            auto t0 = std::chrono::steady_clock::now();
            QuantMatmulReduceScatterKernelE4M3E4M3<<<tilingData.mmTilingData.usedCoreNum, nullptr, stream>>>(
                devContext, deviceA, deviceScaleA, deviceB, deviceScaleB, deviceWorkspace, deviceOutput, tilingData);
            ACL_CHECK(aclrtSynchronizeStream(stream));
            auto t1 = std::chrono::steady_clock::now();
            double elapsedUs = std::chrono::duration<double, std::micro>(t1 - t0).count();
            INFO_LOG("[Rank %d] iteration %d/%d: %.3f us", rankId, i + 1, iterations, elapsedUs);
        }
    }

    aclrtFree(deviceA);
    aclrtFree(deviceScaleA);
    aclrtFree(deviceB);
    aclrtFree(deviceScaleB);
    aclrtFree(deviceOutput);
    aclrtFree(deviceWorkspace);

    HcclCommDestroy(comm);

    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(deviceId));
    ACL_CHECK(aclFinalize());

    return 0;
}

int main(int argc, char *argv[])
{
    int m, k, n, rankNum, headMSize;
    std::string mode;
    try {
        parseArguments(argc, argv, m, k, n, rankNum, mode, headMSize);
    } catch (const std::invalid_argument &e) {
        std::cerr << e.what() << std::endl;
        printUsage(argv[0]);
        return -1;
    }

    INFO_LOG("Master (PID=%d) will fork %d processes (mode=%s, headMSize=%d)", getpid(), rankNum, mode.c_str(),
             headMSize);

    std::vector<pid_t> pids(rankNum);
    for (int rankId = 0; rankId < rankNum; ++rankId) {
        pid_t pid = fork();

        if (pid < 0) {
            ERROR_LOG("Fork failed for rank %d", rankId);
            exit(-1);
        } else if (pid == 0) {
            int ret = runQuantMatmulReduceScatter(rankNum, rankId, m, k, n, mode, headMSize);
            exit(ret);
        } else {
            pids[rankId] = pid;
            INFO_LOG("Forked Rank %d -> PID %d", rankId, pid);
        }
    }

    int status;
    bool all_success = true;
    for (int rankId = 0; rankId < rankNum; ++rankId) {
        pid_t pid = pids[rankId];
        waitpid(pid, &status, 0);
        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
            all_success = false;
            ERROR_LOG("Worker PID %d failed", pid);
        }
    }

    std::cout << "All workers finished. Status: " << (all_success ? "SUCCESS" : "FAILURE") << std::endl;
    return all_success ? 0 : -1;
}
