/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <chrono>
#include <algorithm>
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
#include "apace/tiling/matmul_tiling_swat.h"
#include "apace/kernel/matmul/batch_matmul/all_to_all_mm_tiling_data.h"
#include "apace_a2a_mm_kernel_launcher.h"
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
    std::cerr << "Usage: " << programName << " m k n rankNum [localMatmul] [headMSize] [mode] [dtype] [testMode]"
              << std::endl;
    std::cerr << "Args: " << std::endl;
    std::cerr << "  m: row of matrix A" << std::endl;
    std::cerr << "  k: col of matrix A (total K, distributed across ranks)" << std::endl;
    std::cerr << "  n: col of matrix B" << std::endl;
    std::cerr << "  rankNum: number of ranks" << std::endl;
    std::cerr << "  localMatmul: optional, 0=REMOTE融合 | 1=DEFERRED_SYNC | 2=GM累加(LOCAL+REMOTE) (default 0)"
              << std::endl;
    std::cerr << "  headMSize: optional, long block M size (default 0 = auto from tiling; >0 overrides)" << std::endl;
    std::cerr << "  mode: optional, 'precision' (default) | 'perf'" << std::endl;
    std::cerr << "  dtype: optional, 'fp16' (default) | 'bf16'" << std::endl;
    std::cerr << "  testMode: optional, 0=FUSED (default) | 1=COMM_ONLY | 2=MM_ONLY" << std::endl;
    std::cerr << "Example: " << programName << " 100 200 64 4" << std::endl;
    std::cerr << "         " << programName << " 2048 3584 4096 2 1 512 perf bf16" << std::endl;
}

void parseArguments(int argc, char *argv[], int &m, int &k, int &n, int &rankNum, int &localMatmul, int &headMSize,
                    std::string &mode, std::string &dtype, int &testMode)
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

    localMatmul = (argc >= 6) ? std::stoi(argv[5]) : 0;
    if (localMatmul < 0 || localMatmul > 2) {
        throw std::invalid_argument("ERROR: localMatmul must be in [0, 2]");
    }

    headMSize = (argc >= 7) ? std::stoi(argv[6]) : 0;
    if (headMSize < 0) {
        throw std::invalid_argument("ERROR: headMSize must be non-negative (0 = auto from tiling)");
    }

    mode = (argc >= 8) ? std::string(argv[7]) : std::string("precision");
    if (mode != "precision" && mode != "perf") {
        throw std::invalid_argument("ERROR: mode must be 'precision' or 'perf'");
    }

    dtype = (argc >= 9) ? std::string(argv[8]) : std::string("fp16");
    if (dtype != "fp16" && dtype != "bf16") {
        throw std::invalid_argument("ERROR: dtype must be 'fp16' or 'bf16'");
    }

    testMode = (argc >= 10) ? std::stoi(argv[9]) : 0;
    if (testMode < 0 || testMode > 2) {
        throw std::invalid_argument("ERROR: testMode must be in [0, 2]");
    }

    if (m <= 0 || k <= 0 || n <= 0 || rankNum <= 0) {
        throw std::invalid_argument("ERROR: m k n rankNum must be positive");
    }

    if (k % rankNum != 0) {
        throw std::invalid_argument("ERROR: k must be divisible by rankNum");
    }
}

int runAllToAllMatmul(int rankNum, int rankId, int m, int k, int n, int localMatmul, int headMSizeArg,
                      const std::string &mode, const std::string &dtype, int testMode)
{
    const char *ipport = "tcp://127.0.0.1:8998";
    INFO_LOG("rankNum=%d, rankId=%d, ipport=%s, localMatmul=%d, headMSize=%d, mode=%s, dtype=%s, testMode=%d", rankNum,
             rankId, ipport, localMatmul, headMSizeArg, mode.c_str(), dtype.c_str(), testMode);

    uint32_t ka = k / rankNum;

    allToAllMmMatmulTilingData tilingData;

    MatmulTilingSwat tilingEngine;
    // 非量化 tiling 引擎按数据类型字节宽度推导 baseK（fp16 = 2 字节）
    tilingEngine.GetTilingData(m, n, ka, sizeof(uint16_t), false, true, tilingData.tileMmTilingData, true);

    // headMSize：命令行显式指定（>0）则直接采用（含钳制不超过 m）；未指定（0）则从 tiling 推导：
    // 将 usedCoreNum 个核按 N 方向 tile 列均分，每列核数 × baseM 即 M 方向每个 head 块的高度。
    uint32_t headMSize = static_cast<uint32_t>(headMSizeArg);
    if (headMSize == 0) {
        auto nTile = CeilDiv(n, tilingData.tileMmTilingData.baseN);
        headMSize = CeilDiv(tilingData.tileMmTilingData.usedCoreNum, nTile) *
                    tilingData.tileMmTilingData.baseM;
    }
    if (headMSize > static_cast<uint32_t>(m)) {
        headMSize = static_cast<uint32_t>(m);
    }
    uint32_t headTileCnt = static_cast<uint32_t>(m) / headMSize;
    uint32_t tailMSize = static_cast<uint32_t>(m) % headMSize;
    uint32_t tailTileCnt = (tailMSize > 0) ? 1 : 0;
    uint32_t tileCnt = headTileCnt + tailTileCnt;

    tilingData.commTilingData.splitAxisTileCnt = static_cast<uint64_t>(headTileCnt);
    tilingData.commTilingData.splitAxisTileSize = static_cast<uint64_t>(headMSize);
    tilingData.commTilingData.splitAxisTailSize = static_cast<uint64_t>(tailMSize);
    tilingData.commTilingData.splitAxisTailCnt = static_cast<uint64_t>(tailTileCnt);
    tilingData.commTilingData.nonSplitAxisSize = ka;
    tilingData.localMatmul = static_cast<uint32_t>(localMatmul);
    tilingData.testMode = static_cast<uint32_t>(testMode);

    INFO_LOG("TileCnt=%u, HeadTileCnt=%u, HeadMSize=%u, TailTileCnt=%u, TailMSize=%u", tileCnt, headTileCnt, headMSize,
             tailTileCnt, tailMSize);

    ACL_CHECK(aclInit(nullptr));
    int32_t deviceId = rankId;
    ACL_CHECK(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;
    ACL_CHECK(aclrtCreateStream(&stream));

    RootInfoExchanger exchanger(static_cast<uint32_t>(rankId), static_cast<uint32_t>(rankNum), ipport);
    HcclRootInfo rootInfo;
    if (!exchanger.Exchange(rootInfo)) {
        ERROR_LOG("rank %d exchange rootInfo failed", rankId);
        return -1;
    }

    HcclCommConfig config;
    HcclCommConfigInit(&config);
    config.hcclWorldRankID = static_cast<uint32_t>(rankId);
    HcclComm comm = nullptr;
    HCCL_CHECK(HcclCommInitRootInfoConfig(static_cast<uint32_t>(rankNum), &rootInfo, static_cast<uint32_t>(rankId),
                                          &config, &comm));
    if (comm == nullptr) {
        ERROR_LOG("rank %d HcclCommInitRootInfoConfig failed", rankId);
        return -1;
    }

    CommChannelBuilder builder(comm);
    CommContext hostCtx = {};
    const char *ctxTag = "all_to_all_matmul";

    CommContext *devContext = reinterpret_cast<CommContext *>(
        builder.CreateDeviceContext(&hostCtx, sizeof(CommContext), ctxTag, &hostCtx.udmaCtx, &hostCtx.ubmemCtx));
    if (devContext == nullptr) {
        ERROR_LOG("rank %d CreateDeviceContext failed", rankId);
        return -1;
    }

    exchanger.Barrier();
    exchanger.Close();

    std::vector<uint16_t> hostA(m * k, 0);
    std::vector<uint16_t> hostB(n * k, 0);
    std::vector<float> hostBias(n, 0);
    std::vector<uint16_t> hostOutput(m * n, 0);

    auto sizeA = static_cast<size_t>(1) * hostA.size() * sizeof(uint16_t);
    auto sizeB = static_cast<size_t>(1) * hostB.size() * sizeof(uint16_t);
    auto sizeBias = static_cast<size_t>(1) * hostBias.size() * sizeof(float);
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
    if (!ReadFile(inputDir + "/input_a.bin", hostA.data(), sizeA)) {
        ERROR_LOG("rank %d failed to read input_a.bin", rankId);
        return -1;
    }
    if (!ReadFile(inputDir + "/input_b.bin", hostB.data(), sizeB)) {
        ERROR_LOG("rank %d failed to read input_b.bin", rankId);
        return -1;
    }
    if (!ReadFile(inputDir + "/input_bias.bin", hostBias.data(), sizeBias)) {
        ERROR_LOG("rank %d failed to read input_bias.bin", rankId);
        return -1;
    }

    GM_ADDR deviceA = nullptr;
    GM_ADDR deviceB = nullptr;
    GM_ADDR deviceBias = nullptr;
    GM_ADDR deviceOutput = nullptr;

    ACL_CHECK(aclrtMalloc((void **)&deviceA, sizeA, ACL_MEM_MALLOC_HUGE_ONLY));
    ACL_CHECK(aclrtMalloc((void **)&deviceB, sizeB, ACL_MEM_MALLOC_HUGE_ONLY));
    ACL_CHECK(aclrtMalloc((void **)&deviceBias, sizeBias, ACL_MEM_MALLOC_HUGE_ONLY));
    ACL_CHECK(aclrtMalloc((void **)&deviceOutput, sizeOutput, ACL_MEM_MALLOC_HUGE_ONLY));

    ACL_CHECK(aclrtMemcpy(deviceA, sizeA, hostA.data(), sizeA, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(deviceB, sizeB, hostB.data(), sizeB, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(deviceBias, sizeBias, hostBias.data(), sizeBias, ACL_MEMCPY_HOST_TO_DEVICE));

    // MM_ONLY（testMode==2）：AIV 不通信，需把期望通信窗口预填到设备端 commBuffer
    // （内容 = gen_data 的 comm_window.bin，self 槽位为 0；kernel 自 rank 贡献走 localAGmAddr）
    if (testMode == 2) {
        size_t commPayload = static_cast<size_t>(m) * static_cast<size_t>(k) * sizeof(uint16_t);
        std::vector<uint16_t> commWin(m * k, 0);
        if (!ReadFile(inputDir + "/comm_window.bin", commWin.data(), commPayload)) {
            ERROR_LOG("rank %d failed to read comm_window.bin for MM_ONLY prefill", rankId);
            return -1;
        }
        uint64_t winAddr = hostCtx.udmaCtx.commBufferAddrs[rankId];
        ACL_CHECK(aclrtMemcpy(reinterpret_cast<void *>(winAddr), commPayload, commWin.data(), commPayload,
                              ACL_MEMCPY_HOST_TO_DEVICE));
    }

    // 按 dtype 选择 fp16/bf16 内核（两者数据宽度均为 2 字节，host 侧 buffer/tiling 一致）
    auto LaunchKernel = [&]() {
        if (dtype == "bf16") {
            AllToAllMmKernel_Bf16_Udma<<<tilingData.tileMmTilingData.usedCoreNum, nullptr, stream>>>(
                devContext, deviceA, deviceB, deviceOutput, deviceBias, tilingData);
        } else {
            AllToAllMmKernel_Udma<<<tilingData.tileMmTilingData.usedCoreNum, nullptr, stream>>>(
                devContext, deviceA, deviceB, deviceOutput, deviceBias, tilingData);
        }
    };

    if (mode == "precision") {
        LaunchKernel();

        ACL_CHECK(aclrtSynchronizeStream(stream));

        ACL_CHECK(aclrtMemcpy(hostOutput.data(), sizeOutput, deviceOutput, sizeOutput, ACL_MEMCPY_DEVICE_TO_HOST));

        WriteFile(outputDir + "/npu_out.bin", hostOutput.data(), sizeOutput);

        // ---- 通信窗口 dump（仅通信验证，不影响 matmul 输出）----
        // 窗口[r] 布局 = concat_s a_local[s][r*m:(r+1)*m, :]，即 gen_data 的
        // a_after_alltoall_list[r]，payload 大小 = M_total*Ka*fp16 = m*k*2 字节。
        // 用于与 expected_comm.bin 比对，单独验证通信正确性。
        {
            size_t commPayload = static_cast<size_t>(m) * static_cast<size_t>(k) * sizeof(uint16_t);
            std::vector<uint16_t> hostComm(m * k, 0);
            uint64_t winAddr = hostCtx.udmaCtx.commBufferAddrs[rankId];
            ACL_CHECK(aclrtMemcpy(hostComm.data(), commPayload, reinterpret_cast<void *>(winAddr), commPayload,
                                  ACL_MEMCPY_DEVICE_TO_HOST));
            WriteFile(outputDir + "/comm_buffer.bin", hostComm.data(), commPayload);
        }
    } else {
        constexpr int64_t BOOST_ELEM_COUNT = 10000LL * 10000LL;
        size_t boostSize = static_cast<size_t>(BOOST_ELEM_COUNT) * sizeof(uint16_t);
        GM_ADDR boostIn = nullptr;
        GM_ADDR boostOut = nullptr;
        ACL_CHECK(aclrtMalloc((void **)&boostIn, boostSize, ACL_MEM_MALLOC_HUGE_ONLY));
        ACL_CHECK(aclrtMalloc((void **)&boostOut, boostSize, ACL_MEM_MALLOC_HUGE_ONLY));
        std::vector<uint16_t> boostHost(BOOST_ELEM_COUNT, 0x3F00);
        ACL_CHECK(aclrtMemcpy(boostIn, boostSize, boostHost.data(), boostSize, ACL_MEMCPY_HOST_TO_DEVICE));

        constexpr int64_t CACHE_FLUSH_ELEM_COUNT = 128LL * 1024 * 1024;
        size_t cacheFlushSize = static_cast<size_t>(CACHE_FLUSH_ELEM_COUNT) * sizeof(uint16_t);
        GM_ADDR cacheFlush = nullptr;
        ACL_CHECK(aclrtMalloc((void **)&cacheFlush, cacheFlushSize, ACL_MEM_MALLOC_HUGE_ONLY));
        std::vector<uint16_t> cacheFlushHost(CACHE_FLUSH_ELEM_COUNT, 0x0000);
        ACL_CHECK(
            aclrtMemcpy(cacheFlush, cacheFlushSize, cacheFlushHost.data(), cacheFlushSize, ACL_MEMCPY_HOST_TO_DEVICE));

        constexpr int PERF_LOOP_COUNT = 10;
        std::vector<double> durations(PERF_LOOP_COUNT);
        for (int i = 0; i < PERF_LOOP_COUNT; ++i) {
            auto t0 = std::chrono::steady_clock::now();
            LaunchKernel();
            auto t1 = std::chrono::steady_clock::now();
            durations[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();
        }

        double sum = 0, minv = 1e9, maxv = 0;
        for (int i = 1; i < PERF_LOOP_COUNT; ++i) {
            sum += durations[i];
            minv = std::min(minv, durations[i]);
            maxv = std::max(maxv, durations[i]);
        }
        double avg = sum / (PERF_LOOP_COUNT - 1);
        INFO_LOG("[Rank %d] localMatmul=%d headMSize=%u tileCnt=%u | per-iter ms: avg=%.3f min=%.3f max=%.3f "
                 "(excl warmup)", rankId, localMatmul, headMSize, tileCnt, avg, minv, maxv);

        ACL_CHECK(aclrtSynchronizeStream(stream));
        ACL_CHECK(aclrtMemcpy(hostOutput.data(), sizeOutput, deviceOutput, sizeOutput, ACL_MEMCPY_DEVICE_TO_HOST));
        WriteFile(outputDir + "/npu_out.bin", hostOutput.data(), sizeOutput);

        aclrtFree(boostIn);
        aclrtFree(boostOut);
        aclrtFree(cacheFlush);
    }

    aclrtFree(deviceA);
    aclrtFree(deviceB);
    aclrtFree(deviceBias);
    aclrtFree(deviceOutput);

    HcclCommDestroy(comm);

    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(deviceId));
    ACL_CHECK(aclFinalize());

    return 0;
}

int main(int argc, char *argv[])
{
    int m, k, n, rankNum, localMatmul, headMSize, testMode;
    std::string mode, dtype;
    try {
        parseArguments(argc, argv, m, k, n, rankNum, localMatmul, headMSize, mode, dtype, testMode);
    } catch (const std::invalid_argument &e) {
        std::cerr << e.what() << std::endl;
        printUsage(argv[0]);
        return -1;
    }

    INFO_LOG("Master (PID=%d) will fork %d processes (localMatmul=%d, headMSize=%d, mode=%s, dtype=%s)", getpid(),
             rankNum, localMatmul, headMSize, mode.c_str(), dtype.c_str());

    std::vector<pid_t> pids(rankNum);
    for (int rankId = 0; rankId < rankNum; ++rankId) {
        pid_t pid = fork();

        if (pid < 0) {
            ERROR_LOG("Fork failed for rank %d", rankId);
            exit(-1);
        } else if (pid == 0) {
            int ret = runAllToAllMatmul(rankNum, rankId, m, k, n, localMatmul, headMSize, mode, dtype, testMode);
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
