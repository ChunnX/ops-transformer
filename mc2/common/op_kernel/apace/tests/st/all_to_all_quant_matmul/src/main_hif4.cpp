/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.openhisilicon.com
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the
 * License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file main_hif4.cpp
 * \brief AllToAllQuantMatmul HiF4 Hcomm(ST) 精度/性能入口
 *
 * 与主线 allto_all_matmul_apt.cpp 的 HiF4 分支使用相同
 * Apace::AllToAllMxQuantMatmulHcommImpl(ScaleType=hif4_scale)。
 *
 * 数据布局(gen_data.py hif4 模式生成):
 *   input_a.bin:  hifloat4x2_t[M_total * Ka]      (S1P2 2:1 打包, 1 byte = 2 elements)
 *   input_b.bin:  hifloat4x2_t[N * K] (transB)     (同上; 非 transB 时为 [K * N])
 *   input_scaleA.bin: hif4_scale[M_total * CeilDiv(Ka, 64)]  (uint32, 4 bytes/group)
 *   input_scaleB.bin: hif4_scale[N * CeilDiv(K, 64)]         (uint32, 4 bytes/group)
 *   npu_out.bin:  bf16[m * N]
 *
 * [NOTE] Hcomm 通路依赖 GetHcclContext<0> 提供的 HCCL 上下文(由测试框架按主线方式
 * 注册); Blaze 侧 HiF4 BlockMmad 分发就位前, 本入口仅作静态交付, 编译验证随
 * devkit 更新后进行。
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
#include "../../utils/root_info_exchanger.h"
#include "apace/tiling/quant_matmul_tiling_swat.h"
#include "apace/kernel/fusions/all_to_all_quant_matmul/all_to_all_matmul_tiling_data.h"
#include "apace_a2a_kernel_launcher.h"

#define HCCL_CHECK(status) \
    do { \
        HcclResult ret = (status); \
        if (ret != HCCL_SUCCESS) { \
            std::cerr << __FILE__ << ":" << __LINE__ << " HcclResult:" << ret << std::endl; \
        } \
    } while (0)

inline uint64_t CeilDiv(uint64_t a, uint64_t b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

// HiF4: 2 个 4-bit 元素打包为 1 字节
constexpr uint64_t HIF4_DATA_NUM_PER_BYTE = 2UL;
// HiF4: 每 64 元素组 1 个 hif4_scale(4 字节)元数据
constexpr uint64_t HIF4_GROUP_SIZE = 64UL;
// HiF4: Ka 按 128 对齐统一校验(同时保证 Ka%64==0 且 scale 组数为偶数)
constexpr uint64_t HIF4_KA_ALIGN_SIZE = 128UL;
// HiF4: 每 64 元素组的 scale 字节数
constexpr uint64_t HIF4_SCALE_UNIT_BYTES = 4UL;
// HiF4: 仅支持 Hcomm 通路, rank 上限 8
constexpr uint64_t HIF4_MAX_RANK_NUM = 8UL;
constexpr uint64_t ALIGN_NUM = 512UL;
// DFX 预留空间
constexpr uint64_t DFX_RESERVED_BYTES = 64UL * 1024UL;

void printUsage(const std::string &programName)
{
    std::cerr << "Usage: " << programName << " m k n rankNum [mode] [headMSize] [transB]" << std::endl;
    std::cerr << "Args: " << std::endl;
    std::cerr << "  m: per-rank output M (x1 rows = m * rankNum)" << std::endl;
    std::cerr << "  k: total K (distributed across ranks, k % rankNum == 0)" << std::endl;
    std::cerr << "  n: col of matrix B" << std::endl;
    std::cerr << "  rankNum: number of ranks (<= 8 for HiF4 Hcomm)" << std::endl;
    std::cerr << "  mode: optional, 'precision' (default) | 'perf'" << std::endl;
    std::cerr << "  headMSize: optional, long block M size (default 512)" << std::endl;
    std::cerr << "  transB: optional, 'true' (default) | 'false'" << std::endl;
    std::cerr << "Example: " << programName << " 2048 3584 4096 4" << std::endl;
}

void parseArguments(int argc, char *argv[], int &m, int &k, int &n, int &rankNum, std::string &mode, int &headMSize,
                    bool &transB)
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

    transB = true;
    if (argc >= 8) {
        std::string transBStr = std::string(argv[7]);
        if (transBStr == "true" || transBStr == "1") {
            transB = true;
        } else if (transBStr == "false" || transBStr == "0") {
            transB = false;
        } else {
            throw std::invalid_argument("ERROR: transB must be 'true' or 'false'");
        }
    }

    if (m <= 0 || k <= 0 || n <= 0 || rankNum <= 0) {
        throw std::invalid_argument("ERROR: m k n rankNum must be positive");
    }

    if (static_cast<uint64_t>(rankNum) > HIF4_MAX_RANK_NUM) {
        throw std::invalid_argument("ERROR: HiF4 Hcomm path supports rankNum <= 8");
    }

    if (k % rankNum != 0) {
        throw std::invalid_argument("ERROR: k must be divisible by rankNum");
    }

    uint64_t ka = static_cast<uint64_t>(k) / static_cast<uint64_t>(rankNum);
    // Ka 按 128 对齐统一校验: 同时保证 Ka%64==0 且 scale 组数为偶数
    // (kernel 侧 SCALE_C0 按数据类型导出, HiF4 为 1; 128 对齐是与 MX 统一的对齐规则)
    if (ka % HIF4_KA_ALIGN_SIZE != 0) {
        throw std::invalid_argument("ERROR: Ka (k / rankNum) must be 128-aligned (Ka % 128 == 0)");
    }
}

int runAllToAllMatmulHiF4(int rankNum, int rankId, int m, int k, int n, const std::string &mode, int headMSizeArg,
                          bool transB)
{
    const char *ipport = "tcp://127.0.0.1:8998";
    INFO_LOG("rankNum=%d, rankId=%d, ipport=%s, mode=%s, headMSize=%d, transB=%d", rankNum, rankId, ipport,
             mode.c_str(), headMSizeArg, transB ? 1 : 0);

    const uint64_t mTotal = static_cast<uint64_t>(m) * static_cast<uint64_t>(rankNum);
    const uint64_t ka = static_cast<uint64_t>(k) / static_cast<uint64_t>(rankNum);

    Apace::hcommAllToAllMatmulTilingData tilingData = {};

    // 量化 matmul tiling: 与 OpHost DoHcommMxQuantMMTiling HiF4 分支一致
    QuantMatmulTilingSwat<mm::DataType::DT_HIFLOAT4, mm::DataType::DT_HIFLOAT4, mm::BiasDataType::DT_FLOAT> tilingEngine;
    tilingEngine.GetTilingData(static_cast<uint64_t>(m), static_cast<uint64_t>(n), ka, false, transB,
                               tilingData.tileQbmmTilingData);

    // 通信 tiling: 与 OpHost SetHcommTilingInfo 一致
    // splitAxis 沿 x1 的 M 轴(M_total = m * rankNum), nonSplitAxisSize 为总 K
    auto nTile = CeilDiv(static_cast<uint64_t>(n), tilingData.tileQbmmTilingData.baseN);
    uint64_t headMSize =
        CeilDiv(tilingData.tileQbmmTilingData.usedCoreNum, nTile) * tilingData.tileQbmmTilingData.baseM;
    uint64_t headTileCnt = mTotal / headMSize;
    uint64_t tailMSize = mTotal % headMSize;
    uint64_t tailTileCnt = (tailMSize > 0) ? 1 : 0;
    uint64_t tileCnt = headTileCnt + tailTileCnt;

    tilingData.commTilingData.splitAxisTileCnt = headTileCnt;
    tilingData.commTilingData.splitAxisTileSize = headMSize;
    tilingData.commTilingData.splitAxisTailSize = tailMSize;
    tilingData.commTilingData.splitAxisTailCnt = tailTileCnt;
    tilingData.commTilingData.nonSplitAxisSize = static_cast<uint64_t>(k);
    tilingData.localMatmul = 0;

    INFO_LOG("TileCnt=%lu, HeadTileCnt=%lu, HeadMSize=%lu, TailTileCnt=%lu, TailMSize=%lu", tileCnt, headTileCnt,
             headMSize, tailTileCnt, tailMSize);

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

    exchanger.Barrier();
    exchanger.Close();

    // host 数据: hifloat4x2 2:1 打包, scale 为 uint32 元素(4 bytes/group)
    const uint64_t scaleGroupsA = CeilDiv(ka, HIF4_GROUP_SIZE);
    const uint64_t scaleGroupsB = CeilDiv(static_cast<uint64_t>(k), HIF4_GROUP_SIZE);
    std::vector<uint8_t> hostA(mTotal * ka / HIF4_DATA_NUM_PER_BYTE, 0);
    // transB=true: B 按 [N, K] 行主序写入(= DN 列主序 [K, N]); transB=false: 按 [K, N] 行主序
    std::vector<uint8_t> hostB(static_cast<uint64_t>(k) * n / HIF4_DATA_NUM_PER_BYTE, 0);
    std::vector<uint32_t> hostScaleA(mTotal * scaleGroupsA, 0);
    std::vector<uint32_t> hostScaleB(n * scaleGroupsB, 0);
    std::vector<uint16_t> hostOutput(static_cast<uint64_t>(m) * n, 0);

    auto sizeA = hostA.size() * sizeof(uint8_t);
    auto sizeB = hostB.size() * sizeof(uint8_t);
    auto sizeScaleA = hostScaleA.size() * sizeof(uint32_t);
    auto sizeScaleB = hostScaleB.size() * sizeof(uint32_t);
    auto sizeOutput = hostOutput.size() * sizeof(uint16_t);

    // workspace: scale 通信缓冲(512 对齐) + A 数据通信缓冲 + DFX 预留
    // 与 impl Init 的划分一致: [0, x1ScaleLen) 为 scale, [x1ScaleLen, +commDataLen) 为数据
    uint64_t x1ScaleLen = scaleGroupsA * HIF4_SCALE_UNIT_BYTES * mTotal;
    x1ScaleLen = CeilDiv(x1ScaleLen, ALIGN_NUM) * ALIGN_NUM;
    uint64_t commDataLen = mTotal * ka / HIF4_DATA_NUM_PER_BYTE;
    uint64_t workspaceSize = x1ScaleLen + commDataLen + DFX_RESERVED_BYTES;

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
    ACL_CHECK(aclrtMalloc((void **)&deviceOutput, sizeOutput, ACL_MEM_MALLOC_HUGE_ONLY));
    ACL_CHECK(aclrtMalloc((void **)&deviceWorkspace, workspaceSize, ACL_MEM_MALLOC_HUGE_ONLY));

    ACL_CHECK(aclrtMemcpy(deviceA, sizeA, hostA.data(), sizeA, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(deviceB, sizeB, hostB.data(), sizeB, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(deviceScaleA, sizeScaleA, hostScaleA.data(), sizeScaleA, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(deviceScaleB, sizeScaleB, hostScaleB.data(), sizeScaleB, ACL_MEMCPY_HOST_TO_DEVICE));

    auto launchKernel = [&]() {
        AllToAllQuantMatmulKernelHiF4HiF4_Hcomm<<<tilingData.tileQbmmTilingData.usedCoreNum, nullptr, stream>>>(
            deviceA, deviceScaleA, deviceB, deviceScaleB, deviceOutput, deviceWorkspace, transB, tilingData);
    };

    if (mode == "precision") {
        launchKernel();

        ACL_CHECK(aclrtSynchronizeStream(stream));

        ACL_CHECK(aclrtMemcpy(hostOutput.data(), sizeOutput, deviceOutput, sizeOutput, ACL_MEMCPY_DEVICE_TO_HOST));

        WriteFile(outputDir + "/npu_out.bin", hostOutput.data(), sizeOutput);
    } else {
        constexpr int PERF_LOOP_COUNT = 10;
        std::vector<double> durations(PERF_LOOP_COUNT);
        for (int i = 0; i < PERF_LOOP_COUNT; ++i) {
            auto t0 = std::chrono::steady_clock::now();
            launchKernel();
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
        INFO_LOG("[Rank %d] headMSize=%lu tileCnt=%lu | per-iter ms: avg=%.3f min=%.3f max=%.3f (excl warmup)", rankId,
                 headMSize, tileCnt, avg, minv, maxv);

        ACL_CHECK(aclrtSynchronizeStream(stream));
        ACL_CHECK(aclrtMemcpy(hostOutput.data(), sizeOutput, deviceOutput, sizeOutput, ACL_MEMCPY_DEVICE_TO_HOST));
        WriteFile(outputDir + "/npu_out.bin", hostOutput.data(), sizeOutput);
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
    bool transB = true;
    try {
        parseArguments(argc, argv, m, k, n, rankNum, mode, headMSize, transB);
    } catch (const std::invalid_argument &e) {
        std::cerr << e.what() << std::endl;
        printUsage(argv[0]);
        return -1;
    }

    INFO_LOG("Master (PID=%d) will fork %d processes (mode=%s, headMSize=%d, transB=%d)", getpid(), rankNum,
             mode.c_str(), headMSize, transB ? 1 : 0);

    std::vector<pid_t> pids(rankNum);
    for (int rankId = 0; rankId < rankNum; ++rankId) {
        pid_t pid = fork();

        if (pid < 0) {
            ERROR_LOG("Fork failed for rank %d", rankId);
            exit(-1);
        } else if (pid == 0) {
            int ret = runAllToAllMatmulHiF4(rankNum, rankId, m, k, n, mode, headMSize, transB);
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
