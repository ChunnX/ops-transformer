/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file main.cpp
 * \brief ApaceQuantMatmulReduceScatter ST host 入口
 *
 * ReduceScatter via AllToAll PUT + AtomicAdd:
 *   - 每 rank 持有 A_i[M × Ka], ScaleA_i, B_full[K × N], ScaleB_full
 *   - 输出 C_r[M/rankSize × N] = Σ_i A_i[r*M/R:(r+1)*M/R] × B_i
 *
 * tiling 推导（DESIGN.md §2.8）：
 *   mLocal = M / rankSize           // ReduceScatter per-rank M
 *   ka     = K / rankSize           // per-rank K
 *   GetTilingData(mLocal, N, ka, ...)  // m=M/rankSize, k=per-rank K
 *   commTilingData 基于 mLocal 切分（而非 M）
 *   localMatmul = 1                 // LOCAL+REMOTE+AtomicAdd
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
#include "heavy_kernels.h"
#include "acl/acl.h"
#include "hccl/hccl.h"
#include "hccl/hccl_comm.h"
#include "hccl/hccl_res.h"
#include "hccl/hccl_types.h"
#include "hccl/hccl_rank_graph.h"
#include <cstdlib>
#include "utils.h"
#include "apace/utils/comm_channel_builder.h"
#include "../../utils/root_info_exchanger.h"
#include "apace/tiling/quant_matmul_tiling_swat.h"
#include "apace/kernel/apace_quant_matmul_reduce_scatter/apace_quant_matmul_reduce_scatter_tiling_data.h"
#include "kernel_launcher.h"
#include "apace/block/aiv_comm/collective_comm_context.h"

#define HCCL_CHECK(status) do { \
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

void printUsage(const std::string& programName)
{
    std::cerr << "Usage: " << programName << " m k n rankNum [mode] [headMSize] [dtype]" << std::endl;
    std::cerr << "Args: " << std::endl;
    std::cerr << "  m: row of matrix A (total M, must be divisible by rankNum)" << std::endl;
    std::cerr << "  k: col of matrix A (total K, must be divisible by rankNum)" << std::endl;
    std::cerr << "  n: col of matrix B" << std::endl;
    std::cerr << "  rankNum: number of ranks" << std::endl;
    std::cerr << "  mode: optional, 'precision' (default) | 'perf'" << std::endl;
    std::cerr << "  headMSize: optional, long block M size (default 512, info-only, actual derived by formula)" << std::endl;
    std::cerr << "  dtype: optional, FP8 dtype combination (default: e4m3e4m3)" << std::endl;
    std::cerr << "    supported: e4m3e4m3, e5m2e5m2, e4m3e5m2, e5m2e4m3" << std::endl;
    std::cerr << "Example: " << programName << " 2048 4096 2048 4" << std::endl;
    std::cerr << "         " << programName << " 2048 8192 4096 4 perf 512 e5m2e5m2" << std::endl;
}

void parseArguments(int argc, char* argv[], int& m, int& k, int& n, int& rankNum, std::string& mode, int& headMSize, std::string& dtype)
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
    } catch (const std::invalid_argument&) {
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

    dtype = (argc >= 8) ? std::string(argv[7]) : std::string("e4m3e4m3");
    if (dtype != "e4m3e4m3" && dtype != "e5m2e5m2" && dtype != "e4m3e5m2" && dtype != "e5m2e4m3") {
        throw std::invalid_argument("ERROR: dtype must be one of: e4m3e4m3, e5m2e5m2, e4m3e5m2, e5m2e4m3");
    }

    if (m <= 0 || k <= 0 || n <= 0 || rankNum <= 0) {
        throw std::invalid_argument("ERROR: m k n rankNum must be positive");
    }

    // ReduceScatter 输入校验（DESIGN.md §2.8）
    if (m % rankNum != 0) {
        throw std::invalid_argument("ERROR: M must be divisible by rankNum (ReduceScatter M-split)");
    }
    if (k % rankNum != 0) {
        throw std::invalid_argument("ERROR: K must be divisible by rankNum (K-split)");
    }

    uint32_t ka = k / rankNum;
    // ka % 64 == 0：确保 ka/32 等价于 CeilDiv(ka,64)*2，避免 scale 通信数据量截断
    if (ka % 64 != 0) {
        throw std::invalid_argument("ERROR: ka(=K/rankNum) must be divisible by 64 (MXFP8 scale alignment)");
    }
    if (CeilDiv(ka, 64) % 2 != 0) {
        throw std::invalid_argument("ERROR: CeilDiv(ka, 64) must be an even number (redundant check)");
    }
}

int runReduceScatterMatmul(int rankNum, int rankId, int m, int k, int n, const std::string& mode, int headMSizeArg, const std::string& dtype) {
    const char *ipport = "tcp://127.0.0.1:8998";
    INFO_LOG("rankNum=%d, rankId=%d, ipport=%s, mode=%s, headMSize=%d, dtype=%s",
             rankNum, rankId, ipport, mode.c_str(), headMSizeArg, dtype.c_str());

    // ReduceScatter tiling 推导（DESIGN.md §2.8）
    uint32_t ka = k / rankNum;          // per-rank K
    uint32_t mLocal = m / rankNum;      // ReduceScatter per-rank M（输出行数）

    ApaceQuantMatmulReduceScatterTilingData tilingData;

    // GetTilingData k 参数必须为 per-rank K（ka = K/rankSize），非全量 K
    // PUT 模式硬性不变量（operator-anatomy.md §3.7：PUT 用 ka，AG 用 totalLogicalM）
    // 根据 dtype 选择对应的 tiling 引擎模板实例化
    if (dtype == "e4m3e4m3") {
        QuantMatmulTilingSwat<mm::DataType::DT_FLOAT8_E4M3FN, mm::DataType::DT_FLOAT8_E4M3FN> tilingEngine;
        tilingEngine.GetTilingData(mLocal, n, ka, false, true, tilingData.tileQbmmTilingData);
    } else if (dtype == "e5m2e5m2") {
        QuantMatmulTilingSwat<mm::DataType::DT_FLOAT8_E5M2, mm::DataType::DT_FLOAT8_E5M2> tilingEngine;
        tilingEngine.GetTilingData(mLocal, n, ka, false, true, tilingData.tileQbmmTilingData);
    } else if (dtype == "e4m3e5m2") {
        QuantMatmulTilingSwat<mm::DataType::DT_FLOAT8_E4M3FN, mm::DataType::DT_FLOAT8_E5M2> tilingEngine;
        tilingEngine.GetTilingData(mLocal, n, ka, false, true, tilingData.tileQbmmTilingData);
    } else if (dtype == "e5m2e4m3") {
        QuantMatmulTilingSwat<mm::DataType::DT_FLOAT8_E5M2, mm::DataType::DT_FLOAT8_E4M3FN> tilingEngine;
        tilingEngine.GetTilingData(mLocal, n, ka, false, true, tilingData.tileQbmmTilingData);
    } else {
        ERROR_LOG("Unsupported dtype: %s", dtype.c_str());
        return -1;
    }

    // headMSize 由公式推导（CeilDiv(usedCoreNum, nTile) * baseM），cases.csv head_m_size 列仅信息性参考
    auto nTile = CeilDiv(n, tilingData.tileQbmmTilingData.baseN);
    uint32_t headMSize =
        CeilDiv(tilingData.tileQbmmTilingData.usedCoreNum, nTile) * tilingData.tileQbmmTilingData.baseM;
    // commTilingData 基于 mLocal（M/rankSize）切分，而非 M
    uint32_t headTileCnt = mLocal / headMSize;
    uint32_t tailMSize = mLocal % headMSize;
    uint32_t tailTileCnt = (tailMSize > 0) ? 1 : 0;
    uint32_t tileCnt = headTileCnt + tailTileCnt;

    tilingData.commTilingData.splitAxisTileCnt = static_cast<uint64_t>(headTileCnt);
    tilingData.commTilingData.splitAxisTileSize = static_cast<uint64_t>(headMSize);
    tilingData.commTilingData.splitAxisTailSize = static_cast<uint64_t>(tailMSize);
    tilingData.commTilingData.splitAxisTailCnt = static_cast<uint64_t>(tailTileCnt);
    tilingData.commTilingData.nonSplitAxisSize = ka;
    // localMatmul=1：LOCAL+REMOTE+AtomicAdd（通算并行是 MC2 核心价值）
    tilingData.localMatmul = 1;

    tilingData.scaleCommTilingData = tilingData.commTilingData;
    tilingData.scaleCommTilingData.nonSplitAxisSize = ka / 32;  // MXFP8 scale 压缩比

    INFO_LOG("ReduceScatter: M=%d, mLocal=%u, K=%d, ka=%u, N=%d, tileCnt=%u, headMSize=%u, headTileCnt=%u, tailMSize=%u, localMatmul=%u",
             m, mLocal, k, ka, n, tileCnt, headMSize, headTileCnt, tailMSize, tilingData.localMatmul);

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
    HCCL_CHECK(HcclCommInitRootInfoConfig(static_cast<uint32_t>(rankNum), &rootInfo,
                                          static_cast<uint32_t>(rankId), &config, &comm));
    if (comm == nullptr) {
        ERROR_LOG("rank %d HcclCommInitRootInfoConfig failed", rankId);
        return -1;
    }

    CommChannelBuilder builder(comm);
    CommContext hostCtx = {};
    const char *ctxTag = "apace_quant_matmul_reduce_scatter";

    CommContext *devContext = reinterpret_cast<CommContext *>(
        builder.CreateDeviceContext(&hostCtx, sizeof(CommContext), ctxTag,
                                    &hostCtx.udmaCtx, &hostCtx.ubmemCtx));
    if (devContext == nullptr) {
        ERROR_LOG("rank %d CreateDeviceContext failed", rankId);
        return -1;
    }

    exchanger.Barrier();
    exchanger.Close();

    // ReduceScatter 数据布局（DESIGN.md §1.5）：
    //   A: [M × Ka] per rank（K-split，Ka = K/rankNum）
    //   ScaleA: [M × CeilDiv(Ka,64)*2] per rank
    //   B: [K × N] 全量复制（K = rankNum × Ka）
    //   ScaleB: [CeilDiv(K,64)*2 × N] 全量复制
    //   C: [M/rankNum × N] per rank（M-split 输出）
    std::vector<uint8_t> hostA(m * ka, 0);
    std::vector<uint8_t> hostB(k * n, 0);
    std::vector<uint8_t> hostScaleA(m * CeilDiv(ka, 64) * 2, 0);
    std::vector<uint8_t> hostScaleB(CeilDiv(k, 64) * 2 * n, 0);
    // 输出 C: mLocal × N（ReduceScatter per-rank 输出）
    std::vector<uint16_t> hostOutput(mLocal * n, 0);

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

    ACL_CHECK(aclrtMalloc((void**)&deviceA, sizeA, ACL_MEM_MALLOC_HUGE_ONLY));
    ACL_CHECK(aclrtMalloc((void**)&deviceB, sizeB, ACL_MEM_MALLOC_HUGE_ONLY));
    ACL_CHECK(aclrtMalloc((void**)&deviceScaleA, sizeScaleA, ACL_MEM_MALLOC_HUGE_ONLY));
    ACL_CHECK(aclrtMalloc((void**)&deviceScaleB, sizeScaleB, ACL_MEM_MALLOC_HUGE_ONLY));
    ACL_CHECK(aclrtMalloc((void**)&deviceOutput, sizeOutput, ACL_MEM_MALLOC_HUGE_ONLY));

    ACL_CHECK(aclrtMemcpy(deviceA, sizeA, hostA.data(), sizeA, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(deviceB, sizeB, hostB.data(), sizeB, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(deviceScaleA, sizeScaleA, hostScaleA.data(), sizeScaleA, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(deviceScaleB, sizeScaleB, hostScaleB.data(), sizeScaleB, ACL_MEMCPY_HOST_TO_DEVICE));

    if (mode == "precision") {
        // 根据 dtype 选择对应的 kernel 入口
        if (dtype == "e4m3e4m3") {
            ApaceQuantMatmulReduceScatterKernelE4M3E4M3_Udma<<<
                tilingData.tileQbmmTilingData.usedCoreNum, nullptr, stream>>>(
                    devContext, deviceA, deviceScaleA, deviceB, deviceScaleB, deviceOutput, tilingData);
        } else if (dtype == "e5m2e5m2") {
            ApaceQuantMatmulReduceScatterKernelE5M2E5M2_Udma<<<
                tilingData.tileQbmmTilingData.usedCoreNum, nullptr, stream>>>(
                    devContext, deviceA, deviceScaleA, deviceB, deviceScaleB, deviceOutput, tilingData);
        } else if (dtype == "e4m3e5m2") {
            ApaceQuantMatmulReduceScatterKernelE4M3E5M2_Udma<<<
                tilingData.tileQbmmTilingData.usedCoreNum, nullptr, stream>>>(
                    devContext, deviceA, deviceScaleA, deviceB, deviceScaleB, deviceOutput, tilingData);
        } else if (dtype == "e5m2e4m3") {
            ApaceQuantMatmulReduceScatterKernelE5M2E4M3_Udma<<<
                tilingData.tileQbmmTilingData.usedCoreNum, nullptr, stream>>>(
                    devContext, deviceA, deviceScaleA, deviceB, deviceScaleB, deviceOutput, tilingData);
        }

        ACL_CHECK(aclrtSynchronizeStream(stream));

        ACL_CHECK(aclrtMemcpy(hostOutput.data(), sizeOutput, deviceOutput, sizeOutput, ACL_MEMCPY_DEVICE_TO_HOST));

        WriteFile(outputDir + "/npu_out.bin", hostOutput.data(), sizeOutput);
    } else {
        constexpr int64_t HEAVY_BLOCK_NUM = 56;
        constexpr int64_t BOOST_ELEM_COUNT = 10000LL * 10000LL;
        size_t boostSize = static_cast<size_t>(BOOST_ELEM_COUNT) * sizeof(uint16_t);
        GM_ADDR boostIn = nullptr;
        GM_ADDR boostOut = nullptr;
        ACL_CHECK(aclrtMalloc((void**)&boostIn, boostSize, ACL_MEM_MALLOC_HUGE_ONLY));
        ACL_CHECK(aclrtMalloc((void**)&boostOut, boostSize, ACL_MEM_MALLOC_HUGE_ONLY));
        std::vector<uint16_t> boostHost(BOOST_ELEM_COUNT, 0x3F00);
        ACL_CHECK(aclrtMemcpy(boostIn, boostSize, boostHost.data(), boostSize, ACL_MEMCPY_HOST_TO_DEVICE));

        // L2 cache flush（MC2 性能采集红线③）
        constexpr int64_t CACHE_FLUSH_ELEM_COUNT = 128LL * 1024 * 1024;
        size_t cacheFlushSize = static_cast<size_t>(CACHE_FLUSH_ELEM_COUNT) * sizeof(uint16_t);
        GM_ADDR cacheFlush = nullptr;
        ACL_CHECK(aclrtMalloc((void**)&cacheFlush, cacheFlushSize, ACL_MEM_MALLOC_HUGE_ONLY));
        std::vector<uint16_t> cacheFlushHost(CACHE_FLUSH_ELEM_COUNT, 0x0000);
        ACL_CHECK(aclrtMemcpy(
            cacheFlush, cacheFlushSize, cacheFlushHost.data(), cacheFlushSize, ACL_MEMCPY_HOST_TO_DEVICE));

        constexpr int PERF_LOOP_COUNT = 10;
        int64_t cacheFlushBlockLen = CACHE_FLUSH_ELEM_COUNT / HEAVY_BLOCK_NUM;
        std::vector<double> durations(PERF_LOOP_COUNT);
        for (int i = 0; i < PERF_LOOP_COUNT; ++i) {
            // 1. 刷 L2 cache（MC2 性能采集红线③）：256 MB bf16 扫描，挤出 B 矩阵驻留
            heavy_add_kernel<<<HEAVY_BLOCK_NUM, nullptr, stream>>>(
                cacheFlush, CACHE_FLUSH_ELEM_COUNT, cacheFlushBlockLen);
            ACL_CHECK(aclrtSynchronizeStream(stream));
            // 2. 跑主 kernel（msprof task-based 采集每个 kernel 独立记录）
            auto t0 = std::chrono::steady_clock::now();
            // 根据 dtype 选择对应的 kernel 入口
            if (dtype == "e4m3e4m3") {
                ApaceQuantMatmulReduceScatterKernelE4M3E4M3_Udma<<<
                    tilingData.tileQbmmTilingData.usedCoreNum, nullptr, stream>>>(
                        devContext, deviceA, deviceScaleA, deviceB, deviceScaleB, deviceOutput, tilingData);
            } else if (dtype == "e5m2e5m2") {
                ApaceQuantMatmulReduceScatterKernelE5M2E5M2_Udma<<<
                    tilingData.tileQbmmTilingData.usedCoreNum, nullptr, stream>>>(
                        devContext, deviceA, deviceScaleA, deviceB, deviceScaleB, deviceOutput, tilingData);
            } else if (dtype == "e4m3e5m2") {
                ApaceQuantMatmulReduceScatterKernelE4M3E5M2_Udma<<<
                    tilingData.tileQbmmTilingData.usedCoreNum, nullptr, stream>>>(
                        devContext, deviceA, deviceScaleA, deviceB, deviceScaleB, deviceOutput, tilingData);
            } else if (dtype == "e5m2e4m3") {
                ApaceQuantMatmulReduceScatterKernelE5M2E4M3_Udma<<<
                    tilingData.tileQbmmTilingData.usedCoreNum, nullptr, stream>>>(
                        devContext, deviceA, deviceScaleA, deviceB, deviceScaleB, deviceOutput, tilingData);
            }
            ACL_CHECK(aclrtSynchronizeStream(stream));
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
        INFO_LOG("[Rank %d] mLocal=%u headMSize=%u tileCnt=%u | per-iter ms: avg=%.3f min=%.3f max=%.3f (excl warmup)",
                 rankId, mLocal, headMSize, tileCnt, avg, minv, maxv);

        ACL_CHECK(aclrtSynchronizeStream(stream));
        ACL_CHECK(aclrtMemcpy(hostOutput.data(), sizeOutput, deviceOutput, sizeOutput, ACL_MEMCPY_DEVICE_TO_HOST));
        WriteFile(outputDir + "/npu_out.bin", hostOutput.data(), sizeOutput);

        aclrtFree(boostIn);
        aclrtFree(boostOut);
        aclrtFree(cacheFlush);
    }

    aclrtFree(deviceA);
    aclrtFree(deviceScaleA);
    aclrtFree(deviceB);
    aclrtFree(deviceScaleB);
    aclrtFree(deviceOutput);

    HcclCommDestroy(comm);

    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(deviceId));
    ACL_CHECK(aclFinalize());

    return 0;
}

int main(int argc, char* argv[]) {
    int m, k, n, rankNum, headMSize;
    std::string mode;
    std::string dtype;
    try {
        parseArguments(argc, argv, m, k, n, rankNum, mode, headMSize, dtype);
    } catch (const std::invalid_argument& e) {
        std::cerr << e.what() << std::endl;
        printUsage(argv[0]);
        return -1;
    }

    INFO_LOG("Master (PID=%d) will fork %d processes (mode=%s, headMSize=%d, dtype=%s)", getpid(), rankNum, mode.c_str(), headMSize, dtype.c_str());

    std::vector<pid_t> pids(rankNum);
    for (int rankId = 0; rankId < rankNum; ++rankId) {
        pid_t pid = fork();

        if (pid < 0) {
            ERROR_LOG("Fork failed for rank %d", rankId);
            exit(-1);
        } else if (pid == 0) {
            int ret = runReduceScatterMatmul(rankNum, rankId, m, k, n, mode, headMSize, dtype);
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
