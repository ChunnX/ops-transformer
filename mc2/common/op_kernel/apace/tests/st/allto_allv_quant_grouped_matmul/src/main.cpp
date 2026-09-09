/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
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
#include "apace/kernel/fusions/allto_allv_quant_grouped_matmul/allto_allv_quant_grouped_matmul_tiling_data.h"
#include "apace_a2av_gmm_kernel_launcher.h"
#include "apace/core/aiv_comm/collective_comm_context.h"
#include "tiling/platform/platform_ascendc.h"

#define HCCL_CHECK(status) \
    do { \
        HcclResult ret = (status); \
        if (ret != HCCL_SUCCESS) { \
            std::cerr << __FILE__ << ":" << __LINE__ << " HcclResult:" << ret << std::endl; \
        } \
    } while (0)

inline uint64_t CeilDivU64(uint64_t a, uint64_t b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

void printUsage(const std::string &programName)
{
    std::cerr << "Usage: " << programName << " m k n rankNum expertNum [trans_b] [dtype]" << std::endl;
    std::cerr << "Args: " << std::endl;
    std::cerr << "  m: token rows per rank (total rows of gmm_x on each rank)" << std::endl;
    std::cerr << "  k: hidden size H1 (must be multiple of 1024)" << std::endl;
    std::cerr << "  n: expert output dim N1" << std::endl;
    std::cerr << "  rankNum: number of ranks" << std::endl;
    std::cerr << "  expertNum: local expert num per rank E" << std::endl;
    std::cerr << "  trans_b: optional, 0 (default) weight [E,K,N] | 1 weight [E,N,K]" << std::endl;
    std::cerr << "  dtype: optional, e4m3_e4m3 (default) | e5m2_e5m2 | e4m3_e5m2 | e5m2_e4m3" << std::endl;
    std::cerr << "Example: " << programName << " 2048 4096 4096 4 8" << std::endl;
}

void parseArguments(int argc, char *argv[], int &m, int &k, int &n, int &rankNum, int &expertNum, int &transB,
                    std::string &dtype)
{
    if (argc >= 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
        printUsage(argv[0]);
        exit(1);
    }
    if (argc < 6) {
        throw std::invalid_argument("ERROR: Lacks Arguments");
    }
    try {
        m = std::stoi(argv[1]);
        k = std::stoi(argv[2]);
        n = std::stoi(argv[3]);
        rankNum = std::stoi(argv[4]);
        expertNum = std::stoi(argv[5]);
    } catch (const std::invalid_argument &) {
        throw std::invalid_argument("ERROR: m k n rankNum expertNum must be Integer");
    }

    transB = (argc >= 7) ? std::stoi(argv[6]) : 0;
    if (transB != 0 && transB != 1) {
        throw std::invalid_argument("ERROR: trans_b must be 0 or 1");
    }
    dtype = (argc >= 8) ? std::string(argv[7]) : std::string("e4m3_e4m3");

    if (m <= 0 || k <= 0 || n <= 0 || rankNum <= 0 || expertNum <= 0) {
        throw std::invalid_argument("ERROR: m k n rankNum expertNum must be positive");
    }
    if (k % 1024 != 0) {
        throw std::invalid_argument("ERROR: k must be multiple of 1024 (MX scale comm alignment)");
    }
    if (static_cast<uint32_t>(rankNum) > A2AV_GMM_MAX_RANK_NUM) {
        throw std::invalid_argument("ERROR: rankNum exceeds A2AV_GMM_MAX_RANK_NUM");
    }
    if (static_cast<uint32_t>(expertNum) > A2AV_GMM_MAX_EXPERT_NUM) {
        throw std::invalid_argument("ERROR: expertNum exceeds A2AV_GMM_MAX_EXPERT_NUM");
    }
}

void LaunchKernel(const std::string &dtype, int transB, uint32_t blockNum, __gm__ CommContext *ctx, GM_ADDR xGM,
                  GM_ADDR xScaleGM, GM_ADDR weightGM, GM_ADDR weightScaleGM, GM_ADDR yGM, GM_ADDR tilingGM,
                  aclrtStream stream)
{
    if (dtype == "e4m3_e4m3") {
        if (transB) {
            AllToAllvQuantGmmKernel_E4M3E4M3_Trans<<<blockNum, nullptr, stream>>>(ctx, xGM, xScaleGM, weightGM,
                                                                                  weightScaleGM, yGM, tilingGM);
        } else {
            AllToAllvQuantGmmKernel_E4M3E4M3_NonTrans<<<blockNum, nullptr, stream>>>(ctx, xGM, xScaleGM, weightGM,
                                                                                     weightScaleGM, yGM, tilingGM);
        }
    } else if (dtype == "e5m2_e5m2") {
        if (transB) {
            AllToAllvQuantGmmKernel_E5M2E5M2_Trans<<<blockNum, nullptr, stream>>>(ctx, xGM, xScaleGM, weightGM,
                                                                                  weightScaleGM, yGM, tilingGM);
        } else {
            AllToAllvQuantGmmKernel_E5M2E5M2_NonTrans<<<blockNum, nullptr, stream>>>(ctx, xGM, xScaleGM, weightGM,
                                                                                     weightScaleGM, yGM, tilingGM);
        }
    } else if (dtype == "e4m3_e5m2") {
        if (transB) {
            AllToAllvQuantGmmKernel_E4M3E5M2_Trans<<<blockNum, nullptr, stream>>>(ctx, xGM, xScaleGM, weightGM,
                                                                                  weightScaleGM, yGM, tilingGM);
        } else {
            AllToAllvQuantGmmKernel_E4M3E5M2_NonTrans<<<blockNum, nullptr, stream>>>(ctx, xGM, xScaleGM, weightGM,
                                                                                     weightScaleGM, yGM, tilingGM);
        }
    } else if (dtype == "e5m2_e4m3") {
        if (transB) {
            AllToAllvQuantGmmKernel_E5M2E4M3_Trans<<<blockNum, nullptr, stream>>>(ctx, xGM, xScaleGM, weightGM,
                                                                                  weightScaleGM, yGM, tilingGM);
        } else {
            AllToAllvQuantGmmKernel_E5M2E4M3_NonTrans<<<blockNum, nullptr, stream>>>(ctx, xGM, xScaleGM, weightGM,
                                                                                     weightScaleGM, yGM, tilingGM);
        }
    } else {
        throw std::invalid_argument("ERROR: unsupported dtype: " + dtype);
    }
}

int RunAllToAllvQuantGmm(int rankNum, int rankId, int m, int k, int n, int expertNum, int transB,
                         const std::string &dtype)
{
    const char *ipport = "tcp://127.0.0.1:8998";
    INFO_LOG("rankNum=%d, rankId=%d, m=%d, k=%d, n=%d, expertNum=%d, transB=%d, dtype=%s", rankNum, rankId, m, k, n,
             expertNum, transB, dtype.c_str());

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
    const char *ctxTag = "allto_allv_quant_grouped_matmul";
    CommContext *devContext = reinterpret_cast<CommContext *>(builder.CreateDeviceContext(
        &hostCtx, sizeof(CommContext), ctxTag, &hostCtx.udmaCtx, &hostCtx.ubmemCtx));
    if (devContext == nullptr) {
        ERROR_LOG("rank %d CreateDeviceContext failed", rankId);
        return -1;
    }

    exchanger.Barrier();
    exchanger.Close();

    // ---- 读取输入数据 ----
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

    const uint32_t expertTotal = static_cast<uint32_t>(expertNum);
    const uint32_t rankTotal = static_cast<uint32_t>(rankNum);
    const uint32_t scaleKLen = CeilDivU64(k, 64) * 2;
    // weight: 每专家 [K,N] 或 [N,K]；weightScale: 每专家 N * scaleKLen 字节
    const size_t weightBytes = static_cast<size_t>(expertTotal) * k * n;
    const size_t weightScaleBytes = static_cast<size_t>(expertTotal) * n * scaleKLen;
    const size_t xBytes = static_cast<size_t>(m) * k;
    const size_t xScaleBytes = static_cast<size_t>(m) * scaleKLen;
    // 计数矩阵 [rankNum][rankNum * expertNum] uint32
    const size_t countsBytes = static_cast<size_t>(rankTotal) * rankTotal * expertTotal * sizeof(uint32_t);

    std::vector<uint8_t> hostX(xBytes, 0);
    std::vector<uint8_t> hostXScale(xScaleBytes, 0);
    std::vector<uint8_t> hostWeight(weightBytes, 0);
    std::vector<uint8_t> hostWeightScale(weightScaleBytes, 0);
    std::vector<uint32_t> hostCounts(rankTotal * rankTotal * expertTotal, 0);

    ReadFile(inputDir + "/input_x.bin", hostX.data(), xBytes);
    ReadFile(inputDir + "/input_x_scale.bin", hostXScale.data(), xScaleBytes);
    ReadFile(inputDir + "/input_weight.bin", hostWeight.data(), weightBytes);
    ReadFile(inputDir + "/input_weight_scale.bin", hostWeightScale.data(), weightScaleBytes);
    ReadFile(inputDir + "/counts.bin", hostCounts.data(), countsBytes);

    // ---- 构建 tiling ----
    platform_ascendc::PlatformAscendC *ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    const uint32_t aicNum = static_cast<uint32_t>(ascendcPlatform->GetCoreNumAic());
    AllToAllvQuantGmmTilingData tiling;
    BuildAllToAllvQuantGmmTiling(static_cast<uint32_t>(rankId), rankTotal, expertTotal, static_cast<uint32_t>(k),
                                 static_cast<uint32_t>(n), transB != 0, hostCounts.data(), aicNum, tiling);

    // ---- 窗口容量校验 ----
    {
        void *hcclBuf = nullptr;
        uint64_t hcclBufSize = 0;
        if (HcclGetHcclBuffer(comm, &hcclBuf, &hcclBufSize) != HCCL_SUCCESS || hcclBuf == nullptr) {
            ERROR_LOG("rank %d HcclGetHcclBuffer failed", rankId);
            return -1;
        }
        const uint64_t needBytes =
            static_cast<uint64_t>(tiling.totalRecvRows) * (static_cast<uint64_t>(k) + scaleKLen);
        INFO_LOG("rank %d window need=%llu bytes, hccl buffer=%llu bytes, totalRecvRows=%u", rankId,
                 static_cast<unsigned long long>(needBytes), static_cast<unsigned long long>(hcclBufSize),
                 tiling.totalRecvRows);
        if (needBytes > hcclBufSize) {
            ERROR_LOG("rank %d window overflow: need %llu > hccl buffer %llu", rankId,
                      static_cast<unsigned long long>(needBytes), static_cast<unsigned long long>(hcclBufSize));
            return -1;
        }
    }

    INFO_LOG("rank %d tiling: baseM=%u baseN=%u baseK=%u kAL1=%u usedCoreNum=%u groupList[0]=%u", rankId, tiling.baseM,
             tiling.baseN, tiling.baseK, tiling.kAL1, tiling.usedCoreNum, tiling.groupListCum[0]);

    // ---- 设备内存与数据下发 ----
    GM_ADDR devX = nullptr;
    GM_ADDR devXScale = nullptr;
    GM_ADDR devWeight = nullptr;
    GM_ADDR devWeightScale = nullptr;
    GM_ADDR devY = nullptr;
    GM_ADDR devTiling = nullptr;
    const size_t yBytes = static_cast<size_t>(tiling.totalRecvRows) * n * sizeof(uint16_t);
    const size_t yAllocBytes = std::max<size_t>(yBytes, 32); // 避免 0 字节分配

    ACL_CHECK(aclrtMalloc((void **)&devX, xBytes, ACL_MEM_MALLOC_HUGE_ONLY));
    ACL_CHECK(aclrtMalloc((void **)&devXScale, xScaleBytes, ACL_MEM_MALLOC_HUGE_ONLY));
    ACL_CHECK(aclrtMalloc((void **)&devWeight, weightBytes, ACL_MEM_MALLOC_HUGE_ONLY));
    ACL_CHECK(aclrtMalloc((void **)&devWeightScale, weightScaleBytes, ACL_MEM_MALLOC_HUGE_ONLY));
    ACL_CHECK(aclrtMalloc((void **)&devY, yAllocBytes, ACL_MEM_MALLOC_HUGE_ONLY));
    ACL_CHECK(aclrtMalloc((void **)&devTiling, sizeof(AllToAllvQuantGmmTilingData), ACL_MEM_MALLOC_HUGE_ONLY));

    ACL_CHECK(aclrtMemcpy(devX, xBytes, hostX.data(), xBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(devXScale, xScaleBytes, hostXScale.data(), xScaleBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(devWeight, weightBytes, hostWeight.data(), weightBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(devWeightScale, weightScaleBytes, hostWeightScale.data(), weightScaleBytes,
                          ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(devTiling, sizeof(AllToAllvQuantGmmTilingData), &tiling,
                          sizeof(AllToAllvQuantGmmTilingData), ACL_MEMCPY_HOST_TO_DEVICE));

    LaunchKernel(dtype, transB, tiling.usedCoreNum, devContext, devX, devXScale, devWeight, devWeightScale, devY,
                 devTiling, stream);
    ACL_CHECK(aclrtSynchronizeStream(stream));

    std::vector<uint16_t> hostY(yBytes / sizeof(uint16_t), 0);
    ACL_CHECK(aclrtMemcpy(hostY.data(), yBytes, devY, yBytes, ACL_MEMCPY_DEVICE_TO_HOST));
    WriteFile(outputDir + "/npu_out.bin", hostY.data(), yBytes);

    aclrtFree(devX);
    aclrtFree(devXScale);
    aclrtFree(devWeight);
    aclrtFree(devWeightScale);
    aclrtFree(devY);
    aclrtFree(devTiling);

    HcclCommDestroy(comm);
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(deviceId));
    ACL_CHECK(aclFinalize());
    return 0;
}

int main(int argc, char *argv[])
{
    int m, k, n, rankNum, expertNum, transB;
    std::string dtype;
    try {
        parseArguments(argc, argv, m, k, n, rankNum, expertNum, transB, dtype);
    } catch (const std::invalid_argument &e) {
        std::cerr << e.what() << std::endl;
        printUsage(argv[0]);
        return -1;
    }

    INFO_LOG("Master (PID=%d) will fork %d processes", getpid(), rankNum);
    std::vector<pid_t> pids(rankNum);
    for (int rankId = 0; rankId < rankNum; ++rankId) {
        pid_t pid = fork();
        if (pid < 0) {
            ERROR_LOG("Fork failed for rank %d", rankId);
            exit(-1);
        } else if (pid == 0) {
            int ret = RunAllToAllvQuantGmm(rankNum, rankId, m, k, n, expertNum, transB, dtype);
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
