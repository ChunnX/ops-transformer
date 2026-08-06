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
 * \file test_aclnn_allto_all_matmul_apace.cpp
 * \brief aclnn fork-model test: AlltoAllMatmulApace (MX-quant FP8, apace UDMA)
 *
 * Uses the same fork + RootInfoExchanger + CommChannelBuilder pattern as the
 * kernel direct-invoke ST test, but calls the aclnn two-phase API instead of
 * the kernel directly.
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

#include <acl/acl.h>
#include <hccl/hccl.h>
#include <hccl/hccl_types.h>
#include <hccl/hccl_comm.h>
#include <hccl/hccl_res.h>
#include <hccl/hccl_rank_graph.h>
#include <cstdlib>

#include "utils.h"
#include "../../common/op_kernel/apace/utils/comm_channel_builder.h"
#include "../../common/op_kernel/apace/tests/st/utils/root_info_exchanger.h"
#include "apace/kernel/all_to_all_quant_matmul/all_to_all_matmul_tiling_data.h"
#include "aclnnop/aclnn_allto_all_matmul_apace.h"

#define HCCL_CHECK(status) \
    do { \
        HcclResult ret_ = (status); \
        if (ret_ != HCCL_SUCCESS) { \
            std::cerr << __FILE__ << ":" << __LINE__ << " HcclResult:" << ret_ << std::endl; \
        } \
    } while (0)

#define INFO_LOG(fmt, args...) \
    do { \
        fprintf(stdout, "[INFO] " fmt "\n", ##args); \
        fflush(stdout); \
    } while (0)
#define ERROR_LOG(fmt, args...) \
    do { \
        fprintf(stderr, "[ERROR] " fmt "\n", ##args); \
        fflush(stderr); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t s = 1;
    for (auto i : shape)
        s *= i;
    return s;
}

static int CreateAclTensorFromDevice(void *deviceAddr, const std::vector<int64_t> &shape, aclDataType dataType,
                                     aclTensor **tensor)
{
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), (int64_t)shape.size(), dataType, strides.data(), 0,
                              aclFormat::ACL_FORMAT_ND, shape.data(), (int64_t)shape.size(), deviceAddr);
    return (*tensor != nullptr) ? 0 : -1;
}

static bool ReadBinary(const std::string &path, void *buf, size_t sz)
{
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) {
        ERROR_LOG("open %s failed", path.c_str());
        return false;
    }
    size_t rd = fread(buf, 1, sz, f);
    fclose(f);
    return (rd == sz);
}

static bool WriteBinary(const std::string &path, const void *buf, size_t sz)
{
    FILE *f = fopen(path.c_str(), "wb");
    if (!f)
        return false;
    fwrite(buf, 1, sz, f);
    fclose(f);
    return true;
}

static void printUsage(const char *prog)
{
    std::cerr << "Usage: " << prog << " m k n rankNum [baseDir]" << std::endl;
    std::cerr << "  m: per-rank M (total M = m * rankNum)" << std::endl;
    std::cerr << "  k: total K (per-rank Ka = k / rankNum)" << std::endl;
    std::cerr << "  n: N" << std::endl;
    std::cerr << "  rankNum: number of ranks" << std::endl;
}

int runOneRank(int rankNum, int rankId, int mPerRank, int kTotal, int n, const std::string &baseDir)
{
    const char *ipport = "tcp://127.0.0.1:8998";
    INFO_LOG("rank=%d/%d mPerRank=%d kTotal=%d n=%d", rankId, rankNum, mPerRank, kTotal, n);

    int64_t worldSize = rankNum;
    int64_t ka = kTotal / worldSize;
    int64_t mTotal = mPerRank * worldSize;

    // ---- ACL init ----
    aclInit(nullptr);
    aclrtSetDevice(rankId);
    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);

    // ---- HCCL init via TCP root info exchange (same as ST test) ----
    RootInfoExchanger exchanger(static_cast<uint32_t>(rankId), static_cast<uint32_t>(rankNum), ipport);
    HcclRootInfo rootInfo;
    if (!exchanger.Exchange(rootInfo)) {
        ERROR_LOG("rank %d exchange rootInfo failed", rankId);
        return -1;
    }
    HcclComm comm = nullptr;
    HcclCommConfig config;
    HcclCommConfigInit(&config);
    config.hcclWorldRankID = static_cast<uint32_t>(rankId);
    HCCL_CHECK(HcclCommInitRootInfoConfig(static_cast<uint32_t>(rankNum), &rootInfo, static_cast<uint32_t>(rankId),
                                          &config, &comm));
    if (comm == nullptr) {
        ERROR_LOG("rank %d HcclCommInitRootInfoConfig failed", rankId);
        return -1;
    }
    char hcomName[128] = {0};
    HcclGetCommName(comm, hcomName);
    INFO_LOG("rank %d hcom: %s", rankId, hcomName);

    // ---- Create CommContext via CommChannelBuilder (same as ST test) ----
    CommChannelBuilder builder(comm);
    CommContext hostCtx = {};
    const char *ctxTag = "allto_all_matmul_apace";
    void *devCtx =
        builder.CreateDeviceContext(&hostCtx, sizeof(CommContext), ctxTag, &hostCtx.udmaCtx, &hostCtx.ubmemCtx);
    if (devCtx == nullptr) {
        ERROR_LOG("rank %d CreateDeviceContext failed", rankId);
        return -1;
    }
    INFO_LOG("rank %d devCtx=%p rankId=%u rankSize=%u", rankId, devCtx, hostCtx.udmaCtx.rankId,
             hostCtx.udmaCtx.rankSize);

    exchanger.Barrier();
    exchanger.Close();

    // ---- Build context tensor with real CommContext data ----
    constexpr size_t kCtxSize = sizeof(CommContext);
    void *ctxDev = nullptr;
    aclrtMalloc(&ctxDev, kCtxSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(ctxDev, kCtxSize, &hostCtx, kCtxSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclTensor *ctxTensor = nullptr;
    std::vector<int64_t> ctxShape = {static_cast<int64_t>(kCtxSize)};
    CreateAclTensorFromDevice(ctxDev, ctxShape, ACL_UINT8, &ctxTensor);

    // ---- Read input data ----
    std::vector<int64_t> x1Shape = {mTotal, ka};
    std::vector<int64_t> x2Shape = {static_cast<int64_t>(n), kTotal};
    std::vector<int64_t> s1Shape = {mTotal, (ka + 63) / 64, 2};
    std::vector<int64_t> s2Shape = {static_cast<int64_t>(n), (kTotal + 63) / 64, 2};
    std::vector<int64_t> outShape = {mPerRank, static_cast<int64_t>(n)};

    long long nX1 = GetShapeSize(x1Shape), nX2 = GetShapeSize(x2Shape);
    long long nS1 = GetShapeSize(s1Shape), nS2 = GetShapeSize(s2Shape);
    long long nOut = GetShapeSize(outShape);

    std::vector<uint8_t> hX1(nX1), hX2(nX2), hS1(nS1), hS2(nS2);
    std::vector<uint16_t> hOut(nOut, 0);

    std::string inDir = baseDir + "/input/" + std::to_string(rankId);
    if (!ReadBinary(inDir + "/input_a.bin", hX1.data(), nX1) || !ReadBinary(inDir + "/input_b.bin", hX2.data(), nX2) ||
        !ReadBinary(inDir + "/input_scaleA.bin", hS1.data(), nS1) ||
        !ReadBinary(inDir + "/input_scaleB.bin", hS2.data(), nS2)) {
        return -1;
    }

    // ---- Allocate device tensors ----
    void *dX1 = nullptr, *dX2 = nullptr, *dS1 = nullptr, *dS2 = nullptr, *dOut = nullptr;
    aclrtMalloc(&dX1, nX1, ACL_MEM_MALLOC_HUGE_ONLY);
    aclrtMalloc(&dX2, nX2, ACL_MEM_MALLOC_HUGE_ONLY);
    aclrtMalloc(&dS1, nS1, ACL_MEM_MALLOC_HUGE_ONLY);
    aclrtMalloc(&dS2, nS2, ACL_MEM_MALLOC_HUGE_ONLY);
    aclrtMalloc(&dOut, nOut * sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_ONLY);

    aclrtMemcpy(dX1, nX1, hX1.data(), nX1, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(dX2, nX2, hX2.data(), nX2, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(dS1, nS1, hS1.data(), nS1, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(dS2, nS2, hS2.data(), nS2, ACL_MEMCPY_HOST_TO_DEVICE);

    aclTensor *tX1 = nullptr, *tX2 = nullptr, *tS1 = nullptr, *tS2 = nullptr, *tOut = nullptr;
    CreateAclTensorFromDevice(dX1, x1Shape, ACL_FLOAT8_E4M3FN, &tX1);
    CreateAclTensorFromDevice(dX2, x2Shape, ACL_FLOAT8_E4M3FN, &tX2);
    CreateAclTensorFromDevice(dS1, s1Shape, ACL_FLOAT8_E8M0, &tS1);
    CreateAclTensorFromDevice(dS2, s2Shape, ACL_FLOAT8_E8M0, &tS2);
    CreateAclTensorFromDevice(dOut, outShape, ACL_BF16, &tOut);

    // ---- aclnn two-phase API ----
    const char *commMode = "";
    uint64_t wsSize = 0;
    aclOpExecutor *exec = nullptr;
    aclnnStatus ret = aclnnAlltoAllMatmulApaceGetWorkspaceSize(
        ctxTensor, tX1, tX2, nullptr, tS1, tS2, nullptr, nullptr, nullptr, "", worldSize, 0, nullptr, 6, 6, 0, 28, 28,
        false, true, 0, commMode, 0, tOut, nullptr, &wsSize, &exec);
    if (ret != 0) {
        ERROR_LOG("rank %d GetWorkspaceSize failed ret=%d", rankId, (int)ret);
        return -1;
    }
    INFO_LOG("rank %d workspace=%lu", rankId, wsSize);

    void *ws = nullptr;
    if (wsSize > 0) {
        aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }

    ret = aclnnAlltoAllMatmulApace(ws, wsSize, exec, stream);
    if (ret != 0) {
        ERROR_LOG("rank %d Execute failed ret=%d", rankId, (int)ret);
        return -1;
    }

    aclrtSynchronizeStream(stream);
    INFO_LOG("rank %d execute success", rankId);

    // ---- Input validation tests (rank 0 only, to avoid duplicating logs) ----
    if (rankId == 0) {
#define CHECK_FAIL(name, call) \
    do { \
        uint64_t _ws = 0; \
        aclOpExecutor *_ex = nullptr; \
        aclnnStatus _r = call; \
        INFO_LOG("  [%s] ret=%d %s", name, (int)_r, (_r != 0) ? "PASS" : "FAIL"); \
    } while (0)

        INFO_LOG("--- validation: nullptr checks ---");
        CHECK_FAIL("x1_null", aclnnAlltoAllMatmulApaceGetWorkspaceSize(
                                  ctxTensor, nullptr, tX2, nullptr, tS1, tS2, nullptr, nullptr, nullptr, "", worldSize,
                                  0, nullptr, 6, 6, 0, 28, 28, false, true, 0, commMode, 0, tOut, nullptr, &_ws, &_ex));
        CHECK_FAIL("x2_null", aclnnAlltoAllMatmulApaceGetWorkspaceSize(
                                  ctxTensor, tX1, nullptr, nullptr, tS1, tS2, nullptr, nullptr, nullptr, "", worldSize,
                                  0, nullptr, 6, 6, 0, 28, 28, false, true, 0, commMode, 0, tOut, nullptr, &_ws, &_ex));
        CHECK_FAIL("s1_null", aclnnAlltoAllMatmulApaceGetWorkspaceSize(
                                  ctxTensor, tX1, tX2, nullptr, nullptr, tS2, nullptr, nullptr, nullptr, "", worldSize,
                                  0, nullptr, 6, 6, 0, 28, 28, false, true, 0, commMode, 0, tOut, nullptr, &_ws, &_ex));
        CHECK_FAIL("s2_null", aclnnAlltoAllMatmulApaceGetWorkspaceSize(
                                  ctxTensor, tX1, tX2, nullptr, tS1, nullptr, nullptr, nullptr, nullptr, "", worldSize,
                                  0, nullptr, 6, 6, 0, 28, 28, false, true, 0, commMode, 0, tOut, nullptr, &_ws, &_ex));
        CHECK_FAIL("out_null",
                   aclnnAlltoAllMatmulApaceGetWorkspaceSize(ctxTensor, tX1, tX2, nullptr, tS1, tS2, nullptr, nullptr,
                                                            nullptr, "", worldSize, 0, nullptr, 6, 6, 0, 28, 28, false,
                                                            true, 0, commMode, 0, nullptr, nullptr, &_ws, &_ex));

        INFO_LOG("--- validation: dtype checks ---");
        {
            void *dX1B = nullptr, *dOutB = nullptr;
            aclTensor *tX1Bad = nullptr, *tOutBad = nullptr;
            aclrtMalloc(&dX1B, nX1, ACL_MEM_MALLOC_HUGE_FIRST);
            aclrtMalloc(&dOutB, nOut * 2, ACL_MEM_MALLOC_HUGE_FIRST);
            CreateAclTensorFromDevice(dX1B, x1Shape, ACL_FLOAT, &tX1Bad);
            CreateAclTensorFromDevice(dOutB, outShape, ACL_FLOAT, &tOutBad);
            CHECK_FAIL("x1_not_fp8",
                       aclnnAlltoAllMatmulApaceGetWorkspaceSize(
                           ctxTensor, tX1Bad, tX2, nullptr, tS1, tS2, nullptr, nullptr, nullptr, "", worldSize, 0,
                           nullptr, 6, 6, 0, 28, 28, false, true, 0, commMode, 0, tOut, nullptr, &_ws, &_ex));
            CHECK_FAIL("y_not_bf16",
                       aclnnAlltoAllMatmulApaceGetWorkspaceSize(
                           ctxTensor, tX1, tX2, nullptr, tS1, tS2, nullptr, nullptr, nullptr, "", worldSize, 0, nullptr,
                           6, 6, 0, 28, 28, false, true, 0, commMode, 0, tOutBad, nullptr, &_ws, &_ex));
            aclDestroyTensor(tX1Bad);
            aclDestroyTensor(tOutBad);
            aclrtFree(dX1B);
            aclrtFree(dOutB);
        }

        INFO_LOG("--- validation: transpose_x2 must be true ---");
        CHECK_FAIL("transx2_false",
                   aclnnAlltoAllMatmulApaceGetWorkspaceSize(ctxTensor, tX1, tX2, nullptr, tS1, tS2, nullptr, nullptr,
                                                            nullptr, "", worldSize, 0, nullptr, 6, 6, 0, 28, 28, false,
                                                            false, 0, commMode, 0, tOut, nullptr, &_ws, &_ex));

        INFO_LOG("--- validation: K dim not divisible by 64 ---");
        {
            void *dX1K = nullptr;
            aclTensor *tX1K = nullptr;
            std::vector<int64_t> x1kShape = {mTotal, ka + 1}; // K not aligned to 64
            long long nX1K = GetShapeSize(x1kShape);
            aclrtMalloc(&dX1K, nX1K, ACL_MEM_MALLOC_HUGE_FIRST);
            CreateAclTensorFromDevice(dX1K, x1kShape, ACL_FLOAT8_E4M3FN, &tX1K);
            CHECK_FAIL("k_not_div64",
                       aclnnAlltoAllMatmulApaceGetWorkspaceSize(
                           ctxTensor, tX1K, tX2, nullptr, tS1, tS2, nullptr, nullptr, nullptr, "", worldSize, 0,
                           nullptr, 6, 6, 0, 28, 28, false, true, 0, commMode, 0, tOut, nullptr, &_ws, &_ex));
            aclDestroyTensor(tX1K);
            aclrtFree(dX1K);
        }

        INFO_LOG("--- validation: scale shape mismatch ---");
        {
            void *dS1B = nullptr;
            aclTensor *tS1Bad = nullptr;
            std::vector<int64_t> s1BadShape = {(int64_t)mTotal, (ka + 63) / 64 + 1, 2L}; // wrong K dim
            long long nS1B = GetShapeSize(s1BadShape);
            aclrtMalloc(&dS1B, nS1B, ACL_MEM_MALLOC_HUGE_FIRST);
            CreateAclTensorFromDevice(dS1B, s1BadShape, ACL_FLOAT8_E8M0, &tS1Bad);
            CHECK_FAIL("s1_shape_mismatch",
                       aclnnAlltoAllMatmulApaceGetWorkspaceSize(
                           ctxTensor, tX1, tX2, nullptr, tS1Bad, tS2, nullptr, nullptr, nullptr, "", worldSize, 0,
                           nullptr, 6, 6, 0, 28, 28, false, true, 0, commMode, 0, tOut, nullptr, &_ws, &_ex));
            aclDestroyTensor(tS1Bad);
            aclrtFree(dS1B);
        }

        INFO_LOG("--- validation: scale must be 3D ---");
        {
            void *dS2B = nullptr;
            aclTensor *tS2Bad = nullptr;
            std::vector<int64_t> s2BadShape = {(int64_t)n, (kTotal + 63) / 64}; // 2D instead of 3D
            long long nS2B = GetShapeSize(s2BadShape);
            aclrtMalloc(&dS2B, nS2B, ACL_MEM_MALLOC_HUGE_FIRST);
            CreateAclTensorFromDevice(dS2B, s2BadShape, ACL_FLOAT8_E8M0, &tS2Bad);
            CHECK_FAIL("s2_not_3d",
                       aclnnAlltoAllMatmulApaceGetWorkspaceSize(
                           ctxTensor, tX1, tX2, nullptr, tS1, tS2Bad, nullptr, nullptr, nullptr, "", worldSize, 0,
                           nullptr, 6, 6, 0, 28, 28, false, true, 0, commMode, 0, tOut, nullptr, &_ws, &_ex));
            aclDestroyTensor(tS2Bad);
            aclrtFree(dS2B);
        }

        INFO_LOG("--- validation: scale dtype not E8M0 ---");
        {
            void *dS1D = nullptr;
            aclTensor *tS1D = nullptr;
            aclrtMalloc(&dS1D, nS1, ACL_MEM_MALLOC_HUGE_FIRST);
            CreateAclTensorFromDevice(dS1D, s1Shape, ACL_FLOAT, &tS1D);
            CHECK_FAIL("s1_not_e8m0",
                       aclnnAlltoAllMatmulApaceGetWorkspaceSize(
                           ctxTensor, tX1, tX2, nullptr, tS1D, tS2, nullptr, nullptr, nullptr, "", worldSize, 0,
                           nullptr, 6, 6, 0, 28, 28, false, true, 0, commMode, 0, tOut, nullptr, &_ws, &_ex));
            aclDestroyTensor(tS1D);
            aclrtFree(dS1D);
        }
        INFO_LOG("--- validation: Torch .t() non-contiguous x2 ---");
        {
            // Simulate PyTorch .t(): x2 is [K,N] storage, [N,K] view with strides [1,N]
            std::vector<int64_t> viewShapeT = {static_cast<int64_t>(n), kTotal};
            std::vector<int64_t> storageShapeT = {kTotal, static_cast<int64_t>(n)};
            std::vector<int64_t> stridesT = {1, static_cast<int64_t>(n)};
            aclTensor *tX2Trans =
                aclCreateTensor(viewShapeT.data(), viewShapeT.size(), ACL_FLOAT8_E4M3FN, stridesT.data(), 0,
                                ACL_FORMAT_ND, storageShapeT.data(), storageShapeT.size(), dX2);
            // transposed x2 + transposeX2=true should succeed
            uint64_t _ws2 = 0;
            aclOpExecutor *_ex2 = nullptr;
            aclnnStatus _r2 = aclnnAlltoAllMatmulApaceGetWorkspaceSize(
                ctxTensor, tX1, tX2Trans, nullptr, tS1, tS2, nullptr, nullptr, nullptr, "", worldSize, 0, nullptr, 6, 6,
                0, 28, 28, false, true, 0, commMode, 0, tOut, nullptr, &_ws2, &_ex2);
            INFO_LOG("  [x2_transposed] ret=%d %s", (int)_r2, (_r2 == 0) ? "PASS" : "FAIL");
            aclDestroyTensor(tX2Trans);

            // transposed x2 + transposeX2=false → should auto-detect and succeed
            aclTensor *tX2Trans2 =
                aclCreateTensor(viewShapeT.data(), viewShapeT.size(), ACL_FLOAT8_E4M3FN, stridesT.data(), 0,
                                ACL_FORMAT_ND, storageShapeT.data(), storageShapeT.size(), dX2);
            uint64_t _ws3 = 0;
            aclOpExecutor *_ex3 = nullptr;
            aclnnStatus _r3 = aclnnAlltoAllMatmulApaceGetWorkspaceSize(
                ctxTensor, tX1, tX2Trans2, nullptr, tS1, tS2, nullptr, nullptr, nullptr, "", worldSize, 0, nullptr, 6,
                6, 0, 28, 28, false, false, 0, commMode, 0, tOut, nullptr, &_ws3, &_ex3);
            INFO_LOG("  [x2_transp_autofix] ret=%d %s", (int)_r3, (_r3 == 0) ? "PASS" : "FAIL");
            aclDestroyTensor(tX2Trans2);
        }
#undef CHECK_FAIL
    }

    // ---- Read back output ----
    aclrtMemcpy(hOut.data(), nOut * sizeof(uint16_t), dOut, nOut * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST);

    std::string outDir = baseDir + "/output/" + std::to_string(rankId);
    mkdir(outDir.c_str(), 0755);
    WriteBinary(outDir + "/npu_out.bin", hOut.data(), nOut * sizeof(uint16_t));

    // ---- Cleanup ----
    aclDestroyTensor(ctxTensor);
    aclDestroyTensor(tX1);
    aclDestroyTensor(tX2);
    aclDestroyTensor(tS1);
    aclDestroyTensor(tS2);
    aclDestroyTensor(tOut);
    aclrtFree(ctxDev);
    aclrtFree(dX1);
    aclrtFree(dX2);
    aclrtFree(dS1);
    aclrtFree(dS2);
    aclrtFree(dOut);
    if (wsSize > 0)
        aclrtFree(ws);
    HcclCommDestroy(comm);
    aclrtDestroyStream(stream);
    aclrtResetDevice(rankId);
    aclFinalize();
    return 0;
}

int main(int argc, char *argv[])
{
    if (argc < 5) {
        printUsage(argv[0]);
        return -1;
    }
    int mPerRank = std::stoi(argv[1]);
    int kTotal = std::stoi(argv[2]);
    int n = std::stoi(argv[3]);
    int rankNum = std::stoi(argv[4]);
    std::string baseDir = (argc >= 6) ? argv[5] : ".";

    INFO_LOG("Master PID=%d forking %d ranks", getpid(), rankNum);

    std::vector<pid_t> pids(rankNum);
    for (int r = 0; r < rankNum; ++r) {
        pid_t pid = fork();
        if (pid < 0) {
            ERROR_LOG("fork failed");
            return -1;
        }
        if (pid == 0) {
            int rc = runOneRank(rankNum, r, mPerRank, kTotal, n, baseDir);
            _exit(rc);
        }
        pids[r] = pid;
    }

    bool ok = true;
    for (int r = 0; r < rankNum; ++r) {
        int st;
        waitpid(pids[r], &st, 0);
        if (!WIFEXITED(st) || WEXITSTATUS(st) != 0) {
            ok = false;
            ERROR_LOG("rank %d failed", r);
        }
    }
    std::cout << (ok ? "SUCCESS" : "FAILURE") << std::endl;
    return ok ? 0 : -1;
}
