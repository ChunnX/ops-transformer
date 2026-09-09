/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*
 * AlltoAllvQuantGroupedMatMul MX 量化（quantMode=6）E2E 验证示例。
 *
 * 条件满足时（无共享专家/无 permute_out/E4M3/E5M2/FP4）算子在 tiling 阶段自动切换
 * apace 实现（hcomm 纯 C 核通信 + 段式分组量化 Matmul），本示例验证该路径端到端精度：
 *   - 双 rank（单进程多线程），HcclCommInitAll 建链；
 *   - 非均匀 send/recv counts（确定性全局计数矩阵，各线程一致推导）；
 *   - 输入数据由确定性函数生成，golden 可在任意线程重建任意 rank 的输入；
 *   - host 侧按 expert-major 语义计算 MX 反量化 golden 并抽样比对。
 */

#include <thread>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <memory>
#include <vector>
#include "acl/acl.h"
#include "hccl/hccl.h"
#include "aclnnop/aclnn_allto_allv_quant_grouped_mat_mul.h"

#define CHECK_RET(cond, return_expr) \
    do { \
        if (!(cond)) { \
            return_expr; \
        } \
    } while (0)

#define LOG_PRINT(message, ...) \
    do { \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc failed. ret: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMemcpy failed. ret: %d\n", ret); return ret);
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

struct Args {
    int rankId;
    HcclComm hcclComm;
    aclrtStream stream;
    aclrtContext context;
};

// shape 基本信息
constexpr int64_t EP_WORLD_SIZE = 2;
constexpr int64_t BSK = 1024; // 单 rank token 行数
constexpr int64_t H = 2048;
constexpr int64_t E = 4; // 单 rank 专家数
constexpr int64_t N1 = 2048;
constexpr int64_t MX_GROUP = 64;
constexpr int64_t SCALE_LAST_DIM = 2;
constexpr int64_t MX_QUANT_MODE = 6;
constexpr double REL_ERR_TOL = 0.03;

// E8M0 反量化：2^(byte-127)
inline float E8m0ToFloat(uint8_t v)
{
    return std::ldexp(1.0f, static_cast<int>(v) - 127);
}

// E4M3 反量化（指数偏置 7）
inline float E4m3ToFloat(uint8_t v)
{
    int sign = (v & 0x80) ? -1 : 1;
    int expField = (v >> 3) & 0xF;
    int mant = v & 0x7;
    if (expField == 0) {
        return sign * std::ldexp(static_cast<float>(mant) / 8.0f, -6); // 次正规
    }
    return sign * std::ldexp(1.0f + static_cast<float>(mant) / 8.0f, expField - 7);
}

// 确定性输入：任意 rank 的数据可在线重建
inline uint8_t XQuantVal(int64_t rank, int64_t idx)
{
    return static_cast<uint8_t>(0x30 + ((rank * 131 + idx) % 8)); // e4m3 8~15，精确表示
}
inline uint8_t XScaleVal(int64_t rank, int64_t idx)
{
    return static_cast<uint8_t>(127 + ((rank * 17 + idx) % 4)); // e8m0 1/2/4/8
}
inline uint8_t WQuantVal(int64_t idx)
{
    return static_cast<uint8_t>(0x30 + (idx % 8));
}
inline uint8_t WScaleVal(int64_t idx)
{
    return static_cast<uint8_t>(127 + (idx % 4));
}

// 确定性全局计数矩阵：globalCounts[src * (W*E) + dst * E + e]，每 src 行和为 BSK
static void BuildGlobalCounts(std::vector<int64_t> &globalCounts)
{
    const int64_t segPerSrc = EP_WORLD_SIZE * E;
    globalCounts.assign(EP_WORLD_SIZE * segPerSrc, 0);
    for (int64_t s = 0; s < EP_WORLD_SIZE; s++) {
        int64_t remaining = BSK;
        int64_t segCnt = 0;
        for (int64_t idx = 0; idx < segPerSrc; idx++) {
            int64_t avg = remaining / (segPerSrc - idx);
            int64_t var = ((s + idx) % 3 == 0) ? 7 : ((s + idx) % 3 == 1) ? 3 : -5;
            int64_t cnt = avg + var;
            if (cnt < 0) {
                cnt = 0;
            }
            if (cnt > remaining) {
                cnt = remaining;
            }
            remaining -= cnt;
            globalCounts[s * segPerSrc + idx] = cnt;
            segCnt++;
        }
        globalCounts[s * segPerSrc + segPerSrc - 1] += remaining;
    }
}

static int LaunchOneThreadMx(Args &args)
{
    int ret = aclrtSetCurrentContext(args.context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetCurrentContext failed. ret: %d\n", ret); return ret);
    char hcomName[128] = {0};
    ret = HcclGetCommName(args.hcclComm, hcomName);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetCommName failed. ret: %d\n", ret); return -1);

    const int64_t myRank = args.rankId;
    std::vector<int64_t> globalCounts;
    BuildGlobalCounts(globalCounts);
    const int64_t segPerSrc = EP_WORLD_SIZE * E;

    // 本 rank 发送/接收计数（[dst/src][expert]）
    std::vector<int64_t> sendCountsList(EP_WORLD_SIZE * E, 0);
    std::vector<int64_t> recvCountsList(EP_WORLD_SIZE * E, 0);
    for (int64_t peer = 0; peer < EP_WORLD_SIZE; peer++) {
        for (int64_t e = 0; e < E; e++) {
            sendCountsList[peer * E + e] = globalCounts[myRank * segPerSrc + peer * E + e];
            recvCountsList[peer * E + e] = globalCounts[peer * segPerSrc + myRank * E + e];
        }
    }
    int64_t totalRecv = 0;
    for (auto c : recvCountsList) {
        totalRecv += c;
    }

    // rank s 的 gmm_x 中 (dst=me, expert e) 段起始行
    auto sendRowOf = [&](int64_t s, int64_t e) -> int64_t {
        int64_t off = 0;
        for (int64_t idx = 0; idx < segPerSrc; idx++) {
            int64_t d = idx / E;
            int64_t ee = idx % E;
            if (d < myRank || (d == myRank && ee < e)) {
                off += globalCounts[s * segPerSrc + idx];
            }
        }
        return off;
    };

    std::vector<int64_t> gmmXShape = {BSK, H};
    std::vector<int64_t> gmmWShape = {E, H, N1};
    std::vector<int64_t> gmmYShape = {totalRecv, N1};
    std::vector<int64_t> gmmXScaleShape = {BSK, H / MX_GROUP, SCALE_LAST_DIM};
    std::vector<int64_t> gmmWScaleShape = {E, H / MX_GROUP, N1, SCALE_LAST_DIM};

    const long long xSize = GetShapeSize(gmmXShape);
    const long long wSize = GetShapeSize(gmmWShape);
    const long long xScaleSize = GetShapeSize(gmmXScaleShape);
    const long long wScaleSize = GetShapeSize(gmmWScaleShape);
    const long long ySize = GetShapeSize(gmmYShape);

    std::vector<uint8_t> xHost(xSize);
    std::vector<uint8_t> wHost(wSize);
    std::vector<uint8_t> xScaleHost(xScaleSize);
    std::vector<uint8_t> wScaleHost(wScaleSize);
    for (long long i = 0; i < xSize; i++) {
        xHost[i] = XQuantVal(myRank, i);
    }
    for (long long i = 0; i < wSize; i++) {
        wHost[i] = WQuantVal(i);
    }
    for (long long i = 0; i < xScaleSize; i++) {
        xScaleHost[i] = XScaleVal(myRank, i);
    }
    for (long long i = 0; i < wScaleSize; i++) {
        wScaleHost[i] = WScaleVal(i);
    }
    std::vector<uint16_t> yHost(ySize, 0);

    void *xDev = nullptr;
    void *wDev = nullptr;
    void *yDev = nullptr;
    void *xScaleDev = nullptr;
    void *wScaleDev = nullptr;
    aclTensor *gmmX = nullptr;
    aclTensor *gmmW = nullptr;
    aclTensor *gmmY = nullptr;
    aclTensor *gmmXScale = nullptr;
    aclTensor *gmmWScale = nullptr;

    ret = CreateAclTensor(xHost, gmmXShape, &xDev, ACL_FLOAT8_E4M3FN, &gmmX);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(wHost, gmmWShape, &wDev, ACL_FLOAT8_E4M3FN, &gmmW);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(yHost, gmmYShape, &yDev, ACL_BF16, &gmmY);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(xScaleHost, gmmXScaleShape, &xScaleDev, ACL_FLOAT8_E8M0, &gmmXScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(wScaleHost, gmmWScaleShape, &wScaleDev, ACL_FLOAT8_E8M0, &gmmWScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    aclIntArray *sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray *recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    void *workspaceAddr = nullptr;
    int64_t groupSize = 0;

    ret = aclnnAlltoAllvQuantGroupedMatMulGetWorkspaceSize(
        gmmX, gmmW, gmmXScale, gmmWScale,
        nullptr, // sendCountsTensorOptional
        nullptr, // recvCountsTensorOptional
        nullptr, // mmX（apace 路径不支持共享专家，置空）
        nullptr, // mmWeight
        nullptr, // mmXScale
        nullptr, // mmWeightScale
        MX_QUANT_MODE, MX_QUANT_MODE, MX_QUANT_MODE, MX_QUANT_MODE, hcomName, EP_WORLD_SIZE, sendCounts, recvCounts,
        false, // transGmmWeight
        false, // transMmWeight
        groupSize,
        false, // permuteOutFlag（apace 路径不支持 permute_out）
        gmmY, nullptr, nullptr, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("[ERROR] aclnnAlltoAllvQuantGroupedMatMulGetWorkspaceSize failed. ret = %d \n", ret);
              return ret);

    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret: %d\n", ret); return ret);
    }

    ret = aclnnAlltoAllvQuantGroupedMatMul(workspaceAddr, workspaceSize, executor, args.stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnAlltoAllvQuantGroupedMatMul failed. ret = %d \n", ret);
              return ret);

    ret = aclrtSynchronizeStreamWithTimeout(args.stream, 10000000);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret: %d\n", ret);
              return ret);

    std::vector<uint16_t> yOut(ySize, 0);
    ret = aclrtMemcpy(yOut.data(), ySize * sizeof(uint16_t), yDev, ySize * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMemcpy D2H failed. ret: %d\n", ret); return ret);


    // ---------------- host 侧 golden（expert-major 语义，抽样比对）----------------
    double maxRelErr = 0.0;
    long long badCnt = 0;
    long long sampleCnt = 0;
    int64_t outRow = 0;
    for (int64_t e = 0; e < E; e++) {
        for (int64_t s = 0; s < EP_WORLD_SIZE; s++) {
            const int64_t rows = recvCountsList[s * E + e];
            const int64_t srcRowBase = sendRowOf(s, e);
            for (int64_t r = 0; r < rows; r++) {
                for (int64_t n = 0; n < N1; n += 128) {
                    float acc = 0.0f;
                    for (int64_t k = 0; k < H; k++) {
                        const int64_t aIdx = (srcRowBase + r) * H + k;
                        // x scale 布局 [row, H/64, 2]：col k 对应 (k/64)*2 + (k%64)/32
                        const int64_t aScaleIdx =
                            (aIdx / H) * (H / MX_GROUP * SCALE_LAST_DIM) + (k / MX_GROUP) * SCALE_LAST_DIM +
                            (k % MX_GROUP) / 32;
                        const float a = E4m3ToFloat(XQuantVal(s, aIdx)) * E8m0ToFloat(XScaleVal(s, aScaleIdx));
                        const float b =
                            E4m3ToFloat(WQuantVal((e * H + k) * N1 + n)) *
                            E8m0ToFloat(WScaleVal(((e * (H / MX_GROUP) + k / MX_GROUP) * N1 + n) * SCALE_LAST_DIM +
                                                  (k % MX_GROUP) / 32));
                        acc += a * b;
                    }
                    const uint32_t bits = static_cast<uint32_t>(yOut[outRow * N1 + n]) << 16;
                    float gotF = 0.0f;
                    std::memcpy(&gotF, &bits, sizeof(gotF));
                    const double rel = std::fabs(gotF - acc) / std::max<double>(std::fabs(acc), 1.0);
                    if (rel > maxRelErr) {
                        maxRelErr = rel;
                    }
                    if (rel > REL_ERR_TOL) {
                        badCnt++;
                    }
                    sampleCnt++;
                }
                outRow++;
            }
        }
    }
    const bool pass = (badCnt == 0) && (outRow == totalRecv);
    LOG_PRINT("[Rank %lld] totalRecv=%lld rowsChecked=%lld sampled=%lld maxRelErr=%f badCnt=%lld => %s\n",
              static_cast<long long>(myRank), static_cast<long long>(totalRecv), static_cast<long long>(outRow),
              sampleCnt, maxRelErr, badCnt, pass ? "PASS" : "FAIL");

    aclDestroyTensor(gmmX);
    aclDestroyTensor(gmmW);
    aclDestroyTensor(gmmY);
    aclDestroyTensor(gmmXScale);
    aclDestroyTensor(gmmWScale);
    aclrtFree(xDev);
    aclrtFree(wDev);
    aclrtFree(yDev);
    aclrtFree(xScaleDev);
    aclrtFree(wScaleDev);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    HcclCommDestroy(args.hcclComm);
    aclrtDestroyStream(args.stream);
    aclrtDestroyContext(args.context);
    aclrtResetDevice(args.rankId);
    return pass ? 0 : 1;
}

int main(int argc, char *argv[])
{
    int ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclInit failed. ret: %d\n", ret); return ret);

    constexpr uint32_t WORLD_SIZE = EP_WORLD_SIZE;
    aclrtStream stream[WORLD_SIZE];
    aclrtContext context[WORLD_SIZE];

    for (uint32_t rankId = 0; rankId < WORLD_SIZE; rankId++) {
        ret = aclrtSetDevice(rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice failed. ret: %d\n", ret); return ret);
        ret = aclrtCreateContext(&context[rankId], rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateContext failed. ret: %d\n", ret); return ret);
        ret = aclrtCreateStream(&stream[rankId]);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed. ret: %d\n", ret); return ret);
    }

    int32_t devices[WORLD_SIZE];
    for (uint32_t i = 0; i < WORLD_SIZE; i++) {
        devices[i] = i;
    }

    HcclComm comms[WORLD_SIZE];
    ret = HcclCommInitAll(WORLD_SIZE, devices, comms);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclCommInitAll failed. ret = %d \n", ret); return ret);

    Args args[WORLD_SIZE];
    int threadRc[WORLD_SIZE] = {0};
    std::vector<std::unique_ptr<std::thread>> threads(WORLD_SIZE);
    for (uint32_t rankId = 0; rankId < WORLD_SIZE; rankId++) {
        args[rankId].rankId = rankId;
        args[rankId].hcclComm = comms[rankId];
        args[rankId].stream = stream[rankId];
        args[rankId].context = context[rankId];
        threads[rankId].reset(new std::thread(
            [&threadRc, &a = args[rankId]]() { threadRc[a.rankId] = LaunchOneThreadMx(a); }));
    }

    for (uint32_t rankId = 0; rankId < WORLD_SIZE; rankId++) {
        threads[rankId]->join();
    }
    aclFinalize();
    bool allPass = true;
    for (uint32_t rankId = 0; rankId < WORLD_SIZE; rankId++) {
        allPass = allPass && (threadRc[rankId] == 0);
    }
    LOG_PRINT("E2E MX apace validation: %s\n", allPass ? "ALL PASS" : "FAILED");
    return allPass ? 0 : 1;
}
