/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms of the
 * CANN Open Software License Agreement Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * https://www.hiascend.com/software/clicense/clicense-2.0
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied. See the License for the specific language governing permissions and limitations under
 * the License.
 * Description: 非量化 FP16/BF16 AllGatherMatmulV2 example（种子版，A9 探针）。
 * 环境变量参数化：
 *   AGM_V2_DEV_NUM    rank 数，默认 2（取值 2/4/8/...）
 *   AGM_V2_COMM_MODE  通信模式，默认 ccu（ccu / ai_cpu）
 *   AGM_V2_DTYPE      数据类型，默认 fp16（fp16 / bf16）
 *   AGM_V2_M / AGM_V2_K / AGM_V2_N  矩阵形状，默认 256 / 256 / 128（K 须在 [256, 65535)）
 * 判定：
 *   gatherOut 与各 rank 的 x1 严格位级比对（通信为纯搬运）；
 *   output 与 host 侧 FP32 累加 golden 做混合容差比对（fp16: atol/rtol=2e-3, bf16: 8e-3）；
 *   全 rank 通过进程退出码为 0。
 */

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>
#include "hccl/hccl.h"
#include "aclnn/opdev/fp16_t.h"
#include "aclnnop/aclnn_all_gather_matmul_v2.h"

#define CHECK_RET(cond, return_expr) \
    do { \
        if (!(cond)) { \
            return_expr; \
        } \
    } while (0)

#define LOG_PRINT(message, ...) \
    do { \
        printf(message, ##__VA_ARGS__); \
        fflush(stdout); \
    } while (0)

namespace {
// ---------------- 环境参数 ----------------
int g_devNum = 2;
const char *g_commMode = "ccu";
bool g_isBf16 = false;
int64_t g_m = 256;
int64_t g_k = 256;
int64_t g_n = 128;

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto d : shape) {
        shapeSize *= d;
    }
    return shapeSize;
}

int ParseIntEnv(const char *key, int64_t defVal, int64_t minVal, int64_t maxVal, bool &ok)
{
    const char *raw = getenv(key);
    if (raw == nullptr || raw[0] == '\0') {
        return static_cast<int>(defVal);
    }
    char *end = nullptr;
    long v = std::strtol(raw, &end, 10);
    if (end == nullptr || *end != '\0' || v < minVal || v > maxVal) {
        LOG_PRINT("[ERROR] env %s='%s' invalid, require [%ld, %ld].\n", key, raw, minVal, maxVal);
        ok = false;
        return static_cast<int>(defVal);
    }
    return static_cast<int>(v);
}

bool ParseGlobalParams()
{
    bool ok = true;
    g_devNum = static_cast<int>(ParseIntEnv("AGM_V2_DEV_NUM", 2, 1, 64, ok));
    int64_t m = ParseIntEnv("AGM_V2_M", 256, 1, 1 << 20, ok);
    int64_t k = ParseIntEnv("AGM_V2_K", 256, 256, 65534, ok);
    int64_t n = ParseIntEnv("AGM_V2_N", 128, 1, 1 << 20, ok);
    g_m = m;
    g_k = k;
    g_n = n;
    const char *comm = getenv("AGM_V2_COMM_MODE");
    if (comm != nullptr && comm[0] != '\0') {
        if (std::strcmp(comm, "ccu") != 0 && std::strcmp(comm, "ai_cpu") != 0) {
            LOG_PRINT("[ERROR] AGM_V2_COMM_MODE='%s' invalid, require ccu/ai_cpu.\n", comm);
            ok = false;
        } else {
            g_commMode = comm;
        }
    }
    const char *dtype = getenv("AGM_V2_DTYPE");
    if (dtype != nullptr && dtype[0] != '\0') {
        if (std::strcmp(dtype, "fp16") != 0 && std::strcmp(dtype, "bf16") != 0) {
            LOG_PRINT("[ERROR] AGM_V2_DTYPE='%s' invalid, require fp16/bf16.\n", dtype);
            ok = false;
        } else {
            g_isBf16 = (std::strcmp(dtype, "bf16") == 0);
        }
    }
    return ok;
}

// ---------------- host 侧 dtype 比特工具 ----------------
// bf16 host 侧位级转换（round-to-nearest-even）；不引 bfloat16.h 以兼容 cust 模式 include 路径
uint16_t FloatToBf16Bits(float v)
{
    uint32_t u;
    std::memcpy(&u, &v, sizeof(u));
    uint32_t rounded = u + 0x7FFFU + ((u >> 16U) & 1U);
    return static_cast<uint16_t>(rounded >> 16U);
}

float Bf16BitsToFloat(uint16_t bits)
{
    uint32_t u = static_cast<uint32_t>(bits) << 16U;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

uint16_t FloatToBits(float v)
{
    if (g_isBf16) {
        return FloatToBf16Bits(v);
    }
    op::fp16_t h;
    h = v;
    return h.val;
}

float BitsToFloat(uint16_t bits)
{
    if (g_isBf16) {
        return Bf16BitsToFloat(bits);
    }
    op::fp16_t h(bits);
    return h.toFloat();
}

// ---------------- 确定性输入（全部 rank 生成一致的全体数据） ----------------
float X1RawValue(int rank, int64_t row, int64_t col)
{
    int v = (rank * 131 + static_cast<int>(row) * 7 + static_cast<int>(col) * 3) % 1000;
    return static_cast<float>(v) / 1000.0f - 0.5f; // [-0.5, 0.499]
}

float X2RawValue(int64_t row, int64_t col)
{
    int v = (static_cast<int>(row) * 17 + static_cast<int>(col) * 29) % 1000;
    return (static_cast<float>(v) / 1000.0f - 0.5f) * 0.5f; // [-0.25, 0.2495]
}

// ---------------- ACL tensor 构造 ----------------
int CreateAclTensorFromBits(const std::vector<uint16_t> &hostBits, const std::vector<int64_t> &shape,
                            void **deviceAddr, aclDataType dataType, aclTensor **tensor)
{
    auto size = static_cast<size_t>(GetShapeSize(shape)) * sizeof(uint16_t);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc failed. ret: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostBits.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMemcpy failed. ret: %d\n", ret); return ret);
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
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
    int result = -1; // 0 = pass
};

// 秩序收尾（失败路径也要保证通信域/流/上下文释放，避免仿真进程挂死在收尾）
void FinalizeRank(Args &args)
{
    HcclCommDestroy(args.hcclComm);
    aclrtDestroyStream(args.stream);
    aclrtResetDevice(args.rankId);
    aclrtDestroyContext(args.context);
}

// ---------------- 单 rank 执行 + 内嵌 golden 比对 ----------------
int LaunchOneThreadAllGatherMmV2(Args &args)
{
    const int rankId = args.rankId;
    const int rankDim = g_devNum;
    const int64_t m = g_m;
    const int64_t k = g_k;
    const int64_t n = g_n;
    const aclDataType dtype = g_isBf16 ? aclDataType::ACL_BF16 : aclDataType::ACL_FLOAT16;
    const float atol = g_isBf16 ? 8e-3f : 2e-3f;
    const float rtol = g_isBf16 ? 8e-3f : 2e-3f;

    int ret = aclrtSetCurrentContext(args.context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetCurrentContext failed. ret: %d\n", ret); return ret);
    char hcomName[128] = {0};
    ret = HcclGetCommName(args.hcclComm, hcomName);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetCommName failed. ret: %d\n", ret); return -1);
    LOG_PRINT("[INFO] rank = %d, hcomName = %s, devNum = %d, commMode = %s, dtype = %s, m = %ld, k = %ld, n = %ld\n",
              rankId, hcomName, rankDim, g_commMode, g_isBf16 ? "bf16" : "fp16", static_cast<long>(m),
              static_cast<long>(k), static_cast<long>(n));

    std::vector<int64_t> x1Shape = {m, k};
    std::vector<int64_t> x2Shape = {k, n};
    std::vector<int64_t> outShape = {m * rankDim, n};
    std::vector<int64_t> gatherOutShape = {m * rankDim, k};

    // 确定性生成：本 rank 的 x1 与全量 x2（golden 需要全 rank 数据）
    std::vector<uint16_t> x1Bits(static_cast<size_t>(m) * k);
    std::vector<float> x1Quant(static_cast<size_t>(m) * k);
    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < k; j++) {
            uint16_t bits = FloatToBits(X1RawValue(rankId, i, j));
            x1Bits[static_cast<size_t>(i) * k + j] = bits;
            x1Quant[static_cast<size_t>(i) * k + j] = BitsToFloat(bits);
        }
    }
    std::vector<uint16_t> x2Bits(static_cast<size_t>(k) * n);
    std::vector<float> x2Quant(static_cast<size_t>(k) * n);
    for (int64_t i = 0; i < k; i++) {
        for (int64_t j = 0; j < n; j++) {
            uint16_t bits = FloatToBits(X2RawValue(i, j));
            x2Bits[static_cast<size_t>(i) * n + j] = bits;
            x2Quant[static_cast<size_t>(i) * n + j] = BitsToFloat(bits);
        }
    }
    // 全 rank x1 量化值（gatherOut golden 与 output golden 共用）
    std::vector<float> allX1Quant(static_cast<size_t>(m) * k * rankDim);
    std::vector<uint16_t> allX1Bits(static_cast<size_t>(m) * k * rankDim);
    for (int r = 0; r < rankDim; r++) {
        for (int64_t i = 0; i < m; i++) {
            for (int64_t j = 0; j < k; j++) {
                uint16_t bits = FloatToBits(X1RawValue(r, i, j));
                allX1Bits[(static_cast<size_t>(r) * m + i) * k + j] = bits;
                allX1Quant[(static_cast<size_t>(r) * m + i) * k + j] = BitsToFloat(bits);
            }
        }
    }

    void *x1DeviceAddr = nullptr;
    void *x2DeviceAddr = nullptr;
    void *outDeviceAddr = nullptr;
    void *gatherOutDeviceAddr = nullptr;
    aclTensor *x1 = nullptr;
    aclTensor *x2 = nullptr;
    aclTensor *out = nullptr;
    aclTensor *gatherOut = nullptr;

    const int64_t gatherIndex = 0;
    const int64_t commTurn = 0;
    const int64_t streamMode = 1;
    const int64_t blockSize = 0;
    const int64_t groupSize = 0;
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    void *workspaceAddr = nullptr;

    ret = CreateAclTensorFromBits(x1Bits, x1Shape, &x1DeviceAddr, dtype, &x1);
    CHECK_RET(ret == 0, return ret);
    ret = CreateAclTensorFromBits(x2Bits, x2Shape, &x2DeviceAddr, dtype, &x2);
    CHECK_RET(ret == 0, return ret);
    std::vector<uint16_t> outInit(static_cast<size_t>(m) * rankDim * n, 0);
    ret = CreateAclTensorFromBits(outInit, outShape, &outDeviceAddr, dtype, &out);
    CHECK_RET(ret == 0, return ret);
    std::vector<uint16_t> gatherInit(static_cast<size_t>(m) * rankDim * k, 0);
    ret = CreateAclTensorFromBits(gatherInit, gatherOutShape, &gatherOutDeviceAddr, dtype, &gatherOut);
    CHECK_RET(ret == 0, return ret);

    // 阶段一：非量化场景 scale/amax 均传 nullptr
    ret = aclnnAllGatherMatmulV2GetWorkspaceSize(x1, x2, nullptr, nullptr, nullptr, nullptr, blockSize, hcomName,
                                                 gatherIndex, commTurn, streamMode, groupSize, g_commMode, out,
                                                 gatherOut, nullptr, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR][rank %d] aclnnAllGatherMatmulV2GetWorkspaceSize failed. ret = %d\n",
                                            rankId, ret);
              return ret);
    LOG_PRINT("[INFO][rank %d] GetWorkspaceSize success, workspaceSize = %lu.\n", rankId,
              static_cast<unsigned long>(workspaceSize));
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret: %d \n", ret); return ret);
    }
    // 阶段二
    ret = aclnnAllGatherMatmulV2(workspaceAddr, workspaceSize, executor, args.stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR][rank %d] aclnnAllGatherMatmulV2 failed. ret = %d\n", rankId, ret);
              return ret);
    ret = aclrtSynchronizeStreamWithTimeout(args.stream, 1800000);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("[ERROR][rank %d] aclrtSynchronizeStreamWithTimeout failed. ret: %d\n", rankId, ret);
              return ret);
    LOG_PRINT("[INFO][rank %d] aclnnAllGatherMatmulV2 execute done.\n", rankId);

    // ---------------- 回读 + golden 比对 ----------------
    int checkRet = 0;
    const size_t outElemCnt = static_cast<size_t>(m) * rankDim * n;
    const size_t gatherElemCnt = static_cast<size_t>(m) * rankDim * k;
    std::vector<uint16_t> outDev(outElemCnt, 0);
    std::vector<uint16_t> gatherDev(gatherElemCnt, 0);
    ret = aclrtMemcpy(outDev.data(), outElemCnt * sizeof(uint16_t), outDeviceAddr, outElemCnt * sizeof(uint16_t),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR][rank %d] memcpy out failed. ret: %d\n", rankId, ret); return ret);
    ret = aclrtMemcpy(gatherDev.data(), gatherElemCnt * sizeof(uint16_t), gatherOutDeviceAddr,
                      gatherElemCnt * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR][rank %d] memcpy gatherOut failed. ret: %d\n", rankId, ret);
              return ret);

    // 1) gatherOut：纯通信搬运，严格位级比对
    size_t gatherMismatch = 0;
    for (size_t idx = 0; idx < gatherElemCnt; idx++) {
        if (gatherDev[idx] != allX1Bits[idx]) {
            if (gatherMismatch < 4) {
                LOG_PRINT("[ERROR][rank %d] gatherOut mismatch at flat %zu (row %zu col %zu): expect 0x%04x got 0x%04x\n",
                          rankId, idx, idx / static_cast<size_t>(k), idx % static_cast<size_t>(k), allX1Bits[idx],
                          gatherDev[idx]);
            }
            gatherMismatch++;
        }
    }
    LOG_PRINT("[INFO][rank %d] gatherOut bitwise compare: %zu/%zu mismatch.\n", rankId, gatherMismatch, gatherElemCnt);
    if (gatherMismatch != 0) {
        checkRet = 1;
    }

    // 2) output：FP32 累加 golden，混合容差
    size_t outMismatch = 0;
    float maxAbsErr = 0.0f;
    float maxRelErr = 0.0f;
    for (int64_t row = 0; row < m * rankDim; row++) {
        const float *aRow = &allX1Quant[static_cast<size_t>(row) * k];
        for (int64_t col = 0; col < n; col++) {
            float golden = 0.0f;
            for (int64_t t = 0; t < k; t++) {
                golden += aRow[t] * x2Quant[static_cast<size_t>(t) * n + col];
            }
            float dev = BitsToFloat(outDev[static_cast<size_t>(row) * n + col]);
            float absErr = std::fabs(dev - golden);
            float relErr = (std::fabs(golden) > 1e-6f) ? absErr / std::fabs(golden) : 0.0f;
            if (absErr > maxAbsErr) {
                maxAbsErr = absErr;
            }
            if (relErr > maxRelErr) {
                maxRelErr = relErr;
            }
            if (absErr > atol + rtol * std::fabs(golden)) {
                if (outMismatch < 4) {
                    LOG_PRINT("[ERROR][rank %d] output mismatch at (row %ld, col %ld): golden %.6f dev %.6f absErr %.6f\n",
                              rankId, static_cast<long>(row), static_cast<long>(col), golden, dev, absErr);
                }
                outMismatch++;
            }
        }
    }
    LOG_PRINT("[INFO][rank %d] output compare(tol a=%.1e r=%.1e): %zu/%zu mismatch, maxAbsErr %.6e, maxRelErr %.6e.\n",
              rankId, atol, rtol, outMismatch, outElemCnt, maxAbsErr, maxRelErr);
    if (outMismatch != 0) {
        checkRet = 1;
    }

    LOG_PRINT("[INFO][rank %d] %s: gatherOut %s, output %s.\n", rankId, checkRet == 0 ? "PASS" : "FAIL",
              gatherMismatch == 0 ? "PASS" : "FAIL", outMismatch == 0 ? "PASS" : "FAIL");

    // ---------------- 资源释放 ----------------
    if (x1 != nullptr) {
        aclDestroyTensor(x1);
    }
    if (x2 != nullptr) {
        aclDestroyTensor(x2);
    }
    if (out != nullptr) {
        aclDestroyTensor(out);
    }
    if (gatherOut != nullptr) {
        aclDestroyTensor(gatherOut);
    }
    if (x1DeviceAddr != nullptr) {
        aclrtFree(x1DeviceAddr);
    }
    if (x2DeviceAddr != nullptr) {
        aclrtFree(x2DeviceAddr);
    }
    if (outDeviceAddr != nullptr) {
        aclrtFree(outDeviceAddr);
    }
    if (gatherOutDeviceAddr != nullptr) {
        aclrtFree(gatherOutDeviceAddr);
    }
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    return checkRet;
}
} // namespace

int main(int argc, char *argv[])
{
    class EnvGuard {
    public:
        EnvGuard(const char *key, const char *val, bool enable = true) : key_(enable ? key : nullptr)
        {
            if (!enable) {
                return;
            }
            const char *old = getenv(key);
            if (old != nullptr) {
                strncpy(saved_, old, sizeof(saved_) - 1);
            }
            setenv(key, val, 1);
        }
        ~EnvGuard()
        {
            if (key_ == nullptr) {
                return;
            }
            if (saved_[0] != '\0') {
                setenv(key_, saved_, 1);
            } else {
                unsetenv(key_);
            }
        }

    private:
        const char *key_;
        char saved_[256] = {0};
    };

    if (!ParseGlobalParams()) {
        return 1;
    }
    EnvGuard envGuard("HCCL_OP_EXPANSION_MODE", "CCU_SCHED", std::strcmp(g_commMode, "ccu") == 0);

    int ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclInit failed. ret: %d\n", ret); return ret);
    aclrtStream stream[g_devNum];
    aclrtContext context[g_devNum];
    for (int rankId = 0; rankId < g_devNum; rankId++) {
        ret = aclrtSetDevice(rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice(%d) failed. ret: %d\n", rankId, ret);
                  return ret);
        ret = aclrtCreateContext(&context[rankId], rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateContext failed. ret: %d\n", ret); return ret);
        ret = aclrtCreateStream(&stream[rankId]);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed. ret: %d\n", ret); return ret);
    }
    int32_t devices[g_devNum];
    for (int i = 0; i < g_devNum; i++) {
        devices[i] = i;
    }
    HcclComm comms[g_devNum];
    ret = HcclCommInitAll(g_devNum, devices, comms);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclCommInitAll(devNum=%d) failed. ret: %d\n", g_devNum, ret);
              return ret);

    std::vector<Args> args(g_devNum);
    std::vector<std::unique_ptr<std::thread>> threads(g_devNum);
    for (int rankId = 0; rankId < g_devNum; rankId++) {
        args[rankId].rankId = rankId;
        args[rankId].hcclComm = comms[rankId];
        args[rankId].context = context[rankId];
        args[rankId].stream = stream[rankId];
        threads[rankId].reset(new (std::nothrow) std::thread([rankId, &args]() {
            int r = LaunchOneThreadAllGatherMmV2(args[rankId]);
            args[rankId].result = r;
            FinalizeRank(args[rankId]);
        }));
    }
    for (int rankId = 0; rankId < g_devNum; rankId++) {
        threads[rankId]->join();
    }
    int finalRet = 0;
    for (int rankId = 0; rankId < g_devNum; rankId++) {
        if (args[rankId].result != 0) {
            LOG_PRINT("[RESULT][rank %d] FAIL (ret = %d)\n", rankId, args[rankId].result);
            finalRet = 1;
        } else {
            LOG_PRINT("[RESULT][rank %d] PASS\n", rankId);
        }
    }
    aclFinalize();
    LOG_PRINT("[RESULT] all_gather_matmul_v2 fp16 example %s (devNum=%d, commMode=%s, dtype=%s, m=%ld, k=%ld, n=%ld)\n",
              finalRet == 0 ? "PASS" : "FAIL", g_devNum, g_commMode, g_isBf16 ? "bf16" : "fp16",
              static_cast<long>(g_m), static_cast<long>(g_k), static_cast<long>(g_n));
    return finalRet;
}
