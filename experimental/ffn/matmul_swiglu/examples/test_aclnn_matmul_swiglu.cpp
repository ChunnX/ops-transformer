/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*
 * MatmulSwiglu 性能测试(方案 A: 端到端延迟 + 吞吐)。
 * 对若干 [M,K,N] 形状: warmup 后连续发射 ITERS 次 kernel, 一次 SynchronizeStream,
 * 用墙钟得平均端到端延迟(含 host tiling+launch), 并算 GFLOPS。纯 kernel 细分用 msprof(方案 B)。
 * 数据类型默认 fp16(x/weight/y), bias 固定 fp32; 改 PERF_FP32=1 可切 fp32 对照。
 */
#include <acl/acl.h>
#include <aclnnop/aclnn_matmul_swiglu.h>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <vector>

#define PERF_FP32 0  // 0=fp16(默认), 1=fp32

#define CHECK_RET(cond, expr)                    \
    do {                                         \
        if (!(cond)) {                           \
            expr;                                \
        }                                        \
    } while (0)

#define LOG(fmt, ...)               \
    do {                            \
        printf(fmt, ##__VA_ARGS__); \
        fflush(stdout);             \
    } while (0)

namespace {
#if PERF_FP32
using XT = float;
constexpr aclDataType kXDtype = ACL_FLOAT;
#else
using XT = uint16_t;  // fp16 存储
constexpr aclDataType kXDtype = ACL_FLOAT16;

// float -> IEEE754 half(round-to-nearest, 次正规flush为0), 仅用于造输入
uint16_t FloatToHalf(float f)
{
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    uint32_t sign = (x >> 16) & 0x8000U;
    int32_t exp = static_cast<int32_t>((x >> 23) & 0xFFU) - 127 + 15;
    uint32_t mant = x & 0x7FFFFFU;
    if (exp <= 0) {
        return static_cast<uint16_t>(sign);
    }
    if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7C00U);
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13));
}
#endif

XT MakeX(float v)
{
#if PERF_FP32
    return v;
#else
    return FloatToHalf(v);
#endif
}

int64_t ShapeSize(const std::vector<int64_t>& s)
{
    int64_t n = 1;
    for (auto d : s) {
        n *= d;
    }
    return n;
}

template <typename T>
int CreateTensor(const std::vector<T>& host, const std::vector<int64_t>& shape, void** devAddr,
                 aclDataType dtype, aclTensor** tensor)
{
    auto bytes = ShapeSize(shape) * static_cast<int64_t>(sizeof(T));
    auto ret = aclrtMalloc(devAddr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG("aclrtMalloc failed %d\n", ret); return ret);
    ret = aclrtMemcpy(*devAddr, bytes, host.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG("aclrtMemcpy failed %d\n", ret); return ret);
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0, ACL_FORMAT_ND,
                              shape.data(), shape.size(), *devAddr);
    CHECK_RET(*tensor != nullptr, LOG("aclCreateTensor null\n"); return -1);
    return 0;
}

struct Shape {
    int64_t m;
    int64_t k;
    int64_t n;  // 输出宽度; weight 为 [K, 2N]
};

// 对单个形状测性能。warmup 次预热, iters 次计时。
int RunShape(const Shape& sp, int warmup, int iters, aclrtStream stream)
{
    const std::vector<int64_t> xShape = {sp.m, sp.k};
    const std::vector<int64_t> wShape = {sp.k, 2 * sp.n};
    const std::vector<int64_t> bShape = {2 * sp.n};
    const std::vector<int64_t> yShape = {sp.m, sp.n};

    std::vector<XT> xHost(ShapeSize(xShape), MakeX(0.1f));
    std::vector<XT> wHost(ShapeSize(wShape), MakeX(0.1f));
    std::vector<float> bHost(ShapeSize(bShape), 0.0f);  // bias 固定 fp32
    std::vector<XT> yHost(ShapeSize(yShape), MakeX(0.0f));

    aclTensor *x = nullptr, *weight = nullptr, *bias = nullptr, *out = nullptr;
    void *xDev = nullptr, *wDev = nullptr, *bDev = nullptr, *yDev = nullptr;
    int ret = CreateTensor(xHost, xShape, &xDev, kXDtype, &x);
    ret |= CreateTensor(wHost, wShape, &wDev, kXDtype, &weight);
    ret |= CreateTensor(bHost, bShape, &bDev, ACL_FLOAT, &bias);
    ret |= CreateTensor(yHost, yShape, &yDev, kXDtype, &out);
    CHECK_RET(ret == 0, LOG("create tensor failed\n"); return -1);

    // 先取一次 workspace 大小(各次相同), 预分配复用
    uint64_t wsSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnMatmulSwigluGetWorkspaceSize(x, weight, bias, false, out, &wsSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG("GetWorkspaceSize failed %d\n", ret); return ret);
    void* ws = nullptr;
    if (wsSize > 0) {
        ret = aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG("alloc ws failed %d\n", ret); return ret);
    }

    // 每次迭代重新生成 executor(稳妥, 避免复用限制); GetWorkspaceSize 为 host 侧,
    // 计时段内不入流, 墙钟主要反映连续 launch 的设备吞吐。
    auto launch = [&]() -> int {
        aclOpExecutor* exec = nullptr;
        uint64_t sz = 0;
        int r = aclnnMatmulSwigluGetWorkspaceSize(x, weight, bias, false, out, &sz, &exec);
        if (r != ACL_SUCCESS) {
            return r;
        }
        return aclnnMatmulSwiglu(ws, wsSize, exec, stream);
    };

    for (int i = 0; i < warmup; i++) {
        ret = launch();
        CHECK_RET(ret == ACL_SUCCESS, LOG("warmup launch failed %d\n", ret); return ret);
    }
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG("sync failed %d\n", ret); return ret);

    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; i++) {
        ret = launch();
        CHECK_RET(ret == ACL_SUCCESS, LOG("timed launch failed %d\n", ret); return ret);
    }
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG("sync failed %d\n", ret); return ret);
    auto t1 = std::chrono::steady_clock::now();

    double totalUs = std::chrono::duration<double, std::micro>(t1 - t0).count();
    double avgUs = totalUs / iters;
    // matmul FLOPs = 2 * M * (2N) * K; SwiGLU 逐元素开销可忽略
    double flops = 2.0 * static_cast<double>(sp.m) * (2.0 * static_cast<double>(sp.n)) * static_cast<double>(sp.k);
    double gflops = flops / (avgUs * 1e-6) / 1e9;

    LOG("  M=%-5ld K=%-5ld N=%-6ld | avg %8.2f us | %8.1f GFLOPS\n", sp.m, sp.k, sp.n, avgUs, gflops);

    if (ws != nullptr) {
        aclrtFree(ws);
    }
    aclDestroyTensor(x);
    aclDestroyTensor(weight);
    aclDestroyTensor(bias);
    aclDestroyTensor(out);
    aclrtFree(xDev);
    aclrtFree(wDev);
    aclrtFree(bDev);
    aclrtFree(yDev);
    return 0;
}
}  // namespace

int main()
{
    const int32_t deviceId = 0;
    const int warmup = 10;
    const int iters = 50;

    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG("aclInit failed %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG("aclrtSetDevice failed %d\n", ret); return ret);
    aclrtStream stream = nullptr;
    ret = aclrtCreateStream(&stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG("aclrtCreateStream failed %d\n", ret); return ret);

    LOG("MatmulSwiglu 性能测试 (%s, warmup=%d iters=%d)\n", PERF_FP32 ? "fp32" : "fp16", warmup, iters);
    LOG("端到端延迟(含 host tiling+launch); 纯 kernel 用 msprof 采集\n");

    // 默认 sweep: 由小到大, N 为中间维(输出宽度), weight=[K,2N]
    std::vector<Shape> shapes = {
        {512, 2048, 2048},
        {2048, 4096, 4096},
        {4096, 4096, 11008},
    };
    for (const auto& sp : shapes) {
        RunShape(sp, warmup, iters, stream);
    }

    if (stream != nullptr) {
        aclrtDestroyStream(stream);
    }
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
