/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file test_aclnn_broadcast_nddma_bench.cpp
 * @brief aclnn 二段式调用示例：scene=AB（dim4 广播），自造输入 + host golden，逐字节比对。
 *
 * scene=AB：输入 src(dim0,dim1,dim2,dim3,1)，输出 (dim0,dim1,dim2,dim3,dim4)。
 *   out[i0,i1,i2,i3,i4] = in[i0,i1,i2,i3,0]（沿 dim4 广播复制）。
 * 默认 fp16，shape 输出 56x4x8x8x8。位型比对（fp16 广播是纯拷贝，无数值误差）。
 */

#include "acl/acl.h"
#include "aclnnop/aclnn_broadcast_nddma_bench.h"
#include <cstdint>
#include <cstdio>
#include <vector>

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

namespace {
constexpr int64_t DIM0 = 56;
constexpr int64_t DIM1 = 4;
constexpr int64_t DIM2 = 8;
constexpr int64_t DIM3 = 8;
constexpr int64_t DIM4 = 8;
constexpr int64_t SCENE_AB = 0;
constexpr int64_t IS_MTE3 = 1;

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t s = 1;
    for (auto i : shape) {
        s *= i;
    }
    return s;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
    std::vector<int64_t> stride(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        stride[i] = shape[i + 1] * stride[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, stride.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}
}  // namespace

int main()
{
    constexpr int32_t deviceId = 0;
    using ElemT = uint16_t;  // 承载 fp16 位型，比对按字节
    const aclDataType kAclType = aclDataType::ACL_FLOAT16;

    aclrtStream stream = nullptr;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // scene=AB：输入广播轴 dim4=1
    std::vector<int64_t> xShape = {DIM0, DIM1, DIM2, DIM3, 1};
    std::vector<int64_t> yShape = {DIM0, DIM1, DIM2, DIM3, DIM4};
    int64_t nIn = GetShapeSize(xShape);
    int64_t nOut = GetShapeSize(yShape);

    // 自造输入：递增序列（fp16 位型用连续 uint16 值，保证 host/device 一致的纯拷贝比对）
    std::vector<ElemT> xHost(nIn);
    for (int64_t i = 0; i < nIn; ++i) {
        xHost[i] = static_cast<ElemT>((i * 7 + 1) & 0x3FFF);  // 落在 fp16 正常数区，避免 NaN/Inf 位型
    }
    std::vector<ElemT> yHost(nOut, 0);

    // host golden：out[...,i4] = in[...,0]（AB 沿 dim4 广播）
    std::vector<ElemT> golden(nOut);
    int64_t innerBlocks = DIM0 * DIM1 * DIM2 * DIM3;  // = nIn
    for (int64_t b = 0; b < innerBlocks; ++b) {
        for (int64_t i4 = 0; i4 < DIM4; ++i4) {
            golden[b * DIM4 + i4] = xHost[b];
        }
    }

    void* xDev = nullptr;
    void* yDev = nullptr;
    void* workspaceAddr = nullptr;
    aclTensor* x = nullptr;
    aclTensor* y = nullptr;

    ret = CreateAclTensor(xHost, xShape, &xDev, kAclType, &x);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Create x tensor failed. ERROR: %d\n", ret); return ret);
    ret = CreateAclTensor(yHost, yShape, &yDev, kAclType, &y);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Create y tensor failed. ERROR: %d\n", ret); return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnBroadcastNddmaBenchGetWorkspaceSize(
        x, IS_MTE3, SCENE_AB, DIM2, DIM3, DIM4, y, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnBroadcastNddmaBenchGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    ret = aclnnBroadcastNddmaBench(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBroadcastNddmaBench failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    size_t nBytes = static_cast<size_t>(nOut) * sizeof(ElemT);
    std::vector<ElemT> outHost(nOut);
    ret = aclrtMemcpy(outHost.data(), nBytes, yDev, nBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result D2H failed. ERROR: %d\n", ret); return ret);

    int mismatch = 0;
    for (int64_t i = 0; i < nOut; ++i) {
        if (outHost[i] != golden[i]) {
            if (mismatch < 8) {
                LOG_PRINT("elem mismatch at %lld: out=%u golden=%u\n",
                          static_cast<long long>(i), outHost[i], golden[i]);
            }
            mismatch++;
        }
    }

    aclDestroyTensor(x);
    aclDestroyTensor(y);
    aclrtFree(xDev);
    aclrtFree(yDev);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    if (mismatch == 0) {
        LOG_PRINT("[PASS] broadcast(scene=AB) output matches golden (%lld elems)\n",
                  static_cast<long long>(nOut));
        return 0;
    }
    LOG_PRINT("[FAIL] %d elems mismatch out of %lld\n", mismatch, static_cast<long long>(nOut));
    return 1;
}
