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
 * @file test_aclnn_last_transpose_nddma_bench.cpp
 * @brief aclnn 二段式调用示例：自造输入 + host golden，逐字节比对（无外部数据依赖）。
 *
 *   input  shape (dim0, dim1, dim2, dim3)
 *   output shape (dim0, dim1, dim3, dim2) = transpose[0,1,3,2]（末两轴互换）
 *   y[i0,i1,i3,i2] = x[i0,i1,i2,i3]（纯索引重排，位型比对无数值误差）。
 * 默认 fp16，shape 64x4x256x256。
 */

#include "acl/acl.h"
#include "aclnnop/aclnn_last_transpose_nddma_bench.h"
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
constexpr int64_t DIM0 = 64;
constexpr int64_t DIM1 = 4;
constexpr int64_t DIM2 = 256;
constexpr int64_t DIM3 = 256;
constexpr int64_t IS_MTE3 = 1;  // 写回 GM，做精度比对

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
    // fp16 => 2 bytes/elem
    using ElemT = uint16_t;  // 承载 fp16 位型，比对按字节
    const aclDataType kAclType = aclDataType::ACL_FLOAT16;

    aclrtStream stream = nullptr;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> xShape = {DIM0, DIM1, DIM2, DIM3};
    std::vector<int64_t> yShape = {DIM0, DIM1, DIM3, DIM2};
    int64_t nElem = GetShapeSize(xShape);
    size_t nBytes = static_cast<size_t>(nElem) * sizeof(ElemT);

    // 自造输入：递增序列（fp16 位型用连续 uint16 值，落在正常数区避免 NaN/Inf）
    std::vector<ElemT> xHost(nElem);
    for (int64_t i = 0; i < nElem; ++i) {
        xHost[i] = static_cast<ElemT>((i * 7 + 1) & 0x3FFF);
    }
    std::vector<ElemT> yHost(nElem, 0);

    // host golden：y[i0,i1,i3,i2] = x[i0,i1,i2,i3]（末两轴互换）
    std::vector<ElemT> golden(nElem);
    for (int64_t i0 = 0; i0 < DIM0; ++i0) {
        for (int64_t i1 = 0; i1 < DIM1; ++i1) {
            for (int64_t i2 = 0; i2 < DIM2; ++i2) {
                for (int64_t i3 = 0; i3 < DIM3; ++i3) {
                    int64_t inIdx = ((i0 * DIM1 + i1) * DIM2 + i2) * DIM3 + i3;
                    int64_t outIdx = ((i0 * DIM1 + i1) * DIM3 + i3) * DIM2 + i2;
                    golden[outIdx] = xHost[inIdx];
                }
            }
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
    ret = aclnnLastTransposeNddmaBenchGetWorkspaceSize(x, IS_MTE3, y, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnLastTransposeNddmaBenchGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    ret = aclnnLastTransposeNddmaBench(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLastTransposeNddmaBench failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 拷回输出
    std::vector<ElemT> outHost(nElem);
    ret = aclrtMemcpy(outHost.data(), nBytes, yDev, nBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result D2H failed. ERROR: %d\n", ret); return ret);

    int mismatch = 0;
    for (int64_t i = 0; i < nElem; i++) {
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
        LOG_PRINT("[PASS] transpose[0,1,3,2] output matches golden (%lld elems)\n",
                  static_cast<long long>(nElem));
        return 0;
    }
    LOG_PRINT("[FAIL] %d elems mismatch out of %lld\n", mismatch, static_cast<long long>(nElem));
    return 1;
}
