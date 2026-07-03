/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <memory>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_matmul_swiglu.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)          \
    do {                                 \
        printf(message, ##__VA_ARGS__);  \
        fflush(stdout);                  \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
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

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor returned nullptr.\n"); return -1);
    return 0;
}

int aclnnMatmulSwigluTest(int32_t deviceId, aclrtStream& stream)
{
    LOG_PRINT("[stage] Init ACL\n");
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 形状: x[M,K], weight[K,2N], bias[2N], 输出 y[M,N]
    const int64_t M = 32, K = 128, N = 64;
    std::vector<int64_t> xShape = {M, K};
    std::vector<int64_t> wShape = {K, 2 * N};
    std::vector<int64_t> bShape = {2 * N};
    std::vector<int64_t> yShape = {M, N};

    // fp32 验证: x/weight/y 为 fp32, bias 固定 fp32。x=w=0.1 时 gate=up=K*0.01=1.28,
    // 每个 y 应≈SiLU(1.28)*1.28≈1.282
    std::vector<float> xHost(GetShapeSize(xShape), 0.1f);
    std::vector<float> wHost(GetShapeSize(wShape), 0.1f);
    std::vector<float> bHost(GetShapeSize(bShape), 0.0f);

    aclTensor* x = nullptr;
    aclTensor* weight = nullptr;
    aclTensor* bias = nullptr;
    aclTensor* out = nullptr;
    void* xDev = nullptr;
    void* wDev = nullptr;
    void* biasDev = nullptr;
    void* outDev = nullptr;

    LOG_PRINT("[stage] Create x tensor\n");
    ret = CreateAclTensor(xHost, xShape, &xDev, aclDataType::ACL_FLOAT, &x);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> xPtr(x, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> xDevPtr(xDev, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    LOG_PRINT("[stage] Create weight tensor\n");
    ret = CreateAclTensor(wHost, wShape, &wDev, aclDataType::ACL_FLOAT, &weight);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> wPtr(weight, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> wDevPtr(wDev, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    LOG_PRINT("[stage] Create bias tensor\n");
    ret = CreateAclTensor(bHost, bShape, &biasDev, aclDataType::ACL_FLOAT, &bias);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> biasPtr(bias, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> biasDevPtr(biasDev, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<float> outHost(GetShapeSize(yShape), 0.0f);
    LOG_PRINT("[stage] Create output tensor\n");
    ret = CreateAclTensor(outHost, yShape, &outDev, aclDataType::ACL_FLOAT, &out);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> outPtr(out, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> outDevPtr(outDev, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    LOG_PRINT("[stage] Tensor handles x=%p weight=%p bias=%p out=%p\n",
              static_cast<void*>(x), static_cast<void*>(weight), static_cast<void*>(bias), static_cast<void*>(out));

    // 调用 aclnnMatmulSwiglu 两段式接口。
    // 生成的接口签名(见 op_api/include/aclnn_matmul_swiglu.h)为:
    //   (x, weight, bias, transposeWeight, out, workspaceSize, executor)
    // 即 输入 -> 属性 -> 输出。bias 虽为可选, 但自动生成的 GetWorkspaceSize 会对其做非空校验,
    // 传 nullptr 会返回 161001 (ACLNN_ERR_PARAM_NULLPTR), 故这里传入一个全 0 的 bias。
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    LOG_PRINT("[stage] GetWorkspaceSize\n");
    ret = aclnnMatmulSwigluGetWorkspaceSize(x, weight, bias, false, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMatmulSwigluGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    LOG_PRINT("[stage] Workspace size: %lu\n", workspaceSize);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        LOG_PRINT("[stage] Allocate workspace\n");
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("alloc workspace failed. ERROR: %d\n", ret); return ret);
    }
    std::unique_ptr<void, aclError (*)(void*)> wsPtr(workspaceAddr, aclrtFree);

    LOG_PRINT("[stage] Launch aclnnMatmulSwiglu\n");
    ret = aclnnMatmulSwiglu(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMatmulSwiglu failed. ERROR: %d\n", ret); return ret);

    LOG_PRINT("[stage] Synchronize stream\n");
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    auto size = GetShapeSize(yShape);
    std::vector<float> resultData(size, 0.0f);
    LOG_PRINT("[stage] Copy output to host\n");
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(float), outDev, size * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result failed. ERROR: %d\n", ret); return ret);
    // 打印前若干元素 + 统计, 便于核对是否≈1.282(均匀输入下所有 y 应相等)
    float ymin = resultData[0];
    float ymax = resultData[0];
    double ysum = 0.0;
    for (auto v : resultData) {
        ymin = v < ymin ? v : ymin;
        ymax = v > ymax ? v : ymax;
        ysum += v;
    }
    LOG_PRINT("y[0..3]=%.6f %.6f %.6f %.6f  (expect ~1.282)\n",
              resultData[0], resultData[1], resultData[2], resultData[3]);
    LOG_PRINT("y stats: min=%.6f max=%.6f mean=%.6f count=%ld\n",
              ymin, ymax, ysum / static_cast<double>(size), static_cast<long>(size));
    return 0;
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    auto ret = aclnnMatmulSwigluTest(deviceId, stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("test failed. ERROR: %d\n", ret);
    }
    // 无论成功与否都释放 ACL 资源, 避免失败路径泄漏。
    if (stream != nullptr) {
        aclrtDestroyStream(stream);
    }
    aclrtResetDevice(deviceId);
    aclFinalize();
    return ret;
}
