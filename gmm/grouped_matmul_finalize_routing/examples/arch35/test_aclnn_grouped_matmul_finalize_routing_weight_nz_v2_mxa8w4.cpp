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
 * \file test_aclnn_grouped_matmul_finalize_routing_weight_nz_v2_mxa8w4.cpp
 * \brief MxA8W4数据流示例程序
 *
 * MxA8W4数据流说明：
 * - x1: FLOAT8_E4M3FN, ND格式, shape=(m, k)
 * - x2: FLOAT4_E2M1, NZ_C0_32格式, viewShape=(e, n, k), transposeX2=true
 * - scale: FLOAT8_E8M0, ND格式, shape=(e, n, ceil(k/64), 2)
 * - pertokenScale: FLOAT8_E8M0, ND格式, shape=(m, ceil(k/64), 2)
 *
 * 约束条件：
 * - k % 32 == 0
 * - n % 32 == 0
 * - e <= 1024
 * - transposeX2固定为true
 * - offsetOptional必须为nullptr
 */

#include <iostream>
#include <memory>
#include <vector>
#include <cmath>

#include "acl/acl.h"
#include "aclnnop/aclnn_grouped_matmul_finalize_routing_weight_nz_v2.h"

#define CHECK_RET(cond, return_expr)                                                                                   \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            return_expr;                                                                                               \
        }                                                                                                              \
    } while (0)

#define CHECK_FREE_RET(cond, return_expr)                                                                              \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            Finalize(deviceId, stream);                                                                                \
            return_expr;                                                                                               \
        }                                                                                                              \
    } while (0)

#define LOG_PRINT(message, ...)                                                                                        \
    do {                                                                                                               \
        printf(message, ##__VA_ARGS__);                                                                                \
    } while (0)

// 计算Tensor形状对应的总元素数
int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

// 向上取整除法
int64_t CeilDiv(int64_t a, int64_t b)
{
    return (a + b - 1) / b;
}

// 初始化AscendCL环境
int Init(int32_t deviceId, aclrtStream *stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

// 释放资源
void Finalize(int32_t deviceId, aclrtStream stream)
{
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

// 创建ND格式的aclTensor
template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

// 创建NZ_C0_32格式的weight aclTensor（MxA8W4专用）
// viewShape=(e, n, k), storageShape=(e, ceil(k/32), ceil(n/16), 16, 32)
template <typename T>
int CreateAclTensorWeightNzA8W4(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                                aclDataType dataType, aclTensor **tensor)
{
    int64_t e = shape[0];
    int64_t n = shape[1];
    int64_t k = shape[2];
    // MxA8W4: NZ_C0_32格式，C0=32（按float8_e4m3计算）
    std::vector<int64_t> storageShape = {e, CeilDiv(k, 32), CeilDiv(n, 16), 16L, 32L};

    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_FRACTAL_NZ,
                              storageShape.data(), storageShape.size(), *deviceAddr);
    return 0;
}

int main()
{
    // 1. 初始化device/stream
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init stream failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出（MxA8W4数据流）
    // MxA8W4约束：k % 32 == 0, n % 32 == 0, e <= 1024
    int64_t m = 192;
    int64_t k = 2048;
    int64_t n = 7168;
    int64_t e = 4;
    int64_t batch = 24;
    int64_t bsdp = 8;
    int64_t dtype = 0;
    float shareInputWeight = 1.0;
    int64_t sharedInputOffset = 0;
    bool transposeX = false;
    bool transposeW = true; // MxA8W4场景固定为true
    int64_t groupListType = 1;
    int64_t ceil_k_64 = CeilDiv(k, 64);

    std::vector<int64_t> xShape = {m, k};
    std::vector<int64_t> wShape = {e, n, k}; // viewShape，transposeX2=true
    std::vector<int64_t> scaleShape = {e, n, ceil_k_64, 2};
    std::vector<int64_t> pertokenScaleShape = {m, ceil_k_64, 2};
    std::vector<int64_t> groupListShape = {e};
    std::vector<int64_t> sharedInputShape = {bsdp, n};
    std::vector<int64_t> logitShape = {m};
    std::vector<int64_t> rowIndexShape = {m};
    std::vector<int64_t> outShape = {batch, n};

    void *xDeviceAddr = nullptr;
    void *wDeviceAddr = nullptr;
    void *scaleDeviceAddr = nullptr;
    void *pertokenScaleDeviceAddr = nullptr;
    void *groupListDeviceAddr = nullptr;
    void *sharedInputDeviceAddr = nullptr;
    void *logitDeviceAddr = nullptr;
    void *rowIndexDeviceAddr = nullptr;
    void *outDeviceAddr = nullptr;

    aclTensor *x = nullptr;
    aclTensor *w = nullptr;
    aclTensor *groupList = nullptr;
    aclTensor *scale = nullptr;
    aclTensor *pertokenScale = nullptr;
    aclTensor *sharedInput = nullptr;
    aclTensor *logit = nullptr;
    aclTensor *rowIndex = nullptr;
    aclTensor *out = nullptr;

    // MxA8W4数据类型：x1=FLOAT8_E4M3FN, x2=FLOAT4_E2M1, scale=FLOAT8_E8M0, pertokenScale=FLOAT8_E8M0
    // FLOAT8_E4M3FN和FLOAT8_E8M0使用uint8_t存储（1字节）
    // FLOAT4_E2M1使用uint8_t存储（2个FP4打包为1字节）
    std::vector<uint8_t> xHostData(GetShapeSize(xShape));
    std::vector<uint8_t> wHostData(GetShapeSize(wShape));
    std::vector<uint8_t> scaleHostData(GetShapeSize(scaleShape));
    std::vector<uint8_t> pertokenScaleHostData(GetShapeSize(pertokenScaleShape));
    std::vector<int64_t> groupListHostData(GetShapeSize(groupListShape));
    groupListHostData[0] = 64;

    std::vector<uint16_t> sharedInputHostData(GetShapeSize(sharedInputShape));
    std::vector<float> logitHostData(GetShapeSize(logitShape));
    std::vector<float> rowIndexHostData(GetShapeSize(rowIndexShape));
    std::vector<float> outHostData(GetShapeSize(outShape));

    // 创建x aclTensor (FLOAT8_E4M3FN, ND格式)
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &x);
    CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);
    // 创建NZ_C0_32格式的w aclTensor (FLOAT4_E2M1)
    ret = CreateAclTensorWeightNzA8W4(wHostData, wShape, &wDeviceAddr, aclDataType::ACL_FLOAT4_E2M1, &w);
    CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);
    // 创建scale aclTensor (FLOAT8_E8M0, ND格式)
    ret = CreateAclTensor(scaleHostData, scaleShape, &scaleDeviceAddr, aclDataType::ACL_FLOAT8_E8M0, &scale);
    CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);
    // 创建pertokenScale aclTensor (FLOAT8_E8M0, ND格式)
    ret = CreateAclTensor(pertokenScaleHostData, pertokenScaleShape, &pertokenScaleDeviceAddr,
                          aclDataType::ACL_FLOAT8_E8M0, &pertokenScale);
    CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);
    // 创建groupList aclTensor
    ret = CreateAclTensor(groupListHostData, groupListShape, &groupListDeviceAddr, aclDataType::ACL_INT64, &groupList);
    CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);
    // 创建sharedInput aclTensor
    ret = CreateAclTensor(sharedInputHostData, sharedInputShape, &sharedInputDeviceAddr, aclDataType::ACL_BF16,
                          &sharedInput);
    CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);
    // 创建logit aclTensor
    ret = CreateAclTensor(logitHostData, logitShape, &logitDeviceAddr, aclDataType::ACL_FLOAT, &logit);
    CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);
    // 创建rowIndex aclTensor
    ret = CreateAclTensor(rowIndexHostData, rowIndexShape, &rowIndexDeviceAddr, aclDataType::ACL_INT64, &rowIndex);
    CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);
    // 创建out aclTensor
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
    CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN算子库API
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    void *workspaceAddr = nullptr;

    // 调用aclnnGroupedMatmulFinalizeRoutingWeightNzV2第一段接口
    // MxA8W4场景：offsetOptional、antiquantScaleOptional、antiquantOffsetOptional必须为nullptr
    ret = aclnnGroupedMatmulFinalizeRoutingWeightNzV2GetWorkspaceSize(
        x, w, scale, nullptr, nullptr, nullptr, nullptr, pertokenScale, groupList, sharedInput, logit, rowIndex, dtype,
        shareInputWeight, sharedInputOffset, transposeX, transposeW, groupListType, nullptr, out, &workspaceSize,
        &executor);
    CHECK_FREE_RET(ret == ACL_SUCCESS,
                   LOG_PRINT("aclnnGroupedMatmulFinalizeRoutingWeightNzV2GetWorkspaceSize failed. ERROR: %d\n", ret);
                   return ret);

    // 根据第一段接口计算出的workspaceSize申请device内存
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 调用aclnnGroupedMatmulFinalizeRoutingWeightNzV2第二段接口
    ret = aclnnGroupedMatmulFinalizeRoutingWeightNzV2(workspaceAddr, workspaceSize, executor, stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS,
                   LOG_PRINT("aclnnGroupedMatmulFinalizeRoutingWeightNzV2 failed. ERROR: %d\n", ret);
                   return ret);

    // 4. 同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧
    auto size = GetShapeSize(outShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
                   return ret);
    for (int64_t i = 0; i < std::min((int64_t)10, size); i++) {
        LOG_PRINT("result[%lld] is: %f\n", i, resultData[i]);
    }

    // 6. 释放aclTensor资源
    aclDestroyTensor(x);
    aclDestroyTensor(w);
    aclDestroyTensor(scale);
    aclDestroyTensor(pertokenScale);
    aclDestroyTensor(groupList);
    aclDestroyTensor(sharedInput);
    aclDestroyTensor(logit);
    aclDestroyTensor(rowIndex);
    aclDestroyTensor(out);

    // 7. 释放device资源
    aclrtFree(xDeviceAddr);
    aclrtFree(wDeviceAddr);
    aclrtFree(scaleDeviceAddr);
    aclrtFree(pertokenScaleDeviceAddr);
    aclrtFree(groupListDeviceAddr);
    aclrtFree(sharedInputDeviceAddr);
    aclrtFree(logitDeviceAddr);
    aclrtFree(rowIndexDeviceAddr);
    aclrtFree(outDeviceAddr);

    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
