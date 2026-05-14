/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aclnn_grouped_matmul_swiglu_quant_weight_nz_v2_mxA8W4.cpp
 * \brief
 */

#include <iostream>
#include <memory>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_grouped_matmul_swiglu_quant_weight_nz_v2.h"

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

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

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

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclFormat FormatType, aclTensor **tensor)
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

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, FormatType, shape.data(),
                              shape.size(), *deviceAddr);
    return 0;
}

template <typename T>
int CreateAclTensorList(const std::vector<std::vector<T>> &hostData, const std::vector<std::vector<int64_t>> &shapes,
                        void **deviceAddr, aclDataType dataType, aclFormat FormatType, aclTensorList **tensor)
{
    int size = shapes.size();
    aclTensor *tensors[size];
    for (int i = 0; i < size; i++) {
        int ret = CreateAclTensor<T>(hostData[i], shapes[i], deviceAddr + i, dataType, FormatType, tensors + i);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }
    *tensor = aclCreateTensorList(tensors, size);
    return ACL_SUCCESS;
}

template <typename T1, typename T2>
auto CeilDiv(T1 a, T2 b) -> T1
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

void Finalize(int32_t deviceId, aclrtStream stream)
{
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

int aclnnGroupedMatmulSwigluQuantWeightNzV2MxA8W4Test(int32_t deviceId, aclrtStream &stream)
{
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    int64_t E = 8;
    int64_t M = 2048;
    int64_t N = 4096;
    int64_t K = 4096;

    std::vector<int64_t> xShape = {M, K};
    std::vector<std::vector<int64_t>> weightShape = {{E, N / 32, K / 16, 16, 32}};
    std::vector<std::vector<int64_t>> weightScaleShape = {{E, N, CeilDiv(K, 64), 2}};
    std::vector<int64_t> xScaleShape = {M, CeilDiv(K, 64), 2};
    std::vector<int64_t> groupListShape = {E};
    std::vector<int64_t> outputShape = {M, N / 2};
    std::vector<int64_t> outputScaleShape = {M, CeilDiv((N / 2), 64), 2};

    void *xDeviceAddr = nullptr;
    void *weightDeviceAddr = nullptr;
    void *weightScaleDeviceAddr = nullptr;
    void *xScaleDeviceAddr = nullptr;
    void *groupListDeviceAddr = nullptr;
    void *outputDeviceAddr = nullptr;
    void *outputScaleDeviceAddr = nullptr;

    aclTensor *x = nullptr;
    aclTensorList *weight = nullptr;
    aclTensorList *weightScale = nullptr;
    aclTensor *xScale = nullptr;
    aclTensor *groupList = nullptr;
    aclTensor *output = nullptr;
    aclTensor *outputScale = nullptr;

    std::vector<uint8_t> xHostData(M * K, 1);
    std::vector<uint8_t> weightHostData(E * (N / 32) * (K / 16) * 16 * 32, 1);
    std::vector<uint8_t> weightScaleHostData(E * N * CeilDiv(K, 64) * 2, 1);
    std::vector<uint8_t> xScaleHostData(M * CeilDiv(K, 64) * 2, 1);
    std::vector<int64_t> groupListHostData(E, 256);
    std::vector<uint8_t> outputHostData(M * N / 2, 1);
    std::vector<uint8_t> outputScaleHostData(M * CeilDiv((N / 2), 64) * 2, 1);

    std::vector<int64_t> tuningConfigData = {1};
    aclIntArray *tuningConfig = aclCreateIntArray(tuningConfigData.data(), 1);

    int64_t quantMode = 2;
    int64_t dequantMode = 2;
    int64_t dequantDtype = 0;
    int64_t groupListType = 1;

    ret = CreateAclTensor<uint8_t>(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN,
                                   aclFormat::ACL_FORMAT_ND, &x);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> xTensorPtr(x, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> xDeviceAddrPtr(xDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<std::vector<uint8_t>> weightHostDataList = {weightHostData};
    ret = CreateAclTensorList<uint8_t>(weightHostDataList, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT4_E2M1,
                                       aclFormat::ACL_FORMAT_FRACTAL_NZ, &weight);
    std::unique_ptr<aclTensorList, aclnnStatus (*)(const aclTensorList *)> weightTensorListPtr(weight,
                                                                                               aclDestroyTensorList);
    std::unique_ptr<void, aclError (*)(void *)> weightDeviceAddrPtr(weightDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<std::vector<uint8_t>> weightScaleHostDataList = {weightScaleHostData};
    ret = CreateAclTensorList<uint8_t>(weightScaleHostDataList, weightScaleShape, &weightScaleDeviceAddr,
                                       aclDataType::ACL_FLOAT8_E8M0, aclFormat::ACL_FORMAT_ND, &weightScale);
    std::unique_ptr<aclTensorList, aclnnStatus (*)(const aclTensorList *)> weightScaleTensorListPtr(
        weightScale, aclDestroyTensorList);
    std::unique_ptr<void, aclError (*)(void *)> weightScaleDeviceAddrPtr(weightScaleDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor<uint8_t>(xScaleHostData, xScaleShape, &xScaleDeviceAddr, aclDataType::ACL_FLOAT8_E8M0,
                                   aclFormat::ACL_FORMAT_ND, &xScale);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> xScaleTensorPtr(xScale, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> xScaleDeviceAddrPtr(xScaleDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor<int64_t>(groupListHostData, groupListShape, &groupListDeviceAddr, aclDataType::ACL_INT64,
                                   aclFormat::ACL_FORMAT_ND, &groupList);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> groupListTensorPtr(groupList, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> groupListDeviceAddrPtr(groupListDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor<uint8_t>(outputHostData, outputShape, &outputDeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN,
                                   aclFormat::ACL_FORMAT_ND, &output);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> outputTensorPtr(output, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> outputDeviceAddrPtr(outputDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor<uint8_t>(outputScaleHostData, outputScaleShape, &outputScaleDeviceAddr,
                                   aclDataType::ACL_FLOAT8_E8M0, aclFormat::ACL_FORMAT_ND, &outputScale);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> outputScaleTensorPtr(outputScale, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> outputScaleDeviceAddrPtr(outputScaleDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    void *workspaceAddr = nullptr;

    ret = aclnnGroupedMatmulSwigluQuantWeightNzV2GetWorkspaceSize(
        x, weight, weightScale, nullptr, nullptr, xScale, nullptr, groupList, dequantMode, dequantDtype, quantMode,
        groupListType, tuningConfig, output, outputScale, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnGroupedMatmulSwigluQuantWeightNzV2GetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnGroupedMatmulSwigluQuantWeightNzV2(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulSwigluQuantWeightNzV2 failed. ERROR: %d\n", ret);
              return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    auto size = GetShapeSize(outputShape);
    std::vector<uint8_t> outputData(size, 0);
    ret = aclrtMemcpy(outputData.data(), size * sizeof(outputData[0]), outputDeviceAddr, size * sizeof(outputData[0]),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy outputData from device to host failed. ERROR: %d\n", ret);
              return ret);
    for (int64_t j = 0; j < size; j++) {
        LOG_PRINT("result[%ld] is: %d\n", j, outputData[j]);
    }

    size = GetShapeSize(outputScaleShape);
    std::vector<uint8_t> outputScaleData(size, 0);
    ret = aclrtMemcpy(outputScaleData.data(), size * sizeof(outputScaleData[0]), outputScaleDeviceAddr,
                      size * sizeof(outputScaleData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy outputScaleData from device to host failed. ERROR: %d\n", ret);
              return ret);
    for (int64_t j = 0; j < size; j++) {
        LOG_PRINT("result[%ld] is: %d\n", j, outputScaleData[j]);
    }

    aclDestroyIntArray(tuningConfig);
    return ACL_SUCCESS;
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = aclnnGroupedMatmulSwigluQuantWeightNzV2MxA8W4Test(deviceId, stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS,
                   LOG_PRINT("aclnnGroupedMatmulSwigluQuantWeightNzV2MxA8W4Test failed. ERROR: %d\n", ret);
                   return ret);

    Finalize(deviceId, stream);
    return 0;
}