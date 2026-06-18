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
 * ACLNN smoke demo for GroupedMatmulActivationQuant WeightNz.
 *
 * This demo only checks whether the full aclnn -> tiling -> kernel chain can run
 * and whether y/yScale can be copied back to host. It does not check numerical
 * accuracy; use torch_eager_demo.py together with golden.py for that.
 *
 * Example build command on an Ascend 950 environment, adjust CANN paths if needed:
 *
 *   g++ -std=c++17 aclnn_weight_nz_demo.cpp -o aclnn_weight_nz_demo \
 *       -I/usr/local/Ascend/cann/latest/include \
 *       -I/usr/local/Ascend/cann/latest/include/aclnnop \
 *       -L/usr/local/Ascend/cann/latest/lib64 \
 *       -lascendcl -lopapi_transformer
 *
 * If you install a custom package, replace -lopapi_transformer with the custom
 * opapi library, for example:
 *
 *       /usr/local/Ascend/cann/latest/opp/vendors/custom_transformer/op_api/lib/libcust_opapi.so
 *
 * Run:
 *
 *   ./aclnn_weight_nz_demo
 */

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_grouped_matmul_activation_quant_weight_nz.h"

#define LOG_PRINT(message, ...) \
    do { \
        std::printf("[GMMAQ aclnn demo] " message, ##__VA_ARGS__); \
    } while (0)

#define CHECK_ACL_RET(expr) \
    do { \
        auto _ret = (expr); \
        if (_ret != ACL_SUCCESS) { \
            LOG_PRINT("%s failed, ret=%d\n", #expr, static_cast<int>(_ret)); \
            return _ret; \
        } \
    } while (0)

namespace {
constexpr int64_t E = 1;
constexpr int64_t M = 64;
constexpr int64_t K = 128;
constexpr int64_t N = 128;
constexpr int64_t MX_BLOCK = 64;
constexpr int64_t MX_SCALE_PAIR = 2;
constexpr int64_t NZ_C0 = 16;
constexpr int64_t NZ_LAST = 32;
constexpr int64_t PRINT_NUM = 16;
constexpr uint8_t FP8_E4M3_ONE = 0x38;
constexpr uint8_t E8M0_ONE = 127;

int64_t CeilDiv(int64_t a, int64_t b)
{
    return (a + b - 1) / b;
}

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t size = 1;
    for (auto dim : shape) {
        size *= dim;
    }
    return size;
}

std::vector<int64_t> MakeStrides(const std::vector<int64_t> &shape)
{
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    return strides;
}

template <typename T>
aclError CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, aclDataType dataType,
                         aclFormat formatType, void **deviceAddr, aclTensor **tensor)
{
    const size_t byteSize = static_cast<size_t>(GetShapeSize(shape)) * sizeof(T);
    CHECK_ACL_RET(aclrtMalloc(deviceAddr, byteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL_RET(aclrtMemcpy(*deviceAddr, byteSize, hostData.data(), byteSize, ACL_MEMCPY_HOST_TO_DEVICE));

    auto strides = MakeStrides(shape);
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, formatType, shape.data(),
                              shape.size(), *deviceAddr);
    if (*tensor == nullptr) {
        LOG_PRINT("aclCreateTensor failed\n");
        return ACL_ERROR_FAILURE;
    }
    return ACL_SUCCESS;
}

template <typename T>
aclError CreateAclTensorWithStorageShape(const std::vector<T> &hostData, const std::vector<int64_t> &viewShape,
                                         const std::vector<int64_t> &storageShape,
                                         const std::vector<int64_t> &strides, aclDataType dataType,
                                         aclFormat formatType, void **deviceAddr, aclTensor **tensor)
{
    const size_t byteSize = static_cast<size_t>(GetShapeSize(storageShape)) * sizeof(T);
    CHECK_ACL_RET(aclrtMalloc(deviceAddr, byteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL_RET(aclrtMemcpy(*deviceAddr, byteSize, hostData.data(), byteSize, ACL_MEMCPY_HOST_TO_DEVICE));

    *tensor = aclCreateTensor(viewShape.data(), viewShape.size(), dataType, strides.data(), 0, formatType,
                              storageShape.data(), storageShape.size(), *deviceAddr);
    if (*tensor == nullptr) {
        LOG_PRINT("aclCreateTensor with storage shape failed\n");
        return ACL_ERROR_FAILURE;
    }
    return ACL_SUCCESS;
}

aclError CreateSingleTensorList(aclTensor *tensor, aclTensorList **tensorList)
{
    aclTensor *tensors[] = {tensor};
    *tensorList = aclCreateTensorList(tensors, 1);
    if (*tensorList == nullptr) {
        LOG_PRINT("aclCreateTensorList failed\n");
        return ACL_ERROR_FAILURE;
    }
    return ACL_SUCCESS;
}

void DestroyTensor(aclTensor *tensor)
{
    if (tensor != nullptr) {
        aclDestroyTensor(tensor);
    }
}

void DestroyTensorList(aclTensorList *tensorList)
{
    if (tensorList != nullptr) {
        aclDestroyTensorList(tensorList);
    }
}

void FreeDevice(void *deviceAddr)
{
    if (deviceAddr != nullptr) {
        aclrtFree(deviceAddr);
    }
}

aclError RunDemo(int32_t deviceId)
{
    aclrtStream stream = nullptr;
    void *xDeviceAddr = nullptr;
    void *groupListDeviceAddr = nullptr;
    void *weightDeviceAddr = nullptr;
    void *weightScaleDeviceAddr = nullptr;
    void *xScaleDeviceAddr = nullptr;
    void *yDeviceAddr = nullptr;
    void *yScaleDeviceAddr = nullptr;
    void *workspaceAddr = nullptr;

    aclTensor *x = nullptr;
    aclTensor *groupList = nullptr;
    aclTensor *weightTensor = nullptr;
    aclTensor *weightScaleTensor = nullptr;
    aclTensor *xScale = nullptr;
    aclTensor *y = nullptr;
    aclTensor *yScale = nullptr;
    aclTensorList *weight = nullptr;
    aclTensorList *weightScale = nullptr;

    auto cleanup = [&]() {
        DestroyTensorList(weight);
        DestroyTensorList(weightScale);
        DestroyTensor(x);
        DestroyTensor(groupList);
        DestroyTensor(weightTensor);
        DestroyTensor(weightScaleTensor);
        DestroyTensor(xScale);
        DestroyTensor(y);
        DestroyTensor(yScale);
        FreeDevice(workspaceAddr);
        FreeDevice(xDeviceAddr);
        FreeDevice(groupListDeviceAddr);
        FreeDevice(weightDeviceAddr);
        FreeDevice(weightScaleDeviceAddr);
        FreeDevice(xScaleDeviceAddr);
        FreeDevice(yDeviceAddr);
        FreeDevice(yScaleDeviceAddr);
        if (stream != nullptr) {
            aclrtDestroyStream(stream);
        }
        aclrtResetDevice(deviceId);
        aclFinalize();
    };

    LOG_PRINT("init ACL, deviceId=%d\n", deviceId);
    CHECK_ACL_RET(aclInit(nullptr));
    CHECK_ACL_RET(aclrtSetDevice(deviceId));
    CHECK_ACL_RET(aclrtCreateStream(&stream));

    const int64_t kScaleBlock = CeilDiv(K, MX_BLOCK);
    const int64_t nScaleBlock = CeilDiv(N, MX_BLOCK);
    const std::vector<int64_t> xShape = {M, K};
    const std::vector<int64_t> groupListShape = {E};
    const std::vector<int64_t> weightViewShape = {E, K, N};
    const std::vector<int64_t> weightStorageShape = {E, CeilDiv(N, NZ_LAST), CeilDiv(K, NZ_C0), NZ_C0, NZ_LAST};
    const std::vector<int64_t> weightStrides = {K * N, N, 1};
    const std::vector<int64_t> weightScaleShape = {E, kScaleBlock, N, MX_SCALE_PAIR};
    const std::vector<int64_t> xScaleShape = {M, kScaleBlock, MX_SCALE_PAIR};
    const std::vector<int64_t> yShape = {M, N};
    const std::vector<int64_t> yScaleShape = {M, nScaleBlock, MX_SCALE_PAIR};

    LOG_PRINT("create tensors: E=%ld, M=%ld, K=%ld, N=%ld\n", E, M, K, N);
    std::vector<uint8_t> xHostData(GetShapeSize(xShape), FP8_E4M3_ONE);
    std::vector<int64_t> groupListHostData = {M};
    std::vector<uint8_t> weightHostData(GetShapeSize(weightStorageShape), FP8_E4M3_ONE);
    std::vector<uint8_t> weightScaleHostData(GetShapeSize(weightScaleShape), E8M0_ONE);
    std::vector<uint8_t> xScaleHostData(GetShapeSize(xScaleShape), E8M0_ONE);
    std::vector<uint8_t> yHostData(GetShapeSize(yShape), 0);
    std::vector<uint8_t> yScaleHostData(GetShapeSize(yScaleShape), 0);

    CHECK_ACL_RET(CreateAclTensor(xHostData, xShape, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND, &xDeviceAddr, &x));
    CHECK_ACL_RET(CreateAclTensor(groupListHostData, groupListShape, ACL_INT64, ACL_FORMAT_ND, &groupListDeviceAddr,
                                  &groupList));
    CHECK_ACL_RET(CreateAclTensorWithStorageShape(weightHostData, weightViewShape, weightStorageShape, weightStrides,
                                                  ACL_FLOAT8_E4M3FN, ACL_FORMAT_FRACTAL_NZ, &weightDeviceAddr,
                                                  &weightTensor));
    CHECK_ACL_RET(CreateSingleTensorList(weightTensor, &weight));
    CHECK_ACL_RET(CreateAclTensor(weightScaleHostData, weightScaleShape, ACL_FLOAT8_E8M0, ACL_FORMAT_ND,
                                  &weightScaleDeviceAddr, &weightScaleTensor));
    CHECK_ACL_RET(CreateSingleTensorList(weightScaleTensor, &weightScale));
    CHECK_ACL_RET(CreateAclTensor(xScaleHostData, xScaleShape, ACL_FLOAT8_E8M0, ACL_FORMAT_ND, &xScaleDeviceAddr,
                                  &xScale));
    CHECK_ACL_RET(CreateAclTensor(yHostData, yShape, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND, &yDeviceAddr, &y));
    CHECK_ACL_RET(CreateAclTensor(yScaleHostData, yScaleShape, ACL_FLOAT8_E8M0, ACL_FORMAT_ND, &yScaleDeviceAddr,
                                  &yScale));

    const aclTensorList *biasOptional = nullptr;
    const aclIntArray *tuningConfig = nullptr;
    const char *activationType = "gelu_tanh";
    const bool transposeWeight = false;
    const int64_t groupListType = 0;
    const char *quantMode = "MX";
    const int64_t yDtype = 36;
    const char *roundMode = "rint";
    const int64_t scaleAlg = 0;
    const float dstTypeMax = 0.0F;

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    LOG_PRINT("call aclnnGroupedMatmulActivationQuantWeightNzGetWorkspaceSize\n");
    CHECK_ACL_RET(aclnnGroupedMatmulActivationQuantWeightNzGetWorkspaceSize(
        x, groupList, weight, weightScale, biasOptional, xScale, activationType, transposeWeight, groupListType,
        tuningConfig, quantMode, yDtype, roundMode, scaleAlg, dstTypeMax, y, yScale, &workspaceSize, &executor));
    LOG_PRINT("workspaceSize=%lu, executor=%p\n", workspaceSize, executor);

    if (workspaceSize > 0) {
        CHECK_ACL_RET(aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }

    LOG_PRINT("call aclnnGroupedMatmulActivationQuantWeightNz\n");
    CHECK_ACL_RET(aclnnGroupedMatmulActivationQuantWeightNz(workspaceAddr, workspaceSize, executor, stream));

    LOG_PRINT("synchronize stream to force tiling/kernel execution\n");
    CHECK_ACL_RET(aclrtSynchronizeStream(stream));

    std::vector<uint8_t> yOut(GetShapeSize(yShape), 0);
    std::vector<uint8_t> yScaleOut(GetShapeSize(yScaleShape), 0);
    CHECK_ACL_RET(aclrtMemcpy(yOut.data(), yOut.size() * sizeof(uint8_t), yDeviceAddr,
                              yOut.size() * sizeof(uint8_t), ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL_RET(aclrtMemcpy(yScaleOut.data(), yScaleOut.size() * sizeof(uint8_t), yScaleDeviceAddr,
                              yScaleOut.size() * sizeof(uint8_t), ACL_MEMCPY_DEVICE_TO_HOST));

    const int64_t yPrintNum = std::min<int64_t>(PRINT_NUM, static_cast<int64_t>(yOut.size()));
    const int64_t yScalePrintNum = std::min<int64_t>(PRINT_NUM, static_cast<int64_t>(yScaleOut.size()));
    LOG_PRINT("first %ld y bytes:", yPrintNum);
    for (int64_t i = 0; i < yPrintNum; ++i) {
        std::printf(" %u", static_cast<unsigned>(yOut[i]));
    }
    std::printf("\n");
    LOG_PRINT("first %ld yScale bytes:", yScalePrintNum);
    for (int64_t i = 0; i < yScalePrintNum; ++i) {
        std::printf(" %u", static_cast<unsigned>(yScaleOut[i]));
    }
    std::printf("\n");
    LOG_PRINT("demo passed\n");

    cleanup();
    return ACL_SUCCESS;
}
} // namespace

int main(int argc, char **argv)
{
    int32_t deviceId = 0;
    if (argc > 1) {
        deviceId = std::atoi(argv[1]);
    }
    auto ret = RunDemo(deviceId);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("demo failed, ret=%d\n", static_cast<int>(ret));
    }
    return static_cast<int>(ret);
}
