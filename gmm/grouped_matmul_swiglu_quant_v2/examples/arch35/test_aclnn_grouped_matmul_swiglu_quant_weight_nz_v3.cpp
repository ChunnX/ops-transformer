/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_grouped_matmul_swiglu_quant_weight_nz_v3.h"

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
    for (auto v : shape) {
        shapeSize *= v;
    }
    return shapeSize;
}

template <typename T1, typename T2>
auto Ceil(T1 a, T2 b) -> T1
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

int Init(int32_t deviceId, aclrtStream *stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ret=%d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ret=%d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ret=%d\n", ret); return ret);
    return ACL_SUCCESS;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &logicalShape,
                    const std::vector<int64_t> &storageShape, void **deviceAddr, aclDataType dataType,
                    aclFormat formatType, aclTensor **tensor, const std::vector<int64_t> *customStrides = nullptr)
{
    uint64_t size = GetShapeSize(storageShape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ret=%d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ret=%d\n", ret); return ret);
    std::vector<int64_t> strides(logicalShape.size(), 1);
    for (int64_t i = logicalShape.size() - 2; i >= 0; --i) {
        strides[i] = logicalShape[i + 1] * strides[i + 1];
    }
    if (customStrides != nullptr) {
        strides = *customStrides;
    }
    *tensor = aclCreateTensor(logicalShape.data(), logicalShape.size(), dataType, strides.data(), 0, formatType,
                              storageShape.data(), storageShape.size(), *deviceAddr);
    CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed\n"); return ACL_ERROR_FAILURE);
    return ACL_SUCCESS;
}

template <typename T>
int CreateAclTensorND(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                      aclDataType dataType, aclTensor **tensor)
{
    return CreateAclTensor(hostData, shape, shape, deviceAddr, dataType, ACL_FORMAT_ND, tensor);
}

template <typename T>
void PrintVector(const std::vector<T> &data, const std::string &name, int64_t printNum = 32)
{
    LOG_PRINT("======== %s ========\n", name.c_str());
    int64_t size = static_cast<int64_t>(data.size());
    int64_t limit = std::min(size, printNum);
    for (int64_t i = 0; i < limit; ++i) {
        LOG_PRINT("%s[%ld] = %d\n", name.c_str(), i, static_cast<int32_t>(data[i]));
    }
    LOG_PRINT("\n");
}

int main(int argc, char **argv)
{
    const int64_t dequantMode = 2;
    const int64_t dequantDtype = 0;
    const int64_t quantMode = 2;
    const int64_t groupListType = 0;
    const int64_t swigluMode = 2;
    const double clampLimit = 7.0;
    const double gluAlpha = 1.702;
    const double gluBias = 1.0;
    const char *roundMode = "rint";
    const double dstTypeMax = 0.0;
    const aclTensorList *weightAssistMatrix = nullptr;
    const aclTensor *bias = nullptr;
    const aclTensor *smoothScale = nullptr;
    const aclIntArray *tuningConfigOptional = nullptr;
    const bool transposeWeight = argc > 1 ? std::stoll(argv[1]) != 0 : false;
    const std::string dumpPrefix = argc > 2 ? argv[2] : "";
    const int64_t scaleAlg = argc > 3 ? std::stoll(argv[3]) : 0;
    if (scaleAlg < 0 || scaleAlg > 1) {
        LOG_PRINT("invalid scale_alg=%ld\n", scaleAlg);
        return 2;
    }
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init failed\n"); return ret);
    int64_t E = 1;
    int64_t M = 2048;
    int64_t K = 64;
    int64_t N = 128;

    std::vector<int64_t> xShape = {M, K};
    // The API entry view is always [E, K, N]. A transposed (ZN) weight is
    // distinguished by its view stride and physical FRACTAL_NZ storage.
    std::vector<int64_t> weightLogicalShape = {E, K, N};
    std::vector<int64_t> weightStorageShape = transposeWeight ?
                                                  std::vector<int64_t>{E, Ceil(K, 32), Ceil(N, 16), 16, 32} :
                                                  std::vector<int64_t>{E, Ceil(N, 32), Ceil(K, 16), 16, 32};
    const int64_t scaleK = Ceil(K, 64);
    std::vector<int64_t> weightScaleShape = {E, scaleK, N, 2};
    std::vector<int64_t> weightScaleStorageShape =
        transposeWeight ? std::vector<int64_t>{E, N, scaleK, 2} : weightScaleShape;
    std::vector<int64_t> xScaleShape = {M, Ceil(K, 64), 2};
    std::vector<int64_t> groupListShape = {E};
    std::vector<int64_t> outputShape = {M, N / 2};
    std::vector<int64_t> outputScaleShape = {M, Ceil(N / 2, 64), 2};

    constexpr uint8_t fp8Pattern[] = {0x38, 0xb8, 0x40, 0x30, 0xc0, 0x28, 0x3c, 0xbc, 0x48, 0xc8, 0x50, 0x20};
    std::vector<uint8_t> xHostData(GetShapeSize(xShape));
    std::vector<uint8_t> weightHostData(GetShapeSize(weightStorageShape));
    for (size_t i = 0; i < xHostData.size(); ++i) {
        xHostData[i] = fp8Pattern[(i * 7 + 3) % (sizeof(fp8Pattern) / sizeof(fp8Pattern[0]))];
    }
    for (size_t i = 0; i < weightHostData.size(); ++i) {
        weightHostData[i] = fp8Pattern[(i * 5 + 1) % (sizeof(fp8Pattern) / sizeof(fp8Pattern[0]))];
    }
    std::vector<uint8_t> weightScaleHostData(GetShapeSize(weightScaleShape), 127);
    std::vector<uint8_t> xScaleHostData(GetShapeSize(xScaleShape), 127);
    // std::vector<int64_t> groupListHostData(E, 1);
    std::vector<int64_t> groupListHostData = {M};
    std::vector<uint8_t> outputHostData(GetShapeSize(outputShape), 0);
    std::vector<uint8_t> outputScaleHostData(GetShapeSize(outputScaleShape), 0);

    void *xDeviceAddr = nullptr;
    void *weightDeviceAddr = nullptr;
    void *weightScaleDeviceAddr = nullptr;
    void *xScaleDeviceAddr = nullptr;
    void *groupListDeviceAddr = nullptr;
    void *outputDeviceAddr = nullptr;
    void *outputScaleDeviceAddr = nullptr;

    aclTensor *x = nullptr;
    aclTensor *weightTensor = nullptr;
    aclTensor *weightScaleTensor = nullptr;
    aclTensor *xScale = nullptr;
    aclTensor *groupList = nullptr;
    aclTensor *output = nullptr;
    aclTensor *outputScale = nullptr;

    aclTensorList *weight = nullptr;
    aclTensorList *weightScale = nullptr;

    ret = CreateAclTensorND<uint8_t>(xHostData, xShape, &xDeviceAddr, ACL_FLOAT8_E4M3FN, &x);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> xTensorPtr(x, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> xDeviceAddrPtr(xDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<int64_t> weightViewStrides =
        transposeWeight ? std::vector<int64_t>{K * N, 1, K} : std::vector<int64_t>{K * N, N, 1};
    ret = CreateAclTensor<uint8_t>(weightHostData, weightLogicalShape, weightStorageShape, &weightDeviceAddr,
                                   ACL_FLOAT8_E4M3FN, ACL_FORMAT_FRACTAL_NZ, &weightTensor, &weightViewStrides);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> weightTensorPtr(weightTensor, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> weightDeviceAddrPtr(weightDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    aclTensor *weightArray[] = {weightTensor};
    weight = aclCreateTensorList(weightArray, 1);
    CHECK_RET(weight != nullptr, LOG_PRINT("aclCreateTensorList(weight) failed\n"); return ACL_ERROR_FAILURE);
    std::unique_ptr<aclTensorList, aclnnStatus (*)(const aclTensorList *)> weightTensorListPtr(weight,
                                                                                               aclDestroyTensorList);
    // The ZN scale view is obtained by transposing an [E, N, scaleK, 2]
    // source. Its API entry view remains [E, scaleK, N, 2].
    std::vector<int64_t> weightScaleViewStrides = transposeWeight ?
                                                      std::vector<int64_t>{N * scaleK * 2, 2, scaleK * 2, 1} :
                                                      std::vector<int64_t>{scaleK * N * 2, N * 2, 2, 1};
    ret =
        CreateAclTensor<uint8_t>(weightScaleHostData, weightScaleShape, weightScaleStorageShape, &weightScaleDeviceAddr,
                                 ACL_FLOAT8_E8M0, ACL_FORMAT_ND, &weightScaleTensor, &weightScaleViewStrides);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> weightScaleTensorPtr(weightScaleTensor,
                                                                                        aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> weightScaleDeviceAddrPtr(weightScaleDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    aclTensor *weightScaleArray[] = {weightScaleTensor};
    weightScale = aclCreateTensorList(weightScaleArray, 1);
    CHECK_RET(weightScale != nullptr, LOG_PRINT("aclCreateTensorList(weightScale) failed\n"); return ACL_ERROR_FAILURE);
    std::unique_ptr<aclTensorList, aclnnStatus (*)(const aclTensorList *)> weightScaleTensorListPtr(
        weightScale, aclDestroyTensorList);

    ret = CreateAclTensorND<uint8_t>(xScaleHostData, xScaleShape, &xScaleDeviceAddr, ACL_FLOAT8_E8M0, &xScale);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> xScaleTensorPtr(xScale, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> xScaleDeviceAddrPtr(xScaleDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensorND<int64_t>(groupListHostData, groupListShape, &groupListDeviceAddr, ACL_INT64, &groupList);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> groupListTensorPtr(groupList, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> groupListDeviceAddrPtr(groupListDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensorND<uint8_t>(outputHostData, outputShape, &outputDeviceAddr, ACL_FLOAT8_E4M3FN, &output);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> outputTensorPtr(output, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> outputDeviceAddrPtr(outputDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensorND<uint8_t>(outputScaleHostData, outputScaleShape, &outputScaleDeviceAddr, ACL_FLOAT8_E8M0,
                                     &outputScale);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> outputScaleTensorPtr(outputScale, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> outputScaleDeviceAddrPtr(outputScaleDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    ret = aclnnGroupedMatmulSwigluQuantWeightNzV3GetWorkspaceSize(
        x, weight, weightScale, weightAssistMatrix, bias, xScale, smoothScale, groupList, dequantMode, dequantDtype,
        quantMode, groupListType, tuningConfigOptional, swigluMode, clampLimit, gluAlpha, gluBias, roundMode, scaleAlg,
        dstTypeMax, output, outputScale, &workspaceSize, &executor);

    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("GetWorkspaceSize failed. ret=%d\n", ret); return ret);
    LOG_PRINT("workspaceSize = %lu\n", workspaceSize);
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("workspace malloc failed\n"); return ret);
    }

    ret = aclnnGroupedMatmulSwigluQuantWeightNzV3(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("run op failed. ret=%d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("sync stream failed ret=%d\n", ret);
        std::_Exit(3);
    }
    LOG_PRINT("run success\n");

    ret = aclrtMemcpy(outputHostData.data(), outputHostData.size() * sizeof(uint8_t), outputDeviceAddr,
                      outputHostData.size() * sizeof(uint8_t), ACL_MEMCPY_DEVICE_TO_HOST);

    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy output failed. ret=%d\n", ret); return ret);

    ret = aclrtMemcpy(outputScaleHostData.data(), outputScaleHostData.size() * sizeof(int8_t), outputScaleDeviceAddr,
                      outputScaleHostData.size() * sizeof(int8_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy outputScale failed. ret=%d\n", ret); return ret);

    PrintVector<uint8_t>(outputHostData, "output", 64);

    PrintVector<uint8_t>(outputScaleHostData, "outputScale", 64);

    if (!dumpPrefix.empty()) {
        std::ofstream outputFile(dumpPrefix + "_output.bin", std::ios::binary);
        outputFile.write(reinterpret_cast<const char *>(outputHostData.data()),
                         static_cast<std::streamsize>(outputHostData.size()));
        std::ofstream scaleFile(dumpPrefix + "_output_scale.bin", std::ios::binary);
        scaleFile.write(reinterpret_cast<const char *>(outputScaleHostData.data()),
                        static_cast<std::streamsize>(outputScaleHostData.size()));
        LOG_PRINT("dumpPrefix=%s\n", dumpPrefix.c_str());
    }

    // The short-lived validation demo has already copied and dumped all
    // results.  Avoid device/ACL teardown here: the 950 debug runtime can wait
    // indefinitely while releasing a successful custom-op context.
    std::_Exit(0);
}
