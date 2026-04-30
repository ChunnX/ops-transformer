/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aclnn_dispatch_ffn_combine_w4_a8.cpp
 * \brief aclnnDispatchFFNCombineW4A8测试样例
 */
#include <thread>
#include <iostream>
#include <string>
#include <cstring>
#include <vector>
#include <fstream>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <acl/acl.h>
#include <hccl/hccl.h>
#include "aclnnop/aclnn_dispatch_ffn_combine_w4_a8.h"

int ndev = 2;

#define CHECK_RET(cond, return_expr) \
do {                               \
if (!(cond)) {                   \
return_expr;                   \
}                                \
} while (0)

#define LOG_PRINT(message, ...)     \
do {                              \
printf(message, ##__VA_ARGS__); \
} while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape) {
    int64_t shapeSize = 1;
    for (auto i: shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

template<typename T>
bool ReadFile(const std::string &filePath, std::vector<int64_t> shape, std::vector<T>& hostData)
{
    size_t fileSize = 1;
    for (int64_t i : shape){
        fileSize *= i; 
    }
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "无法打开文件: " << filePath << std::endl;
        return false;
    }
    hostData.resize(fileSize);
    if (file.read(reinterpret_cast<char*>(hostData.data()), fileSize * sizeof(T))) {
    } else {
        std::cerr << "读取文件失败: " << filePath << std::endl;
        return false;
    }
    file.close();
    return true;
}

template<typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor) {
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
    return 0;
}

template<typename T>
int CreateAclTensorList(const std::vector<std::vector<T>> &hostDataList, 
                        const std::vector<std::vector<int64_t>> &shapeList, 
                        std::vector<void*> &deviceAddrList,
                        aclDataType dataType, 
                        aclTensorList **tensorList) {
    std::vector<aclTensor*> tensors;
    for (size_t i = 0; i < hostDataList.size(); i++) {
        void* deviceAddr = nullptr;
        aclTensor* tensor = nullptr;
        auto ret = CreateAclTensor(hostDataList[i], shapeList[i], &deviceAddr, dataType, &tensor);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        tensors.push_back(tensor);
        deviceAddrList.push_back(deviceAddr);
    }
    *tensorList = aclCreateTensorList(tensors.data(), tensors.size());
    return 0;
}

struct Args {
    uint32_t rankId;
    HcclComm hcclComm;
    aclrtStream stream;
    aclrtContext context;
};

int launchOneThreadDispatchFFNCombineW4A8(Args &args)
{
    int ret;
    ret = aclrtSetCurrentContext(args.context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetCurrentContext failed. ERROR: %d\n", ret); return ret);
    char hcom_name[128] = {0};
    ret = HcclGetCommName(args.hcclComm, hcom_name);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetCommName failed. ret = %d \n", ret); return -1);
    LOG_PRINT("[INFO] rank %d hcom: %s stream: %p, context : %p\n", args.rankId, hcom_name, args.stream,
            args.context);

    std::vector<int64_t> xShape = {128, 4096};
    int64_t numExperts = 8;
    int64_t hiddenSize = 4096;
    int64_t intermediateSize = 14336;
    
    std::vector<std::vector<int64_t>> weight1Shapes;
    std::vector<std::vector<int64_t>> weight2Shapes;
    std::vector<std::vector<int64_t>> scale1Shapes;
    std::vector<std::vector<int64_t>> scale2Shapes;
    std::vector<std::vector<int64_t>> bias1Shapes;
    std::vector<std::vector<int64_t>> bias2Shapes;
    
    for (int64_t i = 0; i < numExperts; i++) {
        weight1Shapes.push_back({hiddenSize, intermediateSize});
        weight2Shapes.push_back({intermediateSize, hiddenSize});
        scale1Shapes.push_back({intermediateSize});
        scale2Shapes.push_back({hiddenSize});
        bias1Shapes.push_back({intermediateSize});
        bias2Shapes.push_back({hiddenSize});
    }
    
    std::vector<int64_t> expertIdShape = {128};
    std::vector<int64_t> probsShape = {128, numExperts};
    std::vector<int64_t> outShape = {128, hiddenSize};
    std::vector<int64_t> expertTokenNumsShape = {numExperts};
    
    void *xDeviceAddr = nullptr;
    std::vector<void*> weight1DeviceAddrs;
    std::vector<void*> weight2DeviceAddrs;
    std::vector<void*> scale1DeviceAddrs;
    std::vector<void*> scale2DeviceAddrs;
    std::vector<void*> bias1DeviceAddrs;
    std::vector<void*> bias2DeviceAddrs;
    void *expertIdDeviceAddr = nullptr;
    void *probsDeviceAddr = nullptr;
    void *outDeviceAddr = nullptr;
    void *expertTokenNumsDeviceAddr = nullptr;
    
    aclTensor *x = nullptr;
    aclTensorList *weight1 = nullptr;
    aclTensorList *weight2 = nullptr;
    aclTensorList *scale1 = nullptr;
    aclTensorList *scale2 = nullptr;
    aclTensorList *bias1 = nullptr;
    aclTensorList *bias2 = nullptr;
    aclTensor *expertId = nullptr;
    aclTensor *probs = nullptr;
    aclTensor *out = nullptr;
    aclTensor *expertTokenNums = nullptr;

    int64_t maxOutputSize = 128;
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    void *workspaceAddr = nullptr;

    long long xShapeSize = GetShapeSize(xShape);
    std::vector<int8_t> xHostData(xShapeSize, 1);
    
    std::vector<std::vector<int8_t>> weight1HostDatas;
    std::vector<std::vector<int8_t>> weight2HostDatas;
    std::vector<std::vector<float>> scale1HostDatas;
    std::vector<std::vector<float>> scale2HostDatas;
    std::vector<std::vector<float>> bias1HostDatas;
    std::vector<std::vector<float>> bias2HostDatas;
    
    for (int64_t i = 0; i < numExperts; i++) {
        long long weight1Size = GetShapeSize(weight1Shapes[i]);
        long long weight2Size = GetShapeSize(weight2Shapes[i]);
        long long scale1Size = GetShapeSize(scale1Shapes[i]);
        long long scale2Size = GetShapeSize(scale2Shapes[i]);
        long long bias1Size = GetShapeSize(bias1Shapes[i]);
        long long bias2Size = GetShapeSize(bias2Shapes[i]);
        
        weight1HostDatas.push_back(std::vector<int8_t>(weight1Size, 1));
        weight2HostDatas.push_back(std::vector<int8_t>(weight2Size, 1));
        scale1HostDatas.push_back(std::vector<float>(scale1Size, 1.0f));
        scale2HostDatas.push_back(std::vector<float>(scale2Size, 1.0f));
        bias1HostDatas.push_back(std::vector<float>(bias1Size, 0.0f));
        bias2HostDatas.push_back(std::vector<float>(bias2Size, 0.0f));
    }
    
    long long expertIdShapeSize = GetShapeSize(expertIdShape);
    long long probsShapeSize = GetShapeSize(probsShape);
    long long outShapeSize = GetShapeSize(outShape);
    long long expertTokenNumsShapeSize = GetShapeSize(expertTokenNumsShape);
    
    std::vector<int32_t> expertIdHostData(expertIdShapeSize, 0);
    std::vector<float> probsHostData(probsShapeSize, 0.1f);
    std::vector<float> outHostData(outShapeSize, 0);
    std::vector<int64_t> expertTokenNumsHostData(expertTokenNumsShapeSize, 0);
    
    std::string dataDir = "./data/";
    
    ReadFile(dataDir + "x.bin", xShape, xHostData);
    for (int64_t i = 0; i < numExperts; i++) {
        ReadFile(dataDir + "weight1_" + std::to_string(i) + ".bin", weight1Shapes[i], weight1HostDatas[i]);
        ReadFile(dataDir + "weight2_" + std::to_string(i) + ".bin", weight2Shapes[i], weight2HostDatas[i]);
        ReadFile(dataDir + "scale1_" + std::to_string(i) + ".bin", scale1Shapes[i], scale1HostDatas[i]);
        ReadFile(dataDir + "scale2_" + std::to_string(i) + ".bin", scale2Shapes[i], scale2HostDatas[i]);
        ReadFile(dataDir + "bias1_" + std::to_string(i) + ".bin", bias1Shapes[i], bias1HostDatas[i]);
        ReadFile(dataDir + "bias2_" + std::to_string(i) + ".bin", bias2Shapes[i], bias2HostDatas[i]);
    }
    ReadFile(dataDir + "expertId.bin", expertIdShape, expertIdHostData);
    ReadFile(dataDir + "probs.bin", probsShape, probsHostData);
    
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_INT8, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    
    ret = CreateAclTensorList(weight1HostDatas, weight1Shapes, weight1DeviceAddrs, 
                              aclDataType::ACL_INT4, &weight1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    
    ret = CreateAclTensorList(weight2HostDatas, weight2Shapes, weight2DeviceAddrs, 
                              aclDataType::ACL_INT4, &weight2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    
    ret = CreateAclTensorList(scale1HostDatas, scale1Shapes, scale1DeviceAddrs, 
                              aclDataType::ACL_FLOAT, &scale1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    
    ret = CreateAclTensorList(scale2HostDatas, scale2Shapes, scale2DeviceAddrs, 
                              aclDataType::ACL_FLOAT, &scale2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    
    ret = CreateAclTensorList(bias1HostDatas, bias1Shapes, bias1DeviceAddrs, 
                              aclDataType::ACL_FLOAT, &bias1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    
    ret = CreateAclTensorList(bias2HostDatas, bias2Shapes, bias2DeviceAddrs, 
                              aclDataType::ACL_FLOAT, &bias2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    
    ret = CreateAclTensor(expertIdHostData, expertIdShape, &expertIdDeviceAddr, 
                         aclDataType::ACL_INT32, &expertId);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    
    ret = CreateAclTensor(probsHostData, probsShape, &probsDeviceAddr, 
                         aclDataType::ACL_FLOAT, &probs);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    
    ret = CreateAclTensor(expertTokenNumsHostData, expertTokenNumsShape, &expertTokenNumsDeviceAddr, 
                         aclDataType::ACL_INT64, &expertTokenNums);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    
    ret = aclnnDispatchFFNCombineW4A8GetWorkspaceSize(x, weight1, weight2, expertId, 
                                                     scale1, scale2, bias1, bias2, 
                                                     probs, hcom_name, maxOutputSize,
                                                     out, expertTokenNums, 
                                                     &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnDispatchFFNCombineW4A8GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    
    ret = aclnnDispatchFFNCombineW4A8(workspaceAddr, workspaceSize, executor, args.stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDispatchFFNCombineW4A8 failed. ERROR: %d\n", ret); return ret);
    
    ret = aclrtSynchronizeStreamWithTimeout(args.stream, 10000);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("device%d aclnnDispatchFFNCombineW4A8 execute success \n", args.rankId);
    
    if (x != nullptr) {
        aclDestroyTensor(x);
    }
    if (weight1 != nullptr) {
        aclDestroyTensorList(weight1);
    }
    if (weight2 != nullptr) {
        aclDestroyTensorList(weight2);
    }
    if (scale1 != nullptr) {
        aclDestroyTensorList(scale1);
    }
    if (scale2 != nullptr) {
        aclDestroyTensorList(scale2);
    }
    if (bias1 != nullptr) {
        aclDestroyTensorList(bias1);
    }
    if (bias2 != nullptr) {
        aclDestroyTensorList(bias2);
    }
    if (expertId != nullptr) {
        aclDestroyTensor(expertId);
    }
    if (probs != nullptr) {
        aclDestroyTensor(probs);
    }
    if (out != nullptr) {
        aclDestroyTensor(out);
    }
    if (expertTokenNums != nullptr) {
        aclDestroyTensor(expertTokenNums);
    }
    if (xDeviceAddr != nullptr) {
        aclrtFree(xDeviceAddr);
    }
    for (auto addr : weight1DeviceAddrs) {
        if (addr != nullptr) aclrtFree(addr);
    }
    for (auto addr : weight2DeviceAddrs) {
        if (addr != nullptr) aclrtFree(addr);
    }
    for (auto addr : scale1DeviceAddrs) {
        if (addr != nullptr) aclrtFree(addr);
    }
    for (auto addr : scale2DeviceAddrs) {
        if (addr != nullptr) aclrtFree(addr);
    }
    for (auto addr : bias1DeviceAddrs) {
        if (addr != nullptr) aclrtFree(addr);
    }
    for (auto addr : bias2DeviceAddrs) {
        if (addr != nullptr) aclrtFree(addr);
    }
    if (expertIdDeviceAddr != nullptr) {
        aclrtFree(expertIdDeviceAddr);
    }
    if (probsDeviceAddr != nullptr) {
        aclrtFree(probsDeviceAddr);
    }
    if (outDeviceAddr != nullptr) {
        aclrtFree(outDeviceAddr);
    }
    if (expertTokenNumsDeviceAddr != nullptr) {
        aclrtFree(expertTokenNumsDeviceAddr);
    }
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(args.stream);
    HcclCommDestroy(args.hcclComm);
    aclrtDestroyContext(args.context);
    aclrtResetDevice(args.rankId);
    return 0;
}

int main(int argc, char *argv[])
{
    int ret = aclInit(nullptr);
    int32_t devices[ndev];
    for (int i = 0; i < ndev; i++) {
        devices[i] = i;
    }
    HcclComm comms[128];
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    
    for (int i = 0; i < ndev; i++) {
        ret = aclrtSetDevice(devices[i]);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    }
    ret = HcclCommInitAll(ndev, devices, comms);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("HcclCommInitAll failed. ERROR: %d\n", ret); return ret);
    
    Args args[ndev];
    aclrtStream stream[ndev];
    aclrtContext context[ndev];
    for (uint32_t rankId = 0; rankId < ndev; rankId++) {
        ret = aclrtSetDevice(rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
        ret = aclrtCreateContext(&context[rankId], rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed. ERROR: %d\n", ret); return ret);
        ret = aclrtCreateStream(&stream[rankId]);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    }
    
    std::vector<std::unique_ptr<std::thread>> threads(ndev);
    for (uint32_t rankId = 0; rankId < ndev; rankId++) {
        args[rankId].rankId = rankId;
        args[rankId].hcclComm = comms[rankId];
        args[rankId].stream = stream[rankId];
        args[rankId].context = context[rankId];
        threads[rankId].reset(new(std::nothrow) std::thread(&launchOneThreadDispatchFFNCombineW4A8, std::ref(args[rankId])));
    }
    for (uint32_t rankId = 0; rankId < ndev; rankId++) {
        threads[rankId]->join();
    }
    aclFinalize();
    return 0;
}
