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
bool ReadFile(const std::string &filePath, const std::vector<int64_t> &shape, std::vector<T>& hostData)
{
    size_t fileSize = GetShapeSize(shape);
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "无法打开文件: " << filePath << std::endl;
        return false;
    }
    hostData.resize(fileSize);
    if (!file.read(reinterpret_cast<char*>(hostData.data()), fileSize * sizeof(T))) {
        std::cerr << "读取文件失败: " << filePath << std::endl;
        file.close();
        return false;
    }
    file.close();
    return true;
}

template<typename T>
bool ReadFileWithOffset(const std::string &filePath, int64_t offset, int64_t count, std::vector<T>& hostData)
{
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "无法打开文件: " << filePath << std::endl;
        return false;
    }
    hostData.resize(count);
    file.seekg(offset * sizeof(T), std::ios::beg);
    if (!file.read(reinterpret_cast<char*>(hostData.data()), count * sizeof(T))) {
        std::cerr << "读取文件失败: " << filePath << " offset: " << offset << " count: " << count << std::endl;
        file.close();
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

void DestroyAclTensor(aclTensor *tensor, void *deviceAddr) {
    if (tensor != nullptr) {
        aclDestroyTensor(tensor);
    }
    if (deviceAddr != nullptr) {
        aclrtFree(deviceAddr);
    }
}

template<typename T>
struct TensorManager {
    std::vector<T> hostData;
    void *deviceAddr = nullptr;
    aclTensor *tensor = nullptr;
    std::vector<int64_t> shape;
    
    int Create(const std::vector<int64_t> &tensorShape, aclDataType dataType) {
        shape = tensorShape;
        return CreateAclTensor(hostData, shape, &deviceAddr, dataType, &tensor);
    }
    
    void Destroy() {
        DestroyAclTensor(tensor, deviceAddr);
        tensor = nullptr;
        deviceAddr = nullptr;
    }
};

template<typename T>
struct TensorListManager {
    std::vector<std::vector<T>> hostDataList;
    std::vector<void*> deviceAddrList;
    aclTensorList *tensorList = nullptr;
    std::vector<std::vector<int64_t>> shapeList;
    
    int CreateFromSingleFile(const std::string &filePath, const std::vector<int64_t> &tensorShape, 
                             int64_t numTensors, aclDataType dataType) {
        shapeList.resize(numTensors, tensorShape);
        hostDataList.resize(numTensors);
        deviceAddrList.clear();
        
        int64_t singleTensorSize = GetShapeSize(tensorShape);
        
        for (int64_t i = 0; i < numTensors; i++) {
            int64_t offset = i * singleTensorSize;
            if (!ReadFileWithOffset(filePath, offset, singleTensorSize, hostDataList[i])) {
                return -1;
            }
            
            void* deviceAddr = nullptr;
            aclTensor* tensor = nullptr;
            auto ret = CreateAclTensor(hostDataList[i], tensorShape, &deviceAddr, dataType, &tensor);
            CHECK_RET(ret == ACL_SUCCESS, return ret);
            
            deviceAddrList.push_back(deviceAddr);
        }
        
        std::vector<aclTensor*> tensors;
        for (size_t i = 0; i < deviceAddrList.size(); i++) {
            aclTensor* tensor = nullptr;
            std::vector<int64_t> strides(tensorShape.size(), 1);
            for (int64_t j = tensorShape.size() - 2; j >= 0; j--) {
                strides[j] = tensorShape[j + 1] * strides[j + 1];
            }
            tensor = aclCreateTensor(tensorShape.data(), tensorShape.size(), dataType, 
                                    strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                    tensorShape.data(), tensorShape.size(), deviceAddrList[i]);
            tensors.push_back(tensor);
        }
        
        tensorList = aclCreateTensorList(tensors.data(), tensors.size());
        return 0;
    }
    
    void Destroy() {
        if (tensorList != nullptr) {
            aclDestroyTensorList(tensorList);
            tensorList = nullptr;
        }
        for (auto addr : deviceAddrList) {
            if (addr != nullptr) {
                aclrtFree(addr);
            }
        }
        deviceAddrList.clear();
    }
};

struct TensorConfig {
    std::string name;
    std::string filePath;
    std::vector<int64_t> shape;
    aclDataType dataType;
    bool isTensorList;
    int64_t numTensors;
};

struct AllTensorsManager {
    TensorManager<int8_t> xManager;
    TensorListManager<int8_t> weight1Manager;
    TensorListManager<int8_t> weight2Manager;
    TensorListManager<float> scale1Manager;
    TensorListManager<float> scale2Manager;
    TensorListManager<float> bias1Manager;
    TensorListManager<float> bias2Manager;
    TensorManager<int32_t> expertIdManager;
    TensorManager<float> probsManager;
    TensorManager<float> outManager;
    TensorManager<int64_t> expertTokenNumsManager;
    
    std::vector<TensorConfig> configs;
    
    void AddConfig(const std::string& name, const std::string& filePath, 
                   const std::vector<int64_t>& shape, aclDataType dataType,
                   bool isTensorList = false, int64_t numTensors = 1) {
        configs.push_back({name, filePath, shape, dataType, isTensorList, numTensors});
    }
    
    int CreateAll() {
        for (const auto& config : configs) {
            int ret = CreateTensor(config);
            if (ret != 0) {
                LOG_PRINT("Failed to create tensor: %s\n", config.name.c_str());
                return ret;
            }
        }
        return 0;
    }
    
    int CreateTensor(const TensorConfig& config) {
        if (config.name == "x") {
            if (!ReadFile(config.filePath, config.shape, xManager.hostData)) {
                return -1;
            }
            return xManager.Create(config.shape, config.dataType);
        } else if (config.name == "weight1") {
            return weight1Manager.CreateFromSingleFile(config.filePath, config.shape, 
                                                       config.numTensors, config.dataType);
        } else if (config.name == "weight2") {
            return weight2Manager.CreateFromSingleFile(config.filePath, config.shape, 
                                                       config.numTensors, config.dataType);
        } else if (config.name == "scale1") {
            return scale1Manager.CreateFromSingleFile(config.filePath, config.shape, 
                                                      config.numTensors, config.dataType);
        } else if (config.name == "scale2") {
            return scale2Manager.CreateFromSingleFile(config.filePath, config.shape, 
                                                      config.numTensors, config.dataType);
        } else if (config.name == "bias1") {
            return bias1Manager.CreateFromSingleFile(config.filePath, config.shape, 
                                                     config.numTensors, config.dataType);
        } else if (config.name == "bias2") {
            return bias2Manager.CreateFromSingleFile(config.filePath, config.shape, 
                                                     config.numTensors, config.dataType);
        } else if (config.name == "expertId") {
            if (!ReadFile(config.filePath, config.shape, expertIdManager.hostData)) {
                return -1;
            }
            return expertIdManager.Create(config.shape, config.dataType);
        } else if (config.name == "probs") {
            if (!ReadFile(config.filePath, config.shape, probsManager.hostData)) {
                return -1;
            }
            return probsManager.Create(config.shape, config.dataType);
        } else if (config.name == "out") {
            return outManager.Create(config.shape, config.dataType);
        } else if (config.name == "expertTokenNums") {
            return expertTokenNumsManager.Create(config.shape, config.dataType);
        }
        return -1;
    }
    
    void DestroyAll() {
        xManager.Destroy();
        weight1Manager.Destroy();
        weight2Manager.Destroy();
        scale1Manager.Destroy();
        scale2Manager.Destroy();
        bias1Manager.Destroy();
        bias2Manager.Destroy();
        expertIdManager.Destroy();
        probsManager.Destroy();
        outManager.Destroy();
        expertTokenNumsManager.Destroy();
    }
};

struct Args {
    uint32_t rankId;
    HcclComm hcclComm;
    aclrtStream stream;
    aclrtContext context;
    
    int64_t m;
    int64_t k;
    int64_t n;
    int64_t k2;
    int64_t n2;
    int64_t numExperts;
    int64_t expertPerRank;
    int64_t topK;
    int64_t EP;
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

    std::string dataDir = "./data/";
    
    std::vector<int64_t> xShape = {args.m, args.k};
    std::vector<int64_t> weight1Shape = {args.k, args.n};
    std::vector<int64_t> weight2Shape = {args.k2, args.n2};
    std::vector<int64_t> scale1Shape = {args.n};
    std::vector<int64_t> scale2Shape = {args.n2};
    std::vector<int64_t> bias1Shape = {args.n};
    std::vector<int64_t> bias2Shape = {args.n2};
    std::vector<int64_t> expertIdShape = {args.m};
    std::vector<int64_t> probsShape = {args.m, args.numExperts};
    std::vector<int64_t> outShape = {args.m, args.n2};
    std::vector<int64_t> expertTokenNumsShape = {args.numExperts};

    AllTensorsManager manager;
    
    manager.AddConfig("x", dataDir + "x.bin", xShape, aclDataType::ACL_INT8);
    manager.AddConfig("weight1", dataDir + "weight1.bin", weight1Shape, aclDataType::ACL_INT4, true, args.numExperts);
    manager.AddConfig("weight2", dataDir + "weight2.bin", weight2Shape, aclDataType::ACL_INT4, true, args.numExperts);
    manager.AddConfig("scale1", dataDir + "scale1.bin", scale1Shape, aclDataType::ACL_FLOAT, true, args.numExperts);
    manager.AddConfig("scale2", dataDir + "scale2.bin", scale2Shape, aclDataType::ACL_FLOAT, true, args.numExperts);
    manager.AddConfig("bias1", dataDir + "bias1.bin", bias1Shape, aclDataType::ACL_FLOAT, true, args.numExperts);
    manager.AddConfig("bias2", dataDir + "bias2.bin", bias2Shape, aclDataType::ACL_FLOAT, true, args.numExperts);
    manager.AddConfig("expertId", dataDir + "expertId.bin", expertIdShape, aclDataType::ACL_INT32);
    manager.AddConfig("probs", dataDir + "probs.bin", probsShape, aclDataType::ACL_FLOAT);
    manager.AddConfig("out", "", outShape, aclDataType::ACL_FLOAT);
    manager.AddConfig("expertTokenNums", "", expertTokenNumsShape, aclDataType::ACL_INT64);

    int64_t maxOutputSize = 128;
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    void *workspaceAddr = nullptr;

    ret = manager.CreateAll();
    CHECK_RET(ret == 0, return ret);
    
    ret = aclnnDispatchFFNCombineW4A8GetWorkspaceSize(manager.xManager.tensor, manager.weight1Manager.tensorList, 
                                                     manager.weight2Manager.tensorList, manager.expertIdManager.tensor, 
                                                     manager.scale1Manager.tensorList, manager.scale2Manager.tensorList, 
                                                     manager.bias1Manager.tensorList, manager.bias2Manager.tensorList, 
                                                     manager.probsManager.tensor, hcom_name, maxOutputSize,
                                                     manager.outManager.tensor, manager.expertTokenNumsManager.tensor, 
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
    
    manager.DestroyAll();
    
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
    
    int64_t numExperts = 8;
    int64_t hiddenSize = 4096;
    int64_t intermediateSize = 14336;
    int64_t topK = 8;
    
    std::vector<std::unique_ptr<std::thread>> threads(ndev);
    for (uint32_t rankId = 0; rankId < ndev; rankId++) {
        args[rankId].rankId = rankId;
        args[rankId].hcclComm = comms[rankId];
        args[rankId].stream = stream[rankId];
        args[rankId].context = context[rankId];
        
        args[rankId].m = 128;
        args[rankId].k = hiddenSize;
        args[rankId].n = intermediateSize;
        args[rankId].k2 = intermediateSize;
        args[rankId].n2 = hiddenSize;
        args[rankId].numExperts = numExperts;
        args[rankId].expertPerRank = numExperts / ndev;
        args[rankId].topK = topK;
        args[rankId].EP = ndev;
        
        threads[rankId].reset(new(std::nothrow) std::thread(&launchOneThreadDispatchFFNCombineW4A8, std::ref(args[rankId])));
    }
    for (uint32_t rankId = 0; rankId < ndev; rankId++) {
        threads[rankId]->join();
    }
    aclFinalize();
    return 0;
}
