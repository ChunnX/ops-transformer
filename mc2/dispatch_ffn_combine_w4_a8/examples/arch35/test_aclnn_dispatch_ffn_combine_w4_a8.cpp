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
 * \brief aclnn测试样例
 */
#include <thread>
#include <iostream>
#include <string>
#include <cstring>
#include <vector>
#include <fstream>
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
int ReadBinFile(const std::string &filePath, std::vector<T> &data, int64_t size) {
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        LOG_PRINT("Failed to open file: %s\n", filePath.c_str());
        return -1;
    }
    data.resize(size);
    file.read(reinterpret_cast<char*>(data.data()), size * sizeof(T));
    if (!file) {
        LOG_PRINT("Failed to read file: %s\n", filePath.c_str());
        return -1;
    }
    file.close();
    return 0;
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

int CreateAclTensorList(const std::vector<std::pair<void*, aclTensor*>> &tensors, aclTensorList **tensorList) {
    std::vector<aclTensor*> tensorVec;
    for (auto &pair : tensors) {
        tensorVec.push_back(pair.second);
    }
    *tensorList = aclCreateTensorList(tensorVec.data(), tensorVec.size());
    return 0;
}

struct Args {
    uint32_t rankId;
    HcclComm hcclComm;
    aclrtStream stream;
    aclrtContext context;
    std::string dataDir;
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

    std::vector<int64_t> xShape = {32, 64};
    std::vector<int64_t> weight1Shape = {64, 128};
    std::vector<int64_t> weight2Shape = {128, 64};
    std::vector<int64_t> expertIdShape = {32};
    std::vector<int64_t> scale1Shape = {128};
    std::vector<int64_t> scale2Shape = {64};
    std::vector<int64_t> bias1Shape = {128};
    std::vector<int64_t> bias2Shape = {64};
    std::vector<int64_t> probsShape = {32, ndev};
    std::vector<int64_t> outShape = {32, 64};
    std::vector<int64_t> expertTokenNumsShape = {ndev};

    int64_t numExperts = ndev;

    int64_t xShapeSize = GetShapeSize(xShape);
    int64_t weight1ShapeSize = GetShapeSize(weight1Shape);
    int64_t weight2ShapeSize = GetShapeSize(weight2Shape);
    int64_t expertIdShapeSize = GetShapeSize(expertIdShape);
    int64_t scale1ShapeSize = GetShapeSize(scale1Shape);
    int64_t scale2ShapeSize = GetShapeSize(scale2Shape);
    int64_t bias1ShapeSize = GetShapeSize(bias1Shape);
    int64_t bias2ShapeSize = GetShapeSize(bias2Shape);
    int64_t probsShapeSize = GetShapeSize(probsShape);
    int64_t outShapeSize = GetShapeSize(outShape);
    int64_t expertTokenNumsShapeSize = GetShapeSize(expertTokenNumsShape);

    void *xDeviceAddr = nullptr;
    std::vector<void*> weight1DeviceAddrs(numExperts);
    std::vector<void*> weight2DeviceAddrs(numExperts);
    void *expertIdDeviceAddr = nullptr;
    std::vector<void*> scale1DeviceAddrs(numExperts);
    std::vector<void*> scale2DeviceAddrs(numExperts);
    std::vector<void*> bias1DeviceAddrs(numExperts);
    std::vector<void*> bias2DeviceAddrs(numExperts);
    void *probsDeviceAddr = nullptr;
    void *outDeviceAddr = nullptr;
    void *expertTokenNumsDeviceAddr = nullptr;

    aclTensor *x = nullptr;
    std::vector<aclTensor*> weight1Tensors(numExperts);
    std::vector<aclTensor*> weight2Tensors(numExperts);
    aclTensor *expertId = nullptr;
    std::vector<aclTensor*> scale1Tensors(numExperts);
    std::vector<aclTensor*> scale2Tensors(numExperts);
    std::vector<aclTensor*> bias1Tensors(numExperts);
    std::vector<aclTensor*> bias2Tensors(numExperts);
    aclTensor *probs = nullptr;
    aclTensor *out = nullptr;
    aclTensor *expertTokenNums = nullptr;

    aclTensorList *weight1List = nullptr;
    aclTensorList *weight2List = nullptr;
    aclTensorList *scale1List = nullptr;
    aclTensorList *scale2List = nullptr;
    aclTensorList *bias1List = nullptr;
    aclTensorList *bias2List = nullptr;

    std::vector<int8_t> xHostData;
    std::vector<std::vector<int8_t>> weight1HostData(numExperts);
    std::vector<std::vector<int8_t>> weight2HostData(numExperts);
    std::vector<int64_t> expertIdHostData;
    std::vector<std::vector<float>> scale1HostData(numExperts);
    std::vector<std::vector<float>> scale2HostData(numExperts);
    std::vector<std::vector<float>> bias1HostData(numExperts);
    std::vector<std::vector<float>> bias2HostData(numExperts);
    std::vector<float> probsHostData;
    std::vector<float> outHostData;
    std::vector<int64_t> expertTokenNumsHostData;

    std::string xPath = args.dataDir + "/x_" + std::to_string(args.rankId) + ".bin";
    std::string expertIdPath = args.dataDir + "/expert_id_" + std::to_string(args.rankId) + ".bin";
    std::string probsPath = args.dataDir + "/probs_" + std::to_string(args.rankId) + ".bin";
    std::string outPath = args.dataDir + "/out_" + std::to_string(args.rankId) + ".bin";
    std::string expertTokenNumsPath = args.dataDir + "/expert_token_nums_" + std::to_string(args.rankId) + ".bin";

    ret = ReadBinFile(xPath, xHostData, xShapeSize);
    CHECK_RET(ret == 0, LOG_PRINT("Read x bin file failed\n"); return ret);
    
    ret = ReadBinFile(expertIdPath, expertIdHostData, expertIdShapeSize);
    CHECK_RET(ret == 0, LOG_PRINT("Read expert_id bin file failed\n"); return ret);
    
    ret = ReadBinFile(probsPath, probsHostData, probsShapeSize);
    CHECK_RET(ret == 0, LOG_PRINT("Read probs bin file failed\n"); return ret);
    
    outHostData.resize(outShapeSize, 0);
    expertTokenNumsHostData.resize(expertTokenNumsShapeSize, 0);

    for (int64_t i = 0; i < numExperts; i++) {
        std::string weight1Path = args.dataDir + "/weight1_" + std::to_string(i) + ".bin";
        std::string weight2Path = args.dataDir + "/weight2_" + std::to_string(i) + ".bin";
        std::string scale1Path = args.dataDir + "/scale1_" + std::to_string(i) + ".bin";
        std::string scale2Path = args.dataDir + "/scale2_" + std::to_string(i) + ".bin";
        std::string bias1Path = args.dataDir + "/bias1_" + std::to_string(i) + ".bin";
        std::string bias2Path = args.dataDir + "/bias2_" + std::to_string(i) + ".bin";

        ret = ReadBinFile(weight1Path, weight1HostData[i], weight1ShapeSize);
        CHECK_RET(ret == 0, LOG_PRINT("Read weight1 bin file failed for expert %ld\n", i); return ret);
        
        ret = ReadBinFile(weight2Path, weight2HostData[i], weight2ShapeSize);
        CHECK_RET(ret == 0, LOG_PRINT("Read weight2 bin file failed for expert %ld\n", i); return ret);
        
        ret = ReadBinFile(scale1Path, scale1HostData[i], scale1ShapeSize);
        CHECK_RET(ret == 0, LOG_PRINT("Read scale1 bin file failed for expert %ld\n", i); return ret);
        
        ret = ReadBinFile(scale2Path, scale2HostData[i], scale2ShapeSize);
        CHECK_RET(ret == 0, LOG_PRINT("Read scale2 bin file failed for expert %ld\n", i); return ret);
        
        ret = ReadBinFile(bias1Path, bias1HostData[i], bias1ShapeSize);
        CHECK_RET(ret == 0, LOG_PRINT("Read bias1 bin file failed for expert %ld\n", i); return ret);
        
        ret = ReadBinFile(bias2Path, bias2HostData[i], bias2ShapeSize);
        CHECK_RET(ret == 0, LOG_PRINT("Read bias2 bin file failed for expert %ld\n", i); return ret);
    }

    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_INT8, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    for (int64_t i = 0; i < numExperts; i++) {
        ret = CreateAclTensor(weight1HostData[i], weight1Shape, &weight1DeviceAddrs[i], aclDataType::ACL_INT4, &weight1Tensors[i]);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        
        ret = CreateAclTensor(weight2HostData[i], weight2Shape, &weight2DeviceAddrs[i], aclDataType::ACL_INT4, &weight2Tensors[i]);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        
        ret = CreateAclTensor(scale1HostData[i], scale1Shape, &scale1DeviceAddrs[i], aclDataType::ACL_FLOAT, &scale1Tensors[i]);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        
        ret = CreateAclTensor(scale2HostData[i], scale2Shape, &scale2DeviceAddrs[i], aclDataType::ACL_FLOAT, &scale2Tensors[i]);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        
        ret = CreateAclTensor(bias1HostData[i], bias1Shape, &bias1DeviceAddrs[i], aclDataType::ACL_FLOAT, &bias1Tensors[i]);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        
        ret = CreateAclTensor(bias2HostData[i], bias2Shape, &bias2DeviceAddrs[i], aclDataType::ACL_FLOAT, &bias2Tensors[i]);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }

    ret = CreateAclTensor(expertIdHostData, expertIdShape, &expertIdDeviceAddr, aclDataType::ACL_INT64, &expertId);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(probsHostData, probsShape, &probsDeviceAddr, aclDataType::ACL_FLOAT, &probs);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(expertTokenNumsHostData, expertTokenNumsShape, &expertTokenNumsDeviceAddr, aclDataType::ACL_INT64, &expertTokenNums);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<std::pair<void*, aclTensor*>> weight1Pairs;
    std::vector<std::pair<void*, aclTensor*>> weight2Pairs;
    std::vector<std::pair<void*, aclTensor*>> scale1Pairs;
    std::vector<std::pair<void*, aclTensor*>> scale2Pairs;
    std::vector<std::pair<void*, aclTensor*>> bias1Pairs;
    std::vector<std::pair<void*, aclTensor*>> bias2Pairs;
    
    for (int64_t i = 0; i < numExperts; i++) {
        weight1Pairs.push_back({weight1DeviceAddrs[i], weight1Tensors[i]});
        weight2Pairs.push_back({weight2DeviceAddrs[i], weight2Tensors[i]});
        scale1Pairs.push_back({scale1DeviceAddrs[i], scale1Tensors[i]});
        scale2Pairs.push_back({scale2DeviceAddrs[i], scale2Tensors[i]});
        bias1Pairs.push_back({bias1DeviceAddrs[i], bias1Tensors[i]});
        bias2Pairs.push_back({bias2DeviceAddrs[i], bias2Tensors[i]});
    }

    ret = CreateAclTensorList(weight1Pairs, &weight1List);
    CHECK_RET(ret == 0, LOG_PRINT("Create weight1 tensor list failed\n"); return -1);
    
    ret = CreateAclTensorList(weight2Pairs, &weight2List);
    CHECK_RET(ret == 0, LOG_PRINT("Create weight2 tensor list failed\n"); return -1);
    
    ret = CreateAclTensorList(scale1Pairs, &scale1List);
    CHECK_RET(ret == 0, LOG_PRINT("Create scale1 tensor list failed\n"); return -1);
    
    ret = CreateAclTensorList(scale2Pairs, &scale2List);
    CHECK_RET(ret == 0, LOG_PRINT("Create scale2 tensor list failed\n"); return -1);
    
    ret = CreateAclTensorList(bias1Pairs, &bias1List);
    CHECK_RET(ret == 0, LOG_PRINT("Create bias1 tensor list failed\n"); return -1);
    
    ret = CreateAclTensorList(bias2Pairs, &bias2List);
    CHECK_RET(ret == 0, LOG_PRINT("Create bias2 tensor list failed\n"); return -1);

    uint64_t workspaceSize = 0;
    int64_t maxOutputSize = 64 * 128;
    aclOpExecutor *executor;
    void *workspaceAddr = nullptr;

    ret = aclnnDispatchFFNCombineW4A8GetWorkspaceSize(x, weight1List, weight2List, expertId, scale1List, scale2List,
                                                      bias1List, bias2List, probs, hcom_name, maxOutputSize,
                                                      out, expertTokenNums, &workspaceSize, &executor);
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
    for (int64_t i = 0; i < numExperts; i++) {
        if (weight1Tensors[i] != nullptr) {
            aclDestroyTensor(weight1Tensors[i]);
        }
        if (weight2Tensors[i] != nullptr) {
            aclDestroyTensor(weight2Tensors[i]);
        }
        if (scale1Tensors[i] != nullptr) {
            aclDestroyTensor(scale1Tensors[i]);
        }
        if (scale2Tensors[i] != nullptr) {
            aclDestroyTensor(scale2Tensors[i]);
        }
        if (bias1Tensors[i] != nullptr) {
            aclDestroyTensor(bias1Tensors[i]);
        }
        if (bias2Tensors[i] != nullptr) {
            aclDestroyTensor(bias2Tensors[i]);
        }
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

    if (weight1List != nullptr) {
        aclDestroyTensorList(weight1List);
    }
    if (weight2List != nullptr) {
        aclDestroyTensorList(weight2List);
    }
    if (scale1List != nullptr) {
        aclDestroyTensorList(scale1List);
    }
    if (scale2List != nullptr) {
        aclDestroyTensorList(scale2List);
    }
    if (bias1List != nullptr) {
        aclDestroyTensorList(bias1List);
    }
    if (bias2List != nullptr) {
        aclDestroyTensorList(bias2List);
    }

    if (xDeviceAddr != nullptr) {
        aclrtFree(xDeviceAddr);
    }
    for (int64_t i = 0; i < numExperts; i++) {
        if (weight1DeviceAddrs[i] != nullptr) {
            aclrtFree(weight1DeviceAddrs[i]);
        }
        if (weight2DeviceAddrs[i] != nullptr) {
            aclrtFree(weight2DeviceAddrs[i]);
        }
        if (scale1DeviceAddrs[i] != nullptr) {
            aclrtFree(scale1DeviceAddrs[i]);
        }
        if (scale2DeviceAddrs[i] != nullptr) {
            aclrtFree(scale2DeviceAddrs[i]);
        }
        if (bias1DeviceAddrs[i] != nullptr) {
            aclrtFree(bias1DeviceAddrs[i]);
        }
        if (bias2DeviceAddrs[i] != nullptr) {
            aclrtFree(bias2DeviceAddrs[i]);
        }
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

    std::string dataDir = "./data";
    if (argc > 1) {
        dataDir = argv[1];
    }

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
        args[rankId].dataDir = dataDir;
        threads[rankId].reset(new(std::nothrow) std::thread(&launchOneThreadDispatchFFNCombineW4A8, std::ref(args[rankId])));
    }
    for (uint32_t rankId = 0; rankId < ndev; rankId++) {
        threads[rankId]->join();
    }
    aclFinalize();
    return 0;
}