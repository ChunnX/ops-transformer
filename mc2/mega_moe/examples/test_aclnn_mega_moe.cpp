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
 * \file test_aclnn_mega_moe.cpp
 * \brief
 */

#include <thread>
#include <memory>
#include <chrono>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <iostream>
#include <vector>
#include "hccl/hccl.h"
#include "mega_moe/vendors/MEGA_transformer/op_api/include/aclnnop/aclnn_mega_moe.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while(0)

constexpr int DEV_NUM = 2;
constexpr int E = 4;
constexpr int M = 1024;
constexpr int K = 1024;
constexpr int N = 1024;
constexpr int topK = 2;
constexpr int total_ub_size = 256 * 1024;
int rankSize = DEV_NUM;

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
}

template<typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc failed. ret: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMemcpy failed. ret: %d\n", ret); return ret);
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i +1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
        shape.data(), shape.size(), *deviceAddr);
    return 0;
}

struct Args {
    int rankId;
    HcclComm hcclComm;
    aclrtStream stream;
    aclrtContext context;
};

template <typename T1, typename T2>
auto Ceil(T1 a, T2 b) -> T1
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

template<typename T>
bool loadBinaryFile(const std::string& filename, std::vector<T>& data) {
    // 打开文件
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        return false;
    }

    // 获取文件大小
    std::streamsize fileSize = file.tellg();
    if (fileSize <= 0) {
        std::cerr << "文件大小为0或无法获取大小: " << filename << std::endl;
        return false;
    }

    // 回到文件开头
    file.seekg(0, std::ios::beg);

    // 调整vector大小
    //data.resize(fileSize);

    // 读取数据
    if (!file.read(reinterpret_cast<char*>(data.data()), fileSize)) {
        std::cerr << "读取文件失败: " << filename << std::endl;
        return false;
    }

    std::cout << "成功读取 " << fileSize << " 字节到 vector<int8_t>" << std::endl;
    return true;
}

std::string GetCurrentPath()
{
    std::filesystem::path current_path = std::filesystem::current_path();
    return current_path.string() + "/";
}

std::string GetOutPutDataPath(int32_t rank)
{
    return GetCurrentPath() + "/data/" + std::to_string(rank) + "/";
}

template<typename OutputDtype>
void WriteResultToBin(int32_t rank, std::string actualResultFile, void *buf, int64_t len)
{
    int size = sizeof(OutputDtype) * len;
    std::string dataPath = GetOutPutDataPath(rank);
    std::string path = dataPath + actualResultFile;
    std::vector<char> actualResult(size, 0);
    auto error = aclrtMemcpy(actualResult.data(), size, buf, size, ACL_MEMCPY_DEVICE_TO_HOST);
    if (error != 0) {
        printf("Memcpy device to host failed!, file:%s\n", actualResultFile.c_str());
        return;
    }
    std::ofstream outfile(path, std::ifstream::binary);
    outfile.write(actualResult.data(), size);
    outfile.close();
    printf("Write result to file %s finished!\n", path.c_str());
}

int launchOneThread_MegaMoe(Args &args)
{
    int ret = aclrtSetCurrentContext(args.context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetCurrentContext failed. ret = %d \n", ret); return ret);

    char hcomName[128] = {0};
    ret = HcclGetCommName(args.hcclComm, hcomName);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetCommName failed. ERROR: %d\n", ret); return -1);
    LOG_PRINT("[INFO] rank = %d, hcomName = %s, stream = %p\n", args.rankId, hcomName, args.stream);

    // 初始化输入shape
    std::vector<int64_t> xShape = {M, K};
    std::vector<int64_t> expertIdxShape = {M, topK};
    std::vector<int64_t> weightShape = {E, N, K};
    std::vector<int64_t> weightScaleShape = {E, N, Ceil(K, 64), 2};
    std::vector<int64_t> outputShape = {M, N / 2}; // gmm1输出
    std::vector<int64_t> outputScaleShape = {M, Ceil(N / 2, 64), 2};
    std::vector<int64_t> weight2Shape = {E, K, N / 2};
    std::vector<int64_t> weight2ScaleShape = {E, K, Ceil(N / 2, 64), 2}; // 转置
    std::vector<int64_t> output2Shape = {M * topK * rankSize, K}; // gmm2输出
    std::vector<int64_t> probsShape = {M , topK};
    std::vector<int64_t> expertScalesOptionalShape = {1, 1}; // 没用
    std::vector<int64_t> scalesOptionalShape = {1, 1}; // 没用
    printf("shape constructed!\n");

    // 计算输入shape大小
    long long xShapeSize = GetShapeSize(xShape);
    long long expertIdxShapeSize = GetShapeSize(expertIdxShape);
    long long weightShapeSize = GetShapeSize(weightShape);
    long long weightScaleShapeSize = GetShapeSize(weightScaleShape);
    long long outputShapeSize = GetShapeSize(outputShape);
    long long outputScaleShapeSize = GetShapeSize(outputScaleShape);
    long long weight2ShapeSize = GetShapeSize(weight2Shape);
    long long weight2ScaleShapeSize = GetShapeSize(weight2ScaleShape);
    long long output2ShapeSize = GetShapeSize(output2Shape);
    long long probsHostDataShapeSize = GetShapeSize(probsShape);
    long long expertScalesOptionalShapeSize = GetShapeSize(expertScalesOptionalShape);
    long long scalesOptionalShapeSize = GetShapeSize(scalesOptionalShape);

    // 构造输入数据
    std::vector<int16_t> xHostData(xShapeSize, 15377);
    std::vector<int32_t> expertIdxHostData(expertIdxShapeSize, 60);
    std::vector<int8_t> weightHostData(weightShapeSize, 60);
    std::vector<int8_t> weightScaleHostData(weightScaleShapeSize, 110);
    std::vector<int8_t> weight2HostData(weight2ShapeSize, 60);
    std::vector<int8_t> weight2ScaleHostData(weight2ScaleShapeSize, 116);
    std::vector<float> probsHostData(probsHostDataShapeSize, 60);
    std::vector<int16_t> output2HostData(output2ShapeSize, 0);
    std::vector<float> expertScalesOptionalHostData(expertScalesOptionalShapeSize, 0);
    std::vector<float> scalesOptionalHostData(scalesOptionalShapeSize, 0);

    // 读取bin文件数据
    std::string inputPrefix = "/home/c00588804/0330/a5_dev/dispatch_ffn_combine-a5_dev/gmmsq_com/data/input_";;
    loadBinaryFile(inputPrefix + std::to_string(args.rankId) + "/input_tokens.bin", xHostData);
    loadBinaryFile(inputPrefix + std::to_string(args.rankId) + "/expert_id.bin", expertIdxHostData);
    loadBinaryFile(inputPrefix + std::to_string(args.rankId) + "/b_8.bin", weightHostData);
    loadBinaryFile(inputPrefix + std::to_string(args.rankId) + "/b_scale.bin", weightScaleHostData);
    loadBinaryFile(inputPrefix + std::to_string(args.rankId) + "/b2_8.bin", weight2HostData);
    loadBinaryFile(inputPrefix + std::to_string(args.rankId) + "/b2_scale.bin", weight2ScaleHostData);
    loadBinaryFile(inputPrefix + std::to_string(args.rankId) + "/prob.bin", probsHostData);
    printf("host data constructed!\n");

    // 构造aclnn接口需要的入参
    void *xDeviceAddr = nullptr;
    void *yDeviceAddr = nullptr;
    void *expertIdsDeviceAddr = nullptr;
    void *weight1DeviceAddr = nullptr;
    void *weightscales1DeviceAddr = nullptr;
    void *weight2DeviceAddr = nullptr;
    void *weightscales2DeviceAddr = nullptr;
    void *expertScalesOptionalDeviceAddr = nullptr;
    void *scalesOptionalDeviceAddr = nullptr;
    void *xActiveMaskOptionalDeviceAddr = nullptr;
    aclTensor *x = nullptr;
    aclTensor *y = nullptr;
    aclTensor *expertIds = nullptr;
    aclTensor *weight1 = nullptr;
    aclTensor *weightscales1 = nullptr;
    aclTensor *weight2 = nullptr;
    aclTensor *weightscales2 = nullptr;
    aclTensor *scalesOptional = nullptr;
    aclTensor *xActiveMaskOptional = nullptr;
    aclTensor *expertScalesOptional = nullptr;
    
    // 创建 x
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建 y
    ret = CreateAclTensor(output2HostData, output2Shape, &yDeviceAddr, aclDataType::ACL_FLOAT16, &y);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建 expertIds
    ret = CreateAclTensor(expertIdxHostData, expertIdxShape, &expertIdsDeviceAddr, aclDataType::ACL_INT32, &expertIds);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建 weight1
    ret = CreateAclTensor(weightHostData, weightShape, &weight1DeviceAddr, aclDataType::ACL_FLOAT8_E5M2, &weight1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建 weightscales1
    ret = CreateAclTensor(weightScaleHostData, weightScaleShape, &weightscales1DeviceAddr, aclDataType::ACL_FLOAT8_E8M0, &weightscales1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建 weight2 
    ret = CreateAclTensor(weight2HostData, weight2Shape, &weight2DeviceAddr, aclDataType::ACL_FLOAT8_E5M2, &weight2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建 weightscales2
    ret = CreateAclTensor(weight2ScaleHostData, weight2ScaleShape, &weightscales2DeviceAddr, aclDataType::ACL_FLOAT8_E8M0, &weightscales2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建 expertScalesOptional
    ret = CreateAclTensor(expertScalesOptionalHostData, expertScalesOptionalShape, &expertScalesOptionalDeviceAddr, aclDataType::ACL_BF16, &expertScalesOptional);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建 scalesOptional
    ret = CreateAclTensor(scalesOptionalHostData, scalesOptionalShape, &scalesOptionalDeviceAddr, aclDataType::ACL_FLOAT, &scalesOptional);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    aclOpExecutor *executor = nullptr;
    int64_t epWorldSize = rankSize;
    int64_t groupListType = 1; // 0:cumsum offset 数组 1:count 差分数组
    uint64_t workspaceSize = 0;
    int64_t epRankId = args.rankId;
    int64_t moeExpertNum = E;
    int64_t maxRecvTokenNum = M * topK * rankSize;
    int64_t dispatchQuantMode = 0;
    int64_t combineQuantMode = 0;
    int64_t globalBs = 1;
    std::string commAlg = "mte";
    void *workspaceAddr = nullptr;
    int64_t shareExpertNum = 1;
    int64_t shareExpertRankNum = 0;
    int64_t dispatchQuantOutType = 1;

    // 调用第一阶段接口
    ret = aclnnMegaMoeGetWorkspaceSize(
        x, expertIds, scalesOptional, xActiveMaskOptional, expertScalesOptional, weight1, weightscales1, weight2, weightscales2, hcomName, epWorldSize, epRankId, moeExpertNum,
        maxRecvTokenNum, shareExpertNum, shareExpertRankNum, dispatchQuantMode, dispatchQuantOutType, combineQuantMode, commAlg.c_str(), globalBs, y, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnMegaMoeGetWorkspaceSize failed. ret = %d \n", ret); return ret);

    printf("aclnnMegaMoeGetWorkspaceSize success \n");

    // 根据第一阶段接口计算出的workspaceSize申请device内存
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret); return ret);
    }

    // 调用第二阶段接口
    ret = aclnnMegaMoe(workspaceAddr, workspaceSize, executor, args.stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnMegaMoe failed. ret = %d \n", ret); return ret);

    // （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStreamWithTimeout(args.stream, 10000);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d \n", ret); return ret);
    LOG_PRINT("[INFO] device_%d aclnnMegaMoe execute successfully.\n", args.rankId);

    // 拷贝workspace数据
    WriteResultToBin<int8_t>(args.rankId, "workspace.bin",  workspaceAddr, workspaceSize);
    WriteResultToBin<int16_t>(args.rankId, "output2.bin",  yDeviceAddr, output2ShapeSize);
    printf("Get workspace content done. \n");

    // 释放device资源，需要根据具体API的接口定义修改
    if (x != nullptr) {
        aclDestroyTensor(x);
    }

    if (y != nullptr) {
        aclDestroyTensor(y);
    }

    if (expertIds != nullptr) {
        aclDestroyTensor(expertIds);
    }

    if (weight1 != nullptr) {
        aclDestroyTensor(weight1);
    }

    if (weightscales1 != nullptr) {
        aclrtFree(weightscales1);
    }

    if (weight2 != nullptr) {
        aclrtFree(weight2);
    }

    if (weightscales2 != nullptr) {
        aclrtFree(weightscales2);
    }

    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }

    ret = HcclCommDestroy(args.hcclComm);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclCommDestroy failed. ret = %d \n", ret); return ret);
    ret = aclrtDestroyStream(args.stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtDestroyStream failed. ret = %d \n", ret); return ret);
    ret = aclrtResetDevice(args.rankId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtResetDevice failed. ret = %d \n", ret); return ret);
    ret = aclrtDestroyContext(args.context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtDestroyContext failed. ret = %d \n", ret); return ret);
    return 0;
}

int main(int argc, char *argv[])
{
    // 本样例基于Atlas A5实现，必须在Atlas A5上运行
    int ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclInit failed. ret = %d \n", ret); return ret);
    aclrtStream stream[DEV_NUM];
    aclrtContext context[DEV_NUM];
    for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
        ret = aclrtSetDevice(rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice failed. ret = %d \n", ret); return ret);
        ret = aclrtCreateContext(&context[rankId], rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed. ERROR: %d\n", ret); return ret);
        ret = aclrtCreateStream(&stream[rankId]);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed. ret = %d \n", ret); return ret);
    }

    int32_t devices[DEV_NUM];
    for (int i = 0; i < DEV_NUM; i++) {
        devices[i] = i;
    }

    printf("Begin Init HCCL Common\n");

    // 初始化集合通信域
    HcclComm comms[DEV_NUM];
    ret = HcclCommInitAll(DEV_NUM, devices, comms);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclCommInitAll failed. ret = %d \n", ret); return ret);

    Args args[DEV_NUM];
    // 启动多线程
    std::vector<std::unique_ptr<std::thread>> threads(DEV_NUM);
    for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
        args[rankId].rankId = rankId;
        args[rankId].hcclComm = comms[rankId];
        args[rankId].context = context[rankId];
        args[rankId].stream = stream[rankId];
        threads[rankId].reset(new(std::nothrow) std::thread(&launchOneThread_MegaMoe, std::ref(args[rankId])));
    }

    for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
        threads[rankId]->join();
    }

    aclFinalize();
    return 0;
}
