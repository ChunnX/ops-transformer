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
 * \file test_aclnn_qbmm_reduce_scatter_add_rms_norm_cast.cpp
 * \brief qbmm_reduce_scatter_add_rms_norm_cast 算子测试用例
 */

#include <iostream>
#include <vector>
#include <thread>
#include "hccl/hccl.h"
#include "aclnnop/aclnn_qbmm_reduce_scatter_add_rms_norm_cast.h"

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
constexpr int RANK_SIZE = DEV_NUM;

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
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
        strides[i] = shape[i + 1] * strides[i + 1];
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

int LaunchOneThreadQbmmReduceScatterAddRmsNormCast(Args &args)
{
    int ret = aclrtSetCurrentContext(args.context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetCurrentContext failed. ret = %d\n", ret); return ret);

    char hcomName[128] = {0};
    ret = HcclGetCommName(args.hcclComm, hcomName);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetCommName failed. ret = %d\n", ret); return -1);
    LOG_PRINT("[INFO] rank = %d, hcomName = %s, stream = %p\n", args.rankId, hcomName, args.stream);

    // 定义输入输出shape
    // MX 量化规则：每32个元素共享一个 scale (fp8_e8m0)
    // Scale A shape: (M, K/32) - 用于 A 矩阵反量化
    // Scale B shape: (K/32, N) - 用于 B 矩阵反量化
    std::vector<int64_t> x1Shape = {128, 256};           // M x K (A矩阵)
    std::vector<int64_t> x2Shape = {256, 512};           // K x N (B矩阵, ND格式)
    std::vector<int64_t> yShape = {128, 512};            // residual (BF16)
    std::vector<int64_t> gammaShape = {512};             // RMS Norm gamma
    std::vector<int64_t> scaleShape = {8, 512};          // Scale B: (K/32, N) = (256/32, 512)
    std::vector<int64_t> pertokenScaleShape = {128, 8};  // Scale A: (M, K/32) = (128, 256/32)
    std::vector<int64_t> y1Shape = {128, 512};           // output float
    std::vector<int64_t> y2Shape = {128, 512};           // output bf16
    std::vector<int64_t> xOutShape = {128, 512};         // output bf16

    void *x1DeviceAddr = nullptr;
    void *x2DeviceAddr = nullptr;
    void *yDeviceAddr = nullptr;
    void *gammaDeviceAddr = nullptr;
    void *scaleDeviceAddr = nullptr;
    void *pertokenScaleDeviceAddr = nullptr;
    void *biasDeviceAddr = nullptr;
    void *y1DeviceAddr = nullptr;
    void *y2DeviceAddr = nullptr;
    void *xOutDeviceAddr = nullptr;

    aclTensor *x1 = nullptr;
    aclTensor *x2 = nullptr;
    aclTensor *y = nullptr;
    aclTensor *gamma = nullptr;
    aclTensor *scale = nullptr;
    aclTensor *pertokenScale = nullptr;
    aclTensor *bias = nullptr;
    aclTensor *y1 = nullptr;
    aclTensor *y2 = nullptr;
    aclTensor *xOut = nullptr;

    // 属性参数
    bool transposeX2 = false;
    int64_t dtype = 0;
    float epsilon = 1e-6f;

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    void *workspaceAddr = nullptr;

    // 计算shape size
    long long x1ShapeSize = GetShapeSize(x1Shape);
    long long x2ShapeSize = GetShapeSize(x2Shape);
    long long yShapeSize = GetShapeSize(yShape);
    long long gammaShapeSize = GetShapeSize(gammaShape);
    long long scaleShapeSize = GetShapeSize(scaleShape);
    long long pertokenScaleShapeSize = GetShapeSize(pertokenScaleShape);
    long long y1ShapeSize = GetShapeSize(y1Shape);
    long long y2ShapeSize = GetShapeSize(y2Shape);
    long long xOutShapeSize = GetShapeSize(xOutShape);

    // 创建host数据
    // FP8_E4M3FN: 用 uint8_t 存储（1字节）
    std::vector<uint8_t> x1HostData(x1ShapeSize, 56);
    std::vector<uint8_t> x2HostData(x2ShapeSize, 56);
    // BF16: 用 uint16_t 存储（2字节）
    std::vector<uint16_t> yHostData(yShapeSize, 0x3f80);  // bf16 的 1.0
    // FLOAT: 用 float 存储
    std::vector<float> gammaHostData(gammaShapeSize, 1.0f);
    // FP8_E8M0 (scale): 用 uint8_t 存储（纯指数格式，1字节）
    std::vector<uint8_t> scaleHostData(scaleShapeSize, 127);
    std::vector<uint8_t> pertokenScaleHostData(pertokenScaleShapeSize, 127);
    // 输出初始化为 0
    std::vector<float> y1HostData(y1ShapeSize, 0.0f);
    std::vector<uint16_t> y2HostData(y2ShapeSize, 0);
    std::vector<uint16_t> xOutHostData(xOutShapeSize, 0);

    // 创建输入tensor
    ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &x1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &x2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_BF16, &y);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gammaHostData, gammaShape, &gammaDeviceAddr, aclDataType::ACL_FLOAT, &gamma);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(scaleHostData, scaleShape, &scaleDeviceAddr, aclDataType::ACL_FLOAT8_E8M0, &scale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(pertokenScaleHostData, pertokenScaleShape, &pertokenScaleDeviceAddr, 
                          aclDataType::ACL_FLOAT8_E8M0, &pertokenScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建可选输入tensor (bias 可以为 nullptr，pertokenScale 必须有值)
    bias = nullptr;

    // 创建输出tensor
    ret = CreateAclTensor(y1HostData, y1Shape, &y1DeviceAddr, aclDataType::ACL_FLOAT, &y1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(y2HostData, y2Shape, &y2DeviceAddr, aclDataType::ACL_BF16, &y2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(xOutHostData, xOutShape, &xOutDeviceAddr, aclDataType::ACL_BF16, &xOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    LOG_PRINT("[INFO] rank = %d, Before aclnnQbmmReduceScatterAddRmsNormCastGetWorkspaceSize\n", args.rankId);

    // 调用第一阶段接口
    ret = aclnnQbmmReduceScatterAddRmsNormCastGetWorkspaceSize(
        x1, x2, y, gamma, scale, bias, pertokenScale,
        hcomName, RANK_SIZE, transposeX2, dtype, epsilon,
        y1, y2, xOut,
        &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
        LOG_PRINT("[ERROR] aclnnQbmmReduceScatterAddRmsNormCastGetWorkspaceSize failed. ret = %d \n", ret); return ret);

    LOG_PRINT("[INFO] rank = %d, workspaceSize = %lu\n", args.rankId, workspaceSize);

    // 根据第一阶段接口计算出的workspaceSize申请device内存
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret); return ret);
    }

    LOG_PRINT("[INFO] rank = %d, Before aclnnQbmmReduceScatterAddRmsNormCast\n", args.rankId);

    // 调用第二阶段接口
    ret = aclnnQbmmReduceScatterAddRmsNormCast(workspaceAddr, workspaceSize, executor, args.stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnQbmmReduceScatterAddRmsNormCast failed. ret = %d \n", ret); return ret);

    // 同步等待任务执行结束
    ret = aclrtSynchronizeStreamWithTimeout(args.stream, 10000);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d \n", ret);
        return ret);

    LOG_PRINT("[INFO] rank = %d aclnnQbmmReduceScatterAddRmsNormCast execute successfully.\n", args.rankId);

    // 释放device资源
    if (x1 != nullptr) {
        aclDestroyTensor(x1);
        aclrtFree(x1DeviceAddr);
    }
    if (x2 != nullptr) {
        aclDestroyTensor(x2);
        aclrtFree(x2DeviceAddr);
    }
    if (y != nullptr) {
        aclDestroyTensor(y);
        aclrtFree(yDeviceAddr);
    }
    if (gamma != nullptr) {
        aclDestroyTensor(gamma);
        aclrtFree(gammaDeviceAddr);
    }
    if (scale != nullptr) {
        aclDestroyTensor(scale);
        aclrtFree(scaleDeviceAddr);
    }
    if (pertokenScale != nullptr) {
        aclDestroyTensor(pertokenScale);
        aclrtFree(pertokenScaleDeviceAddr);
    }
    if (y1 != nullptr) {
        aclDestroyTensor(y1);
        aclrtFree(y1DeviceAddr);
    }
    if (y2 != nullptr) {
        aclDestroyTensor(y2);
        aclrtFree(y2DeviceAddr);
    }
    if (xOut != nullptr) {
        aclDestroyTensor(xOut);
        aclrtFree(xOutDeviceAddr);
    }
    if (workspaceSize > 0 && workspaceAddr != nullptr) {
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
        threads[rankId].reset(new(std::nothrow) std::thread(&LaunchOneThreadQbmmReduceScatterAddRmsNormCast,
            std::ref(args[rankId])));
    }

    for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
        threads[rankId]->join();
    }

    aclFinalize();
    return 0;
}