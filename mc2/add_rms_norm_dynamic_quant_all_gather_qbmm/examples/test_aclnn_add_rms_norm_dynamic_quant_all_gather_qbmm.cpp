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
 * \file test_aclnn_add_rms_norm_dynamic_quant_all_gather_qbmm.cpp
 * \brief add_rms_norm_dynamic_quant_all_gather_qbmm 算子测试用例
 */

#include <cmath>
#include <iostream>
#include <vector>
#include <thread>
#include "hccl/hccl.h"
#include "aclnnop/aclnn_add_rms_norm_dynamic_quant_all_gather_qbmm.h"

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

constexpr int DEV_NUM = 4;
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
    aclDataType dataType, aclTensor **tensor, bool isColumnMajor = false, aclFormat format = aclFormat::ACL_FORMAT_ND)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc failed. ret: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMemcpy failed. ret: %d\n", ret); return ret);
    
    std::vector<int64_t> strides(shape.size(), 1);
    if (isColumnMajor) {
        strides[0] = 1;
        for (int64_t i = 1; i < shape.size(); i++) {
            strides[i] = shape[i - 1] * strides[i - 1];
        }
    } else {
        for (int64_t i = shape.size() - 2; i >= 0; i--) {
            strides[i] = shape[i + 1] * strides[i + 1];
        }
    }
    
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, format,
        shape.data(), shape.size(), *deviceAddr);
    return 0;
}

template<typename T>
int LoadBinFile(const std::string &filename, std::vector<T> &hostData) {
    FILE* fp = fopen(filename.c_str(), "rb");
    if (!fp) {
        LOG_PRINT("[ERROR] Failed to open %s\n", filename.c_str());
        return -1;
    }
    
    fseek(fp, 0, SEEK_END);
    long fileSize = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    
    size_t elemCount = fileSize / sizeof(T);
    hostData.resize(elemCount);
    
    size_t readCount = fread(hostData.data(), sizeof(T), elemCount, fp);
    fclose(fp);
    
    if (readCount != elemCount) {
        LOG_PRINT("[ERROR] Read failed: expected %zu, got %zu\n", elemCount, readCount);
        return -1;
    }
    
    LOG_PRINT("[INFO] Loaded %s: %zu elements (%ld bytes)\n", filename.c_str(), elemCount, fileSize);
    return 0;
}

template<typename T>
int SaveBinFile(const std::string &filename, const std::vector<T> &hostData) {
    FILE* fp = fopen(filename.c_str(), "wb");
    if (!fp) {
        LOG_PRINT("[ERROR] Failed to open %s for writing\n", filename.c_str());
        return -1;
    }
    
    size_t writeCount = fwrite(hostData.data(), sizeof(T), hostData.size(), fp);
    fclose(fp);
    
    if (writeCount != hostData.size()) {
        LOG_PRINT("[ERROR] Write failed: expected %zu, got %zu\n", hostData.size(), writeCount);
        return -1;
    }
    
    LOG_PRINT("[INFO] Saved %s: %zu elements (%zu bytes)\n", filename.c_str(), hostData.size(), hostData.size() * sizeof(T));
    return 0;
}

struct Args {
    int rankId;
    HcclComm hcclComm;
    aclrtStream stream;
    aclrtContext context;
};

int LaunchOneThreadAddRmsNormDynamicQuantAllGatherQbmm(Args &args)
{
    int ret = aclrtSetCurrentContext(args.context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetCurrentContext failed. ret = %d\n", ret); return ret);

    char hcomName[128] = {0};
    ret = HcclGetCommName(args.hcclComm, hcomName);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetCommName failed. ret = %d\n", ret); return -1);
    LOG_PRINT("[INFO] rank = %d, hcomName = %s, stream = %p\n", args.rankId, hcomName, args.stream);

    // 测试用例: M_per_rank=63 → AllGather → M_total=128, K=5120, N=3072
    int mPerRank = 63, kDim = 5120, nDim = 3072;
    bool isNzMode = (getenv("NZ_MODE") != nullptr);
    int64_t kd = (kDim + 63) / 64 * 2;  // ceil(K/64)*2
    int64_t nt_nz = (nDim + 31) / 32;   // ceil(N/32) — N blocks for NZ
    int64_t kt_nz = (kDim + 15) / 16;   // ceil(K/16) — K blocks for NZ

    std::vector<int64_t> x1Shape = {mPerRank, kDim};
    std::vector<int64_t> scaleInv1Shape = {mPerRank, kd};
    // NZ: (ceil(N/32), ceil(K/16), 16, 32); ND: (N, K)
    std::vector<int64_t> x2Shape = isNzMode
        ? std::vector<int64_t>{nt_nz, kt_nz, 16, 32}
        : std::vector<int64_t>{nDim, kDim};
    std::vector<int64_t> residualShape = {mPerRank, kDim};
    std::vector<int64_t> yShape = {mPerRank, kDim};
    std::vector<int64_t> gammaShape = {kDim};
    // NZ: ScaleB (kd, N, 2); ND: (N, kd, 2)
    std::vector<int64_t> scaleInv2Shape = isNzMode
        ? std::vector<int64_t>{kd, nDim, 2}
        : std::vector<int64_t>{nDim, kd, 2};
    std::vector<int64_t> outputShape = {mPerRank * RANK_SIZE, nDim};
    std::vector<int64_t> zShape = {mPerRank, kDim};
    std::vector<int64_t> dynamicQuantOutShape = {mPerRank, kDim};
    std::vector<int64_t> allGatherDataOutShape = {mPerRank * RANK_SIZE, kDim};
    std::vector<int64_t> allGatherScalesOutShape = {mPerRank * RANK_SIZE, kd};

    void *x1DeviceAddr = nullptr;
    void *scaleInv1DeviceAddr = nullptr;
    void *x2DeviceAddr = nullptr;
    void *residualDeviceAddr = nullptr;
    void *yDeviceAddr = nullptr;
    void *gammaDeviceAddr = nullptr;
    void *scaleInv2DeviceAddr = nullptr;
    void *biasDeviceAddr = nullptr;
    void *outputDeviceAddr = nullptr;
    void *zDeviceAddr = nullptr;
    void *dynamicQuantOutDeviceAddr = nullptr;
    void *allGatherDataOutDeviceAddr = nullptr;
    void *allGatherScalesOutDeviceAddr = nullptr;

    aclTensor *x1 = nullptr;
    aclTensor *scaleInv1 = nullptr;
    aclTensor *x2 = nullptr;
    aclTensor *residual = nullptr;
    aclTensor *y = nullptr;
    aclTensor *gamma = nullptr;
    aclTensor *scaleInv2 = nullptr;
    aclTensor *smoothScale = nullptr;
    aclTensor *bias = nullptr;
    aclTensor *output = nullptr;
    aclTensor *z = nullptr;
    aclTensor *dynamicQuantOut = nullptr;
    aclTensor *allGatherDataOut = nullptr;
    aclTensor *allGatherScalesOut = nullptr;

    // 属性参数
    bool transposeX2 = false;
    int64_t dtype = -1;  // 当前只支持-1，即int8
    int64_t residualNormMode = 0;
    bool outputOptional = false;

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    void *workspaceAddr = nullptr;

    // 计算shape size
    long long x1ShapeSize = GetShapeSize(x1Shape);
    long long x2ShapeSize = GetShapeSize(x2Shape);
    long long residualShapeSize = GetShapeSize(residualShape);
    long long yShapeSize = GetShapeSize(yShape);
    long long gammaShapeSize = GetShapeSize(gammaShape);
    long long outputShapeSize = GetShapeSize(outputShape);
    long long zShapeSize = GetShapeSize(zShape);
    long long dynamicQuantOutShapeSize = GetShapeSize(dynamicQuantOutShape);
    long long allGatherDataOutShapeSize = GetShapeSize(allGatherDataOutShape);
    long long allGatherScalesOutShapeSize = GetShapeSize(allGatherScalesOutShape);
    
    std::string inputDir = "./input";
    char* envInputDir = getenv("INPUT_DIR");
    if (envInputDir != nullptr) {
        inputDir = std::string(envInputDir);
    }
    LOG_PRINT("[INFO] rank = %d, inputDir = %s\n", args.rankId, inputDir.c_str());
    
    std::vector<uint8_t> x1HostData;
    std::vector<uint8_t> scaleInv1HostData;
    std::vector<uint8_t> x2HostData;
    std::vector<uint16_t> residualHostData = {0};
    std::vector<uint16_t> yHostData = {0};
    std::vector<float> gammaHostData = {0.0f};
    std::vector<uint8_t> scaleInv2HostData;
    
    std::string rankSuffix = "_rank" + std::to_string(args.rankId) + ".bin";

    ret = LoadBinFile(inputDir + "/x1_fp8" + rankSuffix, x1HostData);
    CHECK_RET(ret == 0, return ret);
    ret = LoadBinFile(inputDir + "/scaleInv1_e8m0" + rankSuffix, scaleInv1HostData);
    CHECK_RET(ret == 0, return ret);
    ret = LoadBinFile(inputDir + "/x2_fp8" + rankSuffix, x2HostData);
    CHECK_RET(ret == 0, return ret);
    ret = LoadBinFile(inputDir + "/scaleInv2_e8m0" + rankSuffix, scaleInv2HostData);
    CHECK_RET(ret == 0, return ret);
    
    // 输出初始化为0
    std::vector<uint16_t> outputHostData(outputShapeSize, 0);
    std::vector<uint16_t> zHostData(zShapeSize, 0);
    std::vector<uint8_t> dynamicQuantOutHostData(dynamicQuantOutShapeSize, 0);
    std::vector<uint8_t> allGatherDataOutHostData(allGatherDataOutShapeSize, 0);
    std::vector<uint8_t> allGatherScalesOutHostData(allGatherScalesOutShapeSize, 0);

    ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &x1);
    ret = CreateAclTensor(scaleInv1HostData, scaleInv1Shape, &scaleInv1DeviceAddr, aclDataType::ACL_FLOAT8_E8M0, &scaleInv1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    
    ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &x2, false,
        isNzMode ? aclFormat::ACL_FORMAT_FRACTAL_NZ : aclFormat::ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    
    ret = CreateAclTensor(residualHostData, residualShape, &residualDeviceAddr, aclDataType::ACL_BF16, &residual);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_BF16, &y);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gammaHostData, gammaShape, &gammaDeviceAddr, aclDataType::ACL_FLOAT, &gamma);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    
    // ret = CreateAclTensor(scaleInv1HostData, scaleInv1Shape, &scaleInv1DeviceAddr, aclDataType::ACL_FLOAT8_E8M0, &scaleInv1);
    // CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(scaleInv2HostData, scaleInv2Shape, &scaleInv2DeviceAddr, aclDataType::ACL_FLOAT8_E8M0, &scaleInv2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    smoothScale = nullptr;
    bias = nullptr;

    // 创建输出tensor
    ret = CreateAclTensor(outputHostData, outputShape, &outputDeviceAddr, aclDataType::ACL_BF16, &output);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(zHostData, zShape, &zDeviceAddr, aclDataType::ACL_BF16, &z);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(dynamicQuantOutHostData, dynamicQuantOutShape, &dynamicQuantOutDeviceAddr, 
                          aclDataType::ACL_FLOAT8_E4M3FN, &dynamicQuantOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(allGatherDataOutHostData, allGatherDataOutShape, &allGatherDataOutDeviceAddr, 
                          aclDataType::ACL_FLOAT8_E4M3FN, &allGatherDataOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(allGatherScalesOutHostData, allGatherScalesOutShape, &allGatherScalesOutDeviceAddr, 
                          aclDataType::ACL_FLOAT8_E8M0, &allGatherScalesOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    LOG_PRINT("[INFO] rank = %d, Before aclnnAddRmsNormDynamicQuantAllGatherQbmmGetWorkspaceSize\n", args.rankId);

    // 调用第一阶段接口
    ret = aclnnAddRmsNormDynamicQuantAllGatherQbmmGetWorkspaceSize(
        x1, x2, residual, y, gamma, scaleInv1, scaleInv2, smoothScale, bias,
        hcomName, RANK_SIZE, transposeX2, dtype, residualNormMode, outputOptional,
        output, z, dynamicQuantOut, allGatherDataOut, allGatherScalesOut,
        &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
        LOG_PRINT("[ERROR] aclnnAddRmsNormDynamicQuantAllGatherQbmmGetWorkspaceSize failed. ret = %d \n", ret); return ret);

    LOG_PRINT("[INFO] rank = %d, workspaceSize = %lu\n", args.rankId, workspaceSize);

    // 根据第一阶段接口计算出的workspaceSize申请device内存
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret); return ret);
    }

    LOG_PRINT("[INFO] rank = %d, Before aclnnAddRmsNormDynamicQuantAllGatherQbmm\n", args.rankId);

    // 调用第二阶段接口
    ret = aclnnAddRmsNormDynamicQuantAllGatherQbmm(workspaceAddr, workspaceSize, executor, args.stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnAddRmsNormDynamicQuantAllGatherQbmm failed. ret = %d \n", ret); return ret);

    // 同步等待任务执行结束
    ret = aclrtSynchronizeStreamWithTimeout(args.stream, 10000);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d \n", ret);
        return ret);

    LOG_PRINT("[INFO] rank = %d aclnnAddRmsNormDynamicQuantAllGatherQbmm execute successfully.\n", args.rankId);

    // === 精度测试：保存输出到bin文件 ===
    std::string outputDir = "./output";
    char* envOutputDir = getenv("OUTPUT_DIR");
    if (envOutputDir != nullptr) {
        outputDir = std::string(envOutputDir);
    }
    
    std::vector<uint16_t> outputResult(outputShapeSize);
    std::vector<uint16_t> zResult(zShapeSize);
    std::vector<uint8_t> dynamicQuantOutResult(dynamicQuantOutShapeSize);
    std::vector<uint8_t> allGatherDataOutResult(allGatherDataOutShapeSize);
    std::vector<uint8_t> allGatherScalesOutResult(allGatherScalesOutShapeSize);

    // Copy output (bf16)
    ret = aclrtMemcpy(outputResult.data(), outputShapeSize * sizeof(uint16_t),
                      outputDeviceAddr, outputShapeSize * sizeof(uint16_t),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS,
        LOG_PRINT("[ERROR] aclrtMemcpy output failed. ret = %d\n", ret); return ret);

    // Copy z (bf16)
    ret = aclrtMemcpy(zResult.data(), zShapeSize * sizeof(uint16_t),
                      zDeviceAddr, zShapeSize * sizeof(uint16_t),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS,
        LOG_PRINT("[ERROR] aclrtMemcpy z failed. ret = %d\n", ret); return ret);

    // Copy dynamic_quant_out (fp8 e4m3)
    ret = aclrtMemcpy(dynamicQuantOutResult.data(), dynamicQuantOutShapeSize * sizeof(uint8_t),
                      dynamicQuantOutDeviceAddr, dynamicQuantOutShapeSize * sizeof(uint8_t),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS,
        LOG_PRINT("[ERROR] aclrtMemcpy dynamic_quant_out failed. ret = %d\n", ret); return ret);

    // Copy all_gather_data_out (fp8 e4m3)
    ret = aclrtMemcpy(allGatherDataOutResult.data(), allGatherDataOutShapeSize * sizeof(uint8_t),
                      allGatherDataOutDeviceAddr, allGatherDataOutShapeSize * sizeof(uint8_t),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS,
        LOG_PRINT("[ERROR] aclrtMemcpy all_gather_data_out failed. ret = %d\n", ret); return ret);

    // Copy all_gather_scales_out (fp8 e8m0)
    ret = aclrtMemcpy(allGatherScalesOutResult.data(), allGatherScalesOutShapeSize * sizeof(uint8_t),
                      allGatherScalesOutDeviceAddr, allGatherScalesOutShapeSize * sizeof(uint8_t),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS,
        LOG_PRINT("[ERROR] aclrtMemcpy all_gather_scales_out failed. ret = %d\n", ret); return ret);

    // Save output
    std::string outputFile = outputDir + "/npu_output_rank" + std::to_string(args.rankId) + ".bin";
    ret = SaveBinFile(outputFile, outputResult);
    CHECK_RET(ret == 0, return ret);

    // Save z
    std::string zFile = outputDir + "/z_output_rank" + std::to_string(args.rankId) + ".bin";
    ret = SaveBinFile(zFile, zResult);
    CHECK_RET(ret == 0, return ret);

    // Save dynamic_quant_out
    std::string dynQuantFile = outputDir + "/dynamic_quant_out_rank" + std::to_string(args.rankId) + ".bin";
    ret = SaveBinFile(dynQuantFile, dynamicQuantOutResult);
    CHECK_RET(ret == 0, return ret);

    // Save all_gather_data_out
    std::string allGatherDataFile = outputDir + "/all_gather_data_out_rank" + std::to_string(args.rankId) + ".bin";
    ret = SaveBinFile(allGatherDataFile, allGatherDataOutResult);
    CHECK_RET(ret == 0, return ret);

    // Save all_gather_scales_out
    std::string allGatherScalesFile = outputDir + "/all_gather_scales_out_rank" + std::to_string(args.rankId) + ".bin";
    ret = SaveBinFile(allGatherScalesFile, allGatherScalesOutResult);
    CHECK_RET(ret == 0, return ret);

    // 打印前10个值验证
    LOG_PRINT("[INFO] rank=%d, output first 10 values (bf16 hex):\n", args.rankId);
    for (int i = 0; i < 10 && i < outputShapeSize; i++) {
        LOG_PRINT("  [%d] 0x%04x\n", i, outputResult[i]);
    }
    LOG_PRINT("[INFO] rank=%d, z first 10 values (bf16 hex):\n", args.rankId);
    for (int i = 0; i < 10 && i < zShapeSize; i++) {
        LOG_PRINT("  [%d] 0x%04x\n", i, zResult[i]);
    }
    LOG_PRINT("[INFO] rank=%d, dynamic_quant_out first 10 values (fp8 raw):\n", args.rankId);
    for (int i = 0; i < 10 && i < dynamicQuantOutShapeSize; i++) {
        LOG_PRINT("  [%d] 0x%02x\n", i, dynamicQuantOutResult[i]);
    }

    // 释放device资源
    if (x1 != nullptr) {
        aclDestroyTensor(x1);
        aclrtFree(x1DeviceAddr);
    }
    if (x2 != nullptr) {
        aclDestroyTensor(x2);
        aclrtFree(x2DeviceAddr);
    }
    if (residual != nullptr) {
        aclDestroyTensor(residual);
        aclrtFree(residualDeviceAddr);
    }
    if (y != nullptr) {
        aclDestroyTensor(y);
        aclrtFree(yDeviceAddr);
    }
    if (gamma != nullptr) {
        aclDestroyTensor(gamma);
        aclrtFree(gammaDeviceAddr);
    }
    if (scaleInv2 != nullptr) {
        aclDestroyTensor(scaleInv2);
        aclrtFree(scaleInv2DeviceAddr);
    }
    if (output != nullptr) {
        aclDestroyTensor(output);
        aclrtFree(outputDeviceAddr);
    }
    if (z != nullptr) {
        aclDestroyTensor(z);
        aclrtFree(zDeviceAddr);
    }
    if (dynamicQuantOut != nullptr) {
        aclDestroyTensor(dynamicQuantOut);
        aclrtFree(dynamicQuantOutDeviceAddr);
    }
    if (allGatherDataOut != nullptr) {
        aclDestroyTensor(allGatherDataOut);
        aclrtFree(allGatherDataOutDeviceAddr);
    }
    if (allGatherScalesOut != nullptr) {
        aclDestroyTensor(allGatherScalesOut);
        aclrtFree(allGatherScalesOutDeviceAddr);
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
        // ret = aclrtSetDevice(rankId+2);
        ret = aclrtSetDevice(rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice failed. ret = %d \n", ret); return ret);
        // ret = aclrtCreateContext(&context[rankId], rankId+2);
        ret = aclrtCreateContext(&context[rankId], rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed. ERROR: %d\n", ret); return ret);
        ret = aclrtCreateStream(&stream[rankId]);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed. ret = %d \n", ret); return ret);
    }

    int32_t devices[DEV_NUM];
    for (int i = 0; i < DEV_NUM; i++) {
        // devices[i] = i+2;
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
        // args[rankId].rankId = rankId+2;
        args[rankId].rankId = rankId;
        args[rankId].hcclComm = comms[rankId];
        args[rankId].context = context[rankId];
        args[rankId].stream = stream[rankId];
        threads[rankId].reset(new(std::nothrow) std::thread(&LaunchOneThreadAddRmsNormDynamicQuantAllGatherQbmm,
            std::ref(args[rankId])));
    }

    for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
        threads[rankId]->join();
    }

    aclFinalize();
    return 0;
}