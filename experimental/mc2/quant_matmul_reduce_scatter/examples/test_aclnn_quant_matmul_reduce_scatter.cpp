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
 * \file test_aclnn_quant_matmul_reduce_scatter.cpp
 * \brief quant_matmul_reduce_scatter 算子测试用例
 */

#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include "hccl/hccl.h"
#include "aclnnop/aclnn_quant_matmul_reduce_scatter.h"

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
static bool gPerfMode = false;
constexpr int PERF_LOOP_COUNT = 20;
static std::mutex barrierMtx;
static std::condition_variable barrierCv;
static int barrierCount = 0;

// Setup barrier: ensures all ranks complete MC2 setup before any launches kernel
static std::mutex setupMtx;
static std::condition_variable setupCv;
static int setupCount = 0;

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
        // ColumnMajor stride: [1, shape[0], shape[0]*shape[1], ...]
        strides[0] = 1;
        for (int64_t i = 1; i < shape.size(); i++) {
            strides[i] = shape[i - 1] * strides[i - 1];
        }
        // LOG_PRINT("[DEBUG] ColumnMajor shape [%lld, %lld] → strides [%lld, %lld]\n",
        //     shape[0], shape[1], strides[0], strides[1]);
    } else {
        // RowMajor stride: [shape[1], 1] for 2D
        for (int64_t i = shape.size() - 2; i >= 0; i--) {
            strides[i] = shape[i + 1] * strides[i + 1];
        }
    }
    
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0, format,
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

int LaunchOneThreadQuantMatmulReduceScatter(Args &args)
{
    int ret = aclrtSetCurrentContext(args.context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetCurrentContext failed. ret = %d\n", ret); return ret);

    char hcomName[128] = {0};
    ret = HcclGetCommName(args.hcclComm, hcomName);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetCommName failed. ret = %d\n", ret); return -1);
    LOG_PRINT("[INFO] rank = %d, hcomName = %s, stream = %p\n", args.rankId, hcomName, args.stream);

    bool isNzMode = (getenv("NZ_MODE") != nullptr);
    int64_t M = 252, K = 2560, N = 5120;
    if (getenv("M_PER_RANK")) M = atoi(getenv("M_PER_RANK")) * RANK_SIZE;
    if (getenv("K_DIM")) K = atoi(getenv("K_DIM"));
    if (getenv("N_DIM")) N = atoi(getenv("N_DIM"));
    int64_t kt = (K + 15) / 16, nt = (N + 31) / 32;
    std::vector<int64_t> x1Shape = {M, K};
    std::vector<int64_t> x2Shape = isNzMode ? std::vector<int64_t>{nt, kt, 16, 32} : std::vector<int64_t>{K, N};
    int64_t kd = (K + 63) / 64;
    std::vector<int64_t> scaleShape = isNzMode ? std::vector<int64_t>{kd, N, 2} : std::vector<int64_t>{N, kd, 2};
    std::vector<int64_t> pertokenScaleShape = {M, kd, 2};
    std::vector<int64_t> xOutShape = {M / RANK_SIZE, N};
    void *x1DeviceAddr = nullptr;
    void *x2DeviceAddr = nullptr;
    void *scaleDeviceAddr = nullptr;
    void *pertokenScaleDeviceAddr = nullptr;
    void *xOutDeviceAddr = nullptr;

    aclTensor *x1 = nullptr;
    aclTensor *x2 = nullptr;
    aclTensor *scale = nullptr;
    aclTensor *pertokenScale = nullptr;
    aclTensor *xOut = nullptr;

    bool transposeX2 = false;

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    void *workspaceAddr = nullptr;

    long long x1ShapeSize = GetShapeSize(x1Shape);
    long long x2ShapeSize = GetShapeSize(x2Shape);
    long long scaleShapeSize = GetShapeSize(scaleShape);
    long long pertokenScaleShapeSize = GetShapeSize(pertokenScaleShape);
    long long xOutShapeSize = GetShapeSize(xOutShape);
    
    // 精度测试：从bin文件加载输入数据（而不是硬编码）
    // 支持从环境变量 INPUT_DIR 读取路径，默认 "./input"
    std::string inputDir = "./input";
    char* envInputDir = getenv("INPUT_DIR");
    if (envInputDir != nullptr) {
        inputDir = std::string(envInputDir);
    }
    LOG_PRINT("[INFO] rank = %d, inputDir = %s\n", args.rankId, inputDir.c_str());
    
    std::vector<uint8_t> x1HostData;
    std::vector<uint8_t> x2HostData;
    std::vector<uint8_t> scaleHostData;
    std::vector<uint8_t> pertokenScaleHostData;
    
    // Per-rank inputs: A, scaleA, y, gamma — different per rank
    // Shared inputs: B, scaleB — same for all ranks (weight matrix)
    std::string rankSuffix = "_rank" + std::to_string(args.rankId);
    ret = LoadBinFile(inputDir + "/input_a_fp8" + rankSuffix + ".bin", x1HostData);
    CHECK_RET(ret == 0, return ret);
    ret = LoadBinFile(inputDir + "/input_scaleA" + rankSuffix + ".bin", pertokenScaleHostData);
    CHECK_RET(ret == 0, return ret);
    ret = LoadBinFile(inputDir + "/input_b_fp8" + rankSuffix + ".bin", x2HostData);
    CHECK_RET(ret == 0, return ret);
    ret = LoadBinFile(inputDir + "/input_scaleB" + rankSuffix + ".bin", scaleHostData);
    CHECK_RET(ret == 0, return ret);
    
    // 输出初始化为0
    std::vector<uint16_t> xOutHostData(xOutShapeSize, 0);

    // 创建输入tensor
    ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &x1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    
    ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &x2, false, isNzMode ? aclFormat::ACL_FORMAT_FRACTAL_NZ : aclFormat::ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    
    ret = CreateAclTensor(scaleHostData, scaleShape, &scaleDeviceAddr, aclDataType::ACL_FLOAT8_E8M0, &scale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    
    ret = CreateAclTensor(pertokenScaleHostData, pertokenScaleShape, &pertokenScaleDeviceAddr, 
                          aclDataType::ACL_FLOAT8_E8M0, &pertokenScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建输出tensor
    ret = CreateAclTensor(xOutHostData, xOutShape, &xOutDeviceAddr, aclDataType::ACL_BF16, &xOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    LOG_PRINT("[INFO] rank = %d, Before aclnnQuantMatmulReduceScatterGetWorkspaceSize\n", args.rankId);
    
    // // Setup barrier: wait for all ranks to finish MC2 setup before launching kernels
    // {
    //     std::unique_lock<std::mutex> lock(setupMtx);
    //     setupCount++;
    //     if (setupCount >= static_cast<int>(DEV_NUM)) {
    //         setupCv.notify_all();
    //     } else {
    //         setupCv.wait(lock, [&]() { return setupCount >= static_cast<int>(DEV_NUM); });
    //     }
    // }

    // 性能测试模式：通过环境变量 PERF_MODE=1 开启，默认执行 1 次
    gPerfMode = (getenv("PERF_MODE") != nullptr);
    int loopCount = gPerfMode ? PERF_LOOP_COUNT : 1;
    for (int i = 0; i < loopCount; i++) {
        // 调用第一阶段接口
        ret = aclnnQuantMatmulReduceScatterGetWorkspaceSize(
            x1, x2, scale, pertokenScale,
            hcomName, RANK_SIZE, transposeX2,
            xOut,
            &workspaceSize, &executor);
        CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[ERROR] aclnnQuantMatmulReduceScatterGetWorkspaceSize failed. ret = %d \n", ret); return ret);

        LOG_PRINT("[INFO] rank = %d, workspaceSize = %lu\n", args.rankId, workspaceSize);

        // 根据第一阶段接口计算出的workspaceSize申请device内存
        if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret); return ret);
        }

        LOG_PRINT("[INFO] rank = %d, Before aclnnQuantMatmulReduceScatter\n", args.rankId);

    
        // 调用第二阶段接口
        ret = aclnnQuantMatmulReduceScatter(workspaceAddr, workspaceSize, executor, args.stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnQuantMatmulReduceScatter failed. ret = %d \n", ret); return ret);

        // 同步等待任务执行结束
        ret = aclrtSynchronizeStreamWithTimeout(args.stream, 30000);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d \n", ret);
            return ret);
    }

    LOG_PRINT("[INFO] rank = %d aclnnQuantMatmulReduceScatter execute successfully. loops=%d\n", args.rankId, loopCount);

    // === 精度测试：保存输出到bin文件 ===
    std::string outputDir = "./output";
    char* envOutputDir = getenv("OUTPUT_DIR");
    if (envOutputDir != nullptr) {
        outputDir = std::string(envOutputDir);
    }
    
    std::vector<uint16_t> xOutResult(xOutShapeSize);

    // Copy xOut (bf16)
    ret = aclrtMemcpy(xOutResult.data(), xOutShapeSize * sizeof(uint16_t),
                      xOutDeviceAddr, xOutShapeSize * sizeof(uint16_t),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS,
        LOG_PRINT("[ERROR] aclrtMemcpy xOut failed. ret = %d\n", ret); return ret);

    // Save xOut
    std::string xOutFile = outputDir + "/npu_xout_rank" + std::to_string(args.rankId) + ".bin";
    ret = SaveBinFile(xOutFile, xOutResult);
    CHECK_RET(ret == 0, return ret);

    // 打印前10个值验证
    LOG_PRINT("[INFO] rank=%d, xOut first 10 values (bf16 hex):\n", args.rankId);
    for (int i = 0; i < 10 && i < xOutShapeSize; i++) {
        LOG_PRINT("  [%d] 0x%04x\n", i, xOutResult[i]);
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
    if (scale != nullptr) {
        aclDestroyTensor(scale);
        aclrtFree(scaleDeviceAddr);
    }
    if (pertokenScale != nullptr) {
        aclDestroyTensor(pertokenScale);
        aclrtFree(pertokenScaleDeviceAddr);
    }
    if (xOut != nullptr) {
        aclDestroyTensor(xOut);
        aclrtFree(xOutDeviceAddr);
    }
    if (workspaceSize > 0 && workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }

    // // Wait for all ranks to complete kernel execution before destroying HCCL comm
    // {
    //     std::unique_lock<std::mutex> lock(barrierMtx);
    //     barrierCount++;
    //     if (barrierCount >= static_cast<int>(DEV_NUM)) {
    //         barrierCv.notify_all();
    //     } else {
    //         barrierCv.wait(lock, [&]() { return barrierCount >= static_cast<int>(DEV_NUM); });
    //     }
    // }

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
        threads[rankId].reset(new(std::nothrow) std::thread(&LaunchOneThreadQuantMatmulReduceScatter,
            std::ref(args[rankId])));
    }

    for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
        threads[rankId]->join();
    }

    aclFinalize();
    return 0;
}