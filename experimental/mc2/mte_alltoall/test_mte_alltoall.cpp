/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the CANN Open Software License Agreement Version 2.0.
 */

// ============================================================================
// MTE AllToAll 算子 - 多卡多线程 Host 测试代码
// ============================================================================
//
// AllToAll 语义：
//   卡 myRankId 从卡 remoteRank 接收第 myRankId 块数据：
//     output[remoteRank * sendCount + k] = input_of_rank(remoteRank)[myRankId * sendCount + k]
//
// 验证方法：
//   每张卡的输入: x[i] = (rankId+1)*10 + (i%100)
//   期望输出:
//     output[remoteRank * sendCount + k] = (remoteRank+1)*10 + ((myRankId*sendCount+k)%100)
//
// 对齐约束：sendCount 必须是 32 的倍数（DataCopy 要求 count*sizeof(int8_t) 为 32B 对齐）

#include <thread>
#include <vector>
#include <cstring>
#include "hccl/hccl.h"
#include "acl/acl.h"
#include "hccl/hcom.h"
#include "tiling/hccl/hccl_tiling.h"
#include "./kernel/mte_alltoall_tiling_data.h"
#include "./mte_alltoall_kernel_host.h"

using namespace std;
using namespace AscendC;

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

constexpr int64_t BLOCK_NUM = 64;       // 使用的 AIV 核数
constexpr int64_t DEV_NUM = 4;          // 卡数（需与实际环境匹配）
int streamWithTimeout = 10000;

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
}

template<typename T>
int CreateDeviceTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    return ACL_SUCCESS;
}

// HCCL 外部接口声明
extern "C" HcclResult HcclAllocComResourceByTiling(HcclComm comm, void *stream, void *mc2Tiling, void **commContext);

// ============================================================================
// 创建 Tiling 数据和 MC2 通信 Context
// ============================================================================
void CreatTilingDataAndContext(const char* hcomName, aclrtStream stream,
                                void **deviceTilingAddr, void **deviceContextAddr,
                                int64_t sendCount, int64_t rankDim)
{
    MteAllToAllTilingData tilingData;
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling;
    tilingData.mc2InitTiling = mc2InitTiling;
    tilingData.mc2CcTiling = mc2CcTiling;

    // AllToAll 业务 Tiling 参数
    tilingData.mteAllToAllTilingInfo.sendCount = sendCount;
    tilingData.mteAllToAllTilingInfo.rankDim = rankDim;
    tilingData.mteAllToAllTilingInfo.aivNum = BLOCK_NUM;

    // MC2 通信 Tiling 配置
    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(hcomName, 8, "AlltoAll=level0:fullmesh;level1:pairwise");
    mc2CcTilingConfig.SetCommEngine(3);  // 3 = MTE 通信引擎
    mc2CcTilingConfig.GetTiling(tilingData.mc2InitTiling);
    mc2CcTilingConfig.GetTiling(tilingData.mc2CcTiling);

    // 将 Tiling 数据拷贝到 Device
    int tilingSize = sizeof(MteAllToAllTilingData);
    aclrtMalloc(deviceTilingAddr, tilingSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(*deviceTilingAddr, tilingSize, &tilingData, tilingSize, ACL_MEMCPY_HOST_TO_DEVICE);

    // 通过 HCCL 分配通信资源，生成 mc2Context
    HcclComm commHandle;
    HcomGetCommHandleByGroup(hcomName, &commHandle);
    void *mc2Context = nullptr;
    HcclAllocComResourceByTiling(commHandle, stream, &tilingData, &mc2Context);
    if (mc2Context == nullptr) {
        return;
    }

    int64_t mc2ContextSize = 576;
    aclrtMalloc(deviceContextAddr, mc2ContextSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(*deviceContextAddr, mc2ContextSize, mc2Context, mc2ContextSize, ACL_MEMCPY_HOST_TO_DEVICE);
}

// ============================================================================
// 每个 rank 的线程参数
// ============================================================================
struct Args {
    uint32_t rankId;
    uint32_t rankDim;
    HcclComm hcclComm;
    aclrtStream stream;
    aclrtContext context;
};

// ============================================================================
// 分配 Device 内存
// ============================================================================
void AllocDeviceMemory(Args& args, const char* hcomName, void** xDevice, void** workspace,
                        void** tiling, void** mc2Context, int64_t sendCount, int64_t rankDim,
                        int64_t workSpaceSize)
{
    CreatTilingDataAndContext(hcomName, args.stream, tiling, mc2Context, sendCount, rankDim);

    // AllToAll 输入: [rankDim, sendCount]，总元素数 = rankDim * sendCount
    int64_t totalElements = rankDim * sendCount;
    std::vector<int64_t> xShape = {totalElements};

    if (workSpaceSize > 0) {
        aclrtMalloc(workspace, workSpaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }

    // 生成 rank 差异化的输入数据
    long long xShapeSize = GetShapeSize(xShape);
    std::vector<int8_t> xHostData(xShapeSize);
    for (int64_t i = 0; i < xShapeSize; ++i) {
        xHostData[i] = static_cast<int8_t>((args.rankId + 1) * 10 + (i % 100));
    }
    CreateDeviceTensor(xHostData, xShape, xDevice);
}

// ============================================================================
// 释放 Device 内存
// ============================================================================
int FreeDeviceMemory(Args& args, void *xDeviceAddr, void *workspaceAddr,
                      void *tilingAddr, void *mc2ContextAddr, void *outputDeviceAddr,
                      int64_t workSpaceSize)
{
    if (xDeviceAddr != nullptr)  aclrtFree(xDeviceAddr);
    if (outputDeviceAddr != nullptr) aclrtFree(outputDeviceAddr);
    if (mc2ContextAddr != nullptr) aclrtFree(mc2ContextAddr);
    if (tilingAddr != nullptr) aclrtFree(tilingAddr);
    if (workspaceAddr != nullptr) aclrtFree(workspaceAddr);
    HcclCommDestroy(args.hcclComm);
    aclrtDestroyStream(args.stream);
    aclrtResetDevice(args.rankId);
    aclrtDestroyContext(args.context);
    return ACL_SUCCESS;
}

// ============================================================================
// 单个 rank 的 Kernel 执行和 AllToAll 验证
// ============================================================================
int LaunchOneThread(Args &args)
{
    aclrtSetCurrentContext(args.context);
    char hcomName[128] = {0};
    HcclGetCommName(args.hcclComm, hcomName);
    LOG_PRINT("[INFO] rank = %d, hcomName = %s, stream = %p\n", args.rankId, hcomName, args.stream);

    // AllToAll 参数
    // sendCount 必须是 32 的倍数（DataCopy 32B 对齐约束）
    int64_t sendCount = 1024;   // 每个块的大小（元素数）
    int64_t rankDim = args.rankDim;
    int64_t totalElements = rankDim * sendCount;
    int64_t workSpaceSize = 256 * 1024 * 1024;

    void *xDeviceAddr = nullptr;
    void *workspaceAddr = nullptr;
    void *tilingAddr = nullptr;
    void *mc2ContextAddr = nullptr;
    void *outputDeviceAddr = nullptr;
    AllocDeviceMemory(args, hcomName, &xDeviceAddr, &workspaceAddr,
                      &tilingAddr, &mc2ContextAddr, sendCount, rankDim, workSpaceSize);

    // AllToAll 输出: 与输入大小相同 [rankDim, sendCount]
    std::vector<int64_t> outputShape = {totalElements};
    long long outputShapeSize = totalElements * sizeof(int8_t);
    aclrtMalloc(&outputDeviceAddr, outputShapeSize, ACL_MEM_MALLOC_HUGE_FIRST);

    // 调用 Kernel
    MteAllToAllDemo(BLOCK_NUM, args.stream, (uint8_t*)xDeviceAddr,
                        (uint8_t*)outputDeviceAddr, (uint8_t*)workspaceAddr,
                        (uint8_t*)mc2ContextAddr, (uint8_t*)tilingAddr);
    aclrtSynchronizeStreamWithTimeout(args.stream, streamWithTimeout);
    LOG_PRINT("[INFO] device_%d mte_alltoall execute successfully.\n", args.rankId);

    // ---- 结果验证（AllToAll 语义） ----
    int8_t *output_host;
    aclrtMallocHost(reinterpret_cast<void**>(&output_host), outputShapeSize);
    aclrtMemcpy(output_host, outputShapeSize, outputDeviceAddr, outputShapeSize, ACL_MEMCPY_DEVICE_TO_HOST);

    int64_t totalErrors = 0;
    int64_t maxPrintErrors = 20;
    uint32_t myRankId = args.rankId;

    for (uint32_t remoteRank = 0; remoteRank < rankDim; ++remoteRank) {
        // 本卡应收到 remoteRank 的输入中第 myRankId 块的数据
        for (int64_t k = 0; k < sendCount; ++k) {
            // remoteRank 的输入中索引 myRankId*sendCount+k 处的值
            int64_t srcIdx = myRankId * sendCount + k;
            int8_t expected = static_cast<int8_t>((remoteRank + 1) * 10 + (srcIdx % 100));
            int8_t actual = output_host[remoteRank * sendCount + k];
            if (actual != expected) {
                if (totalErrors < maxPrintErrors) {
                    LOG_PRINT("[ERROR] rank=%d output[remoteRank=%d][%lld] expected=%d actual=%d "
                              "(srcIdx=%lld)\n",
                              args.rankId, remoteRank, (long long)k, (int)expected, (int)actual,
                              (long long)srcIdx);
                }
                ++totalErrors;
            }
        }
    }

    if (totalErrors > 0) {
        LOG_PRINT("[RESULT] rank=%d FAIL, total errors: %lld / %lld\n",
                  args.rankId, (long long)totalErrors, (long long)totalElements);
    } else {
        LOG_PRINT("[RESULT] rank=%d PASS, all %lld elements correct\n",
                  args.rankId, (long long)totalElements);
    }

    aclrtFreeHost(output_host);
    FreeDeviceMemory(args, xDeviceAddr, workspaceAddr, tilingAddr, mc2ContextAddr,
                     outputDeviceAddr, workSpaceSize);
    return totalErrors > 0 ? 1 : 0;
}

// ============================================================================
// 主函数 - 多卡多线程执行
// ============================================================================
int main(int argc, char *argv[])
{
    // 1. ACL 初始化
    aclInit(nullptr);

    // 2. 为每个 rank 创建 Context 和 Stream
    aclrtStream stream[DEV_NUM];
    aclrtContext context[DEV_NUM];
    for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
        aclrtSetDevice(rankId);
        aclrtCreateContext(&context[rankId], rankId);
        aclrtCreateStream(&stream[rankId]);
    }

    // 3. 初始化 HCCL 通信域
    int32_t devices[DEV_NUM];
    for (int i = 0; i < DEV_NUM; i++) {
        devices[i] = i;
    }
    HcclComm comms[DEV_NUM];
    HcclCommInitAll(DEV_NUM, devices, comms);

    // 4. 为每个 rank 启动线程并行执行
    Args args[DEV_NUM];
    std::vector<std::unique_ptr<std::thread>> threads(DEV_NUM);
    int returnCodes[DEV_NUM] = {0};
    for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
        args[rankId].rankId = rankId;
        args[rankId].rankDim = DEV_NUM;
        args[rankId].hcclComm = comms[rankId];
        args[rankId].context = context[rankId];
        args[rankId].stream = stream[rankId];
        threads[rankId].reset(new(std::nothrow) std::thread([&args, &returnCodes, rankId]() {
            returnCodes[rankId] = LaunchOneThread(args[rankId]);
        }));
    }

    // 5. 等待所有线程完成
    for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
        threads[rankId]->join();
    }
    aclFinalize();

    // 6. 汇总结果
    int finalRet = 0;
    for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
        if (returnCodes[rankId] != 0) {
            finalRet = 1;
        }
    }
    if (finalRet == 0) {
        printf("========== All ranks PASSED ==========\n");
    } else {
        printf("========== Some ranks FAILED ==========\n");
    }
    return finalRet;
}
