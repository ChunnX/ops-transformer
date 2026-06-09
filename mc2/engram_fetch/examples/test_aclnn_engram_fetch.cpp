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
 * \file test_aclnn_engram_fetch.cpp
 * \brief engram_fetch 算子测试用例
 */

#include <iostream>
#include <vector>
#include <thread>
#include <memory>
#include <mutex>
#include <condition_variable>
#include "hccl/hccl.h"
#include "aclnnop/aclnn_engram_fetch.h"
#include <algorithm>
#include <cstring>

#include "acl/acl.h"
#include "hccl/hccl.h"
#include "hccl/hccl_comm.h"
#include "hccl/hccl_res.h"
#include "hccl/hccl_rank_graph.h"
#include "hccl/hcomm_primitives.h"
#include "hccl/hcomm_res_defs.h"
#include "hccl/hccl_types.h"
typedef uint64_t ChannelPtr;
static constexpr int COMM_PROTOCOL_UBC_CTP_VALUE = 4; // AIV URMA, local UB fabric (CTP)
static constexpr int COMM_PROTOCOL_UBC_TP_VALUE = 5;  // AIV URMA, local UB fabric (TP)
struct EngramBufferHandle {
    int64_t rankId;
    int64_t rankSize;
    uint64_t hcommHandle_[8];
    uint64_t commBuffer[8];
};

static HcclResult build_channel_descs(HcclComm comm, uint32_t self, uint32_t world, HcclMemHandle *myMemHandle,
                                      std::vector<HcclChannelDesc> &out)
{
    out.clear();
    out.reserve(world > 0 ? world - 1 : 0);

    uint32_t *netLayers = nullptr;
    uint32_t netLayerNum = 0;
    HcclResult r = HcclRankGraphGetLayers(comm, &netLayers, &netLayerNum);
    if (r != HCCL_SUCCESS)
        return r;

    for (uint32_t peer = 0; peer < world; ++peer) {
        if (peer == self)
            continue;
        bool found = false;
        for (uint32_t li = 0; li < netLayerNum && !found; ++li) {
            CommLink *linkList = nullptr;
            uint32_t listSize = 0;
            r = HcclRankGraphGetLinks(comm, netLayers[li], self, peer, &linkList, &listSize);
            if (r != HCCL_SUCCESS)
                continue;
            for (uint32_t i = 0; i < listSize && !found; ++i) {
                const int p = static_cast<int>(linkList[i].linkAttr.linkProtocol);
                if (p != COMM_PROTOCOL_UBC_CTP_VALUE && p != COMM_PROTOCOL_UBC_TP_VALUE)
                    continue;
                HcclChannelDesc desc;
                HcclChannelDescInit(&desc, 1);
                desc.remoteRank = peer;
                desc.channelProtocol = linkList[i].linkAttr.linkProtocol;
                desc.localEndpoint.protocol = linkList[i].srcEndpointDesc.protocol;
                desc.localEndpoint.commAddr = linkList[i].srcEndpointDesc.commAddr;
                desc.localEndpoint.loc = linkList[i].srcEndpointDesc.loc;
                desc.remoteEndpoint.protocol = linkList[i].dstEndpointDesc.protocol;
                desc.remoteEndpoint.commAddr = linkList[i].dstEndpointDesc.commAddr;
                desc.remoteEndpoint.loc = linkList[i].dstEndpointDesc.loc;
                desc.notifyNum = 3;
                // The crucial difference vs `comm_with_urma`'s hccl path:
                // attach MY registered workspace handle so peers' URMA writes
                // land in MY 500 MiB region (and vice versa, after channel
                // memory exchange via HcclChannelAcquire).
                desc.memHandles = myMemHandle;
                desc.memHandleNum = 1;
                out.push_back(desc);
                found = true;
            }
        }
        if (!found)
            return HCCL_E_NOT_FOUND;
    }
    return HCCL_SUCCESS;
}

EngramBufferHandle* get_or_create_ctx(const std::string &groupName, int64_t rank, int64_t worldSize,
                                             uint64_t bufBytes)
{
    const std::string tag = groupName + ".comm_with_urma_reg";
    const uint32_t selfRank = static_cast<uint32_t>(rank);
    const uint32_t worldSizeU = static_cast<uint32_t>(worldSize);

    // 1. HcclComm handle.
    HcclComm comm = nullptr;
    HcclResult r = HcclCommGetHandleWithName(groupName.c_str(), &comm);
    if (r != HCCL_SUCCESS) {
        std::cerr << "[ERROR] HcclCommGetHandleWithName failed. ret: " << r << std::endl;
        return nullptr;
    }

    // 2. Allocate + register OUR URMA-target workspace.

    void *workspaceDev = nullptr;
    aclError ar = aclrtMalloc(&workspaceDev, bufBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ar != ACL_SUCCESS) {
        std::cerr << "[ERROR] aclrtMalloc failed. ret: " << ar << std::endl;
        return nullptr;
    }
    ar = aclrtMemset(workspaceDev, bufBytes, 1, bufBytes);
    if (ar != ACL_SUCCESS) {
        std::cerr << "[ERROR] aclrtMemset failed. ret: " << ar << std::endl;
        aclrtFree(workspaceDev);
        return nullptr;
    }

    CommMem mem;
    mem.type = COMM_MEM_TYPE_DEVICE;
    mem.addr = workspaceDev;
    mem.size = bufBytes;
    HcclMemHandle workspaceHandle = nullptr;
    r = HcclCommMemReg(comm, tag.c_str(), &mem, &workspaceHandle);
    if (r != HCCL_SUCCESS) {
        std::cerr << "[ERROR] HcclCommMemReg failed. ret: " << r << std::endl;
        aclrtFree(workspaceDev);
        return nullptr;
    }

    // 3. Per-peer ChannelDescs (UBC_CTP/UBC_TP, memHandles=&workspaceHandle).
    std::vector<HcclChannelDesc> descs;
    r = build_channel_descs(comm, selfRank, worldSizeU, &workspaceHandle, descs);
    if (r != HCCL_SUCCESS) {
        std::cerr << "[ERROR] build_channel_descs failed. ret: " << r << std::endl;
        // HcclCommMemDereg(comm, workspaceHandle);
        aclrtFree(workspaceDev);
        return nullptr;
    }

    std::vector<ChannelHandle> peerHandles(descs.size(), 0);
    r = HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_AIV, descs.data(), static_cast<uint32_t>(descs.size()),
                           peerHandles.data());
    if (r != HCCL_SUCCESS) {
        std::cerr << "[ERROR] HcclChannelAcquire failed. Engine: AIV. ret: " << r << std::endl;
        // HcclCommMemDereg(comm, workspaceHandle);
        aclrtFree(workspaceDev);
        return nullptr;
    }

    // 4. ChannelPtr lookup (PR 2032). Index by peer rank, with self slot = 0.
    std::vector<ChannelHandle> handlesByRank(worldSizeU, 0);
    for (size_t i = 0; i < descs.size(); ++i) {
        handlesByRank[descs[i].remoteRank] = peerHandles[i];
    }
    std::vector<ChannelHandle> nonSelfHandles;
    std::vector<uint32_t> nonSelfRanks;
    nonSelfHandles.reserve(worldSizeU - 1);
    nonSelfRanks.reserve(worldSizeU - 1);
    for (uint32_t p = 0; p < worldSizeU; ++p) {
        if (p == selfRank)
            continue;
        nonSelfHandles.push_back(handlesByRank[p]);
        nonSelfRanks.push_back(p);
    }
    EngramBufferHandle* ctx = new EngramBufferHandle();
   
    // for (uint32_t i = 0; i < worldSize; ++i) {
    //     if (i == rank) {
    //         ctx->commBuffer[i] = reinterpret_cast<uint64_t>(workspaceDev);
    //         continue;
    //     }
    //     void *peerBuf = nullptr;
    //     uint64_t peerBufSize = 0;
    //     auto hcclRet = HcclChannelGetHcclBuffer(comm, handlesByRank[i], &peerBuf, &peerBufSize);
    //     ctx->commBuffer[i] = reinterpret_cast<uint64_t>(peerBuf);
    // }
    // ctx->rankId = selfRank;
    // ctx->rankSize = worldSize;
    // std::memcpy(ctx->hcommHandle_,                            // 目标地址
    //             nonSelfHandles.data(),                        // 源地址：获取 vector 内部数组的指针
    //             nonSelfHandles.size() * sizeof(ChannelHandle) // 拷贝大小：元素个数 * 单个元素大小
    // );
    for (uint32_t i = 0; i < worldSize; ++i) {
        if (i == rank) {
            ctx->commBuffer[i] = reinterpret_cast<uint64_t>(workspaceDev);
            std::cout << "[DEBUG] [Rank " << rank << "] ctx->commBuffer[" << i << "] = " << ctx->commBuffer[i]
                      << " (Self workspaceDev)" << std::endl;
            continue;
        }
        uint32_t memNum = 0;
        CommMem *remoteMems = nullptr;
        char **memTags = nullptr;
        auto hcclRet = HcclChannelGetRemoteMems(comm, handlesByRank[i], &memNum, &remoteMems, &memTags);

        if (hcclRet != HCCL_SUCCESS) {
            std::cerr << "[ERROR] [Rank " << rank << "] HcclChannelGetRemoteMems failed for rank " << i
                      << ", ret = " << hcclRet << std::endl;
        }
        ctx->commBuffer[i] = reinterpret_cast<uint64_t>(remoteMems->addr);
        std::cout << "[DEBUG] [Rank " << rank << "] ctx->commBuffer[" << i << "] = " << ctx->commBuffer[i]
                  << " (Peer buffer, size: " << remoteMems->size << ")" << std::endl;
    }
    
    ctx->rankId = selfRank;
    std::cout << "[DEBUG] [Rank " << rank << "] ctx->rankId = " << ctx->rankId << std::endl;
   
    ctx->rankSize = worldSize;
     std::cout << "[DEBUG] [Rank " << rank << "] ctx->rankSize = " << ctx->rankSize << std::endl;
    std::cout << "[DEBUG] [Rank " << rank << "] Start copying hcommHandle_..." << std::endl;

    for (size_t i = 0; i < nonSelfHandles.size(); ++i) {
        ctx->hcommHandle_[i] = nonSelfHandles[i];
        std::cout << "[DEBUG] [Rank " << rank << "] ctx->hcommHandle_[" << i << "] = " << nonSelfHandles[i]
                  << std::endl;
    }
    return ctx;
}

#define CHECK_RET(cond, return_expr)                                                                                   \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            return_expr;                                                                                               \
        }                                                                                                              \
    } while (0)

#define LOG_PRINT(message, ...)                                                                                        \
    do {                                                                                                               \
        printf(message, ##__VA_ARGS__);                                                                                \
    } while (0)


constexpr int DEV_NUM = 2;
static std::mutex barrierMtx;
static std::condition_variable barrierCv;
static int barrierCount = 0;

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor, aclFormat format = aclFormat::ACL_FORMAT_ND)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc failed. ret: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMemcpy failed. ret: %d\n", ret); return ret);

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0, format, shape.data(), shape.size(),
                              *deviceAddr);
    return 0;
}

struct Args {
    int rankId;
    int rankSize;
    HcclComm hcclComm;
    aclrtStream stream;
    aclrtContext context;
};

int LaunchOneThreadEngramFetch(Args &args)
{
    int ret = aclrtSetCurrentContext(args.context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetCurrentContext failed. ret = %d\n", ret); return ret);

    char hcomName[128] = {0};
    ret = HcclGetCommName(args.hcclComm, hcomName);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetCommName failed. ret = %d\n", ret); return -1);
    LOG_PRINT("[INFO] rank = %d, hcomName = %s, stream = %p\n", args.rankId, hcomName, args.stream);

    // Test parameters
    int32_t hiddenSize = 512;
    int64_t numEntriesPerRank = 4;
    int64_t numTokens = 2;
    int64_t outputDtype = 1; // BF16

    // Shapes
    // bufferHandle: int32 ND, buffer descriptor (flat int32 array)
    std::vector<int64_t> bufferHandleShape = {numEntriesPerRank * hiddenSize};
    // indices: int32 ND, 1D, contains token ids to fetch
    std::vector<int64_t> indicesShape = {numTokens};
    // fetched: BF16 ND, 2D [numTokens, hiddenSize]
    std::vector<int64_t> fetchedShape = {numTokens, hiddenSize};

    // Host data
    long long bufferHandleSize = GetShapeSize(bufferHandleShape);
    long long indicesSize = GetShapeSize(indicesShape);
    long long fetchedSize = GetShapeSize(fetchedShape);

    std::vector<int32_t> bufferHandleHostData(bufferHandleSize, 0);
    // Fill indices with sequential token ids: 0, 1, 2, ..., numTokens-1
    std::vector<int32_t> indicesHostData(indicesSize);
    for (int64_t i = 0; i < indicesSize; i++) {
        // indicesHostData[i] = static_cast<int32_t>(i);
        indicesHostData[i] = 1;
    }
    std::vector<uint16_t> fetchedHostData(fetchedSize, 0); // BF16 output, init to 0

    // Device memory and tensors
    void *bufferHandleDeviceAddr = nullptr;
    void *indicesDeviceAddr = nullptr;
    void *fetchedDeviceAddr = nullptr;
    aclTensor *bufferHandle = nullptr;
    aclTensor *indices = nullptr;
    aclTensor *fetched = nullptr;

    // Memcpy Mc2Context

    EngramBufferHandle * ctx = get_or_create_ctx(hcomName, args.rankId, args.rankSize, 200 * 1024 * 1024);

     if (ctx != nullptr) {
        std::cout << "rankId_: " << ctx->rankId << std::endl;
        std::cout << "rankSize_: " << ctx->rankSize << std::endl;
        
        // 打印 1 个 handle
        std::cout << "hcommHandle_[0]: " << ctx->hcommHandle_[0] << std::endl;
        
        // 打印 2 个 buffer
        std::cout << "commBuffer_[0]: " << ctx->commBuffer[0] << std::endl;
        std::cout << "commBuffer_[1]: " << ctx->commBuffer[1] << std::endl;
    }
     std::cout << "args.rankId:  "<<args.rankId<< std::endl;
    ret = aclrtMalloc(&bufferHandleDeviceAddr, sizeof(EngramBufferHandle), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc Mc2Context failed. ret: %d\n", ret); return ret);
    ret = aclrtMemcpy(bufferHandleDeviceAddr, sizeof(EngramBufferHandle), 
                         ctx, sizeof(EngramBufferHandle), ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMemcpy Mc2Context failed. ret: %d\n", ret); return ret);  
    auto size = GetShapeSize(bufferHandleShape) * sizeof(int32_t);
    bufferHandle = aclCreateTensor(bufferHandleShape.data(), bufferHandleShape.size(), aclDataType::ACL_INT32, nullptr, 0, aclFormat::ACL_FORMAT_ND, bufferHandleShape.data(), bufferHandleShape.size(),
                              bufferHandleDeviceAddr);

    // Create input tensors
    // ret = CreateAclTensor(bufferHandleHostData, bufferHandleShape, &bufferHandleDeviceAddr, aclDataType::ACL_INT32,
    //                       &bufferHandle);
    // CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(indicesHostData, indicesShape, &indicesDeviceAddr, aclDataType::ACL_INT32, &indices);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Create output tensor (BF16)
    ret = CreateAclTensor(fetchedHostData, fetchedShape, &fetchedDeviceAddr, aclDataType::ACL_BF16, &fetched);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    void *workspaceAddr = nullptr;


     // Copy output back
    std::vector<uint16_t> fetchedResult1(fetchedSize);
    ret = aclrtMemcpy(fetchedResult1.data(), fetchedSize * sizeof(uint16_t), fetchedDeviceAddr,
                      fetchedSize * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMemcpy fetched failed. ret = %d\n", ret); return ret);

    // Print first few output values
    LOG_PRINT("[INFO] before  rank=%d, fetched first 10 values (bf16 hex):\n", args.rankId);
    for (int64_t i = 0; i < 10 && i < fetchedSize; i++) {
        LOG_PRINT("  [%lld] 0x%04x\n", static_cast<long long>(i), fetchedResult1[i]);
    }

    LOG_PRINT("[INFO] rank = %d, Before aclnnEngramFetchGetWorkspaceSize\n", args.rankId);

    // ----------------------------------------
    // Phase 1: Get workspace size
    ret = aclnnEngramFetchGetWorkspaceSize(bufferHandle, indices, hiddenSize, numEntriesPerRank, outputDtype,
    fetched,
                                           &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnEngramFetchGetWorkspaceSize failed. ret = %d\n", ret);
              return ret);

    LOG_PRINT("[INFO] rank = %d, workspaceSize = %lu\n", args.rankId, workspaceSize);

    // Allocate workspace
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d\n", ret); return
        ret);
    }

    LOG_PRINT("[INFO] rank = %d, Before aclnnEngramFetch\n", args.rankId);

    // Phase 2: Execute
    ret = aclnnEngramFetch(workspaceAddr, workspaceSize, executor, args.stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnEngramFetch failed. ret = %d\n", ret); return ret);

    // Sync
    ret = aclrtSynchronizeStreamWithTimeout(args.stream, 30000);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d\n", ret);
              return ret);

    LOG_PRINT("[INFO] rank = %d, aclnnEngramFetch execute successfully.\n", args.rankId);
    // ----------------------------------------


    // Copy output back
    std::vector<uint16_t> fetchedResult(fetchedSize);
    ret = aclrtMemcpy(fetchedResult.data(), fetchedSize * sizeof(uint16_t), fetchedDeviceAddr,
                      fetchedSize * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMemcpy fetched failed. ret = %d\n", ret); return ret);

    // Print first few output values
    LOG_PRINT("[INFO] rank=%d, fetched first 10 values (bf16 hex):\n", args.rankId);
    for (int64_t i = 0; i < 10 && i < fetchedSize; i++) {
        LOG_PRINT("  [%lld] 0x%04x\n", static_cast<long long>(i), fetchedResult[i]);
    }

    // Cleanup
    if (bufferHandle != nullptr) {
        aclDestroyTensor(bufferHandle);
        aclrtFree(bufferHandleDeviceAddr);
    }
    if (indices != nullptr) {
        aclDestroyTensor(indices);
        aclrtFree(indicesDeviceAddr);
    }
    if (fetched != nullptr) {
        aclDestroyTensor(fetched);
        aclrtFree(fetchedDeviceAddr);
    }
    if (workspaceSize > 0 && workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }

    // Barrier: wait for all ranks before cleanup
    {
        std::unique_lock<std::mutex> lock(barrierMtx);
        barrierCount++;
        if (barrierCount >= DEV_NUM) {
            barrierCv.notify_all();
        } else {
            barrierCv.wait(lock, [&]() { return barrierCount >= DEV_NUM; });
        }
    }

    ret = HcclCommDestroy(args.hcclComm);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclCommDestroy failed. ret = %d\n", ret); return ret);
    ret = aclrtDestroyStream(args.stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtDestroyStream failed. ret = %d\n", ret); return ret);
    ret = aclrtResetDevice(args.rankId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtResetDevice failed. ret = %d\n", ret); return ret);
    ret = aclrtDestroyContext(args.context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtDestroyContext failed. ret = %d\n", ret); return ret);

    return 0;
}

int main(int argc, char *argv[])
{
    int ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclInit failed. ret = %d\n", ret); return ret);
    printf("start host\n");
    aclrtStream stream[DEV_NUM];
    aclrtContext context[DEV_NUM];
    for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
        ret = aclrtSetDevice(rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice failed. ret = %d\n", ret); return ret);
        ret = aclrtCreateContext(&context[rankId], rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed. ERROR: %d\n", ret); return ret);
        ret = aclrtCreateStream(&stream[rankId]);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed. ret = %d\n", ret); return ret);
    }

    int32_t devices[DEV_NUM];
    for (int i = 0; i < DEV_NUM; i++) {
        devices[i] = i;
    }

    // Initialize HCCL communication domain
    HcclComm comms[DEV_NUM];
    ret = HcclCommInitAll(DEV_NUM, devices, comms);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclCommInitAll failed. ret = %d\n", ret); return ret);

    Args args[DEV_NUM];
    // Launch threads
    std::vector<std::unique_ptr<std::thread>> threads(DEV_NUM);
    for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
        args[rankId].rankId = rankId;
        args[rankId].rankSize = DEV_NUM;
        args[rankId].hcclComm = comms[rankId];
        args[rankId].context = context[rankId];
        args[rankId].stream = stream[rankId];
        threads[rankId].reset(new (std::nothrow) std::thread(&LaunchOneThreadEngramFetch, std::ref(args[rankId])));
    }

    for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
        threads[rankId]->join();
    }

    aclFinalize();
    return 0;
}