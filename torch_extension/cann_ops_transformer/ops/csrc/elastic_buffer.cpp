/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file elastic_buffer.cpp
 * \brief
 */

#include <torch/extension.h>
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <chrono>
#include <functional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
#include <cstring>
#include <atomic>
#include <cstdint>
#include <algorithm>

// CANN ACL Runtime API
#include "acl/acl.h"

// HCCL types
#include "hccl/hccl_types.h"

// HCCL common utilities
#include "comm_context.h"
#include "hccl_common.h"

// ACLNN common utilities
#include "aclnn_common.h"

// torch_npu stream utilities
#include "torch_npu/csrc/core/npu/NPUStream.h"

namespace Mc2Api {

// Constants
constexpr static uint8_t COMM_ENGINE_AIV = 4;
constexpr uint32_t HCCL_MAX_RANK_SIZE = 1024;
constexpr uint32_t HCCL_MIN_RANK_SIZE = 2;
constexpr int COMM_PROTOCOL_UBC_CTP_VALUE = 4;
constexpr int COMM_PROTOCOL_UBC_TP_VALUE = 5;
constexpr int64_t BUFFER_ALIGNMENT = 2 * 1024 * 1024;
constexpr int DIM_TWO = 2;

// RAII guard for multi-step host buffer allocation
struct HostBufferGuard {
    void *hostPtr = nullptr;
    bool registered = false;

    ~HostBufferGuard()
    {
        if (registered && hostPtr) {
            aclrtHostUnregister(hostPtr);
        }
        if (hostPtr) {
            aclrtFreeHost(hostPtr);
        }
    }

    void Release()
    {
        hostPtr = nullptr;
        registered = false;
    }
};

// Helper functions
static inline int64_t CeilDiv(int64_t x, int64_t y)
{
    TORCH_CHECK(y > 0, "CeilDiv divisor must be positive, got ", y);
    TORCH_CHECK(x <= INT64_MAX - y + 1, "CeilDiv overflow: x=", x, " y=", y);
    return (x + y - 1) / y;
}

static inline int64_t AlignTo(int64_t x, int64_t y)
{
    TORCH_CHECK(y > 0, "AlignTo divisor must be positive, got ", y);
    TORCH_CHECK(x <= INT64_MAX - y + 1, "AlignTo overflow: x=", x, " y=", y);
    return CeilDiv(x, y) * y;
}

// CommContext structure for HCCL communication
// 推理/训练共用：virtualAddrList + hcommHandle 由 EngramContextBuilder 一次性填充
// 训练专用：gmAddrList 由 AllocateAndExchangeGm 每次 fetch 动态填充
struct CommContext {
    uint32_t epRankId = 0;
    uint32_t rankSize = 0;
    uint64_t virtualAddrList[HCCL_MAX_RANK_SIZE] = {};   // → kernel commBuffer: 各 rank host pinned 设备映射地址
    uint64_t hcommHandle[HCCL_MAX_RANK_SIZE] = {};       // → kernel hcommHandle_: 各 rank UBC channel 句柄
    uint64_t gmAddrList[HCCL_MAX_RANK_SIZE] = {};        // → kernel gmCommBuffer: 各 rank GM 缓冲区地址（per-fetch 动态）
};

struct EngramContextResources {
    HcclComm hcclComm = nullptr;
    HcclMemHandle memHandle = nullptr;
    void *hostBufPtr = nullptr;
    void *deviceBufPtr = nullptr;
    CommContext context;
    at::Tensor contextTensor;
};

static at::Tensor CreateCommContextTensor(const CommContext &context)
{
    int64_t numElements = (sizeof(CommContext) + sizeof(int32_t) - 1) / sizeof(int32_t);
    at::Tensor tensor = at::empty({numElements}, at::TensorOptions()
                                                     .dtype(at::kInt)
                                                     .device(c10::DeviceType::PrivateUse1)
                                                     .memory_format(c10::MemoryFormat::Contiguous));
    at::Tensor hostTensor = at::empty({numElements}, at::TensorOptions().dtype(at::kInt));
    errno_t memRet = memcpy_s(hostTensor.data_ptr<int32_t>(), sizeof(CommContext), &context, sizeof(CommContext));
    TORCH_CHECK(memRet == EOK, "memcpy_s failed, ret=", memRet);
    tensor.copy_(hostTensor);
    return tensor;
}

// EngramContextBuilder: 构建 Engram 通信上下文（PR 8541 重构，从 ElasticBuffer 内联方法提取为独立类）
// 职责：HCCL handle 获取 → host pinned 内存分配注册 → UBC channel 建立 → 远端地址交换 → context tensor 创建
// withGrad 参数决定 channel memHandle 槽位数（训练=2, 推理=1）
class EngramContextBuilder {
public:
    EngramContextResources Build(const std::string &groupName, int64_t numCpuBytes, bool withGrad = false)
    {
        EngramContextResources resources;
        AcquireHcclHandle(groupName, resources.hcclComm);

        std::string contextTag = groupName + "engram_embedding";
        TORCH_CHECK(contextTag.size() <= 255, "Mc2ContextTag is too long, max size is 255, got ",
                    contextTag.size());

        HostBufferGuard guard;
        CreateContext(resources, contextTag, numCpuBytes, withGrad, guard);
        resources.contextTensor = CreateCommContextTensor(resources.context);
        guard.Release();
        return resources;
    }

private:
    static void AcquireHcclHandle(const std::string &groupName, HcclComm &hcclComm)
    {
        auto hcclRet = HcomGetCommHandleByGroupFunc(groupName.c_str(), &hcclComm);
        TORCH_CHECK(hcclRet == HCCL_SUCCESS, "Get HCCL handle failed, group: ", groupName.c_str(), ", ret: ",
                    hcclRet);
    }

    static void AllocateAndRegisterBuffer(const HcclComm &commHandle, const std::string &memBufferTag,
                                          int64_t numCpuBytes, EngramContextResources &resources,
                                          HostBufferGuard &guard)
    {
        aclError ar = aclrtMallocHost(&guard.hostPtr, static_cast<uint64_t>(numCpuBytes));
        TORCH_CHECK(ar == ACL_SUCCESS, "aclrtMallocHost(", numCpuBytes, " B) failed, ret=", ar);

        ar = aclrtHostRegisterV2(guard.hostPtr, static_cast<uint64_t>(numCpuBytes), ACL_HOST_REG_MAPPED);
        TORCH_CHECK(ar == ACL_SUCCESS, "aclrtHostRegisterV2(", numCpuBytes, " B) failed, ret=", ar);
        guard.registered = true;

        void *devPtr = nullptr;
        ar = aclrtHostGetDevicePointer(guard.hostPtr, &devPtr, 0);
        TORCH_CHECK(ar == ACL_SUCCESS, "aclrtHostGetDevicePointer failed, ret=", ar);

        CommMem mem;
        mem.type = COMM_MEM_TYPE_DEVICE;
        mem.addr = devPtr;
        mem.size = static_cast<uint64_t>(numCpuBytes);

        auto hcclRet = HcclCommMemRegFunc(commHandle, memBufferTag.c_str(), &mem, &resources.memHandle);
        TORCH_CHECK(hcclRet == HCCL_SUCCESS, "HcclCommMemReg(tag='", memBufferTag, "', size=", numCpuBytes,
                    ") failed, ret=", hcclRet);

        resources.hostBufPtr = guard.hostPtr;
        resources.deviceBufPtr = devPtr;
    }

    static void BuildChannelDescs(const HcclComm &commHandle, uint32_t srcRankId, uint32_t rankDim,
                                  bool withGrad,
                                  HcclMemHandle &memHandle, std::vector<HcclChannelDesc> &channelDesc)
    {
        channelDesc.clear();
        channelDesc.reserve(rankDim > 0 ? rankDim - 1 : 0);

        uint32_t *netLayers = nullptr;
        uint32_t netLayerNum = 0;
        HcclResult r = HcclRankGraphGetLayersFunc(commHandle, &netLayers, &netLayerNum);
        TORCH_CHECK(r == HCCL_SUCCESS, "Get HCCL layers failed, ret: ", r);

        for (uint32_t peer = 0; peer < rankDim; ++peer) {
            if (peer == srcRankId)
                continue;
            bool found = false;
            for (uint32_t li = 0; li < netLayerNum && !found; ++li) {
                CommLink *linkList = nullptr;
                uint32_t listSize = 0;
                r = HcclRankGraphGetLinksFunc(commHandle, netLayers[li], srcRankId, peer, &linkList, &listSize);
                if (r != HCCL_SUCCESS)
                    continue;
                for (uint32_t i = 0; i < listSize && !found; ++i) {
                    const int p = static_cast<int>(linkList[i].linkAttr.linkProtocol);
                    if (p != COMM_PROTOCOL_UBC_CTP_VALUE && p != COMM_PROTOCOL_UBC_TP_VALUE)
                        continue;
                    HcclChannelDesc desc;
                    // 训练态预留 2 槽位（host pinned + GM），推理态 1 槽位（host pinned only）
                    HcclResult initRet = HcclChannelDescInit(&desc, withGrad ? 2 : 1);
                    TORCH_CHECK(initRet == HCCL_SUCCESS, "HcclChannelDescInit failed, ret=", initRet);
                    desc.remoteRank = peer;
                    desc.channelProtocol = linkList[i].linkAttr.linkProtocol;
                    desc.localEndpoint.protocol = linkList[i].srcEndpointDesc.protocol;
                    desc.localEndpoint.commAddr = linkList[i].srcEndpointDesc.commAddr;
                    desc.localEndpoint.loc = linkList[i].srcEndpointDesc.loc;
                    desc.remoteEndpoint.protocol = linkList[i].dstEndpointDesc.protocol;
                    desc.remoteEndpoint.commAddr = linkList[i].dstEndpointDesc.commAddr;
                    desc.remoteEndpoint.loc = linkList[i].dstEndpointDesc.loc;
                    desc.notifyNum = 3;
                    desc.memHandles = &memHandle;
                    desc.memHandleNum = 1;
                    channelDesc.push_back(desc);
                    found = true;
                }
            }
            TORCH_CHECK(found, "No UBC_CTP/UBC_TP link found for srcRankID ", srcRankId, ", dstRankID ", peer);
        }
    }

    static void GetHcclCommChannel(const HcclComm &commHandle, uint32_t rankDim, uint32_t srcRankId,
                                   bool withGrad,
                                   HcclMemHandle &memHandle, ChannelHandle *channels)
    {
        std::vector<HcclChannelDesc> descs;
        ChannelHandle channelBuf[HCCL_MAX_RANK_SIZE] = {};
        BuildChannelDescs(commHandle, srcRankId, rankDim, withGrad, memHandle, descs);
        auto hcclRet = HcclChannelAcquireFunc(commHandle, CommEngine::COMM_ENGINE_AIV, descs.data(),
                                              static_cast<uint32_t>(descs.size()), channelBuf);
        TORCH_CHECK(hcclRet == HCCL_SUCCESS,
                    "HcclChannelAcquire(AIV, UBC_CTP/UBC_TP, memHandle_s=1) failed, ret=", hcclRet);
        for (size_t i = 0; i < descs.size(); ++i) {
            channels[descs[i].remoteRank] = channelBuf[i];
        }
    }

    static void GetHcclCommResource(const HcclComm &commHandle, EngramContextResources &resources,
                                    bool withGrad, const std::string &targetTag)
    {
        uint32_t rankId = resources.context.epRankId;
        ChannelHandle handlesByRank[HCCL_MAX_RANK_SIZE] = {};
        GetHcclCommChannel(commHandle, resources.context.rankSize, rankId, withGrad,
                           resources.memHandle, handlesByRank);

        for (uint32_t peer = 0; peer < resources.context.rankSize; ++peer) {
            if (peer == rankId)
                continue;
            resources.context.hcommHandle[peer] = handlesByRank[peer];
        }

        resources.context.virtualAddrList[rankId] = reinterpret_cast<uint64_t>(resources.deviceBufPtr);

        for (uint32_t i = 0; i < resources.context.rankSize; ++i) {
            if (i == rankId)
                continue;
            uint32_t memNum = 0;
            CommMem *remoteMems = nullptr;
            char **memTags = nullptr;
            auto hcclRet =
                HcclChannelGetRemoteMemsFunc(commHandle, resources.context.hcommHandle[i], &memNum, &remoteMems,
                                             &memTags);
            TORCH_CHECK(hcclRet == HCCL_SUCCESS, "HcclChannelGetRemoteMems(peer=", i, ") failed, ret=", hcclRet);
            // 取自己注册的buffer作为通信buffer
            bool hasTargetMem = false;
            for (uint32_t j = 0; j < memNum; j++) {
                if (memTags == nullptr || remoteMems == nullptr) {
                    break;
                }
                if (memTags[j] != nullptr && targetTag == memTags[j]) {
                    uint64_t targetMemAddr = reinterpret_cast<uint64_t>(remoteMems[j].addr);
                    resources.context.virtualAddrList[i] = targetMemAddr;
                    ASCEND_LOGI("Get Target Mem(%s) Success, Mem id is %d, Addr is %lu", targetTag.c_str(), j,
                                targetMemAddr);
                    hasTargetMem = true;
                    break;
                }
            }
            TORCH_CHECK(hasTargetMem, "Target Mem : ", targetTag, " is not found.");
        }
    }

    static void CreateContext(EngramContextResources &resources, const std::string &contextTag, int64_t numCpuBytes,
                              bool withGrad, HostBufferGuard &guard)
    {
        uint64_t contextSize = sizeof(CommContext);
        void *ctx = nullptr;
        auto hcclRet = HcclEngineCtxCreateFunc(resources.hcclComm, contextTag.c_str(), CommEngine::COMM_ENGINE_AIV,
                                               contextSize, &ctx);
        TORCH_CHECK(hcclRet == HCCL_SUCCESS, "Create HCCL context memory failed, ret: ", hcclRet);

        hcclRet = HcclGetRankIdFunc(resources.hcclComm, &resources.context.epRankId);
        TORCH_CHECK(hcclRet == HCCL_SUCCESS, "Get rank ID failed, ret: ", hcclRet);

        hcclRet = HcclGetRankSizeFunc(resources.hcclComm, &resources.context.rankSize);
        TORCH_CHECK(hcclRet == HCCL_SUCCESS, "Get rank size failed, ret: ", hcclRet);
        TORCH_CHECK(resources.context.rankSize >= HCCL_MIN_RANK_SIZE,
                    "rankSize must be at least HCCL_MIN_RANK_SIZE, got ", resources.context.rankSize, ", min ",
                    HCCL_MIN_RANK_SIZE);
        TORCH_CHECK(resources.context.rankSize <= HCCL_MAX_RANK_SIZE, "rankSize exceeds HCCL_MAX_RANK_SIZE, got ",
                    resources.context.rankSize, ", max ", HCCL_MAX_RANK_SIZE);

        std::string memBufferTag = contextTag + "_buffer";
        AllocateAndRegisterBuffer(resources.hcclComm, memBufferTag, numCpuBytes, resources, guard);
        GetHcclCommResource(resources.hcclComm, resources, withGrad, memBufferTag);

        hcclRet = HcclEngineCtxCopyFunc(resources.hcclComm, CommEngine::COMM_ENGINE_AIV, contextTag.c_str(),
                                        &resources.context, contextSize, 0);
        TORCH_CHECK(hcclRet == HCCL_SUCCESS, "Copy context from host to device failed, ret: ", hcclRet);
    }
};

// ElasticBuffer class - unified interface for Engram storage and MoE EP kernels
// 推理(with_grad=false): RDMA read (one-sided GET), 异步 fetch + wait callable
// 训练(with_grad=true): all-to-all (桶排序→a2a→gather→a2a→还原), save-for-backward 输出 + EngramFetchGrad
class ElasticBuffer {
public:
    ElasticBuffer(const std::string &groupName, int64_t numCpuBytes, bool withGrad = false);
    ~ElasticBuffer();

    // EngramFetch: 推理/训练共用前向接口
    // 推理(with_grad=false): RDMA read, 可选输出全 nullptr, 返回 callable 内调 wait
    // 训练(with_grad=true): all-to-all, 5 个可选输出非空(save-for-backward), 接口内调 wait + GM 申请/释放
    void EngramWrite(const at::Tensor &storage);
    std::function<at::Tensor()> EngramFetch(
        const at::Tensor &indices,
        at::Tensor *permOut = nullptr,
        at::Tensor *sendCountsOut = nullptr,
        at::Tensor *recvCountsOut = nullptr,
        at::Tensor *recvLocalEntryOut = nullptr,
        at::Tensor *numRecvOut = nullptr);
    // EngramFetchGrad: 训练反向（新算子 EngramFetchGrad 的 C++ 封装）
    // 输入: gradFetched + 前向 save-for-backward (perm/sendCounts/recvCounts/recvLocalEntry)
    // 输出: (gradUnique fp32, uniqueLocalEntry int32) — 本 rank owned 分片稀疏梯度
    // 内部: AllocateAndExchangeGm → aclnnEngramFetchGrad → FreeGm → narrow 截取
    std::tuple<at::Tensor, at::Tensor> EngramFetchGrad(
        const at::Tensor &gradFetched,
        const at::Tensor &perm, const at::Tensor &sendCounts,
        const at::Tensor &recvCounts, const at::Tensor &recvLocalEntry);
    void EngramBarrier(bool withDeviceSync = false);
    void Destroy();

    int64_t GetHostBufPtr() const
    {
        return reinterpret_cast<int64_t>(hostBufPtr_);
    }

    static int64_t GetEngramStorageSizeHint(int64_t numEntries, int64_t hiddenSize,
                                            at::ScalarType dtype = at::kBFloat16);

    using DispatchTensorList = std::tuple<at::Tensor, at::Tensor, at::Tensor>;
    using DispatchEpilogueTensorList =
        std::tuple<at::Tensor, at::Tensor, c10::optional<at::Tensor>, c10::optional<at::Tensor>>;
    using CombineTensorList = std::tuple<at::Tensor, c10::optional<at::Tensor>>;

    DispatchTensorList MoeEpDispatch(const at::Tensor &x, const at::Tensor &topkIdx,
                                     const c10::optional<at::Tensor> &topkWeights,
                                     const c10::optional<at::Tensor> &scales,
                                     const c10::optional<at::Tensor> &cachedHandleDstBufferSlotIdx,
                                     int64_t epWorldSize, int64_t epRankId, int64_t numExperts,
                                     int64_t numMaxTokensPerRank, int64_t expertAlignment, bool doCpuSync,
                                     int64_t hostPinnedCounterAddr);
    DispatchEpilogueTensorList MoeEpDispatchEpilogue(
        const at::Tensor &dstBufferSlotIdx, const at::Tensor &numRecvPerRank, const at::Tensor &numRecvPerExpert,
        const c10::optional<at::Tensor> &cachedRecvSrcMetadata, int64_t epWorldSize, int64_t epRankId,
        int64_t numExperts, int64_t numMaxTokensPerRank, int64_t expertAlignment, at::Tensor &recvX,
        at::Tensor &recvSrcMetadata, const c10::optional<at::Tensor> &recvTopkWeightsOpt,
        const c10::optional<at::Tensor> &recvScalesOpt);
    CombineTensorList MoeEpCombine(const at::Tensor &x, const at::Tensor &topkIdx, const at::Tensor &recvSrcMetadata,
                                   const at::Tensor &numRecvTokensPerExpert,
                                   const c10::optional<at::Tensor> &topkWeights,
                                   const c10::optional<at::Tensor> &bias0,
                                   const c10::optional<at::Tensor> &bias1, int64_t epWorldSize, int64_t epRankId,
                                   int64_t numExperts, int64_t numMaxTokensPerRank);

private:
    void EnsureEngramContext();
    void EnsureMoeContext();

    // GM 动态管理（per-fetch 生命周期，不放入 EngramContextBuilder）
    void AllocateAndExchangeGm(int64_t gmBytes);  // 大页申请 GM → HCCL 注册 → 地址交换 → 更新 context
    void FreeGm();                                 // barrier → 释放 GM → 清空 gmAddrList
    void UpdateContextTensor();                    // 把最新 commContext_ 刷到 contextTensor_

    std::string groupName_;
    int64_t numCpuBytes_;
    bool withGrad_ = false;

    void *hostBufPtr_ = nullptr;
    void *deviceBufPtr_ = nullptr;
    HcclMemHandle memHandle_ = nullptr;
    void *gmBufPtr_ = nullptr;
    HcclMemHandle gmMemHandle_ = nullptr;
    std::string gmBufferTag_;
    HcclComm hcclComm_ = nullptr;
    CommContext commContext_;
    at::Tensor contextTensor_; // Cached Engram context tensor
    bool engramContextInitialized_ = false;

    at::Tensor moeContextTensor_;
    int64_t moeCclBufferSize_ = 0;
    bool moeContextInitialized_ = false;

    int64_t hiddenSize_ = 0;
    int64_t numEngramEntries_ = 0;
    at::ScalarType engramDtype_ = at::kBFloat16;

    bool destroyed_ = false;
    bool writeCalled_ = false;
    std::atomic<bool> fetchInProgress_{false};
};

// Constructor
ElasticBuffer::ElasticBuffer(const std::string &groupName, int64_t numCpuBytes, bool withGrad)
    : groupName_(groupName), numCpuBytes_(numCpuBytes), withGrad_(withGrad),
      destroyed_(false), writeCalled_(false)
{
    InitHcclEngineCtxFunctions();
    InitHcclFunctions();
}

// Destructor - automatic resource cleanup
ElasticBuffer::~ElasticBuffer()
{
    try {
        Destroy();
    } catch (const std::exception &e) {
        ASCEND_LOGE("ElasticBuffer destructor cleanup failed: %s", e.what());
    }
}

void ElasticBuffer::EnsureEngramContext()
{
    TORCH_CHECK(!destroyed_, "ElasticBuffer cannot be used after destroy, please create a new ElasticBuffer instance");
    if (engramContextInitialized_) {
        return;
    }
    TORCH_CHECK(numCpuBytes_ > 0, "num_cpu_bytes must be greater than 0 to use Engram operations");
    EngramContextBuilder builder;
    EngramContextResources resources = builder.Build(groupName_, numCpuBytes_, withGrad_);
    hcclComm_ = resources.hcclComm;
    memHandle_ = resources.memHandle;
    hostBufPtr_ = resources.hostBufPtr;
    deviceBufPtr_ = resources.deviceBufPtr;
    commContext_ = resources.context;
    contextTensor_ = resources.contextTensor;
    engramContextInitialized_ = true;
}

void ElasticBuffer::EnsureMoeContext()
{
    TORCH_CHECK(!destroyed_, "ElasticBuffer cannot be used after destroy, please create a new ElasticBuffer instance");
    if (moeContextInitialized_) {
        return;
    }
    int64_t contextSize = (sizeof(op_api::CommContext) + sizeof(int32_t) - 1) / sizeof(int32_t);
    moeContextTensor_ = at::empty({contextSize}, at::TensorOptions()
                                                     .dtype(at::kInt)
                                                     .device(c10::DeviceType::PrivateUse1)
                                                     .memory_format(c10::MemoryFormat::Contiguous));
    op_api::HcclChannelContextBuilder builder;
    builder.Build(groupName_, 0, moeCclBufferSize_, moeContextTensor_, "urma", "moe_dispatch_combine");
    moeContextInitialized_ = true;
}

// EngramWrite - write data with automatic barrier
void ElasticBuffer::EngramWrite(const at::Tensor &storage)
{
    TORCH_CHECK(!destroyed_, "engram_write cannot be called after destroy, "
                             "please create a new ElasticBuffer instance");
    EnsureEngramContext();

    TORCH_CHECK(storage.nbytes() <= static_cast<size_t>(numCpuBytes_), "storage size ", storage.nbytes(),
                " exceeds buffer capacity ", numCpuBytes_);

    constexpr int64_t int32Max = static_cast<int64_t>(INT32_MAX);
    TORCH_CHECK(storage.size(0) * static_cast<int64_t>(commContext_.rankSize) <= int32Max,
                "num_entries * rank_size must not exceed INT32_MAX, got num_entries=", numEngramEntries_,
                ", rank_size=", commContext_.rankSize, ", product=", numEngramEntries_ * commContext_.rankSize);

    EngramBarrier(true);

    hiddenSize_ = storage.size(1);
    numEngramEntries_ = storage.size(0);
    engramDtype_ = storage.scalar_type();

    if (numEngramEntries_ > 0) {
        constexpr size_t MEMCPY_MAX_BYTES = 0x7fffffff;
        size_t totalBytes = storage.nbytes();
        size_t remaining = totalBytes;
        uint8_t *dst = static_cast<uint8_t *>(hostBufPtr_);
        const uint8_t *src = static_cast<const uint8_t *>(storage.data_ptr());
        while (remaining > 0) {
            size_t chunkSize = std::min(remaining, MEMCPY_MAX_BYTES);
            errno_t memRet = memcpy_s(dst, chunkSize, src, chunkSize);
            TORCH_CHECK(memRet == EOK, "memcpy_s failed, ret=", memRet, ", offset=", totalBytes - remaining,
                        ", chunkSize=", chunkSize);
            dst += chunkSize;
            src += chunkSize;
            remaining -= chunkSize;
        }
    }

    EngramBarrier(true);
    writeCalled_ = true;
}

// EngramFetch - fetch data using stored metadata (inference/training shared)
// 推理: ACLNN_CMD(aclnnEngramFetch, nullptr×5) → return callable { aclnnEngramFetchWait }
// 训练: AllocateAndExchangeGm → ACLNN_CMD(aclnnEngramFetch, 5 outputs) → aclnnEngramFetchWait → FreeGm → return [fetched]()
std::function<at::Tensor()> ElasticBuffer::EngramFetch(
    const at::Tensor &indices,
    at::Tensor *permOut, at::Tensor *sendCountsOut,
    at::Tensor *recvCountsOut, at::Tensor *recvLocalEntryOut,
    at::Tensor *numRecvOut)
{
    EnsureEngramContext();
    TORCH_CHECK(!destroyed_, "engram_fetch cannot be called after destroy, please create a new ElasticBuffer instance");
    TORCH_CHECK(writeCalled_, "engram_fetch must be called after at least one engram_write");
    TORCH_CHECK(!fetchInProgress_.load(),
                "Cannot call engram_fetch while previous fetch callback is pending, "
                "please invoke the callback function returned by the previous engram_fetch first");

    int64_t numTokens = indices.size(0);
    if (numTokens == 0) {
        auto emptyTensor =
            at::empty({0, hiddenSize_}, at::TensorOptions().dtype(engramDtype_).device(indices.device()));
        return [=]() { return emptyTensor; };
    }

    fetchInProgress_.store(true);

    auto fetched =
        at::empty({numTokens, hiddenSize_}, at::TensorOptions().dtype(engramDtype_).device(indices.device()));

    if (withGrad_) {
        // ===== 训练态: all-to-all 路径 =====
        uint32_t rankSize = commContext_.rankSize;
        int64_t dtypeSize = static_cast<int64_t>(at::elementSize(engramDtype_));
        int64_t hiddenBytes = hiddenSize_ * dtypeSize;
        int64_t maxR = numTokens * static_cast<int64_t>(rankSize);  // a2a 接收上界

        // 分配 5 个 save-for-backward tensor
        at::Tensor perm = at::empty({numTokens}, indices.options());          // (T,) 桶排列序索引
        at::Tensor sendCounts = at::empty({static_cast<int64_t>(rankSize)}, indices.options());  // (W,)
        at::Tensor recvCounts = at::empty({static_cast<int64_t>(rankSize)}, indices.options());  // (W,)
        at::Tensor recvLocalEntry = at::empty({maxR}, indices.options());     // (R_max,)
        at::Tensor numRecv = at::empty({1}, indices.options());               // (1,)

        // 申请 GM（a2a 收发缓冲区）+ 交换地址 + 更新 contextTensor_
        int64_t gmBytes = AlignTo(maxR * hiddenBytes, BUFFER_ALIGNMENT);
        AllocateAndExchangeGm(gmBytes);

        // issue kernel: ① 桶排序 ② a2a(indices) ③ gather ④ a2a(embedding)
        ACLNN_CMD(aclnnEngramFetch, contextTensor_, indices,
                  hiddenSize_, numEngramEntries_, fetched,
                  perm, sendCounts, recvCounts, recvLocalEntry, numRecv);

        // wait kernel: ⑤ fetched[perm] = gathered（还原）
        ACLNN_CMD(aclnnEngramFetchWait, contextTensor_, fetched);

        // 释放 GM（barrier + aclrtFree）
        FreeGm();

        if (permOut) *permOut = perm;
        if (sendCountsOut) *sendCountsOut = sendCounts;
        if (recvCountsOut) *recvCountsOut = recvCounts;
        if (recvLocalEntryOut) *recvLocalEntryOut = recvLocalEntry;
        if (numRecvOut) *numRecvOut = numRecv;

        fetchInProgress_.store(false);
        return [fetched]() { return fetched; };
    }

    c10::optional<at::Tensor> nullOpt;
    ACLNN_CMD(aclnnEngramFetch, contextTensor_, indices,
              hiddenSize_, numEngramEntries_, fetched,
              nullOpt, nullOpt, nullOpt, nullOpt, nullOpt);

    auto capturedContext = contextTensor_;
    auto fetchFlag = &fetchInProgress_;
    return [capturedContext, fetched, fetchFlag]() {
        ACLNN_CMD(aclnnEngramFetchWait, capturedContext, fetched);
        fetchFlag->store(false);
        return fetched;
    };
}

// EngramFetchGrad - training backward: produce gradUnique (fp32), uniqueLocalEntry (sparse index)
// kernel 内部 3 步: ① grad_sorted = gradFetched[perm] ② a2a(grad_sorted) → recvGrad ③ unique + fp32 scatter-add
// GM 用于 a2a 通信（fp32 规模），per-fetch 动态申请/释放
std::tuple<at::Tensor, at::Tensor> ElasticBuffer::EngramFetchGrad(
    const at::Tensor &gradFetched,
    const at::Tensor &perm, const at::Tensor &sendCounts,
    const at::Tensor &recvCounts, const at::Tensor &recvLocalEntry)
{
    EnsureEngramContext();
    TORCH_CHECK(!destroyed_, "engram_fetch_grad cannot be called after destroy");
    TORCH_CHECK(withGrad_, "engram_fetch_grad requires with_grad=True");
    TORCH_CHECK(writeCalled_, "engram_fetch_grad must be called after at least one engram_write");
    TORCH_CHECK(!fetchInProgress_.load(),
                "Cannot call engram_fetch_grad while previous fetch callback is pending");

    int64_t numTokens = gradFetched.size(0);
    int64_t hidden = gradFetched.size(1);
    uint32_t rankSize = commContext_.rankSize;
    int64_t maxR = numTokens * static_cast<int64_t>(rankSize);

    int64_t hiddenBytesFp32 = hidden * static_cast<int64_t>(sizeof(float));
    int64_t gmBytes = AlignTo(maxR * hiddenBytesFp32, BUFFER_ALIGNMENT);
    AllocateAndExchangeGm(gmBytes);

    auto gradUnique = at::empty({maxR, hidden},
        at::TensorOptions().dtype(at::kFloat).device(gradFetched.device()));
    auto uniqueLocalEntry = at::empty({maxR},
        at::TensorOptions().dtype(at::kInt).device(gradFetched.device()));
    auto numUnique = at::empty({1},
        at::TensorOptions().dtype(at::kInt).device(gradFetched.device()));

    ACLNN_CMD(aclnnEngramFetchGrad, contextTensor_, gradFetched,
              perm, sendCounts, recvCounts, recvLocalEntry,
              hiddenSize_, numEngramEntries_,
              gradUnique, uniqueLocalEntry, numUnique);

    FreeGm();

    int64_t actualK = static_cast<int64_t>(numUnique.item<int32_t>());
    if (actualK < maxR) {
        gradUnique = gradUnique.narrow(0, 0, actualK);
        uniqueLocalEntry = uniqueLocalEntry.narrow(0, 0, actualK);
    }

    fetchInProgress_.store(false);
    return {gradUnique, uniqueLocalEntry};
}

// AllocateAndExchangeGm - dynamically allocate GM, register to HCCL, exchange addresses
// 每次 fetch/backward 调用:
// 1. aclrtMalloc(HUGE_FIRST) 大页申请 GM
// 2. HcclCommMemReg 注册到 HCCL（tag 带序号避免冲突）
// 3. EngramBarrier(true) 等所有 rank 注册完
// 4. HcclChannelGetRemoteMems 交换 GM 地址 → gmAddrList[peer]
// 5. UpdateContextTensor() 刷 contextTensor_ 的 gmCommBuffer
// 6. EngramBarrier(true) 等所有 rank context 更新完
void ElasticBuffer::AllocateAndExchangeGm(int64_t gmBytes)
{
    TORCH_CHECK(!destroyed_, "AllocateAndExchangeGm cannot be called after destroy");
    TORCH_CHECK(gmBytes > 0, "gmBytes must be > 0, got ", gmBytes);
    TORCH_CHECK(gmBufPtr_ == nullptr, "previous GM not freed, call FreeGm first");
    TORCH_CHECK(hcclComm_ != nullptr, "HCCL comm not initialized");

    aclError ar = aclrtMalloc(&gmBufPtr_, static_cast<uint64_t>(gmBytes), ACL_MEM_MALLOC_HUGE_FIRST);
    TORCH_CHECK(ar == ACL_SUCCESS, "aclrtMalloc(HUGE_FIRST, ", gmBytes, " B) failed, ret=", ar);

    static std::atomic<int64_t> gmSeq{0};
    gmBufferTag_ = groupName_ + "engram_embedding_gm_" + std::to_string(gmSeq.fetch_add(1));

    CommMem mem;
    mem.type = COMM_MEM_TYPE_DEVICE;
    mem.addr = gmBufPtr_;
    mem.size = static_cast<uint64_t>(gmBytes);

    auto hcclRet = HcclCommMemRegFunc(hcclComm_, gmBufferTag_.c_str(), &mem, &gmMemHandle_);
    TORCH_CHECK(hcclRet == HCCL_SUCCESS, "HcclCommMemReg(GM tag='", gmBufferTag_, "', size=", gmBytes,
                ") failed, ret=", hcclRet);

    EngramBarrier(true);

    commContext_.gmAddrList[commContext_.epRankId] = reinterpret_cast<uint64_t>(gmBufPtr_);

    for (uint32_t i = 0; i < commContext_.rankSize; ++i) {
        if (i == commContext_.epRankId)
            continue;
        uint32_t memNum = 0;
        CommMem *remoteMems = nullptr;
        char **memTags = nullptr;
        hcclRet = HcclChannelGetRemoteMemsFunc(hcclComm_, commContext_.hcommHandle[i], &memNum, &remoteMems,
                                                &memTags);
        TORCH_CHECK(hcclRet == HCCL_SUCCESS, "HcclChannelGetRemoteMems(peer=", i, ") for GM failed, ret=", hcclRet);

        bool found = false;
        for (uint32_t j = 0; j < memNum; j++) {
            if (memTags == nullptr || remoteMems == nullptr) {
                break;
            }
            if (memTags[j] != nullptr && gmBufferTag_ == memTags[j]) {
                commContext_.gmAddrList[i] = reinterpret_cast<uint64_t>(remoteMems[j].addr);
                found = true;
                break;
            }
        }
        TORCH_CHECK(found, "GM Mem tag '", gmBufferTag_, "' not found for peer ", i);
    }

    UpdateContextTensor();
    EngramBarrier(true);
}

// FreeGm - wait for kernel completion, free GM
// 1. EngramBarrier(true) 等 kernel 完成
// 2. aclrtFree 释放 GM
// 3. 清空 gmAddrList
void ElasticBuffer::FreeGm()
{
    if (gmBufPtr_ == nullptr) {
        return;
    }

    EngramBarrier(true);

    aclError ret = aclrtFree(gmBufPtr_);
    TORCH_CHECK(ret == ACL_SUCCESS, "aclrtFree(GM) failed, ret: ", ret);
    gmBufPtr_ = nullptr;
    gmMemHandle_ = nullptr;
    gmBufferTag_.clear();

    for (uint32_t i = 0; i < commContext_.rankSize; ++i) {
        commContext_.gmAddrList[i] = 0;
    }
}

// UpdateContextTensor - refresh contextTensor_ with latest commContext_
void ElasticBuffer::UpdateContextTensor()
{
    TORCH_CHECK(!destroyed_, "UpdateContextTensor cannot be called after destroy");
    at::Tensor hostTensor = at::from_blob(&commContext_, {sizeof(CommContext) / sizeof(int32_t)}, at::kInt);
    contextTensor_.copy_(hostTensor);
}

// EngramBarrier - cross-rank synchronization
void ElasticBuffer::EngramBarrier(bool withDeviceSync)
{
    TORCH_CHECK(!destroyed_,
                "engram_barrier cannot be called after destroy, please create a new ElasticBuffer instance");
    EnsureEngramContext();
    TORCH_CHECK(hcclComm_ != nullptr, "HCCL comm not initialized");

    if (withDeviceSync) {
        aclError aclRet = aclrtSynchronizeDevice();
        TORCH_CHECK(aclRet == ACL_SUCCESS, "aclrtSynchronizeDevice failed, ret: ", aclRet);
    }

    aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
    HcclResult ret = HcclBarrierFunc(hcclComm_, stream);
    TORCH_CHECK(ret == HCCL_SUCCESS, "HcclBarrier failed, ret: ", ret);

    if (withDeviceSync) {
        aclError aclRet = aclrtSynchronizeDevice();
        TORCH_CHECK(aclRet == ACL_SUCCESS, "aclrtSynchronizeDevice failed, ret: ", aclRet);
    }
}

// Destroy - explicit resource cleanup
void ElasticBuffer::Destroy()
{
    if (destroyed_)
        return;

    if (gmBufPtr_ != nullptr) {
        aclError ret = aclrtFree(gmBufPtr_);
        TORCH_CHECK(ret == ACL_SUCCESS, "aclrtFree(GM) failed, ret: ", ret);
        gmBufPtr_ = nullptr;
        gmMemHandle_ = nullptr;
    }

    if (hostBufPtr_ != nullptr) {
        aclError ret = aclrtHostUnregister(hostBufPtr_);
        TORCH_CHECK(ret == ACL_SUCCESS, "aclrtHostUnregister failed, ret: ", ret);
        ret = aclrtFreeHost(hostBufPtr_);
        TORCH_CHECK(ret == ACL_SUCCESS, "aclrtFreeHost failed, ret: ", ret);
        hostBufPtr_ = nullptr;
        deviceBufPtr_ = nullptr;
    }
    engramContextInitialized_ = false;
    moeContextInitialized_ = false;
    contextTensor_ = at::Tensor();
    moeContextTensor_ = at::Tensor();

    destroyed_ = true;
}

// GetEngramStorageSizeHint - calculate recommended CPU buffer size (static method)
int64_t ElasticBuffer::GetEngramStorageSizeHint(int64_t numEntries, int64_t hiddenSize, at::ScalarType dtype)
{
    int64_t dtypeSize = at::elementSize(dtype);
    TORCH_CHECK(hiddenSize <= INT64_MAX / dtypeSize, "hiddenSize * dtypeSize overflow");
    int64_t hiddenSizeBytes = hiddenSize * dtypeSize;
    int64_t numSfPacks = (dtypeSize <= 1) ? CeilDiv(hiddenSize, 32) : 0;
    TORCH_CHECK(hiddenSizeBytes <= INT64_MAX - numSfPacks * 4, "numBytesPerEntry addition overflow");
    int64_t numBytesPerEntry = AlignTo(hiddenSizeBytes + numSfPacks * 4, 32);
    TORCH_CHECK(numBytesPerEntry > 0 && numEntries <= INT64_MAX / numBytesPerEntry,
                "numBytesPerEntry * numEntries overflow");
    int64_t numCpuBytes = AlignTo(numBytesPerEntry * numEntries, BUFFER_ALIGNMENT);

    return numCpuBytes;
}

} // namespace Mc2Api

namespace OpApi {
namespace {

#define ACL_CHECK(expr)                                                                                                \
    do {                                                                                                               \
        aclError _s = (expr);                                                                                          \
        if (_s != ACL_SUCCESS) {                                                                                       \
            throw std::runtime_error("ACL error: " + std::string(__FILE__) + ":" + std::to_string(__LINE__) +          \
                                     " code=" + std::to_string(_s));                                                   \
        }                                                                                                              \
    } while (0)

} // namespace

class HostPinnedCounter {
public:
    HostPinnedCounter()
    {
        ACL_CHECK(aclrtMallocHost(&hostPtr_, 4 * sizeof(int64_t)));
        ACL_CHECK(aclrtHostRegisterV2(hostPtr_, 4 * sizeof(int64_t), ACL_HOST_REG_MAPPED));
        ACL_CHECK(aclrtHostGetDevicePointer(hostPtr_, &devPtr_, 0));
        Reset();
    }

    ~HostPinnedCounter()
    {
        if (hostPtr_ != nullptr) {
            aclrtHostUnregister(hostPtr_);
            aclrtFreeHost(hostPtr_);
            hostPtr_ = nullptr;
            devPtr_ = nullptr;
        }
    }

    void Reset()
    {
        *reinterpret_cast<volatile int64_t *>(hostPtr_) = -1;
    }

    int64_t SpinWait()
    {
        while (true) {
            int64_t v = *reinterpret_cast<volatile int64_t *>(hostPtr_);
            if (v >= 0) {
                return v;
            }
        }
    }

    uintptr_t DevicePtr() const
    {
        return reinterpret_cast<uintptr_t>(devPtr_);
    }

    uintptr_t HostPtr() const
    {
        return reinterpret_cast<uintptr_t>(hostPtr_);
    }

private:
    void *hostPtr_ = nullptr;
    void *devPtr_ = nullptr;
};

} // namespace OpApi

Mc2Api::ElasticBuffer::DispatchTensorList Mc2Api::ElasticBuffer::MoeEpDispatch(
    const at::Tensor &x, const at::Tensor &topkIdx, const c10::optional<at::Tensor> &topkWeights,
    const c10::optional<at::Tensor> &scales, const c10::optional<at::Tensor> &cachedHandleDstBufferSlotIdx,
    int64_t epWorldSize, int64_t epRankId, int64_t numExperts, int64_t numMaxTokensPerRank, int64_t expertAlignment,
    bool doCpuSync, int64_t hostPinnedCounterAddr)
{
    TORCH_CHECK(x.dim() == DIM_TWO, "x must be 2D");
    TORCH_CHECK((epWorldSize > 1), "The ep_world_sizes should be greater than 1, current is: ", epWorldSize);
    TORCH_CHECK(epRankId >= 0 && epRankId < epWorldSize, "ep_rank_id out of range");
    TORCH_CHECK(numExperts % epWorldSize == 0, "num_experts must be divisible by ep_world_size");
    EnsureMoeContext();

    bool anyCached = cachedHandleDstBufferSlotIdx.has_value();
    TORCH_CHECK(!(anyCached && doCpuSync), "cached mode is incompatible with do_cpu_sync=True");

    auto xSize = x.sizes();
    int64_t numTokens = xSize[0];
    int64_t topK = topkIdx.size(1);
    int64_t numLocalExperts = numExperts / epWorldSize;

    at::Tensor numRecvPerRank = at::empty({epWorldSize}, x.options().dtype(at::kInt));
    at::Tensor numRecvPerExpert = at::empty({numLocalExperts}, x.options().dtype(at::kLong));
    at::Tensor dstSlot = at::empty({numTokens, topK}, x.options().dtype(at::kInt));

    at::Tensor topkWeightsTensor = topkWeights.has_value() ? *topkWeights : at::Tensor();
    at::Tensor cachedSlotTensor =
        cachedHandleDstBufferSlotIdx.has_value() ? *cachedHandleDstBufferSlotIdx : at::Tensor();

    aclDataType scalesDtype = aclDataType::ACL_FLOAT;
    at::Tensor scalesTensor = scales.has_value() ? *scales : at::Tensor();
    if (scales.has_value() && scalesTensor.scalar_type() == at::kByte) {
        scalesDtype = aclDataType::ACL_FLOAT8_E8M0;
    }
    TensorWrapper scalesWrapper = TensorWrapper{scalesTensor, scalesDtype};

    ACLNN_CMD(aclnnMoeEpDispatch, moeContextTensor_, x, topkIdx, topkWeightsTensor, scalesWrapper, cachedSlotTensor,
              epWorldSize, epRankId, numExperts, numMaxTokensPerRank, moeCclBufferSize_, expertAlignment, doCpuSync,
              hostPinnedCounterAddr, numRecvPerRank, numRecvPerExpert, dstSlot);

    return std::tie(numRecvPerRank, numRecvPerExpert, dstSlot);
}

Mc2Api::ElasticBuffer::DispatchEpilogueTensorList Mc2Api::ElasticBuffer::MoeEpDispatchEpilogue(
    const at::Tensor &dstBufferSlotIdx, const at::Tensor &numRecvPerRank, const at::Tensor &numRecvPerExpert,
    const c10::optional<at::Tensor> &cachedRecvSrcMetadata, int64_t epWorldSize, int64_t epRankId,
    int64_t numExperts, int64_t numMaxTokensPerRank, int64_t expertAlignment, at::Tensor &recvX,
    at::Tensor &recvSrcMetadata, const c10::optional<at::Tensor> &recvTopkWeightsOpt,
    const c10::optional<at::Tensor> &recvScalesOpt)
{
    TORCH_CHECK(dstBufferSlotIdx.dim() == DIM_TWO, "dst_buffer_slot_idx must be 2D");
    EnsureMoeContext();

    at::Tensor cachedRecvSrcMetadataTensor = cachedRecvSrcMetadata.has_value() ? *cachedRecvSrcMetadata : at::Tensor();

    aclDataType recvScalesDtype = aclDataType::ACL_FLOAT;
    at::Tensor recvScalesTensor = recvScalesOpt.has_value() ? *recvScalesOpt : at::Tensor();
    if (recvScalesOpt.has_value() && recvScalesTensor.scalar_type() == at::kByte) {
        recvScalesDtype = aclDataType::ACL_FLOAT8_E8M0;
    }
    TensorWrapper recvScalesWrapper = TensorWrapper{recvScalesTensor, recvScalesDtype};

    at::Tensor recvTopkWeightsTensor = recvTopkWeightsOpt.has_value() ? *recvTopkWeightsOpt : at::Tensor();

    ACLNN_CMD(aclnnMoeEpDispatchEpilogue, moeContextTensor_, dstBufferSlotIdx, numRecvPerRank, numRecvPerExpert,
              cachedRecvSrcMetadataTensor, epWorldSize, epRankId, numExperts, numMaxTokensPerRank, moeCclBufferSize_,
              expertAlignment, recvX, recvSrcMetadata, recvTopkWeightsTensor, recvScalesWrapper);

    c10::optional<at::Tensor> recvTopkWeightsOutput;
    if (recvTopkWeightsOpt.has_value()) {
        recvTopkWeightsOutput = *recvTopkWeightsOpt;
    }
    c10::optional<at::Tensor> recvScalesOutput;
    if (recvScalesOpt.has_value()) {
        recvScalesOutput = *recvScalesOpt;
    }
    return std::make_tuple(recvX, recvSrcMetadata, recvTopkWeightsOutput, recvScalesOutput);
}

Mc2Api::ElasticBuffer::CombineTensorList Mc2Api::ElasticBuffer::MoeEpCombine(
    const at::Tensor &x, const at::Tensor &topkIdx, const at::Tensor &recvSrcMetadata,
    const at::Tensor &numRecvTokensPerExpert, const c10::optional<at::Tensor> &topkWeights,
    const c10::optional<at::Tensor> &bias0, const c10::optional<at::Tensor> &bias1, int64_t epWorldSize,
    int64_t epRankId, int64_t numExperts, int64_t numMaxTokensPerRank)
{
    TORCH_CHECK(x.dim() == DIM_TWO, "x must be 2D");
    TORCH_CHECK(!bias0.has_value(), "bias not supported in first release");
    TORCH_CHECK(!bias1.has_value(), "bias not supported in first release");
    EnsureMoeContext();

    int64_t numTokens = topkIdx.size(0);
    int64_t hidden = x.size(1);
    int64_t topK = topkIdx.size(1);

    at::Tensor combinedX = at::empty({numTokens, hidden}, x.options());
    at::Tensor combinedTopkWeights;
    if (topkWeights.has_value()) {
        combinedTopkWeights = at::empty({numTokens, topK}, x.options().dtype(at::kFloat));
    } else {
        combinedTopkWeights = at::Tensor();
    }

    c10::optional<at::Tensor> topkWeightsOpt = topkWeights;
    c10::optional<at::Tensor> bias0Opt = c10::optional<at::Tensor>();
    c10::optional<at::Tensor> bias1Opt = c10::optional<at::Tensor>();

    ACLNN_CMD(aclnnMoeEpCombine, moeContextTensor_, x, topkIdx, recvSrcMetadata, numRecvTokensPerExpert,
              topkWeightsOpt, bias0Opt, bias1Opt, epWorldSize, epRankId, numExperts, numMaxTokensPerRank,
              moeCclBufferSize_, combinedX, combinedTopkWeights);

    c10::optional<at::Tensor> combinedTopkWeightsOpt;
    if (topkWeights.has_value()) {
        combinedTopkWeightsOpt = combinedTopkWeights;
    }
    return std::make_tuple(combinedX, combinedTopkWeightsOpt);
}

// PyBind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    pybind11::class_<OpApi::HostPinnedCounter>(m, "HostPinnedCounter")
        .def(pybind11::init<>())
        .def("spin_wait", &OpApi::HostPinnedCounter::SpinWait)
        .def("reset", &OpApi::HostPinnedCounter::Reset)
        .def("device_ptr", &OpApi::HostPinnedCounter::DevicePtr)
        .def("host_ptr", &OpApi::HostPinnedCounter::HostPtr);

    pybind11::class_<Mc2Api::ElasticBuffer>(m, "ElasticBuffer")
        .def(pybind11::init<const std::string &, int64_t, bool>(),
             pybind11::arg("groupName"), pybind11::arg("numCpuBytes"),
             pybind11::arg("withGrad") = false)
        .def("engram_write", &Mc2Api::ElasticBuffer::EngramWrite, pybind11::arg("storage").noconvert())
        .def("engram_fetch", &Mc2Api::ElasticBuffer::EngramFetch,
             pybind11::arg("indices").noconvert(),
             pybind11::arg("permOut") = pybind11::none(),
             pybind11::arg("sendCountsOut") = pybind11::none(),
             pybind11::arg("recvCountsOut") = pybind11::none(),
             pybind11::arg("recvLocalEntryOut") = pybind11::none(),
             pybind11::arg("numRecvOut") = pybind11::none())
        .def("engram_fetch_grad", &Mc2Api::ElasticBuffer::EngramFetchGrad,
             pybind11::arg("gradFetched").noconvert(),
             pybind11::arg("perm").noconvert(),
             pybind11::arg("sendCounts").noconvert(),
             pybind11::arg("recvCounts").noconvert(),
             pybind11::arg("recvLocalEntry").noconvert())
        .def("engram_barrier", &Mc2Api::ElasticBuffer::EngramBarrier, pybind11::arg("withDeviceSync") = false)
        .def("destroy", &Mc2Api::ElasticBuffer::Destroy)
        .def("get_host_buf_ptr", &Mc2Api::ElasticBuffer::GetHostBufPtr)
        .def("moe_ep_dispatch", &Mc2Api::ElasticBuffer::MoeEpDispatch)
        .def("moe_ep_dispatch_epilogue", &Mc2Api::ElasticBuffer::MoeEpDispatchEpilogue)
        .def("moe_ep_combine", &Mc2Api::ElasticBuffer::MoeEpCombine)
        .def_static("get_engram_storage_size_hint", &Mc2Api::ElasticBuffer::GetEngramStorageSizeHint,
                    pybind11::arg("numEntries"), pybind11::arg("hiddenSize"), pybind11::arg("dtype") = at::kBFloat16);
}
