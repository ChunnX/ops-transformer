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
#include <functional>
#include <string>
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
struct CommContext {
    uint32_t epRankId = 0;
    uint32_t rankSize = 0;
    uint64_t virtualAddrList[HCCL_MAX_RANK_SIZE] = {};
    uint64_t hcommHandle[HCCL_MAX_RANK_SIZE] = {};
};

// ElasticBuffer class - unified interface for distributed Engram storage
class ElasticBuffer {
public:
    ElasticBuffer(const std::string &groupName, int64_t numCpuBytes);
    ~ElasticBuffer();

    void EngramWrite(const at::Tensor &storage);
    std::function<at::Tensor()> EngramFetch(const at::Tensor &indices);
    void EngramBarrier(bool withDeviceSync = false);
    void Destroy();

    int64_t GetHostBufPtr() const
    {
        return reinterpret_cast<int64_t>(hostBufPtr_);
    }

    static int64_t GetEngramStorageSizeHint(int64_t numEntries, int64_t hiddenSize,
                                            at::ScalarType dtype = at::kBFloat16);

private:
    void BuildCommContext();
    void AcquireHcclHandle();
    void AllocateAndRegisterBuffer(const HcclComm &commHandle, const std::string &memBufferTag, uint32_t rankId);
    void BuildChannelDescs(const HcclComm &commHandle, uint32_t srcRankId, uint32_t rankDim,
                           std::vector<HcclChannelDesc> &channelDesc);
    void GetHcclCommChannel(const HcclComm &commHandle, uint32_t rankDim, uint32_t srcRankId, ChannelHandle *channels,
                            uint32_t length);
    void GetHcclCommResource(const HcclComm &commHandle, CommContext *commContextStruct, uint32_t rankSize,
                             const std::string &targetTag);
    void CreateContextInternal(const HcclComm &commHandle, const std::string &mc2ContextTag);
    at::Tensor CreateContext();

    static void CopyContextToTensor(const CommContext &context, at::Tensor &tensor);
    static int64_t ContextTensorSize();

    std::string groupName_;
    int64_t numCpuBytes_;

    void *hostBufPtr_ = nullptr;
    void *deviceBufPtr_ = nullptr;
    HcclMemHandle memHandle_ = nullptr;
    HcclComm hcclComm_ = nullptr;
    CommContext commContext_;
    at::Tensor contextTensor_; // Cached context tensor (created once during init)

    int64_t hiddenSize_ = 0;
    int64_t numEngramEntries_ = 0;
    at::ScalarType engramDtype_ = at::kBFloat16;

    bool destroyed_ = false;
    bool writeCalled_ = false;
    std::atomic<bool> fetchInProgress_{false};
};

// Constructor
ElasticBuffer::ElasticBuffer(const std::string &groupName, int64_t numCpuBytes)
    : groupName_(groupName), numCpuBytes_(numCpuBytes), destroyed_(false), writeCalled_(false)
{
    InitHcclEngineCtxFunctions();
    InitHcclFunctions();
    BuildCommContext();
    contextTensor_ = CreateContext();
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

// Build communication context
void ElasticBuffer::BuildCommContext()
{
    AcquireHcclHandle();
    std::string mc2ContextTag = groupName_ + "engram_embedding";
    TORCH_CHECK(mc2ContextTag.size() <= 255, "Mc2ContextTag is too long, max size is 255, got ", mc2ContextTag.size());
    CreateContextInternal(hcclComm_, mc2ContextTag);
}

// Acquire HCCL handle
void ElasticBuffer::AcquireHcclHandle()
{
    auto hcclRet = HcomGetCommHandleByGroupFunc(groupName_.c_str(), &hcclComm_);
    TORCH_CHECK(hcclRet == HCCL_SUCCESS, "Get HCCL handle failed, group: ", groupName_.c_str(), ", ret: ", hcclRet);
}

// Allocate and register buffer
void ElasticBuffer::AllocateAndRegisterBuffer(const HcclComm &commHandle, const std::string &memBufferTag,
                                              uint32_t rankId)
{
    HostBufferGuard guard;

    aclError ar = aclrtMallocHost(&guard.hostPtr, static_cast<uint64_t>(numCpuBytes_));
    TORCH_CHECK(ar == ACL_SUCCESS, "aclrtMallocHost(", numCpuBytes_, " B) failed, ret=", ar);

    ar = aclrtHostRegisterV2(guard.hostPtr, static_cast<uint64_t>(numCpuBytes_), ACL_HOST_REG_MAPPED);
    TORCH_CHECK(ar == ACL_SUCCESS, "aclrtHostRegisterV2(", numCpuBytes_, " B) failed, ret=", ar);
    guard.registered = true;

    void *devPtr = nullptr;
    ar = aclrtHostGetDevicePointer(guard.hostPtr, &devPtr, 0);
    TORCH_CHECK(ar == ACL_SUCCESS, "aclrtHostGetDevicePointer failed, ret=", ar);

    CommMem mem;
    mem.type = COMM_MEM_TYPE_DEVICE;
    mem.addr = devPtr;
    mem.size = static_cast<uint64_t>(numCpuBytes_);

    auto hcclRet = HcclCommMemRegFunc(commHandle, memBufferTag.c_str(), &mem, &memHandle_);
    TORCH_CHECK(hcclRet == HCCL_SUCCESS, "HcclCommMemReg(tag='", memBufferTag, "', size=", numCpuBytes_,
                ") failed, ret=", hcclRet);

    hostBufPtr_ = guard.hostPtr;
    deviceBufPtr_ = devPtr;
    guard.Release();
}

// Build channel descriptors
void ElasticBuffer::BuildChannelDescs(const HcclComm &commHandle, uint32_t srcRankId, uint32_t rankDim,
                                      std::vector<HcclChannelDesc> &channelDesc)
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
                HcclResult initRet = HcclChannelDescInit(&desc, 1);
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
                desc.memHandles = &memHandle_;
                desc.memHandleNum = 1;
                channelDesc.push_back(desc);
                found = true;
            }
        }
        TORCH_CHECK(found, "No UBC_CTP/UBC_TP link found for srcRankID ", srcRankId, ", dstRankID ", peer);
    }
}

// Get HCCL communication channels
void ElasticBuffer::GetHcclCommChannel(const HcclComm &commHandle, uint32_t rankDim, uint32_t srcRankId,
                                       ChannelHandle *channels, uint32_t length)
{
    std::vector<HcclChannelDesc> descs;
    ChannelHandle channelBuf[HCCL_MAX_RANK_SIZE] = {};
    BuildChannelDescs(commHandle, srcRankId, rankDim, descs);
    auto hcclRet = HcclChannelAcquireFunc(commHandle, CommEngine::COMM_ENGINE_AIV, descs.data(),
                                          static_cast<uint32_t>(descs.size()), channelBuf);
    TORCH_CHECK(hcclRet == HCCL_SUCCESS,
                "HcclChannelAcquire(AIV, UBC_CTP/UBC_TP, memHandle_s=1) failed, ret=", hcclRet);
    for (size_t i = 0; i < descs.size(); ++i) {
        channels[descs[i].remoteRank] = channelBuf[i];
    }
}

// Get HCCL communication resources
void ElasticBuffer::GetHcclCommResource(const HcclComm &commHandle, CommContext *commContextStruct, uint32_t rankSize,
                                        const std::string &targetTag)
{
    uint32_t rankId = commContextStruct->epRankId;
    ChannelHandle handlesByRank[HCCL_MAX_RANK_SIZE] = {};
    GetHcclCommChannel(commHandle, rankSize, rankId, handlesByRank, rankSize);

    for (uint32_t peer = 0; peer < rankSize; ++peer) {
        if (peer == rankId)
            continue;
        commContextStruct->hcommHandle[peer] = handlesByRank[peer];
    }

    commContextStruct->virtualAddrList[rankId] = reinterpret_cast<uint64_t>(deviceBufPtr_);

    for (uint32_t i = 0; i < rankSize; ++i) {
        if (i == rankId)
            continue;
        uint32_t memNum = 0;
        CommMem *remoteMems = nullptr;
        char **memTags = nullptr;
        auto hcclRet = HcclChannelGetRemoteMemsFunc(commHandle, commContextStruct->hcommHandle[i], &memNum,
                                                    &remoteMems, &memTags);
        TORCH_CHECK(hcclRet == HCCL_SUCCESS, "HcclChannelGetRemoteMems(peer=", i, ") failed, ret=", hcclRet);
        // 取自己注册的buffer作为通信buffer
        bool hasTargetMem = false;
        for (uint32_t j = 0; j < memNum; j++) {
            if (memTags == nullptr || remoteMems == nullptr) {
                break;
            }
            if (memTags[j] != nullptr && targetTag == memTags[j]) {
                uint64_t targetMemAddr = reinterpret_cast<uint64_t>(remoteMems[j].addr);
                commContextStruct->virtualAddrList[i] = targetMemAddr;
                ASCEND_LOGI("Get Target Mem(%s) Success, Mem id is %d, Addr is %lu", targetTag.c_str(), j,
                            targetMemAddr);
                hasTargetMem = true;
                break;
            }
        }
        TORCH_CHECK(hasTargetMem, "Target Mem : ", targetTag, " is not found.");
    }
}

// Create context internally
void ElasticBuffer::CreateContextInternal(const HcclComm &commHandle, const std::string &mc2ContextTag)
{
    uint64_t commContext_Size = sizeof(CommContext);
    void *ctx = nullptr;
    auto hcclRet =
        HcclEngineCtxCreateFunc(commHandle, mc2ContextTag.c_str(), CommEngine::COMM_ENGINE_AIV, commContext_Size, &ctx);
    TORCH_CHECK(hcclRet == HCCL_SUCCESS, "Create HCCL context memory failed, ret: ", hcclRet);

    hcclRet = HcclGetRankIdFunc(commHandle, &commContext_.epRankId);
    TORCH_CHECK(hcclRet == HCCL_SUCCESS, "Get rank ID failed, ret: ", hcclRet);

    hcclRet = HcclGetRankSizeFunc(commHandle, &commContext_.rankSize);
    TORCH_CHECK(hcclRet == HCCL_SUCCESS, "Get rank size failed, ret: ", hcclRet);
    TORCH_CHECK(commContext_.rankSize >= HCCL_MIN_RANK_SIZE, "rankSize must be at least HCCL_MIN_RANK_SIZE, got ",
                commContext_.rankSize, ", min ", HCCL_MIN_RANK_SIZE);
    TORCH_CHECK(commContext_.rankSize <= HCCL_MAX_RANK_SIZE, "rankSize exceeds HCCL_MAX_RANK_SIZE, got ",
                commContext_.rankSize, ", max ", HCCL_MAX_RANK_SIZE);

    std::string memBufferTag = mc2ContextTag + "_buffer";
    AllocateAndRegisterBuffer(commHandle, memBufferTag, commContext_.epRankId);

    GetHcclCommResource(commHandle, &commContext_, commContext_.rankSize, memBufferTag);

    hcclRet = HcclEngineCtxCopyFunc(commHandle, CommEngine::COMM_ENGINE_AIV, mc2ContextTag.c_str(), &commContext_,
                                    commContext_Size, 0);
    TORCH_CHECK(hcclRet == HCCL_SUCCESS, "Copy context from host to device failed, ret: ", hcclRet);
}

// EngramWrite - write data with automatic barrier
void ElasticBuffer::EngramWrite(const at::Tensor &storage)
{
    TORCH_CHECK(!destroyed_, "engram_write cannot be called after destroy, "
                            "please create a new ElasticBuffer instance");

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
            TORCH_CHECK(memRet == EOK, "memcpy_s failed, ret=", memRet,
                        ", offset=", totalBytes - remaining, ", chunkSize=", chunkSize);
            dst += chunkSize;
            src += chunkSize;
            remaining -= chunkSize;
        }
    }

    EngramBarrier(true);
    writeCalled_ = true;
}

// EngramFetch - fetch data using stored metadata
std::function<at::Tensor()> ElasticBuffer::EngramFetch(const at::Tensor &indices)
{
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

    ACLNN_CMD(aclnnEngramFetch, contextTensor_, indices, hiddenSize_, numEngramEntries_, fetched);

    auto capturedContext = contextTensor_;
    auto fetchFlag = &fetchInProgress_;
    return [capturedContext, fetched, fetchFlag]() {
        ACLNN_CMD(aclnnEngramFetchWait, capturedContext, fetched);
        fetchFlag->store(false);
        return fetched;
    };
}

// EngramBarrier - cross-rank synchronization
void ElasticBuffer::EngramBarrier(bool withDeviceSync)
{
    TORCH_CHECK(!destroyed_,
                "engram_barrier cannot be called after destroy, please create a new ElasticBuffer instance");
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

    if (hostBufPtr_ != nullptr) {
        aclError ret = aclrtHostUnregister(hostBufPtr_);
        TORCH_CHECK(ret == ACL_SUCCESS, "aclrtHostUnregister failed, ret: ", ret);
        ret = aclrtFreeHost(hostBufPtr_);
        TORCH_CHECK(ret == ACL_SUCCESS, "aclrtFreeHost failed, ret: ", ret);
        hostBufPtr_ = nullptr;
        deviceBufPtr_ = nullptr;
    }

    destroyed_ = true;
}

// CreateContext - create context tensor from commContext_
at::Tensor ElasticBuffer::CreateContext()
{
    TORCH_CHECK(commContext_.rankSize > 0, "CommContext not properly initialized");

    int64_t tensorSize = ContextTensorSize();
    at::Tensor context = at::empty({tensorSize}, at::TensorOptions()
                                                     .dtype(at::kInt)
                                                     .device(c10::DeviceType::PrivateUse1)
                                                     .memory_format(c10::MemoryFormat::Contiguous));

    CopyContextToTensor(commContext_, context);
    return context;
}

// CopyContextToTensor (static helper)
void ElasticBuffer::CopyContextToTensor(const CommContext &context, at::Tensor &tensor)
{
    int64_t numElements = sizeof(CommContext) / sizeof(int32_t);
    at::Tensor hostTensor = at::empty({numElements}, at::TensorOptions().dtype(at::kInt));
    errno_t memRet = memcpy_s(hostTensor.data_ptr<int32_t>(), sizeof(CommContext), &context, sizeof(CommContext));
    TORCH_CHECK(memRet == EOK, "memcpy_s failed, ret=", memRet);
    tensor.copy_(hostTensor);
}

// ContextTensorSize (static helper)
int64_t ElasticBuffer::ContextTensorSize()
{
    return (sizeof(CommContext) + sizeof(int32_t) - 1) / sizeof(int32_t);
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

// PyBind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    pybind11::class_<Mc2Api::ElasticBuffer>(m, "ElasticBuffer")
        .def(pybind11::init<const std::string &, int64_t>(), pybind11::arg("groupName"), pybind11::arg("numCpuBytes"))
        .def("engram_write", &Mc2Api::ElasticBuffer::EngramWrite, pybind11::arg("storage").noconvert())
        .def("engram_fetch", &Mc2Api::ElasticBuffer::EngramFetch, pybind11::arg("indices").noconvert())
        .def("engram_barrier", &Mc2Api::ElasticBuffer::EngramBarrier, pybind11::arg("withDeviceSync") = false)
        .def("destroy", &Mc2Api::ElasticBuffer::Destroy)
        .def("get_host_buf_ptr", &Mc2Api::ElasticBuffer::GetHostBufPtr)
        .def_static("get_engram_storage_size_hint", &Mc2Api::ElasticBuffer::GetEngramStorageSizeHint,
                    pybind11::arg("numEntries"), pybind11::arg("hiddenSize"), pybind11::arg("dtype") = at::kBFloat16);
}