/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * ElasticBuffer - Unified implementation combining buffer management and operations
 */

#include <torch/extension.h>
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <functional>
#include <string>
#include <vector>
#include <cstring>

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

// Ensure ACL runtime API inline wrappers link correctly
extern "C" {
aclError aclrtHostRegisterV2(void *ptr, uint64_t size, uint32_t flag);
aclError aclrtHostUnregister(void *ptr);
aclError aclrtHostGetDevicePointer(void *pHost, void **pDevice, uint32_t flag);
}

namespace op_api {

// Constants
constexpr static uint8_t COMM_ENGINE_AIV = 4;
constexpr uint32_t HCCL_MAX_RANK_SIZE = 1024;
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

    void release()
    {
        hostPtr = nullptr;
        registered = false;
    }
};

// Helper functions
static inline int64_t ceilDiv(int64_t x, int64_t y)
{
    return (x + y - 1) / y;
}

static inline int64_t alignTo(int64_t x, int64_t y)
{
    return ceilDiv(x, y) * y;
}

// CommContext structure for HCCL communication
struct CommContext {
    uint32_t epRankId = 0;
    uint32_t rankSize = 0;
    uint64_t virtualAddrList[HCCL_MAX_RANK_SIZE] = {};
    uint64_t hcommHandle_[HCCL_MAX_RANK_SIZE] = {};
};

// ElasticBuffer class - unified interface for distributed Engram storage
class ElasticBuffer {
public:
    ElasticBuffer(const std::string &groupName, int64_t numCpuBytes);
    ~ElasticBuffer();

    void engramWrite(const at::Tensor &storage);
    std::function<at::Tensor()> engramFetch(const at::Tensor &indices);
    void engramBarrier(bool withDeviceSync = false);
    void destroy();

    int64_t getHostBufPtr() const
    {
        return reinterpret_cast<int64_t>(hostBufPtr);
    }

    static int64_t getEngramStorageSizeHint(int64_t numEntries, int64_t hiddenSize,
                                            at::ScalarType dtype = at::kBFloat16);

private:
    void buildCommContext();
    void acquireHcclHandle();
    void allocateAndRegisterBuffer(const HcclComm &commHandle, const std::string &memBufferTag, uint32_t rankId);
    void buildChannelDescs(const HcclComm &commHandle, uint32_t srcRankId, uint32_t rankDim,
                           std::vector<HcclChannelDesc> &channelDesc);
    void getHcclCommChannel(const HcclComm &commHandle, uint32_t rankDim, uint32_t srcRankId, ChannelHandle *channels,
                            uint32_t length);
    void getHcclCommResource(const HcclComm &commHandle, CommContext *commContextStruct, uint32_t rankSize, const std::string targetTag);
    void createContextInternal(const HcclComm &commHandle, const std::string &mc2ContextTag);
    at::Tensor createContext();

    static void copyContextToTensor(const CommContext &context, at::Tensor &tensor);
    static int64_t contextTensorSize();

    std::string groupName;
    int64_t numCpuBytes;

    void *hostBufPtr = nullptr;
    void *deviceBufPtr = nullptr;
    HcclMemHandle memHandle = nullptr;
    HcclComm hcclComm = nullptr;
    CommContext commContext;
    at::Tensor contextTensor; // Cached context tensor (created once during init)

    int64_t hiddenSize = 0;
    int64_t numEngramEntries = 0;
    at::ScalarType engramDtype = at::kBFloat16;

    bool initialized = false;
    bool destroyed = false;
    bool writeCalled = false;
};

// Constructor
ElasticBuffer::ElasticBuffer(const std::string &groupName, int64_t numCpuBytes)
    : groupName(groupName), numCpuBytes(numCpuBytes), initialized(false), destroyed(false), writeCalled(false)
{
    TORCH_CHECK(numCpuBytes >= 0, "num_cpu_bytes must be non-negative, got ", numCpuBytes);
    TORCH_CHECK(numCpuBytes % BUFFER_ALIGNMENT == 0, "num_cpu_bytes must be 2MB-aligned, got ", numCpuBytes,
                " bytes which is not divisible by ", BUFFER_ALIGNMENT, " bytes");

    InitHcclEngineCtxFunctions();
    buildCommContext();
    contextTensor = createContext();
    initialized = true;
}

// Destructor - automatic resource cleanup
ElasticBuffer::~ElasticBuffer()
{
    try {
        destroy();
    } catch (const std::exception &e) {
        ASCEND_LOGE("ElasticBuffer destructor cleanup failed: %s", e.what());
    }
}

// Build communication context
void ElasticBuffer::buildCommContext()
{
    acquireHcclHandle();
    std::string mc2ContextTag = groupName + "engram_embedding";
    TORCH_CHECK(mc2ContextTag.size() <= 255, "Mc2ContextTag is too long, max size is 255, got ", mc2ContextTag.size());
    createContextInternal(hcclComm, mc2ContextTag);
}

// Acquire HCCL handle
void ElasticBuffer::acquireHcclHandle()
{
    auto hcclRet = HcomGetCommHandleByGroupFunc(groupName.c_str(), &hcclComm);
    TORCH_CHECK(hcclRet == HCCL_SUCCESS, "Get HCCL handle failed, group: ", groupName.c_str(), ", ret: ", hcclRet);
}

// Allocate and register buffer
void ElasticBuffer::allocateAndRegisterBuffer(const HcclComm &commHandle, const std::string &memBufferTag, uint32_t rankId)
{
    HostBufferGuard guard;

    aclError ar = aclrtMallocHost(&guard.hostPtr, static_cast<uint64_t>(numCpuBytes));
    TORCH_CHECK(ar == ACL_SUCCESS, "aclrtMallocHost(", numCpuBytes, " B) failed, ret=", ar);
    memset(guard.hostPtr, 0, static_cast<uint64_t>(numCpuBytes));

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

    auto hcclRet = HcclCommMemRegFunc(commHandle, memBufferTag.c_str(), &mem, &memHandle);
    TORCH_CHECK(hcclRet == HCCL_SUCCESS, "HcclCommMemReg(tag='", memBufferTag, "', size=", numCpuBytes,
                ") failed, ret=", hcclRet);

    hostBufPtr = guard.hostPtr;
    deviceBufPtr = devPtr;
    guard.release();
}

// Build channel descriptors
void ElasticBuffer::buildChannelDescs(const HcclComm &commHandle, uint32_t srcRankId, uint32_t rankDim,
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
                desc.memHandles = &memHandle;
                desc.memHandleNum = 1;
                channelDesc.push_back(desc);
                found = true;
            }
        }
        TORCH_CHECK(found, "No UBC_CTP/UBC_TP link found for srcRankID ", srcRankId, ", dstRankID ", peer);
    }
}

// Get HCCL communication channels
void ElasticBuffer::getHcclCommChannel(const HcclComm &commHandle, uint32_t rankDim, uint32_t srcRankId,
                                       ChannelHandle *channels, uint32_t length)
{
    std::vector<HcclChannelDesc> descs;
    ChannelHandle channelBuf[HCCL_MAX_RANK_SIZE] = {};
    buildChannelDescs(commHandle, srcRankId, rankDim, descs);
    auto hcclRet = HcclChannelAcquireFunc(commHandle, CommEngine::COMM_ENGINE_AIV, descs.data(),
                                          static_cast<uint32_t>(descs.size()), channelBuf);
    for (size_t i = 0; i < descs.size(); ++i) {
        channels[descs[i].remoteRank] = channelBuf[i];
    }
    TORCH_CHECK(hcclRet == HCCL_SUCCESS, "HcclChannelAcquire(AIV, UBC_CTP/UBC_TP, memHandles=1) failed, ret=", hcclRet);
}

// Get HCCL communication resources
void ElasticBuffer::getHcclCommResource(const HcclComm &commHandle, CommContext *commContextStruct, uint32_t rankSize, const std::string targetTag)
{
    uint32_t rankId = commContextStruct->epRankId;
    ChannelHandle handlesByRank[HCCL_MAX_RANK_SIZE] = {};
    getHcclCommChannel(commHandle, rankSize, rankId, handlesByRank, rankSize);

    memset(commContextStruct->hcommHandle_, 0, rankSize * sizeof(uint64_t));
    for (uint32_t peer = 0; peer < rankSize; ++peer) {
        if (peer == rankId)
            continue;
        commContextStruct->hcommHandle_[peer] = handlesByRank[peer];
    }

    commContextStruct->virtualAddrList[rankId] = reinterpret_cast<uint64_t>(deviceBufPtr);

    for (uint32_t i = 0; i < rankSize; ++i) {
        if (i == rankId)
            continue;
        uint32_t memNum = 0;
        CommMem *remoteMems = nullptr;
        char **memTags = nullptr;
        auto hcclRet = HcclChannelGetRemoteMemsFunc(commHandle, commContextStruct->hcommHandle_[i], &memNum,
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
                ASCEND_LOGI("Get Targe Mem(%s) Success, Mem id is %d, Addr is %lu", targetTag.c_str(), j, targetMemAddr);
                hasTargetMem = true;
                break;
            }
        }
        TORCH_CHECK(hasTargetMem, "Target Mem : ", targetTag, " is not found.");
    }
}

// Create context internally
void ElasticBuffer::createContextInternal(const HcclComm &commHandle, const std::string &mc2ContextTag)
{
    uint64_t commContextSize = sizeof(CommContext);
    void *ctx = nullptr;
    auto hcclRet =
        HcclEngineCtxCreateFunc(commHandle, mc2ContextTag.c_str(), CommEngine::COMM_ENGINE_AIV, commContextSize, &ctx);
    TORCH_CHECK(hcclRet == HCCL_SUCCESS, "Create HCCL context memory failed, ret: ", hcclRet);

    hcclRet = HcclGetRankIdFunc(commHandle, &commContext.epRankId);
    TORCH_CHECK(hcclRet == HCCL_SUCCESS, "Get rank ID failed, ret: ", hcclRet);

    hcclRet = HcclGetRankSizeFunc(commHandle, &commContext.rankSize);
    TORCH_CHECK(hcclRet == HCCL_SUCCESS, "Get rank size failed, ret: ", hcclRet);

    std::string memBufferTag = mc2ContextTag + "_buffer";
    allocateAndRegisterBuffer(commHandle, memBufferTag, commContext.epRankId);

    getHcclCommResource(commHandle, &commContext, commContext.rankSize, memBufferTag);

    hcclRet = HcclEngineCtxCopyFunc(commHandle, CommEngine::COMM_ENGINE_AIV, mc2ContextTag.c_str(), &commContext,
                                    commContextSize, 0);
    TORCH_CHECK(hcclRet == HCCL_SUCCESS, "Copy context from host to device failed, ret: ", hcclRet);
}

// EngramWrite - write data with automatic barrier
void ElasticBuffer::engramWrite(const at::Tensor &storage)
{
    TORCH_CHECK(!destroyed, "ElasticBuffer has been destroyed");
    TORCH_CHECK(initialized, "ElasticBuffer not initialized");

    TORCH_CHECK(storage.is_cpu(), "storage must be on CPU, got device: ", storage.device());
    TORCH_CHECK(storage.dim() == 2, "storage must be 2D, got dimensions: ", storage.dim());
    TORCH_CHECK(storage.is_contiguous(), "storage must be contiguous");
    TORCH_CHECK(storage.scalar_type() == at::kBFloat16 || storage.scalar_type() == at::kHalf ||
                    storage.scalar_type() == at::kFloat,
                "storage dtype must be bfloat16/float16/float32, got: ", storage.scalar_type());
    TORCH_CHECK(storage.size(1) % 128 == 0, "storage second dimension must be 128-aligned, got: ", storage.size(1));
    TORCH_CHECK(storage.nbytes() <= static_cast<size_t>(numCpuBytes), "storage size ", storage.nbytes(),
                " exceeds buffer capacity ", numCpuBytes);

    engramBarrier(true);

    hiddenSize = storage.size(1);
    numEngramEntries = storage.size(0);
    engramDtype = storage.scalar_type();
    writeCalled = true;

    if (numEngramEntries > 0) {
        memcpy(hostBufPtr, storage.data_ptr(), storage.nbytes());
    }

    engramBarrier(true);
}

// EngramFetch - fetch data using stored metadata
std::function<at::Tensor()> ElasticBuffer::engramFetch(const at::Tensor &indices)
{
    TORCH_CHECK(!destroyed, "ElasticBuffer has been destroyed");
    TORCH_CHECK(writeCalled, "Must call engramWrite before engramFetch");

    TORCH_CHECK(indices.device().type() == c10::DeviceType::PrivateUse1,
                "indices must be on NPU, got device: ", indices.device());
    TORCH_CHECK(indices.dim() == 1, "indices must be 1D, got dimensions: ", indices.dim());
    TORCH_CHECK(indices.scalar_type() == at::kInt, "indices dtype must be int32, got: ", indices.scalar_type());

    int64_t numTokens = indices.size(0);
    if (numTokens == 0) {
        auto emptyTensor = at::empty({0, hiddenSize}, at::TensorOptions().dtype(engramDtype).device(indices.device()));
        return [=]() { return emptyTensor; };
    }

    auto fetched = at::empty({numTokens, hiddenSize}, at::TensorOptions().dtype(engramDtype).device(indices.device()));

    auto dtypeVal = static_cast<int64_t>(engramDtype);
    ACLNN_CMD(aclnnEngramFetch, contextTensor, indices, fetched, hiddenSize, numEngramEntries, dtypeVal);

    auto capturedContext = contextTensor;
    return [capturedContext, fetched]() { 
        ACLNN_CMD(aclnnEngramFetchWait, capturedContext);
        return fetched; 
    };
}

// EngramBarrier - cross-rank synchronization
void ElasticBuffer::engramBarrier(bool withDeviceSync)
{
    TORCH_CHECK(!destroyed, "ElasticBuffer has been destroyed");
    TORCH_CHECK(hcclComm != nullptr, "HCCL comm not initialized");

    static bool barrierInitialized = false;
    if (!barrierInitialized) {
        InitHcclFunctions();
        barrierInitialized = true;
    }

    if (withDeviceSync) {
        aclError aclRet = aclrtSynchronizeDevice();
        TORCH_CHECK(aclRet == ACL_SUCCESS, "aclrtSynchronizeDevice failed, ret: ", aclRet);
    }

    aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
    HcclResult ret = HcclBarrierFunc(hcclComm, stream);
    TORCH_CHECK(ret == HCCL_SUCCESS, "HcclBarrier failed, ret: ", ret);

    if (withDeviceSync) {
        aclError aclRet = aclrtSynchronizeDevice();
        TORCH_CHECK(aclRet == ACL_SUCCESS, "aclrtSynchronizeDevice failed, ret: ", aclRet);
    }
}

// Destroy - explicit resource cleanup
void ElasticBuffer::destroy()
{
    if (destroyed)
        return;

    if (hostBufPtr != nullptr) {
        aclError ret = aclrtHostUnregister(hostBufPtr);
        TORCH_CHECK(ret == ACL_SUCCESS, "aclrtHostUnregister failed, ret: ", ret);
        ret = aclrtFreeHost(hostBufPtr);
        TORCH_CHECK(ret == ACL_SUCCESS, "aclrtFreeHost failed, ret: ", ret);
        hostBufPtr = nullptr;
        deviceBufPtr = nullptr;
    }

    destroyed = true;
}

// CreateContext - create context tensor from commContext
at::Tensor ElasticBuffer::createContext()
{
    TORCH_CHECK(commContext.rankSize > 0, "CommContext not properly initialized");

    int64_t tensorSize = contextTensorSize();
    at::Tensor context = at::empty({tensorSize}, at::TensorOptions()
                                                     .dtype(at::kInt)
                                                     .device(c10::DeviceType::PrivateUse1)
                                                     .memory_format(c10::MemoryFormat::Contiguous));

    copyContextToTensor(commContext, context);
    return context;
}

// CopyContextToTensor (static helper)
void ElasticBuffer::copyContextToTensor(const CommContext &context, at::Tensor &tensor)
{
    int64_t numElements = sizeof(CommContext) / sizeof(int32_t);
    at::Tensor hostTensor = at::empty({numElements}, at::TensorOptions().dtype(at::kInt));
    memcpy(hostTensor.data_ptr<int32_t>(), &context, sizeof(CommContext));
    tensor.copy_(hostTensor);
}

// ContextTensorSize (static helper)
int64_t ElasticBuffer::contextTensorSize()
{
    return (sizeof(CommContext) + sizeof(int32_t) - 1) / sizeof(int32_t);
}

// GetEngramStorageSizeHint - calculate recommended CPU buffer size (static method)
int64_t ElasticBuffer::getEngramStorageSizeHint(int64_t numEntries, int64_t hiddenSize, at::ScalarType dtype)
{
    TORCH_CHECK(numEntries >= 0, "num_entries must be non-negative, got ", numEntries);
    TORCH_CHECK(hiddenSize >= 0, "hidden_size must be non-negative, got ", hiddenSize);
    TORCH_CHECK(hiddenSize % 128 == 0, "hidden_size must be 128-aligned, got ", hiddenSize);

    int64_t dtypeSize = at::elementSize(dtype);
    int64_t numSfPacks = (dtypeSize <= 1) ? ceilDiv(hiddenSize, 32) : 0;
    int64_t numBytesPerEntry = alignTo(hiddenSize * dtypeSize + numSfPacks * 4, 32);
    int64_t numCpuBytes = alignTo(numBytesPerEntry * numEntries, BUFFER_ALIGNMENT);

    return numCpuBytes;
}

} // namespace op_api

// PyBind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    pybind11::class_<op_api::ElasticBuffer>(m, "ElasticBuffer")
        .def(pybind11::init<const std::string &, int64_t>(), pybind11::arg("groupName"), pybind11::arg("numCpuBytes"))
        .def("engram_write", &op_api::ElasticBuffer::engramWrite, pybind11::arg("storage").noconvert())
        .def("engram_fetch", &op_api::ElasticBuffer::engramFetch, pybind11::arg("indices").noconvert())
        .def("engram_barrier", &op_api::ElasticBuffer::engramBarrier, pybind11::arg("withDeviceSync") = false)
        .def("destroy", &op_api::ElasticBuffer::destroy)
        .def("get_host_buf_ptr", &op_api::ElasticBuffer::getHostBufPtr)
        .def_static("get_engram_storage_size_hint", &op_api::ElasticBuffer::getEngramStorageSizeHint,
                    pybind11::arg("numEntries"), pybind11::arg("hiddenSize"), pybind11::arg("dtype") = at::kBFloat16);
}