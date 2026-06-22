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

    void destroy();

    int64_t getHostBufPtr() const
    {
        return reinterpret_cast<int64_t>(hostBufPtr);
    }


private:
    void buildCommContext();
    void acquireHcclHandle();
    void allocateAndRegisterBuffer(const HcclComm &commHandle, const std::string &memTag, uint32_t rankId);
    void buildChannelDescs(const HcclComm &commHandle, uint32_t srcRankId, uint32_t rankDim,
                           std::vector<HcclChannelDesc> &channelDesc);
    void getHcclCommChannel(const HcclComm &commHandle, uint32_t rankDim, uint32_t srcRankId, ChannelHandle *channels,
                            uint32_t length);
    void getHcclCommResource(const HcclComm &commHandle, CommContext *commContextStruct, uint32_t rankSize);
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

    bool destroyed = false;
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
}

// Destructor - automatic resource cleanup
ElasticBuffer::~ElasticBuffer()
{
    destroy();
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
    void *hostPtr = nullptr;
    aclError ar = aclrtMallocHost(&hostPtr, static_cast<uint64_t>(numCpuBytes));
    TORCH_CHECK(ar == ACL_SUCCESS, "aclrtMallocHost(", numCpuBytes, " B) failed, ret=", ar);
    memset(hostPtr, 0, static_cast<uint64_t>(numCpuBytes));

    ar = aclrtHostRegisterV2(hostPtr, static_cast<uint64_t>(numCpuBytes), ACL_HOST_REG_MAPPED);
    TORCH_CHECK(ar == ACL_SUCCESS, "aclrtHostRegisterV2(", numCpuBytes, " B) failed, ret=", ar);

    void *devPtr = nullptr;
    ar = aclrtHostGetDevicePointer(hostPtr, &devPtr, 0);
    TORCH_CHECK(ar == ACL_SUCCESS, "aclrtHostGetDevicePointer failed, ret=", ar);

    CommMem mem;
    mem.type = COMM_MEM_TYPE_DEVICE;
    mem.addr = devPtr;
    mem.size = static_cast<uint64_t>(numCpuBytes);

    auto hcclRet = HcclCommMemRegFunc(commHandle, memBufferTag.c_str(), &mem, &memHandle);
    TORCH_CHECK(hcclRet == HCCL_SUCCESS, "HcclCommMemReg(tag='", memBufferTag, "', size=", numCpuBytes,
                ") failed, ret=", hcclRet);

    hostBufPtr = hostPtr;
    deviceBufPtr = devPtr;
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

} // namespace op_api

// PyBind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    pybind11::class_<op_api::ElasticBuffer>(m, "ElasticBuffer")
        .def(pybind11::init<const std::string &, int64_t>(), pybind11::arg("groupName"), pybind11::arg("numCpuBytes"))
        .def("destroy", &op_api::ElasticBuffer::destroy)
        .def("get_host_buf_ptr", &op_api::ElasticBuffer::getHostBufPtr)
}