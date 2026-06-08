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
 * \file test.cpp
 * \brief HCCL Channel mode communication context implementation.
 */

#include "comm_context_channel.h"

// torch_npu头文件缺少以下声明，从CANN官方acl_rt.h补充
// extern aclError aclrtHostRegister(void *ptr, uint64_t size, aclrtHostRegisterType type);
// extern aclError aclrtHostUnregister(void *ptr);
// extern aclError aclrtHostGetDevicePointer(void *pHost, void **pDevice, uint32_t flag);

namespace op_api {

void CopyContextToTensor(const CommContext &context, at::Tensor &tensor)
{
    at::Tensor hostContext = at::from_blob(const_cast<CommContext *>(&context),
                                           {sizeof(CommContext) / sizeof(int32_t)}, at::kInt);
    tensor.copy_(hostContext);
}

void HcclChannelContextBuilder::Build(const std::string &group, int64_t worldSize, int64_t &cclBufferSize, at::Tensor &contextTensor)
{
    ASCEND_LOGI("Start to get CommContext Tensor, group: %s", group.c_str());
    InitHcclEngineCtxFunctions();
    HcclComm hcclHandle;
    AcquireHcclHandle(group, hcclHandle);
    CommProtocol protocol;
    GetCommProtocol(hcclHandle, protocol);
    CommContext commContextStruct;
    BuildContext(hcclHandle, group, protocol, commContextStruct, cclBufferSize);
    CopyContextToTensor(commContextStruct, contextTensor);

    peerBufAddrs_.resize(rankSize_);
    channelPtrByRank_.resize(rankSize_);
    memcpy(peerBufAddrs_.data(), commContextStruct.epHcclBuffer_, rankSize_ * sizeof(uint64_t));
    memcpy(channelPtrByRank_.data(), commContextStruct.hcommHandle_, rankSize_ * sizeof(uint64_t));

    ASCEND_LOGI("Get CommContext Tensor Success, group: %s, ccl_buffer_size: %ld",
                group.c_str(), cclBufferSize);
}

CommContext HcclChannelContextBuilder::Build(const std::string &group, int64_t worldSize, int64_t &cclBufferSize)
{
    ASCEND_LOGI("Start to get CommContext struct, group: %s", group.c_str());
    InitHcclEngineCtxFunctions();
    HcclComm hcclHandle;
    AcquireHcclHandle(group, hcclHandle);
    CommProtocol protocol;
    GetCommProtocol(hcclHandle, protocol);
    CommContext commContextStruct;
    BuildContext(hcclHandle, group, protocol, commContextStruct, cclBufferSize);

    peerBufAddrs_.resize(rankSize_);
    channelPtrByRank_.resize(rankSize_);
    memcpy(peerBufAddrs_.data(), commContextStruct.epHcclBuffer_, rankSize_ * sizeof(uint64_t));
    memcpy(channelPtrByRank_.data(), commContextStruct.hcommHandle_, rankSize_ * sizeof(uint64_t));

    ASCEND_LOGI("Get CommContext struct Success, group: %s, ccl_buffer_size: %ld",
                group.c_str(), cclBufferSize);
    return commContextStruct;
}

void HcclChannelContextBuilder::AcquireHcclHandle(const std::string &group, HcclComm &hcclHandle)
{
    auto aclnnRet = HcomGetCommHandleByGroupFunc(group.c_str(), &hcclHandle);
    TORCH_CHECK(aclnnRet == HCCL_SUCCESS, "Get HCCL handle failed, group: ", group.c_str(),
                ", ret: ", aclnnRet);
    ASCEND_LOGI("Get HCCL communication handle success hcclHandle is: %p", hcclHandle);
}

void HcclChannelContextBuilder::BuildContext(const HcclComm &hcclHandle, const std::string &group,
                      const CommProtocol &protocol, CommContext &commContextStruct,
                      int64_t &cclBufferSize)
{
    std::string mc2ContextTag = std::string(group) + "moe_dispatch_ffn_combine";
    TORCH_CHECK(mc2ContextTag.size() <= 255, "Mc2ContextTag is too long, max size is 255, got ",
                mc2ContextTag.size());
    CommEngine engine = CommEngine::COMM_ENGINE_AIV;
    void *ctx = nullptr;
    uint64_t hcclBuffSize = 0;
    CreateContext(hcclHandle, mc2ContextTag, engine, protocol, ctx, commContextStruct, COMM_DEFAULT_BUF_BYTES);
    cclBufferSize = static_cast<int64_t>(bufBytes_);
}

void HcclChannelContextBuilder::GetCommProtocol(const HcclComm &commHandle, CommProtocol &protocol)
{
    uint32_t layerNum = 0;
    uint32_t *layerList = nullptr;
    auto ret = HcclRankGraphGetLayersFunc(commHandle, &layerList, &layerNum);
    TORCH_CHECK(ret == HCCL_SUCCESS, "Get HCCL layers failed, ret: ", ret);
    if (layerNum == HCCL_COMM_LAYERS_MTE_CCU) {
        protocol = CommProtocol::COMM_PROTOCOL_UB_MEM;
        return;
    }
    CheckProtocolSupport(commHandle, layerList, layerNum);
    protocol = CommProtocol::COMM_PROTOCOL_UB_MEM;
}

void HcclChannelContextBuilder::CheckProtocolSupport(const HcclComm &commHandle, uint32_t *&layerList, uint32_t &layerNum)
{
    uint32_t srcRankId = 0;
    auto hcclRet = HcclGetRankIdFunc(commHandle, &srcRankId);
    TORCH_CHECK(hcclRet == HCCL_SUCCESS, "CheckProtocolSupport Get rank ID failed, ret: ", hcclRet);
    for (uint32_t layerIndex = 0; layerIndex < layerNum; ++layerIndex) {
        uint32_t rankNumInLayer = 0;
        uint32_t *rankIdLists = nullptr;
        hcclRet = HcclRankGraphGetRanksByLayerFunc(commHandle, layerList[layerIndex], &rankIdLists, &rankNumInLayer);
        TORCH_CHECK(hcclRet == HCCL_SUCCESS, "Get rank IDs by layer failed, ret: ", hcclRet);
        for (uint32_t rankId = 0; rankId < rankNumInLayer; ++rankId) {
            if (rankIdLists[rankId] == srcRankId || layerMap_.find(rankIdLists[rankId]) != layerMap_.end()) {
                continue;
            }
            CommLink *linksList = nullptr;
            uint32_t netLinkNum = 0;
            hcclRet = HcclRankGraphGetLinksFunc(commHandle, layerList[layerIndex], srcRankId, rankIdLists[rankId],
                                                &linksList, &netLinkNum);
            TORCH_CHECK(hcclRet == HCCL_SUCCESS, "Get HCCL links failed, ret: ", hcclRet);
            TORCH_CHECK(netLinkNum > 0, "No available HCCL links found");
            TORCH_CHECK(CheckLinks(netLinkNum, linksList), "No HCCL links support UB_MEM srcRankID ", srcRankId,
                        ", dstRankID ", rankIdLists[rankId], ", layer ", layerList[layerIndex]);
            layerMap_[rankIdLists[rankId]] = layerList[layerIndex];
        }
    }
}

bool HcclChannelContextBuilder::CheckLinks(uint32_t &netLinkNum, CommLink *linksList)
{
    for (uint32_t j = 0; j < netLinkNum; ++j) {
        if (linksList[j].linkAttr.linkProtocol == CommProtocol::COMM_PROTOCOL_UB_MEM) {
            return true;
        }
    }
    return false;
}

void HcclChannelContextBuilder::AllocateAndRegisterBuffer(const HcclComm &commHandle, const std::string &mc2ContextTag,
                                    uint64_t bufBytes)
{
    void *hostPtr = nullptr;
    aclError ar = aclrtMallocHost(&hostPtr, bufBytes);
    TORCH_CHECK(ar == ACL_SUCCESS, "aclrtMallocHost(", bufBytes, " B) failed, ret=", ar);
    memset(hostPtr, 0, bufBytes);

    // aclrtHostRegisterV2(hostPtr, bufBytes, 0) 等价于 aclrtHostRegister(hostPtr, bufBytes, ACL_HOST_REGISTER_MAPPED)
    ar = aclrtHostRegisterV2(hostPtr, bufBytes, 0);
    TORCH_CHECK(ar == ACL_SUCCESS, "aclrtHostRegister(", bufBytes, " B) failed, ret=", ar);

    void *devPtr = nullptr;
    ar = aclrtHostGetDevicePointer(hostPtr, &devPtr, 0);
    TORCH_CHECK(ar == ACL_SUCCESS, "aclrtHostGetDevicePointer failed, ret=", ar);

    CommMem mem;
    mem.type = COMM_MEM_TYPE_DEVICE;
    mem.addr = devPtr;
    mem.size = bufBytes;
    HcclMemHandle myMemHandle = nullptr;
    auto hcclRet = HcclCommMemRegFunc(commHandle, mc2ContextTag.c_str(), &mem, &myMemHandle);
    TORCH_CHECK(hcclRet == HCCL_SUCCESS,
                "HcclCommMemReg(tag='", mc2ContextTag, "', size=", bufBytes, ") failed, ret=", hcclRet);

    myBufHost_ = hostPtr;
    myBufDev_ = devPtr;
    myMemHandle_ = myMemHandle;
    bufBytes_ = bufBytes;
    ASCEND_LOGI("AllocateAndRegisterBuffer success, hostPtr=%p, devPtr=%p, bufBytes=%lu",
                hostPtr, devPtr, bufBytes);
}

void HcclChannelContextBuilder::InitHcclChannel(const HcclComm &commHandle, uint32_t rankDim, uint32_t srcRankId,
                          const CommProtocol &protocol, std::vector<HcclChannelDesc> &channelDesc)
{
    uint32_t channelNum = channelDesc.size();
    auto hcclRet = HcclChannelDescInit(channelDesc.data(), channelNum);
    TORCH_CHECK(hcclRet == HCCL_SUCCESS, "HCCL channel init failed, ret: ", hcclRet);
    uint32_t netLayerNum = 0;
    uint32_t *netLayerList = nullptr;
    GetNetLayers(commHandle, netLayerList, netLayerNum);
    TORCH_CHECK(netLayerNum > 0, "Get HCCL net layers failed, netLayerNum is ", netLayerNum);
    for (uint32_t i = 0; i < rankDim; ++i) {
        if (i == srcRankId) continue;
        uint32_t dstRank = i;
        uint32_t channelId = (i > srcRankId) ? (i - 1) : i;
        CommLink *links = nullptr;
        uint32_t layerId = netLayerNum == 1 ?
                netLayerList[HCCL_COMM_LAYERS_UB_MEM] : layerMap_[dstRank];
        GetHcclCommLink(commHandle, layerId, srcRankId, dstRank, protocol, links);
        channelDesc[channelId].channelProtocol = protocol;
        channelDesc[channelId].remoteRank = dstRank;
        channelDesc[channelId].notifyNum = channelNum;
        channelDesc[channelId].localEndpoint = links->srcEndpointDesc;
        channelDesc[channelId].remoteEndpoint = links->dstEndpointDesc;
        channelDesc[channelId].memHandles = &myMemHandle_;
        channelDesc[channelId].memHandleNum = 1;
    }
}

void HcclChannelContextBuilder::GetHcclCommChannel(const HcclComm &commHandle, uint32_t rankDim, uint32_t srcRankId,
                             const CommProtocol &protocol, const CommEngine &engine,
                             std::vector<ChannelHandle> &channels)
{
    uint32_t channelNum = rankDim - 1;
    std::vector<HcclChannelDesc> channelDesc(channelNum);
    channels.resize(channelNum);
    InitHcclChannel(commHandle, rankDim, srcRankId, protocol, channelDesc);
    auto hcclRet = HcclChannelAcquireFunc(commHandle, engine, channelDesc.data(), channelNum, channels.data());
    TORCH_CHECK(hcclRet == HCCL_SUCCESS, "Acquire HCCL channel failed, ret: ", hcclRet);
}

void HcclChannelContextBuilder::GetHcclCommResource(const HcclComm &commHandle, const CommEngine &engine, const CommProtocol &protocol,
                              CommContext *commContextStruct, uint32_t rankSize)
{
    uint32_t rankId = commContextStruct->epRankId;
    std::vector<ChannelHandle> channels;
    GetHcclCommChannel(commHandle, rankSize, rankId, protocol, engine, channels);

    commContextStruct->epHcclBuffer_[rankId] = reinterpret_cast<uint64_t>(myBufDev_);

    for (uint32_t i = 0; i < rankSize; ++i) {
        if (i == rankId) continue;
        uint32_t idx = (i < rankId) ? i : (i - 1);
        uint32_t memNum = 0;
        CommMem *remoteMems = nullptr;
        char **memTags = nullptr;
        auto hcclRet = HcclChannelGetRemoteMemsFunc(commHandle, channels[idx], &memNum, &remoteMems, &memTags);
        TORCH_CHECK(hcclRet == HCCL_SUCCESS,
                    "HcclChannelGetRemoteMems(peer=", i, ") failed, ret=", hcclRet);
        TORCH_CHECK(memNum >= 1,
                    "HcclChannelGetRemoteMems(peer=", i, ") returned memNum=0");
        commContextStruct->epHcclBuffer_[i] = reinterpret_cast<uint64_t>(remoteMems[memNum - 1].addr);
    }

    CollectChannelPtrs(commHandle, channels, rankId, rankSize, commContextStruct);
}

void HcclChannelContextBuilder::CollectChannelPtrs(const HcclComm &commHandle,
                             const std::vector<ChannelHandle> &channels,
                             uint32_t srcRankId, uint32_t rankSize,
                             CommContext *commContextStruct)
{
    std::vector<ChannelHandle> nonSelfHandles;
    for (uint32_t p = 0; p < rankSize; ++p) {
        if (p == srcRankId) continue;
        uint32_t idx = (p < srcRankId) ? p : (p - 1);
        nonSelfHandles.push_back(channels[idx]);
    }

    std::vector<ChannelPtr> nonSelfPtrs(nonSelfHandles.size(), 0);
    HcommResult hr = HcommChannelGetPtrByHandleFunc(
        nonSelfHandles.data(),
        static_cast<uint32_t>(nonSelfHandles.size()),
        nonSelfPtrs.data());
    TORCH_CHECK(hr == HCOMM_SUCCESS_VALUE,
        "HcommChannelGetPtrByHandle failed, ret=", hr);

    size_t ptrIdx = 0;
    for (uint32_t p = 0; p < rankSize; ++p) {
        if (p == srcRankId) {
            commContextStruct->hcommHandle_[p] = 0;
            continue;
        }
        commContextStruct->hcommHandle_[p] = static_cast<uint64_t>(nonSelfPtrs[ptrIdx]);
        ptrIdx++;
    }
    ASCEND_LOGI("CollectChannelPtrs success, collected %zu ChannelPtr entries", nonSelfPtrs.size());
}

void HcclChannelContextBuilder::CreateContext(const HcclComm &commHandle, const std::string &mc2ContextTag,
                        const CommEngine &engine, const CommProtocol &protocol, void *&ctx,
                        CommContext &commContextStruct, uint64_t bufBytes)
{
    uint64_t commContextSize = sizeof(CommContext);
    auto hcclRet = HcclEngineCtxCreateFunc(commHandle, mc2ContextTag.c_str(), engine, commContextSize, &ctx);
    TORCH_CHECK(hcclRet == HCCL_SUCCESS, "Create HCCL context memory failed, ret: ", hcclRet);

    hcclRet = HcclGetRankIdFunc(commHandle, &commContextStruct.epRankId);
    TORCH_CHECK(hcclRet == HCCL_SUCCESS, "Get rank ID failed, ret: ", hcclRet);
    rankId_ = commContextStruct.epRankId;

    uint32_t rSize = 0;
    hcclRet = HcclGetRankSizeFunc(commHandle, &rSize);
    TORCH_CHECK(hcclRet == HCCL_SUCCESS, "Get rank size failed, ret: ", hcclRet);
    rankSize_ = rSize;

    AllocateAndRegisterBuffer(commHandle, mc2ContextTag, bufBytes);

    GetHcclCommResource(commHandle, engine, protocol, &commContextStruct, rankSize_);

    hcclRet = HcclEngineCtxCopyFunc(commHandle, engine, mc2ContextTag.c_str(), &commContextStruct, commContextSize, 0);
    TORCH_CHECK(hcclRet == HCCL_SUCCESS, "Copy context from host to device failed, ret: ", hcclRet);
}

void HcclChannelContextBuilder::GetNetLayers(const HcclComm &commHandle, uint32_t *&netLayerList, uint32_t &netLayerNum)
{
    auto hcclRet = HcclRankGraphGetLayersFunc(commHandle, &netLayerList, &netLayerNum);
    TORCH_CHECK(hcclRet == HCCL_SUCCESS, "Get HCCL layers failed, ret: ", hcclRet);
}

void HcclChannelContextBuilder::GetHcclCommLink(const HcclComm &commHandle, uint32_t netLayerId, uint32_t srcRankId,
                                  uint32_t dstRankId, const CommProtocol &protocol, CommLink *&links)
{
    CommLink *linksList = nullptr;
    uint32_t netLinkNum = 0;
    auto hcclRet = HcclRankGraphGetLinksFunc(commHandle, netLayerId, srcRankId, dstRankId, &linksList, &netLinkNum);
    TORCH_CHECK(hcclRet == HCCL_SUCCESS, "Get HCCL Communication link failed, ret: ", hcclRet);
    TORCH_CHECK(netLinkNum > 0, "The Net Link Is nullptr");
    uint32_t index = 0;
    for (; index < netLinkNum; ++index) {
        if (linksList[index].linkAttr.linkProtocol == protocol) {
            links = &linksList[index];
            break;
        }
    }
    TORCH_CHECK(index < netLinkNum, "No matching communication protocol found");
}

CommContextManager::CommContextManager(const std::string &group, int64_t worldSize)
    : group_(group), worldSize_(worldSize), cclBufferSize_(0), initialized_(false),
      myBufDev_(nullptr), myMemHandle_(nullptr), bufBytes_(0),
      rankId_(0), rankSize_(0) {}

CommContextManager::~CommContextManager()
{
    if (myBufHost_ != nullptr) {
        aclrtHostUnregister(myBufHost_);
        aclrtFreeHost(myBufHost_);
        myBufHost_ = nullptr;
        myBufDev_ = nullptr;
    }
}

at::Tensor CommContextManager::create_context()
{
    CommContext commContextStruct = get_comm_context();
    at::Tensor context = at::empty({context_tensor_size()}, at::TensorOptions().dtype(at::kInt)
        .device(c10::DeviceType::PrivateUse1).memory_format(c10::MemoryFormat::Contiguous));
    CopyContextToTensor(commContextStruct, context);
    return context;
}

CommContext CommContextManager::get_comm_context()
{
    if (initialized_) {
        CommContext commContextStruct;
        commContextStruct.epRankId = rankId_;
        memcpy(commContextStruct.epHcclBuffer_, peerBufAddrs_.data(), rankSize_ * sizeof(uint64_t));
        memcpy(commContextStruct.hcommHandle_, channelPtrByRank_.data(), rankSize_ * sizeof(uint64_t));
        return commContextStruct;
    }

    HcclChannelContextBuilder builder;
    CommContext commContextStruct = builder.Build(group_, worldSize_, cclBufferSize_);

    myBufHost_ = builder.myBufHost_;
    myBufDev_ = builder.myBufDev_;
    myMemHandle_ = builder.myMemHandle_;
    bufBytes_ = builder.bufBytes_;
    rankId_ = builder.rankId_;
    rankSize_ = builder.rankSize_;
    peerBufAddrs_ = builder.peerBufAddrs_;
    channelPtrByRank_ = builder.channelPtrByRank_;
    initialized_ = true;

    return commContextStruct;
}

void CommContextManager::update_group(const std::string &group, at::Tensor &contextTensor)
{
    CommContext commContextStruct;
    update_comm_context(group, commContextStruct);
    CopyContextToTensor(commContextStruct, contextTensor);
}

void CommContextManager::update_comm_context(const std::string &group, CommContext &commContext)
{
    if (myBufHost_ != nullptr) {
        aclrtHostUnregister(myBufHost_);
        aclrtFreeHost(myBufHost_);
        myBufHost_ = nullptr;
        myBufDev_ = nullptr;
    }
    initialized_ = false;
    group_ = group;
    cclBufferSize_ = 0;

    HcclChannelContextBuilder builder;
    commContext = builder.Build(group_, worldSize_, cclBufferSize_);

    myBufHost_ = builder.myBufHost_;
    myBufDev_ = builder.myBufDev_;
    myMemHandle_ = builder.myMemHandle_;
    bufBytes_ = builder.bufBytes_;
    rankId_ = builder.rankId_;
    rankSize_ = builder.rankSize_;
    peerBufAddrs_ = builder.peerBufAddrs_;
    channelPtrByRank_ = builder.channelPtrByRank_;
    initialized_ = true;
}

int64_t CommContextManager::ccl_buffer_size() const { return cclBufferSize_; }

int64_t CommContextManager::context_tensor_size()
{
    return (sizeof(CommContext) + sizeof(int32_t) - 1) / sizeof(int32_t);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    py::class_<CommContextManager>(m, "CommContextManager")
        .def(py::init<const std::string &, int64_t>(),
             py::arg("group"), py::arg("worldSize"))
        .def("create_context", &CommContextManager::create_context)
        .def("update_group", &CommContextManager::update_group, py::arg("group"), py::arg("contextTensor").noconvert())
        .def_property_readonly("ccl_buffer_size", &CommContextManager::ccl_buffer_size);
}

} // namespace op_api