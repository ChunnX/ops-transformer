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
 * \file test.cpp
 * \brief HCCL Channel (URMA) mode communication context implementation.
 */

#include "engram_embedding_utils.h"

namespace op_api {

void CopyContextToTensor(const CommContext &context, at::Tensor &tensor)
{
    int64_t numElements = sizeof(CommContext) / sizeof(int32_t);
    at::Tensor hostTensor = at::empty({numElements}, at::TensorOptions().dtype(at::kInt));
    memcpy(hostTensor.data_ptr<int32_t>(), &context, sizeof(CommContext));
    tensor.copy_(hostTensor);
}

CommContext EngramEmbedingUtilsBuilder::Build(const std::string &group, uint64_t bufBytes)
{
    bufBytes_ = bufBytes;
    InitHcclEngineCtxFunctions();
    HcclComm hcclHandle;
    AcquireHcclHandle(group, hcclHandle);
    CommContext commContextStruct;
    BuildContext(hcclHandle, group, commContextStruct);
    return commContextStruct;
}

void EngramEmbedingUtilsBuilder::Build(const std::string &group, uint64_t bufBytes, at::Tensor &contextTensor)
{
    CommContext commContextStruct = Build(group, bufBytes);
    CopyContextToTensor(commContextStruct, contextTensor);
}

void EngramEmbedingUtilsBuilder::AcquireHcclHandle(const std::string &group, HcclComm &hcclHandle)
{
    auto aclnnRet = HcomGetCommHandleByGroupFunc(group.c_str(), &hcclHandle);
    TORCH_CHECK(aclnnRet == HCCL_SUCCESS, "Get HCCL handle failed, group: ", group.c_str(),
                ", ret: ", aclnnRet);
}

void EngramEmbedingUtilsBuilder::BuildContext(const HcclComm &hcclHandle, const std::string &group,
                      CommContext &commContextStruct)
{
    std::string mc2ContextTag = std::string(group) + "engram_embedding";
    TORCH_CHECK(mc2ContextTag.size() <= 255, "Mc2ContextTag is too long, max size is 255, got ",
                mc2ContextTag.size());
    CommEngine engine = CommEngine::COMM_ENGINE_AIV;
    void *ctx = nullptr;
    CreateContext(hcclHandle, mc2ContextTag, engine, ctx, commContextStruct);
}

void EngramEmbedingUtilsBuilder::AllocateAndRegisterBuffer(const HcclComm &commHandle, const std::string &memBufferTag,
                                    uint32_t rankId)
{
    void *hostPtr = nullptr;
    aclError ar = aclrtMallocHost(&hostPtr, bufBytes_);
    TORCH_CHECK(ar == ACL_SUCCESS, "aclrtMallocHost(", bufBytes_, " B) failed, ret=", ar);
    memset(hostPtr, 0, bufBytes_);

    ar = aclrtHostRegisterV2(hostPtr, bufBytes_, ACL_HOST_REG_MAPPED);
    TORCH_CHECK(ar == ACL_SUCCESS, "aclrtHostRegisterV2(", bufBytes_, " B) failed, ret=", ar);

    void *devPtr = nullptr;
    ar = aclrtHostGetDevicePointer(hostPtr, &devPtr, 0);
    TORCH_CHECK(ar == ACL_SUCCESS, "aclrtHostGetDevicePointer failed, ret=", ar);

    CommMem mem;
    mem.type = COMM_MEM_TYPE_DEVICE;
    mem.addr = devPtr;
    mem.size = bufBytes_;
    auto hcclRet = HcclCommMemRegFunc(commHandle, memBufferTag.c_str(), &mem, &memHandle_);
    TORCH_CHECK(hcclRet == HCCL_SUCCESS,
                "HcclCommMemReg(tag='", memBufferTag, "', size=", bufBytes_, ") failed, ret=", hcclRet);

    myBufHost_ = hostPtr;
    selfBufDev_ = devPtr;
}

void EngramEmbedingUtilsBuilder::BuildChannelDescs(const HcclComm &commHandle, uint32_t srcRankId, uint32_t rankDim,
                           std::vector<HcclChannelDesc> &channelDesc)
{
    channelDesc.clear();
    channelDesc.reserve(rankDim > 0 ? rankDim - 1 : 0);

    uint32_t *netLayers = nullptr;
    uint32_t netLayerNum = 0;
    HcclResult r = HcclRankGraphGetLayersFunc(commHandle, &netLayers, &netLayerNum);
    TORCH_CHECK(r == HCCL_SUCCESS, "Get HCCL layers failed, ret: ", r);

    for (uint32_t peer = 0; peer < rankDim; ++peer) {
        if (peer == srcRankId) continue;
        bool found = false;
        for (uint32_t li = 0; li < netLayerNum && !found; ++li) {
            CommLink *linkList = nullptr;
            uint32_t listSize = 0;
            r = HcclRankGraphGetLinksFunc(commHandle, netLayers[li], srcRankId, peer, &linkList, &listSize);
            if (r != HCCL_SUCCESS) continue;
            for (uint32_t i = 0; i < listSize && !found; ++i) {
                const int p = static_cast<int>(linkList[i].linkAttr.linkProtocol);
                if (p != COMM_PROTOCOL_UBC_CTP_VALUE && p != COMM_PROTOCOL_UBC_TP_VALUE) continue;
                HcclChannelDesc desc;
                HcclChannelDescInit(&desc, 1);
                desc.remoteRank              = peer;
                desc.channelProtocol         = linkList[i].linkAttr.linkProtocol;
                desc.localEndpoint.protocol  = linkList[i].srcEndpointDesc.protocol;
                desc.localEndpoint.commAddr  = linkList[i].srcEndpointDesc.commAddr;
                desc.localEndpoint.loc       = linkList[i].srcEndpointDesc.loc;
                desc.remoteEndpoint.protocol = linkList[i].dstEndpointDesc.protocol;
                desc.remoteEndpoint.commAddr = linkList[i].dstEndpointDesc.commAddr;
                desc.remoteEndpoint.loc      = linkList[i].dstEndpointDesc.loc;
                desc.notifyNum               = 3;
                desc.memHandles              = &memHandle_;
                desc.memHandleNum            = 1;
                channelDesc.push_back(desc);
                found = true;
            }
        }
        TORCH_CHECK(found, "No UBC_CTP/UBC_TP link found for srcRankID ", srcRankId,
                     ", dstRankID ", peer);
    }
}

void EngramEmbedingUtilsBuilder::GetHcclCommChannel(const HcclComm &commHandle, uint32_t rankDim, uint32_t srcRankId,
                             ChannelHandle *channels, uint32_t length)
{
    std::vector<HcclChannelDesc> descs;
    ChannelHandle channelBuf[HCCL_MAX_RANK_SIZE] = {};
    BuildChannelDescs(commHandle, srcRankId, rankDim, descs);
    auto hcclRet = HcclChannelAcquireFunc(commHandle, CommEngine::COMM_ENGINE_AIV,
                       descs.data(), static_cast<uint32_t>(descs.size()), channelBuf);
    for (size_t i = 0; i < descs.size(); ++i) {
        channels[descs[i].remoteRank] = channelBuf[i];
    }
    TORCH_CHECK(hcclRet == HCCL_SUCCESS, "HcclChannelAcquire(AIV, UBC_CTP/UBC_TP, memHandles=1) failed, ret=", hcclRet);
}

void EngramEmbedingUtilsBuilder::GetHcclCommResource(const HcclComm &commHandle, const CommEngine &engine,
                              CommContext *commContextStruct, uint32_t rankSize, const std::string &targetTag)
{
    uint32_t rankId = commContextStruct->epRankId;
    ChannelHandle handlesByRank[HCCL_MAX_RANK_SIZE] = {};
    GetHcclCommChannel(commHandle, rankSize, rankId, handlesByRank, rankSize);

    memset(commContextStruct->hcommHandle_, 0, rankSize * sizeof(uint64_t));
    for (uint32_t peer = 0; peer < rankSize; ++peer) {
        if (peer == rankId) continue;
        commContextStruct->hcommHandle_[peer] = handlesByRank[peer];
    }

    commContextStruct->virtualAddrList[rankId] = reinterpret_cast<uint64_t>(selfBufDev_);

    for (uint32_t i = 0; i < rankSize; ++i) {
        if (i == rankId) continue;
        uint32_t memNum = 0;
        CommMem *remoteMems = nullptr;
        char **memTags = nullptr;
        auto hcclRet = HcclChannelGetRemoteMemsFunc(commHandle, commContextStruct->hcommHandle_[i],
                                                      &memNum, &remoteMems, &memTags);
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

void EngramEmbedingUtilsBuilder::CreateContext(const HcclComm &commHandle, const std::string &mc2ContextTag,
                        const CommEngine &engine, void *&ctx,
                        CommContext &commContextStruct)
{
    uint64_t commContextSize = sizeof(CommContext);
    auto hcclRet = HcclEngineCtxCreateFunc(commHandle, mc2ContextTag.c_str(), engine, commContextSize, &ctx);
    TORCH_CHECK(hcclRet == HCCL_SUCCESS, "Create HCCL context memory failed, ret: ", hcclRet);

    hcclRet = HcclGetRankIdFunc(commHandle, &commContextStruct.epRankId);
    TORCH_CHECK(hcclRet == HCCL_SUCCESS, "Get rank ID failed, ret: ", hcclRet);

    hcclRet = HcclGetRankSizeFunc(commHandle, &commContextStruct.rankSize);
    TORCH_CHECK(hcclRet == HCCL_SUCCESS, "Get rank size failed, ret: ", hcclRet);

    std::string memBufferTag = mc2ContextTag + "_buffer";
    AllocateAndRegisterBuffer(commHandle, memBufferTag, commContextStruct.epRankId);

    GetHcclCommResource(commHandle, engine, &commContextStruct, commContextStruct.rankSize, memBufferTag);

    hcclRet = HcclEngineCtxCopyFunc(commHandle, engine, mc2ContextTag.c_str(), &commContextStruct, commContextSize, 0);
    TORCH_CHECK(hcclRet == HCCL_SUCCESS, "Copy context from host to device failed, ret: ", hcclRet);
}

EngramEmbeddingUtils::EngramEmbeddingUtils(const std::string &group, int64_t cclBufferSize)
    : group_(group), cclBufferSize_(cclBufferSize), initialized_(false),
      myBufHost_(nullptr) {}

EngramEmbeddingUtils::~EngramEmbeddingUtils() {}

at::Tensor EngramEmbeddingUtils::create_context()
{
    CommContext commContextStruct = get_comm_context();
    at::Tensor context = at::empty({context_tensor_size()}, at::TensorOptions().dtype(at::kInt)
        .device(c10::DeviceType::PrivateUse1).memory_format(c10::MemoryFormat::Contiguous));
    CopyContextToTensor(commContextStruct, context);
    return context;
}

CommContext EngramEmbeddingUtils::get_comm_context()
{
    if (initialized_) {
        return cachedContext_;
    }

    EngramEmbedingUtilsBuilder builder;
    cachedContext_ = builder.Build(group_, static_cast<uint64_t>(cclBufferSize_));
    myBufHost_ = builder.myBufHost_;
    cclBufferSize_ = static_cast<int64_t>(builder.bufBytes_);
    initialized_ = true;

    return cachedContext_;
}

void EngramEmbeddingUtils::update_group(const std::string &group, at::Tensor &contextTensor)
{
    CommContext commContextStruct;
    update_comm_context(group, commContextStruct);
    CopyContextToTensor(commContextStruct, contextTensor);
}

void EngramEmbeddingUtils::update_comm_context(const std::string &group, CommContext &commContext)
{
    if (myBufHost_ != nullptr) {
        aclrtHostUnregister(myBufHost_);
        aclrtFreeHost(myBufHost_);
        myBufHost_ = nullptr;
    }
    initialized_ = false;
    group_ = group;

    EngramEmbedingUtilsBuilder builder;
    cachedContext_ = builder.Build(group_, static_cast<uint64_t>(cclBufferSize_));
    myBufHost_ = builder.myBufHost_;
    cclBufferSize_ = static_cast<int64_t>(builder.bufBytes_);
    initialized_ = true;
    commContext = cachedContext_;
}

int64_t EngramEmbeddingUtils::ccl_buffer_size() const { return cclBufferSize_; }

int64_t EngramEmbeddingUtils::host_ptr() const { return reinterpret_cast<int64_t>(myBufHost_); }

int64_t EngramEmbeddingUtils::context_tensor_size()
{
    return (sizeof(CommContext) + sizeof(int32_t) - 1) / sizeof(int32_t);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    py::class_<EngramEmbeddingUtils>(m, "EngramEmbeddingUtils")
        .def(py::init<const std::string &, int64_t>(),
             py::arg("group"), py::arg("cclBufferSize"))
        .def("create_context", &EngramEmbeddingUtils::create_context)
        .def("update_group", &EngramEmbeddingUtils::update_group, py::arg("group"), py::arg("contextTensor").noconvert())
        .def("host_ptr", &EngramEmbeddingUtils::host_ptr);
}

} // namespace op_api