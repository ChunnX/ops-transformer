/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <torch/extension.h>
#include "hccl_common.h"
#include "moe_shared_jetty_adapter.h"
#include "moe_shared_jetty_context.h"
#include "torch_npu/csrc/aten/common/from_blob.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Mc2Api {
namespace {
constexpr uint32_t SHARED_JETTY_NUM = Mc2Aclnn::MOE_SHARED_JETTY_NUM;
constexpr size_t SHARED_JETTY_MAX_TAG_SIZE = 255;
constexpr CommEngine CommEngineUse = CommEngine::COMM_ENGINE_AIV;
constexpr uint64_t BUFFER_ALIGNMENT = 2ULL * 1024ULL * 1024ULL;
constexpr uint64_t HUGE1G_SIZE = 1024ULL * 1024ULL * 1024ULL;
constexpr uint64_t VIRTUAL_ADDRESS_RESERVE_FLAGS = 0ULL;

struct MoeSharedBufferEntry {
    HcclComm comm = nullptr;
    void *deviceBuffer = nullptr;
    aclrtDrvMemHandle physicalMemHandle = nullptr;
    HcclMemHandle memHandle = nullptr;
    uint64_t size = 0;
};

// HCCL has no matching unregister API for channel-visible memory. Keep the
// buffer alive for as long as its communicator may reference the registration.
std::mutex gMoeSharedBufferMutex;
std::unordered_map<std::string, MoeSharedBufferEntry> gMoeSharedBuffers;
} // namespace

using Mc2Aclnn::MoeSharedJettyContext;

class MoeSharedJettyContextBuilder {
private:
    HcclComm comm_ = nullptr;
    uint32_t rankSize_ = 0;
    uint32_t layer_ = 0;
    MoeSharedJettyContext context_;
    uint64_t requiredBufferSize_ = 0;
    std::vector<HcclChannelDesc> channelDescs_;
    void *deviceBuffer_ = nullptr;
    aclrtDrvMemHandle physicalMemHandle_ = nullptr;
    HcclMemHandle memHandle_ = nullptr;
    bool bufferRetained_ = false;

    static void CheckHccl(HcclResult ret, const char *operation)
    {
        TORCH_CHECK(ret == HCCL_SUCCESS, operation, " failed, ret=", ret);
    }

    static void LogCleanupFailure(aclError ret, const char *operation)
    {
        if (ret != ACL_SUCCESS) {
            ASCEND_LOGE("%s failed during MoE buffer cleanup, ret=%d", operation, ret);
        }
    }

    void ReleaseBuffer()
    {
        if (deviceBuffer_ == nullptr) {
            return;
        }
        if (physicalMemHandle_ != nullptr) {
            LogCleanupFailure(aclrtUnmapMem(deviceBuffer_), "aclrtUnmapMem");
            LogCleanupFailure(aclrtFreePhysical(physicalMemHandle_), "aclrtFreePhysical");
            physicalMemHandle_ = nullptr;
        }
        LogCleanupFailure(aclrtReleaseMemAddress(deviceBuffer_), "aclrtReleaseMemAddress");
        deviceBuffer_ = nullptr;
        memHandle_ = nullptr;
    }

    static aclError SetBufferAccess(void *address, uint64_t size, uint32_t deviceId)
    {
        aclrtMemAccessDesc access = {};
        access.flags = ACL_RT_MEM_ACCESS_FLAGS_READWRITE;
        access.location.id = deviceId;
        access.location.type = ACL_MEM_LOCATION_TYPE_DEVICE;
        return aclrtMemSetAccess(address, size, &access, 1);
    }

    bool AllocateHugePageBuffer(uint64_t size, aclrtPhysicalMemProp &property, aclrtMemAttr memoryType,
                                uint32_t deviceId)
    {
        void *address = nullptr;
        auto ret = aclrtReserveMemAddress(&address, size, 0, nullptr, VIRTUAL_ADDRESS_RESERVE_FLAGS);
        if (ret != ACL_SUCCESS) {
            ASCEND_LOGI("aclrtReserveMemAddress(%lu) failed, ret=%d", size, ret);
            return false;
        }
        property.memAttr = memoryType;
        aclrtDrvMemHandle handle = nullptr;
        ret = aclrtMallocPhysical(&handle, size, &property, 0);
        if (ret != ACL_SUCCESS) {
            ASCEND_LOGI("aclrtMallocPhysical(%lu) failed, ret=%d", size, ret);
            LogCleanupFailure(aclrtReleaseMemAddress(address), "aclrtReleaseMemAddress");
            return false;
        }
        ret = aclrtMapMem(address, size, 0, handle, 0);
        if (ret != ACL_SUCCESS) {
            ASCEND_LOGI("aclrtMapMem(%lu) failed, ret=%d", size, ret);
            LogCleanupFailure(aclrtFreePhysical(handle), "aclrtFreePhysical");
            LogCleanupFailure(aclrtReleaseMemAddress(address), "aclrtReleaseMemAddress");
            return false;
        }
        const aclError accessRet = SetBufferAccess(address, size, deviceId);
        if (accessRet != ACL_SUCCESS) {
            ASCEND_LOGI("aclrtMemSetAccess(%lu) failed, ret=%d", size, accessRet);
            LogCleanupFailure(aclrtUnmapMem(address), "aclrtUnmapMem");
            LogCleanupFailure(aclrtFreePhysical(handle), "aclrtFreePhysical");
            LogCleanupFailure(aclrtReleaseMemAddress(address), "aclrtReleaseMemAddress");
            return false;
        }
        deviceBuffer_ = address;
        physicalMemHandle_ = handle;
        return true;
    }

    void AllocateDeviceBuffer()
    {
        int32_t deviceId = 0;
        auto ret = aclrtGetDevice(&deviceId);
        TORCH_CHECK(ret == ACL_SUCCESS, "aclrtGetDevice failed, ret=", ret);
        aclrtPhysicalMemProp property = {};
        property.handleType = ACL_MEM_HANDLE_TYPE_NONE;
        property.allocationType = ACL_MEM_ALLOCATION_TYPE_PINNED;
        property.location.id = static_cast<uint32_t>(deviceId);
        property.location.type = ACL_MEM_LOCATION_TYPE_DEVICE;
        TORCH_CHECK(requiredBufferSize_ <= std::numeric_limits<uint64_t>::max() - (HUGE1G_SIZE - 1),
                    "MoE communication buffer size overflows 1GB alignment: ", requiredBufferSize_);
        const uint64_t huge1GBytes = ((requiredBufferSize_ + HUGE1G_SIZE - 1) / HUGE1G_SIZE) * HUGE1G_SIZE;
        const uint32_t currentDeviceId = static_cast<uint32_t>(deviceId);
        if (!AllocateHugePageBuffer(huge1GBytes, property, ACL_MEM_HUGE1G, currentDeviceId)) {
            TORCH_CHECK(AllocateHugePageBuffer(requiredBufferSize_, property, ACL_MEM_HUGE, currentDeviceId),
                        "Allocate MoE communication buffer failed, size=", requiredBufferSize_);
        }
        ret = aclrtMemset(deviceBuffer_, requiredBufferSize_, 0, requiredBufferSize_);
        TORCH_CHECK(ret == ACL_SUCCESS, "Clear MoE communication buffer failed, ret=", ret);
    }

    void RegisterDeviceBuffer(const std::string &contextTag, const std::string &bufferTag)
    {
        AllocateDeviceBuffer();
        CommMem memory = {};
        memory.type = COMM_MEM_TYPE_DEVICE;
        memory.addr = deviceBuffer_;
        memory.size = requiredBufferSize_;
        CheckHccl(HcclCommMemRegFunc(comm_, bufferTag.c_str(), &memory, &memHandle_), "HcclCommMemReg");
        // There is no HCCL unregister API. From this point onward the communicator
        // may retain the address even if later channel/context construction fails.
        bufferRetained_ = true;
        TORCH_CHECK(memHandle_ != nullptr, "HcclCommMemReg returned a null memory handle");
        RetainBuffer(contextTag);
        for (auto &desc : channelDescs_) {
            desc.memHandles = &memHandle_;
            desc.memHandleNum = 1;
        }
    }

    void RetainBuffer(const std::string &contextTag)
    {
        std::lock_guard<std::mutex> lock(gMoeSharedBufferMutex);
        const MoeSharedBufferEntry entry = {comm_, deviceBuffer_, physicalMemHandle_, memHandle_, requiredBufferSize_};
        // A new communicator may reuse the group name after the old communicator
        // and its HCCL context are destroyed. Publish the current generation.
        gMoeSharedBuffers[contextTag] = entry;
    }

    void ReuseBuffer(const std::string &contextTag)
    {
        std::lock_guard<std::mutex> lock(gMoeSharedBufferMutex);
        auto entry = gMoeSharedBuffers.find(contextTag);
        TORCH_CHECK(entry != gMoeSharedBuffers.end(), "MoE shared Jetty context '", contextTag,
                    "' exists but its registered buffer is missing; use a new communication group");
        TORCH_CHECK(entry->second.comm == comm_, "MoE shared Jetty context '", contextTag,
                    "' does not match the current communicator; use a unique HCCL group name");
        TORCH_CHECK(requiredBufferSize_ <= entry->second.size, "MoE communication buffer exceeds the existing ",
                    "buffer for this group, requested=", requiredBufferSize_, ", allocated=", entry->second.size,
                    "; use a new communication group initialized with the larger operator shape");
        deviceBuffer_ = entry->second.deviceBuffer;
        physicalMemHandle_ = entry->second.physicalMemHandle;
        memHandle_ = entry->second.memHandle;
        bufferRetained_ = true;
    }

    void GetRankInfo()
    {
        CheckHccl(HcclGetRankIdFunc(comm_, &context_.epRankId), "HcclGetRankId");
        CheckHccl(HcclGetRankSizeFunc(comm_, &rankSize_), "HcclGetRankSize");
        TORCH_CHECK(rankSize_ >= Mc2Aclnn::MOE_SHARED_JETTY_MIN_RANK_SIZE &&
                        rankSize_ <= Mc2Aclnn::MOE_SHARED_JETTY_MAX_RANK_SIZE,
                    "MoE shared Jetty rank size must be in [", Mc2Aclnn::MOE_SHARED_JETTY_MIN_RANK_SIZE, ", ",
                    Mc2Aclnn::MOE_SHARED_JETTY_MAX_RANK_SIZE,
                    "], got ", rankSize_);
        TORCH_CHECK(context_.epRankId < rankSize_, "Invalid shared Jetty rank ID: ", context_.epRankId);
        context_.epRankSize = rankSize_;
        // Each shared Jetty serves every remote rank.
        context_.sharedJettyNum = SHARED_JETTY_NUM;
    }

    static bool IsSameEndpoint(const EndpointDesc &lhs, const EndpointDesc &rhs)
    {
        return lhs.protocol == rhs.protocol && lhs.commAddr.type == rhs.commAddr.type &&
               std::memcmp(lhs.commAddr.raws, rhs.commAddr.raws, sizeof(lhs.commAddr.raws)) == 0 &&
               lhs.loc.locType == rhs.loc.locType &&
               std::memcmp(lhs.loc.raws, rhs.loc.raws, sizeof(lhs.loc.raws)) == 0;
    }

    bool LayerCoversAllRanks(uint32_t layer)
    {
        uint32_t *ranks = nullptr;
        uint32_t rankNum = 0;
        CheckHccl(HcclRankGraphGetRanksByLayerFunc(comm_, layer, &ranks, &rankNum), "HcclRankGraphGetRanksByLayer");
        TORCH_CHECK(rankNum == 0 || ranks != nullptr, "Null rank list for layer ", layer);
        // Rank count alone cannot detect missing ranks in a malformed topology.
        std::vector<bool> covered(rankSize_, false);
        for (uint32_t i = 0; i < rankNum; ++i) {
            TORCH_CHECK(ranks[i] < rankSize_, "Invalid rank in layer ", layer, ": ", ranks[i]);
            covered[ranks[i]] = true;
        }
        return std::all_of(covered.begin(), covered.end(), [](bool value) { return value; });
    }

    bool CollectPeerLinks(uint32_t layer, CommProtocol protocol, std::vector<std::vector<CommLink>> &peerLinks)
    {
        for (uint32_t peer = 0; peer < rankSize_; ++peer) {
            if (peer == context_.epRankId) {
                continue;
            }
            CommLink *links = nullptr;
            uint32_t linkNum = 0;
            CheckHccl(HcclRankGraphGetLinksFunc(comm_, layer, context_.epRankId, peer, &links, &linkNum),
                      "HcclRankGraphGetLinks");
            TORCH_CHECK(linkNum == 0 || links != nullptr, "Null link list for layer ", layer, ", peer ", peer);
            std::vector<CommLink> candidates;
            for (uint32_t i = 0; i < linkNum; ++i) {
                if (links[i].linkAttr.linkProtocol == protocol) {
                    candidates.push_back(links[i]);
                }
            }
            if (candidates.empty()) {
                return false;
            }
            peerLinks.push_back(std::move(candidates));
        }
        return true;
    }

    bool MatchSharedEndpoint(const EndpointDesc &endpoint, const std::vector<std::vector<CommLink>> &peerLinks,
                             CommProtocol protocol, std::vector<HcclChannelDesc> &descs)
    {
        CheckHccl(HcclChannelDescInit(descs.data(), static_cast<uint32_t>(descs.size())), "HcclChannelDescInit");
        for (uint32_t peer = 0, index = 0; peer < rankSize_; ++peer) {
            if (peer == context_.epRankId) {
                continue;
            }
            const auto &links = peerLinks[index];
            auto link = std::find_if(links.begin(), links.end(), [&](const CommLink &value) {
                return IsSameEndpoint(endpoint, value.srcEndpointDesc);
            });
            if (link == links.end()) {
                return false;
            }
            auto &desc = descs[index++];
            desc.channelProtocol = protocol;
            desc.remoteRank = peer;
            desc.localEndpoint = link->srcEndpointDesc;
            desc.remoteEndpoint = link->dstEndpointDesc;
        }
        return true;
    }

    bool SelectSharedLinks(uint32_t layer, CommProtocol protocol)
    {
        if (!LayerCoversAllRanks(layer)) {
            return false;
        }
        std::vector<std::vector<CommLink>> peerLinks;
        if (!CollectPeerLinks(layer, protocol, peerLinks)) {
            return false;
        }

        // A later link of the first peer may be the one shared by every peer.
        // Ranks must see consistent RankGraph candidate ordering, as in the
        // common Clos plane. This is not a negotiation of asymmetric EID choices.
        for (const auto &candidate : peerLinks.front()) {
            std::vector<HcclChannelDesc> descs(peerLinks.size());
            if (MatchSharedEndpoint(candidate.srcEndpointDesc, peerLinks, protocol, descs)) {
                channelDescs_ = std::move(descs);
                layer_ = layer;
                ASCEND_LOGI("Selected shared Jetty layer %u", layer_);
                return true;
            }
        }
        return false;
    }

    void GetCommProtocol(CommProtocol protocol)
    {
        uint32_t *layers = nullptr;
        uint32_t layerNum = 0;
        CheckHccl(HcclRankGraphGetLayersFunc(comm_, &layers, &layerNum), "HcclRankGraphGetLayers");
        TORCH_CHECK(layerNum > 0 && layers != nullptr, "No HCCL network layers for shared Jetty");
        // Preserve the RankGraph order; never combine links from different layers.
        for (uint32_t i = 0; i < layerNum; ++i) {
            if (SelectSharedLinks(layers[i], protocol)) {
                return;
            }
        }
        TORCH_CHECK(false, "No layer supports shared Jetty to all peers with protocol ",
                    static_cast<int>(protocol), ", local rank ", context_.epRankId);
    }

    std::vector<ChannelHandle> AcquireSharedChannels(const std::string &queueTag)
    {
        std::vector<ChannelHandle> channels(channelDescs_.size());
        HcclChannelConfig config = nullptr;
        CheckHccl(HcclChannelConfigCreateFunc(&config), "HcclChannelConfigCreate");
        const char *operation = "HcclChannelConfigSetInt";
        auto ret = HcclChannelConfigSetIntFunc(config, HCCL_CHANNEL_CONFIG_TYPE_IS_SHARED_QUEUE, 1);
        if (ret == HCCL_SUCCESS) {
            operation = "HcclChannelConfigSetStr";
            ret = HcclChannelConfigSetStrFunc(config, HCCL_CHANNEL_CONFIG_TYPE_SHARED_QUEUE_TAG, queueTag.c_str());
        }
        if (ret == HCCL_SUCCESS) {
            operation = "HcclChannelAcquireWithConfig";
            ret = HcclChannelAcquireWithConfigFunc(comm_, CommEngineUse, channelDescs_.data(),
                                                  static_cast<uint32_t>(channelDescs_.size()), config, channels.data());
        }
        auto destroyRet = HcclChannelConfigDestroyFunc(config);
        TORCH_CHECK(ret == HCCL_SUCCESS, operation, " failed, ret=", ret, ", config cleanup ret=", destroyRet);
        CheckHccl(destroyRet, "HcclChannelConfigDestroy");
        return channels;
    }

    void PopulateRegisteredBufferAddresses(const std::vector<ChannelHandle> &channels,
                                           const std::string &bufferTag)
    {
        context_.epHcclBuffer[context_.epRankId] = reinterpret_cast<uint64_t>(deviceBuffer_);
        for (uint32_t peer = 0, index = 0; peer < rankSize_; ++peer) {
            if (peer == context_.epRankId) {
                continue;
            }
            uint32_t memoryCount = 0;
            CommMem *remoteMemories = nullptr;
            char **memoryTags = nullptr;
            CheckHccl(HcclChannelGetRemoteMemsFunc(comm_, channels[index++], &memoryCount, &remoteMemories,
                                                   &memoryTags),
                      "HcclChannelGetRemoteMems");
            bool found = false;
            for (uint32_t i = 0; i < memoryCount && remoteMemories != nullptr && memoryTags != nullptr; ++i) {
                if (memoryTags[i] != nullptr && bufferTag == memoryTags[i]) {
                    TORCH_CHECK(remoteMemories[i].addr != nullptr && remoteMemories[i].size >= requiredBufferSize_,
                                "Invalid remote MoE buffer, peer=", peer, ", required=", requiredBufferSize_,
                                ", available=", remoteMemories[i].size);
                    context_.epHcclBuffer[peer] = reinterpret_cast<uint64_t>(remoteMemories[i].addr);
                    found = true;
                    break;
                }
            }
            TORCH_CHECK(found, "Registered MoE buffer '", bufferTag, "' not found for peer ", peer);
        }
    }

    void GetHcclCommResource(const std::string &contextTag, const std::string &bufferTag)
    {
        std::vector<ChannelHandle> channels;
        for (uint32_t jetty = 0; jetty < context_.sharedJettyNum; ++jetty) {
            const std::string queueTag = contextTag + "_jetty_" + std::to_string(jetty);
            if (jetty == 0) {
                // The public multi-channel API does not return its constituent channels.
                // Reusing this tag below returns the same channels, without a second Jetty.
                channels = AcquireSharedChannels(queueTag);
            }
            CheckHccl(CreateMoeMultiChannelHandle(comm_, queueTag.c_str(), channelDescs_.data(),
                                                static_cast<uint32_t>(channelDescs_.size()),
                                                &context_.multiChannelHandles[jetty]),
                      "MakeMultiChannelHandle");
            TORCH_CHECK(context_.multiChannelHandles[jetty] != 0, "Null shared Jetty handle, index=", jetty);
        }
        PopulateRegisteredBufferAddresses(channels, bufferTag);
    }

    void *BuildContext(const std::string &contextTag, const std::string &bufferTag)
    {
        void *ctx = nullptr;
        uint64_t size = 0;
        auto ret = HcclEngineCtxGetFunc(comm_, contextTag.c_str(), CommEngineUse, &ctx, &size);
        if (ret == HCCL_SUCCESS) {
            TORCH_CHECK(ctx != nullptr && size == sizeof(MoeSharedJettyContext), "Invalid shared Jetty context cache");
            ReuseBuffer(contextTag);
            return ctx;
        }
        // Build resources before publishing the context so failed channel creation
        // cannot leave a partially initialized context in the HCCL cache.
        CommProtocol protocol = CommProtocol::COMM_PROTOCOL_UBC_CTP;
        GetCommProtocol(protocol);
        RegisterDeviceBuffer(contextTag, bufferTag);
        GetHcclCommResource(contextTag, bufferTag);
        CheckHccl(HcclEngineCtxCreateFunc(comm_, contextTag.c_str(), CommEngineUse, sizeof(context_), &ctx),
                  "HcclEngineCtxCreate");
        TORCH_CHECK(ctx != nullptr, "HcclEngineCtxCreate returned a null context");
        ret = HcclEngineCtxCopyFunc(comm_, CommEngineUse, contextTag.c_str(), &context_, sizeof(context_), 0);
        if (ret != HCCL_SUCCESS) {
            auto destroyRet = HcclEngineCtxDestroyFunc(comm_, contextTag.c_str(), CommEngineUse);
            TORCH_CHECK(false, "Copy shared Jetty context failed, ret=", ret,
                        ", context cleanup ret=", destroyRet, "; use a new communication group");
        }
        return ctx;
    }

    static at::Tensor CreateContextTensor(void *ctx)
    {
        static_assert(sizeof(MoeSharedJettyContext) % sizeof(int32_t) == 0, "Context must fit an int32 tensor");
        const int64_t elements = sizeof(MoeSharedJettyContext) / sizeof(int32_t);
        auto options = at::TensorOptions().dtype(at::kInt).device(c10::DeviceType::PrivateUse1);
        // HCCL owns the context and multi-channel handles. The process-lifetime
        // registry owns the communication window for the same lifetime as the tag.
        return at_npu::native::from_blob(ctx, {elements}, options);
    }

public:
    MoeSharedJettyContextBuilder() = default;
    MoeSharedJettyContextBuilder(const MoeSharedJettyContextBuilder &) = delete;
    MoeSharedJettyContextBuilder &operator=(const MoeSharedJettyContextBuilder &) = delete;

    ~MoeSharedJettyContextBuilder()
    {
        if (!bufferRetained_) {
            ReleaseBuffer();
        }
    }

    at::Tensor Build(const std::string &groupName, uint64_t requiredBufferSize)
    {
        // As with ordinary channel construction, callers must serialize builds on
        // the same communicator and enter them in matching order across its ranks.
        // After a resource build failure, discard the communicator before retrying:
        // HCCL may still own channels or contexts from the failed attempt.
        TORCH_CHECK(!groupName.empty() && groupName.find('\0') == std::string::npos, "Invalid HCCL group name");
        const std::string contextTag = groupName + "_moe_dispatch_combine_shared_jetty_v2_j" +
                                       std::to_string(Mc2Aclnn::MOE_SHARED_JETTY_NUM) + "_sq" +
                                       std::to_string(Mc2Aclnn::MOE_SHARED_JETTY_SQ_DEPTH) + "_cq" +
                                       std::to_string(Mc2Aclnn::MOE_SHARED_JETTY_CQ_DEPTH);
        const std::string bufferTag = contextTag + "_buffer";
        const std::string lastQueueTag = contextTag + "_jetty_" + std::to_string(SHARED_JETTY_NUM - 1);
        TORCH_CHECK(lastQueueTag.size() <= SHARED_JETTY_MAX_TAG_SIZE, "Shared Jetty context tag is too long");
        TORCH_CHECK(bufferTag.size() <= SHARED_JETTY_MAX_TAG_SIZE, "Shared Jetty buffer tag is too long");
        TORCH_CHECK(requiredBufferSize > 0 && requiredBufferSize % BUFFER_ALIGNMENT == 0,
                    "MoE communication buffer must be positive and 2MB aligned, got ", requiredBufferSize);
        InitHcclSharedJettyFunctions();
        CheckHccl(HcomGetCommHandleByGroupFunc(groupName.c_str(), &comm_), "HcomGetCommHandleByGroup");
        TORCH_CHECK(comm_ != nullptr, "Null HCCL communicator");
        context_ = {};
        channelDescs_.clear();
        GetRankInfo();
        requiredBufferSize_ = requiredBufferSize;
        void *ctx = BuildContext(contextTag, bufferTag);
        return CreateContextTensor(ctx);
    }
};

at::Tensor BuildMoeSharedJettyContext(const std::string &groupName, uint64_t requiredBufferSize)
{
    MoeSharedJettyContextBuilder builder;
    return builder.Build(groupName, requiredBufferSize);
}

} // namespace Mc2Api
