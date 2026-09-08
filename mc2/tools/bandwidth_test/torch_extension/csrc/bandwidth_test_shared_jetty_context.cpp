/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <torch/extension.h>
#include "hccl_common.h"

namespace op_api {
HcclResult BuildSharedJettyMultiChannelHandle(HcclComm comm, const char *sharedQueueTag,
                                              const HcclChannelDesc *channelDescs, uint32_t channelNum,
                                              uint64_t *multiChannel);

namespace {
constexpr uint32_t MAX_RANKS = 1024;
constexpr uint32_t MAX_SHARED_JETTYS = 56;
constexpr int64_t WIN_STATE_BYTES = 1024 * 1024;
constexpr size_t HCCL_CONTEXT_TAG_MAX_LEN = 255;

// Wire layout matches SharedJettyContext in bandwidth_test_urma_shared_jetty.h.
struct SharedJettyContext {
    uint32_t epRankId = 0;
    uint32_t sharedJettyNum = 0;
    uint64_t bufferSize = 0;
    uint64_t epHcclBuffer_[MAX_RANKS] = {};
    uint64_t sharedJettyHandle_[MAX_RANKS] = {};
};

struct RankLinkInfo {
    CommProtocol protocol;
    uint32_t layer;
};

struct LayerRanks {
    uint32_t layer;
    std::vector<uint32_t> ranks;
};

struct SharedLayerInfo {
    uint32_t layer;
    size_t rankNum;
    EndpointDesc endpoint;
};

enum class SharedJettyCommLevel : int64_t {
    AUTO = 0,
    UB_CTP = 1,
    UB_RTP = 2,
};

void CheckHccl(HcclResult result, const char *operation)
{
    TORCH_CHECK(result == HCCL_SUCCESS, operation, " failed, ret=", result);
}

class ChannelConfigGuard {
public:
    ChannelConfigGuard()
    {
        auto ret = HcclChannelConfigCreateFunc(&config_);
        if (ret != HCCL_SUCCESS) {
            HcclResult cleanupRet = HCCL_SUCCESS;
            if (config_ != nullptr) {
                cleanupRet = HcclChannelConfigDestroyFunc(config_);
                config_ = nullptr;
            }
            TORCH_CHECK(false, "HcclChannelConfigCreate failed, ret=", ret,
                        "; HcclChannelConfigDestroy cleanup ret=", cleanupRet);
        }
    }

    ~ChannelConfigGuard()
    {
        if (config_ != nullptr) {
            HcclResult ret = HcclChannelConfigDestroyFunc(config_);
            if (ret != HCCL_SUCCESS) {
                ASCEND_LOGE("HcclChannelConfigDestroy cleanup failed, ret=%d", ret);
            }
        }
    }

    ChannelConfigGuard(const ChannelConfigGuard &) = delete;
    ChannelConfigGuard &operator=(const ChannelConfigGuard &) = delete;

    HcclChannelConfig Get() const
    {
        return config_;
    }

    HcclResult Destroy()
    {
        if (config_ == nullptr) {
            return HCCL_SUCCESS;
        }
        HcclResult ret = HcclChannelConfigDestroyFunc(config_);
        if (ret == HCCL_SUCCESS) {
            config_ = nullptr;
        }
        return ret;
    }

private:
    HcclChannelConfig config_ = nullptr;
};

void CopyContextToTensor(const SharedJettyContext &context, at::Tensor &tensor)
{
    at::Tensor hostContext = at::from_blob(const_cast<SharedJettyContext *>(&context),
                                           {sizeof(SharedJettyContext) / sizeof(int32_t)}, at::kInt);
    tensor.copy_(hostContext);
}
} // namespace

class SharedJettyContextBuilder {
public:
    SharedJettyContextBuilder() = default;
    SharedJettyContextBuilder(const SharedJettyContextBuilder &) = delete;
    SharedJettyContextBuilder &operator=(const SharedJettyContextBuilder &) = delete;

    void Build(const std::string &group, uint32_t worldSize, uint32_t sharedJettyNum, SharedJettyCommLevel commLevel,
               uint64_t bufferSize, const std::string &tag, void **localWindow, HcclMemHandle *windowMemHandle,
               at::Tensor &contextTensor)
    {
        ASCEND_LOGI("Start to build shared Jetty context, group: %s", group.c_str());
        InitHcclEngineCtxFunctions();
        InitHcclSharedChannelFunctions();

        worldSize_ = worldSize;
        commLevel_ = commLevel;
        tag_ = tag;
        localWindow_ = localWindow;
        windowMemHandle_ = windowMemHandle;
        TORCH_CHECK(localWindow_ != nullptr && windowMemHandle_ != nullptr,
                    "shared Jetty buffer owner pointers must not be null");
        ValidateTags(sharedJettyNum);
        AcquireHcclHandle(group);

        uint32_t actualWorldSize = 0;
        CheckHccl(HcclGetRankSizeFunc(comm_, &actualWorldSize), "HcclGetRankSize");
        CheckHccl(HcclGetRankIdFunc(comm_, &hostContext_.epRankId), "HcclGetRankId");
        TORCH_CHECK(actualWorldSize == worldSize && hostContext_.epRankId < actualWorldSize,
                    "Context rank or world size does not match the group");
        hostContext_.sharedJettyNum = sharedJettyNum;
        hostContext_.bufferSize = bufferSize;
        auto descriptors = BuildDescriptors();
        CreateDeviceContext();
        AllocateAndRegisterDeviceBuffer(bufferSize);
        AttachRegisteredWindow(descriptors);
        std::vector<ChannelHandle> remoteQueryChannels;
        BuildSharedJettyHandles(descriptors, remoteQueryChannels);
        CollectRegisteredWindows(remoteQueryChannels);
        CopyDeviceContext();
        CopyContextToTensor(hostContext_, contextTensor);
        ASCEND_LOGI("Build shared Jetty context success");
    }

private:
    void ValidateTags(uint32_t sharedJettyNum) const
    {
        const std::string windowTag = tag_ + "_win";
        const std::string longestQueueTag = tag_ + "_jetty_" + std::to_string(sharedJettyNum - 1);
        TORCH_CHECK(tag_.size() <= HCCL_CONTEXT_TAG_MAX_LEN, "Shared Jetty context tag is too long, max size is ",
                    HCCL_CONTEXT_TAG_MAX_LEN, ", got ", tag_.size());
        TORCH_CHECK(windowTag.size() <= HCCL_CONTEXT_TAG_MAX_LEN, "Shared Jetty Win tag is too long, max size is ",
                    HCCL_CONTEXT_TAG_MAX_LEN, ", got ", windowTag.size());
        TORCH_CHECK(longestQueueTag.size() <= HCCL_CONTEXT_TAG_MAX_LEN,
                    "Shared Jetty queue tag is too long, max size is ", HCCL_CONTEXT_TAG_MAX_LEN, ", got ",
                    longestQueueTag.size());
    }

    std::string WindowTag() const
    {
        return tag_ + "_win";
    }

    std::string QueueTag(uint32_t jetty) const
    {
        return tag_ + "_jetty_" + std::to_string(jetty);
    }

    void AcquireHcclHandle(const std::string &group)
    {
        auto hcclRet = HcomGetCommHandleByGroupFunc(group.c_str(), &comm_);
        TORCH_CHECK(hcclRet == HCCL_SUCCESS, "Get HCCL handle failed, group: ", group, ", ret: ", hcclRet);
        TORCH_CHECK(comm_ != nullptr, "HCCL communicator is null");
        ASCEND_LOGI("Get HCCL communication handle success hcclHandle is: %p", comm_);
    }

    void CreateDeviceContext()
    {
        // Match HcclChannelContextBuilder: this EngineCtx stores the operator context only.
        // Win registration and shared-Channel construction are independent resources below.
        void *deviceContext = nullptr;
        uint64_t contextSize = sizeof(SharedJettyContext);
        auto hcclRet = HcclEngineCtxCreateFunc(comm_, tag_.c_str(), COMM_ENGINE_AIV, contextSize, &deviceContext);
        TORCH_CHECK(hcclRet == HCCL_SUCCESS, "Create shared Jetty device context failed, ret: ", hcclRet);
        TORCH_CHECK(deviceContext != nullptr, "HCCL returned a null shared Jetty device context");
    }

    void CopyDeviceContext()
    {
        uint64_t contextSize = sizeof(SharedJettyContext);
        auto hcclRet = HcclEngineCtxCopyFunc(comm_, COMM_ENGINE_AIV, tag_.c_str(), &hostContext_, contextSize, 0);
        TORCH_CHECK(hcclRet == HCCL_SUCCESS, "Copy shared Jetty context from host to device failed, ret: ", hcclRet);
    }

    void AllocateAndRegisterDeviceBuffer(uint64_t bufferSize)
    {
        if (*localWindow_ == nullptr) {
            aclError aclRet = aclrtMalloc(localWindow_, bufferSize, ACL_MEM_MALLOC_HUGE_FIRST);
            TORCH_CHECK(aclRet == ACL_SUCCESS, "aclrtMalloc(", bufferSize, ") failed, ret=", aclRet);
            aclRet = aclrtMemset(*localWindow_, bufferSize, 0, bufferSize);
            if (aclRet != ACL_SUCCESS) {
                aclError freeRet = aclrtFree(*localWindow_);
                if (freeRet == ACL_SUCCESS) {
                    *localWindow_ = nullptr;
                }
                TORCH_CHECK(false, "aclrtMemset(shared Jetty buffer) failed, ret=", aclRet,
                            "; aclrtFree rollback ret=", freeRet);
            }
        }

        if (*windowMemHandle_ == nullptr) {
            CommMem mem{};
            mem.type = COMM_MEM_TYPE_DEVICE;
            mem.addr = *localWindow_;
            mem.size = bufferSize;
            const std::string windowTag = WindowTag();
            auto hcclRet = HcclCommMemRegFunc(comm_, windowTag.c_str(), &mem, windowMemHandle_);
            if (hcclRet != HCCL_SUCCESS) {
                *windowMemHandle_ = nullptr;
                aclError freeRet = aclrtFree(*localWindow_);
                if (freeRet == ACL_SUCCESS) {
                    *localWindow_ = nullptr;
                }
                TORCH_CHECK(false, "HcclCommMemReg(tag='", windowTag, "', size=", bufferSize, ") failed, ret=", hcclRet,
                            "; aclrtFree rollback ret=", freeRet);
            }
        }
    }

    void AttachRegisteredWindow(std::vector<HcclChannelDesc> &descriptors) const
    {
        TORCH_CHECK(*windowMemHandle_ != nullptr, "Shared Jetty Win is not registered");
        for (auto &descriptor : descriptors) {
            descriptor.memHandles = windowMemHandle_;
            descriptor.memHandleNum = 1;
        }
    }

    std::vector<ChannelHandle> AcquireSharedChannels(const std::vector<HcclChannelDesc> &descriptors, uint32_t jetty)
    {
        if (descriptors.empty()) {
            return {};
        }
        const uint32_t channelNum = static_cast<uint32_t>(descriptors.size());
        // The Torch ProcessGroup already owns an HcclComm, so use its standard wrapper; HCCL dispatches the
        // actual endpoint/channel creation to HcommChannelCreateWithConfig internally.
        ChannelConfigGuard config;
        CheckHccl(HcclChannelConfigSetIntFunc(config.Get(), HCCL_CHANNEL_CONFIG_TYPE_IS_SHARED_QUEUE, 1),
                  "HcclChannelConfigSetInt(IS_SHARED_QUEUE)");
        const std::string queueTag = QueueTag(jetty);
        // All peer descriptors in one call use one tag/Endpoint/Jetty; a new tag creates the next Jetty.
        CheckHccl(
            HcclChannelConfigSetStrFunc(config.Get(), HCCL_CHANNEL_CONFIG_TYPE_SHARED_QUEUE_TAG, queueTag.c_str()),
            "HcclChannelConfigSetStr(SHARED_QUEUE_TAG)");

        std::vector<ChannelHandle> channels(channelNum, 0);
        HcclResult acquireRet = HcclChannelAcquireWithConfigFunc(comm_, COMM_ENGINE_AIV, descriptors.data(), channelNum,
                                                                 config.Get(), channels.data());
        HcclResult destroyRet = config.Destroy();
        TORCH_CHECK(acquireRet == HCCL_SUCCESS, "HcclChannelAcquireWithConfig failed for shared Jetty ", jetty,
                    ", ret=", acquireRet, "; HcclChannelConfigDestroy ret=", destroyRet);
        CheckHccl(destroyRet, "HcclChannelConfigDestroy");
        for (uint32_t channel = 0; channel < channelNum; ++channel) {
            TORCH_CHECK(channels[channel] != 0,
                        "HcclChannelAcquireWithConfig returned a null channel for shared Jetty ", jetty, ", channel ",
                        channel);
        }
        return channels;
    }

    void BuildSharedJettyHandles(const std::vector<HcclChannelDesc> &descriptors,
                                 std::vector<ChannelHandle> &remoteQueryChannels)
    {
        if (descriptors.empty()) {
            return;
        }
        const uint32_t channelNum = static_cast<uint32_t>(descriptors.size());
        for (uint32_t jetty = 0; jetty < hostContext_.sharedJettyNum; ++jetty) {
            std::vector<ChannelHandle> channels = AcquireSharedChannels(descriptors, jetty);
            if (jetty == 0) {
                remoteQueryChannels = channels;
            }

            // HCCL performs the control-plane acquire above. asc-comm currently exposes no public
            // from-Channels adapter, so use its public helper only to materialize the Device multi-channel
            // entity. The helper's repeated acquire uses the same tag and reuses this shared-channel pool.
            uint64_t multiChannel = 0;
            const std::string queueTag = QueueTag(jetty);
            HcclResult ret = BuildSharedJettyMultiChannelHandle(comm_, queueTag.c_str(), descriptors.data(), channelNum,
                                                                &multiChannel);
            TORCH_CHECK(ret == HCCL_SUCCESS && multiChannel != 0, "MakeMultiChannelHandle failed for shared Jetty ",
                        jetty, ", ret=", ret);
            hostContext_.sharedJettyHandle_[jetty] = multiChannel;
        }
    }

    void CollectRegisteredWindows(const std::vector<ChannelHandle> &remoteQueryChannels)
    {
        hostContext_.epHcclBuffer_[hostContext_.epRankId] = reinterpret_cast<uint64_t>(*localWindow_);
        const std::string targetTag = WindowTag();
        TORCH_CHECK(worldSize_ == 1 || remoteQueryChannels.size() == worldSize_ - 1,
                    "Missing standard channels for remote Win lookup");
        // Every Jetty carries the same registered Win, so the first standard acquire is sufficient for lookup.
        for (uint32_t peer = 0; peer < worldSize_; ++peer) {
            if (peer == hostContext_.epRankId) {
                continue;
            }
            uint32_t channelIndex = peer < hostContext_.epRankId ? peer : peer - 1;
            uint32_t memNum = 0;
            CommMem *remoteMems = nullptr;
            char **memTags = nullptr;
            auto ret =
                HcclChannelGetRemoteMemsFunc(comm_, remoteQueryChannels[channelIndex], &memNum, &remoteMems, &memTags);
            TORCH_CHECK(ret == HCCL_SUCCESS, "HcclChannelGetRemoteMems(peer=", peer, ") failed, ret=", ret);

            bool found = false;
            if (remoteMems != nullptr && memTags != nullptr) {
                for (uint32_t memIndex = 0; memIndex < memNum; ++memIndex) {
                    if (memTags[memIndex] == nullptr || targetTag != memTags[memIndex]) {
                        continue;
                    }
                    TORCH_CHECK(remoteMems[memIndex].type == COMM_MEM_TYPE_DEVICE,
                                "Remote Win is not device memory for rank ", peer);
                    TORCH_CHECK(remoteMems[memIndex].addr != nullptr, "Remote Win is null for rank ", peer);
                    TORCH_CHECK(remoteMems[memIndex].size >= hostContext_.bufferSize,
                                "Remote Win is too small for rank ", peer, ": required ", hostContext_.bufferSize,
                                ", got ", remoteMems[memIndex].size);
                    hostContext_.epHcclBuffer_[peer] = reinterpret_cast<uint64_t>(remoteMems[memIndex].addr);
                    found = true;
                    break;
                }
            }
            TORCH_CHECK(found, "Registered Win tag '", targetTag, "' is not found for rank ", peer);
        }
    }

    CommLink *FindProtocolLink(uint32_t layer, uint32_t peer, CommProtocol protocol) const
    {
        CommLink *links = nullptr;
        uint32_t linkNum = 0;
        auto ret = HcclRankGraphGetLinksFunc(comm_, layer, hostContext_.epRankId, peer, &links, &linkNum);
        if (ret != HCCL_SUCCESS || links == nullptr) {
            return nullptr;
        }
        for (uint32_t index = 0; index < linkNum; ++index) {
            if (links[index].linkAttr.linkProtocol == protocol) {
                return &links[index];
            }
        }
        return nullptr;
    }

    LayerRanks GetLayerRanks(uint32_t layer) const
    {
        uint32_t *ranks = nullptr;
        uint32_t rankNum = 0;
        CheckHccl(HcclRankGraphGetRanksByLayerFunc(comm_, layer, &ranks, &rankNum), "HcclRankGraphGetRanksByLayer");
        TORCH_CHECK(ranks != nullptr && rankNum > 0, "Layer ", layer, " returned no ranks");
        std::vector<uint32_t> rankList(ranks, ranks + rankNum);
        for (uint32_t rank : rankList) {
            TORCH_CHECK(rank < worldSize_, "Layer ", layer, " contains invalid rank ", rank);
        }
        TORCH_CHECK(std::find(rankList.begin(), rankList.end(), hostContext_.epRankId) != rankList.end(), "Layer ",
                    layer, " does not contain local rank ", hostContext_.epRankId);
        return {layer, std::move(rankList)};
    }

    std::vector<LayerRanks> GetAllLayerRanks(const uint32_t *layers, uint32_t layerNum) const
    {
        std::vector<LayerRanks> layerRanks;
        layerRanks.reserve(layerNum);
        for (uint32_t layerIndex = 0; layerIndex < layerNum; ++layerIndex) {
            layerRanks.push_back(GetLayerRanks(layers[layerIndex]));
        }
        return layerRanks;
    }

    LayerRanks FindUbDomain(const std::vector<LayerRanks> &layers) const
    {
        LayerRanks ubDomain{0, {hostContext_.epRankId}};
        std::vector<bool> resolved(worldSize_, false);
        resolved[hostContext_.epRankId] = true;
        // Match the ordinary context: each layer only checks peers not covered by an inner UB_CTP layer.
        for (const auto &layer : layers) {
            bool allNewPeersSupportCtp = true;
            for (uint32_t peer : layer.ranks) {
                if (!resolved[peer] && FindProtocolLink(layer.layer, peer, COMM_PROTOCOL_UB_CTP) == nullptr) {
                    allNewPeersSupportCtp = false;
                    break;
                }
            }
            if (!allNewPeersSupportCtp) {
                break;
            }
            for (uint32_t peer : layer.ranks) {
                resolved[peer] = true;
            }
            ubDomain = layer;
        }
        return ubDomain;
    }

    std::vector<uint32_t> RemoteRanks(const LayerRanks &layer) const
    {
        std::vector<uint32_t> peers;
        for (uint32_t rank : layer.ranks) {
            if (rank != hostContext_.epRankId) {
                peers.push_back(rank);
            }
        }
        return peers;
    }

    std::vector<uint32_t> AllRemoteRanks() const
    {
        std::vector<uint32_t> peers;
        for (uint32_t peer = 0; peer < worldSize_; ++peer) {
            if (peer != hostContext_.epRankId) {
                peers.push_back(peer);
            }
        }
        return peers;
    }

    static bool ContainsRanks(const LayerRanks &layer, const std::vector<uint32_t> &peers)
    {
        for (uint32_t peer : peers) {
            if (std::find(layer.ranks.begin(), layer.ranks.end(), peer) == layer.ranks.end()) {
                return false;
            }
        }
        return true;
    }

    static bool SameEndpoint(const EndpointDesc &lhs, const EndpointDesc &rhs)
    {
        return lhs.protocol == rhs.protocol && lhs.commAddr.type == rhs.commAddr.type &&
               std::memcmp(lhs.commAddr.raws, rhs.commAddr.raws, sizeof(lhs.commAddr.raws)) == 0 &&
               lhs.loc.locType == rhs.loc.locType && std::memcmp(lhs.loc.raws, rhs.loc.raws, sizeof(lhs.loc.raws)) == 0;
    }

    bool ResolveLayerEndpoint(const LayerRanks &layer, const std::vector<uint32_t> &peers, CommProtocol protocol,
                              EndpointDesc &resolvedEndpoint) const
    {
        if (!ContainsRanks(layer, peers)) {
            return false;
        }
        bool hasEndpoint = false;
        for (uint32_t peer : peers) {
            CommLink *link = FindProtocolLink(layer.layer, peer, protocol);
            if (link == nullptr) {
                return false;
            }
            if (!hasEndpoint) {
                resolvedEndpoint = link->srcEndpointDesc;
                hasEndpoint = true;
            } else if (!SameEndpoint(resolvedEndpoint, link->srcEndpointDesc)) {
                return false;
            }
        }
        return hasEndpoint;
    }

    std::vector<SharedLayerInfo> FindSharedLayers(const std::vector<LayerRanks> &layers,
                                                  const std::vector<uint32_t> &peers, CommProtocol protocol) const
    {
        std::vector<SharedLayerInfo> candidates;
        for (const auto &layer : layers) {
            EndpointDesc candidateEndpoint{};
            if (!ResolveLayerEndpoint(layer, peers, protocol, candidateEndpoint)) {
                continue;
            }
            candidates.push_back({layer.layer, layer.ranks.size(), candidateEndpoint});
        }
        return candidates;
    }

    SharedLayerInfo FindSharedLayer(const std::vector<LayerRanks> &layers, const std::vector<uint32_t> &peers,
                                    CommProtocol protocol) const
    {
        auto candidates = FindSharedLayers(layers, peers, protocol);
        TORCH_CHECK(!candidates.empty(), "No layer can build one shared Jetty with protocol ",
                    static_cast<int>(protocol), " for all selected peers");
        return *std::max_element(
            candidates.begin(), candidates.end(),
            [](const SharedLayerInfo &lhs, const SharedLayerInfo &rhs) { return lhs.rankNum < rhs.rankNum; });
    }

    void RecordRanks(const std::vector<uint32_t> &peers, CommProtocol protocol, uint32_t layer)
    {
        for (uint32_t peer : peers) {
            rankLinkMap_[peer] = {protocol, layer};
        }
    }

    void ResolveAutoLinks(const std::vector<LayerRanks> &layers)
    {
        LayerRanks ubDomain = FindUbDomain(layers);
        std::vector<uint32_t> ubPeers = RemoteRanks(ubDomain);
        std::vector<bool> isUbRank(worldSize_, false);
        for (uint32_t rank : ubDomain.ranks) {
            isUbRank[rank] = true;
        }
        std::vector<uint32_t> crossPeers;
        for (uint32_t peer = 0; peer < worldSize_; ++peer) {
            if (peer != hostContext_.epRankId && !isUbRank[peer]) {
                crossPeers.push_back(peer);
            }
        }

        if (!ubPeers.empty() && !crossPeers.empty()) {
            auto ubLayers = FindSharedLayers(layers, ubPeers, COMM_PROTOCOL_UB_CTP);
            auto crossLayers = FindSharedLayers(layers, crossPeers, COMM_PROTOCOL_UB_RTP);
            bool found = false;
            SharedLayerInfo ubSelection{};
            SharedLayerInfo crossSelection{};
            for (const auto &ubLayer : ubLayers) {
                for (const auto &crossLayer : crossLayers) {
                    if (!SameEndpoint(ubLayer.endpoint, crossLayer.endpoint)) {
                        continue;
                    }
                    if (!found || ubLayer.rankNum > ubSelection.rankNum ||
                        (ubLayer.rankNum == ubSelection.rankNum && crossLayer.rankNum >= crossSelection.rankNum)) {
                        ubSelection = ubLayer;
                        crossSelection = crossLayer;
                        found = true;
                    }
                }
            }
            TORCH_CHECK(found, "No compatible UB_CTP and UB_RTP layers can build one shared Jetty");
            RecordRanks(ubPeers, COMM_PROTOCOL_UB_CTP, ubSelection.layer);
            RecordRanks(crossPeers, COMM_PROTOCOL_UB_RTP, crossSelection.layer);
        } else if (!ubPeers.empty()) {
            auto ubLayer = FindSharedLayer(layers, ubPeers, COMM_PROTOCOL_UB_CTP);
            RecordRanks(ubPeers, COMM_PROTOCOL_UB_CTP, ubLayer.layer);
        } else if (!crossPeers.empty()) {
            auto crossLayer = FindSharedLayer(layers, crossPeers, COMM_PROTOCOL_UB_RTP);
            RecordRanks(crossPeers, COMM_PROTOCOL_UB_RTP, crossLayer.layer);
        }
    }

    void ResolveUniformLinks(const std::vector<LayerRanks> &layers, CommProtocol protocol)
    {
        std::vector<uint32_t> peers = AllRemoteRanks();
        auto layer = FindSharedLayer(layers, peers, protocol);
        RecordRanks(peers, protocol, layer.layer);
    }

    void ResolveProtocolLayers(const std::vector<LayerRanks> &layers)
    {
        rankLinkMap_.clear();
        if (commLevel_ == SharedJettyCommLevel::AUTO) {
            ResolveAutoLinks(layers);
        } else if (commLevel_ == SharedJettyCommLevel::UB_CTP) {
            ResolveUniformLinks(layers, COMM_PROTOCOL_UB_CTP);
        } else {
            ResolveUniformLinks(layers, COMM_PROTOCOL_UB_RTP);
        }
        TORCH_CHECK(rankLinkMap_.size() == worldSize_ - 1, "Incomplete shared Jetty topology: resolved ",
                    rankLinkMap_.size(), " of ", worldSize_ - 1, " remote ranks");
    }

    CommLink *GetRecordedLink(uint32_t peer, const RankLinkInfo &linkInfo) const
    {
        CommLink *links = nullptr;
        uint32_t linkNum = 0;
        CheckHccl(HcclRankGraphGetLinksFunc(comm_, linkInfo.layer, hostContext_.epRankId, peer, &links, &linkNum),
                  "HcclRankGraphGetLinks");
        TORCH_CHECK(links != nullptr && linkNum > 0, "No HCCL link from rank ", hostContext_.epRankId, " to rank ",
                    peer, " on recorded layer ", linkInfo.layer);
        for (uint32_t index = 0; index < linkNum; ++index) {
            if (links[index].linkAttr.linkProtocol == linkInfo.protocol) {
                return &links[index];
            }
        }
        TORCH_CHECK(false, "Recorded protocol ", static_cast<int>(linkInfo.protocol), " is unavailable from rank ",
                    hostContext_.epRankId, " to rank ", peer, " on layer ", linkInfo.layer);
        return nullptr;
    }

    std::vector<HcclChannelDesc> BuildDescriptors()
    {
        std::vector<HcclChannelDesc> descriptors(worldSize_ - 1);
        if (descriptors.empty()) {
            return descriptors;
        }
        uint32_t *layers = nullptr;
        uint32_t layerNum = 0;
        CheckHccl(HcclRankGraphGetLayersFunc(comm_, &layers, &layerNum), "HcclRankGraphGetLayers");
        TORCH_CHECK(layers != nullptr && layerNum > 0, "HCCL returned no network layers");
        auto layerRanks = GetAllLayerRanks(layers, layerNum);
        ResolveProtocolLayers(layerRanks);
        CheckHccl(HcclChannelDescInit(descriptors.data(), descriptors.size()), "HcclChannelDescInit");
        EndpointDesc localEndpoint{};
        bool hasLocalEndpoint = false;
        for (uint32_t peer = 0; peer < worldSize_; ++peer) {
            if (peer == hostContext_.epRankId) {
                continue;
            }
            uint32_t index = peer < hostContext_.epRankId ? peer : peer - 1;
            const auto &linkInfo = rankLinkMap_.at(peer);
            CommLink *link = GetRecordedLink(peer, linkInfo);
            if (!hasLocalEndpoint) {
                localEndpoint = link->srcEndpointDesc;
                hasLocalEndpoint = true;
            } else {
                TORCH_CHECK(SameEndpoint(localEndpoint, link->srcEndpointDesc),
                            "Shared jetty peers do not use one local endpoint; this environment does not support ",
                            "one Jetty connecting every peer");
            }
            auto &desc = descriptors[index];
            desc.remoteRank = peer;
            desc.channelProtocol = linkInfo.protocol;
            desc.localEndpoint = link->srcEndpointDesc;
            desc.remoteEndpoint = link->dstEndpointDesc;
        }
        return descriptors;
    }

    std::string tag_;
    uint32_t worldSize_ = 0;
    HcclComm comm_ = nullptr;
    void **localWindow_ = nullptr;
    HcclMemHandle *windowMemHandle_ = nullptr;
    SharedJettyCommLevel commLevel_ = SharedJettyCommLevel::AUTO;
    std::unordered_map<uint32_t, RankLinkInfo> rankLinkMap_;
    SharedJettyContext hostContext_{};
};

class SharedJettyContextManager {
public:
    SharedJettyContextManager(const std::string &group, int64_t worldSize, int64_t sharedJettyNum,
                              int64_t sharedJettyCommLevel, int64_t bufferSize, const std::string &tag)
        : group_(group),
          tag_(tag),
          worldSize_(worldSize),
          sharedJettyNum_(sharedJettyNum),
          bufferSize_(bufferSize)
    {
        TORCH_CHECK(worldSize > 0 && worldSize <= MAX_RANKS, "world_size must be in [1, 1024]");
        TORCH_CHECK(sharedJettyNum > 0 && sharedJettyNum <= MAX_SHARED_JETTYS, "shared_jetty_num must be in [1, 56]");
        // Matches STATE_SIZE in moe_distribute_v2_constant.h; kernel data starts after this state area.
        TORCH_CHECK(bufferSize > WIN_STATE_BYTES, "shared jetty Win must exceed the 1 MiB state area");
        TORCH_CHECK(!tag.empty(), "shared jetty context tag must not be empty");
        TORCH_CHECK(sharedJettyCommLevel >= static_cast<int64_t>(SharedJettyCommLevel::AUTO) &&
                        sharedJettyCommLevel <= static_cast<int64_t>(SharedJettyCommLevel::UB_RTP),
                    "shared_jetty_comm_level must be 0, 1 or 2");
        commLevel_ = static_cast<SharedJettyCommLevel>(sharedJettyCommLevel);
    }

    ~SharedJettyContextManager()
    {
        try {
            Destroy();
        } catch (const std::exception &error) {
            ASCEND_LOGE("SharedJettyContextManager destroy failed: %s", error.what());
        }
    }
    SharedJettyContextManager(const SharedJettyContextManager &) = delete;
    SharedJettyContextManager &operator=(const SharedJettyContextManager &) = delete;

    at::Tensor CreateContext()
    {
        if (context_.defined()) {
            return context_;
        }
        TORCH_CHECK(!creationStarted_, "Shared jetty construction failed; recreate the communication group");

        at::Tensor context = at::empty({ContextTensorSize()}, at::TensorOptions()
                                                                  .dtype(at::kInt)
                                                                  .device(c10::DeviceType::PrivateUse1)
                                                                  .memory_format(c10::MemoryFormat::Contiguous));
        creationStarted_ = true;
        builder_.Build(group_, static_cast<uint32_t>(worldSize_), static_cast<uint32_t>(sharedJettyNum_), commLevel_,
                       static_cast<uint64_t>(bufferSize_), tag_, &localWindow_, &windowMemHandle_, context);
        context_ = context;
        return context_;
    }

    void AbandonRegisteredWindow()
    {
        // Used only when ProcessGroup destruction itself fails after construction failed. Keeping the
        // allocation alive until process exit is safer than freeing memory still registered in HCCL.
        if (windowMemHandle_ != nullptr || localWindow_ != nullptr) {
            ASCEND_LOGE("Abandon shared Jetty Win because its communicator could not be destroyed");
        }
        windowMemHandle_ = nullptr;
        localWindow_ = nullptr;
    }

private:
    static int64_t ContextTensorSize()
    {
        return (sizeof(SharedJettyContext) + sizeof(int32_t) - 1) / sizeof(int32_t);
    }

    void Destroy()
    {
        if (windowMemHandle_ != nullptr || localWindow_ != nullptr) {
            aclError aclRet = aclrtSynchronizeDevice();
            TORCH_CHECK(aclRet == ACL_SUCCESS, "aclrtSynchronizeDevice failed, ret=", aclRet);
        }
        if (windowMemHandle_ != nullptr) {
            // The public HCCL API has no matching deregistration call. The Python owner must keep the
            // communicator alive until all kernels finish, then destroy it before this Win is released.
            windowMemHandle_ = nullptr;
        }
        if (localWindow_ != nullptr) {
            aclError aclRet = aclrtFree(localWindow_);
            TORCH_CHECK(aclRet == ACL_SUCCESS, "aclrtFree(shared Jetty buffer) failed, ret=", aclRet);
            localWindow_ = nullptr;
        }
    }

    std::string group_;
    std::string tag_;
    int64_t worldSize_;
    int64_t sharedJettyNum_;
    int64_t bufferSize_;
    SharedJettyCommLevel commLevel_ = SharedJettyCommLevel::AUTO;
    void *localWindow_ = nullptr;
    HcclMemHandle windowMemHandle_ = nullptr;
    SharedJettyContextBuilder builder_;
    at::Tensor context_;
    bool creationStarted_ = false;
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    py::class_<SharedJettyContextManager>(m, "SharedJettyContextManager")
        .def(py::init<const std::string &, int64_t, int64_t, int64_t, int64_t, const std::string &>())
        .def("create_context", &SharedJettyContextManager::CreateContext)
        .def("_abandon_registered_window", &SharedJettyContextManager::AbandonRegisteredWindow);
}
} // namespace op_api
