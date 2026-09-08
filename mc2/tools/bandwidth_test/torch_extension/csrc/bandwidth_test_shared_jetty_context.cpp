/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
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
#include "hcomm/hcomm_host.h"

namespace op_api {
namespace {
constexpr uint32_t MAX_RANKS = 1024;
constexpr int64_t MAX_SHARED_JETTYS = 56;
constexpr int64_t WIN_STATE_BYTES = 1024 * 1024;
constexpr size_t HCCL_CONTEXT_TAG_MAX_LEN = 255;

// Wire layout matches SharedJettyContext in bandwidth_test_urma_shared_jetty.h.
// Keep the ordinary context's tensor size so existing graph shape contracts remain valid.
struct SharedJettyContext {
    uint32_t epRankId = 0;
    uint32_t sharedJettyNum = 0;
    uint64_t bufferSize = 0;
    uint64_t epHcclBuffer_[MAX_RANKS] = {};
    uint64_t sharedJettyHandle_[MAX_RANKS] = {};
};
static_assert(sizeof(SharedJettyContext) == 16400, "Shared jetty context wire layout changed");
static_assert(offsetof(SharedJettyContext, epRankId) == 0 &&
                  offsetof(SharedJettyContext, sharedJettyNum) == 4 &&
                  offsetof(SharedJettyContext, bufferSize) == 8 &&
                  offsetof(SharedJettyContext, epHcclBuffer_) == 16 &&
                  offsetof(SharedJettyContext, sharedJettyHandle_) == 8208,
              "Shared jetty context field offsets changed");

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

void CopyContextToTensor(const SharedJettyContext &context, at::Tensor &tensor)
{
    at::Tensor hostContext = at::from_blob(const_cast<SharedJettyContext *>(&context),
                                           {sizeof(SharedJettyContext) / sizeof(int32_t)}, at::kInt);
    tensor.copy_(hostContext);
}
} // namespace

// Preparation and context construction are separate because Python exchanges the registered Win addresses
// between the two phases. HcclChannelContextBuilder does not need this split because it queries remote memory
// from acquired channels.
class SharedJettyContextBuilder {
public:
    SharedJettyContextBuilder() = default;
    SharedJettyContextBuilder(const SharedJettyContextBuilder &) = delete;
    SharedJettyContextBuilder &operator=(const SharedJettyContextBuilder &) = delete;

    void PrepareLocalResources(const std::string &group, uint32_t worldSize, uint32_t sharedJettyNum,
                               SharedJettyCommLevel commLevel, uint64_t bufferSize, const std::string &tag,
                               void **localWindow, HcclMemHandle *windowMemHandle)
    {
        ASCEND_LOGI("Start to prepare shared Jetty resources, group: %s", group.c_str());
        InitHcclEngineCtxFunctions();

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
        CreateDeviceContext();
        AllocateAndRegisterDeviceBuffer(bufferSize);
    }

    void ValidateRemoteBuffers(const std::vector<uint64_t> &remoteBuffers) const
    {
        TORCH_CHECK(remoteBuffers.size() == worldSize_, "Missing remote Win addresses");
        TORCH_CHECK(remoteBuffers[hostContext_.epRankId] == reinterpret_cast<uint64_t>(*localWindow_),
                    "Local Win address mismatch");
        for (uint32_t rank = 0; rank < worldSize_; ++rank) {
            TORCH_CHECK(remoteBuffers[rank] != 0, "Null Win for rank ", rank);
        }
    }

    void BuildSharedContext(const std::vector<uint64_t> &remoteBuffers, at::Tensor &contextTensor)
    {
        std::copy(remoteBuffers.begin(), remoteBuffers.end(), hostContext_.epHcclBuffer_);
        auto descriptors = BuildDescriptors();
        if (!descriptors.empty()) {
            for (uint32_t jetty = 0; jetty < hostContext_.sharedJettyNum; ++jetty) {
                const std::string queueTag = tag_ + "_rank_" + std::to_string(hostContext_.epRankId) +
                    "_jetty_" + std::to_string(jetty);
                MultiChannelHandle handle = 0;
                auto ret = MakeMultiChannelHandle(comm_, queueTag.c_str(), descriptors.data(),
                                                  descriptors.size(), &handle);
                TORCH_CHECK(ret == HCCL_SUCCESS && handle != 0,
                            "MakeMultiChannelHandle failed for shared jetty ", jetty, ", ret=", ret,
                            "; selected peer endpoints/protocols must support one HCCL shared queue");
                hostContext_.sharedJettyHandle_[jetty] = handle;
            }
        }
        CopyDeviceContext();
        CopyContextToTensor(hostContext_, contextTensor);
        ASCEND_LOGI("Build shared Jetty context success");
    }

private:
    void ValidateTags(uint32_t sharedJettyNum) const
    {
        const std::string windowTag = tag_ + "_win";
        const std::string longestQueueTag = tag_ + "_rank_" + std::to_string(worldSize_ - 1) + "_jetty_" +
            std::to_string(sharedJettyNum - 1);
        TORCH_CHECK(tag_.size() <= HCCL_CONTEXT_TAG_MAX_LEN, "Shared Jetty context tag is too long, max size is ",
                    HCCL_CONTEXT_TAG_MAX_LEN, ", got ", tag_.size());
        TORCH_CHECK(windowTag.size() <= HCCL_CONTEXT_TAG_MAX_LEN, "Shared Jetty Win tag is too long, max size is ",
                    HCCL_CONTEXT_TAG_MAX_LEN, ", got ", windowTag.size());
        TORCH_CHECK(longestQueueTag.size() <= HCCL_CONTEXT_TAG_MAX_LEN,
                    "Shared Jetty queue tag is too long, max size is ", HCCL_CONTEXT_TAG_MAX_LEN, ", got ",
                    longestQueueTag.size());
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
        void *context = nullptr;
        uint64_t contextSize = sizeof(SharedJettyContext);
        auto hcclRet = HcclEngineCtxCreateFunc(comm_, tag_.c_str(), COMM_ENGINE_AIV, contextSize, &context);
        TORCH_CHECK(hcclRet == HCCL_SUCCESS, "Create shared Jetty device context failed, ret: ", hcclRet);
        TORCH_CHECK(context != nullptr, "HCCL returned a null shared Jetty device context");
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
            const std::string windowTag = tag_ + "_win";
            auto hcclRet = HcclCommMemRegFunc(comm_, windowTag.c_str(), &mem, windowMemHandle_);
            if (hcclRet != HCCL_SUCCESS) {
                *windowMemHandle_ = nullptr;
                aclError freeRet = aclrtFree(*localWindow_);
                if (freeRet == ACL_SUCCESS) {
                    *localWindow_ = nullptr;
                }
                TORCH_CHECK(false, "HcclCommMemReg(tag='", windowTag, "', size=", bufferSize,
                            ") failed, ret=", hcclRet, "; aclrtFree rollback ret=", freeRet);
            }
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
        CheckHccl(HcclRankGraphGetRanksByLayerFunc(comm_, layer, &ranks, &rankNum),
                  "HcclRankGraphGetRanksByLayer");
        TORCH_CHECK(ranks != nullptr && rankNum > 0, "Layer ", layer, " returned no ranks");
        std::vector<uint32_t> rankList(ranks, ranks + rankNum);
        for (uint32_t rank : rankList) {
            TORCH_CHECK(rank < worldSize_, "Layer ", layer, " contains invalid rank ", rank);
        }
        TORCH_CHECK(std::find(rankList.begin(), rankList.end(), hostContext_.epRankId) != rankList.end(),
                    "Layer ", layer, " does not contain local rank ", hostContext_.epRankId);
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
        return std::memcmp(lhs.commAddr.raws, rhs.commAddr.raws, sizeof(lhs.commAddr.raws)) == 0;
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
                                                  const std::vector<uint32_t> &peers,
                                                  CommProtocol protocol) const
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
                    static_cast<int>(protocol),
                    " for all selected peers");
        return *std::max_element(candidates.begin(), candidates.end(),
                                 [](const SharedLayerInfo &lhs, const SharedLayerInfo &rhs) {
                                     return lhs.rankNum < rhs.rankNum;
                                 });
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
        TORCH_CHECK(links != nullptr && linkNum > 0, "No HCCL link from rank ", hostContext_.epRankId,
                    " to rank ", peer, " on recorded layer ", linkInfo.layer);
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
            desc.memHandles = windowMemHandle_;
            desc.memHandleNum = 1;
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
        TORCH_CHECK(sharedJettyNum > 0 && sharedJettyNum <= MAX_SHARED_JETTYS,
                    "shared_jetty_num must be in [1, 56]");
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

    void Prepare()
    {
        EnsurePrepared();
    }

    uint64_t LocalBufferAddress() const
    {
        TORCH_CHECK(prepared_, "Shared Jetty resources are not prepared");
        return reinterpret_cast<uint64_t>(localWindow_);
    }

    at::Tensor CreateContext(const std::vector<uint64_t> &remoteBuffers)
    {
        EnsurePrepared();
        if (context_.defined()) {
            TORCH_CHECK(remoteBuffers == remoteBuffers_,
                        "Cannot replace Win addresses on a live shared jetty context");
            return context_;
        }
        TORCH_CHECK(!creationStarted_, "Shared jetty construction failed; recreate the communication group");
        builder_.ValidateRemoteBuffers(remoteBuffers);

        at::Tensor context = at::empty({ContextTensorSize()}, at::TensorOptions()
                                                                  .dtype(at::kInt)
                                                                  .device(c10::DeviceType::PrivateUse1)
                                                                  .memory_format(c10::MemoryFormat::Contiguous));
        creationStarted_ = true;
        builder_.BuildSharedContext(remoteBuffers, context);
        remoteBuffers_ = remoteBuffers;
        context_ = context;
        return context_;
    }

private:
    static int64_t ContextTensorSize()
    {
        return (sizeof(SharedJettyContext) + sizeof(int32_t) - 1) / sizeof(int32_t);
    }

    void EnsurePrepared()
    {
        if (prepared_) {
            return;
        }
        TORCH_CHECK(!preparationStarted_, "Shared Jetty preparation failed; recreate the communication group");
        preparationStarted_ = true;
        builder_.PrepareLocalResources(group_, static_cast<uint32_t>(worldSize_),
                                       static_cast<uint32_t>(sharedJettyNum_), commLevel_,
                                       static_cast<uint64_t>(bufferSize_), tag_, &localWindow_, &windowMemHandle_);
        prepared_ = true;
    }

    void Destroy()
    {
        if (windowMemHandle_ != nullptr || localWindow_ != nullptr) {
            aclError aclRet = aclrtSynchronizeDevice();
            TORCH_CHECK(aclRet == ACL_SUCCESS, "aclrtSynchronizeDevice failed, ret=", aclRet);
        }
        if (windowMemHandle_ != nullptr) {
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
    std::vector<uint64_t> remoteBuffers_;
    bool preparationStarted_ = false;
    bool prepared_ = false;
    bool creationStarted_ = false;
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    py::class_<SharedJettyContextManager>(m, "SharedJettyContextManager")
        .def(py::init<const std::string &, int64_t, int64_t, int64_t, int64_t, const std::string &>())
        .def("prepare", &SharedJettyContextManager::Prepare)
        .def("create_context", &SharedJettyContextManager::CreateContext)
        .def_property_readonly("local_buffer_address", &SharedJettyContextManager::LocalBufferAddress);
}
} // namespace op_api
