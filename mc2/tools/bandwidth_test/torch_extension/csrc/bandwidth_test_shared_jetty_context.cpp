/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <torch/extension.h>
#include "hccl_common.h"
#include "hcomm/hcomm_host.h"

namespace op_api {
namespace {
constexpr uint32_t MAX_RANKS = 1024;

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

void CheckHccl(HcclResult result, const char *operation)
{
    TORCH_CHECK(result == HCCL_SUCCESS, operation, " failed, ret=", result);
}
} // namespace

class SharedJettyContextManager {
public:
    SharedJettyContextManager(const std::string &group, int64_t worldSize, int64_t sharedJettyNum,
                              int64_t bufferSize, const std::string &tag)
        : tag_(tag), worldSize_(worldSize)
    {
        TORCH_CHECK(worldSize > 0 && worldSize <= MAX_RANKS, "world_size must be in [1, 1024]");
        TORCH_CHECK(bufferSize > 1024 * 1024, "shared jetty Win must exceed the 1 MiB state area");
        TORCH_CHECK(!tag.empty() && tag.size() < 200, "invalid shared jetty context tag");
        const char *soc = aclrtGetSocName();
        TORCH_CHECK(soc != nullptr && std::strstr(soc, "Ascend950") != nullptr,
                    "BandwidthTest shared jetty requires Ascend950");
        TORCH_CHECK(aclrtGetDevice(&device_) == ACL_SUCCESS, "aclrtGetDevice failed");
        InitHcclEngineCtxFunctions();
        CheckHccl(HcomGetCommHandleByGroupFunc(group.c_str(), &comm_), "HcomGetCommHandleByGroup");
        TORCH_CHECK(comm_ != nullptr, "HCCL communicator is null");
        uint32_t actualWorldSize = 0;
        CheckHccl(HcclGetRankSizeFunc(comm_, &actualWorldSize), "HcclGetRankSize");
        CheckHccl(HcclGetRankIdFunc(comm_, &hostContext_.epRankId), "HcclGetRankId");
        TORCH_CHECK(actualWorldSize == worldSize && hostContext_.epRankId < actualWorldSize,
                    "Context rank or world size does not match the group");
        hostContext_.sharedJettyNum = sharedJettyNum;
        hostContext_.bufferSize = bufferSize;
        // HCCL releases engine memory and registered resources with the communicator,
        // including partial construction failures. Never free memory behind live channels.
        const std::string winTag = tag_ + "_win";
        CheckHccl(HcclEngineCtxCreateFunc(comm_, winTag.c_str(), COMM_ENGINE_AIV, bufferSize, &localBuffer_),
                  "HcclEngineCtxCreate(Win)");
        TORCH_CHECK(localBuffer_ != nullptr, "HCCL returned a null Win");
        TORCH_CHECK(aclrtMemset(localBuffer_, bufferSize, 0, bufferSize) == ACL_SUCCESS, "Win initialization failed");
        CommMem mem{};
        mem.type = COMM_MEM_TYPE_DEVICE;
        mem.addr = localBuffer_;
        mem.size = bufferSize;
        CheckHccl(HcclCommMemRegFunc(comm_, winTag.c_str(), &mem, &memHandle_), "HcclCommMemReg(Win)");
    }

    uint64_t LocalBufferAddress() const
    {
        return reinterpret_cast<uint64_t>(localBuffer_);
    }

    at::Tensor CreateContext(const std::vector<uint64_t> &remoteBuffers)
    {
        const c10::OptionalDeviceGuard guard(c10::Device(c10::DeviceType::PrivateUse1, device_));
        TORCH_CHECK(remoteBuffers.size() == worldSize_, "Missing remote Win addresses");
        TORCH_CHECK(remoteBuffers[hostContext_.epRankId] == LocalBufferAddress(), "Local Win address mismatch");
        for (uint32_t rank = 0; rank < worldSize_; ++rank) {
            TORCH_CHECK(remoteBuffers[rank] != 0, "Null Win for rank ", rank);
        }
        if (context_.defined()) {
            TORCH_CHECK(std::equal(remoteBuffers.begin(), remoteBuffers.end(), hostContext_.epHcclBuffer_),
                        "Cannot replace Win addresses on a live shared jetty context");
            return context_;
        }
        TORCH_CHECK(!creationStarted_, "Shared jetty construction failed; recreate the communication group");
        creationStarted_ = true;
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
        auto host = at::from_blob(&hostContext_, {sizeof(SharedJettyContext) / sizeof(int32_t)}, at::kInt);
        context_ = at::empty_like(host, host.options().device(c10::Device(c10::DeviceType::PrivateUse1, device_)));
        context_.copy_(host);
        return context_;
    }

private:
    std::vector<CommLink> FindUrmaLinks(uint32_t peer, const uint32_t *layers, uint32_t layerNum) const
    {
        // UBC_CTP (also called UB_CTP) stays within a supernode; UB_RTP crosses supernodes.
        // Search actual layer IDs, including single-layer topologies whose ID is not zero.
        for (auto protocol : {COMM_PROTOCOL_UBC_CTP, COMM_PROTOCOL_UB_RTP}) {
            std::vector<CommLink> candidates;
            for (uint32_t i = 0; i < layerNum; ++i) {
                CommLink *links = nullptr;
                uint32_t linkNum = 0;
                auto ret = HcclRankGraphGetLinksFunc(comm_, layers[i], hostContext_.epRankId, peer, &links, &linkNum);
                if (ret != HCCL_SUCCESS || links == nullptr) {
                    continue; // A probed layer need not connect this pair.
                }
                for (uint32_t j = 0; j < linkNum; ++j) {
                    if (links[j].linkAttr.linkProtocol == protocol) {
                        candidates.push_back(links[j]);
                    }
                }
            }
            if (!candidates.empty()) {
                return candidates;
            }
        }
        TORCH_CHECK(false, "No UBC_CTP or UB_RTP link from rank ", hostContext_.epRankId, " to rank ", peer);
    }

    static std::vector<CommLink> SelectSharedLinks(const std::vector<std::vector<CommLink>> &candidates)
    {
        // Match the official example's local-endpoint grouping without multiplying the requested jetty count.
        // Search alternate links as independently choosing each peer's first link can miss a common endpoint.
        for (const auto &first : candidates.front()) {
            std::vector<CommLink> selected;
            for (const auto &peerLinks : candidates) {
                auto match = std::find_if(peerLinks.begin(), peerLinks.end(), [&first](const CommLink &link) {
                    return std::memcmp(first.srcEndpointDesc.commAddr.raws, link.srcEndpointDesc.commAddr.raws,
                                       sizeof(first.srcEndpointDesc.commAddr.raws)) == 0;
                });
                if (match == peerLinks.end()) {
                    break;
                }
                selected.push_back(*match);
            }
            if (selected.size() == candidates.size()) {
                return selected;
            }
        }
        TORCH_CHECK(false, "Shared jetty peers have no common local endpoint; this benchmark requires one endpoint ",
                    "for all peers to preserve the requested shared_jetty_num");
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
        CheckHccl(HcclChannelDescInit(descriptors.data(), descriptors.size()), "HcclChannelDescInit");
        std::vector<std::vector<CommLink>> candidates;
        for (uint32_t peer = 0; peer < worldSize_; ++peer) {
            if (peer != hostContext_.epRankId) {
                candidates.push_back(FindUrmaLinks(peer, layers, layerNum));
            }
        }
        auto sharedLinks = SelectSharedLinks(candidates);
        for (uint32_t peer = 0; peer < worldSize_; ++peer) {
            if (peer == hostContext_.epRankId) {
                continue;
            }
            uint32_t index = peer < hostContext_.epRankId ? peer : peer - 1;
            const auto &link = sharedLinks[index];
            auto &desc = descriptors[index];
            desc.remoteRank = peer;
            desc.channelProtocol = link.linkAttr.linkProtocol;
            desc.localEndpoint = link.srcEndpointDesc;
            desc.remoteEndpoint = link.dstEndpointDesc;
            desc.memHandles = &memHandle_;
            desc.memHandleNum = 1;
        }
        return descriptors;
    }

    std::string tag_;
    uint32_t worldSize_;
    int32_t device_ = 0;
    HcclComm comm_ = nullptr;
    void *localBuffer_ = nullptr;
    HcclMemHandle memHandle_ = nullptr;
    SharedJettyContext hostContext_{};
    at::Tensor context_;
    bool creationStarted_ = false;
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    py::class_<SharedJettyContextManager>(m, "SharedJettyContextManager")
        .def(py::init<const std::string &, int64_t, int64_t, int64_t, const std::string &>())
        .def("create_context", &SharedJettyContextManager::CreateContext)
        .def_property_readonly("local_buffer_address", &SharedJettyContextManager::LocalBufferAddress);
}
} // namespace op_api
