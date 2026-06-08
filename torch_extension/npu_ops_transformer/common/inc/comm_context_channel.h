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
 * \file comm_context_channel.h
 * \brief HCCL Channel mode communication context declarations.
 */

#ifndef NPU_OPS_TRANSFORMER_COMM_CONTEXT_CHANNEL_H
#define NPU_OPS_TRANSFORMER_COMM_CONTEXT_CHANNEL_H

#include <torch/extension.h>
#include <acl/acl_rt.h>
#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include "hccl_common.h"

namespace op_api {

constexpr static uint8_t COMM_ENGINE_AIV = 4;
constexpr uint32_t HCCL_MAX_RANK_SIZE = 1024;
constexpr uint32_t HCCL_COMM_LAYERS_MTE_CCU = 1;
constexpr uint32_t HCCL_COMM_LAYERS_UB_MEM = 0;
constexpr uint64_t COMM_DEFAULT_BUF_BYTES = 500ULL * 1024ULL * 1024ULL;

struct CommContext {
    uint32_t epRankId = 0;
    uint64_t epHcclBuffer_[HCCL_MAX_RANK_SIZE] = {};
    uint64_t hcommHandle_[HCCL_MAX_RANK_SIZE] = {};
};

void CopyContextToTensor(const CommContext &context, at::Tensor &tensor);

class HcclChannelContextBuilder {
public:
    void *myBufHost_ = nullptr;
    void *myBufDev_ = nullptr;
    HcclMemHandle myMemHandle_ = nullptr;
    uint64_t bufBytes_ = 0;
    uint32_t rankId_ = 0;
    uint32_t rankSize_ = 0;
    std::vector<uint64_t> peerBufAddrs_;
    std::vector<uint64_t> channelPtrByRank_;

    void Build(const std::string &group, int64_t worldSize, int64_t &cclBufferSize, at::Tensor &contextTensor);
    CommContext Build(const std::string &group, int64_t worldSize, int64_t &cclBufferSize);
    void AcquireHcclHandle(const std::string &group, HcclComm &hcclHandle);
    void BuildContext(const HcclComm &hcclHandle, const std::string &group,
                      const CommProtocol &protocol, CommContext &commContextStruct,
                      int64_t &cclBufferSize);
    void GetCommProtocol(const HcclComm &commHandle, CommProtocol &protocol);
    void CheckProtocolSupport(const HcclComm &commHandle, uint32_t *&layerList, uint32_t &layerNum);
    static bool CheckLinks(uint32_t &netLinkNum, CommLink *linksList);
    void AllocateAndRegisterBuffer(const HcclComm &commHandle, const std::string &mc2ContextTag,
                                    uint64_t bufBytes);
    void InitHcclChannel(const HcclComm &commHandle, uint32_t rankDim, uint32_t srcRankId,
                          const CommProtocol &protocol, std::vector<HcclChannelDesc> &channelDesc);
    void GetHcclCommChannel(const HcclComm &commHandle, uint32_t rankDim, uint32_t srcRankId,
                             const CommProtocol &protocol, const CommEngine &engine,
                             std::vector<ChannelHandle> &channels);
    void GetHcclCommResource(const HcclComm &commHandle, const CommEngine &engine, const CommProtocol &protocol,
                              CommContext *commContextStruct, uint32_t rankSize);
    void CollectChannelPtrs(const HcclComm &commHandle,
                             const std::vector<ChannelHandle> &channels,
                             uint32_t srcRankId, uint32_t rankSize,
                             CommContext *commContextStruct);
    void CreateContext(const HcclComm &commHandle, const std::string &mc2ContextTag,
                        const CommEngine &engine, const CommProtocol &protocol, void *&ctx,
                        CommContext &commContextStruct, uint64_t bufBytes);
    static void GetNetLayers(const HcclComm &commHandle, uint32_t *&netLayerList, uint32_t &netLayerNum);
    static void GetHcclCommLink(const HcclComm &commHandle, uint32_t netLayerId, uint32_t srcRankId,
                                  uint32_t dstRankId, const CommProtocol &protocol, CommLink *&links);

    std::unordered_map<uint32_t, uint32_t> layerMap_;
};

class CommContextManager {
public:
    CommContextManager(const std::string &group, int64_t worldSize);
    ~CommContextManager();

    at::Tensor create_context();
    CommContext get_comm_context();
    void update_group(const std::string &group, at::Tensor &contextTensor);
    void update_comm_context(const std::string &group, CommContext &commContext);
    int64_t ccl_buffer_size() const;

private:
    static int64_t context_tensor_size();

    std::string group_;
    int64_t worldSize_;
    int64_t cclBufferSize_ = 0;
    bool initialized_ = false;
    void *myBufHost_ = nullptr;
    void *myBufDev_ = nullptr;
    HcclMemHandle myMemHandle_ = nullptr;
    uint64_t bufBytes_ = 0;
    uint32_t rankId_ = 0;
    uint32_t rankSize_ = 0;
    std::vector<uint64_t> peerBufAddrs_;
    std::vector<uint64_t> channelPtrByRank_;
};

} // namespace op_api

#endif // NPU_OPS_TRANSFORMER_COMM_CONTEXT_CHANNEL_H