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
#include "hccl_common.h"


// 确保acl_rt_api.h中的inline wrapper能正确链接
extern "C" {
aclError aclrtHostRegisterV2(void *ptr, uint64_t size, uint32_t flag);
aclError aclrtHostUnregister(void *ptr);
aclError aclrtHostGetDevicePointer(void *pHost, void **pDevice, uint32_t flag);
}

namespace op_api {

constexpr static uint8_t COMM_ENGINE_AIV = 4;
constexpr uint32_t HCCL_MAX_RANK_SIZE = 1024;
// URMA 链路协议值（AIV URMA, local UB fabric）
constexpr int COMM_PROTOCOL_UBC_CTP_VALUE = 4;
constexpr int COMM_PROTOCOL_UBC_TP_VALUE  = 5;

struct CommContext {
    uint32_t epRankId = 0;
    uint32_t rankSize = 0;
    uint64_t virtualAddrList[HCCL_MAX_RANK_SIZE] = {};
    uint64_t hcommHandle_[HCCL_MAX_RANK_SIZE] = {};
};

void CopyContextToTensor(const CommContext &context, at::Tensor &tensor);

class EngramEmbedingUtilsBuilder {
public:
    void *myBufHost_ = nullptr;
    void *selfBufDev_ = nullptr;
    HcclMemHandle memHandle_ = nullptr;
    uint64_t bufBytes_ = 0;

    void Build(const std::string &group, uint64_t bufBytes, at::Tensor &contextTensor);
    CommContext Build(const std::string &group, uint64_t bufBytes);
    void AcquireHcclHandle(const std::string &group, HcclComm &hcclHandle);
    void BuildContext(const HcclComm &hcclHandle, const std::string &group,
                      CommContext &commContextStruct);
    static bool CheckLinks(uint32_t &netLinkNum, CommLink *linksList);
    void AllocateAndRegisterBuffer(const HcclComm &commHandle, const std::string &mc2ContextTag,
                                    uint32_t rankId);
    void BuildChannelDescs(const HcclComm &commHandle, uint32_t srcRankId, uint32_t rankDim,
                           std::vector<HcclChannelDesc> &channelDesc);
    void GetHcclCommChannel(const HcclComm &commHandle, uint32_t rankDim, uint32_t srcRankId,
                             ChannelHandle *channels, uint32_t length);
    void GetHcclCommResource(const HcclComm &commHandle, const CommEngine &engine,
                              CommContext *commContextStruct, uint32_t rankSize);
    void CreateContext(const HcclComm &commHandle, const std::string &mc2ContextTag,
                        const CommEngine &engine, void *&ctx,
                        CommContext &commContextStruct);
};

class EngramEmbeddingUtils {
public:
    EngramEmbeddingUtils(const std::string &group, int64_t cclBufferSize);
    ~EngramEmbeddingUtils();

    at::Tensor create_context();
    CommContext get_comm_context();
    void update_group(const std::string &group, at::Tensor &contextTensor);
    void update_comm_context(const std::string &group, CommContext &commContext);
    int64_t ccl_buffer_size() const;
    int64_t host_ptr() const;

private:
    static int64_t context_tensor_size();

    std::string group_;
    int64_t cclBufferSize_;
    bool initialized_ = false;
    void *myBufHost_ = nullptr;
    CommContext cachedContext_ = {};
};

} // namespace op_api

#endif // NPU_OPS_TRANSFORMER_COMM_CONTEXT_CHANNEL_H