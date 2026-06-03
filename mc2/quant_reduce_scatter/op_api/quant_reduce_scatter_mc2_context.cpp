/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
**/

/*!
 * \file quant_reduce_scatter_mc2_context.cpp
 * \brief quant_reduce_scatter mc2 context management implementation
 */
#include <cstdlib>
#include <dlfcn.h>
#include "common/op_api/mc2_context.h"
#include "quant_reduce_scatter_mc2_context.h"

#if HCOMM_VERSION_NUM >= HCCL_CHANNEL_SUPPORT_VERSION

namespace {
constexpr uint64_t KOPY_DEFAULT_CTX_OFFSET = 0;
constexpr uint64_t MAX_CONTEXT_TAG_SIZE = 255;
constexpr uint32_t HCCL_COMM_LAYERS_MTE_CCU = 1;
constexpr uint32_t HCCL_COMM_LAYERS_UB_MEM = 0;
constexpr uint32_t GET_LOCAL_SERVER_RANK_SIZE_LAYER = 0;
constexpr uint32_t EP_RANK_OFFSET_STEP = 8192;
} // namespace

namespace Mc2Aclnn {

/**
 * @brief GetHcclCommResource for QuantReduceScatter
 */
aclnnStatus Mc2QrsContext::GetHcclCommResourceForQrs(const HcclComm &hcclHandle, const CommEngine &engine,
                                                     const CommProtocol &protocol,
                                                     Mc2QuantReduceScatterContext *mc2ContextStruct)
{
    OP_LOGI("Start to get HCCL communication resource for QuantReduceScatter");

    if (mc2ContextStruct->rankDim > HCCL_MTE_MAX_RANK_NUM) {
        OP_LOGE(ACLNN_ERR_INNER,
                "rankDim %u exceeds HCCL_MTE_MAX_RANK_NUM %u",
                mc2ContextStruct->rankDim, HCCL_MTE_MAX_RANK_NUM);
        return ACLNN_ERR_INNER;
    }

    uint32_t rankId = mc2ContextStruct->rankId;
    std::vector<ChannelHandle> channels;
    auto ret = Mc2Context::GetHcclCommChannel(hcclHandle, mc2ContextStruct->rankDim, rankId, protocol, engine, channels);
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }
    OP_LOGI("Get HCCL communication channel success, channel num is: %u", channels.size());

    for (uint32_t i = 0; i < mc2ContextStruct->rankDim; ++i) {
        void *tempBuffer = nullptr;
        uint64_t bufSize = 0;
        HcclResult hcclRet;

        if (i == rankId) {
            hcclRet = Mc2Context::HcclGetHcclBuffer(hcclHandle, &tempBuffer, &hcclBuffSize_);
            bufSize = hcclBuffSize_;
        } else {
            uint32_t idx = (i < rankId) ? i : (i - 1);
            hcclRet = Mc2Context::HcclChannelGetHcclBuffer(hcclHandle, channels[idx], &tempBuffer, &bufSize);
        }

        if (hcclRet != HCCL_SUCCESS || tempBuffer == nullptr) {
            OP_LOGE(ACLNN_ERR_INNER, "Get HCCL buffer failed, src: %u, dst: %u", rankId, i);
            return ACLNN_ERR_INNER;
        }

        mc2ContextStruct->windowsIn[i] = reinterpret_cast<uint64_t>(tempBuffer);
        mc2ContextStruct->windowsOut[i] = reinterpret_cast<uint64_t>(static_cast<char *>(tempBuffer) + bufSize / 2);
    }

    OP_LOGI("Get HCCL CommResource for QuantReduceScatter success");
    return ACLNN_SUCCESS;
}

/**
 * @brief CreatMc2Context for QuantReduceScatter
 */
aclnnStatus Mc2QrsContext::CreatMc2ContextForQrs(const HcclComm &hcclHandle, const std::string &mc2ContextTag,
                                                 const CommEngine &engine, const CommProtocol &protocol,
                                                 Mc2QuantReduceScatterContext *mc2ContextStruct, void *&ctx,
                                                 uint64_t &hcclBuffSize)
{
    OP_LOGI("Start to create HCCL context for QuantReduceScatter");

    uint64_t ctxSize = sizeof(Mc2QuantReduceScatterContext);
    auto hcclRet = Mc2Context::HcclEngineCtxCreate(hcclHandle, mc2ContextTag.c_str(), engine, ctxSize, &ctx);
    if (hcclRet != HCCL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER, "Create HCCL context memory failed");
        return ACLNN_ERR_INNER;
    }
    OP_LOGI("Create HCCL context for QuantReduceScatter success, context is: %p", ctx);

    hcclRet = Mc2Context::HcclGetRankId(hcclHandle, &mc2ContextStruct->rankId);
    if (hcclRet != HCCL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER, "Get rank ID failed");
        return ACLNN_ERR_INNER;
    }
    OP_LOGI("Get rank ID success for QuantReduceScatter, rankId is: %u", mc2ContextStruct->rankId);

    hcclRet = Mc2Context::HcclGetRankSize(hcclHandle, &mc2ContextStruct->rankDim);
    if (hcclRet != HCCL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER, "Get rank size failed");
        return ACLNN_ERR_INNER;
    }
    OP_LOGI("Get rank size for QuantReduceScatter success, rankSize is: %u", mc2ContextStruct->rankDim);

    auto ret = Mc2Context::GetHcclCommResourceForQrs(hcclHandle, engine, protocol, mc2ContextStruct);
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER, "Get HCCL communication resource failed");
        return ret;
    }

    mc2ContextStruct->workSpace = 0;
    mc2ContextStruct->workSpaceSize = 0;
    mc2ContextStruct->winSize = 0;
    hcclRet = Mc2Context::HcclEngineCtxCopy(hcclHandle, engine, mc2ContextTag.c_str(), mc2ContextStruct,
                                            ctxSize, KOPY_DEFAULT_CTX_OFFSET);
    if (hcclRet != HCCL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER, "Copy context from host to device failed");
        return ACLNN_ERR_INNER;
    }

    hcclBuffSize = hcclBuffSize_;
    OP_LOGI("Copy context for QuantReduceScatter from host to device success");
    return ACLNN_SUCCESS;
}

/**
 * @brief CreatMc2ContextTensor for QuantReduceScatter
 */
aclnnStatus Mc2QrsContext::CreatMc2ContextTensorForQrs(void *ctx, aclTensor *&mc2Context)
{
    OP_LOGI("Start to create Mc2Context Tensor for QuantReduceScatter");

    if (ctx == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER, "Create Mc2Context Tensor failed, context is nullptr.");
        return ACLNN_ERR_INNER;
    }

    uint64_t mc2ContextLength = sizeof(Mc2QuantReduceScatterContext);
    int64_t shape[1] = {static_cast<int64_t>(mc2ContextLength / sizeof(uint32_t))};
    int64_t strides[1] = {1};

    mc2Context = Mc2Context::aclCreateTensor(shape, 1, ACL_INT32, strides, 0, ACL_FORMAT_ND, shape, 1, ctx);
    if (mc2Context == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER, "Create Mc2Context Tensor failed.");
        return ACLNN_ERR_INNER;
    }

    OP_LOGI("CreatMc2ContextTensor for QuantReduceScatter Success");
    return ACLNN_SUCCESS;
}

/**
 * @brief GetMc2ContextTensor for QuantReduceScatter
 */
aclnnStatus Mc2Context::GetMc2ContextTensorForQrs(const char *group, const char *opName, uint64_t &hcclBuffSize,
                                                  aclTensor *&mc2Context, int64_t &worldSize)
{
    OP_LOGI("Start to get Mc2Context Tensor for QuantReduceScatter");

    Mc2Context::Mc2Context instance;
    auto aclnnRet = instance.LoadHcclSymbols();
    CHECK_RET(aclnnRet == ACLNN_SUCCESS, aclnnRet);

    void *ctx = nullptr;
    CommProtocol protocol;
    std::string mc2ContextTag = std::string(group) + std::string(opName);
    CommEngine engine = CommEngine::COMM_ENGINE_AIV;
    hcclBuffSize = 0; // Default to 0, will be updated in CheckContextCache

    aclnnRet = instance.ValidateContextTag(mc2ContextTag);
    CHECK_RET(aclnnRet == ACLNN_SUCCESS, aclnnRet);

    HcclComm hcclHandle;
    aclnnRet = instance.GetCommHandle(group, hcclHandle);
    CHECK_RET(aclnnRet == ACLNN_SUCCESS, aclnnRet);

    uint32_t rankSize = 0;
    auto hcclRet = instance.HcclGetRankSize(hcclHandle, &rankSize);
    if (hcclRet != HCCL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER, "Hccl get worldSize failed");
        return ACLNN_ERR_INNER;
    }
    worldSize = rankSize;
    OP_LOGI("Get worldSize success, worldSize is: %ld", worldSize);

    aclnnRet = instance.CheckContextCache(hcclHandle, mc2ContextTag, engine, ctx, hcclBuffSize);
    CHECK_RET(aclnnRet == ACLNN_SUCCESS, aclnnRet);
    if (hcclBuffSize != 0) {
        // Cache not found, need to create context
        aclnnRet = CreatMc2ContextTensorForQrs(ctx, mc2Context);
        CHECK_RET(aclnnRet == ACLNN_SUCCESS, aclnnRet);
        OP_LOGI("Found context cache, Get Mc2Context Tensor Success");
        return ACLNN_SUCCESS;
    }

    aclnnRet = instance.GetCommProtocol(hcclHandle, protocol);
    CHECK_RET(aclnnRet == ACLNN_SUCCESS, aclnnRet);

    Mc2QuantReduceScatterContext mc2ContextStruct = {};
    aclnnRet = CreatMc2ContextForQrs(hcclHandle, mc2ContextTag, engine, protocol,
                                     &mc2ContextStruct, ctx, hcclBuffSize);
    CHECK_RET(aclnnRet == ACLNN_SUCCESS, aclnnRet);

    aclnnRet = CreatMc2ContextTensorForQrs(ctx, mc2Context);
    CHECK_RET(aclnnRet == ACLNN_SUCCESS, aclnnRet);

    OP_LOGI("Get Mc2QuantReduceScatterContext Tensor Success");
    return ACLNN_SUCCESS;
}

} // namespace Mc2Aclnn

#endif
