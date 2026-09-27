/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file comm_channel_builder.cpp
 * \brief Host 侧 HCCL channel 创建工具，通过 CommChannelBuilderManager 对外暴露接口
 *
 * CommChannelBuilder 直接引用 mc2/common/op_kernel/apace 下的 comm_channel_builder.h 原版实现，
 * CommChannelBuilderManager 负责适配 torch_extension：
 * - 从 group name 获取 HcclComm handle
 * - 动态加载 HCCL 函数指针 (InitHcclEngineCtxFunctions)
 * - 持有 CommContext 成员，调用原版 CreateDeviceContext 填充
 * - 用 at::from_blob 将 device 指针包装为 at::Tensor 返回
 */

#include <torch/extension.h>
#include <cstdint>
#include <cstring>
#include <string>
#include <tuple>
#include "acl/acl.h"
#include "acl/acl_rt.h"
#include "hccl_common.h"
#include "apace/core/aiv_comm/collective_comm_context.h"
#include "apace/utils/comm_channel_builder.h"

namespace op_api {

using ApaceCommUdmaContext = Apace::AivComm::CommUdmaContext;
using ApaceCommUbmemContext = Apace::AivComm::CommUbmemContext;

struct CommContext {
    ApaceCommUdmaContext udmaCtx;
    ApaceCommUbmemContext ubmemCtx;
};

struct QuantMteContext {
    uint32_t epRankId = 0;
    uint32_t rankSizePerServer = 0;
    uint64_t kfcContextAddr = 0;
    uint64_t epHcclBuffer_[1024] = {};
    uint64_t hcommHandle_[1024] = {};
};
static_assert(sizeof(QuantMteContext) == 16400, "QuantMteContext layout must match QuantReduceScatterContext");

using DefaultCommChannelBuilder = CommChannelBuilder<ChannelMode::URMA, ChannelMode::UBMEM>;

class CommChannelBuilderManager {
public:
    explicit CommChannelBuilderManager(const std::string &group)
        : group_(group)
    {
        InitHcclEngineCtxFunctions();

        auto aclnnRet = HcomGetCommHandleByGroupFunc(group.c_str(), &comm_);
        TORCH_CHECK(aclnnRet == HCCL_SUCCESS, "Get HCCL handle failed, group: ", group, ", ret: ", aclnnRet);

        builder_ = std::make_unique<DefaultCommChannelBuilder>(comm_);
    }

    uint32_t GetRankId() const
    {
        TORCH_CHECK(builder_ != nullptr, "builder_ is not initialized");
        return builder_->GetRankId();
    }

    uint32_t GetRankSize() const
    {
        TORCH_CHECK(builder_ != nullptr, "builder_ is not initialized");
        return builder_->GetRankSize();
    }

    /*!
     * \brief 创建 device context 并包装为 at::Tensor 返回
     *
     * 内部流程：
     * 1. 调用原版 CreateDeviceContext(hostCtx, size, tag, dataCtx, barrierCtx) 填充 CommContext
     * 2. 用 at::from_blob 将返回的 device 指针包装为 at::Tensor
     */
    at::Tensor CreateContext(const std::string &ctxTag)
    {
        TORCH_CHECK(builder_ != nullptr, "builder_ is not initialized");
        CommContext hostCtx = {};
        void *devCtx = builder_->CreateDeviceContext(&hostCtx, sizeof(CommContext), ctxTag.c_str(), &hostCtx.udmaCtx,
                                                     &hostCtx.ubmemCtx);

        TORCH_CHECK(devCtx != nullptr, "CreateDeviceContext failed, ctxTag: ", ctxTag);

        at::Tensor context = at::from_blob(devCtx, {sizeof(CommContext) / sizeof(int32_t)}, at::kInt);
        return context;
    }

    /*!
     * \brief 创建 quant MTE 算子族公共 context，使用 HCCL 内置 buffer 作为 win 区
     *
     * 固定 ctxTag 保证 eager/静态图/动态图共享同一份 context。命中时仅查询返回，
     * 未命中时交换各 rank 内置 buffer 地址并拷贝到 engine context。
     */
    std::tuple<at::Tensor, int64_t> CreateQuantMteContext(const std::string &ctxTag)
    {
        TORCH_CHECK(builder_ != nullptr, "builder_ is not initialized");
        constexpr uint32_t quantMteMaxRank = 1024U;
        const CommEngine engine = CommEngine::COMM_ENGINE_AIV;

        void *ctx = nullptr;
        uint64_t ctxSize = 0;
        if (HcclEngineCtxGetFunc(comm_, ctxTag.c_str(), engine, &ctx, &ctxSize) == HCCL_SUCCESS) {
            TORCH_CHECK(ctx != nullptr, "HcclEngineCtxGet returned null context, tag: ", ctxTag);
            TORCH_CHECK(ctxSize >= sizeof(QuantMteContext), "Invalid quant MTE context size, expected at least ",
                        sizeof(QuantMteContext), ", got ", ctxSize);
            return {WrapQuantMteContext(ctx), static_cast<int64_t>(GetHcclBufferSize())};
        }

        TORCH_CHECK(builder_->Init(), "Init CommChannelBuilder failed, tag: ", ctxTag);

        uint64_t remoteAddrs[quantMteMaxRank] = {};
        const std::string memTag = ctxTag + "_winbuf";
        TORCH_CHECK(builder_->AllocRegAndBuildChannels(ChannelMode::URMA, memTag.c_str(), nullptr, remoteAddrs),
                    "Build quant MTE HCCL buffer address table failed, tag: ", ctxTag);

        auto hcclRet = HcclEngineCtxCreateFunc(comm_, ctxTag.c_str(), engine, sizeof(QuantMteContext), &ctx);
        TORCH_CHECK(hcclRet == HCCL_SUCCESS, "HcclEngineCtxCreate failed, tag: ", ctxTag, ", ret: ", hcclRet);
        TORCH_CHECK(ctx != nullptr, "HcclEngineCtxCreate returned null context, tag: ", ctxTag);

        const uint32_t rankId = builder_->GetRankId();
        const uint32_t rankSize = builder_->GetRankSize();
        TORCH_CHECK(rankSize > 0 && rankSize <= quantMteMaxRank, "Invalid quant MTE rank size: ", rankSize);
        TORCH_CHECK(rankId < rankSize, "Invalid quant MTE rank id ", rankId, " for rank size ", rankSize);

        uint32_t *layers = nullptr;
        uint32_t layerNum = 0;
        hcclRet = HcclRankGraphGetLayersFunc(comm_, &layers, &layerNum);
        TORCH_CHECK(hcclRet == HCCL_SUCCESS && layers != nullptr && layerNum > 0,
                    "HcclRankGraphGetLayers failed, ret: ", hcclRet);

        uint32_t rankSizePerServer = 0;
        hcclRet = HcclRankGraphGetRankSizeByLayerFunc(comm_, layers[0], &rankSizePerServer);
        TORCH_CHECK(hcclRet == HCCL_SUCCESS && rankSizePerServer > 0 && rankSizePerServer <= rankSize,
                    "HcclRankGraphGetRankSizeByLayer failed, ret: ", hcclRet);

        QuantMteContext hostContext = {};
        hostContext.epRankId = rankId;
        hostContext.rankSizePerServer = rankSizePerServer;
        for (uint32_t i = 0; i < rankSize; ++i) {
            hostContext.epHcclBuffer_[i] = remoteAddrs[i];
        }

        hcclRet = HcclEngineCtxCopyFunc(comm_, engine, ctxTag.c_str(), &hostContext, sizeof(hostContext), 0);
        TORCH_CHECK(hcclRet == HCCL_SUCCESS, "HcclEngineCtxCopy failed, tag: ", ctxTag, ", ret: ", hcclRet);

        return {WrapQuantMteContext(ctx), static_cast<int64_t>(GetHcclBufferSize())};
    }

    /*!
     * \brief 获取 HCCL 内置 buffer 大小
     */
    uint64_t GetHcclBufferSize()
    {
        void *buf = nullptr;
        uint64_t bufSize = 0;
        auto hcclRet = HcclGetHcclBufferFunc(comm_, &buf, &bufSize);
        TORCH_CHECK(hcclRet == HCCL_SUCCESS && buf != nullptr, "HcclGetHcclBuffer failed, ret: ", hcclRet);
        return bufSize;
    }

private:
    static at::Tensor WrapQuantMteContext(void *ctx)
    {
        auto options = at::TensorOptions()
                           .dtype(at::kInt)
                           .device(c10::DeviceType::PrivateUse1)
                           .memory_format(c10::MemoryFormat::Contiguous);
        return at::from_blob(ctx, {sizeof(QuantMteContext) / sizeof(int32_t)}, options);
    }

    HcclComm comm_ = nullptr;
    std::string group_;
    std::unique_ptr<DefaultCommChannelBuilder> builder_;
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    py::class_<CommChannelBuilderManager>(m, "CommChannelBuilderManager")
        .def(py::init<const std::string &>(), py::arg("group"))
        .def("get_rank_id", &CommChannelBuilderManager::GetRankId)
        .def("get_rank_size", &CommChannelBuilderManager::GetRankSize)
        .def("create_context", &CommChannelBuilderManager::CreateContext, py::arg("ctx_tag"))
        .def("create_quant_mte_context", &CommChannelBuilderManager::CreateQuantMteContext, py::arg("ctx_tag"))
        .def("get_hccl_buffer_size", &CommChannelBuilderManager::GetHcclBufferSize);
}

} // namespace op_api
