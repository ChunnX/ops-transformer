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
#include "acl/acl.h"
#include "acl/acl_rt.h"
#include "torch_npu/csrc/aten/common/from_blob.h"
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

// UBMEM data channel 的窗口通信上下文：data/barrier 均为 UBMEM(UB_MEM) 协议。
// 与 URMA data channel 不同，UB_MEM channel 获取的远端地址可直通 GM 读写
// （参见 apace barrier_ubmem.h 对 commBufferAddrs 的 __gm__ 直写用法），
// 供窗口直写型算子（如 quant_all_reduce/quant_reduce_scatter 的 MTE 窗口协议）使用。
struct UbmemCommContext {
    ApaceCommUbmemContext dataCtx;
    ApaceCommUbmemContext barrierCtx;
};

using DefaultCommChannelBuilder = CommChannelBuilder<ChannelMode::URMA, ChannelMode::UBMEM>;
using UbmemCommChannelBuilder = CommChannelBuilder<ChannelMode::UBMEM, ChannelMode::UBMEM>;

class CommChannelBuilderManager {
public:
    explicit CommChannelBuilderManager(const std::string &group)
        : group_(group)
    {
        InitHcclEngineCtxFunctions();

        auto aclnnRet = HcomGetCommHandleByGroupFunc(group.c_str(), &comm_);
        TORCH_CHECK(aclnnRet == HCCL_SUCCESS, "Get HCCL handle failed, group: ", group, ", ret: ", aclnnRet);

        builder_ = std::make_unique<DefaultCommChannelBuilder>(comm_);
        ubmemBuilder_ = std::make_unique<UbmemCommChannelBuilder>(comm_);
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
     * 2. 用 at_npu::native::from_blob 将返回的 device 指针包装为 NPU at::Tensor。
     *    必须声明 NPU device（HCCL owns ctx，tensor 仅是非拥有的 device 内存视图）：
     *    若用 at::from_blob 默认得到包装 device 指针的 CPU tensor，消费方对其
     *    .cpu() 为 no-op、后续tolist()将按 host 地址解引用 device 显存导致段错误。
     */
    at::Tensor CreateContext(const std::string &ctxTag)
    {
        TORCH_CHECK(builder_ != nullptr, "builder_ is not initialized");
        CommContext hostCtx = {};
        void *devCtx = builder_->CreateDeviceContext(&hostCtx, sizeof(CommContext), ctxTag.c_str(), &hostCtx.udmaCtx,
                                                      &hostCtx.ubmemCtx);

        TORCH_CHECK(devCtx != nullptr, "CreateDeviceContext failed, ctxTag: ", ctxTag);

        auto options = at::TensorOptions().dtype(at::kInt).device(c10::DeviceType::PrivateUse1);
        at::Tensor context = at_npu::native::from_blob(devCtx, {sizeof(CommContext) / sizeof(int32_t)}, options);
        return context;
    }

    /*!
     * \brief 创建 UBMEM data channel 的 device context 并包装为 at::Tensor 返回
     *
     * dataCtx.commBufferAddrs 为各 rank 的 HCCL 内置 buffer 地址（首次创建时整体清零，
     * 大小由 HCCL_BUFFSIZE 决定），可直通 GM 读写；URMA data channel 获取的远端地址
     * 仅可作为 WriteNbi/ReadNbi 的地址参数、不能直通 GM 访问，窗口直写型算子须用本接口。
     * barrierCtx 为 engine ctx 尾部的 2MB 同步区（当前消费方不使用，预留）。
     */
    at::Tensor CreateUbmemContext(const std::string &ctxTag)
    {
        TORCH_CHECK(ubmemBuilder_ != nullptr, "ubmemBuilder_ is not initialized");
        UbmemCommContext hostCtx = {};
        void *devCtx = ubmemBuilder_->CreateDeviceContext(&hostCtx, sizeof(UbmemCommContext), ctxTag.c_str(),
                                                           &hostCtx.dataCtx, &hostCtx.barrierCtx);

        TORCH_CHECK(devCtx != nullptr, "CreateUbmemContext failed, ctxTag: ", ctxTag);

        auto options = at::TensorOptions().dtype(at::kInt).device(c10::DeviceType::PrivateUse1);
        at::Tensor context = at_npu::native::from_blob(devCtx, {sizeof(UbmemCommContext) / sizeof(int32_t)}, options);
        return context;
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

    /*!
     * \brief 判断是否为单机（单服务器）拓扑
     *
     * HCCL rank graph 仅存在 1 个 net layer 时，所有 rank 同属一个通信域（单机）；
     * 多层拓扑意味着跨服务器/超节点组网。供仅支持单机窗口通信的算子做部署约束校验。
     */
    bool IsSingleServerTopology() const
    {
        uint32_t layerNum = 0;
        uint32_t *layerList = nullptr;
        auto hcclRet = HcclRankGraphGetLayersFunc(comm_, &layerList, &layerNum);
        TORCH_CHECK(hcclRet == HCCL_SUCCESS && layerList != nullptr && layerNum > 0,
                    "Get HCCL layers failed, ret: ", hcclRet);
        return layerNum == 1;
    }

private:
    HcclComm comm_ = nullptr;
    std::string group_;
    std::unique_ptr<DefaultCommChannelBuilder> builder_;
    std::unique_ptr<UbmemCommChannelBuilder> ubmemBuilder_;
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    py::class_<CommChannelBuilderManager>(m, "CommChannelBuilderManager")
        .def(py::init<const std::string &>(), py::arg("group"))
        .def("get_rank_id", &CommChannelBuilderManager::GetRankId)
        .def("get_rank_size", &CommChannelBuilderManager::GetRankSize)
        .def("create_context", &CommChannelBuilderManager::CreateContext, py::arg("ctx_tag"))
        .def("create_ubmem_context", &CommChannelBuilderManager::CreateUbmemContext, py::arg("ctx_tag"))
        .def("get_hccl_buffer_size", &CommChannelBuilderManager::GetHcclBufferSize)
        .def("is_single_server_topology", &CommChannelBuilderManager::IsSingleServerTopology);
}

} // namespace op_api
