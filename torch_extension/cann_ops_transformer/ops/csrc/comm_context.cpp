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
 * \file comm_context.cpp
 * \brief comm_context implementation supporting both KFC and HCCL Channel modes.
 *
 */

#include <torch/extension.h>
#include <acl/acl_rt.h>
#include "comm_context.h"
#include "hccl_common.h"

namespace op_api {

// ======================== Common Types and Utilities ========================

enum class BackendMode : uint8_t {
    UNINITIALIZED,
    KFC,
    CHANNEL
};

static const char *GetSocName()
{
    static const char *socName = aclrtGetSocName();
    return socName;
}

BackendMode ResolveBackend(const py::object &backend)
{
    if (py::isinstance<py::str>(backend)) {
        auto mode = backend.cast<std::string>();
        if (mode == "channel")
            return BackendMode::CHANNEL;
        if (mode == "kfc")
            return BackendMode::KFC;
        TORCH_CHECK(false, "backend string must be 'kfc' or 'channel', got '", mode, "'");
    }

    if (py::isinstance<py::dict>(backend)) {
        auto dict = backend.cast<py::dict>();
        TORCH_CHECK(dict.size() > 0, "backend dict must not be empty");

        const char *socName = GetSocName();
        TORCH_CHECK(socName != nullptr, "aclrtGetSocName returned nullptr");

        for (auto item : dict) {
            std::string key = py::cast<std::string>(item.first);
            std::string value = py::cast<std::string>(item.second);
            if (value == "channel" || value == "kfc") {
                if (std::strstr(socName, key.c_str()) != nullptr) {
                    ASCEND_LOGI("Matched SoC name '%s' with key '%s', using backend '%s'", socName, key.c_str(),
                                value.c_str());
                    return value == "channel" ? BackendMode::CHANNEL : BackendMode::KFC;
                }
            } else {
                TORCH_CHECK(false, "backend dict value must be 'kfc' or 'channel', got '", value, "'");
            }
        }

        TORCH_CHECK(false, "No matching SoC name found for '", socName, "' in backend dict");
    }

    TORCH_CHECK(false, "backend must be a string ('kfc' or 'channel') or a dict");
}

// ======================== KFC Mode ========================
class KfcContextBuilder {
public:
    void Build(const std::string &group, int64_t worldSize, int64_t &cclBufferSize, at::Tensor &contextTensor)
    {
        CommContext mc2ContextHost;
        GetMc2Context(mc2ContextHost, worldSize, cclBufferSize, group.c_str());

        CopyContextToTensor(mc2ContextHost, contextTensor);
    }

private:
    void CollectRankBuffers(HcclComm &comm, int64_t worldSize, int64_t &cclBufferSize, CommContext &mc2Context)
    {
        uint32_t ctxIndex = 0;
        uint32_t rankId = 0;
        auto hcclRet = HcclGetRankIdFunc(comm, &rankId);
        TORCH_CHECK(hcclRet == HCCL_SUCCESS, "HcclGetRankIdFunc failed, ret: ", hcclRet);
        mc2Context.epRankId = rankId;
        const char *socName = GetSocName();
        void *remoteAddr = nullptr;
        uint64_t commSize = 0;
        if (socName != nullptr && std::strstr(socName, "Ascend910B") != nullptr && worldSize > 8) {
            HcclResult ret = static_cast<HcclResult>(HcclGetHcclBufferFunc(comm, &remoteAddr, &commSize));
            TORCH_CHECK((ret == HCCL_SUCCESS), "Get HcclBufferSize failed, ret=", ret);
            cclBufferSize = static_cast<int64_t>(commSize);
            mc2Context.epHcclBuffer_[rankId] = (uint64_t)remoteAddr;
            return;
        }
        for (uint64_t remoteRankId = 0; remoteRankId < worldSize; remoteRankId++) {
            commSize = 0;
            remoteAddr = nullptr;
            HcclResult ret;
            if (rankId == remoteRankId) {
                ret = static_cast<HcclResult>(HcclGetHcclBufferFunc(comm, &remoteAddr, &commSize));
                cclBufferSize = commSize;
            } else {
                ret = static_cast<HcclResult>(HcclGetRemoteIpcHcclBufFunc(comm, remoteRankId, &remoteAddr, &commSize));
            }
            TORCH_CHECK((ret == HCCL_SUCCESS), "Get HcclBufferSize failed, ret=", ret);
            mc2Context.epHcclBuffer_[remoteRankId] = (uint64_t)remoteAddr;
        }
    }

    void CreateHcclContext(HcclComm &commHandle, void *opArgs, int64_t worldSize, const char *groupName,
                           std::string algConfig, uint32_t opType, CommContext &mc2Context)
    {
        HcclResult ret =
            static_cast<HcclResult>(HcclKfcOpArgsSetAlgConfigFunc(opArgs, const_cast<char *>(algConfig.c_str())));
        TORCH_CHECK(ret == 0, "HcclKfcOpArgsSetAlgConfig failed, ret:", ret);
        ret = static_cast<HcclResult>(HcclCommGetHandleWithNameFunc(groupName, &commHandle));
        TORCH_CHECK(ret == 0, "HcclGetCommHandle failed, ret:", ret);
        void *opsResCtx;
        ret = static_cast<HcclResult>(HcclCreateOpResCtxFunc(commHandle, opType, opArgs, &opsResCtx));
        TORCH_CHECK(ret == 0, "HcclCreateOpResCtx failed, ret:", ret);
        mc2Context.kfcContextAddr = (uint64_t)opsResCtx;

        uint32_t rankId = 0;
        uint32_t worldSizeHccl = 0;
        ret = static_cast<HcclResult>(HcclGetRankSizeFunc(commHandle, &worldSizeHccl));
        TORCH_CHECK(ret == HCCL_SUCCESS, "HcclGetRankSize failed, ret:", ret);
        ret = static_cast<HcclResult>(HcclGetRankIdFunc(commHandle, &rankId));
        TORCH_CHECK(ret == HCCL_SUCCESS, "HcclGetRankId failed, ret:", ret);
        TORCH_CHECK(rankId < worldSizeHccl, "rankId:", rankId, " worldSizeHccl:", worldSizeHccl,
                    " worldSize:", worldSize);
        TORCH_CHECK(worldSize == worldSizeHccl, "worldSize:", worldSize, " != worldSizeHccl:", worldSizeHccl);
    }

    void GetMc2Context(CommContext &mc2ContextHost, int64_t worldSize, int64_t &cclBufferSize, const char *groupStr)
    {
        InitHcclFunctions();
        void *opArgs = nullptr;
        HcclResult ret = static_cast<HcclResult>(HcclKfcAllocOpArgsFunc(&opArgs));
        TORCH_CHECK(ret == 0, "HcclKfcAllocOpArgs failed, ret:", ret);
        uint8_t commEngine = COMM_ENGINE_AIV;
        ret = static_cast<HcclResult>(HcclKfcOpArgsSetCommEngineFunc(opArgs, (uint8_t)commEngine));
        TORCH_CHECK(ret == 0, "HcclKfcOpArgsSetCommEngine failed, ret:", ret);
        HcclComm commHandle;
        const char *socName = GetSocName();
        const bool is910B = (socName != nullptr && std::strstr(socName, "Ascend910B") != nullptr);
        const bool isMultiServer = worldSize > 8;
        const std::string algConfig =
            is910B && isMultiServer ? "BatchWrite=level1:hierarchy" : "AlltoAll=level0:fullmesh;level1:pairwise";
        const uint32_t opType = is910B && isMultiServer ? 18 : 8; // 18: BatchWrite, 8: AllToAll
        CreateHcclContext(commHandle, opArgs, worldSize, groupStr, algConfig, opType, mc2ContextHost);
        ret = static_cast<HcclResult>(HcclKfcFreeOpArgsFunc(opArgs));
        TORCH_CHECK(ret == 0, "getHcclKfcFreeOpArgs failed, ret:", ret);
        CollectRankBuffers(commHandle, worldSize, cclBufferSize, mc2ContextHost);
    }
};

// ======================== CommContextManager ========================

class CommContextManager {
public:
    CommContextManager(const std::string &group, int64_t worldSize, const py::object &backend = py::str("kfc"),
                       const std::string &commAlg = "ub-mem", const std::string &opName = "moe_dispatch_ffn_combine")
        : group_(group), worldSize_(worldSize), commAlg_(commAlg), opName_(opName), backend_(backend),
          mode_(BackendMode::UNINITIALIZED), cclBufferSize_(0), topoType_(TopoType::INTRA_SUPER_NODE),
          rankNumPerServer_(2)
    {
    }

    at::Tensor CreateContext()
    {
        EnsureResolved();
        at::Tensor context = at::empty({ContextTensorSize()}, at::TensorOptions()
                                                                  .dtype(at::kInt)
                                                                  .device(c10::DeviceType::PrivateUse1)
                                                                  .memory_format(c10::MemoryFormat::Contiguous));
        DispatchBuild(context);
        return context;
    }

    void UpdateGroup(const std::string &group, at::Tensor &contextTensor)
    {
        group_ = group;
        cclBufferSize_ = 0;
        EnsureResolved();
        DispatchBuild(contextTensor);
    }

    int64_t CclBufferSize() const
    {
        return cclBufferSize_;
    }
    int64_t GetTopoType() const
    {
        return static_cast<int64_t>(topoType_);
    }
    int64_t GetRankNumPerServer() const
    {
        return static_cast<int64_t>(rankNumPerServer_);
    }

private:
    static int64_t ContextTensorSize()
    {
        return (sizeof(CommContext) + sizeof(int32_t) - 1) / sizeof(int32_t);
    }

    void EnsureResolved()
    {
        if (mode_ == BackendMode::UNINITIALIZED) {
            mode_ = ResolveBackend(backend_);
        }
    }

    void DispatchBuild(at::Tensor &tensor)
    {
        switch (mode_) {
            case BackendMode::KFC:
                {
                    KfcContextBuilder builder;
                    builder.Build(group_, worldSize_, cclBufferSize_, tensor);
                    return;
                }
            case BackendMode::CHANNEL:
                {
                    HcclChannelContextBuilder builder;
                    builder.Build(group_, worldSize_, cclBufferSize_, tensor, commAlg_, opName_);
                    topoType_ = builder.GetTopoType();
                    rankNumPerServer_ = builder.GetRankNumPerServer();
                    return;
                }
            default:
                TORCH_CHECK(false, "Unknown backend mode: ", static_cast<int>(mode_));
        }
    }

    std::string group_;
    std::string commAlg_;
    std::string opName_;
    int64_t worldSize_;
    py::object backend_;
    BackendMode mode_;
    int64_t cclBufferSize_ = 0;
    TopoType topoType_ = TopoType::INTRA_SUPER_NODE;
    int64_t rankNumPerServer_ = 2;
};

// Bind the CommContextManager class to Python module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    py::class_<CommContextManager>(m, "CommContextManager")
        .def(py::init<const std::string &, int64_t, const py::object &, const std::string &, const std::string &>(),
             py::arg("group"), py::arg("worldSize"), py::arg("backend") = std::string("kfc"),
             py::arg("commAlg") = std::string("ub-mem"), py::arg("opName") = std::string("moe_dispatch_ffn_combine"))
        .def("create_context", &CommContextManager::CreateContext)
        .def("update_group", &CommContextManager::UpdateGroup, py::arg("group"), py::arg("contextTensor").noconvert())
        .def_property_readonly("ccl_buffer_size", &CommContextManager::CclBufferSize)
        .def_property_readonly("topo_type", &CommContextManager::GetTopoType)
        .def_property_readonly("rank_num_per_server", &CommContextManager::GetRankNumPerServer);
}
} // namespace op_api
