/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * ElasticBuffer C++ binding - 重构版本
 */

#include <torch/extension.h>
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <iostream>
#include <iomanip>
#include <functional>

// CANN ACL Runtime API
#include "acl/acl.h"

// HCCL types
#include "hccl/hccl_types.h"

// HCCL common utilities
#include "hccl_common.h"

// ACLNN common utilities
#include "aclnn_common.h"

// torch_npu stream utilities
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "engram_embedding_utils.h"

namespace op_api {

void EngramWrite(uint64_t hostPinnedAddr, const at::Tensor &storage)
{
    TORCH_CHECK(hostPinnedAddr != 0, "Host pinned memory address is null");

    void *dstAddr = reinterpret_cast<void *>(hostPinnedAddr);
    void *srcAddr = storage.data_ptr();
    size_t dataSize = storage.nbytes();
    memcpy(dstAddr, srcAddr, dataSize);
}

std::function<at::Tensor()> EngramFetch(const at::Tensor &commContextTensor, const at::Tensor &indices,
                                        int64_t hiddenSize, int64_t numEngramEntries, at::ScalarType outDtype, uint64_t hostPinnedAddr)
{
    int64_t numTokens = indices.size(0);
    if (numTokens == 0) {
        auto emptyTensor = at::empty({0, hiddenSize}, at::TensorOptions().dtype(outDtype).device(indices.device()));
        return [=]() { return emptyTensor; };
    }
    auto fetched = at::empty({numTokens, hiddenSize}, at::TensorOptions().dtype(outDtype).device(indices.device()));
    auto dtypeVal = 12;
    ACLNN_CMD(aclnnEngramFetch, commContextTensor, indices, fetched, hiddenSize, numEngramEntries, dtypeVal);
    return [=]() {
        // ACLNN_CMD(aclnnEngramFetchWait, commContextTensor);
        return fetched;
    };
}

void EngramDestroy(uint64_t hostPinnedAddr)
{
    if (hostPinnedAddr != 0) {
        void *addr = reinterpret_cast<void *>(hostPinnedAddr);
        aclError ret = aclrtHostUnregister(addr);
        TORCH_CHECK(ret == ACL_SUCCESS, "aclrtHostUnregister failed with error code: ", ret,
                    ", hostPinnedAddr: ", hostPinnedAddr);
        ret = aclrtFreeHost(addr);
        TORCH_CHECK(ret == ACL_SUCCESS, "aclrtFreeHost failed with error code: ", ret,
                    ", hostPinnedAddr: ", hostPinnedAddr);
    }
}

void EngramBarrier(const std::string &groupName, bool withDeviceSync)
{
    static bool initialized = false;
    if (!initialized) {
        InitHcclFunctions();
        initialized = true;
    }

    if (withDeviceSync) {
        aclError aclRet = aclrtSynchronizeDevice();
        TORCH_CHECK(aclRet == ACL_SUCCESS, "aclrtSynchronizeDevice failed with error code: ", aclRet);
    }

    HcclComm comm = nullptr;
    HcclResult ret = HcclCommGetHandleWithNameFunc(groupName.c_str(), &comm);
    TORCH_CHECK(ret == HCCL_SUCCESS && comm != nullptr, "HcclCommGetHandleWithName failed with error code: ", ret,
                ", groupName: ", groupName);

    aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
    ret = HcclBarrierFunc(comm, stream);
    TORCH_CHECK(ret == HCCL_SUCCESS, "HcclBarrier failed with error code: ", ret, ", comm: ", comm,
                ", stream: ", stream);

    if (withDeviceSync) {
        aclError aclRet = aclrtSynchronizeDevice();
        TORCH_CHECK(aclRet == ACL_SUCCESS, "aclrtSynchronizeDevice failed with error code: ", aclRet);
    }
}
} // namespace op_api

// PyBind11 module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("npu_engram_destroy", &op_api::EngramDestroy, "Destroy ElasticBuffer and free host pinned memory");

    m.def("npu_engram_write", &op_api::EngramWrite, "Write CPU tensor data to host pinned memory");

    m.def("npu_engram_barrier", &op_api::EngramBarrier, "Call HCCL barrier with group name and optional device sync");

    m.def("npu_engram_fetch", &op_api::EngramFetch, "EngramFetch with ACLNN_CMD");
}