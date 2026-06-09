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

namespace op_api {

constexpr uint32_t HCCL_MAX_RANK_SIZE = 1024;

/**
 * CommContext - 通信上下文结构体
 * 用于 ACLNN API，包含 RDMA 通信所需的拓扑信息
 */
struct CommContext {
    uint64_t epRankId = 0;
    uint64_t worldSize = 0;
    uint64_t virtualAddrList[HCCL_MAX_RANK_SIZE];
    uint64_t hcommHandle_[HCCL_MAX_RANK_SIZE];
};

/**
 * 获取 CommContext tensor 的大小（以 int32 为单位）
 */
static int64_t GetCommContextSize()
{
    return sizeof(CommContext) / sizeof(int32_t);
}

/**
 * 从 tensor 提取 CommContext 指针
 */
static CommContext *GetCommContextFromTensor(at::Tensor &tensor)
{
    auto data = tensor.accessor<int32_t, 1>();
    return reinterpret_cast<CommContext *>(data.data());
}

/**
 * ElasticBufferInit - 分配 host pinned memory
 *
 * 返回 py::dict:
 *   - "commContext": CommContext tensor（用于 ACLNN API）
 *   - "hostPinnedAddr": host pinned memory 地址（uint64_t）
 */
py::dict ElasticBufferInit(std::string groupName, uint64_t rank, uint64_t worldSize, uint64_t numCpuBytes)
{
    std::cout << "[C++] ElasticBufferInit called" << std::endl;
    std::cout << "[C++] groupName: " << groupName << ", rank: " << rank << ", worldSize: " << worldSize
              << ", numCpuBytes: " << numCpuBytes << std::endl;

    // 创建 CommContext tensor
    auto commContextTensor = at::empty({GetCommContextSize()}, at::TensorOptions().dtype(at::kInt).device(at::kCPU));

    CommContext *commContext = GetCommContextFromTensor(commContextTensor);

    // 分配 host pinned memory
    void *hostPinnedAddr = nullptr;
    aclError ret = aclrtMallocHost(&hostPinnedAddr, numCpuBytes);
    TORCH_CHECK(ret == ACL_SUCCESS && hostPinnedAddr != nullptr, "aclrtMallocHost failed with error code: ", ret,
                ", numCpuBytes: ", numCpuBytes);

    std::cout << "[C++] Allocated host pinned memory: " << hostPinnedAddr << " (" << numCpuBytes << " bytes)"
              << std::endl;

    // 填充 CommContext（可选，根据需要）
    commContext->epRankId = rank;
    commContext->worldSize = worldSize;

    // 返回 dict
    py::dict result;
    result["commContext"] = commContextTensor;
    result["hostPinnedAddr"] = reinterpret_cast<uint64_t>(hostPinnedAddr);

    std::cout << "[C++] ElasticBufferInit completed, returning dict" << std::endl;
    return result;
}

/**
 * ElasticBufferDestroy - 释放 host pinned memory
 *
 * 参数：
 *   hostPinnedAddr: host pinned memory 地址（uint64_t）
 */
void ElasticBufferDestroy(uint64_t hostPinnedAddr)
{
    std::cout << "[C++] ElasticBufferDestroy called" << std::endl;

    if (hostPinnedAddr != 0) {
        void *addr = reinterpret_cast<void *>(hostPinnedAddr);
        std::cout << "[C++] Freeing host pinned memory: " << addr << std::endl;

        aclError ret = aclrtFreeHost(addr);
        TORCH_CHECK(ret == ACL_SUCCESS, "aclrtFreeHost failed with error code: ", ret,
                    ", hostPinnedAddr: ", hostPinnedAddr);

        std::cout << "[C++] Host pinned memory freed successfully" << std::endl;
    }
}

/**
 * ElasticBufferBarrier - 调用 HCCL 底层的 HcclBarrier
 *
 * 参数：
 *   groupName: HCCL 通信组名称
 *   withDeviceSync: 是否进行设备同步
 */
void ElasticBufferBarrier(const std::string &groupName, bool withDeviceSync)
{
    std::cout << "[C++] ElasticBufferBarrier called" << std::endl;

    // 初始化 HCCL 函数（首次调用时）
    static bool initialized = false;
    if (!initialized) {
        InitHcclFunctions();
        initialized = true;
    }

    // 设备同步（可选）
    if (withDeviceSync) {
        aclError aclRet = aclrtSynchronizeDevice();
        TORCH_CHECK(aclRet == ACL_SUCCESS, "aclrtSynchronizeDevice failed with error code: ", aclRet);
    }

    // 通过 groupName 获取 HcclComm
    HcclComm comm = nullptr;
    HcclResult ret = HcclCommGetHandleWithNameFunc(groupName.c_str(), &comm);
    TORCH_CHECK(ret == HCCL_SUCCESS && comm != nullptr, "HcclCommGetHandleWithName failed with error code: ", ret,
                ", groupName: ", groupName);

    // 获取当前 NPU stream
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
    // 调用 HcclBarrier
    ret = HcclBarrierFunc(comm, stream);
    TORCH_CHECK(ret == HCCL_SUCCESS, "HcclBarrier failed with error code: ", ret, ", comm: ", comm,
                ", stream: ", stream);

    // 再次设备同步（可选）
    if (withDeviceSync) {
        aclError aclRet = aclrtSynchronizeDevice();
        TORCH_CHECK(aclRet == ACL_SUCCESS, "aclrtSynchronizeDevice failed with error code: ", aclRet);
    }

    std::cout << "[C++] ElasticBufferBarrier completed" << std::endl;
}

/**
 * ElasticBufferWrite - 将 CPU tensor 数据拷贝到 host pinned memory
 *
 * 参数：
 *   hostPinnedAddr: host pinned memory 地址（uint64_t）
 *   storage: 要写入的数据（CPU tensor）
 */
void ElasticBufferWrite(uint64_t hostPinnedAddr, const at::Tensor &storage)
{
    std::cout << "[C++] ElasticBufferWrite called" << std::endl;

    TORCH_CHECK(storage.is_cpu(), "Storage tensor must be on CPU");
    TORCH_CHECK(storage.is_contiguous(), "Storage tensor must be contiguous");
    TORCH_CHECK(hostPinnedAddr != 0, "Host pinned memory address is null");

    void *dstAddr = reinterpret_cast<void *>(hostPinnedAddr);
    void *srcAddr = storage.data_ptr();
    size_t dataSize = storage.nbytes();

    std::cout << "[C++] Copying " << dataSize << " bytes to host pinned memory" << std::endl;

    memcpy(dstAddr, srcAddr, dataSize);

    // 验证拷贝结果（调试用）
    if (dataSize <= 32) {
        std::cout << "[C++] Verification: first few bytes copied:" << std::endl;
        unsigned char *dstBytes = static_cast<unsigned char *>(dstAddr);
        std::cout << "[C++]   ";
        for (size_t i = 0; i < dataSize && i < 16; ++i) {
            std::cout << std::hex << std::setw(2) << std::setfill('0') << static_cast<unsigned int>(dstBytes[i]) << " ";
        }
        std::cout << std::dec << std::endl;
    }

    std::cout << "[C++] ElasticBufferWrite completed" << std::endl;
}

/**
 * EngramFetch - 通过 RDMA 从远端 rank 抓取数据
 *
 * 参数：
 *   commContextTensor: CommContext tensor（用于 ACLNN API）
 *   indices: 全局索引（NPU tensor）
 *   hiddenSize: 隐藏层维度
 *   numEngramEntries: Engram 条目数（用于计算偏移量）
 *   dtypeVal: 数据类型枚举值（11 = bfloat16）
 *
 * 返回：
 *   callable，调用后返回 fetched tensor
 */
std::function<at::Tensor()> EngramFetch(const at::Tensor &commContextTensor, const at::Tensor &indices,
                                        int64_t hiddenSize, int64_t numEngramEntries, int64_t dtypeVal)
{
    std::cout << "[C++] EngramFetch called" << std::endl;
    std::cout << "[C++] hiddenSize: " << hiddenSize << ", numEngramEntries: " << numEngramEntries
              << ", dtypeVal: " << dtypeVal << std::endl;

    // 创建输出 tensor
    int64_t numTokens = indices.size(0);
    auto fetched =
        at::empty({numTokens, hiddenSize}, at::TensorOptions().dtype(at::kBFloat16).device(indices.device()));

    std::cout << "[C++] Calling ACLNN_CMD(aclnnEngramFetch)" << std::endl;

    // 使用 ACLNN_CMD 宏调用 issue kernel
    // 传入 CommContext tensor 和其他参数
    // ACLNN_CMD(aclnnEngramFetch, commContextTensor, indices, fetched, hiddenSize, numEngramEntries, dtypeVal);
    ACLNN_CMD(aclnnEngramFetch, commContextTensor, indices, fetched);

    std::cout << "[C++] Issue kernel completed" << std::endl;

    // 返回 callable
    return [=]() {
        std::cout << "[C++] Calling ACLNN_CMD(aclnnEngramFetchWait)" << std::endl;

        ACLNN_CMD(aclnnEngramFetchWait, commContextTensor);

        std::cout << "[C++] Wait kernel completed" << std::endl;
        return fetched;
    };
}

} // namespace op_api

// PyBind11 module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("npu_elastic_buffer_init", &op_api::ElasticBufferInit,
          "Initialize ElasticBuffer with host pinned memory, returns dict");

    m.def("npu_elastic_buffer_destroy", &op_api::ElasticBufferDestroy,
          "Destroy ElasticBuffer and free host pinned memory");

    m.def("npu_elastic_buffer_write", &op_api::ElasticBufferWrite, "Write CPU tensor data to host pinned memory");

    m.def("npu_elastic_buffer_barrier", &op_api::ElasticBufferBarrier,
          "Call HCCL barrier with group name and optional device sync");

    m.def("npu_engram_fetch", &op_api::EngramFetch, "EngramFetch with ACLNN_CMD");
}