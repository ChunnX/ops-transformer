/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * ElasticBuffer C++ binding - 使用 ACLNN_CMD 宏调用
 */

#include <torch/extension.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <iostream>
#include <functional>

// CANN ACL Runtime API（host memory allocation）
#include "acl/acl.h"

// HCCL types
#include "hccl/hccl_types.h"

// ACLNN common utilities
#include "aclnn_common.h"

// torch_npu stream utilities
#include "torch_npu/csrc/core/npu/NPUStream.h"

namespace op_api {

constexpr uint32_t HCCL_MAX_RANK_SIZE = 1024;

struct CommContext {
    uint64_t epRankId = 0;
    uint64_t kfcContextAddr = 0;
    uint64_t virtualAddrList[HCCL_MAX_RANK_SIZE];
    uint64_t hcommHandle_[HCCL_MAX_RANK_SIZE];
};

struct ElasticBufferHandle {
    CommContext mc2Context;

    // Host and Device memory management
    void *hostPinnedAddr = nullptr;
    uint64_t hidden = 0;
    uint64_t num_engram_entries = 0;
};

static int64_t GetElasticBufferHandleSize()
{
    return sizeof(ElasticBufferHandle) / sizeof(int32_t);
}

static void CopyHandleToTensor(const ElasticBufferHandle &handle, at::Tensor &tensor)
{
    at::Tensor hostHandle = at::from_blob(const_cast<ElasticBufferHandle *>(&handle),
                                          {sizeof(ElasticBufferHandle) / sizeof(int32_t)}, at::kInt);
    tensor.copy_(hostHandle);
}

static ElasticBufferHandle *GetHandleFromTensor(at::Tensor &tensor)
{
    auto handleData = tensor.accessor<int32_t, 1>();
    return reinterpret_cast<ElasticBufferHandle *>(handleData.data());
}

static const ElasticBufferHandle *GetHandleFromTensorConst(const at::Tensor &tensor)
{
    auto handleData = tensor.accessor<int32_t, 1>();
    return reinterpret_cast<const ElasticBufferHandle *>(handleData.data());
}

/**
 * ElasticBuffer初始化 - 分配 host pinned memory 并填充 handle
 */
at::Tensor npu_elastic_buffer_init(std::string group_name, uint64_t rank, uint64_t world_size, uint64_t num_bytes, uint64_t num_cpu_bytes)
{
    // 创建 handle tensor
    auto handle_tensor =
        at::empty({GetElasticBufferHandleSize()}, at::TensorOptions().dtype(at::kInt).device(at::kCPU));

    ElasticBufferHandle *handle = GetHandleFromTensor(handle_tensor);

    // 分配 host pinned memory
    void *host_pinned_addr = nullptr;
    aclError ret = aclrtMallocHost(&host_pinned_addr, num_cpu_bytes);
    TORCH_CHECK(ret == ACL_SUCCESS && host_pinned_addr != nullptr, "aclrtMallocHost failed with error code: ", ret,
                ", num_cpu_bytes: ", num_cpu_bytes);

    // 填充 handle 字段
    handle->hostPinnedAddr = host_pinned_addr;
    return handle_tensor;
}

/**
 * ElasticBuffer销毁 - 释放 host pinned memory
 */
void npu_elastic_buffer_destroy(at::Tensor &handle_tensor)
{
    std::cout << "[C++] npu_elastic_buffer_destroy called" << std::endl;

    ElasticBufferHandle *handle = GetHandleFromTensor(handle_tensor);

    if (handle->hostPinnedAddr != nullptr) {
        std::cout << "[C++] Freeing host pinned memory: " << handle->hostPinnedAddr << std::endl;

        aclError ret = aclrtFreeHost(handle->hostPinnedAddr);
        TORCH_CHECK(ret == ACL_SUCCESS, "aclrtFreeHost failed with error code: ", ret,
                    ", host_addr: ", handle->hostPinnedAddr);

        std::cout << "[C++] Host pinned memory freed successfully" << std::endl;
        handle->hostPinnedAddr = nullptr;
    }
}

/**
 * ElasticBufferBarrier - 调用 HCCL 底层的 HcclBarrier，支持设备同步
 */
void npu_elastic_buffer_barrier(const std::string &group_name, bool with_device_sync)
{
    // 1. 设备同步（如果需要）
    if (with_device_sync) {
        aclError acl_ret = aclrtSynchronizeDevice();
        TORCH_CHECK(acl_ret == ACL_SUCCESS, "aclrtSynchronizeDevice failed with error code: ", acl_ret);
    }

    // 2. 动态加载 HcclBarrier 函数
    using HcclBarrierFuncType = HcclResult (*)(HcclComm, aclrtStream);
    static HcclBarrierFuncType HcclBarrierFunc = nullptr;

    if (HcclBarrierFunc == nullptr) {
        void *hccl_handler = GetOpApiLibHandler("libhccl.so");
        TORCH_CHECK(hccl_handler != nullptr, "Failed to load libhccl.so");
        HcclBarrierFunc =
            reinterpret_cast<HcclBarrierFuncType>(GetOpApiFuncAddrInLib(hccl_handler, "libhccl.so", "HcclBarrier"));
        TORCH_CHECK(HcclBarrierFunc != nullptr, "Failed to get HcclBarrier function address");
    }

    // 3. 动态加载 HcclCommGetHandleWithName 函数
    using HcclCommGetHandleWithNameFuncType = HcclResult (*)(const char *, HcclComm *);
    static HcclCommGetHandleWithNameFuncType HcclCommGetHandleWithNameFunc = nullptr;

    if (HcclCommGetHandleWithNameFunc == nullptr) {
        void *hccl_handler = GetOpApiLibHandler("libhccl.so");
        TORCH_CHECK(hccl_handler != nullptr, "Failed to load libhccl.so");

        HcclCommGetHandleWithNameFunc = reinterpret_cast<HcclCommGetHandleWithNameFuncType>(
            GetOpApiFuncAddrInLib(hccl_handler, "libhccl.so", "HcclCommGetHandleWithName"));

        TORCH_CHECK(HcclCommGetHandleWithNameFunc != nullptr,
                    "Failed to get HcclCommGetHandleWithName function address");
    }

    // 4. 通过 group_name 获取 HcclComm
    HcclComm comm = nullptr;
    HcclResult ret = HcclCommGetHandleWithNameFunc(group_name.c_str(), &comm);

    TORCH_CHECK(ret == HCCL_SUCCESS && comm != nullptr, "HcclCommGetHandleWithName failed with error code: ", ret,
                ", group_name: ", group_name);

    // 5. 获取当前 NPU stream
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();

    // 6. 调用 HcclBarrier
    ret = HcclBarrierFunc(comm, stream);

    TORCH_CHECK(ret == HCCL_SUCCESS, "HcclBarrier failed with error code: ", ret, ", comm: ", comm,
                ", stream: ", stream);

    // 7. 再次设备同步
    if (with_device_sync) {
        aclError acl_ret = aclrtSynchronizeDevice();
        TORCH_CHECK(acl_ret == ACL_SUCCESS, "aclrtSynchronizeDevice failed with error code: ", acl_ret);
    }
}

/**
 * EngramWrite - 将 CPU tensor 数据拷贝到 host pinned memory
 * Host-to-Host 内存拷贝
 */
void npu_elastic_buffer_write(const at::Tensor &handle_tensor, const at::Tensor &storage)
{
    std::cout << "[C++] npu_elastic_buffer_write called" << std::endl;

    // 验证 storage 必须在 CPU 上
    TORCH_CHECK(storage.is_cpu(), "Storage tensor must be on CPU");

    // 验证 storage 必须是连续的
    TORCH_CHECK(storage.is_contiguous(), "Storage tensor must be contiguous");

    ElasticBufferHandle *handle = GetHandleFromTensor(const_cast<at::Tensor &>(handle_tensor));

    TORCH_CHECK(handle->hostPinnedAddr != nullptr, "Host pinned memory not allocated");

    // 计算数据大小
    size_t data_size = storage.numel() * storage.element_size();

    // 获取 destination 地址
    void *dst_addr = static_cast<char *>(handle->hostPinnedAddr);

    // 获取 source 数据指针
    void *src_addr = storage.data_ptr();

    memcpy(dst_addr, src_addr, data_size);

    // 可选：验证拷贝结果（调试用）
    if (data_size <= 32) {
        std::cout << "[C++] Verification: first few bytes copied:" << std::endl;
        char *dst_bytes = static_cast<char *>(dst_addr);
        std::cout << "[C++]   ";
        for (size_t i = 0; i < data_size && i < 16; ++i) {
            std::cout << std::hex << (int)dst_bytes[i] << " ";
        }
        std::cout << std::dec << std::endl;
    }
}

/**
 * EngramFetch - 使用 ACLNN_CMD 宏调用
 * 参考 DeepEP：返回 callable
 */
std::function<at::Tensor()> npu_engram_fetch(const at::Tensor &handle, const at::Tensor &indices, int64_t hidden_size,
                                             int64_t dtype_val)
{
    std::cout << "[C++] npu_engram_fetch called" << std::endl;

    // 创建输出 tensor
    int64_t num_tokens = indices.size(0);
    auto fetched =
        at::empty({num_tokens, hidden_size}, at::TensorOptions().dtype(at::kBFloat16).device(indices.device()));

    std::cout << "[C++] Calling ACLNN_CMD(aclnnEngramFetch)" << std::endl;

    // 使用 ACLNN_CMD 宏调用 issue kernel
    // 宏会自动处理：GetWorkspaceSize + workspace分配 + Execute
    ACLNN_CMD(aclnnEngramFetch, handle, indices, fetched);

    std::cout << "[C++] Issue kernel completed" << std::endl;

    // 返回 callable，触发 wait phase
    return [=]() {
        std::cout << "[C++] Calling ACLNN_CMD(aclnnEngramFetchWait)" << std::endl;

        // 使用 ACLNN_CMD 宏调用 wait kernel
        ACLNN_CMD(aclnnEngramFetchWait, handle);

        std::cout << "[C++] Wait kernel completed" << std::endl;
        return fetched;
    };
}

} // namespace op_api

// PyBind11 module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("npu_elastic_buffer_init", &op_api::npu_elastic_buffer_init,
          "Initialize ElasticBuffer with host pinned memory");
    m.def("npu_elastic_buffer_destroy", &op_api::npu_elastic_buffer_destroy,
          "Destroy ElasticBuffer and free host pinned memory");
    m.def("npu_elastic_buffer_write", &op_api::npu_elastic_buffer_write, "Write CPU tensor data to host pinned memory");
    m.def("npu_elastic_buffer_barrier", &op_api::npu_elastic_buffer_barrier,
          "Call HCCL barrier with group name and optional device sync");
    m.def("npu_engram_fetch", &op_api::npu_engram_fetch, "EngramFetch with ACLNN_CMD");
}