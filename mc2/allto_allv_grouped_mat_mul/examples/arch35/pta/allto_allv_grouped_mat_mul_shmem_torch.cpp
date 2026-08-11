/*
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <string>
#include <vector>

#include <ATen/ATen.h>
#include <c10/core/DeviceType.h>
#include <torch/library.h>
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include "acl/acl.h"
#include "aclnnop/aclnn_allto_allv_grouped_mat_mul_v2.h"
#include "shmem.h"

namespace allto_allv_grouped_mat_mul_shmem_pta {
namespace {
constexpr int64_t MAX_WORLD_SIZE = 8;
constexpr int64_t MAX_LOCAL_EXPERTS = 64;

struct BridgeState {
    bool initialized = false;
    int64_t rank = -1;
    int64_t worldSize = 0;
    int64_t heapBytes = 0;
    std::string ipPort;
    void *context = nullptr;
};

BridgeState &GetState()
{
    static BridgeState state;
    return state;
}

std::mutex &GetStateMutex()
{
    static std::mutex mutex;
    return mutex;
}

aclrtStream GetCurrentStream()
{
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
    TORCH_CHECK(stream != nullptr, "current torch_npu stream is null");
    return stream;
}

void CheckAcl(int32_t status, const char *api, int64_t rank)
{
    TORCH_CHECK(status == ACL_SUCCESS, api, " failed on rank ", rank, ", ret=", status);
}

void CheckShmem(int64_t status, const char *api, int64_t rank)
{
    TORCH_CHECK(status == ACLSHMEM_SUCCESS, api, " failed on rank ", rank, ", ret=", status);
}

void FillShmemAttr(int64_t rank, int64_t worldSize, int64_t heapBytes, const std::string &ipPort,
                   aclshmemx_init_attr_t &attr, aclshmemx_uniqueid_t &uniqueId)
{
    std::memset(&attr, 0, sizeof(attr));
    std::memset(&uniqueId, 0, sizeof(uniqueId));
    TORCH_CHECK(ipPort.size() < sizeof(attr.ip_port), "ip_port is too long: ", ipPort.size());
    std::copy(ipPort.begin(), ipPort.end(), attr.ip_port);
    attr.ip_port[ipPort.size()] = '\0';
    attr.my_pe = static_cast<int32_t>(rank);
    attr.n_pes = static_cast<int32_t>(worldSize);
    attr.local_mem_size = static_cast<uint64_t>(heapBytes);
    const int32_t attrVersion = (1 << 16) + static_cast<int32_t>(sizeof(aclshmemx_init_attr_t));
    attr.option_attr = {attrVersion, ACLSHMEM_DATA_OP_UDMA, DEFAULT_TIMEOUT, DEFAULT_TIMEOUT, DEFAULT_TIMEOUT};
    uniqueId.my_pe = static_cast<int32_t>(rank);
    uniqueId.n_pes = static_cast<int32_t>(worldSize);
    attr.comm_args = reinterpret_cast<void *>(&uniqueId);
}

std::vector<int64_t> ContiguousStrides(const std::vector<int64_t> &shape)
{
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t index = static_cast<int64_t>(shape.size()) - 2; index >= 0; --index) {
        TORCH_CHECK(shape[index + 1] == 0 || strides[index + 1] <= std::numeric_limits<int64_t>::max() /
                                                          std::max<int64_t>(shape[index + 1], 1),
                    "tensor stride overflow");
        strides[index] = strides[index + 1] * shape[index + 1];
    }
    return strides;
}

aclTensor *CreateBf16Tensor(void *address, const std::vector<int64_t> &shape)
{
    const std::vector<int64_t> strides = ContiguousStrides(shape);
    return aclCreateTensor(shape.data(), shape.size(), ACL_BF16, strides.data(), 0, ACL_FORMAT_ND, shape.data(),
                           shape.size(), address);
}

using TensorGuard = std::unique_ptr<aclTensor, decltype(&aclDestroyTensor)>;
using IntArrayGuard = std::unique_ptr<aclIntArray, decltype(&aclDestroyIntArray)>;

void ValidateTensor(const at::Tensor &tensor, const char *name, int64_t expectedDim)
{
    TORCH_CHECK(tensor.defined(), name, " must be defined");
    TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1, name, " must be an NPU tensor, got ",
                tensor.device());
    TORCH_CHECK(tensor.scalar_type() == at::kBFloat16, name, " must be BF16, got ", tensor.scalar_type());
    TORCH_CHECK(tensor.dim() == expectedDim, name, " must be ", expectedDim, "D, got ", tensor.dim(), "D");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    const aclFormat format = static_cast<aclFormat>(at_npu::native::get_npu_format(tensor));
    TORCH_CHECK(format == ACL_FORMAT_ND, name, " must use ND storage format, got ", static_cast<int32_t>(format));
}

int64_t SumAndValidateCounts(at::IntArrayRef counts, const char *name)
{
    int64_t sum = 0;
    for (const int64_t value : counts) {
        TORCH_CHECK(value >= 0, name, " contains a negative value: ", value);
        TORCH_CHECK(sum <= std::numeric_limits<int64_t>::max() - value, name, " sum overflows int64");
        sum += value;
    }
    return sum;
}

void FreeSymmetric(void *&workspace, void *&symmetricX, aclrtStream stream)
{
    if (workspace == nullptr && symmetricX == nullptr) {
        return;
    }
    if (stream != nullptr) {
        (void)aclrtSynchronizeStream(stream);
    }
    aclshmem_barrier_all();
    if (workspace != nullptr) {
        shmem_free(workspace);
        workspace = nullptr;
    }
    if (symmetricX != nullptr) {
        shmem_free(symmetricX);
        symmetricX = nullptr;
    }
}
} // namespace

int64_t Init(int64_t rank, int64_t worldSize, int64_t heapBytes, const std::string &ipPort)
{
    std::lock_guard<std::mutex> lock(GetStateMutex());
    BridgeState &state = GetState();
    if (state.initialized) {
        TORCH_CHECK(state.rank == rank && state.worldSize == worldSize && state.heapBytes == heapBytes &&
                        state.ipPort == ipPort,
                    "SHMEM is already initialized with different parameters on rank ", state.rank);
        return 0;
    }

    TORCH_CHECK(worldSize >= 1 && worldSize <= MAX_WORLD_SIZE, "world_size must be in [1, 8], got ", worldSize);
    TORCH_CHECK(rank >= 0 && rank < worldSize, "rank must be in [0, world_size), got ", rank);
    TORCH_CHECK(heapBytes > 0, "heap_bytes must be positive, got ", heapBytes);
    TORCH_CHECK(!ipPort.empty(), "ip_port must not be empty");

    CheckShmem(aclshmemx_set_conf_store_tls(false, nullptr, 0), "aclshmemx_set_conf_store_tls", rank);
    aclshmemx_init_attr_t attr{};
    aclshmemx_uniqueid_t uniqueId{};
    FillShmemAttr(rank, worldSize, heapBytes, ipPort, attr, uniqueId);
    CheckShmem(aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attr), "aclshmemx_init_attr", rank);

    TORCH_CHECK(shmem_my_pe() == rank, "SHMEM rank mismatch: expected ", rank, ", got ", shmem_my_pe());
    TORCH_CHECK(shmem_n_pes() == worldSize, "SHMEM world-size mismatch: expected ", worldSize, ", got ", shmem_n_pes());
    void *context = reinterpret_cast<void *>(shmemx_get_ffts_config());
    TORCH_CHECK(context != nullptr, "shmemx_get_ffts_config returned null on rank ", rank);

    state.initialized = true;
    state.rank = rank;
    state.worldSize = worldSize;
    state.heapBytes = heapBytes;
    state.ipPort = ipPort;
    state.context = context;
    return 0;
}

int64_t Barrier()
{
    std::lock_guard<std::mutex> lock(GetStateMutex());
    const BridgeState &state = GetState();
    TORCH_CHECK(state.initialized, "SHMEM is not initialized; call init first");
    aclshmem_barrier_all();
    return 0;
}

int64_t Finalize()
{
    std::lock_guard<std::mutex> lock(GetStateMutex());
    BridgeState &state = GetState();
    if (!state.initialized) {
        return 0;
    }
    const int64_t rank = state.rank;
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
    if (stream != nullptr) {
        CheckAcl(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream(finalize)", rank);
    }
    const int64_t status = shmem_finalize();
    state = BridgeState{};
    CheckShmem(status, "shmem_finalize", rank);
    return 0;
}

at::Tensor Run(const at::Tensor &x, const at::Tensor &weight, at::IntArrayRef sendCounts,
               at::IntArrayRef recvCounts)
{
    std::lock_guard<std::mutex> lock(GetStateMutex());
    const BridgeState &state = GetState();
    TORCH_CHECK(state.initialized, "SHMEM is not initialized; call init first");
    ValidateTensor(x, "x", 2);
    ValidateTensor(weight, "weight", 3);
    TORCH_CHECK(x.device() == weight.device(), "x and weight must be on the same NPU, got ", x.device(), " and ",
                weight.device());

    c10_npu::OptionalNPUGuard deviceGuard(x.device());
    TORCH_CHECK(shmem_my_pe() == state.rank, "SHMEM rank changed after initialization on rank ", state.rank);
    TORCH_CHECK(shmem_n_pes() == state.worldSize, "SHMEM world size changed after initialization on rank ",
                state.rank);

    const int64_t m = x.size(0);
    const int64_t k = x.size(1);
    const int64_t experts = weight.size(0);
    const int64_t n = weight.size(2);
    TORCH_CHECK(m > 0 && k > 0 && n > 0, "M, K, and N must be positive, got M=", m, ", K=", k, ", N=", n);
    TORCH_CHECK(experts >= 1 && experts <= MAX_LOCAL_EXPERTS, "local experts must be in [1, 64], got ", experts);
    TORCH_CHECK(weight.size(1) == k, "K mismatch: x has ", k, ", weight has ", weight.size(1));
    const int64_t expectedCounts = state.worldSize * experts;
    TORCH_CHECK(static_cast<int64_t>(sendCounts.size()) == expectedCounts, "send_counts length must be ",
                expectedCounts, ", got ", sendCounts.size());
    TORCH_CHECK(static_cast<int64_t>(recvCounts.size()) == expectedCounts, "recv_counts length must be ",
                expectedCounts, ", got ", recvCounts.size());
    TORCH_CHECK(SumAndValidateCounts(sendCounts, "send_counts") == m, "sum(send_counts) must equal M=", m);
    const int64_t recvM = SumAndValidateCounts(recvCounts, "recv_counts");

    at::Tensor y = at::empty({recvM, n}, x.options());
    ValidateTensor(y, "y", 2);
    aclrtStream stream = GetCurrentStream();
    const uint64_t xBytes = static_cast<uint64_t>(x.numel()) * static_cast<uint64_t>(x.element_size());
    TORCH_CHECK(xBytes <= static_cast<uint64_t>(std::numeric_limits<size_t>::max()), "x byte size exceeds size_t");

    void *symmetricX = aclshmem_malloc(static_cast<size_t>(xBytes));
    TORCH_CHECK(symmetricX != nullptr, "aclshmem_malloc(x) failed on rank ", state.rank, ", bytes=", xBytes);
    void *workspace = nullptr;
    try {
        CheckAcl(aclrtMemcpyAsync(symmetricX, static_cast<size_t>(xBytes), x.data_ptr(), static_cast<size_t>(xBytes),
                                  ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
                 "aclrtMemcpyAsync(x->symmetric)", state.rank);

        const std::vector<int64_t> xShape{x.size(0), x.size(1)};
        const std::vector<int64_t> weightShape{weight.size(0), weight.size(1), weight.size(2)};
        const std::vector<int64_t> yShape{y.size(0), y.size(1)};
        TensorGuard xDesc(CreateBf16Tensor(symmetricX, xShape), &aclDestroyTensor);
        TensorGuard weightDesc(CreateBf16Tensor(weight.data_ptr(), weightShape), &aclDestroyTensor);
        TensorGuard yDesc(CreateBf16Tensor(y.data_ptr(), yShape), &aclDestroyTensor);
        TORCH_CHECK(xDesc && weightDesc && yDesc, "aclCreateTensor failed on rank ", state.rank);

        const std::vector<int64_t> sendValues(sendCounts.begin(), sendCounts.end());
        const std::vector<int64_t> recvValues(recvCounts.begin(), recvCounts.end());
        IntArrayGuard sendArray(aclCreateIntArray(sendValues.data(), sendValues.size()), &aclDestroyIntArray);
        IntArrayGuard recvArray(aclCreateIntArray(recvValues.data(), recvValues.size()), &aclDestroyIntArray);
        TORCH_CHECK(sendArray && recvArray, "aclCreateIntArray failed on rank ", state.rank);

        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;
        CheckAcl(aclnnAlltoAllvGroupedMatMulV2GetWorkspaceSize(
                     xDesc.get(), weightDesc.get(), state.context, state.worldSize, sendArray.get(), recvArray.get(),
                     yDesc.get(), &workspaceSize, &executor),
                 "aclnnAlltoAllvGroupedMatMulV2GetWorkspaceSize", state.rank);
        TORCH_CHECK(workspaceSize > 0 && executor != nullptr, "V2 workspace query returned invalid metadata on rank ",
                    state.rank, ": workspaceSize=", workspaceSize, ", executor=", executor);
        workspace = aclshmem_calloc(1, workspaceSize);
        TORCH_CHECK(workspace != nullptr, "aclshmem_calloc(workspace) failed on rank ", state.rank, ", bytes=",
                    workspaceSize);

        // A non-repeatable aclOpExecutor is released automatically by the second-stage aclnn call.
        CheckAcl(aclnnAlltoAllvGroupedMatMulV2(workspace, workspaceSize, executor, stream),
                 "aclnnAlltoAllvGroupedMatMulV2", state.rank);
        CheckAcl(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream(run)", state.rank);
        aclshmem_barrier_all();
        shmem_free(workspace);
        workspace = nullptr;
        shmem_free(symmetricX);
        symmetricX = nullptr;
        return y;
    } catch (...) {
        FreeSymmetric(workspace, symmetricX, stream);
        throw;
    }
}
} // namespace allto_allv_grouped_mat_mul_shmem_pta

TORCH_LIBRARY(allto_allv_grouped_mat_mul_shmem_pta, m)
{
    m.def("init(int rank, int world_size, int heap_bytes, str ip_port) -> int");
    m.def("run(Tensor x, Tensor weight, int[] send_counts, int[] recv_counts) -> Tensor");
    m.def("barrier() -> int");
    m.def("finalize() -> int");
    m.impl("init", &allto_allv_grouped_mat_mul_shmem_pta::Init);
    m.impl("run", torch::kPrivateUse1, &allto_allv_grouped_mat_mul_shmem_pta::Run);
    m.impl("barrier", &allto_allv_grouped_mat_mul_shmem_pta::Barrier);
    m.impl("finalize", &allto_allv_grouped_mat_mul_shmem_pta::Finalize);
}
