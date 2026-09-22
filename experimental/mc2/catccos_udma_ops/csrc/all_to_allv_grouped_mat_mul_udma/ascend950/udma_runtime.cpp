/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#include "udma_runtime.h"

#include "shmem.h"

#include <ATen/ATen.h>
#include <torch/library.h>

#include <algorithm>
#include <cstring>

namespace OpsTransformer::CatccosUdma {
namespace {

int32_t FillInitAttributes(int32_t rank, int32_t worldSize, uint64_t heapSize, const char *initMethod,
                           aclshmemx_init_attr_t *attributes, aclshmemx_uniqueid_t *uniqueId)
{
    // Preserve SHMEM's optional-attribute ABI version and timeout defaults.
    // Its version encodes sizeof(aclshmem_init_optional_attr_t), not the
    // enclosing aclshmemx_init_attr_t.
    *attributes = aclshmemx_init_attr_t{};
    *uniqueId = aclshmemx_uniqueid_t{};
    const size_t length = std::min(std::strlen(initMethod), sizeof(attributes->ip_port) - 1);
    std::copy_n(initMethod, length, attributes->ip_port);
    attributes->ip_port[length] = '\0';
    attributes->my_pe = rank;
    attributes->n_pes = worldSize;
    attributes->local_mem_size = heapSize;
    attributes->option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_UDMA;
    uniqueId->version = ACLSHMEM_UNIQUEID_VERSION;
    uniqueId->my_pe = rank;
    uniqueId->n_pes = worldSize;
    attributes->comm_args = reinterpret_cast<void *>(uniqueId);
    return ACLSHMEM_SUCCESS;
}

} // namespace

UdmaRuntime &UdmaRuntime::Instance()
{
    // Keep the allocator state alive through Python/DSO teardown so a late
    // Tensor deleter never references an already-destroyed singleton.
    static UdmaRuntime *runtime = new UdmaRuntime();
    return *runtime;
}

void UdmaRuntime::Initialize(int32_t rank, int32_t worldSize, const std::string &initMethod, uint64_t heapSize)
{
    std::lock_guard<std::mutex> lock(mutex_);
    TORCH_CHECK(rank >= 0 && rank < worldSize, "invalid UDMA rank ", rank, " for world size ", worldSize);
    TORCH_CHECK(worldSize > 0 && worldSize <= 8, "UDMA world size must be in [1, 8]");
    TORCH_CHECK(!initMethod.empty(), "UDMA init method must not be empty");
    TORCH_CHECK(heapSize > kKernelWorkspaceBytes, "UDMA heap is too small: ", heapSize);

    if (initialized_) {
        TORCH_CHECK(rank == rank_ && worldSize == worldSize_ && heapSize == heapSize_ && initMethod == initMethod_,
                    "UDMA runtime is already initialized with a different configuration");
        return;
    }

    aclshmemx_init_attr_t attributes;
    aclshmemx_uniqueid_t uniqueId;
    const int32_t attrStatus = FillInitAttributes(
        rank, worldSize, heapSize, initMethod.c_str(), &attributes, &uniqueId);
    TORCH_CHECK(attrStatus == ACLSHMEM_SUCCESS, "failed to construct UDMA SHMEM attributes: ", attrStatus);
    const int32_t initStatus = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);
    TORCH_CHECK(initStatus == ACLSHMEM_SUCCESS, "aclshmemx_init_attr failed: ", initStatus);

    workspace_ = static_cast<uint8_t *>(aclshmem_calloc(1, kKernelWorkspaceBytes));
    if (workspace_ == nullptr) {
        shmem_finalize();
        TORCH_CHECK(false, "failed to allocate ", kKernelWorkspaceBytes,
                    " bytes of symmetric UDMA workspace");
    }
    fftsAddress_ = shmemx_get_ffts_config();

    rank_ = rank;
    worldSize_ = worldSize;
    heapSize_ = heapSize;
    initMethod_ = initMethod;
    initialized_ = true;
}

void UdmaRuntime::EnsureInitialized() const
{
    TORCH_CHECK(initialized_, "UDMA runtime is not initialized; call init_udma_group() before the operator");
}

void *UdmaRuntime::AllocateSymmetric(size_t bytes)
{
    std::lock_guard<std::mutex> lock(mutex_);
    EnsureInitialized();
    TORCH_CHECK(bytes > 0, "symmetric allocation size must be positive");
    void *pointer = aclshmem_malloc(bytes);
    TORCH_CHECK(pointer != nullptr, "failed to allocate ", bytes,
                " bytes from the symmetric UDMA heap; increase heap_size if necessary");
    externalAllocations_.insert(pointer);
    return pointer;
}

void UdmaRuntime::FreeSymmetric(void *pointer)
{
    if (pointer == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    const auto allocation = externalAllocations_.find(pointer);
    if (allocation == externalAllocations_.end()) {
        return;
    }
    if (lastStream_ != nullptr) {
        const aclError syncStatus = aclrtSynchronizeStream(lastStream_);
        if (syncStatus != ACL_SUCCESS) {
            TORCH_WARN("failed to synchronize before freeing symmetric memory: ", syncStatus);
        }
    }
    aclshmem_free(pointer);
    externalAllocations_.erase(allocation);
}

bool UdmaRuntime::IsSymmetric(const void *pointer) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    EnsureInitialized();
    return externalAllocations_.find(const_cast<void *>(pointer)) != externalAllocations_.end();
}

void UdmaRuntime::RecordStream(aclrtStream stream)
{
    std::lock_guard<std::mutex> lock(mutex_);
    EnsureInitialized();
    if (lastStream_ != nullptr && lastStream_ != stream) {
        const aclError syncStatus = aclrtSynchronizeStream(lastStream_);
        TORCH_CHECK(syncStatus == ACL_SUCCESS, "failed to synchronize the previous UDMA stream: ", syncStatus);
    }
    lastStream_ = stream;
}

void UdmaRuntime::Finalize(aclrtStream stream)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (!initialized_) {
        return;
    }
    TORCH_CHECK(externalAllocations_.empty(), "cannot finalize UDMA while ", externalAllocations_.size(),
                " symmetric output tensor(s) are still alive; delete them before destroy_udma_group()");
    aclrtStream streamToWait = lastStream_ != nullptr ? lastStream_ : stream;
    if (streamToWait != nullptr) {
        const aclError syncStatus = aclrtSynchronizeStream(streamToWait);
        TORCH_CHECK(syncStatus == ACL_SUCCESS, "failed to synchronize UDMA stream during finalization: ", syncStatus);
    }
    if (workspace_ != nullptr) {
        aclshmem_free(workspace_);
    }
    const int32_t status = shmem_finalize();

    initialized_ = false;
    rank_ = -1;
    worldSize_ = 0;
    heapSize_ = 0;
    initMethod_.clear();
    workspace_ = nullptr;
    lastStream_ = nullptr;
    fftsAddress_ = 0;
    TORCH_CHECK(status == ACLSHMEM_SUCCESS, "shmem_finalize failed: ", status);
}

uint8_t *UdmaRuntime::Workspace() const
{
    EnsureInitialized();
    return workspace_;
}

uint64_t UdmaRuntime::FftsAddress() const
{
    EnsureInitialized();
    return fftsAddress_;
}

int32_t UdmaRuntime::Rank() const
{
    EnsureInitialized();
    return rank_;
}

int32_t UdmaRuntime::WorldSize() const
{
    EnsureInitialized();
    return worldSize_;
}

uint64_t UdmaRuntime::HeapSize() const
{
    EnsureInitialized();
    return heapSize_;
}

} // namespace OpsTransformer::CatccosUdma
