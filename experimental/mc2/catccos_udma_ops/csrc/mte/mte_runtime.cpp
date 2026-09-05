/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#include "mte_runtime.h"

#include "shmem.h"
#include "platform/platform_ascendc.h"

#include <ATen/ATen.h>
#include <torch/library.h>

#include <algorithm>
#include <cstring>

namespace OpsTransformer::CatccosMte {
namespace {

int32_t FillInitAttributes(int32_t rank, int32_t worldSize, uint64_t heapSize, const char *initMethod,
                           aclshmemx_init_attr_t *attributes, aclshmemx_uniqueid_t *uniqueId)
{
    std::memset(attributes, 0, sizeof(*attributes));
    std::memset(uniqueId, 0, sizeof(*uniqueId));
    const size_t length = std::min(std::strlen(initMethod), sizeof(attributes->ip_port) - 1);
    std::copy_n(initMethod, length, attributes->ip_port);
    attributes->ip_port[length] = '\0';
    attributes->my_pe = rank;
    attributes->n_pes = worldSize;
    attributes->local_mem_size = heapSize;
    const int32_t version = (1 << 16) + sizeof(aclshmemx_init_attr_t);
    attributes->option_attr = {version, ACLSHMEM_DATA_OP_MTE, DEFAULT_TIMEOUT, DEFAULT_TIMEOUT, DEFAULT_TIMEOUT};
    uniqueId->my_pe = rank;
    uniqueId->n_pes = worldSize;
    attributes->comm_args = reinterpret_cast<void *>(uniqueId);
    return ACLSHMEM_SUCCESS;
}

} // namespace

MteRuntime &MteRuntime::Instance()
{
    // Keep the allocator state alive through Python/DSO teardown so a late
    // Tensor deleter never references an already-destroyed singleton.
    static MteRuntime *runtime = new MteRuntime();
    return *runtime;
}

std::unique_lock<std::mutex> MteRuntime::AcquireLaunchGuard()
{
    return std::unique_lock<std::mutex>(launchMutex_);
}

void MteRuntime::Initialize(int32_t rank, int32_t worldSize, const std::string &initMethod,
                            uint64_t heapSize, aclrtStream stream)
{
    auto launchGuard = AcquireLaunchGuard();
    std::lock_guard<std::mutex> lock(mutex_);
    TORCH_CHECK(!failed_, "MTE encountered an unfinished device operation; restart all worker processes");
    TORCH_CHECK(rank >= 0 && rank < worldSize, "invalid MTE rank ", rank, " for world size ", worldSize);
    TORCH_CHECK(worldSize > 0 && worldSize <= 8, "MTE world size must be in [1, 8]");
    TORCH_CHECK(!initMethod.empty(), "MTE init method must not be empty");
    TORCH_CHECK(heapSize > kKernelWorkspaceBytes, "MTE heap is too small: ", heapSize);

    auto *platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    TORCH_CHECK(platform != nullptr, "MTE platform query is unavailable");
    uint64_t physicalUbBytes = 0;
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, physicalUbBytes);
    // The pinned CATLASS Ascend950 architecture exposes 248 KiB to kernels.
    // Platform providers may report that usable value or the 256 KiB physical
    // capacity; do not reject a valid 248 KiB report or subtract its reserve twice.
    constexpr uint64_t requiredUsableUbBytes = kMtePhysicalUbBytes - kMteSystemReservedBytes;
    TORCH_CHECK(physicalUbBytes >= requiredUsableUbBytes,
                "MTE requires at least ", requiredUsableUbBytes, " bytes of usable UB; platform reported ",
                physicalUbBytes);

    if (initialized_) {
        TORCH_CHECK(rank == rank_ && worldSize == worldSize_ && heapSize == heapSize_ && initMethod == initMethod_,
                    "MTE runtime is already initialized with a different configuration");
        return;
    }

    aclshmemx_init_attr_t attributes;
    aclshmemx_uniqueid_t uniqueId;
    const int32_t attrStatus = FillInitAttributes(
        rank, worldSize, heapSize, initMethod.c_str(), &attributes, &uniqueId);
    TORCH_CHECK(attrStatus == ACLSHMEM_SUCCESS, "failed to construct MTE SHMEM attributes: ", attrStatus);
    const int32_t initStatus = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);
    TORCH_CHECK(initStatus == ACLSHMEM_SUCCESS, "aclshmemx_init_attr failed: ", initStatus);

    try {
        // Scratch belongs to this runtime. Payload buffers must begin at 2048
        // or later, and use different hard-event IDs from the signal path.
        const int32_t configStatus = aclshmemx_set_mte_config(0, 2048, 7);
        TORCH_CHECK(configStatus == ACLSHMEM_SUCCESS, "failed to configure MTE scratch: ", configStatus);
        workspace_ = static_cast<uint8_t *>(aclshmem_calloc(1, kKernelWorkspaceBytes));
        TORCH_CHECK(workspace_ != nullptr, "failed to allocate ", kKernelWorkspaceBytes,
                    " bytes of symmetric MTE workspace");
        TORCH_CHECK(aclrtMemset(workspace_, kKernelWorkspaceBytes, 0, kKernelWorkspaceBytes) == ACL_SUCCESS,
                    "failed to zero MTE workspace");
        TORCH_CHECK(aclrtSynchronizeStream(stream) == ACL_SUCCESS,
                    "failed to synchronize MTE workspace initialization");
        fftsAddress_ = shmemx_get_ffts_config();
        ValidateMtePeers(workspace_ + kKernelWorkspaceDataBytes, rank, worldSize, fftsAddress_, stream);
    } catch (const MteProbeError &error) {
        if (error.UnsafeToRelease()) {
            // A timed-out kernel may still reference the heap. Retain it until
            // process teardown and prohibit another initialization on it.
            failed_ = true;
            throw;
        }
        if (workspace_ != nullptr) {
            aclshmem_free(workspace_);
        }
        workspace_ = nullptr;
        fftsAddress_ = 0;
        shmem_finalize();
        throw;
    } catch (...) {
        if (workspace_ != nullptr) {
            aclshmem_free(workspace_);
        }
        workspace_ = nullptr;
        fftsAddress_ = 0;
        shmem_finalize();
        throw;
    }

    rank_ = rank;
    worldSize_ = worldSize;
    heapSize_ = heapSize;
    initMethod_ = initMethod;
    initialized_ = true;
    lastStream_ = stream;
}

void MteRuntime::EnsureInitialized() const
{
    TORCH_CHECK(!failed_, "MTE device completion failed; restart all worker processes");
    TORCH_CHECK(initialized_, "MTE runtime is not initialized; call init_mte_group() before the operator");
}

void *MteRuntime::AllocateSymmetric(size_t bytes)
{
    std::lock_guard<std::mutex> lock(mutex_);
    EnsureInitialized();
    TORCH_CHECK(bytes > 0, "symmetric allocation size must be positive");
    void *pointer = aclshmem_malloc(bytes);
    TORCH_CHECK(pointer != nullptr, "failed to allocate ", bytes,
                " bytes from the symmetric MTE heap; increase heap_size if necessary");
    externalAllocations_.insert(pointer);
    return pointer;
}

void MteRuntime::FreeSymmetric(void *pointer)
{
    if (pointer == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (failed_) {
        return;
    }
    const auto allocation = externalAllocations_.find(pointer);
    if (allocation == externalAllocations_.end()) {
        return;
    }
    if (lastStream_ != nullptr) {
        const aclError syncStatus = aclrtSynchronizeStream(lastStream_);
        if (syncStatus != ACL_SUCCESS) {
            failed_ = true;
            TORCH_WARN("MTE stream synchronization failed; retain symmetric memory until process exit: ", syncStatus);
            return;
        }
    }
    aclshmem_free(pointer);
    externalAllocations_.erase(allocation);
}

bool MteRuntime::IsSymmetric(const void *pointer) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    EnsureInitialized();
    return externalAllocations_.find(const_cast<void *>(pointer)) != externalAllocations_.end();
}

void MteRuntime::RecordStream(aclrtStream stream)
{
    std::lock_guard<std::mutex> lock(mutex_);
    EnsureInitialized();
    if (lastStream_ != nullptr && lastStream_ != stream) {
        const aclError syncStatus = aclrtSynchronizeStream(lastStream_);
        if (syncStatus != ACL_SUCCESS) {
            failed_ = true;
        }
        TORCH_CHECK(syncStatus == ACL_SUCCESS, "failed to synchronize the previous MTE stream: ", syncStatus);
    }
    lastStream_ = stream;
}

void MteRuntime::Finalize(aclrtStream stream)
{
    auto launchGuard = AcquireLaunchGuard();
    std::lock_guard<std::mutex> lock(mutex_);
    if (!initialized_) {
        return;
    }
    TORCH_CHECK(!failed_, "MTE device completion failed; retain its heap and restart all worker processes");
    TORCH_CHECK(externalAllocations_.empty(), "cannot finalize MTE while ", externalAllocations_.size(),
                " symmetric output tensor(s) are still alive; delete them before destroy_mte_group()");
    aclrtStream streamToWait = lastStream_ != nullptr ? lastStream_ : stream;
    if (streamToWait != nullptr) {
        const aclError syncStatus = aclrtSynchronizeStream(streamToWait);
        if (syncStatus != ACL_SUCCESS) {
            failed_ = true;
        }
        TORCH_CHECK(syncStatus == ACL_SUCCESS, "failed to synchronize MTE stream during finalization: ", syncStatus);
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
    hasTuningOverride_ = false;
    TORCH_CHECK(status == ACLSHMEM_SUCCESS, "shmem_finalize failed: ", status);
}

uint8_t *MteRuntime::Workspace() const
{
    EnsureInitialized();
    return workspace_;
}

uint64_t MteRuntime::FftsAddress() const
{
    EnsureInitialized();
    return fftsAddress_;
}

int32_t MteRuntime::Rank() const
{
    EnsureInitialized();
    return rank_;
}

int32_t MteRuntime::WorldSize() const
{
    EnsureInitialized();
    return worldSize_;
}

uint64_t MteRuntime::HeapSize() const
{
    EnsureInitialized();
    return heapSize_;
}

uint8_t *MteRuntime::ControlWorkspace(uint32_t operatorIndex) const
{
    EnsureInitialized();
    TORCH_CHECK(operatorIndex < 2, "invalid MTE operator control index");
    return workspace_ + kKernelWorkspaceDataBytes + operatorIndex * kOperatorControlBytes;
}

MteCommConfig MteRuntime::RequestedCommConfig() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    EnsureInitialized();
    return hasTuningOverride_ ? tuningOverride_ :
        MteCommConfig{static_cast<uint32_t>(worldSize_ - 1), 1, 65536, 2};
}

void MteRuntime::SetTuningOverride(const MteCommConfig &config)
{
#if defined(CATCCOS_MTE_WITH_TUNING) && CATCCOS_MTE_WITH_TUNING
    std::lock_guard<std::mutex> lock(mutex_);
    EnsureInitialized();
    const uint32_t peers = static_cast<uint32_t>(worldSize_ - 1);
    TORCH_CHECK(config.workersPerPeer > 0 && config.workersPerPeer <= kMteMaxPeerWorkers,
                "workersPerPeer must be in [1, 64]");
    if (peers == 0) {
        TORCH_CHECK(config.commAivCount == 0 && config.workersPerPeer == 1,
                    "single-rank MTE requires commAivCount=0 and workersPerPeer=1");
    } else {
        TORCH_CHECK(config.commAivCount >= config.workersPerPeer &&
                    config.commAivCount % config.workersPerPeer == 0 &&
                    config.commAivCount / config.workersPerPeer <= peers,
                    "MTE worker groups must be positive and must not exceed the remote peer count");
    }
    TORCH_CHECK(config.commChunkBytes >= 32 && config.commChunkBytes % 32 == 0,
                "commChunkBytes must be a positive multiple of 32");
    TORCH_CHECK(config.commUbStages >= 1 && config.commUbStages <= 2,
                "commUbStages must be 1 or 2");
    TORCH_CHECK(static_cast<uint64_t>(config.commChunkBytes) * config.commUbStages + 2048 + 8192 <= 256UL * 1024,
                "MTE communication buffers exceed UB capacity after scratch and system reserve");
    // Initialization checks the platform against the pinned usable UB budget;
    // the selected kernel separately checks its available communication AIVs.
    tuningOverride_ = config;
    hasTuningOverride_ = true;
#else
    (void)config;
    TORCH_CHECK(false, "MTE tuning is disabled; rebuild with --with-tuning");
#endif
}

void MteRuntime::ClearTuningOverride()
{
#if defined(CATCCOS_MTE_WITH_TUNING) && CATCCOS_MTE_WITH_TUNING
    std::lock_guard<std::mutex> lock(mutex_);
    EnsureInitialized();
    hasTuningOverride_ = false;
#else
    TORCH_CHECK(false, "MTE tuning is disabled; rebuild with --with-tuning");
#endif
}

} // namespace OpsTransformer::CatccosMte
