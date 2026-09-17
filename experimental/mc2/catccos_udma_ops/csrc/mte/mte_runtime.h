/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#ifndef OPS_TRANSFORMER_CATCCOS_MTE_RUNTIME_H
#define OPS_TRANSFORMER_CATCCOS_MTE_RUNTIME_H

#include <acl/acl.h>

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <stdexcept>
#include <unordered_set>
#include "catccos/comm/mte_transport_params.hpp"

namespace OpsTransformer::CatccosMte {

class MteProbeError : public std::runtime_error {
public:
    MteProbeError(const std::string &message, bool unsafeToRelease)
        : std::runtime_error(message), unsafeToRelease_(unsafeToRelease) {}
    bool UnsafeToRelease() const noexcept { return unsafeToRelease_; }
private:
    bool unsafeToRelease_;
};

// Implemented in the ASC compilation unit; checks transport capabilities before
// probing all directed peer mappings. A null peer pointer is not a topology test.
void ValidateMtePeers(uint8_t *workspace, int32_t rank, int32_t worldSize,
                      uint64_t fftsAddress, aclrtStream stream);

class MteRuntime {
public:
    static constexpr size_t kKernelWorkspaceDataBytes = 400UL * 1024 * 1024;
    static constexpr size_t kOperatorControlBytes = 64UL * 1024;
    static constexpr size_t kKernelWorkspaceBytes = kKernelWorkspaceDataBytes + 2 * kOperatorControlBytes;

    static MteRuntime &Instance();
    // Hold through stream selection and enqueue; the returned guard must outlive
    // every workspace read in the host operator. Lock order: launch, then state.
    std::unique_lock<std::mutex> AcquireLaunchGuard();

    void Initialize(int32_t rank, int32_t worldSize, const std::string &initMethod,
                    uint64_t heapSize, aclrtStream stream = nullptr);
    void Finalize(aclrtStream stream);
    void EnsureInitialized() const;

    void *AllocateSymmetric(size_t bytes);
    void FreeSymmetric(void *pointer);
    bool IsSymmetric(const void *pointer) const;
    void RecordStream(aclrtStream stream);
    uint8_t *Workspace() const;
    uint8_t *ControlWorkspace(uint32_t operatorIndex) const;
    uint64_t FftsAddress() const;
    int32_t Rank() const;
    int32_t WorldSize() const;
    uint64_t HeapSize() const;
    MteCommConfig RequestedCommConfig() const;
    void SetTuningOverride(const MteCommConfig &config);
    void ClearTuningOverride();

private:
    MteRuntime() = default;
    ~MteRuntime() = default;
    MteRuntime(const MteRuntime &) = delete;
    MteRuntime &operator=(const MteRuntime &) = delete;

    std::mutex launchMutex_;
    mutable std::mutex mutex_;
    bool initialized_{false};
    bool failed_{false};
    int32_t rank_{-1};
    int32_t worldSize_{0};
    uint64_t heapSize_{0};
    std::string initMethod_;
    uint8_t *workspace_{nullptr};
    aclrtStream lastStream_{nullptr};
    uint64_t fftsAddress_{0};
    bool hasTuningOverride_{false};
    MteCommConfig tuningOverride_{};
    std::unordered_set<void *> externalAllocations_;
};

} // namespace OpsTransformer::CatccosMte

#endif // OPS_TRANSFORMER_CATCCOS_MTE_RUNTIME_H
