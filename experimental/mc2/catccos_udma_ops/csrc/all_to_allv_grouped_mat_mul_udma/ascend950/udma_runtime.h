/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#ifndef OPS_TRANSFORMER_CATCCOS_UDMA_RUNTIME_H
#define OPS_TRANSFORMER_CATCCOS_UDMA_RUNTIME_H

#include <acl/acl.h>

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_set>

namespace OpsTransformer::CatccosUdma {

class UdmaRuntime {
public:
    static constexpr size_t kKernelWorkspaceBytes = 200UL * 1024 * 1024 * sizeof(uint16_t) + 4096;

    static UdmaRuntime &Instance();

    void Initialize(int32_t rank, int32_t worldSize, const std::string &initMethod, uint64_t heapSize);
    void Finalize(aclrtStream stream);
    void EnsureInitialized() const;

    void *AllocateSymmetric(size_t bytes);
    void FreeSymmetric(void *pointer);
    bool IsSymmetric(const void *pointer) const;
    void RecordStream(aclrtStream stream);
    uint8_t *Workspace() const;
    uint64_t FftsAddress() const;
    int32_t Rank() const;
    int32_t WorldSize() const;
    uint64_t HeapSize() const;

private:
    UdmaRuntime() = default;
    ~UdmaRuntime() = default;
    UdmaRuntime(const UdmaRuntime &) = delete;
    UdmaRuntime &operator=(const UdmaRuntime &) = delete;

    mutable std::mutex mutex_;
    bool initialized_{false};
    int32_t rank_{-1};
    int32_t worldSize_{0};
    uint64_t heapSize_{0};
    std::string initMethod_;
    uint8_t *workspace_{nullptr};
    aclrtStream lastStream_{nullptr};
    uint64_t fftsAddress_{0};
    std::unordered_set<void *> externalAllocations_;
};

} // namespace OpsTransformer::CatccosUdma

#endif // OPS_TRANSFORMER_CATCCOS_UDMA_RUNTIME_H
