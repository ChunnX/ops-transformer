#pragma once

#include <cstdint>
#include <initializer_list>
#include <limits>
#include "../core/numerical_types.h"

namespace sage::host_runtime {

constexpr uint64_t WORKSPACE_MEMORY_ALIGNMENT = 512ULL;
constexpr uint64_t DEFAULT_SYSTEM_WORKSPACE = 16ULL * 1024ULL * 1024ULL;

class WorkspacePlanner {
public:
    explicit WorkspacePlanner(uint64_t baseOffset = 0) : m_currentOffset(baseOffset) {}

    bool Suballocate(uint64_t &outOffset, std::initializer_list<uint64_t> dimensions)
    {
        uint64_t tensorBytes = 1;
        for (uint64_t dim : dimensions) {
            if (dim > 0 && tensorBytes > std::numeric_limits<uint64_t>::max() / dim) {
                return false;
            }
            tensorBytes *= dim;
        }

        if (tensorBytes > std::numeric_limits<uint64_t>::max() - (WORKSPACE_MEMORY_ALIGNMENT - 1)) {
            return false;
        }
        tensorBytes = core::RoundUpTo(tensorBytes, WORKSPACE_MEMORY_ALIGNMENT);

        if (m_currentOffset > std::numeric_limits<uint64_t>::max() - tensorBytes) {
            return false;
        }
        outOffset = m_currentOffset;
        m_currentOffset += tensorBytes;
        return true;
    }

    uint64_t GetTotalWorkspaceBytes() const noexcept { return m_currentOffset; }

private:
    uint64_t m_currentOffset;
};

} // namespace sage::host_runtime

namespace optiling {
using sage::host_runtime::WORKSPACE_MEMORY_ALIGNMENT;
using sage::host_runtime::DEFAULT_SYSTEM_WORKSPACE;
using sage::host_runtime::WorkspacePlanner;
}
