#ifndef AICPU_STUB_STATUS_H
#define AICPU_STUB_STATUS_H

#include <cstdint>

namespace aicpu {
using KernelStatus = uint32_t;
constexpr uint32_t KERNEL_STATUS_OK = 0U;
constexpr uint32_t KERNEL_STATUS_PARAM_INVALID = 1U;
constexpr uint32_t KERNEL_STATUS_INNER_ERROR = 2U;
constexpr uint32_t KERNEL_STATUS_TIMEOUT = 3U;
}  // namespace aicpu

#endif  // AICPU_STUB_STATUS_H
