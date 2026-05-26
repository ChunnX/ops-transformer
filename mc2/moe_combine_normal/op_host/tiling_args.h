#ifndef TILING_ARGS_H
#define TILING_ARGS_H
#include <cstdint>

namespace Moe {
constexpr uint64_t COMBINE_STATE_WIN_OFFSET = 4U * 1024UL * 1024UL;
constexpr uint64_t NOTIFY_DISPATCH_WIN_OFFSET = 102U * 1024UL * 1024UL;
constexpr uint64_t NOTIFY_DISPATCH_BUFF_OFFSET = 0UL;
} // namespace Moe
#endif // TILING_ARGS_H