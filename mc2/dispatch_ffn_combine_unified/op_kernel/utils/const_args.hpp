
#ifndef CONST_ARGS_HPP
#define CONST_ARGS_HPP
constexpr static uint64_t MB_SIZE = 1024 * 1024UL;
constexpr static int32_t NUMS_PER_FLAG = 16;
constexpr static int32_t CACHE_LINE = 512;
constexpr static int32_t RESET_VAL = 0xffff;
constexpr static int32_t FLAGSTRIDE = 16;
constexpr static int32_t UB_ALIGN = 32;
constexpr uint16_t CROSS_CORE_FLAG_MAX_SET_COUNT = 15;
// Number of ranks contained inside one Atlas A2 server. Used by the unified
// kernel's BatchWrite path (DispatchAndCombine when IS_A2_=true) to decide
// between same-server IPC and cross-server RDMA token sends.
constexpr static uint32_t SERVER_RANK_SIZE_A2 = 8;
#endif