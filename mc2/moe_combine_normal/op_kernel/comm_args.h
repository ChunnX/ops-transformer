#ifndef COMM_ARGS_H
#define COMM_ARGS_H
#include <cstdint>

#define FORCE_INLINE_AICORE __attribute__((always_inline)) inline __aicore__
#include "kernel_operator.h"

namespace Moe {
constexpr int CAM_MAX_RANK_SIZE = 384;

constexpr uint64_t NOTIFY_DISPATCH_BUFF_OFFSET = 102UL * 1024UL * 1024UL;
constexpr int64_t IPC_BUFF_MAX_SIZE = 100 * 1024 * 1024;
constexpr int64_t IPC_DATA_OFFSET = 2 * 1024 * 1024;
constexpr int64_t PING_PONG_SIZE = 2;
constexpr int64_t UB_SINGLE_DMA_SIZE_MAX = 190 * 1024;
constexpr int64_t SMALL_DATA_SIZE = 1 * 1024 * 1024;
constexpr int64_t UB_SINGLE_PING_PONG_ADD_SIZE_MAX = UB_SINGLE_DMA_SIZE_MAX / 2;
constexpr int UB_ALIGN_SIZE = 32;
constexpr int64_t MAGIC_ALIGN_COUNT = UB_ALIGN_SIZE / sizeof(int32_t);

constexpr uint8_t COMM_NUM = 2;
constexpr uint8_t COMM_EP_IDX = 0;
constexpr uint8_t COMM_TP_IDX = 1;

constexpr int DFX_COUNT = 50;
constexpr int64_t WAIT_SUCCESS = 112233445566;
constexpr int64_t IPC_CHUNK_FLAG = 0;
constexpr int64_t MAX_WAIT_ROUND_UNIT = 10 * 1000 * 1000;
constexpr uint64_t ROUND_STATE_MAX_SIZE = 4UL * 1024UL;
constexpr uint64_t BASE_ROUND_STATE_OFFSET = 450UL * 1024UL;

constexpr static int32_t UB_HEAD_OFFSET = 96;
constexpr static int32_t UB_MID_OFFSET = UB_HEAD_OFFSET + UB_SINGLE_PING_PONG_ADD_SIZE_MAX + UB_ALIGN_SIZE;
constexpr static int64_t UB_FLAG_SIZE = 2 * 1024;
constexpr static int64_t MAX_CORE_NUM = 48;
constexpr static uint64_t STATE_WIN_OFFSET = 900 * 1024;
constexpr static int64_t COMPARE_ALIGN_SIZE = 256;

constexpr static int64_t UB_SINGLE_TOTAL_SIZE_MAX = 192 * 1024;
constexpr static int64_t START_OFFSET_FOR_SHARE = 512;

enum Op : int {
    COPYONLY = -1,
    ADD = 0,
    MUL = 1,
    MAX = 2,
    MIN = 3
};

struct CommArgs {
    int rank = 0;
    int localRank = -1;
    int rankSize = 0;
    int localRankSize = -1;
    uint32_t extraFlag = 0;
    int testFlag = 0;
    GM_ADDR peerMems[CAM_MAX_RANK_SIZE] = {};
    int64_t sendCountMatrix[CAM_MAX_RANK_SIZE * CAM_MAX_RANK_SIZE] = {};
    int64_t sendCounts[CAM_MAX_RANK_SIZE] = {};
    int64_t sdispls[CAM_MAX_RANK_SIZE] = {};
    int64_t recvCounts[CAM_MAX_RANK_SIZE] = {};
    int64_t rdispls[CAM_MAX_RANK_SIZE] = {};
    int64_t batchSize;
    int64_t hiddenSize;
    int64_t topk;
    int64_t sharedExpertRankNum;
    int64_t expertNumPerRank;
    int64_t dfx[DFX_COUNT] = {};
};
} // namespace Moe
#endif // COMM_ARGS_H