#ifndef MOE_DISTRIBUTE_BASE_TYPES_H
#define MOE_DISTRIBUTE_BASE_TYPES_H

// ============================================================================
// moe_distribute_base_types.h — HcclCombinOpParam 结构体 + ChannelEntity 偏移常量
//
// 纯 C++ 结构体和常量, 不含 kernel API, Host 侧和 Kernel 侧共用。
// 从 moe_distribute_base_native.h 和 aiv_urma_channel_decode.h 抽出,
// 避免 host 编译器引入 kernel_operator.h。
// ============================================================================

#include <cstdint>
#include <cstring>

constexpr uint32_t MAX_RANK_NUM = 64U;
constexpr uint32_t MAX_OP_NUM = 8U;

constexpr uint64_t WIN_PICI_OFFSET = 1024U * 1024U;
constexpr uint64_t PICI_WIN_SIZE = 512UL;

// ---- ChannelEntity (256B) 偏移 ----
constexpr uint32_t CH_OFF_LOCAL_BUFFER_NUM  = 32U;
constexpr uint32_t CH_OFF_REMOTE_BUFFER_NUM = 36U;
constexpr uint32_t CH_OFF_LOCAL_BUFFER_PTR  = 64U;
constexpr uint32_t CH_OFF_REMOTE_BUFFER_PTR = 72U;
constexpr uint32_t CH_OFF_SQ_CONTEXT_PTR    = 80U;
constexpr uint32_t CH_OFF_CQ_CONTEXT_PTR    = 88U;

// ---- SqContext (128B) 偏移 ----
constexpr uint32_t SQ_OFF_SQVA       = 8U;
constexpr uint32_t SQ_OFF_DBVA       = 32U;
constexpr uint32_t SQ_OFF_WQESIZE    = 44U;
constexpr uint32_t SQ_OFF_SQDEPTH    = 48U;
constexpr uint32_t SQ_OFF_TPID       = 52U;
constexpr uint32_t SQ_OFF_REMOTE_EID = 56U;

// ---- CqContext (128B) 偏移 ----
constexpr uint32_t CQ_OFF_CQVA    = 8U;
constexpr uint32_t CQ_OFF_DBVA    = 32U;
constexpr uint32_t CQ_OFF_CQESIZE = 44U;
constexpr uint32_t CQ_OFF_CQDEPTH = 48U;

// ---- RegedBufferEntity (64B) 偏移 ----
constexpr uint32_t RB_OFF_ADDR         = 8U;
constexpr uint32_t RB_OFF_SIZE         = 16U;
constexpr uint32_t RB_OFF_UB_TOKEN_ID  = 28U;
constexpr uint32_t RB_OFF_UB_TOKEN_VAL = 32U;

// ---- 结构体大小 ----
constexpr uint32_t CHANNEL_ENTITY_SIZE = 256U;
constexpr uint32_t SQ_CONTEXT_SIZE     = 128U;
constexpr uint32_t CQ_CONTEXT_SIZE     = 128U;
constexpr uint32_t REGED_BUFFER_SIZE   = 64U;

struct HcclAiRMAWQ {
    uint32_t jettyId;
    uint64_t sqVA;
    uint32_t wqeSize;
    uint32_t sqDepth;
    uint64_t headAddr;
    uint64_t tailAddr;
    uint64_t dbAddr;
    uint32_t tp_id;
    uint8_t rmtEid[16];
    uint32_t rmtObjId;
    uint32_t rmtTokenValue;
    uint32_t localTokenId;
};

struct HcclAiRMACQ {
    uint32_t jfcId;
    uint64_t cqVA;
    uint32_t cqeSize;
    uint32_t cqDepth;
    uint64_t headAddr;
    uint64_t tailAddr;
    uint64_t dbAddr;
};

struct HcclCombinOpParam {
    uint64_t workSpace;
    uint64_t workSpaceSize;
    uint32_t rankId;
    uint32_t rankDim;
    uint64_t winSize;
    uint64_t windowsIn[MAX_RANK_NUM];
    uint64_t windowsOut[MAX_RANK_NUM];
    uint64_t xnAddr;
    uint64_t ckeAddr;
    uint64_t msAddr;
    uint64_t msSize;
    uint32_t opType[MAX_OP_NUM];
    uint8_t algorithmType[MAX_OP_NUM];
    HcclAiRMAWQ sqs[MAX_RANK_NUM];
    HcclAiRMACQ cqs[MAX_RANK_NUM];
};

#endif // MOE_DISTRIBUTE_BASE_TYPES_H
