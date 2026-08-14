#ifndef MOE_DISTRIBUTE_BASE_TYPES_H
#define MOE_DISTRIBUTE_BASE_TYPES_H

// ============================================================================
// moe_distribute_base_types.h — HcclCombinOpParam 相关结构体定义
//
// 纯 C++ 结构体, 不含 kernel API, Host 侧和 Kernel 侧共用。
// 从 moe_distribute_base_native.h 抽出, 避免 host 编译器引入 kernel_operator.h。
// ============================================================================

#include <cstdint>
#include <cstring>

constexpr uint32_t MAX_RANK_NUM = 64U;
constexpr uint32_t MAX_OP_NUM = 8U;

constexpr uint64_t WIN_PICI_OFFSET = 1024U * 1024U;
constexpr uint64_t PICI_WIN_SIZE = 512UL;

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
