#ifndef GENERIC_BLOCK_SPARSE_ATTENTION_GRAD_METADATA_LAYOUT_H
#define GENERIC_BLOCK_SPARSE_ATTENTION_GRAD_METADATA_LAYOUT_H

#include <cstddef>
#include <cstdint>

namespace gbsag_meta {

constexpr uint32_t MAGIC = 0x47425341U; // "GBSA"
constexpr uint32_t HEADER_WORDS = 32;
constexpr uint32_t KTASK_WORDS = 8;
constexpr uint32_t BASIC_BLOCK_SIZE = 128;
constexpr uint32_t MAX_AIC_CORE_NUM = 64;

constexpr uint32_t OFF_MAGIC = 0;
constexpr uint32_t OFF_TOTAL_LEN = 1;
constexpr uint32_t OFF_BATCH = 2;
constexpr uint32_t OFF_NUM_Q_HEADS = 3;
constexpr uint32_t OFF_NUM_KV_HEADS = 4;
constexpr uint32_t OFF_HEAD_DIM = 5;
constexpr uint32_t OFF_INPUT_LAYOUT = 6;
constexpr uint32_t OFF_MAX_Q_SEQLEN = 7;
constexpr uint32_t OFF_MAX_KV_SEQLEN = 8;
constexpr uint32_t OFF_TOTAL_TASK_NUM = 9;
constexpr uint32_t OFF_TASK_NUM_PER_CORE = 10;
constexpr uint32_t OFF_TAIL_TASK_NUM = 11;
constexpr uint32_t OFF_SEQ_LEN_OFFSET = 12;
constexpr uint32_t OFF_TASK_TABLE_OFFSET = 13;
constexpr uint32_t OFF_CORE_NUM = 14;
constexpr uint32_t OFF_TASK_WORDS = 15;

constexpr uint32_t TASK_BEGIN_BATCH = 0;
constexpr uint32_t TASK_BEGIN_KV_HEAD = 1;
constexpr uint32_t TASK_BEGIN_KV_SEQ_OFFSET = 2;
constexpr uint32_t TASK_RESERVED = 3;
constexpr uint32_t TASK_PRE_Q_LO = 4;
constexpr uint32_t TASK_PRE_Q_HI = 5;
constexpr uint32_t TASK_PRE_KV_LO = 6;
constexpr uint32_t TASK_PRE_KV_HI = 7;

constexpr uint32_t LAYOUT_TND = 0;
constexpr uint32_t LAYOUT_BNSD = 1;
constexpr uint32_t LAYOUT_BSND = 2;

struct Header {
    int64_t magic;
    int64_t totalLen;
    int64_t batch;
    int64_t numQHeads;
    int64_t numKvHeads;
    int64_t headDim;
    int64_t inputLayout;
    int64_t maxQSeqlen;
    int64_t maxKvSeqlen;
    int64_t totalTaskNum;
    int64_t taskNumPerCore;
    int64_t tailTaskNum;
    int64_t seqLenOffset;
    int64_t taskTableOffset;
    int64_t coreNum;
    int64_t taskWords;
    int64_t reserved[16];
};

struct KTask {
    int64_t beginBatch;
    int64_t beginKvHead;
    int64_t beginKvSeqOffset;
    int64_t reserved;
    int64_t preQLo;
    int64_t preQHi;
    int64_t preKvLo;
    int64_t preKvHi;
};

static_assert(sizeof(Header) == HEADER_WORDS * sizeof(int64_t), "metadata header ABI drift");
static_assert(sizeof(KTask) == KTASK_WORDS * sizeof(int64_t), "metadata KTask ABI drift");
static_assert(offsetof(Header, totalTaskNum) / sizeof(int64_t) == OFF_TOTAL_TASK_NUM, "header offset drift");
static_assert(offsetof(Header, taskTableOffset) / sizeof(int64_t) == OFF_TASK_TABLE_OFFSET, "header offset drift");
static_assert(offsetof(KTask, beginKvSeqOffset) / sizeof(int64_t) == TASK_BEGIN_KV_SEQ_OFFSET, "task offset drift");

inline uint32_t MetadataWords(uint32_t coreNum)
{
    return HEADER_WORDS + coreNum * KTASK_WORDS;
}

} // namespace gbsag_meta

#endif // GENERIC_BLOCK_SPARSE_ATTENTION_GRAD_METADATA_LAYOUT_H
