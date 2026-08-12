/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */

/*
 * 本头文件定义 kernel 侧的 tiling 数据结构 SparseAttentionScorePrefillTilingData，
 * 它是 host 侧 TilingData 结构在 kernel 侧的镜像，用于在 kernel 中从 GM（Global Memory）
 * 反序列化 host 下发的 tiling 数据。kernel 通过该结构按字段偏移读取 tiling buffer，
 * 获取执行所需的形状、切分、workspace 布局以及派发 key 等信息。
 */

#ifndef SPARSE_ATTENTION_SCORE_PREFILL_KERNEL_COMMON_HPP
#define SPARSE_ATTENTION_SCORE_PREFILL_KERNEL_COMMON_HPP

#include "kernel_operator.h"

namespace SparseAttnPrefill {

struct SparseAttentionScorePrefillTilingData {
    uint32_t batch;                 // batch 数
    uint32_t numHeads;              // Q head 数
    uint32_t kvHeads;               // KV head 数
    uint32_t groupSize;             // numHeads/kvHeads，GQA 中一个 KV head 对应的 Q head 数
    uint32_t embeddingSize;         // head 维度 D
    uint32_t blockSize;             // KV block token 数，通常为 128
    uint32_t topK;                  // 每个 Q token 选择的 KV block 数
    uint32_t totalQTokens;          // 所有 batch 的 Q token 总数
    uint32_t numKvBlocks;           // totalPackedRows = sum(ceil(kv_seqlen/blockSize))
    uint32_t maxBlocksPerBatch;     // block_table 行 stride
    uint32_t k2qNnzUpperBound;      // CSR 数据区每 head 上界
    uint32_t totalTaskNumP1;        // Phase1 任务数 = numKvBlocks * kvHeads
    uint32_t totalTaskNumP2;        // Phase2 任务数 = totalQTokens * kvHeads
    float scaleValue;               // softmax 缩放因子
    uint32_t innerPrecise;          // 精度模式 (1=bf16 O_partial, 4=fp32 O_partial)
    uint64_t accumOutSize;          // workspace 中 O_partial float 元素总数
    uint64_t lseStatSize;           // workspace 中 softmaxMax/Sum 各自 float 元素总数
    uint64_t workSpaceSize;                         // workspace 总字节数
    uint64_t gmSWorkspaceSize;                      // A2 GM S staging workspace per-core 字节数 (A5=0)
    uint64_t gmPWorkspaceSize;                      // A2 GM P staging workspace per-core 字节数 (A5=0)
    uint64_t tilingKey;                             // tiling key 用于 kernel 派发
};

}  // namespace SparseAttnPrefill

#endif  // SPARSE_ATTENTION_SCORE_PREFILL_KERNEL_COMMON_HPP
