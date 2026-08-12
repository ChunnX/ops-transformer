/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */
/*
 * =========================================================================================
 * sparse_attention_score_prefill —— Kernel 入口文件
 * =========================================================================================
 *
 * 【算子概述】
 * 本算子面向 LLM Prefill 阶段的稀疏注意力计算。采用 KV-Gather-Q 方案：
 *   - 以 KV block 为主轴分任务，同一个 KV block 只加载一次
 *   - 遍历所有选择该 block 的 Q token，计算 partial attention
 *
 * 【两阶段架构】
 *   Phase 1 (Cube + Vector 协同):
 *     - Cube 侧完成 QK matmul 和 PV matmul
 *     - Vector 侧完成 online softmax (计算 rowMax / rowSum / P=exp(S-rowMax))
 *     - 输出未归一化的 O_partial = P×V、rowMax、rowSum 到 workspace
 *     - 任务粒度: 1 logical KV block × 1 KV head
 *
 *   Phase 2 (仅 Vector):
 *     - 复用 IFA FlashDecode 路径完成 rescale 与 combine
 *     - 对每个 Q token 的 topK 个 partial 做: O_norm = O_partial/rowSum, 再加权求和
 *     - 写最终 attention_out
 *     - 任务粒度: 1 Q token × 1 KV head
 *
 * 【执行流程】
 *   InitSyncFlags()           — 初始化 Cube/Vector 跨核同步事件标志
 *   InitWorkspaceStats()      — VEC 侧预置 softmaxMax=-inf, softmaxSum=0 (无效 slot 标记)
 *   SyncAll<false>()          — 全局同步，确保 workspace 初始化完成
 *   Phase1KvCentricCompute()  — Phase1: KV-centric partial compute
 *   SyncAll<false>()          — 全局同步，确保 Phase1 全部完成
 *   Phase2CombineScale()      — Phase2: Vector-only combine (仅 VEC 执行)
 *   ReleaseSyncFlags()        — 释放同步事件标志
 *
 * 【输入参数说明】
 *   query              — [total_q_tokens, num_q_heads, D] bf16, TND 格式 Q
 *   key/value          — [num_physical_blocks, blockSize, kvHeads, D] bf16, Paged KV Cache
 *   blockTable         — [batch, maxBlocksPerBatch] int32, logical→physical block 映射
 *   k2qRowPtr          — [kvHeads, totalPackedRows+1] int32, CSR 行指针
 *   k2qQIndices        — [kvHeads, totalQ*topK] int32, CSR 列数据 (global qToken id)
 *   k2qSlotIndices     — [kvHeads, totalQ*topK] int32, workspace slot 索引 [0, topK)
 *   actualSeqLengths   — [batch] int32, 每个 batch 的 Q seqlen
 *   actualSeqLengthsKv — [batch] int32, 每个 batch 的 KV seqlen
 *   attentionOut       — [total_q_tokens, num_q_heads, D] bf16, 输出
 *   workspace          — libapi workspace + user workspace (O_partial/max/sum 三块 buffer)
 *   tiling             — host 侧下发的 tiling 数据
 *
 * 【Tiling Key 派发】
 *   根据 innerPrecise 属性选择不同的代码路径:
 *     20001 (BF16_D128_TILING):           fp32 O_partial 路径 (innerPrecise!=1)
 *     20002 (BF16_D128_INNER_LOW_TILING): bf16 O_partial 路径 (innerPrecise==1)
 *   两者区别仅在于 REDtype (PV fixpipe 输出精度):
 *     - fp32 路径: REDtype=float, O_partial 以 fp32 存储
 *     - bf16 路径: REDtype=bfloat16_t, PV fixpipe F322BF16 + Phase2 regbase cast
 *
 * 【硬件平台】
 *   目标: Arch35 (Atlas A5, __CCE_AICORE__==310)
 *   核类型: KERNEL_TYPE_MIX_AIC_1_2 (1 AIC : 2 AIV 混合核)
 *   Cube 核执行 Phase1 的 QK/PV matmul; Vector 核执行 softmax 和 Phase2 combine
 * =========================================================================================
 */

#include "kernel_operator.h"
#include "sparse_attention_score_prefill_tilingkey.h"
#include "sparse_attention_score_prefill_kernel_interface.cpp"

// Kernel 入口函数: 由 Ascend C 运行时调用, 接收 GM 上的所有输入/输出/workspace/tiling 指针
extern "C" __global__ __aicore__ void sparse_attention_score_prefill(
    __gm__ uint8_t* query,
    __gm__ uint8_t* key,
    __gm__ uint8_t* value,
    __gm__ uint8_t* blockTable,
    __gm__ uint8_t* k2qRowPtr,
    __gm__ uint8_t* k2qQIndices,
    __gm__ uint8_t* k2qSlotIndices,
    __gm__ uint8_t* actualSeqLengths,
    __gm__ uint8_t* actualSeqLengthsKv,
    __gm__ uint8_t* attentionOut,
    __gm__ uint8_t* workspace,
    __gm__ uint8_t* tiling)
{
    // 校验 tiling key 有效 (所有合法 key >= BASE_TILING=20000)
    if (TILING_KEY_VAR >= SASA_PREFILL_BASE_TILING) {
        // 从 workspace 中分离出 user workspace (跳过 libapi workspace 区域)
        __gm__ uint8_t *user = AscendC::GetUserWorkspace(workspace);
        // 设置核任务类型为 AIC:AIV=1:2 混合模式, Cube 和 Vector 核同时启动
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

#if (__CCE_AICORE__ == 310)
        // === Tiling Key 派发: 编译期静态分支, 只有匹配的分支会被编入 ===

        // 路径 1: fp32 O_partial (innerPrecise != 1, 默认路径)
        // REDtype=float, Phase1 PV fixpipe 输出 fp32, Phase2 直接读 fp32 做 combine
        TILING_KEY_IS(SASA_PREFILL_BF16_D128_TILING);
#if TILING_KEY_VAR == SASA_PREFILL_BF16_D128_TILING
        SasaPrefillInferIntf<bfloat16_t, bfloat16_t, float>(
            query, key, value, blockTable,
            k2qRowPtr, k2qQIndices, k2qSlotIndices,
            actualSeqLengths, actualSeqLengthsKv,
            attentionOut, user, tiling);
#endif

        // 路径 2: bf16 O_partial (innerPrecise == 1, 低精度路径)
        // REDtype=bfloat16_t, PV fixpipe F322BF16 写 bf16, Phase2 regbase cast bf16→fp32
        // 可节省 workspace 中 O_partial buffer 的一半空间 (bf16=2B vs fp32=4B)
        TILING_KEY_IS(SASA_PREFILL_BF16_D128_INNER_LOW_TILING);
#if TILING_KEY_VAR == SASA_PREFILL_BF16_D128_INNER_LOW_TILING
        SasaPrefillInferIntf<bfloat16_t, bfloat16_t, bfloat16_t>(
            query, key, value, blockTable,
            k2qRowPtr, k2qQIndices, k2qSlotIndices,
            actualSeqLengths, actualSeqLengthsKv,
            attentionOut, user, tiling);
#endif
#endif

#if (__CCE_AICORE__ == 220)
        // === A2 (AtlasA2) Tiling Key 派发 ===
        // 与 A5 派发逻辑一致, 仅编译目标为 A2 (__CCE_AICORE__==220)

        // 路径 1: fp32 O_partial (innerPrecise != 1, 默认路径)
        TILING_KEY_IS(SASA_PREFILL_BF16_D128_TILING);
#if TILING_KEY_VAR == SASA_PREFILL_BF16_D128_TILING
        SasaPrefillInferIntf<bfloat16_t, bfloat16_t, float>(
            query, key, value, blockTable,
            k2qRowPtr, k2qQIndices, k2qSlotIndices,
            actualSeqLengths, actualSeqLengthsKv,
            attentionOut, user, tiling);
#endif

        // 路径 2: bf16 O_partial (innerPrecise == 1, 低精度路径)
        TILING_KEY_IS(SASA_PREFILL_BF16_D128_INNER_LOW_TILING);
#if TILING_KEY_VAR == SASA_PREFILL_BF16_D128_INNER_LOW_TILING
        SasaPrefillInferIntf<bfloat16_t, bfloat16_t, bfloat16_t>(
            query, key, value, blockTable,
            k2qRowPtr, k2qQIndices, k2qSlotIndices,
            actualSeqLengths, actualSeqLengthsKv,
            attentionOut, user, tiling);
#endif
#endif
    }
}
