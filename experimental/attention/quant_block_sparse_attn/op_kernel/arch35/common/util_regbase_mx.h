/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file util_regbase_mx.h
 * \brief QuantBlockSparseAttn 的 MXFP8 全量化运行期参数。
 */
#ifndef QUANT_BLOCK_SPARSE_ATTN_UTIL_REGBASE_MX_H_
#define QUANT_BLOCK_SPARSE_ATTN_UTIL_REGBASE_MX_H_

#include <cstdint>
#include "../../quant_block_sparse_attn_common.h"

namespace regbasemx {
constexpr uint32_t BSA_MX_SCALE_LAST_DIM = 2U;

// 单个 MX sparse-block task 的运行期状态。kernel 会为每个
// (B, N1, S1 outer block, sparse S2 block) 创建一个 MxRunInfo，并用三槽环形缓存保存，
// 使延迟执行的 C2/V2 可以消费前序 C1/V1 生成的 P tile。
struct MxRunInfo {
    // 当前有效 sparse-block task 的全局流水序号。
    // 来源：ProcessMainLoop 每处理一个有效 sparse block 后递增。
    // 用途：选择 taskRunInfo 三槽环形缓存、V1 stage1 ping-pong buffer 和 softmaxExp buffer。
    uint32_t loop = 0U;
    // 当前 S1 outer task 的序号。
    // 来源：ProcessMainLoop 每完成一个 s1OuterIdx 后递增。
    // 用途：选择 softmaxMax/softmaxSum 三槽缓存，使同一个 query block 的 S2 online-softmax 统计量可复用。
    uint32_t mLoop = 0U;
    // 当前环形缓存槽是否保存了一个待处理 task。
    // 来源：FillRunInfo 置 true；无效 tail 或 C2/V2 完成后置 false。
    // 用途：HasValidTask 和 ExecuteTask 判断尾部流水是否还需要 drain。
    bool isValid = false;
    // 当前 sparse block 是否是该 query block 的第一个 S2 loop。
    // 来源：s2LoopIdx == 0。
    // 用途：BMM1 首次拷贝 Q/QScale，V1 初始化 online softmax，V2 直接写初始累计输出。
    bool isFirstS2Loop = false;
    // 当前 sparse block 是否是该 query block 的最后一个 S2 loop。
    // 来源：s2LoopIdx == s2LoopEnd。
    // 用途：释放/复用 Q L1 buffer，V1 写 softmax_lse，V2 做最终归一化并写 attentionOut。
    bool isLastS2Loop = false;

    // batch index。
    // 来源：bn1Idx / realN2Size。
    // 用途：索引 actualSeqLengths、cuSeqlens、sparseSeqLen、sparseIndices 和 block_table。
    uint32_t bIdx = 0U;
    // KV head index。
    // 来源：n1Idx / gSize。
    // 用途：PA BNBD 拷贝 K/V/KScale/VScale 时选择 KV head。
    uint32_t n2Idx = 0U;
    // query block 在 G*S1 维度上的起始行。
    // 来源：s1OuterIdx * M_BASE。
    // 用途：保留 BSA/common MX 的 runinfo 语义；当前 MX 路径实际寻址主要使用 s1Idx。
    uint32_t gS1Idx = 0U;
    // GQA group index。
    // 来源：n1Idx % gSize。
    // 用途：保留 GQA 映射信息；当前 MX PA BNBD 路径输出和 KV 选择主要使用 realN2Idx/n2Idx。
    uint32_t gIdx = 0U;
    // 当前 query tile 在 batch 内的 S1 起始 token。
    // 来源：s1OuterIdx * M_BASE。
    // 用途：计算 query/qScale GM offset、输出 GM offset，以及 causal partial-mask 判断。
    uint32_t s1Idx = 0U;
    // query/output head index。
    // 来源：n1Idx。
    // 用途：TND query/qScale/attentionOut/softmaxLse 的 head 维寻址。
    uint32_t realN2Idx = 0U;
    // query head flat index。
    // 来源：bn1Idx % realN2Size。
    // 用途：索引 sparseSeqLen/sparseIndices，并派生 n2Idx/gIdx。
    uint32_t n1Idx = 0U;
    // 当前 batch 的实际 query sequence length。
    // 来源：actualSeqLengths 或 cuSeqlensQ。
    // 用途：计算 S1 tail 的 actMSize，避免 query 尾块越界。
    uint64_t actS1Size = 1U;
    // 当前 batch 的实际 KV sequence length。
    // 来源：actualSeqLengthsKv 或 cuSeqlensKv。
    // 用途：计算 sparse block 在 S2 方向的真实有效长度，处理 KV tail。
    uint64_t actS2Size = 1U;
    // 当前 query tile 的真实 M 行数，最大为 M_BASE=128。
    // 来源：min(M_BASE, actS1Size - s1Idx)。
    // 用途：Q/QScale 拷贝、BMM1 M 维、Fixpipe 和 vector 有效行判断。
    uint32_t actMSize = 0U;
    // actMSize 向上 32 对齐后的行数。
    // 来源：AlignUp(actMSize, 32)。
    // 用途：DN2NZ/ND2NZ 搬运、VF 计算和 subBlock 行拆分。
    uint32_t actMSizeAlign32 = 0U;
    // 当前 AIV subBlock 实际负责的 query 行数。
    // 来源：actMSize 按两个 AIV subBlock 拆分；subBlock0 取前半，subBlock1 取剩余。
    // 用途：V1/V2 有效行判断、FlashUpdate、attentionOut 和 softmaxLse 写回。
    uint32_t actVecMSize = 0U;
    // 当前 AIV subBlock 在 128-row query tile 内的行偏移。
    // 来源：subBlock0 为 0，subBlock1 为前半 actVecMSize。
    // 用途：V2 写 attentionOut/softmaxLse 时定位当前 subBlock 的输出起始 token。
    uint32_t vecMbaseIdx = 0U;

    // 当前 sparse block 的逻辑 S2 有效长度。
    // 来源：s2SparseBlk1RealSize。
    // 用途：判断当前 512-token block 是否需要拆成两个 256-token C1/V1 subLoop。
    uint32_t actSingleLoopS2Size = 0U;
    // actSingleLoopS2Size 向上 64 对齐后的长度。
    // 来源：AlignUp(s2SparseBlk1RealSize, 64)。
    // 用途：V1 softmax/VF 的 N 维对齐长度，以及 MX scale/PScale 的 group 对齐。
    uint32_t actSingleLoopS2SizeAlign = 0U;

    // sparseIndices[B, N1, Qb, 0] 的基地址，单位为 int32 元素。
    // 来源：ProcessMainLoop 根据 bIdx/n1Idx/s1OuterIdx/maxQb/maxKb 计算。
    // 用途：FillRunInfo 通过 sparseBase + s2LoopIdx 读取当前逻辑 KV block id。
    uint64_t sparseBase = 0U;
    // 当前 query block 有效 sparse KV block 数。
    // 来源：sparseSeqLen[B, N1, Qb]。
    // 用途：记录当前 task 对应的 sparse 长度；ProcessMainLoop 已用 sparseLen 控制 s2LoopEnd。
    int32_t actSparseLen = 0;
    // 当前 sparse block 在逻辑 KV token 维度上的起始 offset。
    // 来源：sparseBlkIdx1 * kvSparseBlockSize。
    // 用途：计算 S2 tail 长度，并参与 causal partial-mask 判断。
    int64_t sparseS2TokenOffset = 0;
    // 当前 sparseIndices 读出的逻辑 KV block id。
    // 来源：sparseIndicesGm[sparseBase + s2LoopIdx]。
    // 用途：cube 拷贝 K/V/KScale/VScale 时先得到逻辑 block，再结合 block_table 定位 PA BNBD 物理 block。
    int64_t sparseBlkIdx1 = -1;
    // 当前逻辑 KV block 的真实有效 token 数，最大为 kvSparseBlockSize=512。
    // 来源：min(kvSparseBlockSize, actS2Size - sparseS2TokenOffset)。
    // 用途：BMM1/V1/C2 的 S2/K 维真实长度，尾块为 0 时跳过 task。
    uint32_t s2SparseBlk1RealSize = 0U;
    // s2SparseBlk1RealSize 向上 64 对齐后的长度。
    // 来源：AlignUp(s2SparseBlk1RealSize, 64)。
    // 用途：MX value/P scale、V1 VF 和 BMM2 K 维对齐。
    uint32_t s2SparseBlk1RealAlignedSize = 0U;
    // 当前 sparse block 是否与 causal mask 边界相交。
    // 来源：AttentionMaskFullProcessingOrRequired。
    // 用途：当前 MX vector 尚未接入 attenMask，但保留该标志用于后续 mask 分支。
    bool sparseBlk1PartialMask = false;

    // 当前 batch 在 TND query/output 中的 token 基址。
    // 来源：cuSeqlensQ[bIdx]。
    // 用途：计算 query/qScale/attentionOut/softmaxLse 的全局 token offset。
    uint64_t queryTokenBase = 0U;
    // query 数据在 GM 中的起始 offset，单位为 fp8_e4m3fn 元素。
    // 来源：(queryTokenBase + s1Idx) * n2GD + realN2Idx * dSize。
    // 用途：cube 将 Q 从 GM 以 ND2NZ 搬到 L1。
    uint64_t queryOffset = 0U;
    // query antiquant scale 在 GM 中的起始 offset，单位为 fp8_e8m0 元素。
    // 来源：(queryTokenBase + s1Idx) * qScaleN1D + realN2Idx * queryScaleDSize * scaleLastDim。
    // 用途：cube 将 QScale 从 GM 以 DN2NZ 搬到 L1，并在 BMM1 LoadData 时做 MX 随路反量化。
    uint64_t queryScaleOffset = 0U;
};

// 基础 tensor 尺寸以及常用 TND stride。除非字段明确表示 DataCopyExt 的字节 stride，
// MX 路径中的 scale/data offset 都以元素个数为单位，而不是字节数。
struct MxBaseConstInfo {
    uint64_t t1Size = 0U;
    uint32_t dSize = 0U;
    uint32_t dSizeV = 0U;
    uint32_t gSize = 0U;
    uint32_t n2Size = 0U;
    uint32_t n1Size = 0U;
    uint32_t realGSize = 0U;
    uint32_t realN2Size = 0U;
    uint64_t s1Size = 0U;
    uint64_t s2Size = 0U;
    uint32_t coreNum = 0U;
    uint32_t aicIdx = 0U;
    uint32_t aivIdx = 0U;
    uint8_t subBlockIdx = 0U;
    uint32_t n2GD = 0U;
    uint32_t qScaleN1D = 0U;
    uint32_t kScaleN2D = 0U;
    uint32_t valueScaleN2D = 0U;
    uint32_t attentionOutStride = 0U;
    uint32_t softmaxLseStride = 0U;
    float scaleValue = 0.0F;
};

// MXFP8 全量化固定使用 PA BNBD。maxBlockNumPerBatch 和 paBlockStride
// 会配合 block_table 将逻辑 sparse token block 转换为物理 block。
struct MxPageAttentionConstInfo {
    uint32_t maxBlockNumPerBatch = 0U;
    uint32_t paBlockStride = 0U;
    uint32_t kvSparseBlockSize = 0U;
};

// sparse layout 为 B_N_Qb_Kb。maxQb/maxKb 用于索引 sparseSeqLen 和 sparseIndices。
struct MxSparseConstInfo {
    uint32_t maxQb = 0U;
    uint32_t maxKb = 0U;
};

// MX scale 元信息：
//   query/key scale D size = ceil(D / 64)，scaleLastDim=2，对应 [T, N, D/64, 2]。
//   valueScaleDSize 展平为 DV*2，对应 [T/64, N, DV, 2]。
struct MxScaleConstInfo {
    uint32_t scaleLastDim = BSA_MX_SCALE_LAST_DIM;
    uint32_t queryScaleDSize = 0U;
    uint32_t keyScaleDSize = 0U;
    uint32_t valueScaleDSize = 0U;
};

// 来自 host attrs/可选输入的特性开关。这里只保留当前独立 MX kernel 需要的字段；
// 与本路径无关的 common-MX 字段不引入，避免和 BSA FP8 路径耦合。
struct MxFeatureConstInfo {
    bool isSoftmaxLseEnable = false;
    bool isActualLenDimsNull = false;
    bool isActualLenDimsKVNull = false;
};

struct MxConstInfo : MxBaseConstInfo,
                     MxPageAttentionConstInfo,
                     MxSparseConstInfo,
                     MxScaleConstInfo,
                     MxFeatureConstInfo {};
} // namespace regbasemx

#endif // QUANT_BLOCK_SPARSE_ATTN_UTIL_REGBASE_MX_H_
