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
 * \file quant_block_sparse_attn_mx_kernel.h
 * \brief QuantBlockSparseAttn 的 MXFP8 全量化 kernel 入口。
 */
#ifndef QUANT_BLOCK_SPARSE_ATTN_MX_KERNEL_H_
#define QUANT_BLOCK_SPARSE_ATTN_MX_KERNEL_H_

#include "kernel_operator.h"
#include "../quant_block_sparse_attn_mx_tiling_data.h"
#include "quant_block_sparse_attn_common_arch35.h"
#include "quant_block_sparse_attn_kvcache.h"
#include "quant_block_sparse_attn_mx_block_cube.h"
#include "quant_block_sparse_attn_mx_block_vec.h"
#include "common/buffer_manager.h"
#include "common/buffers_policy.h"
#include "common/util_regbase_mx.h"

using namespace AscendC;
using namespace optiling;

namespace BaseApi {
/*
 * MXFP8 全量化路径与已有 FP8 kernel 路径隔离：
 *   1. quant_mode == 2 时，host 侧下发 QuantBlockSparseAttnMxTilingData；
 *   2. 本入口将精简后的 tiling data 转换为 MxConstInfo；
 *   3. cube 侧负责 QK/PV 矩阵乘，并在 L1->L0 搬运时完成 MX 随路反量化；
 *   4. vector 侧负责 softmax、P 静态量化、输出 update 以及可选 LSE 写回。
 *
 * 当前 MX 特性仅支持 TND query/output、PA BNBD key/value、D=DV=128，
 * kernel 计算 tile 仍为 512 token。当前 sparse_kv_block_size 可以是 128 或 64，
 * 因此一个 512 宽 logical S2 tile 会由多个 sparse KV block 拼接而成；
 * 512 宽 tile 再拆成两个 256 宽 C1/V1 subLoop，随后一次 C2/V2 消费合并后的
 * 128x512 P 矩阵。
 *
 * useDn 与 FP8 模板保持同一设计含义：控制 C1/V1 数据布局选择。当前 kernel
 * 入口固定实例化 useDn=true，MX kernel 内的非 DN 分支保留为后续扩展入口。
 */
template <bool useDn = true>
class QuantBlockSparseAttnMxKernel {
public:
    using INPUT_T = fp8_e4m3fn_t;
    using MM_T = float;
    static constexpr bool USE_DN = useDn;
    static constexpr uint32_t M_BASE = 128U;
    static constexpr uint32_t S2_BASE = 512U;
    static constexpr uint32_t S2_SPLIT = 256U;
    static constexpr uint32_t D_BASE = 128U;
    static constexpr uint32_t DV_BASE = 128U;
    // 延迟两个 task 后再启动 C2/V2，表示 C2/V2 执行时，后续两个 C1/V1 task 已经预取。
    // 每个 512 KV block 又会拆成两个 C1/V1 subLoop，因此启动阶段会先看到六次 C1/V1，
    // 再进入第一轮延迟 C2/V2，匹配当前 MX 流水预热方式。
    static constexpr uint32_t PRELOAD_N = 2U;
    static constexpr uint32_t PRELOAD_TASK_CACHE_SIZE = PRELOAD_N + 1U;

    __aicore__ inline QuantBlockSparseAttnMxKernel() = default;

    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                __gm__ uint8_t *sparseIndices, __gm__ uint8_t *sparseSeqLen,
                                __gm__ uint8_t *, __gm__ uint8_t *metadata, __gm__ uint8_t *cuSeqlensQ,
                                __gm__ uint8_t *cuSeqlensKv, __gm__ uint8_t *seqUsedQ,
                                __gm__ uint8_t *seqUsedKv, __gm__ uint8_t *blockTable, __gm__ uint8_t *qScale,
                                __gm__ uint8_t *kScale, __gm__ uint8_t *vScale, __gm__ uint8_t *pScale,
                                __gm__ uint8_t *softmaxLse, __gm__ uint8_t *attentionOut, __gm__ uint8_t *,
                                const QuantBlockSparseAttnMxTilingData *__restrict tiling, TPipe *tPipe)
    {
        // Q/K/V 保留原始 GM 指针，因为 cube 拷贝点需要按实际矩阵选择 ND2NZ/DN2NZ，
        // 并结合 block_table 做 PA 寻址。其他标量/索引类 tensor 在这里包装成
        // GlobalTensor，便于生成 RunInfo 时直接读取 sparse 元信息。
        pipe = tPipe;
        tilingData = tiling;
        queryPtr = query;
        keyPtr = key;
        valuePtr = value;
        sparseIndicesGm.SetGlobalBuffer((__gm__ int32_t *)sparseIndices);
        sparseSeqLenGm.SetGlobalBuffer((__gm__ int32_t *)sparseSeqLen);
        metadataGm.SetGlobalBuffer((__gm__ int32_t *)metadata);
        cuSeqlensQGm.SetGlobalBuffer((__gm__ int32_t *)cuSeqlensQ);
        cuSeqlensKvGm.SetGlobalBuffer((__gm__ int32_t *)cuSeqlensKv);
        seqUsedQGm.SetGlobalBuffer((__gm__ int32_t *)seqUsedQ);
        seqUsedKvGm.SetGlobalBuffer((__gm__ int32_t *)seqUsedKv);
        InitConstInfo();
        InitMMResBuf();
        cubeBlock.Init(pipe, &l1BufferManager, queryPtr, keyPtr, valuePtr, blockTable, qScale, kScale, vScale);
        vecBlock.Init(pipe, pScale, softmaxLse, attentionOut);
    }

    __aicore__ inline void Process()
    {
        if (constInfo.aicIdx >= constInfo.coreNum) {
            return;
        }
        CrossCoreBufferInit();
        if ASCEND_IS_AIC {
            cubeBlock.InitBuffers();
        } else {
            vecBlock.InitBuffers();
        }
        ProcessMainLoop();
        CrossCoreBufferUninit();
    }

private:
    __aicore__ inline void InitConstInfo()
    {
        // AIC 与 AIV 使用同一个逻辑 core id。AIV 额外记录 subBlockIdx，
        // 用于把 128 行 query 拆给两个 vector sub-block 分别处理。
        if ASCEND_IS_AIC {
            constInfo.aicIdx = GetBlockIdx();
        } else {
            constInfo.aivIdx = GetBlockIdx();
            constInfo.aicIdx = GetBlockIdx() / GetSubBlockNum();
            constInfo.subBlockIdx = GetSubBlockIdx();
        }

        const auto &baseParams = tilingData->baseParams;
        const auto &pageAttentionParams = tilingData->pageAttentionParams;
        const auto &sparseParams = tilingData->sparseParams;
        const auto &scaleParams = tilingData->scaleParams;

        constInfo.t1Size = baseParams.t1Size;
        constInfo.n2Size = baseParams.n2Size;
        constInfo.gSize = baseParams.gSize;
        constInfo.n1Size = baseParams.n2Size * baseParams.gSize;
        constInfo.realN2Size = constInfo.n1Size;
        constInfo.realGSize = 1U;
        constInfo.s1Size = baseParams.s1Size;
        constInfo.s2Size = baseParams.s2Size;
        constInfo.dSize = baseParams.dSize;
        constInfo.dSizeV = baseParams.dSizeV;
        constInfo.scaleValue = baseParams.scaleValue;
        constInfo.isSoftmaxLseEnable = baseParams.isSoftMaxLseEnable != 0U;
        constInfo.isActualLenDimsNull = baseParams.isActualSeqLengthsNull != 0U;
        constInfo.isActualLenDimsKVNull = baseParams.isActualSeqLengthsKVNull != 0U;
        constInfo.coreNum = baseParams.coreNum;
        constInfo.n2GD = constInfo.realN2Size * constInfo.dSize;
        constInfo.attentionOutStride = (constInfo.realN2Size - 1U) * constInfo.dSizeV * sizeof(bfloat16_t);
        constInfo.softmaxLseStride = (constInfo.realN2Size - 1U) * sizeof(float);

        constInfo.maxBlockNumPerBatch = pageAttentionParams.maxBlockNumPerBatch;
        constInfo.paBlockStride = pageAttentionParams.paBlockStride;
        constInfo.kvSparseBlockSize = pageAttentionParams.kvBlockSize;

        constInfo.maxQb = sparseParams.maxQb;
        constInfo.maxKb = sparseParams.maxKb;

        // host 侧下发的 MX scale shape：
        //   Q/K scale: [T, N, D/64, 2]，单个 token 行有 (D/64)*2 个 e8m0 元素；
        //   V scale:   [T/64, N, DV, 2]，host 将最后两个 e8m0 元素展平到
        //              valueScaleDSize = DV * 2。
        // 以下 stride 均以 e8m0 元素个数为单位，供 PA scale 拷贝 helper 使用；
        // scale 搬到 L1 后，再由 LoadData 完成 MX 随路反量化。
        constInfo.scaleLastDim = scaleParams.scaleLastDim;
        constInfo.queryScaleDSize = scaleParams.queryScaleDSize;
        constInfo.keyScaleDSize = scaleParams.keyScaleDSize;
        constInfo.valueScaleDSize = scaleParams.valueScaleDSize;
        constInfo.qScaleN1D = constInfo.realN2Size * constInfo.queryScaleDSize * constInfo.scaleLastDim;
        constInfo.kScaleN2D = constInfo.n2Size * constInfo.keyScaleDSize * constInfo.scaleLastDim;
        constInfo.valueScaleN2D = constInfo.n2Size * constInfo.valueScaleDSize;
    }

    __aicore__ inline void InitMMResBuf()
    {
        // 这里分配的是 AIC/AIV 之间跨核交接用的公共 buffer，设计背景参考 common MX
        // full-quant 的流水组织：C1 结果先从 cube fixpipe 到 UB，再由 V1 做 softmax；
        // V1 量化后的 P/PScale 留在 L1，延迟后的 C2 再从 L1 读取 P/PScale 参与 BMM2。
        //
        // bmm1Buffers 保存 C1 的 fp32 结果。虽然完整逻辑 P 是 [128, 512]，
        // 但 MX 路径会把一个 512 KV block 拆成两个 256 subLoop，且每个 AIV subBlock
        // 只处理 128 行 query 的一半，因此单个 UB C1 tile 实际是：
        //   (M_BASE / CV_RATIO) * (S2_BASE / 2) * sizeof(float)
        // 也就是 [64, 256] 的 fp32 tile。
        constexpr uint32_t mm1ResultSize = M_BASE / CV_RATIO * S2_BASE * sizeof(MM_T) / 2U;
        // bmm2Buffers 保存 C2 的 fp32 结果。C2 已经消费完整 P[128,512]，
        // 但 V2 仍按 AIV subBlock 处理半个 M 方向，所以单个 UB C2 tile 为：
        //   (M_BASE / CV_RATIO) * DV_BASE * sizeof(float)，即 [64, 128]。
        constexpr uint32_t mm2ResultSize = M_BASE / CV_RATIO * DV_BASE * sizeof(MM_T);
        // L1 P buffer 由 V1 和 C2 共享。这样做是为了避免把完整 P[128,512]
        // 写回 GM 再读入，直接在 L1 中完成 V1->C2 交接：
        //   [0, 128*512)                  : 量化后的 P，类型为 fp8_e4m3fn
        //   [128*512, 128*512 + scaleLen) : P scale，类型为 fp8_e8m0
        // scaleLen 按 MX group 计算，每 32 个 fp8 P 元素对应 1 个 e8m0 scale 元素。
        // C2 通过 MX LoadData 路径同时读取 P 和 PScale，并在 L1->L0 时完成随路反量化。
        constexpr uint32_t mm2LeftSize = M_BASE * S2_BASE * sizeof(INPUT_T) + M_BASE * S2_BASE / 32U;
        // 512KB 是 arch35 当前 L1 管理器的统一容量入口。cube 自己还会从同一个
        // l1BufferManager 中切出 Q/K/V 临时 L1 buffer；这里先切出跨 AIV/AIC 共享的 P buffer。
        l1BufferManager.Init(pipe, 512U * 1024U);
        l1PBuffers.Init(l1BufferManager, mm2LeftSize);
        // UB 中同时存在两类跨核结果：C1 使用双 buffer 以覆盖 C1/V1 流水，
        // C2 使用单 buffer，因为同一 delayed task 的 C2/V2 是顺序交接。
        // ubBufferManager 只负责统一申请总容量，具体 DB/single 策略由下面的
        // bmm1Buffers/bmm2Buffers 决定。
        ubBufferManager.Init(pipe, mm1ResultSize * 2U + mm2ResultSize);
        bmm1Buffers.Init(ubBufferManager, mm1ResultSize);
        bmm2Buffers.Init(ubBufferManager, mm2ResultSize);
    }

    __aicore__ inline void CrossCoreBufferInit()
    {
        if ASCEND_IS_AIV {
            bmm1Buffers.Get().SetCrossCore();
            bmm1Buffers.Get().SetCrossCore();
            bmm2Buffers.Get().SetCrossCore();
        }
    }

    __aicore__ inline void CrossCoreBufferUninit()
    {
        if ASCEND_IS_AIC {
            bmm1Buffers.Get().WaitCrossCore();
            bmm1Buffers.Get().WaitCrossCore();
            bmm2Buffers.Get().WaitCrossCore();
        }
    }

    __aicore__ inline uint32_t GetActualQSeqLen(uint32_t bIdx)
    {
        if (!constInfo.isActualLenDimsNull) {
            return static_cast<uint32_t>(seqUsedQGm.GetValue(bIdx));
        }
        return static_cast<uint32_t>(cuSeqlensQGm.GetValue(bIdx + 1U) - cuSeqlensQGm.GetValue(bIdx));
    }

    __aicore__ inline uint32_t GetActualKvSeqLen(uint32_t bIdx)
    {
        if (!constInfo.isActualLenDimsKVNull) {
            return static_cast<uint32_t>(seqUsedKvGm.GetValue(bIdx));
        }
        return static_cast<uint32_t>(cuSeqlensKvGm.GetValue(bIdx + 1U) - cuSeqlensKvGm.GetValue(bIdx));
    }

    __aicore__ inline uint32_t GetQTokenBase(uint32_t bIdx)
    {
        return static_cast<uint32_t>(cuSeqlensQGm.GetValue(bIdx));
    }

    __aicore__ inline uint32_t GetS1LoopStart(uint32_t bn1Idx, uint32_t bn1StartIdx, uint32_t s1StartIdx) const
    {
        return bn1Idx == bn1StartIdx ? s1StartIdx : 0U;
    }

    __aicore__ inline uint32_t GetS1LoopEnd(uint32_t bn1Idx, uint32_t bn1EndIdx, uint32_t s1EndIdx,
                                            uint32_t actualS1Size) const
    {
        const uint32_t loopEnd = (actualS1Size + M_BASE - 1U) / M_BASE;
        if (s1EndIdx != 0U && bn1Idx == bn1EndIdx - 1U) {
            return s1EndIdx;
        }
        return loopEnd;
    }

    __aicore__ inline uint32_t GetSparseBlocksPerTask() const
    {
        // MX C1/V1 的计算 tile 固定为 512 token；PA sparse block 可以是 128 或 64。
        // 因此一个 task 需要从 sparseIndices 连续取 4 个或 8 个 block 拼成 512。
        // host 当前只允许 128/64，这里仍保留数组容量上限，避免后续扩展时越界。
        const uint32_t blockCount = S2_BASE / constInfo.kvSparseBlockSize;
        if (blockCount == 0U) {
            return 1U;
        }
        return blockCount > regbasemx::BSA_MX_MAX_SPARSE_BLOCK_PER_TASK ?
            regbasemx::BSA_MX_MAX_SPARSE_BLOCK_PER_TASK : blockCount;
    }

    __aicore__ inline void FillRunInfo(MxRunInfo &runInfo, uint64_t loop, uint32_t mLoop, uint32_t bn1Idx,
                                       uint32_t s1OuterIdx, uint32_t s2LoopIdx, uint32_t s2LoopEnd,
                                       uint32_t sparseBlockCount, uint64_t sparseBase)
    {
        // 一个 RunInfo 表示一个 (B, N1, S1-outer) tile 下的一个 512-token logical S2 tile。
        // sparse_kv_block_size 为 128/64 时，当前 tile 会由多个 sparseIndices entry 拼接而成；
        // cube 后续逐块结合 block_table，找到 PA BNBD 布局中的物理 page。
        runInfo = {};
        runInfo.loop = static_cast<uint32_t>(loop);
        runInfo.mLoop = mLoop;
        runInfo.isValid = true;
        runInfo.bIdx = bn1Idx / constInfo.realN2Size;
        runInfo.n1Idx = bn1Idx % constInfo.realN2Size;
        runInfo.realN2Idx = runInfo.n1Idx;
        runInfo.n2Idx = runInfo.n1Idx / constInfo.gSize;
        runInfo.gIdx = runInfo.n1Idx % constInfo.gSize;
        runInfo.gS1Idx = s1OuterIdx * M_BASE;
        runInfo.s1Idx = s1OuterIdx * M_BASE;
        runInfo.queryTokenBase = GetQTokenBase(runInfo.bIdx);
        runInfo.actS1Size = GetActualQSeqLen(runInfo.bIdx);
        runInfo.actS2Size = GetActualKvSeqLen(runInfo.bIdx);
        runInfo.actMSize = M_BASE;
        if (runInfo.s1Idx + M_BASE > runInfo.actS1Size) {
            runInfo.actMSize = static_cast<uint32_t>(runInfo.actS1Size) - runInfo.s1Idx;
        }
        runInfo.actMSizeAlign32 = (runInfo.actMSize + 31U) >> 5 << 5;
        runInfo.actVecMSize = runInfo.actMSize <= 16U ? runInfo.actMSize : (runInfo.actMSizeAlign32 >> 1U);
        runInfo.vecMbaseIdx = 0U;
        if (constInfo.subBlockIdx == 1U) {
            runInfo.vecMbaseIdx = runInfo.actVecMSize;
            runInfo.actVecMSize = runInfo.actMSize - runInfo.actVecMSize;
        }
        runInfo.sparseBase = sparseBase;
        runInfo.actSparseLen =
            sparseSeqLenGm.GetValue((static_cast<uint64_t>(runInfo.bIdx) * constInfo.realN2Size + runInfo.n1Idx) *
                                        constInfo.maxQb +
                                    s1OuterIdx);
        runInfo.sparseBlockStartIdx = s2LoopIdx;
        runInfo.sparseBlockCount = sparseBlockCount > regbasemx::BSA_MX_MAX_SPARSE_BLOCK_PER_TASK ?
            regbasemx::BSA_MX_MAX_SPARSE_BLOCK_PER_TASK : sparseBlockCount;
        for (uint32_t i = 0U; i < runInfo.sparseBlockCount; ++i) {
            const int64_t sparseBlkIdx = sparseIndicesGm.GetValue(sparseBase + s2LoopIdx + i);
            const int64_t sparseTokenOffset = sparseBlkIdx * static_cast<int64_t>(constInfo.kvSparseBlockSize);
            const int64_t remainS2 = static_cast<int64_t>(runInfo.actS2Size) - sparseTokenOffset;
            const uint32_t sparseBlockRealSize =
                remainS2 > static_cast<int64_t>(constInfo.kvSparseBlockSize) ? constInfo.kvSparseBlockSize :
                    (remainS2 > 0 ? static_cast<uint32_t>(remainS2) : 0U);
            runInfo.sparseBlockIdx[i] = sparseBlkIdx;
            runInfo.sparseBlockTokenOffset[i] = sparseTokenOffset;
            runInfo.sparseBlockRealSize[i] = sparseBlockRealSize;
            runInfo.actSingleLoopS2Size += sparseBlockRealSize;
        }
        if (runInfo.sparseBlockCount != 0U) {
            runInfo.sparseBlkIdx1 = runInfo.sparseBlockIdx[0];
            runInfo.sparseS2TokenOffset = runInfo.sparseBlockTokenOffset[0];
            runInfo.s2SparseBlk1RealSize = runInfo.sparseBlockRealSize[0];
        }
        runInfo.s2SparseBlk1RealAlignedSize = (runInfo.actSingleLoopS2Size + 63U) >> 6 << 6;
        runInfo.actSingleLoopS2SizeAlign = runInfo.s2SparseBlk1RealAlignedSize;
        runInfo.isFirstS2Loop = s2LoopIdx == 0U;
        runInfo.isLastS2Loop = s2LoopIdx + runInfo.sparseBlockCount - 1U >= s2LoopEnd;
        runInfo.queryOffset =
            (runInfo.queryTokenBase + runInfo.s1Idx) * constInfo.n2GD + runInfo.realN2Idx * constInfo.dSize;
        runInfo.queryScaleOffset = (runInfo.queryTokenBase + runInfo.s1Idx) * constInfo.qScaleN1D +
                                   runInfo.realN2Idx * constInfo.queryScaleDSize * constInfo.scaleLastDim;
        // 保留 BSA sparse-mask 的判断结果到 RunInfo。当前 MX vector 调用路径暂未启用
        // mask 输入，但该标志记录 512 block 是否与 causal 边界相交，后续接入 mask 时
        // 可以直接复用这个入口。
        int64_t delta = static_cast<int64_t>(runInfo.actS2Size) - static_cast<int64_t>(runInfo.actS1Size);
        int64_t sparseBlkIdx = runInfo.sparseBlkIdx1;
        AttentionMaskFullProcessingOrRequired(runInfo.s1Idx, runInfo.sparseS2TokenOffset, delta,
                                              runInfo.actSingleLoopS2Size, sparseBlkIdx,
                                              runInfo.sparseBlk1PartialMask);
    }

    __aicore__ inline bool HasValidTask(MxRunInfo taskRunInfo[PRELOAD_TASK_CACHE_SIZE]) const
    {
        for (uint32_t i = 0U; i < PRELOAD_TASK_CACHE_SIZE; ++i) {
            if (taskRunInfo[i].isValid) {
                return true;
            }
        }
        return false;
    }

    __aicore__ inline void ExecuteTask(uint64_t loop, MxRunInfo taskRunInfo[PRELOAD_TASK_CACHE_SIZE])
    {
        // 当前 task 执行 C1/V1；PRELOAD_N 个 loop 之前的 task 执行 C2/V2。
        // 这对应“两轮 C1/V1 匹配一轮 C2/V2”的流水：
        // 当前 task 可能由 4 个 128-token 或 8 个 64-token sparse block 拼成 512-token tile。
        // C1(sub0)->V1(sub0)、C1(sub1)->V1(sub1) 在 L1 中生成完整 128x512 P，
        // 延迟后的 C2 再读取该 P/PScale，并与 V/VScale 做矩阵乘。
        MxRunInfo &runInfoCur = taskRunInfo[loop % PRELOAD_TASK_CACHE_SIZE];
        MxRunInfo &runInfoPre = taskRunInfo[(loop - PRELOAD_N) % PRELOAD_TASK_CACHE_SIZE];
        if (runInfoCur.isValid) {
            const uint32_t c1v1Loop = (runInfoCur.actSingleLoopS2Size + S2_SPLIT - 1U) / S2_SPLIT;
            if ASCEND_IS_AIC {
                for (uint32_t subLoop = 0U; subLoop < c1v1Loop; ++subLoop) {
                    cubeBlock.IterateBmm1(bmm1Buffers.Get(), runInfoCur, constInfo, subLoop);
                }
            } else {
                Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &pBuf = l1PBuffers.GetVec();
                for (uint32_t subLoop = 0U; subLoop < c1v1Loop; ++subLoop) {
                    vecBlock.ProcessVec1(pBuf, bmm1Buffers.Get(), runInfoCur, constInfo, subLoop);
                }
            }
        }
        if (loop >= PRELOAD_N && runInfoPre.isValid) {
            if ASCEND_IS_AIC {
                cubeBlock.IterateBmm2(bmm2Buffers.Get(), l1PBuffers, runInfoPre, constInfo);
            } else {
                vecBlock.ProcessVec2(bmm2Buffers.Get(), runInfoPre, constInfo);
            }
            runInfoPre.isValid = false;
        }
    }

    __aicore__ inline void ProcessMainLoop()
    {
        // 主循环的外层任务划分参考 BSA FP8 路径：
        //   experimental/attention/quant_block_sparse_attn/op_kernel/arch35/quant_block_sparse_attn_kernel.h
        //   中的 QuantBlockSparseAttnKernel::ProcessMainLoop。
        // host 已经在 metadata 中按 core 切好连续的 BN1/S1-outer 区间，这里只读取
        // bn1Start/bn1End/s1Start/s1End，不重新做切核策略，保证 MX 与 FP8 使用同一套
        // sparse_indices、sparseSeqLen 和 block_table 任务分配方式。
        uint32_t bn1StartIdx = GetBsaAttrMetadata(metadataGm, constInfo.aicIdx, BSA_FA_BN1_START_INDEX);
        uint32_t bn1EndIdx = GetBsaAttrMetadata(metadataGm, constInfo.aicIdx, BSA_FA_BN1_END_INDEX);
        uint32_t s1StartIdx = GetBsaAttrMetadata(metadataGm, constInfo.aicIdx, BSA_FA_S1_START_INDEX);
        uint32_t s1EndIdx = GetBsaAttrMetadata(metadataGm, constInfo.aicIdx, BSA_FA_S1_END_INDEX);
        if (s1EndIdx != 0U) {
            ++bn1EndIdx;
        }

        // taskRunInfo 是三槽环形缓存，参考 common MX full-quant 的流水组织：
        //   attention/common/op_kernel/arch35/fia_kernel_fullquant_mx.h
        //   attention/common/op_kernel/arch35/fia_block_cube_fullquant_mx.h
        //   attention/common/op_kernel/arch35/fia_block_vec_fullquant_mx.h
        // 当前 loop 的 task 先执行 C1/V1；PRELOAD_N 个 loop 之前的 task 再执行 C2/V2。
        // PRELOAD_N=2 表示流水启动时先预热两组 task。每个 task 仍是 512-token
        // logical tile，并拆成两个 256-token C1/V1 subLoop，所以实际开头会先看到多次 C1/V1，
        // 再由延迟 task 消费已经完整写入 L1 的 P/PScale 做 C2/V2。
        MxRunInfo taskRunInfo[PRELOAD_TASK_CACHE_SIZE] = {};
        uint64_t loop = 0U;
        uint32_t mLoop = 0U;
        for (uint32_t bn1Idx = bn1StartIdx; bn1Idx < bn1EndIdx; ++bn1Idx) {
            // bn1Idx 展开为 batch 和 query head。MX 当前 TND + GQA 场景下，
            // realN2Size 等价于 query head 数；n1/n2/g 的映射在 FillRunInfo 中完成。
            const uint32_t bIdx = bn1Idx / constInfo.realN2Size;
            const uint32_t actualS1Size = GetActualQSeqLen(bIdx);
            // 当前 core 可能只负责起始/结束 BN1 的一部分 S1 outer block；
            // 中间 BN1 则处理完整的 [0, ceil(actualS1/128))。
            const uint32_t s1LoopStart = GetS1LoopStart(bn1Idx, bn1StartIdx, s1StartIdx);
            const uint32_t s1LoopEnd = GetS1LoopEnd(bn1Idx, bn1EndIdx, s1EndIdx, actualS1Size);
            for (uint32_t s1OuterIdx = s1LoopStart; s1OuterIdx < s1LoopEnd; ++s1OuterIdx) {
                const uint32_t n1Idx = bn1Idx % constInfo.realN2Size;
                // sparseSeqLen[B, N1, Qb] 表示当前 128-row query block 有多少个有效
                // sparse KV block。长度为 0 时不需要进入 C1/V1/C2/V2。
                const int32_t sparseLen =
                    sparseSeqLenGm.GetValue((static_cast<uint64_t>(bIdx) * constInfo.realN2Size + n1Idx) *
                                                constInfo.maxQb +
                                            s1OuterIdx);
                if (sparseLen <= 0) {
                    continue;
                }
                // sparseBase 指向 sparseIndices[B, N1, Qb, 0]。后续 s2LoopIdx 会从该基址
                // 读取逻辑 KV block id；cube 再结合 block_table 找到 PA BNBD 的物理 block。
                const uint64_t sparseBase =
                    (static_cast<uint64_t>(bIdx) * constInfo.realN2Size + n1Idx) * constInfo.maxQb *
                        constInfo.maxKb +
                    static_cast<uint64_t>(s1OuterIdx) * constInfo.maxKb;
                const uint32_t s2LoopEnd = static_cast<uint32_t>(sparseLen - 1);
                const uint32_t sparseBlocksPerTask = GetSparseBlocksPerTask();
                for (uint32_t s2LoopIdx = 0U; s2LoopIdx <= s2LoopEnd; s2LoopIdx += sparseBlocksPerTask) {
                    MxRunInfo &runInfo = taskRunInfo[loop % PRELOAD_TASK_CACHE_SIZE];
                    const uint32_t remainSparseBlockCount = static_cast<uint32_t>(sparseLen) - s2LoopIdx;
                    const uint32_t sparseBlockCount =
                        remainSparseBlockCount > sparseBlocksPerTask ? sparseBlocksPerTask : remainSparseBlockCount;
                    // FillRunInfo 把当前 S2 tile 的所有运行期地址一次性算好：
                    // query/qScale 的 TND offset、当前 512 tile 内多个 sparseIndices 逻辑 block id、
                    // 实际 S2 tail 长度以及 causal partial-mask 标志。
                    FillRunInfo(runInfo, loop, mLoop, bn1Idx, s1OuterIdx, s2LoopIdx, s2LoopEnd, sparseBlockCount,
                                sparseBase);
                    if (runInfo.actSingleLoopS2Size != 0U && runInfo.actMSize != 0U) {
                        ExecuteTask(loop, taskRunInfo);
                        ++loop;
                    } else {
                        runInfo.isValid = false;
                    }
                }
                ++mLoop;
            }
        }
        // 主遍历结束后，环形缓存中仍可能有已经完成 C1/V1、但因 PRELOAD_N 延迟
        // 尚未执行 C2/V2 的 task。这里继续推进空 loop，把尾部 task 全部 drain 掉。
        while (HasValidTask(taskRunInfo)) {
            ExecuteTask(loop, taskRunInfo);
            ++loop;
        }
    }

    TPipe *pipe = nullptr;
    const QuantBlockSparseAttnMxTilingData *__restrict tilingData = nullptr;
    regbasemx::MxConstInfo constInfo;
    __gm__ uint8_t *queryPtr = nullptr;
    __gm__ uint8_t *keyPtr = nullptr;
    __gm__ uint8_t *valuePtr = nullptr;
    GlobalTensor<int32_t> sparseIndicesGm;
    GlobalTensor<int32_t> sparseSeqLenGm;
    GlobalTensor<int32_t> metadataGm;
    GlobalTensor<int32_t> cuSeqlensQGm;
    GlobalTensor<int32_t> cuSeqlensKvGm;
    GlobalTensor<int32_t> seqUsedQGm;
    GlobalTensor<int32_t> seqUsedKvGm;
    BufferManager<BufferType::UB> ubBufferManager;
    BufferManager<BufferType::L1> l1BufferManager;
    BuffersPolicyDB<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> bmm1Buffers;
    BuffersPolicySingleBuffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> bmm2Buffers;
    BuffersPolicy3buff<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> l1PBuffers;
    QuantBlockSparseAttnMxBlockCube<USE_DN> cubeBlock;
    QuantBlockSparseAttnMxBlockVec<USE_DN> vecBlock;
};
} // namespace BaseApi

#endif // QUANT_BLOCK_SPARSE_ATTN_MX_KERNEL_H_
