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
 * \file msa_index_score_epilogue.h
 * \brief AIV 侧 Epilogue：可选反量化列乘 -> atten_mask -> 分段 RowMax -> local_mask -> 写回。
 *
 * 数值路径：
 *   score = Maxpool[ (scale·) Q@Kᵀ + atten_mask ] + local_mask
 * atten_mask（sparse_mode=3）按 rightDownCausal 解析，不加载 [2048,2048] 模板。
 * local_mask 由 start_loc（query 所在逻辑 block）+ init/local_blocks 生成强制 +∞。
 */

#ifndef MSA_INDEX_SCORE_EPILOGUE_H
#define MSA_INDEX_SCORE_EPILOGUE_H

#include "kernel_operator.h"

#include "catlass/arch/arch.hpp"
#include "catlass/arch/resource.hpp"

#include "../msa_index_score_common.h"
#include "../msa_index_score_debug.h"
#include "msa_index_score_task.h"

namespace MsaIndexScoreNs {

class MsaSegRowMaxEpilogue {
public:
    using ArchTag = Catlass::Arch::AtlasA2;

    // UB 手工布局（字节偏移），catlass 不提供 UB allocator。
    static constexpr uint32_t UB_OFF_S = 0;
    static constexpr uint32_t UB_SIZE_S = MSA_ROWS_PER_PASS * MSA_BLOCKS_PER_STILE * MSA_BLOCK_SIZE * sizeof(float);
    static constexpr uint32_t UB_OFF_T64 = UB_OFF_S + UB_SIZE_S;
    static constexpr uint32_t UB_SIZE_T64 = MSA_REDUCE_ROWS * (MSA_BLOCK_SIZE / 2) * sizeof(float);
    static constexpr uint32_t UB_OFF_T8 = UB_OFF_T64 + UB_SIZE_T64;
    static constexpr uint32_t UB_SIZE_T8 = MSA_REDUCE_ROWS * MSA_FP32_PER_BLOCK * sizeof(float);
    static constexpr uint32_t UB_OFF_ROWMAX = UB_OFF_T8 + UB_SIZE_T8;
    static constexpr uint32_t UB_SIZE_ROWMAX = 512;
    // 反量化 scale 一页：block_size 个 fp32 = 512B。
    static constexpr uint32_t UB_OFF_DEQ = UB_OFF_ROWMAX + UB_SIZE_ROWMAX;
    static constexpr uint32_t UB_SIZE_DEQ = MSA_BLOCK_SIZE * sizeof(float);
    static constexpr uint32_t UB_TOTAL = UB_OFF_DEQ + UB_SIZE_DEQ;
    static_assert(UB_TOTAL <= ArchTag::UB_SIZE, "MsaSegRowMaxEpilogue UB out of bounds");

    static constexpr uint32_t STILE_WIDTH = MSA_BLOCKS_PER_STILE * MSA_BLOCK_SIZE;

    __aicore__ inline MsaSegRowMaxEpilogue() {}

    __aicore__ inline void Init(Catlass::Arch::Resource<ArchTag> &resource, uint32_t numQHeads,
                                uint32_t strideOutHead, uint32_t strideOutToken, uint32_t maxBlocksPerBatch,
                                uint32_t strideScalePage, uint32_t strideScaleHead, bool isQuant,
                                const AscendC::GlobalTensor<float> &gScore,
                                const AscendC::GlobalTensor<float> &gScale,
                                const AscendC::GlobalTensor<int32_t> &gBlockTable)
    {
        numQHeads_ = numQHeads;
        strideOutHead_ = strideOutHead;
        strideOutToken_ = strideOutToken;
        maxBlocksPerBatch_ = maxBlocksPerBatch;
        strideScalePage_ = strideScalePage;
        strideScaleHead_ = strideScaleHead;
        isQuant_ = isQuant;
        gScore_ = gScore;
        gScale_ = gScale;
        gBlockTable_ = gBlockTable;

        ubS_ = resource.ubBuf.template GetBufferByByte<float>(UB_OFF_S);
        ubT64_ = resource.ubBuf.template GetBufferByByte<float>(UB_OFF_T64);
        ubT8_ = resource.ubBuf.template GetBufferByByte<float>(UB_OFF_T8);
        ubRowMax_ = resource.ubBuf.template GetBufferByByte<float>(UB_OFF_ROWMAX);
        ubDeqScale_ = resource.ubBuf.template GetBufferByByte<float>(UB_OFF_DEQ);
    }

    /// 处理一个 S tile（MSA_BLOCKS_PER_STILE 个 sparse block）中属于本 subcore 的行。
    __aicore__ inline void ProcessSTile(const AscendC::GlobalTensor<float> &gS, const MsaTask &task,
                                        uint32_t blkBase, uint32_t subIdx, uint32_t subBlockNum,
                                        bool dumpDebug = false)
    {
        const uint32_t subM = MsaCeilDiv(task.mActual, subBlockNum);
        const uint32_t mOff = subIdx * subM;
        if (mOff >= task.mActual) {
            return;
        }
        const uint32_t mSub = MsaMinU32(subM, task.mActual - mOff);

#if MSA_INDEX_SCORE_DEBUG
        if (dumpDebug) {
            AscendC::printf(
                "[MSA_DBG][AIV] ProcessSTile blkBase=%u subIdx=%u mOff=%u mSub=%u "
                "fullEndBlk=%u visibleEndBlk=%u isQuant=%u\n",
                blkBase, subIdx, mOff, mSub, task.fullEndBlk, task.visibleEndBlk,
                static_cast<uint32_t>(isQuant_));
        }
#endif

        for (uint32_t p0 = 0; p0 < mSub; p0 += MSA_ROWS_PER_PASS) {
            const uint32_t rows = MsaMinU32(MSA_ROWS_PER_PASS, mSub - p0);
            ProcessPass(gS, task, blkBase, mOff + p0, rows, dumpDebug && (p0 == 0));
        }
    }

private:
    /// 处理一个 pass（最多 MSA_ROWS_PER_PASS 行 × MSA_BLOCKS_PER_STILE 个 block）。
    __aicore__ inline void ProcessPass(const AscendC::GlobalTensor<float> &gS, const MsaTask &task,
                                       uint32_t blkBase, uint32_t rowOff, uint32_t rows, bool dumpDebug)
    {
        const bool allInvalid = (blkBase >= task.visibleEndBlk);

#if MSA_INDEX_SCORE_DEBUG
        if (dumpDebug) {
            AscendC::printf(
                "[MSA_DBG][AIV] ProcessPass rowOff=%u rows=%u allInvalid=%u STILE_WIDTH=%u\n", rowOff,
                rows, static_cast<uint32_t>(allInvalid), STILE_WIDTH);
        }
#endif

        if (!allInvalid) {
#if MSA_INDEX_SCORE_DEBUG
            uint32_t sDumpN = MSA_DEBUG_DUMP_ELEMS;
            if (task.kvLen > 0 && task.kvLen <= 8) {
                sDumpN = static_cast<uint32_t>(task.kvLen) + 3U;
            }
#endif
            AscendC::DataCopy(ubS_, gS[static_cast<uint64_t>(rowOff) * STILE_WIDTH], rows * STILE_WIDTH);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

#if MSA_INDEX_SCORE_DEBUG
            if (dumpDebug) {
                AscendC::printf(
                    "[MSA_DBG][AIV] after DOT load S ub (row0, blk0 first %u fp32):\n", sDumpN);
                MsaDumpUbSample(ubS_, MSA_DUMP_DESC_S_DOT, sDumpN);
                AscendC::printf("[MSA_DBG][AIV] S_after_dot:");
                MsaPrintUbFloats(ubS_, sDumpN);
            }
#endif

            if (isQuant_) {
                ApplyDequantScale(task, blkBase, rows);
#if MSA_INDEX_SCORE_DEBUG
                if (dumpDebug) {
                    AscendC::printf("[MSA_DBG][AIV] S_after_dequant:");
                    MsaPrintUbFloats(ubS_, sDumpN);
                }
#endif
            }

            ApplyMask(task, blkBase, rowOff, rows);

#if MSA_INDEX_SCORE_DEBUG
            if (dumpDebug) {
                AscendC::printf(
                    "[MSA_DBG][AIV] after MASK S ub (row0, blk0 first %u fp32), "
                    "maskRange=[fullEndBlk,visibleEndBlk)=[%u,%u):\n",
                    sDumpN, task.fullEndBlk, task.visibleEndBlk);
                MsaDumpUbSample(ubS_, MSA_DUMP_DESC_S_MASK, sDumpN);
                AscendC::printf("[MSA_DBG][AIV] S_after_mask:");
                MsaPrintUbFloats(ubS_, sDumpN);

                // 逐行打印本 pass 的 validCount，核对 MASK 切分点。
                for (uint32_t r = 0; r < rows && r < 8U; ++r) {
                    const uint32_t rowInReq = task.mStart + rowOff + r;
                    const uint32_t tOff = rowInReq / numQHeads_;
                    const int32_t visibleKeyEnd = VisibleKeyEnd(task, static_cast<int32_t>(tOff));
                    const uint32_t blk = blkBase; // 小用例通常只有 blk0
                    int32_t validCount = visibleKeyEnd - static_cast<int32_t>(blk * MSA_BLOCK_SIZE);
                    if (validCount < 0) {
                        validCount = 0;
                    }
                    if (validCount > static_cast<int32_t>(MSA_BLOCK_SIZE)) {
                        validCount = static_cast<int32_t>(MSA_BLOCK_SIZE);
                    }
                    AscendC::printf(
                        "[MSA_DBG][AIV] row r=%u flat=%u tokenOff=%u visibleEnd=%d validCount=%d\n", r,
                        rowInReq, tOff, visibleKeyEnd, validCount);
                    if (validCount < static_cast<int32_t>(MSA_BLOCK_SIZE) && validCount >= 0) {
                        const uint32_t base = (r * MSA_BLOCKS_PER_STILE) * MSA_BLOCK_SIZE;
                        const uint32_t lo = (validCount >= 2) ? static_cast<uint32_t>(validCount - 2) : 0U;
                        const uint32_t hi = static_cast<uint32_t>(validCount) + 3U;
                        AscendC::printf("[MSA_DBG][AIV]   S_after_mask[%u:%u]:", lo, hi);
                        AscendC::PipeBarrier<PIPE_ALL>();
                        for (uint32_t i = lo; i < hi && i < MSA_BLOCK_SIZE; ++i) {
                            AscendC::printf(" %f", ubS_.GetValue(base + i));
                        }
                        AscendC::printf("\n");
                    }
                }
            }
#endif

            SegRowMax();

#if MSA_INDEX_SCORE_DEBUG
            if (dumpDebug) {
                AscendC::printf(
                    "[MSA_DBG][AIV] after MAX rowmax (rows[0:%u) x %u blocks):\n", rows,
                    MSA_BLOCKS_PER_STILE);
                MsaDumpUbSample(ubRowMax_, MSA_DUMP_DESC_SCORE_MAX, rows * MSA_BLOCKS_PER_STILE);
                for (uint32_t r = 0; r < rows && r < 8U; ++r) {
                    AscendC::printf("[MSA_DBG][AIV] score_after_max[r=%u]:", r);
                    AscendC::PipeBarrier<PIPE_ALL>();
                    for (uint32_t j = 0; j < MSA_BLOCKS_PER_STILE; ++j) {
                        AscendC::printf(" %f", ubRowMax_.GetValue(r * MSA_BLOCKS_PER_STILE + j));
                    }
                    AscendC::printf("\n");
                }
            }
#endif
        } else {
            AscendC::Duplicate(ubRowMax_, MSA_FILL_VALUE, MSA_REDUCE_ROWS);
            AscendC::PipeBarrier<PIPE_V>();
#if MSA_INDEX_SCORE_DEBUG
            if (dumpDebug) {
                AscendC::printf("[MSA_DBG][AIV] allInvalid stile -> fill score with -inf\n");
            }
#endif
        }

        FillInvalidBlocks(task, blkBase);
        ApplyLocalMaskUb(task, blkBase, rows);

#if MSA_INDEX_SCORE_DEBUG
        if (dumpDebug) {
            AscendC::printf(
                "[MSA_DBG][AIV] final score before store (after fillInvalid+localMask), "
                "rows[0:%u) x %u blocks:\n",
                rows, MSA_BLOCKS_PER_STILE);
            MsaDumpUbSample(ubRowMax_, MSA_DUMP_DESC_SCORE_FINAL, rows * MSA_BLOCKS_PER_STILE);
            for (uint32_t r = 0; r < rows && r < 8U; ++r) {
                AscendC::printf("[MSA_DBG][AIV] score_final[r=%u]:", r);
                AscendC::PipeBarrier<PIPE_ALL>();
                for (uint32_t j = 0; j < MSA_BLOCKS_PER_STILE; ++j) {
                    AscendC::printf(" %f", ubRowMax_.GetValue(r * MSA_BLOCKS_PER_STILE + j));
                }
                AscendC::printf("\n");
            }
        }
#endif

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        StoreScore(task, blkBase, rowOff, rows);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
    }

    /// sparse_mode=3：rightDownCausal；否则仅 kv_len 截断。
    __aicore__ inline int32_t VisibleKeyEnd(const MsaTask &task, int32_t tOff) const
    {
        if (task.sparseMode == MSA_SPARSE_MODE_RIGHT_DOWN) {
            return MsaClampI32(task.kvLen - task.qLen + tOff + 1, 0, task.kvLen < 0 ? 0 : task.kvLen);
        }
        return task.kvLen < 0 ? 0 : task.kvLen;
    }

    /// 只对 [fullEndBlk, visibleEndBlk) 的边界 block 逐行置 -inf。
    /// 全可见 block 走无 mask 快通道；全不可见 block 的结果在 FillInvalidBlocks 中整体覆盖。
    __aicore__ inline void ApplyMask(const MsaTask &task, uint32_t blkBase, uint32_t rowOff, uint32_t rows)
    {
        if (task.fullEndBlk >= task.visibleEndBlk) {
            return;
        }
        const uint32_t jLo = (task.fullEndBlk > blkBase) ? (task.fullEndBlk - blkBase) : 0;
        if (jLo >= MSA_BLOCKS_PER_STILE) {
            return;
        }
        const uint32_t jHi = MsaMinU32(MSA_BLOCKS_PER_STILE,
                                       (task.visibleEndBlk > blkBase) ? (task.visibleEndBlk - blkBase) : 0);

        bool touched = false;
        for (uint32_t r = 0; r < rows; ++r) {
            const uint32_t rowInReq = task.mStart + rowOff + r;
            const uint32_t tOff = rowInReq / numQHeads_;
            const int32_t visibleKeyEnd = VisibleKeyEnd(task, static_cast<int32_t>(tOff));
            for (uint32_t j = jLo; j < jHi; ++j) {
                const uint32_t blk = blkBase + j;
                int32_t validCount = visibleKeyEnd - static_cast<int32_t>(blk * MSA_BLOCK_SIZE);
                if (validCount < 0) {
                    validCount = 0;
                }
                if (validCount >= static_cast<int32_t>(MSA_BLOCK_SIZE)) {
                    continue;
                }
                FillBlockTail((r * MSA_BLOCKS_PER_STILE + j) * MSA_BLOCK_SIZE,
                              static_cast<uint32_t>(validCount));
                touched = true;
            }
        }
        if (touched) {
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    /// Maxpool 之后、Store 之前施加 local_mask。
    /// 用与 FillInvalidBlocks 相同的位掩码 Duplicate，保证 32B 对齐访问。
    __aicore__ inline void ApplyLocalMaskUb(const MsaTask &task, uint32_t blkBase, uint32_t rows)
    {
        if (task.initBlocks == 0 && task.localBlocks == 0) {
            return;
        }
        const int32_t qBlock = task.startLoc;
        const int32_t localStart =
            (qBlock + 1 > static_cast<int32_t>(task.localBlocks))
                ? (qBlock + 1 - static_cast<int32_t>(task.localBlocks))
                : 0;

        uint64_t initBits = 0;
        uint64_t localBits = 0;
        for (uint32_t j = 0; j < MSA_BLOCKS_PER_STILE; ++j) {
            const int32_t blk = static_cast<int32_t>(blkBase + j);
            if (blk < 0 || blk >= static_cast<int32_t>(task.visibleEndBlk)) {
                continue;
            }
            if (static_cast<uint32_t>(blk) < task.initBlocks) {
                initBits |= (1ULL << j);
            }
            if (blk >= localStart && blk <= qBlock) {
                localBits |= (1ULL << j);
            }
        }
        initBits &= ~localBits; // local 覆盖 init

        const uint8_t repeats =
            static_cast<uint8_t>((rows * MSA_BLOCKS_PER_STILE + MSA_FP32_PER_REPEAT - 1) / MSA_FP32_PER_REPEAT);
        if (initBits != 0) {
            uint64_t bits = 0;
            for (uint32_t g = 0; g < MSA_FP32_PER_REPEAT / MSA_BLOCKS_PER_STILE; ++g) {
                bits |= (initBits << (g * MSA_BLOCKS_PER_STILE));
            }
            uint64_t mask[2] = {bits, 0};
            AscendC::Duplicate<float>(ubRowMax_, MSA_LOCAL_SCORE_INIT, mask, repeats, 1,
                                      MSA_FP32_PER_REPEAT / MSA_FP32_PER_BLOCK);
            AscendC::PipeBarrier<PIPE_V>();
        }
        if (localBits != 0) {
            uint64_t bits = 0;
            for (uint32_t g = 0; g < MSA_FP32_PER_REPEAT / MSA_BLOCKS_PER_STILE; ++g) {
                bits |= (localBits << (g * MSA_BLOCKS_PER_STILE));
            }
            uint64_t mask[2] = {bits, 0};
            AscendC::Duplicate<float>(ubRowMax_, MSA_LOCAL_SCORE_LOCAL, mask, repeats, 1,
                                      MSA_FP32_PER_REPEAT / MSA_FP32_PER_BLOCK);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    /// 把 ubS_[base + validCount, base + MSA_BLOCK_SIZE) 置为 -inf。
    /// 拆成"整 repeat 尾段"+"单 repeat 内的位掩码"两步，保证每次访问都是 32B 对齐的。
    __aicore__ inline void FillBlockTail(uint32_t base, uint32_t validCount)
    {
        const uint32_t alignedStart = MsaCeilDiv(validCount, MSA_FP32_PER_REPEAT) * MSA_FP32_PER_REPEAT;
        if (alignedStart < MSA_BLOCK_SIZE) {
            AscendC::Duplicate(ubS_[base + alignedStart], MSA_FILL_VALUE, MSA_BLOCK_SIZE - alignedStart);
        }
        if (validCount < alignedStart) {
            const uint32_t repeatBase = base + (validCount / MSA_FP32_PER_REPEAT) * MSA_FP32_PER_REPEAT;
            const uint32_t lane = validCount % MSA_FP32_PER_REPEAT;
            uint64_t bits = (lane == 0) ? ~0ULL : (~0ULL << lane);
            uint64_t mask[2] = {bits, 0};
            AscendC::Duplicate<float>(ubS_[repeatBase], MSA_FILL_VALUE, mask, 1, 1, 8);
        }
    }

    /// int8 路径：S 已是 Q · cast(K_int8)，再按列乘 scale[page,0,n] 完成反量化。
    /// 数学上 Q·(K_int8*s) = (Q·cast(K_int8))*s（int8→fp 对整数精确）。
    __aicore__ inline void ApplyDequantScale(const MsaTask &task, uint32_t blkBase, uint32_t rows)
    {
        for (uint32_t j = 0; j < MSA_BLOCKS_PER_STILE; ++j) {
            const uint32_t blk = blkBase + j;
            if (blk >= task.visibleEndBlk) {
                break;
            }
            const int32_t page = gBlockTable_.GetValue(task.batchIdx * maxBlocksPerBatch_ + blk);
            // Hkv==1：取 scale[page, 0, :]
            const uint64_t scaleOff = static_cast<uint64_t>(page) * strideScalePage_;
            AscendC::DataCopy(ubDeqScale_, gScale_[scaleOff], MSA_BLOCK_SIZE);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);

            for (uint32_t r = 0; r < rows; ++r) {
                const uint32_t base = r * STILE_WIDTH + j * MSA_BLOCK_SIZE;
                AscendC::Mul(ubS_[base], ubS_[base], ubDeqScale_, static_cast<int32_t>(MSA_BLOCK_SIZE));
            }
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    /// 把 ubS_ 视作 [MSA_REDUCE_ROWS, MSA_BLOCK_SIZE] 做行归约，结果落在 ubRowMax_[MSA_REDUCE_ROWS]，
    /// 其排布恰为 [MSA_ROWS_PER_PASS][MSA_BLOCKS_PER_STILE] 行主序。
    __aicore__ inline void SegRowMax()
    {
        constexpr uint32_t HALF_BLOCK = MSA_BLOCK_SIZE / 2;
        constexpr uint8_t SRC_REP_BLK = MSA_BLOCK_SIZE / MSA_FP32_PER_BLOCK; // 16
        constexpr uint8_t DST_REP_BLK = HALF_BLOCK / MSA_FP32_PER_BLOCK;     // 8

        AscendC::Max<float>(ubT64_, ubS_, ubS_[HALF_BLOCK], static_cast<int32_t>(MSA_FP32_PER_REPEAT),
                            static_cast<uint8_t>(MSA_REDUCE_ROWS),
                            AscendC::BinaryRepeatParams(1, 1, 1, DST_REP_BLK, SRC_REP_BLK, SRC_REP_BLK));
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::BlockReduceMax<float, true>(ubT8_, ubT64_, static_cast<uint8_t>(MSA_REDUCE_ROWS),
                                             static_cast<int32_t>(MSA_FP32_PER_REPEAT), 1, 1, DST_REP_BLK);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::BlockReduceMax<float, true>(ubRowMax_, ubT8_,
                                             static_cast<uint8_t>(MSA_REDUCE_ROWS / MSA_FP32_PER_BLOCK),
                                             static_cast<int32_t>(MSA_FP32_PER_REPEAT), 1, 1, DST_REP_BLK);
        AscendC::PipeBarrier<PIPE_V>();
    }

    /// blk >= visibleEndBlk 的位置整体置为 -inf。
    __aicore__ inline void FillInvalidBlocks(const MsaTask &task, uint32_t blkBase)
    {
        const uint32_t invStart = (task.visibleEndBlk > blkBase) ? (task.visibleEndBlk - blkBase) : 0;
        if (invStart >= MSA_BLOCKS_PER_STILE) {
            return;
        }
        if (invStart == 0) {
            AscendC::Duplicate(ubRowMax_, MSA_FILL_VALUE, MSA_REDUCE_ROWS);
            AscendC::PipeBarrier<PIPE_V>();
            return;
        }
        uint64_t groupBits = 0;
        for (uint32_t k = invStart; k < MSA_BLOCKS_PER_STILE; ++k) {
            groupBits |= (1ULL << k);
        }
        uint64_t bits = 0;
        for (uint32_t g = 0; g < MSA_FP32_PER_REPEAT / MSA_BLOCKS_PER_STILE; ++g) {
            bits |= (groupBits << (g * MSA_BLOCKS_PER_STILE));
        }
        uint64_t mask[2] = {bits, 0};
        AscendC::Duplicate<float>(ubRowMax_, MSA_FILL_VALUE, mask,
                                  static_cast<uint8_t>(MSA_REDUCE_ROWS / MSA_FP32_PER_REPEAT), 1,
                                  MSA_FP32_PER_REPEAT / MSA_FP32_PER_BLOCK);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void StoreScore(const MsaTask &task, uint32_t blkBase, uint32_t rowOff, uint32_t rows)
    {
        for (uint32_t r = 0; r < rows; ++r) {
            const uint32_t rowInReq = task.mStart + rowOff + r;
            const uint32_t tOff = rowInReq / numQHeads_;
            const uint32_t h = rowInReq - tOff * numQHeads_;
            const uint64_t off = static_cast<uint64_t>(h) * strideOutHead_ +
                                 static_cast<uint64_t>(task.cuQStart + tOff) * strideOutToken_ + blkBase;
            AscendC::DataCopy(gScore_[off], ubRowMax_[r * MSA_BLOCKS_PER_STILE], MSA_BLOCKS_PER_STILE);
        }
    }

    AscendC::GlobalTensor<float> gScore_;
    AscendC::GlobalTensor<float> gScale_;
    AscendC::GlobalTensor<int32_t> gBlockTable_;
    AscendC::LocalTensor<float> ubS_;
    AscendC::LocalTensor<float> ubT64_;
    AscendC::LocalTensor<float> ubT8_;
    AscendC::LocalTensor<float> ubRowMax_;
    AscendC::LocalTensor<float> ubDeqScale_;

    uint32_t numQHeads_ = 1;
    uint32_t strideOutHead_ = 0;
    uint32_t strideOutToken_ = 0;
    uint32_t maxBlocksPerBatch_ = 0;
    uint32_t strideScalePage_ = 0;
    uint32_t strideScaleHead_ = 1;
    bool isQuant_ = false;
};

} // namespace MsaIndexScoreNs

#endif // MSA_INDEX_SCORE_EPILOGUE_H
