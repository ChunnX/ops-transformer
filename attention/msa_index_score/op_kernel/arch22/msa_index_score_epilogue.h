/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file msa_index_score_epilogue.h
 * \brief AIV 侧 Epilogue：mask -> 分段 RowMax -> scale -> 写回。
 *
 * Atlas A2 没有 L0C->UB 直通路径，Cube 产出的 S 必须经 GM workspace 中转，
 * 本 Epilogue 从 workspace 读取 S tile 并完成后续全部矢量计算。
 *
 * 归约的段长恰等于 sparse block 宽度（MSA_BLOCK_SIZE），且一个 S tile 的
 * [rows, BLOCKS, BLOCK_SIZE] 在内存上等价于 [rows*BLOCKS, BLOCK_SIZE]，
 * 因此"逐 block 求最大值"退化为对该二维视图做一次标准行归约，
 * 归约结果天然按 [row][block] 行主序排布，可直接整块写回。
 */

#ifndef MSA_INDEX_SCORE_EPILOGUE_H
#define MSA_INDEX_SCORE_EPILOGUE_H

#include "kernel_operator.h"

#include "catlass/arch/arch.hpp"
#include "catlass/arch/resource.hpp"

#include "../msa_index_score_common.h"
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
    static constexpr uint32_t UB_OFF_SCALE = UB_OFF_ROWMAX + UB_SIZE_ROWMAX;
    static constexpr uint32_t UB_SIZE_SCALE = 512;
    static constexpr uint32_t UB_OFF_SCALE_BRC = UB_OFF_SCALE + UB_SIZE_SCALE;
    static constexpr uint32_t UB_SIZE_SCALE_BRC = 512;
    static constexpr uint32_t UB_TOTAL = UB_OFF_SCALE_BRC + UB_SIZE_SCALE_BRC;
    static_assert(UB_TOTAL <= ArchTag::UB_SIZE, "MsaSegRowMaxEpilogue UB out of bounds");

    static constexpr uint32_t STILE_WIDTH = MSA_BLOCKS_PER_STILE * MSA_BLOCK_SIZE;

    __aicore__ inline MsaSegRowMaxEpilogue() {}

    __aicore__ inline void Init(Catlass::Arch::Resource<ArchTag> &resource, uint32_t numQHeads,
                                uint32_t strideOutHead, uint32_t strideOutToken, uint32_t strideScaleT,
                                uint32_t strideScaleN, bool hasScale,
                                const AscendC::GlobalTensor<float> &gScore,
                                const AscendC::GlobalTensor<float> &gScale)
    {
        numQHeads_ = numQHeads;
        strideOutHead_ = strideOutHead;
        strideOutToken_ = strideOutToken;
        strideScaleT_ = strideScaleT;
        strideScaleN_ = strideScaleN;
        hasScale_ = hasScale;
        gScore_ = gScore;
        gScale_ = gScale;

        ubS_ = resource.ubBuf.template GetBufferByByte<float>(UB_OFF_S);
        ubT64_ = resource.ubBuf.template GetBufferByByte<float>(UB_OFF_T64);
        ubT8_ = resource.ubBuf.template GetBufferByByte<float>(UB_OFF_T8);
        ubRowMax_ = resource.ubBuf.template GetBufferByByte<float>(UB_OFF_ROWMAX);
        ubScale_ = resource.ubBuf.template GetBufferByByte<float>(UB_OFF_SCALE);
        ubScaleBrc_ = resource.ubBuf.template GetBufferByByte<float>(UB_OFF_SCALE_BRC);
    }

    /// 处理一个 S tile（MSA_BLOCKS_PER_STILE 个 sparse block）中属于本 subcore 的行。
    __aicore__ inline void ProcessSTile(const AscendC::GlobalTensor<float> &gS, const MsaTask &task,
                                        uint32_t blkBase, uint32_t subIdx, uint32_t subBlockNum)
    {
        const uint32_t subM = MsaCeilDiv(task.mActual, subBlockNum);
        const uint32_t mOff = subIdx * subM;
        if (mOff >= task.mActual) {
            return;
        }
        const uint32_t mSub = MsaMinU32(subM, task.mActual - mOff);

        for (uint32_t p0 = 0; p0 < mSub; p0 += MSA_ROWS_PER_PASS) {
            const uint32_t rows = MsaMinU32(MSA_ROWS_PER_PASS, mSub - p0);
            ProcessPass(gS, task, blkBase, mOff + p0, rows);
        }
    }

private:
    /// 处理一个 pass（最多 MSA_ROWS_PER_PASS 行 × MSA_BLOCKS_PER_STILE 个 block）。
    __aicore__ inline void ProcessPass(const AscendC::GlobalTensor<float> &gS, const MsaTask &task,
                                       uint32_t blkBase, uint32_t rowOff, uint32_t rows)
    {
        const bool allInvalid = (blkBase >= task.visibleEndBlk);

        if (!allInvalid) {
            AscendC::DataCopy(ubS_, gS[static_cast<uint64_t>(rowOff) * STILE_WIDTH], rows * STILE_WIDTH);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

            ApplyMask(task, blkBase, rowOff, rows);
            SegRowMax();
        } else {
            AscendC::Duplicate(ubRowMax_, MSA_FILL_VALUE, MSA_REDUCE_ROWS);
            AscendC::PipeBarrier<PIPE_V>();
        }

        if (hasScale_) {
            ApplyScale(task, rowOff, rows);
        }
        FillInvalidBlocks(task, blkBase);

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        StoreScore(task, blkBase, rowOff, rows);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
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
            const int32_t qPos = task.startLoc + static_cast<int32_t>(tOff);
            int32_t visibleKeyEnd = (qPos + 1) < task.kvLen ? (qPos + 1) : task.kvLen;
            if (visibleKeyEnd < 0) {
                visibleKeyEnd = 0;
            }
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

    /// 把 ubS_ 视作 [MSA_REDUCE_ROWS, MSA_BLOCK_SIZE] 做行归约，结果落在 ubRowMax_[MSA_REDUCE_ROWS]，
    /// 其排布恰为 [MSA_ROWS_PER_PASS][MSA_BLOCKS_PER_STILE] 行主序。
    __aicore__ inline void SegRowMax()
    {
        constexpr uint32_t HALF_BLOCK = MSA_BLOCK_SIZE / 2;
        constexpr uint8_t SRC_REP_BLK = MSA_BLOCK_SIZE / MSA_FP32_PER_BLOCK; // 16
        constexpr uint8_t DST_REP_BLK = HALF_BLOCK / MSA_FP32_PER_BLOCK;     // 8

        // 128 -> 64：两半取 max，一次 repeat 覆盖一行的前/后 64 个元素。
        AscendC::Max<float>(ubT64_, ubS_, ubS_[HALF_BLOCK], static_cast<int32_t>(MSA_FP32_PER_REPEAT),
                            static_cast<uint8_t>(MSA_REDUCE_ROWS),
                            AscendC::BinaryRepeatParams(1, 1, 1, DST_REP_BLK, SRC_REP_BLK, SRC_REP_BLK));
        AscendC::PipeBarrier<PIPE_V>();

        // 64 -> 8：每个 32B datablock 归约成 1 个。
        AscendC::BlockReduceMax<float, true>(ubT8_, ubT64_, static_cast<uint8_t>(MSA_REDUCE_ROWS),
                                             static_cast<int32_t>(MSA_FP32_PER_REPEAT), 1, 1, DST_REP_BLK);
        AscendC::PipeBarrier<PIPE_V>();

        // 8 -> 1：一行的 8 个中间值恰好占满一个 datablock。
        AscendC::BlockReduceMax<float, true>(ubRowMax_, ubT8_,
                                             static_cast<uint8_t>(MSA_REDUCE_ROWS / MSA_FP32_PER_BLOCK),
                                             static_cast<int32_t>(MSA_FP32_PER_REPEAT), 1, 1, DST_REP_BLK);
        AscendC::PipeBarrier<PIPE_V>();
    }

    /// scale 为 per-token per-head 正值，max 具有正齐次性，故在归约之后施加，
    /// 作用对象从 [rows, BLOCK_SIZE*BLOCKS] 缩小到 [rows, BLOCKS]。
    __aicore__ inline void ApplyScale(const MsaTask &task, uint32_t rowOff, uint32_t rows)
    {
        const uint32_t rowBase = task.globalRowBase + rowOff;
        if (strideScaleN_ == 1) {
            AscendC::DataCopyExtParams params{1, rows * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
            AscendC::DataCopyPad(ubScale_, gScale_[rowBase], params, padParams);
        } else {
            for (uint32_t r = 0; r < rows; ++r) {
                const uint32_t rowInReq = task.mStart + rowOff + r;
                const uint32_t tOff = rowInReq / numQHeads_;
                const uint32_t h = rowInReq - tOff * numQHeads_;
                const uint64_t off = static_cast<uint64_t>(task.cuQStart + tOff) * strideScaleT_ +
                                     static_cast<uint64_t>(h) * strideScaleN_;
                ubScale_.SetValue(r, gScale_.GetValue(off));
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);

        AscendC::Brcb(ubScaleBrc_, ubScale_, static_cast<uint8_t>(MsaCeilDiv(rows, MSA_FP32_PER_BLOCK)),
                      AscendC::BrcbRepeatParams(1, MSA_FP32_PER_BLOCK));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(ubRowMax_, ubRowMax_, ubScaleBrc_, static_cast<int32_t>(rows * MSA_BLOCKS_PER_STILE));
        AscendC::PipeBarrier<PIPE_V>();
    }

    /// blk >= visibleEndBlk 的位置整体置为 -inf（在 scale 之后，避免被缩放污染）。
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
        // 一个 repeat 覆盖 MSA_FP32_PER_REPEAT 个元素 = 8 行输出，按 (idx % BLOCKS) >= invStart 置位。
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

    /// 每行的 MSA_BLOCKS_PER_STILE 个结果恰好 32B，且 GM 目的地址天然 32B 对齐，
    /// 因此可以用整块 DataCopy 写回，无需 DataCopyPad。
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
    AscendC::LocalTensor<float> ubS_;
    AscendC::LocalTensor<float> ubT64_;
    AscendC::LocalTensor<float> ubT8_;
    AscendC::LocalTensor<float> ubRowMax_;
    AscendC::LocalTensor<float> ubScale_;
    AscendC::LocalTensor<float> ubScaleBrc_;

    uint32_t numQHeads_ = 1;
    uint32_t strideOutHead_ = 0;
    uint32_t strideOutToken_ = 0;
    uint32_t strideScaleT_ = 0;
    uint32_t strideScaleN_ = 1;
    bool hasScale_ = false;
};

} // namespace MsaIndexScoreNs

#endif // MSA_INDEX_SCORE_EPILOGUE_H
