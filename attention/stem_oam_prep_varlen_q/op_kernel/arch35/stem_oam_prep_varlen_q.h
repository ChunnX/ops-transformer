/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * stem_oam_prep_varlen_q kernel (arch35 regbase, __simd_vf__)
 *
 * GM 输入:
 *   q:           [total_tokens, H_q, D]       FP8_e4m3fn
 *   qscale:      [total_tokens, H_q]          FP32
 *   qSeqLens:    [batch]                       INT32
 *   cuSeqLensQ:  [batch+1]                     INT32
 *
 * GM 输出:
 *   qFlat:       [batch, H_q, max_Qb, S*D]    BF16
 *
 * 参数: B=stemBlockSize(128), S=stemStride(16), R=B/S(8), D=dim_qk(128)
 *
 * 计算流程 (每核处理 1 个 qb block):
 *
 *   1. CopyInQBlock:
 *      GM → UB: q[cu_off + qb*B : cu_off + qb*B + B, h, :]
 *      qBlockBuf: [B, D] FP8, 超出 q_len 的行填零
 *
 *   2. CastBlockToFP32:
 *      UB: qBlockBuf [B, D] FP8 → qBlockFP32Buf [B, D] FP32
 *
 *   3. CopyInScalesBulk:
 *      GM → UB: qscale[cu_off + qb*B : cu_off + qb*B + B, h]
 *      scalesBuf: [B] FP32
 *
 *   4. AccumulateBlock (Weighted Group Sum):
 *      qBlockFP32Buf 逻辑 reshape 为 [R, S, D]
 *      scalesBuf 逻辑 reshape 为 [R, S]
 *
 *      对每个 g ∈ [0, S-1]:
 *        groupSum[g, :] = Σ_{r=0}^{R-1} qBlockFP32[r*S+g, :] × scales[r*S+g]
 *                         ↑ 沿 R 轴累加, 每行 D=128 个 FP32
 *
 *      groupSumBuf: [S, D] FP32 = [S*D] FP32
 *
 *   5. FlattenAndCast:
 *      UB: groupSumBuf [S*D] FP32 → Cast → outQueueQflat [S*D] BF16
 *
 *   6. CopyOutBlock:
 *      UB → GM: qFlat[b, h, qb, :] = outQueueQflat [S*D] BF16
 */

#ifndef STEM_OAM_PREP_VARLEN_Q_REGBASE_H
#define STEM_OAM_PREP_VARLEN_Q_REGBASE_H

#include "kernel_operator.h"
#include "stem_oam_prep_varlen_q_tiling_data.h"

using namespace AscendC;

constexpr uint32_t DIM_QK = 128;
constexpr uint32_t MAX_STEM_BLOCK_SIZE = 256;
constexpr uint32_t MAX_STEM_STRIDE = 64;
constexpr uint32_t MAX_QFLAT_DIM = MAX_STEM_STRIDE * DIM_QK;
constexpr uint32_t MAX_BATCH_SIZE = 1024;
constexpr uint32_t VARLEN_READ_ALIGN = 8;

constexpr static AscendC::MicroAPI::CastTrait CAST_FP8_TO_FP32 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN
};

constexpr static AscendC::MicroAPI::CastTrait CAST_FP32_TO_BF16 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT
};

/**
 * CastFP8ToFP32Vf: UB 内 FP8 → FP32 类型转换 (VF 寄存器)
 *   srcPtr:  [count] FP8_e4m3fn
 *   dstPtr:  [count] FP32
 */
__simd_vf__ inline void CastFP8ToFP32Vf(
    __ubuf__ int8_t* srcPtr,
    __ubuf__ float* dstPtr,
    uint32_t count)
{
    using namespace AscendC::MicroAPI;

    constexpr uint16_t VL_F = VECTOR_REG_WIDTH / sizeof(float);
    uint16_t vlIter = (count + VL_F - 1) / VL_F;

    // RegTensor 通过 fp8_e4m3fn_t 类型声明告知编译器 bit-pattern 语义；
    // 实际 load/store 经由 int8_t 指针中转，规避 fp8 标量语义后端限制。
    RegTensor<fp8_e4m3fn_t> srcReg;
    RegTensor<float> dstReg;
    MaskReg maskAll = CreateMask<float, MaskPattern::ALL>();

    __ubuf__ fp8_e4m3fn_t* srcRegSrc = (__ubuf__ fp8_e4m3fn_t*)srcPtr;

    for (uint16_t vi = 0; vi < vlIter; vi++) {
        DataCopy<fp8_e4m3fn_t, LoadDist::DIST_UNPACK_B8>(
            srcReg, srcRegSrc + vi * VL_F);
        Cast<float, fp8_e4m3fn_t, CAST_FP8_TO_FP32>(dstReg, srcReg, maskAll);
        DataCopy(dstPtr + vi * VL_F, dstReg, maskAll);
    }
}

/**
 * CastFP32ToBF16Vf: UB 内 FP32 → BF16 类型转换 (VF 寄存器)
 *   srcPtr:  [count] FP32
 *   dstPtr:  [count] BF16
 */
__simd_vf__ inline void CastFP32ToBF16Vf(
    __ubuf__ float* srcPtr,
    __ubuf__ bfloat16_t* dstPtr,
    uint32_t count)
{
    using namespace AscendC::MicroAPI;

    constexpr uint16_t VL_F = VECTOR_REG_WIDTH / sizeof(float);
    uint16_t vlIter = (count + VL_F - 1) / VL_F;

    RegTensor<float> srcReg;
    RegTensor<bfloat16_t> dstReg;
    MaskReg maskAll = CreateMask<float, MaskPattern::ALL>();

    for (uint16_t vi = 0; vi < vlIter; vi++) {
        DataCopy(srcReg, srcPtr + vi * VL_F);
        Cast<bfloat16_t, float, CAST_FP32_TO_BF16>(dstReg, srcReg, maskAll);
        DataCopy<bfloat16_t, StoreDist::DIST_PACK_B32>(
            dstPtr + vi * VL_F, dstReg, maskAll);
    }
}

/**
 * ComputeWeightedGroupSumVf: 在 VF 寄存器上完成 Weighted Group Sum
 *
 * 输入 (UB):
 *   qFP32Ptr:  [B, D] FP32 = [R*S, D] FP32  (qBlockFP32Buf)
 *   scalePtr:  [B]    FP32 = [R*S]    FP32  (scalesBuf)
 *
 * 输出 (UB):
 *   groupSumPtr: [S*D] FP32 = [S, D] FP32   (groupSumBuf)
 *
 * 对每个 g ∈ [0, S-1]:
 *   groupSum[g, :] = Σ_{r=0}^{R-1} qFP32[r*S+g, :] × scale[r*S+g]
 *                    ↑ 沿 R 轴累加
 *
 * UB 中数据布局:
 *   qFP32: row 0, row 1, ..., row B-1  (每行 D 个 FP32, 连续存储)
 *   同一个 g 的 R 行: row g, row S+g, row 2S+g, ..., row (R-1)S+g
 *   行间隔: S*D = 16*128 = 2048 个 float
 */
__simd_vf__ inline void ComputeWeightedGroupSumVf(
    __ubuf__ float* qFP32Ptr,
    __ubuf__ float* scalePtr,
    __ubuf__ float* groupSumPtr,
    uint16_t S, uint16_t R, uint16_t dimQk)
{
    using namespace AscendC::MicroAPI;

    constexpr uint16_t VL_F = VECTOR_REG_WIDTH / sizeof(float);
    uint16_t vlIterF = (dimQk + VL_F - 1) / VL_F;

    RegTensor<float> qReg;
    RegTensor<float> scaleBrcReg;
    RegTensor<float> accReg;
    MaskReg maskAll = CreateMask<float, MaskPattern::ALL>();

    for (uint16_t g = 0; g < S; g++) {
        for (uint16_t vi = 0; vi < vlIterF; vi++) {
            Duplicate(accReg, 0.0f, maskAll);

            for (uint16_t r = 0; r < R; r++) {
                uint16_t row = r * S + g;

                DataCopy<float, LoadDist::DIST_BRC_B32>(
                    scaleBrcReg, scalePtr + row);
                uint32_t qOffset = static_cast<uint32_t>(row) * dimQk + vi * VL_F;
                DataCopy(qReg, qFP32Ptr + qOffset);
                Mul(qReg, qReg, scaleBrcReg, maskAll);
                Add(accReg, accReg, qReg, maskAll);
            }

            uint32_t outOffset = static_cast<uint32_t>(g) * dimQk + vi * VL_F;
            DataCopy(groupSumPtr + outOffset, accReg, maskAll);
        }
    }
}

class KernelStemPrepQ {
public:
    __aicore__ inline KernelStemPrepQ() {}

    __aicore__ inline void Init(
        GM_ADDR q, GM_ADDR qscale,
        GM_ADDR q_seq_lens, GM_ADDR cu_seqlens_q,
        GM_ADDR qflat,
        const StemPrepQTilingData* tiling)
    {
        tiling_ = tiling;
        task_id_ = GetBlockIdx();
        if (task_id_ >= tiling->used_core_num) return;

        num_q_heads_ = tiling->num_q_heads;
        dim_qk_     = tiling->dim_qk;
        B_ = tiling->stem_block_size;
        S_ = tiling->stem_stride;
        R_ = tiling->R_val;
        qflat_dim_ = tiling->kflat_dim;
        ubFactor_ = tiling->ub_factor;

        qSeqLensBase_ = q_seq_lens;
        cuSeqLensBase_ = cu_seqlens_q;

        // qb 粒度切核: 总任务 = H_q × Σ_b ceil(qSeqLens[b]/B)
        // 每核分配 [qb_start_, qb_end_) 范围的连续 qb
        uint32_t tasks_per_core_base = tiling->blocks_per_core_base;
        uint32_t tasks_remainder = tiling->blocks_remainder;
        
        if (task_id_ < tasks_remainder) {
            qb_start_ = task_id_ * (tasks_per_core_base + 1);
            qb_end_ = qb_start_ + tasks_per_core_base + 1;
        } else {
            qb_start_ = tasks_remainder * (tasks_per_core_base + 1) + 
                        (task_id_ - tasks_remainder) * tasks_per_core_base;
            qb_end_ = qb_start_ + tasks_per_core_base;
        }

        qBase_ = q;
        qscaleBase_ = qscale;
        qflatBase_ = qflat;

        // UB Buffer 分配:
        // qBlockBuf_:     [B, D] FP8  = 128×128×1 = 16 KB
        // qBlockFP32Buf_: [B, D] FP32 = 128×128×4 = 64 KB
        // scalesBuf_:     [B,Hq] FP32 = 128×Hq×4
        // groupSumBuf_:   [S*D]  FP32 = 2048×4    = 8 KB
        // outQueueQflat_: [S*D]  BF16 = 2048×2    = 4 KB
        // varlenInfoBuf_: [MAX_BATCH_SIZE] INT32 = 4 KB
        pipe_.InitBuffer(qBlockBuf_, ubFactor_ * B_ * dim_qk_ * sizeof(int8_t));
        pipe_.InitBuffer(qBlockFP32Buf_, ubFactor_ * B_ * dim_qk_ * sizeof(float));
        pipe_.InitBuffer(scalesBuf_, ubFactor_ * B_ * num_q_heads_ * sizeof(float));
        pipe_.InitBuffer(groupSumBuf_, ubFactor_ * qflat_dim_ * sizeof(float));
        pipe_.InitBuffer(outQueueQflat_, ubFactor_ * qflat_dim_ * sizeof(bfloat16_t));
        pipe_.InitBuffer(varlenInfoBuf_, MAX_BATCH_SIZE * sizeof(int32_t) * 2);

        LocalTensor<int32_t> varlenLocal = varlenInfoBuf_.Get<int32_t>();
        uint32_t batchBytes = tiling->batch_size * sizeof(int32_t);
        uint32_t batchBytesAligned = (batchBytes + 31) & ~31u;
        if (batchBytesAligned == 0) batchBytesAligned = 32;

        qSeqLensGm_.SetGlobalBuffer((__gm__ int32_t*)qSeqLensBase_, tiling->batch_size);
        for (uint32_t i = 0; i < tiling->batch_size; i++) {
            varlenLocal.SetValue(i, qSeqLensGm_.GetValue(i));
        }

        cuSeqLensGm_.SetGlobalBuffer((__gm__ int32_t*)cuSeqLensBase_, tiling->batch_size + 1);
        for (uint32_t i = 0; i <= tiling->batch_size; i++) {
            varlenLocal.SetValue(MAX_BATCH_SIZE + i, cuSeqLensGm_.GetValue(i));
        }
    }

    __aicore__ inline void Process()
    {
        if (task_id_ >= tiling_->used_core_num) return;

        for (uint32_t pos = qb_start_; pos < qb_end_; pos += ubFactor_) {
            curBatch_ = (qb_end_ - pos < ubFactor_) ? qb_end_ - pos : ubFactor_;
            PrecomputeTaskMeta(pos);

            CopyInQ();
            CastAll();
            CopyInScales();
            AccumulateAndFlatten();
            CopyOutAll();
        }
    }

private:
    __aicore__ inline void PrecomputeTaskMeta(uint32_t pos)
    {
        LocalTensor<int32_t> varlenLocal = varlenInfoBuf_.Get<int32_t>();
        uint32_t blocks_per_batch = num_q_heads_ * tiling_->max_Qb;

        uint32_t sumValid = 0;
        for (uint32_t i = 0; i < curBatch_; i++) {
            uint32_t task_idx = pos + i;
            uint32_t bIdx = task_idx / blocks_per_batch;
            uint32_t within_batch = task_idx % blocks_per_batch;
            uint32_t hIdx = within_batch / tiling_->max_Qb;
            uint32_t qbLcl = within_batch % tiling_->max_Qb;

            uint32_t cuOff = static_cast<uint32_t>(varlenLocal.GetValue(MAX_BATCH_SIZE + bIdx));
            uint32_t qLen = static_cast<uint32_t>(varlenLocal.GetValue(bIdx));
            uint32_t startRow = qbLcl * B_;
            uint32_t validRows = (startRow + B_ <= qLen) ? B_ :
                                 (qLen > startRow ? qLen - startRow : 0);

            if (i == 0) {
                chunkGmPos_ = cuOff + startRow;
                chunkHead_ = hIdx;
                firstBatch_ = bIdx;
                firstHead_ = hIdx;
                firstQb_ = qbLcl;
            }

            taskMetaHead_[i] = hIdx;
            taskMetaCuOff_[i] = cuOff;
            taskMetaStartRow_[i] = startRow;
            taskMetaValidRows_[i] = validRows;

            sumValid += validRows;
        }
        chunkTotalRows_ = sumValid;
    }

    __aicore__ inline void CopyInQ()
    {
        LocalTensor<int8_t> qBlockI8 = qBlockBuf_.Get<int8_t>();
        uint32_t totalElems = curBatch_ * B_ * dim_qk_;
        Duplicate(qBlockI8, (int8_t)0, totalElems);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);

        uint32_t qOffset = chunkGmPos_ * num_q_heads_ * dim_qk_ + chunkHead_ * dim_qk_;
        qGm_.SetGlobalBuffer((__gm__ int8_t*)qBase_ + qOffset * sizeof(int8_t),
            chunkTotalRows_ * num_q_heads_ * dim_qk_);
        DataCopyPadExtParams<int8_t> padParams{false, 0, 0, 0};
        DataCopyExtParams params{
            static_cast<uint16_t>(chunkTotalRows_),
            static_cast<uint32_t>(dim_qk_ * sizeof(int8_t)),
            static_cast<uint32_t>((num_q_heads_ - 1) * dim_qk_ * sizeof(int8_t)),
            0, 0};
        DataCopyPad(qBlockI8, qGm_, params, padParams);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    }

    __aicore__ inline void CastAll()
    {
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

        // FP8 → FP32: Vector Cast API does NOT support FP8 scalar semantics on arch35.
        // 必须经由 RegBase DataCopy through RegTensor + MicroAPI::Cast, implemented in CastFP8ToFP32Vf.
        // 使用 int8_t* 作为底层 UB buffer type 规避 fp8 标量语义后端限制.
        const uint32_t elemsPerBlock = B_ * dim_qk_;
        __ubuf__ int8_t* srcBase = (__ubuf__ int8_t*)qBlockBuf_.Get<int8_t>().GetPhyAddr();
        __ubuf__ float* dstBase =
            (__ubuf__ float*)qBlockFP32Buf_.Get<float>().GetPhyAddr();
        for (uint32_t i = 0; i < curBatch_; i++) {
            CastFP8ToFP32Vf(
                srcBase + i * elemsPerBlock,
                dstBase + i * elemsPerBlock,
                elemsPerBlock);
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void CopyInScales()
    {
        LocalTensor<float> scalesLocal = scalesBuf_.Get<float>();
        uint32_t totalScaleElems = curBatch_ * B_;
        Duplicate(scalesLocal, 0.0f, totalScaleElems);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);

        DataCopyPadExtParams<float> padParams{false, 0, 0, 0.0f};
        DataCopyExtParams params{
            static_cast<uint16_t>(1),                      // count: per task 1 scale
            static_cast<uint32_t>(sizeof(float)),          // blockLen = 4 bytes
            static_cast<uint32_t>((num_q_heads_ - 1) * sizeof(float)),  // srcStride: skip other heads
            0, 0};

        for (uint32_t i = 0; i < curBatch_; i++) {
            uint32_t valid = taskMetaValidRows_[i];
            if (valid == 0) continue;

            uint32_t gmPos = taskMetaCuOff_[i] + taskMetaStartRow_[i];
            uint32_t scaleBase = gmPos * num_q_heads_ + taskMetaHead_[i];
            qscaleGm_.SetGlobalBuffer((__gm__ float*)qscaleBase_ + scaleBase,
                                      valid * num_q_heads_);

            params.blockCount = static_cast<uint16_t>(valid);
            LocalTensor<float> dstElem = scalesLocal[i * B_];
            DataCopyPad(dstElem, qscaleGm_, params, padParams);
        }

        SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
    }

    __aicore__ inline void AccumulateAndFlatten()
    {
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);

        for (uint32_t i = 0; i < curBatch_; i++) {
            __ubuf__ float* qFP32Ptr =
                (__ubuf__ float*)qBlockFP32Buf_.Get<float>().GetPhyAddr() + i * B_ * dim_qk_;
            __ubuf__ float* scalePtr =
                (__ubuf__ float*)scalesBuf_.Get<float>().GetPhyAddr() + i * B_;
            __ubuf__ float* groupSumPtr =
                (__ubuf__ float*)groupSumBuf_.Get<float>().GetPhyAddr() + i * qflat_dim_;

            ComputeWeightedGroupSumVf(qFP32Ptr, scalePtr, groupSumPtr,
                static_cast<uint16_t>(S_),
                static_cast<uint16_t>(R_),
                static_cast<uint16_t>(dim_qk_));
        }
        PipeBarrier<PIPE_V>();

        for (uint32_t i = 0; i < curBatch_; i++) {
            LocalTensor<float> groupSumLocal = groupSumBuf_.Get<float>();
            LocalTensor<bfloat16_t> qflatLocal = outQueueQflat_.Get<bfloat16_t>();
            uint32_t gsOffset = i * qflat_dim_;
            Cast<bfloat16_t, float>(qflatLocal[gsOffset], groupSumLocal[gsOffset],
                RoundMode::CAST_RINT, qflat_dim_);
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void CopyOutAll()
    {
        LocalTensor<bfloat16_t> qflatLocal = outQueueQflat_.Get<bfloat16_t>();
        uint32_t gmOutOffset = firstBatch_ * (num_q_heads_ * tiling_->max_Qb * qflat_dim_)
                             + firstHead_ * (tiling_->max_Qb * qflat_dim_)
                             + firstQb_ * qflat_dim_;
        uint32_t blockLen = qflat_dim_ * sizeof(bfloat16_t);
        qflatGm_.SetGlobalBuffer((__gm__ bfloat16_t*)qflatBase_ + gmOutOffset, 
                                  curBatch_ * qflat_dim_);
        DataCopyPad(qflatGm_, qflatLocal,
            {static_cast<uint16_t>(curBatch_), static_cast<uint16_t>(blockLen), 0, 0});
    }

    TPipe pipe_;
    const StemPrepQTilingData* tiling_ = nullptr;

    uint32_t task_id_ = 0;
    uint32_t num_q_heads_ = 0;
    uint32_t dim_qk_ = 0;
    uint32_t B_ = 0;
    uint32_t S_ = 0;
    uint32_t R_ = 0;
    uint32_t qflat_dim_ = 0;
    uint32_t ubFactor_ = 1;

    uint32_t qb_start_ = 0;
    uint32_t qb_end_ = 0;

    uint32_t curBatch_ = 0;
    uint32_t chunkGmPos_ = 0;
    uint32_t chunkHead_ = 0;
    uint32_t chunkTotalRows_ = 0;
    uint32_t firstBatch_ = 0;
    uint32_t firstHead_ = 0;
    uint32_t firstQb_ = 0;

    uint32_t taskMetaHead_[4] = {0};
    uint32_t taskMetaCuOff_[4] = {0};
    uint32_t taskMetaStartRow_[4] = {0};
    uint32_t taskMetaValidRows_[4] = {0};

    GM_ADDR qSeqLensBase_ = nullptr;
    GM_ADDR cuSeqLensBase_ = nullptr;
    GM_ADDR qBase_ = nullptr;
    GM_ADDR qscaleBase_ = nullptr;
    GM_ADDR qflatBase_ = nullptr;

    GlobalTensor<int8_t> qGm_;
    GlobalTensor<float> qscaleGm_;
    GlobalTensor<int32_t> qSeqLensGm_;
    GlobalTensor<int32_t> cuSeqLensGm_;
    GlobalTensor<bfloat16_t> qflatGm_;

    TBuf<TPosition::VECCALC> qBlockBuf_;
    TBuf<TPosition::VECCALC> qBlockFP32Buf_;
    TBuf<TPosition::VECCALC> scalesBuf_;
    TBuf<TPosition::VECCALC> groupSumBuf_;
    TBuf<TPosition::VECCALC> outQueueQflat_;
    TBuf<TPosition::VECCALC> varlenInfoBuf_;
};

#endif
