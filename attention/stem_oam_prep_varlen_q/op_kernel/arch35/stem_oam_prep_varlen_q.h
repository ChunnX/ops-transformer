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
    __ubuf__ fp8_e4m3fn_t* srcPtr,
    __ubuf__ float* dstPtr,
    uint32_t count)
{
    using namespace AscendC::MicroAPI;

    constexpr uint16_t VL_F = VECTOR_REG_WIDTH / sizeof(float);
    uint16_t vlIter = (count + VL_F - 1) / VL_F;

    RegTensor<fp8_e4m3fn_t> srcReg;
    RegTensor<float> dstReg;
    MaskReg maskAll = CreateMask<float, MaskPattern::ALL>();

    for (uint16_t vi = 0; vi < vlIter; vi++) {
        DataCopy<fp8_e4m3fn_t, LoadDist::DIST_UNPACK_B8>(
            srcReg, srcPtr + vi * VL_F);
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
                uint16_t row_idx = r * S + g;
                DataCopy<float, LoadDist::DIST_BRC_B32>(
                    scaleBrcReg, scalePtr + row_idx);
                uint32_t qOffset = static_cast<uint32_t>(row_idx) * dimQk + vi * VL_F;
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
        // scalesBuf_:     [B]    FP32 = 128×4          = 0.5 KB  (只存当前 head 的 scale)
        // groupSumBuf_:   [S*D]  FP32 = 2048×4    = 8 KB
        // outQueueQflat_: [S*D]  BF16 = 2048×2    = 4 KB
        // varlenInfoBuf_: [MAX_BATCH_SIZE] INT32 = 4 KB
        pipe_.InitBuffer(qBlockBuf_, B_ * dim_qk_ * sizeof(fp8_e4m3fn_t));
        pipe_.InitBuffer(qBlockFP32Buf_, B_ * dim_qk_ * sizeof(float));
        pipe_.InitBuffer(scalesBuf_, B_ * sizeof(float));
        pipe_.InitBuffer(groupSumBuf_, qflat_dim_ * sizeof(float));
        pipe_.InitBuffer(outQueueQflat_, 1, qflat_dim_ * sizeof(bfloat16_t));
        pipe_.InitBuffer(varlenInfoBuf_, MAX_BATCH_SIZE * sizeof(int32_t));

        // 一次性把 qSeqLens[0..batch-1] 搬到 UB
        LocalTensor<int32_t> varlenLocal = varlenInfoBuf_.Get<int32_t>();
        uint32_t batchBytes = tiling->batch_size * sizeof(int32_t);
        qSeqLensGm_.SetGlobalBuffer((__gm__ int32_t*)qSeqLensBase_, tiling->batch_size);
        DataCopyPad(varlenLocal, qSeqLensGm_,
            {1, static_cast<uint16_t>(batchBytes), 0, 0},
            {false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    }

    __aicore__ inline void Process()
    {
        if (task_id_ >= tiling_->used_core_num) return;

        for (uint32_t task_idx = qb_start_; task_idx < qb_end_; task_idx++) {
            DecodeTaskId(task_idx);
            ProcessOneBlock(task_idx);
        }
    }

    /**
     * DecodeTaskId: 从全局 task_idx 反解 (batch, head, qb_local)
     *
     * task_idx 的排列顺序:
     *   (b=0,h=0,qb=0), (b=0,h=0,qb=1), ..., (b=0,h=0,qb=N0-1),
     *   (b=0,h=1,qb=0), ..., (b=0,h=1,qb=N0-1),
     *   ...
     *   (b=1,h=0,qb=0), ..., (b=1,h=H_q-1,qb=N1-1),
     *   ...
     * 其中 Nb = ceil(qSeqLens[b] / B)
     */
    __aicore__ inline void DecodeTaskId(uint32_t task_idx)
    {
        uint32_t total_qb = 0;
        uint32_t acc_cu_off = 0;
        LocalTensor<int32_t> varlenLocal = varlenInfoBuf_.Get<int32_t>();
        for (uint32_t b = 0; b < tiling_->batch_size; b++) {
            uint32_t q_len_b = static_cast<uint32_t>(varlenLocal.GetValue(b));

            uint32_t q_padded_b = ((q_len_b + B_ - 1) / B_) * B_;
            uint32_t num_Qb_b = q_padded_b / B_;

            for (uint32_t h = 0; h < num_q_heads_; h++) {
                if (task_idx < total_qb + num_Qb_b) {
                    batch_idx_ = b;
                    head_idx_ = h;
                    qb_local_ = task_idx - total_qb;
                    cu_off_ = acc_cu_off;
                    q_len_ = q_len_b;
                    q_padded_ = q_padded_b;
                    num_Qb_ = num_Qb_b;
                    AscendC::PRINTF("[D]task=%u b=%u h=%u qb=%u cu=%u qlen=%u\n",
                        task_idx, b, h, qb_local_, cu_off_, q_len_);
                    return;
                }
                total_qb += num_Qb_b;
            }
            acc_cu_off += q_len_b;
        }
    }

private:
    /**
     * ReadVarlenInfo: 读取 varlen 元信息
     *
     * 从 GM 读取:
     *   qSeqLens[b]  → q_len_:  该 batch 的 Q 序列长度
     *   cuSeqLensQ[b] → cu_off_: 该 batch 在 q 中的起始 token 偏移
     *
     * 计算:
     *   q_padded_ = ceil(q_len / B) × B
     *   num_Qb_   = q_padded / B
     */
    __aicore__ inline void ReadVarlenInfo(uint32_t b)
    {
        LocalTensor<int32_t> varlenLocal = varlenInfoBuf_.Get<int32_t>();
        qSeqLensGm_.SetGlobalBuffer((__gm__ int32_t*)qSeqLensBase_ + b, VARLEN_READ_ALIGN);
        DataCopyPad(varlenLocal, qSeqLensGm_,
            {1, static_cast<uint16_t>(VARLEN_READ_ALIGN * sizeof(int32_t)), 0, 0},
            {false, 0, 0, 0});
        PipeBarrier<PIPE_MTE2>();
        q_len_ = static_cast<uint32_t>(varlenLocal.GetValue(0));

        cuSeqLensGm_.SetGlobalBuffer((__gm__ int32_t*)cuSeqLensBase_ + b, VARLEN_READ_ALIGN);
        DataCopyPad(varlenLocal, cuSeqLensGm_,
            {1, static_cast<uint16_t>(VARLEN_READ_ALIGN * sizeof(int32_t)), 0, 0},
            {false, 0, 0, 0});
        PipeBarrier<PIPE_MTE2>();
        cu_off_ = static_cast<uint32_t>(varlenLocal.GetValue(0));

        q_padded_ = ((q_len_ + B_ - 1) / B_) * B_;
        num_Qb_   = q_padded_ / B_;
    }

    __aicore__ inline void ProcessOneBlock(uint32_t task_idx)
    {
        CopyInQBlock();
        CastBlockToFP32();
        CopyInScalesBulk();
        AccumulateBlock();
        FlattenAndCast();
        CopyOutBlock();
    }

    /**
     * CopyInQBlock: GM → UB, 搬入 1 个 qb block 的 Q 数据
     *
     * GM 源: q[cu_off + qb*B + i, h, :]  for i = 0..valid_rows-1
     *        q shape: [total_tokens, H_q, D]
     *        GM offset per row = globalPos * H_q * D + h * D
     *
     * UB 目标: qBlockBuf [B, D] FP8
     *          超出 q_len 的行保持零 (预填零)
     */
    __aicore__ inline void CopyInQBlock()
    {
        LocalTensor<int8_t> qBlockI8 = qBlockBuf_.Get<int8_t>();
        Duplicate(qBlockI8, (int8_t)0, B_ * dim_qk_);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);

        uint32_t start_row = qb_local_ * B_;
        uint32_t valid_rows = (start_row + B_ <= q_len_) ? B_ :
                              (q_len_ > start_row ? q_len_ - start_row : 0);

        LocalTensor<fp8_e4m3fn_t> qBlockLocal = qBlockBuf_.Get<fp8_e4m3fn_t>();
        if (valid_rows > 0) {
            uint32_t qOffset = (cu_off_ + start_row) * num_q_heads_ * dim_qk_
                              + head_idx_ * dim_qk_;
            DataCopyParams copyParams;
            copyParams.blockCount = static_cast<uint16_t>(valid_rows);
            copyParams.blockLen = static_cast<uint16_t>(dim_qk_ * sizeof(fp8_e4m3fn_t));
            copyParams.srcStride = static_cast<uint16_t>((num_q_heads_ - 1) * dim_qk_ * sizeof(fp8_e4m3fn_t));
            copyParams.dstStride = 0;
            qGm_.SetGlobalBuffer((__gm__ fp8_e4m3fn_t*)qBase_ + qOffset,
                                 valid_rows * num_q_heads_ * dim_qk_);
            DataCopyPad(qBlockLocal, qGm_, copyParams,
                        DataCopyPadParams{false, 0, 0, 0});
        }
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    }


    /**
     * CastBlockToFP32: UB 内 FP8 → FP32 类型转换
     *
     * qBlockBuf [B, D] FP8 → qBlockFP32Buf [B, D] FP32
     * 先预填零, 再 Cast valid_rows*D 个元素
     */
    __aicore__ inline void CastBlockToFP32()
    {
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

        LocalTensor<fp8_e4m3fn_t> qBlockLocal = qBlockBuf_.Get<fp8_e4m3fn_t>();
        LocalTensor<float> qBlockFP32Local = qBlockFP32Buf_.Get<float>();
        Duplicate(qBlockFP32Local, 0.0f, B_ * dim_qk_);
        PipeBarrier<PIPE_V>();

        uint32_t start_row = qb_local_ * B_;
        uint32_t valid_rows = (start_row + B_ <= q_len_) ? B_ :
                              (q_len_ > start_row ? q_len_ - start_row : 0);
        uint32_t cast_count = valid_rows * dim_qk_;
        if (cast_count > 0) {
            Cast<float, fp8_e4m3fn_t>(qBlockFP32Local, qBlockLocal,
                RoundMode::CAST_NONE, cast_count);
        }
        PipeBarrier<PIPE_V>();
    }

    /**
     * CopyInScalesBulk: GM → UB, 搬入 1 个 qb block 的 scale 数据
     *
     * GM 源: qScale[cu_off + qb*B : cu_off + qb*B + B, h]
     *        qscale shape: [total_tokens, H_q]
     *        GM offset = (cu_off + qb*B) * H_q + h
     *
     * UB 目标: scalesBuf [B] FP32
     */
    __aicore__ inline void CopyInScalesBulk()
    {
        LocalTensor<float> scalesLocal = scalesBuf_.Get<float>();
        Duplicate(scalesLocal, 0.0f, B_);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        event_t vsEvent = static_cast<event_t>(pipe_.FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(vsEvent);
        WaitFlag<HardEvent::V_S>(vsEvent);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);

        uint32_t start_row = qb_local_ * B_;
        uint32_t valid_rows = (start_row + B_ <= q_len_) ? B_ :
                              (q_len_ > start_row ? q_len_ - start_row : 0);
        if (valid_rows > 0) {
            uint32_t base_offset = (cu_off_ + start_row) * num_q_heads_ + head_idx_;
            uint32_t gm_range = valid_rows * num_q_heads_;
            qscaleGm_.SetGlobalBuffer(
                (__gm__ float*)qscaleBase_ + base_offset,
                gm_range);
            for (uint32_t i = 0; i < valid_rows; i++) {
                scalesLocal.SetValue(i,
                    qscaleGm_.GetValue(i * num_q_heads_));
            }
            event_t svEvent = static_cast<event_t>(pipe_.FetchEventID(HardEvent::S_V));
            SetFlag<HardEvent::S_V>(svEvent);
            WaitFlag<HardEvent::S_V>(svEvent);
            if (valid_rows >= 4) {
                AscendC::PRINTF("[SVAL]qb=%u b=%u h=%u s[0]=%f s[1]=%f s[vr-1]=%f s[vr-2]=%f\n",
                    qb_local_, batch_idx_, head_idx_,
                    scalesLocal.GetValue(0), scalesLocal.GetValue(1),
                    scalesLocal.GetValue(valid_rows - 1), scalesLocal.GetValue(valid_rows - 2));
            }
        }
        SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
    }

    /**
     * AccumulateBlock: Weighted Group Sum (沿 R 轴累加)
     *
     * qBlockFP32Buf [B, D] 逻辑 reshape 为 [R, S, D]
     * scalesBuf     [B]    逻辑 reshape 为 [R, S]
     *
     * 对每个 g ∈ [0, S-1]:
     *   groupSum[g, :] = Σ_{r=0}^{R-1} qBlockFP32[r*S+g, :] × scales[r*S+g]
     *
     * groupSumBuf: [S, D] FP32 = [S*D] FP32
     */
    __aicore__ inline void AccumulateBlock()
    {
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);

        __ubuf__ float* qFP32Ptr =
            (__ubuf__ float*)qBlockFP32Buf_.Get<float>().GetPhyAddr();
        __ubuf__ float* scalePtr =
            (__ubuf__ float*)scalesBuf_.Get<float>().GetPhyAddr();
        __ubuf__ float* groupSumPtr =
            (__ubuf__ float*)groupSumBuf_.Get<float>().GetPhyAddr();

        VF_CALL<ComputeWeightedGroupSumVf>(qFP32Ptr, scalePtr, groupSumPtr,
            static_cast<uint16_t>(S_),
            static_cast<uint16_t>(R_),
            static_cast<uint16_t>(dim_qk_));
        PipeBarrier<PIPE_V>();
        LocalTensor<float> groupSumLocal = groupSumBuf_.Get<float>();
        uint32_t S_D = S_ * dim_qk_;
        AscendC::PRINTF("[GSUM]qb=%u b=%u h=%u g[0]=%f g[D]=%f g[S*D-D]=%f g[S*D-1]=%f\n",
            qb_local_, batch_idx_, head_idx_,
            groupSumLocal.GetValue(0),
            groupSumLocal.GetValue(dim_qk_),
            groupSumLocal.GetValue(S_D - dim_qk_),
            groupSumLocal.GetValue(S_D - 1));
    }

    /**
     * FlattenAndCast: UB 内 FP32 → BF16 类型转换
     *
     * groupSumBuf [S*D] FP32 → Cast(RINT) → outQueueQflat [S*D] BF16
     */
    __aicore__ inline void FlattenAndCast()
    {
        LocalTensor<float> groupSumLocal = groupSumBuf_.Get<float>();
        LocalTensor<bfloat16_t> qflatLocal = outQueueQflat_.AllocTensor<bfloat16_t>();
        Cast<bfloat16_t, float>(qflatLocal, groupSumLocal,
            RoundMode::CAST_RINT, qflat_dim_);
        outQueueQflat_.EnQue<bfloat16_t>(qflatLocal);
    }

    /**
     * CopyOutBlock: UB → GM, 写出 1 个 qb block 的 qFlat 结果
     *
     * UB 源: outQueueQflat [S*D] BF16
     *
     * GM 目标: qFlat[b, h, qb, :]
     *          qFlat shape: [batch, H_q, max_Qb, S*D]
     *          GM offset = b * (H_q * max_Qb * S*D) + h * (max_Qb * S*D) + qb * S*D
     */
    __aicore__ inline void CopyOutBlock()
    {
        LocalTensor<bfloat16_t> qflatLocal = outQueueQflat_.DeQue<bfloat16_t>();
        uint32_t outOffset = batch_idx_ * (num_q_heads_ * tiling_->max_Qb * qflat_dim_)
                            + head_idx_  * (tiling_->max_Qb * qflat_dim_)
                            + qb_local_  * qflat_dim_;
        qflatGm_.SetGlobalBuffer((__gm__ bfloat16_t*)qflatBase_ + outOffset, qflat_dim_);
        DataCopyPad(qflatGm_, qflatLocal,
            {1, static_cast<uint16_t>(qflat_dim_ * sizeof(bfloat16_t)), 0, 0});
        outQueueQflat_.FreeTensor(qflatLocal);
    }

    TPipe pipe_;
    const StemPrepQTilingData* tiling_ = nullptr;

    uint32_t task_id_ = 0;
    uint32_t batch_idx_ = 0;
    uint32_t head_idx_ = 0;
    uint32_t num_q_heads_ = 0;
    uint32_t dim_qk_ = 0;
    uint32_t q_len_ = 0;
    uint32_t cu_off_ = 0;
    uint32_t q_padded_ = 0;
    uint32_t num_Qb_ = 0;
    uint32_t B_ = 0;
    uint32_t S_ = 0;
    uint32_t R_ = 0;
    uint32_t qflat_dim_ = 0;

    uint32_t qb_start_ = 0;
    uint32_t qb_end_ = 0;
    uint32_t qb_local_ = 0;
    uint32_t core_in_bh_ = 0;
    uint32_t cores_per_bh_this_ = 1;

    GM_ADDR qSeqLensBase_ = nullptr;
    GM_ADDR cuSeqLensBase_ = nullptr;
    GM_ADDR qBase_ = nullptr;
    GM_ADDR qscaleBase_ = nullptr;
    GM_ADDR qflatBase_ = nullptr;

    GlobalTensor<fp8_e4m3fn_t> qGm_;
    GlobalTensor<float> qscaleGm_;
    GlobalTensor<int32_t> qSeqLensGm_;
    GlobalTensor<int32_t> cuSeqLensGm_;
    GlobalTensor<bfloat16_t> qflatGm_;

    TBuf<TPosition::VECCALC> qBlockBuf_;
    TBuf<TPosition::VECCALC> qBlockFP32Buf_;
    TBuf<TPosition::VECCALC> scalesBuf_;
    TBuf<TPosition::VECCALC> groupSumBuf_;
    TQue<TPosition::VECOUT, 1> outQueueQflat_;
    TBuf<TPosition::VECCALC> varlenInfoBuf_;
};

#endif
