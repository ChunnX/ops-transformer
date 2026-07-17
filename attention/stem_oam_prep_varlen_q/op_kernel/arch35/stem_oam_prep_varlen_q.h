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
 *   qSeqLens:    [batch]                       INT64
 *   cuSeqLensQ:  [batch+1]                     INT64
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
 *   4. AccumulateBlock (Weighted Group Sum, in-place binary reduce):
 *      qBlockFP32Buf 逻辑 reshape 为 [R, S, D]
 *      scalesBuf 逻辑 reshape 为 [R, S]
 *
 *      Phase 1: qBlockFP32Buf[(r*S+g)*D : (r*S+g+1)*D] *= scales[r*S+g]
 *                对每个 (r, g) 的原地乘 scale
 *      Phase 2: 沿 R 轴反向 reduce（stride 从 R_main 递减到 1）
 *                结果保留在 qBlockFP32Buf 的前 S 行 (偏移 0..S*D-1)
 *
 *   5. FlattenAndCast:
 *      UB: qBlockFP32Buf[0 : S*D] FP32 → Cast → outQueueQflat [S*D] BF16
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
constexpr uint32_t DOUBLE_BUFFER_DEPTH = 2;

__simd_callee__ inline void MulsOnAllRows(__ubuf__ float *qFP32Ptr, __ubuf__ float *scalePtr, uint16_t R, uint16_t S,
                                          uint16_t D)
{
    using namespace AscendC::MicroAPI;

    constexpr uint16_t VL_F = VECTOR_REG_WIDTH / sizeof(float);
    uint16_t vlIter = (D + VL_F - 1) / VL_F;

    MaskReg maskAll = CreateMask<float, MaskPattern::ALL>();
    RegTensor<float> regScale;
    RegTensor<float> regA;

    for (uint16_t r = 0; r < R; r++) {
        for (uint16_t g = 0; g < S; g++) {
            uint16_t rowIdx = static_cast<uint16_t>(r * S + g);
            uint32_t rowOff = static_cast<uint32_t>(rowIdx) * D;
            DataCopy<float, LoadDist::DIST_BRC_B32>(regScale, scalePtr + rowIdx);
            for (uint16_t vi = 0; vi < vlIter; vi++) {
                uint32_t off = rowOff + static_cast<uint32_t>(vi) * VL_F;
                DataCopy(regA, qFP32Ptr + off);
                Mul(regA, regA, regScale, maskAll);
                DataCopy(qFP32Ptr + off, regA, maskAll);
            }
        }
    }
}

__simd_vf__ inline void WeightedBinaryReduceFull(__ubuf__ float *qFP32Ptr, __ubuf__ float *scalePtr, uint16_t R,
                                                  uint16_t S, uint16_t D)
{
    MulsOnAllRows(qFP32Ptr, scalePtr, R, S, D);

    using namespace AscendC::MicroAPI;

    constexpr uint16_t VL_F = VECTOR_REG_WIDTH / sizeof(float);
    uint16_t vlIter = (D + VL_F - 1) / VL_F;

    MaskReg maskAll = CreateMask<float, MaskPattern::ALL>();
    RegTensor<float> regA;
    RegTensor<float> regB;

    uint16_t R_main = 1;
    while ((R_main << 1) <= R) { R_main <<= 1; }
    uint16_t R_tail = R - R_main;

    for (uint16_t stride = R_main; stride >= 1; stride >>= 1) {
        uint16_t ops = (stride == R_main) ? R_tail : stride;
        if (ops == 0) continue;
        for (uint16_t g = 0; g < S; g++) {
            for (uint16_t r = 0; r < ops; r++) {
                uint32_t offDst = static_cast<uint32_t>(r * S + g) * D;
                uint32_t offSrc = static_cast<uint32_t>((r + stride) * S + g) * D;
                for (uint16_t vi = 0; vi < vlIter; vi++) {
                    uint32_t d = static_cast<uint32_t>(vi) * VL_F;
                    DataCopy(regA, qFP32Ptr + offDst + d);
                    DataCopy(regB, qFP32Ptr + offSrc + d);
                    Add(regA, regA, regB, maskAll);
                    DataCopy(qFP32Ptr + offDst + d, regA, maskAll);
                }
            }
        }
    }
}

class KernelStemPrepQ {
public:
    __aicore__ inline KernelStemPrepQ()
    {
    }

    __aicore__ inline void Init(GM_ADDR q, GM_ADDR qscale, GM_ADDR cu_seqlens_q, GM_ADDR qflat,
                                const StemPrepQTilingData *tiling)
    {
        tiling_ = tiling;
        task_id_ = GetBlockIdx();
        if (task_id_ >= tiling->used_core_num)
            return;

        num_q_heads_ = tiling->num_q_heads;
        dim_qk_ = tiling->dim_qk;
        B_ = tiling->stem_block_size;
        S_ = tiling->stem_stride;
        R_ = tiling->R_val;
        qflat_dim_ = tiling->kflat_dim;
        pp_num_ = tiling->ub_factor > 1 ? 2 : 1;

        InitTaskRange(tiling->blocks_per_core_base, tiling->blocks_remainder);
        InitGlobalBuffers(q, qscale, cu_seqlens_q, qflat);
        InitUBBuffers();
        LoadCuSeqLensQ();
    }

    __aicore__ inline void Process()
    {
        if (task_id_ >= tiling_->used_core_num)
            return;

        for (uint32_t task_idx = qb_start_; task_idx < qb_end_; task_idx++) {
            DecodeTaskId(task_idx);
            ProcessOneBlock();
        }
    }

    __aicore__ inline void DecodeTaskId(uint32_t task_idx)
    {
        // 整数除法定位 batch（替代线性扫描，O(1) 复杂度）
        uint32_t tasks_per_batch = tiling_->max_Qb * num_q_heads_;
        batch_idx_ = task_idx / tasks_per_batch;
        uint32_t in_batch = task_idx % tasks_per_batch;
        head_idx_ = in_batch / tiling_->max_Qb;
        qb_local_ = in_batch % tiling_->max_Qb;

        // 从 cuSeqLensQ 读取当前 batch 的信息
        LocalTensor<int64_t> cuSeqLensQ = cuSeqLensQBuf_.Get<int64_t>();
        int64_t cu_start = cuSeqLensQ.GetValue(batch_idx_);
        int64_t cu_end = cuSeqLensQ.GetValue(batch_idx_ + 1);

        cu_off_ = static_cast<uint32_t>(cu_start);
        q_len_ = static_cast<uint32_t>(cu_end - cu_start);
        num_Qb_ = (q_len_ + B_ - 1) / B_;
    }

    __aicore__ inline void ProcessOneBlock()
    {
        if (qb_local_ >= num_Qb_) {
            WriteZeroBlock();
            return;
        }
        CopyInQBlock();
        CastBlockToFP32();
        CopyInScalesBulk();
        AccumulateBlock();
        FlattenAndCast();
        CopyOutBlock();
    }

private:
    __aicore__ inline void InitTaskRange(uint32_t tasks_per_core_base, uint32_t tasks_remainder)
    {
        if (task_id_ < tasks_remainder) {
            qb_start_ = task_id_ * (tasks_per_core_base + 1);
            qb_end_ = qb_start_ + tasks_per_core_base + 1;
        } else {
            qb_start_ =
                tasks_remainder * (tasks_per_core_base + 1) + (task_id_ - tasks_remainder) * tasks_per_core_base;
            qb_end_ = qb_start_ + tasks_per_core_base;
        }
    }

    __aicore__ inline void InitGlobalBuffers(GM_ADDR q, GM_ADDR qscale, GM_ADDR cu_seqlens_q, GM_ADDR qflat)
    {
        qBase_ = q;
        qscaleBase_ = qscale;
        cuSeqLensQBase_ = cu_seqlens_q;
        qflatBase_ = qflat;
    }

    __aicore__ inline void InitUBBuffers()
    {
        pipe_.InitBuffer(qBlockBuf_, pp_num_, B_ * dim_qk_ * sizeof(fp8_e4m3fn_t));
        pipe_.InitBuffer(qBlockFP32Buf_, pp_num_, B_ * dim_qk_ * sizeof(float));
        pipe_.InitBuffer(scalesBuf_, pp_num_, B_ * num_q_heads_ * sizeof(float));
        pipe_.InitBuffer(outQueueQflat_, pp_num_, qflat_dim_ * sizeof(bfloat16_t));
    }

    __aicore__ inline void LoadCuSeqLensQ()
    {
        uint32_t cuSeqLensQCount = tiling_->batch_size + 1;
        pipe_.InitBuffer(cuSeqLensQBuf_, cuSeqLensQCount * sizeof(int64_t));
        LocalTensor<int64_t> cuSeqLensQLocal = cuSeqLensQBuf_.Get<int64_t>();
        cuSeqLensGm_.SetGlobalBuffer((__gm__ int64_t *)cuSeqLensQBase_, cuSeqLensQCount);
        DataCopyPad(cuSeqLensQLocal, cuSeqLensGm_,
                    {1, static_cast<uint16_t>(cuSeqLensQCount * sizeof(int64_t)), 0, 0}, {0, 0, 0, 0});
        PipeBarrier<PIPE_ALL>();
    }

public:

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
        LocalTensor<fp8_e4m3fn_t> qBlockLocal = qBlockBuf_.AllocTensor<fp8_e4m3fn_t>();
        Duplicate((LocalTensor<int8_t> &)qBlockLocal, (int8_t)0, B_ * dim_qk_);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);

        uint32_t start_row = qb_local_ * B_;
        uint32_t valid_rows = (start_row + B_ <= q_len_) ? B_ : (q_len_ > start_row ? q_len_ - start_row : 0);

        if (valid_rows > 0) {
            uint32_t qOffset = (cu_off_ + start_row) * num_q_heads_ * dim_qk_ + head_idx_ * dim_qk_;
            DataCopyParams copyParams;
            copyParams.blockCount = static_cast<uint16_t>(valid_rows);
            copyParams.blockLen = static_cast<uint16_t>(dim_qk_ * sizeof(fp8_e4m3fn_t));
            copyParams.srcStride = static_cast<uint16_t>((num_q_heads_ - 1) * dim_qk_ * sizeof(fp8_e4m3fn_t));
            copyParams.dstStride = 0;
            qGm_.SetGlobalBuffer((__gm__ fp8_e4m3fn_t *)qBase_ + qOffset, valid_rows * num_q_heads_ * dim_qk_);
            DataCopyPad(qBlockLocal, qGm_, copyParams, DataCopyPadParams{false, 0, 0, 0});
        }
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        qBlockBuf_.EnQue(qBlockLocal);
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

        LocalTensor<fp8_e4m3fn_t> qBlockLocal = qBlockBuf_.DeQue<fp8_e4m3fn_t>();
        LocalTensor<float> qBlockFP32Local = qBlockFP32Buf_.AllocTensor<float>();
        Duplicate(qBlockFP32Local, 0.0f, B_ * dim_qk_);
        PipeBarrier<PIPE_V>();

        uint32_t start_row = qb_local_ * B_;
        uint32_t valid_rows = (start_row + B_ <= q_len_) ? B_ : (q_len_ > start_row ? q_len_ - start_row : 0);
        uint32_t cast_count = valid_rows * dim_qk_;
        if (cast_count > 0) {
            Cast<float, fp8_e4m3fn_t>(qBlockFP32Local, qBlockLocal, RoundMode::CAST_NONE, cast_count);
        }
        PipeBarrier<PIPE_V>();
        qBlockBuf_.FreeTensor(qBlockLocal);
        qBlockFP32Buf_.EnQue(qBlockFP32Local);
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
        LocalTensor<float> scalesLocal = scalesBuf_.AllocTensor<float>();
        Duplicate(scalesLocal, 0.0f, B_);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
        event_t vsEvent = static_cast<event_t>(pipe_.FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(vsEvent);
        WaitFlag<HardEvent::V_S>(vsEvent);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);

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
        }
        scalesBuf_.EnQue(scalesLocal);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
    }

    __aicore__ inline void AccumulateBlock()
    {
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);

        LocalTensor<float> fp32Buf = qBlockFP32Buf_.DeQue<float>();
        LocalTensor<float> scaleBuf = scalesBuf_.DeQue<float>();
        __ubuf__ float *qFP32Ptr = (__ubuf__ float *)fp32Buf.GetPhyAddr();
        __ubuf__ float *scalePtr = (__ubuf__ float *)scaleBuf.GetPhyAddr();

        VF_CALL<WeightedBinaryReduceFull>(qFP32Ptr, scalePtr, static_cast<uint16_t>(R_),
                                           static_cast<uint16_t>(S_), static_cast<uint16_t>(dim_qk_));
        PipeBarrier<PIPE_V>();

        scalesBuf_.FreeTensor(scaleBuf);
        qBlockFP32Buf_.EnQue(fp32Buf);
    }

    /**
     * FlattenAndCast: UB 内 FP32 → BF16 类型转换
     *
     * qBlockFP32Buf 的前 S*D 个 FP32 元素
     * (即 AccumulateBlock 后留在那里的 groupSum 结果)
     * → Cast(RINT) → outQueueQflat [S*D] BF16
     */
    __aicore__ inline void FlattenAndCast()
    {
        LocalTensor<float> groupSumLocal = qBlockFP32Buf_.DeQue<float>();
        LocalTensor<bfloat16_t> qflatLocal = outQueueQflat_.AllocTensor<bfloat16_t>();
        Cast<bfloat16_t, float>(qflatLocal, groupSumLocal, RoundMode::CAST_RINT, qflat_dim_);
        qBlockFP32Buf_.FreeTensor(groupSumLocal);
        outQueueQflat_.EnQue(qflatLocal);
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
        uint32_t outOffset = batch_idx_ * (num_q_heads_ * tiling_->max_Qb * qflat_dim_) +
                             head_idx_ * (tiling_->max_Qb * qflat_dim_) + qb_local_ * qflat_dim_;
        qflatGm_.SetGlobalBuffer((__gm__ bfloat16_t *)qflatBase_ + outOffset, qflat_dim_);
        DataCopyPad(qflatGm_, qflatLocal, {1, static_cast<uint16_t>(qflat_dim_ * sizeof(bfloat16_t)), 0, 0});
        outQueueQflat_.FreeTensor(qflatLocal);
    }

    /**
     * WriteZeroBlock: 虚拟块输出补 0
     *
     * 当 qb_local_ >= num_Qb_（含 q_len_b == 0 的情况），
     * 向 GM 对应位置写入 qflat_dim_ 个 0（BF16）。
     *
     * GM 目标: qFlat[b, h, qb, :]
     *          GM offset = b * (H_q * max_Qb * S*D) + h * (max_Qb * S*D) + qb * S*D
     */
    __aicore__ inline void WriteZeroBlock()
    {
        LocalTensor<bfloat16_t> qflatLocal = outQueueQflat_.AllocTensor<bfloat16_t>();
        Duplicate(qflatLocal, (bfloat16_t)0, qflat_dim_);
        outQueueQflat_.EnQue<bfloat16_t>(qflatLocal);
        LocalTensor<bfloat16_t> qflatOut = outQueueQflat_.DeQue<bfloat16_t>();
        uint32_t outOffset = batch_idx_ * (num_q_heads_ * tiling_->max_Qb * qflat_dim_) +
                             head_idx_ * (tiling_->max_Qb * qflat_dim_) + qb_local_ * qflat_dim_;
        qflatGm_.SetGlobalBuffer((__gm__ bfloat16_t *)qflatBase_ + outOffset, qflat_dim_);
        DataCopyPad(qflatGm_, qflatOut, {1, static_cast<uint16_t>(qflat_dim_ * sizeof(bfloat16_t)), 0, 0});
        outQueueQflat_.FreeTensor(qflatOut);
    }

    TPipe pipe_;
    const StemPrepQTilingData *tiling_ = nullptr;

    uint32_t task_id_ = 0;
    uint32_t batch_idx_ = 0;
    uint32_t head_idx_ = 0;
    uint32_t num_q_heads_ = 0;
    uint32_t dim_qk_ = 0;
    uint32_t q_len_ = 0;
    uint32_t cu_off_ = 0;
    uint32_t num_Qb_ = 0;
    uint32_t B_ = 0;
    uint32_t S_ = 0;
    uint32_t R_ = 0;
    uint32_t qflat_dim_ = 0;

    uint32_t qb_start_ = 0;
    uint32_t qb_end_ = 0;
    uint32_t qb_local_ = 0;
    uint32_t pp_num_ = 2;

    GM_ADDR cuSeqLensQBase_ = nullptr;
    GM_ADDR qBase_ = nullptr;
    GM_ADDR qscaleBase_ = nullptr;
    GM_ADDR qflatBase_ = nullptr;

    GlobalTensor<fp8_e4m3fn_t> qGm_;
    GlobalTensor<float> qscaleGm_;
    GlobalTensor<int64_t> cuSeqLensGm_;
    GlobalTensor<bfloat16_t> qflatGm_;

    TQue<QuePosition::VECIN, DOUBLE_BUFFER_DEPTH> qBlockBuf_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER_DEPTH> qBlockFP32Buf_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER_DEPTH> scalesBuf_;
    TQue<TPosition::VECOUT, DOUBLE_BUFFER_DEPTH> outQueueQflat_;
    TBuf<QuePosition::VECCALC> cuSeqLensQBuf_;
};

#endif
