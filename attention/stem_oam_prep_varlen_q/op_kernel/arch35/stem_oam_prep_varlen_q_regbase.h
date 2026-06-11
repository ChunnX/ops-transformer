/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef STEM_OAM_PREP_VARLEN_Q_REGBASE_H
#define STEM_OAM_PREP_VARLEN_Q_REGBASE_H

#include "kernel_operator.h"

using namespace AscendC;

constexpr uint32_t DIM_QK = 128;
constexpr uint32_t MAX_STEM_BLOCK_SIZE = 256;
constexpr uint32_t MAX_STEM_STRIDE = 64;
constexpr uint32_t MAX_QFLAT_DIM = MAX_STEM_STRIDE * DIM_QK;
constexpr uint32_t VARLEN_READ_ALIGN = 8;

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

    static constexpr CastTrait castTrait = {
        RegLayout::ZERO, SatMode::NO_SAT,
        MaskMergeMode::MERGING, RoundMode::CAST_NONE
    };

    MaskReg maskF = CreateMask<float>();

    for (uint16_t g = 0; g < S; g++) {
        for (uint16_t vi = 0; vi < vlIterF; vi++) {
            uint32_t remainCount = static_cast<uint32_t>(dimQk - vi * VL_F);
            maskF = UpdateMask<float>(remainCount);
            Duplicate(accReg, 0.0f, maskF);
        }

        for (uint16_t r = 0; r < R; r++) {
            uint16_t row_idx = r * S + g;

            DataCopy<float, LoadDist::DIST_BRC_B32>(
                scaleBrcReg, scalePtr + row_idx);

            for (uint16_t vi = 0; vi < vlIterF; vi++) {
                uint32_t remainCount = static_cast<uint32_t>(dimQk - vi * VL_F);
                maskF = UpdateMask<float>(remainCount);
                uint32_t qOffset = static_cast<uint32_t>(row_idx) * dimQk + vi * VL_F;

                DataCopy(qReg, qFP32Ptr + qOffset);
                Mul(qReg, qReg, scaleBrcReg, maskF);
                Add(accReg, accReg, qReg, maskF);
            }
        }

        for (uint16_t vi = 0; vi < vlIterF; vi++) {
            uint32_t remainCount = static_cast<uint32_t>(dimQk - vi * VL_F);
            maskF = UpdateMask<float>(remainCount);
            uint32_t outOffset = static_cast<uint32_t>(g) * dimQk + vi * VL_F;
            DataCopy(groupSumPtr + outOffset, accReg, maskF);
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
        StemPrepQTilingData* tiling)
    {
        tiling_ = tiling;
        task_id_ = GetBlockIdx();
        if (task_id_ >= tiling->total_tasks) return;

        num_q_heads_ = tiling->num_q_heads;
        dim_qk_     = tiling->dim_qk;
        max_q_padded_ = tiling->max_q_padded;
        B_ = tiling->stem_block_size;
        S_ = tiling->stem_stride;
        R_ = tiling->R_val;
        qflat_dim_ = tiling->kflat_dim;

        qSeqLensBase_ = q_seq_lens;
        cuSeqLensBase_ = cu_seqlens_q;

        if (tiling->split_mode == 0) {
            batch_idx_ = task_id_ / num_q_heads_;
            head_idx_  = task_id_ % num_q_heads_;
        } else if (tiling->split_mode == 1) {
            uint32_t base = tiling->cores_per_bh_base;
            uint32_t rem  = tiling->remainder_bh;
            uint32_t big  = base + 1;
            uint32_t big_total = rem * big;
            uint32_t bh_idx;
            if (task_id_ < big_total) {
                bh_idx      = task_id_ / big;
                core_in_bh_ = task_id_ % big;
            } else {
                uint32_t adj = task_id_ - big_total;
                bh_idx      = rem + adj / base;
                core_in_bh_ = adj % base;
            }
            batch_idx_ = bh_idx / num_q_heads_;
            head_idx_  = bh_idx % num_q_heads_;
            cores_per_bh_this_ = (bh_idx < rem) ? big : base;
        } else {
            batch_idx_ = 0;
            head_idx_ = 0;
            qb_start_ = 0;
            core_in_bh_ = 0;
            cores_per_bh_this_ = 1;
        }

        ReadVarlenInfo(batch_idx_);

        if (tiling->split_mode == 0) {
            qb_start_ = 0;
            qb_end_   = num_Qb_;
        } else if (tiling->split_mode == 1) {
            uint32_t cph  = cores_per_bh_this_;
            uint32_t qpc  = num_Qb_ / cph;
            uint32_t qrem = num_Qb_ % cph;
            if (core_in_bh_ < qrem) {
                qb_start_ = core_in_bh_ * (qpc + 1);
                qb_end_   = qb_start_ + qpc + 1;
            } else {
                qb_start_ = qrem * (qpc + 1) + (core_in_bh_ - qrem) * qpc;
                qb_end_   = qb_start_ + qpc;
            }
        }

        qBase_ = q;
        qscaleBase_ = qscale;
        qflatBase_ = qflat;

        pipe_.InitBuffer(qBlockBuf_, B_ * dim_qk_ * sizeof(fp8_e4m3fn_t));
        pipe_.InitBuffer(qBlockFP32Buf_, B_ * dim_qk_ * sizeof(float));
        pipe_.InitBuffer(scalesBuf_, B_ * sizeof(float));
        pipe_.InitBuffer(groupSumBuf_, qflat_dim_ * sizeof(float));
        pipe_.InitBuffer(outQueueQflat_, 1, qflat_dim_ * sizeof(bfloat16_t));
        pipe_.InitBuffer(varlenInfoBuf_, VARLEN_READ_ALIGN * sizeof(int32_t));
    }

    __aicore__ inline void Process()
    {
        if (task_id_ >= tiling_->total_tasks) return;

        for (uint32_t qb = qb_start_; qb < qb_end_; qb++) {
            ProcessOneBlock(qb);
        }
    }

private:
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

    __aicore__ inline void ProcessOneBlock(uint32_t qb)
    {
        CopyInQBlock(qb);
        CastBlockToFP32(qb);
        CopyInScalesBulk(qb);
        AccumulateBlock(qb);
        FlattenAndCast();
        CopyOutBlock(qb);
    }

    __aicore__ inline void CopyInQBlock(uint32_t qb)
    {
        LocalTensor<int8_t> qBlockI8 = qBlockBuf_.Get<int8_t>();
        Duplicate(qBlockI8, (int8_t)0, B_ * dim_qk_);
        PipeBarrier<PIPE_V>();

        uint32_t start_row = qb * B_;
        uint32_t valid_rows = (start_row + B_ <= q_len_) ? B_ :
                              (q_len_ > start_row ? q_len_ - start_row : 0);

        LocalTensor<fp8_e4m3fn_t> qBlockLocal = qBlockBuf_.Get<fp8_e4m3fn_t>();
        for (uint32_t i = 0; i < valid_rows; i++) {
            uint32_t globalPos = cu_off_ + start_row + i;
            uint32_t qOffset = globalPos * num_q_heads_ * dim_qk_ + head_idx_ * dim_qk_;
            qGm_.SetGlobalBuffer((__gm__ fp8_e4m3fn_t*)qBase_ + qOffset, dim_qk_);
            DataCopyPad(qBlockLocal[i * dim_qk_], qGm_,
                {1, static_cast<uint16_t>(dim_qk_ * sizeof(fp8_e4m3fn_t)), 0, 0},
                {false, 0, 0, 0});
        }
        PipeBarrier<PIPE_MTE2>();
    }

    __aicore__ inline void CastBlockToFP32(uint32_t qb)
    {
        LocalTensor<fp8_e4m3fn_t> qBlockLocal = qBlockBuf_.Get<fp8_e4m3fn_t>();
        LocalTensor<float> qBlockFP32Local = qBlockFP32Buf_.Get<float>();
        Duplicate(qBlockFP32Local, 0.0f, B_ * dim_qk_);

        uint32_t start_row = qb * B_;
        uint32_t valid_rows = (start_row + B_ <= q_len_) ? B_ :
                              (q_len_ > start_row ? q_len_ - start_row : 0);
        uint32_t cast_count = valid_rows * dim_qk_;
        if (cast_count > 0) {
            Cast<float, fp8_e4m3fn_t>(qBlockFP32Local, qBlockLocal,
                RoundMode::CAST_NONE, cast_count);
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void CopyInScalesBulk(uint32_t qb)
    {
        LocalTensor<float> scalesLocal = scalesBuf_.Get<float>();
        uint32_t scaleBase = batch_idx_ * (num_q_heads_ * max_q_padded_)
                           + head_idx_  * max_q_padded_
                           + qb * B_;
        qscaleGm_.SetGlobalBuffer((__gm__ float*)qscaleBase_ + scaleBase, B_);
        DataCopyPad(scalesLocal, qscaleGm_,
            {1, static_cast<uint16_t>(B_ * sizeof(float)), 0, 0},
            {false, 0, 0, 0});
        PipeBarrier<PIPE_MTE2>();
    }

    __aicore__ inline void AccumulateBlock(uint32_t qb)
    {
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
    }

    __aicore__ inline void FlattenAndCast()
    {
        LocalTensor<float> groupSumLocal = groupSumBuf_.Get<float>();
        LocalTensor<bfloat16_t> qflatLocal = outQueueQflat_.AllocTensor<bfloat16_t>();
        Cast<bfloat16_t, float>(qflatLocal, groupSumLocal,
            RoundMode::CAST_RINT, qflat_dim_);
        outQueueQflat_.EnQue<bfloat16_t>(qflatLocal);
    }

    __aicore__ inline void CopyOutBlock(uint32_t qb)
    {
        LocalTensor<bfloat16_t> qflatLocal = outQueueQflat_.DeQue<bfloat16_t>();
        uint32_t outOffset = batch_idx_ * (num_q_heads_ * tiling_->max_Qb * qflat_dim_)
                            + head_idx_  * (tiling_->max_Qb * qflat_dim_)
                            + qb         * qflat_dim_;
        qflatGm_.SetGlobalBuffer((__gm__ bfloat16_t*)qflatBase_ + outOffset, qflat_dim_);
        DataCopyPad(qflatGm_, qflatLocal,
            {1, static_cast<uint16_t>(qflat_dim_ * sizeof(bfloat16_t)), 0, 0});
        outQueueQflat_.FreeTensor(qflatLocal);
    }

    TPipe pipe_;
    StemPrepQTilingData* tiling_;

    uint32_t task_id_ = 0;
    uint32_t batch_idx_ = 0;
    uint32_t head_idx_ = 0;
    uint32_t num_q_heads_ = 0;
    uint32_t dim_qk_ = 0;
    uint32_t max_q_padded_ = 0;
    uint32_t q_len_ = 0;
    uint32_t num_Qb_ = 0;
    uint32_t q_padded_ = 0;
    uint32_t cu_off_ = 0;
    uint32_t B_ = 0;
    uint32_t S_ = 0;
    uint32_t R_ = 0;
    uint32_t qflat_dim_ = 0;

    uint32_t qb_start_ = 0;
    uint32_t qb_end_ = 0;
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
