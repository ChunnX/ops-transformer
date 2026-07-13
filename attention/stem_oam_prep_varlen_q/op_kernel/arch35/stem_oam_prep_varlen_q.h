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
 *   4. AccumulateBlock (Weighted Group Sum, in-place binary reduce):
 *      qBlockFP32Buf 逻辑 reshape 为 [R, S, D]
 *      scalesBuf 逻辑 reshape 为 [R, S]
 *
 *      Phase 1: qBlockFP32Buf[(r*S+g)*D : (r*S+g+1)*D] *= scales[r*S+g]
 *                对每个 (r, g) 的原地乘 scale
 *      Phase 2: 沿 R 轴二进制分解 + 原地配对归约（任意 R 均支持）
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
 * __simd_callee__ helpers for WeightedBinaryReduceChunkVf.
 *
 * Each helper is pure — uses MicroAPI directly, no if statements.
 *
 * MulsOnAllRows:
 *   qFP32[(r*S + g)*D + d] *= scale[r*S + g]  for all r,g,d
 *
 * ReduceChunkInPlace:
 *   pair-wise reduce chunk (size 2^p, starting at base_r) so that
 *   row[(base_r*S + g), :] holds the intra-chunk sum
 *
 * AddMergeToAccumulator:
 *   row[g, :] += row[(base_r*S + g), :]  for g in [0, S)
 */

__simd_callee__ inline void MulsOnAllRows(__ubuf__ float* qFP32Ptr, __ubuf__ float* scalePtr,
                                          uint16_t R, uint16_t S, uint16_t D)
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

__simd_callee__ inline void ReduceChunkInPlace(__ubuf__ float* qFP32Ptr,
                                               uint16_t base_r, uint16_t p,
                                               uint16_t S, uint16_t D)
{
    using namespace AscendC::MicroAPI;

    constexpr uint16_t VL_F = VECTOR_REG_WIDTH / sizeof(float);
    uint16_t vlIter = (D + VL_F - 1) / VL_F;

    MaskReg maskAll = CreateMask<float, MaskPattern::ALL>();
    RegTensor<float> regA;
    RegTensor<float> regB;

    uint16_t chunk_size = 1;
    for (uint16_t i = 0; i < p; i++) { chunk_size <<= 1; }

    for (uint16_t k = 0; k < p; k++) {
        uint16_t stride = static_cast<uint16_t>(1 << k);
        uint16_t step = stride << 1;
        for (uint16_t g = 0; g < S; g++) {
            for (uint16_t r_local = 0; r_local < chunk_size; r_local += step) {
                uint16_t r_dst = base_r + r_local;
                uint16_t r_src = base_r + r_local + stride;
                uint32_t offDst = static_cast<uint32_t>(r_dst * S + g) * D;
                uint32_t offSrc = static_cast<uint32_t>(r_src * S + g) * D;
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

__simd_callee__ inline void AddMergeToAccumulator(__ubuf__ float* qFP32Ptr,
                                                  uint16_t base_r,
                                                  uint16_t S, uint16_t D)
{
    using namespace AscendC::MicroAPI;

    constexpr uint16_t VL_F = VECTOR_REG_WIDTH / sizeof(float);
    uint16_t vlIter = (D + VL_F - 1) / VL_F;

    MaskReg maskAll = CreateMask<float, MaskPattern::ALL>();
    RegTensor<float> regA;
    RegTensor<float> regB;

    for (uint16_t g = 0; g < S; g++) {
        uint32_t offAcc   = static_cast<uint32_t>(g) * D;
        uint32_t offChunk = static_cast<uint32_t>(base_r * S + g) * D;
        for (uint16_t vi = 0; vi < vlIter; vi++) {
            uint32_t d = static_cast<uint32_t>(vi) * VL_F;
            DataCopy(regA, qFP32Ptr + offAcc + d);
            DataCopy(regB, qFP32Ptr + offChunk + d);
            Add(regA, regA, regB, maskAll);
            DataCopy(qFP32Ptr + offAcc + d, regA, maskAll);
        }
    }
}

/**
 * WeightedBinaryReduceChunkVf: one chunk of the binary decomposition
 *
 * Single VF entry; no if inside. Control flow is bounded by uint16_t flags
 * (do_muls / merge_count in {0,1}) via for-loop iteration counts.
 *   - First chunk (base_r=0): do_muls=1, merge_count=0  → Muls all rows, reduce chunk, NO merge
 *     (row[g] already holds the chunk sum after reduce, no merge needed)
 *   - Later chunks (base_r>0): do_muls=0, merge_count=1 → reduce chunk, then Add merge
 *     (row[g] is accumulator, just Add the new chunk in)
 *
 * Muls is fused into the same entry as reduce+merge, avoiding extra VF calls.
 */
__simd_vf__ inline void WeightedBinaryReduceChunkVf(
    __ubuf__ float* qFP32Ptr,
    __ubuf__ float* scalePtr,
    uint16_t R,
    uint16_t base_r, uint16_t p,
    uint16_t S, uint16_t D,
    uint16_t do_muls, uint16_t merge_count)
{
    for (uint16_t m = 0; m < do_muls; m++) {
        MulsOnAllRows(qFP32Ptr, scalePtr, R, S, D);
    }

    ReduceChunkInPlace(qFP32Ptr, base_r, p, S, D);

    for (uint16_t m = 0; m < merge_count; m++) {
        AddMergeToAccumulator(qFP32Ptr, base_r, S, D);
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
        // qBlockFP32Buf_: [B, D] FP32 = 128×128×4 = 64 KB  (also stores binary-reduced result in first S rows)
        // scalesBuf_:     [B]    FP32 = 128×4          = 0.5 KB  (只存当前 head 的 scale)
        // outQueueQflat_: [S*D]  BF16 = 2048×2    = 4 KB
        // varlenInfoBuf_: [MAX_BATCH_SIZE] INT32 = 4 KB
        pipe_.InitBuffer(qBlockBuf_, B_ * dim_qk_ * sizeof(fp8_e4m3fn_t));
        pipe_.InitBuffer(qBlockFP32Buf_, B_ * dim_qk_ * sizeof(float));
        pipe_.InitBuffer(scalesBuf_, B_ * sizeof(float));
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
        // 虚拟块：qb_local_ >= num_Qb_，输出补 0
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
     * AccumulateBlock: Binary-reduce based Weighted Group Sum (in-place)
     *
     * qBlockFP32Buf [B, D] 逻辑 reshape 为 [R, S, D]
     * scalesBuf     [B]    逻辑 reshape 为 [R, S]
     *
     * 对每个 g ∈ [0, S-1]:
     *   groupSum[g, :] = Σ_{r=0}^{R-1} qBlockFP32[r*S+g, :] × scales[r*S+g]
     *
     * Binary decompose R (MSB→LSB) and dispatch one chunk per VF call.
     * Each VF call fuses:
     *   - Muls on all R*S rows   (only the first chunk, do_muls=1)
     *   - Pairwise reduce in p layers on this chunk (always)
     *   - Add-merge into row[g]  (only later chunks, merge_count=1)
     *
     * 结果留在 qBlockFP32Buf 的前 S 行 (行 g, 偏移 g*D, g in [0, S))
     */
    __aicore__ inline void AccumulateBlock()
    {
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);

        __ubuf__ float* qFP32Ptr =
            (__ubuf__ float*)qBlockFP32Buf_.Get<float>().GetPhyAddr();
        __ubuf__ float* scalePtr =
            (__ubuf__ float*)scalesBuf_.Get<float>().GetPhyAddr();

        uint16_t base_r = 0;
        uint16_t R_rem = static_cast<uint16_t>(R_);
        while (R_rem > 0) {
            // floor_log2(R_rem) — compute highest bit position
            uint16_t p = 0;
            uint16_t tmp = R_rem;
            while (tmp >= 2) {
                tmp >>= 1;
                p++;
            }
            uint16_t chunk = static_cast<uint16_t>(1 << p);

            // First chunk: Muls all rows + reduce chunk + (no merge, row[g] already holds sum)
            // Later chunks: reduce chunk + Add merge into accumulator row[g]
            uint16_t do_muls     = (base_r == 0) ? 1 : 0;
            uint16_t merge_count = (base_r > 0)  ? 1 : 0;

            VF_CALL<WeightedBinaryReduceChunkVf>(
                qFP32Ptr, scalePtr,
                static_cast<uint16_t>(R_),
                base_r, p,
                static_cast<uint16_t>(S_),
                static_cast<uint16_t>(dim_qk_),
                do_muls, merge_count);
            PipeBarrier<PIPE_V>();

            base_r += chunk;
            R_rem -= chunk;
        }
        // qBlockFP32Buf 的前 S*D 个 float (即 row[g] for g in [0,S))
        // 保存了 groupSum[g, :] 的结果
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
        LocalTensor<float> groupSumLocal = qBlockFP32Buf_.Get<float>();
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
        uint32_t outOffset = batch_idx_ * (num_q_heads_ * tiling_->max_Qb * qflat_dim_)
                           + head_idx_  * (tiling_->max_Qb * qflat_dim_)
                           + qb_local_  * qflat_dim_;
        qflatGm_.SetGlobalBuffer((__gm__ bfloat16_t*)qflatBase_ + outOffset, qflat_dim_);
        DataCopyPad(qflatGm_, qflatOut,
            {1, static_cast<uint16_t>(qflat_dim_ * sizeof(bfloat16_t)), 0, 0});
        outQueueQflat_.FreeTensor(qflatOut);
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
    TQue<TPosition::VECOUT, 1> outQueueQflat_;
    TBuf<TPosition::VECCALC> varlenInfoBuf_;
};

#endif
