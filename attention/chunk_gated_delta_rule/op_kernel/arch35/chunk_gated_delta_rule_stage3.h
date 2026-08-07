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
 * \file chunk_gated_delta_rule_stage3.h
 * \brief
 */
#ifndef CHUNK_GATED_DELTA_RULE_STAGE3_H
#define CHUNK_GATED_DELTA_RULE_STAGE3_H

#include "kernel_tiling/kernel_tiling.h"
#include "chunk_gated_delta_rule_utils.h"
#include "../chunk_gated_delta_rule_tiling_data.h"

namespace ChunkGatedDeltaRule {
using namespace AscendC;
using namespace matmul;

using aT3 = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t, true>;
using bT3 = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t, true>;
using cT3 = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t>;
using StageThreeMT = matmul::MatmulImpl<aT3, bT3, cT3>;

struct StageThreeParams {
    // qkt: Stage1 Q @ K^T result.  Workspace capacity is [Nv, Sp, C]; the
    // current chunk payload is [L, L] BF16 (tail chunks are compact in GM).
    GlobalTensor<bfloat16_t> qkt;
    // Cumulative log-gate G, not exp(G): [Nv, Sp] FP32, current slice [L].
    GlobalTensor<float> gCumExp;
    // Z after Stage2's in-place state correction: [Nv, Sp, Dv] BF16.
    GlobalTensor<bfloat16_t> vInner;
    // Inclusive lower-triangular mask: [C, C] FP32.
    GlobalTensor<float> maskTensor;
    // Stage workspace base.  Stage3 uses one [C,C] BF16 tile per logical Core.
    GM_ADDR ws;
    // Final TND output [T,Nv,Dv] BF16.  Stage2 has already written O_inter.
    GlobalTensor<bfloat16_t> attnOut;
    StageThreeMT *mm3;
    TPipe *pipe;
    ChunkGroup *cg;
    float scale; // attention scale, normally 1/sqrt(Dk)
    int64_t Nv;  // value-head count
    int64_t Nk;  // query/key-head count (not used directly in Stage3)
    int64_t Dv;  // value width
    int64_t Dk;  // query/key width (qkt already contracted over Dk)
};

template <bool gOptional>
class Stage3 {
public:
    __aicore__ inline void Init(StageThreeParams *initParams, int32_t coreNum)
    {
        coreId_ = GetBlockIdx();
        sTP_ = initParams;
        pipe_ = sTP_->pipe;
        chunkSize_ = sTP_->cg->chunkSize;
        seqLength_ = sTP_->cg->length;
        Sp_ = (seqLength_ + chunkSize_ - 1) / chunkSize_ * chunkSize_;
        chunkNum_ = (seqLength_ + chunkSize_ - 1) / chunkSize_;
        coreNum_ = coreNum;
        Nv_ = sTP_->Nv;
        Nk_ = sTP_->Nk;
        Dv_ = sTP_->Dv;
        Dk_ = sTP_->Dk;
        paddedDv_ = Ceil(Dv_, BLOCK_SIZE / sizeof(bfloat16_t)) * (BLOCK_SIZE / sizeof(bfloat16_t));

        // The Stage workspace starts with the region reserved by the common
        // stage layout.  Each logical Core owns one [C,C] BF16 tile after
        // this offset; AIV writes it and AIC reads it.
        uint64_t workSpaceOffset = 0;
        tmpGM_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(
            initParams->ws + workSpaceOffset + coreNum_ * chunkSize_ * chunkSize_ * sizeof(float)));
        if ASCEND_IS_AIC {
            // AIC only uses tmpGM and the matmul object.  No vector queues or
            // mask UB are needed on this side.
            return;
        }
        coreId_ /= AIC_AIV_1_1;
        if (GetSubBlockIdx() == 1) {
            // AIV1 deliberately skips UB initialization.  It still enters
            // Process() later to send/wait the same cross-core flags as AIV0.
            return;
        }
        uint64_t inQueueSize =
            static_cast<uint64_t>(chunkSize_) * AscendC::Std::max((int64_t)chunkSize_, paddedDv_) * sizeof(bfloat16_t);
        pipe_->InitBuffer(inQueue_, BUFFER_NUM_ONE, inQueueSize);
        pipe_->InitBuffer(outQueue_, BUFFER_NUM_ONE,
                          chunkSize_ > paddedDv_ ? chunkSize_ * chunkSize_ * sizeof(float) :
                                                   chunkSize_ * paddedDv_ * sizeof(bfloat16_t));
        pipe_->InitBuffer(tmpBuff_, (STAGE3_BUFFER_COUNT * chunkSize_ * chunkSize_ * sizeof(float)));
        uint64_t buffOffset = 0;
        uint64_t tmpOffset = chunkSize_ * chunkSize_;
        tmpBuffer1_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(tmpOffset), buffOffset);
        buffOffset += tmpOffset * sizeof(float);
        tmpBuffer2_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(tmpOffset), buffOffset);
        buffOffset += tmpOffset * sizeof(float);
        maskBuffer_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(tmpOffset), buffOffset);

        // Copy the prebuilt inclusive lower-triangular [C,C] FP32 mask to
        // UB once per Stage3 invocation.  The mask is reused for every
        // (value-head, chunk) task assigned to this logical Core.
        DataCopyExtParams inParams{static_cast<uint16_t>(chunkSize_), static_cast<uint32_t>(chunkSize_ * sizeof(float)),
                                   0, 0, 0};
        DataCopyPadExtParams<float> copyPadParams{false, 0, 0, 0};
        DataCopyPad(maskBuffer_, sTP_->maskTensor, inParams, copyPadParams);
        int32_t eventID = static_cast<int32_t>(pipe_->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventID);
        WaitFlag<HardEvent::MTE2_V>(eventID);
    }

    __aicore__ inline void Process()
    {
        // Stage2 has serialized the state dependency across chunks.  Once
        // Stage2 is complete, every (value-head, chunk) pair is independent
        // for output synthesis, so Stage3 flattens the two dimensions.
        int64_t totalChunks = Nv_ * chunkNum_; // Nv * number of chunks
        int64_t chunksPerCore = (totalChunks + coreNum_ - 1) / coreNum_;
        int64_t lastChunkSize = seqLength_ % chunkSize_ == 0 ? chunkSize_ : seqLength_ % chunkSize_;
        int64_t startChunk = coreId_ * chunksPerCore;
        int64_t endChunk = startChunk + chunksPerCore > totalChunks ? totalChunks : startChunk + chunksPerCore;
        for (int64_t idx = startChunk; idx < endChunk; idx++) {
            // Head-major task mapping: one head's chunks are consecutive.
            int64_t nvId = idx / chunkNum_;
            int64_t chunkId = idx % chunkNum_;
            int64_t chunkPos = chunkId * chunkSize_;                                 // 当前chunk起始位置
            curChunkSize_ = (chunkId == chunkNum_ - 1) ? lastChunkSize : chunkSize_; // 尾块
            if ASCEND_IS_AIV {
                if (GetSubBlockIdx() == 0) {
                    // AIV0 produces A in FP32 UB and stores BF16 A to the
                    // per-core tmpGM tile.  AIV1 has no arithmetic work.
                    CalMaskedQKTV2(tmpGM_[coreId_ * chunkSize_ * chunkSize_], nvId, chunkPos);
                }
                // Both AIVs must arrive before AIC consumes tmpGM.  The
                // template parameter 0x2 denotes the two-AIV participants.
                CrossCoreSetFlag<0x2, PIPE_MTE3>(0x4);
                CrossCoreWaitFlag(0x3);
            }

            if ASCEND_IS_AIC {
                // 0x4 is emitted by both AIVs.  Only after both arrivals may
                // AIC read the weight tile and run A @ Z.
                CrossCoreWaitFlag(0x4);
                AICProcess(tmpGM_[coreId_ * chunkSize_ * chunkSize_], sTP_->vInner[nvId * Sp_ * Dv_ + chunkPos * Dv_],
                           sTP_->attnOut[nvId * Dv_ + chunkPos * Nv_ * Dv_]);
                // IterateAll(..., 1) accumulates into the existing output:
                // attnOut <- O_inter + A @ Z.  This flag releases both AIVs
                // so the single tmpGM tile can be overwritten by the next
                // task.
                CrossCoreSetFlag<0x2, PIPE_FIX>(0x3);
            }
        }
    }

    template <bool HAS_GATE>
    static __simd_vf__ inline void CalMaskedQKTVf(__ubuf__ bfloat16_t *out, __ubuf__ bfloat16_t *qkt,
                                                  __ubuf__ float *gCum, __ubuf__ float *causalMask, float scale,
                                                  uint16_t validSize, uint16_t rowStride)
    {
        using namespace AscendC::MicroAPI;

        // DIST_UNPACK_B16/DIST_PACK_B32 按 FP32 lane 排布，一个寄存器可处理
        // 64 个 BF16 元素。有效列数 L 大于 64 时，每行由多个 repeat 完成。
        constexpr uint16_t FP32_ELEMS_PER_REPEAT = VECTOR_REG_WIDTH / sizeof(float);
        static constexpr CastTrait CAST_BF16_TO_FP32 = {
            RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        static constexpr CastTrait CAST_FP32_TO_BF16 = {
            RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

        RegTensor<bfloat16_t> qktB16;
        RegTensor<bfloat16_t> outB16;
        RegTensor<float> qktF32;
        RegTensor<float> gRowF32;
        RegTensor<float> gColF32;
        RegTensor<float> decayF32;
        RegTensor<float> maskF32;
        MaskReg activeMask;

        // 每个 repeat 处理 64 列，最后一个 repeat 通过 activeMask 屏蔽越过 L
        // 的 lane
        const uint16_t repeatTimes =
            static_cast<uint16_t>((validSize + FP32_ELEMS_PER_REPEAT - 1) / FP32_ELEMS_PER_REPEAT);

        // 无 gate 时 decay 恒为 1
        if constexpr (!HAS_GATE) {
            Duplicate(decayF32, 1.0f);
        }

        for (uint16_t row = 0; row < validSize; ++row) {
            if constexpr (HAS_GATE) {
                // G_i 在当前行的所有列分段中不变，因此每行只广播一次。
                DataCopy<float, LoadDist::DIST_BRC_B32>(gRowF32, gCum + row);
            }

            // UpdateMask 会消耗 remainingCount，因此每一行开始时重置为 L。
            uint32_t remainingCount = validSize;
            for (uint16_t repeat = 0; repeat < repeatTimes; ++repeat) {
                activeMask = UpdateMask<float>(remainingCount);
                const uint32_t colOffset = static_cast<uint32_t>(repeat) * FP32_ELEMS_PER_REPEAT;
                const uint32_t offset = static_cast<uint32_t>(row) * rowStride + colOffset;

                DataCopy(maskF32, causalMask + offset);
                if constexpr (HAS_GATE) {
                    // 当前寄存器覆盖列 [colOffset, colOffset+64)，加载对应的 G_j。
                    DataCopy(gColF32, gCum + colOffset);

                    // decay[i,j] = exp((G_i-G_j) * mask[i,j])。
                    // 上三角位置先得到 exp(0)=1，随后仍由最终一次乘 mask 清零。
                    Sub(decayF32, gRowF32, gColF32, activeMask);
                    Mul(decayF32, decayF32, maskF32, activeMask);
                    Exp(decayF32, decayF32, activeMask);
                }

                // A[i,j] = BF16(FP32(qkt[i,j]) * scale * decay[i,j] * mask[i,j])。
                // HAS_GATE=false 时 decayF32 为循环外构造的全 1；HAS_GATE=true
                // 时 decayF32 为上方当前分段计算的 exp(G_i-G_j)。
                DataCopy<bfloat16_t, LoadDist::DIST_UNPACK_B16>(qktB16, qkt + offset);
                Cast<float, bfloat16_t, CAST_BF16_TO_FP32>(qktF32, qktB16, activeMask);
                Muls(qktF32, qktF32, scale, activeMask);
                Mul(qktF32, qktF32, decayF32, activeMask);
                Mul(qktF32, qktF32, maskF32, activeMask);
                Cast<bfloat16_t, float, CAST_FP32_TO_BF16>(outB16, qktF32, activeMask);
                DataCopy<bfloat16_t, StoreDist::DIST_PACK_B32>(out + offset, outB16, activeMask);
            }
        }
    }

    __aicore__ inline void CalMaskedQKTV2(GlobalTensor<bfloat16_t> outGM, int nvId, int chunkPos)
    {
        // 当前 chunk 的有效长度为 L=curChunkSize_，容量记为 C=chunkSize_；
        // 计算：
        //   A[i,j] = BF16(FP32(qkt[i,j]) * scale * exp(G_i-G_j) * 1(j<=i))，[L,L]。
        if (gOptional_) {
            // gCumExp，GM 中实际保存的是 Stage1 写出的cumsum log-gate：
            //   G_i = sum_{k=0}^{i} g_k，不是 exp(G_i)。
            // workspace [Nv,Sp] FP32；当前地址指向第 nvId 个 head、
            // 当前 chunk 起点，AlignedCopyIn(1,L) 搬入当前切片 [L]。
            // 前 L 个元素有效。
            AlignedCopyIn(sTP_->gCumExp[nvId * Sp_ + chunkPos], 1, curChunkSize_);
            auto gCum = inQueue_.DeQue<float>();
            // inQueue_ 复用搬入 qkt，gCumSum 保存临时位置 tmpBuffer1_
            // VF 只读取 [0,L)，尾部 [L,C) 的对齐填充不会参与计算。
            DataCopy(tmpBuffer1_, gCum, chunkSize_);
            PipeBarrier<PIPE_V>();
            // 释放输入队列供 qkt 搬入使用。
            inQueue_.FreeTensor(gCum);
        }

        // qkt 是 Stage1 计算的 QK^T，layout为 [Nv,Sp,C] BF16。
        // 当前 chunk 的有效载荷是紧凑的 [L,L]；AlignedCopyIn(L,L) 搬入后，
        // UB 使用固定行跨度 C，物理计算视图为 [L,C]，每行仅前 L 列有效。
        AlignedCopyIn(sTP_->qkt[nvId * Sp_ * chunkSize_ + chunkPos * chunkSize_], curChunkSize_, curChunkSize_);
        auto qkt = inQueue_.DeQue<bfloat16_t>();
        // VF 输出同样采用 [L,C] BF16 的 UB 布局，以便按固定行跨度写入每一行。
        auto scaleQkt = outQueue_.AllocTensor<bfloat16_t>();

        // VF_CALL 取出物理地址：
        //   outPtr  : 输出 A，[L,C] BF16，前 L 列有效；
        //   qktPtr  : 输入 qkt，[L,C] BF16；
        //   gCumPtr : 累计 log-gate G，[C] FP32，前 L 个元素有效；
        //   maskPtr : 预先生成的包含对角线的下三角 mask，[C,C] FP32。
        auto outPtr = reinterpret_cast<__ubuf__ bfloat16_t *>(scaleQkt.GetPhyAddr());
        auto qktPtr = reinterpret_cast<__ubuf__ bfloat16_t *>(qkt.GetPhyAddr());
        auto gCumPtr = reinterpret_cast<__ubuf__ float *>(tmpBuffer1_.GetPhyAddr());
        auto maskPtr = reinterpret_cast<__ubuf__ float *>(maskBuffer_.GetPhyAddr());
        // validSize=L 控制有效行列数，rowStride=C 控制 UB 行 stride
        //   BF16 qkt -> FP32 -> *scale -> *decay -> *mask -> BF16(RINT)。
        if (gOptional_) {
            AscendC::VF_CALL<CalMaskedQKTVf<true>>(outPtr, qktPtr, gCumPtr, maskPtr, sTP_->scale,
                                                  static_cast<uint16_t>(curChunkSize_),
                                                  static_cast<uint16_t>(chunkSize_));
        } else {
            AscendC::VF_CALL<CalMaskedQKTVf<false>>(outPtr, qktPtr, gCumPtr, maskPtr, sTP_->scale,
                                                   static_cast<uint16_t>(curChunkSize_),
                                                   static_cast<uint16_t>(chunkSize_));
        }

        // AlignedCopyOut 从 [L,C] UB 中搬出数据
        // 每行前 L 列，写成紧凑的 [L,L] BF16 到每个核 tmpGM，供 AIC 做 A@Z。
        outQueue_.EnQue(scaleQkt);
        AlignedCopyOut(outGM, curChunkSize_, curChunkSize_);
        inQueue_.FreeTensor(qkt);
    }

    __aicore__ inline void CalMaskedQKT(GlobalTensor<bfloat16_t> outGM, int nvId, int chunkPos)
    {
        // chunkSize 大小进行自动补齐
        //
        // Effective matrix shape is [L,L], where L=curChunkSize_.  UB uses a
        // fixed row stride C, so vector operations below cover L*C elements;
        // columns >= L are padding and are removed by the inclusive mask.
        //
        // With g: G is the [L] cumulative log-gate slice written by Stage1.
        // Broadcast creates G_i and G_j, then decay[i,j]=exp(G_i-G_j).
        // Without g, decay is one and only the causal mask remains.
        if constexpr (gOptional) {
            AlignedCopyIn(sTP_->gCumExp[nvId * Sp_ + chunkPos], 1, curChunkSize_); // 自动补齐
            auto g_cum = inQueue_.DeQue<float>();
            const uint32_t srcShape1[] = {static_cast<uint32_t>(chunkSize_), static_cast<uint32_t>(1)};
            const uint32_t srcShape2[] = {static_cast<uint32_t>(1), static_cast<uint32_t>(chunkSize_)};
            const uint32_t dstShape[] = {static_cast<uint32_t>(chunkSize_), static_cast<uint32_t>(chunkSize_)};
            // tmpBuffer1_: broadcast along columns, so row i contains G_i.
            // tmpBuffer2_: broadcast along rows, so column j contains G_j.
            Broadcast<float, BROADCAST_AXIS, 1>(tmpBuffer1_, g_cum, dstShape, srcShape1);
            Broadcast<float, BROADCAST_AXIS, 0>(tmpBuffer2_, g_cum, dstShape, srcShape2);
            PipeBarrier<PIPE_V>();
            // [L,C] FP32: delta[i,j] = G_i - G_j.  Multiplying by mask
            // before Exp makes masked entries harmless; the final mask below
            // still explicitly zeros the upper triangle.
            Sub(tmpBuffer1_, tmpBuffer1_, tmpBuffer2_, curChunkSize_ * chunkSize_);
            PipeBarrier<PIPE_V>();
            Mul(tmpBuffer1_, tmpBuffer1_, maskBuffer_, curChunkSize_ * chunkSize_);
            PipeBarrier<PIPE_V>();
            // [L,C] FP32 decay = exp(G_i-G_j).
            Exp<float, 0, true>(tmpBuffer1_, tmpBuffer1_, curChunkSize_ * chunkSize_);
            PipeBarrier<PIPE_V>();
            inQueue_.FreeTensor(g_cum);
        } else {
            // No optional gate means G_i-G_j=0, hence decay=1.
            Duplicate(tmpBuffer1_, static_cast<float>(1.0f), curChunkSize_ * chunkSize_);
            PipeBarrier<PIPE_V>();
        }

        // qkt is Stage1's QK^T result.  Full chunks are [C,C] in GM; a tail
        // chunk is a compact [L,L] payload in its reserved C*C slot.  The
        // copy helper puts either form into a fixed [L,C] UB row stride.
        AlignedCopyIn(sTP_->qkt[nvId * Sp_ * chunkSize_ + chunkPos * chunkSize_], curChunkSize_, curChunkSize_);
        auto qkt = inQueue_.DeQue<bfloat16_t>();
        auto scale_qkt = outQueue_.AllocTensor<bfloat16_t>();
        // Actual precision path:
        //   qkt BF16 -> FP32 -> *scale -> *decay -> *trilMask -> BF16.
        // scale_qkt is the causal weight matrix A consumed by AIC.
        Cast(tmpBuffer2_, qkt, RoundMode::CAST_NONE, curChunkSize_ * chunkSize_);
        Muls(tmpBuffer2_, tmpBuffer2_, sTP_->scale, curChunkSize_ * chunkSize_);
        Mul(tmpBuffer2_, tmpBuffer2_, tmpBuffer1_, curChunkSize_ * chunkSize_);
        // Keep only j<=i.  This mask includes the diagonal because the
        // current token's Z must contribute to its own output.
        Mul(tmpBuffer2_, tmpBuffer2_, maskBuffer_, curChunkSize_ * chunkSize_);
        Cast(scale_qkt, tmpBuffer2_, RoundMode::CAST_RINT, curChunkSize_ * chunkSize_);
        outQueue_.EnQue(scale_qkt);
        // Write compact [L,L] BF16 back to the per-core GM tile.  AIC reads
        // this tile with M=L, K=L immediately after the 0x4 handshake.
        AlignedCopyOut(outGM, curChunkSize_, curChunkSize_);
        inQueue_.FreeTensor(qkt);
    }

    __aicore__ inline void AICProcess(GlobalTensor<bfloat16_t> tmpGM, GlobalTensor<bfloat16_t> vInner,
                                      GlobalTensor<bfloat16_t> attnInter)
    {
        // A [L,L] BF16 @ Z [L,Dv] BF16 -> [L,Dv] BF16.
        // attnInter points into the final TND output with row stride Nv*Dv;
        // Stage2 has already placed O_inter there.  The final argument 1 to
        // IterateAll requests accumulate mode, so this operation is:
        //     attnOut <- attnOut + A @ Z.
        sTP_->mm3->SetOrgShape(curChunkSize_, Dv_, curChunkSize_, curChunkSize_, Nv_ * Dv_); // MNK
        sTP_->mm3->SetSingleShape(curChunkSize_, Dv_, curChunkSize_);                        // SingleCoreMNK
        sTP_->mm3->SetTensorA(tmpGM);
        sTP_->mm3->SetTensorB(vInner);
        sTP_->mm3->IterateAll(attnInter, 1);
        sTP_->mm3->End();
    }

    template <typename inType>
    __aicore__ inline void AlignedCopyIn(GlobalTensor<inType> tmpGM, int32_t row, int32_t col)
    {
        LocalTensor<inType> inLocal = inQueue_.AllocTensor<inType>();
        // Copy a compact row-major GM matrix into a fixed C-column UB view.
        // ``row`` is L and ``col`` is L for both full and tail chunks.  The
        // source rows are compact; dstStride inserts the padding columns up
        // to the C-wide calculation view.  DataCopyPad also handles the
        // 32-byte transfer alignment required by the hardware.
        int paddingCol = Ceil(col, BLOCK_SIZE / sizeof(inType)) * (BLOCK_SIZE / sizeof(inType));
        DataCopyExtParams inParams{static_cast<uint16_t>(row), static_cast<uint32_t>(col * sizeof(inType)),
                                   static_cast<uint32_t>(0),
                                   static_cast<uint32_t>((chunkSize_ - paddingCol) * sizeof(inType) / BLOCK_SIZE), 0};
        DataCopyPadExtParams<inType> copyPadParams{false, 0, 0, 0};
        DataCopyPad(inLocal, tmpGM, inParams, copyPadParams);
        inQueue_.EnQue(inLocal);
    }

    template <typename outType>
    __aicore__ inline void AlignedCopyOut(GlobalTensor<outType> tmpGM, int32_t row, int32_t col)
    {
        auto outLocal = outQueue_.DeQue<outType>();
        // Reverse AlignedCopyIn: read the first L columns of each C-stride
        // UB row and write a compact [L,L] payload to GM.  The AIC matmul is
        // configured with M=L and K=L, so it never consumes the padding.
        int paddingCol = Ceil(col, BLOCK_SIZE / sizeof(outType)) * (BLOCK_SIZE / sizeof(outType));
        DataCopyExtParams copyParams;
        copyParams.blockCount = static_cast<uint16_t>(row);
        copyParams.blockLen = static_cast<uint32_t>(col * sizeof(outType));
        copyParams.srcStride = static_cast<uint32_t>((chunkSize_ - paddingCol) * sizeof(outType) / BLOCK_SIZE);
        copyParams.dstStride = static_cast<uint32_t>((0) * sizeof(outType));
        DataCopyPad(tmpGM, outLocal, copyParams);
        outQueue_.FreeTensor(outLocal);
    }

private:
    StageThreeParams *sTP_;
    TPipe *pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM_ONE> inQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM_ONE> outQueue_;
    TBuf<TPosition::VECCALC> tmpBuff_;
    GlobalTensor<bfloat16_t> tmpGM_;
    LocalTensor<float> tmpBuffer1_;
    LocalTensor<float> tmpBuffer2_;
    LocalTensor<float> maskBuffer_;
    int64_t seqLength_;
    int64_t Sp_;
    int64_t Nv_;
    int64_t Nk_;
    int64_t Dv_;
    int64_t Dk_;
    int64_t paddedDv_;
    int32_t chunkNum_;
    int32_t coreNum_;
    int32_t curChunkSize_;
    int32_t chunkSize_;
    int32_t coreId_;
};

} // namespace ChunkGatedDeltaRule
#endif // CHUNK_GATED_DELTA_RULE_STAGE3_H
