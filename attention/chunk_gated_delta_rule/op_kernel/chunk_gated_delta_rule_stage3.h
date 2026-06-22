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
#include "chunk_gated_delta_rule_tiling_data.h"

namespace ChunkGatedDeltaRule {
using namespace AscendC;
using namespace matmul;

constexpr uint8_t STAGE3_AIC_AIV_SYNC_MODE = 0x2;
constexpr uint32_t STAGE3_TMP_GM_SLOT_COUNT = 4;
constexpr uint16_t STAGE3_AIV_READY_AIC_FLAG_BASE = 0x4;
constexpr uint16_t STAGE3_AIC_DONE_AIV_FLAG_BASE = 0x8;

using aT3 = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t, true>;
using bT3 = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t, true>;
using cT3 = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t>;
using StageThreeMT = matmul::MatmulImpl<aT3, bT3, cT3>;

struct StageThreeParams {
    GlobalTensor<bfloat16_t> qkt;        // (Nv, Sp, Dk)
    GlobalTensor<float> gCumExp;         // (Nv, Sp)
    GlobalTensor<bfloat16_t> vInner;
    GlobalTensor<float> maskTensor;
    GM_ADDR ws;
    GlobalTensor<bfloat16_t> attnOut;
    StageThreeMT *mm3;
    TPipe *pipe;
    ChunkGroup *cg;
    float scale;
    int64_t Nv;
    int64_t Nk;
    int64_t Dv;
    int64_t Dk;
    bool gOptional;
};

class Stage3 {
public:
    __aicore__ inline void Init(StageThreeParams *initParams, int32_t coreNum)
    {
        coreId_ = GetBlockIdx();
        sTP_ = initParams;
        pipe_ = sTP_->pipe;
        chunkSize_ = sTP_->cg->chunkSize;
        seqLength_ = sTP_->cg->length;
        Sp_ = (seqLength_ + chunkSize_ - 1) / chunkSize_  * chunkSize_;
        chunkNum_ = (seqLength_ + chunkSize_ - 1) / chunkSize_ ;
        coreNum_ = coreNum;
        Nv_ = sTP_->Nv;
        Nk_ = sTP_->Nk;
        Dv_ = sTP_->Dv;
        Dk_ = sTP_->Dk;
        paddedDv_ = Ceil(Dv_, BLOCK_SIZE / sizeof(bfloat16_t)) * (BLOCK_SIZE / sizeof(bfloat16_t));
        gOptional_ = sTP_->gOptional;
        subBlockIdx_ = GetSubBlockIdx();
        halfChunkSize_ = chunkSize_ / TASK_RATIO;
        subOffset_ = subBlockIdx_ * halfChunkSize_;
        uint64_t workSpaceOffset = 0;
        tmpGM_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(
            initParams->ws + workSpaceOffset + coreNum_ * chunkSize_ * chunkSize_ * sizeof(float)));
        if ASCEND_IS_AIC {
            return;
        }
        coreId_ /= AIC_AIV_1_1;
        if (subBlockIdx_ != 0) {
            return;
        }
        uint64_t subBlockElementNum = static_cast<uint64_t>(halfChunkSize_) *
                                      AscendC::Std::max((int64_t)chunkSize_, paddedDv_) * sizeof(bfloat16_t);
        pipe_->InitBuffer(inQueue_, BUFFER_NUM_ONE, subBlockElementNum);
        pipe_->InitBuffer(outQueue_, BUFFER_NUM_ONE, halfChunkSize_ * chunkSize_ * sizeof(bfloat16_t));
        pipe_->InitBuffer(tmpBuff_, (STAGE3_BUFFER_COUNT * halfChunkSize_ * chunkSize_ * sizeof(float)));
        uint64_t buffOffset = 0;
        uint64_t tmpOffset = halfChunkSize_ * chunkSize_;
        tmpBuffer1_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(tmpOffset), buffOffset);
        buffOffset += tmpOffset * sizeof(float);
        tmpBuffer2_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(tmpOffset), buffOffset);
        buffOffset += tmpOffset * sizeof(float);
        maskBuffer_ = tmpBuff_.GetWithOffset<float>(static_cast<uint32_t>(tmpOffset), buffOffset);
        LoadMaskToBuffer(subOffset_);
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIV {
            if (subBlockIdx_ != 0) {
                return;
            }
        }
        int64_t totalChunks = Nv_ * chunkNum_;  // Nv Nc 融合
        int64_t chunksPerCore = (totalChunks + coreNum_ - 1) / coreNum_;
        int64_t lastChunkSize = seqLength_ % chunkSize_ == 0 ? chunkSize_ : seqLength_ % chunkSize_;
        int64_t startChunk = coreId_ * chunksPerCore;
        int64_t endChunk = startChunk + chunksPerCore > totalChunks ? totalChunks : startChunk + chunksPerCore;
        int64_t localIdx = 0;
        for (int64_t idx = startChunk; idx < endChunk; idx++) {
            int64_t nvId = idx / chunkNum_;
            int64_t chunkId = idx % chunkNum_;
            int64_t chunkPos = chunkId * chunkSize_;                                 // 当前chunk起始位置
            curChunkSize_ = (chunkId == chunkNum_ - 1) ? lastChunkSize : chunkSize_; // 尾块
            uint32_t slotIdx = static_cast<uint32_t>(localIdx % STAGE3_TMP_GM_SLOT_COUNT);
            uint64_t slotOffset = GetTmpGmSlotOffset(slotIdx);
            if ASCEND_IS_AIV {
                if (localIdx >= STAGE3_TMP_GM_SLOT_COUNT) {
                    CrossCoreWaitFlag<STAGE3_AIC_AIV_SYNC_MODE, PIPE_MTE3>(GetAicDoneAivFlag(slotIdx));
                }
                CalFullMaskedQKT(tmpGM_[slotOffset + coreId_ * chunkSize_ * chunkSize_], nvId, chunkPos);
                CrossCoreSetFlag<STAGE3_AIC_AIV_SYNC_MODE, PIPE_MTE3>(GetAivReadyAicFlag(slotIdx));
            }

            if ASCEND_IS_AIC {
                CrossCoreWaitFlag<STAGE3_AIC_AIV_SYNC_MODE, PIPE_FIX>(GetAivReadyAicFlag(slotIdx));
                AICProcess(tmpGM_[slotOffset + coreId_ * chunkSize_ * chunkSize_],
                           sTP_->vInner[nvId * Sp_ * Dv_ + chunkPos * Dv_],
                           sTP_->attnOut[nvId * Dv_ + chunkPos * Nv_ * Dv_]);
                CrossCoreSetFlag<STAGE3_AIC_AIV_SYNC_MODE, PIPE_FIX>(GetAicDoneAivFlag(slotIdx));
            }
            localIdx++;
        }
        if ASCEND_IS_AIV {
            int64_t drainCount = localIdx < static_cast<int64_t>(STAGE3_TMP_GM_SLOT_COUNT) ?
                                 localIdx : static_cast<int64_t>(STAGE3_TMP_GM_SLOT_COUNT);
            for (int64_t drainIdx = localIdx - drainCount; drainIdx < localIdx; drainIdx++) {
                uint32_t slotIdx = static_cast<uint32_t>(drainIdx % STAGE3_TMP_GM_SLOT_COUNT);
                CrossCoreWaitFlag<STAGE3_AIC_AIV_SYNC_MODE, PIPE_MTE3>(GetAicDoneAivFlag(slotIdx));
            }
        }
    }

    __aicore__ inline void CalFullMaskedQKT(GlobalTensor<bfloat16_t> outGM, int nvId, int chunkPos)
    {
        for (int32_t rowBlock = 0; rowBlock < TASK_RATIO; rowBlock++) {
            subOffset_ = rowBlock * halfChunkSize_;
            LoadMaskToBuffer(subOffset_);
            SetSubValidRows();
            CalMaskedQKT(outGM[subOffset_ * curChunkSize_], nvId, chunkPos);
        }
    }

    __aicore__ inline uint64_t GetTmpGmSlotOffset(uint32_t slotIdx)
    {
        return static_cast<uint64_t>(slotIdx) * coreNum_ * chunkSize_ * chunkSize_;
    }

    __aicore__ inline uint16_t GetAivReadyAicFlag(uint32_t slotIdx)
    {
        return STAGE3_AIV_READY_AIC_FLAG_BASE + static_cast<uint16_t>(slotIdx);
    }

    __aicore__ inline uint16_t GetAicDoneAivFlag(uint32_t slotIdx)
    {
        return STAGE3_AIC_DONE_AIV_FLAG_BASE + static_cast<uint16_t>(slotIdx);
    }

    __aicore__ inline void LoadMaskToBuffer(int32_t rowOffset)
    {
        if (chunkSize_ % BLOCK_FLOAT_NUM == 0) {
            DataCopy(maskBuffer_, sTP_->maskTensor[rowOffset * chunkSize_],
                     static_cast<uint32_t>(halfChunkSize_ * chunkSize_));
            int32_t eventID = static_cast<int32_t>(pipe_->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eventID);
            WaitFlag<HardEvent::MTE2_V>(eventID);
            return;
        }
        DataCopyExtParams inParams{static_cast<uint16_t>(halfChunkSize_),
                                   static_cast<uint32_t>(chunkSize_ * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> copyPadParams{false, 0, 0, 0};
        DataCopyPad(maskBuffer_, sTP_->maskTensor[rowOffset * chunkSize_], inParams, copyPadParams);
        int32_t eventID = static_cast<int32_t>(pipe_->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventID);
        WaitFlag<HardEvent::MTE2_V>(eventID);
    }

    __aicore__ inline void SetSubValidRows()
    {
        if (curChunkSize_ <= subOffset_) {
            subValidRows_ = 0;
        } else {
            subValidRows_ = AscendC::Std::min(halfChunkSize_, curChunkSize_ - subOffset_);
        }
    }

    __aicore__ inline void CalMaskedQKT(GlobalTensor<bfloat16_t> outGM, int nvId, int chunkPos)
    {
        if (subValidRows_ == 0) {
            return;
        }
        // chunkSize 大小进行自动补齐
        if (gOptional_) {
            AlignedCopyIn(sTP_->gCumExp[nvId * Sp_ + chunkPos], 1, curChunkSize_);  // 自动补齐
            auto g_cum = inQueue_.DeQue<float>();
            const uint32_t srcShape1[] = {static_cast<uint32_t>(subValidRows_), static_cast<uint32_t>(1)};
            const uint32_t srcShape2[] = {static_cast<uint32_t>(1), static_cast<uint32_t>(chunkSize_)};
            const uint32_t dstShape[] = {static_cast<uint32_t>(subValidRows_),
                                         static_cast<uint32_t>(chunkSize_)};
            Broadcast<float, BROADCAST_AXIS, 1>(tmpBuffer1_, g_cum[subOffset_], dstShape, srcShape1);
            Broadcast<float, BROADCAST_AXIS, 0>(tmpBuffer2_, g_cum, dstShape, srcShape2);
            PipeBarrier<PIPE_V>();
            inQueue_.FreeTensor(g_cum);
            AlignedCopyIn(sTP_->qkt[nvId * Sp_ * chunkSize_ + chunkPos * chunkSize_ + subOffset_ * curChunkSize_],
                          subValidRows_, curChunkSize_);
            Sub(tmpBuffer1_, tmpBuffer1_, tmpBuffer2_, subValidRows_ * chunkSize_);
            PipeBarrier<PIPE_V>();
            Mul(tmpBuffer1_, tmpBuffer1_, maskBuffer_, subValidRows_ * chunkSize_);
            PipeBarrier<PIPE_V>();
            Exp<float, 0, true>(tmpBuffer1_, tmpBuffer1_, subValidRows_ * chunkSize_);
            PipeBarrier<PIPE_V>();
        } else {
            Duplicate(tmpBuffer1_, static_cast<float>(1.0f), subValidRows_ * chunkSize_);
            PipeBarrier<PIPE_V>();
            AlignedCopyIn(sTP_->qkt[nvId * Sp_ * chunkSize_ + chunkPos * chunkSize_ + subOffset_ * curChunkSize_],
                          subValidRows_, curChunkSize_);
        }
        Muls(tmpBuffer1_, tmpBuffer1_, sTP_->scale, subValidRows_ * chunkSize_);
        Mul(tmpBuffer1_, tmpBuffer1_, maskBuffer_, subValidRows_ * chunkSize_);

        // qkt
        auto qkt = inQueue_.DeQue<bfloat16_t>();
        auto scale_qkt = outQueue_.AllocTensor<bfloat16_t>();
        Cast(tmpBuffer2_, qkt, RoundMode::CAST_NONE, subValidRows_ * chunkSize_);
        Mul(tmpBuffer2_, tmpBuffer2_, tmpBuffer1_, subValidRows_ * chunkSize_);
        Cast(scale_qkt, tmpBuffer2_, RoundMode::CAST_RINT, subValidRows_ * chunkSize_);
        outQueue_.EnQue(scale_qkt);
        AlignedCopyOut(outGM, subValidRows_, curChunkSize_);
        inQueue_.FreeTensor(qkt);
    }

    __aicore__ inline void AICProcess(GlobalTensor<bfloat16_t>tmpGM,
                                      GlobalTensor<bfloat16_t>vInner,
                                      GlobalTensor<bfloat16_t>attnInter)
    {
        // masked_qkt @ v_inner
        sTP_->mm3->SetOrgShape(curChunkSize_, Dv_, curChunkSize_, curChunkSize_, Nv_ * Dv_);    // MNK
        sTP_->mm3->SetSingleShape(curChunkSize_, Dv_, curChunkSize_); // SingleCoreMNK
        sTP_->mm3->SetTensorA(tmpGM);
        sTP_->mm3->SetTensorB(vInner);
        sTP_->mm3->IterateAll(attnInter, 1);
        sTP_->mm3->End();
    }

    template <typename inType>
    __aicore__ inline void AlignedCopyIn(GlobalTensor<inType> tmpGM, int32_t row, int32_t col)
    {
        LocalTensor<inType> inLocal = inQueue_.AllocTensor<inType>();
        if (row == halfChunkSize_ && col == chunkSize_) {
            DataCopy(inLocal, tmpGM, static_cast<uint32_t>(row * chunkSize_));
            inQueue_.EnQue(inLocal);
            return;
        }
        // 非对齐拷入会自动对齐, 然后离散拷入UB
        int paddingCol = Ceil(col, BLOCK_SIZE / sizeof(inType)) * (BLOCK_SIZE / sizeof(inType));
        DataCopyExtParams inParams{static_cast<uint16_t>(row),
                                   static_cast<uint32_t>(col * sizeof(inType)),
                                   static_cast<uint32_t>(0),
                                   static_cast<uint32_t>((chunkSize_ - paddingCol) * sizeof(inType) / BLOCK_SIZE),
                                   0};
        DataCopyPadExtParams<inType> copyPadParams{false, 0, 0, 0};
        DataCopyPad(inLocal, tmpGM, inParams, copyPadParams);
        inQueue_.EnQue(inLocal);
    }

    template <typename outType>
    __aicore__ inline void AlignedCopyOut(GlobalTensor<outType> tmpGM, int32_t row, int32_t col)
    {
        auto outLocal = outQueue_.DeQue<outType>();
        if (row == halfChunkSize_ && col == chunkSize_) {
            DataCopy(tmpGM, outLocal, static_cast<uint32_t>(row * chunkSize_));
            outQueue_.FreeTensor(outLocal);
            return;
        }
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
    int32_t subValidRows_;
    int32_t halfChunkSize_;
    int32_t subBlockIdx_;
    int32_t subOffset_;
    int32_t chunkSize_;
    int32_t coreId_;
    bool gOptional_;
};

} // namespace ChunkGatedDeltaRule
#endif // CHUNK_GATED_DELTA_RULE_STAGE3_H
