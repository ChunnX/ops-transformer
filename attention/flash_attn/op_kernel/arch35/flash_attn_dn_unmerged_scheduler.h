/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLASH_ATTN_DN_UNMERGED_SCHEDULER_H_
#define FLASH_ATTN_DN_UNMERGED_SCHEDULER_H_
#include "../utils/flash_attn_type.h"
#include "flash_attn_tiling_data.h"
#include "memory_copy_arch35.h"
#include "../../../common/op_kernel/arch35/flash_attention_score_common_regbase_arch35.h"
#include "../../../common/op_kernel/arch_info.h"
#include "../../../common/op_kernel/vector_common.h"

namespace FlashAttnKernel {
using namespace optiling;
using ArchInfo::CV_RATIO;
template <typename FA_T, typename ConstInfo, typename RunInfo>
class DnUnmergedSchedulerHooks {
public:
    using INPUT_T = typename FA_T::inputType;
    using SEQLEN_T = uint32_t;
    static constexpr uint32_t mBaseSize = static_cast<uint32_t>(FA_T::mBaseSize);
    static constexpr uint32_t s2BaseSize = static_cast<uint32_t>(FA_T::s2BaseSize);
    static constexpr FA_LAYOUT LAYOUT_T = FA_T::qLayout;
    static constexpr FA_LAYOUT LAYOUT_KV = FA_T::kvLayout;
    static constexpr bool PAGE_ATTENTION = FA_T::pageAttention;
    static constexpr bool HAS_MASK = FA_T::hasMask;
    ConstInfo &constInfo_;
    SeqLensTool<LAYOUT_T, SEQLEN_T> &qSeqLensTool_;
    SeqLensTool<LAYOUT_KV, SEQLEN_T> &kvSeqLensTool_;
    __tiling_data_ptr__ FlashAttnNoQuantTilingArch35 *tilingData_;
    // 分核信息
    GlobalTensor<uint32_t> faMetaDataGm_;
    // metadata
    uint32_t sectionNum_;
    uint32_t metadata_aic_num_;
    uint32_t metadata_aiv_num_;
    // fa metadata(BN1_S1 不合轴分核: bn 轴 = b*n1, M 轴 = 纯 s1 的 mBaseSize 块序号)
    uint32_t bn1Start_;
    uint32_t bn1End_;
    uint32_t s1OStart_;
    uint32_t s1OEnd_;
    uint32_t s2OStart_;
    uint32_t s2OEnd_;
    uint32_t coreFirstTmpOutWsPos_;

    // schduler params
    uint64_t actSeqLensKv_ = 0;
    uint64_t actSeqLensQ_ = 0;
    uint32_t curS2Start_;
    uint32_t curS2End_ = 0;
    uint32_t prevBIdx_;
    uint32_t prevBn1Idx_;
    uint32_t prevS1Idx_;
    uint32_t mloop_ = 0;
    bool headS2Split_ = false;
    bool tailS2Split_ = false;
    uint32_t s2FirstToken_ = 0;

    uint32_t bn1Cur_ = 0;
    uint32_t s1Cur_ = 0;
    uint32_t s2Cur_ = 0;
    __aicore__ inline DnUnmergedSchedulerHooks(ConstInfo &info, SeqLensTool<LAYOUT_T, SEQLEN_T> &q,
                                               SeqLensTool<LAYOUT_KV, SEQLEN_T> &kv)
        : constInfo_(info),
          qSeqLensTool_(q),
          kvSeqLensTool_(kv)
    {}
    __aicore__ inline void Init(__gm__ uint8_t *cuSeqLensQ, __gm__ uint8_t *cuSeqLensKv, __gm__ uint8_t *seqUsedQ,
                                __gm__ uint8_t *seqUsedKv, __gm__ uint8_t *learnableSink, __gm__ uint8_t *metadata,
                                __tiling_data_ptr__ FlashAttnNoQuantTilingArch35 *tiling)
    {
        tilingData_ = tiling;
        InitConstInfo();
        sectionNum_ = ((__gm__ uint32_t *)metadata)[METADATA_HEADER_SECTION_NUM_INDEX];
        metadata_aic_num_ = ((__gm__ uint32_t *)metadata)[METADATA_HEADER_AIC_NUM_INDEX];
        metadata_aiv_num_ = ((__gm__ uint32_t *)metadata)[METADATA_HEADER_AIV_NUM_INDEX];
        constInfo_.enableFlashDecode = ((__gm__ uint32_t *)metadata)[METADATA_HEADER_HAS_FD_INDEX] != 0;
        constInfo_.learnableSinkFlag = learnableSink != nullptr;
        faMetaDataGm_.SetGlobalBuffer((__gm__ uint32_t *)(metadata + METADATA_HEADER_OFFSET),
                                      sectionNum_ * metadata_aic_num_ * METADATA_STRIDE);
        qSeqLensTool_.Init(cuSeqLensQ, constInfo_.cuSeqLensQSize, seqUsedQ, constInfo_.seqUsedQSize, constInfo_.s1Size);
        kvSeqLensTool_.Init(cuSeqLensKv, constInfo_.cuSeqLensKVSize, seqUsedKv, constInfo_.seqUsedKvSize,
                            constInfo_.s2Size);
    }
    __aicore__ inline void BeginSection(uint32_t section)
    {
        GetFASectionInfo(section);
        mloop_ = 0;
        headS2Split_ = false;
        tailS2Split_ = false;
        bn1Cur_ = bn1Start_;
        s1Cur_ = s1OStart_ * mBaseSize;
        s2Cur_ = s2OStart_;
        prevBn1Idx_ = bn1Cur_;
        prevS1Idx_ = s1Cur_;
    }
    __aicore__ inline bool NextTask(uint32_t loop, RunInfo &task)
    {
        while (ShouldDispatchTask(bn1Cur_, s1Cur_, s2Cur_)) {
            TASK_DEAL_MODE mode = GetTaskDealMode(bn1Cur_, s1Cur_, s2Cur_);
            if (mode == TASK_DEAL_MODE::CREATE_TASK) {
                CalcParams(loop, bn1Cur_, s1Cur_, s2Cur_, task);
                task.isValid = true;
                UpdateAxisInfo(mode, bn1Cur_, s1Cur_, s2Cur_);
                return true;
            }
            UpdateAxisInfo(mode, bn1Cur_, s1Cur_, s2Cur_);
        }
        return false;
    }
    __aicore__ inline void InitConstInfo()
    {
        if ASCEND_IS_AIC {
            constInfo_.aicIdx = GetBlockIdx();
        } else {
            constInfo_.aivIdx = GetBlockIdx();
            constInfo_.aicIdx = GetBlockIdx() / GetSubBlockNum();
            constInfo_.subBlockIdx = GetSubBlockIdx();
        }
        constInfo_.bSize = tilingData_->flashAttnBaseParams.bSize;
        constInfo_.t1Size = tilingData_->flashAttnBaseParams.t1Size;
        constInfo_.t2Size = tilingData_->flashAttnBaseParams.t2Size;
        constInfo_.n1Size = tilingData_->flashAttnBaseParams.n1Size;
        constInfo_.n2Size = tilingData_->flashAttnBaseParams.n2Size;
        constInfo_.gSize = tilingData_->flashAttnBaseParams.gSize;
        constInfo_.s1Size = tilingData_->flashAttnBaseParams.s1Size;
        constInfo_.s2Size = tilingData_->flashAttnBaseParams.s2Size;
        constInfo_.dSize = tilingData_->flashAttnBaseParams.dSize;
        constInfo_.dSizeV = tilingData_->flashAttnBaseParams.dSizeV;
        constInfo_.cuSeqLensQSize = tilingData_->flashAttnBaseParams.cuSeqLensQSize;
        constInfo_.cuSeqLensKVSize = tilingData_->flashAttnBaseParams.cuSeqLensKVSize;
        constInfo_.seqUsedQSize = tilingData_->flashAttnBaseParams.seqUsedQSize;
        constInfo_.seqUsedKvSize = tilingData_->flashAttnBaseParams.seqUsedKvSize;
        constInfo_.scaleValue = static_cast<float>(tilingData_->flashAttnBaseParams.scaleValue);
        constInfo_.coreNum = tilingData_->flashAttnBaseParams.coreNum;
        constInfo_.outputLayout = static_cast<FA_LAYOUT>(tilingData_->flashAttnBaseParams.outputLayout);
        constInfo_.needInitOutput = tilingData_->flashAttnBaseParams.needInitOutput;

        constInfo_.sparseMode = tilingData_->flashAttnAttenMaskParams.sparseMode;
        constInfo_.preTokens = tilingData_->flashAttnAttenMaskParams.winLefts;
        constInfo_.nextTokens = tilingData_->flashAttnAttenMaskParams.winRights;
        constInfo_.attenMaskBatch = tilingData_->flashAttnAttenMaskParams.attenMaskBatch;
        constInfo_.attenMaskS1Size = tilingData_->flashAttnAttenMaskParams.attenMaskS1Size;
        constInfo_.attenMaskS2Size = tilingData_->flashAttnAttenMaskParams.attenMaskS2Size;
        constInfo_.isExistRowInvalid = tilingData_->flashAttnAttenMaskParams.isExistRowInvalid;

        constInfo_.accumOutSize = tilingData_->flashAttnWorkspaceParams.accumOutSize;
        constInfo_.logSumExpSize = tilingData_->flashAttnWorkspaceParams.logSumExpSize;

        // pageAttention
        if constexpr (PAGE_ATTENTION) {
            constInfo_.maxBlockNumPerBatch = tilingData_->flashAttnPageAttentionParams.maxBlockNumPerBatch;
            constInfo_.blockSize = tilingData_->flashAttnPageAttentionParams.blockSize;
            constInfo_.paLayoutType = tilingData_->flashAttnPageAttentionParams.paLayoutType;
            // kvcache非连续场景的strides, 透传给offsetCalculator
            constInfo_.keyBnStride = tilingData_->flashAttnBaseParams.keyBnStride;
            constInfo_.keyN2Stride = tilingData_->flashAttnBaseParams.keyN2Stride;
            constInfo_.valueBnStride = tilingData_->flashAttnBaseParams.valueBnStride;
            constInfo_.valueN2Stride = tilingData_->flashAttnBaseParams.valueN2Stride;
        }
        // LSE
        constInfo_.isSoftmaxLseEnable = tilingData_->flashAttnBaseParams.isSoftMaxLseEnable;

        constInfo_.dBasicBlock = BaseApi::Align64Func((uint16_t)constInfo_.dSizeV);
    }

    __aicore__ inline uint32_t GetFAMetaDataIndex(uint32_t coreIdx, uint32_t metaIdx, uint32_t sectionIdx)
    {
        // AICPU metadata format: 16 fields per AIC core, 0-indexed (no leading CORE_ENABLE).
        // Kernel field constants ( FLASH_ATTN_BN2_START_INDEX=1, etc.) are 1-based, so subtract 1.
        return METADATA_STRIDE * metadata_aic_num_ * sectionIdx + METADATA_STRIDE * coreIdx + metaIdx;
    }

    __aicore__ inline bool ShouldDispatchTask(uint32_t bn1Cur, uint32_t s1Cur, uint32_t s2Cur)
    {
        if (bn1Cur != bn1End_) {
            return bn1Cur < bn1End_;
        }
        uint32_t s1EndCoord = s1OEnd_ * mBaseSize;
        if (s1Cur != s1EndCoord) {
            return s1Cur < s1EndCoord;
        }
        return s2Cur < s2OEnd_;
    }

    __aicore__ inline uint32_t GetS1Step(uint32_t s1Cur)
    {
        uint32_t step = mBaseSize;
        if (s1Cur + step > actSeqLensQ_) {
            step = static_cast<uint32_t>(actSeqLensQ_) - s1Cur;
        }
        return step;
    }

    __aicore__ inline TASK_DEAL_MODE GetTaskDealMode(uint32_t bn1Cur, uint32_t s1Cur, uint32_t s2Cur)
    {
        bool isFirstTask = (bn1Cur == bn1Start_) && (s1Cur == s1OStart_ * mBaseSize) && (s2Cur == s2OStart_);
        uint32_t bIdx = bn1Cur / constInfo_.n1Size;
        if (isFirstTask || prevBIdx_ != bIdx) {
            prevBIdx_ = bIdx;
            actSeqLensQ_ = qSeqLensTool_.GetActualSeqLength(bIdx);
            actSeqLensKv_ = kvSeqLensTool_.GetActualSeqLength(bIdx);
        }
        uint64_t s2LoopTimes = (actSeqLensKv_ + s2BaseSize - 1) / s2BaseSize;
        uint64_t s1LoopTimes = (actSeqLensQ_ + mBaseSize - 1) / mBaseSize;
        if (s2LoopTimes == 0 || s1LoopTimes == 0) {
            if (s1Cur == 0 && s2Cur == 0) {
                return TASK_DEAL_MODE::DEAL_ZERO;
            }
            return TASK_DEAL_MODE::SKIP_ZERO;
        }
        // 计算每一行的起止点，只有当换行时（bn1Cur、s1Cur更新）才需要重新计算
        if (isFirstTask || bn1Cur != prevBn1Idx_ || s1Cur != prevS1Idx_) {
            if constexpr (!HAS_MASK) {
                CalcCurS2StartEndNoSparse(bn1Cur, s1Cur);
            } else {
                CalcCurS2StartEndWithSparse(bn1Cur, s1Cur);
            }
            prevBn1Idx_ = bn1Cur;
            prevS1Idx_ = s1Cur;
        }
        // S2有效块区间为[curS2Start_, curS2End_), s2Cur尚未到达有效区间且该行有有效块,
        // 需快进到curS2Start继续计算, 不能跳行 (BAND等sparse模式curS2Start常>0)
        if (s2Cur < curS2Start_ && curS2Start_ < curS2End_) {
            return TASK_DEAL_MODE::NOT_START;
        }
        // 该行无有效块(curS2Start_>=curS2End_)或s2Cur已越过有效区间, 跳过当前行
        if (s2Cur < curS2Start_ || s2Cur >= curS2End_) {
            return TASK_DEAL_MODE::SKIP_REMAINING_S2;
        }

        if (s2Cur == curS2Start_) {
            mloop_++;
        }

        return TASK_DEAL_MODE::CREATE_TASK;
    }

    __aicore__ inline void GetPreNextTokenLeftUp(int64_t actSeqLensQ, int64_t actSeqLensKv, int64_t &preTokenLeftUp,
                                                 int64_t &nextTokenLeftUp)
    {
        preTokenLeftUp = constInfo_.preTokens;
        nextTokenLeftUp = constInfo_.nextTokens;
        fa_base_vector::GetSafeActToken(actSeqLensQ, actSeqLensKv, preTokenLeftUp, nextTokenLeftUp,
                                        constInfo_.sparseMode);

        if (constInfo_.sparseMode == fa_base_vector::BAND) {
            preTokenLeftUp = static_cast<int64_t>(actSeqLensQ) - static_cast<int64_t>(actSeqLensKv) + preTokenLeftUp;
        }

        if (constInfo_.sparseMode == fa_base_vector::RIGHT_DOWN_CAUSAL) {
            nextTokenLeftUp = static_cast<int64_t>(actSeqLensKv) - static_cast<int64_t>(actSeqLensQ);
        } else if (constInfo_.sparseMode == fa_base_vector::BAND) {
            nextTokenLeftUp = static_cast<int64_t>(actSeqLensKv) - static_cast<int64_t>(actSeqLensQ) + nextTokenLeftUp;
        }
    }

    __aicore__ inline void CalcCurS2StartEndNoSparse(uint32_t bn1Cur, uint32_t s1Cur)
    {
        // s1Cur 为纯 s1 坐标(元素粒度)
        s2FirstToken_ = 0;
        curS2Start_ = 0U;
        curS2End_ = (static_cast<uint32_t>(actSeqLensKv_) + s2BaseSize - 1) / s2BaseSize;

        if ((bn1Cur == bn1Start_) && (s1Cur == s1OStart_ * mBaseSize)) {
            headS2Split_ = s2OStart_ != 0U;
            curS2Start_ = s2OStart_;
        }

        if ((bn1Cur == bn1End_) && (s1Cur == s1OEnd_ * mBaseSize)) {
            tailS2Split_ = s2OEnd_ != 0U;
            curS2End_ = s2OEnd_;
        }
    }

    __aicore__ inline void CalcCurS2StartEndWithSparse(uint32_t bn1Cur, uint32_t s1Cur)
    {
        s2FirstToken_ = 0;
        // 1. Calc preTokenLeftUp, nextTokenLeftUp
        int64_t preTokenLeftUp = 0;
        int64_t nextTokenLeftUp = 0;
        GetPreNextTokenLeftUp(actSeqLensQ_, actSeqLensKv_, preTokenLeftUp, nextTokenLeftUp);

        // 2. calc index of s2FirstToken, s2LastToken by s1 token range(纯 s1, 不合轴无 gs1 概念)
        int64_t s1FirstToken = static_cast<int64_t>(s1Cur);
        int64_t s1LastToken = AttentionCommon::Min(s1FirstToken + static_cast<int64_t>(GetS1Step(s1Cur)),
                                                   static_cast<int64_t>(actSeqLensQ_)) -
                              1;

        // 3. trans index of token to index of block
        uint32_t s2StartWithSparse = 0U;
        uint32_t s2EndWithSparse = 0U;
        int64_t s2FirstToken = s1FirstToken - preTokenLeftUp;
        int64_t s2LastToken = s1LastToken + nextTokenLeftUp;
        // no valid token
        if (s2FirstToken >= static_cast<int64_t>(actSeqLensKv_) || s2LastToken < 0 || s2LastToken < s2FirstToken) {
            curS2Start_ = 0U;
            curS2End_ = 0U;
            s2FirstToken_ = s2FirstToken;
            return;
        }
        // get valid range
        s2FirstToken = ClipSInnerToken(s2FirstToken, 0, static_cast<int64_t>(actSeqLensKv_ - 1));
        s2LastToken = ClipSInnerToken(s2LastToken, 0, static_cast<int64_t>(actSeqLensKv_ - 1));
        s2FirstToken_ = s2FirstToken;

        s2StartWithSparse = static_cast<uint32_t>(s2FirstToken) / s2BaseSize;
        s2EndWithSparse = static_cast<uint32_t>(s2LastToken) / s2BaseSize + 1U;

        // 4. Calc curS2Start_, curS2End_
        curS2Start_ = s2StartWithSparse;
        curS2End_ = s2EndWithSparse;

        if (bn1Cur == bn1Start_ && s1Cur == s1OStart_ * mBaseSize) { // first line
            headS2Split_ = s2OStart_ > s2StartWithSparse ? true : false;
            curS2Start_ = AttentionCommon::Max(s2StartWithSparse, s2OStart_);
        }
        if (bn1Cur == bn1End_ && s1Cur == s1OEnd_ * mBaseSize) { // last line
            tailS2Split_ = s2OEnd_ > 0U ? true : false;
            curS2End_ = s2OEnd_ > 0U ? AttentionCommon::Min(s2EndWithSparse, s2OEnd_) : s2EndWithSparse;
        }
        return;
    }

    __aicore__ inline void CalcParams(uint64_t loop, uint32_t bn1Cur, uint32_t s1Cur, uint32_t s2Cur, RunInfo &info)
    {
        info.loop = loop;
        info.mloop = mloop_;
        info.isFirstFdBlock = (s2Cur == (s2FirstToken_ / s2BaseSize));
        // 不合轴: bn1 = b*n1, 任务为单个 query 头(n1)的纯 s1 块; K/V 按 n2 = n1/gSize 共享
        info.bIdx = bn1Cur / constInfo_.n1Size;
        info.n1Idx = bn1Cur % constInfo_.n1Size;
        info.n2Idx = info.n1Idx / constInfo_.gSize;
        info.gIdx = info.n1Idx % constInfo_.gSize;
        info.s1Idx = s1Cur;
        info.s2Idx = s2Cur * s2BaseSize;
        info.actS1Size = actSeqLensQ_;
        info.actS2Size = actSeqLensKv_;

        info.actMSize = GetS1Step(s1Cur);
        info.actSingleLoopS2Size = s2BaseSize;
        if (((s2Cur + 1) * s2BaseSize) > info.actS2Size) {
            info.actSingleLoopS2Size = info.actS2Size - s2Cur * s2BaseSize;
        }
        info.actSingleLoopS2SizeAlign = AttentionCommon::Align(
            (uint32_t)info.actSingleLoopS2Size, (uint32_t)(AttentionCommon::BYTE_BLOCK / sizeof(INPUT_T)));

        GetPreNextTokenLeftUp(actSeqLensQ_, actSeqLensKv_, info.preTokensLeftUp, info.nextTokensLeftUp);

        // 情况1: loop不等于0时, 第一个S2 inner循环就是第一个S2 outer循环, 即s2Cur=0
        // 情况2: loop=0时, 如果(bn1Start, s1OStart, s2Start)任务有效, 对于当前核, 为第一个S2 inner循环
        // 情况3: loop=0时, 如果(bn1Start, s1OStart, s2Start)任务无效,
        // 下一个有效任务一定是某个head的第一个S2外切块，s2Cur=0
        info.isFirstS2Loop = ((loop == 0) || (s2Cur == curS2Start_));
        info.isS2SplitCore = false;
        info.faTmpOutWsPos = coreFirstTmpOutWsPos_;
        info.isLastS2Loop = (s2Cur + 1 == curS2End_);
        info.actMSizeAlign32 = (info.actMSize + 31) >> 5 << 5;
        info.actVecMSize = info.actMSize <= 16 ? info.actMSize : (info.actMSizeAlign32 / CV_RATIO);
        info.vecMbaseIdx = 0;
        if (constInfo_.subBlockIdx == 1) {
            info.vecMbaseIdx = info.actVecMSize;
            info.actVecMSize = info.actMSize - info.actVecMSize;
        }

        if (bn1Start_ == bn1End_ && s1OStart_ == s1OEnd_) {
            // 所有任务属于同一个 (b,n1)
            info.isS2SplitCore = true;
        } else {
            if (headS2Split_ && (bn1Cur == bn1Start_) && (s1Cur == s1OStart_ * mBaseSize)) {
                // 当前任务属于第一个 (b,n1), 并且它的 S2 被切分了
                info.isS2SplitCore = true;
            } else if (tailS2Split_ && (bn1Cur == bn1End_) && (s1Cur == s1OEnd_ * mBaseSize)) {
                // 当前任务属于最后一个 (b,n1), 并且它的 S2 被切分了
                info.isS2SplitCore = true;
                info.faTmpOutWsPos = headS2Split_ ? (info.faTmpOutWsPos + 1) : info.faTmpOutWsPos;
            }
        }
    }

    __aicore__ inline void UpdateAxisInfo(TASK_DEAL_MODE taskDealMode, uint32_t &bn1Cur, uint32_t &s1Cur,
                                          uint32_t &s2Cur)
    {
        uint64_t s2LoopTimes = (actSeqLensKv_ + s2BaseSize - 1) / s2BaseSize;

        // 尚未到达有效区间, 快进s2Cur到curS2Start, 不跳行
        if (taskDealMode == TASK_DEAL_MODE::NOT_START) {
            s2Cur = curS2Start_;
            return;
        }
        if (taskDealMode != TASK_DEAL_MODE::SKIP_REMAINING_S2) {
            // 当前S2未处理完
            if (s2Cur + 1 < s2LoopTimes) {
                s2Cur++;
                return;
            }
        }
        // 当前 (b,n1) 未处理完
        s2Cur = 0;
        if (s1Cur < actSeqLensQ_) {
            s1Cur += GetS1Step(s1Cur);
            return;
        }
        // 当前 (b,n1) 已处理完
        s1Cur = 0;
        bn1Cur++;
    }

    __aicore__ inline void GetFASectionInfo(uint32_t sectionIdx)
    {
        bn1Start_ =
            faMetaDataGm_.GetValue(GetFAMetaDataIndex(constInfo_.aicIdx, FLASH_ATTN_BN1_START_INDEX, sectionIdx));
        s1OStart_ = faMetaDataGm_.GetValue(GetFAMetaDataIndex(constInfo_.aicIdx, FLASH_ATTN_M_START_INDEX, sectionIdx));
        s2OStart_ =
            faMetaDataGm_.GetValue(GetFAMetaDataIndex(constInfo_.aicIdx, FLASH_ATTN_S2_START_INDEX, sectionIdx));
        bn1End_ = faMetaDataGm_.GetValue(GetFAMetaDataIndex(constInfo_.aicIdx, FLASH_ATTN_BN1_END_INDEX, sectionIdx));
        s1OEnd_ = faMetaDataGm_.GetValue(GetFAMetaDataIndex(constInfo_.aicIdx, FLASH_ATTN_M_END_INDEX, sectionIdx));
        s2OEnd_ = faMetaDataGm_.GetValue(GetFAMetaDataIndex(constInfo_.aicIdx, FLASH_ATTN_S2_END_INDEX, sectionIdx));
        coreFirstTmpOutWsPos_ = faMetaDataGm_.GetValue(
            GetFAMetaDataIndex(constInfo_.aicIdx, FLASH_ATTN_FIRST_FD_DATA_WORKSPACE_IDX_INDEX, sectionIdx));
    }
};
} // namespace FlashAttnKernel
#endif
