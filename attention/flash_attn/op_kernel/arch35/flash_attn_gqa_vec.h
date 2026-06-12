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
 * \file flash_attn_gqa_vec.h
 * \brief Merged: flash_attn_block_vec_noquant_gqa (compute block) +
 *        flash_attn_kernel_base_gqa_vec (kernel orchestrator)
 */

#ifndef FLASH_ATTENTION_NOQUANT_GQA_KERNEL_VEC_H_
#define FLASH_ATTENTION_NOQUANT_GQA_KERNEL_VEC_H_

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#include "../../../common/op_kernel/fia_public_define.h"
#include "../../../common/op_kernel/arch35/flash_attention_score_common_regbase.h"

#include "kernel_operator_list_tensor_intf.h"
#include "../utils/flash_attn_common_def.h"
#include "../utils/flash_attn_vec_def.h"
#include "../utils/flash_attn_utils.h"
#include "../../../common/op_kernel/buffers_policy.h"
using namespace fa_base_matmul;
#include "../../../common/op_kernel/memory_copy_arch35.h"

#include "../utils/attenmask_gs1.h"
#include "adv_api/activation/softmax.h"
#include "../../../common/op_kernel/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz.h"
#include "../../../common/op_kernel/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz_dn.h"
#include "../../../common/op_kernel/arch35/vf/vf_flashupdate_new.h"
#include "../../../common/op_kernel/arch35/vf/vf_div_cast.h"
#include "../../../common/op_kernel/arch35/vf/vf_flash_decode.h"
#include "../../../common/op_kernel/vector_common.h"

using namespace AscendC;
using namespace FaVectorApi;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;
using namespace AttentionCommon;

#include "flash_attention_noquant_block_vec_flashdecode.h"
#include "flash_attn_tiling_data.h"

using namespace optiling;

namespace BaseApi {


template <
    typename INPUT_T, typename T, typename OUTPUT_T, typename VecFdBlockType,
    LayOutTypeEnum layout = LayOutTypeEnum::None,
    LayOutTypeEnum outLayout = LayOutTypeEnum::None,
    S1TemplateType s1TemplateType = S1TemplateType::Aligned128,
    S2TemplateType s2TemplateType = S2TemplateType::Aligned128,
    DTemplateType dTemplateType = DTemplateType::Aligned128,
    DTemplateType dVTemplateType = DTemplateType::Aligned128,
    bool hasAtten = false, bool hasDrop = false,
    uint8_t KvLayoutType = 0, bool useDn = false,
    bool bmm2Write2Ub = true, bool splitD = false>
class FlashAttentionNoQuantGqaKernelVec {
public:
    static constexpr uint32_t mBaseSize = (uint32_t)s1TemplateType;
    static constexpr uint32_t s2BaseSize = (uint32_t)s2TemplateType;
    static constexpr uint32_t vec1S2CopyLenDn = s2BaseSize >> 1;
    static constexpr uint32_t vec1HalfS1BaseSize = mBaseSize >> 1;
    static constexpr uint32_t vec1S2CopyCountDn = mBaseSize >> 5;
    static constexpr uint32_t vec1S2strideDn = s2BaseSize * 8;
    static constexpr uint32_t vec1ResOffsetDn = s2BaseSize * 32 + 64;
    static constexpr uint32_t vec1Srcstride = (mBaseSize >> 1) + 1;
    static constexpr uint32_t dTemplateAlign64 = Align64Func((uint16_t)dVTemplateType);
    static constexpr uint32_t DB = 2;
    static_assert(DB > 0 && (DB & (DB - 1)) == 0, "DB must be a power of two for bitmask indexing");
    static constexpr uint32_t initOutputEventId = 0U;
    static constexpr LAYOUT_Q MASK_LAYOUT =
        (layout == LayOutTypeEnum::LAYOUT_BSH || layout == LayOutTypeEnum::LAYOUT_TND ||
         layout == LayOutTypeEnum::LAYOUT_NTD || layout == LayOutTypeEnum::LAYOUT_SBH) ? LAYOUT_Q::SG : LAYOUT_Q::GS;
    static constexpr T BOOL_ATTEN_MASK_SCALAR_VALUE = -1000000000000.0;
    static constexpr uint32_t dVBaseSize = (uint32_t)dVTemplateType;

    static constexpr bool USE_DN = useDn;
    static constexpr bool BMM2_TOUB = bmm2Write2Ub;
    static constexpr bool HAS_MASK = hasAtten;

    static constexpr uint32_t PRELOAD_N = 2; // C1 C1 C1 C2
    // task ring buffer 至少需要 PRELOAD_N+1(=3) 个slot；为了用位掩码(loop & MASK)代替取模(loop % SIZE)，
    // 将容量向上取整到最近的2的幂(=4)。代价仅为多1个RunInfoXVec slot，换取消除内层循环的取模scalar bound。
    static constexpr uint32_t PRELOAD_TASK_CACHE_SIZE = 4;
    static constexpr uint32_t PRELOAD_TASK_CACHE_MASK = PRELOAD_TASK_CACHE_SIZE - 1;
    static_assert(PRELOAD_TASK_CACHE_SIZE >= PRELOAD_N + 1,
                  "PRELOAD_TASK_CACHE_SIZE must be at least PRELOAD_N + 1");
    static_assert((PRELOAD_TASK_CACHE_SIZE & PRELOAD_TASK_CACHE_MASK) == 0,
                  "PRELOAD_TASK_CACHE_SIZE must be a power of two for bitmask indexing");


    static constexpr bool PAGE_ATTENTION = (KvLayoutType > 0);

    static constexpr bool IS_TND_LAYOUT = (layout == LayOutTypeEnum::LAYOUT_TND || layout == LayOutTypeEnum::LAYOUT_NTD);
    static constexpr ActualSeqLensMode KV_PARSER_MODE = GetKvActSeqMode<layout, PAGE_ATTENTION>();

    using QSeqParserType =
        typename std::conditional<IS_TND_LAYOUT, ActualSeqLensParser<ActualSeqLensMode::ACCUM, int32_t, true>,
                                  ActualSeqLensParser<ActualSeqLensMode::BY_BATCH, int32_t>>::type;
    using KvSeqParserType = typename std::conditional<IS_TND_LAYOUT, ActualSeqLensParser<KV_PARSER_MODE, int32_t, true>,
                                                      ActualSeqLensParser<KV_PARSER_MODE, int32_t>>::type;

    using mm2ResPos = typename std::conditional<bmm2Write2Ub,
        Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>,
        Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_FORWARD>>::type;
    using attenMaskGmType = typename std::conditional<hasAtten, GlobalTensor<uint8_t>, int8_t>::type;
    using vec2ResGmType = typename std::conditional<splitD, GlobalTensor<float>, int8_t>::type;
    using flashdecodeGmType = GlobalTensor<float>;

    /* =================来自原Block的 GM tensors================= */
    GlobalTensor<OUTPUT_T> attentionOutGm;
    GlobalTensor<float> softmaxLseGm;
    attenMaskGmType attenMaskGmInt;
    vec2ResGmType vec2ResGm[3];
    GlobalTensor<INPUT_T> sinkGm;
    flashdecodeGmType accumOutGm;
    flashdecodeGmType softmaxFDSumGm;
    flashdecodeGmType softmaxFDMaxGm;
    /* =================来自原Block的 UB buffers================= */
    TBuf<> commonTBuf;
    TQue<QuePosition::VECOUT, 1> stage1OutQue[2];
    TQue<QuePosition::VECIN, 1> attenMaskInQue[2];
    TBuf<> stage2OutBuf;
    TEventID mte3ToVId[2];
    TEventID vToMte3Id[2];
    TBuf<> softmaxMaxBuf[PRELOAD_N + 1];
    TBuf<> softmaxSumBuf[PRELOAD_N + 1];
    TBuf<> softmaxExpBuf[PRELOAD_N + 1];
    TBuf<> vselrIndexesBuf[4];
    TBuf<> mm2InBuf;
    TQue<QuePosition::VECOUT, 1> maxBrdcst;
    TQue<QuePosition::VECOUT, 1> sumBrdcst;
    TQue<QuePosition::VECOUT, 1> softmaxLseQueue;
    TQue<QuePosition::VECIN, 1> sinkQue;
    /* =================来自原Block的其他成员================= */
    uint32_t negativeIntScalar = *((uint32_t *)&BOOL_ATTEN_MASK_SCALAR_VALUE);
    T negativeFloatScalar;
    int64_t vec2SubBlockOffset = 0;
// CV buffers
    BufferManager<BufferType::GM> gmBufferManager;
    BufferManager<BufferType::UB> ubBufferManager;
    BufferManager<BufferType::L1> l1BufferManager;
    BuffersPolicy3buff<BufferType::GM, SyncType::CROSS_CORE_SYNC_FORWARD> bmm2ResGmBuffers;
    BuffersPolicyDB<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> bmm1Buffers;
    BuffersPolicyDB<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> bmm2Buffers;
    BuffersPolicy3buff<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> l1PBuffers;

    GlobalTensor<uint32_t> faMetaDataGm;
    GlobalTensor<uint32_t> fdMetaDataGm;
    __gm__ uint8_t *keyPtr = nullptr;

    ConstInfoXVec constInfoXVec;

    TPipe *pipe;
    VecFdBlockType vecFdBlock;

    uint32_t createdTaskCount = 0U;

    uint64_t actSeqLensKv = 0;
    uint64_t actSeqLensQ = 0;
    uint32_t curS2Start;
    uint32_t curS2End = 0;
    uint32_t prevBIdx;
    uint32_t prevBN2Idx;
    uint32_t prevGS1Idx;
    uint32_t mloop = 0;
    bool headS2Split = false;
    bool tailS2Split = false;

    uint32_t sectionNum_;
    bool isFd;
    // fa metadata
    uint32_t bN2Start_;
    uint32_t bN2End_;
    uint32_t gS1OStart_;
    uint32_t gS1OEnd_;
    uint32_t s2OStart_;
    uint32_t s2OEnd_;
    uint32_t coreFirstTmpOutWsPos_;
    // fd metadata
    FDparamsX fdParams_;

    //  Shared buffer between fa and fd
    TQue<QuePosition::VECOUT, 1> SharedBuffer1[2]; // 共用
    TQue<QuePosition::VECIN, 1> SharedBuffer2[2];  // 共用
    TBuf<> SharedBuffer3;                          // 共用

    // 非tnd时不使用cuSeqLens，这里保证定义
    QSeqParserType qSeqParser;
    KvSeqParserType kvSeqParser;

    // ==============================fuction=======================================================
    __aicore__ inline FlashAttentionNoQuantGqaKernelVec()
        : vecFdBlock(constInfoXVec){};


    // ==================== Vec Compute Methods (from FANoQuantGqaBlockVec) ====================

    

    __aicore__ inline void InitVecInput(__gm__ uint8_t *attenMask, __gm__ uint8_t *learnableSink,
                                        __gm__ uint8_t *softmaxLse, __gm__ uint8_t *attentionOut,
                                        __gm__ uint8_t *workspace)
    {
        this->attentionOutGm.SetGlobalBuffer((__gm__ OUTPUT_T *)attentionOut);

        if (constInfoXVec.isSoftmaxLseEnable) {
            softmaxLseGm.SetGlobalBuffer((__gm__ float *)softmaxLse);
        }

        if constexpr (hasAtten) {
            attenMaskGmInt.SetGlobalBuffer((__gm__ uint8_t *)attenMask);
        }

        if constexpr (splitD) {
            int64_t vec2ResultSize = mBaseSize * constInfoXVec.dBasicBlock;
            vec2SubBlockOffset = constInfoXVec.subBlockIdx * vec2ResultSize >> 1;
            int64_t prevCoretotalOffset = constInfoXVec.aicIdx * 3 * vec2ResultSize; // 3为preload次数
            vec2ResGm[0].SetGlobalBuffer((__gm__ T *)workspace + prevCoretotalOffset);
            vec2ResGm[1].SetGlobalBuffer((__gm__ T *)workspace + prevCoretotalOffset + vec2ResultSize);
            vec2ResGm[2].SetGlobalBuffer((__gm__ T *)workspace + prevCoretotalOffset + 2 * vec2ResultSize);
            workspace = workspace + constInfoXVec.coreNum * 3 * vec2ResultSize * sizeof(T);
        }

        if (learnableSink != nullptr) {
            sinkGm.SetGlobalBuffer((__gm__ INPUT_T *)learnableSink);
        }

        if (this->isFd) {
            accumOutGm.SetGlobalBuffer((__gm__ float *)workspace);
            softmaxFDSumGm.SetGlobalBuffer((__gm__ float *)workspace + constInfoXVec.accumOutSize);
            softmaxFDMaxGm.SetGlobalBuffer((__gm__ float *)workspace + constInfoXVec.accumOutSize +
                                           constInfoXVec.logSumExpSize);
        }
    }
    __aicore__ inline void ProcessVec1(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
                                       Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf,
                                       RunInfoXVec runInfo)
    {
#ifdef SKIP_V1
        bmm1ResBuf.WaitCrossCore();
        bmm1ResBuf.SetCrossCore();
        outputBuf.SetCrossCore();
        return;
#endif

        bmm1ResBuf.WaitCrossCore();
        if (unlikely(runInfo.actVecMSize == 0)) {
            bmm1ResBuf.SetCrossCore();
            outputBuf.SetCrossCore();
            return;
        }

        if constexpr (useDn) {
            ProcessVec1Dn(outputBuf, bmm1ResBuf, runInfo);
        } else {
            ProcessVec1Nd(outputBuf, bmm1ResBuf, runInfo);
        }
    }

    __aicore__ inline void ClearOutput()
    {
        if (IsInitAttentionOutGm()) {
            SetFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId); // 释放剩余ub
            InitOutputSingleCore();
            if (constInfoXVec.isSoftmaxLseEnable) {
                InitLseOutputSingleCore();
            }
            WaitFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
            SyncAll();
        }
    }

    __aicore__ inline bool IsInitAttentionOutGm()
    {
        return constInfoXVec.needInitOutput;
    }

    __aicore__ inline void InitOutputSingleCore()
    {
        int64_t tSize = constInfoXVec.bSize * constInfoXVec.s1Size;
        if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
            tSize = qSeqParser.GetTSize();
        }
        int64_t totalOutputSize = tSize * constInfoXVec.n2Size * constInfoXVec.gSize * constInfoXVec.dSizeV;
        int64_t singleCoreSize =
            (totalOutputSize + (2 * constInfoXVec.coreNum) - 1) / (2 * constInfoXVec.coreNum); // 2 means c:v = 1:2
        int64_t tailSize = totalOutputSize - constInfoXVec.aivIdx * singleCoreSize;
        int64_t singleInitOutputSize = tailSize < singleCoreSize ? tailSize : singleCoreSize;

        if (singleInitOutputSize > 0) {
            WaitFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
            if constexpr (IsSameType<OUTPUT_T, int8_t>::value) {
                GlobalTensor<half> attentionOutTmpGm;
                attentionOutTmpGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(attentionOutGm.GetPhyAddr(0)));
                matmul::InitOutput<half>(attentionOutTmpGm[constInfoXVec.aivIdx * singleCoreSize], singleInitOutputSize, 0);
            } else {
                matmul::InitOutput<OUTPUT_T>(attentionOutGm[constInfoXVec.aivIdx * singleCoreSize], singleInitOutputSize,
                    0);
            }
            SetFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
        }
    }

    __aicore__ inline void InitLseOutputSingleCore()
    {
        int64_t tSize = constInfoXVec.bSize * constInfoXVec.s1Size;
        if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
            tSize = qSeqParser.GetTSize();
        }
        int64_t totalOutputSize = tSize * constInfoXVec.n2Size * constInfoXVec.gSize;
        int64_t singleCoreSize =
            (totalOutputSize + (2 * constInfoXVec.coreNum) - 1) / (2 * constInfoXVec.coreNum); // 2 means c:v = 1:2
        int64_t tailSize = totalOutputSize - constInfoXVec.aivIdx * singleCoreSize;
        int64_t singleInitOutputSize = tailSize < singleCoreSize ? tailSize : singleCoreSize;

        if (singleInitOutputSize > 0) {
            WaitFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
            matmul::InitOutput<float>(softmaxLseGm[constInfoXVec.aivIdx * singleCoreSize], singleInitOutputSize,
                                      3e+99); // 3e+99: set the value of invalid batch to inf
            SetFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
        }
    }

    // =================================Private Functions=================================

    __aicore__ inline void ProcessVec1Dn(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
                                         Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf,
                                         RunInfoXVec runInfo)
    {
        LocalTensor<uint8_t> attenMaskUb;

        LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.mloop % (PRELOAD_N + 1)].template Get<float>()[0];
        LocalTensor<float> maxUb = this->softmaxMaxBuf[runInfo.mloop % (PRELOAD_N + 1)].template Get<float>()[0];

        auto expUb = this->softmaxExpBuf[runInfo.loop % (PRELOAD_N + 1)].template Get<T>()[0];
        int64_t stage1Offset = runInfo.loop & (DB - 1);

        LocalTensor<T> mmRes = bmm1ResBuf.template GetTensor<T>();
        auto stage1CastTensor = this->stage1OutQue[stage1Offset].template AllocTensor<INPUT_T>();
        if (unlikely(runInfo.isFirstS2Loop)) {
            FaVectorApi::ProcessVec1VfDn<T, INPUT_T, false, false, s2BaseSize>(
                stage1CastTensor, sumUb, maxUb, mmRes, expUb, this->vselrIndexesBuf, attenMaskUb,
                runInfo.actMSizeAlign32 >> 1, runInfo.actSingleLoopS2SizeAlign, runInfo.actSingleLoopS2Size,
                static_cast<T>(constInfoXVec.scaleValue), 1.0f, negativeFloatScalar, 0.0F, false);
        } else {
            FaVectorApi::ProcessVec1VfDn<T, INPUT_T, true, false, s2BaseSize>(
                stage1CastTensor, sumUb, maxUb, mmRes, expUb, this->vselrIndexesBuf, attenMaskUb,
                runInfo.actMSizeAlign32 >> 1, runInfo.actSingleLoopS2SizeAlign, runInfo.actSingleLoopS2Size,
                static_cast<T>(constInfoXVec.scaleValue), 1.0f, negativeFloatScalar, 0.0F, false);
        }
        bmm1ResBuf.SetCrossCore();

        this->stage1OutQue[stage1Offset].template EnQue(stage1CastTensor);
        this->stage1OutQue[stage1Offset].template DeQue<INPUT_T>();
        //-------------------------Data copy to l1-------------------------
        LocalTensor<INPUT_T> mm2AL1Tensor = outputBuf.GetTensor<INPUT_T>();

        if (runInfo.actSingleLoopS2Size > vec1S2CopyLenDn) {
            DataCopy(mm2AL1Tensor[constInfoXVec.subBlockIdx * vec1HalfS1BaseSize * runInfo.actSingleLoopS2SizeAlign],
                     stage1CastTensor,
                     {vec1S2CopyCountDn, vec1S2CopyLenDn, 1,
                      static_cast<uint16_t>(runInfo.actSingleLoopS2SizeAlign - vec1S2CopyLenDn)});
            DataCopy(mm2AL1Tensor[constInfoXVec.subBlockIdx * vec1HalfS1BaseSize * runInfo.actSingleLoopS2SizeAlign +
                                  vec1S2strideDn],
                     stage1CastTensor[vec1ResOffsetDn],
                     {vec1S2CopyCountDn, static_cast<uint16_t>(runInfo.actSingleLoopS2SizeAlign - vec1S2CopyLenDn),
                      static_cast<uint16_t>(s2BaseSize - runInfo.actSingleLoopS2SizeAlign + 1), vec1S2CopyLenDn});
        } else {
            DataCopy(mm2AL1Tensor[constInfoXVec.subBlockIdx * vec1HalfS1BaseSize * runInfo.actSingleLoopS2SizeAlign],
                     stage1CastTensor,
                     {vec1S2CopyCountDn, static_cast<uint16_t>(runInfo.actSingleLoopS2SizeAlign),
                      static_cast<uint16_t>(vec1S2CopyLenDn - runInfo.actSingleLoopS2SizeAlign + 1), 0});
        }
        outputBuf.SetCrossCore();
        //-----------------------------------------------------------------
        this->stage1OutQue[stage1Offset].template FreeTensor(stage1CastTensor);
        if (unlikely(runInfo.isLastS2Loop)) {
            SoftmaxDataCopyOut(runInfo, sumUb, maxUb);
        }
        return;
    }

    __aicore__ inline void SoftmaxDataCopyOut(RunInfoXVec runInfo, LocalTensor<float> &sumUb, LocalTensor<float> &maxUb)
    {
        if (this->isFd && runInfo.isS2SplitCore) {
            ComputeLogSumExpAndCopyToGm(runInfo, sumUb, maxUb);
        }
        if (constInfoXVec.isSoftmaxLseEnable && !(this->isFd && runInfo.isS2SplitCore)) {
            SoftmaxLseCopyOut(sumUb, maxUb, runInfo);
        }

        if (constInfoXVec.learnableSinkFlag) {
            this->Vec1SinkComputeGSFused(runInfo, sumUb, maxUb);
        }
    }

    __aicore__ inline void SoftmaxLseCopyOut(LocalTensor<float> &softmaxSumTmp, LocalTensor<float> &softmaxMaxTmp,
                                             RunInfoXVec &runInfo)
    {
        if (unlikely(runInfo.actVecMSize == 0)) {
            return;
        }

        uint32_t vecMIdx = runInfo.gS1Idx + runInfo.vecMbaseIdx;
        LocalTensor<float> lseUb = this->softmaxLseQueue.template AllocTensor<float>();
        ComputeLseOutputVF(lseUb, softmaxSumTmp, softmaxMaxTmp, runInfo.actVecMSize);
        softmaxLseQueue.template EnQue(lseUb);
        softmaxLseQueue.DeQue<float>();

        if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
            uint32_t prefixBS1 = qSeqParser.GetTBase(runInfo.bIdx);
            uint64_t bN2Offset = runInfo.n2Idx * constInfoXVec.gSize * constInfoXVec.t1Size + prefixBS1;
            DataCopySoftmaxLseTNDtoNTArch35<T, ConstInfoXVec>(softmaxLseGm, lseUb, bN2Offset, vecMIdx, runInfo.actVecMSize,
                                                           constInfoXVec);
        } else if constexpr (layout == LayOutTypeEnum::LAYOUT_BSH) {
            uint64_t bN2Offset = runInfo.bIdx * constInfoXVec.n2Size * constInfoXVec.gSize * constInfoXVec.s1Size +
                                 runInfo.n2Idx * constInfoXVec.gSize * constInfoXVec.s1Size;
            DataCopySoftmaxLseBSNDArch35<T, ConstInfoXVec>(softmaxLseGm, lseUb, bN2Offset, vecMIdx, runInfo.actVecMSize,
                                                        constInfoXVec);
        } else { // BNSD
            uint64_t bN2Offset = runInfo.bIdx * constInfoXVec.n2Size * constInfoXVec.gSize * constInfoXVec.s1Size +
                                 runInfo.n2Idx * constInfoXVec.gSize * constInfoXVec.s1Size;
            uint64_t qActSeqLens = qSeqParser.GetActualSeqLength(runInfo.bIdx);
            DataCopySoftmaxLseBNSDArch35<T, ConstInfoXVec>(softmaxLseGm, lseUb, bN2Offset, vecMIdx, runInfo.actVecMSize,
                                                        constInfoXVec, qActSeqLens);
        }

        softmaxLseQueue.FreeTensor(lseUb);
    }

    __aicore__ inline void Vec1SinkComputeGSFused(RunInfoXVec &runInfo, LocalTensor<float> &sumUb,
                                                  LocalTensor<float> &maxUb)
    {
        CopySinkIn(runInfo);
        LocalTensor<INPUT_T> sinkUb = sinkQue.DeQue<INPUT_T>();
        SinkSubExpAddGSFusedVF<float, INPUT_T>(sinkUb, sumUb, maxUb, runInfo.actVecMSize);
        sinkQue.FreeTensor(sinkUb);
    }

    __aicore__ inline void CopySinkIn(RunInfoXVec &runInfo)
    {
        LocalTensor<INPUT_T> sinkUbBf16 = sinkQue.AllocTensor<INPUT_T>();
        int64_t sinkOffset =
            runInfo.n2Idx * constInfoXVec.gSize + constInfoXVec.subBlockIdx * (runInfo.actMSize - runInfo.actVecMSize);
        DataCopyExtParams sinkCopyParams;
        sinkCopyParams.blockCount = 1;                                   // 进行一次连续拷贝
        sinkCopyParams.blockLen = runInfo.actVecMSize * sizeof(INPUT_T); // 实际需要拷贝的字节数
        sinkCopyParams.srcStride = 0;                                    // 源地址连续
        sinkCopyParams.dstStride = 0;                                    // 目的地址连续

        DataCopyPadExtParams<INPUT_T> sinkCopyPadParams{};
        DataCopyPad(sinkUbBf16, this->sinkGm[sinkOffset], sinkCopyParams, sinkCopyPadParams);
        sinkQue.EnQue(sinkUbBf16);
    }

    __aicore__ inline void ProcessVec1Nd(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
                                         Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf,
                                         RunInfoXVec runInfo)
    {
        LocalTensor<uint8_t> dropMaskUb;
        LocalTensor<INPUT_T> nonePseUb; // PSE不支持，占位
        LocalTensor<uint8_t> attenMaskUb;
        if constexpr (hasAtten == true) {
            attenMaskUb = this->attenMaskInQue[runInfo.loop & (DB - 1)].template AllocTensor<uint8_t>();
            AttenMaskCopyIn(attenMaskUb, 0, runInfo.actVecMSize, runInfo); // 全量拷贝
        }

        LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.mloop % (PRELOAD_N + 1)].template Get<float>();
        LocalTensor<float> maxUb = this->softmaxMaxBuf[runInfo.mloop % (PRELOAD_N + 1)].template Get<float>();
        LocalTensor<float> expUb = this->softmaxExpBuf[runInfo.loop % (PRELOAD_N + 1)].template Get<T>();
        LocalTensor<T> pScaleUb;
        LocalTensor<T> queryScaleUb;
        LocalTensor<uint8_t> apiTmpBuffer;

        apiTmpBuffer = this->commonTBuf.template Get<uint8_t>();

        int64_t stage1Offset = runInfo.loop & (DB - 1);

        LocalTensor<T> mmRes = bmm1ResBuf.template GetTensor<T>();
        auto stage1CastTensor = this->stage1OutQue[stage1Offset].template AllocTensor<INPUT_T>();
        if (runInfo.isFirstS2Loop) {
            if (likely(runInfo.actSingleLoopS2Size == 128)) {
                ProcessVec1Vf<T, INPUT_T, INPUT_T /*pseShiftType*/, false, mBaseSize, s2BaseSize, EQ_128, hasAtten,
                              PseTypeEnum::PSE_NONE_TYPE, hasDrop, false, false>(
                    stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, mmRes, expUb, sumUb, maxUb, attenMaskUb,
                    nonePseUb, dropMaskUb, apiTmpBuffer, pScaleUb, runInfo.actVecMSize, runInfo.actSingleLoopS2Size,
                    0 /*pseStride*/, 0.0f /*slopes*/, 0.0f /*posShift*/, static_cast<T>(constInfoXVec.scaleValue),
                    1.0f, negativeFloatScalar, 0.0F, queryScaleUb, 1.0f);
            } else if (runInfo.actSingleLoopS2Size <= 64) {
                ProcessVec1Vf<T, INPUT_T, INPUT_T /*pseShiftType*/, false, mBaseSize, s2BaseSize, GT_0_AND_LTE_64,
                              hasAtten, PseTypeEnum::PSE_NONE_TYPE, hasDrop, false, false>(
                    stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, mmRes, expUb, sumUb, maxUb, attenMaskUb,
                    nonePseUb, dropMaskUb, apiTmpBuffer, pScaleUb, runInfo.actVecMSize, runInfo.actSingleLoopS2Size,
                    0 /*pseStride*/, 0.0f /*slopes*/, 0.0f /*posShift*/, static_cast<T>(constInfoXVec.scaleValue),
                    1.0f, negativeFloatScalar, 0.0F, queryScaleUb, 1.0f);
            } else if (runInfo.actSingleLoopS2Size < 128) {
                ProcessVec1Vf<T, INPUT_T, INPUT_T /*pseShiftType*/, false, mBaseSize, s2BaseSize, GT_64_AND_LTE_128,
                              hasAtten, PseTypeEnum::PSE_NONE_TYPE, hasDrop, false, false>(
                    stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, mmRes, expUb, sumUb, maxUb, attenMaskUb,
                    nonePseUb, dropMaskUb, apiTmpBuffer, pScaleUb, runInfo.actVecMSize, runInfo.actSingleLoopS2Size,
                    0 /*pseStride*/, 0.0f /*slopes*/, 0.0f /*posShift*/, static_cast<T>(constInfoXVec.scaleValue),
                    1.0f, negativeFloatScalar, 0.0F, queryScaleUb, 1.0f);
            } else {
                if constexpr (s2BaseSize == 256) {
                    ProcessVec1Vf<T, INPUT_T, INPUT_T /*pseShiftType*/, false, mBaseSize, s2BaseSize,
                                  GT_128_AND_LTE_256, hasAtten, PseTypeEnum::PSE_NONE_TYPE, hasDrop>(
                        stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, mmRes, expUb, sumUb, maxUb, attenMaskUb,
                        nonePseUb, dropMaskUb, apiTmpBuffer, expUb, runInfo.actVecMSize, runInfo.actSingleLoopS2Size,
                        0 /*pseStride*/, 0.0f /*slopes*/, 0.0f /*posShift*/, static_cast<T>(constInfoXVec.scaleValue),
                        1.0f, negativeFloatScalar, 0.0F);
                }
            }
        } else {
            if (likely(runInfo.actSingleLoopS2Size == 128)) {
                ProcessVec1Vf<T, INPUT_T, INPUT_T /*pseShiftType*/, true, mBaseSize, s2BaseSize, EQ_128, hasAtten,
                              PseTypeEnum::PSE_NONE_TYPE, hasDrop, false, false>(
                    stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, mmRes, expUb, sumUb, maxUb, attenMaskUb,
                    nonePseUb, dropMaskUb, apiTmpBuffer, pScaleUb, runInfo.actVecMSize, runInfo.actSingleLoopS2Size,
                    0 /*pseStride*/, 0.0f /*slopes*/, 0.0f /*posShift*/, static_cast<T>(constInfoXVec.scaleValue),
                    1.0f, negativeFloatScalar, 0.0F, queryScaleUb, 1.0f);
            } else if (runInfo.actSingleLoopS2Size <= 64) {
                ProcessVec1Vf<T, INPUT_T, INPUT_T /*pseShiftType*/, true, mBaseSize, s2BaseSize, GT_0_AND_LTE_64,
                              hasAtten, PseTypeEnum::PSE_NONE_TYPE, hasDrop, false, false>(
                    stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, mmRes, expUb, sumUb, maxUb, attenMaskUb,
                    nonePseUb, dropMaskUb, apiTmpBuffer, pScaleUb, runInfo.actVecMSize, runInfo.actSingleLoopS2Size,
                    0 /*pseStride*/, 0.0f /*slopes*/, 0.0f /*posShift*/, static_cast<T>(constInfoXVec.scaleValue),
                    1.0f, negativeFloatScalar, 0.0F, queryScaleUb, 1.0f);
            } else if (runInfo.actSingleLoopS2Size < 128) {
                ProcessVec1Vf<T, INPUT_T, INPUT_T /*pseShiftType*/, true, mBaseSize, s2BaseSize, GT_64_AND_LTE_128,
                              hasAtten, PseTypeEnum::PSE_NONE_TYPE, hasDrop, false, false>(
                    stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, mmRes, expUb, sumUb, maxUb, attenMaskUb,
                    nonePseUb, dropMaskUb, apiTmpBuffer, pScaleUb, runInfo.actVecMSize, runInfo.actSingleLoopS2Size,
                    0 /*pseStride*/, 0.0f /*slopes*/, 0.0f /*posShift*/, static_cast<T>(constInfoXVec.scaleValue),
                    1.0f, negativeFloatScalar, 0.0F, queryScaleUb, 1.0f);
            } else {
                if constexpr (s2BaseSize == 256) {
                    ProcessVec1Vf<T, INPUT_T, INPUT_T /*pseShiftType*/, true, mBaseSize, s2BaseSize, GT_128_AND_LTE_256,
                                  hasAtten, PseTypeEnum::PSE_NONE_TYPE, hasDrop>(
                        stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, mmRes, expUb, sumUb, maxUb, attenMaskUb,
                        nonePseUb, dropMaskUb, apiTmpBuffer, expUb, runInfo.actVecMSize, runInfo.actSingleLoopS2Size,
                        0 /*pseStride*/, 0.0f /*slopes*/, 0.0f /*posShift*/, static_cast<T>(constInfoXVec.scaleValue),
                        1.0f, negativeFloatScalar, 0.0F);
                }
            }
        }
        bmm1ResBuf.SetCrossCore();
        if constexpr (hasAtten) {
            this->attenMaskInQue[runInfo.loop & (DB - 1)].template FreeTensor(attenMaskUb);
        }

        // ===================DataCopy to L1 ====================
        this->stage1OutQue[stage1Offset].template EnQue(stage1CastTensor);
        this->stage1OutQue[stage1Offset].template DeQue<INPUT_T>();
        LocalTensor<INPUT_T> mm2AL1Tensor;
        mm2AL1Tensor = outputBuf.GetTensor<INPUT_T>();

        if (likely(runInfo.actVecMSize != 0)) {
            DataCopy(mm2AL1Tensor[constInfoXVec.subBlockIdx * (blockBytes / sizeof(INPUT_T)) *
                                  (runInfo.actMSize - runInfo.actVecMSize)],
                     stage1CastTensor,
                     {s2BaseSize / 16, (uint16_t)runInfo.actVecMSize, (uint16_t)(vec1Srcstride - runInfo.actVecMSize),
                      (uint16_t)(mBaseSize - runInfo.actVecMSize)});
        }
        this->stage1OutQue[stage1Offset].template FreeTensor(stage1CastTensor);

        outputBuf.SetCrossCore();
        // ======================================================
        if (!runInfo.isFirstS2Loop) {
            UpdateExpSumAndExpMax<T>(sumUb, maxUb, expUb, sumUb, maxUb, apiTmpBuffer, runInfo.actVecMSize);
        }

        if (unlikely(runInfo.isLastS2Loop)) {
            SoftmaxDataCopyOut(runInfo, sumUb, maxUb);
        }
    }

    __aicore__ inline void ProcessVec2OnUb(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm2ResBuf,
                                           RunInfoXVec runInfo)
    {
        if (unlikely(runInfo.actVecMSize == 0)) {
            bmm2ResBuf.SetCrossCore();
            return;
        }

        int64_t vec2CalcSize = runInfo.actVecMSize * dTemplateAlign64;

        LocalTensor<T> vec2ResUb = this->stage2OutBuf.template Get<T>();
        LocalTensor<T> mmRes = bmm2ResBuf.template GetTensor<T>();
        WaitFlag<HardEvent::MTE3_V>(mte3ToVId[0]);
        if (unlikely(runInfo.isFirstS2Loop)) {
            DataCopy(vec2ResUb, mmRes, vec2CalcSize);
        } else {
            LocalTensor<T> expUb = softmaxExpBuf[runInfo.loop % (PRELOAD_N + 1)].template Get<T>();
            LocalTensor<T> pScaleUb;

            if (!runInfo.isLastS2Loop) {
                FlashUpdateNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, false, false>(
                    vec2ResUb, mmRes, vec2ResUb, expUb, pScaleUb, runInfo.actVecMSize, dTemplateAlign64, 1.0, 1.0);
            } else {
                LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.mloop % (PRELOAD_N + 1)].template Get<float>();
                FlashUpdateLastNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, false, false>(
                    vec2ResUb, mmRes, vec2ResUb, expUb, pScaleUb, sumUb, runInfo.actVecMSize, dTemplateAlign64, 1.0,
                    1.0);
            }
        }
        bmm2ResBuf.SetCrossCore();
        if (runInfo.isLastS2Loop) {
            if (unlikely(runInfo.isFirstS2Loop)) {
                LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.mloop % (PRELOAD_N + 1)].template Get<float>();
                LastDivNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, false>(
                    vec2ResUb, vec2ResUb, sumUb, runInfo.actVecMSize, (uint16_t)dTemplateAlign64, 0.0F);
            }
            CopyOutAttentionOut(runInfo, vec2ResUb, 0, runInfo.actVecMSize);
        }
        SetFlag<HardEvent::MTE3_V>(mte3ToVId[0]);
    }

    __aicore__ inline void Bmm2ResCastAndCopyOut(RunInfoXVec &runInfo, LocalTensor<T> &vec2ResUb, uint32_t mStartVec,
                                                 uint32_t mDealSize)
    {
        LocalTensor<OUTPUT_T> attenOut;
        int64_t dSizeAligned64 = (int64_t)dVTemplateType;
        if constexpr (splitD) {
            dSizeAligned64 = constInfoXVec.dBasicBlock;
        }

        attenOut.SetAddr(vec2ResUb.address_);

        RowInvalid(vec2ResUb, mStartVec, mDealSize, runInfo, dSizeAligned64);
        Cast(attenOut, vec2ResUb, RoundMode::CAST_ROUND, mDealSize * dSizeAligned64);
        SetFlag<HardEvent::V_MTE3>(vToMte3Id[0]);
        WaitFlag<HardEvent::V_MTE3>(vToMte3Id[0]);

        Bmm2DataCopyOutTrans(runInfo, attenOut, mStartVec, mDealSize);
    }

    template <typename VEC2_RES_T>
    __aicore__ inline void ProcessVec2DSplit(GlobalTensor<T> &mmRes, const RunInfoXVec &runInfo)
    {
        // bmm2 result is on GM and global update data on UB
        uint32_t subMBase = 8192 / constInfoXVec.dBasicBlock;
        int64_t vec2LoopLimit = CeilDiv(runInfo.actVecMSize, subMBase);
        LocalTensor<T> vec2ResUb = this->stage2OutBuf.template Get<T>();
        WaitFlag<HardEvent::MTE3_V>(mte3ToVId[0]);
        LocalTensor<T> bmm2Ub = mm2InBuf.template Get<T>();

        event_t mte2ToV = static_cast<event_t>(pipe->FetchEventID(HardEvent::MTE2_V));
        event_t vToMte2 = static_cast<event_t>(pipe->FetchEventID(HardEvent::V_MTE2));
        event_t mte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));

        for (int64_t mOIdx = 0; mOIdx < vec2LoopLimit; mOIdx++) {
            int32_t mIdx = mOIdx * subMBase;
            uint32_t subMRealsize = subMBase;
            if (mOIdx == vec2LoopLimit - 1) {
                subMRealsize = runInfo.actVecMSize - mIdx;
            }

            int64_t vec2CalcSize = subMRealsize * constInfoXVec.dBasicBlock;
            // Gm地址偏移是按照实际DSize实现的，虽然我们设置了KC=192
            int64_t mm2ResInnerOffset = mIdx * dTemplateAlign64;
            SetFlag<HardEvent::V_MTE2>(vToMte2);
            WaitFlag<HardEvent::V_MTE2>(vToMte2);
            if (constInfoXVec.dSizeV == dTemplateAlign64) {
                if constexpr (useDn) {
                    DataCopy(bmm2Ub, mmRes[mm2ResInnerOffset], vec2CalcSize);
                } else {
                    DataCopy(bmm2Ub,
                             mmRes[constInfoXVec.subBlockIdx * (runInfo.actMSize - runInfo.actVecMSize) *
                                       (int64_t)dVTemplateType +
                                   mm2ResInnerOffset],
                             vec2CalcSize);
                }
            } else {
                DataCopyParams dataCopyParams;
                DataCopyPadParams dataCopyPadParams;
                dataCopyParams.blockCount = subMRealsize;
                dataCopyParams.dstStride = (constInfoXVec.dBasicBlock - constInfoXVec.dSizeV) * sizeof(T) / blockBytes;
                dataCopyParams.srcStride = (dTemplateAlign64 - constInfoXVec.dSizeV) * sizeof(T);
                dataCopyParams.blockLen = constInfoXVec.dSizeV * sizeof(T);
                if constexpr (useDn) {
                    DataCopyPad(bmm2Ub, mmRes[mm2ResInnerOffset], dataCopyParams,
                                dataCopyPadParams);
                } else {
                    DataCopyPad(bmm2Ub,
                                mmRes[constInfoXVec.subBlockIdx * (runInfo.actMSize - runInfo.actVecMSize) *
                                          (int64_t)dVTemplateType +
                                      mm2ResInnerOffset],
                                dataCopyParams, dataCopyPadParams);
                }
            }

            // 经过了跳读，UB上每行是按照dTemplateAlign64对齐的
            int64_t vec2ResInnerOffset = mIdx * constInfoXVec.dBasicBlock;
            if (vec2LoopLimit > 1) {
                SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2);
                WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2);
                if constexpr (useDn) {
                    DataCopy(vec2ResUb,
                             this->vec2ResGm[runInfo.loop & (DB - 1)][vec2SubBlockOffset + vec2ResInnerOffset],
                             vec2CalcSize);
                } else {
                    DataCopy(vec2ResUb,
                             this->vec2ResGm[runInfo.loop & (DB - 1)][constInfoXVec.subBlockIdx *
                                                                    (runInfo.actMSize - runInfo.actVecMSize) *
                                                                    constInfoXVec.dBasicBlock +
                                                                vec2ResInnerOffset],
                             vec2CalcSize);
                }
            }
            SetFlag<HardEvent::MTE2_V>(mte2ToV);
            WaitFlag<HardEvent::MTE2_V>(mte2ToV);
            if (unlikely(runInfo.isFirstS2Loop)) {
                DataCopy(vec2ResUb, bmm2Ub, vec2CalcSize);
            } else {
                int64_t vec2ExpBufOffset = mIdx;
                LocalTensor<T> expUb =
                    softmaxExpBuf[runInfo.loop % (PRELOAD_N + 1)].template Get<T>()[vec2ExpBufOffset];
                if (!runInfo.isLastS2Loop) {
                    FlashUpdateNew<T, INPUT_T, OUTPUT_T, 0xFF, false, false>(vec2ResUb, bmm2Ub, vec2ResUb, expUb, expUb,
                                                                             subMRealsize, constInfoXVec.dBasicBlock, 0.0F,
                                                                             1.0f);
                } else {
                    int64_t vec2SumBufOffset = mIdx;
                    LocalTensor<float> sumUb =
                        this->softmaxSumBuf[runInfo.mloop % (PRELOAD_N + 1)].template Get<float>()[vec2SumBufOffset];
                    FlashUpdateLastNew<T, INPUT_T, OUTPUT_T, 0xFF, false, false>(
                        vec2ResUb, bmm2Ub, vec2ResUb, expUb, expUb, sumUb, subMRealsize, constInfoXVec.dBasicBlock, 0.0F,
                        1.0f);
                }
            }

            if (unlikely(runInfo.isLastS2Loop)) {
                if (unlikely(runInfo.isFirstS2Loop)) {
                    int64_t vec2SumBufOffset = mIdx;
                    LocalTensor<float> sumUb =
                        this->softmaxSumBuf[runInfo.mloop % (PRELOAD_N + 1)].template Get<float>()[vec2SumBufOffset];
                    LastDivNew<T, INPUT_T, OUTPUT_T, 0xFF, false>(vec2ResUb, vec2ResUb, sumUb, subMRealsize,
                                                                  constInfoXVec.dBasicBlock, 0.0F);
                }
                CopyOutAttentionOut(runInfo, vec2ResUb, mIdx, subMRealsize);
            } else if (vec2LoopLimit > 1) {
                SetFlag<HardEvent::V_MTE3>(vToMte3Id[0]);
                WaitFlag<HardEvent::V_MTE3>(vToMte3Id[0]);
                if constexpr (useDn) {
                    DataCopy(this->vec2ResGm[runInfo.loop & (DB - 1)][vec2SubBlockOffset + vec2ResInnerOffset],
                             vec2ResUb,
                             vec2CalcSize);
                } else {
                    DataCopy(this->vec2ResGm[runInfo.loop & (DB - 1)][constInfoXVec.subBlockIdx *
                                                                    (runInfo.actMSize - runInfo.actVecMSize) *
                                                                    constInfoXVec.dBasicBlock +
                                                                vec2ResInnerOffset],
                             vec2ResUb, vec2CalcSize);
                }
            }
        }
        SetFlag<HardEvent::MTE3_V>(mte3ToVId[0]);
    }

    __aicore__ inline void CopyOutAttentionOut(RunInfoXVec runInfo, LocalTensor<T> &vec2ResUb, uint32_t mStartVec,
                                               uint32_t mDealSize)
    {
        if (this->isFd && runInfo.isS2SplitCore) {
            Bmm2ResForFDCopyOut(runInfo, vec2ResUb, mStartVec, mDealSize);
        } else {
            Bmm2ResCastAndCopyOut(runInfo, vec2ResUb, mStartVec, mDealSize);
        }
    }

    __aicore__ inline bool CalcBlockNeedRowInvalid(RunInfoXVec &runInfo, int64_t s1FirstValidToken,
                                                   int64_t s1LastValidToken)
    {
        int32_t vecMStartIdx = runInfo.gS1Idx + runInfo.vecMbaseIdx;
        int32_t vecMEndIdx = vecMStartIdx + runInfo.actVecMSize - 1;
        int32_t s1StartTdx, s1EndTdx;
        bool ret = false;
        if constexpr (layout == LayOutTypeEnum::LAYOUT_BSH || layout == LayOutTypeEnum::LAYOUT_SBH ||
                      layout == LayOutTypeEnum::LAYOUT_TND) {
            // S1G layout
            s1StartTdx = vecMStartIdx / constInfoXVec.gSize;
            s1EndTdx = vecMEndIdx / constInfoXVec.gSize;
            ret = (s1StartTdx < s1FirstValidToken) || (s1EndTdx > s1LastValidToken);
        } else {
            // GS1 layout
            s1StartTdx = vecMStartIdx % runInfo.actS1Size;
            s1EndTdx = vecMEndIdx % runInfo.actS1Size;
            int32_t gStartIdx = vecMStartIdx / runInfo.actS1Size;
            int32_t gEndIdx = vecMEndIdx / runInfo.actS1Size;
            if (gStartIdx == gEndIdx) {
                // 只跨1个G
                ret = (s1StartTdx < s1FirstValidToken) || (s1EndTdx > s1LastValidToken);
            } else {
                // 跨多个G
                ret = (s1StartTdx < s1FirstValidToken);
                ret = ret || (s1EndTdx < s1FirstValidToken) || (s1EndTdx > s1LastValidToken);
            }
        }
        return ret;
    }

    template <typename VEC2_RES_T>
    __aicore__ inline void RowInvalid(LocalTensor<VEC2_RES_T> &vec2ResUb, int64_t mStartVec, int64_t mDealSize,
                                      RunInfoXVec &runInfo, int64_t dSizeAligned64)
    {
        if constexpr (hasAtten) {
            int64_t s1FirstValidToken = AttentionCommon::Min(AttentionCommon::Max(-runInfo.nextTokensLeftUp, 0), runInfo.actS1Size);
            int64_t s1LastValidToken = AttentionCommon::Min(AttentionCommon::Max(runInfo.preTokensLeftUp + runInfo.actS2Size, 0), runInfo.actS1Size);
            s1LastValidToken = AttentionCommon::Max(s1LastValidToken - 1, 0);
            bool hasValidRow = (s1FirstValidToken > 0) || (s1LastValidToken < runInfo.actS1Size);
            bool batchNeedRowInvalid = ((constInfoXVec.sparseMode != SparseMode::LEFT_UP_CAUSAL) &&
                                        hasValidRow); // sparse = 0 or 3 or 4，preToekens or nextTokens负数
            if (!batchNeedRowInvalid) {
                return;
            }

            bool blockNeedRowInvalid = CalcBlockNeedRowInvalid(runInfo, s1FirstValidToken, s1LastValidToken);

            if (blockNeedRowInvalid) {
                LocalTensor<float> maxTensor =
                    softmaxMaxBuf[runInfo.mloop % (PRELOAD_N + 1)].template Get<float>()[mStartVec];
                RowInvalidUpdateVF<float>(vec2ResUb, maxTensor, mDealSize, constInfoXVec.dSizeV,
                                          static_cast<uint32_t>(dSizeAligned64));
            }
        }
    }

    __aicore__ inline void Bmm2DataCopyOutTrans(const RunInfoXVec &info, LocalTensor<OUTPUT_T> &attenOutUb,
                                                uint32_t vecMIdx, uint32_t dealRowCount)
    {
        FaUbTensor<OUTPUT_T> ubTensor{.tensor = attenOutUb,
                                      .rowCount = dealRowCount,
                                      .colCount = (uint32_t)(splitD ? constInfoXVec.dBasicBlock : dTemplateAlign64)};
        GmCoord gmCoord{.bIdx = info.bIdx,
                        .n2Idx = info.n2Idx,
                        .gS1Idx = info.gS1Idx + info.vecMbaseIdx + vecMIdx,
                        .dIdx = 0,
                        .gS1DealSize = dealRowCount,
                        .dDealSize = (uint32_t)constInfoXVec.dSizeV};
        CopyAttentionOut(ubTensor, gmCoord);
    }

    __aicore__ inline void CopyAttentionOut(FaUbTensor<OUTPUT_T> &ubTensor, GmCoord &gmCoord)
    {
        if constexpr (outLayout == LayOutTypeEnum::LAYOUT_BSH) {
            constexpr GmFormat OUT_FORMAT = GmFormat::BSNGD;
            FaGmTensor<OUTPUT_T, OUT_FORMAT, int32_t> outGmTensor;
            outGmTensor.gmTensor = attentionOutGm;
            outGmTensor.offsetCalculator.Init(constInfoXVec.bSize, constInfoXVec.n2Size, constInfoXVec.gSize, constInfoXVec.s1Size,
                                              constInfoXVec.dSizeV, qSeqParser);
            CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, GetOutUbFormat<layout>()> copyAttenOutUbToGm;
            copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
        } else if constexpr (outLayout == LayOutTypeEnum::LAYOUT_BNSD) {
            constexpr GmFormat OUT_FORMAT = GmFormat::BNGSD;
            FaGmTensor<OUTPUT_T, OUT_FORMAT, int32_t> outGmTensor;
            outGmTensor.gmTensor = attentionOutGm;
            outGmTensor.offsetCalculator.Init(constInfoXVec.bSize, constInfoXVec.n2Size, constInfoXVec.gSize, constInfoXVec.s1Size,
                                              constInfoXVec.dSizeV, qSeqParser);
            CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, GetOutUbFormat<layout>()> copyAttenOutUbToGm;
            copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
        } else if constexpr (outLayout == LayOutTypeEnum::LAYOUT_TND) {
            constexpr GmFormat OUT_FORMAT = GmFormat::TNGD;
            FaGmTensor<OUTPUT_T, OUT_FORMAT, int32_t, true> outGmTensor;
            outGmTensor.gmTensor = attentionOutGm;
            outGmTensor.offsetCalculator.Init(constInfoXVec.n2Size, constInfoXVec.gSize, constInfoXVec.dSizeV,
                                              qSeqParser);
            CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, GetOutUbFormat<layout>()> copyAttenOutUbToGm;
            copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
        }
    }

    __aicore__ inline void BroadCastAndCopyOut(const RunInfoXVec &runInfo, LocalTensor<float> &sumUb,
                                               LocalTensor<float> &maxUb, int64_t gmOffset, int64_t calculateSize)
    {
        // Copy sum to gm
        LocalTensor<float> sumOutTensor = sumBrdcst.template AllocTensor<float>();
        FaVectorApi::BroadcastMaxSum(sumOutTensor, sumUb, runInfo.actVecMSize);
        sumBrdcst.template EnQue(sumOutTensor);
        sumBrdcst.template DeQue<float>();
        DataCopy(softmaxFDSumGm[gmOffset], sumOutTensor, calculateSize);
        sumBrdcst.template FreeTensor(sumOutTensor);

        // Copy max to gm
        LocalTensor<float> maxOutTensor = maxBrdcst.template AllocTensor<float>();
        FaVectorApi::BroadcastMaxSum(maxOutTensor, maxUb, runInfo.actVecMSize);
        maxBrdcst.template EnQue(maxOutTensor);
        maxBrdcst.template DeQue<float>();
        DataCopy(softmaxFDMaxGm[gmOffset], maxOutTensor, calculateSize);
        maxBrdcst.template FreeTensor(maxOutTensor);
    }

    __aicore__ inline void ComputeLogSumExpAndCopyToGm(const RunInfoXVec &runInfo, LocalTensor<float> &sumUb,
                                                       LocalTensor<float> &maxUb)
    {
        if (unlikely(runInfo.actVecMSize == 0)) {
            return;
        }
        int64_t calculateSize = runInfo.actVecMSize * fp32BaseSize;
        int64_t gmOffset = runInfo.faTmpOutWsPos * mBaseSize * fp32BaseSize + runInfo.vecMbaseIdx * fp32BaseSize;
        // Copy sum to gm
        BroadCastAndCopyOut(runInfo, sumUb, maxUb, gmOffset, calculateSize);
    }

    __aicore__ inline void Bmm2ResForFDCopyOut(const RunInfoXVec &runInfo, LocalTensor<T> &vec2ResUb, uint32_t mStartVec,
                                               uint32_t mDealSize)
    {
        int64_t dSizeAligned64 = (int64_t)dVTemplateType;
        if constexpr (splitD) {
            dSizeAligned64 = constInfoXVec.dBasicBlock;
        }
        SetFlag<HardEvent::V_MTE3>(vToMte3Id[runInfo.loop & (DB - 1)]);
        WaitFlag<HardEvent::V_MTE3>(vToMte3Id[runInfo.loop & (DB - 1)]);
        uint64_t gmOffset =
            runInfo.faTmpOutWsPos * mBaseSize * constInfoXVec.dSizeV + (runInfo.vecMbaseIdx + mStartVec) * constInfoXVec.dSizeV;

        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = mDealSize;
        dataCopyParams.blockLen = constInfoXVec.dSizeV * sizeof(T);
        dataCopyParams.srcStride = (dSizeAligned64 - constInfoXVec.dSizeV) / (FA_BYTE_BLOCK / sizeof(T));
        dataCopyParams.dstStride = 0;

        DataCopyPad(accumOutGm[gmOffset], vec2ResUb, dataCopyParams);
    }

    __aicore__ inline void ProcessVec2(mm2ResPos &bmm2ResBuf, RunInfoXVec runInfo)
    {
#ifdef SKIP_V2
        bmm2ResBuf.WaitCrossCore();
        if constexpr (bmm2Write2Ub) {
            bmm2ResBuf.SetCrossCore();
        }
        return;
#endif

        bmm2ResBuf.WaitCrossCore();

        if constexpr (bmm2Write2Ub) {
            ProcessVec2OnUb(bmm2ResBuf, runInfo);
        } else if constexpr (splitD) {
            GlobalTensor<T> mmRes = bmm2ResBuf.template GetTensor<T>();
            ProcessVec2DSplit(mmRes, runInfo);
        } else {
            GlobalTensor<T> mmRes = bmm2ResBuf.template GetTensor<T>();
            ProcessVec2_C2GmUpdateUb(mmRes, runInfo);
        }
        return;
    }

    __aicore__ inline void ProcessVec2_C2GmUpdateUb(GlobalTensor<T> &mmRes, RunInfoXVec runInfo)
    {
        // bmm2 result is on GM and global update data on UB
        uint32_t subMBase = 8192 / dTemplateAlign64;
        int64_t vec2LoopLimit = CeilDiv(runInfo.actVecMSize, subMBase);
        LocalTensor<T> vec2ResUb = this->stage2OutBuf.template Get<T>();
        WaitFlag<HardEvent::MTE3_V>(mte3ToVId[0]);
        LocalTensor<T> bmm2Ub = mm2InBuf.template Get<T>();

        event_t mte2ToV = static_cast<event_t>(pipe->FetchEventID(HardEvent::MTE2_V));
        event_t vToMte2 = static_cast<event_t>(pipe->FetchEventID(HardEvent::V_MTE2));

        for (int64_t mOIdx = 0; mOIdx < vec2LoopLimit; mOIdx++) {
            int32_t mIdx = mOIdx * subMBase;
            uint32_t subMRealsize = subMBase;
            if (mOIdx == vec2LoopLimit - 1) {
                subMRealsize = runInfo.actVecMSize - mIdx;
            }

            int64_t vec2CalcSize = subMRealsize * dTemplateAlign64;
            // Gm地址偏移是按照实际DSize实现的，虽然我们设置了KC=192
            int64_t mm2ResInnerOffset = mIdx * constInfoXVec.dSizeV;
            SetFlag<HardEvent::V_MTE2>(vToMte2);
            WaitFlag<HardEvent::V_MTE2>(vToMte2);
            if (constInfoXVec.dSizeV == dTemplateAlign64) {
                if constexpr (useDn) {
                    DataCopy(bmm2Ub, mmRes[mm2ResInnerOffset], vec2CalcSize);
                } else {
                    DataCopy(bmm2Ub,
                             mmRes[constInfoXVec.subBlockIdx * (runInfo.actMSize - runInfo.actVecMSize) * constInfoXVec.dSizeV +
                                   mm2ResInnerOffset],
                             vec2CalcSize);
                }
            } else {
                DataCopyParams dataCopyParams;
                DataCopyPadParams dataCopyPadParams;
                dataCopyParams.blockCount = subMRealsize;
                dataCopyParams.dstStride = (dTemplateAlign64 - constInfoXVec.dSizeV) * sizeof(T) / blockBytes;
                dataCopyParams.srcStride = 0;
                dataCopyParams.blockLen = constInfoXVec.dSizeV * sizeof(T);
                if constexpr (useDn) {
                    DataCopyPad(bmm2Ub, mmRes[mm2ResInnerOffset], dataCopyParams,
                                dataCopyPadParams);
                } else {
                    DataCopyPad(
                        bmm2Ub,
                        mmRes[constInfoXVec.subBlockIdx * (runInfo.actMSize - runInfo.actVecMSize) * constInfoXVec.dSizeV +
                              mm2ResInnerOffset],
                        dataCopyParams, dataCopyPadParams);
                }
            }
            SetFlag<HardEvent::MTE2_V>(mte2ToV);
            WaitFlag<HardEvent::MTE2_V>(mte2ToV);
            // 经过了跳读，UB上每行是按照dTemplateAlign64对齐的
            LocalTensor vec2ResInner = vec2ResUb[mIdx * dTemplateAlign64];

            if (unlikely(runInfo.isFirstS2Loop)) {
                DataCopy(vec2ResInner, bmm2Ub, vec2CalcSize);
            } else {
                int64_t vec2ExpBufOffset = mIdx;
                LocalTensor<T> expUb =
                    softmaxExpBuf[runInfo.loop % (PRELOAD_N + 1)].template Get<T>()[vec2ExpBufOffset];
                if (!runInfo.isLastS2Loop) {
                    FlashUpdateNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, false, false>(
                        vec2ResInner, bmm2Ub, vec2ResInner, expUb, expUb, subMRealsize, dTemplateAlign64, 0.0F,
                        1.0f);
                } else {
                    int64_t vec2SumBufOffset = mIdx;
                    LocalTensor<float> sumUb =
                        softmaxSumBuf[runInfo.mloop % (PRELOAD_N + 1)].template Get<float>()[vec2SumBufOffset];
                    FlashUpdateLastNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, false, false>(
                        vec2ResInner, bmm2Ub, vec2ResInner, expUb, expUb, sumUb, subMRealsize, dTemplateAlign64, 0.0F,
                        1.0f);
                }
            }

            if (unlikely(runInfo.isLastS2Loop)) {
                if (unlikely(runInfo.isFirstS2Loop)) {
                    int64_t vec2SumBufOffset = mIdx;
                    LocalTensor<float> sumUb =
                        softmaxSumBuf[runInfo.mloop % (PRELOAD_N + 1)].template Get<float>()[vec2SumBufOffset];
                    LastDivNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, false>(vec2ResInner, vec2ResInner, sumUb,
                                                                              subMRealsize, dTemplateAlign64, 0.0F);
                }
                CopyOutAttentionOut(runInfo, vec2ResInner, mIdx, subMRealsize);
            }
        }
        SetFlag<HardEvent::MTE3_V>(mte3ToVId[0]);
    }

    __aicore__ inline void SoftmaxInitBuffer()
    {
        pipe->InitBuffer(softmaxSumBuf[0], 256); // [64, 1]
        pipe->InitBuffer(softmaxSumBuf[1], 256); // [64, 1]
        pipe->InitBuffer(softmaxSumBuf[2], 256); // [64, 1]

        pipe->InitBuffer(softmaxMaxBuf[0], 256); // [64, 1]
        pipe->InitBuffer(softmaxMaxBuf[1], 256); // [64, 1]
        pipe->InitBuffer(softmaxMaxBuf[2], 256); // [64, 1]

        pipe->InitBuffer(softmaxExpBuf[0], 256); // [64, 1]
        pipe->InitBuffer(softmaxExpBuf[1], 256); // [64, 1]
        pipe->InitBuffer(softmaxExpBuf[2], 256); // [64, 1]

        pipe->InitBuffer(maxBrdcst, 1, 2048); // [64, 8]
        pipe->InitBuffer(sumBrdcst, 1, 2048); // [64, 8]
    }

    __aicore__ inline void InitBuffers(TQue<QuePosition::VECOUT, 1> SharedBuffer1[],
                                       TQue<QuePosition::VECIN, 1> SharedBuffer2[], TBuf<> &SharedBuffer3)
    {
        uint32_t mm1ResultSize = mBaseSize / CV_RATIO * s2BaseSize * sizeof(T);
        uint32_t mm2ResultSize = mBaseSize / CV_RATIO * dTemplateAlign64 * sizeof(T);
        if constexpr (!bmm2Write2Ub) {
            pipe->InitBuffer(mm2InBuf, 32768); // bmm2结果在Gm，vector2开启多层循环，每次处理32KB
        }
        pipe->InitBuffer(SharedBuffer2[0], 1, 8192);
        pipe->InitBuffer(SharedBuffer2[1], 1, 8192);
        pipe->InitBuffer(SharedBuffer3, 32768);
        pipe->InitBuffer(SharedBuffer1[0], 1, 33024);
        pipe->InitBuffer(SharedBuffer1[1], 1, 33024);
        if constexpr (s2BaseSize == 256) {
            if constexpr (mBaseSize == 128) {
                SoftmaxInitBuffer();
            } else {
                SoftmaxInitBuffer();
                pipe->InitBuffer(commonTBuf, 512);
            }
        } else {
            SoftmaxInitBuffer();
            if constexpr (!useDn) {
                pipe->InitBuffer(commonTBuf, 512);
            }
        }

        if (constInfoXVec.isSoftmaxLseEnable) {
            // 8: 适配TND，每行的结果存为8个重复lse元素（32B对齐）
            this->pipe->InitBuffer(softmaxLseQueue, 1, (mBaseSize >> 1U) * sizeof(float) * 8);
        }

        if (constInfoXVec.learnableSinkFlag) {
            this->pipe->InitBuffer(sinkQue, 1, 256); // buffer size = 256 bytes
        }
        stage1OutQue[0] = SharedBuffer1[0];
        stage1OutQue[1] = SharedBuffer1[1];
        attenMaskInQue[0] = SharedBuffer2[0];
        attenMaskInQue[1] = SharedBuffer2[1];
        stage2OutBuf = SharedBuffer3;
    }

    __aicore__ inline void AllocEventID()
    {
        mte3ToVId[0] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
        mte3ToVId[1] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
        vToMte3Id[0] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
        vToMte3Id[1] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
        SetFlag<HardEvent::MTE3_V>(mte3ToVId[0]);
        SetFlag<HardEvent::MTE3_V>(mte3ToVId[1]);
    }

    __aicore__ inline void FreeEventID()
    {
        WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToVId[0]);
        WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToVId[1]);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(mte3ToVId[0]);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(mte3ToVId[1]);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(vToMte3Id[0]);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(vToMte3Id[1]);
    }

    __aicore__ inline void AttenMaskCopyIn(LocalTensor<uint8_t> attenMaskUb, uint32_t vecMIdx, uint32_t mDealSize,
                                           RunInfoXVec &runInfo)
    {
        MaskInfo maskInfo;
        maskInfo.gs1StartIdx = runInfo.gS1Idx + runInfo.vecMbaseIdx + vecMIdx;
        maskInfo.gs1dealNum = mDealSize;
        maskInfo.s1Size = runInfo.actS1Size;
        maskInfo.gSize = constInfoXVec.gSize;
        maskInfo.s2StartIdx = runInfo.s2Idx;
        maskInfo.s2dealNum = runInfo.actSingleLoopS2Size;
        maskInfo.s2Size = runInfo.actS2Size;
        maskInfo.nBaseSize = s2BaseSize;
        maskInfo.preToken = constInfoXVec.preTokens;
        maskInfo.nextToken = constInfoXVec.nextTokens;
        maskInfo.sparseMode = static_cast<SparseMode>(constInfoXVec.sparseMode);
        maskInfo.batchIdx = (constInfoXVec.attenMaskBatch == 1) ? 0 : runInfo.bIdx;
        maskInfo.attenMaskBatchStride = constInfoXVec.attenMaskS1Size * constInfoXVec.attenMaskS2Size;
        maskInfo.attenMaskS1Stride = constInfoXVec.attenMaskS2Size;
        maskInfo.attenMaskDstStride = (s2BaseSize - AttentionCommon::Align(maskInfo.s2dealNum, 32U)) / 32;
        maskInfo.maskValue = negativeIntScalar;
        maskInfo.s1LeftPaddingSize = 0;
        maskInfo.s2LeftPaddingSize = 0;
        maskInfo.layout = MASK_LAYOUT;
        maskInfo.attenMaskType = MASK_BOOL; // compatible with int8/uint8

        bool IsSkipMask = IsSkipAttentionmask(maskInfo);
        bool IsSkipMaskForPre = IsSkipAttentionmaskForPre(maskInfo);

        if (IsSkipMask && IsSkipMaskForPre) {
            Duplicate(attenMaskUb, static_cast<uint8_t>(0U), maskInfo.gs1dealNum * s2BaseSize);
            return;
        }

        if (!IsSkipMask) {
            AttentionmaskCopyIn<uint8_t, MASK_LAYOUT, true, s2BaseSize>(attenMaskUb, attenMaskGmInt, maskInfo);
        } else {
            Duplicate(attenMaskUb, static_cast<uint8_t>(0U), maskInfo.gs1dealNum * s2BaseSize);
        }

        if (!IsSkipMaskForPre) {
            LocalTensor<uint8_t> attenMaskUbPre =
                this->attenMaskInQue[1 - (runInfo.loop & (DB - 1))].template AllocTensor<uint8_t>();
            AttentionmaskCopyIn<uint8_t, MASK_LAYOUT, true, s2BaseSize>(attenMaskUbPre, attenMaskGmInt, maskInfo, true);
            MergeMask(attenMaskUb, attenMaskUbPre, maskInfo.gs1dealNum, s2BaseSize);
            this->attenMaskInQue[1 - (runInfo.loop & (DB - 1))].template FreeTensor(attenMaskUbPre);
        }
    }

    __aicore__ inline void DealZeroActSeqLen(uint32_t bN2Cur)
    {
        uint32_t n2Idx = bN2Cur % constInfoXVec.n2Size;
        uint32_t bIdx = bN2Cur / constInfoXVec.n2Size;
        if constexpr (outLayout == LayOutTypeEnum::LAYOUT_BSH) {
            OffsetCalculator<GmFormat::BSNGD, int32_t> offsetCalculator;
            offsetCalculator.Init(constInfoXVec.bSize, constInfoXVec.n2Size, constInfoXVec.gSize, constInfoXVec.s1Size,
                                  constInfoXVec.dSizeV, qSeqParser);
            DealActSeqLenIsZero<GmFormat::BSNGD, OUTPUT_T>(bIdx, n2Idx, offsetCalculator, attentionOutGm);
        } else if constexpr (outLayout == LayOutTypeEnum::LAYOUT_BNSD) {
            OffsetCalculator<GmFormat::BNGSD, int32_t> offsetCalculator;
            offsetCalculator.Init(constInfoXVec.bSize, constInfoXVec.n2Size, constInfoXVec.gSize, constInfoXVec.s1Size,
                                  constInfoXVec.dSizeV, qSeqParser);
            DealActSeqLenIsZero<GmFormat::BNGSD, OUTPUT_T>(bIdx, n2Idx, offsetCalculator, attentionOutGm);
        } else if constexpr (outLayout == LayOutTypeEnum::LAYOUT_TND) {
            OffsetCalculator<GmFormat::TNGD, int32_t, true> offsetCalculator;
            offsetCalculator.Init(constInfoXVec.n2Size, constInfoXVec.gSize, constInfoXVec.dSizeV, qSeqParser);
            DealActSeqLenIsZero<GmFormat::TNGD, OUTPUT_T>(bIdx, n2Idx, offsetCalculator, attentionOutGm);
        }
    }

    // ==================== Kernel Orchestrator Methods ====================

private:
    __aicore__ inline void InitCommon(__gm__ uint8_t *key,
                                       __gm__ uint8_t *cuSeqLensQ, __gm__ uint8_t *cuSeqLensKv,
                                       __gm__ uint8_t *seqUsedQ, __gm__ uint8_t *seqUsedKv,
                                       __gm__ uint8_t *&workspace, __gm__ uint8_t *fiaMetaData,
                                       const __gm__ FlashAttnNoQuantTilingArch35 *__restrict tiling, TPipe *tPipe)
    {
        this->pipe = tPipe;

        InitConstInfo(tiling);

        keyPtr = key;

        GlobalTensor<int32_t> localCuSeqLensGmQ;
        GlobalTensor<int32_t> localSeqUsedGmQ;
        GlobalTensor<int32_t> localCuSeqLensGmKv;
        GlobalTensor<int32_t> localSeqUsedGmKv;

        if constexpr (IS_TND_LAYOUT) {
            localCuSeqLensGmQ.SetGlobalBuffer((__gm__ int32_t *)cuSeqLensQ, constInfoXVec.cuSeqLensQSize + 1);
            localSeqUsedGmQ.SetGlobalBuffer((__gm__ int32_t *)seqUsedQ, constInfoXVec.seqUsedQSize);
            localCuSeqLensGmKv.SetGlobalBuffer((__gm__ int32_t *)cuSeqLensKv, constInfoXVec.cuSeqLensKVSize + 1);
            localSeqUsedGmKv.SetGlobalBuffer((__gm__ int32_t *)seqUsedKv, constInfoXVec.seqUsedKvSize);
        } else {
            localSeqUsedGmQ.SetGlobalBuffer((__gm__ int32_t *)seqUsedQ, constInfoXVec.seqUsedQSize);
            localSeqUsedGmKv.SetGlobalBuffer((__gm__ int32_t *)seqUsedKv, constInfoXVec.seqUsedKvSize);
        }

        InitQSeqParser(localCuSeqLensGmQ, localSeqUsedGmQ);
        InitKvSeqParser(localCuSeqLensGmKv, localSeqUsedGmKv);

        sectionNum_ = ((__gm__ uint32_t *)fiaMetaData)[0];
        isFd = static_cast<bool>(((__gm__ uint32_t *)fiaMetaData)[1]);

        faMetaDataGm.SetGlobalBuffer((__gm__ uint32_t *)(fiaMetaData + FA_METADATA_HEADER_OFFSET),
                                      FA_AIC_CORE_NUM * 16U * sectionNum_);
        fdMetaDataGm.SetGlobalBuffer(
            (__gm__ uint32_t *)(fiaMetaData + FA_METADATA_HEADER_OFFSET +
                                FLASH_ATTN_METADATA_SIZE * FA_AIC_CORE_NUM * sectionNum_ * sizeof(uint32_t)),
            FA_AIV_CORE_NUM * 16U * sectionNum_);

        InitMMResBuf(workspace);
    }

public:
    __aicore__ inline void InitVec(__gm__ uint8_t *key,
                                   __gm__ uint8_t *cuSeqLensQ, __gm__ uint8_t *cuSeqLensKv,
                                   __gm__ uint8_t *seqUsedQ, __gm__ uint8_t *seqUsedKv,
                                   __gm__ uint8_t *workspace, __gm__ uint8_t *fiaMetaData,
                                   const __gm__ FlashAttnNoQuantTilingArch35 *__restrict tiling, TPipe *tPipe,
                                   __gm__ uint8_t *attenMask, __gm__ uint8_t *learnableSink,
                                   __gm__ uint8_t *softmaxLse, __gm__ uint8_t *attentionOut)
    {
        __gm__ uint8_t *ws = workspace;
        InitCommon(key, cuSeqLensQ, cuSeqLensKv, seqUsedQ, seqUsedKv, ws, fiaMetaData, tiling, tPipe);
        uint32_t tmp1 = NEGATIVE_MIN_VALUE_FP32;
        this->negativeFloatScalar = *((T *)&tmp1);
        InitVecInput(attenMask, learnableSink, softmaxLse, attentionOut, ws);
        ClearOutput();
        if (isFd) {
            vecFdBlock.InitParams();
            vecFdBlock.InitGlobalTensor(softmaxFDMaxGm, softmaxFDSumGm,
                                        accumOutGm, attentionOutGm, keyPtr);
            if (learnableSink != nullptr) {
                constInfoXVec.learnableSinkFlag = true;
                vecFdBlock.InitLearnableSinkGm(sinkGm);
            }
            if (constInfoXVec.isSoftmaxLseEnable) {
                vecFdBlock.InitSoftmaxLseGm(softmaxLseGm);
            }
            vecFdBlock.SetCuSeqLensParsers(qSeqParser, kvSeqParser);
        }
    }

    __aicore__ inline void InitMMResBuf(__gm__ uint8_t *&workspace)
    {
        uint32_t mm1OutDtype = sizeof(T);

        uint32_t mm1ResultSize = mBaseSize / CV_RATIO * s2BaseSize * mm1OutDtype;
        constexpr uint32_t mm2ResultSize = mBaseSize / CV_RATIO * dVBaseSize * sizeof(T);
        constexpr uint32_t mm2LeftSize = mBaseSize * s2BaseSize * sizeof(INPUT_T);
        l1BufferManager.Init(pipe, 524288); // 512 * 1024
        // 保存p结果的L1内存必须放在第一个L1 policy上，保证和vec申请的地址相同
        l1PBuffers.Init(l1BufferManager, mm2LeftSize);
        if constexpr (BMM2_TOUB) {
            ubBufferManager.Init(pipe, mm1ResultSize * 2 + mm2ResultSize * 2);
            bmm2Buffers.Init(ubBufferManager, mm2ResultSize);
        } else {
            ubBufferManager.Init(pipe, mm1ResultSize * 2);
        }
        bmm1Buffers.Init(ubBufferManager, mm1ResultSize);

        // GM Buffer
        if constexpr (!BMM2_TOUB) {
            int64_t mm2ResultSize =
                mBaseSize * constInfoXVec.dBasicBlock; // 使用Cube计算的总大小，Gm上的数据按照实际的dSize存储
            int64_t prevCoretotalOffset = constInfoXVec.aicIdx * 3 * mm2ResultSize; // 3为preload次数
            // SameB模式下V0和V1调用IterateAll的时候填写的地址相同
            gmBufferManager.Init(workspace + prevCoretotalOffset * sizeof(T));
            bmm2ResGmBuffers.Init(gmBufferManager, mm2ResultSize * sizeof(T));
            workspace = workspace + constInfoXVec.coreNum * 3 * mm2ResultSize * sizeof(T);
        }
    }

    __aicore__ inline void InitQSeqParser(GlobalTensor<int32_t> &cuSeqLensGmQ, GlobalTensor<int32_t> &seqUsedGmQ)
    {
        if constexpr (IS_TND_LAYOUT) {
            qSeqParser.Init(cuSeqLensGmQ, seqUsedGmQ, constInfoXVec.cuSeqLensQSize, constInfoXVec.seqUsedQSize);
        } else {
            qSeqParser.Init(seqUsedGmQ, constInfoXVec.seqUsedQSize, constInfoXVec.s1Size);
        }
    }

    __aicore__ inline void InitKvSeqParser(GlobalTensor<int32_t> &cuSeqLensGmKv, GlobalTensor<int32_t> &seqUsedGmKv)
    {
        if constexpr (IS_TND_LAYOUT) {
            kvSeqParser.Init(cuSeqLensGmKv, seqUsedGmKv, constInfoXVec.cuSeqLensKVSize, constInfoXVec.seqUsedKvSize);
        } else {
            kvSeqParser.Init(seqUsedGmKv, constInfoXVec.seqUsedKvSize, constInfoXVec.s2Size);
        }
    }

    __aicore__ inline void InitConstInfo(const __gm__ FlashAttnNoQuantTilingArch35 *__restrict tiling)
    {
        constInfoXVec.aivIdx = GetBlockIdx();
        constInfoXVec.aicIdx = GetBlockIdx() / GetSubBlockNum();
        constInfoXVec.subBlockIdx = GetSubBlockIdx();
        constInfoXVec.bSize = tiling->flashAttnBaseParams.bSize;
        constInfoXVec.t1Size = tiling->flashAttnBaseParams.t1Size;
        constInfoXVec.n2Size = tiling->flashAttnBaseParams.n2Size;
        constInfoXVec.gSize = tiling->flashAttnBaseParams.gSize;
        constInfoXVec.s1Size = tiling->flashAttnBaseParams.s1Size;
        constInfoXVec.s2Size = tiling->flashAttnBaseParams.s2Size;
        constInfoXVec.dSizeV = tiling->flashAttnBaseParams.dSizeV;
        constInfoXVec.cuSeqLensQSize = tiling->flashAttnBaseParams.cuSeqLensQSize;
        constInfoXVec.cuSeqLensKVSize = tiling->flashAttnBaseParams.cuSeqLensKVSize;
        constInfoXVec.seqUsedQSize = tiling->flashAttnBaseParams.seqUsedQSize;
        constInfoXVec.seqUsedKvSize = tiling->flashAttnBaseParams.seqUsedKvSize;
        constInfoXVec.scaleValue = static_cast<float>(tiling->flashAttnBaseParams.scaleValue);
        constInfoXVec.isKvContinuous = tiling->flashAttnBaseParams.isKvContinuous != 0;
        constInfoXVec.coreNum = tiling->flashAttnBaseParams.coreNum;
        constInfoXVec.needInitOutput = tiling->flashAttnBaseParams.needInitOutput;

        constInfoXVec.sparseMode = tiling->flashAttnAttenMaskParams.sparseMode;
        constInfoXVec.preTokens = tiling->flashAttnAttenMaskParams.winLefts;
        constInfoXVec.nextTokens = tiling->flashAttnAttenMaskParams.winRights;
        constInfoXVec.attenMaskBatch = tiling->flashAttnAttenMaskParams.attenMaskBatch;
        constInfoXVec.attenMaskS1Size = tiling->flashAttnAttenMaskParams.attenMaskS1Size;
        constInfoXVec.attenMaskS2Size = tiling->flashAttnAttenMaskParams.attenMaskS2Size;

        constInfoXVec.accumOutSize = tiling->flashAttnWorkspaceParams.accumOutSize;
        constInfoXVec.logSumExpSize = tiling->flashAttnWorkspaceParams.logSumExpSize;

        constInfoXVec.isSoftmaxLseEnable = tiling->flashAttnBaseParams.isSoftMaxLseEnable;

        constInfoXVec.dBasicBlock = Align64Func((uint16_t)constInfoXVec.dSizeV);
    }

    __aicore__ inline uint32_t GetFAMetaDataIndex(uint32_t coreIdx, uint32_t metaIdx, uint32_t sectionIdx)
    {
        // AICPU metadata format: 16 fields per AIC core, 0-indexed (no leading CORE_ENABLE).
        // Kernel field constants ( FLASH_ATTN_BN2_START_INDEX=1, etc.) are 1-based, so subtract 1.
        return FLASH_ATTN_METADATA_SIZE * FA_AIC_CORE_NUM * sectionIdx + 16U * coreIdx + metaIdx;
    }
    __aicore__ inline uint32_t GetFDMetaDataIndex(uint32_t coreIdx, uint32_t metaIdx, uint32_t sectionIdx)
    {
        return FA_FD_METADATA_SIZE * FA_AIV_CORE_NUM * sectionIdx + FA_FD_METADATA_SIZE * coreIdx + metaIdx;
    }

    __aicore__ inline void CrossCoreBufferInit()
    {
        if constexpr (BMM2_TOUB) {
            bmm2Buffers.Get().SetCrossCore();
            bmm2Buffers.Get().SetCrossCore();
        }
        bmm1Buffers.Get().SetCrossCore();
        bmm1Buffers.Get().SetCrossCore();
    }

    __aicore__ inline void FlashAttention(uint32_t sectionIdx)
    {
        if (constInfoXVec.aicIdx >= constInfoXVec.coreNum) {
            return;
        }

        GetFASectionInfo(sectionIdx);
        RunInfoXVec taskRunInfo[PRELOAD_TASK_CACHE_SIZE] = {};

        // Reset pipeline state for each section to avoid cross-section deadlock
        createdTaskCount = 0;
        uint32_t executedTaskCount = 0;
        mloop = 0;
        headS2Split = false;
        tailS2Split = false;

        uint32_t bN2Cur = bN2Start_;
        uint32_t gS1Cur = gS1OStart_;
        uint32_t s2Cur = s2OStart_;
        prevBN2Idx = bN2Cur;
        prevGS1Idx = gS1Cur;

        bool shouldDispatchTask = true;
        uint32_t validTaskCount = 0; // 未执行(有效)的任务数
        while (shouldDispatchTask || validTaskCount) {
            // 分发任务
            shouldDispatchTask = ShouldDispatchTask(bN2Cur, gS1Cur, s2Cur);
            if (shouldDispatchTask) {
                TASK_DEAL_MODE taskDealMode = GetTaskDealMode(bN2Cur, gS1Cur, s2Cur);
                if (taskDealMode == TASK_DEAL_MODE::CREATE_TASK) {
                    // 创建任务
                    CreateTask(createdTaskCount, bN2Cur, gS1Cur, s2Cur, taskRunInfo);
                    createdTaskCount++;
                    validTaskCount++;
                    UpdateAxisInfo(taskDealMode, bN2Cur, gS1Cur, s2Cur);
                } else if (taskDealMode == TASK_DEAL_MODE::DEAL_ZERO) {
                    DealZeroActSeqLen(bN2Cur);
                    UpdateAxisInfo(taskDealMode, bN2Cur, gS1Cur, s2Cur);
                    continue;
                } else {
                    UpdateAxisInfo(taskDealMode, bN2Cur, gS1Cur, s2Cur);
                    continue;
                }
            }
            // 执行任务
            if (validTaskCount) {
                ExecuteTask(executedTaskCount, taskRunInfo);
                executedTaskCount++;
                if (executedTaskCount > PRELOAD_N) {
                    validTaskCount--;
                }
            }
        }
    }

    __aicore__ inline bool ShouldDispatchTask(uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur)
    {
        return ((bN2Cur != bN2End_) || (gS1Cur != gS1OEnd_) || (s2Cur != s2OEnd_));
    }

    __aicore__ inline TASK_DEAL_MODE GetTaskDealMode(uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur)
    {
        bool isFirstTask = (bN2Cur == bN2Start_) && (gS1Cur == gS1OStart_) && (s2Cur == s2OStart_);
        uint32_t bIdx = bN2Cur / constInfoXVec.n2Size;
        if (isFirstTask || prevBIdx != bIdx) {
            prevBIdx = bIdx;
            if (constInfoXVec.cuSeqLensKVSize == 0 && constInfoXVec.seqUsedKvSize == 0 && !constInfoXVec.isKvContinuous) {
                actSeqLensKv = SeqLenFromTensorList<layout>(keyPtr, bIdx);
            } else {
                actSeqLensKv = kvSeqParser.GetActualSeqLength(bIdx);
            }
            actSeqLensQ = qSeqParser.GetActualSeqLength(bIdx);
        }
        uint64_t s2LoopTimes = (actSeqLensKv + s2BaseSize - 1) / s2BaseSize;
        uint64_t gS1Size = actSeqLensQ * constInfoXVec.gSize;
        uint64_t gS1LoopTimes = (gS1Size + mBaseSize - 1) / mBaseSize;
        if (s2LoopTimes == 0 || gS1LoopTimes == 0) {
            if (gS1Cur == 0 && s2Cur == 0) {
                return TASK_DEAL_MODE::DEAL_ZERO;
            }
            return TASK_DEAL_MODE::SKIP_ZERO;
        }
        // 计算每一行的起止点，只有当换行时（bN2Cur、gS1Cur更新）才需要重新计算
        if (isFirstTask || bN2Cur != prevBN2Idx || gS1Cur != prevGS1Idx) {
            if constexpr (!HAS_MASK) {
                CalcCurS2StartEndNoSparse(bN2Cur, gS1Cur);
            } else {
                CalcCurS2StartEndWithSparse(bN2Cur, gS1Cur);
            }
            prevBN2Idx = bN2Cur;
            prevGS1Idx = gS1Cur;
        }

        if (curS2Start < curS2End && s2Cur >= curS2End) {
            return TASK_DEAL_MODE::S2_END;
        }

        if (s2Cur < curS2Start || s2Cur >= curS2End) {
            return TASK_DEAL_MODE::SKIP;
        }

        if (s2Cur == curS2Start) {
            mloop++;
        }

        return TASK_DEAL_MODE::CREATE_TASK;
    }

    __aicore__ inline void GetPreNextTokenLeftUp(int64_t actSeqLensQ, int64_t actSeqLensKv, int64_t &preTokenLeftUp,
                                                 int64_t &nextTokenLeftUp)
    {
        preTokenLeftUp = constInfoXVec.preTokens;
        nextTokenLeftUp = constInfoXVec.nextTokens;
        fa_base_vector::GetSafeActToken(actSeqLensQ, actSeqLensKv, preTokenLeftUp, nextTokenLeftUp,
                                        constInfoXVec.sparseMode);

        if (constInfoXVec.sparseMode == fa_base_vector::BAND) {
            preTokenLeftUp = static_cast<int64_t>(actSeqLensQ) - static_cast<int64_t>(actSeqLensKv) + preTokenLeftUp;
        }

        if (constInfoXVec.sparseMode == fa_base_vector::RIGHT_DOWN_CAUSAL) {
            nextTokenLeftUp = static_cast<int64_t>(actSeqLensKv) - static_cast<int64_t>(actSeqLensQ);
        } else if (constInfoXVec.sparseMode == fa_base_vector::BAND) {
            nextTokenLeftUp = static_cast<int64_t>(actSeqLensKv) - static_cast<int64_t>(actSeqLensQ) + nextTokenLeftUp;
        }
    }

    __aicore__ inline void CalcCurS2StartEndNoSparse(uint32_t bN2Cur, uint32_t gS1Cur)
    {
        curS2Start = 0U;
        curS2End = (static_cast<uint32_t>(actSeqLensKv) + s2BaseSize - 1) / s2BaseSize;

        if ((bN2Cur == bN2Start_) && (gS1Cur == gS1OStart_)) {
            headS2Split = s2OStart_ != 0U;
            curS2Start = s2OStart_;
        }

        if ((bN2Cur == bN2End_) && (gS1Cur == gS1OEnd_)) {
            tailS2Split = s2OEnd_ != 0U;
            curS2End = s2OEnd_;
        }
    }

    __aicore__ inline void CalcCurS2StartEndWithSparse(uint32_t bN2Cur, uint32_t gS1Cur)
    {
        // 1. Calc preTokenLeftUp, nextTokenLeftUp
        int64_t preTokenLeftUp = 0;
        int64_t nextTokenLeftUp = 0;
        GetPreNextTokenLeftUp(actSeqLensQ, actSeqLensKv, preTokenLeftUp, nextTokenLeftUp);

        // 2. calc index of s2FirstToken, s2LastToken by index of s1GFirstToken, s1GLastToken
        int64_t s1GFirstToken = static_cast<int64_t>(gS1Cur) * static_cast<int64_t>(mBaseSize);
        int64_t s1GLastToken = AttentionCommon::Min(s1GFirstToken + static_cast<int64_t>(mBaseSize),
                                   static_cast<int64_t>(actSeqLensQ) * static_cast<int64_t>(constInfoXVec.gSize)) -
                               1;
        int64_t s1FirstToken = 0;
        int64_t s1LastToken = 0;
        if constexpr (GetOutUbFormat<layout>() == UbFormat::S1G) {
            s1FirstToken = static_cast<int64_t>(s1GFirstToken / constInfoXVec.gSize);
            s1LastToken = static_cast<int64_t>(s1GLastToken / constInfoXVec.gSize);
        } else {
            if (s1GFirstToken / static_cast<int64_t>(actSeqLensQ) == s1GLastToken / static_cast<int64_t>(actSeqLensQ)) {
                // start and end locate in one G
                s1FirstToken = s1GFirstToken % static_cast<int64_t>(actSeqLensQ);
                s1LastToken = s1GLastToken % static_cast<int64_t>(actSeqLensQ);
            } else {
                // start and end locate in tow or more G, but working same as crossing one complete block
                s1FirstToken = 0;
                s1LastToken = static_cast<int64_t>(actSeqLensQ);
            }
        }

        // 3. trans index of token to index of block
        uint32_t s2StartWithSparse = 0U;
        uint32_t s2EndWithSparse = 0U;
        int64_t s2FirstToken = s1FirstToken - preTokenLeftUp;
        int64_t s2LastToken = s1LastToken + nextTokenLeftUp;
        // no valid token
        if (s2FirstToken >= static_cast<int64_t>(actSeqLensKv) || s2LastToken < 0 || s2LastToken < s2FirstToken) {
            curS2Start = 0U;
            curS2End = 0U;
            return;
        }
        // get valid range
        s2FirstToken = ClipSInnerToken(s2FirstToken, 0, static_cast<int64_t>(actSeqLensKv - 1));
        s2LastToken = ClipSInnerToken(s2LastToken, 0, static_cast<int64_t>(actSeqLensKv - 1));

        s2StartWithSparse = static_cast<uint32_t>(s2FirstToken) / s2BaseSize;
        s2EndWithSparse = static_cast<uint32_t>(s2LastToken) / s2BaseSize + 1U;

        // 4. Calc curS2Start, curS2End
        curS2Start = s2StartWithSparse;
        curS2End = s2EndWithSparse;

        if (bN2Cur == bN2Start_ && gS1Cur == gS1OStart_) { // first line
            headS2Split = s2OStart_ > s2StartWithSparse ? true : false;
            curS2Start = AttentionCommon::Max(s2StartWithSparse, s2OStart_);
        }
        if (bN2Cur == bN2End_ && gS1Cur == gS1OEnd_) { // last line
            tailS2Split = s2OEnd_ > 0U ? true : false;
            curS2End = s2OEnd_ > 0U ? AttentionCommon::Min(s2EndWithSparse, s2OEnd_) : s2EndWithSparse;
        }
        return;
    }

    __aicore__ inline void ExecuteTask(uint64_t loop, RunInfoXVec taskRunInfo[PRELOAD_TASK_CACHE_SIZE])
    {
        RunInfoXVec &runInfo0 = taskRunInfo[loop & PRELOAD_TASK_CACHE_MASK];                  // 本轮任务
        RunInfoXVec &runInfoNegN = taskRunInfo[(loop - PRELOAD_N) & PRELOAD_TASK_CACHE_MASK]; // 上PRELOAD_N轮任务
        if (runInfo0.isValid) {
            ComputeVec1(runInfo0);
        }
        if (loop >= PRELOAD_N) {
            if (runInfoNegN.isValid) {
                ComputeVec2(runInfoNegN);
                runInfoNegN.isValid = false;
            }
        }
    }

    __aicore__ inline void ComputeVec1(RunInfoXVec &runInfo)
    {
        ProcessVec1(this->l1PBuffers.Get(), this->bmm1Buffers.Get(), runInfo);
    }

    __aicore__ inline void ComputeVec2(RunInfoXVec &runInfo)
    {
        if constexpr (BMM2_TOUB) {
            ProcessVec2(this->bmm2Buffers.Get(), runInfo);
        } else {
            ProcessVec2(this->bmm2ResGmBuffers.Get(), runInfo);
        }
    }

    __aicore__ inline void CreateTask(uint64_t loop, uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur,
                                      RunInfoXVec taskRunInfo[PRELOAD_TASK_CACHE_SIZE])
    {
        RunInfoXVec &runInfo = taskRunInfo[loop & PRELOAD_TASK_CACHE_MASK]; // 本轮任务
        CalcParams(loop, bN2Cur, gS1Cur, s2Cur, runInfo);
        runInfo.isValid = true;
    }

    __aicore__ inline void CalcParams(uint64_t loop, uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur, RunInfoXVec &info)
    {
        info.loop = loop;
        info.mloop = mloop;
        info.bIdx = bN2Cur / constInfoXVec.n2Size;
        info.n2Idx = bN2Cur % constInfoXVec.n2Size;
        info.gS1Idx = gS1Cur * mBaseSize;
        info.gIdx = info.gS1Idx % constInfoXVec.gSize;
        info.s2Idx = s2Cur * s2BaseSize;
        info.actS1Size = actSeqLensQ;
        info.actS2Size = actSeqLensKv;

        info.actMSize = mBaseSize;
        uint64_t gS1Size = info.actS1Size * constInfoXVec.gSize;
        if (((gS1Cur + 1) * mBaseSize) > gS1Size) {
            info.actMSize = gS1Size - gS1Cur * mBaseSize;
        }
        info.actSingleLoopS2Size = s2BaseSize;
        if (((s2Cur + 1) * s2BaseSize) > info.actS2Size) {
            info.actSingleLoopS2Size = info.actS2Size - s2Cur * s2BaseSize;
        }
        info.actSingleLoopS2SizeAlign =
            AttentionCommon::Align((uint32_t)info.actSingleLoopS2Size, (uint32_t)(FA_BYTE_BLOCK / sizeof(INPUT_T)));

        GetPreNextTokenLeftUp(actSeqLensQ, actSeqLensKv, info.preTokensLeftUp, info.nextTokensLeftUp);

        info.isFirstS2Loop = ((loop == 0) || (s2Cur == curS2Start));
        info.isS2SplitCore = false;
        info.faTmpOutWsPos = coreFirstTmpOutWsPos_;
        info.isLastS2Loop = (s2Cur + 1 == curS2End);

        if constexpr (USE_DN) {
            info.actMSizeAlign32 = (info.actMSize + 31) >> 5 << 5;
            info.actVecMSize = info.actMSize <= 16 ? info.actMSize : (info.actMSizeAlign32 >> 1);
        } else {
            info.actVecMSize = (info.actMSize + 1) >> 1;
        }
        info.vecMbaseIdx = 0;
        if (constInfoXVec.subBlockIdx == 1) {
            info.vecMbaseIdx = info.actVecMSize;
            info.actVecMSize = info.actMSize - info.actVecMSize;
        }

        if (bN2Start_ == bN2End_ && gS1OStart_ == gS1OEnd_) {
            info.isS2SplitCore = true;
        } else {
            if (headS2Split && (bN2Cur == bN2Start_) && (gS1Cur == gS1OStart_)) {
                info.isS2SplitCore = true;
            } else if (tailS2Split && (bN2Cur == bN2End_) && (gS1Cur == gS1OEnd_)) {
                info.isS2SplitCore = true;
                info.faTmpOutWsPos = headS2Split ? (info.faTmpOutWsPos + 1) : info.faTmpOutWsPos;
            }
        }
    }

    __aicore__ inline void UpdateAxisInfo(TASK_DEAL_MODE taskDealMode, uint32_t &bN2Cur, uint32_t &gS1Cur,
                                          uint32_t &s2Cur)
    {
        uint64_t s2LoopTimes = (actSeqLensKv + s2BaseSize - 1) / s2BaseSize;
        uint64_t gS1Size = actSeqLensQ * constInfoXVec.gSize;
        uint64_t gS1LoopTimes = (gS1Size + mBaseSize - 1) / mBaseSize;

        // 当前S2未处理完
        if (s2Cur + 1 < s2LoopTimes) {
            s2Cur++;
            return;
        }

        // 当前BN2未处理完
        s2Cur = 0;
        if (gS1Cur + 1 < gS1LoopTimes) {
            gS1Cur++;
            return;
        }

        // 当前BN2已处理完
        gS1Cur = 0;
        bN2Cur++;
    }

    __aicore__ inline void FlashDecode(uint32_t sectionIdx)
    {
        if (!isFd) {
            return;
        }
        GetFDSectionInfo(sectionIdx);
        vecFdBlock.InitBuffers(this->pipe, SharedBuffer1, SharedBuffer2, SharedBuffer3);
        AscendC::ICachePreLoad(2);
        vecFdBlock.AllocEventID();
        vecFdBlock.InitDecodeParams();
        SyncAll();
        vecFdBlock.FlashDecode(fdParams_);
        SyncAll();
        vecFdBlock.FreeBuffers(SharedBuffer1, SharedBuffer2);
        vecFdBlock.FreeEventID();
    }

    __aicore__ inline void GetFASectionInfo(uint32_t sectionIdx)
    {
        bN2Start_ = faMetaDataGm.GetValue(GetFAMetaDataIndex(constInfoXVec.aicIdx, FLASH_ATTN_BN2_START_INDEX, sectionIdx));
        gS1OStart_ = faMetaDataGm.GetValue(GetFAMetaDataIndex(constInfoXVec.aicIdx, FLASH_ATTN_M_START_INDEX, sectionIdx));
        s2OStart_ = faMetaDataGm.GetValue(GetFAMetaDataIndex(constInfoXVec.aicIdx, FLASH_ATTN_S2_START_INDEX, sectionIdx));
        bN2End_ = faMetaDataGm.GetValue(GetFAMetaDataIndex(constInfoXVec.aicIdx, FLASH_ATTN_BN2_END_INDEX, sectionIdx));
        gS1OEnd_ = faMetaDataGm.GetValue(GetFAMetaDataIndex(constInfoXVec.aicIdx, FLASH_ATTN_M_END_INDEX, sectionIdx));
        s2OEnd_ = faMetaDataGm.GetValue(GetFAMetaDataIndex(constInfoXVec.aicIdx, FLASH_ATTN_S2_END_INDEX, sectionIdx));
        coreFirstTmpOutWsPos_ = faMetaDataGm.GetValue(
            GetFAMetaDataIndex(constInfoXVec.aicIdx, FLASH_ATTN_FIRST_FD_DATA_WORKSPACE_IDX_INDEX, sectionIdx));
    }

    __aicore__ inline void GetFDSectionInfo(uint32_t sectionIdx)
    {
        fdParams_.mLen = fdMetaDataGm.GetValue(GetFDMetaDataIndex(constInfoXVec.aivIdx, FA_FD_M_NUM_INDEX, sectionIdx));
        fdParams_.fdCoreEnable = fdParams_.mLen > 0 ? 1U : 0U;
        if (!fdParams_.fdCoreEnable) {
            return;
        }
        fdParams_.fdBN2Idx =
            fdMetaDataGm.GetValue(GetFDMetaDataIndex(constInfoXVec.aivIdx, FA_FD_BN2_IDX_INDEX, sectionIdx));
        fdParams_.fdMIdx = fdMetaDataGm.GetValue(GetFDMetaDataIndex(constInfoXVec.aivIdx, FA_FD_M_IDX_INDEX, sectionIdx));
        fdParams_.fdWorkspaceIdx =
            fdMetaDataGm.GetValue(GetFDMetaDataIndex(constInfoXVec.aivIdx, FA_FD_WORKSPACE_IDX_INDEX, sectionIdx));
        fdParams_.fdS2SplitNum =
            fdMetaDataGm.GetValue(GetFDMetaDataIndex(constInfoXVec.aivIdx, FA_FD_WORKSPACE_NUM_INDEX, sectionIdx));
        fdParams_.mStart = fdMetaDataGm.GetValue(GetFDMetaDataIndex(constInfoXVec.aivIdx, FA_FD_M_START_INDEX, sectionIdx));
    }


    __aicore__ inline void Process()
    {
        if (constInfoXVec.aicIdx < constInfoXVec.coreNum) {
            CrossCoreBufferInit();
            InitBuffers(SharedBuffer1, SharedBuffer2, SharedBuffer3);
            AllocEventID();
        }
        for (uint32_t sectionIdx = 0; sectionIdx < sectionNum_; sectionIdx++) {
            if (constInfoXVec.aicIdx < constInfoXVec.coreNum) {
                FlashAttention(sectionIdx);
            }
            FlashDecode(sectionIdx);
        }

        if (constInfoXVec.aicIdx < constInfoXVec.coreNum) {
            FreeEventID();
        }
    }
}; // FlashAttentionNoQuantGqaKernelVec


} // namespace BaseApi

#endif // FLASH_ATTENTION_NOQUANT_GQA_KERNEL_VEC_H_