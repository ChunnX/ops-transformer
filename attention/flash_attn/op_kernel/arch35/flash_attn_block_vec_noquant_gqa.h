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
 * \file flash_attn_block_vec_noquant_gqa.h
 * \brief
 */
#ifndef FLASH_ATTENTION_NOQUANT_GQA_BLOCK_VEC_H_
#define FLASH_ATTENTION_NOQUANT_GQA_BLOCK_VEC_H_

#include "kernel_operator.h"

#include "../utils/attenmask_gs1.h"

#include "../../../common/op_kernel/arch35/flash_attention_score_common_regbase.h"
#include "adv_api/activation/softmax.h"
#include "../../../common/op_kernel/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz.h"
#include "fa_vec1_vf_static.h"
#include "../../../common/op_kernel/arch35/vf/vf_flashupdate_new.h"
#include "../../../common/op_kernel/arch35/vf/vf_div_cast.h"
#include "../../../common/op_kernel/vector_common.h"
#include "../../../common/op_kernel/memory_copy_arch35.h"

using namespace AscendC;
using namespace FaVectorApi;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;
using namespace AttentionCommon;

namespace BaseApi {

template <
    typename INPUT_T, typename T, typename OUTPUT_T, LayOutTypeEnum layout = LayOutTypeEnum::None,
    LayOutTypeEnum outLayout = LayOutTypeEnum::None, S1TemplateType s1TemplateType = S1TemplateType::Aligned128,
    S2TemplateType s2TemplateType = S2TemplateType::Aligned128, DTemplateType dTemplateType = DTemplateType::Aligned128,
    DTemplateType dVTemplateType = DTemplateType::Aligned128, bool hasAtten = false, bool hasDrop = false,
    uint8_t KvLayoutType = 0>
class VecBlockBase {
public:
    /* =================编译期常量的基本块信息================= */
    static constexpr uint32_t mBaseSize = (uint32_t)s1TemplateType;
    static constexpr uint32_t s2BaseSize = (uint32_t)s2TemplateType;
    static constexpr uint32_t vec1HalfS1BaseSize = mBaseSize >> 1;
    static constexpr uint32_t vec1Srcstride = (mBaseSize >> 1) + 1;
    static constexpr uint32_t dTemplateAlign64 = Align64Func((uint16_t)dVTemplateType);

    static constexpr uint32_t DB = 2;
    // 索引使用 loop & (DB - 1) 代替 loop % DB，要求 DB 必须是2的幂，否则位掩码结果错误
    static_assert(DB > 0 && (DB & (DB - 1)) == 0, "DB must be a power of two for bitmask indexing");
    static constexpr uint32_t PRELOAD_N = 2; // C1 C1 C2
    static constexpr bool HAS_MASK = hasAtten;

    static constexpr uint32_t initOutputEventId = 0U; // attenOut和lse，刷无效行会用到剩余ub，需要加同步

    static constexpr ActualSeqLensMode Q_MODE = GetQActSeqMode<layout>();
    static constexpr LAYOUT_Q MASK_LAYOUT =
        (layout == LayOutTypeEnum::LAYOUT_BSH || layout == LayOutTypeEnum::LAYOUT_TND ||
         layout == LayOutTypeEnum::LAYOUT_SBH) ?
            LAYOUT_Q::SG :
            LAYOUT_Q::GS;

    static constexpr T BOOL_ATTEN_MASK_SCALAR_VALUE = -1000000000000.0; // 用于mask为bool类型
    uint32_t negativeIntScalar = *((uint32_t *)&BOOL_ATTEN_MASK_SCALAR_VALUE);

    using attenMaskGmType = typename std::conditional<hasAtten, GlobalTensor<uint8_t>, int8_t>::type;

    using ConstInfoX = ConstInfo_t<FiaKernelType::NO_QUANT>;

    __aicore__ inline VecBlockBase(ConstInfoX &constInfo){};
};

template <
    typename INPUT_T, typename T, typename OUTPUT_T, LayOutTypeEnum layout = LayOutTypeEnum::None,
    LayOutTypeEnum outLayout = LayOutTypeEnum::None, S1TemplateType s1TemplateType = S1TemplateType::Aligned128,
    S2TemplateType s2TemplateType = S2TemplateType::Aligned128, DTemplateType dTemplateType = DTemplateType::Aligned128,
    DTemplateType dVTemplateType = DTemplateType::Aligned128, bool hasAtten = false, bool hasDrop = false,
    uint8_t KvLayoutType = 0>
class FANoQuantGqaBlockVec
    : public VecBlockBase<INPUT_T, T, OUTPUT_T, layout, outLayout, s1TemplateType, s2TemplateType, dTemplateType,
                          dVTemplateType, hasAtten, hasDrop, KvLayoutType> {
public:
    /* =================编译期常量的基本块信息================= */
    static constexpr uint32_t mBaseSize = (uint32_t)s1TemplateType;
    static constexpr uint32_t s2BaseSize = (uint32_t)s2TemplateType;
    static constexpr uint32_t vec1HalfS1BaseSize = mBaseSize >> 1;
    static constexpr uint32_t vec1Srcstride = (mBaseSize >> 1) + 1;
    static constexpr uint32_t dTemplateAlign64 = Align64Func((uint16_t)dVTemplateType);

    static constexpr uint32_t DB = 2;
    // 索引使用 loop & (DB - 1) 代替 loop % DB，要求 DB 必须是2的幂，否则位掩码结果错误
    static_assert(DB > 0 && (DB & (DB - 1)) == 0, "DB must be a power of two for bitmask indexing");
    static constexpr uint32_t PRELOAD_N = 2; // C1 C1 C2
    static constexpr bool HAS_MASK = hasAtten;

    static constexpr uint32_t initOutputEventId = 0U; // attenOut和lse，刷无效行会用到剩余ub，需要加同步
    // 静态 event ID 常量（避开 6/7，参考 quant_flash_attn：BASE + idx 数学公式直接计算）
    // MTE3_V 通道（避开 6/7）：
    //   [0]      Vec2 stage2OutBuf 的反向同步（vec2 单点）
    //   [1]      LSE / softmax 数据搬出后的反向同步
    //   [2..3]   stage1 双缓冲 ping/pong 的反向同步（FA_STAGE1_MTE3_V_FLAG_0 + stage1Offset）
    static constexpr uint32_t FA_VEC2_MTE3_V_FLAG    = 0U;
    static constexpr uint32_t FA_LSE_MTE3_V_FLAG     = 1U;
    static constexpr uint32_t FA_STAGE1_MTE3_V_FLAG_0 = 2U; // stage1Offset=0
    static constexpr uint32_t FA_STAGE1_MTE3_V_FLAG_1 = 3U; // stage1Offset=1
    // V_MTE3 通道：[0..1] stage1 双缓冲与 vec2 cast 共用（不同时刻），(FA_V_MTE3_FLAG_0 + idx)
    static constexpr uint32_t FA_V_MTE3_FLAG_0       = 0U;
    static constexpr uint32_t FA_V_MTE3_FLAG_1       = 1U;
    // 单点 MTE2_V / V_MTE2 / MTE3_MTE2 通道：均用 ID 2，互不冲突
    static constexpr uint32_t FA_MTE2_TO_V_ID        = 2U;
    static constexpr uint32_t FA_V_TO_MTE2_ID        = 2U;
    static constexpr uint32_t FA_MTE3_TO_MTE2_ID     = 2U;
    // MTE2_V 通道 sink 数据搬运同步（独占 ID 4，避免与 attenMask 0/1 冲突）
    static constexpr uint32_t FA_MTE2_TO_V_SINK      = 4U;
    // attenMask ping/pong 的 MTE2→V 正向同步（MTE2_V 通道 0/1），(FA_ATTEN_MASK_MTE2_V_FLAG_0 + bufIdx)
    static constexpr uint32_t FA_ATTEN_MASK_MTE2_V_FLAG_0 = 0U;
    static constexpr uint32_t FA_ATTEN_MASK_MTE2_V_FLAG_1 = 1U;
    // attenMask 整块清零(V)先于 mask DMA(MTE2) 的反向同步（V_MTE2 通道 0，未被其他路径占用）
    static constexpr uint32_t FA_ATTEN_MASK_V_MTE2_0 = 0U;

    // ===== 核间同步常量（避开 6、7，Set/Wait 对称）=====
    static constexpr uint64_t CROSS_CORE_SYNC_MODE = 4U;
    static constexpr uint32_t AIV0_AIV1_OFFSET = 16U;
    // 注：跨核 barrier 沿用 AscendC 内置 SyncAll() 硬同步接口（占用 flagId [11-14]），
    // 与上面的 mode 4 flagId [0-9] 完全不冲突，是 Mix 算子的标准做法。

    static constexpr ActualSeqLensMode Q_MODE = GetQActSeqMode<layout>();
    static constexpr LAYOUT_Q MASK_LAYOUT =
        (layout == LayOutTypeEnum::LAYOUT_BSH || layout == LayOutTypeEnum::LAYOUT_TND ||
         layout == LayOutTypeEnum::LAYOUT_SBH) ?
            LAYOUT_Q::SG :
            LAYOUT_Q::GS;

    static constexpr T BOOL_ATTEN_MASK_SCALAR_VALUE = -1000000000000.0; // 用于mask为bool类型
    uint32_t negativeIntScalar = *((uint32_t *)&BOOL_ATTEN_MASK_SCALAR_VALUE);

    using attenMaskGmType = typename std::conditional<hasAtten, GlobalTensor<uint8_t>, int8_t>::type;

    using ConstInfoX = ConstInfo_t<FiaKernelType::NO_QUANT>;

    // gm
    GlobalTensor<OUTPUT_T> attentionOutGm;
    GlobalTensor<float> softmaxLseGm;

    GlobalTensor<int32_t> actualSeqLengthsGmQ;
    using QSeqParserType =
        typename std::conditional<(layout == LayOutTypeEnum::LAYOUT_TND),
                                  ActualSeqLensParser<Q_MODE, int32_t, true>,
                                  ActualSeqLensParser<Q_MODE, int32_t>>::type;
    QSeqParserType *qActSeqLensParser = nullptr;

    __aicore__ inline void SetCuSeqLensParser(QSeqParserType &qParser)
    {
        this->qActSeqLensParser = &qParser;
    }

    attenMaskGmType attenMaskGmInt;

    GlobalTensor<INPUT_T> sinkGm;

    // ub
    LocalTensor<uint8_t> commonBuf;
    LocalTensor<INPUT_T> stage1OutBuf[2];
    LocalTensor<uint8_t> attenMaskBuf[2];
    LocalTensor<T> stage2OutBuf;
    LocalTensor<float> softmaxMaxBuf[PRELOAD_N + 1];
    LocalTensor<float> softmaxSumBuf[PRELOAD_N + 1];
    LocalTensor<T> softmaxExpBuf[PRELOAD_N + 1];
    LocalTensor<uint8_t> vselrIndexesBuf[4]; // 静态Tensor编程：LocalTensor 直接替代 TBuf<>
    LocalTensor<float> lseBuf;
    LocalTensor<INPUT_T> sinkBuf;

    const ConstInfoX &constInfo;
    T negativeFloatScalar;

    // ==================== Functions ======================
    __aicore__ inline FANoQuantGqaBlockVec(ConstInfoX &constInfo)
        : VecBlockBase<INPUT_T, T, OUTPUT_T, layout, outLayout, s1TemplateType, s2TemplateType, dTemplateType,
                       dVTemplateType, hasAtten, hasDrop, KvLayoutType>(constInfo),
          constInfo(constInfo){};

    __aicore__ inline void InitVecBlock(__gm__ uint8_t *actualSeqQlenAddr,
                                        __gm__ uint8_t *actualSeqKvlenAddr, __gm__ uint8_t *attenMask,
                                        __gm__ uint8_t *learnableSink, __gm__ uint8_t *softmaxLse,
                                        __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace)
    {
        uint32_t tmp1 = NEGATIVE_MIN_VALUE_FP32;
        this->negativeFloatScalar = *((T *)&tmp1);

        InitVecInput(actualSeqQlenAddr, actualSeqKvlenAddr, attenMask, learnableSink, softmaxLse, attentionOut,
                     workspace);
    }

    __aicore__ inline void InitVecInput(__gm__ uint8_t *actualSeqQlenAddr, __gm__ uint8_t *actualSeqKvlenAddr,
                                        __gm__ uint8_t *attenMask, __gm__ uint8_t *learnableSink,
                                        __gm__ uint8_t *softmaxLse, __gm__ uint8_t *attentionOut,
                                        __gm__ uint8_t *workspace)
    {
        this->attentionOutGm.SetGlobalBuffer((__gm__ OUTPUT_T *)attentionOut);

        if (constInfo.isSoftmaxLseEnable) {
            softmaxLseGm.SetGlobalBuffer((__gm__ float *)softmaxLse);
        }

        uint64_t actualLenQSize =
            (layout == LayOutTypeEnum::LAYOUT_TND) ? constInfo.cuSeqLensQSize : constInfo.seqUsedQSize;
        actualSeqLengthsGmQ.SetGlobalBuffer((__gm__ int32_t *)actualSeqQlenAddr, actualLenQSize);

        if constexpr (hasAtten) {
            attenMaskGmInt.SetGlobalBuffer((__gm__ uint8_t *)attenMask);
        }

        if (learnableSink != nullptr) {
            sinkGm.SetGlobalBuffer((__gm__ INPUT_T *)learnableSink);
        }
    }
    __aicore__ inline void ProcessVec1(LocalTensor<INPUT_T> &l1PTensor, uint32_t l1PCcId,
                                       LocalTensor<T> &bmm1UbTensor, uint32_t bmm1CcId,
                                       RunInfoX runInfo)
    {
#ifdef SKIP_V1
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(bmm1CcId);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(bmm1CcId);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE3>(l1PCcId);
        return;
#endif

        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(bmm1CcId);
        if (unlikely(runInfo.actVecMSize == 0)) {
            CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(bmm1CcId);
            CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE3>(l1PCcId);
            return;
        }

        ProcessVec1Nd(l1PTensor, l1PCcId, bmm1UbTensor, bmm1CcId, runInfo);
    }

    __aicore__ inline void ClearOutput()
    {
        bool needOutputInit = IsInitAttentionOutGm();
        if (needOutputInit) {
            SetFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId); // 释放剩余ub
            InitOutputSingleCore();
            if (constInfo.isSoftmaxLseEnable) {
                InitLseOutputSingleCore();
            }
            WaitFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
            AscendC::SyncAll();
        }
    }

    __aicore__ inline bool IsInitAttentionOutGm()
    {
        return constInfo.needInitOutput;
    }

    __aicore__ inline void InitOutputSingleCore()
    {
        int64_t tSize = constInfo.bSize * constInfo.s1Size;
        if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
            tSize = qActSeqLensParser->GetTSize();
        }
        int64_t totalOutputSize = tSize * constInfo.n2Size * constInfo.gSize * constInfo.dSizeV;
        int64_t singleCoreSize =
            (totalOutputSize + (2 * constInfo.coreNum) - 1) / (2 * constInfo.coreNum); // 2 means c:v = 1:2
        int64_t tailSize = totalOutputSize - constInfo.aivIdx * singleCoreSize;
        int64_t singleInitOutputSize = tailSize < singleCoreSize ? tailSize : singleCoreSize;

        if (singleInitOutputSize > 0) {
            WaitFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
            if constexpr (IsSameType<OUTPUT_T, int8_t>::value) {
                GlobalTensor<half> attentionOutTmpGm;
                attentionOutTmpGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(attentionOutGm.GetPhyAddr(0)));
                matmul::InitOutput<half>(attentionOutTmpGm[constInfo.aivIdx * singleCoreSize], singleInitOutputSize, 0);
            } else {
                matmul::InitOutput<OUTPUT_T>(attentionOutGm[constInfo.aivIdx * singleCoreSize], singleInitOutputSize,
                    0);
            }
            SetFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
        }
    }

    __aicore__ inline void InitLseOutputSingleCore()
    {
        int64_t tSize = constInfo.bSize * constInfo.s1Size;
        if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
            tSize = qActSeqLensParser->GetTSize();
        }
        int64_t totalOutputSize = tSize * constInfo.n2Size * constInfo.gSize;
        int64_t singleCoreSize =
            (totalOutputSize + (2 * constInfo.coreNum) - 1) / (2 * constInfo.coreNum); // 2 means c:v = 1:2
        int64_t tailSize = totalOutputSize - constInfo.aivIdx * singleCoreSize;
        int64_t singleInitOutputSize = tailSize < singleCoreSize ? tailSize : singleCoreSize;

        if (singleInitOutputSize > 0) {
            WaitFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
            matmul::InitOutput<float>(softmaxLseGm[constInfo.aivIdx * singleCoreSize], singleInitOutputSize,
                                      3e+99); // 3e+99: set the value of invalid batch to inf
            SetFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
        }
    }

    // =================================Private Functions=================================

    __aicore__ inline void SoftmaxDataCopyOut(RunInfoX runInfo, LocalTensor<float> &sumUb, LocalTensor<float> &maxUb)
    {
        if (constInfo.isSoftmaxLseEnable) {
            SoftmaxLseCopyOut(sumUb, maxUb, runInfo);
        }

        if (constInfo.learnableSinkFlag) {
            this->Vec1SinkComputeGSFused(runInfo, sumUb, maxUb);
        }
    }

    __aicore__ inline void SoftmaxLseCopyOut(LocalTensor<float> &softmaxSumTmp, LocalTensor<float> &softmaxMaxTmp,
                                             RunInfoX &runInfo)
    {
        if (unlikely(runInfo.actVecMSize == 0)) {
            return;
        }

        uint32_t vecMIdx = runInfo.gS1Idx + runInfo.vecMbaseIdx;
        LocalTensor<float> &lseUb = this->lseBuf;
        ComputeLseOutputVF(lseUb, softmaxSumTmp, softmaxMaxTmp, runInfo.actVecMSize);
        // V → MTE3 正向：等待 lse 计算完成后才可搬出
        SetFlag<HardEvent::V_MTE3>(FA_V_MTE3_FLAG_0);
        WaitFlag<HardEvent::V_MTE3>(FA_V_MTE3_FLAG_0);
        {
            if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
                uint32_t prefixBS1 = qActSeqLensParser->GetTBase(runInfo.bIdx);
                uint64_t bN2Offset = runInfo.n2Idx * constInfo.gSize * constInfo.t1Size + prefixBS1;
                DataCopySoftmaxLseTNDtoNTArch35<T, ConstInfoX>(softmaxLseGm, lseUb, bN2Offset, vecMIdx,
                                                               runInfo.actVecMSize, constInfo);
            } else if constexpr (layout == LayOutTypeEnum::LAYOUT_BSH) {
                uint64_t bN2Offset = runInfo.bIdx * constInfo.n2Size * constInfo.gSize * constInfo.s1Size +
                                     runInfo.n2Idx * constInfo.gSize * constInfo.s1Size;
                uint64_t qActSeqLens = qActSeqLensParser->GetActualSeqLength(runInfo.bIdx);
                DataCopySoftmaxLseBSNDArch35<T, ConstInfoX>(softmaxLseGm, lseUb, bN2Offset, vecMIdx,
                                                            runInfo.actVecMSize, constInfo);
            } else { // BNSD
                uint64_t bN2Offset = runInfo.bIdx * constInfo.n2Size * constInfo.gSize * constInfo.s1Size +
                                     runInfo.n2Idx * constInfo.gSize * constInfo.s1Size;
                uint64_t qActSeqLens = qActSeqLensParser->GetActualSeqLength(runInfo.bIdx);
                DataCopySoftmaxLseBNSDArch35<T, ConstInfoX>(softmaxLseGm, lseUb, bN2Offset, vecMIdx,
                                                            runInfo.actVecMSize, constInfo, qActSeqLens);
            }
        }
        // MTE3 → V 反向：等待 MTE3 搬运结束后下一轮 V 才能复用 lseBuf
        SetFlag<HardEvent::MTE3_V>(FA_LSE_MTE3_V_FLAG);
        WaitFlag<HardEvent::MTE3_V>(FA_LSE_MTE3_V_FLAG);
    }

    __aicore__ inline void Vec1SinkCompute(RunInfoX &runInfo, LocalTensor<float> &sumUb, LocalTensor<float> &maxUb)
    {
        int64_t sinkOffset = runInfo.n2Idx * constInfo.gSize + runInfo.gIdx;
        auto sinkRaw = this->sinkGm.GetValue(sinkOffset);
        float sinkValue;
        if constexpr (IsSameType<decltype(sinkRaw), half>::value) {
            sinkValue = static_cast<float>(sinkRaw);
        } else {
            sinkValue = ToFloat(sinkRaw);
        }
        SinkSubExpAddVF<float>(sumUb, maxUb, sinkValue, runInfo.actVecMSize);
    }

    __aicore__ inline void Vec1SinkComputeGSFused(RunInfoX &runInfo, LocalTensor<float> &sumUb,
                                                  LocalTensor<float> &maxUb)
    {
        CopySinkIn(runInfo);
        WaitFlag<HardEvent::MTE2_V>(FA_MTE2_TO_V_SINK);
        SinkSubExpAddGSFusedVF<float, INPUT_T>(this->sinkBuf, sumUb, maxUb, runInfo.actVecMSize);
    }

    __aicore__ inline void CopySinkIn(RunInfoX &runInfo)
    {
        int64_t sinkOffset =
            runInfo.n2Idx * constInfo.gSize + constInfo.subBlockIdx * (runInfo.actMSize - runInfo.actVecMSize);
        DataCopyExtParams sinkCopyParams;
        sinkCopyParams.blockCount = 1;                                   // 进行一次连续拷贝
        sinkCopyParams.blockLen = runInfo.actVecMSize * sizeof(INPUT_T); // 实际需要拷贝的字节数
        sinkCopyParams.srcStride = 0;                                    // 源地址连续
        sinkCopyParams.dstStride = 0;                                    // 目的地址连续

        DataCopyPadExtParams<INPUT_T> sinkCopyPadParams{};
        DataCopyPad(this->sinkBuf, this->sinkGm[sinkOffset], sinkCopyParams, sinkCopyPadParams);
        SetFlag<HardEvent::MTE2_V>(FA_MTE2_TO_V_SINK);
    }

    __aicore__ inline void ProcessVec1Nd(LocalTensor<INPUT_T> &l1PTensor, uint32_t l1PCcId,
                                          LocalTensor<T> &bmm1UbTensor, uint32_t bmm1CcId,
                                          RunInfoX runInfo)
    {
        LocalTensor<uint8_t> dropMaskUb;
        LocalTensor<INPUT_T> nonePseUb; // PSE不支持，占位
        LocalTensor<uint8_t> attenMaskUb;
        LocalTensor<uint8_t> attenMaskUbPre;
        if constexpr (hasAtten == true) {
            // 替代 attenMaskInQue.AllocTensor：等上一轮 V 读完同一 ping/pong 槽后再 MTE2 写入
            PipeBarrier<PIPE_ALL>();
            attenMaskUb = this->attenMaskBuf[runInfo.loop & (DB - 1)];
            AttenMaskCopyIn(attenMaskUb, 0, runInfo.actVecMSize, runInfo);
            PipeBarrier<PIPE_MTE2>();
            PipeBarrier<PIPE_V>();
        }

        LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.mloop % (PRELOAD_N + 1)];
        LocalTensor<float> maxUb = this->softmaxMaxBuf[runInfo.mloop % (PRELOAD_N + 1)];
        LocalTensor<float> expUb = this->softmaxExpBuf[runInfo.loop % (PRELOAD_N + 1)].template ReinterpretCast<T>();
        LocalTensor<T> pScaleUb;
        LocalTensor<T> queryScaleUb;
        LocalTensor<uint8_t> apiTmpBuffer;

        apiTmpBuffer = this->commonBuf;

        int64_t stage1Offset = 0;
        stage1Offset = runInfo.loop & (DB - 1);

        // stage1 双缓冲反向同步：上一轮搬运到 L1 完成后才能复用当前 ping/pong 槽
        WaitFlag<HardEvent::MTE3_V>(FA_STAGE1_MTE3_V_FLAG_0 + stage1Offset);

        float descaleQK = 1.0;
        float deSCaleKValue = 1.0;

        LocalTensor<T> mmRes = bmm1UbTensor;
        LocalTensor<INPUT_T> stage1CastTensor = this->stage1OutBuf[stage1Offset];
        if (runInfo.isFirstS2Loop) {
            if (likely(runInfo.actSingleLoopS2Size == 128)) {
                fa_static::ProcessVec1Vf<T, INPUT_T, INPUT_T /*pseShiftType*/, false, mBaseSize, s2BaseSize, EQ_128, hasAtten,
                              PseTypeEnum::PSE_NONE_TYPE, hasDrop, false, false>(
                    stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, mmRes, expUb, sumUb, maxUb, attenMaskUb,
                    nonePseUb, dropMaskUb, apiTmpBuffer, pScaleUb, runInfo.actVecMSize, runInfo.actSingleLoopS2Size,
                    0 /*pseStride*/, 0.0f /*slopes*/, 0.0f /*posShift*/, static_cast<T>(constInfo.scaleValue),
                    descaleQK, negativeFloatScalar, 0.0F, queryScaleUb, deSCaleKValue);
            } else if (runInfo.actSingleLoopS2Size <= 64) {
                fa_static::ProcessVec1Vf<T, INPUT_T, INPUT_T /*pseShiftType*/, false, mBaseSize, s2BaseSize, GT_0_AND_LTE_64,
                              hasAtten, PseTypeEnum::PSE_NONE_TYPE, hasDrop, false, false>(
                    stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, mmRes, expUb, sumUb, maxUb, attenMaskUb,
                    nonePseUb, dropMaskUb, apiTmpBuffer, pScaleUb, runInfo.actVecMSize, runInfo.actSingleLoopS2Size,
                    0 /*pseStride*/, 0.0f /*slopes*/, 0.0f /*posShift*/, static_cast<T>(constInfo.scaleValue),
                    descaleQK, negativeFloatScalar, 0.0F, queryScaleUb, deSCaleKValue);
            } else if (runInfo.actSingleLoopS2Size < 128) {
                fa_static::ProcessVec1Vf<T, INPUT_T, INPUT_T /*pseShiftType*/, false, mBaseSize, s2BaseSize, GT_64_AND_LTE_128,
                              hasAtten, PseTypeEnum::PSE_NONE_TYPE, hasDrop, false, false>(
                    stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, mmRes, expUb, sumUb, maxUb, attenMaskUb,
                    nonePseUb, dropMaskUb, apiTmpBuffer, pScaleUb, runInfo.actVecMSize, runInfo.actSingleLoopS2Size,
                    0 /*pseStride*/, 0.0f /*slopes*/, 0.0f /*posShift*/, static_cast<T>(constInfo.scaleValue),
                    descaleQK, negativeFloatScalar, 0.0F, queryScaleUb, deSCaleKValue);
            } else {
                if constexpr (s2BaseSize == 256) {
                    fa_static::ProcessVec1Vf<T, INPUT_T, INPUT_T /*pseShiftType*/, false, mBaseSize, s2BaseSize,
                                  GT_128_AND_LTE_256, hasAtten, PseTypeEnum::PSE_NONE_TYPE, hasDrop>(
                        stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, mmRes, expUb, sumUb, maxUb, attenMaskUb,
                        nonePseUb, dropMaskUb, apiTmpBuffer, expUb, runInfo.actVecMSize, runInfo.actSingleLoopS2Size,
                        0 /*pseStride*/, 0.0f /*slopes*/, 0.0f /*posShift*/, static_cast<T>(constInfo.scaleValue),
                        descaleQK, negativeFloatScalar, 0.0F);
                }
            }
        } else {
            if (likely(runInfo.actSingleLoopS2Size == 128)) {
                fa_static::ProcessVec1Vf<T, INPUT_T, INPUT_T /*pseShiftType*/, true, mBaseSize, s2BaseSize, EQ_128, hasAtten,
                              PseTypeEnum::PSE_NONE_TYPE, hasDrop, false, false>(
                    stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, mmRes, expUb, sumUb, maxUb, attenMaskUb,
                    nonePseUb, dropMaskUb, apiTmpBuffer, pScaleUb, runInfo.actVecMSize, runInfo.actSingleLoopS2Size,
                    0 /*pseStride*/, 0.0f /*slopes*/, 0.0f /*posShift*/, static_cast<T>(constInfo.scaleValue),
                    descaleQK, negativeFloatScalar, 0.0F, queryScaleUb, deSCaleKValue);
            } else if (runInfo.actSingleLoopS2Size <= 64) {
                fa_static::ProcessVec1Vf<T, INPUT_T, INPUT_T /*pseShiftType*/, true, mBaseSize, s2BaseSize, GT_0_AND_LTE_64,
                              hasAtten, PseTypeEnum::PSE_NONE_TYPE, hasDrop, false, false>(
                    stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, mmRes, expUb, sumUb, maxUb, attenMaskUb,
                    nonePseUb, dropMaskUb, apiTmpBuffer, pScaleUb, runInfo.actVecMSize, runInfo.actSingleLoopS2Size,
                    0 /*pseStride*/, 0.0f /*slopes*/, 0.0f /*posShift*/, static_cast<T>(constInfo.scaleValue),
                    descaleQK, negativeFloatScalar, 0.0F, queryScaleUb, deSCaleKValue);
            } else if (runInfo.actSingleLoopS2Size < 128) {
                fa_static::ProcessVec1Vf<T, INPUT_T, INPUT_T /*pseShiftType*/, true, mBaseSize, s2BaseSize, GT_64_AND_LTE_128,
                              hasAtten, PseTypeEnum::PSE_NONE_TYPE, hasDrop, false, false>(
                    stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, mmRes, expUb, sumUb, maxUb, attenMaskUb,
                    nonePseUb, dropMaskUb, apiTmpBuffer, pScaleUb, runInfo.actVecMSize, runInfo.actSingleLoopS2Size,
                    0 /*pseStride*/, 0.0f /*slopes*/, 0.0f /*posShift*/, static_cast<T>(constInfo.scaleValue),
                    descaleQK, negativeFloatScalar, 0.0F, queryScaleUb, deSCaleKValue);
            } else {
                if constexpr (s2BaseSize == 256) {
                    fa_static::ProcessVec1Vf<T, INPUT_T, INPUT_T /*pseShiftType*/, true, mBaseSize, s2BaseSize, GT_128_AND_LTE_256,
                                  hasAtten, PseTypeEnum::PSE_NONE_TYPE, hasDrop>(
                        stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, mmRes, expUb, sumUb, maxUb, attenMaskUb,
                        nonePseUb, dropMaskUb, apiTmpBuffer, expUb, runInfo.actVecMSize, runInfo.actSingleLoopS2Size,
                        0 /*pseStride*/, 0.0f /*slopes*/, 0.0f /*posShift*/, static_cast<T>(constInfo.scaleValue),
                        descaleQK, negativeFloatScalar, 0.0F);
                }
            }
        }
        if constexpr (hasAtten) {
            // 替代 attenMaskInQue.FreeTensor：V 读完 mask 后再允许 MTE2 覆写该槽
            PipeBarrier<PIPE_V>();
            PipeBarrier<PIPE_ALL>();
        }
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(bmm1CcId);

        // ===================DataCopy to L1 ====================
        // 对应原版 stage1OutQue EnQue/DeQue：V 写完后 MTE3 才能读 stage1OutBuf
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE3>(FA_V_MTE3_FLAG_0 + stage1Offset);
        WaitFlag<HardEvent::V_MTE3>(FA_V_MTE3_FLAG_0 + stage1Offset);
        LocalTensor<INPUT_T> mm2AL1Tensor;
        mm2AL1Tensor = l1PTensor;

        if (likely(runInfo.actVecMSize != 0)) {
            DataCopy(mm2AL1Tensor[constInfo.subBlockIdx * (blockBytes / sizeof(INPUT_T)) *
                                  (runInfo.actMSize - runInfo.actVecMSize)],
                     stage1CastTensor,
                     {s2BaseSize / 16, (uint16_t)runInfo.actVecMSize, (uint16_t)(vec1Srcstride - runInfo.actVecMSize),
                      (uint16_t)(mBaseSize - runInfo.actVecMSize)});
        }

        // stage1 双缓冲反向同步：MTE3 写完 L1 后释放当前 ping/pong 槽，下轮 V 可覆写
        SetFlag<HardEvent::MTE3_V>(FA_STAGE1_MTE3_V_FLAG_0 + stage1Offset);

        // L1 P 矩阵由 UB→L1 的 DataCopy(MTE3 流水) 产出，必须用 PIPE_MTE3 通知 AIC，
        // 否则 AIC 会在 MTE3 搬运完成前读取 L1，读到脏数据（首次运行尤为明显）。
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE3>(l1PCcId);
        // ======================================================
        if (!runInfo.isFirstS2Loop) {
            UpdateExpSumAndExpMax<T>(sumUb, maxUb, expUb, sumUb, maxUb, apiTmpBuffer, runInfo.actVecMSize);
        }

        if (unlikely(runInfo.isLastS2Loop)) {
            SoftmaxDataCopyOut(runInfo, sumUb, maxUb);
        }
    }

    __aicore__ inline void ProcessVec2OnUb(LocalTensor<T> &bmm2UbTensor, uint32_t bmm2CcId,
                                           RunInfoX runInfo)
    {
        if (unlikely(runInfo.actVecMSize == 0)) {
            CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(bmm2CcId);
            return;
        }

        int64_t vec2CalcSize = runInfo.actVecMSize * dTemplateAlign64;

        LocalTensor<T> vec2ResUb = this->stage2OutBuf;
        LocalTensor<T> mmRes = bmm2UbTensor;
        // stage2OutBuf 反向同步：等待上一轮 MTE3 搬出后，本轮 V 才能覆写
        WaitFlag<HardEvent::MTE3_V>(FA_VEC2_MTE3_V_FLAG);
        if (unlikely(runInfo.isFirstS2Loop)) {
            DataCopy(vec2ResUb, mmRes, vec2CalcSize);
        } else {
            LocalTensor<T> expUb = softmaxExpBuf[runInfo.loop % (PRELOAD_N + 1)];
            LocalTensor<T> pScaleUb;

            float deSCalePreVValue = 1.0f;
            if (!runInfo.isLastS2Loop) {
                FlashUpdateNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, false, false>(
                    vec2ResUb, mmRes, vec2ResUb, expUb, pScaleUb, runInfo.actVecMSize, dTemplateAlign64, 1.0, 1.0);
            } else {
                LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.mloop % (PRELOAD_N + 1)];
                FlashUpdateLastNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, false, false>(
                    vec2ResUb, mmRes, vec2ResUb, expUb, pScaleUb, sumUb, runInfo.actVecMSize, dTemplateAlign64, 1.0,
                    1.0);
            }
        }
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(bmm2CcId);
        if (runInfo.isLastS2Loop) {
            if (unlikely(runInfo.isFirstS2Loop)) {
                LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.mloop % (PRELOAD_N + 1)];
                LastDivNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, false>(
                    vec2ResUb, vec2ResUb, sumUb, runInfo.actVecMSize, (uint16_t)dTemplateAlign64, 0.0F);
            }
            CopyOutAttentionOut(runInfo, vec2ResUb, 0, runInfo.actVecMSize);
        }
        // stage2OutBuf 反向同步：MTE3 搬出后释放，下一轮 V 才能覆写
        SetFlag<HardEvent::MTE3_V>(FA_VEC2_MTE3_V_FLAG);
    }

    __aicore__ inline void Bmm2ResCastAndCopyOut(RunInfoX &runInfo, LocalTensor<T> &vec2ResUb, uint32_t mStartVec,
                                                 uint32_t mDealSize)
    {
        LocalTensor<OUTPUT_T> attenOut;
        int64_t dSizeAligned64 = (int64_t)dVTemplateType;

        attenOut.SetAddr(vec2ResUb.address_);

        RowInvalid(vec2ResUb, mStartVec, mDealSize, runInfo, dSizeAligned64);
        Cast(attenOut, vec2ResUb, RoundMode::CAST_ROUND, mDealSize * dSizeAligned64);
        // V → MTE3 正向：等待 cast 完成后再搬运到 GM
        SetFlag<HardEvent::V_MTE3>(FA_V_MTE3_FLAG_0);
        WaitFlag<HardEvent::V_MTE3>(FA_V_MTE3_FLAG_0);
        {
            Bmm2DataCopyOutTrans(runInfo, attenOut, mStartVec, mDealSize);
        }
    }

    __aicore__ inline void CopyOutAttentionOut(RunInfoX runInfo, LocalTensor<T> &vec2ResUb, uint32_t mStartVec,
                                               uint32_t mDealSize)
    {
        Bmm2ResCastAndCopyOut(runInfo, vec2ResUb, mStartVec, mDealSize);
    }

    __aicore__ inline bool CalcBlockNeedRowInvalid(RunInfoX &runInfo, int64_t s1FirstValidToken,
                                                   int64_t s1LastValidToken)
    {
        int32_t vecMStartIdx = runInfo.gS1Idx + runInfo.vecMbaseIdx;
        int32_t vecMEndIdx = vecMStartIdx + runInfo.actVecMSize - 1;
        int32_t s1StartTdx, s1EndTdx;
        bool ret = false;
        if constexpr (layout == LayOutTypeEnum::LAYOUT_BSH || layout == LayOutTypeEnum::LAYOUT_SBH ||
                      layout == LayOutTypeEnum::LAYOUT_TND) {
            // S1G layout
            s1StartTdx = vecMStartIdx / constInfo.gSize;
            s1EndTdx = vecMEndIdx / constInfo.gSize;
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
                                      RunInfoX &runInfo, int64_t dSizeAligned64)
    {
        if constexpr (hasAtten) {
            int64_t s1FirstValidToken = AttentionCommon::Min(AttentionCommon::Max(-runInfo.nextTokensLeftUp, 0), runInfo.actS1Size);
            int64_t s1LastValidToken = AttentionCommon::Min(AttentionCommon::Max(runInfo.preTokensLeftUp + runInfo.actS2Size, 0), runInfo.actS1Size);
            s1LastValidToken = AttentionCommon::Max(s1LastValidToken - 1, 0);
            bool hasValidRow = (s1FirstValidToken > 0) || (s1LastValidToken < runInfo.actS1Size);
            bool batchNeedRowInvalid = ((constInfo.sparseMode != SparseMode::LEFT_UP_CAUSAL) &&
                                        hasValidRow); // sparse = 0 or 3 or 4，preToekens or nextTokens负数
            if (!batchNeedRowInvalid) {
                return;
            }

            bool blockNeedRowInvalid = CalcBlockNeedRowInvalid(runInfo, s1FirstValidToken, s1LastValidToken);

            if (blockNeedRowInvalid) {
                LocalTensor<float> maxTensor =
                    softmaxMaxBuf[runInfo.mloop % (PRELOAD_N + 1)][mStartVec];
                RowInvalidUpdateVF<float>(vec2ResUb, maxTensor, mDealSize, constInfo.dSizeV,
                                          static_cast<uint32_t>(dSizeAligned64));
            }
        }
    }

    __aicore__ inline void Bmm2DataCopyOutTrans(const RunInfoX &info, LocalTensor<OUTPUT_T> &attenOutUb,
                                                uint32_t vecMIdx, uint32_t dealRowCount)
    {
        FaUbTensor<OUTPUT_T> ubTensor{.tensor = attenOutUb,
                                      .rowCount = dealRowCount,
                                      .colCount = (uint32_t)dTemplateAlign64};
        GmCoord gmCoord{.bIdx = info.bIdx,
                        .n2Idx = info.n2Idx,
                        .gS1Idx = info.gS1Idx + info.vecMbaseIdx + vecMIdx,
                        .dIdx = 0,
                        .gS1DealSize = dealRowCount,
                        .dDealSize = (uint32_t)constInfo.dSizeV};
        CopyAttentionOut(ubTensor, gmCoord);
    }

    __aicore__ inline void CopyAttentionOut(FaUbTensor<OUTPUT_T> &ubTensor, GmCoord &gmCoord)
    {
        if constexpr (outLayout == LayOutTypeEnum::LAYOUT_BSH) {
            constexpr GmFormat OUT_FORMAT = GmFormat::BSNGD;
            FaGmTensor<OUTPUT_T, OUT_FORMAT, int32_t> outGmTensor;
            outGmTensor.gmTensor = attentionOutGm;
            outGmTensor.offsetCalculator.Init(constInfo.bSize, constInfo.n2Size, constInfo.gSize, constInfo.s1Size,
                                              constInfo.dSizeV, *qActSeqLensParser);
            CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, GetOutUbFormat<layout>()> copyAttenOutUbToGm;
            copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
        } else if constexpr (outLayout == LayOutTypeEnum::LAYOUT_BNSD) {
            constexpr GmFormat OUT_FORMAT = GmFormat::BNGSD;
            FaGmTensor<OUTPUT_T, OUT_FORMAT, int32_t> outGmTensor;
            outGmTensor.gmTensor = attentionOutGm;
            outGmTensor.offsetCalculator.Init(constInfo.bSize, constInfo.n2Size, constInfo.gSize, constInfo.s1Size,
                                              constInfo.dSizeV, *qActSeqLensParser);
            CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, GetOutUbFormat<layout>()> copyAttenOutUbToGm;
            copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
        } else if constexpr (outLayout == LayOutTypeEnum::LAYOUT_TND) {
            constexpr GmFormat OUT_FORMAT = GmFormat::TNGD;
            FaGmTensor<OUTPUT_T, OUT_FORMAT, int32_t, true> outGmTensor;
            outGmTensor.gmTensor = attentionOutGm;
            outGmTensor.offsetCalculator.Init(constInfo.n2Size, constInfo.gSize, constInfo.dSizeV,
                                              *qActSeqLensParser);
            CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, GetOutUbFormat<layout>()> copyAttenOutUbToGm;
            copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
        }
    }

    template<typename BmmResT>
    __aicore__ inline void ProcessVec2(BmmResT &bmm2ResTensor, uint32_t bmm2CcId, RunInfoX runInfo)
    {
#ifdef SKIP_V2
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(bmm2CcId);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(bmm2CcId);
        return;
#endif

        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(bmm2CcId);
        ProcessVec2OnUb(bmm2ResTensor, bmm2CcId, runInfo);
        return;
    }

    __aicore__ inline void SoftmaxInitBuffer(uint32_t off_sm)
    {
        // softmaxSum×3 + softmaxMax×3 + softmaxExp×3，各 256 bytes；maxBrdcst/sumBrdcst 各 2048 bytes
        softmaxSumBuf[0] = LocalTensor<uint8_t>(TPosition::VECIN, off_sm,          256U).template ReinterpretCast<float>();
        softmaxSumBuf[1] = LocalTensor<uint8_t>(TPosition::VECIN, off_sm + 256U,   256U).template ReinterpretCast<float>();
        softmaxSumBuf[2] = LocalTensor<uint8_t>(TPosition::VECIN, off_sm + 512U,   256U).template ReinterpretCast<float>();
        softmaxMaxBuf[0] = LocalTensor<uint8_t>(TPosition::VECIN, off_sm + 768U,   256U).template ReinterpretCast<float>();
        softmaxMaxBuf[1] = LocalTensor<uint8_t>(TPosition::VECIN, off_sm + 1024U,  256U).template ReinterpretCast<float>();
        softmaxMaxBuf[2] = LocalTensor<uint8_t>(TPosition::VECIN, off_sm + 1280U,  256U).template ReinterpretCast<float>();
        softmaxExpBuf[0] = LocalTensor<uint8_t>(TPosition::VECIN, off_sm + 1536U,  256U).template ReinterpretCast<T>();
        softmaxExpBuf[1] = LocalTensor<uint8_t>(TPosition::VECIN, off_sm + 1792U,  256U).template ReinterpretCast<T>();
        softmaxExpBuf[2] = LocalTensor<uint8_t>(TPosition::VECIN, off_sm + 2048U,  256U).template ReinterpretCast<T>();
    }

    __aicore__ inline void InitBuffers()
    {
        // 完全自主管理：使用 LocalTensor 构造函数直接指定绝对字节偏移，
        // 无需 LocalMemAllocator 线性分配器。所有偏移为编译期常量。
        constexpr uint32_t mm1Sz = mBaseSize / CV_RATIO * s2BaseSize * sizeof(T);
        constexpr uint32_t mm2Sz = mBaseSize / CV_RATIO * (uint32_t)dVTemplateType * sizeof(T);

        constexpr uint32_t BASE = mm1Sz * 2U + mm2Sz * 2U;

        // attenMask ping/pong（各 8192 bytes）
        attenMaskBuf[0] = LocalTensor<uint8_t>(TPosition::VECIN, BASE,          8192U);
        attenMaskBuf[1] = LocalTensor<uint8_t>(TPosition::VECIN, BASE + 8192U,  8192U);
        // stage2 输出（32768 bytes）
        stage2OutBuf    = LocalTensor<uint8_t>(TPosition::VECIN, BASE + 16384U, 32768U).template ReinterpretCast<T>();
        // stage1 输出 ping/pong（各 33024 bytes）
        stage1OutBuf[0] = LocalTensor<uint8_t>(TPosition::VECIN, BASE + 49152U, 33024U).template ReinterpretCast<INPUT_T>();
        stage1OutBuf[1] = LocalTensor<uint8_t>(TPosition::VECIN, BASE + 82176U, 33024U).template ReinterpretCast<INPUT_T>();

        // Softmax 中间值：2304 bytes，紧跟 stage1[1] 末尾（BASE + 115200）
        SoftmaxInitBuffer(BASE + 115200U);

        // commonBuf：512 bytes
        commonBuf = LocalTensor<uint8_t>(TPosition::VECIN, BASE + 117504U, 512U);

        // lseBuf / sinkBuf：runtime 条件，地址已在编译期固定
        constexpr uint32_t OFF_LSE = BASE + 118016U;
        constexpr uint32_t LSE_BUF_SZ = (mBaseSize >> 1U) * 8U * sizeof(float);
        if (constInfo.isSoftmaxLseEnable) {
            lseBuf = LocalTensor<uint8_t>(TPosition::VECIN, OFF_LSE,
                         LSE_BUF_SZ).template ReinterpretCast<float>();
        }
        if (constInfo.learnableSinkFlag) {
            sinkBuf = LocalTensor<uint8_t>(TPosition::VECIN, OFF_LSE + LSE_BUF_SZ,
                          256U).template ReinterpretCast<INPUT_T>();
        }
    }

    // 反向同步事件预置 SetFlag：完全静态、直接使用编译期常量（参考 quant_flash_attn AllocEventID 风格）
    __aicore__ inline void AllocEventID()
    {
        SetFlag<HardEvent::MTE3_V>(FA_VEC2_MTE3_V_FLAG);
        SetFlag<HardEvent::MTE3_V>(FA_LSE_MTE3_V_FLAG);
        SetFlag<HardEvent::MTE3_V>(FA_STAGE1_MTE3_V_FLAG_0);
        SetFlag<HardEvent::MTE3_V>(FA_STAGE1_MTE3_V_FLAG_1);
    }

    __aicore__ inline void FreeEventID()
    {
        WaitFlag<AscendC::HardEvent::MTE3_V>(FA_VEC2_MTE3_V_FLAG);
        WaitFlag<AscendC::HardEvent::MTE3_V>(FA_LSE_MTE3_V_FLAG);
        WaitFlag<AscendC::HardEvent::MTE3_V>(FA_STAGE1_MTE3_V_FLAG_0);
        WaitFlag<AscendC::HardEvent::MTE3_V>(FA_STAGE1_MTE3_V_FLAG_1);
    }

    __aicore__ inline void AttenMaskCopyIn(LocalTensor<uint8_t> attenMaskUb, uint32_t vecMIdx, uint32_t mDealSize,
                                           RunInfoX &runInfo)
    {
        const uint32_t bufIdx = runInfo.loop & (DB - 1);
        MaskInfo maskInfo;
        maskInfo.gs1StartIdx = runInfo.gS1Idx + runInfo.vecMbaseIdx + vecMIdx;
        maskInfo.gs1dealNum = mDealSize;
        maskInfo.s1Size = runInfo.actS1Size;
        maskInfo.gSize = constInfo.gSize;
        maskInfo.s2StartIdx = runInfo.s2Idx;
        maskInfo.s2dealNum = runInfo.actSingleLoopS2Size;
        maskInfo.s2Size = runInfo.actS2Size;
        maskInfo.nBaseSize = s2BaseSize;
        maskInfo.preToken = constInfo.preTokens;
        maskInfo.nextToken = constInfo.nextTokens;
        maskInfo.sparseMode = static_cast<SparseMode>(constInfo.sparseMode);
        maskInfo.batchIdx = (constInfo.attenMaskBatch == 1) ? 0 : runInfo.bIdx;
        maskInfo.attenMaskBatchStride = constInfo.attenMaskS1Size * constInfo.attenMaskS2Size;
        maskInfo.attenMaskS1Stride = constInfo.attenMaskS2Size;
        maskInfo.attenMaskDstStride = (s2BaseSize - AttentionCommon::Align(maskInfo.s2dealNum, 32U)) / 32;
        maskInfo.maskValue = negativeIntScalar;
        maskInfo.s1LeftPaddingSize = 0;
        maskInfo.s2LeftPaddingSize = 0;
        maskInfo.layout = MASK_LAYOUT;
        maskInfo.attenMaskType = MASK_BOOL; // compatible with int8/uint8

        bool IsSkipMask = IsSkipAttentionmask(maskInfo);
        bool IsSkipMaskForPre = IsSkipAttentionmaskForPre(maskInfo);

        // ping/pong 直接通过 BASE + bufIdx 数学公式得到（参考 quant_flash_attn 的 SYNC_xxx + buffIdx 风格）
        const uint32_t mte2ToVId    = FA_ATTEN_MASK_MTE2_V_FLAG_0 + bufIdx;
        const uint32_t preMte2ToVId = FA_ATTEN_MASK_MTE2_V_FLAG_0 + (bufIdx ^ 1U);

        // 静态Tensor迁移：attenMaskBuf 为固定地址裸 UB，运行时不会在 kernel 启动间清零。
        // mask 的 DMA 仅写入有效 s1 行 × s2dealNum 列，行尾 padding / 无效行不写。
        // 批跑时这些未写区域残留上一个用例的脏 mask 位，会被 MergeMask / softmax 误当作屏蔽位读入，
        // 导致整行 attention/LSE 崩坏（单跑因 UB 为全新清零内存而恰好通过）。
        // 因此在 DMA 覆盖前整块清零，保证批跑与单跑行为一致。
        ZeroMaskBuf(attenMaskUb, maskInfo.gs1dealNum * s2BaseSize);

        if (IsSkipMask && IsSkipMaskForPre) {
            return; // 已清零，等价于全不屏蔽
        }

        if (!IsSkipMask) {
            AttentionmaskCopyIn<uint8_t, MASK_LAYOUT, true, s2BaseSize>(
                attenMaskUb, attenMaskGmInt, maskInfo, false, mte2ToVId);
        }
        // else: 已整块清零，无需再次 Duplicate

        if (!IsSkipMaskForPre) {
            const uint32_t preBufIdx = 1U - bufIdx;
            LocalTensor<uint8_t> attenMaskUbPre = this->attenMaskBuf[preBufIdx];
            ZeroMaskBuf(attenMaskUbPre, maskInfo.gs1dealNum * s2BaseSize);
            AttentionmaskCopyIn<uint8_t, MASK_LAYOUT, true, s2BaseSize>(
                attenMaskUbPre, attenMaskGmInt, maskInfo, true, preMte2ToVId);
            MergeMask(attenMaskUb, attenMaskUbPre, maskInfo.gs1dealNum, s2BaseSize);
            PipeBarrier<PIPE_V>();
        }
    }

    // 整块清零 mask buffer，并插入 V→MTE2 同步，保证清零(V)先于后续 mask DMA(MTE2) 写入同一 UB。
    __aicore__ inline void ZeroMaskBuf(LocalTensor<uint8_t> &maskUb, uint32_t numElements)
    {
        Duplicate(maskUb, static_cast<uint8_t>(0U), numElements);
        SetFlag<HardEvent::V_MTE2>(FA_ATTEN_MASK_V_MTE2_0);
        WaitFlag<HardEvent::V_MTE2>(FA_ATTEN_MASK_V_MTE2_0);
    }

    __aicore__ inline void DealZeroActSeqLen(uint32_t bN2Cur)
    {
        uint32_t n2Idx = bN2Cur % constInfo.n2Size;
        uint32_t bIdx = bN2Cur / constInfo.n2Size;
        if constexpr (outLayout == LayOutTypeEnum::LAYOUT_BSH) {
            OffsetCalculator<GmFormat::BSNGD, int32_t> offsetCalculator;
            offsetCalculator.Init(constInfo.bSize, constInfo.n2Size, constInfo.gSize, constInfo.s1Size,
                                  constInfo.dSizeV, *qActSeqLensParser);
            DealActSeqLenIsZero<GmFormat::BSNGD, OUTPUT_T>(bIdx, n2Idx, offsetCalculator, attentionOutGm);
        } else if constexpr (outLayout == LayOutTypeEnum::LAYOUT_BNSD) {
            OffsetCalculator<GmFormat::BNGSD, int32_t> offsetCalculator;
            offsetCalculator.Init(constInfo.bSize, constInfo.n2Size, constInfo.gSize, constInfo.s1Size,
                                  constInfo.dSizeV, *qActSeqLensParser);
            DealActSeqLenIsZero<GmFormat::BNGSD, OUTPUT_T>(bIdx, n2Idx, offsetCalculator, attentionOutGm);
        } else if constexpr (outLayout == LayOutTypeEnum::LAYOUT_TND) {
            OffsetCalculator<GmFormat::TNGD, int32_t, true> offsetCalculator;
            offsetCalculator.Init(constInfo.n2Size, constInfo.gSize, constInfo.dSizeV, *qActSeqLensParser);
            DealActSeqLenIsZero<GmFormat::TNGD, OUTPUT_T>(bIdx, n2Idx, offsetCalculator, attentionOutGm);
        }
    }
};
} // namespace BaseApi
#endif // FLASH_ATTENTION_NOQUANT_GQA_BLOCK_VEC_H_