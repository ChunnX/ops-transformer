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
 * \brief FlashAttn arch35 AIV-side kernel (non-quant GQA)
 */

#ifndef FLASH_ATTN_GQA_VEC_H_
#define FLASH_ATTN_GQA_VEC_H_

#include "../../../common/op_kernel/fia_public_define.h"

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#include "../../../common/op_kernel/arch35/util_regbase.h"
#include "../../../common/op_kernel/const_def.h"
using regbaseutil::regBytes;
using AttentionCommon::Align;

#include "../utils/attenmask_gs1.h"

#include "kernel_operator_list_tensor_intf.h"
#include "../utils/flash_attn_common_def.h"
#include "../utils/flash_attn_vec_def.h"
#include "../utils/flash_attn_utils.h"

#include "../../../common/op_kernel/arch35/flash_attention_score_common_regbase.h"
#include "adv_api/activation/softmax.h"
#include "../../../common/op_kernel/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz.h"
#include "../../../common/op_kernel/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz_dn.h"
#include "fa_vec1_vf_static.h"
#include "../../../common/op_kernel/arch35/vf/vf_flashupdate_new.h"
#include "../../../common/op_kernel/arch35/vf/vf_div_cast.h"
#include "../../../common/op_kernel/arch35/vf/vf_flash_decode.h"
#include "../../../common/op_kernel/vector_common.h"

#include "../../../common/op_kernel/memory_copy_arch35.h"

using namespace AscendC;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;
using namespace FaVectorApi;
using namespace AttentionCommon;

#include "flash_attention_noquant_block_vec_flashdecode.h"

#include "flash_attn_tiling_data.h"
#include "flash_attn_task_loop_hooks.h"

using namespace optiling;

namespace BaseApi {
template <
    typename INPUT_T, typename T, typename OUTPUT_T, typename VecFdBlockType,
    LayOutTypeEnum layout = LayOutTypeEnum::None,
    LayOutTypeEnum outLayout = LayOutTypeEnum::None, S1TemplateType s1TemplateType = S1TemplateType::Aligned128,
    S2TemplateType s2TemplateType = S2TemplateType::Aligned128, DTemplateType dTemplateType = DTemplateType::Aligned128,
    DTemplateType dVTemplateType = DTemplateType::Aligned128, bool hasAtten = false, bool hasDrop = false,
    uint8_t KvLayoutType = 0, bool useDn = false, bool bmm2Write2Ub = true, bool splitD = false>
class FlashAttentionNoQuantGqaKernel {
public:
    /* =================编译期常量的基本块信息================= */
    static constexpr uint32_t mBaseSize = (uint32_t)s1TemplateType;
    static constexpr uint32_t s2BaseSize = (uint32_t)s2TemplateType;
    static constexpr uint32_t dBaseSize = (uint32_t)dTemplateType;
    static constexpr uint32_t dVBaseSize = (uint32_t)dVTemplateType;
    static constexpr uint32_t vec1S2CopyLenDn = s2BaseSize >> 1;
    static constexpr uint32_t vec1HalfS1BaseSize = mBaseSize >> 1;
    static constexpr uint32_t vec1S2CopyCountDn = mBaseSize >> 5;
    static constexpr uint32_t vec1S2strideDn = s2BaseSize * 8;
    static constexpr uint32_t vec1ResOffsetDn = s2BaseSize * 32 + 64;
    static constexpr uint32_t vec1Srcstride = (mBaseSize >> 1) + 1;
    static constexpr uint32_t dTemplateAlign64 = Align64Func((uint16_t)dVTemplateType);

    static constexpr bool USE_DN = useDn;
    static constexpr bool BMM2_TOUB = bmm2Write2Ub;
    static constexpr bool HAS_PREFIX = false;
    static constexpr bool PAGE_ATTENTION = (KvLayoutType > 0);
    static constexpr LayOutTypeEnum LAYOUT = layout;

    using Q_T = INPUT_T;
    using MM_T = T;

    static constexpr uint32_t DB = 2;
    // 索引使用 loop & (DB - 1) 代替 loop % DB，要求 DB 必须是2的幂，否则位掩码结果错误
    static_assert(DB > 0 && (DB & (DB - 1)) == 0, "DB must be a power of two for bitmask indexing");
    static constexpr uint32_t PRELOAD_N = 2;
    static constexpr uint32_t PRELOAD_TASK_CACHE_SIZE = 4;
    static constexpr uint32_t PRELOAD_TASK_CACHE_MASK = PRELOAD_TASK_CACHE_SIZE - 1;
    static_assert(PRELOAD_TASK_CACHE_SIZE >= PRELOAD_N + 1,
                  "PRELOAD_TASK_CACHE_SIZE must be at least PRELOAD_N + 1");
    static_assert((PRELOAD_TASK_CACHE_SIZE & PRELOAD_TASK_CACHE_MASK) == 0,
                   "PRELOAD_TASK_CACHE_SIZE must be a power of two for bitmask indexing");
    static constexpr bool HAS_MASK = hasAtten;

    static constexpr uint32_t initOutputEventId = 0U; // attenOut和lse，刷无效行会用到剩余ub，需要加同步
    // 静态 event ID 常量（避开 6/7，都在各自硬件通道独立）
    static constexpr uint32_t FA_MTE3_TO_V_0     = 0U;
    static constexpr uint32_t FA_MTE3_TO_V_1     = 1U;
    static constexpr uint32_t FA_V_TO_MTE3_0     = 0U;
    static constexpr uint32_t FA_V_TO_MTE3_1     = 1U;
    static constexpr uint32_t FA_MTE2_TO_V_ID    = 2U;
    static constexpr uint32_t FA_V_TO_MTE2_ID    = 2U;
    static constexpr uint32_t FA_MTE3_TO_MTE2_ID = 2U;
    static constexpr uint32_t FA_MTE2_TO_V_SINK  = 4U;
    // stage1 双缓冲的 MTE3→V 反向同步（MTE3_V 通道 3/5）
    static constexpr uint32_t FA_STAGE1_MTE3_TO_V_0 = 3U;
    static constexpr uint32_t FA_STAGE1_MTE3_TO_V_1 = 5U;
    // attenMask ping/pong 的 MTE2→V 正向同步（MTE2_V 通道 0/1，与 vec2=2、sink=4 错开）
    static constexpr uint32_t FA_ATTEN_MASK_MTE2_V_0 = 0U;
    static constexpr uint32_t FA_ATTEN_MASK_MTE2_V_1 = 1U;
    // attenMask 整块清零(V)先于 mask DMA(MTE2) 的反向同步（V_MTE2 通道 0，未被其他路径占用）
    static constexpr uint32_t FA_ATTEN_MASK_V_MTE2_0 = 0U;

    // ===== 核间同步常量（避开 6、7，Set/Wait 对称）=====
    // 注：跨核 barrier 沿用 AscendC 内置 SyncAll() 硬同步接口（占用 flagId [11-14]），
    // 与上面的 mode 4 flagId [0-9] 完全不冲突，是 Mix 算子的标准做法。
    static constexpr uint64_t CROSS_CORE_SYNC_MODE = 4U;

    static constexpr ActualSeqLensMode Q_MODE = GetQActSeqMode<layout>();
    static constexpr ActualSeqLensMode KV_MODE = GetKvActSeqMode<layout, PAGE_ATTENTION>();
    static constexpr LAYOUT_Q MASK_LAYOUT =
        (layout == LayOutTypeEnum::LAYOUT_BSH || layout == LayOutTypeEnum::LAYOUT_TND ||
         layout == LayOutTypeEnum::LAYOUT_SBH) ?
            LAYOUT_Q::SG :
            LAYOUT_Q::GS;

    using attenMaskGmType = typename std::conditional<hasAtten, GlobalTensor<uint8_t>, int8_t>::type;

    using vec2ResGmType = typename std::conditional<splitD, GlobalTensor<float>, int8_t>::type;

    using flashdecodeGmType = GlobalTensor<float>;

    using ConstInfoX = ConstInfoXVec;
    using RunInfoType = RunInfoXVec;

    struct TaskLoopHooks {
        static constexpr bool HAS_DEAL_ZERO = true;

        static __aicore__ inline void InitLoopState(FlashAttentionNoQuantGqaKernel &) {}

        static __aicore__ inline void ResetLoopState(FlashAttentionNoQuantGqaKernel &self)
        {
            self.mloop = 0;
            self.accGS1Loops = 0;
            self.headS2Split = false;
            self.tailS2Split = false;
        }

        static __aicore__ inline void GetSectionInfo(FlashAttentionNoQuantGqaKernel &self, uint32_t sectionIdx)
        {
            self.GetFASectionInfo(sectionIdx);
        }

        static __aicore__ inline void CreateTask(FlashAttentionNoQuantGqaKernel &self,
                                                  RunInfoXVec taskRunInfo[PRELOAD_TASK_CACHE_SIZE],
                                                  uint64_t loop, uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur)
        {
            self.CreateTask(loop, bN2Cur, gS1Cur, s2Cur, taskRunInfo);
        }

        static __aicore__ inline void ExecuteTask(FlashAttentionNoQuantGqaKernel &self,
                                                   RunInfoXVec taskRunInfo[PRELOAD_TASK_CACHE_SIZE],
                                                   uint64_t loop)
        {
            self.ExecuteTask(loop, taskRunInfo);
        }

        static __aicore__ inline void UpdateAxisInfo(FlashAttentionNoQuantGqaKernel &self, TASK_DEAL_MODE taskDealMode,
                                                      uint32_t &bN2Cur, uint32_t &gS1Cur, uint32_t &s2Cur)
        {
            self.UpdateAxisInfo(taskDealMode, bN2Cur, gS1Cur, s2Cur);
        }

        static __aicore__ inline void DealZeroActSeqLen(FlashAttentionNoQuantGqaKernel &self, uint32_t bN2Cur)
        {
            self.DealZeroActSeqLen(bN2Cur);
        }
    };

    // gm
    GlobalTensor<OUTPUT_T> attentionOutGm;

    using QSeqParserType =
        typename std::conditional<(LAYOUT == LayOutTypeEnum::LAYOUT_TND),
                                  ActualSeqLensParser<Q_MODE, int32_t, true>,
                                  ActualSeqLensParser<Q_MODE, int32_t>>::type;

    using KvSeqParserType =
        typename std::conditional<(LAYOUT == LayOutTypeEnum::LAYOUT_TND),
                                  ActualSeqLensParser<KV_MODE, int32_t, true>,
                                  ActualSeqLensParser<KV_MODE, int32_t>>::type;

    QSeqParserType qSeqLensParser;
    KvSeqParserType kvSeqLensParser;

    attenMaskGmType attenMaskGmInt;

    vec2ResGmType vec2ResGm[3];

    flashdecodeGmType accumOutGm;
    flashdecodeGmType softmaxFDSumGm;
    flashdecodeGmType softmaxFDMaxGm;

    // ub
    LocalTensor<uint8_t> commonBuf;
    LocalTensor<INPUT_T> stage1OutBuf[2];
    LocalTensor<uint8_t> attenMaskBuf[2];
    LocalTensor<T> stage2OutBuf;
    static constexpr uint32_t kMte3ToVId[2] = {FA_MTE3_TO_V_0, FA_MTE3_TO_V_1};
    static constexpr uint32_t kVToMte3Id[2] = {FA_V_TO_MTE3_0, FA_V_TO_MTE3_1};
    static constexpr uint32_t kStage1MTE3toV[2] = {FA_STAGE1_MTE3_TO_V_0, FA_STAGE1_MTE3_TO_V_1};
    LocalTensor<float> softmaxMaxBuf[PRELOAD_N + 1];
    LocalTensor<float> softmaxSumBuf[PRELOAD_N + 1];
    LocalTensor<T> softmaxExpBuf[PRELOAD_N + 1];
    LocalTensor<uint8_t> vselrIndexesBuf[4]; // 静态Tensor编程：LocalTensor 直接替代 TBuf<>
    LocalTensor<T> mm2InBuf;
    LocalTensor<float> maxBrdcstBuf;
    LocalTensor<float> sumBrdcstBuf;
    LocalTensor<float> lseBuf;
    LocalTensor<INPUT_T> sinkBuf;

    int64_t bmm2SubBlockOffset = 0;
    int64_t vec2SubBlockOffset = 0;

    // ==================== Functions ======================

    __aicore__ inline void InitVecBlock(__gm__ uint8_t *actualSeqQlenAddr,
                                        __gm__ uint8_t *actualSeqKvlenAddr, __gm__ uint8_t *attenMask,
                                        __gm__ uint8_t *learnableSink, __gm__ uint8_t *softmaxLse,
                                        __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace)
    {
        InitVecInput(actualSeqQlenAddr, actualSeqKvlenAddr, attenMask, learnableSink, softmaxLse, attentionOut,
                     workspace);
    }

    __aicore__ inline void InitVecInput(__gm__ uint8_t *actualSeqQlenAddr, __gm__ uint8_t *actualSeqKvlenAddr,
                                        __gm__ uint8_t *attenMask, __gm__ uint8_t *learnableSink,
                                        __gm__ uint8_t *softmaxLse, __gm__ uint8_t *attentionOut,
                                        __gm__ uint8_t *workspace)
    {
        attentionOutGm.SetGlobalBuffer((__gm__ OUTPUT_T *)attentionOut);

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
    __aicore__ inline void ProcessVec1(LocalTensor<INPUT_T> &l1PTensor, uint32_t l1PCcId,
                                       LocalTensor<T> &bmm1UbTensor, uint32_t bmm1CcId,
                                       RunInfoXVec runInfo)
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

        if constexpr (useDn) {
            ProcessVec1Dn(l1PTensor, l1PCcId, bmm1UbTensor, bmm1CcId, runInfo);
        } else {
            ProcessVec1Nd(l1PTensor, l1PCcId, bmm1UbTensor, bmm1CcId, runInfo);
        }
    }

    __aicore__ inline void ClearOutput()
    {
        bool needOutputInit = IsInitAttentionOutGm();
        // 静态Tensor迁移：FD 中间结果 GM workspace(accumOut/softmaxFDSum/softmaxFDMax) 复用同一段
        // user workspace，运行时不会在 kernel 启动间清零。稀疏(mask)场景下 FA 实际写入的 split 槽位
        // 可能少于 FD combine 读取的槽位；未写槽位在批跑时残留上一个用例的大值(attention/LSE)，导致
        // 第二个及之后的 mask 用例 FD 规约读到脏数据而精度崩坏（单跑/首个用例因 workspace 为全新清零
        // 内存而恰好通过）。因此 isFd 时在 kernel 入口整体清零该 workspace，使批跑与单跑行为一致。
        bool needFdWsInit = this->isFd;
        if (needOutputInit || needFdWsInit) {
            SetFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId); // 释放剩余ub
            if (needOutputInit) {
                InitOutputSingleCore();
                if (constInfoXVec.isSoftmaxLseEnable) {
                    InitLseOutputSingleCore();
                }
            }
            if (needFdWsInit) {
                InitFdWorkspaceSingleCore();
            }
            WaitFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
            // 跨核 barrier：使用 AscendC 内置 SyncAll() 硬同步接口。
            // - 默认 isAIVOnly=true，仅同步全部 AIV 核，与 ClearOutput 仅 AIV 进入的语义一致；
            // - SyncAll 内部占用 flagId [11-14]，与本算子静态分配的 mode 4 flagId [0-9] 不冲突；
            // - 相比手动 CrossCoreSetFlag<0,PIPE_ALL>，SyncAll 内部 pipe/Set/Wait 配对经过充分验证，
            //   不会触发 aicore subErrType 0x4 timeout-trap。
            AscendC::SyncAll();
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
            tSize = qSeqLensParser.GetTSize();
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
            tSize = qSeqLensParser.GetTSize();
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

    __aicore__ inline void InitFdWorkspaceSingleCore()
    {
        // accumOut / softmaxFDSum / softmaxFDMax 在 user workspace 上连续排布
        // （见 InitVecInput：accumOut@workspace, softmaxFDSum@+accumOutSize, softmaxFDMax@+accumOutSize+logSumExpSize），
        // 总大小为 accumOutSize + 2 * logSumExpSize（单位：float 元素）。整体清零以消除批跑跨启动的脏残留。
        int64_t totalFdWsSize = (int64_t)constInfoXVec.accumOutSize + 2 * (int64_t)constInfoXVec.logSumExpSize;
        if (totalFdWsSize <= 0) {
            return;
        }
        int64_t singleCoreSize =
            (totalFdWsSize + (2 * constInfoXVec.coreNum) - 1) / (2 * constInfoXVec.coreNum); // 2 means c:v = 1:2
        int64_t tailSize = totalFdWsSize - constInfoXVec.aivIdx * singleCoreSize;
        int64_t singleInitSize = tailSize < singleCoreSize ? tailSize : singleCoreSize;

        if (singleInitSize > 0) {
            WaitFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
            matmul::InitOutput<float>(accumOutGm[constInfoXVec.aivIdx * singleCoreSize], singleInitSize, 0);
            SetFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
        }
    }

    // =================================Private Functions=================================

    __aicore__ inline void ProcessVec1Dn(LocalTensor<INPUT_T> &l1PTensor, uint32_t l1PCcId,
                                          LocalTensor<T> &bmm1UbTensor, uint32_t bmm1CcId,
                                          RunInfoXVec runInfo)
    {
        const T negativeFloatScalar = -3.4028235e38F;
        LocalTensor<uint8_t> attenMaskUb;

        LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.mloop % (PRELOAD_N + 1)][0];
        LocalTensor<float> maxUb = this->softmaxMaxBuf[runInfo.mloop % (PRELOAD_N + 1)][0];

        auto expUb = this->softmaxExpBuf[runInfo.loop % (PRELOAD_N + 1)][0];
        int64_t stage1Offset = runInfo.loop & (DB - 1);

        WaitFlag<HardEvent::MTE3_V>(kStage1MTE3toV[stage1Offset]);

        float descaleQK = 1.0;

        LocalTensor<T> mmRes = bmm1UbTensor;
        LocalTensor<INPUT_T> stage1CastTensor = this->stage1OutBuf[stage1Offset];
        if (unlikely(runInfo.isFirstS2Loop)) {
            fa_static::ProcessVec1VfDn<T, INPUT_T, false, false, s2BaseSize>(
                stage1CastTensor, sumUb, maxUb, mmRes, expUb, this->vselrIndexesBuf, attenMaskUb,
                runInfo.actMSizeAlign32 >> 1, runInfo.actSingleLoopS2SizeAlign, runInfo.actSingleLoopS2Size,
                static_cast<T>(constInfoXVec.scaleValue), descaleQK, negativeFloatScalar, 0.0F, false);
        } else {
            fa_static::ProcessVec1VfDn<T, INPUT_T, true, false, s2BaseSize>(
                stage1CastTensor, sumUb, maxUb, mmRes, expUb, this->vselrIndexesBuf, attenMaskUb,
                runInfo.actMSizeAlign32 >> 1, runInfo.actSingleLoopS2SizeAlign, runInfo.actSingleLoopS2Size,
                static_cast<T>(constInfoXVec.scaleValue), descaleQK, negativeFloatScalar, 0.0F, false);
        }
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(bmm1CcId);

        SetFlag<HardEvent::V_MTE3>(kVToMte3Id[stage1Offset]);
        WaitFlag<HardEvent::V_MTE3>(kVToMte3Id[stage1Offset]);
        //-------------------------Data copy to l1-------------------------
        LocalTensor<INPUT_T> mm2AL1Tensor = l1PTensor;

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
        SetFlag<HardEvent::MTE3_V>(kStage1MTE3toV[stage1Offset]);
        // L1 P 矩阵由 UB→L1 的 DataCopy(MTE3 流水) 产出，必须用 PIPE_MTE3 通知 AIC，
        // 否则 AIC 会在 MTE3 搬运完成前读取 L1，读到脏数据（首次运行尤为明显）。
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE3>(l1PCcId);
        //----------------------------------------------------------------
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

        if (constInfoXVec.isSoftmaxLseEnable && (!this->isFd || !runInfo.isS2SplitCore)) {
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
        LocalTensor<float> &lseUb = this->lseBuf;
        ComputeLseOutputVF(lseUb, softmaxSumTmp, softmaxMaxTmp, runInfo.actVecMSize);
        SetFlag<HardEvent::V_MTE3>(kVToMte3Id[0]);
        WaitFlag<HardEvent::V_MTE3>(kVToMte3Id[0]);

        if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
            uint32_t prefixBS1 = qSeqLensParser.GetTBase(runInfo.bIdx);
            uint64_t bN2Offset = runInfo.n2Idx * constInfoXVec.gSize * constInfoXVec.t1Size + prefixBS1;
            DataCopySoftmaxLseTNDtoNTArch35<T, ConstInfoXVec>(softmaxLseGm, lseUb, bN2Offset, vecMIdx, runInfo.actVecMSize,
                                                           constInfoXVec);
        } else if constexpr (layout == LayOutTypeEnum::LAYOUT_BSH) {
            uint64_t bN2Offset = runInfo.bIdx * constInfoXVec.n2Size * constInfoXVec.gSize * constInfoXVec.s1Size +
                                 runInfo.n2Idx * constInfoXVec.gSize * constInfoXVec.s1Size;
            uint64_t qActSeqLens = qSeqLensParser.GetActualSeqLength(runInfo.bIdx);
            DataCopySoftmaxLseBSNDArch35<T, ConstInfoXVec>(softmaxLseGm, lseUb, bN2Offset, vecMIdx, runInfo.actVecMSize,
                                                         constInfoXVec);
        } else { // BNSD
            uint64_t bN2Offset = runInfo.bIdx * constInfoXVec.n2Size * constInfoXVec.gSize * constInfoXVec.s1Size +
                                 runInfo.n2Idx * constInfoXVec.gSize * constInfoXVec.s1Size;
            uint64_t qActSeqLens = qSeqLensParser.GetActualSeqLength(runInfo.bIdx);
            DataCopySoftmaxLseBNSDArch35<T, ConstInfoXVec>(softmaxLseGm, lseUb, bN2Offset, vecMIdx, runInfo.actVecMSize,
                                                         constInfoXVec, qActSeqLens);
        }
        SetFlag<HardEvent::MTE3_V>(kMte3ToVId[1]);
        WaitFlag<HardEvent::MTE3_V>(kMte3ToVId[1]);
    }

    __aicore__ inline void Vec1SinkCompute(RunInfoXVec &runInfo, LocalTensor<float> &sumUb, LocalTensor<float> &maxUb)
    {
        int64_t sinkOffset = runInfo.n2Idx * constInfoXVec.gSize + runInfo.gIdx;
        auto sinkRaw = this->sinkGm.GetValue(sinkOffset);
        float sinkValue;
        if constexpr (IsSameType<decltype(sinkRaw), half>::value) {
            sinkValue = static_cast<float>(sinkRaw);
        } else {
            sinkValue = ToFloat(sinkRaw);
        }
        SinkSubExpAddVF<float>(sumUb, maxUb, sinkValue, runInfo.actVecMSize);
    }

    __aicore__ inline void Vec1SinkComputeGSFused(RunInfoXVec &runInfo, LocalTensor<float> &sumUb,
                                                  LocalTensor<float> &maxUb)
    {
        CopySinkIn(runInfo);
        WaitFlag<HardEvent::MTE2_V>(FA_MTE2_TO_V_SINK);
        SinkSubExpAddGSFusedVF<float, INPUT_T>(this->sinkBuf, sumUb, maxUb, runInfo.actVecMSize);
    }

    __aicore__ inline void CopySinkIn(RunInfoXVec &runInfo)
    {
        int64_t sinkOffset =
            runInfo.n2Idx * constInfoXVec.gSize + constInfoXVec.subBlockIdx * (runInfo.actMSize - runInfo.actVecMSize);
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
                                          RunInfoXVec runInfo)
    {
        const T negativeFloatScalar = -3.4028235e38F;
        LocalTensor<uint8_t> dropMaskUb;
        LocalTensor<INPUT_T> nonePseUb; // PSE不支持，占位
        LocalTensor<uint8_t> attenMaskUb;
        LocalTensor<uint8_t> attenMaskUbPre;
        if constexpr (hasAtten) {
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

        WaitFlag<HardEvent::MTE3_V>(kStage1MTE3toV[stage1Offset]);

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
                    0 /*pseStride*/, 0.0f /*slopes*/, 0.0f /*posShift*/, static_cast<T>(constInfoXVec.scaleValue),
                    descaleQK, negativeFloatScalar, 0.0F, queryScaleUb, deSCaleKValue);
            } else if (runInfo.actSingleLoopS2Size <= 64) {
                fa_static::ProcessVec1Vf<T, INPUT_T, INPUT_T /*pseShiftType*/, false, mBaseSize, s2BaseSize, GT_0_AND_LTE_64,
                              hasAtten, PseTypeEnum::PSE_NONE_TYPE, hasDrop, false, false>(
                    stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, mmRes, expUb, sumUb, maxUb, attenMaskUb,
                    nonePseUb, dropMaskUb, apiTmpBuffer, pScaleUb, runInfo.actVecMSize, runInfo.actSingleLoopS2Size,
                    0 /*pseStride*/, 0.0f /*slopes*/, 0.0f /*posShift*/, static_cast<T>(constInfoXVec.scaleValue),
                    descaleQK, negativeFloatScalar, 0.0F, queryScaleUb, deSCaleKValue);
            } else if (runInfo.actSingleLoopS2Size < 128) {
                fa_static::ProcessVec1Vf<T, INPUT_T, INPUT_T /*pseShiftType*/, false, mBaseSize, s2BaseSize, GT_64_AND_LTE_128,
                              hasAtten, PseTypeEnum::PSE_NONE_TYPE, hasDrop, false, false>(
                    stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, mmRes, expUb, sumUb, maxUb, attenMaskUb,
                    nonePseUb, dropMaskUb, apiTmpBuffer, pScaleUb, runInfo.actVecMSize, runInfo.actSingleLoopS2Size,
                    0 /*pseStride*/, 0.0f /*slopes*/, 0.0f /*posShift*/, static_cast<T>(constInfoXVec.scaleValue),
                    descaleQK, negativeFloatScalar, 0.0F, queryScaleUb, deSCaleKValue);
            } else {
                if constexpr (s2BaseSize == 256) {
                    fa_static::ProcessVec1Vf<T, INPUT_T, INPUT_T /*pseShiftType*/, false, mBaseSize, s2BaseSize,
                                  GT_128_AND_LTE_256, hasAtten, PseTypeEnum::PSE_NONE_TYPE, hasDrop>(
                        stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, mmRes, expUb, sumUb, maxUb, attenMaskUb,
                        nonePseUb, dropMaskUb, apiTmpBuffer, expUb, runInfo.actVecMSize, runInfo.actSingleLoopS2Size,
                        0 /*pseStride*/, 0.0f /*slopes*/, 0.0f /*posShift*/, static_cast<T>(constInfoXVec.scaleValue),
                        descaleQK, negativeFloatScalar, 0.0F);
                }
            }
        } else {
            if (likely(runInfo.actSingleLoopS2Size == 128)) {
                fa_static::ProcessVec1Vf<T, INPUT_T, INPUT_T /*pseShiftType*/, true, mBaseSize, s2BaseSize, EQ_128, hasAtten,
                              PseTypeEnum::PSE_NONE_TYPE, hasDrop, false, false>(
                    stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, mmRes, expUb, sumUb, maxUb, attenMaskUb,
                    nonePseUb, dropMaskUb, apiTmpBuffer, pScaleUb, runInfo.actVecMSize, runInfo.actSingleLoopS2Size,
                    0 /*pseStride*/, 0.0f /*slopes*/, 0.0f /*posShift*/, static_cast<T>(constInfoXVec.scaleValue),
                    descaleQK, negativeFloatScalar, 0.0F, queryScaleUb, deSCaleKValue);
            } else if (runInfo.actSingleLoopS2Size <= 64) {
                fa_static::ProcessVec1Vf<T, INPUT_T, INPUT_T /*pseShiftType*/, true, mBaseSize, s2BaseSize, GT_0_AND_LTE_64,
                              hasAtten, PseTypeEnum::PSE_NONE_TYPE, hasDrop, false, false>(
                    stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, mmRes, expUb, sumUb, maxUb, attenMaskUb,
                    nonePseUb, dropMaskUb, apiTmpBuffer, pScaleUb, runInfo.actVecMSize, runInfo.actSingleLoopS2Size,
                    0 /*pseStride*/, 0.0f /*slopes*/, 0.0f /*posShift*/, static_cast<T>(constInfoXVec.scaleValue),
                    descaleQK, negativeFloatScalar, 0.0F, queryScaleUb, deSCaleKValue);
            } else if (runInfo.actSingleLoopS2Size < 128) {
                fa_static::ProcessVec1Vf<T, INPUT_T, INPUT_T /*pseShiftType*/, true, mBaseSize, s2BaseSize, GT_64_AND_LTE_128,
                              hasAtten, PseTypeEnum::PSE_NONE_TYPE, hasDrop, false, false>(
                    stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, mmRes, expUb, sumUb, maxUb, attenMaskUb,
                    nonePseUb, dropMaskUb, apiTmpBuffer, pScaleUb, runInfo.actVecMSize, runInfo.actSingleLoopS2Size,
                    0 /*pseStride*/, 0.0f /*slopes*/, 0.0f /*posShift*/, static_cast<T>(constInfoXVec.scaleValue),
                    descaleQK, negativeFloatScalar, 0.0F, queryScaleUb, deSCaleKValue);
            } else {
                if constexpr (s2BaseSize == 256) {
                    fa_static::ProcessVec1Vf<T, INPUT_T, INPUT_T /*pseShiftType*/, true, mBaseSize, s2BaseSize, GT_128_AND_LTE_256,
                                  hasAtten, PseTypeEnum::PSE_NONE_TYPE, hasDrop>(
                        stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, mmRes, expUb, sumUb, maxUb, attenMaskUb,
                        nonePseUb, dropMaskUb, apiTmpBuffer, expUb, runInfo.actVecMSize, runInfo.actSingleLoopS2Size,
                        0 /*pseStride*/, 0.0f /*slopes*/, 0.0f /*posShift*/, static_cast<T>(constInfoXVec.scaleValue),
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
        SetFlag<HardEvent::V_MTE3>(kVToMte3Id[stage1Offset]);
        WaitFlag<HardEvent::V_MTE3>(kVToMte3Id[stage1Offset]);
        LocalTensor<INPUT_T> mm2AL1Tensor;
        mm2AL1Tensor = l1PTensor;

        if (likely(runInfo.actVecMSize != 0)) {
            DataCopy(mm2AL1Tensor[constInfoXVec.subBlockIdx * (blockBytes / sizeof(INPUT_T)) *
                                  (runInfo.actMSize - runInfo.actVecMSize)],
                     stage1CastTensor,
                     {s2BaseSize / 16, (uint16_t)runInfo.actVecMSize, (uint16_t)(vec1Srcstride - runInfo.actVecMSize),
                      (uint16_t)(mBaseSize - runInfo.actVecMSize)});
        }

        SetFlag<HardEvent::MTE3_V>(kStage1MTE3toV[stage1Offset]);

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
                                           RunInfoXVec runInfo)
    {
        if (unlikely(runInfo.actVecMSize == 0)) {
            CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(bmm2CcId);
            return;
        }

        int64_t vec2CalcSize = runInfo.actVecMSize * dTemplateAlign64;

        LocalTensor<T> vec2ResUb = this->stage2OutBuf;
        LocalTensor<T> mmRes = bmm2UbTensor;
        WaitFlag<HardEvent::MTE3_V>(kMte3ToVId[0]);
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
        SetFlag<HardEvent::MTE3_V>(kMte3ToVId[0]);
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
        SetFlag<HardEvent::V_MTE3>(kVToMte3Id[0]);
        WaitFlag<HardEvent::V_MTE3>(kVToMte3Id[0]);

        Bmm2DataCopyOutTrans(runInfo, attenOut, mStartVec, mDealSize);
    }

    template <typename VEC2_RES_T>
    __aicore__ inline void ProcessVec2DSplit(GlobalTensor<T> &mmRes, const RunInfoXVec &runInfo)
    {
        // bmm2 result is on GM and global update data on UB
        uint32_t subMBase = 8192 / constInfoXVec.dBasicBlock;
        int64_t vec2LoopLimit = CeilDiv(runInfo.actVecMSize, subMBase);
        LocalTensor<T> vec2ResUb = this->stage2OutBuf;
        WaitFlag<HardEvent::MTE3_V>(kMte3ToVId[0]);
        LocalTensor<T> bmm2Ub = mm2InBuf;

        constexpr uint32_t mte2ToV    = FA_MTE2_TO_V_ID;
        constexpr uint32_t vToMte2    = FA_V_TO_MTE2_ID;
        constexpr uint32_t mte3ToMte2 = FA_MTE3_TO_MTE2_ID;

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
                    DataCopy(bmm2Ub, mmRes[bmm2SubBlockOffset + mm2ResInnerOffset], vec2CalcSize);
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
                    DataCopyPad(bmm2Ub, mmRes[bmm2SubBlockOffset + mm2ResInnerOffset], dataCopyParams,
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
                float deSCalePreVValue = 1.0f;
                LocalTensor<T> expUb = softmaxExpBuf[runInfo.loop % (PRELOAD_N + 1)][vec2ExpBufOffset];
                if (!runInfo.isLastS2Loop) {
                    FlashUpdateNew<T, INPUT_T, OUTPUT_T, 0xFF, false, false>(vec2ResUb, bmm2Ub, vec2ResUb, expUb, expUb,
                                                                             subMRealsize, constInfoXVec.dBasicBlock, 0.0F,
                                                                             deSCalePreVValue);
                } else {
                    int64_t vec2SumBufOffset = mIdx;
                    LocalTensor<float> sumUb =
                        this->softmaxSumBuf[runInfo.mloop % (PRELOAD_N + 1)][vec2SumBufOffset];
                    FlashUpdateLastNew<T, INPUT_T, OUTPUT_T, 0xFF, false, false>(
                        vec2ResUb, bmm2Ub, vec2ResUb, expUb, expUb, sumUb, subMRealsize, constInfoXVec.dBasicBlock, 0.0F,
                        deSCalePreVValue);
                }
            }

            if (unlikely(runInfo.isLastS2Loop)) {
                if (unlikely(runInfo.isFirstS2Loop)) {
                    int64_t vec2SumBufOffset = mIdx;
                    LocalTensor<float> sumUb =
                        this->softmaxSumBuf[runInfo.mloop % (PRELOAD_N + 1)][vec2SumBufOffset];
                    LastDivNew<T, INPUT_T, OUTPUT_T, 0xFF, false>(vec2ResUb, vec2ResUb, sumUb, subMRealsize,
                                                                  constInfoXVec.dBasicBlock, 0.0F);
                }
                CopyOutAttentionOut(runInfo, vec2ResUb, mIdx, subMRealsize);
            } else if (vec2LoopLimit > 1) {
                SetFlag<HardEvent::V_MTE3>(kVToMte3Id[0]);
                WaitFlag<HardEvent::V_MTE3>(kVToMte3Id[0]);
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
        SetFlag<HardEvent::MTE3_V>(kMte3ToVId[0]);
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
                    softmaxMaxBuf[runInfo.mloop % (PRELOAD_N + 1)][mStartVec];
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
                                              constInfoXVec.dSizeV, qSeqLensParser);
            CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, GetOutUbFormat<layout>()> copyAttenOutUbToGm;
            copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
        } else if constexpr (outLayout == LayOutTypeEnum::LAYOUT_BNSD) {
            constexpr GmFormat OUT_FORMAT = GmFormat::BNGSD;
            FaGmTensor<OUTPUT_T, OUT_FORMAT, int32_t> outGmTensor;
            outGmTensor.gmTensor = attentionOutGm;
            outGmTensor.offsetCalculator.Init(constInfoXVec.bSize, constInfoXVec.n2Size, constInfoXVec.gSize, constInfoXVec.s1Size,
                                              constInfoXVec.dSizeV, qSeqLensParser);
            CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, GetOutUbFormat<layout>()> copyAttenOutUbToGm;
            copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
        } else if constexpr (outLayout == LayOutTypeEnum::LAYOUT_TND) {
            constexpr GmFormat OUT_FORMAT = GmFormat::TNGD;
            FaGmTensor<OUTPUT_T, OUT_FORMAT, int32_t, true> outGmTensor;
            outGmTensor.gmTensor = attentionOutGm;
            outGmTensor.offsetCalculator.Init(constInfoXVec.n2Size, constInfoXVec.gSize, constInfoXVec.dSizeV,
                                              qSeqLensParser);
            CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, GetOutUbFormat<layout>()> copyAttenOutUbToGm;
            copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
        }
    }

    __aicore__ inline void BroadCastAndCopyOut(const RunInfoXVec &runInfo, LocalTensor<float> &sumUb,
                                               LocalTensor<float> &maxUb, int64_t gmOffset, int64_t calculateSize)
    {
        // Copy sum to gm
        FaVectorApi::BroadcastMaxSum(this->sumBrdcstBuf, sumUb, runInfo.actVecMSize);
        SetFlag<HardEvent::V_MTE3>(kVToMte3Id[0]);
        WaitFlag<HardEvent::V_MTE3>(kVToMte3Id[0]);
        DataCopy(softmaxFDSumGm[gmOffset], this->sumBrdcstBuf, calculateSize);
        SetFlag<HardEvent::MTE3_V>(kMte3ToVId[1]);
        WaitFlag<HardEvent::MTE3_V>(kMte3ToVId[1]);

        // Copy max to gm
        FaVectorApi::BroadcastMaxSum(this->maxBrdcstBuf, maxUb, runInfo.actVecMSize);
        SetFlag<HardEvent::V_MTE3>(kVToMte3Id[0]);
        WaitFlag<HardEvent::V_MTE3>(kVToMte3Id[0]);
        DataCopy(softmaxFDMaxGm[gmOffset], this->maxBrdcstBuf, calculateSize);
        SetFlag<HardEvent::MTE3_V>(kMte3ToVId[1]);
        WaitFlag<HardEvent::MTE3_V>(kMte3ToVId[1]);
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
        SetFlag<HardEvent::V_MTE3>(kVToMte3Id[runInfo.loop & (DB - 1)]);
        WaitFlag<HardEvent::V_MTE3>(kVToMte3Id[runInfo.loop & (DB - 1)]);
        uint64_t gmOffset =
            runInfo.faTmpOutWsPos * mBaseSize * constInfoXVec.dSizeV + (runInfo.vecMbaseIdx + mStartVec) * constInfoXVec.dSizeV;

        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = mDealSize;
        dataCopyParams.blockLen = constInfoXVec.dSizeV * sizeof(T);
        dataCopyParams.srcStride = (dSizeAligned64 - constInfoXVec.dSizeV) / (FA_BYTE_BLOCK / sizeof(T));
        dataCopyParams.dstStride = 0;

        DataCopyPad(accumOutGm[gmOffset], vec2ResUb, dataCopyParams);
    }

    template<typename BmmResT>
    __aicore__ inline void ProcessVec2(BmmResT &bmm2ResTensor, uint32_t bmm2CcId, RunInfoXVec runInfo)
    {
#ifdef SKIP_V2
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(bmm2CcId);
        if constexpr (bmm2Write2Ub) {
            CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(bmm2CcId);
        }
        return;
#endif

        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(bmm2CcId);

        if constexpr (bmm2Write2Ub) {
            ProcessVec2OnUb(bmm2ResTensor, bmm2CcId, runInfo);
        } else if constexpr (splitD) {
            ProcessVec2DSplit(bmm2ResTensor, runInfo);
        } else {
            ProcessVec2_C2GmUpdateUb(bmm2ResTensor, runInfo);
        }
        return;
    }

    __aicore__ inline void ProcessVec2_C2GmUpdateUb(GlobalTensor<T> &mmRes, RunInfoXVec runInfo)
    {
        // bmm2 result is on GM and global update data on UB
        uint32_t subMBase = 8192 / dTemplateAlign64;
        int64_t vec2LoopLimit = CeilDiv(runInfo.actVecMSize, subMBase);
        LocalTensor<T> vec2ResUb = this->stage2OutBuf;
        WaitFlag<HardEvent::MTE3_V>(kMte3ToVId[0]);
        LocalTensor<T> bmm2Ub = mm2InBuf;

        constexpr uint32_t mte2ToV = FA_MTE2_TO_V_ID;
        constexpr uint32_t vToMte2 = FA_V_TO_MTE2_ID;

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
                    DataCopy(bmm2Ub, mmRes[bmm2SubBlockOffset + mm2ResInnerOffset], vec2CalcSize);
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
                    DataCopyPad(bmm2Ub, mmRes[bmm2SubBlockOffset + mm2ResInnerOffset], dataCopyParams,
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
                float deSCalePreVValue = 1.0f;
                LocalTensor<T> expUb = softmaxExpBuf[runInfo.loop % (PRELOAD_N + 1)][vec2ExpBufOffset];
                if (!runInfo.isLastS2Loop) {
                    FlashUpdateNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, false, false>(
                        vec2ResInner, bmm2Ub, vec2ResInner, expUb, expUb, subMRealsize, dTemplateAlign64, 0.0F,
                        deSCalePreVValue);
                } else {
                    int64_t vec2SumBufOffset = mIdx;
                    LocalTensor<float> sumUb =
                        softmaxSumBuf[runInfo.mloop % (PRELOAD_N + 1)][vec2SumBufOffset];
                    FlashUpdateLastNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, false, false>(
                        vec2ResInner, bmm2Ub, vec2ResInner, expUb, expUb, sumUb, subMRealsize, dTemplateAlign64, 0.0F,
                        deSCalePreVValue);
                }
            }

            if (unlikely(runInfo.isLastS2Loop)) {
                if (unlikely(runInfo.isFirstS2Loop)) {
                    int64_t vec2SumBufOffset = mIdx;
                    LocalTensor<float> sumUb =
                        softmaxSumBuf[runInfo.mloop % (PRELOAD_N + 1)][vec2SumBufOffset];
                    LastDivNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, false>(vec2ResInner, vec2ResInner, sumUb,
                                                                              subMRealsize, dTemplateAlign64, 0.0F);
                }
                CopyOutAttentionOut(runInfo, vec2ResInner, mIdx, subMRealsize);
            }
        }
        SetFlag<HardEvent::MTE3_V>(kMte3ToVId[0]);
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
        maxBrdcstBuf     = LocalTensor<uint8_t>(TPosition::VECIN, off_sm + 2304U,  2048U).template ReinterpretCast<float>();
        sumBrdcstBuf     = LocalTensor<uint8_t>(TPosition::VECIN, off_sm + 4352U,  2048U).template ReinterpretCast<float>();
    }

    __aicore__ inline void InitBuffers()
    {
        // 完全自主管理：使用 LocalTensor 构造函数直接指定绝对字节偏移，
        // 无需 LocalMemAllocator 线性分配器。所有偏移为编译期常量。
        constexpr uint32_t mm1Sz = mBaseSize / CV_RATIO * s2BaseSize * sizeof(T);
        constexpr uint32_t mm2Sz = bmm2Write2Ub ?
            (mBaseSize / CV_RATIO * (uint32_t)dVTemplateType * sizeof(T)) : 0U;

        // mm2InBuf：仅 !bmm2Write2Ub 时存在，固定偏移在 bmm1/2 结果区末尾
        if constexpr (!bmm2Write2Ub) {
            mm2InBuf = LocalTensor<uint8_t>(TPosition::VECIN, mm1Sz * 2U + mm2Sz * 2U,
                           32768U).template ReinterpretCast<T>();
        }

        // FA/FD 共享区起始偏移（跳过 bmm1/bmm2/mm2In 区域）
        constexpr uint32_t mm2InBufSz = bmm2Write2Ub ? 0U : 32768U;
        constexpr uint32_t BASE = mm1Sz * 2U + mm2Sz * 2U + mm2InBufSz;

        // attenMask ping/pong（各 8192 bytes）
        attenMaskBuf[0] = LocalTensor<uint8_t>(TPosition::VECIN, BASE,          8192U);
        attenMaskBuf[1] = LocalTensor<uint8_t>(TPosition::VECIN, BASE + 8192U,  8192U);
        // stage2 输出（32768 bytes）
        stage2OutBuf    = LocalTensor<uint8_t>(TPosition::VECIN, BASE + 16384U, 32768U).template ReinterpretCast<T>();
        // stage1 输出 ping/pong（各 33024 bytes）
        stage1OutBuf[0] = LocalTensor<uint8_t>(TPosition::VECIN, BASE + 49152U, 33024U).template ReinterpretCast<INPUT_T>();
        stage1OutBuf[1] = LocalTensor<uint8_t>(TPosition::VECIN, BASE + 82176U, 33024U).template ReinterpretCast<INPUT_T>();

        // Softmax 中间值：6400 bytes，紧跟 stage1[1] 末尾（BASE + 115200）
        SoftmaxInitBuffer(BASE + 115200U);

        // commonBuf：compile-time 条件，512 bytes（BASE + 121600）
        constexpr bool hasCommonBuf = (s2BaseSize == 256U) ? (mBaseSize != 128U) : (!useDn);
        if constexpr (hasCommonBuf) {
            commonBuf = LocalTensor<uint8_t>(TPosition::VECIN, BASE + 121600U, 512U);
        }

        // lseBuf / sinkBuf：runtime 条件，地址已在编译期固定（始终位于 commonBuf 区末尾之后）
        constexpr uint32_t commonBufSz = hasCommonBuf ? 512U : 0U;
        constexpr uint32_t OFF_LSE = BASE + 121600U + commonBufSz;
        constexpr uint32_t LSE_BUF_SZ = (mBaseSize >> 1U) * 8U * sizeof(float);
        if (constInfoXVec.isSoftmaxLseEnable) {
            lseBuf = LocalTensor<uint8_t>(TPosition::VECIN, OFF_LSE,
                         LSE_BUF_SZ).template ReinterpretCast<float>();
        }
        if (constInfoXVec.learnableSinkFlag) {
            sinkBuf = LocalTensor<uint8_t>(TPosition::VECIN, OFF_LSE + LSE_BUF_SZ,
                          256U).template ReinterpretCast<INPUT_T>();
        }
    }

    __aicore__ inline void AllocEventID()
    {
        SetFlag<HardEvent::MTE3_V>(kMte3ToVId[0]);
        SetFlag<HardEvent::MTE3_V>(kMte3ToVId[1]);
        SetFlag<HardEvent::MTE3_V>(kStage1MTE3toV[0]);
        SetFlag<HardEvent::MTE3_V>(kStage1MTE3toV[1]);
    }

    __aicore__ inline void FreeEventID()
    {
        WaitFlag<AscendC::HardEvent::MTE3_V>(kMte3ToVId[0]);
        WaitFlag<AscendC::HardEvent::MTE3_V>(kMte3ToVId[1]);
        WaitFlag<AscendC::HardEvent::MTE3_V>(kStage1MTE3toV[0]);
        WaitFlag<AscendC::HardEvent::MTE3_V>(kStage1MTE3toV[1]);
    }

    __aicore__ inline void AttenMaskCopyIn(LocalTensor<uint8_t> attenMaskUb, uint32_t vecMIdx, uint32_t mDealSize,
                                           RunInfoXVec &runInfo)
    {
        const uint32_t bufIdx = runInfo.loop & (DB - 1);
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
        maskInfo.maskValue = NEGATIVE_MIN_VALUE_FP32;
        maskInfo.s1LeftPaddingSize = 0;
        maskInfo.s2LeftPaddingSize = 0;
        maskInfo.layout = MASK_LAYOUT;
        maskInfo.attenMaskType = MASK_BOOL; // compatible with int8/uint8

        bool IsSkipMask = IsSkipAttentionmask(maskInfo);
        bool IsSkipMaskForPre = IsSkipAttentionmaskForPre(maskInfo);

        const uint32_t mte2ToVId =
            (bufIdx == 0U) ? FA_ATTEN_MASK_MTE2_V_0 : FA_ATTEN_MASK_MTE2_V_1;
        const uint32_t preMte2ToVId =
            (bufIdx == 0U) ? FA_ATTEN_MASK_MTE2_V_1 : FA_ATTEN_MASK_MTE2_V_0;

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
        uint32_t n2Idx = bN2Cur % constInfoXVec.n2Size;
        uint32_t bIdx = bN2Cur / constInfoXVec.n2Size;
        if constexpr (outLayout == LayOutTypeEnum::LAYOUT_BSH) {
            OffsetCalculator<GmFormat::BSNGD, int32_t> offsetCalculator;
            offsetCalculator.Init(constInfoXVec.bSize, constInfoXVec.n2Size, constInfoXVec.gSize, constInfoXVec.s1Size,
                                  constInfoXVec.dSizeV, qSeqLensParser);
            DealActSeqLenIsZero<GmFormat::BSNGD, OUTPUT_T>(bIdx, n2Idx, offsetCalculator, attentionOutGm);
        } else if constexpr (outLayout == LayOutTypeEnum::LAYOUT_BNSD) {
            OffsetCalculator<GmFormat::BNGSD, int32_t> offsetCalculator;
            offsetCalculator.Init(constInfoXVec.bSize, constInfoXVec.n2Size, constInfoXVec.gSize, constInfoXVec.s1Size,
                                  constInfoXVec.dSizeV, qSeqLensParser);
            DealActSeqLenIsZero<GmFormat::BNGSD, OUTPUT_T>(bIdx, n2Idx, offsetCalculator, attentionOutGm);
        } else if constexpr (outLayout == LayOutTypeEnum::LAYOUT_TND) {
            OffsetCalculator<GmFormat::TNGD, int32_t, true> offsetCalculator;
            offsetCalculator.Init(constInfoXVec.n2Size, constInfoXVec.gSize, constInfoXVec.dSizeV, qSeqLensParser);
            DealActSeqLenIsZero<GmFormat::TNGD, OUTPUT_T>(bIdx, n2Idx, offsetCalculator, attentionOutGm);
        }
    }

    // ===== 核间共享 buffer slots =====
    LocalTensor<uint8_t> bmm1UbSlot[2];
    LocalTensor<uint8_t> bmm2UbSlot[2];
    LocalTensor<uint8_t> l1PSlot[3];
    GlobalTensor<T>      bmm2GmSlot[3];

    // ===== 静态 cross-core 事件 ID =====
    static constexpr uint32_t CC_BMM1_0 = 0U;
    static constexpr uint32_t CC_BMM1_1 = 1U;
    static constexpr uint32_t CC_BMM2_0 = 2U;
    static constexpr uint32_t CC_BMM2_1 = 3U;
    static constexpr uint32_t CC_BMM2_2 = 4U;
    static constexpr uint32_t CC_L1P_0  = 5U;
    static constexpr uint32_t CC_L1P_1  = 8U;
    static constexpr uint32_t CC_L1P_2  = 9U;

    GlobalTensor<int32_t> cuSeqLensGmQ;
    GlobalTensor<int32_t> cuSeqLensGmKv;
    GlobalTensor<int32_t> seqUsedGmQ;
    GlobalTensor<int32_t> seqUsedGmKv;
    GlobalTensor<uint32_t> faMetaDataGm;
    GlobalTensor<uint32_t> fdMetaDataGm;
    GlobalTensor<float> softmaxLseGm;
    GlobalTensor<INPUT_T> sinkGm;
    __gm__ uint8_t *keyPtr = nullptr;

    ConstInfoXVec constInfoXVec;

    const __gm__ FlashAttnNoQuantTilingArch35 *__restrict tilingData;
    VecFdBlockType vecFdBlock;

    uint32_t accGS1Loops = 0U;
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



    // ==============================Functions============================================
    __aicore__ inline FlashAttentionNoQuantGqaKernel()
        : vecFdBlock(constInfoXVec){};
    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                __gm__ uint8_t *attenMask, __gm__ uint8_t *cuSeqLensQ, __gm__ uint8_t *cuSeqLensKv,
                                __gm__ uint8_t *blockTable, __gm__ uint8_t *learnableSink, __gm__ uint8_t *softmaxLse,
                                __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace, __gm__ uint8_t *fiaMetaData,
                                __gm__ uint8_t *seqUsedQ, __gm__ uint8_t *seqUsedKv,
                                const __gm__ FlashAttnNoQuantTilingArch35 *__restrict tiling)
    {
        this->tilingData = tiling;

        InitConstInfo();

        keyPtr = key;

        if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_TND) {
            cuSeqLensGmQ.SetGlobalBuffer((__gm__ int32_t *)cuSeqLensQ, constInfoXVec.cuSeqLensQSize + 1);
            seqUsedGmQ.SetGlobalBuffer((__gm__ int32_t *)seqUsedQ, constInfoXVec.seqUsedQSize);
            cuSeqLensGmKv.SetGlobalBuffer((__gm__ int32_t *)cuSeqLensKv, constInfoXVec.cuSeqLensKVSize + 1);
            seqUsedGmKv.SetGlobalBuffer((__gm__ int32_t *)seqUsedKv, constInfoXVec.seqUsedKvSize);
        } else {
            seqUsedGmQ.SetGlobalBuffer((__gm__ int32_t *)seqUsedQ, constInfoXVec.seqUsedQSize);
            seqUsedGmKv.SetGlobalBuffer((__gm__ int32_t *)seqUsedKv, constInfoXVec.seqUsedKvSize);
        }
        sectionNum_ = ((__gm__ uint32_t *)fiaMetaData)[0];
        isFd = static_cast<bool>(((__gm__ uint32_t *)fiaMetaData)[1]);

        faMetaDataGm.SetGlobalBuffer((__gm__ uint32_t *)(fiaMetaData + FA_METADATA_HEADER_OFFSET),
                                     FA_AIC_CORE_NUM * 16U * sectionNum_);
        
        fdMetaDataGm.SetGlobalBuffer(
            (__gm__ uint32_t *)(fiaMetaData + FA_METADATA_HEADER_OFFSET +
                                FLASH_ATTN_METADATA_SIZE * FA_AIC_CORE_NUM * sectionNum_ * sizeof(uint32_t)),
            FA_AIV_CORE_NUM * 16U * sectionNum_);

        InitQSeqLensParser();
        InitKvSeqLensParser();

        InitMMResBuf(workspace);

        if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_TND) {
            InitVecBlock(cuSeqLensQ, cuSeqLensKv, attenMask, learnableSink, softmaxLse,
                                    attentionOut, workspace);
        } else {
            InitVecBlock(seqUsedQ, seqUsedKv, attenMask, learnableSink, softmaxLse, attentionOut,
                                    workspace);
        }
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
                softmaxLseGm.SetGlobalBuffer((__gm__ float *)softmaxLse);
                vecFdBlock.InitSoftmaxLseGm(softmaxLseGm);
            }
            vecFdBlock.SetCuSeqLensParsers(qSeqLensParser, kvSeqLensParser);
        }
    }

    __aicore__ inline void InitMMResBuf(__gm__ uint8_t *&workspace)
    {
        constexpr uint32_t mm1SlotSize = mBaseSize / CV_RATIO * s2BaseSize * sizeof(T);
        constexpr uint32_t mm2SlotSize = BMM2_TOUB ?
            (mBaseSize / CV_RATIO * dVBaseSize * sizeof(T)) : 0U;
        constexpr uint32_t l1PSlotSize = mBaseSize * s2BaseSize * sizeof(INPUT_T);
        constexpr uint32_t bmm1UbBase  = 2U * mm2SlotSize;

        if constexpr (BMM2_TOUB) {
            bmm2UbSlot[0] = LocalTensor<uint8_t>(TPosition::VECIN, 0U,          mm2SlotSize);
            bmm2UbSlot[1] = LocalTensor<uint8_t>(TPosition::VECIN, mm2SlotSize, mm2SlotSize);
        }
        bmm1UbSlot[0] = LocalTensor<uint8_t>(TPosition::VECIN, bmm1UbBase,               mm1SlotSize);
        bmm1UbSlot[1] = LocalTensor<uint8_t>(TPosition::VECIN, bmm1UbBase + mm1SlotSize, mm1SlotSize);

        l1PSlot[0] = LocalTensor<uint8_t>(TPosition::A1, 0U,               l1PSlotSize);
        l1PSlot[1] = LocalTensor<uint8_t>(TPosition::A1, l1PSlotSize,      l1PSlotSize);
        l1PSlot[2] = LocalTensor<uint8_t>(TPosition::A1, 2U * l1PSlotSize, l1PSlotSize);

        if constexpr (!BMM2_TOUB) {
            int64_t mm2GmSlotElems = mBaseSize * constInfoXVec.dBasicBlock;
            int64_t prevCoreOffset = constInfoXVec.aicIdx * 3 * mm2GmSlotElems;
            bmm2GmSlot[0].SetGlobalBuffer((__gm__ T *)workspace + prevCoreOffset);
            bmm2GmSlot[1].SetGlobalBuffer((__gm__ T *)workspace + prevCoreOffset + mm2GmSlotElems);
            bmm2GmSlot[2].SetGlobalBuffer((__gm__ T *)workspace + prevCoreOffset + 2 * mm2GmSlotElems);
            workspace = workspace + constInfoXVec.coreNum * 3 * mm2GmSlotElems * sizeof(T);
        }
    }

    __aicore__ inline void InitQSeqLensParser()
    {
        if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_TND) {
            qSeqLensParser.Init(cuSeqLensGmQ, seqUsedGmQ, constInfoXVec.cuSeqLensQSize, constInfoXVec.seqUsedQSize);
        } else {
            qSeqLensParser.Init(seqUsedGmQ, constInfoXVec.seqUsedQSize, constInfoXVec.s1Size);
        }
    }

    __aicore__ inline void InitKvSeqLensParser()
    {
        if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_TND) {
            kvSeqLensParser.Init(cuSeqLensGmKv, seqUsedGmKv, constInfoXVec.cuSeqLensKVSize, constInfoXVec.seqUsedKvSize);
        } else {
            kvSeqLensParser.Init(seqUsedGmKv, constInfoXVec.seqUsedKvSize, constInfoXVec.s2Size);
        }
    }

     __aicore__ inline void InitConstInfo()
    {
        constInfoXVec.aivIdx = GetBlockIdx();
        constInfoXVec.aicIdx = GetBlockIdx() / GetSubBlockNum();
        constInfoXVec.subBlockIdx = GetSubBlockIdx();
        constInfoXVec.bSize = this->tilingData->flashAttnBaseParams.bSize;
        constInfoXVec.t1Size = this->tilingData->flashAttnBaseParams.t1Size;
        constInfoXVec.n2Size = this->tilingData->flashAttnBaseParams.n2Size;
        constInfoXVec.gSize = this->tilingData->flashAttnBaseParams.gSize;
        constInfoXVec.s1Size = this->tilingData->flashAttnBaseParams.s1Size;
        constInfoXVec.s2Size = this->tilingData->flashAttnBaseParams.s2Size;
        constInfoXVec.dSizeV = this->tilingData->flashAttnBaseParams.dSizeV;
        constInfoXVec.cuSeqLensQSize = this->tilingData->flashAttnBaseParams.cuSeqLensQSize;
        constInfoXVec.cuSeqLensKVSize = this->tilingData->flashAttnBaseParams.cuSeqLensKVSize;
        constInfoXVec.seqUsedQSize = this->tilingData->flashAttnBaseParams.seqUsedQSize;
        constInfoXVec.seqUsedKvSize = this->tilingData->flashAttnBaseParams.seqUsedKvSize;
        constInfoXVec.scaleValue = static_cast<float>(this->tilingData->flashAttnBaseParams.scaleValue);
        constInfoXVec.isKvContinuous = this->tilingData->flashAttnBaseParams.isKvContinuous != 0;
        constInfoXVec.coreNum = this->tilingData->flashAttnBaseParams.coreNum;
        constInfoXVec.needInitOutput = this->tilingData->flashAttnBaseParams.needInitOutput;
        
        constInfoXVec.sparseMode = this->tilingData->flashAttnAttenMaskParams.sparseMode;
        constInfoXVec.preTokens = this->tilingData->flashAttnAttenMaskParams.winLefts;
        constInfoXVec.nextTokens = this->tilingData->flashAttnAttenMaskParams.winRights;
        constInfoXVec.attenMaskBatch = this->tilingData->flashAttnAttenMaskParams.attenMaskBatch;
        constInfoXVec.attenMaskS1Size = this->tilingData->flashAttnAttenMaskParams.attenMaskS1Size;
        constInfoXVec.attenMaskS2Size = this->tilingData->flashAttnAttenMaskParams.attenMaskS2Size;

        constInfoXVec.accumOutSize = this->tilingData->flashAttnWorkspaceParams.accumOutSize;
        constInfoXVec.logSumExpSize = this->tilingData->flashAttnWorkspaceParams.logSumExpSize;

        constInfoXVec.isSoftmaxLseEnable = this->tilingData->flashAttnBaseParams.isSoftMaxLseEnable;

        constInfoXVec.dBasicBlock = Align64Func((uint16_t)constInfoXVec.dSizeV);
    }

    __aicore__ inline uint32_t GetFAMetaDataIndex(uint32_t coreIdx, uint32_t metaIdx, uint32_t sectionIdx)
    {
        return FLASH_ATTN_METADATA_SIZE * FA_AIC_CORE_NUM * sectionIdx + 16U * coreIdx + metaIdx;
    }
    __aicore__ inline uint32_t GetFDMetaDataIndex(uint32_t coreIdx, uint32_t metaIdx, uint32_t sectionIdx)
    {
        return FA_FD_METADATA_SIZE * FA_AIV_CORE_NUM * sectionIdx + FA_FD_METADATA_SIZE * coreIdx + metaIdx;
    }

    __aicore__ inline void CrossCoreBufferInit()
    {
        if constexpr (BMM2_TOUB) {
            CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CC_BMM2_0);
            CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CC_BMM2_1);
        }
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CC_BMM1_0);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CC_BMM1_1);
    }

    __aicore__ inline void FlashAttention(uint32_t sectionIdx)
    {
        if (constInfoXVec.aicIdx >= constInfoXVec.coreNum) {
            return;
        }
        CommonTaskLoop(*this, sectionIdx);
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
                actSeqLensKv = SeqLenFromTensorList<LAYOUT>(keyPtr, bIdx);
            } else {
                actSeqLensKv = kvSeqLensParser.GetActualSeqLength(bIdx);
            }
            actSeqLensQ = qSeqLensParser.GetActualSeqLength(bIdx);
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
            nextTokenLeftUp = static_cast<int64_t>(actSeqLensKv) - static_cast<int64_t>(actSeqLensQ) + nextTokenLeftUp;
        } else if (constInfoXVec.sparseMode == fa_base_vector::RIGHT_DOWN_CAUSAL) {
            nextTokenLeftUp = static_cast<int64_t>(actSeqLensKv) - static_cast<int64_t>(actSeqLensQ);
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
        int64_t preTokenLeftUp = 0;
        int64_t nextTokenLeftUp = 0;
        GetPreNextTokenLeftUp(actSeqLensQ, actSeqLensKv, preTokenLeftUp, nextTokenLeftUp);

        int64_t s1GFirstToken = static_cast<int64_t>(gS1Cur) * static_cast<int64_t>(mBaseSize);
        int64_t s1GLastToken = AttentionCommon::Min(s1GFirstToken + static_cast<int64_t>(mBaseSize),
                                   static_cast<int64_t>(actSeqLensQ) * static_cast<int64_t>(constInfoXVec.gSize)) -
                               1;
        int64_t s1FirstToken = 0;
        int64_t s1LastToken = 0;
        if constexpr (GetOutUbFormat<LAYOUT>() == UbFormat::S1G) {
            s1FirstToken = static_cast<int64_t>(s1GFirstToken / constInfoXVec.gSize);
            s1LastToken = static_cast<int64_t>(s1GLastToken / constInfoXVec.gSize);
        } else {
            if (s1GFirstToken / static_cast<int64_t>(actSeqLensQ) == s1GLastToken / static_cast<int64_t>(actSeqLensQ)) {
                s1FirstToken = s1GFirstToken % static_cast<int64_t>(actSeqLensQ);
                s1LastToken = s1GLastToken % static_cast<int64_t>(actSeqLensQ);
            } else {
                s1FirstToken = 0;
                s1LastToken = static_cast<int64_t>(actSeqLensQ);
            }
        }

        uint32_t s2StartWithSparse = 0U;
        uint32_t s2EndWithSparse = 0U;
        int64_t s2FirstToken = s1FirstToken - preTokenLeftUp;
        int64_t s2LastToken = s1LastToken + nextTokenLeftUp;
        if (s2FirstToken >= static_cast<int64_t>(actSeqLensKv) || s2LastToken < 0 || s2LastToken < s2FirstToken) {
            curS2Start = 0U;
            curS2End = 0U;
            return;
        }
        s2FirstToken = ClipSInnerToken(s2FirstToken, 0, static_cast<int64_t>(actSeqLensKv - 1));
        s2LastToken = ClipSInnerToken(s2LastToken, 0, static_cast<int64_t>(actSeqLensKv - 1));

        s2StartWithSparse = static_cast<uint32_t>(s2FirstToken) / s2BaseSize;
        s2EndWithSparse = static_cast<uint32_t>(s2LastToken) / s2BaseSize + 1U;

        curS2Start = s2StartWithSparse;
        curS2End = s2EndWithSparse;

        if (bN2Cur == bN2Start_ && gS1Cur == gS1OStart_) {
            headS2Split = s2OStart_ > s2StartWithSparse ? true : false;
            curS2Start = AttentionCommon::Max(s2StartWithSparse, s2OStart_);
        }
        if (bN2Cur == bN2End_ && gS1Cur == gS1OEnd_) {
            tailS2Split = s2OEnd_ > 0U ? true : false;
            curS2End = s2OEnd_ > 0U ? AttentionCommon::Min(s2EndWithSparse, s2OEnd_) : s2EndWithSparse;
        }
        return;
    }

    __aicore__ inline void ExecuteTask(uint64_t loop, RunInfoXVec taskRunInfo[PRELOAD_TASK_CACHE_SIZE])
    {
        RunInfoXVec &runInfo0 = taskRunInfo[loop & PRELOAD_TASK_CACHE_MASK];
        RunInfoXVec &runInfoNegN = taskRunInfo[(loop - PRELOAD_N) & PRELOAD_TASK_CACHE_MASK];
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
        uint32_t bmm1Slot = runInfo.loop % 2U;
        uint32_t l1PIdx   = runInfo.loop % 3U;
        uint32_t l1PCcId  = (l1PIdx == 0U) ? CC_L1P_0 : (l1PIdx == 1U) ? CC_L1P_1 : CC_L1P_2;
        auto l1PTensor    = l1PSlot[l1PIdx].template ReinterpretCast<INPUT_T>();
        auto bmm1Tensor   = bmm1UbSlot[bmm1Slot].template ReinterpretCast<T>();
        ProcessVec1(l1PTensor, l1PCcId, bmm1Tensor, CC_BMM1_0 + bmm1Slot, runInfo);
    }

    __aicore__ inline void ComputeVec2(RunInfoXVec &runInfo)
    {
        uint32_t outSlot = runInfo.loop % 2U;
        if constexpr (BMM2_TOUB) {
            auto bmm2Tensor = bmm2UbSlot[outSlot].template ReinterpretCast<T>();
            ProcessVec2(bmm2Tensor, CC_BMM2_0 + outSlot, runInfo);
        } else {
            ProcessVec2(bmm2GmSlot[runInfo.loop % 3U],
                                   CC_BMM2_0 + (runInfo.loop % 3U), runInfo);
        }
    }

    __aicore__ inline void CreateTask(uint64_t loop, uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur,
                                      RunInfoXVec taskRunInfo[PRELOAD_TASK_CACHE_SIZE])
    {
        RunInfoXVec &runInfo = taskRunInfo[loop & PRELOAD_TASK_CACHE_MASK];
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

        if (s2Cur + 1 < s2LoopTimes) {
            s2Cur++;
            return;
        }

        if (taskDealMode == TASK_DEAL_MODE::CREATE_TASK || taskDealMode == TASK_DEAL_MODE::S2_END ||
            taskDealMode == TASK_DEAL_MODE::SKIP_S1OUT) {
            accGS1Loops++;
        }

        s2Cur = 0;
        if (gS1Cur + 1 < gS1LoopTimes) {
            gS1Cur++;
            return;
        }

        if ((taskDealMode == TASK_DEAL_MODE::DEAL_ZERO || taskDealMode == TASK_DEAL_MODE::SKIP_ZERO)) {
            accGS1Loops++;
        }

        gS1Cur = 0;
        bN2Cur++;
    }

    __aicore__ inline void FlashDecode(uint32_t sectionIdx)
    {
        if (!isFd) {
            return;
        }
        GetFDSectionInfo(sectionIdx);
        vecFdBlock.InitBuffers();
        AscendC::ICachePreLoad(2);
        vecFdBlock.AllocEventID();
        vecFdBlock.InitDecodeParams();
        if (constInfoXVec.aicIdx < constInfoXVec.coreNum) {
            WaitFlag<AscendC::HardEvent::MTE3_V>(FA_MTE3_TO_V_0);
            SetFlag<AscendC::HardEvent::MTE3_V>(FA_MTE3_TO_V_0);
            AscendC::PipeBarrier<PIPE_MTE3>();
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::SyncAll();
        vecFdBlock.FlashDecode(fdParams_);
        AscendC::SyncAll();
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
            InitBuffers();
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
}; // FlashAttentionNoQuantGqaKernel

} // namespace BaseApi

#endif // FLASH_ATTN_GQA_VEC_H_
