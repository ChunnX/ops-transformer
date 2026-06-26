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
 * \file flash_attn_block_vec_noquant_gqa_nd.h
 * \brief FANoQuantGqaBlockVecNd —— Nd 路径专用 Vec Block 模板（独立类，无 base 基类）。
 */
#ifndef FLASH_ATTENTION_NOQUANT_GQA_BLOCK_VEC_ND_H_
#define FLASH_ATTENTION_NOQUANT_GQA_BLOCK_VEC_ND_H_

#include "kernel_operator.h"

#include "../utils/attenmask_gs1.h"

#include "../../../common/op_kernel/arch35/flash_attention_score_common_regbase.h"
#include "adv_api/activation/softmax.h"
#include "../../../common/op_kernel/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz.h"
#include "../../../common/op_kernel/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz_dn.h"
// 静态 Tensor 适配的 ProcessVec1Vf / ProcessVec1VfDn(Causal) 重载已直接合入上面两个 common vf 头文件，
// 由 LocalTensor<uint8_t>* 重载与原 TBuf<>* 版本并存；这里通过 using namespace FaVectorApi 调用即可。
#include "../../../common/op_kernel/arch35/vf/vf_flashupdate_new.h"
#include "../../../common/op_kernel/arch35/vf/vf_div_cast.h"
#include "../../../common/op_kernel/arch35/vf/vf_flash_decode.h"
#include "../../../common/op_kernel/vector_common.h"
#include "../../../common/op_kernel/memory_copy_arch35.h"
#include "fa_static_init_output.h"  // 静态Tensor版 InitOutput，替代 matmul::InitOutput（其内部用 TPipe 动态分配 eventId，与静态 eventId 冲突）

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
    uint8_t KvLayoutType = 0, bool bmm2Write2Ub = true, bool splitD = false>
class FANoQuantGqaBlockVecNd {
public:
    /* =================编译期常量的基本块信息================= */
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
    // 索引使用 loop & (DB - 1) 代替 loop % DB，要求 DB 必须是2的幂，否则位掩码结果错误
    static_assert(DB > 0 && (DB & (DB - 1)) == 0, "DB must be a power of two for bitmask indexing");
    static constexpr uint32_t PRELOAD_N = 2; // C1 C1 C2
    static constexpr bool HAS_MASK = hasAtten;

    // =============================================================================
    // 静态 event ID 常量 —— 全部按守护的 buff 命名（不再以通路 MTE3/V/MTE2 命名）
    //   命名规范：小驼峰 + EventId 后缀；同一 buff 多 slot 时以数字后缀区分（与原 `_0/_1` 一一对应）
    //   语义后缀：
    //     ReadyEventId   生产者→消费者正向同步（生产者写完 buff → 消费者可读 / 可搬出）
    //     FreeEventId    消费者→生产者反向同步（消费者用完 buff → 生产者可重新写入）
    //     RecycleEventId 同一 buff 上 MTE3→MTE2 的回写复用
    //
    //   物理 flag id 复用规则：同一 buff 的 forward/back-sync 可同号（位于不同硬件通道）；
    //     不同 buff 在不重叠时序窗口内也可复用同号。值与重命名前完全一致，逻辑不变。
    //     统一避开通道 6/7 与 SyncAll() 使用的 [11-14]。
    // =============================================================================

    // ----- initOutputScratch：ClearOutput / Init*SingleCore 临时 UB -----
    static constexpr uint32_t initOutputBufFreeEventId   = 0U; // MTE3_V

    // ----- stage1OutBuf[2]：vec1 输出，UB→L1 P 矩阵 ping/pong -----
    static constexpr uint32_t stage1OutBufReadyEventId0  = 0U; // V_MTE3, stage1OutBuf[0] 写完
    static constexpr uint32_t stage1OutBufReadyEventId1  = 1U; // V_MTE3, stage1OutBuf[1] 写完
    static constexpr uint32_t stage1OutBufFreeEventId0   = 3U; // MTE3_V, stage1OutBuf[0] 搬到 L1 完成
    static constexpr uint32_t stage1OutBufFreeEventId1   = 5U; // MTE3_V, stage1OutBuf[1] 搬到 L1 完成

    // ----- stage2OutBuf：vec2 主结果区，UB→GM 输出 -----
    static constexpr uint32_t stage2OutBufFreeEventId    = 0U; // MTE3_V, 上一轮 CopyOut 完成
    static constexpr uint32_t vec2OutBufReadyEventId     = 0U; // V_MTE3, vec2 cast/UB 写完 → MTE3 出 GM 局部栅栏

    // ----- lseBuf / maxBrdcstBuf / sumBrdcstBuf：softmax / FD 输出区 -----
    static constexpr uint32_t lseBufFreeEventId          = 1U; // MTE3_V, lse / brdcst 出 GM 完成
    static constexpr uint32_t lseBufReadyEventId         = 0U; // V_MTE3, lseBuf 写完 → 出 GM
    static constexpr uint32_t brdcstBufReadyEventId      = 0U; // V_MTE3, brdcstBufs 写完 → 出 GM

    // ----- FD ping/pong 输出 (Bmm2ResForFDCopyOut)：accumOutGm 写出 -----
    static constexpr uint32_t fdOutBufReadyEventId0      = 0U; // V_MTE3, ping slot 写完
    static constexpr uint32_t fdOutBufReadyEventId1      = 1U; // V_MTE3, pong slot 写完

    // ----- mm2InBuf：bmm2 结果 GM→UB（非 BMM2_TOUB / splitD 路径）-----
    static constexpr uint32_t mm2InBufReadyEventId       = 2U; // MTE2_V, GM→UB 完成
    static constexpr uint32_t mm2InBufFreeEventId        = 2U; // V_MTE2, V 读完, MTE2 可覆写

    // ----- vec2ResGm slot：splitD 子块 D 上的 UB→GM→UB 回写 -----
    static constexpr uint32_t vec2ResGmRecycleEventId    = 2U; // MTE3_MTE2, 出 GM → 下轮入 GM

    // ----- attenMaskBuf[2]：mask GM→UB ping/pong -----
    static constexpr uint32_t attenMaskBufReadyEventId0  = 0U; // MTE2_V, ping 已搬入
    static constexpr uint32_t attenMaskBufReadyEventId1  = 1U; // MTE2_V, pong 已搬入
    static constexpr uint32_t attenMaskBufFreeEventId0   = 0U; // V_MTE2, ping V 读完
    static constexpr uint32_t attenMaskBufFreeEventId1   = 1U; // V_MTE2, pong V 读完

    // ----- sinkBuf：learnable sink GM→UB -----
    static constexpr uint32_t sinkBufReadyEventId        = 4U; // MTE2_V, sinkBuf 已搬入

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

    using vec2ResGmType = typename std::conditional<splitD, GlobalTensor<float>, int8_t>::type;

    using flashdecodeGmType = GlobalTensor<float>;

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

    vec2ResGmType vec2ResGm[3];
    GlobalTensor<INPUT_T> sinkGm;

    flashdecodeGmType accumOutGm;
    flashdecodeGmType softmaxFDSumGm;
    flashdecodeGmType softmaxFDMaxGm;

    // ub
    LocalTensor<uint8_t> vec1ApiTmpBuf; // ProcessVec1Vf API 的 sharedTmpBuffer（vec1 计算临时区，仅特定模板分支需要）
    LocalTensor<INPUT_T> stage1OutBuf[2];
    LocalTensor<uint8_t> attenMaskBuf[2];
    LocalTensor<T> stage2OutBuf;
    LocalTensor<float> softmaxMaxBuf[PRELOAD_N + 1];
    LocalTensor<float> softmaxSumBuf[PRELOAD_N + 1];
    LocalTensor<T> softmaxExpBuf[PRELOAD_N + 1];
    LocalTensor<uint8_t> vselrIndexesBuf[4]; // 静态Tensor编程：LocalTensor 直接替代 TBuf<>
    LocalTensor<T> mm2InBuf;
    LocalTensor<float> maxBrdcstBuf;
    LocalTensor<float> sumBrdcstBuf;
    LocalTensor<float> lseBuf;
    LocalTensor<INPUT_T> sinkBuf;

    const ConstInfoX &constInfo;
    T negativeFloatScalar;
    int64_t bmm2SubBlockOffset = 0;
    int64_t vec2SubBlockOffset = 0;
    bool isFd = false;

    // ==================== Functions ======================
    __aicore__ inline FANoQuantGqaBlockVecNd(ConstInfoX &constInfo) constInfo(constInfo){};

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

        if constexpr (splitD) {
            int64_t vec2ResultSize = mBaseSize * constInfo.dBasicBlock;
            vec2SubBlockOffset = constInfo.subBlockIdx * vec2ResultSize >> 1;
            int64_t prevCoretotalOffset = constInfo.aicIdx * 3 * vec2ResultSize; // 3为preload次数
            vec2ResGm[0].SetGlobalBuffer((__gm__ T *)workspace + prevCoretotalOffset);
            vec2ResGm[1].SetGlobalBuffer((__gm__ T *)workspace + prevCoretotalOffset + vec2ResultSize);
            vec2ResGm[2].SetGlobalBuffer((__gm__ T *)workspace + prevCoretotalOffset + 2 * vec2ResultSize);
            workspace = workspace + constInfo.coreNum * 3 * vec2ResultSize * sizeof(T);
        }

        if constexpr (!bmm2Write2Ub) {
            int64_t bmm2ResBlock = constInfo.dSizeV;
            if constexpr (splitD) {
                bmm2ResBlock = (int64_t)dVTemplateType;
            }
            int64_t mm2ResultSize = (mBaseSize) * bmm2ResBlock;  // 使用cube计算的总大小, Gm上的数据按照实际的dSize存储
            bmm2SubBlockOffset = constInfo.subBlockIdx * mm2ResultSize >> 1;  // mBaseSize一定可以被2整除
        }

        if (learnableSink != nullptr) {
            sinkGm.SetGlobalBuffer((__gm__ INPUT_T *)learnableSink);
        }

        if (this->isFd) {
            accumOutGm.SetGlobalBuffer((__gm__ float *)workspace);
            softmaxFDSumGm.SetGlobalBuffer((__gm__ float *)workspace + constInfo.accumOutSize);
            softmaxFDMaxGm.SetGlobalBuffer((__gm__ float *)workspace + constInfo.accumOutSize +
                                           constInfo.logSumExpSize);
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
        if (IsInitAttentionOutGm()) {
            // 此时 InitBuffers() 尚未运行，整个 UB 空闲，从偏移 0 处直接借用 scratch UB
            // 供 fa_static::InitOutput 内部 Duplicate→DataCopyPad UB→GM 使用。
            LocalTensor<uint8_t> initOutputScratch =
                LocalTensor<uint8_t>(TPosition::VECIN, 0U, fa_static::INIT_OUTPUT_SCRATCH_BYTES);
            SetFlag<AscendC::HardEvent::MTE3_V>(initOutputBufFreeEventId); // 释放 initOutputScratch
            InitOutputSingleCore(initOutputScratch);
            if (constInfo.isSoftmaxLseEnable) {
                InitLseOutputSingleCore(initOutputScratch);
            }
            WaitFlag<AscendC::HardEvent::MTE3_V>(initOutputBufFreeEventId);
            SyncAll();
        }
    }

    __aicore__ inline bool IsInitAttentionOutGm()
    {
        return constInfo.needInitOutput;
    }

    __aicore__ inline void InitOutputSingleCore(LocalTensor<uint8_t> initOutputScratch)
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
            WaitFlag<AscendC::HardEvent::MTE3_V>(initOutputBufFreeEventId);
            if constexpr (IsSameType<OUTPUT_T, int8_t>::value) {
                GlobalTensor<half> attentionOutTmpGm;
                attentionOutTmpGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(attentionOutGm.GetPhyAddr(0)));
                fa_static::InitOutput<half>(attentionOutTmpGm[constInfo.aivIdx * singleCoreSize],
                                            singleInitOutputSize, 0, initOutputScratch);
            } else {
                fa_static::InitOutput<OUTPUT_T>(attentionOutGm[constInfo.aivIdx * singleCoreSize],
                                                singleInitOutputSize, 0, initOutputScratch);
            }
            SetFlag<AscendC::HardEvent::MTE3_V>(initOutputBufFreeEventId);
        }
    }

    __aicore__ inline void InitLseOutputSingleCore(LocalTensor<uint8_t> initOutputScratch)
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
            WaitFlag<AscendC::HardEvent::MTE3_V>(initOutputBufFreeEventId);
            fa_static::InitOutput<float>(softmaxLseGm[constInfo.aivIdx * singleCoreSize], singleInitOutputSize,
                                         3e+99, initOutputScratch); // 3e+99: set the value of invalid batch to inf
            SetFlag<AscendC::HardEvent::MTE3_V>(initOutputBufFreeEventId);
        }
    }

    __aicore__ inline void InitFdWorkspaceSingleCore(LocalTensor<uint8_t> initOutputScratch)
    {
        // accumOut / softmaxFDSum / softmaxFDMax 在 user workspace 上连续排布
        // （见 InitVecInput：accumOut@workspace, softmaxFDSum@+accumOutSize, softmaxFDMax@+accumOutSize+logSumExpSize），
        // 总大小为 accumOutSize + 2 * logSumExpSize（单位：float 元素）。整体清零以消除批跑跨启动的脏残留。
        int64_t totalFdWsSize = (int64_t)constInfo.accumOutSize + 2 * (int64_t)constInfo.logSumExpSize;
        if (totalFdWsSize <= 0) {
            return;
        }
        int64_t singleCoreSize =
            (totalFdWsSize + (2 * constInfo.coreNum) - 1) / (2 * constInfo.coreNum); // 2 means c:v = 1:2
        int64_t tailSize = totalFdWsSize - constInfo.aivIdx * singleCoreSize;
        int64_t singleInitSize = tailSize < singleCoreSize ? tailSize : singleCoreSize;

        if (singleInitSize > 0) {
            WaitFlag<AscendC::HardEvent::MTE3_V>(initOutputBufFreeEventId);
            fa_static::InitOutput<float>(accumOutGm[constInfo.aivIdx * singleCoreSize], singleInitSize, 0,
                                         initOutputScratch);
            SetFlag<AscendC::HardEvent::MTE3_V>(initOutputBufFreeEventId);
        }
    }

    // =================================Private Functions=================================

    __aicore__ inline void SoftmaxDataCopyOut(RunInfoX runInfo, LocalTensor<float> &sumUb, LocalTensor<float> &maxUb)
    {
        if (this->isFd) {
            if (runInfo.isS2SplitCore) {
                ComputeLogSumExpAndCopyToGm(runInfo, sumUb, maxUb);
            }
        }

        if (this->isFd) {
            if (!runInfo.isS2SplitCore && constInfo.isSoftmaxLseEnable) {
                SoftmaxLseCopyOut(sumUb, maxUb, runInfo);
            }
        } else {
            if (constInfo.isSoftmaxLseEnable) {
                SoftmaxLseCopyOut(sumUb, maxUb, runInfo);
            }
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

        // 入口 Wait：跨调用 lseBuf free 同步 —— 等上轮本函数 / BroadCastAndCopyOut 出口 SetFlag<MTE3_V>
        // 实际发出（必然在上轮 DataCopy lseBuf→GM 完成之后），首轮消费 AllocEventID 预置的 SetFlag。
        // 镜像 master 中 softmaxLseQueue.AllocTensor() 内部的 free WaitFlag 语义。
        WaitFlag<HardEvent::MTE3_V>(lseBufFreeEventId);

        uint32_t vecMIdx = runInfo.gS1Idx + runInfo.vecMbaseIdx;
        LocalTensor<float> &lseUb = this->lseBuf;
        ComputeLseOutputVF(lseUb, softmaxSumTmp, softmaxMaxTmp, runInfo.actVecMSize);
        // V→MTE3 ready fence：V 写完 lseBuf → MTE3 可读
        SetFlag<HardEvent::V_MTE3>(lseBufReadyEventId);
        WaitFlag<HardEvent::V_MTE3>(lseBufReadyEventId);

        if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
            uint32_t prefixBS1 = qActSeqLensParser->GetTBase(runInfo.bIdx);
            uint64_t bN2Offset = runInfo.n2Idx * constInfo.gSize * constInfo.t1Size + prefixBS1;
            DataCopySoftmaxLseTNDtoNTArch35<T, ConstInfoX>(softmaxLseGm, lseUb, bN2Offset, vecMIdx, runInfo.actVecMSize,
                                                           constInfo);
        } else if constexpr (layout == LayOutTypeEnum::LAYOUT_BSH) {
            uint64_t bN2Offset = runInfo.bIdx * constInfo.n2Size * constInfo.gSize * constInfo.s1Size +
                                 runInfo.n2Idx * constInfo.gSize * constInfo.s1Size;
            uint64_t qActSeqLens = qActSeqLensParser->GetActualSeqLength(runInfo.bIdx);
            DataCopySoftmaxLseBSNDArch35<T, ConstInfoX>(softmaxLseGm, lseUb, bN2Offset, vecMIdx, runInfo.actVecMSize,
                                                        constInfo);
        } else { // BNSD
            uint64_t bN2Offset = runInfo.bIdx * constInfo.n2Size * constInfo.gSize * constInfo.s1Size +
                                 runInfo.n2Idx * constInfo.gSize * constInfo.s1Size;
            uint64_t qActSeqLens = qActSeqLensParser->GetActualSeqLength(runInfo.bIdx);
            DataCopySoftmaxLseBNSDArch35<T, ConstInfoX>(softmaxLseGm, lseUb, bN2Offset, vecMIdx, runInfo.actVecMSize,
                                                        constInfo, qActSeqLens);
        }
        // 出口 Set：issue 到 MTE3 pipe，硬件流水语义保证其在本轮 DataCopy 完成之后才"发出"。
        // 镜像 master 中 softmaxLseQueue.FreeTensor() 内部 SetFlag 语义，留给下轮 entry Wait 消费；
        // 末次调用由 FreeEventID 的 WaitFlag 排干。
        SetFlag<HardEvent::MTE3_V>(lseBufFreeEventId);
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
        WaitFlag<HardEvent::MTE2_V>(sinkBufReadyEventId);
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
        SetFlag<HardEvent::MTE2_V>(sinkBufReadyEventId);
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
            const uint32_t maskBufIdx = runInfo.loop & (DB - 1);
            attenMaskUb = this->attenMaskBuf[maskBufIdx];
            // V → MTE2 反向同步：等上一轮 V 读完该 ping/pong 槽后，本轮 MTE2 才能覆写
            // (AllocEventID 预置过 V_MTE2 ping/pong，使首轮 Wait 能立即通过)
            WaitFlag<HardEvent::V_MTE2>(attenMaskBufFreeEventId0 + maskBufIdx);
            AttenMaskCopyIn(attenMaskUb, 0, runInfo.actVecMSize, runInfo);
            // AttenMaskCopyIn 内部已通过 MTE2_V 事件保证 mask 数据 V 可见
        }

        LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.mloop % (PRELOAD_N + 1)];
        LocalTensor<float> maxUb = this->softmaxMaxBuf[runInfo.mloop % (PRELOAD_N + 1)];
        LocalTensor<float> expUb = this->softmaxExpBuf[runInfo.loop % (PRELOAD_N + 1)].template ReinterpretCast<T>();
        LocalTensor<T> pScaleUb;
        LocalTensor<T> queryScaleUb;
        LocalTensor<uint8_t> apiTmpBuffer;

        apiTmpBuffer = this->vec1ApiTmpBuf;

        int64_t stage1Offset = 0;
        stage1Offset = runInfo.loop & (DB - 1);

        WaitFlag<HardEvent::MTE3_V>(stage1Offset == 0 ? stage1OutBufFreeEventId0 : stage1OutBufFreeEventId1);

        float descaleQK = 1.0;
        float deSCaleKValue = 1.0;

        LocalTensor<T> mmRes = bmm1UbTensor;
        LocalTensor<INPUT_T> stage1CastTensor = this->stage1OutBuf[stage1Offset];
        if (runInfo.isFirstS2Loop) {
            if (likely(runInfo.actSingleLoopS2Size == 128)) {
                FaVectorApi::ProcessVec1Vf<T, INPUT_T, INPUT_T /*pseShiftType*/, false, mBaseSize, s2BaseSize, EQ_128, hasAtten,
                              PseTypeEnum::PSE_NONE_TYPE, hasDrop, false, false>(
                    stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, mmRes, expUb, sumUb, maxUb, attenMaskUb,
                    nonePseUb, dropMaskUb, apiTmpBuffer, pScaleUb, runInfo.actVecMSize, runInfo.actSingleLoopS2Size,
                    0 /*pseStride*/, 0.0f /*slopes*/, 0.0f /*posShift*/, static_cast<T>(constInfo.scaleValue),
                    descaleQK, negativeFloatScalar, 0.0F, queryScaleUb, deSCaleKValue);
            } else if (runInfo.actSingleLoopS2Size <= 64) {
                FaVectorApi::ProcessVec1Vf<T, INPUT_T, INPUT_T /*pseShiftType*/, false, mBaseSize, s2BaseSize, GT_0_AND_LTE_64,
                              hasAtten, PseTypeEnum::PSE_NONE_TYPE, hasDrop, false, false>(
                    stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, mmRes, expUb, sumUb, maxUb, attenMaskUb,
                    nonePseUb, dropMaskUb, apiTmpBuffer, pScaleUb, runInfo.actVecMSize, runInfo.actSingleLoopS2Size,
                    0 /*pseStride*/, 0.0f /*slopes*/, 0.0f /*posShift*/, static_cast<T>(constInfo.scaleValue),
                    descaleQK, negativeFloatScalar, 0.0F, queryScaleUb, deSCaleKValue);
            } else if (runInfo.actSingleLoopS2Size < 128) {
                FaVectorApi::ProcessVec1Vf<T, INPUT_T, INPUT_T /*pseShiftType*/, false, mBaseSize, s2BaseSize, GT_64_AND_LTE_128,
                              hasAtten, PseTypeEnum::PSE_NONE_TYPE, hasDrop, false, false>(
                    stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, mmRes, expUb, sumUb, maxUb, attenMaskUb,
                    nonePseUb, dropMaskUb, apiTmpBuffer, pScaleUb, runInfo.actVecMSize, runInfo.actSingleLoopS2Size,
                    0 /*pseStride*/, 0.0f /*slopes*/, 0.0f /*posShift*/, static_cast<T>(constInfo.scaleValue),
                    descaleQK, negativeFloatScalar, 0.0F, queryScaleUb, deSCaleKValue);
            } else {
                if constexpr (s2BaseSize == 256) {
                    FaVectorApi::ProcessVec1Vf<T, INPUT_T, INPUT_T /*pseShiftType*/, false, mBaseSize, s2BaseSize,
                                  GT_128_AND_LTE_256, hasAtten, PseTypeEnum::PSE_NONE_TYPE, hasDrop>(
                        stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, mmRes, expUb, sumUb, maxUb, attenMaskUb,
                        nonePseUb, dropMaskUb, apiTmpBuffer, expUb, runInfo.actVecMSize, runInfo.actSingleLoopS2Size,
                        0 /*pseStride*/, 0.0f /*slopes*/, 0.0f /*posShift*/, static_cast<T>(constInfo.scaleValue),
                        descaleQK, negativeFloatScalar, 0.0F);
                }
            }
        } else {
            if (likely(runInfo.actSingleLoopS2Size == 128)) {
                FaVectorApi::ProcessVec1Vf<T, INPUT_T, INPUT_T /*pseShiftType*/, true, mBaseSize, s2BaseSize, EQ_128, hasAtten,
                              PseTypeEnum::PSE_NONE_TYPE, hasDrop, false, false>(
                    stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, mmRes, expUb, sumUb, maxUb, attenMaskUb,
                    nonePseUb, dropMaskUb, apiTmpBuffer, pScaleUb, runInfo.actVecMSize, runInfo.actSingleLoopS2Size,
                    0 /*pseStride*/, 0.0f /*slopes*/, 0.0f /*posShift*/, static_cast<T>(constInfo.scaleValue),
                    descaleQK, negativeFloatScalar, 0.0F, queryScaleUb, deSCaleKValue);
            } else if (runInfo.actSingleLoopS2Size <= 64) {
                FaVectorApi::ProcessVec1Vf<T, INPUT_T, INPUT_T /*pseShiftType*/, true, mBaseSize, s2BaseSize, GT_0_AND_LTE_64,
                              hasAtten, PseTypeEnum::PSE_NONE_TYPE, hasDrop, false, false>(
                    stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, mmRes, expUb, sumUb, maxUb, attenMaskUb,
                    nonePseUb, dropMaskUb, apiTmpBuffer, pScaleUb, runInfo.actVecMSize, runInfo.actSingleLoopS2Size,
                    0 /*pseStride*/, 0.0f /*slopes*/, 0.0f /*posShift*/, static_cast<T>(constInfo.scaleValue),
                    descaleQK, negativeFloatScalar, 0.0F, queryScaleUb, deSCaleKValue);
            } else if (runInfo.actSingleLoopS2Size < 128) {
                FaVectorApi::ProcessVec1Vf<T, INPUT_T, INPUT_T /*pseShiftType*/, true, mBaseSize, s2BaseSize, GT_64_AND_LTE_128,
                              hasAtten, PseTypeEnum::PSE_NONE_TYPE, hasDrop, false, false>(
                    stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, mmRes, expUb, sumUb, maxUb, attenMaskUb,
                    nonePseUb, dropMaskUb, apiTmpBuffer, pScaleUb, runInfo.actVecMSize, runInfo.actSingleLoopS2Size,
                    0 /*pseStride*/, 0.0f /*slopes*/, 0.0f /*posShift*/, static_cast<T>(constInfo.scaleValue),
                    descaleQK, negativeFloatScalar, 0.0F, queryScaleUb, deSCaleKValue);
            } else {
                if constexpr (s2BaseSize == 256) {
                    FaVectorApi::ProcessVec1Vf<T, INPUT_T, INPUT_T /*pseShiftType*/, true, mBaseSize, s2BaseSize, GT_128_AND_LTE_256,
                                  hasAtten, PseTypeEnum::PSE_NONE_TYPE, hasDrop>(
                        stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, mmRes, expUb, sumUb, maxUb, attenMaskUb,
                        nonePseUb, dropMaskUb, apiTmpBuffer, expUb, runInfo.actVecMSize, runInfo.actSingleLoopS2Size,
                        0 /*pseStride*/, 0.0f /*slopes*/, 0.0f /*posShift*/, static_cast<T>(constInfo.scaleValue),
                        descaleQK, negativeFloatScalar, 0.0F);
                }
            }
        }
        if constexpr (hasAtten) {
            // V → MTE2 反向同步：V 读完 mask 后释放该 ping/pong 槽，下一轮 MTE2 可覆写
            const uint32_t maskBufIdx = runInfo.loop & (DB - 1);
            SetFlag<HardEvent::V_MTE2>(attenMaskBufFreeEventId0 + maskBufIdx);
        }
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(bmm1CcId);

        // ===================DataCopy to L1 ====================
        // V → MTE3 正向同步：等 V 写完 stage1CastTensor 后 MTE3 才能搬运到 L1
        SetFlag<HardEvent::V_MTE3>(stage1OutBufReadyEventId0 + stage1Offset);
        WaitFlag<HardEvent::V_MTE3>(stage1OutBufReadyEventId0 + stage1Offset);
        LocalTensor<INPUT_T> mm2AL1Tensor;
        mm2AL1Tensor = l1PTensor;

        if (likely(runInfo.actVecMSize != 0)) {
            DataCopy(mm2AL1Tensor[constInfo.subBlockIdx * (blockBytes / sizeof(INPUT_T)) *
                                  (runInfo.actMSize - runInfo.actVecMSize)],
                     stage1CastTensor,
                     {s2BaseSize / 16, (uint16_t)runInfo.actVecMSize, (uint16_t)(vec1Srcstride - runInfo.actVecMSize),
                      (uint16_t)(mBaseSize - runInfo.actVecMSize)});
        }

        SetFlag<HardEvent::MTE3_V>(stage1Offset == 0 ? stage1OutBufFreeEventId0 : stage1OutBufFreeEventId1);

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
        WaitFlag<HardEvent::MTE3_V>(stage2OutBufFreeEventId);
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
        SetFlag<HardEvent::MTE3_V>(stage2OutBufFreeEventId);
    }

    __aicore__ inline void Bmm2ResCastAndCopyOut(RunInfoX &runInfo, LocalTensor<T> &vec2ResUb, uint32_t mStartVec,
                                                 uint32_t mDealSize)
    {
        LocalTensor<OUTPUT_T> attenOut;
        int64_t dSizeAligned64 = (int64_t)dVTemplateType;
        if constexpr (splitD) {
            dSizeAligned64 = constInfo.dBasicBlock;
        }

        attenOut.SetAddr(vec2ResUb.address_);

        RowInvalid(vec2ResUb, mStartVec, mDealSize, runInfo, dSizeAligned64);
        Cast(attenOut, vec2ResUb, RoundMode::CAST_ROUND, mDealSize * dSizeAligned64);
        SetFlag<HardEvent::V_MTE3>(vec2OutBufReadyEventId);
        WaitFlag<HardEvent::V_MTE3>(vec2OutBufReadyEventId);

        Bmm2DataCopyOutTrans(runInfo, attenOut, mStartVec, mDealSize);
    }

    template <typename VEC2_RES_T>
    __aicore__ inline void ProcessVec2DSplit(GlobalTensor<T> &mmRes, const RunInfoX &runInfo)
    {
        // bmm2 result is on GM and global update data on UB
        uint32_t subMBase = 8192 / constInfo.dBasicBlock;
        int64_t vec2LoopLimit = CeilDiv(runInfo.actVecMSize, subMBase);
        LocalTensor<T> vec2ResUb = this->stage2OutBuf;
        WaitFlag<HardEvent::MTE3_V>(stage2OutBufFreeEventId);
        LocalTensor<T> bmm2Ub = mm2InBuf;

        constexpr uint32_t mte2ToV    = mm2InBufReadyEventId;
        constexpr uint32_t vToMte2    = mm2InBufFreeEventId;
        constexpr uint32_t mte3ToMte2 = vec2ResGmRecycleEventId;

        for (int64_t mOIdx = 0; mOIdx < vec2LoopLimit; mOIdx++) {
            int32_t mIdx = mOIdx * subMBase;
            uint32_t subMRealsize = subMBase;
            if (mOIdx == vec2LoopLimit - 1) {
                subMRealsize = runInfo.actVecMSize - mIdx;
            }

            int64_t vec2CalcSize = subMRealsize * constInfo.dBasicBlock;
            // Gm地址偏移是按照实际DSize实现的，虽然我们设置了KC=192
            int64_t mm2ResInnerOffset = mIdx * dTemplateAlign64;
            SetFlag<HardEvent::V_MTE2>(vToMte2);
            WaitFlag<HardEvent::V_MTE2>(vToMte2);
            if (constInfo.dSizeV == dTemplateAlign64) {
                
                    DataCopy(bmm2Ub,
                             mmRes[constInfo.subBlockIdx * (runInfo.actMSize - runInfo.actVecMSize) *
                                       (int64_t)dVTemplateType +
                                   mm2ResInnerOffset],
                             vec2CalcSize);
                
            } else {
                DataCopyParams dataCopyParams;
                DataCopyPadParams dataCopyPadParams;
                dataCopyParams.blockCount = subMRealsize;
                dataCopyParams.dstStride = (constInfo.dBasicBlock - constInfo.dSizeV) * sizeof(T) / blockBytes;
                dataCopyParams.srcStride = (dTemplateAlign64 - constInfo.dSizeV) * sizeof(T);
                dataCopyParams.blockLen = constInfo.dSizeV * sizeof(T);
                
                    DataCopyPad(bmm2Ub,
                                mmRes[constInfo.subBlockIdx * (runInfo.actMSize - runInfo.actVecMSize) *
                                          (int64_t)dVTemplateType +
                                      mm2ResInnerOffset],
                                dataCopyParams, dataCopyPadParams);
                
            }

            // 经过了跳读，UB上每行是按照dTemplateAlign64对齐的
            int64_t vec2ResInnerOffset = mIdx * constInfo.dBasicBlock;
            if (vec2LoopLimit > 1) {
                SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2);
                WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2);
                
                    DataCopy(vec2ResUb,
                             this->vec2ResGm[runInfo.loop & (DB - 1)][constInfo.subBlockIdx *
                                                                    (runInfo.actMSize - runInfo.actVecMSize) *
                                                                    constInfo.dBasicBlock +
                                                                vec2ResInnerOffset],
                             vec2CalcSize);
                
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
                                                                             subMRealsize, constInfo.dBasicBlock, 0.0F,
                                                                             deSCalePreVValue);
                } else {
                    int64_t vec2SumBufOffset = mIdx;
                    LocalTensor<float> sumUb =
                        this->softmaxSumBuf[runInfo.mloop % (PRELOAD_N + 1)][vec2SumBufOffset];
                    FlashUpdateLastNew<T, INPUT_T, OUTPUT_T, 0xFF, false, false>(
                        vec2ResUb, bmm2Ub, vec2ResUb, expUb, expUb, sumUb, subMRealsize, constInfo.dBasicBlock, 0.0F,
                        deSCalePreVValue);
                }
            }

            if (unlikely(runInfo.isLastS2Loop)) {
                if (unlikely(runInfo.isFirstS2Loop)) {
                    int64_t vec2SumBufOffset = mIdx;
                    LocalTensor<float> sumUb =
                        this->softmaxSumBuf[runInfo.mloop % (PRELOAD_N + 1)][vec2SumBufOffset];
                    LastDivNew<T, INPUT_T, OUTPUT_T, 0xFF, false>(vec2ResUb, vec2ResUb, sumUb, subMRealsize,
                                                                  constInfo.dBasicBlock, 0.0F);
                }
                CopyOutAttentionOut(runInfo, vec2ResUb, mIdx, subMRealsize);
            } else if (vec2LoopLimit > 1) {
                SetFlag<HardEvent::V_MTE3>(vec2OutBufReadyEventId);
                WaitFlag<HardEvent::V_MTE3>(vec2OutBufReadyEventId);
                
                    DataCopy(this->vec2ResGm[runInfo.loop & (DB - 1)][constInfo.subBlockIdx *
                                                                    (runInfo.actMSize - runInfo.actVecMSize) *
                                                                    constInfo.dBasicBlock +
                                                                vec2ResInnerOffset],
                             vec2ResUb, vec2CalcSize);
                
            }
        }
        SetFlag<HardEvent::MTE3_V>(stage2OutBufFreeEventId);
    }

    __aicore__ inline void CopyOutAttentionOut(RunInfoX runInfo, LocalTensor<T> &vec2ResUb, uint32_t mStartVec,
                                               uint32_t mDealSize)
    {
        if (this->isFd) {
            if (runInfo.isS2SplitCore) {
                Bmm2ResForFDCopyOut(runInfo, vec2ResUb, mStartVec, mDealSize);
            } else {
                Bmm2ResCastAndCopyOut(runInfo, vec2ResUb, mStartVec, mDealSize);
            }
        } else {
            Bmm2ResCastAndCopyOut(runInfo, vec2ResUb, mStartVec, mDealSize);
        }
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
                                      .colCount = (uint32_t)(splitD ? constInfo.dBasicBlock : dTemplateAlign64)};
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

    __aicore__ inline void BroadCastAndCopyOut(const RunInfoX &runInfo, LocalTensor<float> &sumUb,
                                               LocalTensor<float> &maxUb, int64_t gmOffset, int64_t calculateSize)
    {
        // 入口 Wait：跨调用 brdcstBuf free 同步 —— 等上轮本函数 / SoftmaxLseCopyOut 出口 SetFlag<MTE3_V>
        // 实际发出（必然在上轮 DataCopy(brdcstBuf/lseBuf→GM) 完成之后），首轮消费 AllocEventID 预置。
        // 镜像 master 中 sumBrdcst.AllocTensor() 内部的 free WaitFlag 语义。
        WaitFlag<HardEvent::MTE3_V>(lseBufFreeEventId);

        // Copy sum to gm
        FaVectorApi::BroadcastMaxSum(this->sumBrdcstBuf, sumUb, runInfo.actVecMSize);
        // V→MTE3 ready fence：V 写完 sumBrdcstBuf → MTE3 可读
        SetFlag<HardEvent::V_MTE3>(brdcstBufReadyEventId);
        WaitFlag<HardEvent::V_MTE3>(brdcstBufReadyEventId);
        DataCopy(softmaxFDSumGm[gmOffset], this->sumBrdcstBuf, calculateSize);

        // Copy max to gm —— sum / max 写不同 buf 不冲突；MTE3 pipe FIFO 保证 max DataCopy 排在 sum 之后，
        // 因此 sum 与 max 之间无需额外 V↔MTE3 fence（与 master 中 sumBrdcst / maxBrdcst 独立 TQue 一致）
        FaVectorApi::BroadcastMaxSum(this->maxBrdcstBuf, maxUb, runInfo.actVecMSize);
        SetFlag<HardEvent::V_MTE3>(brdcstBufReadyEventId);
        WaitFlag<HardEvent::V_MTE3>(brdcstBufReadyEventId);
        DataCopy(softmaxFDMaxGm[gmOffset], this->maxBrdcstBuf, calculateSize);

        // 出口 Set：issue 到 MTE3 pipe，硬件流水语义保证其在 sum + max 两次 DataCopy 都完成之后才"发出"。
        // 镜像 master 中 maxBrdcst.FreeTensor() 的 SetFlag 语义；留给下轮 entry Wait 消费；
        // 末次调用由 FreeEventID 的 WaitFlag 排干。
        SetFlag<HardEvent::MTE3_V>(lseBufFreeEventId);
    }

    __aicore__ inline void ComputeLogSumExpAndCopyToGm(const RunInfoX &runInfo, LocalTensor<float> &sumUb,
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

    __aicore__ inline void Bmm2ResForFDCopyOut(const RunInfoX &runInfo, LocalTensor<T> &vec2ResUb, uint32_t mStartVec,
                                               uint32_t mDealSize)
    {
        int64_t dSizeAligned64 = (int64_t)dVTemplateType;
        if constexpr (splitD) {
            dSizeAligned64 = constInfo.dBasicBlock;
        }
        SetFlag<HardEvent::V_MTE3>(fdOutBufReadyEventId0 + (runInfo.loop & (DB - 1)));
        WaitFlag<HardEvent::V_MTE3>(fdOutBufReadyEventId0 + (runInfo.loop & (DB - 1)));
        uint64_t gmOffset =
            runInfo.faTmpOutWsPos * mBaseSize * constInfo.dSizeV + (runInfo.vecMbaseIdx + mStartVec) * constInfo.dSizeV;

        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = mDealSize;
        dataCopyParams.blockLen = constInfo.dSizeV * sizeof(T);
        dataCopyParams.srcStride = (dSizeAligned64 - constInfo.dSizeV) / (FA_BYTE_BLOCK / sizeof(T));
        dataCopyParams.dstStride = 0;

        DataCopyPad(accumOutGm[gmOffset], vec2ResUb, dataCopyParams);
    }

    template<typename BmmResT>
    __aicore__ inline void ProcessVec2(BmmResT &bmm2ResTensor, uint32_t bmm2CcId, RunInfoX runInfo)
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

    __aicore__ inline void ProcessVec2_C2GmUpdateUb(GlobalTensor<T> &mmRes, RunInfoX runInfo)
    {
        // bmm2 result is on GM and global update data on UB
        uint32_t subMBase = 8192 / dTemplateAlign64;
        int64_t vec2LoopLimit = CeilDiv(runInfo.actVecMSize, subMBase);
        LocalTensor<T> vec2ResUb = this->stage2OutBuf;
        WaitFlag<HardEvent::MTE3_V>(stage2OutBufFreeEventId);
        LocalTensor<T> bmm2Ub = mm2InBuf;

        constexpr uint32_t mte2ToV = mm2InBufReadyEventId;
        constexpr uint32_t vToMte2 = mm2InBufFreeEventId;

        for (int64_t mOIdx = 0; mOIdx < vec2LoopLimit; mOIdx++) {
            int32_t mIdx = mOIdx * subMBase;
            uint32_t subMRealsize = subMBase;
            if (mOIdx == vec2LoopLimit - 1) {
                subMRealsize = runInfo.actVecMSize - mIdx;
            }

            int64_t vec2CalcSize = subMRealsize * dTemplateAlign64;
            // Gm地址偏移是按照实际DSize实现的，虽然我们设置了KC=192
            int64_t mm2ResInnerOffset = mIdx * constInfo.dSizeV;
            SetFlag<HardEvent::V_MTE2>(vToMte2);
            WaitFlag<HardEvent::V_MTE2>(vToMte2);
            if (constInfo.dSizeV == dTemplateAlign64) {
                
                    DataCopy(bmm2Ub,
                             mmRes[constInfo.subBlockIdx * (runInfo.actMSize - runInfo.actVecMSize) * constInfo.dSizeV +
                                   mm2ResInnerOffset],
                             vec2CalcSize);
                
            } else {
                DataCopyParams dataCopyParams;
                DataCopyPadParams dataCopyPadParams;
                dataCopyParams.blockCount = subMRealsize;
                dataCopyParams.dstStride = (dTemplateAlign64 - constInfo.dSizeV) * sizeof(T) / blockBytes;
                dataCopyParams.srcStride = 0;
                dataCopyParams.blockLen = constInfo.dSizeV * sizeof(T);
                
                    DataCopyPad(
                        bmm2Ub,
                        mmRes[constInfo.subBlockIdx * (runInfo.actMSize - runInfo.actVecMSize) * constInfo.dSizeV +
                              mm2ResInnerOffset],
                        dataCopyParams, dataCopyPadParams);
                
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
        SetFlag<HardEvent::MTE3_V>(stage2OutBufFreeEventId);
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

        // vec1ApiTmpBuf：compile-time 条件，512 bytes（BASE + 121600）
        //   仅 ProcessVec1Nd 分支需要（s2=256 && m!=128，或 !useDn）, 作为 ProcessVec1Vf 的 sharedTmpBuffer
        constexpr bool hasVec1ApiTmpBuf = (s2BaseSize == 256U) ? (mBaseSize != 128U) : true;
        if constexpr (hasVec1ApiTmpBuf) {
            vec1ApiTmpBuf = LocalTensor<uint8_t>(TPosition::VECIN, BASE + 121600U, 512U);
        }

        // lseBuf / sinkBuf：runtime 条件，地址已在编译期固定（始终位于 vec1ApiTmpBuf 区末尾之后）
        constexpr uint32_t vec1ApiTmpBufSz = hasVec1ApiTmpBuf ? 512U : 0U;
        constexpr uint32_t OFF_LSE = BASE + 121600U + vec1ApiTmpBufSz;
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
    // 这里预置的都是 *FreeEventId* 类事件，让对应 buff 首轮 Wait 能立即通过
    __aicore__ inline void AllocEventID()
    {
        SetFlag<HardEvent::MTE3_V>(stage2OutBufFreeEventId);
        SetFlag<HardEvent::MTE3_V>(lseBufFreeEventId);
        SetFlag<HardEvent::MTE3_V>(stage1OutBufFreeEventId0);
        SetFlag<HardEvent::MTE3_V>(stage1OutBufFreeEventId1);
        if constexpr (HAS_MASK) {
            SetFlag<HardEvent::V_MTE2>(attenMaskBufFreeEventId0);
            SetFlag<HardEvent::V_MTE2>(attenMaskBufFreeEventId1);
        }
    }

    __aicore__ inline void FreeEventID()
    {
        WaitFlag<AscendC::HardEvent::MTE3_V>(stage2OutBufFreeEventId);
        WaitFlag<AscendC::HardEvent::MTE3_V>(lseBufFreeEventId);
        WaitFlag<AscendC::HardEvent::MTE3_V>(stage1OutBufFreeEventId0);
        WaitFlag<AscendC::HardEvent::MTE3_V>(stage1OutBufFreeEventId1);
        if constexpr (HAS_MASK) {
            WaitFlag<AscendC::HardEvent::V_MTE2>(attenMaskBufFreeEventId0);
            WaitFlag<AscendC::HardEvent::V_MTE2>(attenMaskBufFreeEventId1);
        }
    }

    // FD 兼容：进入 FlashDecode 前排干 mask V→MTE2 ping/pong 反向同步预置。
    // FA 的 AllocEventID 预置了 V_MTE2[0]/[1] 且这两个 flag 会一直挂到所有 section 结束才由
    // FreeEventID 排干，因此 FD 运行时 FA 仍残留 2 个 V_MTE2 pending flag。FD 自身的 V_MTE2 同步
    // （SYNC_* = 8~13）在物理 flag 空间上与 FA 的 [0]/[1] 冲突，导致 FD 的 WaitFlag<V_MTE2> 卡死。
    // 进入 FD 前先 Wait 排干，使 FD 期间 FA 无任何残留 V_MTE2 flag。
    __aicore__ inline void DrainMaskReverseSync()
    {
        if constexpr (HAS_MASK) {
            WaitFlag<AscendC::HardEvent::V_MTE2>(attenMaskBufFreeEventId0);
            WaitFlag<AscendC::HardEvent::V_MTE2>(attenMaskBufFreeEventId1);
        }
    }

    // FD 结束后恢复 mask V→MTE2 ping/pong 预置，供下一 section 的 FA ProcessVec1Nd 首轮 Wait 使用，
    // 并保证 kernel 退出时 FreeEventID 的 WaitFlag 配平。与 DrainMaskReverseSync 成对调用。
    __aicore__ inline void PresetMaskReverseSync()
    {
        if constexpr (HAS_MASK) {
            SetFlag<AscendC::HardEvent::V_MTE2>(attenMaskBufFreeEventId0);
            SetFlag<AscendC::HardEvent::V_MTE2>(attenMaskBufFreeEventId1);
        }
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
        const uint32_t mte2ToVId    = attenMaskBufReadyEventId0 + bufIdx;
        const uint32_t preMte2ToVId = attenMaskBufReadyEventId0 + (bufIdx ^ 1U);

        // 回归主线风格：跳过 mask 的场景才显式 Duplicate 0；否则 AttentionmaskCopyIn 自身完成 DMA 覆盖。
        // 入口处 V→MTE2 反向同步已保证 ping/pong slot 此时可以安全覆写。
        if (IsSkipMask && IsSkipMaskForPre) {
            Duplicate(attenMaskUb, static_cast<uint8_t>(0U), maskInfo.gs1dealNum * s2BaseSize);
            return; // 全不屏蔽
        }

        if (!IsSkipMask) {
            AttentionmaskCopyIn<uint8_t, MASK_LAYOUT, true, s2BaseSize>(
                attenMaskUb, attenMaskGmInt, maskInfo, false, mte2ToVId);
        } else {
            Duplicate(attenMaskUb, static_cast<uint8_t>(0U), maskInfo.gs1dealNum * s2BaseSize);
        }

        if (!IsSkipMaskForPre) {
            const uint32_t preBufIdx = bufIdx ^ 1U;
            LocalTensor<uint8_t> attenMaskUbPre = this->attenMaskBuf[preBufIdx];
            AttentionmaskCopyIn<uint8_t, MASK_LAYOUT, true, s2BaseSize>(
                attenMaskUbPre, attenMaskGmInt, maskInfo, true, preMte2ToVId);
            MergeMask(attenMaskUb, attenMaskUbPre, maskInfo.gs1dealNum, s2BaseSize);
        }
    }

    __aicore__ inline void DealZeroActSeqLen(uint32_t bN2Cur, LocalTensor<uint8_t> initOutputScratch)
    {
        uint32_t n2Idx = bN2Cur % constInfo.n2Size;
        uint32_t bIdx = bN2Cur / constInfo.n2Size;
        if constexpr (outLayout == LayOutTypeEnum::LAYOUT_BSH) {
            OffsetCalculator<GmFormat::BSNGD, int32_t> offsetCalculator;
            offsetCalculator.Init(constInfo.bSize, constInfo.n2Size, constInfo.gSize, constInfo.s1Size,
                                  constInfo.dSizeV, *qActSeqLensParser);
            fa_static::DealActSeqLenIsZero<GmFormat::BSNGD, OUTPUT_T>(bIdx, n2Idx, offsetCalculator, attentionOutGm,
                                                                      initOutputScratch);
        } else if constexpr (outLayout == LayOutTypeEnum::LAYOUT_BNSD) {
            OffsetCalculator<GmFormat::BNGSD, int32_t> offsetCalculator;
            offsetCalculator.Init(constInfo.bSize, constInfo.n2Size, constInfo.gSize, constInfo.s1Size,
                                  constInfo.dSizeV, *qActSeqLensParser);
            fa_static::DealActSeqLenIsZero<GmFormat::BNGSD, OUTPUT_T>(bIdx, n2Idx, offsetCalculator, attentionOutGm,
                                                                      initOutputScratch);
        } else if constexpr (outLayout == LayOutTypeEnum::LAYOUT_TND) {
            OffsetCalculator<GmFormat::TNGD, int32_t, true> offsetCalculator;
            offsetCalculator.Init(constInfo.n2Size, constInfo.gSize, constInfo.dSizeV, *qActSeqLensParser);
            fa_static::DealActSeqLenIsZero<GmFormat::TNGD, OUTPUT_T>(bIdx, n2Idx, offsetCalculator, attentionOutGm,
                                                                     initOutputScratch);
        }
    }
};

// AIC/AIV 分编译占位（Mix kernel 在 AIC 侧重编译时使用）
template <typename INPUT_T, typename T, typename OUTPUT_T, LayOutTypeEnum layout = LayOutTypeEnum::None,
          LayOutTypeEnum outLayout = LayOutTypeEnum::None, S1TemplateType s1TemplateType = S1TemplateType::Aligned128,
          S2TemplateType s2TemplateType = S2TemplateType::Aligned128, DTemplateType dTemplateType = DTemplateType::Aligned128,
          DTemplateType dVTemplateType = DTemplateType::Aligned128, bool hasAtten = false, bool hasDrop = false,
          uint8_t KvLayoutType = 0, bool bmm2Write2Ub = true, bool splitD = false>
class FANoQuantGqaBlockVecDummyNd {
public:
    static constexpr bool HAS_MASK = hasAtten;
    static constexpr uint32_t stage2OutBufFreeEventId = 0U;

    using ConstInfoX = ConstInfo_t<FiaKernelType::NO_QUANT>;

    __aicore__ inline FANoQuantGqaBlockVecDummyNd(ConstInfoX &constInfo){};
};

} // namespace BaseApi
#endif // FLASH_ATTENTION_NOQUANT_GQA_BLOCK_VEC_ND_H_