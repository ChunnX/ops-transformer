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
 * \file flash_attn_block_vec_noquant_gqa_dn.h
 * \brief FANoQuantGqaBlockVecDn —— Dn 路径专用 Vec Block 模板（独立类，无 base 基类）。
 */
#ifndef FLASH_ATTENTION_NOQUANT_GQA_BLOCK_VEC_DN_H_
#define FLASH_ATTENTION_NOQUANT_GQA_BLOCK_VEC_DN_H_

#include "kernel_operator.h"

#include "../../../common/op_kernel/arch35/flash_attention_score_common_regbase.h"
#include "fa_static_init_output.h"  // 静态Tensor版 InitOutput，替代 matmul::InitOutput（其内部用 TPipe 动态分配 eventId，与静态 eventId 冲突）

using namespace AscendC;
using namespace regbaseutil;
using namespace AttentionCommon;

namespace BaseApi {

template <
    typename INPUT_T, typename T, typename OUTPUT_T, LayOutTypeEnum layout = LayOutTypeEnum::None,
    LayOutTypeEnum outLayout = LayOutTypeEnum::None, S1TemplateType s1TemplateType = S1TemplateType::Aligned128,
    S2TemplateType s2TemplateType = S2TemplateType::Aligned128, DTemplateType dTemplateType = DTemplateType::Aligned128,
    DTemplateType dVTemplateType = DTemplateType::Aligned128, bool hasAtten = false, bool hasDrop = false,
    uint8_t KvLayoutType = 0, bool bmm2Write2Ub = true, bool splitD = false>
class FANoQuantGqaBlockVecDn {
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
    // hasAtten 仅用于 HAS_MASK，供 kernel 侧 sparse S2 调度；Vec 侧不执行 FA 计算，仅保留流水同步。
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

    // ----- sinkBuf：learnable sink GM→UB -----
    static constexpr uint32_t sinkBufReadyEventId        = 4U; // MTE2_V, sinkBuf 已搬入

    // ===== 核间同步常量（避开 6、7，Set/Wait 对称）=====
    static constexpr uint64_t CROSS_CORE_SYNC_MODE = 4U;
    static constexpr uint32_t AIV0_AIV1_OFFSET = 16U;
    // 注：跨核 barrier 沿用 AscendC 内置 SyncAll() 硬同步接口（占用 flagId [11-14]），
    // 与上面的 mode 4 flagId [0-9] 完全不冲突，是 Mix 算子的标准做法。

    static constexpr ActualSeqLensMode Q_MODE = GetQActSeqMode<layout>();

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

    vec2ResGmType vec2ResGm[3];
    GlobalTensor<INPUT_T> sinkGm;

    flashdecodeGmType accumOutGm;
    flashdecodeGmType softmaxFDSumGm;
    flashdecodeGmType softmaxFDMaxGm;

    // ub
    LocalTensor<uint8_t> vec1ApiTmpBuf; // UB 布局占位（原 vec1 API tmp）
    LocalTensor<INPUT_T> stage1OutBuf[2];
    LocalTensor<T> stage2OutBuf;
    LocalTensor<float> softmaxMaxBuf[PRELOAD_N + 1];
    LocalTensor<float> softmaxSumBuf[PRELOAD_N + 1];
    LocalTensor<T> softmaxExpBuf[PRELOAD_N + 1];
    LocalTensor<T> mm2InBuf;
    LocalTensor<float> maxBrdcstBuf;
    LocalTensor<float> sumBrdcstBuf;
    LocalTensor<float> lseBuf;
    LocalTensor<INPUT_T> sinkBuf;

    const ConstInfoX &constInfo;
    int64_t bmm2SubBlockOffset = 0;
    int64_t vec2SubBlockOffset = 0;
    bool isFd = false;

    // ==================== Functions ======================
    __aicore__ inline FANoQuantGqaBlockVecDn(ConstInfoX &constInfo) : constInfo(constInfo){};

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
        (void)attenMask;
        this->attentionOutGm.SetGlobalBuffer((__gm__ OUTPUT_T *)attentionOut);

        if (constInfo.isSoftmaxLseEnable) {
            softmaxLseGm.SetGlobalBuffer((__gm__ float *)softmaxLse);
        }

        uint64_t actualLenQSize =
            (layout == LayOutTypeEnum::LAYOUT_TND) ? constInfo.cuSeqLensQSize : constInfo.seqUsedQSize;
        actualSeqLengthsGmQ.SetGlobalBuffer((__gm__ int32_t *)actualSeqQlenAddr, actualLenQSize);

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
                                       uint64_t loop)
    {
        (void)l1PTensor;
        (void)bmm1UbTensor;
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(bmm1CcId);
        int64_t stage1Offset = loop & (DB - 1);
        WaitFlag<HardEvent::MTE3_V>(stage1Offset == 0 ? stage1OutBufFreeEventId0 : stage1OutBufFreeEventId1);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(bmm1CcId);
        SetFlag<HardEvent::V_MTE3>(stage1OutBufReadyEventId0 + stage1Offset);
        WaitFlag<HardEvent::V_MTE3>(stage1OutBufReadyEventId0 + stage1Offset);
        SetFlag<HardEvent::MTE3_V>(stage1Offset == 0 ? stage1OutBufFreeEventId0 : stage1OutBufFreeEventId1);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE3>(l1PCcId);
        Vec1TailSyncOnly();
    }

    __aicore__ inline void Vec1TailSyncOnly()
    {
        if (this->isFd && constInfo.isSoftmaxLseEnable) {
            WaitFlag<HardEvent::MTE3_V>(lseBufFreeEventId);
            SetFlag<HardEvent::MTE3_V>(lseBufFreeEventId);
        } else if (constInfo.isSoftmaxLseEnable) {
            WaitFlag<HardEvent::MTE3_V>(lseBufFreeEventId);
            SetFlag<HardEvent::MTE3_V>(lseBufFreeEventId);
        }
        if (constInfo.learnableSinkFlag) {
            WaitFlag<HardEvent::MTE2_V>(sinkBufReadyEventId);
            SetFlag<HardEvent::MTE2_V>(sinkBufReadyEventId);
        }
    }

    template <typename BmmResT>
    __aicore__ inline void ProcessVec2(BmmResT &bmm2ResTensor, uint32_t bmm2CcId)
    {
        (void)bmm2ResTensor;
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(bmm2CcId);
        WaitFlag<HardEvent::MTE3_V>(stage2OutBufFreeEventId);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(bmm2CcId);
        SetFlag<HardEvent::MTE3_V>(stage2OutBufFreeEventId);
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

        // FA/FD 共享 UB 布局：保留 16K 空洞（原 attenMask ping/pong），仅作地址对齐与 FD 复用，不做 mask 搬运。
        // stage2 输出（32768 bytes）
        stage2OutBuf    = LocalTensor<uint8_t>(TPosition::VECIN, BASE + 16384U, 32768U).template ReinterpretCast<T>();
        // stage1 输出 ping/pong（各 33024 bytes）
        stage1OutBuf[0] = LocalTensor<uint8_t>(TPosition::VECIN, BASE + 49152U, 33024U).template ReinterpretCast<INPUT_T>();
        stage1OutBuf[1] = LocalTensor<uint8_t>(TPosition::VECIN, BASE + 82176U, 33024U).template ReinterpretCast<INPUT_T>();

        // Softmax 中间值：6400 bytes，紧跟 stage1[1] 末尾（BASE + 115200）
        SoftmaxInitBuffer(BASE + 115200U);

        // vec1ApiTmpBuf：UB 布局占位，保留原地址区间（Vec 侧仅同步，不再用于 VF 计算）
        constexpr bool hasVec1ApiTmpBuf = (s2BaseSize == 256U) ? (mBaseSize != 128U) : false;
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
    }

    __aicore__ inline void FreeEventID()
    {
        WaitFlag<AscendC::HardEvent::MTE3_V>(stage2OutBufFreeEventId);
        WaitFlag<AscendC::HardEvent::MTE3_V>(lseBufFreeEventId);
        WaitFlag<AscendC::HardEvent::MTE3_V>(stage1OutBufFreeEventId0);
        WaitFlag<AscendC::HardEvent::MTE3_V>(stage1OutBufFreeEventId1);
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
class FANoQuantGqaBlockVecDummyDn {
public:
    static constexpr uint32_t stage2OutBufFreeEventId = 0U;
    static constexpr bool HAS_MASK = hasAtten;
    static constexpr bool APPLY_ATTEN_MASK_IN_VF = false;

    using ConstInfoX = ConstInfo_t<FiaKernelType::NO_QUANT>;

    __aicore__ inline FANoQuantGqaBlockVecDummyDn(ConstInfoX &constInfo){};
};

} // namespace BaseApi
#endif // FLASH_ATTENTION_NOQUANT_GQA_BLOCK_VEC_DN_H_