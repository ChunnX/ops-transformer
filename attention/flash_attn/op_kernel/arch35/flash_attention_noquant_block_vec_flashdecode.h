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
 * \file flash_attention_noquant_block_vec_flashdecode.h
 * \brief
 */
#ifndef FLASH_ATTN_BLOCK_VEC_FLASHDECODE_H
#define FLASH_ATTN_BLOCK_VEC_FLASHDECODE_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"

#include "../../../common/op_kernel/arch35/infer_flash_attention_comm.h"
#include "../../../common/op_kernel/arch35/vf/vf_flash_decode.h"
#include "../../../common/op_kernel/fia_public_define.h"
#include "../../../common/op_kernel/memory_copy_arch35.h"
#include "../../../common/op_kernel/memory_copy.h"

namespace BaseApi {
struct TaskInfo {
    uint32_t bIdx;
    uint32_t n2Idx;
    uint32_t gS1Idx;
    uint32_t actualCombineLoopSize;
};

template <typename INPUT_T, typename T, typename OUTPUT_T, LayOutTypeEnum layout = LayOutTypeEnum::None,
          LayOutTypeEnum outLayout = LayOutTypeEnum::None, S1TemplateType s1TemplateType = S1TemplateType::Aligned128,
          S2TemplateType s2TemplateType = S2TemplateType::Aligned128,
          DTemplateType dTemplateType = DTemplateType::Aligned128,
          DTemplateType dVTemplateType = DTemplateType::Aligned128, bool hasAtten = false, bool hasDrop = false,
          uint8_t KvLayoutType = 0, bool useDn = false, bool bmm2Write2Ub = true, bool splitD = false>
class FiaBlockVecFlashDecode {
public:
    // =================================类型定义区=================================
    // 中间计算数据类型为float，高精度模式
    // using T = float;
    // using OUTPUT_T = OUTPUT_T;
    using SINK_T = INPUT_T;

private:
    // =================================常量区=================================
    static constexpr int64_t BYTE_BLOCK = 32UL;
    static constexpr int64_t REPEAT_BLOCK_BYTE = 256U;
    static constexpr uint64_t SYNC_LSE_MAX_SUM_BUF1_FLAG = 8;
    static constexpr uint64_t SYNC_LSE_MAX_SUM_BUF2_FLAG = 9;
    static constexpr uint64_t SYNC_MM2RES_BUF1_FLAG = 10;
    static constexpr uint64_t SYNC_MM2RES_BUF2_FLAG = 11;
    static constexpr uint64_t SYNC_FDOUTPUT_BUF_FLAG = 2;
    static constexpr uint64_t SYNC_LSEOUTPUT_BUF_FLAG = 4;
    static constexpr uint64_t SYNC_SINK_BUF1_FLAG = 12;
    static constexpr uint64_t SYNC_SINK_BUF2_FLAG = 13;

    static constexpr uint32_t BUFFER_SIZE_BYTE_32B = 32;
    static constexpr uint32_t BUFFER_SIZE_BYTE_64B = 64;
    static constexpr uint32_t BUFFER_SIZE_BYTE_256B = 256;
    static constexpr uint32_t BUFFER_SIZE_BYTE_512B = 512;
    static constexpr uint32_t BUFFER_SIZE_BYTE_1K = 1024;
    static constexpr uint32_t BUFFER_SIZE_BYTE_2K = 2048;
    static constexpr uint32_t BUFFER_SIZE_BYTE_4K = 4096;
    static constexpr uint32_t BUFFER_SIZE_BYTE_8K = 8192;
    static constexpr uint32_t BUFFER_SIZE_BYTE_16K = 16384;

    static constexpr uint32_t BLOCK_ELEMENT_NUM = BYTE_BLOCK / sizeof(T); // 32/4=8
    static constexpr uint32_t FP32_BLOCK_ELEMENT_NUM = BYTE_BLOCK / sizeof(float);
    static constexpr uint32_t FP32_REPEAT_ELEMENT_NUM = REPEAT_BLOCK_BYTE / sizeof(float);

    static constexpr float FLOAT_INF = 3e+99;
    uint32_t preLoadNum = 2U;
    uint32_t dSizeV_Align;
    static constexpr bool attenMaskFlag = hasAtten;
    using ConstInfoX = ConstInfo_t<FiaKernelType::NO_QUANT>;
    // 基本块大小
    static constexpr uint32_t s1BaseSize = (uint32_t)s1TemplateType;

protected:
    GlobalTensor<float> lseSumFdGm;
    GlobalTensor<float> lseMaxFdGm;
    GlobalTensor<float> accumOutGm;
    GlobalTensor<OUTPUT_T> attentionOutGm;
    GlobalTensor<float> softmaxLseGm;
    GlobalTensor<SINK_T> sinkGm;

    static constexpr UbFormat UB_FORMAT = GetOutUbFormat<layout>();
    static constexpr bool isS1G = (UB_FORMAT == UbFormat::S1G);
    static constexpr bool isPa = KvLayoutType > 0;
    T scale2Value = 0;
    T offset2Value = 0;

    // =======================获取实际Act_S===========================
    static constexpr ActualSeqLensMode Q_MODE = GetQActSeqMode<layout>();
    static constexpr ActualSeqLensMode KV_MODE = GetKvActSeqMode<layout, isPa>();
    // tensorlist
    __gm__ uint8_t *keyPtr = nullptr;

    using QSeqParserType =
        typename std::conditional<(layout == LayOutTypeEnum::LAYOUT_TND || layout == LayOutTypeEnum::LAYOUT_NTD),
                                  ActualSeqLensParser<Q_MODE, int32_t, true>,
                                  ActualSeqLensParser<Q_MODE, int32_t>>::type;

    using KvSeqParserType =
        typename std::conditional<(layout == LayOutTypeEnum::LAYOUT_TND || layout == LayOutTypeEnum::LAYOUT_NTD),
                                  ActualSeqLensParser<KV_MODE, int32_t, true>,
                                  ActualSeqLensParser<KV_MODE, int32_t>>::type;

    QSeqParserType *qActSeqLensParser = nullptr;
    KvSeqParserType *kvActSeqLensParser = nullptr;

    int64_t preTokensPerBatch = 0;
    int64_t nextTokensPerBatch = 0;

    static constexpr T BOOL_ATTEN_MASK_SCALAR_VALUE = -1000000000000.0; // 用于mask为bool类型
    uint32_t negativeIntScalar = *((uint32_t *)&BOOL_ATTEN_MASK_SCALAR_VALUE);
    bool learnableSinkFlag = false;

    uint64_t actSeqLensKv = 0;
    uint64_t actSeqLensQ = 0;
    // ================================类成员变量====================================
    // aic、aiv核信息
    uint32_t blockIdx = 0U;
    const ConstInfoX &constInfo;
    TaskInfo taskInfo{};

private:
    // ================================FD Local Buffer区====================================
    LocalTensor<T> fdSumBuf1;    // 1.5k: 16*24*4
    LocalTensor<T> fdSumBuf2;    // 1.5k: 16*24*4
    LocalTensor<T> fdMaxBuf1;    // 1.5k: 16*24*4
    LocalTensor<T> fdMaxBuf2;    // 1.5k: 16*24*4
    LocalTensor<T> fdLseExpBuf;  // 1.5k: 16*24*4
    LocalTensor<T> fdMm2ResBuf1; // 32k: 16*512*4
    LocalTensor<T> fdMm2ResBuf2; // 32k: 16*512*4
    LocalTensor<T> fdReduceBuf;  // 32k: 16*512*4
    LocalTensor<OUTPUT_T> fdOutputBuf;  // 32k: 16*512*4
    LocalTensor<T> fdSinkCopyInBuf;
    LocalTensor<T> fdSinkValueBuf;
    LocalTensor<T> fdSinkExpBuf;
    LocalTensor<T> fdSinkTmpBuf;

    LocalTensor<T> fdLseMaxUbBuf1;
    LocalTensor<T> fdLseMaxUbBuf2;
    LocalTensor<T> fdLseUbBuf;

public:
    __aicore__ inline FiaBlockVecFlashDecode(ConstInfoX &constInfo) : constInfo(constInfo){};

    template <typename U> // 避免重名用U
    __aicore__ inline U Align(U num, U rnd)
    {
        return (((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd) * (rnd)));
    }

    __aicore__ inline void InitGlobalTensor(GlobalTensor<float> lseMaxFdGm, GlobalTensor<float> lseSumFdGm,
                                            GlobalTensor<float> accumOutGm, GlobalTensor<OUTPUT_T> attentionOutGm,
                                            __gm__ uint8_t *key)
    {
        this->lseMaxFdGm = lseMaxFdGm;
        this->lseSumFdGm = lseSumFdGm;
        this->accumOutGm = accumOutGm;
        this->attentionOutGm = attentionOutGm;

        this->keyPtr = key;
    }

    __aicore__ inline void SetCuSeqLensParsers(QSeqParserType &qParser, KvSeqParserType &kvParser)
    {
        this->qActSeqLensParser = &qParser;
        this->kvActSeqLensParser = &kvParser;
    }

    __aicore__ inline void InitSoftmaxLseGm(GlobalTensor<float> softmaxLseGm)
    {
        this->softmaxLseGm = softmaxLseGm;
    }
    __aicore__ inline void InitLearnableSinkGm(GlobalTensor<SINK_T> learnableSink)
    {
        learnableSinkFlag = true;
        this->sinkGm = learnableSink;
    }
    __aicore__ inline void InitParams()
    {
        this->dSizeV_Align = this->Align(constInfo.dSizeV, FP32_REPEAT_ELEMENT_NUM);
    }
    __aicore__ inline void InitDecodeParams()
    {
        this->blockIdx = GetBlockIdx();
    }
    __aicore__ inline void InitBuffers()
    {
        if ASCEND_IS_AIV {
            // 与 FA block 共享UB布局：bmm1/bmm2 区域在前，FD 业务缓冲区紧随其后。
            // 使用 LocalTensor 构造函数直接指定绝对字节偏移，实现内存完全自主管理，
            // 无需 LocalMemAllocator 线性分配器。
            constexpr uint32_t mm1Sz = s1BaseSize / 2U * (uint32_t)s2TemplateType * sizeof(T);
            constexpr uint32_t mm2Sz = bmm2Write2Ub ?
                (s1BaseSize / 2U * (uint32_t)dVTemplateType * sizeof(T)) : 0U;
            // mm2InBuf 区域：仅 !bmm2Write2Ub 时存在（32768 bytes）
            constexpr uint32_t mm2InBufSz = bmm2Write2Ub ? 0U : 32768U;

            // FD 业务区起始字节偏移（跳过 bmm1/bmm2/mm2In 区域）
            constexpr uint32_t BASE = mm1Sz * 2U + mm2Sz * 2U + mm2InBufSz;

            // 各共享区块的绝对字节偏移（与原 LocalMemAllocator 分配顺序完全一致）
            // SharedBuffer2[0]：attenMaskBuf[0](FA) / fdLseMaxUb(FD)，8192 bytes
            constexpr uint32_t OFF_BUF2A = BASE;
            // Skip attenMaskBuf[1]：8192 bytes
            // SharedBuffer3：stage2OutBuf(FA) / fdSum/Max/LseExp(FD)，32768 bytes
            constexpr uint32_t OFF_BUF3  = BASE + BUFFER_SIZE_BYTE_8K + BUFFER_SIZE_BYTE_8K;
            // SharedBuffer1[0]：stage1OutBuf[0](FA) / fdMm2Res(FD)，33024 bytes
            constexpr uint32_t OFF_BUF1A = OFF_BUF3 + 32768U;
            // SharedBuffer1[1]：stage1OutBuf[1](FA) / fdReduce/Output(FD)，33024 bytes
            constexpr uint32_t OFF_BUF1B = OFF_BUF1A + 33024U;

            // sharedBuf1a 内：fdMm2ResBuf1（前16K）、fdMm2ResBuf2（后16K）
            fdMm2ResBuf1 = LocalTensor<uint8_t>(TPosition::VECIN, OFF_BUF1A,
                               BUFFER_SIZE_BYTE_16K).template ReinterpretCast<T>();
            fdMm2ResBuf2 = LocalTensor<uint8_t>(TPosition::VECIN, OFF_BUF1A + BUFFER_SIZE_BYTE_16K,
                               BUFFER_SIZE_BYTE_16K).template ReinterpretCast<T>();
            // sharedBuf1b 内：fdReduceBuf（前16K）、fdOutputBuf（后16K）
            fdReduceBuf  = LocalTensor<uint8_t>(TPosition::VECIN, OFF_BUF1B,
                               BUFFER_SIZE_BYTE_16K).template ReinterpretCast<T>();
            fdOutputBuf  = LocalTensor<uint8_t>(TPosition::VECIN, OFF_BUF1B + BUFFER_SIZE_BYTE_16K,
                               BUFFER_SIZE_BYTE_16K).template ReinterpretCast<OUTPUT_T>();

            // sharedBuf3 内：5 个 6144-byte 槽（fdSum1/2, fdMax1/2, fdLseExp）
            constexpr uint32_t STRIDE = BUFFER_SIZE_BYTE_4K + BUFFER_SIZE_BYTE_2K; // 6144 bytes
            fdSumBuf1   = LocalTensor<uint8_t>(TPosition::VECIN, OFF_BUF3,
                              STRIDE).template ReinterpretCast<T>();
            fdSumBuf2   = LocalTensor<uint8_t>(TPosition::VECIN, OFF_BUF3 + STRIDE,
                              STRIDE).template ReinterpretCast<T>();
            fdMaxBuf1   = LocalTensor<uint8_t>(TPosition::VECIN, OFF_BUF3 + 2U * STRIDE,
                              STRIDE).template ReinterpretCast<T>();
            fdMaxBuf2   = LocalTensor<uint8_t>(TPosition::VECIN, OFF_BUF3 + 3U * STRIDE,
                              STRIDE).template ReinterpretCast<T>();
            fdLseExpBuf = LocalTensor<uint8_t>(TPosition::VECIN, OFF_BUF3 + 4U * STRIDE,
                              STRIDE).template ReinterpretCast<T>();

            // sharedBuf2a 内：fdLseMaxUbBuf1/2、fdLseUbBuf，各 256 bytes
            fdLseMaxUbBuf1 = LocalTensor<uint8_t>(TPosition::VECIN, OFF_BUF2A,
                                 BUFFER_SIZE_BYTE_256B).template ReinterpretCast<T>();
            fdLseMaxUbBuf2 = LocalTensor<uint8_t>(TPosition::VECIN, OFF_BUF2A + BUFFER_SIZE_BYTE_256B,
                                 BUFFER_SIZE_BYTE_256B).template ReinterpretCast<T>();
            fdLseUbBuf     = LocalTensor<uint8_t>(TPosition::VECIN, OFF_BUF2A + 2U * BUFFER_SIZE_BYTE_256B,
                                 BUFFER_SIZE_BYTE_256B).template ReinterpretCast<T>();

            // Sink 缓冲区（紧接 fdLseUbBuf）。即使 learnableSinkFlag=false，
            // ComputeScaleValue 仍会将 fdSinkExpBuf 作为参数传入 __simd_vf__ 函数；
            // 该 VF 函数虽然在 `for(i=0; i<learnableSinkFlag; ++i)` 中保护了 lseSink 的访问，
            // 但编译器/调度器可能在循环外提前准备地址寄存器，未初始化的 LocalTensor
            // (address_==0) 会让 VEC 取到非法地址 → aicore subErrType 0x4 / errcode 340
            // (The address for VEC to access UB is not aligned)。因此无条件按编译期常量
            // 初始化 sink 4 槽位，保证地址在硬件视角下始终合法（位于 UB 内、32B 对齐）。
            constexpr uint32_t SINK_BASE = OFF_BUF2A + 3U * BUFFER_SIZE_BYTE_256B; // OFF_BUF2A + 768
            fdSinkCopyInBuf = LocalTensor<uint8_t>(TPosition::VECIN, SINK_BASE,
                                  BUFFER_SIZE_BYTE_2K).template ReinterpretCast<T>();
            fdSinkValueBuf  = LocalTensor<uint8_t>(TPosition::VECIN, SINK_BASE + BUFFER_SIZE_BYTE_2K,
                                  BUFFER_SIZE_BYTE_2K).template ReinterpretCast<T>();
            fdSinkExpBuf    = LocalTensor<uint8_t>(TPosition::VECIN,
                                  SINK_BASE + 2U * BUFFER_SIZE_BYTE_2K,
                                  BUFFER_SIZE_BYTE_256B).template ReinterpretCast<T>();
            fdSinkTmpBuf    = LocalTensor<uint8_t>(TPosition::VECIN,
                                  SINK_BASE + 2U * BUFFER_SIZE_BYTE_2K + BUFFER_SIZE_BYTE_256B,
                                  BUFFER_SIZE_BYTE_256B).template ReinterpretCast<T>();
        }
    }
    __aicore__ inline void AllocEventID()
    {
        SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_LSE_MAX_SUM_BUF1_FLAG);
        SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_LSE_MAX_SUM_BUF2_FLAG);
        SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_MM2RES_BUF1_FLAG);
        SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_MM2RES_BUF2_FLAG);
        SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_FDOUTPUT_BUF_FLAG);
        SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_LSEOUTPUT_BUF_FLAG);
        SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_SINK_BUF1_FLAG);
        SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_SINK_BUF2_FLAG);
    }
    __aicore__ inline void FreeEventID()
    {
        WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_LSE_MAX_SUM_BUF1_FLAG);
        WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_LSE_MAX_SUM_BUF2_FLAG);
        WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_MM2RES_BUF1_FLAG);
        WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_MM2RES_BUF2_FLAG);
        WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_FDOUTPUT_BUF_FLAG);
        WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_LSEOUTPUT_BUF_FLAG);
        WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_SINK_BUF1_FLAG);
        WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_SINK_BUF2_FLAG);
    }

protected:
    __aicore__ inline void CopyAccumOutIn(LocalTensor<T> &accumOutLocal, uint32_t splitKVIndex, uint32_t startRow,
                                          uint32_t dealRowCount)
    {
        DataCopyExtParams copyInParams;
        DataCopyPadExtParams<T> copyInPadParams;
        copyInParams.blockCount = dealRowCount;
        copyInParams.blockLen = constInfo.dSizeV * sizeof(T);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = (this->dSizeV_Align - constInfo.dSizeV) / BLOCK_ELEMENT_NUM;

        copyInPadParams.isPad = true;
        copyInPadParams.leftPadding = 0;
        copyInPadParams.rightPadding = (this->dSizeV_Align - constInfo.dSizeV) % BLOCK_ELEMENT_NUM;
        copyInPadParams.paddingValue = 0;
        uint64_t combineAccumOutOffset = startRow * constInfo.dSizeV +                 // taskoffset + g轴offset
                                         splitKVIndex * s1BaseSize * constInfo.dSizeV; // 份数offset

        DataCopyPad(accumOutLocal, accumOutGm[combineAccumOutOffset], copyInParams, copyInPadParams);
    }
    __aicore__ inline void CopyLseIn(uint32_t startRow, uint32_t dealRowCount, uint64_t baseOffset, uint32_t cntM)
    {
        LocalTensor<T> lseSum = (cntM & 1) == 0 ? fdSumBuf1 : fdSumBuf2;
        LocalTensor<T> lseMax = (cntM & 1) == 0 ? fdMaxBuf1 : fdMaxBuf2;

        uint64_t combineLseOffset = (baseOffset + startRow) * FP32_BLOCK_ELEMENT_NUM;
        uint64_t combineLoopOffset = s1BaseSize * FP32_BLOCK_ELEMENT_NUM;
        uint64_t dealRowCountAlign = dealRowCount * FP32_BLOCK_ELEMENT_NUM;

        for (uint32_t i = 0; i < taskInfo.actualCombineLoopSize; i++) {
            DataCopy(lseSum[i * dealRowCountAlign], lseSumFdGm[combineLseOffset + i * combineLoopOffset],
                     dealRowCountAlign); // 份数offset

            DataCopy(lseMax[i * dealRowCountAlign], lseMaxFdGm[combineLseOffset + i * combineLoopOffset],
                     dealRowCountAlign);
        }
    }
    __aicore__ inline void ComputeScaleValue(LocalTensor<T> &lseExp, uint32_t dealRowCount,
                                             uint32_t actualCombineLoopSize, uint32_t cntM, uint32_t startRow)
    {
        LocalTensor<T> lseSum = (cntM & 1) == 0 ? fdSumBuf1 : fdSumBuf2;
        LocalTensor<T> lseMax = (cntM & 1) == 0 ? fdMaxBuf1 : fdMaxBuf2;
        if (unlikely(learnableSinkFlag)) {
            SinkMax(startRow, dealRowCount);
        }
        LocalTensor<T> lseMaxUb = (cntM & 1) == 0 ? fdLseMaxUbBuf1 : fdLseMaxUbBuf2;

        LocalTensor<T> sinkExpBuf = fdSinkExpBuf;
        LocalTensor<T> maxLseUb = fdLseUbBuf;
        ComputeScaleValue_VF_FD(sinkExpBuf, lseMax, lseSum, lseExp, maxLseUb, lseMaxUb, dealRowCount,
                                actualCombineLoopSize, constInfo.isSoftmaxLseEnable, learnableSinkFlag);
    }

    __aicore__ inline void Bmm2DataCopyOutTrans(LocalTensor<OUTPUT_T> &attenOutUb, uint32_t startRow,
                                                uint32_t dealRowCount, uint32_t columnCount)
    {
        FaUbTensor<OUTPUT_T> ubTensor{
            .tensor = attenOutUb,
            .rowCount = dealRowCount,
            .colCount = columnCount,
        };
        GmCoord gmCoord{.bIdx = taskInfo.bIdx,
                        .n2Idx = taskInfo.n2Idx,
                        .gS1Idx = taskInfo.gS1Idx + startRow,
                        .dIdx = 0,
                        .gS1DealSize = dealRowCount,
                        .dDealSize = (uint32_t)constInfo.dSizeV};

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
            outGmTensor.offsetCalculator.Init(constInfo.n2Size, constInfo.gSize, constInfo.dSizeV, *qActSeqLensParser);
            CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, GetOutUbFormat<layout>()> copyAttenOutUbToGm;
            copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
        } else if constexpr (outLayout == LayOutTypeEnum::LAYOUT_NTD) {
            constexpr GmFormat OUT_FORMAT = GmFormat::NGTD;
            FaGmTensor<OUTPUT_T, OUT_FORMAT, int32_t, true> outGmTensor;
            outGmTensor.gmTensor = attentionOutGm;
            outGmTensor.offsetCalculator.Init(constInfo.n2Size, constInfo.gSize, constInfo.dSizeV, *qActSeqLensParser);
            CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, GetOutUbFormat<layout>()> copyAttenOutUbToGm;
            copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
        }
    }
    __aicore__ inline void ReduceFinalRes(LocalTensor<T> &reduceOut, LocalTensor<T> &mm2Res, LocalTensor<T> &lseLocal,
                                          uint32_t cntKV, uint32_t dealRowCount)
    {
        uint64_t dSizeV_Align = (uint64_t)this->dSizeV_Align;
        ReduceFinalRes_VF<T>(reduceOut, lseLocal, mm2Res, dealRowCount, dSizeV_Align, cntKV);
    }
    __aicore__ inline void CopyFinalResOut(LocalTensor<T> &accumOutLocal, uint32_t startRow, uint32_t dealRowCount,
                                           uint32_t cntM)
    {
        LocalTensor<OUTPUT_T> tmpBmm2ResCastTensor = fdOutputBuf;
        AscendC::PipeBarrier<PIPE_V>();
        DealInvalidRows(accumOutLocal, startRow, dealRowCount, this->dSizeV_Align);
        DealInvalidMaskRows(accumOutLocal, startRow, dealRowCount, this->dSizeV_Align, cntM);
        WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_FDOUTPUT_BUF_FLAG);
        uint32_t shapeArray[] = {dealRowCount, (uint32_t)constInfo.dSizeV};
        tmpBmm2ResCastTensor.SetShapeInfo(ShapeInfo(2, shapeArray, DataFormat::ND));
        if constexpr (IsSameType<OUTPUT_T, bfloat16_t>::value) { // bf16 采取四舍六入五成双模式
            Cast(tmpBmm2ResCastTensor, accumOutLocal, AscendC::RoundMode::CAST_RINT, dealRowCount * this->dSizeV_Align);
        } else {
            Cast(tmpBmm2ResCastTensor, accumOutLocal, AscendC::RoundMode::CAST_ROUND,
                 dealRowCount * this->dSizeV_Align);
        }
        SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_FDOUTPUT_BUF_FLAG);
        WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_FDOUTPUT_BUF_FLAG);
        Bmm2DataCopyOutTrans(tmpBmm2ResCastTensor, startRow, dealRowCount, this->dSizeV_Align);
        SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_FDOUTPUT_BUF_FLAG);
    }
    __aicore__ inline void CalcPreNextTokens()
    {
        actSeqLensQ = qActSeqLensParser->GetActualSeqLength(taskInfo.bIdx);
        if (constInfo.cuSeqLensKVSize == 0 && constInfo.seqUsedKvSize == 0 && !constInfo.isKvContinuous) {
            actSeqLensKv = SeqLenFromTensorList<layout>(keyPtr, taskInfo.bIdx);
        } else {
            actSeqLensKv = kvActSeqLensParser->GetActualSeqLength(taskInfo.bIdx);
        }
        int64_t safePreToken = constInfo.preTokens;
        int64_t safeNextToken = constInfo.nextTokens;

        fa_base_vector::GetSafeActToken(actSeqLensQ, actSeqLensKv, safePreToken, safeNextToken, constInfo.sparseMode);

        if (constInfo.sparseMode == BAND) {
            preTokensPerBatch = safePreToken;
            nextTokensPerBatch = actSeqLensKv - actSeqLensQ + safeNextToken;
        } else if ((constInfo.sparseMode == DEFAULT_MASK) && attenMaskFlag) {
            nextTokensPerBatch = safeNextToken;
            preTokensPerBatch = actSeqLensKv - actSeqLensQ + safePreToken;
        } else {
            nextTokensPerBatch = actSeqLensKv - actSeqLensQ;
            preTokensPerBatch = 0;
        }
    }
    __aicore__ inline void CopySinkIn(uint32_t cntM)
    {

    }
    __aicore__ inline void SinkMax(uint32_t startRow, uint32_t dealRowCount)
    {
        constexpr GmFormat Q_FORMAT = GetQueryGmFormat<layout>();
        int64_t gIdx = 0;
        LocalTensor<T> sinkBrcbBuf = fdSinkValueBuf;
        LocalTensor<T> sinkExpBuf = fdSinkExpBuf;

        for (int64_t row = 0; row < dealRowCount; ++row) {
            if constexpr ((Q_FORMAT == GmFormat::BSNGD) || (Q_FORMAT == GmFormat::TNGD)) { // 内存按照S1G排布
                gIdx = (taskInfo.gS1Idx + startRow + row) % constInfo.gSize;
            } else if constexpr ((Q_FORMAT == GmFormat::BNGSD) || (Q_FORMAT == GmFormat::NGTD)) { // 内存按照GS1排布
                int64_t actS1Size = qActSeqLensParser->GetActualSeqLength(taskInfo.bIdx);
                gIdx = (taskInfo.gS1Idx + startRow + row) / actS1Size;
            }
            DataCopy(sinkExpBuf[row * BLOCK_ELEMENT_NUM], sinkBrcbBuf[gIdx * BLOCK_ELEMENT_NUM], BLOCK_ELEMENT_NUM);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    template <typename UBOUT_T>
    __aicore__ inline void DealInvalidRows(LocalTensor<UBOUT_T> &attenOutUb, uint32_t startRow, uint32_t dealRowCount,
                                           uint32_t columnCount)
    {
        if (!attenMaskFlag) {
            return;
        }

        if (constInfo.sparseMode == ALL_MASK || constInfo.sparseMode == LEFT_UP_CAUSAL) {
            return;
        }

        fa_base_vector::InvalidRowParams params{
            .actS1Size = actSeqLensQ,
            .gSize = static_cast<uint64_t>(constInfo.gSize),
            .gS1Idx = taskInfo.gS1Idx + startRow,
            .dealRowCount = dealRowCount,
            .columnCount = columnCount,
            .preTokensPerBatch = preTokensPerBatch,
            .nextTokensPerBatch = nextTokensPerBatch,
        };

        fa_base_vector::InvalidRows<UBOUT_T, fa_base_vector::GeInputUbFormat<layout>()> invalidRows;
        invalidRows(attenOutUb, params);
    }

    template <typename UBOUT_T>
    __aicore__ inline void DealInvalidMaskRows(LocalTensor<UBOUT_T> &attenOutUb, uint32_t startRow,
                                               uint32_t dealRowCount, uint32_t columnCount, uint32_t cntM)
    {
        if (!attenMaskFlag) {
            return;
        }
        if (constInfo.sparseMode != DEFAULT_MASK && constInfo.sparseMode != ALL_MASK) {
            return;
        }
        LocalTensor<T> lseMaxUb = (cntM & 1) == 0 ? fdLseMaxUbBuf1 : fdLseMaxUbBuf2;

        fa_base_vector::InvalidMaskRows<UBOUT_T, T, true>(0, dealRowCount, columnCount, lseMaxUb, negativeIntScalar,
                                                          attenOutUb);
    }

public:
    __aicore__ inline void FlashDecode(FDparamsX &fd)
    {
        if (!fd.fdCoreEnable) {
            return;
        }
        if (unlikely(fd.fdS2SplitNum == 0)) {
            // metadata 未携带有效 split 数时不做 combine（与 GetFDSectionInfo 未赋值时行为一致）
            return;
        }
        uint32_t fdBalanceMBaseSize = 8U;
        uint32_t fdBalanceMSplitNum = (fd.mLen + fdBalanceMBaseSize - 1) / fdBalanceMBaseSize;
        uint32_t fdBalanceMTailSize =
            (fd.mLen % fdBalanceMBaseSize == 0) ? fdBalanceMBaseSize : fd.mLen % fdBalanceMBaseSize;

        uint32_t reduceGlobaLoop = 0;
        uint32_t reduceMLoop = 0;

        uint32_t tmpFdS1gOuterMStart = 0;
        uint32_t tmpFdS1gOuterMEnd = fdBalanceMSplitNum - 1;
        taskInfo.bIdx = fd.fdBN2Idx / constInfo.n2Size;
        taskInfo.n2Idx = fd.fdBN2Idx % constInfo.n2Size;
        taskInfo.gS1Idx = fd.fdMIdx * s1BaseSize;
        taskInfo.actualCombineLoopSize = fd.fdS2SplitNum; // 当前规约任务 kv 方向有几份（metadata FA_FD_WORKSPACE_NUM）
        // 稀疏 mask 场景下单个 combine 任务的实际 split 数可能为 1（小于 preLoadNum=2）。
        // 预取必须按 actualCombineLoopSize 截断，否则预取 2 份、主循环只消费 1 份，
        // SYNC_MM2RES ping/pong 的 V_MTE2/MTE2_V Set/Wait 不配对，FreeEventID 永久阻塞 → AIV 卡死，
        // AIC 在下一 section / 下一轮 bmm CrossCoreWaitFlag 处全部 timeout(subErrType 0x4)。
        // 注释掉 GetFDSectionInfo 中 fdS2SplitNum 赋值后该值为 0、combine 被跳过，故看似“不再 aicore”。
        const uint32_t effPreLoadNum =
            preLoadNum < taskInfo.actualCombineLoopSize ? preLoadNum : taskInfo.actualCombineLoopSize;
        uint64_t combineTaskPrefixSum = fd.fdWorkspaceIdx;
        uint64_t taskOffset = combineTaskPrefixSum * s1BaseSize;

        for (uint32_t fdS1gOuterMIdx = tmpFdS1gOuterMStart; fdS1gOuterMIdx <= tmpFdS1gOuterMEnd;
             fdS1gOuterMIdx++) { // 左闭右闭
            uint32_t actualGSplitSize = fdBalanceMBaseSize;
            if (fdS1gOuterMIdx == fdBalanceMSplitNum - 1) {
                actualGSplitSize = fdBalanceMTailSize;
            }
            uint32_t startRow = fd.mStart + fdS1gOuterMIdx * fdBalanceMBaseSize;

            LocalTensor<T> lseExp = fdLseExpBuf;
            LocalTensor<T> reduceOut = fdReduceBuf;
            WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_LSE_MAX_SUM_BUF1_FLAG + (reduceMLoop & 1));
            CopyLseIn(startRow, actualGSplitSize, taskOffset, reduceMLoop);
            SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_LSE_MAX_SUM_BUF1_FLAG + (reduceMLoop & 1));
            WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_LSE_MAX_SUM_BUF1_FLAG + (reduceMLoop & 1));
            if (unlikely(learnableSinkFlag)) {
                CopySinkIn(reduceMLoop);
            }
            for (uint32_t preLoadIdx = 0; preLoadIdx < effPreLoadNum; preLoadIdx++) {
                LocalTensor<T> mm2Res = (((reduceGlobaLoop + preLoadIdx) & 1) == 0) ? fdMm2ResBuf1 : fdMm2ResBuf2;
                WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_MM2RES_BUF1_FLAG + ((reduceGlobaLoop + preLoadIdx) & 1));
                CopyAccumOutIn(mm2Res, preLoadIdx, taskOffset + startRow, actualGSplitSize);
                SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_MM2RES_BUF1_FLAG + ((reduceGlobaLoop + preLoadIdx) & 1));
            }
            ComputeScaleValue(lseExp, actualGSplitSize, taskInfo.actualCombineLoopSize, reduceMLoop, startRow);
            CalcPreNextTokens();
            if (constInfo.isSoftmaxLseEnable) {
                LocalTensor<T> maxLseUb = fdLseUbBuf;
                SetFlag<HardEvent::V_MTE3>(SYNC_LSEOUTPUT_BUF_FLAG);
                WaitFlag<HardEvent::V_MTE3>(SYNC_LSEOUTPUT_BUF_FLAG);
                uint32_t mOffset = taskInfo.gS1Idx + startRow;
                if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
                    uint32_t prefixBS1 = qActSeqLensParser->GetTBase(taskInfo.bIdx);
                    uint64_t bN2Offset = taskInfo.n2Idx * constInfo.gSize * constInfo.t1Size + prefixBS1;
                    DataCopySoftmaxLseTNDtoNTArch35<T, ConstInfoX>(softmaxLseGm, maxLseUb, bN2Offset, mOffset,
                                                                   actualGSplitSize, constInfo);
                } else if constexpr (layout == LayOutTypeEnum::LAYOUT_NTD) {
                    uint32_t prefixBS1 = qActSeqLensParser->GetTBase(taskInfo.bIdx);
                    uint32_t s1Size = qActSeqLensParser->GetActualSeqLength(taskInfo.bIdx);
                    uint64_t bN2Offset =
                        prefixBS1 * constInfo.gSize * constInfo.n2Size + taskInfo.n2Idx * constInfo.gSize;
                    DataCopySoftmaxLseNTDArch35<T, ConstInfoX>(softmaxLseGm, maxLseUb, bN2Offset, mOffset,
                                                               actualGSplitSize, constInfo, s1Size);
                } else if constexpr (layout == LayOutTypeEnum::LAYOUT_BSH) {
                    uint64_t bN2Offset = taskInfo.bIdx * constInfo.gSize * constInfo.n2Size * constInfo.s1Size +
                                         taskInfo.n2Idx * constInfo.gSize * constInfo.s1Size;
                    uint64_t qActSeqLens = qActSeqLensParser->GetActualSeqLength(taskInfo.bIdx);
                    DataCopySoftmaxLseBSNDArch35<T, ConstInfoX>(softmaxLseGm, maxLseUb, bN2Offset, mOffset,
                                                                actualGSplitSize, constInfo);
                } else { // BNSD
                    uint64_t bN2Offset = taskInfo.bIdx * constInfo.gSize * constInfo.n2Size * constInfo.s1Size +
                                         taskInfo.n2Idx * constInfo.gSize * constInfo.s1Size;
                    uint64_t qActSeqLens = qActSeqLensParser->GetActualSeqLength(taskInfo.bIdx);
                    DataCopySoftmaxLseBNSDArch35<T, ConstInfoX>(softmaxLseGm, maxLseUb, bN2Offset, mOffset,
                                                                actualGSplitSize, constInfo, qActSeqLens);
                }
            }
            SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_LSE_MAX_SUM_BUF1_FLAG + (reduceMLoop & 1));

            for (uint32_t i = 0; i < taskInfo.actualCombineLoopSize; i++) {
                LocalTensor<T> mm2Res = (reduceGlobaLoop & 1) == 0 ? fdMm2ResBuf1 : fdMm2ResBuf2;
                if (i >= effPreLoadNum) {
                    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_MM2RES_BUF1_FLAG + (reduceGlobaLoop & 1));
                    CopyAccumOutIn(mm2Res, i, taskOffset + startRow, actualGSplitSize);
                    SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_MM2RES_BUF1_FLAG + (reduceGlobaLoop & 1));
                }
                WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_MM2RES_BUF1_FLAG + (reduceGlobaLoop & 1));
                ReduceFinalRes(reduceOut, mm2Res, lseExp, i, actualGSplitSize);
                SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_MM2RES_BUF1_FLAG + (reduceGlobaLoop & 1));
                reduceGlobaLoop += 1;
            }
            CopyFinalResOut(reduceOut, startRow, actualGSplitSize, reduceMLoop);
            reduceMLoop += 1;
        }
    }
};

template <typename INPUT_T, typename T, typename OUTPUT_T, LayOutTypeEnum layout = LayOutTypeEnum::None,
          LayOutTypeEnum outLayout = LayOutTypeEnum::None, S1TemplateType s1TemplateType = S1TemplateType::Aligned128,
          S2TemplateType s2TemplateType = S2TemplateType::Aligned128,
          DTemplateType dTemplateType = DTemplateType::Aligned128,
          DTemplateType dVTemplateType = DTemplateType::Aligned128, bool hasAtten = false, bool hasDrop = false,
          uint8_t KvLayoutType = 0, bool useDn = false, bool bmm2Write2Ub = true, bool splitD = false>
class FiaBlockVecFlashDecodeDummy {
public:
    // =================================类型定义区=================================
    // 中间计算数据类型为float，高精度模式
    using SINK_T = bfloat16_t;
    using ConstInfoX = ConstInfo_t<FiaKernelType::NO_QUANT>;
    __aicore__ inline FiaBlockVecFlashDecodeDummy(){};
    __aicore__ inline void InitGlobalTensor(GlobalTensor<float> lseMaxFdGm, GlobalTensor<float> lseSumFdGm,
                                            GlobalTensor<float> accumOutGm, GlobalTensor<OUTPUT_T> attentionOutGm) {};
    __aicore__ inline void InitParams() {};
    __aicore__ inline void InitSoftmaxLseGm(GlobalTensor<float> softmaxLseGm) {};
    __aicore__ inline void InitLearnableSinkGm(GlobalTensor<SINK_T> learnableSink) {};
    __aicore__ inline void InitDecodeParams() {};
    __aicore__ inline void InitBuffers() {};
    __aicore__ inline void AllocEventID() {};
    __aicore__ inline void FreeEventID() {};
    __aicore__ inline void FlashDecode(FDparamsX &fd) {};

protected:
    __aicore__ inline void CopyAccumOutIn(LocalTensor<T> &accumOutLocal, uint32_t splitKVIndex, uint32_t startRow,
                                          uint32_t dealRowCount) {};
    __aicore__ inline void CopyLseIn(uint32_t startRow, uint32_t dealRowCount, uint64_t baseOffset, uint32_t cntM) {};
    __aicore__ inline void ComputeScaleValue(LocalTensor<T> &lseExp, uint32_t dealRowCount,
                                             uint32_t actualCombineLoopSize, uint32_t cntM);
    __aicore__ inline void Bmm2DataCopyOutTrans(LocalTensor<OUTPUT_T> &attenOutUb, uint32_t startRow,
                                                uint32_t dealRowCount, uint32_t columnCount) {};
    __aicore__ inline void ReduceFinalRes(LocalTensor<T> &reduceOut, LocalTensor<T> &mm2Res, LocalTensor<T> &lseLocal,
                                          uint32_t cntKV, uint32_t dealRowCount) {};
    __aicore__ inline void CopyFinalResOut(LocalTensor<T> &accumOutLocal, uint32_t startRow, uint32_t dealRowCount,
                                           uint32_t cntM) {};
    __aicore__ inline void SinkMax(LocalTensor<T> lseMaxUb, uint32_t startRow, uint32_t dealRowCount);

    __aicore__ inline void DealInvalidRows(LocalTensor<T> &attenOutUb, uint32_t startRow, uint32_t dealRowCount,
                                           uint32_t columnCount);
    __aicore__ inline void DealInvalidMaskRows(LocalTensor<T> &attenOutUb, uint32_t startRow, uint32_t dealRowCount,
                                               uint32_t columnCount, uint32_t cntM);
};

} // namespace BaseApi
#endif
