/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_flash_mla_block_vec_flashdecode_fp8.h
 * \brief QuantFlashMlaWithKvcache FD阶段vec block（自FIA MLA/QFA FD裁剪适配，
 *        规约workspace中间结果到输出，适配int32序列解析与多section metadata调度）
 */

#ifndef QUANT_FLASH_MLA_BLOCK_VEC_FLASHDECODE_FP8_H_
#define QUANT_FLASH_MLA_BLOCK_VEC_FLASHDECODE_FP8_H_

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#if __has_include("../../../common/op_kernel/arch35/infer_flash_attention_comm_arch35.h")
#include "../../../common/op_kernel/arch35/infer_flash_attention_comm_arch35.h"
#include "../../../common/op_kernel/arch35/vf/vf_flash_decode_arch35.h"
#else
#include "../../common/op_kernel/arch35/infer_flash_attention_comm_arch35.h"
#include "../../common/op_kernel/arch35/vf/vf_flash_decode_arch35.h"
#endif
#include "../../../common/op_kernel/vector_common.h"
#include "memory_copy_arch35_quant_flash_mla_with_kvcache.h"
#include "quant_flash_mla_with_kvcache_public_def.h"

using namespace AscendC;
using namespace FaVectorApi;
using namespace AttentionCommon;

namespace BaseApi {

template <typename INPUT_T, typename T, typename OUTPUT_T, LayOutTypeEnum layout = LayOutTypeEnum::LAYOUT_TND,
          LayOutTypeEnum outLayout = LayOutTypeEnum::LAYOUT_TND, S1TemplateType s1TemplateType = S1TemplateType::Aligned64,
          S2TemplateType s2TemplateType = S2TemplateType::Aligned128,
          DTemplateType dTemplateType = DTemplateType::Aligned576,
          DTemplateType dVTemplateType = DTemplateType::Aligned512>
class QuantFlashMlaBlockVecFlashDecodeFp8 {
public:
    // =================================类型定义区=================================
    struct TaskInfo {
        uint32_t bIdx;
        uint32_t n2Idx;
        uint32_t gS1Idx;
        uint32_t actualCombineLoopSize;
    };

private:
    // =================================常量区=================================
    static constexpr int64_t BYTE_BLOCK = 32UL;
    static constexpr int64_t REPEAT_BLOCK_BYTE = 256U;
    static constexpr uint64_t SYNC_LSE_MAX_SUM_BUF1_FLAG = 0;
    static constexpr uint64_t SYNC_LSE_MAX_SUM_BUF2_FLAG = 1;
    static constexpr uint64_t SYNC_MM2RES_BUF1_FLAG = 2;
    static constexpr uint64_t SYNC_MM2RES_BUF2_FLAG = 3;
    static constexpr uint64_t SYNC_FDOUTPUT_BUF_FLAG = 4;
    static constexpr uint64_t SYNC_LSEOUTPUT_BUF_FLAG = 5;
    static constexpr uint64_t SYNC_SINK_BUF1_FLAG = 6;
    static constexpr uint64_t SYNC_SINK_BUF2_FLAG = 7;

    static constexpr uint32_t BUFFER_SIZE_BYTE_256B = 256;
    static constexpr uint32_t BUFFER_SIZE_BYTE_2K = 2048;
    static constexpr uint32_t BUFFER_SIZE_BYTE_4K = 4096;
    static constexpr uint32_t BUFFER_SIZE_BYTE_16K = 16384;

    static constexpr uint32_t BLOCK_ELEMENT_NUM = BYTE_BLOCK / sizeof(T); // 32/4=8
    static constexpr uint32_t FP32_REPEAT_ELEMENT_NUM = REPEAT_BLOCK_BYTE / sizeof(float);

    static constexpr float FLOAT_INF = 3e+99;
    uint32_t preLoadNum = 2U;
    uint32_t dSizeV_Align;
    using ConstInfoX = QmlaConstInfo;
    // 基本块大小（与metadata分核mBaseSize=64一致）
    static constexpr uint32_t s1BaseSize = (uint32_t)s1TemplateType;

protected:
    GlobalTensor<float> lseSumFdGm_;
    GlobalTensor<float> lseMaxFdGm_;
    GlobalTensor<float> accumOutGm_;
    GlobalTensor<OUTPUT_T> attentionOutGm_;
    GlobalTensor<float> softmaxLseGm_;

    static constexpr UbFormat UB_FORMAT = GetOutUbFormat<layout>();

    // Q: TND, cu_seq(含前导0)+seq_used, int32（与FA vec block保持一致的解析器）
    using QSeqParserType = ActualSeqLensParser<ActualSeqLensMode::ACCUM, int32_t, true>;
    QSeqParserType *qActSeqLensParser_ = nullptr;

    // ================================类成员变量====================================
    // 结构体
    const ConstInfoX &constInfo_;
    TaskInfo taskInfo_{};

private:
    // ================================FD Local Buffer区====================================
    TBuf<> fdSumBuf1_;    // 1.5k: 16*24*4
    TBuf<> fdSumBuf2_;    // 1.5k: 16*24*4
    TBuf<> fdMaxBuf1_;    // 1.5k: 16*24*4
    TBuf<> fdMaxBuf2_;    // 1.5k: 16*24*4
    TBuf<> fdLseExpBuf_;  // 1.5k: 16*24*4
    TBuf<> fdMm2ResBuf1_; // 16k: 8*512*4
    TBuf<> fdMm2ResBuf2_; // 16k: 8*512*4
    TBuf<> fdReduceBuf_;  // 16k: 8*512*4
    TBuf<> fdOutputBuf_;  // 16k: 8*512*4

    TBuf<> fdLseMaxUbBuf1_;
    TBuf<> fdLseMaxUbBuf2_;
    TBuf<> fdLseUbBuf_;
    TBuf<> fdSinkExpBuf_;

public:
    __aicore__ inline QuantFlashMlaBlockVecFlashDecodeFp8(ConstInfoX &constInfo)
        : constInfo_(constInfo){};

    __aicore__ inline void InitGlobalTensor(GlobalTensor<float> lseMaxFdGm, GlobalTensor<float> lseSumFdGm,
                                            GlobalTensor<float> accumOutGm, GlobalTensor<OUTPUT_T> attentionOutGm)
    {
        this->lseMaxFdGm_ = lseMaxFdGm;
        this->lseSumFdGm_ = lseSumFdGm;
        this->accumOutGm_ = accumOutGm;
        this->attentionOutGm_ = attentionOutGm;
    }

    __aicore__ inline void SetCuSeqLensParsers(QSeqParserType &qParser)
    {
        this->qActSeqLensParser_ = &qParser;
    }

    __aicore__ inline void InitSoftmaxLseGm(GlobalTensor<float> softmaxLseGm)
    {
        this->softmaxLseGm_ = softmaxLseGm;
    }

    __aicore__ inline void InitParams()
    {
        this->dSizeV_Align = AttentionCommon::Align(constInfo_.dSizeV, FP32_REPEAT_ELEMENT_NUM);
    }

    __aicore__ inline void InitBuffers(TPipe *pipe)
    {
        if ASCEND_IS_AIV {
            pipe->Reset();
            // InQue, DB, SYNC_LSE_MAX_SUM_BUF1_FLAG SYNC_LSE_MAX_SUM_BUF2_FLAG
            pipe->InitBuffer(fdSumBuf1_, BUFFER_SIZE_BYTE_4K + BUFFER_SIZE_BYTE_2K);
            pipe->InitBuffer(fdSumBuf2_, BUFFER_SIZE_BYTE_4K + BUFFER_SIZE_BYTE_2K);
            pipe->InitBuffer(fdMaxBuf1_, BUFFER_SIZE_BYTE_4K + BUFFER_SIZE_BYTE_2K);
            pipe->InitBuffer(fdMaxBuf2_, BUFFER_SIZE_BYTE_4K + BUFFER_SIZE_BYTE_2K);
            // TmpBuf
            pipe->InitBuffer(fdLseExpBuf_, BUFFER_SIZE_BYTE_4K + BUFFER_SIZE_BYTE_2K);
            // InQue, DB, SYNC_MM2RES_BUF1_FLAG SYNC_MM2RES_BUF2_FLAG
            pipe->InitBuffer(fdMm2ResBuf1_, BUFFER_SIZE_BYTE_16K);
            pipe->InitBuffer(fdMm2ResBuf2_, BUFFER_SIZE_BYTE_16K);
            // TmpBuf
            pipe->InitBuffer(fdReduceBuf_, BUFFER_SIZE_BYTE_16K);
            // OutQue, SYNC_FDOUTPUT_BUF_FLAG
            pipe->InitBuffer(fdOutputBuf_, BUFFER_SIZE_BYTE_16K);

            pipe->InitBuffer(fdLseMaxUbBuf1_, BUFFER_SIZE_BYTE_256B);
            pipe->InitBuffer(fdLseMaxUbBuf2_, BUFFER_SIZE_BYTE_256B);
            // OutQue, SYNC_LSEOUTPUT_BUF_FLAG
            pipe->InitBuffer(fdLseUbBuf_, BUFFER_SIZE_BYTE_256B);

            // 后面要取地址，放到外面
            pipe->InitBuffer(fdSinkExpBuf_, BUFFER_SIZE_BYTE_256B);
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
        copyInParams.blockLen = constInfo_.dSizeV * sizeof(T);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = (this->dSizeV_Align - constInfo_.dSizeV) / BLOCK_ELEMENT_NUM;

        copyInPadParams.isPad = true;
        copyInPadParams.leftPadding = 0;
        copyInPadParams.rightPadding = (this->dSizeV_Align - constInfo_.dSizeV) % BLOCK_ELEMENT_NUM;
        copyInPadParams.paddingValue = 0;
        uint64_t combineAccumOutOffset = startRow * constInfo_.dSizeV +                 // taskoffset + g轴offset
                                         splitKVIndex * s1BaseSize * constInfo_.dSizeV; // 份数offset

        DataCopyPad(accumOutLocal, accumOutGm_[combineAccumOutOffset], copyInParams, copyInPadParams);
    }

    __aicore__ inline void CopyLseIn(uint32_t startRow, uint32_t dealRowCount, uint64_t baseOffset, uint32_t cntM)
    {
        LocalTensor<T> lseSum = (cntM & 1) == 0 ? fdSumBuf1_.Get<T>() : fdSumBuf2_.Get<T>();
        LocalTensor<T> lseMax = (cntM & 1) == 0 ? fdMaxBuf1_.Get<T>() : fdMaxBuf2_.Get<T>();

        uint64_t combineLseOffset = (baseOffset + startRow) * BLOCK_ELEMENT_NUM;
        uint64_t combineLoopOffset = s1BaseSize * BLOCK_ELEMENT_NUM;
        uint64_t dealRowCountAlign = dealRowCount * BLOCK_ELEMENT_NUM;

        for (uint32_t i = 0; i < taskInfo_.actualCombineLoopSize; ++i) {
            DataCopy(lseSum[i * dealRowCountAlign], lseSumFdGm_[combineLseOffset + i * combineLoopOffset],
                     dealRowCountAlign); // 份数offset

            DataCopy(lseMax[i * dealRowCountAlign], lseMaxFdGm_[combineLseOffset + i * combineLoopOffset],
                     dealRowCountAlign);
        }
    }

    __aicore__ inline void ComputeScaleValue(LocalTensor<T> &lseExp, uint32_t dealRowCount,
                                             uint32_t actualCombineLoopSize, uint32_t cntM)
    {
        LocalTensor<T> lseSum = (cntM & 1) == 0 ? fdSumBuf1_.Get<T>() : fdSumBuf2_.Get<T>();
        LocalTensor<T> lseMax = (cntM & 1) == 0 ? fdMaxBuf1_.Get<T>() : fdMaxBuf2_.Get<T>();
        LocalTensor<T> lseMaxUb = (cntM & 1) == 0 ? fdLseMaxUbBuf1_.Get<T>() : fdLseMaxUbBuf2_.Get<T>();

        LocalTensor<T> sinkExpBuf = fdSinkExpBuf_.Get<T>();
        LocalTensor<T> maxLseUb = fdLseUbBuf_.Get<T>();
        ComputeScaleValue_VF_FD(sinkExpBuf, lseMax, lseSum, lseExp, maxLseUb, lseMaxUb, dealRowCount,
                                actualCombineLoopSize, constInfo_.isSoftmaxLseEnable, false);
    }

    __aicore__ inline void Bmm2DataCopyOutTrans(LocalTensor<OUTPUT_T> &attenOutUb, uint32_t startRow,
                                               uint32_t dealRowCount, uint32_t columnCount)
    {
        FaUbTensor<OUTPUT_T> ubTensor{
            .tensor = attenOutUb,
            .rowCount = dealRowCount,
            .colCount = columnCount,
        };
        GmCoord gmCoord{.bIdx = taskInfo_.bIdx,
                        .n2Idx = taskInfo_.n2Idx,
                        .gS1Idx = taskInfo_.gS1Idx + startRow,
                        .dIdx = 0,
                        .gS1DealSize = dealRowCount,
                        .dDealSize = (uint32_t)constInfo_.dSizeV};
        CopyAttentionOut(ubTensor, gmCoord);
    }

    __aicore__ inline void CopyAttentionOut(FaUbTensor<OUTPUT_T> &ubTensor, GmCoord &gmCoord)
    {
        if constexpr (outLayout == LayOutTypeEnum::LAYOUT_TND) {
            constexpr GmFormat OUT_FORMAT = GmFormat::TNGD;
            FaGmTensor<OUTPUT_T, OUT_FORMAT, int32_t, true> outGmTensor;
            outGmTensor.gmTensor = attentionOutGm_;
            outGmTensor.offsetCalculator.Init(constInfo_.realN2Size, constInfo_.realGSize, constInfo_.dSizeV,
                                              *qActSeqLensParser_);
            CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, GetOutUbFormat<layout>()> copyAttenOutUbToGm;
            copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
        } else if constexpr (outLayout == LayOutTypeEnum::LAYOUT_NTD) {
            constexpr GmFormat OUT_FORMAT = GmFormat::NGTD;
            FaGmTensor<OUTPUT_T, OUT_FORMAT, int32_t, true> outGmTensor;
            outGmTensor.gmTensor = attentionOutGm_;
            outGmTensor.offsetCalculator.Init(constInfo_.realN2Size, constInfo_.realGSize, constInfo_.dSizeV,
                                               *qActSeqLensParser_);
            CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, GetOutUbFormat<layout>()> copyAttenOutUbToGm;
            copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
        } else if constexpr (outLayout == LayOutTypeEnum::LAYOUT_BSH) {
            constexpr GmFormat OUT_FORMAT = GmFormat::BSNGD;
            FaGmTensor<OUTPUT_T, OUT_FORMAT> outGmTensor;
            outGmTensor.gmTensor = attentionOutGm_;
            outGmTensor.offsetCalculator.Init(constInfo_.bSize, constInfo_.realN2Size, constInfo_.realGSize,
                                              constInfo_.s1Size, constInfo_.dSizeV);
            CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, GetOutUbFormat<layout>()> copyAttenOutUbToGm;
            copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
        } else { // BNSD
            constexpr GmFormat OUT_FORMAT = GmFormat::BNGSD;
            FaGmTensor<OUTPUT_T, OUT_FORMAT> outGmTensor;
            outGmTensor.gmTensor = attentionOutGm_;
            outGmTensor.offsetCalculator.Init(constInfo_.bSize, constInfo_.realN2Size, constInfo_.realGSize,
                                              constInfo_.s1Size, constInfo_.dSizeV);
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

    __aicore__ inline void CopyFinalResOut(LocalTensor<T> &accumOutLocal, uint32_t startRow, uint32_t dealRowCount)
    {
        LocalTensor<OUTPUT_T> tmpBmm2ResCastTensor = fdOutputBuf_.Get<OUTPUT_T>();
        AscendC::PipeBarrier<PIPE_V>();
        WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_FDOUTPUT_BUF_FLAG);
        uint32_t shapeArray[] = {dealRowCount, (uint32_t)constInfo_.dSizeV};
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

    __aicore__ inline void CopySoftmaxLseOutput(uint32_t actualGSplitSize, uint32_t startRow)
    {
        if (constInfo_.isSoftmaxLseEnable) {
            // lse行无效在ComputeScaleValue的VF计算时已经进行了赋值inf处理
            LocalTensor<T> maxLseUb = fdLseUbBuf_.Get<T>();
            // 判断是否行无效
            SetFlag<HardEvent::V_MTE3>(SYNC_LSEOUTPUT_BUF_FLAG);
            WaitFlag<HardEvent::V_MTE3>(SYNC_LSEOUTPUT_BUF_FLAG);
            uint32_t mOffset = taskInfo_.gS1Idx + startRow;
            // 新接口LSE输出统一为NT格式[N2*G, T]，与FA vec block一致
            uint32_t prefixBS1 = qActSeqLensParser_->GetTBase(taskInfo_.bIdx);
            uint64_t bN2Offset = taskInfo_.n2Idx * constInfo_.realGSize * constInfo_.t1Size + prefixBS1;
            DataCopySoftmaxLseTNDtoNTArch35NoGS1Merge<T, ConstInfoX>(softmaxLseGm_, maxLseUb, bN2Offset, mOffset,
                                                                     actualGSplitSize, constInfo_);
        }
    }

public:
    __aicore__ inline void FlashDecode(FDparamsX &fd)
    {
        if (!fd.fdCoreEnable) {
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
        // MLA: realN2Size = n2Size = 1, bIdx即fdBN2Idx
        taskInfo_.bIdx = fd.fdBN2Idx / constInfo_.realN2Size;
        taskInfo_.n2Idx = fd.fdBN2Idx % constInfo_.realN2Size;
        taskInfo_.gS1Idx = fd.fdMIdx * s1BaseSize;
        taskInfo_.actualCombineLoopSize = fd.fdS2SplitNum; // 当前规约任务kv方向有几份
        uint64_t combineTaskPrefixSum = fd.fdWorkspaceIdx;
        uint64_t taskOffset = combineTaskPrefixSum * s1BaseSize;

        for (uint32_t fdS1gOuterMIdx = tmpFdS1gOuterMStart; fdS1gOuterMIdx <= tmpFdS1gOuterMEnd;
             ++fdS1gOuterMIdx) { // 左闭右闭
            uint32_t actualGSplitSize = fdBalanceMBaseSize;
            if (fdS1gOuterMIdx == fdBalanceMSplitNum - 1) {
                actualGSplitSize = fdBalanceMTailSize;
            }
            uint32_t startRow = fd.mStart + fdS1gOuterMIdx * fdBalanceMBaseSize;

            LocalTensor<T> lseExp = fdLseExpBuf_.Get<T>();
            LocalTensor<T> reduceOut = fdReduceBuf_.Get<T>();
            WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_LSE_MAX_SUM_BUF1_FLAG + (reduceMLoop & 1));
            CopyLseIn(startRow, actualGSplitSize, taskOffset, reduceMLoop);
            SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_LSE_MAX_SUM_BUF1_FLAG + (reduceMLoop & 1));
            WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_LSE_MAX_SUM_BUF1_FLAG + (reduceMLoop & 1));
            for (uint32_t preLoadIdx = 0; preLoadIdx < preLoadNum; ++preLoadIdx) {
                LocalTensor<T> mm2Res =
                    ((reduceGlobaLoop + preLoadIdx) & 1) == 0 ? fdMm2ResBuf1_.Get<T>() : fdMm2ResBuf2_.Get<T>();
                WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_MM2RES_BUF1_FLAG + ((reduceGlobaLoop + preLoadIdx) & 1));
                CopyAccumOutIn(mm2Res, preLoadIdx, taskOffset + startRow, actualGSplitSize);
                SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_MM2RES_BUF1_FLAG + ((reduceGlobaLoop + preLoadIdx) & 1));
            }
            ComputeScaleValue(lseExp, actualGSplitSize, taskInfo_.actualCombineLoopSize, reduceMLoop);
            CopySoftmaxLseOutput(actualGSplitSize, startRow);
            SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_LSE_MAX_SUM_BUF1_FLAG + (reduceMLoop & 1));

            for (uint32_t i = 0; i < taskInfo_.actualCombineLoopSize; ++i) {
                LocalTensor<T> mm2Res = (reduceGlobaLoop & 1) == 0 ? fdMm2ResBuf1_.Get<T>() : fdMm2ResBuf2_.Get<T>();
                if (i >= preLoadNum) {
                    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_MM2RES_BUF1_FLAG + (reduceGlobaLoop & 1));
                    CopyAccumOutIn(mm2Res, i, taskOffset + startRow, actualGSplitSize);
                    SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_MM2RES_BUF1_FLAG + (reduceGlobaLoop & 1));
                }
                WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_MM2RES_BUF1_FLAG + (reduceGlobaLoop & 1));
                ReduceFinalRes(reduceOut, mm2Res, lseExp, i, actualGSplitSize);
                SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_MM2RES_BUF1_FLAG + (reduceGlobaLoop & 1));
                reduceGlobaLoop += 1;
            }
            CopyFinalResOut(reduceOut, startRow, actualGSplitSize);
            reduceMLoop += 1;
        }
    }
};

// AIC侧编译占位（避免cube编译单元实例化FD vec逻辑）
template <typename INPUT_T, typename T, typename OUTPUT_T, LayOutTypeEnum layout = LayOutTypeEnum::LAYOUT_TND,
          LayOutTypeEnum outLayout = LayOutTypeEnum::LAYOUT_TND, S1TemplateType s1TemplateType = S1TemplateType::Aligned64,
          S2TemplateType s2TemplateType = S2TemplateType::Aligned128,
          DTemplateType dTemplateType = DTemplateType::Aligned576,
          DTemplateType dVTemplateType = DTemplateType::Aligned512>
class QuantFlashMlaBlockVecFlashDecodeFp8Dummy {
public:
    using ConstInfoX = QmlaConstInfo;
    __aicore__ inline QuantFlashMlaBlockVecFlashDecodeFp8Dummy(ConstInfoX &constInfo){};
};

} // namespace BaseApi
#endif // QUANT_FLASH_MLA_BLOCK_VEC_FLASHDECODE_FP8_H_
