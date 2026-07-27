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
 * \file flash_attn_block_vec_noquant_flashdecode.h
 * \brief
 */
#ifndef FLASH_ATTN_BLOCK_VEC_FLASHDECODE_H
#define FLASH_ATTN_BLOCK_VEC_FLASHDECODE_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"

#if __has_include("../../../common/op_kernel/arch35/infer_flash_attention_comm.h")
#include "../../../common/op_kernel/arch35/infer_flash_attention_comm.h"
#include "../../../common/op_kernel/arch35/vf/vf_flash_decode.h"
#include "../../../common/op_kernel/fia_public_define.h"
#include "../../../common/op_kernel/memory_copy_arch35.h"
#include "../../../common/op_kernel/memory_copy.h"
#else
#include "../../common/arch35/infer_flash_attention_comm.h"
#include "../../common/arch35/vf/vf_flash_decode.h"
#include "../../common/fia_public_define.h"
#include "../../common/memory_copy_arch35.h"
#include "../../common/memory_copy.h"
#endif

namespace BaseApi {
struct TaskInfo {
    uint32_t bIdx;
    uint32_t n2Idx;
    uint32_t gS1Idx;
    uint32_t actualCombineLoopSize;
};

template <FA_LAYOUT LAYOUT_T>
__aicore__ inline constexpr fa_base_vector::UbInputFormat GeInputUbFormat()
{
    static_assert((LAYOUT_T == FA_LAYOUT::BSND) || (LAYOUT_T == FA_LAYOUT::BNSD) || (LAYOUT_T == FA_LAYOUT::TND),
                  "Get Query GmFormat fail, LAYOUT_T is incorrect");
    if constexpr (LAYOUT_T == FA_LAYOUT::TND || LAYOUT_T == FA_LAYOUT::BSND) {
        return fa_base_vector::UbInputFormat::S1G;
    } else if constexpr (LAYOUT_T == FA_LAYOUT::BNSD) {
        return fa_base_vector::UbInputFormat::GS1;
    }
}

template <typename FA_T>
class FiaBlockVecFlashDecode {
public:
    using INPUT_T = typename FA_T::inputType;
    using OUTPUT_T = typename FA_T::outputType;
    static constexpr uint32_t mBaseSize = (uint32_t)FA_T::mBaseSize;
    static constexpr uint32_t s2BaseSize = (uint32_t)FA_T::s2BaseSize;
    static constexpr uint32_t dVBaseSize = (uint32_t)FA_T::dVBaseSize;
    static constexpr FA_LAYOUT LAYOUT_T = FA_T::qLayout;
    static constexpr FA_LAYOUT LAYOUT_KV = FA_T::kvLayout;
    static constexpr FA_LAYOUT LAYOUT_OUT = FA_T::attnOutLayout;
    static constexpr bool PAGE_ATTENTION = FA_T::pageAttention;
    static constexpr bool HAS_MASK = FA_T::hasMask;
    // =================================类型定义区=================================
    using T = float;
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
    using ConstInfoX = ConstInfo_t<FiaKernelType::NO_QUANT>;

protected:
    GlobalTensor<float> lseSumFdGm;
    GlobalTensor<float> lseMaxFdGm;
    GlobalTensor<float> accumOutGm;
    GlobalTensor<float> softmaxLseGm;

    static constexpr UbFormat UB_FORMAT = GetOutUbFormat<LAYOUT_T>();
    int64_t preTokensPerBatch = 0;
    int64_t nextTokensPerBatch = 0;

    static constexpr T BOOL_ATTEN_MASK_SCALAR_VALUE = -1000000000000.0; // 用于mask为bool类型
    uint32_t negativeIntScalar = *((uint32_t *)&BOOL_ATTEN_MASK_SCALAR_VALUE);

    uint64_t actSeqLensKv = 0;
    uint64_t actSeqLensQ = 0;
    // ================================类成员变量====================================
    const ConstInfoX &constInfo;
    TaskInfo taskInfo{};

    using SEQLEN_T = uint32_t;
    SeqLensTool<LAYOUT_T, SEQLEN_T> &qSeqLensTool;
    SeqLensTool<LAYOUT_KV, SEQLEN_T> &kvSeqLensTool;

    static constexpr GmFormat OUT_FORMAT = GetAttentionOutGmFormat<LAYOUT_OUT>();
    using FaGmTensorOut = FaGmTensor<OUTPUT_T, OUT_FORMAT, SEQLEN_T, IS_TND<LAYOUT_OUT>()>;
    FaGmTensorOut outGmTensor;
    CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, GetOutUbFormat<LAYOUT_T>()> copyAttenOutUbToGm;

private:
    // ================================FD Local Buffer区====================================
    LocalTensor<T> fdSumBuf1;          // 1.5k: 16*24*4
    LocalTensor<T> fdSumBuf2;          // 1.5k: 16*24*4
    LocalTensor<T> fdMaxBuf1;          // 1.5k: 16*24*4
    LocalTensor<T> fdMaxBuf2;          // 1.5k: 16*24*4
    LocalTensor<T> fdLseExpBuf;        // 1.5k: 16*24*4
    LocalTensor<T> fdMm2ResBuf1;       // 32k: 16*512*4
    LocalTensor<T> fdMm2ResBuf2;       // 32k: 16*512*4
    LocalTensor<T> fdReduceBuf;        // 32k: 16*512*4
    LocalTensor<OUTPUT_T> fdOutputBuf; // 32k: 16*512*4

    LocalTensor<T> fdLseMaxUbBuf1;
    LocalTensor<T> fdLseMaxUbBuf2;
    LocalTensor<T> fdLseUbBuf;

public:
    __aicore__ inline FiaBlockVecFlashDecode(ConstInfoX &constInfo, SeqLensTool<LAYOUT_T, SEQLEN_T> &qSeqLensTool,
                                             SeqLensTool<LAYOUT_KV, SEQLEN_T> &kvSeqLensTool)
        : constInfo(constInfo), qSeqLensTool(qSeqLensTool), kvSeqLensTool(kvSeqLensTool){};

    template <typename U> // 避免重名用U
    __aicore__ inline U Align(U num, U rnd)
    {
        return (((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd) * (rnd)));
    }

    __aicore__ inline void InitBlock(__gm__ uint8_t *learnableSink, __gm__ uint8_t *softmaxLse,
                                     __gm__ uint8_t *attentionOut)
    {
        this->dSizeV_Align = this->Align(constInfo.dSizeV, FP32_REPEAT_ELEMENT_NUM);

        InitAttenOutBuffer(constInfo.bSize, constInfo.n2Size, constInfo.gSize, constInfo.s1Size, constInfo.dSizeV,
                           outGmTensor, attentionOut);

        if (constInfo.isSoftmaxLseEnable) {
            softmaxLseGm.SetGlobalBuffer((__gm__ float *)softmaxLse);
        }
    }

    __aicore__ inline void InitGlobalTensor(GlobalTensor<float> lseMaxFdGm, GlobalTensor<float> lseSumFdGm,
                                            GlobalTensor<float> accumOutGm)
    {
        this->lseMaxFdGm = lseMaxFdGm;
        this->lseSumFdGm = lseSumFdGm;
        this->accumOutGm = accumOutGm;
    }

    __aicore__ inline void InitBuffers()
    {
        if ASCEND_IS_AIV {
            // 与 FA block 共享UB布局：bmm1/bmm2 区域在前，FD 业务缓冲区紧随其后。
            // 使用 LocalTensor 构造函数直接指定绝对字节偏移，实现内存完全自主管理，
            // 无需 LocalMemAllocator 线性分配器。
            constexpr uint32_t mm1Sz = mBaseSize / 2U * s2BaseSize * sizeof(T);
            constexpr uint32_t mm2Sz = mBaseSize / 2U * dVBaseSize * sizeof(T);

            // FD 业务区起始字节偏移（跳过 bmm1/bmm2/mm2In 区域）
            constexpr uint32_t BASE = mm1Sz * 2U + mm2Sz * 2U;

            // 各共享区块的绝对字节偏移（与原 LocalMemAllocator 分配顺序完全一致）
            // SharedBuffer2[0]：attenMaskBuf[0](FA) / fdLseMaxUb(FD)，8192 bytes
            constexpr uint32_t OFF_BUF2A = BASE;
            // Skip attenMaskBuf[1]：8192 bytes
            // SharedBuffer3：stage2OutBuf(FA) / fdSum/Max/LseExp(FD)，32768 bytes
            constexpr uint32_t OFF_BUF3 = BASE + BUFFER_SIZE_BYTE_8K + BUFFER_SIZE_BYTE_8K;
            // SharedBuffer1[0]：stage1OutBuf[0](FA) / fdMm2Res(FD)，33024 bytes
            constexpr uint32_t OFF_BUF1A = OFF_BUF3 + 32768U;
            // SharedBuffer1[1]：stage1OutBuf[1](FA) / fdReduce/Output(FD)，33024 bytes
            constexpr uint32_t OFF_BUF1B = OFF_BUF1A + 33024U;

            // sharedBuf1a 内：fdMm2ResBuf1（前16K）、fdMm2ResBuf2（后16K）
            fdMm2ResBuf1 =
                LocalTensor<uint8_t>(TPosition::VECIN, OFF_BUF1A, BUFFER_SIZE_BYTE_16K).template ReinterpretCast<T>();
            fdMm2ResBuf2 =
                LocalTensor<uint8_t>(TPosition::VECIN, OFF_BUF1A + BUFFER_SIZE_BYTE_16K, BUFFER_SIZE_BYTE_16K)
                    .template ReinterpretCast<T>();
            // sharedBuf1b 内：fdReduceBuf（前16K）、fdOutputBuf（后16K）
            fdReduceBuf =
                LocalTensor<uint8_t>(TPosition::VECIN, OFF_BUF1B, BUFFER_SIZE_BYTE_16K).template ReinterpretCast<T>();
            fdOutputBuf = LocalTensor<uint8_t>(TPosition::VECIN, OFF_BUF1B + BUFFER_SIZE_BYTE_16K, BUFFER_SIZE_BYTE_16K)
                              .template ReinterpretCast<OUTPUT_T>();

            // sharedBuf3 内：5 个 6144-byte 槽（fdSum1/2, fdMax1/2, fdLseExp）
            constexpr uint32_t STRIDE = BUFFER_SIZE_BYTE_4K + BUFFER_SIZE_BYTE_2K; // 6144 bytes
            fdSumBuf1 = LocalTensor<uint8_t>(TPosition::VECIN, OFF_BUF3, STRIDE).template ReinterpretCast<T>();
            fdSumBuf2 = LocalTensor<uint8_t>(TPosition::VECIN, OFF_BUF3 + STRIDE, STRIDE).template ReinterpretCast<T>();
            fdMaxBuf1 =
                LocalTensor<uint8_t>(TPosition::VECIN, OFF_BUF3 + 2U * STRIDE, STRIDE).template ReinterpretCast<T>();
            fdMaxBuf2 =
                LocalTensor<uint8_t>(TPosition::VECIN, OFF_BUF3 + 3U * STRIDE, STRIDE).template ReinterpretCast<T>();
            fdLseExpBuf =
                LocalTensor<uint8_t>(TPosition::VECIN, OFF_BUF3 + 4U * STRIDE, STRIDE).template ReinterpretCast<T>();

            // sharedBuf2a 内：fdLseMaxUbBuf1/2、fdLseUbBuf，各 256 bytes
            fdLseMaxUbBuf1 =
                LocalTensor<uint8_t>(TPosition::VECIN, OFF_BUF2A, BUFFER_SIZE_BYTE_256B).template ReinterpretCast<T>();
            fdLseMaxUbBuf2 =
                LocalTensor<uint8_t>(TPosition::VECIN, OFF_BUF2A + BUFFER_SIZE_BYTE_256B, BUFFER_SIZE_BYTE_256B)
                    .template ReinterpretCast<T>();
            fdLseUbBuf =
                LocalTensor<uint8_t>(TPosition::VECIN, OFF_BUF2A + 2U * BUFFER_SIZE_BYTE_256B, BUFFER_SIZE_BYTE_256B)
                    .template ReinterpretCast<T>();
        }
    }

protected:
    __aicore__ inline void InitAttenOutBuffer(uint32_t batchSize, uint32_t n2Size, uint32_t gSize, uint32_t qSeqSize,
                                              uint32_t headDim, FaGmTensorOut &outGmTensor, __gm__ uint8_t *gm)
    {
        outGmTensor.gmTensor.SetGlobalBuffer((__gm__ OUTPUT_T *)gm);
        if constexpr (GmLayoutParams<OUT_FORMAT>::CATEGORY == FormatCategory::GM_Q_OUT_BNGSD) {
            outGmTensor.offsetCalculator.Init(batchSize, n2Size, gSize, qSeqSize, headDim, qSeqLensTool.seqUsedParser);
        } else {
            outGmTensor.offsetCalculator.Init(n2Size, gSize, headDim, qSeqLensTool.cuSeqLensParser);
        }
    }

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
        uint64_t combineAccumOutOffset = startRow * constInfo.dSizeV +                // taskoffset + g轴offset
                                         splitKVIndex * mBaseSize * constInfo.dSizeV; // 份数offset

        DataCopyPad(accumOutLocal, accumOutGm[combineAccumOutOffset], copyInParams, copyInPadParams);
    }
    __aicore__ inline void CopyLseIn(uint32_t startRow, uint32_t dealRowCount, uint64_t baseOffset, uint32_t cntM)
    {
        LocalTensor<T> lseSum = (cntM & 1) == 0 ? fdSumBuf1 : fdSumBuf2;
        LocalTensor<T> lseMax = (cntM & 1) == 0 ? fdMaxBuf1 : fdMaxBuf2;

        uint64_t combineLseOffset = (baseOffset + startRow) * FP32_BLOCK_ELEMENT_NUM;
        uint64_t combineLoopOffset = mBaseSize * FP32_BLOCK_ELEMENT_NUM;
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
        LocalTensor<T> lseMaxUb = (cntM & 1) == 0 ? fdLseMaxUbBuf1 : fdLseMaxUbBuf2;

        LocalTensor<T> sinkExpBuf;
        LocalTensor<T> maxLseUb = fdLseUbBuf;
        bool learnableSinkFlag = false;
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
        copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
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
        Mutex::Lock<PIPE_V>(SYNC_FDOUTPUT_BUF_FLAG);
        uint32_t shapeArray[] = {dealRowCount, (uint32_t)constInfo.dSizeV};
        tmpBmm2ResCastTensor.SetShapeInfo(ShapeInfo(2, shapeArray, DataFormat::ND));
        if constexpr (IsSameType<OUTPUT_T, bfloat16_t>::value) { // bf16 采取四舍六入五成双模式
            Cast(tmpBmm2ResCastTensor, accumOutLocal, AscendC::RoundMode::CAST_RINT, dealRowCount * this->dSizeV_Align);
        } else {
            Cast(tmpBmm2ResCastTensor, accumOutLocal, AscendC::RoundMode::CAST_ROUND,
                 dealRowCount * this->dSizeV_Align);
        }
        Mutex::Unlock<PIPE_V>(SYNC_FDOUTPUT_BUF_FLAG);
        Mutex::Lock<PIPE_MTE3>(SYNC_FDOUTPUT_BUF_FLAG);
        Bmm2DataCopyOutTrans(tmpBmm2ResCastTensor, startRow, dealRowCount, this->dSizeV_Align);
        Mutex::Unlock<PIPE_MTE3>(SYNC_FDOUTPUT_BUF_FLAG);
    }
    __aicore__ inline void CalcPreNextTokens()
    {
        actSeqLensQ = qSeqLensTool.GetActualSeqLength(taskInfo.bIdx);
        actSeqLensKv = kvSeqLensTool.GetActualSeqLength(taskInfo.bIdx);

        int64_t safePreToken = constInfo.preTokens;
        int64_t safeNextToken = constInfo.nextTokens;

        fa_base_vector::GetSafeActToken(actSeqLensQ, actSeqLensKv, safePreToken, safeNextToken, constInfo.sparseMode);

        if (constInfo.sparseMode == BAND) {
            preTokensPerBatch = safePreToken;
            nextTokensPerBatch = actSeqLensKv - actSeqLensQ + safeNextToken;
        } else if ((constInfo.sparseMode == DEFAULT_MASK) && HAS_MASK) {
            nextTokensPerBatch = safeNextToken;
            preTokensPerBatch = actSeqLensKv - actSeqLensQ + safePreToken;
        } else {
            nextTokensPerBatch = actSeqLensKv - actSeqLensQ;
            preTokensPerBatch = 0;
        }
    }

    template <typename UBOUT_T>
    __aicore__ inline void DealInvalidRows(LocalTensor<UBOUT_T> &attenOutUb, uint32_t startRow, uint32_t dealRowCount,
                                           uint32_t columnCount)
    {
        if constexpr (!HAS_MASK) {
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

        fa_base_vector::InvalidRows<UBOUT_T, GeInputUbFormat<LAYOUT_T>()> invalidRows;
        invalidRows(attenOutUb, params);
    }

    template <typename UBOUT_T>
    __aicore__ inline void DealInvalidMaskRows(LocalTensor<UBOUT_T> &attenOutUb, uint32_t startRow,
                                               uint32_t dealRowCount, uint32_t columnCount, uint32_t cntM)
    {
        if constexpr (!HAS_MASK) {
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
        taskInfo.gS1Idx = fd.fdMIdx * mBaseSize;
        taskInfo.actualCombineLoopSize = fd.fdS2SplitNum; // 当前规约任务kv方向有几份
        uint64_t combineTaskPrefixSum = fd.fdWorkspaceIdx;
        uint64_t taskOffset = combineTaskPrefixSum * mBaseSize;

        for (uint32_t fdS1gOuterMIdx = tmpFdS1gOuterMStart; fdS1gOuterMIdx <= tmpFdS1gOuterMEnd;
             fdS1gOuterMIdx++) { // 左闭右闭
            uint32_t actualGSplitSize = fdBalanceMBaseSize;
            if (fdS1gOuterMIdx == fdBalanceMSplitNum - 1) {
                actualGSplitSize = fdBalanceMTailSize;
            }
            uint32_t startRow = fd.mStart + fdS1gOuterMIdx * fdBalanceMBaseSize;

            LocalTensor<T> lseExp = fdLseExpBuf;
            LocalTensor<T> reduceOut = fdReduceBuf;
            Mutex::Lock<PIPE_MTE2>(SYNC_LSE_MAX_SUM_BUF1_FLAG + (reduceMLoop & 1));
            CopyLseIn(startRow, actualGSplitSize, taskOffset, reduceMLoop);
            Mutex::Unlock<PIPE_MTE2>(SYNC_LSE_MAX_SUM_BUF1_FLAG + (reduceMLoop & 1));
            for (uint32_t preLoadIdx = 0; preLoadIdx < preLoadNum; preLoadIdx++) {
                LocalTensor<T> mm2Res = (((reduceGlobaLoop + preLoadIdx) & 1) == 0) ? fdMm2ResBuf1 : fdMm2ResBuf2;
                Mutex::Lock<PIPE_MTE2>(SYNC_MM2RES_BUF1_FLAG + ((reduceGlobaLoop + preLoadIdx) & 1));
                CopyAccumOutIn(mm2Res, preLoadIdx, taskOffset + startRow, actualGSplitSize);
                Mutex::Unlock<PIPE_MTE2>(SYNC_MM2RES_BUF1_FLAG + ((reduceGlobaLoop + preLoadIdx) & 1));
            }
            Mutex::Lock<PIPE_V>(SYNC_LSE_MAX_SUM_BUF1_FLAG + (reduceMLoop & 1));
            Mutex::Lock<PIPE_V>(SYNC_LSEOUTPUT_BUF_FLAG);
            ComputeScaleValue(lseExp, actualGSplitSize, taskInfo.actualCombineLoopSize, reduceMLoop, startRow);
            Mutex::Unlock<PIPE_V>(SYNC_LSEOUTPUT_BUF_FLAG);
            Mutex::Unlock<PIPE_V>(SYNC_LSE_MAX_SUM_BUF1_FLAG + (reduceMLoop & 1));
            CalcPreNextTokens();
            if (constInfo.isSoftmaxLseEnable) {
                LocalTensor<T> maxLseUb = fdLseUbBuf;
                Mutex::Lock<PIPE_MTE3>(SYNC_LSEOUTPUT_BUF_FLAG);
                uint32_t mOffset = taskInfo.gS1Idx + startRow;
                if constexpr (LAYOUT_T == FA_LAYOUT::TND) {
                    uint32_t prefixBS1 = qSeqLensTool.cuSeqLensParser.GetTBase(taskInfo.bIdx);
                    uint64_t bN2Offset = taskInfo.n2Idx * constInfo.gSize * constInfo.t1Size + prefixBS1;
                    DataCopySoftmaxLseTNDtoNTArch35<T, ConstInfoX>(softmaxLseGm, maxLseUb, bN2Offset, mOffset,
                                                                   actualGSplitSize, constInfo);
                } else if constexpr (LAYOUT_T == FA_LAYOUT::BSND) {
                    uint64_t bN2Offset = taskInfo.bIdx * constInfo.gSize * constInfo.n2Size * constInfo.s1Size +
                                         taskInfo.n2Idx * constInfo.gSize * constInfo.s1Size;
                    uint64_t qActSeqLens = qSeqLensTool.seqUsedParser.GetActualSeqLength(taskInfo.bIdx);
                    DataCopySoftmaxLseBSNDArch35<T, ConstInfoX>(softmaxLseGm, maxLseUb, bN2Offset, mOffset,
                                                                actualGSplitSize, constInfo);
                } else if constexpr (LAYOUT_T == FA_LAYOUT::BNSD) {
                    uint64_t bN2Offset = taskInfo.bIdx * constInfo.gSize * constInfo.n2Size * constInfo.s1Size +
                                         taskInfo.n2Idx * constInfo.gSize * constInfo.s1Size;
                    uint64_t qActSeqLens = qSeqLensTool.seqUsedParser.GetActualSeqLength(taskInfo.bIdx);
                    DataCopySoftmaxLseBNSDArch35<T, ConstInfoX>(softmaxLseGm, maxLseUb, bN2Offset, mOffset,
                                                                actualGSplitSize, constInfo, qActSeqLens);
                }
                Mutex::Unlock<PIPE_MTE3>(SYNC_LSEOUTPUT_BUF_FLAG);
            }

            for (uint32_t i = 0; i < taskInfo.actualCombineLoopSize; i++) {
                LocalTensor<T> mm2Res = (reduceGlobaLoop & 1) == 0 ? fdMm2ResBuf1 : fdMm2ResBuf2;
                if (i >= preLoadNum) {
                    Mutex::Lock<PIPE_MTE2>(SYNC_MM2RES_BUF1_FLAG + (reduceGlobaLoop & 1));
                    CopyAccumOutIn(mm2Res, i, taskOffset + startRow, actualGSplitSize);
                    Mutex::Unlock<PIPE_MTE2>(SYNC_MM2RES_BUF1_FLAG + (reduceGlobaLoop & 1));
                }
                Mutex::Lock<PIPE_V>(SYNC_MM2RES_BUF1_FLAG + (reduceGlobaLoop & 1));
                ReduceFinalRes(reduceOut, mm2Res, lseExp, i, actualGSplitSize);
                Mutex::Unlock<PIPE_V>(SYNC_MM2RES_BUF1_FLAG + (reduceGlobaLoop & 1));
                reduceGlobaLoop += 1;
            }
            CopyFinalResOut(reduceOut, startRow, actualGSplitSize, reduceMLoop);
            reduceMLoop += 1;
        }
    }
};

} // namespace BaseApi
#endif
