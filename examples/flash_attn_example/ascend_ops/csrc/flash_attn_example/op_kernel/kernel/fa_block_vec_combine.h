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
 * \file fa_block_vec_combine.h
 * \brief
 */
#ifndef FA_BLOCK_VEC_COMBINE_H
#define FA_BLOCK_VEC_COMBINE_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "../vector/vf/vf_combine.h"
#include "../fa_kernel_public.h"
#include "../vector/vector.h"
#include "../memcopy/memory_copy.h"

using namespace fa_kernel::config;
using namespace fa_kernel::layout;
using namespace fa_kernel;

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
          DTemplateType dVTemplateType = DTemplateType::Aligned128, bool hasAttn = false>
class FaBlockVecCombine {
public:
    // =================================类型定义区=================================

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

    static constexpr uint32_t BUFFER_SIZE_BYTE_256B = 256;
    static constexpr uint32_t BUFFER_SIZE_BYTE_1K = 1024;
    static constexpr uint32_t BUFFER_SIZE_BYTE_2K = 2048;
    static constexpr uint32_t BUFFER_SIZE_BYTE_4K = 4096;
    static constexpr uint32_t BUFFER_SIZE_BYTE_16K = 16384;

    static constexpr uint32_t BLOCK_ELEMENT_NUM = BYTE_BLOCK / sizeof(T); // 32/4=8
    static constexpr uint32_t FP32_REPEAT_ELEMENT_NUM = REPEAT_BLOCK_BYTE / sizeof(float);

    static constexpr float FLOAT_INF = 3e+99;
    uint32_t preLoadNum = 2U;
    uint32_t dSizeV_Align;
    static constexpr bool attnMaskFlag = hasAttn;
    using ConstInfoX = ConstInfo_t<FaKernelType::NO_QUANT>;
    // 基本块大小
    static constexpr uint32_t s1BaseSize = (uint32_t)s1TemplateType;

protected:
    GlobalTensor<float> lseSumCombineGm;
    GlobalTensor<float> lseMaxCombineGm;
    GlobalTensor<float> accumOutGm;
    GlobalTensor<OUTPUT_T> attentionOutGm;
    GlobalTensor<uint64_t> actualSeqLengthsGmQ;
    GlobalTensor<uint64_t> actualSeqLengthsGm;

    // tensorlist
    __gm__ uint8_t *keyPtr = nullptr;

    int64_t preTokensPerBatch = 0;
    int64_t nextTokensPerBatch = 0;

    static constexpr T BOOL_ATTN_MASK_SCALAR_VALUE = -1000000000000.0; // 用于mask为bool类型
    uint32_t negativeIntScalar = *((uint32_t *)&BOOL_ATTN_MASK_SCALAR_VALUE);

    uint64_t actSeqLensKv = 0;
    uint64_t actSeqLensQ = 0;
    // ================================类成员变量====================================
    // 结构体
    const ConstInfoX &constInfo;
    TaskInfo taskInfo{};

private:
    // ================================FD Local Buffer区====================================
    TBuf<> combineSumBuf1;    // 1.5k: 16*24*4
    TBuf<> combineSumBuf2;    // 1.5k: 16*24*4
    TBuf<> combineMaxBuf1;    // 1.5k: 16*24*4
    TBuf<> combineMaxBuf2;    // 1.5k: 16*24*4
    TBuf<> combineLseExpBuf;  // 1.5k: 16*24*4
    TBuf<> combineMm2ResBuf1; // 32k: 16*512*4
    TBuf<> combineMm2ResBuf2; // 32k: 16*512*4
    TBuf<> combineReduceBuf;  // 32k: 16*512*4
    TBuf<> combineOutputBuf;  // 32k: 16*512*4

    TBuf<> combineLseMaxUbBuf1;
    TBuf<> combineLseMaxUbBuf2;
    TBuf<> combineLseUbBuf;

public:
    __aicore__ inline FaBlockVecCombine(ConstInfoX &constInfo) : constInfo(constInfo){};

    template <typename U> // 避免重名用U
    __aicore__ inline U Align(U num, U rnd)
    {
        return (((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd) * (rnd)));
    }

    __aicore__ inline void InitGlobalTensor(GlobalTensor<float> lseMaxCombineGm, GlobalTensor<float> lseSumCombineGm,
                                            GlobalTensor<float> accumOutGm, GlobalTensor<OUTPUT_T> attentionOutGm,
                                            GlobalTensor<uint64_t> actualSeqLengthsGmQ,
                                            GlobalTensor<uint64_t> actualSeqLengthsGm, __gm__ uint8_t *key)
    {
        this->lseMaxCombineGm = lseMaxCombineGm;
        this->lseSumCombineGm = lseSumCombineGm;
        this->accumOutGm = accumOutGm;
        this->attentionOutGm = attentionOutGm;
        this->actualSeqLengthsGmQ = actualSeqLengthsGmQ;
        this->actualSeqLengthsGm = actualSeqLengthsGm;
        this->keyPtr = key;
    }
    __aicore__ inline void InitParams()
    {
        this->dSizeV_Align = this->Align(constInfo.dSizeV, FP32_REPEAT_ELEMENT_NUM);
    }
    __aicore__ inline void InitBuffers(TPipe *pipe)
    {
        if ASCEND_IS_AIV {
            pipe->Reset();
            // InQue, DB, SYNC_LSE_MAX_SUM_BUF1_FLAG SYNC_LSE_MAX_SUM_BUF2_FLAG
            pipe->InitBuffer(combineSumBuf1, BUFFER_SIZE_BYTE_4K + BUFFER_SIZE_BYTE_2K);

            pipe->InitBuffer(combineSumBuf2, BUFFER_SIZE_BYTE_4K + BUFFER_SIZE_BYTE_2K);
            pipe->InitBuffer(combineMaxBuf1, BUFFER_SIZE_BYTE_4K + BUFFER_SIZE_BYTE_2K);
            pipe->InitBuffer(combineMaxBuf2, BUFFER_SIZE_BYTE_4K + BUFFER_SIZE_BYTE_2K);
            // TmpBuf
            pipe->InitBuffer(combineLseExpBuf, BUFFER_SIZE_BYTE_4K + BUFFER_SIZE_BYTE_2K);
            // InQue, DB, SYNC_MM2RES_BUF1_FLAG SYNC_MM2RES_BUF2_FLAG
            pipe->InitBuffer(combineMm2ResBuf1, BUFFER_SIZE_BYTE_16K);
            pipe->InitBuffer(combineMm2ResBuf2, BUFFER_SIZE_BYTE_16K);
            // TmpBuf
            pipe->InitBuffer(combineReduceBuf, BUFFER_SIZE_BYTE_16K);
            // OutQue, SYNC_FDOUTPUT_BUF_FLAG
            pipe->InitBuffer(combineOutputBuf, BUFFER_SIZE_BYTE_16K);

            pipe->InitBuffer(combineLseMaxUbBuf1, BUFFER_SIZE_BYTE_256B);
            pipe->InitBuffer(combineLseMaxUbBuf2, BUFFER_SIZE_BYTE_256B);
            // OutQue, SYNC_LSEOUTPUT_BUF_FLAG
            pipe->InitBuffer(combineLseUbBuf, BUFFER_SIZE_BYTE_256B);
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
    }
    __aicore__ inline void FreeEventID()
    {
        WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_LSE_MAX_SUM_BUF1_FLAG);
        WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_LSE_MAX_SUM_BUF2_FLAG);
        WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_MM2RES_BUF1_FLAG);
        WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_MM2RES_BUF2_FLAG);
        WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_FDOUTPUT_BUF_FLAG);
        WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_LSEOUTPUT_BUF_FLAG);
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
        LocalTensor<T> lseSum = (cntM & 1) == 0 ? combineSumBuf1.Get<T>() : combineSumBuf2.Get<T>();
        LocalTensor<T> lseMax = (cntM & 1) == 0 ? combineMaxBuf1.Get<T>() : combineMaxBuf2.Get<T>();

        uint64_t combineLseOffset = (baseOffset + startRow) * BLOCK_ELEMENT_NUM;
        uint64_t combineLoopOffset = s1BaseSize * BLOCK_ELEMENT_NUM;
        uint64_t dealRowCountAlign = dealRowCount * BLOCK_ELEMENT_NUM;

        for (uint32_t i = 0; i < taskInfo.actualCombineLoopSize; ++i) {
            DataCopy(lseSum[i * dealRowCountAlign], lseSumCombineGm[combineLseOffset + i * combineLoopOffset],
                     dealRowCountAlign); // 份数offset

            DataCopy(lseMax[i * dealRowCountAlign], lseMaxCombineGm[combineLseOffset + i * combineLoopOffset],
                     dealRowCountAlign);
        }
    }
    __aicore__ inline void ComputeScaleValue(LocalTensor<T> &lseExp, uint32_t dealRowCount,
                                             uint32_t actualCombineLoopSize, uint32_t cntM, uint32_t startRow)
    {
        LocalTensor<T> lseSum = (cntM & 1) == 0 ? combineSumBuf1.Get<T>() : combineSumBuf2.Get<T>();
        LocalTensor<T> lseMax = (cntM & 1) == 0 ? combineMaxBuf1.Get<T>() : combineMaxBuf2.Get<T>();
        LocalTensor<T> lseMaxUb = (cntM & 1) == 0 ? combineLseMaxUbBuf1.Get<T>() : combineLseMaxUbBuf2.Get<T>();

        LocalTensor<T> maxLseUb = combineLseUbBuf.Get<T>();
        ComputeScaleValue_VF_Combine(lseMax, lseSum, lseExp, maxLseUb, lseMaxUb, dealRowCount, actualCombineLoopSize);
    }

    __aicore__ inline void Bmm2DataCopyOutTrans(LocalTensor<OUTPUT_T> &attnOutUb, uint32_t startRow,
                                                uint32_t dealRowCount, uint32_t columnCount)
    {
        FaUbTensor<OUTPUT_T> ubTensor{
            .tensor = attnOutUb,
            .rowCount = dealRowCount,
            .colCount = columnCount,
        };
        GmCoord gmCoord{.bIdx = taskInfo.bIdx,
                        .n2Idx = taskInfo.n2Idx,
                        .gS1Idx = taskInfo.gS1Idx + startRow,
                        .dIdx = 0,
                        .gS1DealSize = dealRowCount,
                        .dDealSize = (uint32_t)constInfo.dSizeV};

        constexpr GmFormat OUT_FORMAT = GmFormat::BSNGD;
        FaGmTensor<OUTPUT_T, OUT_FORMAT> outGmTensor;
        outGmTensor.gmTensor = attentionOutGm;
        InitOffset(outGmTensor.offsetInfo, constInfo.bSize, constInfo.n2Size, constInfo.gSize, constInfo.s1Size,
                   constInfo.dSizeV);
        CopyAttnOutUbToGm<OUTPUT_T, OUT_FORMAT>(outGmTensor, ubTensor, gmCoord);
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
        LocalTensor<OUTPUT_T> tmpBmm2ResCastTensor = combineOutputBuf.Get<OUTPUT_T>();
        AscendC::PipeBarrier<PIPE_V>();
        DealInvalidRows(accumOutLocal, startRow, dealRowCount, this->dSizeV_Align);
        DealInvalidMaskRows(accumOutLocal, startRow, dealRowCount, this->dSizeV_Align, cntM);
        WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_FDOUTPUT_BUF_FLAG);
        uint32_t shapeArray[] = {dealRowCount, (uint32_t)constInfo.dSizeV};
        tmpBmm2ResCastTensor.SetShapeInfo(ShapeInfo(2, shapeArray, DataFormat::ND));
        if constexpr (IsSameType<OUTPUT_T, bfloat16_t>::value) {
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
        actSeqLensQ = constInfo.s1Size;
        actSeqLensKv = constInfo.s2Size;
        int64_t safePreToken = constInfo.preTokens;
        int64_t safeNextToken = constInfo.nextTokens;

        fa_base_vector::GetSafeActToken(actSeqLensQ, actSeqLensKv, safePreToken, safeNextToken, constInfo.sparseMode);

        if (constInfo.sparseMode == BAND) {
            preTokensPerBatch = safePreToken;
            nextTokensPerBatch = actSeqLensKv - actSeqLensQ + safeNextToken;
        } else if ((constInfo.sparseMode == DEFAULT_MASK) && attnMaskFlag) {
            nextTokensPerBatch = safeNextToken;
            preTokensPerBatch = actSeqLensKv - actSeqLensQ + safePreToken;
        } else {
            nextTokensPerBatch = actSeqLensKv - actSeqLensQ;
            preTokensPerBatch = 0;
        }
    }

    template <typename UBOUT_T>
    __aicore__ inline void DealInvalidRows(LocalTensor<UBOUT_T> &attnOutUb, uint32_t startRow, uint32_t dealRowCount,
                                           uint32_t columnCount)
    {
        if (!attnMaskFlag) {
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

        fa_base_vector::InvalidRows<UBOUT_T>(attnOutUb, params);
    }

    template <typename UBOUT_T>
    __aicore__ inline void DealInvalidMaskRows(LocalTensor<UBOUT_T> &attnOutUb, uint32_t startRow,
                                               uint32_t dealRowCount, uint32_t columnCount, uint32_t cntM)
    {
        if (!constInfo.isRowInvalidOpen || !attnMaskFlag) {
            return;
        }
        if (constInfo.sparseMode != DEFAULT_MASK && constInfo.sparseMode != ALL_MASK) {
            return;
        }
        LocalTensor<T> lseMaxUb = (cntM & 1) == 0 ? combineLseMaxUbBuf1.Get<T>() : combineLseMaxUbBuf2.Get<T>();
        fa_base_vector::InvalidMaskRows<UBOUT_T, T, true>(0, dealRowCount, columnCount, lseMaxUb, negativeIntScalar,
                                                          attnOutUb);
    }

public:
    __aicore__ inline void Combine(CombineParamsX &fd)
    {
        if (!fd.combineCoreEnable) {
            return;
        }
        uint32_t combineBalanceMBaseSize = 8U;
        uint32_t combineBalanceMSplitNum = (fd.mLen + combineBalanceMBaseSize - 1) / combineBalanceMBaseSize;
        uint32_t combineBalanceMTailSize =
            (fd.mLen % combineBalanceMBaseSize == 0) ? combineBalanceMBaseSize : fd.mLen % combineBalanceMBaseSize;

        uint32_t reduceGlobaLoop = 0;
        uint32_t reduceMLoop = 0;

        uint32_t tmpCombineS1gOuterMStart = 0;
        uint32_t tmpCombineS1gOuterMEnd = combineBalanceMSplitNum - 1;
        taskInfo.bIdx = fd.combineBN2Idx / constInfo.n2Size;
        taskInfo.n2Idx = fd.combineBN2Idx % constInfo.n2Size;
        taskInfo.gS1Idx = fd.combineMIdx * s1BaseSize;
        taskInfo.actualCombineLoopSize = fd.combineS2SplitNum; // 当前规约任务kv方向有几份
        uint64_t combineTaskPrefixSum = fd.combineWorkspaceIdx;
        uint64_t taskOffset = combineTaskPrefixSum * s1BaseSize;

        for (uint32_t combineS1gOuterMIdx = tmpCombineS1gOuterMStart; combineS1gOuterMIdx <= tmpCombineS1gOuterMEnd;
             ++combineS1gOuterMIdx) { // 左闭右闭
            uint32_t actualGSplitSize = combineBalanceMBaseSize;
            if (combineS1gOuterMIdx == combineBalanceMSplitNum - 1) {
                actualGSplitSize = combineBalanceMTailSize;
            }
            uint32_t startRow = fd.mStart + combineS1gOuterMIdx * combineBalanceMBaseSize;

            LocalTensor<T> lseExp = combineLseExpBuf.Get<T>();
            LocalTensor<T> reduceOut = combineReduceBuf.Get<T>();
            WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_LSE_MAX_SUM_BUF1_FLAG + (reduceMLoop & 1));
            CopyLseIn(startRow, actualGSplitSize, taskOffset, reduceMLoop);
            SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_LSE_MAX_SUM_BUF1_FLAG + (reduceMLoop & 1));
            WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_LSE_MAX_SUM_BUF1_FLAG + (reduceMLoop & 1));
            for (uint32_t preLoadIdx = 0; preLoadIdx < preLoadNum; ++preLoadIdx) {
                LocalTensor<T> mm2Res =
                    ((reduceGlobaLoop + preLoadIdx) & 1) == 0 ? combineMm2ResBuf1.Get<T>() : combineMm2ResBuf2.Get<T>();
                WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_MM2RES_BUF1_FLAG + ((reduceGlobaLoop + preLoadIdx) & 1));
                CopyAccumOutIn(mm2Res, preLoadIdx, taskOffset + startRow, actualGSplitSize);
                SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_MM2RES_BUF1_FLAG + ((reduceGlobaLoop + preLoadIdx) & 1));
            }
            ComputeScaleValue(lseExp, actualGSplitSize, taskInfo.actualCombineLoopSize, reduceMLoop, startRow);
            CalcPreNextTokens();
            SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_LSE_MAX_SUM_BUF1_FLAG + (reduceMLoop & 1));

            for (uint32_t i = 0; i < taskInfo.actualCombineLoopSize; ++i) {
                LocalTensor<T> mm2Res =
                    (reduceGlobaLoop & 1) == 0 ? combineMm2ResBuf1.Get<T>() : combineMm2ResBuf2.Get<T>();
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
            CopyFinalResOut(reduceOut, startRow, actualGSplitSize, reduceMLoop);
            reduceMLoop += 1;
        }
    }
};

template <typename INPUT_T, typename T, typename OUTPUT_T, LayOutTypeEnum layout = LayOutTypeEnum::None,
          LayOutTypeEnum outLayout = LayOutTypeEnum::None, S1TemplateType s1TemplateType = S1TemplateType::Aligned128,
          S2TemplateType s2TemplateType = S2TemplateType::Aligned128,
          DTemplateType dTemplateType = DTemplateType::Aligned128,
          DTemplateType dVTemplateType = DTemplateType::Aligned128, bool hasAttn = false>
class FaBlockVecCombineDummy {
public:
    using ConstInfoX = ConstInfo_t<FaKernelType::NO_QUANT>;
    __aicore__ inline FaBlockVecCombineDummy(ConstInfoX &constInfo){};
};

} // namespace BaseApi
#endif
