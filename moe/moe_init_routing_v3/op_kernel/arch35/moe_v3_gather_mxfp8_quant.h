/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_v3_gather_mxfp8_quant.h
 * \brief
 */
#ifndef MOE_V3_GATHER_MXFP8_QUANT_H_REGBASE
#define MOE_V3_GATHER_MXFP8_QUANT_H_REGBASE

#include "moe_v3_common.h"
#include "op_kernel/platform_util.h"
#include "op_kernel/math_util.h"

namespace MoeInitRoutingV3 {
using namespace AscendC;

template <typename T, typename U, bool ClampAmax = false>
class MoeGatherOutMxfp8Quant {
public:
    __aicore__ inline MoeGatherOutMxfp8Quant(){};
    __aicore__ inline void Init(GM_ADDR xAddr, GM_ADDR unused_ScaleAddr, GM_ADDR expandedRowIdxAddr,
                                GM_ADDR expandedXAddr, GM_ADDR expandedScaleAddr, GM_ADDR sortedExpertIdxAddr,
                                const MoeInitRoutingV3Arch35TilingData *tilingData, TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitKernelTiling(GM_ADDR sortedExpertIdxAddr, const MoeInitRoutingV3Arch35TilingData *tilingData);
    __aicore__ inline void CopyInExpandedExpertIdx(int64_t progress);
    __aicore__ inline void CopyExpandedXandMXQuant(int64_t progress);
    __aicore__ inline void CopyIn(int64_t srcIdx, int64_t colIdx, int64_t loopCols);
    __aicore__ inline void Compute(uint32_t xElemNum, uint32_t scaleElemNum, uint32_t validScaleElemNum);
    __aicore__ inline void CopyOut(int64_t dstIdx, int64_t colIdx, int64_t loopCols, int64_t loopScaleCols,
                                   int64_t loopValidScaleCols);

private:
    TPipe *pipe_;
    TQue<QuePosition::VECIN, 1> xInQueue_;
    TQue<QuePosition::VECIN, 1> sortedRowIdxInQueue_;
    TQue<QuePosition::VECOUT, 1> xQuantOutQueue_;
    TQue<QuePosition::VECOUT, 1> mxScaleOutQueue_;
    TBuf<QuePosition::VECCALC> maxExpBuffer_;
    TBuf<QuePosition::VECCALC> invScaleBuffer_;

    GlobalTensor<T> xInGm_;
    GlobalTensor<uint8_t> expandedXOutGm_;
    GlobalTensor<uint8_t> expandedScaleOutGm_;
    GlobalTensor<int32_t> sortedRowIdxGm_;
    GlobalTensor<int32_t> expertTotalCountGm_;

    const MoeV3Arch35GatherOutComputeTilingData *gatherOutTilingData_;

    int64_t needCoreNum_;
    int64_t blockIdx_;
    int64_t cols_;
    int64_t validScaleCols_; // 一个token实际有意义的scale有多少个元素（列）
    int64_t scaleCols_; // 一个token的scale有多少个元素（列），即CeilAlign(CeliDiv(h,32),2)，在actualScaleCols_为奇数时，scaleCols_=validScaleCols_+1
    int64_t n_;
    int64_t k_;
    int64_t perCoreRow_;
    int64_t currentLoopRows_;
    int64_t coreRows_;
    int64_t perLoopRows_;
    int64_t lastLoopRows_;
    int64_t rowLoops_;
    int64_t perLoopCols_;
    int64_t lastLoopCols_;
    int64_t colLoops_;
    int64_t perLoopScaleCols_;
    int64_t lastLoopValidScaleCols_;
    int64_t lastLoopScaleCols_;
    int64_t indicesOffset_;
    int64_t rowIdxType_ = 0;

    // 会赋值为计算maxExp时，maxExp对应不同的目标fp8类型U的下限
    uint16_t lowerBoundOfB16MaxExp_ = 0;

    // 一个RegTensor的长度Bytes
    const uint32_t vRegSize_ = Ops::Base::GetVRegSize();
    // ub的一个块大小，通常是32B
    const uint32_t ubBlockSize_ = Ops::Base::GetUbBlockSize();
    // 一个RegTensor能塞下多少T(B16)类型的元素
    const uint32_t vlForB16_ = vRegSize_ / sizeof(T);
    // 一个RegTensor有几个UbBlock，即ReduceWithBlock后的元素个数（256B/32B=8）
    // 这里如果ubBlockSize_获取的值为0，应该让kernel挂掉
    const uint32_t numUbBlocksPerVReg_ = vRegSize_ / ubBlockSize_; // 8个元素
    // 每个ubBlock对应多少T类型的元素
    const uint32_t numElemPerUbBlock_ = ubBlockSize_ / sizeof(T);
};

template <typename T>
__aicore__ inline void LoadInputData(RegTensor<float>& dst, __local_mem__ T* src, MaskReg pregLoop, uint32_t srcOffset)
{
    if constexpr (IsSameType<T, float>::value) {
        DataCopy(dst, src + srcOffset);
    } else if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
        RegTensor<T> tmp;
        DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(tmp, src + srcOffset);
        Cast<float, T, castTraitB162B32Even>(dst, tmp, pregLoop);
    }
}

template <typename T>
__aicore__ inline void StoreOutputData(__local_mem__ T* dst,
                                       RegTensor<float>& src, MaskReg pregLoop, uint32_t dstOffset)
{
    if constexpr (IsSameType<T, float>::value) {
        DataCopy(dst + dstOffset, src, pregLoop);
    } else if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
        RegTensor<T> tmp;
        Cast<T, float, castTraitB322B16Even>(tmp, src, pregLoop);
        DataCopy<T, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(dst + dstOffset, tmp, pregLoop);
    } else if constexpr (IsSameType<T, fp8_e4m3fn_t>::value || IsSameType<T, fp8_e5m2_t>::value) {
        RegTensor<T> tmp;
        Cast<T, float, castTraitF32toFp8Even>(tmp, src, pregLoop);
        DataCopy<T, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(dst + dstOffset, tmp, pregLoop);
    }
}

template <typename T>
__aicore__ inline void StoreOuputDataUnalign(RegTensor<float>& src, __local_mem__ T*& dst, UnalignReg& uDst,
                                             MaskReg pregLoop, uint32_t postUpdateStride)
{
    if constexpr (IsSameType<T, float>::value) {
        DataCopyUnAlign(dst, src, uDst, postUpdateStride);
    } else if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
        RegTensor<T> tmp;
        RegTensor<T> tmpPack;
        Cast<T, float, castTraitB322B16Even>(tmp, src, pregLoop);
        Pack((RegTensor<uint16_t>&)tmpPack, (RegTensor<uint32_t>&)tmp);
        DataCopyUnAlign(dst, tmpPack, uDst, postUpdateStride);
    }
}

template <typename T, typename U, bool ClampAmax>
__aicore__ inline void VFProcessSwigluMxFp8InvScale(const LocalTensor<float>& yLocal,
                                                    const LocalTensor<uint8_t>& scaleLocal,
                                                    const LocalTensor<float>& invScaleLocal,
                                                    const LocalTensor<T>& x0Local,
                                                    const uint16_t curRowNum, const uint32_t curColNum)
{
    __local_mem__ float* yLocalAddr = (__local_mem__ float*)yLocal.GetPhyAddr();
    __local_mem__ uint8_t* scaleOriginLocalAddr = (__local_mem__ uint8_t*)scaleLocal.GetPhyAddr();
    __local_mem__ float* invScaleLocalAddr = (__local_mem__ float*)invScaleLocal.GetPhyAddr();
    __local_mem__ T* x0LocalAddr = (__local_mem__ T*)x0Local.GetPhyAddr();
    __local_mem__ uint8_t* scaleLocalAddr = scaleOriginLocalAddr;
    uint32_t maxValueInt = 0;
    if constexpr (IsSameType<U, fp8_e5m2_t>::value) {
        maxValueInt = INV_FP8_E5M2_MAX_VALUE;
    } else if constexpr (IsSameType<U, fp8_e4m3fn_t>::value) {
        maxValueInt = INV_FP8_E4M3_MAX_VALUE;
    }
    uint16_t vlLen = REPEAT_SIZE / sizeof(T);
    uint16_t loopCount = CeilDiv(curColNum, vlLen);
    uint32_t curColNumAlignT = RoundUp<T>(curColNum);
    uint32_t curColNumAlignFloat = RoundUp<float>(curColNum);
    uint32_t scaleColNum = CeilDiv(curColNum, 32);
    uint32_t invScaleColNumAlign = RoundUp<float>(curColNumAlignT / 8);

    __VEC_SCOPE__
    {
        RegTensor<float> one;
        RegTensor<uint32_t> oneUint32;
        RegTensor<float> zero;
        RegTensor<uint32_t> zeroUint32;
        RegTensor<T> x0;
        RegTensor<float> x0Layout0;
        RegTensor<float> x0Layout1;
        RegTensor<float> yLayout0; // 奇偶交错 偶数部分
        RegTensor<float> yLayout1; // 奇偶交错 奇数部分
        RegTensor<float> y0; // 还原交错逻辑 高64位
        RegTensor<float> y1; // 还原交错逻辑 低64位
        RegTensor<float> yMax0;
        RegTensor<float> yMax1;
        RegTensor<float> yMax1Layout0;
        RegTensor<float> yMax1Layout1;
        RegTensor<float> scale;
        RegTensor<float> clampScale;
        RegTensor<float> invScale;
        RegTensor<float> invScale0;
        RegTensor<float> invScale1;
        RegTensor<float> dupInvScale;
        RegTensor<uint32_t> scale2;
        RegTensor<uint32_t> scale3;
        RegTensor<int32_t> scale4;
        RegTensor<int32_t> scale5;
        RegTensor<uint8_t> scale6;
        RegTensor<uint16_t> scale7;
        RegTensor<uint32_t> coeff;
        RegTensor<float> tmp;
        RegTensor<uint32_t> tmp0;
        RegTensor<uint32_t> tmp1;
        RegTensor<uint32_t> tmp2;
        RegTensor<uint32_t> tmp3;
        RegTensor<int32_t> tmp4;
        RegTensor<uint8_t> tmp5;
        UnalignReg uReg;
        uint32_t sreg = (REPEAT_SIZE / sizeof(T)) / 32;
        uint32_t sreg1 = 4 * (REPEAT_SIZE / sizeof(T)) / 32;
        MaskReg pregMain0 = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregMain1 = CreateMask<T, MaskPattern::ALL>();
        MaskReg pregMerge = UpdateMask<float>(sreg);
        MaskReg pregMerge1 = UpdateMask<float>(sreg1);
        MaskReg compareMask0;
        Duplicate(one, 1.0f, pregMain0);
        Duplicate(zeroUint32, static_cast<uint32_t>(0), pregMain0);
        Duplicate(zero, 0.0f, pregMain0);
        Duplicate(oneUint32, static_cast<uint32_t>(1), pregMain0);
        Duplicate(coeff, maxValueInt, pregMain0);
        Duplicate(tmp0, FAST_LOG_AND_VALUE1, pregMerge);
        Duplicate(tmp1, FAST_LOG_AND_VALUE2, pregMerge);
        Duplicate(tmp3, static_cast<uint32_t>(127), pregMerge);
        Duplicate(tmp4, static_cast<int32_t>(127), pregMerge);
        for (uint16_t i = 0; i < curRowNum; i++) {
            for (uint16_t j = 0; j < loopCount; j++) {
                DataCopy(x0, x0LocalAddr + i * curColNumAlignT + j * vlLen);
                Cast<float, T, traitB16ToB32Layout0>(x0Layout0, x0, pregMain1);
                Cast<float, T, traitB16ToB32Layout1>(x0Layout1, x0, pregMain1);

                // 合并奇偶位置
                Interleave(y0, y1, x0Layout0, x0Layout1);
                StoreOutputData<float>(yLocalAddr, y0, pregMain0, 2 * j * VL_FP32 + i * curColNumAlignFloat);
                StoreOutputData<float>(yLocalAddr, y1, pregMain0, (2 * j + 1) * VL_FP32 + i * curColNumAlignFloat);

                Abs(x0Layout0, x0Layout0, pregMain0);
                Abs(x0Layout1, x0Layout1, pregMain0);

                // fp32场景，32个数对应4个Block；先做一次Max，接着ReduceMaxWithBlock，然后DeInterLeave并且Max，得到每4个Block的最大值
                Max(yMax0, x0Layout0, x0Layout1, pregMain0); // 4 --> 2
                ReduceMaxWithDataBlock(yMax1, yMax0, pregMain0);
                DeInterleave(yMax1Layout0, yMax1Layout1, yMax1, yMax1);
                Max(scale, yMax1Layout0, yMax1Layout1, pregMerge);
                if constexpr (ClampAmax) {
                    Maxs(clampScale, scale, 0.0001f, pregMerge); // amax clamp
                    Mul(scale, clampScale, (RegTensor<float>&)coeff, pregMerge); // sf = amax / 448.0
                } else {
                    Mul(scale, scale, (RegTensor<float>&)coeff, pregMerge); // sf = amax / 448.0
                }

                // 量化逻辑
                ShiftRights(scale2, (RegTensor<uint32_t>&)scale, static_cast<int16_t>(FAST_LOG_SHIFT_BITS), pregMerge);
                And(scale2, scale2, tmp0, pregMerge); // exp
                And(scale3, (RegTensor<uint32_t>&)scale, tmp1, pregMerge); // man_bits
                Compare<uint32_t, AscendC::CMPMODE::NE>(compareMask0, scale3, zeroUint32, pregMerge);
                Select(tmp2, oneUint32, zeroUint32, compareMask0); // man_bits != 0
                Sub(scale3, scale2, tmp3, pregMerge); // exp - 127
                Add((RegTensor<uint32_t>&)scale4, scale3, tmp2, pregMerge);
                Adds(scale5, scale4, 127, pregMerge); // sf_uint32

                Cast<uint8_t, int32_t, castTraitU32toU8Even>(tmp5, scale5, pregMerge);
                Pack(scale7, (RegTensor<uint32_t>&)tmp5);
                Pack(scale6, scale7);
                DataCopyUnAlign<uint8_t, PostLiteral::POST_MODE_UPDATE>(scaleLocalAddr, scale6, uReg, 4);

                Sub(scale5, tmp4, scale4, pregMerge); // 127 - exp_scale
                ShiftLefts((RegTensor<int32_t>&)invScale, scale5, static_cast<int16_t>(23), pregMerge);
                Interleave(invScale0, invScale1, invScale, invScale);
                Interleave(invScale1, invScale, invScale0, invScale0);
                StoreOutputData<float>(invScaleLocalAddr, invScale1, pregMerge1, j * 16 + i * invScaleColNumAlign);
            }
            DataCopyUnAlignPost(scaleLocalAddr, uReg, 0);
        }
    }
}

template <typename T>
__aicore__ inline void VFProcessSwigluMxFp8Quant(const LocalTensor<T>& yQuantLocal, const LocalTensor<float>& yLocal,
                                                 const LocalTensor<float>& invScaleLocal, const uint16_t curRowNum,
                                                 const uint32_t curColNum)
{
    __local_mem__ T* yQuantLocalAddr = (__local_mem__ T*)yQuantLocal.GetPhyAddr();
    __local_mem__ float* yLocalAddr = (__local_mem__ float*)yLocal.GetPhyAddr();
    __local_mem__ float* scaleLocalAddr = (__local_mem__ float*)invScaleLocal.GetPhyAddr();

    uint16_t loopCount = CeilDiv(curColNum, VL_FP32);
    uint32_t curColNumAlign = RoundUp<float>(curColNum);
    uint32_t dstCurColNumAlign = RoundUp<T>(curColNum);
    uint32_t invScaleColNumAlign = RoundUp<float>(CeilDiv(curColNum, 8));
    __VEC_SCOPE__
    {
        RegTensor<float> y;
        RegTensor<float> invScale;
        RegTensor<float> dupInvScale;
        MaskReg pregMain = CreateMask<float, MaskPattern::ALL>();
        for (uint16_t i = 0; i < curRowNum; i++) {
            for (uint16_t j = 0; j < loopCount; j++) {
                LoadInputData<float>(y, yLocalAddr, pregMain, i * curColNumAlign + j * VL_FP32);
                DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_E2B_B32>(invScale,
                         scaleLocalAddr + j * 8 + i * invScaleColNumAlign);
                Mul(y, y, invScale, pregMain);
                StoreOutputData<T>(yQuantLocalAddr, y, pregMain, j * VL_FP32 + i * dstCurColNumAlign);
            }
        }
    }
}

template <typename T, typename U, bool ClampAmax>
__aicore__ inline void MoeGatherOutMxfp8Quant<T, U, ClampAmax>::Init(GM_ADDR xAddr, GM_ADDR unused_ScaleAddr,
                                                          GM_ADDR sortedExpertIdxAddr, GM_ADDR expandedRowIdxAddr,
                                                          GM_ADDR expandedXAddr, GM_ADDR expandedScaleAddr,
                                                          const MoeInitRoutingV3Arch35TilingData *tilingData, TPipe *tPipe)
{
#if (__NPU_ARCH__ == 3510)
    SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(0);
#endif

    pipe_ = tPipe;
    blockIdx_ = GetBlockIdx();
    InitKernelTiling(sortedExpertIdxAddr, tilingData);

    xInGm_.SetGlobalBuffer((__gm__ T *)xAddr);
    expandedXOutGm_.SetGlobalBuffer((__gm__ uint8_t *)expandedXAddr);
    if (rowIdxType_ == SCATTER) {
        sortedRowIdxGm_.SetGlobalBuffer((__gm__ int32_t *)expandedRowIdxAddr + blockIdx_ * perCoreRow_,
                                        Align(perCoreRow_, sizeof(int32_t)));
    } else {
        sortedRowIdxGm_.SetGlobalBuffer((__gm__ int32_t *)sortedExpertIdxAddr + Align(n_ * k_, sizeof(int32_t)) +
                                            blockIdx_ * perCoreRow_,
                                        Align(perCoreRow_, sizeof(int32_t)));
    }
    expandedScaleOutGm_.SetGlobalBuffer((__gm__ uint8_t *)expandedScaleAddr);

    // perrows * 2 * 2 * 4 expandRowIdx + sortedExpertId
    pipe_->InitBuffer(sortedRowIdxInQueue_, 1, AlignBytes(perLoopRows_, sizeof(int32_t)));
    pipe_->InitBuffer(xInQueue_, 1, AlignBytes(perLoopCols_, sizeof(T)));
    pipe_->InitBuffer(xQuantOutQueue_, 1, AlignBytes(perLoopCols_ / 4, sizeof(int8_t)) * 4);
    pipe_->InitBuffer(mxScaleOutQueue_, 1, AlignBytes(perLoopScaleCols_, sizeof(int8_t)));
    pipe_->InitBuffer(maxExpBuffer_, AlignBytes(perLoopCols_, sizeof(float)));
    pipe_->InitBuffer(invScaleBuffer_, AlignBytes(perLoopScaleCols_, sizeof(float)));
    if constexpr (IsSameType<U, fp8_e4m3fn_t>::value) {
        lowerBoundOfB16MaxExp_ = LOWER_BOUND_OF_MAX_EXP_FOR_E4M3;
    } else {
        lowerBoundOfB16MaxExp_ = LOWER_BOUND_OF_MAX_EXP_FOR_E5M2;
    }
}

template <typename T, typename U, bool ClampAmax>
__aicore__ inline void MoeGatherOutMxfp8Quant<T, U, ClampAmax>::InitKernelTiling(GM_ADDR sortedExpertIdxAddr, const MoeInitRoutingV3Arch35TilingData *tilingData)
{
    gatherOutTilingData_ = &(tilingData->gatherOutComputeParamsOp);
    cols_ = tilingData->cols;
    validScaleCols_ = Ops::Base::CeilDiv<int64_t>(cols_, MX_BLOCK_SIZE);
    scaleCols_ = Ops::Base::CeilAlign<int64_t>(validScaleCols_, 2); // CeilDiv(h, 32)后再向上到2的倍数（偶数）
    n_ = tilingData->n;
    k_ = tilingData->k;
    rowIdxType_ = tilingData->rowIdxType;

    // core split
    int64_t actualExpertNum_ = tilingData->actualExpertNum;
    expertTotalCountGm_.SetGlobalBuffer((__gm__ int32_t *)sortedExpertIdxAddr + Align(n_ * k_, sizeof(int32_t)) * 2 +
                                            Align(actualExpertNum_, sizeof(int32_t)),
                                        actualExpertNum_);
    int64_t expertTotalCount_ = expertTotalCountGm_.GetValue(0);
    perCoreRow_ = Ceil(expertTotalCount_, tilingData->coreNum);
    needCoreNum_ = Ceil(expertTotalCount_, perCoreRow_);
    int64_t lastCoreIndicesElements = expertTotalCount_ - (needCoreNum_ - 1) * perCoreRow_;

    // inner core split
    coreRows_ = perCoreRow_;
    int64_t originPerLoopElements = gatherOutTilingData_->perCorePerLoopIndicesElements;
    if (blockIdx_ == needCoreNum_ - 1) {
        coreRows_ = lastCoreIndicesElements;
        originPerLoopElements = gatherOutTilingData_->lastCorePerLoopIndicesElements;
    }
    perLoopRows_ = Min(coreRows_, originPerLoopElements);
    rowLoops_ = Ceil(coreRows_, perLoopRows_);
    lastLoopRows_ = coreRows_ - (rowLoops_ - 1) * perLoopRows_;

    // cols split
    perLoopCols_ = gatherOutTilingData_->perLoopCols;
    lastLoopCols_ = gatherOutTilingData_->lastLoopCols;
    colLoops_ = gatherOutTilingData_->colsLoops;
    perLoopScaleCols_ = perLoopCols_ / MX_BLOCK_SIZE; // perLoopCols_在tiling侧计算，已经对齐到32的整数倍了
    lastLoopValidScaleCols_ = validScaleCols_ - (colLoops_ - 1) * perLoopScaleCols_;
    lastLoopScaleCols_ = scaleCols_ - (colLoops_ - 1) * perLoopScaleCols_;
}

template <typename T, typename U, bool ClampAmax>
__aicore__ inline void MoeGatherOutMxfp8Quant<T, U, ClampAmax>::Process()
{
    if (blockIdx_ < needCoreNum_) {
        currentLoopRows_ = perLoopRows_;
        for (int64_t loop = 0; loop < rowLoops_ - 1; loop++) {
            CopyInExpandedExpertIdx(loop);
            CopyExpandedXandMXQuant(loop);
        }

        currentLoopRows_ = lastLoopRows_;
        CopyInExpandedExpertIdx(rowLoops_ - 1);
        CopyExpandedXandMXQuant(rowLoops_ - 1);
    }
}

template <typename T, typename U, bool ClampAmax>
__aicore__ inline void MoeGatherOutMxfp8Quant<T, U, ClampAmax>::CopyInExpandedExpertIdx(int64_t progress)
{
    indicesOffset_ = progress * perLoopRows_;
    LocalTensor<int32_t> indicesLocal = sortedRowIdxInQueue_.AllocTensor<int32_t>();
    DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(currentLoopRows_ * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams<int32_t> dataCopyPadParams{false, 0, 0, 0};
    DataCopyPad(indicesLocal, sortedRowIdxGm_[indicesOffset_], dataCopyParams, dataCopyPadParams);
    sortedRowIdxInQueue_.EnQue<int32_t>(indicesLocal);
}

template <typename T, typename U, bool ClampAmax>
__aicore__ inline void MoeGatherOutMxfp8Quant<T, U, ClampAmax>::CopyExpandedXandMXQuant(int64_t progress)
{
    LocalTensor<int32_t> indicesLocal = sortedRowIdxInQueue_.DeQue<int32_t>();
    SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
    for (int64_t index = 0; index < currentLoopRows_; index++) {
        // 对加载进ub的indicesLocal按行循环
        int32_t srcIdx = indicesLocal.GetValue(index);
        int64_t dstIdx = perCoreRow_ * blockIdx_ + perLoopRows_ * progress + index;
        for (int64_t j = 0; j < colLoops_; j++) {
            // 每行切分成cols，按cols读入-计算量化-拷出
            int64_t loopCols = (j == colLoops_ - 1) ? lastLoopCols_ : perLoopCols_;
            uint32_t loopScaleCols = (j == colLoops_ - 1) ? lastLoopScaleCols_ : perLoopScaleCols_;
            uint32_t loopValidScaleCols = (j == colLoops_ - 1) ? lastLoopValidScaleCols_ : perLoopScaleCols_;
            CopyIn(srcIdx / k_, j, loopCols);
            Compute(loopCols, loopScaleCols, loopValidScaleCols);
            CopyOut(dstIdx, j, loopCols, loopScaleCols, loopValidScaleCols);
        }
    }
}

template <typename T, typename U, bool ClampAmax>
__aicore__ inline void MoeGatherOutMxfp8Quant<T, U, ClampAmax>::CopyIn(int64_t srcIdx, int64_t colIdx, int64_t loopCols)
{
    LocalTensor<T> inLocal = xInQueue_.AllocTensor<T>();
    DataCopyExtParams copyInParam = {1, static_cast<uint32_t>(loopCols * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams = {false, 0, 0, 0};
    int64_t loopColsTail = loopCols % MX_BLOCK_SIZE;
    if (loopColsTail != 0) {
        padParams.isPad = true;
        if (loopColsTail > numElemPerUbBlock_) {
            padParams.rightPadding = numElemPerUbBlock_ - (loopColsTail - numElemPerUbBlock_);
        } else {
            padParams.rightPadding = numElemPerUbBlock_;
        }
    }
    DataCopyPad(inLocal, xInGm_[srcIdx * cols_ + colIdx * perLoopCols_], copyInParam, padParams);
    xInQueue_.EnQue(inLocal);
}

template <typename T, typename U, bool ClampAmax>
__aicore__ inline void MoeGatherOutMxfp8Quant<T, U, ClampAmax>::Compute(uint32_t xElemNum, uint32_t scaleElemNum, uint32_t validScaleElemNum)
{
    // deque input
    LocalTensor<T> xLocal = xInQueue_.DeQue<T>();
    auto xLocalAddr = reinterpret_cast<__ubuf__ T *>(xLocal.GetPhyAddr());
    // alloc outputs
    LocalTensor<int8_t> xQuantLocal = xQuantOutQueue_.AllocTensor<int8_t>();
    LocalTensor<U> xQuantRecastLocal = xQuantLocal.ReinterpretCast<U>();
    LocalTensor<uint8_t> mxScaleLocal = mxScaleOutQueue_.AllocTensor<uint8_t>();
    // get tmp buffers
    LocalTensor<float> maxExpLocal = maxExpBuffer_.Get<float>();
    LocalTensor<float> invScaleLocal = invScaleBuffer_.Get<float>();

    uint16_t vfLoopNumForX = (xElemNum + vlForB16_ - 1) / vlForB16_;
    uint16_t vfLoopNumForScale = (scaleElemNum + vlForB16_ - 1) / vlForB16_;

    VFProcessSwigluMxFp8InvScale<T, U, ClampAmax>(maxExpLocal, mxScaleLocal, invScaleLocal, xLocal, 1, xElemNum);
    VFProcessSwigluMxFp8Quant<U>(xQuantRecastLocal, maxExpLocal, invScaleLocal, 1, xElemNum);

    // free input
    xInQueue_.FreeTensor(xLocal);
    // enque outputs
    xQuantOutQueue_.EnQue(xQuantLocal);
    mxScaleOutQueue_.EnQue(mxScaleLocal);
}

template <typename T, typename U, bool ClampAmax>
__aicore__ inline void MoeGatherOutMxfp8Quant<T, U, ClampAmax>::CopyOut(int64_t dstIdx, int64_t colIdx, int64_t loopCols,
                                                             int64_t loopScaleCols, int64_t loopValidScaleCols)
{
    LocalTensor<uint8_t> mxScaleLocal = mxScaleOutQueue_.DeQue<uint8_t>();
    LocalTensor<uint8_t> outLocal = xQuantOutQueue_.DeQue<uint8_t>();

    DataCopyExtParams copyOutParams = {1, static_cast<uint32_t>(loopCols * sizeof(uint8_t)), 0, 0, 0};
    DataCopyPad<uint8_t>(expandedXOutGm_[dstIdx * cols_ + colIdx * perLoopCols_], outLocal, copyOutParams);

    DataCopyExtParams copyScaleParams = {1, static_cast<uint32_t>(loopValidScaleCols * sizeof(uint8_t)), 0, 0, 0};
    DataCopyPad<uint8_t>(expandedScaleOutGm_[dstIdx * scaleCols_ + colIdx * perLoopScaleCols_], mxScaleLocal,
                         copyScaleParams);
    if (loopScaleCols > loopValidScaleCols) {
        auto paddingOffset = dstIdx * scaleCols_ + colIdx * perLoopScaleCols_ + loopValidScaleCols;
        GlobalTensor<uint8_t> zeroScaleGm;
        zeroScaleGm.SetGlobalBuffer((__gm__ uint8_t *)expandedScaleOutGm_.GetPhyAddr() + paddingOffset,
                                    loopScaleCols - loopValidScaleCols);
        InitGlobalMemory(zeroScaleGm, loopScaleCols - loopValidScaleCols, static_cast<uint8_t>(0));
    }

    xQuantOutQueue_.FreeTensor(outLocal);
    mxScaleOutQueue_.FreeTensor(mxScaleLocal);
}

} // namespace MoeInitRoutingV3
#endif // MOE_V3_GATHER_MXFP8_QUANT_H_REGBASE
