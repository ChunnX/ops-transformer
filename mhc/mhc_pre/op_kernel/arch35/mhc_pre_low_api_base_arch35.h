#ifndef MHC_PRE_LOW_API_BASE_ARCH35_H
#define MHC_PRE_LOW_API_BASE_ARCH35_H

#include "kernel_operator.h"

namespace MhcPre {

using namespace AscendC;

constexpr uint32_t MHC_PRE_LOW_API_BLOCK_SIZE = 32;
constexpr uint32_t MHC_PRE_LOW_API_C0_SIZE = 16;
constexpr uint32_t MHC_PRE_LOW_API_N4 = 4;
constexpr uint32_t MHC_PRE_LOW_API_PRE_OFFSET = 0;
constexpr uint32_t MHC_PRE_LOW_API_POST_OFFSET_N4 = 4;
constexpr uint32_t MHC_PRE_LOW_API_RES_OFFSET_N4 = 8;
constexpr uint32_t MHC_PRE_LOW_API_FUSION_N4 = 24;
constexpr uint32_t MHC_PRE_LOW_API_VL_FP32 = 64;

constexpr AscendC::MicroAPI::CastTrait MHC_PRE_LOW_API_CAST_B16_TO_B32 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};

constexpr AscendC::MicroAPI::CastTrait MHC_PRE_LOW_API_CAST_B32_TO_B16 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

__aicore__ inline uint32_t LowApiCeilDiv(uint32_t value, uint32_t div)
{
    return (value + div - 1) / div;
}

__aicore__ inline uint32_t LowApiAlign(uint32_t value, uint32_t align)
{
    return LowApiCeilDiv(value, align) * align;
}

template <typename T>
__aicore__ inline uint32_t LowApiRoundUp(uint32_t value)
{
    return LowApiAlign(value, MHC_PRE_LOW_API_BLOCK_SIZE / sizeof(T));
}

template <typename T>
__aicore__ inline void MhcPreLowApiCastToFloat(const LocalTensor<float>& dstLocal,
    const LocalTensor<T>& srcLocal, uint16_t rowCount, uint16_t colCount)
{
    __local_mem__ float* dst = (__local_mem__ float*)dstLocal.GetPhyAddr();
    __local_mem__ T* src = (__local_mem__ T*)srcLocal.GetPhyAddr();
    uint16_t loopCount = LowApiCeilDiv(colCount, MHC_PRE_LOW_API_VL_FP32);
    uint16_t srcStride = LowApiRoundUp<T>(colCount);
    uint16_t dstStride = LowApiRoundUp<float>(colCount);
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<float> dstReg;
        AscendC::MicroAPI::RegTensor<T> srcReg;
        AscendC::MicroAPI::MaskReg mask;
        uint32_t remaining;
        for (uint16_t row = 0; row < rowCount; ++row) {
            remaining = colCount;
            for (uint16_t loop = 0; loop < loopCount; ++loop) {
                mask = AscendC::MicroAPI::UpdateMask<float>(remaining);
                AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                    srcReg, src + row * srcStride + loop * MHC_PRE_LOW_API_VL_FP32);
                AscendC::MicroAPI::Cast<float, T, MHC_PRE_LOW_API_CAST_B16_TO_B32>(dstReg, srcReg, mask);
                AscendC::MicroAPI::DataCopy(dst + row * dstStride + loop * MHC_PRE_LOW_API_VL_FP32,
                    dstReg, mask);
            }
        }
    }
}

template <typename T>
__aicore__ inline void MhcPreLowApiLoad(AscendC::MicroAPI::RegTensor<float>& dst,
    __local_mem__ T* src, AscendC::MicroAPI::MaskReg mask, uint32_t offset)
{
    AscendC::MicroAPI::RegTensor<T> tmp;
    AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(tmp, src + offset);
    AscendC::MicroAPI::Cast<float, T, MHC_PRE_LOW_API_CAST_B16_TO_B32>(dst, tmp, mask);
}

__aicore__ inline void MhcPreLowApiLoadBroadcast(AscendC::MicroAPI::RegTensor<float>& dst,
    __local_mem__ float* src, AscendC::MicroAPI::MaskReg mask, uint32_t offset)
{
    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(dst, src + offset);
}

template <typename T>
__aicore__ inline void MhcPreLowApiStore(__local_mem__ T* dst,
    AscendC::MicroAPI::RegTensor<float>& src, AscendC::MicroAPI::MaskReg mask, uint32_t offset)
{
    AscendC::MicroAPI::RegTensor<T> tmp;
    AscendC::MicroAPI::Cast<T, float, MHC_PRE_LOW_API_CAST_B32_TO_B16>(tmp, src, mask);
    AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(dst + offset, tmp, mask);
}

template <typename T>
__aicore__ inline void LowApiCopyIn(const GlobalTensor<T>& inputGm, const LocalTensor<T>& inputLocal,
    uint16_t blockCount, uint32_t copyLen, uint32_t srcStride = 0)
{
    DataCopyExtParams copyParams;
    copyParams.blockCount = blockCount;
    copyParams.blockLen = copyLen * sizeof(T);
    copyParams.srcStride = srcStride * sizeof(T);
    copyParams.dstStride = 0;
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(inputLocal, inputGm, copyParams, padParams);
}

template <typename T>
__aicore__ inline void LowApiCopyToL1(const LocalTensor<T>& srcLocal, const LocalTensor<T>& dstLocal,
    const DataCopyParams& copyParams)
{
    DataCopy(dstLocal, srcLocal, copyParams);
}
template <typename T>
__aicore__ inline void LowApiCopyOut(const LocalTensor<T>& outputLocal, const GlobalTensor<T>& outputGm,
    uint16_t blockCount, uint32_t copyLen, uint32_t dstStride = 0, uint32_t srcStride = 0)
{
    DataCopyExtParams copyParams;
    copyParams.blockCount = blockCount;
    copyParams.blockLen = copyLen * sizeof(T);
    copyParams.srcStride = srcStride * sizeof(T);
    copyParams.dstStride = dstStride * sizeof(T);
    DataCopyPad(outputGm, outputLocal, copyParams);
}

__aicore__ inline void MhcPreLowApiSigmoid(const LocalTensor<float>& dstLocal, const LocalTensor<float>& srcLocal,
    uint32_t elemCount)
{
    LocalTensor<float> tmpLocal = dstLocal[LowApiRoundUp<float>(elemCount)];
    Duplicate(tmpLocal, 1.0f, elemCount);
    Muls(dstLocal, srcLocal, -1.0f, elemCount);
    Exp(dstLocal, dstLocal, elemCount);
    Adds(dstLocal, dstLocal, 1.0f, elemCount);
    Div(dstLocal, tmpLocal, dstLocal, elemCount);
}

__aicore__ inline void MhcPreLowApiVFTransND2NZ(const LocalTensor<float>& yLocal, const LocalTensor<float>& xLocal,
    uint16_t curRowNum, uint16_t curColNum)
{
    __local_mem__ float* yLocalAddr = (__local_mem__ float*)yLocal.GetPhyAddr();
    __local_mem__ float* xLocalAddr = (__local_mem__ float*)xLocal.GetPhyAddr();
    uint16_t curRowNumAlign = LowApiAlign(curRowNum, MHC_PRE_LOW_API_C0_SIZE);
    uint16_t curColNumAlign = LowApiRoundUp<float>(curColNum);
    for (uint32_t col = 0; col < curColNum; ++col) {
        for (uint32_t row = 0; row < curRowNum; ++row) {
            yLocalAddr[(col / MHC_PRE_LOW_API_C0_SIZE) * MHC_PRE_LOW_API_C0_SIZE * curRowNumAlign +
                row * MHC_PRE_LOW_API_C0_SIZE + col % MHC_PRE_LOW_API_C0_SIZE] =
                xLocalAddr[row * curColNumAlign + col];
        }
    }
}

__aicore__ inline void MhcPreLowApiProcessPreN4(const LocalTensor<float>& preLocal,
    const LocalTensor<float>& mmLocal, const LocalTensor<float>& invRmsLocal,
    const LocalTensor<float>& alphaLocal, const LocalTensor<float>& biasLocal,
    uint32_t rowCount, uint32_t fusionAlign, float hcEps)
{
    uint32_t preStride = LowApiRoundUp<float>(MHC_PRE_LOW_API_N4);
    float alpha = alphaLocal.GetValue(0);
    for (uint32_t row = 0; row < rowCount; ++row) {
        float invRms = invRmsLocal.GetValue(row);
        for (uint32_t nIdx = 0; nIdx < MHC_PRE_LOW_API_N4; ++nIdx) {
            float value = mmLocal.GetValue(row * fusionAlign + MHC_PRE_LOW_API_PRE_OFFSET + nIdx);
            float bias = biasLocal.GetValue(nIdx);
            preLocal.SetValue(row * preStride + nIdx, value * invRms * alpha + bias);
        }
    }
    MhcPreLowApiSigmoid(preLocal, preLocal, rowCount * preStride);
    Adds(preLocal, preLocal, hcEps, rowCount * preStride);
}

__aicore__ inline void MhcPreLowApiStageN4(const LocalTensor<float>& dstLocal, const LocalTensor<float>& mmLocal,
    uint32_t rowCount, uint32_t fusionAlign, uint32_t srcOffset, uint32_t elemPerRow)
{
    uint32_t dstStride = LowApiRoundUp<float>(elemPerRow);
    for (uint32_t row = 0; row < rowCount; ++row) {
        for (uint32_t idx = 0; idx < elemPerRow; ++idx) {
            dstLocal.SetValue(row * dstStride + idx, mmLocal.GetValue(row * fusionAlign + srcOffset + idx));
        }
    }
}

__aicore__ inline void MhcPreLowApiProcessSegmentN4(const LocalTensor<float>& dstLocal,
    const LocalTensor<float>& mmLocal, const LocalTensor<float>& invRmsLocal,
    const LocalTensor<float>& alphaLocal, const LocalTensor<float>& biasLocal,
    uint32_t rowCount, uint32_t fusionAlign, uint32_t srcOffset, uint32_t elemPerRow,
    uint32_t alphaIndex, uint32_t biasOffset, bool applySigmoid, bool multiplyTwo, float addEps)
{
    uint32_t dstStride = LowApiRoundUp<float>(elemPerRow);
    float alpha = alphaLocal.GetValue(alphaIndex);
    for (uint32_t row = 0; row < rowCount; ++row) {
        float invRms = invRmsLocal.GetValue(row);
        for (uint32_t idx = 0; idx < elemPerRow; ++idx) {
            float value = mmLocal.GetValue(row * fusionAlign + srcOffset + idx);
            float bias = biasLocal.GetValue(biasOffset + idx);
            dstLocal.SetValue(row * dstStride + idx, value * invRms * alpha + bias);
        }
    }
    if (applySigmoid) {
        MhcPreLowApiSigmoid(dstLocal, dstLocal, rowCount * dstStride);
    }
    if (multiplyTwo) {
        Muls(dstLocal, dstLocal, 2.0f, rowCount * dstStride);
    }
    if (addEps != 0.0f) {
        Adds(dstLocal, dstLocal, addEps, rowCount * dstStride);
    }
}
template <typename T>
__aicore__ inline void MhcPreLowApiProcessHinN4(const LocalTensor<T>& yLocal, const LocalTensor<T>& xLocal,
    const LocalTensor<float>& preLocal, uint32_t rowCount, uint32_t dSize)
{
    __local_mem__ T* y = (__local_mem__ T*)yLocal.GetPhyAddr();
    __local_mem__ T* x = (__local_mem__ T*)xLocal.GetPhyAddr();
    __local_mem__ float* pre = (__local_mem__ float*)preLocal.GetPhyAddr();
    uint32_t dAlign = LowApiRoundUp<T>(dSize);
    uint32_t preStride = LowApiRoundUp<float>(MHC_PRE_LOW_API_N4);
    uint16_t loopCount = LowApiCeilDiv(dSize, MHC_PRE_LOW_API_VL_FP32);
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<float> xReg;
        AscendC::MicroAPI::RegTensor<float> preReg;
        AscendC::MicroAPI::RegTensor<float> sumReg;
        AscendC::MicroAPI::MaskReg mask;
        for (uint16_t row = 0; row < rowCount; ++row) {
            uint32_t remaining = dSize;
            for (uint16_t loop = 0; loop < loopCount; ++loop) {
                mask = AscendC::MicroAPI::UpdateMask<float>(remaining);
                AscendC::MicroAPI::Duplicate(sumReg, 0.0f, mask);
                for (uint16_t nIdx = 0; nIdx < MHC_PRE_LOW_API_N4; ++nIdx) {
                    MhcPreLowApiLoadBroadcast(preReg, pre, mask, row * preStride + nIdx);
                    MhcPreLowApiLoad(xReg, x, mask,
                        row * MHC_PRE_LOW_API_N4 * dAlign + nIdx * dAlign + loop * MHC_PRE_LOW_API_VL_FP32);
                    AscendC::MicroAPI::Mul(xReg, preReg, xReg, mask);
                    AscendC::MicroAPI::Add(sumReg, sumReg, xReg, mask);
                }
                MhcPreLowApiStore(y, sumReg, mask, row * dAlign + loop * MHC_PRE_LOW_API_VL_FP32);
            }
        }
    }
}

} // namespace MhcPre

#endif // MHC_PRE_LOW_API_BASE_ARCH35_H
