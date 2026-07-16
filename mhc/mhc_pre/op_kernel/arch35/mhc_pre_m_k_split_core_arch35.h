#ifndef MHC_PRE_M_K_SPLIT_CORE_ARCH35_H
#define MHC_PRE_M_K_SPLIT_CORE_ARCH35_H

#include "mhc_pre_cube_compute_arch35.h"
#include "mhc_pre_split_bs.h"

namespace MhcPre {

__aicore__ inline void MhcPreMkVFTransND2NZ(const LocalTensor<float>& dstLocal,
    const LocalTensor<float>& srcLocal, uint16_t rowCount, uint16_t columnCount)
{
    __local_mem__ float* dst = (__local_mem__ float*)dstLocal.GetPhyAddr();
    __local_mem__ float* src = (__local_mem__ float*)srcLocal.GetPhyAddr();
    uint16_t rowAlign = LowApiAlign(rowCount, MHC_PRE_LOW_API_C0_SIZE);
    uint16_t dataBlockElements = MHC_PRE_LOW_API_BLOCK_SIZE / sizeof(float);
    uint16_t columnAlign = LowApiRoundUp<float>(columnCount);
    uint16_t mainRowBlocks = rowCount / MHC_PRE_LOW_API_C0_SIZE;
    uint16_t tailRows = rowCount % MHC_PRE_LOW_API_C0_SIZE;
    uint16_t columnLoops = columnAlign / dataBlockElements;
    uint32_t srcBlockStride = columnAlign / dataBlockElements;

    if (tailRows == 0) {
        __VEC_SCOPE__
        {
            AscendC::MicroAPI::RegTensor<float> data;
            AscendC::MicroAPI::MaskReg mask = AscendC::MicroAPI::CreateMask<float>();
            for (uint16_t rowBlock = 0; rowBlock < mainRowBlocks; ++rowBlock) {
                for (uint16_t columnBlock = 0; columnBlock < columnLoops; ++columnBlock) {
                    AscendC::MicroAPI::DataCopy<float,
                        AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(data,
                        src + rowBlock * MHC_PRE_LOW_API_C0_SIZE * columnAlign +
                            columnBlock * dataBlockElements,
                        srcBlockStride, mask);
                    AscendC::MicroAPI::DataCopy(
                        dst + rowBlock * MHC_PRE_LOW_API_C0_SIZE * dataBlockElements +
                            columnBlock * rowAlign * dataBlockElements,
                        data, mask);
                }
            }
        }
        return;
    }

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<float> data;
        AscendC::MicroAPI::MaskReg fullMask = AscendC::MicroAPI::CreateMask<float>();
        uint32_t tailElements = tailRows * MHC_PRE_LOW_API_C0_SIZE;
        AscendC::MicroAPI::MaskReg tailMask =
            AscendC::MicroAPI::UpdateMask<float>(tailElements);
        for (uint16_t rowBlock = 0; rowBlock < mainRowBlocks; ++rowBlock) {
            for (uint16_t columnBlock = 0; columnBlock < columnLoops; ++columnBlock) {
                AscendC::MicroAPI::DataCopy<float,
                    AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(data,
                    src + rowBlock * MHC_PRE_LOW_API_C0_SIZE * columnAlign +
                        columnBlock * dataBlockElements,
                    srcBlockStride, fullMask);
                AscendC::MicroAPI::DataCopy(
                    dst + rowBlock * MHC_PRE_LOW_API_C0_SIZE * dataBlockElements +
                        columnBlock * rowAlign * dataBlockElements,
                    data, fullMask);
            }
        }
        src += mainRowBlocks * MHC_PRE_LOW_API_C0_SIZE * columnAlign;
        dst += mainRowBlocks * MHC_PRE_LOW_API_C0_SIZE * dataBlockElements;
        for (uint16_t columnBlock = 0; columnBlock < columnLoops; ++columnBlock) {
            AscendC::MicroAPI::DataCopy<float,
                AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(data,
                src + columnBlock * dataBlockElements, srcBlockStride, tailMask);
            AscendC::MicroAPI::DataCopy(dst + columnBlock * rowAlign * dataBlockElements,
                data, tailMask);
        }
    }
}

template <typename T, bool INIT_RMS>
__aicore__ inline void MhcPreMkVFProcessXInGammaLoops(const LocalTensor<float>& xOutLocal,
    const LocalTensor<float>& rmsLocal, const LocalTensor<T>& xInLocal,
    const LocalTensor<float>& gammaLocal, uint16_t rowCount, uint16_t columnCount,
    uint16_t loopStart, uint16_t processLoops)
{
    __local_mem__ float* xOut = (__local_mem__ float*)xOutLocal.GetPhyAddr();
    __local_mem__ float* rms = (__local_mem__ float*)rmsLocal.GetPhyAddr();
    __local_mem__ T* xIn = (__local_mem__ T*)xInLocal.GetPhyAddr();
    __local_mem__ float* gamma = (__local_mem__ float*)gammaLocal.GetPhyAddr();
    uint32_t srcStride = LowApiRoundUp<T>(columnCount);
    uint32_t dstStride = LowApiRoundUp<float>(columnCount);
    uint16_t loopEnd = loopStart + processLoops;

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<T> xInReg;
        AscendC::MicroAPI::RegTensor<float> gammaReg;
        AscendC::MicroAPI::RegTensor<float> xReg;
        AscendC::MicroAPI::RegTensor<float> xGammaReg;
        AscendC::MicroAPI::RegTensor<float> squareReg;
        AscendC::MicroAPI::RegTensor<float> partialReg;
        AscendC::MicroAPI::RegTensor<float> sumReg;
        uint32_t remaining = columnCount -
            static_cast<uint32_t>(loopStart) * MHC_PRE_LOW_API_VL_FP32;
        for (uint16_t loop = loopStart; loop < loopEnd; ++loop) {
            AscendC::MicroAPI::MaskReg mask =
                AscendC::MicroAPI::UpdateMask<float>(remaining);
            uint32_t vectorOffset =
                static_cast<uint32_t>(loop) * MHC_PRE_LOW_API_VL_FP32;
            AscendC::MicroAPI::LoadAlign(gammaReg, gamma + vectorOffset);
            for (uint16_t row = 0; row < rowCount; ++row) {
                uint32_t srcOffset = static_cast<uint32_t>(row) * srcStride + vectorOffset;
                AscendC::MicroAPI::LoadAlign<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                    xInReg, xIn + srcOffset);
                AscendC::MicroAPI::Cast<float, T, MHC_PRE_LOW_API_CAST_B16_TO_B32>(
                    xReg, xInReg, mask);
                AscendC::MicroAPI::Mul(xGammaReg, gammaReg, xReg, mask);
                uint32_t dstOffset = static_cast<uint32_t>(row) * dstStride + vectorOffset;
                AscendC::MicroAPI::StoreAlign(xOut + dstOffset, xGammaReg, mask);
                AscendC::MicroAPI::Mul(squareReg, xReg, xReg, mask);
                AscendC::MicroAPI::Reduce<AscendC::MicroAPI::ReduceType::SUM>(
                    partialReg, squareReg, mask);
                if constexpr (INIT_RMS) {
                    AscendC::MicroAPI::Duplicate(sumReg, 0.0f);
                } else {
                    AscendC::MicroAPI::Load(sumReg, rms + row);
                }
                AscendC::MicroAPI::Add(sumReg, sumReg, partialReg, mask);
                AscendC::MicroAPI::Store(rms + row, sumReg, 1U);
            }
        }
    }
}

template <typename T, bool IS_FIRST_K>
__aicore__ inline void MhcPreMkVFProcessXInGammaFull128(const LocalTensor<float>& xOutLocal,
    const LocalTensor<float>& rmsLocal, const LocalTensor<T>& xInLocal,
    const LocalTensor<float>& gammaLocal, uint16_t rowCount)
{
    __local_mem__ float* xOut = (__local_mem__ float*)xOutLocal.GetPhyAddr();
    __local_mem__ float* rms = (__local_mem__ float*)rmsLocal.GetPhyAddr();
    __local_mem__ T* xIn = (__local_mem__ T*)xInLocal.GetPhyAddr();
    __local_mem__ float* gamma = (__local_mem__ float*)gammaLocal.GetPhyAddr();
    constexpr uint32_t columnCount = 128U;
    constexpr uint32_t vectorOffset = MHC_PRE_LOW_API_VL_FP32;
    constexpr uint32_t srcStride = columnCount;
    constexpr uint32_t dstStride = columnCount;

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<T> xInReg;
        AscendC::MicroAPI::RegTensor<float> gamma0Reg;
        AscendC::MicroAPI::RegTensor<float> gamma1Reg;
        AscendC::MicroAPI::RegTensor<float> xReg;
        AscendC::MicroAPI::RegTensor<float> xGammaReg;
        AscendC::MicroAPI::RegTensor<float> squareReg;
        AscendC::MicroAPI::RegTensor<float> squareSumReg;
        AscendC::MicroAPI::RegTensor<float> partialReg;
        AscendC::MicroAPI::RegTensor<float> sumReg;
        AscendC::MicroAPI::MaskReg mask = AscendC::MicroAPI::CreateMask<float>();

        AscendC::MicroAPI::LoadAlign(gamma0Reg, gamma);
        AscendC::MicroAPI::LoadAlign(gamma1Reg, gamma + vectorOffset);
        for (uint16_t row = 0; row < rowCount; ++row) {
            uint32_t rowSrcOffset = static_cast<uint32_t>(row) * srcStride;
            uint32_t rowDstOffset = static_cast<uint32_t>(row) * dstStride;
            AscendC::MicroAPI::Duplicate(squareSumReg, 0.0f);

            AscendC::MicroAPI::LoadAlign<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                xInReg, xIn + rowSrcOffset);
            AscendC::MicroAPI::Cast<float, T, MHC_PRE_LOW_API_CAST_B16_TO_B32>(xReg, xInReg, mask);
            AscendC::MicroAPI::Mul(xGammaReg, gamma0Reg, xReg, mask);
            AscendC::MicroAPI::StoreAlign(xOut + rowDstOffset, xGammaReg, mask);
            AscendC::MicroAPI::Mul(squareReg, xReg, xReg, mask);
            AscendC::MicroAPI::Add(squareSumReg, squareSumReg, squareReg, mask);

            AscendC::MicroAPI::LoadAlign<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                xInReg, xIn + rowSrcOffset + vectorOffset);
            AscendC::MicroAPI::Cast<float, T, MHC_PRE_LOW_API_CAST_B16_TO_B32>(xReg, xInReg, mask);
            AscendC::MicroAPI::Mul(xGammaReg, gamma1Reg, xReg, mask);
            AscendC::MicroAPI::StoreAlign(xOut + rowDstOffset + vectorOffset, xGammaReg, mask);
            AscendC::MicroAPI::Mul(squareReg, xReg, xReg, mask);
            AscendC::MicroAPI::Add(squareSumReg, squareSumReg, squareReg, mask);

            AscendC::MicroAPI::Reduce<AscendC::MicroAPI::ReduceType::SUM>(
                partialReg, squareSumReg, mask);
            if constexpr (IS_FIRST_K) {
                AscendC::MicroAPI::Store(rms + row, partialReg, 1U);
            } else {
                AscendC::MicroAPI::Load(sumReg, rms + row);
                AscendC::MicroAPI::Add(sumReg, sumReg, partialReg, mask);
                AscendC::MicroAPI::Store(rms + row, sumReg, 1U);
            }
        }
    }
}

template <typename T, bool IS_FIRST_K>
__aicore__ inline void MhcPreMkVFProcessXInGamma(const LocalTensor<float>& xOutLocal,
    const LocalTensor<float>& rmsLocal, const LocalTensor<T>& xInLocal,
    const LocalTensor<float>& gammaLocal, uint16_t rowCount, uint16_t columnCount)
{
    if (columnCount == 128U) {
        MhcPreMkVFProcessXInGammaFull128<T, IS_FIRST_K>(xOutLocal, rmsLocal, xInLocal,
            gammaLocal, rowCount);
        return;
    }
    uint16_t loopCount = LowApiCeilDiv(columnCount, MHC_PRE_LOW_API_VL_FP32);
    if constexpr (IS_FIRST_K) {
        MhcPreMkVFProcessXInGammaLoops<T, true>(xOutLocal, rmsLocal, xInLocal,
            gammaLocal, rowCount, columnCount, 0U, 1U);
        if (loopCount > 1U) {
            MhcPreMkVFProcessXInGammaLoops<T, false>(xOutLocal, rmsLocal, xInLocal,
                gammaLocal, rowCount, columnCount, 1U, loopCount - 1U);
        }
    } else {
        MhcPreMkVFProcessXInGammaLoops<T, false>(xOutLocal, rmsLocal, xInLocal,
            gammaLocal, rowCount, columnCount, 0U, loopCount);
    }
}
__aicore__ inline void MhcPreMkVFReducePartials(const LocalTensor<float>& mmOutLocal,
    const LocalTensor<float>& invRmsOutLocal, const LocalTensor<float>& mmPartialLocal,

    const LocalTensor<float>& rmsPartialLocal, uint16_t groupK, uint16_t rowCount,
    uint16_t fusionSize, uint16_t rmsStride, float scaleMean, float normEps)
{
    __local_mem__ float* mmOut = (__local_mem__ float*)mmOutLocal.GetPhyAddr();
    __local_mem__ float* invRmsOut = (__local_mem__ float*)invRmsOutLocal.GetPhyAddr();
    __local_mem__ float* mmPartial = (__local_mem__ float*)mmPartialLocal.GetPhyAddr();
    __local_mem__ float* rmsPartial = (__local_mem__ float*)rmsPartialLocal.GetPhyAddr();
    uint16_t fourLoopCount = groupK / 4U;
    uint16_t tailLoopCount = groupK % 4U;
    uint32_t mmGroupStride = static_cast<uint32_t>(rowCount) * fusionSize;

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<float> rms0;
        AscendC::MicroAPI::RegTensor<float> rms1;
        AscendC::MicroAPI::RegTensor<float> rms2;
        AscendC::MicroAPI::RegTensor<float> rms3;
        AscendC::MicroAPI::RegTensor<float> mm0;
        AscendC::MicroAPI::RegTensor<float> mm1;
        AscendC::MicroAPI::RegTensor<float> mm2;
        AscendC::MicroAPI::RegTensor<float> mm3;
        AscendC::MicroAPI::RegTensor<float> sumRms0;
        AscendC::MicroAPI::RegTensor<float> sumRms1;
        AscendC::MicroAPI::RegTensor<float> sumRms2;
        AscendC::MicroAPI::RegTensor<float> sumRms3;
        AscendC::MicroAPI::RegTensor<float> sumMm0;
        AscendC::MicroAPI::RegTensor<float> sumMm1;
        AscendC::MicroAPI::RegTensor<float> sumMm2;
        AscendC::MicroAPI::RegTensor<float> sumMm3;
        AscendC::MicroAPI::RegTensor<float> one;
        uint32_t rmsMaskSize = 1U;
        uint32_t mmMaskSize = fusionSize;
        AscendC::MicroAPI::MaskReg rmsMask = AscendC::MicroAPI::UpdateMask<float>(rmsMaskSize);
        AscendC::MicroAPI::MaskReg mmMask = AscendC::MicroAPI::UpdateMask<float>(mmMaskSize);
        AscendC::MicroAPI::Duplicate(one, 1.0f, rmsMask);

        for (uint16_t row = 0; row < rowCount; ++row) {
            AscendC::MicroAPI::Duplicate(sumRms0, 0.0f, rmsMask);
            AscendC::MicroAPI::Duplicate(sumRms1, 0.0f, rmsMask);
            AscendC::MicroAPI::Duplicate(sumRms2, 0.0f, rmsMask);
            AscendC::MicroAPI::Duplicate(sumRms3, 0.0f, rmsMask);
            AscendC::MicroAPI::Duplicate(sumMm0, 0.0f, mmMask);
            AscendC::MicroAPI::Duplicate(sumMm1, 0.0f, mmMask);
            AscendC::MicroAPI::Duplicate(sumMm2, 0.0f, mmMask);
            AscendC::MicroAPI::Duplicate(sumMm3, 0.0f, mmMask);

            for (uint16_t loop = 0; loop < fourLoopCount; ++loop) {
                uint16_t kBase = loop * 4U;
                MhcPreLowApiLoadBroadcast(rms0, rmsPartial, rmsMask,
                    static_cast<uint32_t>(kBase) * rmsStride + row);
                MhcPreLowApiLoadBroadcast(rms1, rmsPartial, rmsMask,
                    static_cast<uint32_t>(kBase + 1U) * rmsStride + row);
                MhcPreLowApiLoadBroadcast(rms2, rmsPartial, rmsMask,
                    static_cast<uint32_t>(kBase + 2U) * rmsStride + row);
                MhcPreLowApiLoadBroadcast(rms3, rmsPartial, rmsMask,
                    static_cast<uint32_t>(kBase + 3U) * rmsStride + row);
                AscendC::MicroAPI::Add(sumRms0, sumRms0, rms0, rmsMask);
                AscendC::MicroAPI::Add(sumRms1, sumRms1, rms1, rmsMask);
                AscendC::MicroAPI::Add(sumRms2, sumRms2, rms2, rmsMask);
                AscendC::MicroAPI::Add(sumRms3, sumRms3, rms3, rmsMask);

                AscendC::MicroAPI::Load<float>(mm0,
                    mmPartial + static_cast<uint32_t>(kBase) * mmGroupStride + row * fusionSize);
                AscendC::MicroAPI::Load<float>(mm1,
                    mmPartial + static_cast<uint32_t>(kBase + 1U) * mmGroupStride + row * fusionSize);
                AscendC::MicroAPI::Load<float>(mm2,
                    mmPartial + static_cast<uint32_t>(kBase + 2U) * mmGroupStride + row * fusionSize);
                AscendC::MicroAPI::Load<float>(mm3,
                    mmPartial + static_cast<uint32_t>(kBase + 3U) * mmGroupStride + row * fusionSize);
                AscendC::MicroAPI::Add(sumMm0, sumMm0, mm0, mmMask);
                AscendC::MicroAPI::Add(sumMm1, sumMm1, mm1, mmMask);
                AscendC::MicroAPI::Add(sumMm2, sumMm2, mm2, mmMask);
                AscendC::MicroAPI::Add(sumMm3, sumMm3, mm3, mmMask);
            }

            for (uint16_t tail = 0; tail < tailLoopCount; ++tail) {
                uint16_t kIndex = fourLoopCount * 4U + tail;
                MhcPreLowApiLoadBroadcast(rms0, rmsPartial, rmsMask,
                    static_cast<uint32_t>(kIndex) * rmsStride + row);
                AscendC::MicroAPI::Add(sumRms0, sumRms0, rms0, rmsMask);
                AscendC::MicroAPI::Load<float>(mm0,
                    mmPartial + static_cast<uint32_t>(kIndex) * mmGroupStride + row * fusionSize);
                AscendC::MicroAPI::Add(sumMm0, sumMm0, mm0, mmMask);
            }

            AscendC::MicroAPI::Add(sumRms0, sumRms0, sumRms3, rmsMask);
            AscendC::MicroAPI::Add(sumRms1, sumRms1, sumRms2, rmsMask);
            AscendC::MicroAPI::Add(sumRms0, sumRms0, sumRms1, rmsMask);
            AscendC::MicroAPI::Muls(sumRms0, sumRms0, scaleMean, rmsMask);
            AscendC::MicroAPI::Adds(sumRms0, sumRms0, normEps, rmsMask);
            AscendC::MicroAPI::Sqrt(sumRms0, sumRms0, rmsMask);
            AscendC::MicroAPI::Div(sumRms0, one, sumRms0, rmsMask);

            AscendC::MicroAPI::Add(sumMm0, sumMm0, sumMm3, mmMask);
            AscendC::MicroAPI::Add(sumMm1, sumMm1, sumMm2, mmMask);
            AscendC::MicroAPI::Add(sumMm0, sumMm0, sumMm1, mmMask);
            AscendC::MicroAPI::Store<float>(invRmsOut + row, sumRms0, 1U);
            AscendC::MicroAPI::Store<float>(mmOut + row * fusionSize, sumMm0, fusionSize);
        }
    }
}

template <int8_t HAS_RESI>
__aicore__ inline void MhcPreMkVFPostSegmentsN4(const LocalTensor<float>& hPreLocal,
    const LocalTensor<float>& hPostLocal, const LocalTensor<float>& hResLocal,
    const LocalTensor<float>& mmLocal, const LocalTensor<float>& invRmsLocal,
    const LocalTensor<float>& alphaLocal, const LocalTensor<float>& biasLocal,
    uint16_t rowCount, uint32_t invRmsOffset)
{
    __local_mem__ float* hPre = (__local_mem__ float*)hPreLocal.GetPhyAddr();
    __local_mem__ float* hPost = (__local_mem__ float*)hPostLocal.GetPhyAddr();
    __local_mem__ float* mm = (__local_mem__ float*)mmLocal.GetPhyAddr();
    __local_mem__ float* invRms = (__local_mem__ float*)invRmsLocal.GetPhyAddr();
    __local_mem__ float* alpha = (__local_mem__ float*)alphaLocal.GetPhyAddr();
    __local_mem__ float* bias = (__local_mem__ float*)biasLocal.GetPhyAddr();
    __local_mem__ float* hRes = nullptr;
    if constexpr (HAS_RESI == MHC_PRE_HAS_RESI) {
        hRes = (__local_mem__ float*)hResLocal.GetPhyAddr();
    }

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<float> dataReg;
        AscendC::MicroAPI::RegTensor<float> invRmsReg;
        AscendC::MicroAPI::RegTensor<float> invRmsBroadReg;
        AscendC::MicroAPI::RegTensor<float> alphaPreReg;
        AscendC::MicroAPI::RegTensor<float> alphaPostReg;
        AscendC::MicroAPI::RegTensor<float> alphaResReg;
        AscendC::MicroAPI::RegTensor<float> biasPreReg;
        AscendC::MicroAPI::RegTensor<float> biasPostReg;
        AscendC::MicroAPI::RegTensor<float> biasResReg;
        uint32_t nMaskSize = MHC_PRE_LOW_API_N4;
        uint32_t resMaskSize = MHC_PRE_LOW_API_N4 * MHC_PRE_LOW_API_N4;
        AscendC::MicroAPI::MaskReg nMask =
            AscendC::MicroAPI::UpdateMask<float>(nMaskSize);
        AscendC::MicroAPI::MaskReg resMask =
            AscendC::MicroAPI::UpdateMask<float>(resMaskSize);

        AscendC::MicroAPI::Load<float>(alphaPreReg, alpha);
        AscendC::MicroAPI::Load<float>(alphaPostReg, alpha + MHC_PRE_LOW_API_N4);
        AscendC::MicroAPI::Load<float>(biasPreReg, bias);
        AscendC::MicroAPI::Load<float>(biasPostReg, bias + MHC_PRE_LOW_API_N4);
        if constexpr (HAS_RESI == MHC_PRE_HAS_RESI) {
            AscendC::MicroAPI::Load<float>(alphaResReg, alpha + 2U * MHC_PRE_LOW_API_N4);
            AscendC::MicroAPI::Load<float>(biasResReg, bias + 2U * MHC_PRE_LOW_API_N4);
        }

        for (uint16_t row = 0; row < rowCount; ++row) {
            AscendC::MicroAPI::Load<float>(invRmsReg, invRms + invRmsOffset + row);
            AscendC::MicroAPI::Duplicate(invRmsBroadReg, invRmsReg, nMask);
            AscendC::MicroAPI::Load<float>(dataReg,
                mm + static_cast<uint32_t>(row) * MHC_PRE_LOW_API_FUSION_N4);
            AscendC::MicroAPI::Mul(dataReg, dataReg, invRmsBroadReg, nMask);
            AscendC::MicroAPI::Mul(dataReg, dataReg, alphaPreReg, nMask);
            AscendC::MicroAPI::Add(dataReg, dataReg, biasPreReg, nMask);
            AscendC::MicroAPI::Store<float>(hPre + row * MHC_PRE_LOW_API_N4,
                dataReg, MHC_PRE_LOW_API_N4);

            AscendC::MicroAPI::Load<float>(dataReg,
                mm + static_cast<uint32_t>(row) * MHC_PRE_LOW_API_FUSION_N4 +
                MHC_PRE_LOW_API_N4);
            AscendC::MicroAPI::Mul(dataReg, dataReg, invRmsBroadReg, nMask);
            AscendC::MicroAPI::Mul(dataReg, dataReg, alphaPostReg, nMask);
            AscendC::MicroAPI::Add(dataReg, dataReg, biasPostReg, nMask);
            AscendC::MicroAPI::Store<float>(hPost + row * MHC_PRE_LOW_API_N4,
                dataReg, MHC_PRE_LOW_API_N4);

            if constexpr (HAS_RESI == MHC_PRE_HAS_RESI) {
                AscendC::MicroAPI::Duplicate(invRmsBroadReg, invRmsReg, resMask);
                AscendC::MicroAPI::Load<float>(dataReg,
                    mm + static_cast<uint32_t>(row) * MHC_PRE_LOW_API_FUSION_N4 +
                    2U * MHC_PRE_LOW_API_N4);
                AscendC::MicroAPI::Mul(dataReg, dataReg, invRmsBroadReg, resMask);
                AscendC::MicroAPI::Mul(dataReg, dataReg, alphaResReg, resMask);
                AscendC::MicroAPI::Add(dataReg, dataReg, biasResReg, resMask);
                AscendC::MicroAPI::Store<float>(
                    hRes + row * MHC_PRE_LOW_API_N4 * MHC_PRE_LOW_API_N4,
                    dataReg, MHC_PRE_LOW_API_N4 * MHC_PRE_LOW_API_N4);
            }
        }
    }
}

template <class T, class P, int8_t HAS_RESI>
class MhcPreKernelMKPart1 : public MhcPreKernelSplitBS<T, P, HAS_RESI> {
public:
    using Parent = MhcPreKernelSplitBS<T, P, HAS_RESI>;

    __aicore__ inline explicit MhcPreKernelMKPart1(MT &matmul) : Parent(matmul) {}

    __aicore__ inline void Init(InitParams initParams)
    {
        this->BindGlobalTensors(initParams);
        this->InitFromTilingData(initParams.tilingData);
        this->InitMNConfig();
        this->pipe_ = initParams.tPipeIn;

        partialMmGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(
            initParams.workspace + this->tiling_->mkWorkspaceMmOffset));
        partialRmsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(
            initParams.workspace + this->tiling_->mkWorkspaceRmsOffset));

        this->pipe_->InitBuffer(l1Buffer_, MHC_PRE_LOW_API_L1_ALLOC_SIZE);
        aL1_ = l1Buffer_.Get<P>();
        bL1_ = aL1_[MHC_PRE_LOW_API_L1_BUF_NUM * MHC_PRE_LOW_API_L1_BUF_OFFSET];

        if ASCEND_IS_AIC {
            mmService_.Init();
            CrossCoreSetFlag<MHC_PRE_LOW_API_SYNC_MODE4, PIPE_MTE1>(SYNC_AIC_AIV_FLAG);
            CrossCoreSetFlag<MHC_PRE_LOW_API_SYNC_MODE4, PIPE_MTE1>(SYNC_AIC_AIV_FLAG + FLAG_ID_MAX);
            CrossCoreSetFlag<MHC_PRE_LOW_API_SYNC_MODE4, PIPE_MTE1>(SYNC_AIC_AIV_FLAG);
            CrossCoreSetFlag<MHC_PRE_LOW_API_SYNC_MODE4, PIPE_MTE1>(SYNC_AIC_AIV_FLAG + FLAG_ID_MAX);
        }
        if ASCEND_IS_AIV {
            uint32_t rowCapacity = (this->tiling_->mL1Size + 1U) / 2U;
            uint32_t kCapacity = LowApiRoundUp<T>(this->tiling_->kUbSize);
            this->pipe_->InitBuffer(this->xInQueue_, 2, rowCapacity * kCapacity * sizeof(T));
            this->pipe_->InitBuffer(this->outQueue_, 1,
                rowCapacity * LowApiRoundUp<P>(this->tiling_->kUbSize) * sizeof(P));
            this->pipe_->InitBuffer(this->invRmsOutQueue_, 1,
                LowApiRoundUp<P>(rowCapacity) * sizeof(P));
            if (this->hasGamma_) {
                this->pipe_->InitBuffer(this->gammaInQueue_, 2,
                    LowApiRoundUp<P>(this->tiling_->kUbSize) * sizeof(P));
            }
            this->pipe_->InitBuffer(nd2NzBuf_,
                LowApiAlign(rowCapacity, MHC_PRE_LOW_API_C0_SIZE) *
                LowApiRoundUp<P>(this->tiling_->kUbSize) * sizeof(P));
            nd2NzLocal_ = nd2NzBuf_.Get<P>();
            this->invRmsUb_ = this->invRmsOutQueue_.template AllocTensor<P>();
        }
    }

    __aicore__ inline void Process()
    {
        uint32_t aicIndex = GetBlockIdx();
        if ASCEND_IS_AIV {
            aicIndex /= 2U;
        }
        uint32_t usedCoreNum = this->tiling_->cubeBlockDimM * this->tiling_->cubeBlockDimK;
        if (aicIndex < usedCoreNum) {
            uint32_t mIndex = aicIndex / this->tiling_->cubeBlockDimK;
            uint32_t kIndex = aicIndex % this->tiling_->cubeBlockDimK;
            uint32_t mStart = mIndex * this->tiling_->mL1Size;
            uint32_t mReal = AscendC::Std::min(
                static_cast<uint64_t>(this->tiling_->mL1Size), this->totalLength_ - mStart);
            uint32_t kStart = kIndex * this->tiling_->multCoreSplitKSize;
            uint32_t kEnd = AscendC::Std::min(
                static_cast<uint64_t>(kStart + this->tiling_->multCoreSplitKSize),
                static_cast<uint64_t>(this->matrixInfo_.nD));
            if ASCEND_IS_AIC {
                ProcessAic(mStart, mReal, kIndex, kStart, kEnd);
            } else {
                ProcessAiv(mStart, mReal, kIndex, kStart, kEnd);
            }
        }
        SyncAll<false>();
        if ASCEND_IS_AIC {
            mmService_.End();
        }
    }

private:
    __aicore__ inline void ProcessAic(uint32_t mStart, uint32_t mReal, uint32_t kIndex,
        uint32_t kStart, uint32_t kEnd)
    {
        uint8_t aL1BufferId = 0;
        for (uint32_t kOffset = kStart; kOffset < kEnd; kOffset += this->tiling_->kL1Size) {
            uint32_t currentK = AscendC::Std::min(
                static_cast<uint64_t>(this->tiling_->kL1Size), static_cast<uint64_t>(kEnd - kOffset));
            uint8_t bL1BufferId = mmService_.GetBL1BufferId();
            LocalTensor<P> currentAL1 = aL1_[aL1BufferId * MHC_PRE_LOW_API_L1_BUF_OFFSET];
            LocalTensor<P> currentBL1 = bL1_[bL1BufferId * MHC_PRE_LOW_API_L1_BUF_OFFSET];
            mmService_.CopyInB1Nd2Nz(this->matrixInfo_.nD, currentK, this->matrixInfo_.fusionSize,
                this->phiGm_[kOffset], currentBL1);
            CrossCoreWaitFlag<MHC_PRE_LOW_API_SYNC_MODE4, PIPE_MTE1>(SYNC_AIV_AIC_FLAG);
            CrossCoreWaitFlag<MHC_PRE_LOW_API_SYNC_MODE4, PIPE_MTE1>(SYNC_AIV_AIC_FLAG + FLAG_ID_MAX);

            bool isFirst = kOffset == kStart;
            bool isLast = kOffset + currentK >= kEnd;
            uint64_t mAlign = LowApiAlign(mReal, AscendC::BLOCK_CUBE);
            uint64_t nAlign = LowApiAlign(this->matrixInfo_.fusionSize, AscendC::BLOCK_CUBE);
            uint64_t baseK = (256U / AscendC::Std::max(mAlign, nAlign)) * 32U;
            mmService_.Process(mReal, this->matrixInfo_.fusionSize, mReal, baseK,
                isFirst, isLast, currentAL1, currentBL1);
            if (isLast) {
                uint64_t outputOffset = static_cast<uint64_t>(kIndex) * this->totalLength_ *
                    this->matrixInfo_.fusionSize + static_cast<uint64_t>(mStart) * this->matrixInfo_.fusionSize;
                mmService_.CopyOut(partialMmGm_[outputOffset], this->matrixInfo_.fusionSize);
            }
            CrossCoreSetFlag<MHC_PRE_LOW_API_SYNC_MODE4, PIPE_MTE1>(SYNC_AIC_AIV_FLAG);
            CrossCoreSetFlag<MHC_PRE_LOW_API_SYNC_MODE4, PIPE_MTE1>(SYNC_AIC_AIV_FLAG + FLAG_ID_MAX);
            aL1BufferId ^= 1U;
        }
    }

    __aicore__ inline void ProcessAiv(uint32_t mStart, uint32_t mReal, uint32_t kIndex,
        uint32_t kStart, uint32_t kEnd)
    {

        uint32_t firstRows = mReal / 2U;
        uint32_t rowStart = GetSubBlockIdx() == 0 ? 0U : firstRows;
        uint32_t rowCount = GetSubBlockIdx() == 0 ? firstRows : mReal - firstRows;
        uint8_t aL1BufferId = 0;
        this->globalOffsetM_ = mStart;
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        uint32_t firstK = AscendC::Std::min(
            static_cast<uint64_t>(this->tiling_->kL1Size), static_cast<uint64_t>(kEnd - kStart));
        this->xLocal_ = this->xInQueue_.template AllocTensor<T>();
        this->DataCopyX(rowCount, firstK, rowStart, kStart);
        if (this->hasGamma_) {
            this->gammaUb_ = this->gammaInQueue_.template AllocTensor<P>();
            this->DataCopyGamma(firstK, kStart);
        }

        for (uint32_t kOffset = kStart; kOffset < kEnd; kOffset += this->tiling_->kL1Size) {
            uint32_t currentK = AscendC::Std::min(
                static_cast<uint64_t>(this->tiling_->kL1Size), static_cast<uint64_t>(kEnd - kOffset));
            CrossCoreWaitFlag<MHC_PRE_LOW_API_SYNC_MODE4, PIPE_MTE3>(SYNC_AIC_AIV_FLAG);

            LocalTensor<T> currentXLocal = this->xInQueue_.template DeQue<T>();
            LocalTensor<P> xFloatLocal = this->outQueue_.template AllocTensor<P>();
            bool initializeRms = kOffset == kStart;
            LocalTensor<P> currentGammaLocal;
            if (this->hasGamma_) {
                currentGammaLocal = this->gammaInQueue_.template DeQue<P>();
            }

            uint32_t nextKOffset = kOffset + this->tiling_->kL1Size;
            if (nextKOffset < kEnd) {
                uint32_t nextK = AscendC::Std::min(
                    static_cast<uint64_t>(this->tiling_->kL1Size),
                    static_cast<uint64_t>(kEnd - nextKOffset));
                this->xLocal_ = this->xInQueue_.template AllocTensor<T>();
                this->DataCopyX(rowCount, nextK, rowStart, nextKOffset);
                if (this->hasGamma_) {
                    this->gammaUb_ = this->gammaInQueue_.template AllocTensor<P>();
                    this->DataCopyGamma(nextK, nextKOffset);
                }
            }

            if (this->hasGamma_) {
                if (initializeRms) {
                    MhcPreMkVFProcessXInGamma<T, true>(xFloatLocal, this->invRmsUb_,
                        currentXLocal, currentGammaLocal, rowCount, currentK);
                } else {
                    MhcPreMkVFProcessXInGamma<T, false>(xFloatLocal, this->invRmsUb_,
                        currentXLocal, currentGammaLocal, rowCount, currentK);
                }
                this->gammaInQueue_.FreeTensor(currentGammaLocal);
            } else if (initializeRms) {
                this->template VFDoV0ProcessXIn<false, true>((__ubuf__ P *)xFloatLocal.GetPhyAddr(),
                    (__ubuf__ P *)this->invRmsUb_.GetPhyAddr(),
                    (__ubuf__ T *)currentXLocal.GetPhyAddr(), nullptr, rowCount, currentK);
            } else {
                this->template VFDoV0ProcessXIn<false, false>((__ubuf__ P *)xFloatLocal.GetPhyAddr(),
                    (__ubuf__ P *)this->invRmsUb_.GetPhyAddr(),
                    (__ubuf__ T *)currentXLocal.GetPhyAddr(), nullptr, rowCount, currentK);
            }
            this->xInQueue_.FreeTensor(currentXLocal);

            constexpr event_t eventId = EVENT_ID0;
            WaitFlag<HardEvent::MTE3_V>(eventId);
            LocalTensor<P> currentNd2Nz = nd2NzLocal_;
            MhcPreMkVFTransND2NZ(currentNd2Nz, xFloatLocal, rowCount, currentK);
            this->outQueue_.FreeTensor(xFloatLocal);
            SetFlag<HardEvent::V_MTE3>(eventId);
            WaitFlag<HardEvent::V_MTE3>(eventId);
            DataCopyParams copyParams;
            copyParams.blockCount = LowApiCeilDiv(currentK, MHC_PRE_LOW_API_C0_SIZE);
            copyParams.blockLen =
                rowCount * MHC_PRE_LOW_API_C0_SIZE * sizeof(P) / MHC_PRE_LOW_API_BLOCK_SIZE;
            copyParams.srcStride = LowApiAlign(rowCount, MHC_PRE_LOW_API_C0_SIZE) - rowCount;
            copyParams.dstStride = LowApiAlign(mReal, AscendC::BLOCK_CUBE) - rowCount;
            uint32_t dstRowOffset = GetSubBlockIdx() == 0 ? 0U :
                firstRows * (MHC_PRE_LOW_API_BLOCK_SIZE / sizeof(P));
            LowApiCopyToL1(currentNd2Nz,
                aL1_[aL1BufferId * MHC_PRE_LOW_API_L1_BUF_OFFSET + dstRowOffset], copyParams);
            SetFlag<HardEvent::MTE3_V>(eventId);
            CrossCoreSetFlag<MHC_PRE_LOW_API_SYNC_MODE4, PIPE_MTE3>(SYNC_AIV_AIC_FLAG);
            aL1BufferId ^= 1U;
        }
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);

        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = rowCount * sizeof(P);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        uint64_t rmsOffset = static_cast<uint64_t>(kIndex) * this->totalLength_ + mStart + rowStart;
        DataCopyPad(partialRmsGm_[rmsOffset], this->invRmsUb_, copyParams);
        this->invRmsOutQueue_.FreeTensor(this->invRmsUb_);
    }

private:
    GlobalTensor<P> partialMmGm_;
    GlobalTensor<P> partialRmsGm_;
    TBuf<TPosition::A1> l1Buffer_;
    TBuf<TPosition::VECCALC> nd2NzBuf_;
    LocalTensor<P> aL1_;
    LocalTensor<P> bL1_;
    LocalTensor<P> nd2NzLocal_;
    MhcPreCubeCompute mmService_;
    static constexpr uint64_t SYNC_AIV_AIC_FLAG = 8;
    static constexpr uint64_t SYNC_AIC_AIV_FLAG = 9;
    static constexpr uint64_t FLAG_ID_MAX = 16;
};

template <class T, class P, int8_t HAS_RESI>
class MhcPreKernelMKPart2 : public MhcPreKernelSplitBS<T, P, HAS_RESI> {
public:
    using Parent = MhcPreKernelSplitBS<T, P, HAS_RESI>;

    __aicore__ inline explicit MhcPreKernelMKPart2(MT &matmul) : Parent(matmul) {}

    __aicore__ inline void Init(InitParams initParams)
    {
        this->BindGlobalTensors(initParams);
        this->InitFromTilingData(initParams.tilingData);
        this->InitMNConfig();
        this->pipe_ = initParams.tPipeIn;
        partialMmGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(
            initParams.workspace + this->tiling_->mkWorkspaceMmOffset));
        partialRmsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(
            initParams.workspace + this->tiling_->mkWorkspaceRmsOffset));
        if (this->outFlag_) {
            this->hMixGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(initParams.h_mix));
        } else {
            this->hMixGm_.SetGlobalBuffer(reinterpret_cast<__gm__ P *>(
                initParams.workspace + this->tiling_->mkWorkspaceFinalOffset));
        }
        this->InitUbBuffers();

        if ASCEND_IS_AIV {
            uint32_t aivIndex = GetBlockIdx();
            selected_ = aivIndex < this->tiling_->stage2UsedAivNum;
            if (selected_) {
                uint64_t absoluteMStart =
                    static_cast<uint64_t>(aivIndex) * this->tiling_->stage2RowsPerCore;
                uint64_t mIndex = absoluteMStart / this->tiling_->mL1Size;
                uint64_t localMStart = absoluteMStart - mIndex * this->tiling_->mL1Size;
                this->coreIdx_ = mIndex;
                this->globalOffsetM_ = static_cast<uint64_t>(mIndex) * this->tiling_->mL1Size;
                this->curSingleT_ = AscendC::Std::min(
                    static_cast<uint64_t>(this->tiling_->stage2RowsPerCore),
                    this->totalLength_ - absoluteMStart);
                this->mnConfig_.curSingleCoreM = this->curSingleT_;
                this->vectorOffset_.singleCoreM = this->curSingleT_;
                this->vectorOffset_.offsetMStart = localMStart;
                this->vectorOffset_.offsetMEnd = localMStart + this->curSingleT_;
                this->AIVPreLoad();
            }
        }
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_NOT_AIV {
            return;
        }
        if (!selected_ || this->vectorOffset_.singleCoreM == 0) {
            return;
        }
        ReducePartials();
        SetFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(0));
        WaitFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(0));
        PostProcess();
        this->invRmsOutQueue_.FreeTensor(this->invRmsUb_);
        this->biasInQue_.FreeTensor(this->biasInUb_);
    }

private:
    __aicore__ inline void ReducePartials()
    {
        for (uint32_t offsetT = this->vectorOffset_.offsetMStart;
             offsetT < this->vectorOffset_.offsetMEnd; offsetT += this->V1_BASE_T) {
            uint32_t lenT = AscendC::Std::min(
                static_cast<uint64_t>(this->V1_BASE_T), this->vectorOffset_.offsetMEnd - offsetT);
            uint32_t groupK = this->tiling_->cubeBlockDimK;
            uint32_t fusionSize = this->matrixInfo_.fusionSize;
            uint32_t rmsStride = LowApiRoundUp<P>(lenT);
            uint32_t mmPartialElements = groupK * lenT * fusionSize;
            LocalTensor<P> partialLocal = this->xInQueue_.template AllocTensor<P>();
            LocalTensor<P> rmsPartialLocal = partialLocal[mmPartialElements];

            DataCopyExtParams mmCopyParams;
            mmCopyParams.blockCount = groupK;
            mmCopyParams.blockLen = lenT * fusionSize * sizeof(P);
            mmCopyParams.srcStride = (this->totalLength_ - lenT) * fusionSize * sizeof(P);
            mmCopyParams.dstStride = 0;
            DataCopyPadExtParams<P> mmPadParams{true, 0, 0, 0};
            uint64_t mmSourceOffset = (this->globalOffsetM_ + offsetT) * fusionSize;
            DataCopyPad(partialLocal, partialMmGm_[mmSourceOffset], mmCopyParams, mmPadParams);

            DataCopyExtParams rmsCopyParams;
            rmsCopyParams.blockCount = groupK;
            rmsCopyParams.blockLen = lenT * sizeof(P);
            rmsCopyParams.srcStride = (this->totalLength_ - lenT) * sizeof(P);
            rmsCopyParams.dstStride = 0;
            DataCopyPadExtParams<P> rmsPadParams{true, 0,
                static_cast<uint8_t>(rmsStride - lenT), 0};
            uint64_t rmsSourceOffset = this->globalOffsetM_ + offsetT;
            DataCopyPad(rmsPartialLocal, partialRmsGm_[rmsSourceOffset], rmsCopyParams, rmsPadParams);

            this->xInQueue_.template EnQue<P>(partialLocal);
            partialLocal = this->xInQueue_.template DeQue<P>();
            rmsPartialLocal = partialLocal[mmPartialElements];
            LocalTensor<P> accumulator = this->outQueue_.template AllocTensor<P>();
            uint32_t invRmsOffset = offsetT - this->vectorOffset_.offsetMStart;
            MhcPreMkVFReducePartials(accumulator, this->invRmsUb_[invRmsOffset], partialLocal,
                rmsPartialLocal, static_cast<uint16_t>(groupK), static_cast<uint16_t>(lenT),
                static_cast<uint16_t>(fusionSize), static_cast<uint16_t>(rmsStride),
                this->scaleMean_, this->matrixInfo_.normEps);
            this->xInQueue_.FreeTensor(partialLocal);
            this->outQueue_.template EnQue<P>(accumulator);
            accumulator = this->outQueue_.template DeQue<P>();
            DataCopyExtParams outputParams;
            outputParams.blockCount = 1;
            outputParams.blockLen = lenT * fusionSize * sizeof(P);
            outputParams.srcStride = 0;
            outputParams.dstStride = 0;
            uint64_t outputOffset = (this->globalOffsetM_ + offsetT) * this->matrixInfo_.fusionSize;
            DataCopyPad(this->hMixGm_[outputOffset], accumulator, outputParams);
            this->outQueue_.FreeTensor(accumulator);
        }
        this->DataCopyOutInvRmsUb(this->vectorOffset_.singleCoreM,
            this->vectorOffset_.offsetMStart);
    }

    __aicore__ inline void PostProcessSegments(uint64_t offsetT, uint64_t lenT,
        uint64_t localOffset)
    {
        uint64_t outputOffset = this->globalOffsetM_ + offsetT;
        uint64_t hMixOffset = outputOffset * this->matrixInfo_.fusionSize;
        LocalTensor<P> hResLocal;
        if constexpr (HAS_RESI == MHC_PRE_HAS_RESI) {
            hResLocal = this->outQueue_.template AllocTensor<P>();
        }

        this->HMixCopyIn(hMixOffset, lenT);
        this->matmulRes_ = this->xInQueue_.template DeQue<P>();
        MhcPreMkVFPostSegmentsN4<HAS_RESI>(this->hPreBuff_, this->hPostBuff_, hResLocal,
            this->matmulRes_, this->invRmsUb_, this->alphaInUb_, this->biasInUb_,
            static_cast<uint16_t>(lenT), static_cast<uint32_t>(localOffset));
        PipeBarrier<PIPE_V>();

        if constexpr (HAS_RESI == MHC_PRE_HAS_RESI) {
            this->outQueue_.template EnQue<P>(hResLocal);
            hResLocal = this->outQueue_.template DeQue<P>();
            DataCopyExtParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = static_cast<uint32_t>(
                lenT * MHC_PRE_LOW_API_N4 * MHC_PRE_LOW_API_N4 * sizeof(P));
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            DataCopyPad(this->hResGm_[
                outputOffset * MHC_PRE_LOW_API_N4 * MHC_PRE_LOW_API_N4],
                hResLocal, copyParams);
            this->outQueue_.FreeTensor(hResLocal);
        }
        this->xInQueue_.FreeTensor(this->matmulRes_);
    }

    __aicore__ inline void PostProcess()
    {
        uint64_t localOffset = 0;
        for (uint64_t offsetT = this->vectorOffset_.offsetMStart;
             offsetT < this->vectorOffset_.offsetMEnd; offsetT += this->V1_BASE_T) {
            uint64_t lenT = AscendC::Std::min(
                static_cast<uint64_t>(this->V1_BASE_T), this->vectorOffset_.offsetMEnd - offsetT);
            PostProcessSegments(offsetT, lenT, localOffset);
            this->AIV1ProcessHPost(offsetT, lenT);
            this->AIV1ProcessHPre(offsetT, lenT);
            this->AIV1ProcessHIn(offsetT, lenT, this->v1ChunkDSize_);
            localOffset += this->V1_BASE_T;
        }
    }

private:
    GlobalTensor<P> partialMmGm_;
    GlobalTensor<P> partialRmsGm_;
    bool selected_ = false;
};

} // namespace MhcPre

#endif // MHC_PRE_M_K_SPLIT_CORE_ARCH35_H
