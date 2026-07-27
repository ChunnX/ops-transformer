#ifndef MHC_PRE_LOW_API_BASE_ARCH35_H
#define MHC_PRE_LOW_API_BASE_ARCH35_H

#include "kernel_operator.h"

namespace MhcPre {

using namespace AscendC;

constexpr uint32_t MHC_PRE_LOW_API_BLOCK_SIZE = 32;
constexpr uint32_t MHC_PRE_LOW_API_C0_SIZE = 8;
constexpr uint32_t MHC_PRE_LOW_API_VL_FP32 = 64;

constexpr AscendC::MicroAPI::CastTrait MHC_PRE_LOW_API_CAST_B16_TO_B32 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
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

__aicore__ inline void MhcPreLowApiLoadBroadcast(AscendC::MicroAPI::RegTensor<float>& dst,
    __local_mem__ float* src, AscendC::MicroAPI::MaskReg mask, uint32_t offset)
{
    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(dst, src + offset);
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


} // namespace MhcPre

#endif // MHC_PRE_LOW_API_BASE_ARCH35_H
