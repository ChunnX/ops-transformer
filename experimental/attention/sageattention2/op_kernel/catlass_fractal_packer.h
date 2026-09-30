#pragma once

#include "platform_traits.h"
#include "ring_buffer_layout.h"

#if defined(__CCE__)
#include "kernel_operator.h"

namespace sage::memory {

template <typename ElementType, uint32_t ALIGN_BYTES = 32U>
struct TensorRowChannel {
    using Element = ElementType;
    static constexpr uint32_t ELEMS_PER_BLOCK = ALIGN_BYTES / sizeof(Element);

    __aicore__ static inline void FetchPadded(AscendC::LocalTensor<Element> &destination, const AscendC::GlobalTensor<Element> &source, uint64_t linearOffset, uint32_t validElements, uint32_t paddedElements) {
        AscendC::Duplicate(destination, static_cast<Element>(0), paddedElements);
        Sync<AscendC::HardEvent::V_MTE2>();

        const uint32_t byteSize = validElements * static_cast<uint32_t>(sizeof(Element));
        const uint32_t unalignedElems = validElements % ELEMS_PER_BLOCK;
        const uint8_t tailPad = static_cast<uint8_t>(unalignedElems != 0U ? (ELEMS_PER_BLOCK - unalignedElems) : 0U);

        AscendC::DataCopyExtParams cpyParams{1, byteSize, 0, 0, 0};
        AscendC::DataCopyPadExtParams<Element> padParams{true, 0, tailPad, static_cast<Element>(0)};
        AscendC::DataCopyPad(destination, source[linearOffset], cpyParams, padParams);

        Sync<AscendC::HardEvent::MTE2_V>();
    }

    __aicore__ static inline void Flush(AscendC::GlobalTensor<Element> &destination, uint64_t linearOffset, const AscendC::LocalTensor<Element> &source, uint32_t count) {
        Sync<AscendC::HardEvent::V_MTE3>();
        const uint32_t byteCount = count * static_cast<uint32_t>(sizeof(Element));
        AscendC::DataCopyExtParams cpyParams{1, byteCount, 0, 0, 0};
        AscendC::DataCopyPad(destination[linearOffset], source, cpyParams);
        Sync<AscendC::HardEvent::MTE3_V>();
    }
};

template <uint32_t NZ_STRIDE, uint32_t PLANE_ELEMS = 32U>
struct NZFractalPacker {
    static constexpr uint32_t STRIDE = NZ_STRIDE;
    static constexpr uint32_t BLOCK_ELEMS = PLANE_ELEMS;

    template <typename RegT, typename MaskT>
    __simd_callee__ static inline void StorePair(__ubuf__ float8_e4m3_t *outPtr, uint32_t planeIdx, uint32_t rowIdx, RegT &regTensor, MaskT &maskLower, MaskT &maskUpper) {
        using namespace AscendC::MicroAPI;
        const uint32_t primaryOffset = planeIdx * STRIDE + rowIdx * BLOCK_ELEMS;
        const uint32_t secondaryOffset = (planeIdx + 1U) * STRIDE + rowIdx * BLOCK_ELEMS - BLOCK_ELEMS;

        StoreAlign<float8_e4m3_t, StoreDist::DIST_PACK4_B32>(outPtr + primaryOffset, regTensor, maskLower);
        StoreAlign<float8_e4m3_t, StoreDist::DIST_PACK4_B32>(outPtr + secondaryOffset, regTensor, maskUpper);
    }
};

template <uint32_t STRIDE, uint32_t PLANE_ELEMS = 32U>
using PackFractal = NZFractalPacker<STRIDE, PLANE_ELEMS>;

template <typename T, uint32_t ALIGN_BYTES = 32U>
__aicore__ inline void ReadRow(AscendC::LocalTensor<T> dst, AscendC::GlobalTensor<T> src, uint64_t off, uint32_t d, uint32_t dp) {
    TensorRowChannel<T, ALIGN_BYTES>::FetchPadded(dst, src, off, d, dp);
}

template <typename T, uint32_t ALIGN_BYTES = 32U>
__aicore__ inline void WriteRow(AscendC::GlobalTensor<T> dst, uint64_t off, AscendC::LocalTensor<T> src, uint32_t n) {
    TensorRowChannel<T, ALIGN_BYTES>::Flush(dst, off, src, n);
}

template <uint32_t STRIDE, uint32_t PLANE_ELEMS = 32U>
__simd_callee__ inline void StoreNZPlanePair(__ubuf__ float8_e4m3_t *out, uint32_t plane, uint32_t row, AscendC::MicroAPI::RegTensor<float8_e4m3_t> &packed, AscendC::MicroAPI::MaskReg &first32, AscendC::MicroAPI::MaskReg &last32) {
    NZFractalPacker<STRIDE, PLANE_ELEMS>::StorePair(out, plane, row, packed, first32, last32);
}

} // namespace sage::memory

namespace sage {
using memory::TensorRowChannel, memory::NZFractalPacker, memory::PackFractal, memory::ReadRow, memory::WriteRow, memory::StoreNZPlanePair;
}
#endif // __CCE__
