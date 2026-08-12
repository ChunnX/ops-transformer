#ifndef FTSELF_CATLASS_GEMM_TILE_ATLASA2_COPY_L1_TO_L0B_HPP
#define FTSELF_CATLASS_GEMM_TILE_ATLASA2_COPY_L1_TO_L0B_HPP

#include_next "catlass/gemm/tile/atlasa2/copy_l1_to_l0b.hpp"

namespace FTSelf::Gemm::Tile {

using namespace Catlass;

template <class ArchTag, class SrcType, class DstType = void>
struct CopyL1ToL0B : Catlass::Gemm::Tile::CopyL1ToL0B<ArchTag, SrcType, DstType> {};

// Compatibility specialization required by the matmul_ft GEMV path. This
// A1(zN) -> B2(zN) form is not provided by the current upstream Catlass.
template <class ArchTag, class Element>
struct CopyL1ToL0B<ArchTag,
    Catlass::Gemm::GemmType<Element, layout::zN, AscendC::TPosition::A1>,
    Catlass::Gemm::GemmType<Element, layout::zN, AscendC::TPosition::B2>> {
    using LayoutDst = layout::zN;
    using LayoutSrc = layout::zN;

    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BYTE_PER_FRACTAL / sizeof(Element);

    CATLASS_DEVICE
    CopyL1ToL0B() {}

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::LocalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst,
        LayoutSrc const &layoutSrc)
    {
        AscendC::LoadData2DParams loadDataParams;
        loadDataParams.startIndex = 0;
        loadDataParams.repeatTimes = static_cast<uint16_t>(layoutDst.shape(1));
        loadDataParams.srcStride = layoutSrc.stride(1) / ELE_NUM_PER_FRACTAL;
        loadDataParams.sid = 0;
        loadDataParams.dstGap = layoutDst.stride(1) / ELE_NUM_PER_FRACTAL - 1;
        loadDataParams.ifTranspose = false;
        loadDataParams.addrMode = 0;

        for (uint32_t i = 0; i < layoutDst.shape(3); i++) {
            AscendC::LoadData(
                dstTensor[i * layoutDst.stride(3)],
                srcTensor[i * layoutSrc.stride(3)],
                loadDataParams);
        }
    }
};


} // namespace FTSelf::Gemm::Tile

#endif // FTSELF_CATLASS_GEMM_TILE_ATLASA2_COPY_L1_TO_L0B_HPP
