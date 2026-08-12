#ifndef FTSELF_CATLASS_GEMV_TILE_TILE_VMAD_HPP
#define FTSELF_CATLASS_GEMV_TILE_TILE_VMAD_HPP

#include_next "catlass/gemv/tile/tile_vmad.hpp"

namespace FTSelf::Gemv::Tile {

using namespace Catlass;

template <
    class ArchTag,
    class AType,
    class XType,
    class YType,
    class BiasType = void
>
struct TileVmad : Catlass::Gemv::Tile::TileVmad<ArchTag, AType, XType, YType, BiasType> {};

} // namespace FTSelf::Gemv::Tile

namespace FTSelf::Gemv::Tile {

template <>
struct TileVmad<Arch::AtlasA2,
                Catlass::Gemm::GemmType<half, layout::RowMajor>,
                Catlass::Gemm::GemmType<half, layout::VectorLayout>,
                Catlass::Gemm::GemmType<float, layout::VectorLayout>,
                void>
{
    using ElementA = half;
    using ElementX = half;
    using ElementY = float;

    using ElementAccumulator = ElementY;
        // typename Catlass::Gemm::helper::ElementAccumulatorSelector<ElementA, ElementX>::ElementAccumulator;

    using LayoutDst = layout::RowMajor;
    using LayoutSrc = layout::RowMajor;
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementA);

    // Methods

    CATLASS_DEVICE
    TileVmad() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementY> dstTensor,
        AscendC::LocalTensor<ElementX> srcTensor_v,
        AscendC::LocalTensor<ElementA> srcTensor_m,
        AscendC::LocalTensor<ElementAccumulator> temp,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc
    )
    {
        uint32_t m_actual = layoutSrc.shape(0);
        uint32_t n_actual = layoutSrc.shape(1);
        uint32_t m_round = layoutDst.shape(0);
        uint32_t n_round = layoutDst.shape(1);
        uint32_t temp_repeat_size = BYTE_PER_C0 * 8 / sizeof(ElementAccumulator);
        uint32_t elem_repeat_size = ELE_NUM_PER_C0 * 8;
        uint32_t mask = temp_repeat_size;
        uint32_t repeattimes = CeilDiv(m_actual, temp_repeat_size);
        AscendC::Duplicate<ElementAccumulator>(
            temp,
            (ElementAccumulator)0.0,
            temp_repeat_size,
            CeilDiv(m_round * temp_repeat_size, temp_repeat_size),
            1,
            8
        );

        uint32_t repeat_num = n_actual / temp_repeat_size;
        uint32_t remain = n_actual % temp_repeat_size;

        AscendC::PipeBarrier<PIPE_V>();
        AscendC::BinaryRepeatParams params;
        params.dstBlkStride = 1;
        params.src0BlkStride = 1;
        params.src1BlkStride = 1;
        params.dstRepStride = RoundUp(temp_repeat_size, temp_repeat_size) / (BYTE_PER_C0 / sizeof(ElementAccumulator));
        params.src0RepStride = RoundUp(n_round, elem_repeat_size) / ELE_NUM_PER_C0;
        params.src1RepStride = 0;
        AscendC::SetMaskCount();
        AscendC::SetVectorMask<ElementAccumulator, AscendC::MaskMode::COUNTER>(m_actual * temp_repeat_size);
        for (uint32_t i = 0; i < repeat_num; i++)
        {
            uint32_t offset = i * temp_repeat_size;
            AscendC::MulAddDst<ElementAccumulator, ElementA, false>(
                temp,
                srcTensor_m[offset],
                srcTensor_v[offset],
                AscendC::MASK_PLACEHOLDER,
                1,
                params);

            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::SetMaskNorm();
        AscendC::ResetMask();

        if (remain > 0)
        {
            uint32_t offset = repeat_num * temp_repeat_size;
            if (offset + remain > n_round)
            {
                remain = n_round - offset;
            }
            uint64_t remain_mask = remain;
            AscendC::MulAddDst<ElementAccumulator, ElementA, true>(
                temp,
                srcTensor_m[offset],
                srcTensor_v[offset],
                remain_mask,
                m_actual,
                params);
        }

        uint64_t reduce_mask = (repeat_num == 0) ? remain : temp_repeat_size;
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::WholeReduceSum<ElementAccumulator, true>(
            temp,
            temp,
            reduce_mask,
            m_actual,
            1,
            1,
            8);
        AscendC::PipeBarrier<PIPE_V>();

        // AscendC::UnaryRepeatParams castparams;
        // castparams.dstBlkStride = 1;
        // castparams.srcBlkStride = 1;
        // castparams.dstRepStride = 4;
        // castparams.srcRepStride = 8;
        // AscendC::Cast<ElementA, ElementAccumulator, true>(
        //     srcTensor_m,
        //     temp,
        //     AscendC::RoundMode::CAST_NONE,
        //     (uint64_t)mask,
        //     repeattimes,
        //     castparams);
        // AscendC::PipeBarrier<PIPE_V>();

        uint64_t add_mask = (m_actual < temp_repeat_size) ? m_actual : temp_repeat_size;
        params.dstRepStride = 8;
        params.src0RepStride = 8;
        params.src1RepStride = 8;
        AscendC::Add<ElementY, true>(
            dstTensor,
            temp,
            dstTensor,
            (uint64_t)add_mask,
            CeilDiv(m_round, temp_repeat_size),
            params);
    }
};

} // namespace FTSelf::Gemv::Tile

#endif // FTSELF_CATLASS_GEMV_TILE_TILE_VMAD_HPP
