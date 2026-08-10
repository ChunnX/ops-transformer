#ifndef CATLASS_GEMV_TILE_TILE_THRESHOLD_MEAN_MAX_STD_FUSED_HPP
#define CATLASS_GEMV_TILE_TILE_THRESHOLD_MEAN_MAX_STD_FUSED_HPP

#include "../../catlass.hpp"
#include "../../arch/arch.hpp"
#include "../../gemm/helper.hpp"
#include "../helper.hpp"
#include "../../layout/layout.hpp"
#include "../../gemm/gemm_type.hpp"

namespace Vecself::Gemv::Tile {

using namespace Catlass;
// template <
//     /// Tag indicating architecture
//     class ArchTag,
//     class AType,
//     class XType,
//     class YType,
//     class BiasType = void
// >
// struct TileThreCalc
// {
//     static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported TileThreCalc, can not find the specialization.");
// };

template <
    class ElementA,
    class ElementX,
    class ElementY
>
struct TileThreCalc<Arch::AtlasA2,
                Vecself::Gemv::helper::FT_THRESHOLD_ALGORITHM::ASVAR,
                Gemm::GemmType<ElementA, layout::RowMajor>,
                Gemm::GemmType<ElementX, layout::VectorLayout>,
                Gemm::GemmType<ElementY, layout::VectorLayout>,
                void>
{
    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementX, ElementY>::ElementAccumulator;

    using FT_THRESHOLD_ALGORITHM = Vecself::Gemv::helper::FT_THRESHOLD_ALGORITHM;

    static constexpr FT_THRESHOLD_ALGORITHM ALGO_TYPE = Vecself::Gemv::helper::FT_THRESHOLD_ALGORITHM::ASVAR;
    using LayoutDst = layout::VectorLayout;
    using LayoutSrc = layout::VectorLayout;
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementY);

    // Mehtods

    CATLASS_DEVICE
    TileThreCalc() {};

    // 
    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementY> dstTensor,
        AscendC::LocalTensor<ElementX> srcMeanTensor,
        AscendC::LocalTensor<ElementY> srcStdTensor,
        AscendC::LocalTensor<ElementY> thre_workspace,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc,
        ElementY kn_ratio_factor, ElementY kn_sqrt_ratio_factor,
        ElementY k_sqrt_n_ratio_factor, ElementY B_slice_mean_abs,
        ElementY B_slice_std, ElementY e_max)
    {
        uint32_t m_actual = layoutSrc.shape(0);
        uint32_t m_round = layoutDst.shape(0);

        uint32_t temp_repeat_size = BYTE_PER_C0 * 8 / sizeof(ElementX);
        uint32_t elem_repeat_size = ELE_NUM_PER_C0 * 8;
        uint32_t mask = elem_repeat_size;

        uint64_t add_mask = elem_repeat_size;
        uint32_t repeattimes = CeilDiv(m_actual, elem_repeat_size);


        AscendC::Duplicate<ElementY>(
            thre_workspace,
            (ElementY)0.0,
            int32_t(m_actual) * 5
        );

        /*
        template <typename T>
        __aicore__ inline void Abs(
            const LocalTensor<T>& dstLocal, 
            const LocalTensor<T>& srcLocal, const int32_t& calCount)
        */
        AscendC::Abs(srcMeanTensor, srcMeanTensor, int32_t(m_actual));
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::UnaryRepeatParams castparams;
        castparams.dstBlkStride = 1;
        castparams.srcBlkStride = 1;
        castparams.dstRepStride = 8;
        castparams.srcRepStride = 4;

        /*
        template <typename T1, typename T2>
        __aicore__ inline void Cast(const LocalTensor<T1>& dstLocal, 
            const LocalTensor<T2>& srcLocal, 
            const RoundMode& round_mode, const uint32_t calCount)

        (uint64_t)mask, repeattimes, castparams
        */
        AscendC::Cast<ElementY, ElementX>(
            thre_workspace[4 * m_actual],
            srcMeanTensor,
            AscendC::RoundMode::CAST_NONE, m_actual);

        AscendC::PipeBarrier<PIPE_V>();

        AscendC::SetMaskCount();
        AscendC::SetVectorMask<ElementY, AscendC::MaskMode::COUNTER>(m_actual);
        AscendC::Muls<ElementY,false>(
            thre_workspace,
            thre_workspace[4 * m_actual],
            kn_ratio_factor*B_slice_mean_abs,
            AscendC::MASK_PLACEHOLDER,
            1,
            AscendC::UnaryRepeatParams{}
        );

        AscendC::Muls<ElementY,false>(
            thre_workspace[m_actual],
            thre_workspace[4 * m_actual],
            4 * kn_sqrt_ratio_factor * B_slice_std,
            AscendC::MASK_PLACEHOLDER,
            1,
            AscendC::UnaryRepeatParams{}
        );

        AscendC::Muls<ElementY,false>(
            thre_workspace[m_actual*2],
            srcStdTensor,
            4 * k_sqrt_n_ratio_factor * B_slice_mean_abs,
            AscendC::MASK_PLACEHOLDER,
            1,
            AscendC::UnaryRepeatParams{}
        );

        AscendC::Muls<ElementY,false>(
            thre_workspace[m_actual*3],
            srcStdTensor,
            4 * kn_sqrt_ratio_factor * B_slice_std,
            AscendC::MASK_PLACEHOLDER,
            1,
            AscendC::UnaryRepeatParams{}
        );
        AscendC::SetMaskNorm();
        AscendC::ResetMask();

        AscendC::PipeBarrier<PIPE_V>();

        add_mask = (m_actual < elem_repeat_size) ? m_actual : elem_repeat_size;

        AscendC::BinaryRepeatParams params;
        params.dstBlkStride = 1;
        params.src0BlkStride = 1;
        params.src1BlkStride = 1;
        params.dstRepStride = 8;
        params.src0RepStride = 8;
        params.src1RepStride = 8;

        AscendC::Add<ElementY, true>(
            dstTensor,
            thre_workspace,
            thre_workspace[m_actual],
            add_mask,
            CeilDiv(m_round, elem_repeat_size),
            params);

        AscendC::Add<ElementY, true>(
            thre_workspace[2 * m_actual],
            thre_workspace[3 * m_actual],
            thre_workspace[2 * m_actual],
            add_mask,
            CeilDiv(m_round, elem_repeat_size),
            params);

        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Add<ElementY, true>(
            dstTensor,
            thre_workspace[m_actual * 2],
            dstTensor,
            add_mask,
            CeilDiv(m_round, elem_repeat_size),
            params);
        
        AscendC::PipeBarrier<PIPE_V>();


        AscendC::SetMaskCount();
        AscendC::SetVectorMask<ElementY, AscendC::MaskMode::COUNTER>(m_actual);
        AscendC::Muls<ElementY,false>(
            dstTensor,
            dstTensor,
            e_max,
            AscendC::MASK_PLACEHOLDER,
            1,
            AscendC::UnaryRepeatParams{}
        );
        AscendC::SetMaskNorm();
        AscendC::ResetMask();
            
    }
};



/*
    class ElementA,
    class ElementX,
    class ElementY
*/

template <
    class ElementA
>
struct TileThreCalc<Arch::AtlasA2,
                Vecself::Gemv::helper::FT_THRESHOLD_ALGORITHM::ASVAR,
                Gemm::GemmType<ElementA, layout::RowMajor>,
                Gemm::GemmType<float, layout::VectorLayout>,
                Gemm::GemmType<float, layout::VectorLayout>,
                void>
{
    using ElementX = float;
    using ElementY = float;
    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementX, ElementY>::ElementAccumulator;

    using FT_THRESHOLD_ALGORITHM = Vecself::Gemv::helper::FT_THRESHOLD_ALGORITHM;

    static constexpr FT_THRESHOLD_ALGORITHM ALGO_TYPE = Vecself::Gemv::helper::FT_THRESHOLD_ALGORITHM::ASVAR;
    using LayoutDst = layout::VectorLayout;
    using LayoutSrc = layout::VectorLayout;
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementY);

    // Mehtods

    CATLASS_DEVICE
    TileThreCalc() {};

    // 
    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementY> dstTensor,
        AscendC::LocalTensor<ElementX> srcMeanTensor,
        AscendC::LocalTensor<ElementY> srcStdTensor,
        AscendC::LocalTensor<ElementY> thre_workspace,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc,
        ElementY kn_ratio_factor, ElementY kn_sqrt_ratio_factor,
        ElementY k_sqrt_n_ratio_factor, ElementY B_slice_mean_abs,
        ElementY B_slice_std, ElementY e_max)
    {
        uint32_t m_actual = layoutSrc.shape(0);
        uint32_t m_round = layoutDst.shape(0);

        uint32_t temp_repeat_size = BYTE_PER_C0 * 8 / sizeof(ElementX);
        uint32_t elem_repeat_size = ELE_NUM_PER_C0 * 8;
        uint32_t mask = elem_repeat_size;

        uint64_t add_mask = elem_repeat_size;
        uint32_t repeattimes = CeilDiv(m_actual, elem_repeat_size);


        AscendC::Duplicate<ElementY>(
            thre_workspace,
            (ElementY)0.0,
            int32_t(m_actual) * 5
        );

        /*
        template <typename T>
        __aicore__ inline void Abs(
            const LocalTensor<T>& dstLocal, 
            const LocalTensor<T>& srcLocal, const int32_t& calCount)
        */
        AscendC::Abs(thre_workspace[4 * m_actual], srcMeanTensor, int32_t(m_actual));
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::SetMaskCount();
        AscendC::SetVectorMask<ElementY, AscendC::MaskMode::COUNTER>(m_actual);
        AscendC::Muls<ElementY,false>(
            thre_workspace,
            thre_workspace[4 * m_actual],
            kn_ratio_factor*B_slice_mean_abs,
            AscendC::MASK_PLACEHOLDER,
            1,
            AscendC::UnaryRepeatParams{}
        );

        AscendC::Muls<ElementY,false>(
            thre_workspace[m_actual],
            thre_workspace[4 * m_actual],
            4 * kn_sqrt_ratio_factor * B_slice_std,
            AscendC::MASK_PLACEHOLDER,
            1,
            AscendC::UnaryRepeatParams{}
        );

        AscendC::Muls<ElementY,false>(
            thre_workspace[m_actual*2],
            srcStdTensor,
            4 * k_sqrt_n_ratio_factor * B_slice_mean_abs,
            AscendC::MASK_PLACEHOLDER,
            1,
            AscendC::UnaryRepeatParams{}
        );

        AscendC::Muls<ElementY,false>(
            thre_workspace[m_actual*3],
            srcStdTensor,
            4 * kn_sqrt_ratio_factor * B_slice_std,
            AscendC::MASK_PLACEHOLDER,
            1,
            AscendC::UnaryRepeatParams{}
        );
        AscendC::SetMaskNorm();
        AscendC::ResetMask();

        AscendC::PipeBarrier<PIPE_V>();

        add_mask = (m_actual < elem_repeat_size) ? m_actual : elem_repeat_size;

        AscendC::BinaryRepeatParams params;
        params.dstBlkStride = 1;
        params.src0BlkStride = 1;
        params.src1BlkStride = 1;
        params.dstRepStride = 8;
        params.src0RepStride = 8;
        params.src1RepStride = 8;

        AscendC::Add<ElementY, true>(
            dstTensor,
            thre_workspace,
            thre_workspace[m_actual],
            add_mask,
            CeilDiv(m_round, elem_repeat_size),
            params);

        AscendC::Add<ElementY, true>(
            thre_workspace[2 * m_actual],
            thre_workspace[3 * m_actual],
            thre_workspace[2 * m_actual],
            add_mask,
            CeilDiv(m_round, elem_repeat_size),
            params);

        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Add<ElementY, true>(
            dstTensor,
            thre_workspace[m_actual * 2],
            dstTensor,
            add_mask,
            CeilDiv(m_round, elem_repeat_size),
            params);
        
        AscendC::PipeBarrier<PIPE_V>();


        AscendC::SetMaskCount();
        AscendC::SetVectorMask<ElementY, AscendC::MaskMode::COUNTER>(m_actual);
        AscendC::Muls<ElementY,false>(
            dstTensor,
            dstTensor,
            e_max,
            AscendC::MASK_PLACEHOLDER,
            1,
            AscendC::UnaryRepeatParams{}
        );
        AscendC::SetMaskNorm();
        AscendC::ResetMask();
            
    }
};


template <
    class ElementA
>
struct TileThreCalc<Arch::AtlasA2,
                Vecself::Gemv::helper::FT_THRESHOLD_ALGORITHM::ASVAR,
                Gemm::GemmType<ElementA, layout::RowMajor>,
                Gemm::GemmType<half, layout::VectorLayout>,
                Gemm::GemmType<half, layout::VectorLayout>,
                void>
{
    using ElementX = half;
    using ElementY = half;
    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementX, ElementY>::ElementAccumulator;

    using FT_THRESHOLD_ALGORITHM = Vecself::Gemv::helper::FT_THRESHOLD_ALGORITHM;

    static constexpr FT_THRESHOLD_ALGORITHM ALGO_TYPE = Vecself::Gemv::helper::FT_THRESHOLD_ALGORITHM::ASVAR;
    using LayoutDst = layout::VectorLayout;
    using LayoutSrc = layout::VectorLayout;
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementY);

    // Mehtods

    CATLASS_DEVICE
    TileThreCalc() {};

    // 
    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementY> dstTensor,
        AscendC::LocalTensor<ElementX> srcMeanTensor,
        AscendC::LocalTensor<ElementY> srcStdTensor,
        AscendC::LocalTensor<ElementY> thre_workspace,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc,
        ElementY kn_ratio_factor, ElementY kn_sqrt_ratio_factor,
        ElementY k_sqrt_n_ratio_factor, ElementY B_slice_mean_abs,
        ElementY B_slice_std, ElementY e_max)
    {
        uint32_t m_actual = layoutSrc.shape(0);
        uint32_t m_round = layoutDst.shape(0);

        uint32_t temp_repeat_size = BYTE_PER_C0 * 8 / sizeof(ElementX);
        uint32_t elem_repeat_size = ELE_NUM_PER_C0 * 8;
        uint32_t mask = elem_repeat_size;

        uint64_t add_mask = elem_repeat_size;
        uint32_t repeattimes = CeilDiv(m_actual, elem_repeat_size);


        AscendC::Duplicate<ElementY>(
            thre_workspace,
            (ElementY)0.0,
            int32_t(m_actual) * 5
        );

        /*
        template <typename T>
        __aicore__ inline void Abs(
            const LocalTensor<T>& dstLocal, 
            const LocalTensor<T>& srcLocal, const int32_t& calCount)
        */
        AscendC::Abs(thre_workspace[4 * m_actual], srcMeanTensor, int32_t(m_actual));
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::SetMaskCount();
        AscendC::SetVectorMask<ElementY, AscendC::MaskMode::COUNTER>(m_actual);
        AscendC::Muls<ElementY,false>(
            thre_workspace,
            thre_workspace[4 * m_actual],
            (ElementY)((float)kn_ratio_factor*(float)B_slice_mean_abs),
            AscendC::MASK_PLACEHOLDER,
            1,
            AscendC::UnaryRepeatParams{}
        );

        AscendC::Muls<ElementY,false>(
            thre_workspace[m_actual],
            thre_workspace[4 * m_actual],
            (ElementY)(4.0f * (float)kn_sqrt_ratio_factor * (float)B_slice_std),
            AscendC::MASK_PLACEHOLDER,
            1,
            AscendC::UnaryRepeatParams{}
        );

        AscendC::Muls<ElementY,false>(
            thre_workspace[m_actual*2],
            srcStdTensor,
            (ElementY)(4.0f * (float)k_sqrt_n_ratio_factor * (float)B_slice_mean_abs),
            AscendC::MASK_PLACEHOLDER,
            1,
            AscendC::UnaryRepeatParams{}
        );

        AscendC::Muls<ElementY,false>(
            thre_workspace[m_actual*3],
            srcStdTensor,
            (ElementY)(4.0f * (float)kn_sqrt_ratio_factor * (float)B_slice_std),
            AscendC::MASK_PLACEHOLDER,
            1,
            AscendC::UnaryRepeatParams{}
        );
        AscendC::SetMaskNorm();
        AscendC::ResetMask();

        AscendC::PipeBarrier<PIPE_V>();

        add_mask = (m_actual < elem_repeat_size) ? m_actual : elem_repeat_size;

        AscendC::BinaryRepeatParams params;
        params.dstBlkStride = 1;
        params.src0BlkStride = 1;
        params.src1BlkStride = 1;
        params.dstRepStride = 8;
        params.src0RepStride = 8;
        params.src1RepStride = 8;

        AscendC::Add<ElementY, true>(
            dstTensor,
            thre_workspace,
            thre_workspace[m_actual],
            add_mask,
            CeilDiv(m_round, elem_repeat_size),
            params);

        AscendC::Add<ElementY, true>(
            thre_workspace[2 * m_actual],
            thre_workspace[3 * m_actual],
            thre_workspace[2 * m_actual],
            add_mask,
            CeilDiv(m_round, elem_repeat_size),
            params);

        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Add<ElementY, true>(
            dstTensor,
            thre_workspace[m_actual * 2],
            dstTensor,
            add_mask,
            CeilDiv(m_round, elem_repeat_size),
            params);
        
        AscendC::PipeBarrier<PIPE_V>();


        AscendC::SetMaskCount();
        AscendC::SetVectorMask<ElementY, AscendC::MaskMode::COUNTER>(m_actual);
        AscendC::Muls<ElementY,false>(
            dstTensor,
            dstTensor,
            e_max,
            AscendC::MASK_PLACEHOLDER,
            1,
            AscendC::UnaryRepeatParams{}
        );
        AscendC::SetMaskNorm();
        AscendC::ResetMask();
            
    }
};

}

#endif
