#ifndef CATLASS_GEMV_TILE_TILE_THRESHOLD_COMPUTE_HPP
#define CATLASS_GEMV_TILE_TILE_THRESHOLD_COMPUTE_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/gemm/helper.hpp"
#include "catlass/gemv/helper.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/gemm/gemm_type.hpp"

namespace FTSelf::Gemv::Tile {

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
                FTSelf::Gemv::helper::FT_THRESHOLD_ALGORITHM::AABFT,
                Catlass::Gemm::GemmType<ElementA, layout::RowMajor>,
                Catlass::Gemm::GemmType<ElementX, layout::VectorLayout>,
                Catlass::Gemm::GemmType<ElementY, layout::VectorLayout>,
                void>
{
    using ElementAccumulator =
        typename Catlass::Gemm::helper::ElementAccumulatorSelector<ElementA, ElementX>::ElementAccumulator;

    using FT_THRESHOLD_ALGORITHM = FTSelf::Gemv::helper::FT_THRESHOLD_ALGORITHM;

    static constexpr FT_THRESHOLD_ALGORITHM ALGO_TYPE = FTSelf::Gemv::helper::FT_THRESHOLD_ALGORITHM::AABFT;
    using LayoutDst = layout::RowMajor;
    using LayoutSrc = layout::RowMajor;
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementA);

    // Mehtods

    CATLASS_DEVICE
    TileThreCalc() {};

    //
    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementY> dstTensor,
        AscendC::LocalTensor<ElementA> srcTensor_m,
        AscendC::LocalTensor<ElementX> reduceTensor_v,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc,
        uint32_t dst_offset_ratio = 2
    )
    {
        uint32_t m_actual = layoutSrc.shape(0);
        uint32_t n_actual = layoutSrc.shape(1);
        uint32_t m_round = layoutDst.shape(0);
        uint32_t n_round = layoutDst.shape(1);
        uint32_t temp_repeat_size = BYTE_PER_C0 * 8 / sizeof(ElementX);
        uint32_t elem_repeat_size = ELE_NUM_PER_C0 * 8;
        uint32_t mask = temp_repeat_size;
        uint32_t repeattimes = CeilDiv(m_actual, temp_repeat_size);


        AscendC::Duplicate<ElementX>(
            reduceTensor_v,
            (ElementX)0.0,
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
        params.dstRepStride = RoundUp(temp_repeat_size, temp_repeat_size) / (BYTE_PER_C0 / sizeof(ElementX));
        params.src0RepStride = RoundUp(n_round, elem_repeat_size) / ELE_NUM_PER_C0;
        params.src1RepStride = 0;

        AscendC::BinaryRepeatParams max_params;
        max_params.dstBlkStride = 1;
        max_params.src0BlkStride = 1;
        max_params.src1BlkStride = 1;
        max_params.dstRepStride = RoundUp(temp_repeat_size, temp_repeat_size) / (BYTE_PER_C0 / sizeof(ElementX));
        max_params.src0RepStride = RoundUp(temp_repeat_size, temp_repeat_size) / (BYTE_PER_C0 / sizeof(ElementX));;
        max_params.src1RepStride = RoundUp(n_round, elem_repeat_size) / ELE_NUM_PER_C0;

        AscendC::UnaryRepeatParams abs_params;
        abs_params.dstBlkStride = 1;
        abs_params.srcBlkStride = 1;
        abs_params.dstRepStride = RoundUp(n_round, elem_repeat_size) / ELE_NUM_PER_C0;
        abs_params.srcRepStride = RoundUp(n_round, elem_repeat_size) / ELE_NUM_PER_C0;

        uint64_t abs_mask = temp_repeat_size;
        uint8_t unit_repeatTimes = m_actual;
        for (uint32_t i = 0; i < repeat_num; i++)
        {
            uint32_t offset = i * temp_repeat_size;
            // AscendC::Abs(dstLocal, srcLocal, mask, 4, { 1, 1, 8, 8 });
            /*
            __aicore__ inline void Abs(const LocalTensor<T>& dstLocal,
                const LocalTensor<T>& srcLocal, uint64_t mask, const uint8_t repeatTimes,
                const UnaryRepeatParams& repeatParams)
            */
            AscendC::Abs(
                srcTensor_m,
                srcTensor_m[offset],
                abs_mask,
                m_actual,
                abs_params);

            AscendC::PipeBarrier<PIPE_V>();

            /*
            __aicore__ inline void Max(const LocalTensor<T>& dstLocal,
            const LocalTensor<T>& src0Local, const LocalTensor<T>& src1Local,
            uint64_t mask, const uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
            */

            AscendC::Max(
                reduceTensor_v,
                reduceTensor_v,
                srcTensor_m,
                abs_mask,
                m_actual,
                max_params);

            AscendC::PipeBarrier<PIPE_V>();

        }
        // AscendC::SetMaskNorm();
        // AscendC::ResetMask();

        if (remain > 0)
        {
            uint32_t offset = repeat_num * temp_repeat_size;
            if (offset + remain > n_round)
            {
                remain = n_round - offset;
            }
            uint64_t remain_mask = remain;

            AscendC::Abs(
                srcTensor_m,
                srcTensor_m[offset],
                remain_mask,
                m_actual,
                abs_params);

            AscendC::PipeBarrier<PIPE_V>();

            AscendC::Max(
                reduceTensor_v,
                reduceTensor_v,
                srcTensor_m,
                remain_mask,
                m_actual,
                max_params);

            // AscendC::PipeBarrier<PIPE_V>();

        }

        int32_t reduce_mask = (repeat_num == 0) ? remain : temp_repeat_size;
        AscendC::PipeBarrier<PIPE_V>();
        // for(uint32_t i=0; i < m_actual){

        // }
        /*
        template <typename T, bool isSetMask = true>
        __aicore__ inline void WholeReduceMax(const LocalTensor<T>& dstLocal,
        const LocalTensor<T>& srcLocal, const int32_t mask,
        const int32_t repeatTimes, const int32_t dstRepStride,
        const int32_t srcBlkStride, const int32_t srcRepStride,
        ReduceOrder order = ReduceOrder::ORDER_VALUE_INDEX)
        */
        AscendC::WholeReduceMax<ElementX, true>(
            srcTensor_m,
            reduceTensor_v,
            reduce_mask,
            m_actual,
            1,
            1,
            8,
            AscendC::ReduceOrder::ORDER_ONLY_VALUE);


        AscendC::PipeBarrier<PIPE_V>();

        /*
        每个repeat能处理的数据量取决于数据精度、AI处理器型号，如float->half转换每次迭代操作64个源/目的元素。
        当源操作数和目的操作数位数不同时，计算输入参数以数据类型的字节较大的为准。例如，源操作数为half类型，目的操作数为int32_t类型时，为保证输出和输入是连续的，dstRepStride应设置为8，srcRepStride应设置为4。
        dst与src的应为不同Tensor，或同一Tensor的同一元素，不支持同一Tensor的不同元素。
        src为float，dst为float时，取整模式表示向整数取整（仍为float类型），其他情况表示向dst数据类型所能表示的数字取整。
        */

        uint32_t dstOffset = m_round * dst_offset_ratio;

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
            dstTensor[dstOffset],
            srcTensor_m,
            AscendC::RoundMode::CAST_NONE, m_actual);
        AscendC::PipeBarrier<PIPE_V>();

        /*
        template <typename T, bool isSetMask = true>
        __aicore__ inline void Muls(const LocalTensor<T>& dstLocal,
            const LocalTensor<T>& srcLocal, const T& scalarValue, const int32_t& calCount)
        */
        // __aicore__ inline void Max(const LocalTensor<T>& dstLocal,
        // const LocalTensor<T>& src0Local,
        // const LocalTensor<T>& src1Local, const int32_t& calCount)

        AscendC::Max(dstTensor, dstTensor, dstTensor[dstOffset], m_round);
        AscendC::PipeBarrier<PIPE_V>();

        // AscendC::Muls(dstTensor, dstTensor, alpha, m_actual);
        // AscendC::PipeBarrier<PIPE_V>();
    }
};



/*
    class ElementA,
    class ElementX,
    class ElementY
*/

template <>
struct TileThreCalc<Arch::AtlasA2,
                FTSelf::Gemv::helper::FT_THRESHOLD_ALGORITHM::AABFT,
                Catlass::Gemm::GemmType<float, layout::RowMajor>,
                Catlass::Gemm::GemmType<float, layout::VectorLayout>,
                Catlass::Gemm::GemmType<float, layout::VectorLayout>,
                void>
{
    using ElementA = float;
    using ElementX = float;
    using ElementY = float;

    using ElementAccumulator =
        typename Catlass::Gemm::helper::ElementAccumulatorSelector<ElementA, ElementX>::ElementAccumulator;

    using FT_THRESHOLD_ALGORITHM = FTSelf::Gemv::helper::FT_THRESHOLD_ALGORITHM;

    static constexpr FT_THRESHOLD_ALGORITHM ALGO_TYPE = FT_THRESHOLD_ALGORITHM::AABFT;
    using LayoutDst = layout::RowMajor;
    using LayoutSrc = layout::RowMajor;
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementA);

    // Mehtods

    CATLASS_DEVICE
    TileThreCalc() {};

    //
    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementY> dstTensor,
        AscendC::LocalTensor<ElementA> srcTensor_m,
        AscendC::LocalTensor<ElementX> reduceTensor_v,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc,
        uint32_t dst_offset_ratio = 2
    )
    {
        uint32_t m_actual = layoutSrc.shape(0);
        uint32_t n_actual = layoutSrc.shape(1);
        uint32_t m_round = layoutDst.shape(0);
        uint32_t n_round = layoutDst.shape(1);
        uint32_t temp_repeat_size = BYTE_PER_C0 * 8 / sizeof(ElementX);
        uint32_t elem_repeat_size = ELE_NUM_PER_C0 * 8;
        uint32_t mask = temp_repeat_size;
        uint32_t repeattimes = CeilDiv(m_actual, temp_repeat_size);

        AscendC::Duplicate<ElementX>(
            reduceTensor_v,
            (ElementX)0.0,
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
        params.dstRepStride = RoundUp(temp_repeat_size, temp_repeat_size) / (BYTE_PER_C0 / sizeof(ElementX));
        params.src0RepStride = RoundUp(n_round, elem_repeat_size) / ELE_NUM_PER_C0;
        params.src1RepStride = 0;

        AscendC::BinaryRepeatParams max_params;
        max_params.dstBlkStride = 1;
        max_params.src0BlkStride = 1;
        max_params.src1BlkStride = 1;
        max_params.dstRepStride = RoundUp(temp_repeat_size, temp_repeat_size) / (BYTE_PER_C0 / sizeof(ElementX));
        max_params.src0RepStride = RoundUp(temp_repeat_size, temp_repeat_size) / (BYTE_PER_C0 / sizeof(ElementX));;
        max_params.src1RepStride = RoundUp(n_round, elem_repeat_size) / ELE_NUM_PER_C0;

        AscendC::UnaryRepeatParams abs_params;
        abs_params.dstBlkStride = 1;
        abs_params.srcBlkStride = 1;
        abs_params.dstRepStride = RoundUp(n_round, elem_repeat_size) / ELE_NUM_PER_C0;
        abs_params.srcRepStride = RoundUp(n_round, elem_repeat_size) / ELE_NUM_PER_C0;

        uint64_t abs_mask = temp_repeat_size;
        uint8_t unit_repeatTimes = m_actual;
        for (uint32_t i = 0; i < repeat_num; i++)
        {
            uint32_t offset = i * temp_repeat_size;

            AscendC::Abs(
                srcTensor_m,
                srcTensor_m[offset],
                abs_mask,
                m_actual,
                abs_params);

            AscendC::PipeBarrier<PIPE_V>();

            AscendC::Max(
                reduceTensor_v,
                reduceTensor_v,
                srcTensor_m,
                abs_mask,
                m_actual,
                max_params);

            AscendC::PipeBarrier<PIPE_V>();

        }

        if (remain > 0)
        {
            uint32_t offset = repeat_num * temp_repeat_size;
            if (offset + remain > n_round)
            {
                remain = n_round - offset;
            }
            uint64_t remain_mask = remain;

            AscendC::Abs(
                srcTensor_m,
                srcTensor_m[offset],
                remain_mask,
                m_actual,
                abs_params);

            AscendC::PipeBarrier<PIPE_V>();

            AscendC::Max(
                reduceTensor_v,
                reduceTensor_v,
                srcTensor_m,
                remain_mask,
                m_actual,
                max_params);

            // AscendC::PipeBarrier<PIPE_V>();

        }

        int32_t reduce_mask = (repeat_num == 0) ? remain : temp_repeat_size;
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::WholeReduceMax<ElementY, true>(
            srcTensor_m,
            reduceTensor_v,
            reduce_mask,
            m_actual,
            1,
            1,
            8,
            AscendC::ReduceOrder::ORDER_ONLY_VALUE);


        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Max(dstTensor, dstTensor, srcTensor_m, m_round);
        AscendC::PipeBarrier<PIPE_V>();
    }
};


template <>
struct TileThreCalc<Arch::AtlasA2,
                FTSelf::Gemv::helper::FT_THRESHOLD_ALGORITHM::AABFT,
                Catlass::Gemm::GemmType<half, layout::RowMajor>,
                Catlass::Gemm::GemmType<half, layout::VectorLayout>,
                Catlass::Gemm::GemmType<float, layout::VectorLayout>,
                void>
{
    using ElementA = half;
    using ElementX = half;
    using ElementY = float;

    using FT_THRESHOLD_ALGORITHM = FTSelf::Gemv::helper::FT_THRESHOLD_ALGORITHM;

    static constexpr FT_THRESHOLD_ALGORITHM ALGO_TYPE = FT_THRESHOLD_ALGORITHM::AABFT;

    using ElementAccumulator =
        typename Catlass::Gemm::helper::ElementAccumulatorSelector<ElementA, ElementX>::ElementAccumulator;

    using LayoutDst = layout::RowMajor;
    using LayoutSrc = layout::RowMajor;
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementA);

    // Mehtods

    CATLASS_DEVICE
    TileThreCalc() {};

    //
    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementY> dstTensor,
        AscendC::LocalTensor<ElementA> srcTensor_m,
        AscendC::LocalTensor<ElementX> reduceTensor_v,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc,
        uint32_t dst_offset_ratio = 2
    )
    {
        uint32_t m_actual = layoutSrc.shape(0);
        uint32_t n_actual = layoutSrc.shape(1);
        uint32_t m_round = layoutDst.shape(0);
        uint32_t n_round = layoutDst.shape(1);
        uint32_t temp_repeat_size = BYTE_PER_C0 * 8 / sizeof(ElementX);
        uint32_t elem_repeat_size = ELE_NUM_PER_C0 * 8;
        uint32_t mask = temp_repeat_size;
        uint32_t repeattimes = CeilDiv(m_actual, temp_repeat_size);

        AscendC::Duplicate<ElementX>(
            reduceTensor_v,
            (ElementX)0.0,
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
        params.dstRepStride = RoundUp(temp_repeat_size, temp_repeat_size) / (BYTE_PER_C0 / sizeof(ElementX));
        params.src0RepStride = RoundUp(n_round, elem_repeat_size) / ELE_NUM_PER_C0;
        params.src1RepStride = 0;

        AscendC::BinaryRepeatParams max_params;
        max_params.dstBlkStride = 1;
        max_params.src0BlkStride = 1;
        max_params.src1BlkStride = 1;
        max_params.dstRepStride = RoundUp(temp_repeat_size, temp_repeat_size) / (BYTE_PER_C0 / sizeof(ElementX));
        max_params.src0RepStride = RoundUp(temp_repeat_size, temp_repeat_size) / (BYTE_PER_C0 / sizeof(ElementX));;
        max_params.src1RepStride = RoundUp(n_round, elem_repeat_size) / ELE_NUM_PER_C0;

        AscendC::UnaryRepeatParams abs_params;
        abs_params.dstBlkStride = 1;
        abs_params.srcBlkStride = 1;
        abs_params.dstRepStride = RoundUp(n_round, elem_repeat_size) / ELE_NUM_PER_C0;
        abs_params.srcRepStride = RoundUp(n_round, elem_repeat_size) / ELE_NUM_PER_C0;

        uint64_t abs_mask = temp_repeat_size;
        uint8_t unit_repeatTimes = m_actual;
        for (uint32_t i = 0; i < repeat_num; i++)
        {
            uint32_t offset = i * temp_repeat_size;

            AscendC::Abs(
                srcTensor_m,
                srcTensor_m[offset],
                abs_mask,
                m_actual,
                abs_params);

            AscendC::PipeBarrier<PIPE_V>();

            AscendC::Max(
                reduceTensor_v,
                reduceTensor_v,
                srcTensor_m,
                abs_mask,
                m_actual,
                max_params);

            AscendC::PipeBarrier<PIPE_V>();

        }

        if (remain > 0)
        {
            uint32_t offset = repeat_num * temp_repeat_size;
            if (offset + remain > n_round)
            {
                remain = n_round - offset;
            }
            uint64_t remain_mask = remain;

            AscendC::Abs(
                srcTensor_m,
                srcTensor_m[offset],
                remain_mask,
                m_actual,
                abs_params);

            AscendC::PipeBarrier<PIPE_V>();

            AscendC::Max(
                reduceTensor_v,
                reduceTensor_v,
                srcTensor_m,
                remain_mask,
                m_actual,
                max_params);

            // AscendC::PipeBarrier<PIPE_V>();

        }

        int32_t reduce_mask = (repeat_num == 0) ? remain : temp_repeat_size;
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::WholeReduceMax<ElementX, true>(
            srcTensor_m,
            reduceTensor_v,
            reduce_mask,
            m_actual,
            1,
            1,
            8,
            AscendC::ReduceOrder::ORDER_ONLY_VALUE);


        AscendC::PipeBarrier<PIPE_V>();

        /*
        每个repeat能处理的数据量取决于数据精度、AI处理器型号，如float->half转换每次迭代操作64个源/目的元素。
        当源操作数和目的操作数位数不同时，计算输入参数以数据类型的字节较大的为准。例如，源操作数为half类型，目的操作数为int32_t类型时，为保证输出和输入是连续的，dstRepStride应设置为8，srcRepStride应设置为4。
        dst与src的应为不同Tensor，或同一Tensor的同一元素，不支持同一Tensor的不同元素。
        src为float，dst为float时，取整模式表示向整数取整（仍为float类型），其他情况表示向dst数据类型所能表示的数字取整。
        */

        uint32_t dstOffset = m_round * dst_offset_ratio;

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
            dstTensor[dstOffset],
            srcTensor_m,
            AscendC::RoundMode::CAST_NONE, m_actual);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Max(dstTensor, dstTensor, dstTensor[dstOffset], m_round);
        AscendC::PipeBarrier<PIPE_V>();
    }
};


template <>
struct TileThreCalc<Arch::AtlasA2,
                FTSelf::Gemv::helper::FT_THRESHOLD_ALGORITHM::AABFT,
                Catlass::Gemm::GemmType<half, layout::RowMajor>,
                Catlass::Gemm::GemmType<half, layout::VectorLayout>,
                Catlass::Gemm::GemmType<half, layout::VectorLayout>,
                void>
{
    using ElementA = half;
    using ElementX = half;
    using ElementY = half;

    using FT_THRESHOLD_ALGORITHM = FTSelf::Gemv::helper::FT_THRESHOLD_ALGORITHM;
    static constexpr FT_THRESHOLD_ALGORITHM ALGO_TYPE = FT_THRESHOLD_ALGORITHM::AABFT;

    using ElementAccumulator =
        typename Catlass::Gemm::helper::ElementAccumulatorSelector<ElementA, ElementX>::ElementAccumulator;

    using LayoutDst = layout::RowMajor;
    using LayoutSrc = layout::RowMajor;
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementA);

    // Mehtods

    CATLASS_DEVICE
    TileThreCalc() {};

    //
    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementY> dstTensor,
        AscendC::LocalTensor<ElementA> srcTensor_m,
        AscendC::LocalTensor<ElementX> reduceTensor_v,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc,
        uint32_t dst_offset_ratio = 2
    )
    {
        uint32_t m_actual = layoutSrc.shape(0);
        uint32_t n_actual = layoutSrc.shape(1);
        uint32_t m_round = layoutDst.shape(0);
        uint32_t n_round = layoutDst.shape(1);
        uint32_t temp_repeat_size = BYTE_PER_C0 * 8 / sizeof(ElementX);
        uint32_t elem_repeat_size = ELE_NUM_PER_C0 * 8;
        uint32_t mask = temp_repeat_size;
        uint32_t repeattimes = CeilDiv(m_actual, temp_repeat_size);

        // ElementY aim_weight =

        //  AscendC::LocalTensor<ElementX> temp_space,

        AscendC::Duplicate<ElementX>(
            reduceTensor_v,
            (ElementX)0.0,
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
        params.dstRepStride = RoundUp(temp_repeat_size, temp_repeat_size) / (BYTE_PER_C0 / sizeof(ElementX));
        params.src0RepStride = RoundUp(n_round, elem_repeat_size) / ELE_NUM_PER_C0;
        params.src1RepStride = 0;

        AscendC::BinaryRepeatParams max_params;
        max_params.dstBlkStride = 1;
        max_params.src0BlkStride = 1;
        max_params.src1BlkStride = 1;
        max_params.dstRepStride = RoundUp(temp_repeat_size, temp_repeat_size) / (BYTE_PER_C0 / sizeof(ElementX));
        max_params.src0RepStride = RoundUp(temp_repeat_size, temp_repeat_size) / (BYTE_PER_C0 / sizeof(ElementX));;
        max_params.src1RepStride = RoundUp(n_round, elem_repeat_size) / ELE_NUM_PER_C0;

        // RoundUp(temp_repeat_size, temp_repeat_size) / (BYTE_PER_C0 / sizeof(ElementX));;

        AscendC::UnaryRepeatParams abs_params;
        abs_params.dstBlkStride = 1;
        abs_params.srcBlkStride = 1;
        abs_params.dstRepStride = RoundUp(n_round, elem_repeat_size) / ELE_NUM_PER_C0;
        abs_params.srcRepStride = RoundUp(n_round, elem_repeat_size) / ELE_NUM_PER_C0;

        uint64_t abs_mask = temp_repeat_size;
        uint8_t unit_repeatTimes = m_actual;
        for (uint32_t i = 0; i < repeat_num; i++)
        {
            uint32_t offset = i * temp_repeat_size;

            AscendC::Abs(
                srcTensor_m,
                srcTensor_m[offset],
                abs_mask,
                m_actual,
                abs_params);

            AscendC::PipeBarrier<PIPE_V>();

            AscendC::Max(
                reduceTensor_v,
                reduceTensor_v,
                srcTensor_m,
                abs_mask,
                m_actual,
                max_params);

            AscendC::PipeBarrier<PIPE_V>();

        }

        if (remain > 0)
        {
            uint32_t offset = repeat_num * temp_repeat_size;
            if (offset + remain > n_round)
            {
                remain = n_round - offset;
            }
            uint64_t remain_mask = remain;

            AscendC::Abs(
                srcTensor_m,
                srcTensor_m[offset],
                remain_mask,
                m_actual,
                abs_params);

            AscendC::PipeBarrier<PIPE_V>();

            AscendC::Max(
                reduceTensor_v,
                reduceTensor_v,
                srcTensor_m,
                remain_mask,
                m_actual,
                max_params);

            // AscendC::PipeBarrier<PIPE_V>();

        }

        int32_t reduce_mask = (repeat_num == 0) ? remain : temp_repeat_size;
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::WholeReduceMax<ElementY, true>(
            srcTensor_m,
            reduceTensor_v,
            reduce_mask,
            m_actual,
            1,
            1,
            8,
            AscendC::ReduceOrder::ORDER_ONLY_VALUE);


        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Max(dstTensor, dstTensor, srcTensor_m, m_round);
        AscendC::PipeBarrier<PIPE_V>();
    }
};


}

#endif
