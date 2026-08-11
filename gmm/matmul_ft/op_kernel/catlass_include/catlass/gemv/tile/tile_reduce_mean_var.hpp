#ifndef CATLASS_GEMV_TILE_TILE_REDUCE_MEAN_VAR_STD_FUSED_HPP_BAK
#define CATLASS_GEMV_TILE_TILE_REDUCE_MEAN_VAR_STD_FUSED_HPP_BAK

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/gemm/helper.hpp"
#include "catlass/gemv/helper.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/gemm/gemm_type.hpp"

namespace FTSelf::Gemv::Tile {

using namespace Catlass;
template <
    /// Tag indicating architecture
    class ArchTag,
    FTSelf::Gemv::helper::FT_THRESHOLD_ALGORITHM ALGO_TYPE_,
    FTSelf::Gemv::helper::FT_REDUCE_TYPE REDUCE_TYPE_,
    class AType,
    class XType,
    class YType,
    class BiasType = void
>
struct TileReduce
{
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported TileReduce, can not find the specialization.");
};

template <
    class ElementA,
    class ElementX,
    class ElementY
>
struct TileReduce<Arch::AtlasA2,
                FTSelf::Gemv::helper::FT_THRESHOLD_ALGORITHM::ASVAR_ROBUST,
                FTSelf::Gemv::helper::FT_REDUCE_TYPE::MEAN_SQUARE,
                Catlass::Gemm::GemmType<ElementA, layout::RowMajor>,
                Catlass::Gemm::GemmType<ElementX, layout::VectorLayout>,
                Catlass::Gemm::GemmType<ElementY, layout::VectorLayout>,
                void>
{
    using ElementAccumulator =
        typename Catlass::Gemm::helper::ElementAccumulatorSelector<ElementA, ElementA>::ElementAccumulator;

    using FT_THRESHOLD_ALGORITHM = FTSelf::Gemv::helper::FT_THRESHOLD_ALGORITHM;

    static constexpr FT_THRESHOLD_ALGORITHM ALGO_TYPE = FTSelf::Gemv::helper::FT_THRESHOLD_ALGORITHM::ASVAR_ROBUST;
    using LayoutDst = layout::RowMajor;
    using LayoutSrc = layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementA);
    static constexpr uint32_t DST_ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementY);

    // Mehtods

    CATLASS_DEVICE
    TileReduce() {};

    //
    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementY> dstTensorMeanAbs,
        AscendC::LocalTensor<ElementY> dstTensorMeanSquare,
        AscendC::LocalTensor<ElementA> srcMeanTensor,
        AscendC::LocalTensor<ElementY> red_workspace,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc,
        ElementA n_ratio_factor)
    {
        uint32_t m_actual = layoutSrc.shape(0);
        uint32_t n_actual = layoutSrc.shape(1);
        uint32_t m_round = layoutDst.shape(0);
        uint32_t n_round = layoutDst.shape(1);

        // ElementY n_square_ratio_factor = n_ratio_factor * n_ratio_factor;

        uint32_t repeat_size = ELE_NUM_PER_C0 * 8;
        uint32_t dst_repeat_size = DST_ELE_NUM_PER_C0 * 8;

        uint32_t mask = repeat_size;
        uint32_t dst_mask = dst_repeat_size;

        uint32_t repeat_num = n_actual / repeat_size;
        uint32_t remain = n_actual % repeat_size;

        uint32_t dst_repeat_num = n_actual / dst_repeat_size;
        uint32_t dst_remain = n_actual % dst_repeat_size;

        uint64_t abs_mask = dst_repeat_size;
        uint64_t square_mask = dst_repeat_size;
        uint64_t add_mask = dst_repeat_size;

        AscendC::UnaryRepeatParams abs_params;
        abs_params.dstBlkStride = 1;
        abs_params.srcBlkStride = 1;
        abs_params.dstRepStride = RoundUp(n_round, dst_repeat_size) / ELE_NUM_PER_C0;
        abs_params.srcRepStride = RoundUp(n_round, dst_repeat_size) / ELE_NUM_PER_C0;

        AscendC::BinaryRepeatParams square_params;
        square_params.dstBlkStride = 1;
        square_params.src0BlkStride = 1;
        square_params.src1BlkStride = 1;

        square_params.dstRepStride = RoundUp(dst_repeat_size, dst_repeat_size) / DST_ELE_NUM_PER_C0;
        square_params.src0RepStride = RoundUp(n_round, dst_repeat_size) / ELE_NUM_PER_C0;
        square_params.src1RepStride = RoundUp(n_round, dst_repeat_size) / ELE_NUM_PER_C0;

        AscendC::BinaryRepeatParams add_params;
        add_params.dstBlkStride = 1;
        add_params.src0BlkStride = 1;
        add_params.src1BlkStride = 1;

        add_params.dstRepStride = RoundUp(n_round, dst_repeat_size) / ELE_NUM_PER_C0;
        add_params.src0RepStride = RoundUp(n_round, dst_repeat_size) / ELE_NUM_PER_C0;
        add_params.src1RepStride = RoundUp(n_round, dst_repeat_size) / ELE_NUM_PER_C0;

        AscendC::Duplicate<ElementY>(
                red_workspace,
                (ElementY)0.0,
                dst_repeat_size, // 每个迭代内部要对256 byte 的元素全部赋值为0，
                CeilDiv(m_actual * dst_repeat_size, dst_repeat_size), // 求行square和
                1, // 单次赋值迭代内，矢量目的操作数不同datablock间地址步长。
                8 // 相邻赋值迭代间，矢量目的操作数相同datablock地址步长。
        );

        AscendC::PipeBarrier<PIPE_V>();
        for(uint32_t i=0; i < dst_repeat_num; i++){
            uint32_t offset = i * dst_repeat_size;
            /*
            template <typename T, bool isSetMask = true>
            __aicore__ inline void Muls(const LocalTensor<T>& dstLocal,
                const LocalTensor<T>& srcLocal,
                const T& scalarValue,
                uint64_t mask,
                const uint8_t repeatTimes,
                const UnaryRepeatParams& repeatParams)
            */
            AscendC::Muls<ElementA, true>(
                srcMeanTensor[offset],
                srcMeanTensor[offset],
                n_ratio_factor,
                abs_mask,
                m_actual,
                abs_params);

            AscendC::PipeBarrier<PIPE_V>();
            /*
            template <typename T, bool isSetMask = true>
            __aicore__ inline void Abs(
                const LocalTensor<T>& dstLocal,
                const LocalTensor<T>& srcLocal,
                uint64_t mask,
                const uint8_t repeatTimes,
                const UnaryRepeatParams& repeatParams)
            */
            AscendC::Abs<ElementA, true>(
                srcMeanTensor[offset],
                srcMeanTensor[offset],
                abs_mask,
                m_actual,
                abs_params);

            AscendC::PipeBarrier<PIPE_V>();

            /*
            template <typename T, typename U, bool isSetMask = true>
            __aicore__ inline void MulAddDst(
                const LocalTensor<T>& dstLocal,
                const LocalTensor<U>& src0Local,
                const LocalTensor<U>& src1Local,
                uint64_t mask, const uint8_t repeatTimes,
                const BinaryRepeatParams& repeatParams)
            */
            AscendC::MulAddDst<ElementY, ElementA, true>(
                red_workspace,
                srcMeanTensor[offset],
                srcMeanTensor[offset],
                square_mask, m_actual,
                square_params);

            if(i > 0){
                AscendC::Add<ElementA, true>(
                    srcMeanTensor,
                    srcMeanTensor[offset],
                    srcMeanTensor,
                    add_mask,
                    m_actual,
                    add_params);
            }

            AscendC::PipeBarrier<PIPE_V>();
        }

        if (dst_remain > 0)
        {
            uint32_t offset = dst_repeat_num * dst_repeat_size;

            if (offset + dst_repeat_size > n_actual)
            {
                dst_remain = n_actual - offset;
            }
            // m_actual *
            uint64_t remain_mask = dst_remain;

            AscendC::Muls<ElementA, true>(
                srcMeanTensor[offset],
                srcMeanTensor[offset],
                n_ratio_factor,
                remain_mask,
                m_actual,
                abs_params);

            AscendC::PipeBarrier<PIPE_V>();

            AscendC::Abs<ElementA, true>(
                srcMeanTensor[offset],
                srcMeanTensor[offset],
                remain_mask,
                m_actual,
                abs_params);

            AscendC::PipeBarrier<PIPE_V>();

            AscendC::MulAddDst<ElementY, ElementA, true>(
                red_workspace,
                srcMeanTensor[offset],
                srcMeanTensor[offset],
                remain_mask,
                m_actual,
                square_params);

            AscendC::Add<ElementA, true>(
                srcMeanTensor,
                srcMeanTensor[offset],
                srcMeanTensor,
                remain_mask,
                m_actual,
                add_params);
        }

        uint64_t reduce_mask = (dst_repeat_num == 0) ? dst_remain : dst_repeat_size;
        AscendC::PipeBarrier<PIPE_V>();
        /*
        template <typename T, bool isSetMask = true>
        __aicore__ inline void WholeReduceSum(
            const LocalTensor<T>& dstLocal,
            const LocalTensor<T>& srcLocal,
            const int32_t mask,
            const int32_t repeatTimes,
            const int32_t dstRepStride,
            const int32_t srcBlkStride,
            const int32_t srcRepStride)
        */
        AscendC::WholeReduceSum<ElementA, true>(
            srcMeanTensor,
            srcMeanTensor,
            reduce_mask,
            m_actual,
            1,
            1,
            RoundUp(n_round, dst_repeat_size) / ELE_NUM_PER_C0
        );

        AscendC::WholeReduceSum<ElementY, true>(
            red_workspace,
            red_workspace,
            reduce_mask,
            m_actual,
            1,
            1,
            8);

        AscendC::PipeBarrier<PIPE_V>();

        uint32_t final_add_mask = (m_actual < dst_repeat_size) ? m_actual : dst_repeat_size;

        uint32_t dstOffset = m_round;

        AscendC::Cast<ElementY, ElementA>(
            dstTensorMeanAbs[dstOffset],
            srcMeanTensor,
            AscendC::RoundMode::CAST_NONE, m_actual);
        AscendC::PipeBarrier<PIPE_V>();

        add_params.dstRepStride = 8;
        add_params.src0RepStride = 8;
        add_params.src1RepStride = 8;

        square_params.dstRepStride = 8;
        square_params.src0RepStride = 8;
        square_params.src1RepStride = 8;

        AscendC::Add<ElementY, true>(
            dstTensorMeanAbs,
            dstTensorMeanAbs[dstOffset],
            dstTensorMeanAbs,
            final_add_mask,
            CeilDiv(m_round, dst_repeat_size),
            add_params);

        AscendC::Add<ElementY, true>(
            dstTensorMeanSquare,
            red_workspace,
            dstTensorMeanSquare,
            final_add_mask,
            CeilDiv(m_round, dst_repeat_size),
            square_params);

        AscendC::PipeBarrier<PIPE_V>();
    }
};


template <
>
struct TileReduce<Arch::AtlasA2,
                FTSelf::Gemv::helper::FT_THRESHOLD_ALGORITHM::ASVAR_ROBUST,
                FTSelf::Gemv::helper::FT_REDUCE_TYPE::MEAN_SQUARE,
                Catlass::Gemm::GemmType<float, layout::RowMajor>,
                Catlass::Gemm::GemmType<float, layout::VectorLayout>,
                Catlass::Gemm::GemmType<float, layout::VectorLayout>,
                void>
{
    using ElementA = float;
    using ElementX = float;
    using ElementY = float;
    using ElementAccumulator =
        typename Catlass::Gemm::helper::ElementAccumulatorSelector<ElementA, ElementA>::ElementAccumulator;

    using FT_THRESHOLD_ALGORITHM = FTSelf::Gemv::helper::FT_THRESHOLD_ALGORITHM;

    static constexpr FT_THRESHOLD_ALGORITHM ALGO_TYPE = FTSelf::Gemv::helper::FT_THRESHOLD_ALGORITHM::ASVAR_ROBUST;
    using LayoutDst = layout::RowMajor;
    using LayoutSrc = layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementA);
    static constexpr uint32_t DST_ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementY);

    // Mehtods

    CATLASS_DEVICE
    TileReduce() {};

    //
    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementY> dstTensorMeanAbs,
        AscendC::LocalTensor<ElementY> dstTensorMeanSquare,
        AscendC::LocalTensor<ElementA> srcMeanTensor,
        AscendC::LocalTensor<ElementY> red_workspace,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc,
        ElementA n_ratio_factor)
    {
        uint32_t m_actual = layoutSrc.shape(0);
        uint32_t n_actual = layoutSrc.shape(1);
        uint32_t m_round = layoutDst.shape(0);
        uint32_t n_round = layoutDst.shape(1);

        // ElementY n_square_ratio_factor = n_ratio_factor * n_ratio_factor;

        uint32_t repeat_size = ELE_NUM_PER_C0 * 8;
        uint32_t dst_repeat_size = DST_ELE_NUM_PER_C0 * 8;

        uint32_t mask = repeat_size;
        uint32_t dst_mask = dst_repeat_size;

        uint32_t repeat_num = n_actual / repeat_size;
        uint32_t remain = n_actual % repeat_size;

        uint32_t dst_repeat_num = n_actual / dst_repeat_size;
        uint32_t dst_remain = n_actual % dst_repeat_size;

        uint64_t abs_mask = dst_repeat_size;
        uint64_t square_mask = dst_repeat_size;
        uint64_t add_mask = dst_repeat_size;

        AscendC::UnaryRepeatParams abs_params;
        abs_params.dstBlkStride = 1;
        abs_params.srcBlkStride = 1;
        abs_params.dstRepStride = RoundUp(n_round, dst_repeat_size) / ELE_NUM_PER_C0;
        abs_params.srcRepStride = RoundUp(n_round, dst_repeat_size) / ELE_NUM_PER_C0;

        AscendC::BinaryRepeatParams square_params;
        square_params.dstBlkStride = 1;
        square_params.src0BlkStride = 1;
        square_params.src1BlkStride = 1;

        square_params.dstRepStride = RoundUp(dst_repeat_size, dst_repeat_size) / DST_ELE_NUM_PER_C0;
        square_params.src0RepStride = RoundUp(n_round, dst_repeat_size) / ELE_NUM_PER_C0;
        square_params.src1RepStride = RoundUp(n_round, dst_repeat_size) / ELE_NUM_PER_C0;

        AscendC::BinaryRepeatParams add_params;
        add_params.dstBlkStride = 1;
        add_params.src0BlkStride = 1;
        add_params.src1BlkStride = 1;

        add_params.dstRepStride = RoundUp(n_round, dst_repeat_size) / ELE_NUM_PER_C0;
        add_params.src0RepStride = RoundUp(n_round, dst_repeat_size) / ELE_NUM_PER_C0;
        add_params.src1RepStride = RoundUp(n_round, dst_repeat_size) / ELE_NUM_PER_C0;

        AscendC::Duplicate<ElementY>(
                red_workspace,
                (ElementY)0.0,
                dst_repeat_size, // 每个迭代内部要对256 byte 的元素全部赋值为0，
                CeilDiv(m_actual * dst_repeat_size, dst_repeat_size), // 求行square和
                1, // 单次赋值迭代内，矢量目的操作数不同datablock间地址步长。
                8 // 相邻赋值迭代间，矢量目的操作数相同datablock地址步长。
        );

        AscendC::PipeBarrier<PIPE_V>();
        for(uint32_t i=0; i < dst_repeat_num; i++){
            uint32_t offset = i * dst_repeat_size;
            /*
            template <typename T, bool isSetMask = true>
            __aicore__ inline void Muls(const LocalTensor<T>& dstLocal,
                const LocalTensor<T>& srcLocal,
                const T& scalarValue,
                uint64_t mask,
                const uint8_t repeatTimes,
                const UnaryRepeatParams& repeatParams)
            */
            AscendC::Muls<ElementA, true>(
                srcMeanTensor[offset],
                srcMeanTensor[offset],
                n_ratio_factor,
                abs_mask,
                m_actual,
                abs_params);

            AscendC::PipeBarrier<PIPE_V>();
            /*
            template <typename T, bool isSetMask = true>
            __aicore__ inline void Abs(
                const LocalTensor<T>& dstLocal,
                const LocalTensor<T>& srcLocal,
                uint64_t mask,
                const uint8_t repeatTimes,
                const UnaryRepeatParams& repeatParams)
            */
            AscendC::Abs<ElementA, true>(
                srcMeanTensor[offset],
                srcMeanTensor[offset],
                abs_mask,
                m_actual,
                abs_params);

            AscendC::PipeBarrier<PIPE_V>();

            /*
            template <typename T, typename U, bool isSetMask = true>
            __aicore__ inline void MulAddDst(
                const LocalTensor<T>& dstLocal,
                const LocalTensor<U>& src0Local,
                const LocalTensor<U>& src1Local,
                uint64_t mask, const uint8_t repeatTimes,
                const BinaryRepeatParams& repeatParams)
            */
            AscendC::MulAddDst<ElementY, ElementA, true>(
                red_workspace,
                srcMeanTensor[offset],
                srcMeanTensor[offset],
                square_mask, m_actual,
                square_params);

            if(i > 0){
                AscendC::Add<ElementA, true>(
                    srcMeanTensor,
                    srcMeanTensor[offset],
                    srcMeanTensor,
                    add_mask,
                    m_actual,
                    add_params);
            }

            AscendC::PipeBarrier<PIPE_V>();
        }

        if (dst_remain > 0)
        {
            uint32_t offset = dst_repeat_num * dst_repeat_size;

            if (offset + dst_repeat_size > n_actual)
            {
                dst_remain = n_actual - offset;
            }
            // m_actual *
            uint64_t remain_mask = dst_remain;

            AscendC::Muls<ElementA, true>(
                srcMeanTensor[offset],
                srcMeanTensor[offset],
                n_ratio_factor,
                remain_mask,
                m_actual,
                abs_params);

            AscendC::PipeBarrier<PIPE_V>();

            AscendC::Abs<ElementA, true>(
                srcMeanTensor[offset],
                srcMeanTensor[offset],
                remain_mask,
                m_actual,
                abs_params);

            AscendC::PipeBarrier<PIPE_V>();

            AscendC::MulAddDst<ElementY, ElementA, true>(
                red_workspace,
                srcMeanTensor[offset],
                srcMeanTensor[offset],
                remain_mask,
                m_actual,
                square_params);

            AscendC::Add<ElementA, true>(
                srcMeanTensor,
                srcMeanTensor[offset],
                srcMeanTensor,
                remain_mask,
                m_actual,
                add_params);
        }

        uint64_t reduce_mask = (dst_repeat_num == 0) ? dst_remain : dst_repeat_size;
        AscendC::PipeBarrier<PIPE_V>();
        /*
        template <typename T, bool isSetMask = true>
        __aicore__ inline void WholeReduceSum(
            const LocalTensor<T>& dstLocal,
            const LocalTensor<T>& srcLocal,
            const int32_t mask,
            const int32_t repeatTimes,
            const int32_t dstRepStride,
            const int32_t srcBlkStride,
            const int32_t srcRepStride)
        */
        AscendC::WholeReduceSum<ElementA, true>(
            srcMeanTensor,
            srcMeanTensor,
            reduce_mask,
            m_actual,
            1,
            1,
            RoundUp(n_round, dst_repeat_size) / ELE_NUM_PER_C0
        );

        AscendC::WholeReduceSum<ElementY, true>(
            red_workspace,
            red_workspace,
            reduce_mask,
            m_actual,
            1,
            1,
            8);

        AscendC::PipeBarrier<PIPE_V>();

        uint32_t final_add_mask = (m_actual < dst_repeat_size) ? m_actual : dst_repeat_size;

        uint32_t dstOffset = m_round;

        add_params.dstRepStride = 8;
        add_params.src0RepStride = 8;
        add_params.src1RepStride = 8;

        square_params.dstRepStride = 8;
        square_params.src0RepStride = 8;
        square_params.src1RepStride = 8;

        AscendC::Add<ElementY, true>(
            dstTensorMeanAbs,
            srcMeanTensor,
            dstTensorMeanAbs,
            final_add_mask,
            CeilDiv(m_round, dst_repeat_size),
            add_params);

        AscendC::Add<ElementY, true>(
            dstTensorMeanSquare,
            red_workspace,
            dstTensorMeanSquare,
            final_add_mask,
            CeilDiv(m_round, dst_repeat_size),
            square_params);

        AscendC::PipeBarrier<PIPE_V>();
    }
};



template <
>
struct TileReduce<Arch::AtlasA2,
                FTSelf::Gemv::helper::FT_THRESHOLD_ALGORITHM::ASVAR_ROBUST,
                FTSelf::Gemv::helper::FT_REDUCE_TYPE::MEAN_SQUARE,
                Catlass::Gemm::GemmType<half, layout::RowMajor>,
                Catlass::Gemm::GemmType<half, layout::VectorLayout>,
                Catlass::Gemm::GemmType<half, layout::VectorLayout>,
                void>
{
    using ElementA = half;
    using ElementX = half;
    using ElementY = half;
    using ElementAccumulator =
        typename Catlass::Gemm::helper::ElementAccumulatorSelector<ElementA, ElementA>::ElementAccumulator;

    using FT_THRESHOLD_ALGORITHM = FTSelf::Gemv::helper::FT_THRESHOLD_ALGORITHM;

    static constexpr FT_THRESHOLD_ALGORITHM ALGO_TYPE = FTSelf::Gemv::helper::FT_THRESHOLD_ALGORITHM::ASVAR_ROBUST;
    using LayoutDst = layout::RowMajor;
    using LayoutSrc = layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementA);
    static constexpr uint32_t DST_ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementY);

    // Mehtods

    CATLASS_DEVICE
    TileReduce() {};

    //
    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementY> dstTensorMeanAbs,
        AscendC::LocalTensor<ElementY> dstTensorMeanSquare,
        AscendC::LocalTensor<ElementA> srcMeanTensor,
        AscendC::LocalTensor<ElementY> red_workspace,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc,
        ElementA n_ratio_factor)
    {
        uint32_t m_actual = layoutSrc.shape(0);
        uint32_t n_actual = layoutSrc.shape(1);
        uint32_t m_round = layoutDst.shape(0);
        uint32_t n_round = layoutDst.shape(1);

        // ElementY n_square_ratio_factor = n_ratio_factor * n_ratio_factor;

        uint32_t repeat_size = ELE_NUM_PER_C0 * 8;
        uint32_t dst_repeat_size = DST_ELE_NUM_PER_C0 * 8;

        uint32_t mask = repeat_size;
        uint32_t dst_mask = dst_repeat_size;

        uint32_t repeat_num = n_actual / repeat_size;
        uint32_t remain = n_actual % repeat_size;

        uint32_t dst_repeat_num = n_actual / dst_repeat_size;
        uint32_t dst_remain = n_actual % dst_repeat_size;

        uint64_t abs_mask = dst_repeat_size;
        uint64_t square_mask = dst_repeat_size;
        uint64_t add_mask = dst_repeat_size;

        AscendC::UnaryRepeatParams abs_params;
        abs_params.dstBlkStride = 1;
        abs_params.srcBlkStride = 1;
        abs_params.dstRepStride = RoundUp(n_round, dst_repeat_size) / ELE_NUM_PER_C0;
        abs_params.srcRepStride = RoundUp(n_round, dst_repeat_size) / ELE_NUM_PER_C0;

        AscendC::BinaryRepeatParams square_params;
        square_params.dstBlkStride = 1;
        square_params.src0BlkStride = 1;
        square_params.src1BlkStride = 1;

        square_params.dstRepStride = RoundUp(dst_repeat_size, dst_repeat_size) / DST_ELE_NUM_PER_C0;
        square_params.src0RepStride = RoundUp(n_round, dst_repeat_size) / ELE_NUM_PER_C0;
        square_params.src1RepStride = RoundUp(n_round, dst_repeat_size) / ELE_NUM_PER_C0;

        AscendC::BinaryRepeatParams add_params;
        add_params.dstBlkStride = 1;
        add_params.src0BlkStride = 1;
        add_params.src1BlkStride = 1;

        add_params.dstRepStride = RoundUp(n_round, dst_repeat_size) / ELE_NUM_PER_C0;
        add_params.src0RepStride = RoundUp(n_round, dst_repeat_size) / ELE_NUM_PER_C0;
        add_params.src1RepStride = RoundUp(n_round, dst_repeat_size) / ELE_NUM_PER_C0;

        AscendC::Duplicate<ElementY>(
                red_workspace,
                (ElementY)0.0,
                dst_repeat_size, // 每个迭代内部要对256 byte 的元素全部赋值为0，
                CeilDiv(m_actual * dst_repeat_size, dst_repeat_size), // 求行square和
                1, // 单次赋值迭代内，矢量目的操作数不同datablock间地址步长。
                8 // 相邻赋值迭代间，矢量目的操作数相同datablock地址步长。
        );

        AscendC::PipeBarrier<PIPE_V>();
        for(uint32_t i=0; i < dst_repeat_num; i++){
            uint32_t offset = i * dst_repeat_size;
            /*
            template <typename T, bool isSetMask = true>
            __aicore__ inline void Muls(const LocalTensor<T>& dstLocal,
                const LocalTensor<T>& srcLocal,
                const T& scalarValue,
                uint64_t mask,
                const uint8_t repeatTimes,
                const UnaryRepeatParams& repeatParams)
            */
            AscendC::Muls<ElementA, true>(
                srcMeanTensor[offset],
                srcMeanTensor[offset],
                n_ratio_factor,
                abs_mask,
                m_actual,
                abs_params);

            AscendC::PipeBarrier<PIPE_V>();
            /*
            template <typename T, bool isSetMask = true>
            __aicore__ inline void Abs(
                const LocalTensor<T>& dstLocal,
                const LocalTensor<T>& srcLocal,
                uint64_t mask,
                const uint8_t repeatTimes,
                const UnaryRepeatParams& repeatParams)
            */
            AscendC::Abs<ElementA, true>(
                srcMeanTensor[offset],
                srcMeanTensor[offset],
                abs_mask,
                m_actual,
                abs_params);

            AscendC::PipeBarrier<PIPE_V>();

            /*
            template <typename T, typename U, bool isSetMask = true>
            __aicore__ inline void MulAddDst(
                const LocalTensor<T>& dstLocal,
                const LocalTensor<U>& src0Local,
                const LocalTensor<U>& src1Local,
                uint64_t mask, const uint8_t repeatTimes,
                const BinaryRepeatParams& repeatParams)
            */
            AscendC::MulAddDst<ElementY, ElementA, true>(
                red_workspace,
                srcMeanTensor[offset],
                srcMeanTensor[offset],
                square_mask, m_actual,
                square_params);

            if(i > 0){
                AscendC::Add<ElementA, true>(
                    srcMeanTensor,
                    srcMeanTensor[offset],
                    srcMeanTensor,
                    add_mask,
                    m_actual,
                    add_params);
            }

            AscendC::PipeBarrier<PIPE_V>();
        }

        if (dst_remain > 0)
        {
            uint32_t offset = dst_repeat_num * dst_repeat_size;

            if (offset + dst_repeat_size > n_actual)
            {
                dst_remain = n_actual - offset;
            }
            // m_actual *
            uint64_t remain_mask = dst_remain;

            AscendC::Muls<ElementA, true>(
                srcMeanTensor[offset],
                srcMeanTensor[offset],
                n_ratio_factor,
                remain_mask,
                m_actual,
                abs_params);

            AscendC::PipeBarrier<PIPE_V>();

            AscendC::Abs<ElementA, true>(
                srcMeanTensor[offset],
                srcMeanTensor[offset],
                remain_mask,
                m_actual,
                abs_params);

            AscendC::PipeBarrier<PIPE_V>();

            AscendC::MulAddDst<ElementY, ElementA, true>(
                red_workspace,
                srcMeanTensor[offset],
                srcMeanTensor[offset],
                remain_mask,
                m_actual,
                square_params);

            AscendC::Add<ElementA, true>(
                srcMeanTensor,
                srcMeanTensor[offset],
                srcMeanTensor,
                remain_mask,
                m_actual,
                add_params);

            // AscendC::PipeBarrier<PIPE_V>();
        }

        uint64_t reduce_mask = (dst_repeat_num == 0) ? dst_remain : dst_repeat_size;
        AscendC::PipeBarrier<PIPE_V>();
        /*
        template <typename T, bool isSetMask = true>
        __aicore__ inline void WholeReduceSum(
            const LocalTensor<T>& dstLocal,
            const LocalTensor<T>& srcLocal,
            const int32_t mask,
            const int32_t repeatTimes,
            const int32_t dstRepStride,
            const int32_t srcBlkStride,
            const int32_t srcRepStride)
        */
        AscendC::WholeReduceSum<ElementA, true>(
            srcMeanTensor,
            srcMeanTensor,
            reduce_mask,
            m_actual,
            1,
            1,
            RoundUp(n_round, dst_repeat_size) / ELE_NUM_PER_C0
        );

        AscendC::WholeReduceSum<ElementY, true>(
            red_workspace,
            red_workspace,
            reduce_mask,
            m_actual,
            1,
            1,
            8);

        AscendC::PipeBarrier<PIPE_V>();

        uint32_t final_add_mask = (m_actual < dst_repeat_size) ? m_actual : dst_repeat_size;

        uint32_t dstOffset = m_round;

        add_params.dstRepStride = 8;
        add_params.src0RepStride = 8;
        add_params.src1RepStride = 8;

        square_params.dstRepStride = 8;
        square_params.src0RepStride = 8;
        square_params.src1RepStride = 8;

        AscendC::Add<ElementY, true>(
            dstTensorMeanAbs,
            srcMeanTensor,
            dstTensorMeanAbs,
            final_add_mask,
            CeilDiv(m_round, dst_repeat_size),
            add_params);

        AscendC::Add<ElementY, true>(
            dstTensorMeanSquare,
            red_workspace,
            dstTensorMeanSquare,
            final_add_mask,
            CeilDiv(m_round, dst_repeat_size),
            square_params);

        AscendC::PipeBarrier<PIPE_V>();
    }
};

template <
>
struct TileReduce<Arch::AtlasA2,
                FTSelf::Gemv::helper::FT_THRESHOLD_ALGORITHM::ASVAR_ROBUST,
                FTSelf::Gemv::helper::FT_REDUCE_TYPE::MEAN_SQUARE,
                Catlass::Gemm::GemmType<half, layout::RowMajor>,
                Catlass::Gemm::GemmType<float, layout::VectorLayout>,
                Catlass::Gemm::GemmType<float, layout::VectorLayout>,
                void>
{
    using ElementA = half;
    using ElementX = float;
    using ElementY = float;

    using ElementAccumulator =
        typename Catlass::Gemm::helper::ElementAccumulatorSelector<ElementA, ElementA>::ElementAccumulator;

    using FT_THRESHOLD_ALGORITHM = FTSelf::Gemv::helper::FT_THRESHOLD_ALGORITHM;

    static constexpr FT_THRESHOLD_ALGORITHM ALGO_TYPE = FTSelf::Gemv::helper::FT_THRESHOLD_ALGORITHM::ASVAR_ROBUST;
    using LayoutDst = layout::RowMajor;
    using LayoutSrc = layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementA);
    static constexpr uint32_t DST_ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementY);

    // Mehtods

    CATLASS_DEVICE
    TileReduce() {};

    //
    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementY> dstTensorMeanAbs,
        AscendC::LocalTensor<ElementY> dstTensorMeanSquare,
        AscendC::LocalTensor<ElementA> srcMeanTensor,
        AscendC::LocalTensor<ElementY> red_workspace,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc,
        ElementA n_ratio_factor)
    {
        uint32_t m_actual = layoutSrc.shape(0);
        uint32_t n_actual = layoutSrc.shape(1);
        uint32_t m_round = layoutDst.shape(0);
        uint32_t n_round = layoutDst.shape(1);

        // ElementY n_square_ratio_factor = n_ratio_factor * n_ratio_factor;

        uint32_t repeat_size = ELE_NUM_PER_C0 * 8;
        uint32_t dst_repeat_size = DST_ELE_NUM_PER_C0 * 8;

        uint32_t mask = repeat_size;
        uint32_t dst_mask = dst_repeat_size;

        uint32_t repeat_num = n_actual / repeat_size;
        uint32_t remain = n_actual % repeat_size;

        uint32_t dst_repeat_num = n_actual / dst_repeat_size;
        uint32_t dst_remain = n_actual % dst_repeat_size;

        uint64_t abs_mask = dst_repeat_size;
        uint64_t square_mask = dst_repeat_size;
        uint64_t add_mask = dst_repeat_size;

        AscendC::UnaryRepeatParams abs_params;
        abs_params.dstBlkStride = 1;
        abs_params.srcBlkStride = 1;
        abs_params.dstRepStride = RoundUp(n_round, dst_repeat_size) / ELE_NUM_PER_C0;
        abs_params.srcRepStride = RoundUp(n_round, dst_repeat_size) / ELE_NUM_PER_C0;

        AscendC::BinaryRepeatParams square_params;
        square_params.dstBlkStride = 1;
        square_params.src0BlkStride = 1;
        square_params.src1BlkStride = 1;

        square_params.dstRepStride = RoundUp(dst_repeat_size, dst_repeat_size) / DST_ELE_NUM_PER_C0;
        square_params.src0RepStride = RoundUp(n_round, dst_repeat_size) / ELE_NUM_PER_C0;
        square_params.src1RepStride = RoundUp(n_round, dst_repeat_size) / ELE_NUM_PER_C0;

        AscendC::BinaryRepeatParams add_params;
        add_params.dstBlkStride = 1;
        add_params.src0BlkStride = 1;
        add_params.src1BlkStride = 1;

        add_params.dstRepStride = RoundUp(n_round, dst_repeat_size) / ELE_NUM_PER_C0;
        add_params.src0RepStride = RoundUp(n_round, dst_repeat_size) / ELE_NUM_PER_C0;
        add_params.src1RepStride = RoundUp(n_round, dst_repeat_size) / ELE_NUM_PER_C0;

        AscendC::Duplicate<ElementY>(
                red_workspace,
                (ElementY)0.0,
                dst_repeat_size, // 每个迭代内部要对256 byte 的元素全部赋值为0，
                CeilDiv(m_actual * dst_repeat_size, dst_repeat_size), // 求行square和
                1, // 单次赋值迭代内，矢量目的操作数不同datablock间地址步长。
                8 // 相邻赋值迭代间，矢量目的操作数相同datablock地址步长。
        );

        AscendC::PipeBarrier<PIPE_V>();
        for(uint32_t i=0; i < dst_repeat_num; i++){
            uint32_t offset = i * dst_repeat_size;
            /*
            template <typename T, bool isSetMask = true>
            __aicore__ inline void Muls(const LocalTensor<T>& dstLocal,
                const LocalTensor<T>& srcLocal,
                const T& scalarValue,
                uint64_t mask,
                const uint8_t repeatTimes,
                const UnaryRepeatParams& repeatParams)
            */
            AscendC::Muls<ElementA, true>(
                srcMeanTensor[offset],
                srcMeanTensor[offset],
                n_ratio_factor,
                abs_mask,
                m_actual,
                abs_params);

            AscendC::PipeBarrier<PIPE_V>();
            /*
            template <typename T, bool isSetMask = true>
            __aicore__ inline void Abs(
                const LocalTensor<T>& dstLocal,
                const LocalTensor<T>& srcLocal,
                uint64_t mask,
                const uint8_t repeatTimes,
                const UnaryRepeatParams& repeatParams)
            */
            AscendC::Abs<ElementA, true>(
                srcMeanTensor[offset],
                srcMeanTensor[offset],
                abs_mask,
                m_actual,
                abs_params);

            AscendC::PipeBarrier<PIPE_V>();

            /*
            template <typename T, typename U, bool isSetMask = true>
            __aicore__ inline void MulAddDst(
                const LocalTensor<T>& dstLocal,
                const LocalTensor<U>& src0Local,
                const LocalTensor<U>& src1Local,
                uint64_t mask, const uint8_t repeatTimes,
                const BinaryRepeatParams& repeatParams)
            */
            AscendC::MulAddDst<ElementY, ElementA, true>(
                red_workspace,
                srcMeanTensor[offset],
                srcMeanTensor[offset],
                square_mask, m_actual,
                square_params);

            if(i > 0){
                AscendC::Add<ElementA, true>(
                    srcMeanTensor,
                    srcMeanTensor[offset],
                    srcMeanTensor,
                    add_mask,
                    m_actual,
                    add_params);
            }

            AscendC::PipeBarrier<PIPE_V>();
        }

        if (dst_remain > 0)
        {
            uint32_t offset = dst_repeat_num * dst_repeat_size;

            if (offset + dst_repeat_size > n_actual)
            {
                dst_remain = n_actual - offset;
            }
            // m_actual *
            uint64_t remain_mask = dst_remain;

            AscendC::Muls<ElementA, true>(
                srcMeanTensor[offset],
                srcMeanTensor[offset],
                n_ratio_factor,
                remain_mask,
                m_actual,
                abs_params);

            AscendC::PipeBarrier<PIPE_V>();

            AscendC::Abs<ElementA, true>(
                srcMeanTensor[offset],
                srcMeanTensor[offset],
                remain_mask,
                m_actual,
                abs_params);

            AscendC::PipeBarrier<PIPE_V>();

            AscendC::MulAddDst<ElementY, ElementA, true>(
                red_workspace,
                srcMeanTensor[offset],
                srcMeanTensor[offset],
                remain_mask,
                m_actual,
                square_params);

            AscendC::Add<ElementA, true>(
                srcMeanTensor,
                srcMeanTensor[offset],
                srcMeanTensor,
                remain_mask,
                m_actual,
                add_params);
        }

        uint64_t reduce_mask = (dst_repeat_num == 0) ? dst_remain : dst_repeat_size;
        AscendC::PipeBarrier<PIPE_V>();
        /*
        template <typename T, bool isSetMask = true>
        __aicore__ inline void WholeReduceSum(
            const LocalTensor<T>& dstLocal,
            const LocalTensor<T>& srcLocal,
            const int32_t mask,
            const int32_t repeatTimes,
            const int32_t dstRepStride,
            const int32_t srcBlkStride,
            const int32_t srcRepStride)
        */
        AscendC::WholeReduceSum<ElementA, true>(
            srcMeanTensor,
            srcMeanTensor,
            reduce_mask,
            m_actual,
            1,
            1,
            RoundUp(n_round, dst_repeat_size) / ELE_NUM_PER_C0
        );

        AscendC::WholeReduceSum<ElementY, true>(
            red_workspace,
            red_workspace,
            reduce_mask,
            m_actual,
            1,
            1,
            8);

        AscendC::PipeBarrier<PIPE_V>();

        uint32_t final_add_mask = (m_actual < dst_repeat_size) ? m_actual : dst_repeat_size;

        uint32_t dstOffset = m_round;

        AscendC::Cast<ElementY, ElementA>(
            dstTensorMeanAbs[dstOffset],
            srcMeanTensor,
            AscendC::RoundMode::CAST_NONE, m_actual);
        AscendC::PipeBarrier<PIPE_V>();

        add_params.dstRepStride = 8;
        add_params.src0RepStride = 8;
        add_params.src1RepStride = 8;

        square_params.dstRepStride = 8;
        square_params.src0RepStride = 8;
        square_params.src1RepStride = 8;

        AscendC::Add<ElementY, true>(
            dstTensorMeanAbs,
            dstTensorMeanAbs[dstOffset],
            dstTensorMeanAbs,
            final_add_mask,
            CeilDiv(m_round, dst_repeat_size),
            add_params);

        AscendC::Add<ElementY, true>(
            dstTensorMeanSquare,
            red_workspace,
            dstTensorMeanSquare,
            final_add_mask,
            CeilDiv(m_round, dst_repeat_size),
            square_params);

        AscendC::PipeBarrier<PIPE_V>();
    }
};

template <
    class ElementA,
    class ElementX,
    class ElementY
>
struct TileReduce<Arch::AtlasA2,
                FTSelf::Gemv::helper::FT_THRESHOLD_ALGORITHM::ASVAR_ROBUST,
                FTSelf::Gemv::helper::FT_REDUCE_TYPE::VAR,
                Catlass::Gemm::GemmType<ElementA, layout::RowMajor>,
                Catlass::Gemm::GemmType<ElementX, layout::VectorLayout>,
                Catlass::Gemm::GemmType<ElementY, layout::VectorLayout>,
                void>
{
    using ElementAccumulator =
        typename Catlass::Gemm::helper::ElementAccumulatorSelector<ElementA, ElementA>::ElementAccumulator;

    using FT_THRESHOLD_ALGORITHM = FTSelf::Gemv::helper::FT_THRESHOLD_ALGORITHM;

    static constexpr FT_THRESHOLD_ALGORITHM ALGO_TYPE = FTSelf::Gemv::helper::FT_THRESHOLD_ALGORITHM::ASVAR_ROBUST;
    using LayoutDst = layout::RowMajor;
    using LayoutSrc = layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementA);
    static constexpr uint32_t DST_ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementY);

    // Mehtods

    CATLASS_DEVICE
    TileReduce() {};

    //
    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementY> dstTensorVar,
        AscendC::LocalTensor<ElementA> srcMeanTensor,
        AscendC::LocalTensor<ElementA> srcMaxTensor,
        AscendC::LocalTensor<ElementA> srcMinTensor,
        AscendC::LocalTensor<ElementY> red_workspace,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc,
        ElementA n_ratio_factor)
    {
        uint32_t m_actual = layoutSrc.shape(0);
        uint32_t n_actual = layoutSrc.shape(1);
        uint32_t m_round = layoutDst.shape(0);
        uint32_t n_round = layoutDst.shape(1);

        // ElementY n_square_ratio_factor = n_ratio_factor * n_ratio_factor;

        uint32_t repeat_size = ELE_NUM_PER_C0 * 8;
        uint32_t dst_repeat_size = DST_ELE_NUM_PER_C0 * 8;

        uint32_t mask = repeat_size;
        uint32_t dst_mask = dst_repeat_size;

        uint32_t repeat_num = n_actual / repeat_size;
        uint32_t remain = n_actual % repeat_size;

        uint32_t dst_repeat_num = n_actual / dst_repeat_size;
        uint32_t dst_remain = n_actual % dst_repeat_size;

        uint64_t mean_mask = dst_repeat_size;
        uint64_t var_mask = dst_repeat_size;
        uint64_t sub_mask = dst_repeat_size;
        uint64_t abs_mask = dst_repeat_size;

        AscendC::UnaryRepeatParams mean_params;
        mean_params.dstBlkStride = 1;
        mean_params.srcBlkStride = 1;
        mean_params.dstRepStride = RoundUp(n_round, dst_repeat_size) / ELE_NUM_PER_C0;
        mean_params.srcRepStride = RoundUp(n_round, dst_repeat_size) / ELE_NUM_PER_C0;

        AscendC::BinaryRepeatParams var_params;
        var_params.dstBlkStride = 1;
        var_params.src0BlkStride = 1;
        var_params.src1BlkStride = 1;

        var_params.dstRepStride = RoundUp(dst_repeat_size, dst_repeat_size) / DST_ELE_NUM_PER_C0;
        var_params.src0RepStride = RoundUp(n_round, dst_repeat_size) / ELE_NUM_PER_C0;
        var_params.src1RepStride = RoundUp(n_round, dst_repeat_size) / ELE_NUM_PER_C0;

        AscendC::BinaryRepeatParams sub_params;
        sub_params.dstBlkStride = 1;
        sub_params.src0BlkStride = 1;
        sub_params.src1BlkStride = 1;

        sub_params.dstRepStride = RoundUp(n_round, dst_repeat_size) / ELE_NUM_PER_C0;
        sub_params.src0RepStride = RoundUp(n_round, dst_repeat_size) / ELE_NUM_PER_C0;
        sub_params.src1RepStride = RoundUp(n_round, dst_repeat_size) / ELE_NUM_PER_C0;

        AscendC::UnaryRepeatParams abs_params;
        abs_params.dstBlkStride = 1;
        abs_params.srcBlkStride = 1;
        abs_params.dstRepStride = RoundUp(n_round, dst_repeat_size) / ELE_NUM_PER_C0;
        abs_params.srcRepStride = RoundUp(n_round, dst_repeat_size) / ELE_NUM_PER_C0;

        AscendC::Duplicate<ElementY>(
                red_workspace,
                (ElementY)0.0,
                dst_repeat_size, // 每个迭代内部要对256 byte 的元素全部赋值为0，
                CeilDiv(m_actual * dst_repeat_size, dst_repeat_size), // 求行square和
                1, // 单次赋值迭代内，矢量目的操作数不同datablock间地址步长。
                8 // 相邻赋值迭代间，矢量目的操作数相同datablock地址步长。
        );

        AscendC::PipeBarrier<PIPE_V>();
        for(uint32_t i=0; i < dst_repeat_num; i++){
            uint32_t offset = i * dst_repeat_size;
            /*
            template <typename T, bool isSetMask = true>
            __aicore__ inline void Muls(const LocalTensor<T>& dstLocal,
                const LocalTensor<T>& srcLocal,
                const T& scalarValue,
                uint64_t mask,
                const uint8_t repeatTimes,
                const UnaryRepeatParams& repeatParams)
            */
            AscendC::Muls<ElementA, true>(
                srcMeanTensor[offset],
                srcMeanTensor[offset],
                n_ratio_factor,
                mean_mask,
                m_actual,
                mean_params);

            AscendC::PipeBarrier<PIPE_V>();
            /*
            template <typename T, bool isSetMask = true>
            __aicore__ inline void Sub(const LocalTensor<T>& dstLocal,
                const LocalTensor<T>& src0Local,
                const LocalTensor<T>& src1Local,
                uint64_t mask, const uint8_t repeatTimes,
                const BinaryRepeatParams& repeatParams)
            */
            AscendC::Sub<ElementA, true>(
                srcMaxTensor[offset],
                srcMaxTensor[offset],
                srcMeanTensor[offset],
                sub_mask,
                m_actual,
                sub_params);

            AscendC::Sub<ElementA, true>(
                srcMinTensor[offset],
                srcMeanTensor[offset],
                srcMinTensor[offset],
                sub_mask,
                m_actual,
                sub_params);

            AscendC::PipeBarrier<PIPE_V>();

            // AscendC::Abs<ElementA,true>(
            //     srcMaxTensor[offset],
            //     srcMaxTensor[offset],
            //     abs_mask,
            //     m_actual,
            //     abs_params);

            // AscendC::Abs<ElementA,true>(
            //     srcMinTensor[offset],
            //     srcMinTensor[offset],
            //     abs_mask,
            //     m_actual,
            //     abs_params);

            // AscendC::PipeBarrier<PIPE_V>();
            /*
            template <typename T, typename U, bool isSetMask = true>
            __aicore__ inline void MulAddDst(
                const LocalTensor<T>& dstLocal,
                const LocalTensor<U>& src0Local,
                const LocalTensor<U>& src1Local,
                uint64_t mask, const uint8_t repeatTimes,
                const BinaryRepeatParams& repeatParams)
            */

            AscendC::MulAddDst<ElementY, ElementA, true>(
                red_workspace,
                srcMaxTensor[offset],
                srcMinTensor[offset],
                var_mask, m_actual,
                var_params);

            AscendC::PipeBarrier<PIPE_V>();
        }

        if (dst_remain > 0)
        {
            uint32_t offset = dst_repeat_num * dst_repeat_size;

            if (offset + dst_repeat_size > n_actual)
            {
                dst_remain = n_actual - offset;
            }
            // m_actual *
            uint64_t remain_mask = dst_remain;

            AscendC::Muls<ElementA, true>(
                srcMeanTensor[offset],
                srcMeanTensor[offset],
                n_ratio_factor,
                dst_remain,
                m_actual,
                mean_params);

            AscendC::PipeBarrier<PIPE_V>();

            AscendC::Sub<ElementA, true>(
                srcMaxTensor[offset],
                srcMaxTensor[offset],
                srcMeanTensor[offset],
                remain_mask,
                m_actual,
                sub_params);

            AscendC::Sub<ElementA, true>(
                srcMinTensor[offset],
                srcMeanTensor[offset],
                srcMinTensor[offset],
                remain_mask,
                m_actual,
                sub_params);

            AscendC::PipeBarrier<PIPE_V>();

            // AscendC::Abs<ElementA,true>(
            //     srcMaxTensor[offset],
            //     srcMaxTensor[offset],
            //     remain_mask,
            //     m_actual,
            //     abs_params);

            // AscendC::Abs<ElementA,true>(
            //     srcMinTensor[offset],
            //     srcMinTensor[offset],
            //     remain_mask,
            //     m_actual,
            //     abs_params);

            // AscendC::PipeBarrier<PIPE_V>();

            AscendC::MulAddDst<ElementY, ElementA, true>(
                red_workspace,
                srcMaxTensor[offset],
                srcMinTensor[offset],
                remain_mask, m_actual,
                var_params);

            // AscendC::PipeBarrier<PIPE_V>();
        }

        uint64_t reduce_mask = (dst_repeat_num == 0) ? dst_remain : dst_repeat_size;
        AscendC::PipeBarrier<PIPE_V>();
        /*
        template <typename T, bool isSetMask = true>
        __aicore__ inline void WholeReduceSum(
            const LocalTensor<T>& dstLocal,
            const LocalTensor<T>& srcLocal,
            const int32_t mask,
            const int32_t repeatTimes,
            const int32_t dstRepStride,
            const int32_t srcBlkStride,
            const int32_t srcRepStride)
        */

        AscendC::WholeReduceSum<ElementY, true>(
            red_workspace,
            red_workspace,
            reduce_mask,
            m_actual,
            1,
            1,
            8);

        AscendC::PipeBarrier<PIPE_V>();

        uint32_t final_add_mask = (m_actual < dst_repeat_size) ? m_actual : dst_repeat_size;

        uint32_t dstOffset = m_round;

        var_params.dstRepStride = 8;
        var_params.src0RepStride = 8;
        var_params.src1RepStride = 8;

        // AscendC::Add<ElementY, true>(
        //     dstTensorVar,
        //     red_workspace,
        //     dstTensorVar,
        //     final_add_mask,
        //     CeilDiv(m_round, dst_repeat_size),
        //     var_params);

        AscendC::Add<ElementY>(
            dstTensorVar,
            red_workspace,
            dstTensorVar,
            final_add_mask);

        AscendC::PipeBarrier<PIPE_V>();
    }
};


template <>
struct TileReduce<Arch::AtlasA2,
                FTSelf::Gemv::helper::FT_THRESHOLD_ALGORITHM::ASVAR_ROBUST,
                FTSelf::Gemv::helper::FT_REDUCE_TYPE::VAR,
                Catlass::Gemm::GemmType<float, layout::RowMajor>,
                Catlass::Gemm::GemmType<float, layout::VectorLayout>,
                Catlass::Gemm::GemmType<float, layout::VectorLayout>,
                void>
{
    using ElementA = float;
    using ElementX = float;
    using ElementY = float;

    using ElementAccumulator =
        typename Catlass::Gemm::helper::ElementAccumulatorSelector<ElementA, ElementA>::ElementAccumulator;

    using FT_THRESHOLD_ALGORITHM = FTSelf::Gemv::helper::FT_THRESHOLD_ALGORITHM;

    static constexpr FT_THRESHOLD_ALGORITHM ALGO_TYPE = FTSelf::Gemv::helper::FT_THRESHOLD_ALGORITHM::ASVAR_ROBUST;
    using LayoutDst = layout::RowMajor;
    using LayoutSrc = layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementA);
    static constexpr uint32_t DST_ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementY);

    // Mehtods

    CATLASS_DEVICE
    TileReduce() {};

    //
    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementY> dstTensorVar,
        AscendC::LocalTensor<ElementA> srcMeanTensor,
        AscendC::LocalTensor<ElementA> srcMaxTensor,
        AscendC::LocalTensor<ElementA> srcMinTensor,
        AscendC::LocalTensor<ElementY> red_workspace,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc,
        ElementA n_ratio_factor)
    {
        uint32_t m_actual = layoutSrc.shape(0);
        uint32_t n_actual = layoutSrc.shape(1);
        uint32_t m_round = layoutDst.shape(0);
        uint32_t n_round = layoutDst.shape(1);

        // ElementY n_square_ratio_factor = n_ratio_factor * n_ratio_factor;

        uint32_t repeat_size = ELE_NUM_PER_C0 * 8;
        uint32_t dst_repeat_size = DST_ELE_NUM_PER_C0 * 8;

        uint32_t mask = repeat_size;
        uint32_t dst_mask = dst_repeat_size;

        uint32_t repeat_num = n_actual / repeat_size;
        uint32_t remain = n_actual % repeat_size;

        uint32_t dst_repeat_num = n_actual / dst_repeat_size;
        uint32_t dst_remain = n_actual % dst_repeat_size;

        uint64_t mean_mask = dst_repeat_size;
        uint64_t var_mask = dst_repeat_size;
        uint64_t sub_mask = dst_repeat_size;
        uint64_t abs_mask = dst_repeat_size;

        AscendC::UnaryRepeatParams mean_params;
        mean_params.dstBlkStride = 1;
        mean_params.srcBlkStride = 1;
        mean_params.dstRepStride = RoundUp(n_round, dst_repeat_size) / ELE_NUM_PER_C0;
        mean_params.srcRepStride = RoundUp(n_round, dst_repeat_size) / ELE_NUM_PER_C0;

        AscendC::UnaryRepeatParams abs_params;
        abs_params.dstBlkStride = 1;
        abs_params.srcBlkStride = 1;
        abs_params.dstRepStride = RoundUp(n_round, dst_repeat_size) / ELE_NUM_PER_C0;
        abs_params.srcRepStride = RoundUp(n_round, dst_repeat_size) / ELE_NUM_PER_C0;

        AscendC::BinaryRepeatParams var_params;
        var_params.dstBlkStride = 1;
        var_params.src0BlkStride = 1;
        var_params.src1BlkStride = 1;

        var_params.dstRepStride = RoundUp(dst_repeat_size, dst_repeat_size) / DST_ELE_NUM_PER_C0;
        var_params.src0RepStride = RoundUp(n_round, dst_repeat_size) / ELE_NUM_PER_C0;
        var_params.src1RepStride = RoundUp(n_round, dst_repeat_size) / ELE_NUM_PER_C0;

        AscendC::BinaryRepeatParams sub_params;
        sub_params.dstBlkStride = 1;
        sub_params.src0BlkStride = 1;
        sub_params.src1BlkStride = 1;

        sub_params.dstRepStride = RoundUp(n_round, dst_repeat_size) / ELE_NUM_PER_C0;
        sub_params.src0RepStride = RoundUp(n_round, dst_repeat_size) / ELE_NUM_PER_C0;
        sub_params.src1RepStride = RoundUp(n_round, dst_repeat_size) / ELE_NUM_PER_C0;

        AscendC::Duplicate<ElementY>(
                red_workspace,
                (ElementY)0.0,
                dst_repeat_size, // 每个迭代内部要对256 byte 的元素全部赋值为0，
                CeilDiv(m_actual * dst_repeat_size, dst_repeat_size), // 求行square和
                1, // 单次赋值迭代内，矢量目的操作数不同datablock间地址步长。
                8 // 相邻赋值迭代间，矢量目的操作数相同datablock地址步长。
        );

        AscendC::PipeBarrier<PIPE_V>();
        for(uint32_t i=0; i < dst_repeat_num; i++){
            uint32_t offset = i * dst_repeat_size;
            /*
            template <typename T, bool isSetMask = true>
            __aicore__ inline void Muls(const LocalTensor<T>& dstLocal,
                const LocalTensor<T>& srcLocal,
                const T& scalarValue,
                uint64_t mask,
                const uint8_t repeatTimes,
                const UnaryRepeatParams& repeatParams)
            */
            AscendC::Muls<ElementA, true>(
                srcMeanTensor[offset],
                srcMeanTensor[offset],
                n_ratio_factor,
                mean_mask,
                m_actual,
                mean_params);

            AscendC::PipeBarrier<PIPE_V>();
            /*
            template <typename T, bool isSetMask = true>
            __aicore__ inline void Sub(const LocalTensor<T>& dstLocal,
                const LocalTensor<T>& src0Local,
                const LocalTensor<T>& src1Local,
                uint64_t mask, const uint8_t repeatTimes,
                const BinaryRepeatParams& repeatParams)
            */
            AscendC::Sub<ElementA, true>(
                srcMaxTensor[offset],
                srcMaxTensor[offset],
                srcMeanTensor[offset],
                sub_mask,
                m_actual,
                sub_params);

            AscendC::Sub<ElementA, true>(
                srcMinTensor[offset],
                srcMeanTensor[offset],
                srcMinTensor[offset],
                sub_mask,
                m_actual,
                sub_params);

            AscendC::PipeBarrier<PIPE_V>();

            // template <typename T, bool isSetMask = true>
            // __aicore__ inline void Abs(const LocalTensor<T>& dstLocal,
            //     const LocalTensor<T>& srcLocal,
            //     uint64_t mask,
            //     const uint8_t repeatTimes,
            //     const UnaryRepeatParams& repeatParams)

            // AscendC::Abs<ElementA,true>(
            //     srcMaxTensor[offset],
            //     srcMaxTensor[offset],
            //     abs_mask,
            //     m_actual,
            //     abs_params);

            // AscendC::Abs<ElementA,true>(
            //     srcMinTensor[offset],
            //     srcMinTensor[offset],
            //     abs_mask,
            //     m_actual,
            //     abs_params);

            // AscendC::PipeBarrier<PIPE_V>();

            /*
            template <typename T, typename U, bool isSetMask = true>
            __aicore__ inline void MulAddDst(
                const LocalTensor<T>& dstLocal,
                const LocalTensor<U>& src0Local,
                const LocalTensor<U>& src1Local,
                uint64_t mask, const uint8_t repeatTimes,
                const BinaryRepeatParams& repeatParams)
            */

            AscendC::MulAddDst<ElementY, ElementA, true>(
                red_workspace,
                srcMaxTensor[offset],
                srcMinTensor[offset],
                var_mask, m_actual,
                var_params);

            AscendC::PipeBarrier<PIPE_V>();
        }

        if (dst_remain > 0)
        {
            uint32_t offset = dst_repeat_num * dst_repeat_size;

            if (offset + dst_repeat_size > n_actual)
            {
                dst_remain = n_actual - offset;
            }
            // m_actual *
            uint64_t remain_mask = dst_remain;

            AscendC::Muls<ElementA, true>(
                srcMeanTensor[offset],
                srcMeanTensor[offset],
                n_ratio_factor,
                dst_remain,
                m_actual,
                mean_params);

            AscendC::PipeBarrier<PIPE_V>();

            AscendC::Sub<ElementA, true>(
                srcMaxTensor[offset],
                srcMaxTensor[offset],
                srcMeanTensor[offset],
                remain_mask,
                m_actual,
                sub_params);

            AscendC::Sub<ElementA, true>(
                srcMinTensor[offset],
                srcMeanTensor[offset],
                srcMinTensor[offset],
                remain_mask,
                m_actual,
                sub_params);

            AscendC::PipeBarrier<PIPE_V>();

            // AscendC::Abs<ElementA,true>(
            //     srcMaxTensor[offset],
            //     srcMaxTensor[offset],
            //     remain_mask,
            //     m_actual,
            //     abs_params);

            // AscendC::Abs<ElementA,true>(
            //     srcMinTensor[offset],
            //     srcMinTensor[offset],
            //     remain_mask,
            //     m_actual,
            //     abs_params);

            // AscendC::PipeBarrier<PIPE_V>();

            AscendC::MulAddDst<ElementY, ElementA, true>(
                red_workspace,
                srcMaxTensor[offset],
                srcMinTensor[offset],
                remain_mask, m_actual,
                var_params);

            // AscendC::PipeBarrier<PIPE_V>();
        }

        uint64_t reduce_mask = (dst_repeat_num == 0) ? dst_remain : dst_repeat_size;
        AscendC::PipeBarrier<PIPE_V>();
        /*
        template <typename T, bool isSetMask = true>
        __aicore__ inline void WholeReduceSum(
            const LocalTensor<T>& dstLocal,
            const LocalTensor<T>& srcLocal,
            const int32_t mask,
            const int32_t repeatTimes,
            const int32_t dstRepStride,
            const int32_t srcBlkStride,
            const int32_t srcRepStride)
        */

        AscendC::WholeReduceSum<ElementY, true>(
            red_workspace,
            red_workspace,
            reduce_mask,
            m_actual,
            1,
            1,
            8);

        AscendC::PipeBarrier<PIPE_V>();

        uint32_t final_add_mask = (m_actual < dst_repeat_size) ? m_actual : dst_repeat_size;

        uint32_t dstOffset = m_round;

        var_params.dstRepStride = 8;
        var_params.src0RepStride = 8;
        var_params.src1RepStride = 8;

        // AscendC::Add<ElementY, true>(
        //     dstTensorVar,
        //     red_workspace,
        //     dstTensorVar,
        //     final_add_mask,
        //     CeilDiv(m_round, dst_repeat_size),
        //     var_params);

        AscendC::Add<ElementY>(
            dstTensorVar,
            red_workspace,
            dstTensorVar,
            final_add_mask);

        /*
        template <typename T>
        __aicore__ inline void Add(const LocalTensor<T>& dstLocal,
            const LocalTensor<T>& src0Local,
            const LocalTensor<T>& src1Local,
            const int32_t& calCount)
        */

        AscendC::PipeBarrier<PIPE_V>();
    }
};


template <>
struct TileReduce<Arch::AtlasA2,
                FTSelf::Gemv::helper::FT_THRESHOLD_ALGORITHM::ASVAR_ROBUST,
                FTSelf::Gemv::helper::FT_REDUCE_TYPE::VAR,
                Catlass::Gemm::GemmType<half, layout::RowMajor>,
                Catlass::Gemm::GemmType<half, layout::VectorLayout>,
                Catlass::Gemm::GemmType<half, layout::VectorLayout>,
                void>
{
    using ElementA = half;
    using ElementX = half;
    using ElementY = half;

    using ElementAccumulator =
        typename Catlass::Gemm::helper::ElementAccumulatorSelector<ElementA, ElementA>::ElementAccumulator;

    using FT_THRESHOLD_ALGORITHM = FTSelf::Gemv::helper::FT_THRESHOLD_ALGORITHM;

    static constexpr FT_THRESHOLD_ALGORITHM ALGO_TYPE = FTSelf::Gemv::helper::FT_THRESHOLD_ALGORITHM::ASVAR_ROBUST;
    using LayoutDst = layout::RowMajor;
    using LayoutSrc = layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementA);
    static constexpr uint32_t DST_ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementY);

    // Mehtods

    CATLASS_DEVICE
    TileReduce() {};

    //
    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementY> dstTensorVar,
        AscendC::LocalTensor<ElementA> srcMeanTensor,
        AscendC::LocalTensor<ElementA> srcMaxTensor,
        AscendC::LocalTensor<ElementA> srcMinTensor,
        AscendC::LocalTensor<ElementY> red_workspace,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc,
        ElementA n_ratio_factor)
    {
        uint32_t m_actual = layoutSrc.shape(0);
        uint32_t n_actual = layoutSrc.shape(1);
        uint32_t m_round = layoutDst.shape(0);
        uint32_t n_round = layoutDst.shape(1);

        // ElementY n_square_ratio_factor = n_ratio_factor * n_ratio_factor;

        uint32_t repeat_size = ELE_NUM_PER_C0 * 8;
        uint32_t dst_repeat_size = DST_ELE_NUM_PER_C0 * 8;

        uint32_t mask = repeat_size;
        uint32_t dst_mask = dst_repeat_size;

        uint32_t repeat_num = n_actual / repeat_size;
        uint32_t remain = n_actual % repeat_size;

        uint32_t dst_repeat_num = n_actual / dst_repeat_size;
        uint32_t dst_remain = n_actual % dst_repeat_size;

        uint64_t mean_mask = dst_repeat_size;
        uint64_t var_mask = dst_repeat_size;
        uint64_t sub_mask = dst_repeat_size;

        AscendC::UnaryRepeatParams mean_params;
        mean_params.dstBlkStride = 1;
        mean_params.srcBlkStride = 1;
        mean_params.dstRepStride = RoundUp(n_round, dst_repeat_size) / ELE_NUM_PER_C0;
        mean_params.srcRepStride = RoundUp(n_round, dst_repeat_size) / ELE_NUM_PER_C0;

        AscendC::BinaryRepeatParams var_params;
        var_params.dstBlkStride = 1;
        var_params.src0BlkStride = 1;
        var_params.src1BlkStride = 1;

        var_params.dstRepStride = RoundUp(dst_repeat_size, dst_repeat_size) / DST_ELE_NUM_PER_C0;
        var_params.src0RepStride = RoundUp(n_round, dst_repeat_size) / ELE_NUM_PER_C0;
        var_params.src1RepStride = RoundUp(n_round, dst_repeat_size) / ELE_NUM_PER_C0;

        AscendC::BinaryRepeatParams sub_params;
        sub_params.dstBlkStride = 1;
        sub_params.src0BlkStride = 1;
        sub_params.src1BlkStride = 1;

        sub_params.dstRepStride = RoundUp(n_round, dst_repeat_size) / ELE_NUM_PER_C0;
        sub_params.src0RepStride = RoundUp(n_round, dst_repeat_size) / ELE_NUM_PER_C0;
        sub_params.src1RepStride = RoundUp(n_round, dst_repeat_size) / ELE_NUM_PER_C0;

        AscendC::Duplicate<ElementY>(
                red_workspace,
                (ElementY)0.0,
                dst_repeat_size, // 每个迭代内部要对256 byte 的元素全部赋值为0，
                CeilDiv(m_actual * dst_repeat_size, dst_repeat_size), // 求行square和
                1, // 单次赋值迭代内，矢量目的操作数不同datablock间地址步长。
                8 // 相邻赋值迭代间，矢量目的操作数相同datablock地址步长。
        );

        AscendC::PipeBarrier<PIPE_V>();
        for(uint32_t i=0; i < dst_repeat_num; i++){
            uint32_t offset = i * dst_repeat_size;
            /*
            template <typename T, bool isSetMask = true>
            __aicore__ inline void Muls(const LocalTensor<T>& dstLocal,
                const LocalTensor<T>& srcLocal,
                const T& scalarValue,
                uint64_t mask,
                const uint8_t repeatTimes,
                const UnaryRepeatParams& repeatParams)
            */
            AscendC::Muls<ElementA, true>(
                srcMeanTensor[offset],
                srcMeanTensor[offset],
                n_ratio_factor,
                mean_mask,
                m_actual,
                mean_params);

            AscendC::PipeBarrier<PIPE_V>();
            /*
            template <typename T, bool isSetMask = true>
            __aicore__ inline void Sub(const LocalTensor<T>& dstLocal,
                const LocalTensor<T>& src0Local,
                const LocalTensor<T>& src1Local,
                uint64_t mask, const uint8_t repeatTimes,
                const BinaryRepeatParams& repeatParams)
            */
            AscendC::Sub<ElementA, true>(
                srcMaxTensor[offset],
                srcMaxTensor[offset],
                srcMeanTensor[offset],
                sub_mask,
                m_actual,
                sub_params);

            AscendC::Sub<ElementA, true>(
                srcMinTensor[offset],
                srcMeanTensor[offset],
                srcMinTensor[offset],
                sub_mask,
                m_actual,
                sub_params);

            AscendC::PipeBarrier<PIPE_V>();
            /*
            template <typename T, typename U, bool isSetMask = true>
            __aicore__ inline void MulAddDst(
                const LocalTensor<T>& dstLocal,
                const LocalTensor<U>& src0Local,
                const LocalTensor<U>& src1Local,
                uint64_t mask, const uint8_t repeatTimes,
                const BinaryRepeatParams& repeatParams)
            */

            AscendC::MulAddDst<ElementY, ElementA, true>(
                red_workspace,
                srcMaxTensor[offset],
                srcMinTensor[offset],
                var_mask, m_actual,
                var_params);

            AscendC::PipeBarrier<PIPE_V>();
        }

        if (dst_remain > 0)
        {
            uint32_t offset = dst_repeat_num * dst_repeat_size;

            if (offset + dst_repeat_size > n_actual)
            {
                dst_remain = n_actual - offset;
            }
            // m_actual *
            uint64_t remain_mask = dst_remain;

            AscendC::Muls<ElementA, true>(
                srcMeanTensor[offset],
                srcMeanTensor[offset],
                n_ratio_factor,
                dst_remain,
                m_actual,
                mean_params);

            AscendC::PipeBarrier<PIPE_V>();

            AscendC::Sub<ElementA, true>(
                srcMaxTensor[offset],
                srcMaxTensor[offset],
                srcMeanTensor[offset],
                remain_mask,
                m_actual,
                sub_params);

            AscendC::Sub<ElementA, true>(
                srcMinTensor[offset],
                srcMeanTensor[offset],
                srcMinTensor[offset],
                remain_mask,
                m_actual,
                sub_params);

            AscendC::PipeBarrier<PIPE_V>();

            AscendC::MulAddDst<ElementY, ElementA, true>(
                red_workspace,
                srcMaxTensor[offset],
                srcMinTensor[offset],
                remain_mask, m_actual,
                var_params);

            // AscendC::PipeBarrier<PIPE_V>();
        }

        uint64_t reduce_mask = (dst_repeat_num == 0) ? dst_remain : dst_repeat_size;
        AscendC::PipeBarrier<PIPE_V>();
        /*
        template <typename T, bool isSetMask = true>
        __aicore__ inline void WholeReduceSum(
            const LocalTensor<T>& dstLocal,
            const LocalTensor<T>& srcLocal,
            const int32_t mask,
            const int32_t repeatTimes,
            const int32_t dstRepStride,
            const int32_t srcBlkStride,
            const int32_t srcRepStride)
        */

        AscendC::WholeReduceSum<ElementY, true>(
            red_workspace,
            red_workspace,
            reduce_mask,
            m_actual,
            1,
            1,
            8);

        AscendC::PipeBarrier<PIPE_V>();

        uint32_t final_add_mask = (m_actual < dst_repeat_size) ? m_actual : dst_repeat_size;

        uint32_t dstOffset = m_round;

        var_params.dstRepStride = 8;
        var_params.src0RepStride = 8;
        var_params.src1RepStride = 8;

        AscendC::Add<ElementY, true>(
            dstTensorVar,
            red_workspace,
            dstTensorVar,
            final_add_mask,
            CeilDiv(m_round, dst_repeat_size),
            var_params);

        AscendC::PipeBarrier<PIPE_V>();
    }
};

template <>
struct TileReduce<Arch::AtlasA2,
                FTSelf::Gemv::helper::FT_THRESHOLD_ALGORITHM::ASVAR_ROBUST,
                FTSelf::Gemv::helper::FT_REDUCE_TYPE::VAR,
                Catlass::Gemm::GemmType<half, layout::RowMajor>,
                Catlass::Gemm::GemmType<half, layout::VectorLayout>,
                Catlass::Gemm::GemmType<float, layout::VectorLayout>,
                void>
{
    using ElementA = half;
    using ElementX = half;
    using ElementY = float;

    using ElementAccumulator =
        typename Catlass::Gemm::helper::ElementAccumulatorSelector<ElementA, ElementA>::ElementAccumulator;

    using FT_THRESHOLD_ALGORITHM = FTSelf::Gemv::helper::FT_THRESHOLD_ALGORITHM;

    static constexpr FT_THRESHOLD_ALGORITHM ALGO_TYPE = FTSelf::Gemv::helper::FT_THRESHOLD_ALGORITHM::ASVAR_ROBUST;
    using LayoutDst = layout::RowMajor;
    using LayoutSrc = layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementA);
    static constexpr uint32_t DST_ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementY);

    // Mehtods

    CATLASS_DEVICE
    TileReduce() {};

    //
    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementY> dstTensorVar,
        AscendC::LocalTensor<ElementA> srcMeanTensor,
        AscendC::LocalTensor<ElementA> srcMaxTensor,
        AscendC::LocalTensor<ElementA> srcMinTensor,
        AscendC::LocalTensor<ElementY> red_workspace,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc,
        ElementA n_ratio_factor)
    {
        uint32_t m_actual = layoutSrc.shape(0);
        uint32_t n_actual = layoutSrc.shape(1);
        uint32_t m_round = layoutDst.shape(0);
        uint32_t n_round = layoutDst.shape(1);

        // ElementY n_square_ratio_factor = n_ratio_factor * n_ratio_factor;

        uint32_t repeat_size = ELE_NUM_PER_C0 * 8;
        uint32_t dst_repeat_size = DST_ELE_NUM_PER_C0 * 8;

        uint32_t mask = repeat_size;
        uint32_t dst_mask = dst_repeat_size;

        uint32_t repeat_num = n_actual / repeat_size;
        uint32_t remain = n_actual % repeat_size;

        uint32_t dst_repeat_num = n_actual / dst_repeat_size;
        uint32_t dst_remain = n_actual % dst_repeat_size;

        uint64_t mean_mask = dst_repeat_size;
        uint64_t var_mask = dst_repeat_size;
        uint64_t sub_mask = dst_repeat_size;

        AscendC::UnaryRepeatParams mean_params;
        mean_params.dstBlkStride = 1;
        mean_params.srcBlkStride = 1;
        mean_params.dstRepStride = RoundUp(n_round, dst_repeat_size) / ELE_NUM_PER_C0;
        mean_params.srcRepStride = RoundUp(n_round, dst_repeat_size) / ELE_NUM_PER_C0;

        AscendC::BinaryRepeatParams var_params;
        var_params.dstBlkStride = 1;
        var_params.src0BlkStride = 1;
        var_params.src1BlkStride = 1;

        var_params.dstRepStride = RoundUp(dst_repeat_size, dst_repeat_size) / DST_ELE_NUM_PER_C0;
        var_params.src0RepStride = RoundUp(n_round, dst_repeat_size) / ELE_NUM_PER_C0;
        var_params.src1RepStride = RoundUp(n_round, dst_repeat_size) / ELE_NUM_PER_C0;

        AscendC::BinaryRepeatParams sub_params;
        sub_params.dstBlkStride = 1;
        sub_params.src0BlkStride = 1;
        sub_params.src1BlkStride = 1;

        sub_params.dstRepStride = RoundUp(n_round, dst_repeat_size) / ELE_NUM_PER_C0;
        sub_params.src0RepStride = RoundUp(n_round, dst_repeat_size) / ELE_NUM_PER_C0;
        sub_params.src1RepStride = RoundUp(n_round, dst_repeat_size) / ELE_NUM_PER_C0;

        AscendC::Duplicate<ElementY>(
                red_workspace,
                (ElementY)0.0,
                dst_repeat_size, // 每个迭代内部要对256 byte 的元素全部赋值为0，
                CeilDiv(m_actual * dst_repeat_size, dst_repeat_size), // 求行square和
                1, // 单次赋值迭代内，矢量目的操作数不同datablock间地址步长。
                8 // 相邻赋值迭代间，矢量目的操作数相同datablock地址步长。
        );

        AscendC::PipeBarrier<PIPE_V>();
        for(uint32_t i=0; i < dst_repeat_num; i++){
            uint32_t offset = i * dst_repeat_size;
            /*
            template <typename T, bool isSetMask = true>
            __aicore__ inline void Muls(const LocalTensor<T>& dstLocal,
                const LocalTensor<T>& srcLocal,
                const T& scalarValue,
                uint64_t mask,
                const uint8_t repeatTimes,
                const UnaryRepeatParams& repeatParams)
            */
            AscendC::Muls<ElementA, true>(
                srcMeanTensor[offset],
                srcMeanTensor[offset],
                n_ratio_factor,
                mean_mask,
                m_actual,
                mean_params);

            AscendC::PipeBarrier<PIPE_V>();
            /*
            template <typename T, bool isSetMask = true>
            __aicore__ inline void Sub(const LocalTensor<T>& dstLocal,
                const LocalTensor<T>& src0Local,
                const LocalTensor<T>& src1Local,
                uint64_t mask, const uint8_t repeatTimes,
                const BinaryRepeatParams& repeatParams)
            */
            AscendC::Sub<ElementA, true>(
                srcMaxTensor[offset],
                srcMaxTensor[offset],
                srcMeanTensor[offset],
                sub_mask,
                m_actual,
                sub_params);

            AscendC::Sub<ElementA, true>(
                srcMinTensor[offset],
                srcMeanTensor[offset],
                srcMinTensor[offset],
                sub_mask,
                m_actual,
                sub_params);

            AscendC::PipeBarrier<PIPE_V>();
            /*
            template <typename T, typename U, bool isSetMask = true>
            __aicore__ inline void MulAddDst(
                const LocalTensor<T>& dstLocal,
                const LocalTensor<U>& src0Local,
                const LocalTensor<U>& src1Local,
                uint64_t mask, const uint8_t repeatTimes,
                const BinaryRepeatParams& repeatParams)
            */

            AscendC::MulAddDst<ElementY, ElementA, true>(
                red_workspace,
                srcMaxTensor[offset],
                srcMinTensor[offset],
                var_mask, m_actual,
                var_params);

            AscendC::PipeBarrier<PIPE_V>();
        }

        if (dst_remain > 0)
        {
            uint32_t offset = dst_repeat_num * dst_repeat_size;

            if (offset + dst_repeat_size > n_actual)
            {
                dst_remain = n_actual - offset;
            }
            // m_actual *
            uint64_t remain_mask = dst_remain;

            AscendC::Muls<ElementA, true>(
                srcMeanTensor[offset],
                srcMeanTensor[offset],
                n_ratio_factor,
                dst_remain,
                m_actual,
                mean_params);

            AscendC::PipeBarrier<PIPE_V>();

            AscendC::Sub<ElementA, true>(
                srcMaxTensor[offset],
                srcMaxTensor[offset],
                srcMeanTensor[offset],
                remain_mask,
                m_actual,
                sub_params);

            AscendC::Sub<ElementA, true>(
                srcMinTensor[offset],
                srcMeanTensor[offset],
                srcMinTensor[offset],
                remain_mask,
                m_actual,
                sub_params);

            AscendC::PipeBarrier<PIPE_V>();

            AscendC::MulAddDst<ElementY, ElementA, true>(
                red_workspace,
                srcMaxTensor[offset],
                srcMinTensor[offset],
                remain_mask, m_actual,
                var_params);

            // AscendC::PipeBarrier<PIPE_V>();
        }

        uint64_t reduce_mask = (dst_repeat_num == 0) ? dst_remain : dst_repeat_size;
        AscendC::PipeBarrier<PIPE_V>();
        /*
        template <typename T, bool isSetMask = true>
        __aicore__ inline void WholeReduceSum(
            const LocalTensor<T>& dstLocal,
            const LocalTensor<T>& srcLocal,
            const int32_t mask,
            const int32_t repeatTimes,
            const int32_t dstRepStride,
            const int32_t srcBlkStride,
            const int32_t srcRepStride)
        */

        AscendC::WholeReduceSum<ElementY, true>(
            red_workspace,
            red_workspace,
            reduce_mask,
            m_actual,
            1,
            1,
            8);

        AscendC::PipeBarrier<PIPE_V>();

        uint32_t final_add_mask = (m_actual < dst_repeat_size) ? m_actual : dst_repeat_size;

        uint32_t dstOffset = m_round;

        var_params.dstRepStride = 8;
        var_params.src0RepStride = 8;
        var_params.src1RepStride = 8;

        AscendC::Add<ElementY, true>(
            dstTensorVar,
            red_workspace,
            dstTensorVar,
            final_add_mask,
            CeilDiv(m_round, dst_repeat_size),
            var_params);

        AscendC::PipeBarrier<PIPE_V>();
    }
};

template <
    class ElementA,
    class ElementX,
    class ElementY
>
struct TileReduce<Arch::AtlasA2,
                FTSelf::Gemv::helper::FT_THRESHOLD_ALGORITHM::ASVAR_SIMPLIFIED,
                FTSelf::Gemv::helper::FT_REDUCE_TYPE::VAR_SIMPLIFIED,
                Catlass::Gemm::GemmType<ElementA, layout::RowMajor>,
                Catlass::Gemm::GemmType<ElementX, layout::VectorLayout>,
                Catlass::Gemm::GemmType<ElementY, layout::VectorLayout>,
                void>
{
    using ElementAccumulator =
        typename Catlass::Gemm::helper::ElementAccumulatorSelector<ElementA, ElementA>::ElementAccumulator;

    using FT_THRESHOLD_ALGORITHM = FTSelf::Gemv::helper::FT_THRESHOLD_ALGORITHM;

    static constexpr FT_THRESHOLD_ALGORITHM ALGO_TYPE = FTSelf::Gemv::helper::FT_THRESHOLD_ALGORITHM::ASVAR_SIMPLIFIED;
    using LayoutDst = layout::RowMajor;
    using LayoutSrc = layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementA);
    static constexpr uint32_t DST_ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementY);

    // Mehtods

    CATLASS_DEVICE
    TileReduce() {};

    //
    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementY> dstTensorVar,
        AscendC::LocalTensor<ElementA> srcMaxTensor,
        AscendC::LocalTensor<ElementY> red_workspace,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        uint32_t m_actual = layoutSrc.shape(0);
        uint32_t n_actual = layoutSrc.shape(1);
        uint32_t m_round = layoutDst.shape(0);
        uint32_t n_round = layoutDst.shape(1);

        // ElementY n_square_ratio_factor = n_ratio_factor * n_ratio_factor;

        uint32_t repeat_size = ELE_NUM_PER_C0 * 8;
        uint32_t dst_repeat_size = DST_ELE_NUM_PER_C0 * 8;

        uint32_t mask = repeat_size;
        uint32_t dst_mask = dst_repeat_size;

        uint32_t repeat_num = n_actual / repeat_size;
        uint32_t remain = n_actual % repeat_size;

        uint32_t dst_repeat_num = n_actual / dst_repeat_size;
        uint32_t dst_remain = n_actual % dst_repeat_size;

        uint64_t mean_mask = dst_repeat_size;
        uint64_t var_mask = dst_repeat_size;
        uint64_t sub_mask = dst_repeat_size;

        AscendC::BinaryRepeatParams var_params;
        var_params.dstBlkStride = 1;
        var_params.src0BlkStride = 1;
        var_params.src1BlkStride = 1;

        var_params.dstRepStride = RoundUp(dst_repeat_size, dst_repeat_size) / DST_ELE_NUM_PER_C0;
        var_params.src0RepStride = RoundUp(n_round, dst_repeat_size) / ELE_NUM_PER_C0;
        var_params.src1RepStride = RoundUp(n_round, dst_repeat_size) / ELE_NUM_PER_C0;

        AscendC::Duplicate<ElementY>(
                red_workspace,
                (ElementY)0.0,
                dst_repeat_size, // 每个迭代内部要对256 byte 的元素全部赋值为0，
                CeilDiv(m_actual * dst_repeat_size, dst_repeat_size), // 求行square和
                1, // 单次赋值迭代内，矢量目的操作数不同datablock间地址步长。
                8 // 相邻赋值迭代间，矢量目的操作数相同datablock地址步长。
        );

        AscendC::PipeBarrier<PIPE_V>();
        for(uint32_t i=0; i < dst_repeat_num; i++){
            uint32_t offset = i * dst_repeat_size;
            /*
            template <typename T, bool isSetMask = true>
            __aicore__ inline void Sub(const LocalTensor<T>& dstLocal,
                const LocalTensor<T>& src0Local,
                const LocalTensor<T>& src1Local,
                uint64_t mask, const uint8_t repeatTimes,
                const BinaryRepeatParams& repeatParams)
            */

            /*
            template <typename T, typename U, bool isSetMask = true>
            __aicore__ inline void MulAddDst(
                const LocalTensor<T>& dstLocal,
                const LocalTensor<U>& src0Local,
                const LocalTensor<U>& src1Local,
                uint64_t mask, const uint8_t repeatTimes,
                const BinaryRepeatParams& repeatParams)
            */

            AscendC::MulAddDst<ElementY, ElementA, true>(
                red_workspace,
                srcMaxTensor[offset],
                srcMaxTensor[offset],
                var_mask, m_actual,
                var_params);

            AscendC::PipeBarrier<PIPE_V>();
        }

        if (dst_remain > 0)
        {
            uint32_t offset = dst_repeat_num * dst_repeat_size;

            if (offset + dst_repeat_size > n_actual)
            {
                dst_remain = n_actual - offset;
            }
            // m_actual *
            uint64_t remain_mask = dst_remain;

            AscendC::MulAddDst<ElementY, ElementA, true>(
                red_workspace,
                srcMaxTensor[offset],
                srcMaxTensor[offset],
                remain_mask, m_actual,
                var_params);

            // AscendC::PipeBarrier<PIPE_V>();
        }

        uint64_t reduce_mask = (dst_repeat_num == 0) ? dst_remain : dst_repeat_size;
        AscendC::PipeBarrier<PIPE_V>();
        /*
        template <typename T, bool isSetMask = true>
        __aicore__ inline void WholeReduceSum(
            const LocalTensor<T>& dstLocal,
            const LocalTensor<T>& srcLocal,
            const int32_t mask,
            const int32_t repeatTimes,
            const int32_t dstRepStride,
            const int32_t srcBlkStride,
            const int32_t srcRepStride)
        */

        AscendC::WholeReduceSum<ElementY, true>(
            red_workspace,
            red_workspace,
            reduce_mask,
            m_actual,
            1,
            1,
            8);

        AscendC::PipeBarrier<PIPE_V>();

        uint32_t final_add_mask = (m_actual < dst_repeat_size) ? m_actual : dst_repeat_size;

        uint32_t dstOffset = m_round;

        var_params.dstRepStride = 8;
        var_params.src0RepStride = 8;
        var_params.src1RepStride = 8;

        AscendC::Add<ElementY, true>(
            dstTensorVar,
            red_workspace,
            dstTensorVar,
            final_add_mask,
            CeilDiv(m_round, dst_repeat_size),
            var_params);

        AscendC::PipeBarrier<PIPE_V>();
    }
};

template <>
struct TileReduce<Arch::AtlasA2,
                FTSelf::Gemv::helper::FT_THRESHOLD_ALGORITHM::ASVAR_SIMPLIFIED,
                FTSelf::Gemv::helper::FT_REDUCE_TYPE::VAR_SIMPLIFIED,
                Catlass::Gemm::GemmType<float, layout::RowMajor>,
                Catlass::Gemm::GemmType<float, layout::VectorLayout>,
                Catlass::Gemm::GemmType<float, layout::VectorLayout>,
                void>
{
    using ElementA = float;
    using ElementY = float;
    using ElemetnX = float;

    using ElementAccumulator =
        typename Catlass::Gemm::helper::ElementAccumulatorSelector<ElementA, ElementA>::ElementAccumulator;

    using FT_THRESHOLD_ALGORITHM = FTSelf::Gemv::helper::FT_THRESHOLD_ALGORITHM;

    static constexpr FT_THRESHOLD_ALGORITHM ALGO_TYPE = FTSelf::Gemv::helper::FT_THRESHOLD_ALGORITHM::ASVAR_SIMPLIFIED;
    using LayoutDst = layout::RowMajor;
    using LayoutSrc = layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementA);
    static constexpr uint32_t DST_ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementY);

    // Mehtods

    CATLASS_DEVICE
    TileReduce() {};

    //
    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementY> dstTensorVar,
        AscendC::LocalTensor<ElementA> srcMaxTensor,
        AscendC::LocalTensor<ElementY> red_workspace,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        uint32_t m_actual = layoutSrc.shape(0);
        uint32_t n_actual = layoutSrc.shape(1);
        uint32_t m_round = layoutDst.shape(0);
        uint32_t n_round = layoutDst.shape(1);

        // ElementY n_square_ratio_factor = n_ratio_factor * n_ratio_factor;

        uint32_t repeat_size = ELE_NUM_PER_C0 * 8;
        uint32_t dst_repeat_size = DST_ELE_NUM_PER_C0 * 8;

        uint32_t mask = repeat_size;
        uint32_t dst_mask = dst_repeat_size;

        uint32_t repeat_num = n_actual / repeat_size;
        uint32_t remain = n_actual % repeat_size;

        uint32_t dst_repeat_num = n_actual / dst_repeat_size;
        uint32_t dst_remain = n_actual % dst_repeat_size;

        uint64_t mean_mask = dst_repeat_size;
        uint64_t var_mask = dst_repeat_size;
        uint64_t sub_mask = dst_repeat_size;

        AscendC::BinaryRepeatParams var_params;
        var_params.dstBlkStride = 1;
        var_params.src0BlkStride = 1;
        var_params.src1BlkStride = 1;

        var_params.dstRepStride = RoundUp(dst_repeat_size, dst_repeat_size) / DST_ELE_NUM_PER_C0;
        var_params.src0RepStride = RoundUp(n_round, dst_repeat_size) / ELE_NUM_PER_C0;
        var_params.src1RepStride = RoundUp(n_round, dst_repeat_size) / ELE_NUM_PER_C0;

        AscendC::Duplicate<ElementY>(
                red_workspace,
                (ElementY)0.0,
                dst_repeat_size, // 每个迭代内部要对256 byte 的元素全部赋值为0，
                CeilDiv(m_actual * dst_repeat_size, dst_repeat_size), // 求行square和
                1, // 单次赋值迭代内，矢量目的操作数不同datablock间地址步长。
                8 // 相邻赋值迭代间，矢量目的操作数相同datablock地址步长。
        );

        AscendC::PipeBarrier<PIPE_V>();
        for(uint32_t i=0; i < dst_repeat_num; i++){
            uint32_t offset = i * dst_repeat_size;
            /*
            template <typename T, bool isSetMask = true>
            __aicore__ inline void Sub(const LocalTensor<T>& dstLocal,
                const LocalTensor<T>& src0Local,
                const LocalTensor<T>& src1Local,
                uint64_t mask, const uint8_t repeatTimes,
                const BinaryRepeatParams& repeatParams)
            */

            /*
            template <typename T, typename U, bool isSetMask = true>
            __aicore__ inline void MulAddDst(
                const LocalTensor<T>& dstLocal,
                const LocalTensor<U>& src0Local,
                const LocalTensor<U>& src1Local,
                uint64_t mask, const uint8_t repeatTimes,
                const BinaryRepeatParams& repeatParams)
            */

            AscendC::MulAddDst<ElementY, ElementA, true>(
                red_workspace,
                srcMaxTensor[offset],
                srcMaxTensor[offset],
                var_mask, m_actual,
                var_params);

            AscendC::PipeBarrier<PIPE_V>();
        }

        if (dst_remain > 0)
        {
            uint32_t offset = dst_repeat_num * dst_repeat_size;

            if (offset + dst_repeat_size > n_actual)
            {
                dst_remain = n_actual - offset;
            }
            // m_actual *
            uint64_t remain_mask = dst_remain;

            AscendC::MulAddDst<ElementY, ElementA, true>(
                red_workspace,
                srcMaxTensor[offset],
                srcMaxTensor[offset],
                remain_mask, m_actual,
                var_params);

            // AscendC::PipeBarrier<PIPE_V>();
        }

        uint64_t reduce_mask = (dst_repeat_num == 0) ? dst_remain : dst_repeat_size;
        AscendC::PipeBarrier<PIPE_V>();
        /*
        template <typename T, bool isSetMask = true>
        __aicore__ inline void WholeReduceSum(
            const LocalTensor<T>& dstLocal,
            const LocalTensor<T>& srcLocal,
            const int32_t mask,
            const int32_t repeatTimes,
            const int32_t dstRepStride,
            const int32_t srcBlkStride,
            const int32_t srcRepStride)
        */

        AscendC::WholeReduceSum<ElementY, true>(
            red_workspace,
            red_workspace,
            reduce_mask,
            m_actual,
            1,
            1,
            8);

        AscendC::PipeBarrier<PIPE_V>();

        uint32_t final_add_mask = (m_actual < dst_repeat_size) ? m_actual : dst_repeat_size;

        uint32_t dstOffset = m_round;

        var_params.dstRepStride = 8;
        var_params.src0RepStride = 8;
        var_params.src1RepStride = 8;

        AscendC::Add<ElementY, true>(
            dstTensorVar,
            red_workspace,
            dstTensorVar,
            final_add_mask,
            CeilDiv(m_round, dst_repeat_size),
            var_params);

        AscendC::PipeBarrier<PIPE_V>();
    }
};


template <>
struct TileReduce<Arch::AtlasA2,
                FTSelf::Gemv::helper::FT_THRESHOLD_ALGORITHM::ASVAR_SIMPLIFIED,
                FTSelf::Gemv::helper::FT_REDUCE_TYPE::VAR_SIMPLIFIED,
                Catlass::Gemm::GemmType<half, layout::RowMajor>,
                Catlass::Gemm::GemmType<half, layout::VectorLayout>,
                Catlass::Gemm::GemmType<half, layout::VectorLayout>,
                void>
{
    using ElementA = half;
    using ElementY = half;
    using ElemetnX = half;

    using ElementAccumulator =
        typename Catlass::Gemm::helper::ElementAccumulatorSelector<ElementA, ElementA>::ElementAccumulator;

    using FT_THRESHOLD_ALGORITHM = FTSelf::Gemv::helper::FT_THRESHOLD_ALGORITHM;

    static constexpr FT_THRESHOLD_ALGORITHM ALGO_TYPE = FTSelf::Gemv::helper::FT_THRESHOLD_ALGORITHM::ASVAR_SIMPLIFIED;
    using LayoutDst = layout::RowMajor;
    using LayoutSrc = layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementA);
    static constexpr uint32_t DST_ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementY);

    // Mehtods

    CATLASS_DEVICE
    TileReduce() {};

    //
    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementY> dstTensorVar,
        AscendC::LocalTensor<ElementA> srcMaxTensor,
        AscendC::LocalTensor<ElementY> red_workspace,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        uint32_t m_actual = layoutSrc.shape(0);
        uint32_t n_actual = layoutSrc.shape(1);
        uint32_t m_round = layoutDst.shape(0);
        uint32_t n_round = layoutDst.shape(1);

        // ElementY n_square_ratio_factor = n_ratio_factor * n_ratio_factor;

        uint32_t repeat_size = ELE_NUM_PER_C0 * 8;
        uint32_t dst_repeat_size = DST_ELE_NUM_PER_C0 * 8;

        uint32_t mask = repeat_size;
        uint32_t dst_mask = dst_repeat_size;

        uint32_t repeat_num = n_actual / repeat_size;
        uint32_t remain = n_actual % repeat_size;

        uint32_t dst_repeat_num = n_actual / dst_repeat_size;
        uint32_t dst_remain = n_actual % dst_repeat_size;

        uint64_t mean_mask = dst_repeat_size;
        uint64_t var_mask = dst_repeat_size;
        uint64_t sub_mask = dst_repeat_size;

        AscendC::BinaryRepeatParams var_params;
        var_params.dstBlkStride = 1;
        var_params.src0BlkStride = 1;
        var_params.src1BlkStride = 1;

        var_params.dstRepStride = RoundUp(dst_repeat_size, dst_repeat_size) / DST_ELE_NUM_PER_C0;
        var_params.src0RepStride = RoundUp(n_round, dst_repeat_size) / ELE_NUM_PER_C0;
        var_params.src1RepStride = RoundUp(n_round, dst_repeat_size) / ELE_NUM_PER_C0;

        AscendC::Duplicate<ElementY>(
                red_workspace,
                (ElementY)0.0,
                dst_repeat_size, // 每个迭代内部要对256 byte 的元素全部赋值为0，
                CeilDiv(m_actual * dst_repeat_size, dst_repeat_size), // 求行square和
                1, // 单次赋值迭代内，矢量目的操作数不同datablock间地址步长。
                8 // 相邻赋值迭代间，矢量目的操作数相同datablock地址步长。
        );

        AscendC::PipeBarrier<PIPE_V>();
        for(uint32_t i=0; i < dst_repeat_num; i++){
            uint32_t offset = i * dst_repeat_size;
            /*
            template <typename T, bool isSetMask = true>
            __aicore__ inline void Sub(const LocalTensor<T>& dstLocal,
                const LocalTensor<T>& src0Local,
                const LocalTensor<T>& src1Local,
                uint64_t mask, const uint8_t repeatTimes,
                const BinaryRepeatParams& repeatParams)
            */

            /*
            template <typename T, typename U, bool isSetMask = true>
            __aicore__ inline void MulAddDst(
                const LocalTensor<T>& dstLocal,
                const LocalTensor<U>& src0Local,
                const LocalTensor<U>& src1Local,
                uint64_t mask, const uint8_t repeatTimes,
                const BinaryRepeatParams& repeatParams)
            */

            AscendC::MulAddDst<ElementY, ElementA, true>(
                red_workspace,
                srcMaxTensor[offset],
                srcMaxTensor[offset],
                var_mask, m_actual,
                var_params);

            AscendC::PipeBarrier<PIPE_V>();
        }

        if (dst_remain > 0)
        {
            uint32_t offset = dst_repeat_num * dst_repeat_size;

            if (offset + dst_repeat_size > n_actual)
            {
                dst_remain = n_actual - offset;
            }
            // m_actual *
            uint64_t remain_mask = dst_remain;

            AscendC::MulAddDst<ElementY, ElementA, true>(
                red_workspace,
                srcMaxTensor[offset],
                srcMaxTensor[offset],
                remain_mask, m_actual,
                var_params);

            // AscendC::PipeBarrier<PIPE_V>();
        }

        uint64_t reduce_mask = (dst_repeat_num == 0) ? dst_remain : dst_repeat_size;
        AscendC::PipeBarrier<PIPE_V>();
        /*
        template <typename T, bool isSetMask = true>
        __aicore__ inline void WholeReduceSum(
            const LocalTensor<T>& dstLocal,
            const LocalTensor<T>& srcLocal,
            const int32_t mask,
            const int32_t repeatTimes,
            const int32_t dstRepStride,
            const int32_t srcBlkStride,
            const int32_t srcRepStride)
        */

        AscendC::WholeReduceSum<ElementY, true>(
            red_workspace,
            red_workspace,
            reduce_mask,
            m_actual,
            1,
            1,
            8);

        AscendC::PipeBarrier<PIPE_V>();

        uint32_t final_add_mask = (m_actual < dst_repeat_size) ? m_actual : dst_repeat_size;

        uint32_t dstOffset = m_round;

        var_params.dstRepStride = 8;
        var_params.src0RepStride = 8;
        var_params.src1RepStride = 8;

        AscendC::Add<ElementY, true>(
            dstTensorVar,
            red_workspace,
            dstTensorVar,
            final_add_mask,
            CeilDiv(m_round, dst_repeat_size),
            var_params);

        AscendC::PipeBarrier<PIPE_V>();
    }
};

template <>
struct TileReduce<Arch::AtlasA2,
                FTSelf::Gemv::helper::FT_THRESHOLD_ALGORITHM::ASVAR_SIMPLIFIED,
                FTSelf::Gemv::helper::FT_REDUCE_TYPE::VAR_SIMPLIFIED,
                Catlass::Gemm::GemmType<half, layout::RowMajor>,
                Catlass::Gemm::GemmType<half, layout::VectorLayout>,
                Catlass::Gemm::GemmType<float, layout::VectorLayout>,
                void>
{
    using ElementA = half;
    using ElementY = float;
    using ElemetnX = half;

    using ElementAccumulator =
        typename Catlass::Gemm::helper::ElementAccumulatorSelector<ElementA, ElementA>::ElementAccumulator;

    using FT_THRESHOLD_ALGORITHM = FTSelf::Gemv::helper::FT_THRESHOLD_ALGORITHM;

    static constexpr FT_THRESHOLD_ALGORITHM ALGO_TYPE = FTSelf::Gemv::helper::FT_THRESHOLD_ALGORITHM::ASVAR_SIMPLIFIED;
    using LayoutDst = layout::RowMajor;
    using LayoutSrc = layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementA);
    static constexpr uint32_t DST_ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementY);

    // Mehtods

    CATLASS_DEVICE
    TileReduce() {};

    //
    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementY> dstTensorVar,
        AscendC::LocalTensor<ElementA> srcMaxTensor,
        AscendC::LocalTensor<ElementY> red_workspace,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        uint32_t m_actual = layoutSrc.shape(0);
        uint32_t n_actual = layoutSrc.shape(1);
        uint32_t m_round = layoutDst.shape(0);
        uint32_t n_round = layoutDst.shape(1);

        // ElementY n_square_ratio_factor = n_ratio_factor * n_ratio_factor;

        uint32_t repeat_size = ELE_NUM_PER_C0 * 8;
        uint32_t dst_repeat_size = DST_ELE_NUM_PER_C0 * 8;

        uint32_t mask = repeat_size;
        uint32_t dst_mask = dst_repeat_size;

        uint32_t repeat_num = n_actual / repeat_size;
        uint32_t remain = n_actual % repeat_size;

        uint32_t dst_repeat_num = n_actual / dst_repeat_size;
        uint32_t dst_remain = n_actual % dst_repeat_size;

        uint64_t mean_mask = dst_repeat_size;
        uint64_t var_mask = dst_repeat_size;
        uint64_t sub_mask = dst_repeat_size;

        AscendC::BinaryRepeatParams var_params;
        var_params.dstBlkStride = 1;
        var_params.src0BlkStride = 1;
        var_params.src1BlkStride = 1;

        var_params.dstRepStride = RoundUp(dst_repeat_size, dst_repeat_size) / DST_ELE_NUM_PER_C0;
        var_params.src0RepStride = RoundUp(n_round, dst_repeat_size) / ELE_NUM_PER_C0;
        var_params.src1RepStride = RoundUp(n_round, dst_repeat_size) / ELE_NUM_PER_C0;

        AscendC::Duplicate<ElementY>(
                red_workspace,
                (ElementY)0.0,
                dst_repeat_size, // 每个迭代内部要对256 byte 的元素全部赋值为0，
                CeilDiv(m_actual * dst_repeat_size, dst_repeat_size), // 求行square和
                1, // 单次赋值迭代内，矢量目的操作数不同datablock间地址步长。
                8 // 相邻赋值迭代间，矢量目的操作数相同datablock地址步长。
        );

        AscendC::PipeBarrier<PIPE_V>();
        for(uint32_t i=0; i < dst_repeat_num; i++){
            uint32_t offset = i * dst_repeat_size;
            /*
            template <typename T, bool isSetMask = true>
            __aicore__ inline void Sub(const LocalTensor<T>& dstLocal,
                const LocalTensor<T>& src0Local,
                const LocalTensor<T>& src1Local,
                uint64_t mask, const uint8_t repeatTimes,
                const BinaryRepeatParams& repeatParams)
            */

            /*
            template <typename T, typename U, bool isSetMask = true>
            __aicore__ inline void MulAddDst(
                const LocalTensor<T>& dstLocal,
                const LocalTensor<U>& src0Local,
                const LocalTensor<U>& src1Local,
                uint64_t mask, const uint8_t repeatTimes,
                const BinaryRepeatParams& repeatParams)
            */

            AscendC::MulAddDst<ElementY, ElementA, true>(
                red_workspace,
                srcMaxTensor[offset],
                srcMaxTensor[offset],
                var_mask, m_actual,
                var_params);

            AscendC::PipeBarrier<PIPE_V>();
        }

        if (dst_remain > 0)
        {
            uint32_t offset = dst_repeat_num * dst_repeat_size;

            if (offset + dst_repeat_size > n_actual)
            {
                dst_remain = n_actual - offset;
            }
            // m_actual *
            uint64_t remain_mask = dst_remain;

            AscendC::MulAddDst<ElementY, ElementA, true>(
                red_workspace,
                srcMaxTensor[offset],
                srcMaxTensor[offset],
                remain_mask, m_actual,
                var_params);

            // AscendC::PipeBarrier<PIPE_V>();
        }

        uint64_t reduce_mask = (dst_repeat_num == 0) ? dst_remain : dst_repeat_size;
        AscendC::PipeBarrier<PIPE_V>();
        /*
        template <typename T, bool isSetMask = true>
        __aicore__ inline void WholeReduceSum(
            const LocalTensor<T>& dstLocal,
            const LocalTensor<T>& srcLocal,
            const int32_t mask,
            const int32_t repeatTimes,
            const int32_t dstRepStride,
            const int32_t srcBlkStride,
            const int32_t srcRepStride)
        */

        AscendC::WholeReduceSum<ElementY, true>(
            red_workspace,
            red_workspace,
            reduce_mask,
            m_actual,
            1,
            1,
            8);

        AscendC::PipeBarrier<PIPE_V>();

        uint32_t final_add_mask = (m_actual < dst_repeat_size) ? m_actual : dst_repeat_size;

        uint32_t dstOffset = m_round;

        var_params.dstRepStride = 8;
        var_params.src0RepStride = 8;
        var_params.src1RepStride = 8;

        AscendC::Add<ElementY, true>(
            dstTensorVar,
            red_workspace,
            dstTensorVar,
            final_add_mask,
            CeilDiv(m_round, dst_repeat_size),
            var_params);

        AscendC::PipeBarrier<PIPE_V>();
    }
};

}

#endif
