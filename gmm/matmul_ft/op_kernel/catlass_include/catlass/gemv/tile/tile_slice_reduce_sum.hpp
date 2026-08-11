#ifndef CATLASS_GEMV_TILE_TILE_SLICE_SUM_HPP
#define CATLASS_GEMV_TILE_TILE_SLICE_SUM_HPP

#include "../../catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "../../gemm/helper.hpp"
#include "../../layout/layout.hpp"
#include "../../gemm/gemm_type.hpp"

namespace Vecself::Gemv::Tile {

using namespace Catlass;
template <
    /// Tag indicating architecture
    class ArchTag,
    class AType,
    class YType,
    class BiasType = void,
    bool  MakeDich = false
>
struct TileSliceSum
{
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported TileSliceSum, can not find the specialization.");
};

template <
    class ElementA,
    class ElementY
>
struct TileSliceSum<Arch::AtlasA2,
                Gemm::GemmType<ElementA, layout::RowMajor>,
                Gemm::GemmType<ElementY, layout::VectorLayout>,
                void,false>
{
    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementY>::ElementAccumulator;

    using LayoutDst = layout::RowMajor;
    using LayoutSrc = layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementA);

    // Mehtods

    CATLASS_DEVICE
    TileSliceSum() {};

    // 
    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementY> dstTensor,
        AscendC::LocalTensor<ElementA> srcTensor_m,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc
    )
    {
        uint32_t m_actual = layoutSrc.shape(0);
        uint32_t n_actual = layoutSrc.shape(1);
        uint32_t m_round = layoutDst.shape(0);
        uint32_t n_round = layoutDst.shape(1);
        uint32_t temp_repeat_size = BYTE_PER_C0 * 8 / sizeof(ElementA);
        uint32_t elem_repeat_size = ELE_NUM_PER_C0 * 8;
        uint32_t mask = elem_repeat_size;
        uint32_t add_mask = elem_repeat_size;
        uint32_t repeattimes = CeilDiv(n_actual, elem_repeat_size);

        uint32_t repeat_num = n_actual / temp_repeat_size;
        uint32_t remain = n_actual % temp_repeat_size;
        
        AscendC::BinaryRepeatParams add_params;
        add_params.dstBlkStride = 1;
        add_params.src0BlkStride = 1;
        add_params.src1BlkStride = 1;
        add_params.dstRepStride =  8;
        add_params.src0RepStride = 8;
        add_params.src1RepStride = 8;

        if(repeat_num > 0){
            for (uint32_t i = 1; i < m_actual; i++) {
                uint32_t offset = i * n_actual;
                AscendC::Add<ElementA, true>(
                    srcTensor_m,
                    srcTensor_m[offset],
                    srcTensor_m,
                    add_mask,
                    repeat_num,
                    add_params);
                
                AscendC::PipeBarrier<PIPE_V>();
            }
        }

        if (remain > 0)
        {
            uint32_t col_offset = repeat_num * elem_repeat_size;
            if (col_offset + remain > n_actual)
            {
                remain = n_actual - col_offset;
            }
            // m_actual * 
            uint64_t remain_mask = remain;
            for (uint32_t i = 1; i < m_actual; i++){
                uint32_t row_offset = i * n_actual;
                /*
                template <typename T>
                __aicore__ inline void 
                Add(const LocalTensor<T>& dstLocal, 
                    const LocalTensor<T>& src0Local, 
                    const LocalTensor<T>& src1Local,
                    const int32_t& calCount)
                */
                AscendC::Add<ElementA>(srcTensor_m[col_offset], 
                    srcTensor_m[row_offset+col_offset], srcTensor_m[col_offset], remain);
                AscendC::PipeBarrier<PIPE_V>();
            }
        }

        AscendC::PipeBarrier<PIPE_V>();

        /*
        每个repeat能处理的数据量取决于数据精度、AI处理器型号，如float->half转换每次迭代操作64个源/目的元素。
        当源操作数和目的操作数位数不同时，计算输入参数以数据类型的字节较大的为准。例如，源操作数为half类型，目的操作数为int32_t类型时，为保证输出和输入是连续的，dstRepStride应设置为8，srcRepStride应设置为4。
        dst与src的应为不同Tensor，或同一Tensor的同一元素，不支持同一Tensor的不同元素。
        src为float，dst为float时，取整模式表示向整数取整（仍为float类型），其他情况表示向dst数据类型所能表示的数字取整。
        */

        uint32_t dstOffset = n_round * 2;

        AscendC::UnaryRepeatParams castparams;
        castparams.dstBlkStride = 1;
        castparams.srcBlkStride = 1;
        castparams.dstRepStride = 4;
        castparams.srcRepStride = 8;

        /*
        template <typename T1, typename T2>
        __aicore__ inline void Cast(const LocalTensor<T1>& dstLocal, 
            const LocalTensor<T2>& srcLocal, 
            const RoundMode& round_mode, const uint32_t calCount)

        (uint64_t)mask, repeattimes, castparams
        */
        AscendC::Cast<ElementY, ElementA>(
            dstTensor[dstOffset],
            srcTensor_m,
            AscendC::RoundMode::CAST_NONE, n_actual);
        AscendC::PipeBarrier<PIPE_V>();

        /*
        template <typename T, bool isSetMask = true>
        __aicore__ inline void Muls(const LocalTensor<T>& dstLocal, 
            const LocalTensor<T>& srcLocal, const T& scalarValue, const int32_t& calCount)
        */
        // __aicore__ inline void Max(const LocalTensor<T>& dstLocal, 
        // const LocalTensor<T>& src0Local, 
        // const LocalTensor<T>& src1Local, const int32_t& calCount)

        AscendC::Add(dstTensor, dstTensor, dstTensor[dstOffset], n_round);
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
struct TileSliceSum<Arch::AtlasA2,
                Gemm::GemmType<float, layout::RowMajor>,
                Gemm::GemmType<float, layout::VectorLayout>,
                void,false>
{
    using ElementAccumulator = float;
    using ElementA = float;
    using ElementY = float;
        // typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementY>::ElementAccumulator;

    using LayoutDst = layout::RowMajor;
    using LayoutSrc = layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementA);

    // Mehtods

    CATLASS_DEVICE
    TileSliceSum() {};

    // 
    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<float> dstTensor,
        AscendC::LocalTensor<float> srcTensor_m,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc
    )
    {
        uint32_t m_actual = layoutSrc.shape(0);
        uint32_t n_actual = layoutSrc.shape(1);
        uint32_t m_round = layoutDst.shape(0);
        uint32_t n_round = layoutDst.shape(1);
        uint32_t temp_repeat_size = BYTE_PER_C0 * 8 / sizeof(ElementA);
        uint32_t elem_repeat_size = ELE_NUM_PER_C0 * 8;
        uint32_t mask = elem_repeat_size;
        uint32_t add_mask = elem_repeat_size;
        uint32_t repeattimes = CeilDiv(n_actual, elem_repeat_size);

        uint32_t repeat_num = n_actual / temp_repeat_size;
        uint32_t remain = n_actual % temp_repeat_size;
        
        AscendC::BinaryRepeatParams add_params;
        add_params.dstBlkStride = 1;
        add_params.src0BlkStride = 1;
        add_params.src1BlkStride = 1;
        add_params.dstRepStride =  8;
        add_params.src0RepStride = 8;
        add_params.src1RepStride = 8;

        if(repeat_num > 0){
            for (uint32_t i = 1; i < m_actual; i++) {
                uint32_t offset = i * n_actual;
                AscendC::Add<float, true>(
                    srcTensor_m,
                    srcTensor_m[offset],
                    srcTensor_m,
                    add_mask,
                    repeat_num,
                    add_params);
                
                AscendC::PipeBarrier<PIPE_V>();
            }
        }

        if (remain > 0)
        {
            uint32_t col_offset = repeat_num * elem_repeat_size;
            if (col_offset + remain > n_actual)
            {
                remain = n_actual - col_offset;
            }
            // m_actual * 
            uint64_t remain_mask = remain;
            for (uint32_t i = 1; i < m_actual; i++){
                uint32_t row_offset = i * n_actual;
                /*
                template <typename T>
                __aicore__ inline void 
                Add(const LocalTensor<T>& dstLocal, 
                    const LocalTensor<T>& src0Local, 
                    const LocalTensor<T>& src1Local,
                    const int32_t& calCount)
                */
                AscendC::Add<ElementA>(srcTensor_m[col_offset], 
                    srcTensor_m[row_offset+col_offset], srcTensor_m[col_offset], remain);
                AscendC::PipeBarrier<PIPE_V>();
            }
        }

        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Add(dstTensor, dstTensor, srcTensor_m, n_round);
        AscendC::PipeBarrier<PIPE_V>();
        
        // AscendC::Muls(dstTensor, dstTensor, alpha, m_actual);
        // AscendC::PipeBarrier<PIPE_V>();
    }
};


template <>
struct TileSliceSum<Arch::AtlasA2,
                Gemm::GemmType<half, layout::RowMajor>,
                Gemm::GemmType<float, layout::VectorLayout>,
                void,false>
{
    using ElementAccumulator = float;
    // typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementY>::ElementAccumulator;

    using LayoutDst = layout::RowMajor;
    using LayoutSrc = layout::RowMajor;

    using ElementA = half;
    using ElementY = float;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementA);

    // Mehtods

    CATLASS_DEVICE
    TileSliceSum() {};

    // 
    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<float> dstTensor,
        AscendC::LocalTensor<half> srcTensor_m,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc
    )
    {
        uint32_t m_actual = layoutSrc.shape(0);
        uint32_t n_actual = layoutSrc.shape(1);
        uint32_t m_round = layoutDst.shape(0);
        uint32_t n_round = layoutDst.shape(1);
        uint32_t temp_repeat_size = BYTE_PER_C0 * 8 / sizeof(ElementA);
        uint32_t elem_repeat_size = ELE_NUM_PER_C0 * 8;
        uint32_t mask = elem_repeat_size;
        uint32_t add_mask = elem_repeat_size;
        uint32_t repeattimes = CeilDiv(n_actual, elem_repeat_size);

        uint32_t repeat_num = n_actual / temp_repeat_size;
        uint32_t remain = n_actual % temp_repeat_size;
        
        AscendC::BinaryRepeatParams add_params;
        add_params.dstBlkStride = 1;
        add_params.src0BlkStride = 1;
        add_params.src1BlkStride = 1;
        add_params.dstRepStride =  8;
        add_params.src0RepStride = 8;
        add_params.src1RepStride = 8;

        if(repeat_num > 0){
            for (uint32_t i = 1; i < m_actual; i++) {
                uint32_t offset = i * n_actual;
                AscendC::Add<ElementA, true>(
                    srcTensor_m,
                    srcTensor_m[offset],
                    srcTensor_m,
                    add_mask,
                    repeat_num,
                    add_params);
                
                AscendC::PipeBarrier<PIPE_V>();
            }
        }

        if (remain > 0)
        {
            uint32_t col_offset = repeat_num * elem_repeat_size;
            if (col_offset + remain > n_actual)
            {
                remain = n_actual - col_offset;
            }
            // m_actual * 
            uint64_t remain_mask = remain;
            for (uint32_t i = 1; i < m_actual; i++){
                uint32_t row_offset = i * n_actual;
                /*
                template <typename T>
                __aicore__ inline void 
                Add(const LocalTensor<T>& dstLocal, 
                    const LocalTensor<T>& src0Local, 
                    const LocalTensor<T>& src1Local,
                    const int32_t& calCount)
                */
                AscendC::Add<ElementA>(srcTensor_m[col_offset], 
                    srcTensor_m[row_offset+col_offset], srcTensor_m[col_offset], remain);
                AscendC::PipeBarrier<PIPE_V>();
            }
        }

        AscendC::PipeBarrier<PIPE_V>();

        /*
        每个repeat能处理的数据量取决于数据精度、AI处理器型号，如float->half转换每次迭代操作64个源/目的元素。
        当源操作数和目的操作数位数不同时，计算输入参数以数据类型的字节较大的为准。例如，源操作数为half类型，目的操作数为int32_t类型时，为保证输出和输入是连续的，dstRepStride应设置为8，srcRepStride应设置为4。
        dst与src的应为不同Tensor，或同一Tensor的同一元素，不支持同一Tensor的不同元素。
        src为float，dst为float时，取整模式表示向整数取整（仍为float类型），其他情况表示向dst数据类型所能表示的数字取整。
        */

        uint32_t dstOffset = n_round * 2;

        AscendC::UnaryRepeatParams castparams;
        castparams.dstBlkStride = 1;
        castparams.srcBlkStride = 1;
        castparams.dstRepStride = 4;
        castparams.srcRepStride = 8;

        /*
        template <typename T1, typename T2>
        __aicore__ inline void Cast(const LocalTensor<T1>& dstLocal, 
            const LocalTensor<T2>& srcLocal, 
            const RoundMode& round_mode, const uint32_t calCount)

        (uint64_t)mask, repeattimes, castparams
        */
        AscendC::Cast<ElementY, ElementA>(
            dstTensor[dstOffset],
            srcTensor_m,
            AscendC::RoundMode::CAST_NONE, n_actual);
        AscendC::PipeBarrier<PIPE_V>();

        /*
        template <typename T, bool isSetMask = true>
        __aicore__ inline void Muls(const LocalTensor<T>& dstLocal, 
            const LocalTensor<T>& srcLocal, const T& scalarValue, const int32_t& calCount)
        */
        // __aicore__ inline void Max(const LocalTensor<T>& dstLocal, 
        // const LocalTensor<T>& src0Local, 
        // const LocalTensor<T>& src1Local, const int32_t& calCount)

        AscendC::Add(dstTensor, dstTensor, dstTensor[dstOffset], n_round);
        AscendC::PipeBarrier<PIPE_V>();
        
        // AscendC::Muls(dstTensor, dstTensor, alpha, m_actual);
        // AscendC::PipeBarrier<PIPE_V>();
    }
};


template <>
struct TileSliceSum<Arch::AtlasA2,
                Gemm::GemmType<half, layout::RowMajor>,
                Gemm::GemmType<half, layout::VectorLayout>,
                void,false>
{
    using ElementAccumulator = float;
    using ElementA = half;
    using ElementY = half;
        // typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementY>::ElementAccumulator;

    using LayoutDst = layout::RowMajor;
    using LayoutSrc = layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementA);

    // Mehtods

    CATLASS_DEVICE
    TileSliceSum() {};

    // 
    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<half> dstTensor,
        AscendC::LocalTensor<half> srcTensor_m,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc
    )
    {
        uint32_t m_actual = layoutSrc.shape(0);
        uint32_t n_actual = layoutSrc.shape(1);
        uint32_t m_round = layoutDst.shape(0);
        uint32_t n_round = layoutDst.shape(1);
        uint32_t temp_repeat_size = BYTE_PER_C0 * 8 / sizeof(ElementA);
        uint32_t elem_repeat_size = ELE_NUM_PER_C0 * 8;
        uint32_t mask = elem_repeat_size;
        uint32_t add_mask = elem_repeat_size;
        uint32_t repeattimes = CeilDiv(n_actual, elem_repeat_size);

        uint32_t repeat_num = n_actual / temp_repeat_size;
        uint32_t remain = n_actual % temp_repeat_size;
        
        AscendC::BinaryRepeatParams add_params;
        add_params.dstBlkStride = 1;
        add_params.src0BlkStride = 1;
        add_params.src1BlkStride = 1;
        add_params.dstRepStride =  8;
        add_params.src0RepStride = 8;
        add_params.src1RepStride = 8;

        if(repeat_num > 0){
            for (uint32_t i = 1; i < m_actual; i++) {
                uint32_t offset = i * n_actual;
                AscendC::Add<half, true>(
                    srcTensor_m,
                    srcTensor_m[offset],
                    srcTensor_m,
                    add_mask,
                    repeat_num,
                    add_params);
                
                AscendC::PipeBarrier<PIPE_V>();
            }
        }

        if (remain > 0)
        {
            uint32_t col_offset = repeat_num * elem_repeat_size;
            if (col_offset + remain > n_actual)
            {
                remain = n_actual - col_offset;
            }
            // m_actual * 
            uint64_t remain_mask = remain;
            for (uint32_t i = 1; i < m_actual; i++){
                uint32_t row_offset = i * n_actual;
                /*
                template <typename T>
                __aicore__ inline void 
                Add(const LocalTensor<T>& dstLocal, 
                    const LocalTensor<T>& src0Local, 
                    const LocalTensor<T>& src1Local,
                    const int32_t& calCount)
                */
                AscendC::Add<ElementA>(srcTensor_m[col_offset], 
                    srcTensor_m[row_offset+col_offset], srcTensor_m[col_offset], remain);
                AscendC::PipeBarrier<PIPE_V>();
            }
        }

        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Add(dstTensor, dstTensor, srcTensor_m, n_round);
        AscendC::PipeBarrier<PIPE_V>();
        
        // AscendC::Muls(dstTensor, dstTensor, alpha, m_actual);
        // AscendC::PipeBarrier<PIPE_V>();
    }
};


template <
    class ElementA,
    class ElementY
>
struct TileSliceSum<Arch::AtlasA2,
                Gemm::GemmType<ElementA, layout::RowMajor>,
                Gemm::GemmType<ElementY, layout::VectorLayout>,
                void,true>
{
    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementY>::ElementAccumulator;

    using LayoutDst = layout::RowMajor;
    using LayoutSrc = layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementA);

    // Mehtods

    CATLASS_DEVICE
    TileSliceSum() {};

    // 
    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementY> dstTensor,
        AscendC::LocalTensor<ElementA> srcTensor_m,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc
    )
    {
        uint32_t m_actual = layoutSrc.shape(0);
        uint32_t n_actual = layoutSrc.shape(1);
        uint32_t m_round = layoutDst.shape(0);
        uint32_t n_round = layoutDst.shape(1);
        uint32_t temp_repeat_size = BYTE_PER_C0 * 8 / sizeof(ElementA);
        uint32_t elem_repeat_size = ELE_NUM_PER_C0 * 8;
        uint32_t mask = elem_repeat_size;
        uint32_t add_mask = elem_repeat_size;
        uint32_t repeattimes = CeilDiv(n_actual, elem_repeat_size);

        uint32_t repeat_num = n_actual / temp_repeat_size;
        uint32_t remain = n_actual % temp_repeat_size;
        
        AscendC::BinaryRepeatParams add_params;
        add_params.dstBlkStride = 1;
        add_params.src0BlkStride = 1;
        add_params.src1BlkStride = 1;
        add_params.dstRepStride =  8;
        add_params.src0RepStride = 8;
        add_params.src1RepStride = 8;

        for(uint32_t grows=m_actual; grows>1; grows=grows/2){
            uint32_t remain_rows = grows % 2;
            uint32_t aligned_rows = grows - remain_rows;
            uint32_t aligned_rows_half = aligned_rows / 2;

            uint32_t aligned_offset = aligned_rows_half * n_actual;

            repeat_num = aligned_offset / temp_repeat_size;
            remain = aligned_offset % temp_repeat_size;

            uint32_t aligned_offset_remain = aligned_offset - remain;

            if(repeat_num > 0){
                AscendC::Add<ElementA, true>(
                    srcTensor_m,
                    srcTensor_m[aligned_offset],
                    srcTensor_m,
                    add_mask,
                    repeat_num,
                    add_params  
                );

                AscendC::PipeBarrier<PIPE_V>();
            }

            if(remain > 0){
                AscendC::Add<ElementA>(srcTensor_m[aligned_offset_remain], 
                    srcTensor_m[aligned_offset_remain + aligned_offset], 
                    srcTensor_m[aligned_offset_remain], remain);
                AscendC::PipeBarrier<PIPE_V>();
            }

            if(remain_rows > 0){
                uint32_t total_remain_size = remain_rows * n_actual;
                AscendC::Add<ElementA>(srcTensor_m, 
                    srcTensor_m[aligned_rows * n_actual], 
                    srcTensor_m, total_remain_size);
                AscendC::PipeBarrier<PIPE_V>();
            }
        }

        AscendC::PipeBarrier<PIPE_V>();

        /*
        每个repeat能处理的数据量取决于数据精度、AI处理器型号，如float->half转换每次迭代操作64个源/目的元素。
        当源操作数和目的操作数位数不同时，计算输入参数以数据类型的字节较大的为准。例如，源操作数为half类型，目的操作数为int32_t类型时，为保证输出和输入是连续的，dstRepStride应设置为8，srcRepStride应设置为4。
        dst与src的应为不同Tensor，或同一Tensor的同一元素，不支持同一Tensor的不同元素。
        src为float，dst为float时，取整模式表示向整数取整（仍为float类型），其他情况表示向dst数据类型所能表示的数字取整。
        */

        uint32_t dstOffset = n_round * 2;

        AscendC::UnaryRepeatParams castparams;
        castparams.dstBlkStride = 1;
        castparams.srcBlkStride = 1;
        castparams.dstRepStride = 4;
        castparams.srcRepStride = 8;

        /*
        template <typename T1, typename T2>
        __aicore__ inline void Cast(const LocalTensor<T1>& dstLocal, 
            const LocalTensor<T2>& srcLocal, 
            const RoundMode& round_mode, const uint32_t calCount)

        (uint64_t)mask, repeattimes, castparams
        */
        AscendC::Cast<ElementY, ElementA>(
            dstTensor[dstOffset],
            srcTensor_m,
            AscendC::RoundMode::CAST_NONE, n_actual);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Add(dstTensor, dstTensor, dstTensor[dstOffset], n_round);
        AscendC::PipeBarrier<PIPE_V>();
        
        // AscendC::Muls(dstTensor, dstTensor, alpha, m_actual);
        // AscendC::PipeBarrier<PIPE_V>();
    }
};

template <>
struct TileSliceSum<Arch::AtlasA2,
                Gemm::GemmType<float, layout::RowMajor>,
                Gemm::GemmType<float, layout::VectorLayout>,
                void,true>
{
    using ElementA = float;
    using ElementY = float;

    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementY>::ElementAccumulator;

    using LayoutDst = layout::RowMajor;
    using LayoutSrc = layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementA);

    // Mehtods

    CATLASS_DEVICE
    TileSliceSum() {};

    // 
    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<float> dstTensor,
        AscendC::LocalTensor<float> srcTensor_m,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc
    )
    {
        uint32_t m_actual = layoutSrc.shape(0);
        uint32_t n_actual = layoutSrc.shape(1);
        uint32_t m_round = layoutDst.shape(0);
        uint32_t n_round = layoutDst.shape(1);
        uint32_t temp_repeat_size = BYTE_PER_C0 * 8 / sizeof(ElementA);
        uint32_t elem_repeat_size = ELE_NUM_PER_C0 * 8;
        uint32_t mask = elem_repeat_size;
        uint32_t add_mask = elem_repeat_size;
        uint32_t repeattimes = CeilDiv(n_actual, elem_repeat_size);

        uint32_t repeat_num = n_actual / temp_repeat_size;
        uint32_t remain = n_actual % temp_repeat_size;
        
        AscendC::BinaryRepeatParams add_params;
        add_params.dstBlkStride = 1;
        add_params.src0BlkStride = 1;
        add_params.src1BlkStride = 1;
        add_params.dstRepStride =  8;
        add_params.src0RepStride = 8;
        add_params.src1RepStride = 8;

        for(uint32_t grows=m_actual; grows>1; grows=grows/2){
            uint32_t remain_rows = grows % 2;
            uint32_t aligned_rows = grows - remain_rows;
            uint32_t aligned_rows_half = aligned_rows / 2;

            uint32_t aligned_offset = aligned_rows_half * n_actual;

            repeat_num = aligned_offset / temp_repeat_size;
            remain = aligned_offset % temp_repeat_size;

            uint32_t aligned_offset_remain = aligned_offset - remain;

            if(repeat_num > 0){
                AscendC::Add<ElementA, true>(
                    srcTensor_m,
                    srcTensor_m[aligned_offset],
                    srcTensor_m,
                    add_mask,
                    repeat_num,
                    add_params  
                );

                AscendC::PipeBarrier<PIPE_V>();
            }

            if(remain > 0){
                AscendC::Add<ElementA>(srcTensor_m[aligned_offset_remain], 
                    srcTensor_m[aligned_offset_remain + aligned_offset], 
                    srcTensor_m[aligned_offset_remain], remain);
                AscendC::PipeBarrier<PIPE_V>();
            }

            if(remain_rows > 0){
                uint32_t total_remain_size = remain_rows * n_actual;
                AscendC::Add<ElementA>(srcTensor_m, 
                    srcTensor_m[aligned_rows * n_actual], 
                    srcTensor_m, total_remain_size);
                AscendC::PipeBarrier<PIPE_V>();
            }
        }

        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Add(dstTensor, dstTensor, srcTensor_m, n_round);
        AscendC::PipeBarrier<PIPE_V>();
        
        // AscendC::Muls(dstTensor, dstTensor, alpha, m_actual);
        // AscendC::PipeBarrier<PIPE_V>();
    }
};

template <>
struct TileSliceSum<Arch::AtlasA2,
                Gemm::GemmType<half, layout::RowMajor>,
                Gemm::GemmType<half, layout::VectorLayout>,
                void,true>
{
    using ElementA = half;
    using ElementY = half;

    using ElementAccumulator = float;
        // typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementY>::ElementAccumulator;

    using LayoutDst = layout::RowMajor;
    using LayoutSrc = layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementA);

    // Mehtods

    CATLASS_DEVICE
    TileSliceSum() {};

    // 
    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<half> dstTensor,
        AscendC::LocalTensor<half> srcTensor_m,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc
    )
    {
        uint32_t m_actual = layoutSrc.shape(0);
        uint32_t n_actual = layoutSrc.shape(1);
        uint32_t m_round = layoutDst.shape(0);
        uint32_t n_round = layoutDst.shape(1);
        uint32_t temp_repeat_size = BYTE_PER_C0 * 8 / sizeof(ElementA);
        uint32_t elem_repeat_size = ELE_NUM_PER_C0 * 8;
        uint32_t mask = elem_repeat_size;
        uint32_t add_mask = elem_repeat_size;
        uint32_t repeattimes = CeilDiv(n_actual, elem_repeat_size);

        uint32_t repeat_num = n_actual / temp_repeat_size;
        uint32_t remain = n_actual % temp_repeat_size;
        
        AscendC::BinaryRepeatParams add_params;
        add_params.dstBlkStride = 1;
        add_params.src0BlkStride = 1;
        add_params.src1BlkStride = 1;
        add_params.dstRepStride =  8;
        add_params.src0RepStride = 8;
        add_params.src1RepStride = 8;

        for(uint32_t grows=m_actual; grows>1; grows=grows/2){
            uint32_t remain_rows = grows % 2;
            uint32_t aligned_rows = grows - remain_rows;
            uint32_t aligned_rows_half = aligned_rows / 2;

            uint32_t aligned_offset = aligned_rows_half * n_actual;

            repeat_num = aligned_offset / temp_repeat_size;
            remain = aligned_offset % temp_repeat_size;

            uint32_t aligned_offset_remain = aligned_offset - remain;

            if(repeat_num > 0){
                AscendC::Add<ElementA, true>(
                    srcTensor_m,
                    srcTensor_m[aligned_offset],
                    srcTensor_m,
                    add_mask,
                    repeat_num,
                    add_params  
                );

                AscendC::PipeBarrier<PIPE_V>();
            }

            if(remain > 0){
                AscendC::Add<ElementA>(srcTensor_m[aligned_offset_remain], 
                    srcTensor_m[aligned_offset_remain + aligned_offset], 
                    srcTensor_m[aligned_offset_remain], remain);
                AscendC::PipeBarrier<PIPE_V>();
            }

            if(remain_rows > 0){
                uint32_t total_remain_size = remain_rows * n_actual;
                AscendC::Add<ElementA>(srcTensor_m, 
                    srcTensor_m[aligned_rows * n_actual], 
                    srcTensor_m, total_remain_size);
                AscendC::PipeBarrier<PIPE_V>();
            }
        }

        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Add(dstTensor, dstTensor, srcTensor_m, n_round);
        AscendC::PipeBarrier<PIPE_V>();
        
        // AscendC::Muls(dstTensor, dstTensor, alpha, m_actual);
        // AscendC::PipeBarrier<PIPE_V>();
    }
};

template <>
struct TileSliceSum<Arch::AtlasA2,
                Gemm::GemmType<half, layout::RowMajor>,
                Gemm::GemmType<float, layout::VectorLayout>,
                void,true>
{
    using ElementA = half;
    using ElementY = float;
    using ElementAccumulator = float;
    
    // typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementY>::ElementAccumulator;

    using LayoutDst = layout::RowMajor;
    using LayoutSrc = layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementA);

    // Mehtods

    CATLASS_DEVICE
    TileSliceSum() {};

    // 
    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<float> dstTensor,
        AscendC::LocalTensor<half> srcTensor_m,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc
    )
    {
        uint32_t m_actual = layoutSrc.shape(0);
        uint32_t n_actual = layoutSrc.shape(1);
        uint32_t m_round = layoutDst.shape(0);
        uint32_t n_round = layoutDst.shape(1);
        uint32_t temp_repeat_size = BYTE_PER_C0 * 8 / sizeof(ElementA);
        uint32_t elem_repeat_size = ELE_NUM_PER_C0 * 8;
        uint32_t mask = elem_repeat_size;
        uint32_t add_mask = elem_repeat_size;
        uint32_t repeattimes = CeilDiv(n_actual, elem_repeat_size);

        uint32_t repeat_num = n_actual / temp_repeat_size;
        uint32_t remain = n_actual % temp_repeat_size;
        
        AscendC::BinaryRepeatParams add_params;
        add_params.dstBlkStride = 1;
        add_params.src0BlkStride = 1;
        add_params.src1BlkStride = 1;
        add_params.dstRepStride =  8;
        add_params.src0RepStride = 8;
        add_params.src1RepStride = 8;

        for(uint32_t grows=m_actual; grows>1; grows=grows/2){
            uint32_t remain_rows = grows % 2;
            uint32_t aligned_rows = grows - remain_rows;
            uint32_t aligned_rows_half = aligned_rows / 2;

            uint32_t aligned_offset = aligned_rows_half * n_actual;

            repeat_num = aligned_offset / temp_repeat_size;
            remain = aligned_offset % temp_repeat_size;

            uint32_t aligned_offset_remain = aligned_offset - remain;

            if(repeat_num > 0){
                AscendC::Add<ElementA, true>(
                    srcTensor_m,
                    srcTensor_m[aligned_offset],
                    srcTensor_m,
                    add_mask,
                    repeat_num,
                    add_params  
                );

                AscendC::PipeBarrier<PIPE_V>();
            }

            if(remain > 0){
                AscendC::Add<ElementA>(srcTensor_m[aligned_offset_remain], 
                    srcTensor_m[aligned_offset_remain + aligned_offset], 
                    srcTensor_m[aligned_offset_remain], remain);
                AscendC::PipeBarrier<PIPE_V>();
            }

            if(remain_rows > 0){
                uint32_t total_remain_size = remain_rows * n_actual;
                AscendC::Add<ElementA>(srcTensor_m, 
                    srcTensor_m[aligned_rows * n_actual], 
                    srcTensor_m, total_remain_size);
                AscendC::PipeBarrier<PIPE_V>();
            }
        }

        AscendC::PipeBarrier<PIPE_V>();

        /*
        每个repeat能处理的数据量取决于数据精度、AI处理器型号，如float->half转换每次迭代操作64个源/目的元素。
        当源操作数和目的操作数位数不同时，计算输入参数以数据类型的字节较大的为准。例如，源操作数为half类型，目的操作数为int32_t类型时，为保证输出和输入是连续的，dstRepStride应设置为8，srcRepStride应设置为4。
        dst与src的应为不同Tensor，或同一Tensor的同一元素，不支持同一Tensor的不同元素。
        src为float，dst为float时，取整模式表示向整数取整（仍为float类型），其他情况表示向dst数据类型所能表示的数字取整。
        */

        uint32_t dstOffset = n_round * 2;

        AscendC::UnaryRepeatParams castparams;
        castparams.dstBlkStride = 1;
        castparams.srcBlkStride = 1;
        castparams.dstRepStride = 4;
        castparams.srcRepStride = 8;

        /*
        template <typename T1, typename T2>
        __aicore__ inline void Cast(const LocalTensor<T1>& dstLocal, 
            const LocalTensor<T2>& srcLocal, 
            const RoundMode& round_mode, const uint32_t calCount)

        (uint64_t)mask, repeattimes, castparams
        */
        AscendC::Cast<float, half>(
            dstTensor[dstOffset],
            srcTensor_m,
            AscendC::RoundMode::CAST_NONE, n_actual);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Add(dstTensor, dstTensor, dstTensor[dstOffset], n_round);
        AscendC::PipeBarrier<PIPE_V>();
        
        // AscendC::Muls(dstTensor, dstTensor, alpha, m_actual);
        // AscendC::PipeBarrier<PIPE_V>();
    }
};


}

#endif
