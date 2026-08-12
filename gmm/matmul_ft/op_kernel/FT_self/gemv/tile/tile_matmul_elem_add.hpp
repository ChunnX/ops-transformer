#ifndef CATLASS_GEMV_TILE_TILE_MATMUL_ELEM_ADD_HPP_SELF
#define CATLASS_GEMV_TILE_TILE_MATMUL_ELEM_ADD_HPP_SELF

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/gemm/helper.hpp"
#include "FT_self/gemv/helper.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/gemm/gemm_type.hpp"

namespace FTSelf::Gemv::Tile {

using namespace Catlass;

template <
    /// Tag indicating architecture
    class ArchTag,
    class AType,
    class YType,
    class BiasType = void
>
struct TileMatmulAdd
{
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported TileMatmulAdd, can not find the specialization.");
};

template <
    class ElementA,
    class ElementY
>
struct TileMatmulAdd<Arch::AtlasA2,
                Catlass::Gemm::GemmType<ElementA, layout::RowMajor>,
                Catlass::Gemm::GemmType<ElementY, layout::RowMajor>,
                void>
{

};

template <
    class ElementA,
    class ElementY
>
struct TileMatmulAdd<Arch::AtlasA2,
                Catlass::Gemm::GemmType<ElementA, layout::VectorLayout>,
                Catlass::Gemm::GemmType<ElementY, layout::VectorLayout>,
                void>
{

};

template <
    class ElementA,
    class ElementY
>
struct TileMatmulAdd<Arch::AtlasA2,
                Catlass::Gemm::GemmType<ElementA, layout::RowMajor>,
                Catlass::Gemm::GemmType<ElementY, layout::VectorLayout>,
                void>
{

};

template <>
struct TileMatmulAdd<Arch::AtlasA2,
                Catlass::Gemm::GemmType<float, layout::RowMajor>,
                Catlass::Gemm::GemmType<float, layout::RowMajor>,
                void>
{
    using ElementA = float;
    using ElementX = float;
    using ElementY = float;
    using ElementAccumulator = ElementY;

    using LayoutDst = layout::RowMajor;
    using LayoutSrc = layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementA);

    // Methods

    CATLASS_DEVICE
    TileMatmulAdd() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementY> dstTensor,
        AscendC::LocalTensor<ElementA> srcTensor_m1,
        AscendC::LocalTensor<ElementA> srcTensor_m2,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc
    )
    {
        uint32_t m_actual = layoutSrc.shape(0);
        uint32_t n_actual = layoutSrc.shape(1);
        uint32_t m_round = layoutDst.shape(0);
        uint32_t n_round = layoutDst.shape(1);

        uint32_t repeat_size = ELE_NUM_PER_C0 * 8;
        uint32_t mask = repeat_size;
        uint32_t repeat_num = n_actual / repeat_size;
        uint32_t remain = n_actual % repeat_size;

        // m_actual *
        uint64_t add_mask = repeat_size;

        /*
        控制操作数地址步长的参数。BinaryRepeatParams类型，
        包含操作数相邻迭代间相同datablock的地址步长，
        操作数同一迭代内不同datablock的地址步长等参数。

        相邻迭代间的地址步长参数说明请参考repeatStride；
        同一迭代内datablock的地址步长参数说明请参考dataBlockStride。
        */

        AscendC::BinaryRepeatParams params;
        params.dstBlkStride = 1;
        params.src0BlkStride = 1;
        params.src1BlkStride = 1;
        params.dstRepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;
        params.src0RepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;
        params.src1RepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;

        for (uint32_t i = 0; i < repeat_num; i++) {
            uint32_t offset = i * repeat_size;
            AscendC::Add<ElementA, true>(
                dstTensor[offset],
                srcTensor_m1[offset],
                srcTensor_m2[offset],
                add_mask,
                m_actual,
                params);
        }

        if (remain > 0)
        {
            uint32_t remain_offset = repeat_num * repeat_size;
            if (remain_offset + remain > n_actual)
            {
                remain = n_actual - remain_offset;
            }
            // m_actual *
            uint64_t remain_mask = remain;
            AscendC::Add<ElementA, true>(
                dstTensor[remain_offset],
                srcTensor_m1[remain_offset],
                srcTensor_m2[remain_offset],
                remain_mask,
                m_actual,
                params);
        }

        AscendC::PipeBarrier<PIPE_V>();
    }
};

template <>
struct TileMatmulAdd<Arch::AtlasA2,
                Catlass::Gemm::GemmType<float, layout::VectorLayout>,
                Catlass::Gemm::GemmType<float, layout::VectorLayout>,
                void>
{
    using ElementA = float;
    using ElementX = float;
    using ElementY = float;
    using ElementAccumulator = ElementY;

    using LayoutDst = layout::VectorLayout;
    using LayoutSrc = layout::VectorLayout;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementA);
    static constexpr uint32_t MAX_COMPUTE_LENGTH = 8192;

    // Methods

    CATLASS_DEVICE
    TileMatmulAdd() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementY> dstTensor,
        AscendC::LocalTensor<ElementA> srcTensor_m1,
        AscendC::LocalTensor<ElementA> srcTensor_m2,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc
    )
    {
        uint32_t total_ele_num_round = layoutDst.shape(0);
        uint32_t total_ele_num = layoutSrc.shape(0);

        uint32_t repeat_size = ELE_NUM_PER_C0 * 8;
        uint32_t mask = repeat_size;
        uint32_t repeat_num = total_ele_num / MAX_COMPUTE_LENGTH;
        uint32_t remain = total_ele_num % MAX_COMPUTE_LENGTH;

        for(uint32_t i = 0; i < repeat_num; i++){
            // Do the calculation
            uint32_t offset = i * MAX_COMPUTE_LENGTH;
            AscendC::Add(dstTensor[offset], srcTensor_m1[offset], srcTensor_m2[offset], MAX_COMPUTE_LENGTH);
        }

        if(remain > 0){
            uint32_t remain_offset = repeat_num * MAX_COMPUTE_LENGTH;

            if (remain_offset + remain > total_ele_num)
            {
                remain = total_ele_num - remain_offset;
            }
            AscendC::Add(dstTensor[remain_offset], srcTensor_m1[remain_offset], srcTensor_m2[remain_offset], remain);
        }

        AscendC::PipeBarrier<PIPE_V>();
    }
};



template <>
struct TileMatmulAdd<Arch::AtlasA2,
                Catlass::Gemm::GemmType<half, layout::RowMajor>,
                Catlass::Gemm::GemmType<half, layout::RowMajor>,
                void>
{
    using ElementA = half;
    using ElementX = half;
    using ElementY = half;
    using ElementAccumulator = ElementY;

    using LayoutDst = layout::RowMajor;
    using LayoutSrc = layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementA);

    // Methods

    CATLASS_DEVICE
    TileMatmulAdd() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementY> dstTensor,
        AscendC::LocalTensor<ElementA> srcTensor_m1,
        AscendC::LocalTensor<ElementA> srcTensor_m2,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc
    )
    {
        uint32_t m_actual = layoutSrc.shape(0);
        uint32_t n_actual = layoutSrc.shape(1);
        uint32_t m_round = layoutDst.shape(0);
        uint32_t n_round = layoutDst.shape(1);

        uint32_t repeat_size = ELE_NUM_PER_C0 * 8;
        uint32_t mask = repeat_size;
        uint32_t repeat_num = n_actual / repeat_size;
        uint32_t remain = n_actual % repeat_size;

        // m_actual *
        uint64_t add_mask = repeat_size;

        /*
        控制操作数地址步长的参数。BinaryRepeatParams类型，
        包含操作数相邻迭代间相同datablock的地址步长，
        操作数同一迭代内不同datablock的地址步长等参数。

        相邻迭代间的地址步长参数说明请参考repeatStride；
        同一迭代内datablock的地址步长参数说明请参考dataBlockStride。
        */

        AscendC::BinaryRepeatParams params;
        params.dstBlkStride = 1;
        params.src0BlkStride = 1;
        params.src1BlkStride = 1;
        params.dstRepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;
        params.src0RepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;
        params.src1RepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;

        for (uint32_t i = 0; i < repeat_num; i++) {
            uint32_t offset = i * repeat_size;
            AscendC::Add<ElementA, true>(
                dstTensor[offset],
                srcTensor_m1[offset],
                srcTensor_m2[offset],
                add_mask,
                m_actual,
                params);
        }

        if (remain > 0)
        {
            uint32_t remain_offset = repeat_num * repeat_size;
            if (remain_offset + remain > n_actual)
            {
                remain = n_actual - remain_offset;
            }
            // m_actual *
            uint64_t remain_mask = remain;
            AscendC::Add<ElementA, true>(
                dstTensor[remain_offset],
                srcTensor_m1[remain_offset],
                srcTensor_m2[remain_offset],
                remain_mask,
                m_actual,
                params);
        }

        AscendC::PipeBarrier<PIPE_V>();
    }
};

template <>
struct TileMatmulAdd<Arch::AtlasA2,
                Catlass::Gemm::GemmType<half, layout::VectorLayout>,
                Catlass::Gemm::GemmType<half, layout::VectorLayout>,
                void>
{
    using ElementA = half;
    using ElementX = half;
    using ElementY = half;
    using ElementAccumulator = ElementY;

    using LayoutDst = layout::VectorLayout;
    using LayoutSrc = layout::VectorLayout;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementA);
    static constexpr uint32_t MAX_COMPUTE_LENGTH = 8192;

    // Methods

    CATLASS_DEVICE
    TileMatmulAdd() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementY> dstTensor,
        AscendC::LocalTensor<ElementA> srcTensor_m1,
        AscendC::LocalTensor<ElementA> srcTensor_m2,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc
    )
    {
        uint32_t total_ele_num_round = layoutDst.shape(0);
        uint32_t total_ele_num = layoutSrc.shape(0);

        uint32_t repeat_size = ELE_NUM_PER_C0 * 8;
        uint32_t mask = repeat_size;
        uint32_t repeat_num = total_ele_num / MAX_COMPUTE_LENGTH;
        uint32_t remain = total_ele_num % MAX_COMPUTE_LENGTH;

        for(uint32_t i = 0; i < repeat_num; i++){
            // Do the calculation
            uint32_t offset = i * MAX_COMPUTE_LENGTH;
            AscendC::Add(dstTensor[offset], srcTensor_m1[offset], srcTensor_m2[offset], MAX_COMPUTE_LENGTH);
        }

        if(remain > 0){
            uint32_t remain_offset = repeat_num * MAX_COMPUTE_LENGTH;

            if (remain_offset + remain > total_ele_num)
            {
                remain = total_ele_num - remain_offset;
            }
            AscendC::Add(dstTensor[remain_offset], srcTensor_m1[remain_offset], srcTensor_m2[remain_offset], remain);
        }

        AscendC::PipeBarrier<PIPE_V>();
    }
};


template <>
struct TileMatmulAdd<Arch::AtlasA2,
                Catlass::Gemm::GemmType<half, layout::RowMajor>,
                Catlass::Gemm::GemmType<float, layout::RowMajor>,
                void>
{
    using ElementA = half;
    using ElementX = half;
    using ElementY = float;
    using ElementAccumulator = ElementY;

    using LayoutDst = layout::RowMajor;
    using LayoutSrc = layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementA);
    static constexpr uint32_t DST_ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementY);

    // Methods

    CATLASS_DEVICE
    TileMatmulAdd() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementY> dstTensor,
        AscendC::LocalTensor<ElementA> srcTensor_m1,
        AscendC::LocalTensor<ElementA> srcTensor_m2,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc
    )
    {
        uint32_t m_actual = layoutSrc.shape(0);
        uint32_t n_actual = layoutSrc.shape(1);
        uint32_t m_round = layoutDst.shape(0);
        uint32_t n_round = layoutDst.shape(1);

        uint32_t repeat_size = ELE_NUM_PER_C0 * 8;
        uint32_t dst_repeat_size = DST_ELE_NUM_PER_C0 * 8;
        uint32_t mask = repeat_size;
        uint32_t repeat_num = n_actual / repeat_size;
        uint32_t dst_repeat_num = n_actual / dst_repeat_size;
        uint32_t remain = n_actual % repeat_size;
        uint32_t dst_remain = n_actual % dst_repeat_size;

        // m_actual *
        uint64_t add_mask = repeat_size;
        uint64_t dst_cast_mask = dst_repeat_size;

        /*
        控制操作数地址步长的参数。BinaryRepeatParams类型，
        包含操作数相邻迭代间相同datablock的地址步长，
        操作数同一迭代内不同datablock的地址步长等参数。

        相邻迭代间的地址步长参数说明请参考repeatStride；
        同一迭代内datablock的地址步长参数说明请参考dataBlockStride。
        */

        AscendC::BinaryRepeatParams params;
        params.dstBlkStride = 1;
        params.src0BlkStride = 1;
        params.src1BlkStride = 1;
        params.dstRepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;
        params.src0RepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;
        params.src1RepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;

        for (uint32_t i = 0; i < repeat_num; i++) {
            uint32_t offset = i * repeat_size;
            AscendC::Add<ElementA, true>(
                srcTensor_m1[offset],
                srcTensor_m1[offset],
                srcTensor_m2[offset],
                add_mask,
                m_actual,
                params);
        }

        if (remain > 0)
        {
            uint32_t remain_offset = repeat_num * repeat_size;
            if (remain_offset + remain > n_actual)
            {
                remain = n_actual - remain_offset;
            }
            // m_actual *
            uint64_t remain_mask = remain;
            AscendC::Add<ElementA, true>(
                srcTensor_m1[remain_offset],
                srcTensor_m1[remain_offset],
                srcTensor_m2[remain_offset],
                remain_mask,
                m_actual,
                params);
        }

        AscendC::PipeBarrier<PIPE_V>();

        AscendC::UnaryRepeatParams castparams;
        castparams.dstBlkStride = 1;
        castparams.srcBlkStride = 1;
        castparams.dstRepStride = RoundUp(n_round, dst_repeat_size) / DST_ELE_NUM_PER_C0;
        castparams.srcRepStride = RoundUp(n_round, dst_repeat_size) / ELE_NUM_PER_C0;

        for (uint32_t i = 0; i < dst_repeat_num; i++) {
            uint32_t offset = i * dst_repeat_size;
            AscendC::Cast<ElementY, ElementA, true>(
                dstTensor[offset],
                srcTensor_m1[offset],
                AscendC::RoundMode::CAST_NONE,
                dst_cast_mask,
                m_actual,
                castparams);
        }

        if (dst_remain > 0)
        {
            uint32_t remain_offset = dst_repeat_num * dst_repeat_size;
            if (remain_offset + dst_remain > n_actual)
            {
                dst_remain = n_actual - remain_offset;
            }
            // m_actual *
            uint64_t remain_mask = dst_remain;

            AscendC::Cast<ElementY, ElementA, true>(
                dstTensor[remain_offset],
                srcTensor_m1[remain_offset],
                AscendC::RoundMode::CAST_NONE,
                remain_mask,
                m_actual,
                castparams);
        }

        AscendC::PipeBarrier<PIPE_V>();
    }
};

template <>
struct TileMatmulAdd<Arch::AtlasA2,
                Catlass::Gemm::GemmType<half, layout::VectorLayout>,
                Catlass::Gemm::GemmType<float, layout::VectorLayout>,
                void>
{
    using ElementA = half;
    using ElementX = half;
    using ElementY = float;
    using ElementAccumulator = ElementY;

    using LayoutDst = layout::VectorLayout;
    using LayoutSrc = layout::VectorLayout;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementA);
    static constexpr uint32_t MAX_COMPUTE_LENGTH = 8192;

    // Methods

    CATLASS_DEVICE
    TileMatmulAdd() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementY> dstTensor,
        AscendC::LocalTensor<ElementA> srcTensor_m1,
        AscendC::LocalTensor<ElementA> srcTensor_m2,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc
    )
    {
        uint32_t total_ele_num_round = layoutDst.shape(0);
        uint32_t total_ele_num = layoutSrc.shape(0);

        uint32_t repeat_size = ELE_NUM_PER_C0 * 8;
        uint32_t mask = repeat_size;
        uint32_t repeat_num = total_ele_num / MAX_COMPUTE_LENGTH;
        uint32_t remain = total_ele_num % MAX_COMPUTE_LENGTH;

        for(uint32_t i = 0; i < repeat_num; i++){
            // Do the calculation
            uint32_t offset = i * MAX_COMPUTE_LENGTH;
            AscendC::Add(srcTensor_m1[offset], srcTensor_m1[offset], srcTensor_m2[offset], MAX_COMPUTE_LENGTH);
            AscendC::Cast<ElementY, ElementA>(
                dstTensor[offset],
                srcTensor_m1[offset],
                AscendC::RoundMode::CAST_NONE,
                MAX_COMPUTE_LENGTH);
        }

        if(remain > 0){
            uint32_t remain_offset = repeat_num * MAX_COMPUTE_LENGTH;

            if (remain_offset + remain > total_ele_num)
            {
                remain = total_ele_num - remain_offset;
            }

            AscendC::Add(srcTensor_m1[remain_offset], srcTensor_m1[remain_offset], srcTensor_m2[remain_offset], remain);

            AscendC::Cast<ElementY, ElementA>(
                dstTensor[remain_offset],
                srcTensor_m1[remain_offset],
                AscendC::RoundMode::CAST_NONE,
                remain);
        }

        AscendC::PipeBarrier<PIPE_V>();
    }
};

}
#endif // CATLASS_GEMV_TILE_TILE_FAULT_SUM_HPP_SELF
