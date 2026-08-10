#ifndef CATLASS_GEMV_TILE_TILE_FAULT_VMAD_HPP_SELF
#define CATLASS_GEMV_TILE_TILE_FAULT_VMAD_HPP_SELF

#include "../../catlass.hpp"
#include "../../arch/arch.hpp"
#include "../../gemm/helper.hpp"
#include "../../layout/layout.hpp"
#include "../../gemm/gemm_type.hpp"

namespace Vecself::Gemv::Tile {

using namespace Catlass;

template <
    /// Tag indicating architecture
    class ArchTag,
    class AType,
    class XType,
    class YType,
    class BiasType = void
>
struct TileFaultVmad
{
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported TileFaultVmad, can not find the specialization.");
};

template <
    class ElementA,
    class ElementX,
    class ElementY
>
struct TileFaultVmad<Arch::AtlasA2,
                Gemm::GemmType<ElementA, layout::RowMajor>,
                Gemm::GemmType<ElementX, layout::VectorLayout>,
                Gemm::GemmType<ElementY, layout::VectorLayout>,
                void>
{
    using ElementAccumulator = ElementY;

    using LayoutDst = layout::RowMajor;
    using LayoutSrc = layout::RowMajor;
    
    // 这里是在矩阵上每个datablock 包含的element的数量
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementA);

    // Mehtods

    CATLASS_DEVICE
    TileFaultVmad() {};

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

        // 这里是输出向量上每次256byte对应的元素的数量，要以此为准确定迭代次数，规模等
        uint32_t temp_repeat_size = BYTE_PER_C0 * 8 / sizeof(ElementAccumulator);
        uint32_t elem_repeat_size = ELE_NUM_PER_C0 * 8;
        uint32_t mask = temp_repeat_size;

        uint32_t repeattimes = CeilDiv(m_actual, temp_repeat_size);

        // 将输出向量赋值为0
        AscendC::Duplicate<ElementAccumulator>(
            temp,
            (ElementAccumulator)0.0,
            // 这是为了后续我们每次求全部行的局部行和，
            // 我们对每行局部和的存储空间设定为256byte对齐，
            // 这是因为每次赋值操作的 repeat 迭代内的运算能力是256byte
            temp_repeat_size, // 每个迭代内部要对256 byte 的元素全部赋值为0，
            // 我们的目标是求行和，所以我们的输出空间是行的倍数，且是datablock的倍数
            // 因此在这里，我们在赋值0时通过每次内部迭代对应一个输出行的局部和存储与accum空间
            CeilDiv(m_round * temp_repeat_size, temp_repeat_size), // 求行和
            1, // 单次赋值迭代内，矢量目的操作数不同datablock间地址步长。
            8 // 相邻赋值迭代间，矢量目的操作数相同datablock地址步长。
        ); // 总的来说，我们开辟了256 byte x m_round的空间，全部赋值为0，每256byte对应一个行的局部和存储与accum

        uint32_t repeat_num = n_actual / temp_repeat_size; // 这里是因为每次在行上做temp_repeat_size个column的元素求行和，
        // 然后每次迭代在多行上做完，再滑动到下一批列上去做

        uint32_t remain = n_actual % temp_repeat_size; // 剩余的数量，多做一次

        AscendC::PipeBarrier<PIPE_V>();
        /*
        BinaryRepeatParams为用于控制操作数地址步长的数据结构。
        结构体内包含操作数相邻迭代间相同datablock的地址步长，
        操作数同一迭代内不同datablock的地址步长等参数。
        这里我们用BinaryRepeatParams控制求多个局部行和的内层repeat 迭代的属性
        */
        AscendC::BinaryRepeatParams params;
        /*
        内层repeat 迭代中，每个迭代处理256byte的数据的乘加操作。共8个datablock，
        所以blkstride设置的每次迭代内各个datablock起始地址之间的距离，单位为datablock，
        设置方法为：
        dataBlockStride
        dataBlockStride是指同一迭代内不同datablock的地址步长。
        1. 连续计算，dataBlockStride设置为1，对同一迭代内的8个datablock数据连续进行处理。
        2. 非连续计算，dataBlockStride值大于1（如取2），
            同一迭代内不同datablock之间在读取数据时出现一个datablock的间隔。
        */
        params.dstBlkStride = 1;
        params.src0BlkStride = 1;
        params.src1BlkStride = 1;

        // 目标输出中每个内部repeat iteration之间起始地址的差异，
        // 单位为datablock（32 byte）
        // 这里我们设置每个内部repeat iteration之间的地址差异为8个datablock的空间
        // 这是因为我们每行上留了8个datablock的accum空间，
        // 每次向量乘计算每行上8个datablock元素的相乘结果，
        // 按照元素对应累加到dst的accum空间上完成局部和的运算
        params.dstRepStride = RoundUp(temp_repeat_size, temp_repeat_size) / (BYTE_PER_C0 / sizeof(ElementAccumulator));

        // 如上文所示，在我们的计算中包括内外两个repeat iteration
        // 对于内层 repeat 而言，每次每行上的256byte的元素与对应vector的元素相乘，结果累加到相应行对应的accum空间上，
        // 重复m_actual次，将每行上的局部累加结果计算完成，放到输出的目标操作数空间的对应行存储空间上
        // 对于外层iteration而言，重复在列上做滑动窗口即可，
        // 这里的params设置的是每次外层iteration内进行操作的repeat参数，即设置的是内层repeat的属性。
        // 内层 repeat 中，每个repeat在矩阵A上的stride就是一行的大小，单位为datablock
        params.src0RepStride = RoundUp(n_round, elem_repeat_size) / ELE_NUM_PER_C0;
        // 在内层repeat 中，每次循环在矩阵上换行，但是处理的列是固定的，这意味着在向量上是不滑动的，
        // 只有在外层循环之间才要滑动完成最终整体行和。
        params.src1RepStride = 0;
        AscendC::SetMaskCount();
        AscendC::SetVectorMask<ElementAccumulator, AscendC::MaskMode::COUNTER>(m_actual * temp_repeat_size);
        for(uint32_t i=0; i < repeat_num; i++)
        {
            uint32_t offset = i * temp_repeat_size;
            /*
            两个源操作数对应元素相乘后加到目标操作数相应位置上做聚合
            */
            AscendC::MulAddDst<ElementAccumulator, ElementA, false>(
                temp,
                srcTensor_m[offset], // 这里是起始地址，每次外层迭代都是从第一行的相应列偏移开始的
                srcTensor_v[offset],
                AscendC::MASK_PLACEHOLDER,
                1,
                params);
            
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::SetMaskNorm();
        AscendC::ResetMask();

        if(remain > 0)
        {
            uint32_t offset = repeat_num * temp_repeat_size;
            if (offset + remain > n_round)
            {
                remain = n_round - offset;
            }

            uint64_t remain_mask = remain;
            /*
            bool 类型模版参数：
            isSetMask：是否在接口内部设置mask。
                true，表示在接口内部设置mask。
                false，表示在接口外部设置mask，开发者需要使用SetVectorMask接口设置mask值。
                这种模式下，本接口入参中的mask值必须设置为占位符MASK_PLACEHOLDER。如上方代码所示
            */
            AscendC::MulAddDst<ElementAccumulator, ElementA, true>(
                temp,
                srcTensor_m[offset],
                srcTensor_v[offset],
                remain_mask, // 每次内层repeat 迭代内参与计算的元素的数量，单位是元素
                m_actual, // 内层 repeat 的迭代次数
                params); // 内层repeat迭代的stride属性
        }

        // 进行归约聚合的元素的数量，若进行的repeat次数多于一次，
        // 则代表每次内部repeat中计算的局部列数超过了256byte对应的元素，
        // 可以完成一次默认的repeat最大规模，反之则仅为remain个列
        uint64_t reduce_mask = (repeat_num == 0) ? remain : temp_repeat_size;
        /*
        函数功能: 在每个迭代内，将所有数据求和。
        */
        AscendC::PipeBarrier<PIPE_V>();
        /*
        isSetMask = true
        */
        AscendC::WholeReduceSum<ElementAccumulator,true>(
            temp, // 目标tensor
            temp, // src tensor
            reduce_mask, // 每次repeat 迭代内参与元素的数量，这里一般为256byte对应的列数，即每行上预留的accum空间，除非列数过少，少于256byte对应的列数
            m_actual, // repeat 迭代次数，即只考虑前面m_actual个行作为输出向量的有效值，即矩阵的行数
            1,  // 目标tensor相邻repeat迭代操作间步长，单位为目标tensor元素类型字节数（即目标tensor元素的数量）
            1, // 源操作数单次迭代内相邻datablock的步长，单位为datablock
            8); // 源操作数相邻repeat迭代间的地址步长，即源操作数每次repeat迭代跳过的datablock数目
        
        AscendC::PipeBarrier<PIPE_V>();
        // 完成数据类型的转换
        AscendC::UnaryRepeatParams castparams;

        castparams.dstBlkStride = 1;
        castparams.srcBlkStride = 1;
        castparams.dstRepStride = 4;
        castparams.srcRepStride = 8;
        /*
        template <typename T1, typename T2, bool isSetMask = true>
        __aicore__ inline void Cast(const LocalTensor<T1>& dstLocal, 
            const LocalTensor<T2>& srcLocal, const RoundMode& round_mode, 
            const uint64_t mask, const uint8_t repeatTimes, 
            const UnaryRepeatParams& repeatParams)
        T1: 目的操作数类型
        T2：源操作数类型
        函数功能
        根据源操作数和目的操作数Tensor的数据类型进行精度转换。
        将源操作数精度转换为目的操作数的精度之后写入目的操作数

        这里是将中间聚合完成的行和向量降精度后存放在输入向量的头部。
        */
        AscendC::Cast<ElementA, ElementAccumulator, true>(
            srcTensor_m, // 目的操作数
            temp, // 源操作数
            AscendC::RoundMode::CAST_NONE, // 舍入方式
            (uint64_t)mask, // 每次repeat iteration 内 256 byte 的输出向量元素参与计算
            repeattimes, // 重复 实际行数 / 256 byte 元素数量个迭代，即将整个输出向量完成
            castparams);
        /*
        CAST_NONE = 0,  // 在转换有精度损失时表示CAST_RINT模式，不涉及精度损失时表示不舍入
        CAST_RINT,      // rint，四舍六入五成双舍入
        CAST_RINT模式下，若待舍入部分的第一位为0，则不进位；
        若第一位为1且后续位不全为0，则进位；
        若第一位为1且后续位全为0，
        当M（即位数中保留位数的最后一位）的最后一位为0则不进位，当M的最后一位为1则进位。
        */
        AscendC::PipeBarrier<PIPE_V>();
        // 此时temp，即中间结果已经存放到矩阵上了，且元素精度已经转化为矩阵的元素精度了
        // 这里默认其实精度会下降float -> half
        // 将结果累加到输出向量中，每个repeat iteration内参与元素的数量
        // 若实际行向量的规模小于256byte 对应的矩阵元素数量则为m_actual，否则为对应矩阵元素数量
        uint64_t add_mask = (m_actual < elem_repeat_size) ? m_actual : elem_repeat_size;

        params.dstRepStride = 8;
        params.src0RepStride = 8;
        params.src1RepStride = 8; // 这里每次输出8个datablock了，其中元素类型为ElementA，即矩阵精度
        // 将结果写出，即累加到到dstTensor完成运算
        // 此处按照ElementA的精度进行累加
        AscendC::Add<ElementA, true>(
            dstTensor, // 将 m_round 个元素，即目标行和向量累加到目标向量中即可
            srcTensor_m,
            dstTensor,
            (uint64_t)add_mask, // 每次repeat iteration 的规模即 256 byte 下元素类型为ElementA的元素规模
            CeilDiv(m_round, elem_repeat_size), // 目标行向量的规模
            params);
    }
};

template <>
struct TileFaultVmad<Arch::AtlasA2,
                Gemm::GemmType<float, layout::RowMajor>,
                Gemm::GemmType<float, layout::VectorLayout>,
                Gemm::GemmType<float, layout::VectorLayout>,
                void>
{
    using ElementA = float;
    using ElementX = float;
    using ElementY = float;
    using ElementAccumulator = ElementY;

    using LayoutDst = layout::RowMajor;
    using LayoutSrc = layout::RowMajor;
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementA);

    // Mehtods

    CATLASS_DEVICE
    TileFaultVmad() {};

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

        uint32_t repeat_size = ELE_NUM_PER_C0 * 8;
        uint32_t mask = repeat_size;
        uint32_t repeat_num = n_actual / repeat_size;
        uint32_t remain = n_actual % repeat_size;

        AscendC::BinaryRepeatParams params;
        params.dstBlkStride = 1;
        params.src0BlkStride = 1;
        params.src1BlkStride = 1;
        params.dstRepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;
        params.src0RepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;
        params.src1RepStride = 0;
        AscendC::SetMaskCount();
        AscendC::SetVectorMask<ElementA, AscendC::MaskMode::COUNTER>(m_actual * repeat_size);
        for (uint32_t i = 0; i < repeat_num; i++)
        {
            uint32_t offset = i * repeat_size;
            if (i == 0)
            {
                AscendC::Mul<ElementA, false>(
                    srcTensor_m,
                    srcTensor_m,
                    srcTensor_v,
                    AscendC::MASK_PLACEHOLDER,
                    1,
                    params);
            }
            else
            {
                AscendC::MulAddDst<ElementA, ElementA, false>(
                    srcTensor_m,
                    srcTensor_m[offset],
                    srcTensor_v[offset],
                    AscendC::MASK_PLACEHOLDER,
                    1,
                    params);
            }
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::SetMaskNorm();
        AscendC::ResetMask();

        if (remain > 0)
        {
            uint32_t offset = repeat_num * repeat_size;
            if (offset + remain > n_round)
            {
                remain = n_round - offset;
            }
            uint64_t remain_mask = remain;
            if (repeat_num == 0)
            {
                AscendC::Mul<ElementA, true>(
                    srcTensor_m,
                    srcTensor_m,
                    srcTensor_v,
                    remain_mask,
                    m_actual,
                    params);
            }
            else
            {
                AscendC::MulAddDst<ElementA, ElementA, true>(
                    srcTensor_m,
                    srcTensor_m[offset],
                    srcTensor_v[offset],
                    remain_mask,
                    m_actual,
                    params);
            }
        }

        uint64_t reduce_mask = (repeat_num == 0) ? remain : repeat_size;
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::WholeReduceSum<ElementA, true>(
            srcTensor_m,
            srcTensor_m,
            reduce_mask,
            m_actual,
            1,
            1,
            RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0);

        uint64_t add_mask = (m_actual < repeat_size) ? m_actual : repeat_size;
        params.dstRepStride = 8;
        params.src0RepStride = 8;
        params.src1RepStride = 8;

        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add<ElementA, true>(
            dstTensor,
            srcTensor_m,
            dstTensor,
            add_mask,
            CeilDiv(m_round, repeat_size),
            params);
    }
};

template <>
struct TileFaultVmad<Arch::AtlasA2,
                Gemm::GemmType<half, layout::RowMajor>,
                Gemm::GemmType<half, layout::VectorLayout>,
                Gemm::GemmType<half, layout::VectorLayout>,
                void>
{
    using ElementA = half;
    using ElementX = half;
    using ElementY = half;
    using ElementAccumulator = ElementY;

    using LayoutDst = layout::RowMajor;
    using LayoutSrc = layout::RowMajor;
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementA);

    // Mehtods

    CATLASS_DEVICE
    TileFaultVmad() {};

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

        uint32_t repeat_size = ELE_NUM_PER_C0 * 8;
        uint32_t mask = repeat_size;
        uint32_t repeat_num = n_actual / repeat_size;
        uint32_t remain = n_actual % repeat_size;

        AscendC::BinaryRepeatParams params;
        params.dstBlkStride = 1;
        params.src0BlkStride = 1;
        params.src1BlkStride = 1;
        params.dstRepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;
        params.src0RepStride = RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0;
        params.src1RepStride = 0;
        AscendC::SetMaskCount();
        AscendC::SetVectorMask<ElementA, AscendC::MaskMode::COUNTER>(m_actual * repeat_size);
        for (uint32_t i = 0; i < repeat_num; i++)
        {
            uint32_t offset = i * repeat_size;
            if (i == 0)
            {
                AscendC::Mul<ElementA, false>(
                    srcTensor_m,
                    srcTensor_m,
                    srcTensor_v,
                    AscendC::MASK_PLACEHOLDER,
                    1,
                    params);
            }
            else
            {
                AscendC::MulAddDst<ElementA, ElementA, false>(
                    srcTensor_m,
                    srcTensor_m[offset],
                    srcTensor_v[offset],
                    AscendC::MASK_PLACEHOLDER,
                    1,
                    params);
            }
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::SetMaskNorm();
        AscendC::ResetMask();

        if (remain > 0)
        {
            uint32_t offset = repeat_num * repeat_size;
            if (offset + remain > n_round)
            {
                remain = n_round - offset;
            }
            uint64_t remain_mask = remain;
            if (repeat_num == 0)
            {
                AscendC::Mul<ElementA, true>(
                    srcTensor_m,
                    srcTensor_m,
                    srcTensor_v,
                    remain_mask,
                    m_actual,
                    params);
            }
            else
            {
                AscendC::MulAddDst<ElementA, ElementA, true>(
                    srcTensor_m,
                    srcTensor_m[offset],
                    srcTensor_v[offset],
                    remain_mask,
                    m_actual,
                    params);
            }
        }

        uint64_t reduce_mask = (repeat_num == 0) ? remain : repeat_size;
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::WholeReduceSum<ElementA, true>(
            srcTensor_m,
            srcTensor_m,
            reduce_mask,
            m_actual,
            1,
            1,
            RoundUp(n_round, repeat_size) / ELE_NUM_PER_C0);

        uint64_t add_mask = (m_actual < repeat_size) ? m_actual : repeat_size;
        params.dstRepStride = 8;
        params.src0RepStride = 8;
        params.src1RepStride = 8;

        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add<ElementA, true>(
            dstTensor,
            srcTensor_m,
            dstTensor,
            add_mask,
            CeilDiv(m_round, repeat_size),
            params);
    }
};

template <
    class ElementA,
    class ElementX,
    class ElementY
>
struct TileFaultVmad<Arch::AtlasA2,
                Gemm::GemmType<ElementA, layout::ColumnMajor>,
                Gemm::GemmType<ElementX, layout::VectorLayout>,
                Gemm::GemmType<ElementY, layout::VectorLayout>,
                void>
{
    using ElementAccumulator = ElementY;
    using LayoutDst = layout::ColumnMajor;
    using LayoutSrc = layout::ColumnMajor;
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementA);

    // Mehtods

    CATLASS_DEVICE
    TileFaultVmad() {};

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
        AscendC::SetMaskCount();
        AscendC::SetVectorMask<ElementAccumulator, AscendC::MaskMode::COUNTER>(m_actual);
        AscendC::Duplicate<ElementAccumulator, false>(
            temp,
            (ElementAccumulator)0.0,
            AscendC::MASK_PLACEHOLDER,
            1,
            1,
            8);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::SetFlag<AscendC::HardEvent::V_S>((event_t)(0));
        AscendC::WaitFlag<AscendC::HardEvent::V_S>((event_t)(0));

        AscendC::UnaryRepeatParams params;
        params.dstBlkStride = 1;
        params.srcBlkStride = 1;
        params.dstRepStride = 8;
        params.srcRepStride = 4;
        for (uint32_t i = 0; i < n_actual; i++)
        {
            AscendC::Axpy<ElementAccumulator, ElementA, false>(
                temp,
                srcTensor_m[i * m_round],
                srcTensor_v.GetValue(i),
                AscendC::MASK_PLACEHOLDER,
                1,
                params);
            AscendC::PipeBarrier<PIPE_V>();
        }
        params.dstRepStride = 4;
        params.srcRepStride = 8;
        AscendC::Cast<ElementA, ElementAccumulator, false>(
            srcTensor_m,
            temp,
            AscendC::RoundMode::CAST_NONE,
            AscendC::MASK_PLACEHOLDER,
            1,
            params);
        AscendC::BinaryRepeatParams addparams;
        addparams.dstBlkStride = 1;
        addparams.src0BlkStride = 1;
        addparams.src1BlkStride = 1;
        addparams.dstRepStride = 8;
        addparams.src0RepStride = 8;
        addparams.src1RepStride = 8;
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add<ElementA, false>(
            dstTensor,
            srcTensor_m,
            dstTensor,
            AscendC::MASK_PLACEHOLDER,
            1,
            addparams);
        AscendC::SetMaskNorm();
        AscendC::ResetMask();
    }
};

template <>
struct TileFaultVmad<Arch::AtlasA2,
                Gemm::GemmType<float, layout::ColumnMajor>,
                Gemm::GemmType<float, layout::VectorLayout>,
                Gemm::GemmType<float, layout::VectorLayout>,
                void>
{
    using ElementA = float;
    using ElementX = float;
    using ElementY = float;
    using ElementAccumulator = ElementY;
    using LayoutDst = layout::ColumnMajor;
    using LayoutSrc = layout::ColumnMajor;
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementA);

    // Mehtods

    CATLASS_DEVICE
    TileFaultVmad() {};

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
        AscendC::SetFlag<AscendC::HardEvent::V_S>((event_t)(0));
        AscendC::WaitFlag<AscendC::HardEvent::V_S>((event_t)(0));
        AscendC::UnaryRepeatParams params;
        params.dstBlkStride = 1;
        params.srcBlkStride = 1;
        params.dstRepStride = 8;
        params.srcRepStride = 8;
        AscendC::SetMaskCount();
        AscendC::SetVectorMask<ElementA, AscendC::MaskMode::COUNTER>(m_actual);
        for (uint32_t i = 0; i < n_actual; i++)
        {
            AscendC::Axpy<ElementY, ElementA, false>(
                dstTensor,
                srcTensor_m[i * m_round],
                srcTensor_v.GetValue(i),
                AscendC::MASK_PLACEHOLDER,
                1,
                params);
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::SetMaskNorm();
        AscendC::ResetMask();
    }
};

template <>
struct TileFaultVmad<Arch::AtlasA2,
                Gemm::GemmType<half, layout::ColumnMajor>,
                Gemm::GemmType<half, layout::VectorLayout>,
                Gemm::GemmType<half, layout::VectorLayout>,
                void>
{
    using ElementA = half;
    using ElementX = half;
    using ElementY = half;
    using ElementAccumulator = ElementY;
    using LayoutDst = layout::ColumnMajor;
    using LayoutSrc = layout::ColumnMajor;
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementA);

    // Mehtods

    CATLASS_DEVICE
    TileFaultVmad() {};

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
        AscendC::SetFlag<AscendC::HardEvent::V_S>((event_t)(0));
        AscendC::WaitFlag<AscendC::HardEvent::V_S>((event_t)(0));
        AscendC::UnaryRepeatParams params;
        params.dstBlkStride = 1;
        params.srcBlkStride = 1;
        params.dstRepStride = 8;
        params.srcRepStride = 8;
        AscendC::SetMaskCount();
        AscendC::SetVectorMask<ElementA, AscendC::MaskMode::COUNTER>(m_actual);
        for (uint32_t i = 0; i < n_actual; i++)
        {
            AscendC::Axpy<ElementY, ElementA, false>(
                dstTensor,
                srcTensor_m[i * m_round],
                srcTensor_v.GetValue(i),
                AscendC::MASK_PLACEHOLDER,
                1,
                params);
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::SetMaskNorm();
        AscendC::ResetMask();
    }
};

template <>
struct TileFaultVmad<Arch::AtlasA2,
                Gemm::GemmType<half, layout::RowMajor>,
                Gemm::GemmType<half, layout::VectorLayout>,
                Gemm::GemmType<float, layout::VectorLayout>,
                void>
{
    using ElementA = half;
    using ElementX = half;
    using ElementY = float;

    using ElementAccumulator = ElementY;

    using LayoutDst = layout::RowMajor;
    using LayoutSrc = layout::RowMajor;
    
    // 这里是在矩阵上每个datablock 包含的element的数量
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(ElementA);

    // Mehtods

    CATLASS_DEVICE
    TileFaultVmad() {};

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

        // 这里是输出向量上每次256byte对应的元素的数量，要以此为准确定迭代次数，规模等
        uint32_t temp_repeat_size = BYTE_PER_C0 * 8 / sizeof(ElementAccumulator);
        uint32_t elem_repeat_size = ELE_NUM_PER_C0 * 8;
        uint32_t mask = temp_repeat_size;

        uint32_t repeattimes = CeilDiv(m_actual, temp_repeat_size);

        // 将输出向量赋值为0
        AscendC::Duplicate<ElementAccumulator>(
            temp,
            (ElementAccumulator)0.0,
            // 这是为了后续我们每次求全部行的局部行和，
            // 我们对每行局部和的存储空间设定为256byte对齐，
            // 这是因为每次赋值操作的 repeat 迭代内的运算能力是256byte
            temp_repeat_size, // 每个迭代内部要对256 byte 的元素全部赋值为0，
            // 我们的目标是求行和，所以我们的输出空间是行的倍数，且是datablock的倍数
            // 因此在这里，我们在赋值0时通过每次内部迭代对应一个输出行的局部和存储与accum空间
            CeilDiv(m_round * temp_repeat_size, temp_repeat_size), // 求行和
            1, // 单次赋值迭代内，矢量目的操作数不同datablock间地址步长。
            8 // 相邻赋值迭代间，矢量目的操作数相同datablock地址步长。
        ); // 总的来说，我们开辟了256 byte x m_round的空间，全部赋值为0，每256byte对应一个行的局部和存储与accum

        uint32_t repeat_num = n_actual / temp_repeat_size; // 这里是因为每次在行上做temp_repeat_size个column的元素求行和，
        // 然后每次迭代在多行上做完，再滑动到下一批列上去做

        uint32_t remain = n_actual % temp_repeat_size; // 剩余的数量，多做一次

        AscendC::PipeBarrier<PIPE_V>();
        /*
        BinaryRepeatParams为用于控制操作数地址步长的数据结构。
        结构体内包含操作数相邻迭代间相同datablock的地址步长，
        操作数同一迭代内不同datablock的地址步长等参数。
        这里我们用BinaryRepeatParams控制求多个局部行和的内层repeat 迭代的属性
        */
        AscendC::BinaryRepeatParams params;
        /*
        内层repeat 迭代中，每个迭代处理256byte的数据的乘加操作。共8个datablock，
        所以blkstride设置的每次迭代内各个datablock起始地址之间的距离，单位为datablock，
        设置方法为：
        dataBlockStride
        dataBlockStride是指同一迭代内不同datablock的地址步长。
        1. 连续计算，dataBlockStride设置为1，对同一迭代内的8个datablock数据连续进行处理。
        2. 非连续计算，dataBlockStride值大于1（如取2），
            同一迭代内不同datablock之间在读取数据时出现一个datablock的间隔。
        */
        params.dstBlkStride = 1;
        params.src0BlkStride = 1;
        params.src1BlkStride = 1;

        // 目标输出中每个内部repeat iteration之间起始地址的差异，
        // 单位为datablock（32 byte）
        // 这里我们设置每个内部repeat iteration之间的地址差异为8个datablock的空间
        // 这是因为我们每行上留了8个datablock的accum空间，
        // 每次向量乘计算每行上8个datablock元素的相乘结果，
        // 按照元素对应累加到dst的accum空间上完成局部和的运算
        params.dstRepStride = RoundUp(temp_repeat_size, temp_repeat_size) / (BYTE_PER_C0 / sizeof(ElementAccumulator));

        // 如上文所示，在我们的计算中包括内外两个repeat iteration
        // 对于内层 repeat 而言，每次每行上的256byte的元素与对应vector的元素相乘，结果累加到相应行对应的accum空间上，
        // 重复m_actual次，将每行上的局部累加结果计算完成，放到输出的目标操作数空间的对应行存储空间上
        // 对于外层iteration而言，重复在列上做滑动窗口即可，
        // 这里的params设置的是每次外层iteration内进行操作的repeat参数，即设置的是内层repeat的属性。
        // 内层 repeat 中，每个repeat在矩阵A上的stride就是一行的大小，单位为datablock
        params.src0RepStride = RoundUp(n_round, elem_repeat_size) / ELE_NUM_PER_C0;
        // 在内层repeat 中，每次循环在矩阵上换行，但是处理的列是固定的，这意味着在向量上是不滑动的，
        // 只有在外层循环之间才要滑动完成最终整体行和。
        params.src1RepStride = 0;
        AscendC::SetMaskCount();
        AscendC::SetVectorMask<ElementAccumulator, AscendC::MaskMode::COUNTER>(m_actual * temp_repeat_size);
        for(uint32_t i=0; i < repeat_num; i++)
        {
            uint32_t offset = i * temp_repeat_size;
            /*
            两个源操作数对应元素相乘后加到目标操作数相应位置上做聚合
            */
            AscendC::MulAddDst<ElementAccumulator, ElementA, false>(
                temp,
                srcTensor_m[offset], // 这里是起始地址，每次外层迭代都是从第一行的相应列偏移开始的
                srcTensor_v[offset],
                AscendC::MASK_PLACEHOLDER,
                1,
                params);
            
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::SetMaskNorm();
        AscendC::ResetMask();

        if(remain > 0)
        {
            uint32_t offset = repeat_num * temp_repeat_size;
            if (offset + remain > n_round)
            {
                remain = n_round - offset;
            }

            uint64_t remain_mask = remain;
            /*
            bool 类型模版参数：
            isSetMask：是否在接口内部设置mask。
                true，表示在接口内部设置mask。
                false，表示在接口外部设置mask，开发者需要使用SetVectorMask接口设置mask值。
                这种模式下，本接口入参中的mask值必须设置为占位符MASK_PLACEHOLDER。如上方代码所示
            */
            AscendC::MulAddDst<ElementAccumulator, ElementA, true>(
                temp,
                srcTensor_m[offset],
                srcTensor_v[offset],
                remain_mask, // 每次内层repeat 迭代内参与计算的元素的数量，单位是元素
                m_actual, // 内层 repeat 的迭代次数
                params); // 内层repeat迭代的stride属性
        }

        // 进行归约聚合的元素的数量，若进行的repeat次数多于一次，
        // 则代表每次内部repeat中计算的局部列数超过了256byte对应的元素，
        // 可以完成一次默认的repeat最大规模，反之则仅为remain个列
        uint64_t reduce_mask = (repeat_num == 0) ? remain : temp_repeat_size;
        /*
        函数功能: 在每个迭代内，将所有数据求和。
        */
        AscendC::PipeBarrier<PIPE_V>();
        /*
        isSetMask = true
        */
        AscendC::WholeReduceSum<ElementAccumulator,true>(
            temp, // 目标tensor
            temp, // src tensor
            reduce_mask, // 每次repeat 迭代内参与元素的数量，这里一般为256byte对应的列数，即每行上预留的accum空间，除非列数过少，少于256byte对应的列数
            m_actual, // repeat 迭代次数，即只考虑前面m_actual个行作为输出向量的有效值，即矩阵的行数
            1,  // 目标tensor相邻repeat迭代操作间步长，单位为目标tensor元素类型字节数（即目标tensor元素的数量）
            1, // 源操作数单次迭代内相邻datablock的步长，单位为datablock
            8); // 源操作数相邻repeat迭代间的地址步长，即源操作数每次repeat迭代跳过的datablock数目
        
        AscendC::PipeBarrier<PIPE_V>();
        // 完成数据类型的转换
        // AscendC::UnaryRepeatParams castparams;

        // castparams.dstBlkStride = 1;
        // castparams.srcBlkStride = 1;
        // castparams.dstRepStride = 4;
        // castparams.srcRepStride = 8;
        /*
        template <typename T1, typename T2, bool isSetMask = true>
        __aicore__ inline void Cast(const LocalTensor<T1>& dstLocal, 
            const LocalTensor<T2>& srcLocal, const RoundMode& round_mode, 
            const uint64_t mask, const uint8_t repeatTimes, 
            const UnaryRepeatParams& repeatParams)
        T1: 目的操作数类型
        T2：源操作数类型
        函数功能
        根据源操作数和目的操作数Tensor的数据类型进行精度转换。
        将源操作数精度转换为目的操作数的精度之后写入目的操作数

        这里是将中间聚合完成的行和向量降精度后存放在输入向量的头部。
        */

        // AscendC::Cast<ElementA, ElementAccumulator, true>(
        //     srcTensor_m, // 目的操作数
        //     temp, // 源操作数
        //     AscendC::RoundMode::CAST_NONE, // 舍入方式
        //     (uint64_t)mask, // 每次repeat iteration 内 256 byte 的输出向量元素参与计算
        //     repeattimes, // 重复 实际行数 / 256 byte 元素数量个迭代，即将整个输出向量完成
        //     castparams);

        /*
        CAST_NONE = 0,  // 在转换有精度损失时表示CAST_RINT模式，不涉及精度损失时表示不舍入
        CAST_RINT,      // rint，四舍六入五成双舍入
        CAST_RINT模式下，若待舍入部分的第一位为0，则不进位；
        若第一位为1且后续位不全为0，则进位；
        若第一位为1且后续位全为0，
        当M（即位数中保留位数的最后一位）的最后一位为0则不进位，当M的最后一位为1则进位。
        */
        // AscendC::PipeBarrier<PIPE_V>();
        // 此时temp，即中间结果已经存放到矩阵上了，且元素精度已经转化为矩阵的元素精度了
        // 这里默认其实精度会下降float -> half
        // 将结果累加到输出向量中，每个repeat iteration内参与元素的数量
        // 若实际行向量的规模小于256byte 对应的矩阵元素数量则为m_actual，否则为对应矩阵元素数量
        uint64_t add_mask = (m_actual < temp_repeat_size) ? m_actual : temp_repeat_size;

        params.dstRepStride = 8;
        params.src0RepStride = 8;
        params.src1RepStride = 8; // 这里每次输出8个datablock了，其中元素类型为ElementA，即矩阵精度
        // 将结果写出，即累加到到dstTensor完成运算
        // 此处按照ElementA的精度进行累加
        AscendC::Add<ElementY, true>(
            dstTensor, // 将 m_round 个元素，即目标行和向量累加到目标向量中即可
            temp,
            dstTensor,
            (uint64_t)add_mask, // 每次repeat iteration 的规模即 256 byte 下元素类型为ElementA的元素规模
            CeilDiv(m_round, temp_repeat_size), // 目标行向量的规模
            params);
    }
};
}

#endif //CATLASS_GEMV_TILE_TILE_FAULT_VMAD_HPP_SELF
