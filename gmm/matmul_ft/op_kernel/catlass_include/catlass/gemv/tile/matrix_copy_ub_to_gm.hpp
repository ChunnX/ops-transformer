#ifndef CATLASS_GEMV_TILE_TILE_MATRIX_COPY_UB_TO_GM_HPP
#define CATLASS_GEMV_TILE_TILE_MATRIX_COPY_UB_TO_GM_HPP

#include "../../catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/layout/layout.hpp"
#include "../../gemm/gemm_type.hpp"


namespace Vecself::Gemv::Tile {

using namespace Catlass;

template <
    class ArchTag,
    class GmType
>
struct MatrixCopyUBToGm {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy ub to gm for the matrices, can not find the specialization.");
};

template <typename Element>
struct MatrixCopyUBToGm<Arch::AtlasA2, Gemm::GemmType<Element, layout::RowMajor>> {
    using LayoutDst = layout::RowMajor;
    using LayoutSrc = layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);

    // Mehtods
    CATLASS_DEVICE
    MatrixCopyUBToGm() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<Element> const &dstTensor,
        AscendC::LocalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        /*
                            表4 DataCopyExtParams结构体参数定义
            参数名称                                含义
            blockCount          指定该指令包含的连续传输数据块个数，数据类型为uint16_t，
                                取值范围：blockCount∈[1, 4095]。
            
            blockLen            指定该指令每个连续传输数据块长度，该指令支持非对齐搬运，
                                每个连续传输数据块长度单位为Byte。数据类型为uint32_t，
                                取值范围：blockLen∈[1, 2097151]。

            srcStride           源操作数，相邻连续数据块的间隔（前面一个数据块的尾与后面数据块的头的间隔）。
                                如果源操作数的逻辑位置为VECIN/VECOUT，则单位为dataBlock(32Bytes)。
                                如果源操作数的逻辑位置为GM，则单位为Byte。数据类型为uint32_t，
                                srcStride不要超出该数据类型的取值范围。

            dstStride           目的操作数，相邻连续数据块间的间隔（前面一个数据块的尾与后面数据块的头的间隔）。
                                如果目的操作数的逻辑位置为VECIN/VECOUT，则单位为dataBlock(32Bytes)，
                                如果目的操作数的逻辑位置为GM，则单位为Byte。数据类型为uint32_t，
                                dstStride不要超出该数据类型的取值范围。

            rsv                 保留字段。
        */
        AscendC::DataCopyExtParams dataCopyParams(
            layoutDst.shape(0),
            layoutDst.shape(1) * sizeof(Element),
            (layoutSrc.stride(0) - layoutSrc.shape(1)) / ELE_NUM_PER_C0,
            (layoutDst.stride(0) - layoutDst.shape(1)) * sizeof(Element),
            0
        );
        AscendC::DataCopyPad(dstTensor, srcTensor, dataCopyParams);
    }
};

// new add vectorlayout version
template <typename Element>
struct MatrixCopyUBToGm<Arch::AtlasA2, Gemm::GemmType<Element, layout::VectorLayout>> {
    using LayoutSrc = layout::VectorLayout;
    using LayoutDst = layout::VectorLayout;
    
    static constexpr uint32_t ELE_NUM_PER_BLK = BYTE_PER_BLK / sizeof(Element);

    // Mehtods
    CATLASS_DEVICE
    MatrixCopyUBToGm() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<Element> const &dstTensor,
        AscendC::LocalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        AscendC::DataCopyExtParams dataCopyParams(
            1,
            layoutDst.shape(0) * sizeof(Element),
            0,
            0,
            0
        );
        AscendC::DataCopyPad(dstTensor, srcTensor, dataCopyParams);
    }
};

template <
    class ArchTag,
    class GmType
>
struct MatrixCopyUBToGmAligned {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy ub to gm for the matrices, can not find the specialization.");
};

template <typename Element>
struct MatrixCopyUBToGmAligned<Arch::AtlasA2, Gemm::GemmType<Element, layout::RowMajor>> {
    using LayoutDst = layout::RowMajor;
    using LayoutSrc = layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_BLK = BYTE_PER_BLK / sizeof(Element);
    static constexpr uint32_t BLOCK_LEN_LIMIT = 65536;
    static constexpr uint32_t MAX_REPEAT = 4095;
    static constexpr uint32_t STRIDE_LIMIT = 65536;

    CATLASS_DEVICE
    MatrixCopyUBToGmAligned() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::GlobalTensor<Element> const &dstTensor,
        AscendC::LocalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        uint32_t rows = layoutDst.shape(0);
        uint32_t cols = layoutDst.shape(1);
        uint32_t srcStride = (layoutSrc.stride(0) - layoutSrc.shape(1)) / ELE_NUM_PER_BLK;
        uint32_t dstStride = (layoutDst.stride(0) - layoutDst.shape(1)) / ELE_NUM_PER_BLK;

        if ((layoutSrc.shape(1) == layoutSrc.stride(0)) && (layoutDst.shape(1) == layoutDst.stride(0))) {
            DataCopy(dstTensor, srcTensor, rows * cols);
        } else if (srcStride < STRIDE_LIMIT && dstStride < STRIDE_LIMIT && (cols / ELE_NUM_PER_BLK) < BLOCK_LEN_LIMIT) {
            uint32_t rLoops = CeilDiv(rows, MAX_REPEAT);
            for (uint32_t i = 0; i < rLoops; ++i) {
                uint32_t rActual = (i < rLoops - 1) ? MAX_REPEAT : rows - i * MAX_REPEAT;
                AscendC::DataCopyParams dataCopyParams(
                    rActual, cols / ELE_NUM_PER_BLK, srcStride, dstStride
                );
                DataCopy(dstTensor[i * MAX_REPEAT * layoutDst.stride(0)],
                         srcTensor[i * MAX_REPEAT * layoutSrc.stride(0)], dataCopyParams);
            }
        } else {
            for (uint32_t i = 0; i < rows; ++i) {
                DataCopy(dstTensor[i * layoutDst.stride(0)], srcTensor[i * layoutSrc.stride(0)], cols);
            }
        }
    }
};
} // namespace Vecself::Gemv::Tile

#endif // CATLASS_GEMV_TILE_TILE_MATRIX_COPY_UB_TO_GM_HPP
