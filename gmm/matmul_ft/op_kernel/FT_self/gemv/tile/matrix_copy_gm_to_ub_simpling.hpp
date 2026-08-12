/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMV_TILE_MATRIX_COPY_GM_TO_UB_HPP_CONTINUE_SIMPLING
#define CATLASS_GEMV_TILE_MATRIX_COPY_GM_TO_UB_HPP_CONTINUE_SIMPLING

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/gemm/gemm_type.hpp"

namespace FTSelf::Gemv::Tile {

using namespace Catlass;

template <
    class ArchTag,
    class GmType,
    FTSelf::Gemv::helper::MATRIX_SIMPLING_TYPE SimplingType
>
struct MatrixCopyGmToUBSimpling
{
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy gm to UB with the simplining strategy, can not find the specialization.");
};

/// Partial specialization for AtlasA2, RowMajor in and RowMajor out.
/// Matrix A confirm
template <class Element>
struct MatrixCopyGmToUBSimpling<
    Arch::AtlasA2,
    Catlass::Gemm::GemmType<Element, layout::RowMajor>,
    FTSelf::Gemv::helper::MATRIX_SIMPLING_TYPE::CONTINUOUS_SIMPLING>
{
    using LayoutDst = layout::RowMajor;
    using LayoutSrc = layout::RowMajor;

    using MATRIX_SIMPLING_TYPE = FTSelf::Gemv::helper::MATRIX_SIMPLING_TYPE;

    static constexpr MATRIX_SIMPLING_TYPE SimplingType = MATRIX_SIMPLING_TYPE::CONTINUOUS_SIMPLING;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);

    // Mehtods

    CATLASS_DEVICE
    MatrixCopyGmToUBSimpling() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> dstTensor,
        AscendC::GlobalTensor<Element> srcTensor,
        LayoutDst const &layoutDst,
        LayoutSrc const &layoutSrc, uint32_t simpling_stride)
    {
        /*
        simpling stride 为采样的间隔，连续模式下则为在每次读取数据时，取前(1/simpling_stride)的数据来进行读取。
        */
        uint32_t m_actual = layoutSrc.shape(0);
        uint32_t n_actual = layoutSrc.shape(1);

        uint32_t m_round = layoutDst.shape(0);
        uint32_t n_round = layoutDst.shape(1);
        uint32_t stride = layoutSrc.stride(0);

        uint32_t n_actual_simpled = (n_actual < n_round) ? n_actual : (n_actual  / simpling_stride);

        /*
            表7 DataCopyParams结构体参数定义

        参数名称                    含义
        blockCount          指定该指令包含的连续传输数据块个数，
                            取值范围：blockCount∈[1, 4095]。

        blockLen            指定该指令每个连续传输数据块长度，单位为datablock(32B)。取值范围：blockLen∈[1, 65535]。
                            特别的，当dstLocal位于C2PIPE2GM时，单位为128B；
                            当dstLocal位于C2时，表示源操作数的连续传输数据块长度，单位为64B。

        srcStride           源操作数，相邻连续数据块的间隔（前面一个数据块的尾与后面数据块的头的间隔），
                            单位为datablock(32B)。数据类型为uint16_t，
                            srcStride不要超出该数据类型的取值范围。

        dstStride           目的操作数，相邻连续数据块间的间隔（前面一个数据块的尾与后面数据块的头的间隔），
                            单位为datablock(32B)。数据类型为uint16_t，
                            dstStride不要超出该数据类型的取值范围。
                            特别的，当dstLocal位于C2PIPE2GM时，单位为128B；
                            当dstLocal位于C2时，单位为64B。
        */

        AscendC::DataCopyParams params;
        if ((n_actual_simpled % ELE_NUM_PER_C0 == 0) && (stride % ELE_NUM_PER_C0 == 0) &&
            (stride < Catlass::STRIDE_LIMIT))
        {
            params.blockCount = m_actual;
            params.blockLen = CeilDiv(n_actual_simpled, ELE_NUM_PER_C0);
            params.srcStride = (stride - n_actual) / ELE_NUM_PER_C0;
            params.dstStride = (n_round - n_actual_simpled) / ELE_NUM_PER_C0;
            AscendC::DataCopy(
                dstTensor,
                srcTensor,
                params);
        }
        else if ((n_actual_simpled % ELE_NUM_PER_C0 == 0) &&
                 (stride * ELE_NUM_PER_C0 < Catlass::STRIDE_LIMIT))
        {
            uint32_t counts = m_actual / ELE_NUM_PER_C0;
            uint32_t remain = m_actual % ELE_NUM_PER_C0;
            if (counts > 0)
            {
                params.blockCount = counts;
                params.blockLen = CeilDiv(n_actual_simpled, ELE_NUM_PER_C0);
                params.srcStride = (ELE_NUM_PER_C0 * stride - n_actual) / ELE_NUM_PER_C0;
                params.dstStride = (ELE_NUM_PER_C0 * n_round - n_actual_simpled) / ELE_NUM_PER_C0;
                for (uint32_t i = 0; i < ELE_NUM_PER_C0; i++)
                {
                    AscendC::DataCopy(
                        dstTensor[i * n_round],
                        srcTensor[i * stride],
                        params);
                }
            }
            if (remain > 0)
            {
                params.blockCount = 1;
                params.blockLen = CeilDiv(n_actual_simpled, ELE_NUM_PER_C0);
                params.srcStride = 0;
                params.dstStride = 0;
                for (uint32_t i = 0; i < remain; i++)
                {
                    AscendC::DataCopy(
                        dstTensor[counts * n_round * ELE_NUM_PER_C0 + i * n_round],
                        srcTensor[counts * stride * ELE_NUM_PER_C0 + i * stride],
                        params);
                }
            }
        }
        else
        {
            params.blockCount = 1;
            params.blockLen = CeilDiv(n_actual_simpled, ELE_NUM_PER_C0);
            params.srcStride = 0;
            params.dstStride = 0;
            for (uint32_t i = 0; i < m_actual; i++)
            {
                AscendC::DataCopy(
                    dstTensor[i * n_round],
                    srcTensor[i * stride],
                    params);
            }
        }
    }
};

/// Partial specialization for AtlasA2, RowMajor in and RowMajor out.
/// Matrix A confirm
template <class Element>
struct MatrixCopyGmToUBSimpling<
    Arch::AtlasA2,
    Catlass::Gemm::GemmType<Element, layout::RowMajor>,
    FTSelf::Gemv::helper::MATRIX_SIMPLING_TYPE::STRIDED_SIMPLING>
{
    using LayoutDst = layout::RowMajor;
    using LayoutSrc = layout::RowMajor;

    using MATRIX_SIMPLING_TYPE = FTSelf::Gemv::helper::MATRIX_SIMPLING_TYPE;

    static constexpr MATRIX_SIMPLING_TYPE SimplingType = MATRIX_SIMPLING_TYPE::CONTINUOUS_SIMPLING;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);

    // Mehtods

    CATLASS_DEVICE
    MatrixCopyGmToUBSimpling() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> dstTensor,
        AscendC::GlobalTensor<Element> srcTensor,
        LayoutDst const &layoutDst,
        LayoutSrc const &layoutSrc, uint32_t simpling_stride, uint32_t stride_unit)
    {
        /*
        simpling stride 为采样的间隔，即每隔多少个stride单元采集一次数据
        stride_unit: 单位为元素，即采样时以多少个连续元素为单位
        */

        uint32_t stride_unit_aligned = RoundUp(stride_unit, ELE_NUM_PER_C0);
        uint32_t stride_chunk_aligned = simpling_stride * stride_unit_aligned;
        uint32_t stride_chunk_align_blk = stride_chunk_aligned / ELE_NUM_PER_C0;
        uint32_t stride_unit_align_blk = stride_unit_aligned / ELE_NUM_PER_C0;

        uint32_t m_actual = layoutSrc.shape(0);
        uint32_t n_actual = layoutSrc.shape(1);

        uint32_t n_total_blk = CeilDiv(n_actual, ELE_NUM_PER_C0);
        uint32_t n_chunk_num = CeilDiv(n_total_blk, stride_chunk_align_blk);

        uint32_t m_round = layoutDst.shape(0);
        uint32_t n_round = layoutDst.shape(1);
        uint32_t stride = layoutSrc.stride(0);

        /*
            表7 DataCopyParams结构体参数定义

        参数名称                    含义
        blockCount          指定该指令包含的连续传输数据块个数，
                            取值范围：blockCount∈[1, 4095]。

        blockLen            指定该指令每个连续传输数据块长度，单位为datablock(32B)。取值范围：blockLen∈[1, 65535]。
                            特别的，当dstLocal位于C2PIPE2GM时，单位为128B；
                            当dstLocal位于C2时，表示源操作数的连续传输数据块长度，单位为64B。

        srcStride           源操作数，相邻连续数据块的间隔（前面一个数据块的尾与后面数据块的头的间隔），
                            单位为datablock(32B)。数据类型为uint16_t，
                            srcStride不要超出该数据类型的取值范围。

        dstStride           目的操作数，相邻连续数据块间的间隔（前面一个数据块的尾与后面数据块的头的间隔），
                            单位为datablock(32B)。数据类型为uint16_t，
                            dstStride不要超出该数据类型的取值范围。
                            特别的，当dstLocal位于C2PIPE2GM时，单位为128B；
                            当dstLocal位于C2时，单位为64B。
        */

        AscendC::DataCopyParams params;
        AscendC::DataCopyParams final_params;
        if ((n_actual % ELE_NUM_PER_C0 == 0) && (stride % ELE_NUM_PER_C0 == 0) &&
            (stride < Catlass::STRIDE_LIMIT))
        {
            params.blockCount = m_actual;
            params.blockLen = stride_unit_align_blk;
            // CeilDiv(n_actual_simpled, ELE_NUM_PER_C0);
            params.srcStride = (stride - stride_chunk_aligned) / ELE_NUM_PER_C0;
            params.dstStride = (n_round - stride_unit_aligned) / ELE_NUM_PER_C0;



            for(int32_t j=0;j<(n_chunk_num-1);j++){
                AscendC::DataCopy(
                dstTensor[j*stride_unit_aligned],
                srcTensor[j*stride_chunk_aligned],
                params);
            }

            uint32_t final_chunk_blk_len = n_total_blk - (n_chunk_num - 1)*stride_chunk_align_blk;
            uint32_t final_unit_blk_len = (final_chunk_blk_len < stride_unit_align_blk) ? final_chunk_blk_len : stride_unit_align_blk;
            uint32_t final_chunk_size = final_chunk_blk_len * ELE_NUM_PER_C0;
            uint32_t final_unit_size = final_unit_blk_len * ELE_NUM_PER_C0;

            final_params.blockCount = m_actual;
            final_params.blockLen = final_unit_blk_len;
            final_params.srcStride = (stride - final_chunk_size) / ELE_NUM_PER_C0;
            final_params.dstStride = (n_round - final_unit_size) / ELE_NUM_PER_C0;

            AscendC::DataCopy(
               dstTensor[(n_chunk_num - 1) * stride_unit_aligned],
               srcTensor[(n_chunk_num - 1) * stride_chunk_aligned],
               final_params);
        }
        else if ((n_actual % ELE_NUM_PER_C0 == 0) &&
                 (stride * ELE_NUM_PER_C0 < Catlass::STRIDE_LIMIT))
        {
            uint32_t counts = m_actual / ELE_NUM_PER_C0;
            uint32_t remain = m_actual % ELE_NUM_PER_C0;
            if (counts > 0)
            {
                params.blockCount = counts;
                params.blockLen = stride_unit_align_blk;
                params.srcStride = (ELE_NUM_PER_C0 * stride - stride_chunk_aligned) / ELE_NUM_PER_C0;
                params.dstStride = (ELE_NUM_PER_C0 * n_round - stride_unit_aligned) / ELE_NUM_PER_C0;

                uint32_t final_chunk_blk_len = n_total_blk - (n_chunk_num - 1)*stride_chunk_align_blk;
                uint32_t final_unit_blk_len = (final_chunk_blk_len < stride_unit_align_blk) ? final_chunk_blk_len : stride_unit_align_blk;
                uint32_t final_chunk_size = final_chunk_blk_len * ELE_NUM_PER_C0;
                uint32_t final_unit_size = final_unit_blk_len * ELE_NUM_PER_C0;

                final_params.blockCount = counts;
                final_params.blockLen = final_unit_blk_len;
                final_params.srcStride = (ELE_NUM_PER_C0 * stride - final_chunk_size) / ELE_NUM_PER_C0;
                final_params.dstStride = (ELE_NUM_PER_C0 * n_round - final_unit_size) / ELE_NUM_PER_C0;


                for (uint32_t i = 0; i < ELE_NUM_PER_C0; i++)
                {
                    for(int32_t j=0; j<(n_chunk_num-1); j++){
                        AscendC::DataCopy(
                        dstTensor[i * n_round + j * stride_unit_aligned],
                        srcTensor[i * stride + j * stride_chunk_aligned],
                        params);
                    }

                    AscendC::DataCopy(
                        dstTensor[i * n_round + (n_chunk_num - 1) * stride_unit_aligned],
                        srcTensor[i * stride + (n_chunk_num - 1) * stride_chunk_aligned],
                        final_params
                    );
                }
            }
            if (remain > 0)
            {
                uint32_t final_chunk_blk_len = n_total_blk - (n_chunk_num - 1)*stride_chunk_align_blk;
                uint32_t final_unit_blk_len = (final_chunk_blk_len < stride_unit_align_blk) ? final_chunk_blk_len : stride_unit_align_blk;
                uint32_t final_chunk_size = final_chunk_blk_len * ELE_NUM_PER_C0;
                uint32_t final_unit_size = final_unit_blk_len * ELE_NUM_PER_C0;

                if(n_chunk_num > 1){
                    params.blockCount = (n_chunk_num-1);
                    params.blockLen = stride_unit_align_blk;
                    // CeilDiv(n_actual_simpled, ELE_NUM_PER_C0);
                    params.srcStride = stride_chunk_align_blk - stride_unit_align_blk;
                    params.dstStride = 0;
                }else{
                    params.blockCount = 1;
                    params.blockLen = final_unit_blk_len;
                    // CeilDiv(n_actual_simpled, ELE_NUM_PER_C0);
                    params.srcStride = 0;
                    params.dstStride = 0;
                }

                final_params.blockCount = 1;
                final_params.blockLen = final_unit_blk_len;
                final_params.srcStride = 0;
                final_params.dstStride = 0;

                for (uint32_t i = 0; i < remain; i++)
                {
                    if(n_chunk_num > 1){
                        AscendC::DataCopy(
                            dstTensor[counts * n_round * ELE_NUM_PER_C0 + i * n_round],
                            srcTensor[counts * stride * ELE_NUM_PER_C0 + i * stride],
                            params);
                    }

                    AscendC::DataCopy(
                        dstTensor[counts * n_round * ELE_NUM_PER_C0 + i * n_round + (n_chunk_num - 1) * stride_unit_aligned],
                        srcTensor[counts * stride * ELE_NUM_PER_C0 + i * stride  + (n_chunk_num - 1) * stride_chunk_aligned],
                        final_params
                    );
                }
            }
        }
        else
        {
            uint32_t final_chunk_blk_len = n_total_blk - (n_chunk_num - 1)*stride_chunk_align_blk;
            uint32_t final_unit_blk_len = (final_chunk_blk_len < stride_unit_align_blk) ? final_chunk_blk_len : stride_unit_align_blk;
            uint32_t final_chunk_size = final_chunk_blk_len * ELE_NUM_PER_C0;
            uint32_t final_unit_size = final_unit_blk_len * ELE_NUM_PER_C0;

            if(n_chunk_num > 1){
                params.blockCount = (n_chunk_num-1);
                params.blockLen = stride_unit_align_blk;
                // CeilDiv(n_actual_simpled, ELE_NUM_PER_C0);
                params.srcStride = stride_chunk_align_blk - stride_unit_align_blk;
                params.dstStride = 0;
            }else{
                params.blockCount = 1;
                params.blockLen = final_unit_blk_len;
                // CeilDiv(n_actual_simpled, ELE_NUM_PER_C0);
                params.srcStride = 0;
                params.dstStride = 0;
            }

            final_params.blockCount = 1;
            final_params.blockLen = final_unit_blk_len;
            final_params.srcStride = 0;
            final_params.dstStride = 0;

            for (uint32_t i = 0; i < m_actual; i++)
            {
                AscendC::DataCopy(
                    dstTensor[i * n_round],
                    srcTensor[i * stride],
                    params);

                if(n_chunk_num > 1){
                    AscendC::DataCopy(
                        dstTensor[i * n_round],
                        srcTensor[i * stride],
                        params
                    );
                }

                AscendC::DataCopy(
                    dstTensor[i * n_round + (n_chunk_num - 1) * stride_unit_aligned],
                    srcTensor[i * stride  + (n_chunk_num - 1) * stride_chunk_aligned],
                    final_params
                );
            }
        }
    }
};

/// Partial specialization for AtlasA2, ColumnMajor in and ColumnMajor out.
/// Matrix A confirm
template <class Element>
struct MatrixCopyGmToUBSimpling<
    Arch::AtlasA2,
    Catlass::Gemm::GemmType<Element, layout::ColumnMajor>,
    FTSelf::Gemv::helper::MATRIX_SIMPLING_TYPE::CONTINUOUS_SIMPLING
    >
{
    using LayoutDst = layout::ColumnMajor;
    using LayoutSrc = layout::ColumnMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);

    using MATRIX_SIMPLING_TYPE = FTSelf::Gemv::helper::MATRIX_SIMPLING_TYPE;

    static constexpr MATRIX_SIMPLING_TYPE SimplingType = MATRIX_SIMPLING_TYPE::CONTINUOUS_SIMPLING;

    // Mehtods

    CATLASS_DEVICE
    MatrixCopyGmToUBSimpling() {};

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc,
        uint32_t simpling_stride)
    {
        uint32_t m_actual = layoutSrc.shape(0);
        uint32_t m_actual_simpled = m_actual / simpling_stride;
        uint32_t n_actual = layoutSrc.shape(1);
        uint32_t m_round = layoutDst.shape(0);
        uint32_t n_round = layoutDst.shape(1);
        uint32_t stride = layoutSrc.stride(1);

        AscendC::DataCopyParams params;
        if ((m_actual_simpled % ELE_NUM_PER_C0 == 0) && (stride % ELE_NUM_PER_C0 == 0) &&
            (stride < Catlass::STRIDE_LIMIT))
        {
            params.blockCount = n_actual;
            params.blockLen = CeilDiv(m_actual_simpled, ELE_NUM_PER_C0);
            params.srcStride = (stride - m_actual) / ELE_NUM_PER_C0;
            params.dstStride = (m_round - m_actual_simpled) / ELE_NUM_PER_C0;
            AscendC::DataCopy(
                dstTensor,
                srcTensor,
                params);
        }
        else if ((m_actual_simpled % ELE_NUM_PER_C0 == 0) &&
                 (stride * ELE_NUM_PER_C0 < Catlass::STRIDE_LIMIT))
        {
            uint32_t counts = n_actual / ELE_NUM_PER_C0;
            uint32_t remain = n_actual % ELE_NUM_PER_C0;
            if (counts > 0)
            {
                params.blockCount = counts;
                params.blockLen = CeilDiv(m_actual_simpled, ELE_NUM_PER_C0);
                params.srcStride = (ELE_NUM_PER_C0 * stride - m_actual) / ELE_NUM_PER_C0;
                params.dstStride = (ELE_NUM_PER_C0 * m_round - m_actual_simpled) / ELE_NUM_PER_C0;
                for (uint32_t i = 0; i < ELE_NUM_PER_C0; i++)
                {
                    AscendC::DataCopy(
                        dstTensor[i * m_round],
                        srcTensor[i * stride],
                        params);
                }
            }
            if (remain > 0)
            {
                params.blockCount = 1;
                params.blockLen = CeilDiv(m_actual_simpled, ELE_NUM_PER_C0);
                params.srcStride = 0;
                params.dstStride = 0;
                for (uint32_t i = 0; i < remain; i++)
                {
                    AscendC::DataCopy(
                        dstTensor[counts * m_round * ELE_NUM_PER_C0 + i * m_round],
                        srcTensor[counts * stride * ELE_NUM_PER_C0 + i * stride],
                        params);
                }
            }
        }
        else
        {
            params.blockCount = 1;
            params.blockLen = CeilDiv(m_actual_simpled, ELE_NUM_PER_C0);
            params.srcStride = 0;
            params.dstStride = 0;
            for (uint32_t i = 0; i < n_actual; i++)
            {
                AscendC::DataCopy(
                    dstTensor[i * m_round],
                    srcTensor[i * stride],
                    params);
            }
        }
    }
};

} // namespace FTSelf::Gemv::Tile

#endif // CATLASS_GEMV_TILE_MATRIX_COPY_GM_TO_UB_HPP
