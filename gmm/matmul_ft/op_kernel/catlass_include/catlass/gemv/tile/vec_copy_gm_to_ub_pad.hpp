/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMV_TILE_VEC_COPY_GM_TO_UB_PADDING_HPP
#define CATLASS_GEMV_TILE_VEC_COPY_GM_TO_UB_PADDING_HPP

#include "catlass/catlass.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/gemm/gemm_type.hpp"

// constexpr uint32_t STRIDE_LIMIT = 65536;

namespace FTSelf::Gemv::Tile {

using namespace Catlass;

template <
    class ArchTag,
    class VType,
    FTSelf::Gemv::helper::VEC_PADDING_TYPE PaddingType
>
struct VecCopyGmToUBPadding
{
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy gm to UB with the simplining strategy, can not find the specialization.");
};

template <
    class ArchTag_,
    class VType_
>
struct VecCopyGmToUBPadding<
    ArchTag_,
    VType_,
    FTSelf::Gemv::helper::VEC_PADDING_TYPE::ALIGNED
> {
    using Element = typename VType_::Element;
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);

    // Mehtods

    CATLASS_DEVICE
    VecCopyGmToUBPadding() {};

    CATLASS_DEVICE
    void operator()(
    AscendC::LocalTensor<Element> dstTensor,
    AscendC::GlobalTensor<Element> srcTensor,
    uint32_t len
    ) {
        AscendC::DataCopyParams params;
        params.blockCount = 1;
        params.blockLen = CeilDiv(len, ELE_NUM_PER_C0);
        params.srcStride = 0;
        params.dstStride = 0;

        AscendC::DataCopy(
            dstTensor,
            srcTensor,
            params);
    }
};

template <
    class ArchTag_,
    class VType_
>
struct VecCopyGmToUBPadding<
    ArchTag_,
    VType_,
    FTSelf::Gemv::helper::VEC_PADDING_TYPE::PADDING
> {
    using Element = typename VType_::Element;
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);

    // Mehtods

    CATLASS_DEVICE
    VecCopyGmToUBPadding() {};

    CATLASS_DEVICE
    void operator()(
    AscendC::LocalTensor<Element> dstTensor,
    AscendC::GlobalTensor<Element> srcTensor,
    uint32_t len
    ) {
        /*
                表3 DataCopyExtParams结构体参数定义
        参数名称                        含义
        blockCount      指定该指令包含的连续传输数据块个数，数据类型为uint16_t，
                        取值范围：blockCount∈[1, 4095]。

        blockLen        指定该指令每个连续传输数据块长度，该指令支持非对齐搬运，
                        每个连续传输数据块长度单位为Byte。数据类型为uint32_t，
                        取值范围：blockLen∈[1, 2097151]。

        srcStride       源操作数，相邻连续数据块的间隔（前面一个数据块的尾与后面数据块的头的间隔），
                        如果源操作数的逻辑位置为VECIN/VECOUT，则单位为dataBlock(32Bytes)。
                        如果源操作数的逻辑位置为GM，则单位为Byte。
                        数据类型为uint32_t，srcStride不要超出该数据类型的取值范围。

        dstStride       目的操作数，相邻连续数据块间的间隔（前面一个数据块的尾与后面数据块的头的间隔），
                        如果目的操作数的逻辑位置为VECIN/VECOUT，则单位为dataBlock(32Bytes)，
                        如果目的操作数的逻辑位置为GM，则单位为Byte。
                        数据类型为uint32_t，dstStride不要超出该数据类型的取值范围。

        rsv             保留字段。
        */
        AscendC::DataCopyExtParams dataCopyParams(1, len * sizeof(Element), 0, 0, 0);

        /*
                表5 DataCopyPadExtParams结构体参数定义
        参数名称                        含义
        isPad               是否需要填充用户自定义的数据，取值范围：true，false。
                            true：填充padding value。
                            false：表示用户不需要指定填充值，会默认填充随机值。

        leftPadding         连续搬运数据块左侧需要补充的数据范围，单位为元素个数。
                            leftPadding、rightPadding的字节数均不能超过32Bytes。

        rightPadding        连续搬运数据块右侧需要补充的数据范围，单位为元素个数。
                            leftPadding、rightPadding的字节数均不能超过32Bytes。

        paddingValue        左右两侧需要填充的数据值，需要保证在数据占用字节范围内。
                            数据类型和源操作数保持一致，T数据类型。
                            当数据类型长度为64位时，该参数只能设置为0。
        */
        AscendC::DataCopyPadExtParams<Element> padParams(false, 0, 0, 0);
        // AscendC::DataCopy(
        //     dstTensor,
        //     srcTensor,
        //     params);
        AscendC::DataCopyPad(dstTensor, srcTensor, dataCopyParams, padParams);
    }

    // CATLASS_DEVICE
    // void Gm2Ub(AscendC::LocalTensor<ElementAccumulator> const &dst,
    //     AscendC::GlobalTensor<ElementAccumulator> const &src,
    //     uint32_t dataNum)
    // {
    //     AscendC::DataCopyExtParams dataCopyParams(1, dataNum * sizeof(ElementAccumulator), 0, 0, 0);
    //     AscendC::DataCopyPadExtParams<ElementAccumulator> padParams(false, 0, 0, 0);
    //     AscendC::DataCopyPad(dst, src, dataCopyParams, padParams);
    // }
};


} // namespace FTSelf::Gemv::Tile

#endif // CATLASS_GEMV_TILE_VEC_COPY_GM_TO_UB_HPP
