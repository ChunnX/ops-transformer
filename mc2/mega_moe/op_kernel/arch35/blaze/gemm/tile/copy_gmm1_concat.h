/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once

#include "tensor_api/tensor.h"
#include "blaze/gemm/utils/common_utils.h"
#include "mega_moe_qgmm/blaze/gemm/tile/arch35/copy_gm_to_l1.h"

namespace MegaMoeImpl {

// Source is a slice starting
// at gate[j], retaining the full GM strides. Its logical N is the COMPUTE width
// (gate + up); only its left half is read before jumping to up[j]. Destination
// already describes the concatenated compute tile. No global weight permutation.
template <typename Tensor>
__aicore__ inline uint64_t ConcatTensorRows(const Tensor &tensor)
{
    const auto &shape = AscendC::Std::get<0>(tensor.Layout().Shape());
    return AscendC::Std::get<0>(shape) * AscendC::Std::get<1>(shape);
}

template <typename Tensor>
__aicore__ inline uint64_t ConcatTensorColumns(const Tensor &tensor)
{
    const auto &shape = AscendC::Std::get<1>(tensor.Layout().Shape());
    return AscendC::Std::get<0>(shape) * AscendC::Std::get<1>(shape);
}

template <typename Dst, typename Src>
__aicore__ inline void CopyGmm1WeightConcatToUb(const Dst &dst, const Src &src, uint64_t halfN)
{
    using Pattern = AscendC::Te::GetLayoutPattern<typename Src::layoutType>;
    static_assert(AscendC::Std::is_same_v<Pattern, AscendC::Te::ZNLayoutPtn>,
                  "A8W4 prologue consumes packed C0=32 ZN weights");
    const uint64_t c0 = AscendC::Std::get<0>(AscendC::Std::get<0>(src.Layout().Shape()));
    const uint64_t columnBytes = c0 / 2U;
    const uint32_t blockCount = 2U * ConcatTensorRows(src) / c0;
    const uint32_t blockBytes = ConcatTensorColumns(src) / 2U * columnBytes;
    // Unlike tensor strides, these intrinsics take byte distances between block starts.
    asc_copy_gm2ub_align(reinterpret_cast<__ubuf__ uint8_t *>(dst.Data().Get()),
                         reinterpret_cast<__gm__ uint8_t *>(src.Data().Get()), blockCount, blockBytes, 0, 0, false,
                         src.Engine().GetCacheMode(), halfN * columnBytes, blockBytes);
}

template <typename Dst, typename Src>
__aicore__ inline void CopyGmm1ScaleConcatToL1(const Dst &dst, const Src &src, uint64_t halfN, uint64_t fullK)
{
    using Pattern = AscendC::Te::GetLayoutPattern<typename Src::layoutType>;
    static_assert(AscendC::Std::is_same_v<Pattern, AscendC::Te::ScaleBDNLayoutPtn>,
                  "MegaMoe GMM1 scale concat expects ScaleBDN");
    const uint64_t singleN = ConcatTensorColumns(src) / 2U;
    const uint64_t scaleSpan = ConcatTensorRows(src);
    auto left = src.Slice(AscendC::Te::MakeCoord(uint64_t(0), uint64_t(0)), AscendC::Te::MakeShape(scaleSpan, singleN));
    Blaze::Gemm::Tile::CopyConcatGM2L1::Copy(dst, left, {halfN * 2U, fullK});
}

} // namespace MegaMoeImpl
