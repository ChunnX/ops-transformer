/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMV_BLOCK_BLOCK_GEMV_HPP
#define CATLASS_GEMV_BLOCK_BLOCK_GEMV_HPP

#include "../../catlass.hpp"
namespace Catlass::Gemv::Block {

template <
    class DispatchPolicy,
    class... Args
>
struct BlockGemv {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockGemv is not implemented for this DispatchPolicy");
};
}  // namespace Catlass::Gemv::Block

// The standalone matmul_ft package carries only the Catlass headers used by its
// self-implemented block specializations. Keep Catlass' aggregation list as
// part of the upstream source, but do not pull unavailable blocks into it.
#if 0
#include "catlass/gemv/block/block_gemv_aiv.hpp"
#include "catlass/gemv/block/block_gemv_aic.hpp"
#endif

#endif

// self implemented
#ifndef VECSELF_GEMV_BLOCK_BLOCK_GEMV_HPP
#define VECSELF_GEMV_BLOCK_BLOCK_GEMV_HPP

#include "../helper.hpp"
#include "../tile/tile_copy.hpp"
#include "../tile/tile_fault_compare.hpp"
#include "../tile/tile_fault_sum.hpp"
#include "../tile/tile_matmul_elem_add.hpp"

namespace Vecself::Gemv::Block {

using namespace Catlass;

/*
struct BlockSumMaxNoSplitK <
    Gemm::GemvAtlasA2,
    Vecself::Gemv::helper::FT_THRESHOLD_ALGORITHM::ASVAR,
    UBTileShape_,
    AType_,
    XType_,
    YType_,
    BiasType_,
    TileCopy_,
    TileFaultSum_
>
*/
template <
    class DispatchPolicy,
    Vecself::Gemv::helper::FT_THRESHOLD_ALGORITHM ALGO_TYPE,
    class... Args
>
struct BlockSumMaxNoSplitK {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockSumMaxNoSplitK is not implemented for this DispatchPolicy");
};

template <
    class DispatchPolicy,
    Vecself::Gemv::helper::FT_THRESHOLD_ALGORITHM ALGO_TYPE,
    class... Args
>
struct BlockSumMaxNoSplitKBF {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockSumMaxNoSplitKBF is not implemented for this DispatchPolicy");
};

template <
    class DispatchPolicy,
    Vecself::Gemv::helper::FT_THRESHOLD_ALGORITHM ALGO_TYPE_,
    Vecself::Gemv::helper::FT_AIV_PIPE_FUSE_TYPE FUSE_TYPE_,
    class... Args
>
struct BlockFTSumNoSplitK {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockFTSumNoSplitK is not implemented for this DispatchPolicy");
};

template <
    class DispatchPolicy,
    Vecself::Gemv::helper::FT_THRESHOLD_ALGORITHM ALGO_TYPE_,
    Vecself::Gemv::helper::FT_AIV_PIPE_FUSE_TYPE FUSE_TYPE_,
    Vecself::Gemv::helper::FT_ENC_TYPE ENC_TYPE_,
    Vecself::Gemv::helper::FT_COMP_TYPE COMP_TYPE_,
    class... Args
>
struct BlockFTGemvNoSplitK {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockFTGemvNoSplitK is not implemented for this DispatchPolicy");
};

// Vecself::Gemv::helper::FT_ABE_TYPE ABE_TYPE_,
template <
    class DispatchPolicy,
    Vecself::Gemv::helper::FT_THRESHOLD_ALGORITHM ALGO_TYPE_,
    Vecself::Gemv::helper::FT_AIV_PIPE_FUSE_TYPE FUSE_TYPE_,
    Vecself::Gemv::helper::FT_ENC_TYPE ENC_TYPE_,
    Vecself::Gemv::helper::FT_COMP_TYPE COMP_TYPE_,
    Vecself::Gemv::helper::FT_ABE_TYPE ABE_TYPE_,
    class... Args
>
struct BlockFTGemvCENoSplitK {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockFTGemvCENoSplitK is not implemented for this DispatchPolicy");
};

template <
    class DispatchPolicy,
    Vecself::Gemv::helper::FT_THRESHOLD_ALGORITHM ALGO_TYPE_,
    Vecself::Gemv::helper::FT_AIV_PIPE_FUSE_TYPE FUSE_TYPE_,
    Vecself::Gemv::helper::FT_ENC_TYPE ENC_TYPE_,
    Vecself::Gemv::helper::FT_COMP_TYPE COMP_TYPE_,
    Vecself::Gemv::helper::FT_ABE_TYPE ABE_TYPE_,
    class... Args
>
struct BlockFTGemvCENoSplitKPreload {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockFTGemvCENoSplitKPreload is not implemented for this DispatchPolicy");
};

template <
    class DispatchPolicy,
    Vecself::Gemv::helper::FT_THRESHOLD_ALGORITHM ALGO_TYPE_,
    Vecself::Gemv::helper::FT_AIV_PIPE_FUSE_TYPE FUSE_TYPE_,
    Vecself::Gemv::helper::FT_ENC_TYPE ENC_TYPE_,
    Vecself::Gemv::helper::FT_COMP_TYPE COMP_TYPE_,
    Vecself::Gemv::helper::FT_ABE_TYPE ABE_TYPE_,
    class... Args
>
struct BlockFTGemvCENoSplitKPreloadFI {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockFTGemvCENoSplitKPreloadFI is not implemented for this DispatchPolicy");
};

template <
    class DispatchPolicy,
    class... Args
>
struct BlockSumGemv {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockSumGemv is not implemented for this DispatchPolicy");
};

template <
    class DispatchPolicy,
    Vecself::Gemv::helper::FT_COMP_TYPE COMP_TYPE_,
    class... Args
>
struct BlockCompare {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockCompare is not implemented for this DispatchPolicy");
};

template <
    class DispatchPolicy,
    Vecself::Gemv::helper::FT_ENC_TYPE ENC_TYPE_,
    class... Args
>
struct BlockThresholdCalc {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockThresholdCalc is not implemented for this DispatchPolicy");
};

template <
    class DispatchPolicy,
    Vecself::Gemv::helper::FT_THRESHOLD_ALGORITHM ALGO_TYPE_,
    Vecself::Gemv::helper::FT_AIV_PIPE_FUSE_TYPE FUSE_TYPE_,
    Vecself::Gemv::helper::FT_ENC_TYPE ENC_TYPE_,
    Vecself::Gemv::helper::FT_COMP_TYPE COMP_TYPE_,
    class... Args
>
struct BlockThresholdCalcFused {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockThresholdCalc is not implemented for this DispatchPolicy");
};

template <
    class DispatchPolicy,
    class... Args
>
struct BlockSliceSum {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockSliceSum is not implemented for this DispatchPolicy");
};

template <
    class DispatchPolicy,
    Vecself::Gemv::helper::FT_ENC_TYPE ENC_TYPE_,
    class... Args
>
struct BlockFTGemv {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockFTGemv is not implemented for this DispatchPolicy");
};

template <
    class DispatchPolicy,
    Vecself::Gemv::helper::FT_ENC_TYPE ENC_TYPE_,
    class... Args
>
struct BlockFTGemvDouble {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockFTGemv is not implemented for this DispatchPolicy");
};


template <
    class DispatchPolicy,
    Vecself::Gemv::helper::FT_AIC_BE_SCHEME BE_SCHEME_,
    class... Args
>
struct BlockFTGemvBe {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockFTGemv is not implemented for this DispatchPolicy");
};

template <
    class DispatchPolicy,
    class... Args
>
struct BlockSumGemvPingPong {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockSumGemvPingPong is not implemented for this DispatchPolicy");
};

template <
    class DispatchPolicy,
    Vecself::Gemv::helper::FT_ENC_TYPE ENC_TYPE_,
    bool  VECTORIZED_TRANS_,
    class... Args
>
struct BlockFTGemvVectorized {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockFTGemvVectorized is not implemented for this DispatchPolicy");
};

template <
    class DispatchPolicy,
    class... Args
>
struct BlockMatrixAdd{
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockMatrixAdd is not implemented for this DispatchPolicy");
};

template <
    class DispatchPolicy,
    class... Args
>
struct BlockMatrixTranspose{
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockMatrixTranspose is not implemented for this DispatchPolicy");
};

template <
    class DispatchPolicy,
    class... Args
>
struct BlockMatrixAddVectorized{
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockMatrixAddVectorized is not implemented for this DispatchPolicy");
};

template <
    class DispatchPolicy,
    Vecself::Gemv::helper::FT_AIV_PIPE_FUSE_TYPE FUSE_TYPE_,
    class... Args
>
struct BlockSliceKMNSum {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockSliceKMNSum is not implemented for this DispatchPolicy");
};

template <
    class DispatchPolicy,
    Vecself::Gemv::helper::FT_AIV_PIPE_FUSE_TYPE FUSE_TYPE_,
    class... Args
>
struct BlockSliceKMNSumVectorized {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockSliceKMNSumVectorized is not implemented for this DispatchPolicy");
};

}  // namespace Vecself::Gemv::Block

#include "block_sum_gemv_aiv.hpp"
#include "block_gemv_aic_FT_BE_row_complete.hpp"
#include "block_b_a_red_no_splitk_robust.hpp"
#include "block_b_a_red_no_splitk_robust_bf.hpp"
#include "block_slicekmn_reduce_sum.hpp"
#include "block_slicekmn_reduce_sum_fused.hpp"
#include "block_slicekmn_reduce_sum_fused_robust.hpp"
#include "block_gemv_aic_FT_BE_row_complete_bf.hpp"
#include "block_sum_aiv_ceft_no_splitk_thre_robust_preload.hpp"

#endif  // VECSELF_GEMV_BLOCK_BLOCK_GEMV_HPP
