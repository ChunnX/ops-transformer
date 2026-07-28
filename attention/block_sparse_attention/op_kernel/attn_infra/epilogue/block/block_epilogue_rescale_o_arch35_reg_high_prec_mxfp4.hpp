/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EPILOGUE_BLOCK_BLOCK_EPILOGUE_RESCALE_O_ARCH35_REG_HIGH_PREC_MXFP4
#define EPILOGUE_BLOCK_BLOCK_EPILOGUE_RESCALE_O_ARCH35_REG_HIGH_PREC_MXFP4

#include "../../../attn_infra/bsa_base_defs.hpp"
#include "../../../attn_infra/arch/bsa_resource.hpp"
#include "../../../attn_infra/epilogue/bsa_epilogue_dispatch_policy.hpp"
#include "../../../attn_infra/epilogue/tile_common/bsa_epilogue_tile_copy.hpp"
#include "../../../attn_infra/bsa_gemm_coord.hpp"
#include "../../../attn_infra/bsa_matrix_coord.hpp"
#include "../../../tla/tensor.hpp"
#include "../../../tla/layout.hpp"
#include "block_epilogue_arch35_utils.hpp"

namespace NpuArch::Epilogue::Block {
using namespace MXFP4Kernel;
template <
    class ElementO_,
    class ElementOTmp_,
    class ElementDm_,
    class TileCopy_,
    class OTmpSrcPos_,
    LseMode LSE_MODE_,
    LseFormat LSE_FORMAT_>
class BlockEpilogue<
    EpilogueAtlasA5BsaRescaleO<LSE_MODE_, LSE_FORMAT_, true>,
    ElementO_,
    ElementOTmp_,
    ElementDm_,
    TileCopy_,
    OTmpSrcPos_>
{
public:
    using DispatchPolicy = EpilogueAtlasA5BsaRescaleO<LSE_MODE_, LSE_FORMAT_, true>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementO = ElementO_;
    using ElementOTmp = ElementOTmp_;
    using ElementLse = ElementOTmp_;
    using ElementDm = ElementDm_;
    using TileCopy = TileCopy_;
    using OTmpSrcPos = OTmpSrcPos_;
    using LayoutO = typename TileCopy::LayoutO;

    using CopyUbToGmO = typename TileCopy::CopyUbToGmO;

    __aicore__ inline BlockEpilogue(Arch::Resource<ArchTag> &resource) {
        // FIXPIPE<->V 区

    }

    __aicore__ inline ~BlockEpilogue() {}

    template <class TensorO, class TensorLse>
    __aicore__ inline
    void operator()(TensorO &gOTensor,
                    TensorLse &gLseTensor,
                    GemmCoord &actualOriShape,
                    TaskInfo &taskInfo,
                    TileInfo &tileInfo)
    { 
        

    }

private:


};

}  // namespace NpuArch::Epilogue::Block

#endif  // EPILOGUE_BLOCK_BLOCK_EPILOGUE_RESCALE_O_ARCH35_REG_HIGH_PREC_MXFP4
