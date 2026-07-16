/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EPILOGUE_BLOCK_BLOCK_EPILOGUE_RESCALE_O_ARCH35_REG_HIGH_PREC
#define EPILOGUE_BLOCK_BLOCK_EPILOGUE_RESCALE_O_ARCH35_REG_HIGH_PREC

#include "../../../attn_infra/bsa_base_defs.hpp"
#include "../../../attn_infra/arch/bsa_resource.hpp"
#include "../../../attn_infra/epilogue/bsa_epilogue_dispatch_policy.hpp"
#include "../../../attn_infra/epilogue/tile_common/bsa_epilogue_tile_copy.hpp"
#include "../../../attn_infra/bsa_gemm_coord.hpp"
#include "../../../attn_infra/bsa_matrix_coord.hpp"
#include "../../../tla/tensor.hpp"
#include "../../../tla/layout.hpp"

namespace NpuArch::Epilogue::Block {

template <
    class ElementO_,
    class ElementOTmp_,
    class ElementS_,
    class ElementKV_,
    class TileCopy_,
    class OTmpSrcPos_,
    LseMode LSE_MODE_,
    LseFormat LSE_FORMAT_>
class BlockEpilogue<
    EpilogueAtlasA5BsaRescaleO<LSE_MODE_, LSE_FORMAT_>,
    ElementO_,
    ElementOTmp_,
    ElementS_,
    ElementKV_,
    TileCopy_,
    OTmpSrcPos_>
{
public:
    using DispatchPolicy = EpilogueAtlasA5BsaRescaleO<LSE_MODE_, LSE_FORMAT_>;
    using ArchTag = typename DispatchPolicy::ArchTag;

    static constexpr uint32_t RESCALE_ROW_MAX_ELEM_NUM = 64;
    static constexpr uint32_t RESCALE_COL_MAX_ELEM_NUM = 128;

    __aicore__ inline BlockEpilogue(Arch::Resource<ArchTag> &resource) {}
    __aicore__ inline ~BlockEpilogue() {}
};

}  // namespace NpuArch::Epilogue::Block

#endif  // EPILOGUE_BLOCK_BLOCK_EPILOGUE_RESCALE_O_ARCH35_REG_HIGH_PREC
