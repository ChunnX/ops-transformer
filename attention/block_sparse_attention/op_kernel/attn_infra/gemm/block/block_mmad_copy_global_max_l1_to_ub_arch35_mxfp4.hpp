/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GEMM_BLOCK_COPY_GLOBAL_MAX_L1_TO_UB_ARCH35_MXFP4_HPP
#define GEMM_BLOCK_COPY_GLOBAL_MAX_L1_TO_UB_ARCH35_MXFP4_HPP

#include "../../../attn_infra/bsa_base_defs.hpp"
#include "../../../attn_infra/arch/bsa_resource.hpp"
#include "../../../attn_infra/arch/bsa_cross_core_sync.hpp"
#include "../../../attn_infra/bsa_coord.hpp"
#include "../../../attn_infra/gemm/bsa_gemm_dispatch_policy.hpp"
#include "../../../attn_infra/gemm/bsa_helper.hpp"
#include "../../../attn_infra/bsa_gemm_coord.hpp"
#include "../../../attn_infra/gemm/tile_common/bsa_gemm_tile_copy.hpp"
#include "../../../attn_infra/gemm/tile_common/bsa_tile_mmad.hpp"
#include "../../../tla/layout.hpp"
#include "../../../tla/tensor.hpp"

namespace NpuArch::Gemm::Block {

template <
    class ElementLocalGlobalMax_>
struct BlockMmadTla<
    CopyGlobalMaxL1ToUBBsa,
    void,
    void,
    ElementLocalGlobalMax_,
    void,
    void,
    void,
    void,
    void>
{
public:
    using DispatchPolicy = CopyGlobalMaxL1ToUBBsa;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementLocalGlobalMax = ElementLocalGlobalMax_;

    static constexpr uint32_t L0_STAGES = DispatchPolicy::L0_STAGES;

    __aicore__ inline BlockMmadTla(Arch::Resource<ArchTag> &resource) {}
    __aicore__ inline ~BlockMmadTla() {}
};

}  // namespace NpuArch::Gemm::Block

#endif  // GEMM_BLOCK_COPY_GLOBAL_MAX_L1_TO_UB_ARCH35_MXFP4_HPP
