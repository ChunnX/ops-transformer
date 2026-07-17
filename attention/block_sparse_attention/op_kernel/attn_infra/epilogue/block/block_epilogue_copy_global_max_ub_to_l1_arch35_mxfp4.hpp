/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EPILOGUE_BLOCK_BLOCK_EPILOGUE_COPY_GLOBAL_MAX_UB_TO_L1_ARCH35_MXFP4
#define EPILOGUE_BLOCK_BLOCK_EPILOGUE_COPY_GLOBAL_MAX_UB_TO_L1_ARCH35_MXFP4

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
    class ElementLocalGlobalMax_>
class BlockEpilogue<
    EpilogueCopyGlobalMaxUbToL1Bsa,
    ElementLocalGlobalMax_>
{
public:
    using DispatchPolicy = EpilogueCopyGlobalMaxUbToL1Bsa;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementLocalGlobalMax = ElementLocalGlobalMax_;

    __aicore__ inline BlockEpilogue(Arch::Resource<ArchTag> &resource) {}
    __aicore__ inline ~BlockEpilogue() {}
};

}  // namespace NpuArch::Epilogue::Block

#endif  // EPILOGUE_BLOCK_BLOCK_EPILOGUE_COPY_GLOBAL_MAX_UB_TO_L1_ARCH35_MXFP4
