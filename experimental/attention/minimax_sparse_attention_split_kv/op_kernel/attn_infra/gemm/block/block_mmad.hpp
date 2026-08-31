/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BLOCK_MMAD_HPP
#define BLOCK_MMAD_HPP

#if (__CCE_AICORE__ == 220)
#include "../../../attn_infra/base_defs.hpp"
#include "../../../attn_infra/detail/dependent_false.hpp"
#include "../../../attn_infra/gemm/dispatch_policy.hpp"

namespace NpuArch::Gemm::Block {

template <class DispatchPolicy, class... Args>
struct BlockMmad {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "Could not find a BlockMmad specialization");
};

}  // namespace NpuArch::Gemm::Block

#include "../../../attn_infra/gemm/block/block_mmad_qk.hpp"
#include "../../../attn_infra/gemm/block/block_mmad_pv.hpp"
#include "../../../attn_infra/gemm/block/block_mmad_qk_prefill_a2.hpp"
#include "../../../attn_infra/gemm/block/block_mmad_pv_prefill_a2.hpp"
#endif

#if (__CCE_AICORE__ == 310)
#include "../../../attn_infra/gemm/block/block_mmad_qk_split_kv_arch35.hpp"
#include "../../../attn_infra/gemm/block/block_mmad_pv_split_kv_arch35.hpp"
#endif

#endif // GEMM_BLOCK_BLOCK_MMAD_HPP
