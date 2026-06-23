/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */
#ifndef SASA_PREFILL_ARCH35_KERNEL_UTILS
#define SASA_PREFILL_ARCH35_KERNEL_UTILS

#include "../attn_infra/base_defs.hpp"
#include "../attn_infra/arch/arch.hpp"
#include "../attn_infra/layout/layout.hpp"

#include "../attn_infra/gemm/block/block_mmad.hpp"
#include "../attn_infra/gemm/dispatch_policy.hpp"
#include "../attn_infra/gemm/gemm_type.hpp"

#include "../attn_infra/arch/cross_core_sync.hpp"
#include "../attn_infra/arch/resource.hpp"
#include "../attn_infra/epilogue/block/block_epilogue.hpp"
#include "../attn_infra/epilogue/dispatch_policy.hpp"
#include "../tla/tensor.hpp"
#include "../tla/layout.hpp"
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "kernel_tiling/kernel_tiling.h"

namespace SparseAttnPrefill {

__aicore__ inline
uint32_t CeilDiv(uint32_t a, uint32_t b)
{
    return (a + b - 1) / b;
}

__aicore__ inline
uint32_t RoundUp(uint32_t a, uint32_t b)
{
    return CeilDiv(a, b) * b;
}

}  // namespace SparseAttnPrefill

#endif  // SASA_PREFILL_ARCH35_KERNEL_UTILS
