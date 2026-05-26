/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file tensor_block_mmad_mx_adapter.h
 * \brief Adapt ops-tensor BlockMmad MX specialization shape to the 9-parameter primary template used by this branch.
 */

#pragma once

#include "blaze/gemm/block/block_mmad_mx.h"
#include "tensor_block_mmad_mx_compat.h"

namespace Blaze {
namespace Gemm {
namespace Block {

template <
    uint64_t FULL_LOAD_MODE_, class AType_, class LayoutA_, class BType_, class LayoutB_, class CType_,
    class LayoutC_, class BiasType_, class LayoutBias_>
class BlockMmad<
    MatmulWithScaleMx<FULL_LOAD_MODE_>, AType_, LayoutA_, BType_, LayoutB_, CType_, LayoutC_, BiasType_, LayoutBias_>
    : public TensorBlockMmadMxCompat<
          MatmulWithScaleMx<FULL_LOAD_MODE_>, AType_, LayoutA_, BType_, LayoutB_, CType_, LayoutC_, BiasType_,
          LayoutBias_> {};

} // namespace Block
} // namespace Gemm
} // namespace Blaze
