/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EPILOGUE_BLOCK_BLOCK_EPILOGUE_ONLINE_SOFTMAX_ARCH35_REG_LOW_PREC_FP16_MXFP4_HPP
#define EPILOGUE_BLOCK_BLOCK_EPILOGUE_ONLINE_SOFTMAX_ARCH35_REG_LOW_PREC_FP16_MXFP4_HPP

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
    MXQuantMode MX_QUANT_MODE_,
    class OutputType_,
    class LayoutS_,
    class ScaleType_>
class BlockEpilogue<
    EpilogueOnlineSoftmaxBsa<true, MX_QUANT_MODE_>,
    OutputType_,
    Gemm::GemmType<half, LayoutS_>,
    ScaleType_>
{
public:
    using DispatchPolicy = EpilogueOnlineSoftmaxBsa<true, MX_QUANT_MODE_>;
    static constexpr MXQuantMode mxQuantMode = DispatchPolicy::MX_QUANT_MODE;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementOutput = typename OutputType_::Element;
    using ElementInput = half;
    using LayoutOutput = typename OutputType_::Layout;
    using LayoutInput = LayoutS_;

    static constexpr uint32_t SM_ROW_MAX_ELEM_NUM = 64;
    static constexpr uint32_t SM_COL_MAX_ELEM_NUM = 256;

    __aicore__ inline BlockEpilogue(Arch::Resource<ArchTag> &resource, float scaleValue) {}
    __aicore__ inline ~BlockEpilogue() {}
};

}  // namespace NpuArch::Epilogue::Block

#endif  // EPILOGUE_BLOCK_BLOCK_EPILOGUE_ONLINE_SOFTMAX_ARCH35_REG_LOW_PREC_FP16_MXFP4_HPP
