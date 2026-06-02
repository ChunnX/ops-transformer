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
 * \file dispatch_ffn_combine.h (host-side config)
 * \brief Per-op input/output/attribute index configuration, mirroring the
 *        layout used by `mc2/mega_moe/op_host/op_tiling/mega_moe.h`.
 *        Centralizing these indices in a struct lets the tiling code refer
 *        to inputs/attrs symbolically and keeps the indices trivially
 *        editable when the op def changes.
 *
 *        Indices match the order declared in
 *        `mc2/dispatch_ffn_combine_unified/op_host/dispatch_ffn_combine_def.cpp`.
 */

#ifndef OP_HOST_OP_TILING_DISPATCH_FFN_COMBINE_H
#define OP_HOST_OP_TILING_DISPATCH_FFN_COMBINE_H

#include <cstdint>

namespace optiling {

struct DispatchFFNCombineConfig {
    // ----------------------------------------------------------------
    // Input indices (matching def order):
    //   0: context        (REQUIRED, INT32 ND)
    //   1: x              (REQUIRED, FP16/BF16 ND)
    //   2: topk_ids       (REQUIRED, INT32 ND)
    //   3: topk_weights   (REQUIRED, FP32 ND)
    //   4: weight1        (DYNAMIC, INT4/INT32/FP16/BF16/INT8 ND/NZ)
    //   5: weight2        (DYNAMIC, same as weight1)
    //   6: weight_scales1 (DYNAMIC, INT64 ND)
    //   7: weight_scales2 (DYNAMIC, INT64 ND)
    //   8: bias1          (OPTIONAL, FP32 ND)
    //   9: bias2          (OPTIONAL, FP32 ND)
    //  10: x_active_mask  (OPTIONAL, INT8 ND)
    //  11: scales         (OPTIONAL, FP32 ND)
    // ----------------------------------------------------------------
    uint32_t contextIndex            = 0U;
    uint32_t xIndex                  = 1U;
    uint32_t topkIdsIndex            = 2U;
    uint32_t topkWeightsIndex        = 3U;
    uint32_t weightOneIndex          = 4U;
    uint32_t weightTwoIndex          = 5U;
    uint32_t weightScalesOneIndex    = 6U;
    uint32_t weightScalesTwoIndex    = 7U;
    uint32_t biasOneIndex            = 8U;
    uint32_t biasTwoIndex            = 9U;
    uint32_t xActiveMaskIndex        = 10U;
    uint32_t scalesIndex             = 11U;

    // ----------------------------------------------------------------
    // Output indices (matching def order):
    //   0: y                (REQUIRED, FP16/BF16 ND)
    //   1: expert_token_nums(REQUIRED, INT32 ND)
    // ----------------------------------------------------------------
    uint32_t yIndex                  = 0U;
    uint32_t expertTokenNumsIndex    = 1U;

    // ----------------------------------------------------------------
    // Attribute indices (matching def order):
    //   0: moe_expert_num            (REQUIRED Int)
    //   1: ep_world_size             (REQUIRED Int)
    //   2: ccl_buffer_size           (REQUIRED Int)
    //   3: max_recv_token_num        (OPTIONAL Int, default 0)
    //   4: dispatch_quant_mode       (OPTIONAL Int, default 0)
    //   5: dispatch_quant_out_type   (OPTIONAL Int, default DT_UNDEFINED)
    //   6: combine_quant_mode        (OPTIONAL Int, default 0)
    //   7: comm_alg                  (OPTIONAL String, default "")
    //   8: num_max_token_per_rank    (OPTIONAL Int, default 0)
    //   9: activation                (OPTIONAL String, default "")
    //  10: activation_clamp          (OPTIONAL Float, default 0.0f)
    //  11: gmm2_x_dtype              (OPTIONAL Int, default DT_UNDEFINED)
    //  12: transpose_weight1         (OPTIONAL Bool, default false)
    //  13: transpose_weight2         (OPTIONAL Bool, default false)
    //  14: weight1_interleave        (OPTIONAL Int, default 0)
    // ----------------------------------------------------------------
    uint32_t attrMoeExpertNumIndex          = 0U;
    uint32_t attrEpWorldSizeIndex           = 1U;
    uint32_t attrCclBufferSizeIndex         = 2U;
    uint32_t attrMaxRecvTokenNumIndex       = 3U;
    uint32_t attrDispatchQuantModeIndex     = 4U;
    uint32_t attrDispatchQuantOutTypeIndex  = 5U;
    uint32_t attrCombineQuantModeIndex      = 6U;
    uint32_t attrCommAlgIndex               = 7U;
    uint32_t attrNumMaxTokenPerRankIndex    = 8U;
    uint32_t attrActivationIndex            = 9U;
    uint32_t attrActivationClampIndex       = 10U;
    uint32_t attrActivationOutDtypeIndex      = 11U;
    uint32_t attrTransposeWeight1Index      = 12U;
    uint32_t attrTransposeWeight2Index      = 13U;
    uint32_t attrWeight1InterleaveIndex     = 14U;
};

}  // namespace optiling

#endif  // OP_HOST_OP_TILING_DISPATCH_FFN_COMBINE_H
