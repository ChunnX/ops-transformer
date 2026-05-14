/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dispatch_ffn_combine_tiling.h
 * \brief
 */

#include "moe_init_routing_quant_v2/moe_init_routing_quant_v2_tiling.h"
#include "kernel_tiling/kernel_tiling.h"

#ifndef ASCENDC_DISPATCH_FFN_COMBINE_TILING_H
#define ASCENDC_DISPATCH_FFN_COMBINE_TILING_H
// Numeric codes for the `activation` string attribute. The kernel uses the
// numeric code to select Swiglu / GeLU / etc. branches at runtime without
// having to ship the original string.
enum DispatchFFNCombineActivationCode : uint32_t {
    DISPATCH_FFN_COMBINE_ACT_NONE     = 0U,
    DISPATCH_FFN_COMBINE_ACT_SWIGLU   = 1U,
    DISPATCH_FFN_COMBINE_ACT_GELU     = 2U,
    DISPATCH_FFN_COMBINE_ACT_RELU     = 3U,
    DISPATCH_FFN_COMBINE_ACT_SILU     = 4U,
    DISPATCH_FFN_COMBINE_ACT_TANH     = 5U,
};

// Numeric codes for the `comm_alg` string attribute. The exact mapping must
// match what `dispatch_ffn_combine_tiling.cpp` writes into `info.commAlgCode`.
enum DispatchFFNCombineCommAlgCode : uint32_t {
    DISPATCH_FFN_COMBINE_COMM_DEFAULT      = 0U,  // empty string -> derived from isA2
    DISPATCH_FFN_COMBINE_COMM_ALLTOALL     = 1U,
    DISPATCH_FFN_COMBINE_COMM_BATCHWRITE   = 2U,
    DISPATCH_FFN_COMBINE_COMM_HIERARCHY    = 3U,
    DISPATCH_FFN_COMBINE_COMM_FULLMESH     = 4U,
    DISPATCH_FFN_COMBINE_COMM_PAIRWISE     = 5U,
};

struct DispatchFFNCombineInfo {
    // ---- Shape / platform ----
    uint32_t M;
    uint32_t K;
    uint32_t N;
    uint32_t expertPerRank;
    uint32_t aivNum;
    uint32_t totalUbSize;
    uint32_t topK;
    uint32_t worldSize;
    uint32_t listLen;
    bool isA2;

    // ---- Attributes (in def order, all read by tiling, all forwarded to kernel) ----
    uint32_t moeExpertNum;           // attr 0: moe_expert_num
    uint32_t epWorldSize;            // attr 1: ep_world_size
    uint32_t cclBufferSize;          // attr 2: ccl_buffer_size
    uint32_t maxRecvTokenNum;        // attr 3: max_recv_token_num
    uint32_t dispatchQuantMode;      // attr 4: dispatch_quant_mode
    int32_t  dispatchQuantOutType;   // attr 5: dispatch_quant_out_type (ge::DataType code)
    uint32_t combineQuantMode;       // attr 6: combine_quant_mode
    uint32_t commAlgCode;            // attr 7: comm_alg (encoded via DispatchFFNCombineCommAlgCode)
    uint32_t numMaxTokenPerRank;     // attr 8: num_max_token_per_rank
    uint32_t activationCode;         // attr 9: activation (encoded via DispatchFFNCombineActivationCode)
    float    activationClamp;        // attr 10: activation_clamp
    uint32_t isTransposeW1;          // attr 11: transpose_weight1
    uint32_t isTransposeW2;          // attr 12: transpose_weight2

    // ---- Optional inputs presence ----
    uint32_t hasXActiveMask;         // input 8 (x_active_mask): 1 if present
    uint32_t hasScales;              // input 9 (scales): 1 if present

    // ---- Internal routing decision ----
    // 1 -> quant routing (input X needs to be dynamically quantized before
    //      MoeInitRoutingV2, used for W4A8 / INT8 weight paths).
    // 0 -> non-quant routing (BF16 / FP16 weight path: X is forwarded as-is).
    // The host tiling decides this from the `weight1` data type (see
    // `dispatch_ffn_combine_tiling.cpp`), because `MoeInitRoutingV2` produces
    // a different gather/srcToDst layout depending on whether a per-token
    // quant pass is fused into routing.
    uint32_t isQuantRouting;
};

struct CoCTiling {
    int32_t m0 = -1;
    int32_t k0 = -1;
    int32_t n0 = -1;
    int32_t swizzleDirect = -1;
    int32_t swizzleOffset = -1;
    int32_t ubMoveNum = -1;
    int32_t pValue = -1;
    int32_t commNpuSplit = -1;
    int32_t commDataSplit = -1;
    int32_t lenPerLoop = -1;
    uint64_t initRoutingQuantTilingKey;
    // Quant routing tiling data. Always present in the struct (zero-filled
    // when the path is non-quant) so the on-device layout of
    // `DispatchFFNCombineTilingData` remains stable.
    optiling::MoeInitRoutingQuantV2TilingData moeInitRoutingQuantV2TilingData;
    // Non-quant routing tiling data. Populated when `isQuantRouting == 0`.
    // The struct shape is `MoeInitRoutingV2TilingData` (defined alongside the
    // Quant variant); both share the same Inner* compute-params layout, so
    // the kernel can read either one through the matching type alias.
    optiling::InnerMoeInitRoutingV2TilingData moeInitRoutingV2TilingData;
};

struct DispatchFFNCombineTilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling;
    DispatchFFNCombineInfo dispatchFFNCombineInfo;
    CoCTiling cocTiling;
};
#endif