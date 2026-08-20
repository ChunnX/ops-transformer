/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the LICENSE file in the root of the software repository for the full text of the License.
 */

/*!
 * \file stem_oam_prep_paged_kv_proto.h
 * \brief stem_oam_prep_paged_kv operator prototype definition
 */

#ifndef OPS_BUILT_IN_OP_PROTO_INC_STEM_OAM_PREP_PAGED_KV_PROTO_H_
#define OPS_BUILT_IN_OP_PROTO_INC_STEM_OAM_PREP_PAGED_KV_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {

/**
 * @brief Stem OAM K/V-side preprocessing for block-sparse attention. \n
 *
 * Extracts K/V data from Paged KV Cache, applies K Processing (per-token K-scale × group sum
 * + anti-diagonal flip) and V Processing (per-head vScale × L2 Norm → Log → Global Normalize
 * → ReLU → Block Average), outputs kFlat and vBias for OAM score computation.
 *
 * @par Inputs:
 * @li kCache: Paged K cache. Shape [total_blocks, H_kv, kv_block_size, 128] (BNBD layout).
 *             Data type: DT_FLOAT8_E4M3FN.
 * @li vCache: Paged V cache. Shape same as kCache. Data type: DT_FLOAT8_E4M3FN.
 * @li kvIndices: Per-batch KV block index array. Shape [batch, max_kv_blocks]. Data type: DT_INT32.
 * @li kvSeqLens: Per-batch KV sequence lengths. Shape [batch]. Data type: DT_INT32.
 *                Value depend: optional (tiling reads actual values to compute max_Kb).
 * @li kScaleCache: Optional. Per-token per-head K scale. Shape [total_blocks, H_kv, kv_block_size, 1].
 *                  Data type: DT_FLOAT. Required when kCache is FLOAT8_E4M3FN.
 * @li vScale: Optional. Per-head V scale. Shape [H_kv]. Data type: DT_FLOAT.
 *             Required when vCache is FLOAT8_E4M3FN.
 *
 * @par Attributes:
 * @li lambdaMag: Optional. V bias multiplier, range [0, 1]. Default 0.3.
 * @li kvLayout: Optional. KV cache layout, "BNBD" or "BBND". Default "BNBD".
 * @li stemBlockSize: Optional. Stem block size, must be %32==0 and ≤256. Default 128.
 * @li stemStride: Optional. Stride size, must be %16==0, ≤64, ≤stemBlockSize. Default 16.
 *
 * @par Outputs:
 * @li kFlat: K group sum + anti-diag flip result. Shape [batch, H_kv, max_Kb, kflat_dim].
 *            Where kflat_dim = stemStride × 128. Data type: DT_BF16.
 * @li vBias: V block bias result. Shape [batch, H_kv, max_Kb]. Data type: DT_FLOAT.
 *
 * @attention Constraints:
 * @code{.c}
 * - kCache/vCache data type must be DT_FLOAT8_E4M3FN.
 * - kScaleCache/vScale data type must be DT_FLOAT, mandatory when kCache is FP8.
 * - kv_block_size from kCache shape: {64, 128}.
 * - max_kv_blocks max 2048; kv sequence length max 262144.
 * - stemBlockSize must be divisible by stemStride.
 * - When kvSeqLens[b] == 0, the corresponding kFlat/vBias output is zero-filled.
 * - Only arch35 (Ascend 950PR/Ascend 950DT) supported.
 * @endcode
 */

REG_OP(StemOamPrepPagedKv)
    .INPUT(kCache, TensorType({DT_FLOAT8_E4M3FN}))
    .INPUT(vCache, TensorType({DT_FLOAT8_E4M3FN}))
    .INPUT(kvIndices, TensorType({DT_INT32}))
    .INPUT(kvSeqLens, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(kScaleCache, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(vScale, TensorType({DT_FLOAT}))
    .ATTR(lambdaMag, Float, 0.3)
    .ATTR(kvLayout, String, "BNBD")
    .ATTR(stemBlockSize, Int, 128)
    .ATTR(stemStride, Int, 16)
    .OUTPUT(kFlat, TensorType({DT_BF16}))
    .OUTPUT(vBias, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(StemOamPrepPagedKv)

} // namespace ge

#endif // OPS_BUILT_IN_OP_PROTO_INC_STEM_OAM_PREP_PAGED_KV_PROTO_H_
