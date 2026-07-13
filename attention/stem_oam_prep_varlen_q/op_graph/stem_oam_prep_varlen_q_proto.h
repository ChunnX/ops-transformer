/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file stem_oam_prep_varlen_q_proto.h
 * \brief stem_oam_prep_varlen_q operator prototype definition
 */

#ifndef OPS_BUILT_IN_OP_PROTO_INC_STEM_OAM_PREP_VARLEN_Q_PROTO_H_
#define OPS_BUILT_IN_OP_PROTO_INC_STEM_OAM_PREP_VARLEN_Q_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {

/**
 * @brief Stem OAM Q-side preprocessing for block-sparse attention. \n
 *
 * Converts a variable-length paged Q tensor into a flattened qFlat output grouped by stem blocks,
 * for downstream OAM score computation. The pipeline is:
 * Scale Fusion -> De-page Varlen -> Weighted Group Sum (natural order, no flip) -> Flatten -> Cast to output dtype.
 *
 * @par Inputs:
 * @li q: Variable-length Q tensor (all batch tokens concatenated). Shape [total_tokens, H_q, D].
 *        Last dim D must equal stemBlockSize (default 128).
 *        Supported data types: DT_FLOAT8_E4M3FN, DT_BF16, DT_FLOAT16.
 * @li qSeqLens: Per-batch Q sequence lengths. Shape [batch]. Data type: DT_INT32.
 * @li cuSeqLensQ: Cumulative sequence-length offsets for varlen indexing. Shape [batch + 1]. Data type: DT_INT32.
 *                cuSeqLensQ[0] must be 0; cuSeqLensQ[batch] must equal total_tokens; monotonically increasing.
 * @li qScale: Optional. Per-token scale factor for Q. Shape [total_tokens, H_q]. Data type: DT_FLOAT.
 *             Required when q is FP8; pass nullptr when q is BF16 or FP16.
 *
 * @par Attributes:
 * @li stemBlockSize: Optional. Stem block size (B). Default: 128. Must be a multiple of 32; max 256.
 * @li stemStride: Optional. Stem stride (S). Default: 16. Must be a multiple of 16; max 64; must satisfy stemStride <= stemBlockSize.
 *
 * @par Outputs:
 * @li qFlat: Flattened Q output for OAM score computation. Shape [batch, H_q, max_Qb, kflat_dim].
 *            Where max_Qb = ceil(max(qSeqLens) / stemBlockSize), kflat_dim = stemStride * D.
 *            Data type: DT_BF16 (or DT_FLOAT16 if input q is FP16).
 *
 * @attention Constraints:
 * @code{.c}
 * - q last dim D must equal 128 (hardcoded DIM_QK).
 * - stemBlockSize % 32 == 0, stemBlockSize <= 256.
 * - stemStride % 16 == 0, stemStride <= 64, stemStride <= stemBlockSize.
 * - cuSeqLensQ length must be batch + 1, monotonically increasing, starting at 0.
 * - When q is FP8, qScale is mandatory; when q is BF16/FP16, qScale must be null.
 * - Batch max = 1024.
 * - Q-side stride dimension uses natural order (g in [0, S-1]), unlike the K-side which flips.
 * - When qSeqLens[b] == 0, the corresponding qFlat output is zero-filled.
 * @endcode
 */

REG_OP(StemOamPrepVarlenQ)
    .INPUT(q, TensorType({DT_FLOAT8_E4M3FN, DT_BF16, DT_FLOAT16}))
    .INPUT(qSeqLens, TensorType({DT_INT32}))
    .INPUT(cuSeqLensQ, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(qScale, TensorType({DT_FLOAT}))
    .ATTR(stemBlockSize, Int, 128)
    .ATTR(stemStride, Int, 16)
    .OUTPUT(qFlat, TensorType({DT_BF16, DT_FLOAT16}))
    .OP_END_FACTORY_REG(StemOamPrepVarlenQ)

} // namespace ge

#endif // OPS_BUILT_IN_OP_PROTO_INC_STEM_OAM_PREP_VARLEN_Q_PROTO_H_
