/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LEVEL0_MIXED_QUANT_SPARSE_FLASH_MLA_H_
#define OP_API_INC_LEVEL0_MIXED_QUANT_SPARSE_FLASH_MLA_H_

#include <array>
#include "opdev/op_executor.h"

namespace l0op {

/**
 * @brief MixedQuantSparseFlashMla level-0 operator.
 *        Encapsulates the low-level scheduling of MixedQuantSparseFlashMla, completing InferShape and Kernel Launch
 * registration. This interface is internal and only for use by the aclnn layer.
 *
 * @param q                         query tensor (BF16)
 * @param oriKvOptional             optional original KV tensor (FLOAT8_E4M3FN)
 * @param cmpKvOptional             optional compressed KV tensor (FLOAT8_E4M3FN)
 * @param oriSparseIndicesOptional  optional original KV sparse indices (INT32)
 * @param cmpSparseIndicesOptional  optional compressed KV sparse indices (INT32)
 * @param oriBlockTableOptional     optional original KV block table (INT32)
 * @param cmpBlockTableOptional     optional compressed KV block table (INT32)
 * @param cuSeqlensQOptional        optional query cumulative sequence lengths (INT32)
 * @param cuSeqlensOriKvOptional    optional original KV cumulative sequence lengths (INT32)
 * @param cuSeqlensCmpKvOptional    optional compressed KV cumulative sequence lengths (INT32)
 * @param sequsedQOptional          optional actual query sequence lengths (INT32)
 * @param sequsedOriKvOptional      optional actual original KV sequence lengths (INT32)
 * @param sequsedCmpKvOptional      optional actual compressed KV sequence lengths (INT32)
 * @param cmpResidualKvOptional     optional compressed KV residual lengths (INT32)
 * @param oriTopkLengthOptional     optional original KV top-k lengths (INT32)
 * @param cmpTopkLengthOptional     optional compressed KV top-k lengths (INT32)
 * @param sinksOptional             optional attention sink weights (FLOAT32)
 * @param metadataOptional          optional pre-computed tiling metadata (INT32)
 * @param quantMode                 quantization mode
 * @param ropeHeadDim               RoPE head dimension
 * @param softmaxScale              softmax scaling factor
 * @param cmpRatio                  compressed KV ratio
 * @param oriMaskMode               original KV mask mode
 * @param cmpMaskMode               compressed KV mask mode
 * @param oriWinLeft                original KV left window size
 * @param oriWinRight               original KV right window size
 * @param layoutQOptional           query layout string
 * @param layoutKvOptional          KV layout string
 * @param topkValueMode             top-k value mode
 * @param returnSoftmaxLse          whether to output softmax LSE
 * @param executor                  op executor
 * @return std::array<const aclTensor*, 2> [attnOut, softmaxLse]
 *         Any element being nullptr indicates InferShape or Launch failure for that output.
 */
const std::array<const aclTensor *, 2> MixedQuantSparseFlashMla(
    const aclTensor *q, const aclTensor *oriKvOptional, const aclTensor *cmpKvOptional,
    const aclTensor *oriSparseIndicesOptional, const aclTensor *cmpSparseIndicesOptional,
    const aclTensor *oriBlockTableOptional, const aclTensor *cmpBlockTableOptional, const aclTensor *cuSeqlensQOptional,
    const aclTensor *cuSeqlensOriKvOptional, const aclTensor *cuSeqlensCmpKvOptional, const aclTensor *sequsedQOptional,
    const aclTensor *sequsedOriKvOptional, const aclTensor *sequsedCmpKvOptional,
    const aclTensor *cmpResidualKvOptional, const aclTensor *oriTopkLengthOptional,
    const aclTensor *cmpTopkLengthOptional, const aclTensor *sinksOptional, const aclTensor *metadataOptional,
    int64_t quantMode, int64_t ropeHeadDim, double softmaxScale, int64_t cmpRatio, int64_t oriMaskMode,
    int64_t cmpMaskMode, int64_t oriWinLeft, int64_t oriWinRight, const char *layoutQOptional,
    const char *layoutKvOptional, int64_t topkValueMode, bool returnSoftmaxLse, aclOpExecutor *executor);

} // namespace l0op

#endif // OP_API_INC_LEVEL0_MIXED_QUANT_SPARSE_FLASH_MLA_H_
