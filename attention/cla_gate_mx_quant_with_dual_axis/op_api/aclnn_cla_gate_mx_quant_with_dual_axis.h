/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_ACLNN_CLA_GATE_MX_QUANT_WITH_DUAL_AXIS_H_
#define OP_API_INC_ACLNN_CLA_GATE_MX_QUANT_WITH_DUAL_AXIS_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnClaGateMxQuantWithDualAxis 的第一段接口。
 */
ACLNN_API aclnnStatus aclnnClaGateMxQuantWithDualAxisGetWorkspaceSize(
    const aclTensor* globalOut,
    const aclTensor* localOut,
    const aclTensor* globalGateLogits,
    const aclTensor* localGateLogits,
    const char* roundModeOptional,
    int64_t scaleAlg,
    int64_t dstType,
    aclTensor* rowData,
    aclTensor* rowScale,
    aclTensor* colData,
    aclTensor* colScale,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief aclnnClaGateMxQuantWithDualAxis 的第二段接口。
 */
ACLNN_API aclnnStatus aclnnClaGateMxQuantWithDualAxis(void* workspace, uint64_t workspaceSize,
                                                      aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_ACLNN_CLA_GATE_MX_QUANT_WITH_DUAL_AXIS_H_
