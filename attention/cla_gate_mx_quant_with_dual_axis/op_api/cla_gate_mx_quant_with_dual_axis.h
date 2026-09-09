/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef PTA_NPU_OP_API_INC_LEVEL0_OP_CLA_GATE_MX_QUANT_WITH_DUAL_AXIS_H_
#define PTA_NPU_OP_API_INC_LEVEL0_OP_CLA_GATE_MX_QUANT_WITH_DUAL_AXIS_H_

#include <tuple>
#include "opdev/op_executor.h"

namespace l0op {

std::tuple<aclTensor *, aclTensor *, aclTensor *, aclTensor *> ClaGateMxQuantWithDualAxis(
    const aclTensor *globalOut, const aclTensor *localOut, const aclTensor *globalGateLogits,
    const aclTensor *localGateLogits, const char *roundModeOptional, int64_t scaleAlg, int64_t dstType,
    int64_t quantMode, aclOpExecutor *executor);

} // namespace l0op

#endif // PTA_NPU_OP_API_INC_LEVEL0_OP_CLA_GATE_MX_QUANT_WITH_DUAL_AXIS_H_
