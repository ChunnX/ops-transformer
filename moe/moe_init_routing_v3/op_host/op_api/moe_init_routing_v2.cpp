/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <tuple>
#include "moe_init_routing_v2.h"
#include "opdev/make_op_executor.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(MoeInitRoutingV2);

std::tuple<aclTensor*, aclTensor*, aclTensor*, aclTensor*> MoeInitRoutingV2(const aclTensor *x, const aclTensor *expertIdx, int64_t activeNum, 
                                                                            int64_t expertCapacity, int64_t expertNum, int64_t dropPadMode, int64_t expertTokensCountOrCumsumFlag, 
                                                                            bool expertTokensBeforeCapacityFlag, const aclTensor *expandedXOut, const aclTensor *expandedRowIdxOut, 
                                                                            const aclTensor *expertTokensCountOrCumsumOut, const aclTensor *expertTokensBeforeCapacityOut, aclOpExecutor *executor)
{
    L0_DFX(MoeInitRoutingV2, x, expertIdx, activeNum, expertCapacity, expertNum, dropPadMode, expertTokensCountOrCumsumFlag, 
           expertTokensBeforeCapacityFlag, expandedXOut, expandedRowIdxOut, expertTokensCountOrCumsumOut, expertTokensBeforeCapacityOut);
    
    auto expandedXOut_ = executor->AllocTensor(expandedXOut->GetViewShape(), expandedXOut->GetDataType(), Format::FORMAT_ND); 
    auto expandedRowIdxOut_ = executor->AllocTensor(expandedRowIdxOut->GetViewShape(), expandedRowIdxOut->GetDataType(), Format::FORMAT_ND);
    // expertTokensCountOrCumsumOut_，v3仅支持int64，v2仅支持int32，因此需要在创建tensor时直接写死int32
    auto expertTokensCountOrCumsumOut_ = executor->AllocTensor(expertTokensCountOrCumsumOut->GetViewShape(), op::DataType::DT_INT32, Format::FORMAT_ND);
    auto expertTokensBeforeCapacityOut_ = executor->AllocTensor(expertTokensCountOrCumsumOut->GetViewShape(), op::DataType::DT_INT32, Format::FORMAT_ND);

    if (expandedXOut_ == nullptr || expandedRowIdxOut_ == nullptr || expertTokensCountOrCumsumOut_ == nullptr || expertTokensBeforeCapacityOut_ == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "alloc expandedXOut or expandedRowIdxOut or expertTokensCountOrCumsumOut or expertTokensBeforeCapacityOut tensor failed.");
        return std::tuple<aclTensor*, aclTensor*, aclTensor*, aclTensor*>(nullptr, nullptr, nullptr, nullptr);
    }

    ADD_TO_LAUNCHER_LIST_AICORE(
        MoeInitRoutingV2, OP_INPUT(x, expertIdx), OP_OUTPUT(expandedXOut_, expandedRowIdxOut_, expertTokensCountOrCumsumOut_, expertTokensBeforeCapacityOut_), OP_ATTR(activeNum, expertCapacity, expertNum, dropPadMode, expertTokensCountOrCumsumFlag, expertTokensBeforeCapacityFlag));

    return std::tuple<aclTensor*, aclTensor*, aclTensor*, aclTensor*>(expandedXOut_, expandedRowIdxOut_, expertTokensCountOrCumsumOut_, expertTokensBeforeCapacityOut_); //OP_OUTPUT
}

}  // namespace l0op
