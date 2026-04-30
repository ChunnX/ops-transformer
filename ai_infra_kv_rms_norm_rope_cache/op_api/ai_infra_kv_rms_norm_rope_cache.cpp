/**
 * Copyright (c) 2023-2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(AiInfraKvRmsNormRopeCache);

bool AiInfraKvRmsNormRopeCache(
    const aclTensor* kv, const aclTensor* gamma, const aclTensor* cos, const aclTensor* sin,
    const aclTensor* index, aclTensor* k_cache, aclTensor* ckv_cache,
    const aclTensor* k_rope_scale, const aclTensor* c_kv_scale,
    const aclTensor* k_rope_offset, const aclTensor* c_kv_offset, const aclTensor* v, 
    double epsilon, char* cache_mode, char* rotary_mode, char* quant_mode, bool is_output_kv,
    const aclTensor* k_rope_out, const aclTensor* c_kv_out,
    aclOpExecutor *executor)
{
    // L0接口时延统计以及入参打印
    L0_DFX(AiInfraKvRmsNormRopeCache, kv, gamma, cos, sin, index, k_cache, ckv_cache, 
            k_rope_scale, c_kv_scale, k_rope_offset, c_kv_offset, v, 
            epsilon, cache_mode, rotary_mode, quant_mode, is_output_kv);

    // 构造输出张量


    // 构造输入列表处理可选输入
    std::vector<const aclTensor*> inputs = {kv, gamma, cos, sin, index, k_cache, ckv_cache};
    if (k_rope_scale != nullptr) {
        inputs.push_back(k_rope_scale);
    } else {
        inputs.push_back(executor->AllocTensor(DataType::DT_FLOAT, Format::FORMAT_ND, Format::FORMAT_ND));
    }
    if (c_kv_scale != nullptr) {
        inputs.push_back(c_kv_scale);
    } else {
        inputs.push_back(executor->AllocTensor(DataType::DT_FLOAT, Format::FORMAT_ND, Format::FORMAT_ND));
    }
    if (k_rope_offset != nullptr) {
        inputs.push_back(k_rope_offset);
    } else {
        inputs.push_back(executor->AllocTensor(DataType::DT_FLOAT, Format::FORMAT_ND, Format::FORMAT_ND));
    }
    if (c_kv_offset != nullptr) {
        inputs.push_back(c_kv_offset);
    } else {
        inputs.push_back(executor->AllocTensor(DataType::DT_FLOAT, Format::FORMAT_ND, Format::FORMAT_ND));
    }
    if (v != nullptr) {
        inputs.push_back(v);
    } else {
        inputs.push_back(executor->AllocTensor(kv->GetDataType(), Format::FORMAT_ND, Format::FORMAT_ND));
    }



    // 发起aicore任务，调用ADD_TO_LAUNCHER_LIST_AICORE接口找到kernel
    ADD_TO_LAUNCHER_LIST_AICORE(AiInfraKvRmsNormRopeCache,
                                OP_INPUT(kv, gamma, cos, sin, index, k_cache, ckv_cache,
                                        k_rope_scale, c_kv_scale, k_rope_offset, c_kv_offset, v),
                                //OP_OUTPUT(k_cache_out, ckv_cache_out, k_rope_out, c_kv_out),
                                OP_OUTPUT(k_cache, ckv_cache, k_rope_out, c_kv_out),
                                OP_ATTR(static_cast<float>(epsilon), cache_mode, rotary_mode, quant_mode, is_output_kv));
    
    return true;
}

} // namespace l0op
