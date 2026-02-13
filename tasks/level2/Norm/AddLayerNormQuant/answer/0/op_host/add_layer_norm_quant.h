/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 */
#ifndef OP_API_INC_LEVEL0_ADD_LAYER_NORM_QUANT_H_
#define OP_API_INC_LEVEL0_ADD_LAYER_NORM_QUANT_H_

#include "opdev/op_executor.h"

namespace l0op {
constexpr size_t ADD_LAYER_NORM_QUANT_OUT_NUM = 5;

const std::array<aclTensor*, ADD_LAYER_NORM_QUANT_OUT_NUM> AddLayerNormQuant(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* gamma, const aclTensor* beta,
    const aclTensor* biasOptional, const aclTensor* scales1Optional, const aclTensor* scales2Optional,
    const aclTensor* zeroPoints1Optional, const aclTensor* zeroPoints2Optional, const char* quantMode, double epsilon,
    bool additionalOutput, bool divMode, aclOpExecutor* executor);
} // namespace l0op

#endif // OP_API_INC_LEVEL0_ADD_LAYER_NORM_QUANT_H_
