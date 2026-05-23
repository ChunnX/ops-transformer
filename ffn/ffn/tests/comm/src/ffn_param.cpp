/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file ffn_param.cpp
 * \brief FFN 参数信息.
 */

#include "ffn_param.h"

using namespace ops::adv::tests::ffn;

namespace {
std::string NormalizeTensorName(const std::string &name)
{
    if (name == "expertTokens") {
        return "expert_tokens";
    }
    if (name == "deqScale1") {
        return "deq_scale1";
    }
    if (name == "deqScale2") {
        return "deq_scale2";
    }
    return name;
}

Tensor CloneTensorWithName(const Tensor &tensor, const std::string &name)
{
    if (name == tensor.Name()) {
        return tensor;
    }
    return Tensor(name.c_str(), tensor.ShapeView(), tensor.ShapeType().c_str(), tensor.GetDataType(), tensor.GetFormat(),
                  tensor.GetTensorType());
}
} // namespace

Param::Param(std::vector<Tensor> inputs, std::vector<int64_t> expertTokensData, std::string activation,
             int32_t innerPrecise, int32_t outputDtype, bool tokensIndexFlag)
    : mExpertTokensData(std::move(expertTokensData)), mActivation(activation), mInnerPrecise(innerPrecise),
      mOutputDtype(outputDtype), mTokensIndexFlag(tokensIndexFlag)
{
    for (auto &tensor : inputs) {
        auto normalizedName = NormalizeTensorName(tensor.Name());
        mTensors[normalizedName] = CloneTensorWithName(tensor, normalizedName);
    }
}