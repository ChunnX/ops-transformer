/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "infershape_context_faker.h"

namespace gert {

InferShapeContextFaker& InferShapeContextFaker::SetOpType(const std::string opType)
{
    OpInferShapeContextBuilder::MutableOpInfo().OpType(opType.c_str()).OpName(opType.c_str());
    return *this;
}

InferShapeContextFaker& InferShapeContextFaker::NodeIoNum(size_t inputNum, size_t outputNum)
{
    OpInferShapeContextBuilder::MutableOpInfo().IONum(inputNum, outputNum);
    return *this;
}

InferShapeContextFaker& InferShapeContextFaker::IrInstanceNum(const std::vector<uint32_t>& inputInstanceNum,
                                                              const std::vector<uint32_t>& outputInstanceNum)
{
    OpInferShapeContextBuilder::MutableOpInfo().IOInstanceNum(inputInstanceNum, outputInstanceNum);
    return *this;
}

InferShapeContextFaker& InferShapeContextFaker::NodeInputTd(int32_t index, ge::DataType dtype, ge::Format originFormat,
                                                            ge::Format storageFormat)
{
    OpInferShapeContextBuilder::MutableOpInfo().SetInputTd(index, dtype, originFormat, storageFormat);
    return *this;
}

InferShapeContextFaker& InferShapeContextFaker::NodeOutputTd(int32_t index, ge::DataType dtype, ge::Format originFormat,
                                                             ge::Format storageFormat)
{
    OpInferShapeContextBuilder::MutableOpInfo().SetOutputTd(index, dtype, originFormat, storageFormat);
    return *this;
}

InferShapeContextFaker& InferShapeContextFaker::InputTensors(const std::vector<Tensor *>& inputTensors)
{
    OpInferShapeContextBuilder::InputTensors(inputTensors);
    return *this;
}

InferShapeContextFaker& InferShapeContextFaker::OutputShapes(const std::vector<StorageShape *>& outputShapes)
{
    OpInferShapeContextBuilder::OutputShapes(outputShapes);
    return *this;
}

ContextHolder<InferShapeContext> InferShapeContextFaker::Build()
{
    return OpInferShapeContextBuilder::Build();
}

} // namespace gert
