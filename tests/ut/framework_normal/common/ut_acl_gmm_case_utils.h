/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TESTS_UT_FRAMEWORK_NORMAL_COMMON_UT_ACL_GMM_CASE_UTILS_H
#define TESTS_UT_FRAMEWORK_NORMAL_COMMON_UT_ACL_GMM_CASE_UTILS_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "op_api_ut_common/tensor_desc.h"
#include "ut_acl_parse_utils.h"
#include "ut_parse_utils.h"

namespace ops::ut {

struct AclTensorSpec {
    std::vector<int64_t> viewDims;
    std::vector<int64_t> stride;
    std::vector<int64_t> storageDims;
};

inline AclTensorSpec ParseAclTensorSpec(const std::string &value)
{
    const std::string trimmed = Trim(value);
    if (trimmed.empty() || trimmed == "NONE") {
        return {};
    }

    std::vector<std::string> sections;
    SplitStr2Vec(trimmed, "#", sections);

    AclTensorSpec spec;
    if (!sections.empty()) {
        spec.viewDims = ParseDims(sections[0]);
    }
    if (sections.size() > 1U) {
        spec.stride = ParseDims(sections[1]);
    }
    if (sections.size() > 2U) {
        spec.storageDims = ParseDims(sections[2]);
    }
    return spec;
}

inline std::vector<int64_t> ParseAclTensorViewDims(const std::string &value)
{
    return ParseAclTensorSpec(value).viewDims;
}

inline std::vector<std::string> ParseAclTensorListSpecs(const std::string &value)
{
    std::vector<std::string> specs;
    SplitStr2Vec(Trim(value), ";", specs);
    if (specs.empty()) {
        specs.emplace_back("NONE");
    }
    for (auto &spec : specs) {
        spec = Trim(spec);
        if (spec.empty()) {
            spec = "NONE";
        }
    }
    return specs;
}

inline TensorDesc MakeAclTensorDesc(const std::vector<int64_t> &shape, aclDataType dtype, aclFormat format,
                                    const std::vector<int64_t> &storageShape = {},
                                    const std::vector<int64_t> &stride = {})
{
    return TensorDesc(shape, dtype, format, stride, 0, storageShape).ValueRange(-10, 10);
}

inline TensorDesc BuildAclTensorDescFromSpec(const std::string &shapeSpec, const std::string &dtype,
                                             const std::string &format)
{
    const AclTensorSpec spec = ParseAclTensorSpec(shapeSpec);
    return TensorDesc(spec.viewDims, ParseAclDtype(dtype), ParseAclFormat(format), spec.stride, 0, spec.storageDims);
}

inline std::vector<int64_t> GetOptionalDims(const std::vector<std::vector<int64_t>> &dimsList, size_t index)
{
    return index < dimsList.size() ? dimsList[index] : std::vector<int64_t>{};
}

inline TensorListDesc BuildAclTensorListDesc(const std::vector<std::vector<int64_t>> &shapes, aclDataType dtype,
                                             aclFormat format,
                                             const std::vector<std::vector<int64_t>> &storageShapes = {},
                                             const std::vector<std::vector<int64_t>> &strides = {})
{
    std::vector<TensorDesc> tensorVec;
    tensorVec.reserve(shapes.size());
    for (size_t i = 0; i < shapes.size(); ++i) {
        tensorVec.emplace_back(MakeAclTensorDesc(shapes[i], dtype, format, GetOptionalDims(storageShapes, i),
                                                 GetOptionalDims(strides, i)));
    }
    return TensorListDesc(tensorVec);
}

inline TensorListDesc BuildAclTensorListDesc(const std::string &shapeSpecList, const std::string &dtype,
                                             const std::string &format, bool withValueRange = true)
{
    std::vector<TensorDesc> tensors;
    for (const auto &shapeSpec : ParseAclTensorListSpecs(shapeSpecList)) {
        TensorDesc tensor = BuildAclTensorDescFromSpec(shapeSpec, dtype, format);
        if (withValueRange) {
            tensor.ValueRange(-10, 10);
        }
        tensors.emplace_back(tensor);
    }
    return TensorListDesc(tensors);
}

} // namespace ops::ut

#endif // TESTS_UT_FRAMEWORK_NORMAL_COMMON_UT_ACL_GMM_CASE_UTILS_H
