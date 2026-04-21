/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TESTS_UT_FRAMEWORK_NORMAL_COMMON_UT_ACL_PARSE_UTILS_H
#define TESTS_UT_FRAMEWORK_NORMAL_COMMON_UT_ACL_PARSE_UTILS_H

#include <map>
#include <string>

#include "opdev/common_types.h"
#include "ut_parse_utils.h"

namespace ops::ut {

inline aclDataType ParseAclDtype(const std::string &dtype)
{
    static const std::map<std::string, aclDataType> dtypeMap = {
        {"FLOAT", ACL_FLOAT},
        {"FLOAT16", ACL_FLOAT16},
        {"BF16", ACL_BF16},
        {"INT8", ACL_INT8},
        {"INT4", ACL_INT4},
        {"INT32", ACL_INT32},
        {"INT64", ACL_INT64},
        {"UINT64", ACL_UINT64},
        {"FLOAT8_E4M3FN", ACL_FLOAT8_E4M3FN},
        {"FLOAT8_E5M2", ACL_FLOAT8_E5M2},
        {"FLOAT8_E8M0", ACL_FLOAT8_E8M0},
        {"HIFLOAT8", ACL_HIFLOAT8},
        {"FLOAT4_E2M1", ACL_FLOAT4_E2M1},
        {"UNDEFINED", ACL_DT_UNDEFINED},
    };
    const auto it = dtypeMap.find(Trim(dtype));
    return it == dtypeMap.end() ? ACL_DT_UNDEFINED : it->second;
}

inline aclFormat ParseAclFormat(const std::string &format)
{
    static const std::map<std::string, aclFormat> formatMap = {
        {"ND", ACL_FORMAT_ND},
        {"FRACTAL_NZ", ACL_FORMAT_FRACTAL_NZ},
        {"FRACTAL_NZ_C0_2", ACL_FORMAT_FRACTAL_NZ_C0_2},
        {"FRACTAL_NZ_C0_4", ACL_FORMAT_FRACTAL_NZ_C0_4},
        {"UNDEFINED", ACL_FORMAT_UNDEFINED},
    };
    const auto it = formatMap.find(Trim(format));
    return it == formatMap.end() ? ACL_FORMAT_ND : it->second;
}

} // namespace ops::ut

#endif // TESTS_UT_FRAMEWORK_NORMAL_COMMON_UT_ACL_PARSE_UTILS_H
