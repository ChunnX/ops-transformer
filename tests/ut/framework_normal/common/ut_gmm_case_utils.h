/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TESTS_UT_FRAMEWORK_NORMAL_COMMON_UT_GMM_CASE_UTILS_H
#define TESTS_UT_FRAMEWORK_NORMAL_COMMON_UT_GMM_CASE_UTILS_H

#include <algorithm>
#include <cctype>
#include <exception>
#include <sstream>
#include <string>
#include <vector>

#include "exe_graph/runtime/shape.h"
#include "exe_graph/runtime/storage_shape.h"
#include "ut_parse_utils.h"

namespace ops::ut {

std::string NormalizePath(std::string path);
std::string JoinPath(const std::string &lhs, const std::string &rhs);
bool FileExists(const std::string &path);
std::string GetDirName(std::string path);
std::string GetExeDirPath();
std::string GetCurrentFileDir(const char *file);
std::string GetCwd();
std::string ResolveCsvPath(const std::string &csvName, const std::string &repoRelativeDir, const char *currentFile);

inline gert::Shape BuildGertShape(const std::vector<int64_t> &dims)
{
    gert::Shape shape;
    shape.SetDimNum(dims.size());
    for (size_t i = 0; i < dims.size(); ++i) {
        shape.SetDim(i, dims[i]);
    }
    return shape;
}

inline gert::StorageShape MakeGertStorageShape(const std::vector<int64_t> &originDims,
                                               const std::vector<int64_t> &storageDims)
{
    gert::StorageShape shape;
    shape.MutableOriginShape() = BuildGertShape(originDims);
    shape.MutableStorageShape() = BuildGertShape(storageDims);
    return shape;
}

inline gert::StorageShape MakeGertStorageShape(const std::vector<int64_t> &dims)
{
    return MakeGertStorageShape(dims, dims);
}

inline std::string MakeSafeParamName(std::string name)
{
    std::transform(name.begin(), name.end(), name.begin(),
                   [](unsigned char c) { return std::isalnum(c) ? static_cast<char>(c) : '_'; });
    return name;
}

inline std::string BuildCsvParseErrorMessage(const std::string &csvPath, size_t lineNo,
                                             const std::string &caseName, const std::exception &error)
{
    std::ostringstream oss;
    oss << "Failed to parse CSV case. file=" << csvPath << ", line=" << lineNo;
    if (!caseName.empty()) {
        oss << ", case=" << caseName;
    }
    oss << ", error=" << error.what();
    return oss.str();
}

} // namespace ops::ut

#endif // TESTS_UT_FRAMEWORK_NORMAL_COMMON_UT_GMM_CASE_UTILS_H
