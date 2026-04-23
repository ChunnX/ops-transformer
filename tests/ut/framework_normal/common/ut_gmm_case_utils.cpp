/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ut_gmm_case_utils.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>

#if defined(_WIN32)
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace ops::ut {

std::string NormalizePath(std::string path)
{
    std::replace(path.begin(), path.end(), '\\', '/');
    return path;
}

std::string JoinPath(const std::string &lhs, const std::string &rhs)
{
    if (lhs.empty()) {
        return rhs;
    }
    if (rhs.empty()) {
        return lhs;
    }
    if (lhs.back() == '/' || lhs.back() == '\\') {
        return lhs + rhs;
    }
    return lhs + "/" + rhs;
}

bool FileExists(const std::string &path)
{
    std::ifstream file(path, std::ios::in);
    return file.good();
}

std::string GetDirName(std::string path)
{
    path = NormalizePath(path);
    const auto pos = path.find_last_of('/');
    return pos == std::string::npos ? "." : path.substr(0, pos);
}

std::string GetExeDirPath()
{
#if defined(_WIN32)
    char path[MAX_PATH] = {0};
    const auto len = GetModuleFileNameA(nullptr, path, MAX_PATH);
    return GetDirName(std::string(path, len));
#else
    char path[4096] = {0};
    const ssize_t len = readlink("/proc/self/exe", path, sizeof(path) - 1);
    if (len <= 0) {
        return ".";
    }
    path[len] = '\0';
    return GetDirName(std::string(path));
#endif
}

std::string GetCurrentFileDir(const char *file)
{
    return GetDirName(file);
}

std::string GetCwd()
{
#if defined(_WIN32)
    char path[MAX_PATH] = {0};
    const auto len = GetCurrentDirectoryA(MAX_PATH, path);
    return len == 0 ? std::string(".") : std::string(path, len);
#else
    char path[4096] = {0};
    if (getcwd(path, sizeof(path)) == nullptr) {
        return ".";
    }
    return std::string(path);
#endif
}

std::string ResolveCsvPath(const std::string &csvName, const std::string &repoRelativeDir, const char *currentFile)
{
    const std::string exeDir = GetExeDirPath();
    const std::string cwd = GetCwd();
    const std::string currentFileDir = GetCurrentFileDir(currentFile);

    std::vector<std::string> candidates;
    candidates.emplace_back(JoinPath(currentFileDir, csvName));
    candidates.emplace_back(JoinPath(exeDir, csvName));
    candidates.emplace_back(JoinPath(cwd, csvName));

    const char *codePath = std::getenv("CODE_PATH");
    if (codePath != nullptr) {
        candidates.emplace_back(JoinPath(codePath, JoinPath(repoRelativeDir, csvName)));
    }

    candidates.emplace_back(JoinPath(".", JoinPath(repoRelativeDir, csvName)));
    candidates.emplace_back(JoinPath(exeDir, JoinPath("../../../../../", JoinPath(repoRelativeDir, csvName))));
    candidates.emplace_back(JoinPath(cwd, JoinPath("../../../../../", JoinPath(repoRelativeDir, csvName))));
    candidates.emplace_back(csvName);
    candidates.emplace_back(JoinPath(".", csvName));

    for (const auto &candidate : candidates) {
        if (FileExists(candidate)) {
            return candidate;
        }
    }
    return candidates.front();
}

} // namespace ops::ut
