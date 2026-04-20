/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CSV_TEST_UTILS_H
#define CSV_TEST_UTILS_H

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace csv_test_utils {

inline std::string Trim(std::string value)
{
    auto isNotSpace = [](unsigned char c) { return !std::isspace(c); };
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), isNotSpace));
    value.erase(std::find_if(value.rbegin(), value.rend(), isNotSpace).base(), value.end());
    return value;
}

inline std::string ToLower(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

inline bool ParseBool(const std::string &value)
{
    const std::string lower = ToLower(Trim(value));
    return lower == "true" || lower == "1" || lower == "yes";
}

inline void SplitStr2Vec(const std::string &input, const std::string &delimiter, std::vector<std::string> &output)
{
    const auto delimiterLen = delimiter.size();
    std::string::size_type currPos = 0;
    std::string::size_type nextPos = input.find(delimiter, currPos);
    while (nextPos != std::string::npos) {
        output.emplace_back(input.substr(currPos, nextPos - currPos));
        currPos = nextPos + delimiterLen;
        nextPos = input.find(delimiter, currPos);
    }
    if (currPos <= input.size()) {
        output.emplace_back(input.substr(currPos));
    }
}

inline std::string GetExeDirPath()
{
#if defined(_WIN32)
    char path[MAX_PATH] = {0};
    DWORD len = GetModuleFileNameA(nullptr, path, MAX_PATH);
    std::string exePath(path, len);
    const auto pos = exePath.find_last_of("\\/");
    return pos == std::string::npos ? std::string(".\\") : exePath.substr(0, pos + 1);
#else
    constexpr size_t kBuf = 4096;
    char path[kBuf] = {0};
    const ssize_t len = readlink("/proc/self/exe", path, sizeof(path) - 1);
    if (len <= 0) {
        return "./";
    }
    path[len] = '\0';
    std::string exePath(path);
    const auto pos = exePath.find_last_of('/');
    return pos == std::string::npos ? std::string("./") : exePath.substr(0, pos + 1);
#endif
}

inline std::string GetCurrentFileDir(const char *filePath)
{
    std::string fp(filePath);
    const auto pos = fp.find_last_of("\\/");
    return pos == std::string::npos ? std::string(".") : fp.substr(0, pos);
}

inline std::string GetCwd()
{
#if defined(_WIN32)
    char path[MAX_PATH] = {0};
    DWORD len = GetCurrentDirectoryA(MAX_PATH, path);
    return len == 0 ? std::string(".") : std::string(path, len);
#else
    constexpr size_t kBuf = 4096;
    char path[kBuf] = {0};
    if (getcwd(path, sizeof(path)) == nullptr) {
        return ".";
    }
    return std::string(path);
#endif
}

inline std::string JoinPath(const std::string &dir, const std::string &file)
{
    if (dir.empty()) {
        return file;
    }
    const char last = dir.back();
    if (last == '/' || last == '\\') {
        return dir + file;
    }
    return dir + "/" + file;
}

inline bool FileExists(const std::string &path)
{
    std::ifstream in(path);
    return in.is_open();
}

/** Return the first path that exists; if none exist, return the first candidate (caller may log failure). */
inline std::string FindFirstExistingPath(const std::vector<std::string> &candidates)
{
    for (const auto &p : candidates) {
        if (FileExists(p)) {
            return p;
        }
    }
    return candidates.empty() ? std::string{} : candidates.front();
}

/** Parse colon-separated dimensions; empty or "NONE" yields empty vector. */
inline std::vector<int64_t> ParseDims(const std::string &value)
{
    const std::string trimmed = Trim(value);
    if (trimmed.empty() || trimmed == "NONE") {
        return {};
    }
    std::vector<std::string> tokens;
    SplitStr2Vec(trimmed, ":", tokens);
    std::vector<int64_t> dims;
    dims.reserve(tokens.size());
    for (const auto &token : tokens) {
        dims.emplace_back(static_cast<int64_t>(std::stoll(Trim(token))));
    }
    return dims;
}

/** Parse pipe-separated int64 list; empty or "NONE" yields empty vector. */
inline std::vector<int64_t> ParseI64List(const std::string &value)
{
    const std::string trimmed = Trim(value);
    if (trimmed.empty() || trimmed == "NONE") {
        return {};
    }
    std::vector<std::string> tokens;
    SplitStr2Vec(trimmed, "|", tokens);
    std::vector<int64_t> out;
    for (const auto &token : tokens) {
        out.emplace_back(static_cast<int64_t>(std::stoll(Trim(token))));
    }
    return out;
}

} // namespace csv_test_utils

#endif // CSV_TEST_UTILS_H
