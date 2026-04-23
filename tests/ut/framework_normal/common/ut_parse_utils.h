/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TESTS_UT_FRAMEWORK_NORMAL_COMMON_UT_PARSE_UTILS_H
#define TESTS_UT_FRAMEWORK_NORMAL_COMMON_UT_PARSE_UTILS_H

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace ops::ut {

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

inline int64_t ParseScalarToken(const std::string &value, const std::map<std::string, int64_t> &symbols = {})
{
    const std::string trimmed = Trim(value);
    if (trimmed.empty()) {
        return 0;
    }

    const auto it = symbols.find(trimmed);
    if (it != symbols.end()) {
        return it->second;
    }

    const auto divPos = trimmed.find('/');
    if (divPos != std::string::npos) {
        const int64_t lhs = ParseScalarToken(trimmed.substr(0, divPos), symbols);
        const int64_t rhs = ParseScalarToken(trimmed.substr(divPos + 1), symbols);
        return rhs == 0 ? 0 : lhs / rhs;
    }

    return std::stoll(trimmed);
}

inline std::vector<int64_t> ParseDims(const std::string &value,
                                      const std::map<std::string, int64_t> &symbols = {},
                                      bool noneAsZero = false)
{
    const std::string trimmed = Trim(value);
    if (trimmed.empty() || trimmed == "NONE") {
        return noneAsZero ? std::vector<int64_t>{0} : std::vector<int64_t>{};
    }
    if (trimmed == "ZERO") {
        return {0};
    }

    std::vector<std::string> dimTokens;
    SplitStr2Vec(trimmed, ":", dimTokens);

    std::vector<int64_t> dims;
    dims.reserve(dimTokens.size());
    for (const auto &token : dimTokens) {
        dims.emplace_back(ParseScalarToken(token, symbols));
    }
    return dims;
}

inline std::vector<int64_t> ParseI64List(const std::string &value, const std::string &sep = "|",
                                         const std::map<std::string, int64_t> &symbols = {})
{
    const std::string trimmed = Trim(value);
    if (trimmed.empty() || trimmed == "NONE") {
        return {};
    }

    std::vector<std::string> tokens;
    SplitStr2Vec(trimmed, sep, tokens);

    std::vector<int64_t> out;
    out.reserve(tokens.size());
    for (const auto &token : tokens) {
        out.emplace_back(ParseScalarToken(token, symbols));
    }
    return out;
}

inline std::vector<std::vector<int64_t>> ParseDimsList(const std::string &value)
{
    const std::string trimmed = Trim(value);
    if (trimmed.empty() || trimmed == "NONE") {
        return {};
    }

    std::vector<std::string> items;
    SplitStr2Vec(trimmed, ";", items);

    std::vector<std::vector<int64_t>> out;
    out.reserve(items.size());
    for (const auto &item : items) {
        out.emplace_back(ParseDims(item));
    }
    return out;
}

} // namespace ops::ut

#endif // TESTS_UT_FRAMEWORK_NORMAL_COMMON_UT_PARSE_UTILS_H
