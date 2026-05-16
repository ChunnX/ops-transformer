/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CSV_CASE_LOAD_UTILS_H
#define CSV_CASE_LOAD_UTILS_H

#include <iostream>
#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <filesystem>
#include <cstdlib>

using csv_map = std::unordered_map<std::string, std::string>;

template <typename Map>
inline typename Map::mapped_type ReadMap(const Map& m, const typename Map::key_type& key,
    const typename Map::mapped_type& defaultValue = typename Map::mapped_type{})
{
    auto it = m.find(key);
    return it != m.end() ? it->second : defaultValue;
}

inline std::vector<int64_t> GetShapeArr(const std::string& shapeArrStr)
{
    std::vector<int64_t> shapeArr;
    std::istringstream iss(shapeArrStr);
    int64_t num;
    while (iss >> num) {
        shapeArr.emplace_back(num);
    }
    return shapeArr;
}

inline std::string FindCsvFile(const std::string& csvName)
{
    std::vector<std::filesystem::path> candidates;
    std::filesystem::path cwd = std::filesystem::current_path();
    
    // 1. 当前工作目录（最常见）
    candidates.push_back(cwd / csvName);
    
    // 2. 构建输出目录
    candidates.push_back(cwd / "build" / csvName);
    candidates.push_back(cwd / "build_out" / csvName);
    
    // 3. 环境变量指定的源码目录
    if (const char* srcDir = std::getenv("OPS_TRANSFORMER_DIR")) {
        candidates.push_back(std::filesystem::path(srcDir) / csvName);
    }
    if (const char* projectDir = std::getenv("PROJECT_SOURCE_DIR")) {
        candidates.push_back(std::filesystem::path(projectDir) / csvName);
    }
    
    // 4. 从当前目录向上查找源码根目录（最多向上 5 层）
    std::filesystem::path searchPath = cwd;
    for (int i = 0; i < 5 && !searchPath.empty(); ++i) {
        // 检查是否是源码根目录（包含 tests/ut 子目录）
        if (std::filesystem::exists(searchPath / "tests" / "ut")) {
            candidates.push_back(searchPath / csvName);
            // 也检查可能的子目录结构
            std::filesystem::path testsDir = searchPath / "tests" / "ut" / "framework_normal";
            if (std::filesystem::exists(testsDir)) {
                candidates.push_back(testsDir / csvName);
            }
        }
        searchPath = searchPath.parent_path();
    }
    
    // 5. 遍历候选路径，返回第一个存在的
    for (const auto& candidate : candidates) {
        if (std::filesystem::exists(candidate)) {
            return candidate.string();
        }
    }
    
    // 6. 如果候选路径都找不到，在当前目录及其子目录中快速查找（最多 3 层）
    try {
        for (int depth = 1; depth <= 3; ++depth) {
            std::filesystem::path searchBase = cwd;
            for (int d = 0; d < depth && std::filesystem::exists(searchBase); ++d) {
                for (const auto& entry : std::filesystem::directory_iterator(searchBase)) {
                    if (entry.is_directory()) {
                        std::filesystem::path candidate = entry.path() / csvName;
                        if (std::filesystem::exists(candidate)) {
                            return candidate.string();
                        }
                    }
                }
                // 下一层
                if (std::filesystem::exists(searchBase)) {
                    for (const auto& subEntry : std::filesystem::directory_iterator(searchBase)) {
                        if (subEntry.is_directory()) {
                            searchBase = subEntry.path();
                            break;
                        }
                    }
                }
            }
        }
    } catch (const std::filesystem::filesystem_error&) {
        // 忽略错误
    }
    
    return "";
}

inline std::string ReplaceFileExtension2Csv(const char* file)
{
    std::filesystem::path filePath(file);
    std::string csvName = filePath.filename().replace_extension("csv").string();
    
    // 如果 __FILE__ 包含完整路径且源文件存在，直接替换扩展名
    if (filePath.has_parent_path() && std::filesystem::exists(filePath)) {
        std::filesystem::path csvPath = filePath;
        csvPath.replace_extension("csv");
        if (std::filesystem::exists(csvPath)) {
            return csvPath.string();
        }
    }
    
    // 使用优化后的查找函数
    std::string found = FindCsvFile(csvName);
    if (!found.empty()) {
        return found;
    }
    
    // 如果都找不到，返回当前目录下的文件名（后续会报错提示）
    return (std::filesystem::current_path() / csvName).string();
}

template<typename T> // T 需要支持 T(const csv_map&) 构造函数
std::vector<T> GetCasesFromCsv(const std::string& csvPath)
{
    std::vector<T> cases;
    std::ifstream csvFile(csvPath, std::ios::in);
    if (!csvFile.is_open()) {
        std::cout << "[ERROR] Cannot open case file \"" << csvPath << "\"!" << std::endl;
        return cases;
    }

    std::vector<std::string> keys;
    std::string line;
    std::getline(csvFile, line);
    std::istringstream stream(line);
    for (std::string data; std::getline(stream, data, ','); ) {
        keys.emplace_back(data);
    }
    // lineNum = 1 是表头
    for (int lineNum = 2; std::getline(csvFile, line); ++lineNum) {
        if (line.empty()) {
            std::cout << "[ERROR] " << csvPath << ":" << lineNum
                      << " Row data is empty!" << std::endl;
            return std::vector<T>();
        }
        std::vector<std::string> values;
        std::istringstream stream(line);
        for (std::string data; std::getline(stream, data, ','); ) {
            values.emplace_back(data);
        }
        if (line.back() == ',') {
            values.emplace_back("");
        }
        if (keys.size() != values.size()) {
            std::cout << "[ERROR] " << csvPath << ":" << lineNum
                      << " Row data does not match CSV header columns. Row has " << values.size()
                      << " columns, but header has " << keys.size() << " columns." << std::endl;
            return std::vector<T>();
        }
        csv_map csvMap;
        for (size_t i = 0; i < keys.size(); ++i) {
            if (! values[i].empty()) {
                csvMap[keys[i]] = values[i];
            }
        }
        cases.emplace_back(csvMap);
    }
    return cases;
}

template<typename T>
inline std::string PrintCaseInfoString(const testing::TestParamInfo<T>& info)
{
    return info.param.case_name;
}

#endif // CSV_CASE_LOAD_UTILS_H
