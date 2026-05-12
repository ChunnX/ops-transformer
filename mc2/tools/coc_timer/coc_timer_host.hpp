/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCEND_TIMER_HOST_H
#define ASCEND_TIMER_HOST_H

#include "cocTimer.hpp"
#include <vector>
#include <iostream>
#include <unordered_map>
#include <fstream>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <sstream>
#include "acl/acl_rt.h"

class CocTimerHost
{
public:
    // ========== 常量定义 ==========
    static constexpr float kCycleToTimeBase = 50.0f;           // 周期转微秒的基准值
    static constexpr int kCoresPerGroup = 3;                   // 每组的核数
    static constexpr int kSubGroupIdForAIC = 0;                // AIC 的子组 ID

    // ========== 1. 总缓冲区大小计算 ==========
    static inline int GetTotalTimerBufferSize()
    {
        return CocTimer::N_TIMING_COUNTER + CocTimer::DYNAMIC_ITER_TOTAL_SIZE;
    }

    // ========== 2. 动态项名称解析 ==========
    static inline bool ParseDynamicName(const std::string &name, CocTimer::DynamicTimingType &type, int &iter)
    {
        size_t lastUnderscore = name.find_last_of('_');
        if (lastUnderscore == std::string::npos)
        {
            return false;
        }

        std::string prefix = name.substr(0, lastUnderscore);
        std::string iterStr = name.substr(lastUnderscore + 1);

        try
        {
            iter = std::stoi(iterStr);
        }
        catch (...)
        {
            return false;
        }

        for (const auto &pair : dynamicNameMap)
        {
            if (pair.second == prefix)
            {
                type = pair.first;
                return true;
            }
        }

        return false;
    }

    // ========== 3. 名称映射表 ==========
    static inline const std::unordered_map<int, std::string> fixedNameMap = {
        {CocTimer::KERNEL_TIMING_IDX, "KERNEL_TIMING"},
    };

    static inline const std::unordered_map<CocTimer::DynamicTimingType, std::string> dynamicNameMap = {
        {CocTimer::AIV, "AIV"},
        {CocTimer::AIC, "AIC"},
    };

    // ========== 4. 计时结果结构体 ==========
    struct TimerResult
    {
        std::vector<int64_t> baseCycleData;            // 基础计时数据（N_TIMING_COUNTER）
        std::vector<int64_t> dynamicActualIterPerCore; // 每核动态迭代次数（DYNAMIC_ITER_TOTAL_SIZE）
        std::vector<int64_t> dynamicActualIterMax;     // 每个动态项的全局最大迭代次数（DYNAMIC_TYPE_COUNT）
    };

    // ========== 5. 动态项索引计算 ==========
    static inline int GetDynamicTimingIdx(CocTimer::DynamicTimingType type, int iter)
    {
        int targetIdx = CocTimer::FIXED_TIMING_COUNT + static_cast<int>(type) * CocTimer::MAX_DYNAMIC_ITER + iter;
        if (iter >= CocTimer::MAX_DYNAMIC_ITER)
        {
            std::cerr << "[WARNING] 动态项 " << dynamicNameMap.at(type)
                      << " 迭代次数 " << iter
                      << " 超出全局最大迭代数 " << CocTimer::MAX_DYNAMIC_ITER << std::endl;
        }
        if (targetIdx >= CocTimer::N_TIMING_COUNTER_PER_CORE_ALIGN)
        {
            std::cerr << "[WARNING] 动态项 " << dynamicNameMap.at(type)
                      << " 索引 " << targetIdx
                      << " 超出每核对齐后最大计数器数 " << CocTimer::N_TIMING_COUNTER_PER_CORE_ALIGN << std::endl;
        }
        return targetIdx;
    }

    // ========== 6. 计时项名称获取 ==========
    static inline std::vector<std::string> GetTimingNames(const std::vector<int64_t> &dynamicActualIterMax)
    {
        std::vector<std::string> names;
        const auto &fixedMap = fixedNameMap;

        // 固定项名称
        for (int i = 0; i < CocTimer::FIXED_TIMING_COUNT; ++i)
        {
            auto it = fixedMap.find(i);
            names.push_back(it != fixedMap.end() ? it->second : "UNKNOWN_FIXED_" + std::to_string(i));
        }

        // 动态项名称
        for (int t = 0; t < CocTimer::DYNAMIC_TYPE_COUNT; ++t)
        {
            auto type = static_cast<CocTimer::DynamicTimingType>(t);
            if (!dynamicNameMap.count(type))
                continue;

            const std::string &prefix = dynamicNameMap.at(type);
            int64_t actualMaxIter = CocTimer::MAX_DYNAMIC_ITER;
            if (t < dynamicActualIterMax.size() && dynamicActualIterMax[t] > 0)
            {
                actualMaxIter = dynamicActualIterMax[t];
            }

            for (int iter = 0; iter < actualMaxIter; ++iter)
            {
                names.push_back(prefix + "_" + std::to_string(iter));
            }
        }

        return names;
    }

    static inline std::vector<std::string> GetTimingNames()
    {
        return GetTimingNames(std::vector<int64_t>());
    }

    // ========== 7. 从Device拷贝计时结果 ==========
    static inline TimerResult GetTimeResult(void *deviceAddr)
    {
        TimerResult result;
        int totalBufferSize = GetTotalTimerBufferSize();
        std::vector<int64_t> allData(totalBufferSize, 0);

        auto ret = aclrtMemcpy(
            allData.data(), totalBufferSize * sizeof(int64_t),
            deviceAddr, totalBufferSize * sizeof(int64_t),
            ACL_MEMCPY_DEVICE_TO_HOST);

        if (ret != ACL_SUCCESS)
        {
            std::cerr << "Memcpy device→host failed! ERROR: " << ret << std::endl;
            return result;
        }

        result.baseCycleData = std::vector<int64_t>(allData.begin(), allData.begin() + CocTimer::N_TIMING_COUNTER);
        result.dynamicActualIterPerCore = std::vector<int64_t>(allData.begin() + CocTimer::N_TIMING_COUNTER, allData.end());

        // 计算每个动态项的全局最大迭代次数
        result.dynamicActualIterMax.resize(CocTimer::DYNAMIC_TYPE_COUNT, 0);
        for (int coreId = 0; coreId < CocTimer::N_CORE_COUNT; ++coreId)
        {
            int coreOffset = coreId * CocTimer::DYNAMIC_ITER_PER_CORE_ALIGN;
            for (int type = 0; type < CocTimer::DYNAMIC_TYPE_COUNT; ++type)
            {
                int idx = coreOffset + type;
                if (idx >= result.dynamicActualIterPerCore.size())
                    continue;
                if (result.dynamicActualIterPerCore[idx] > result.dynamicActualIterMax[type])
                {
                    result.dynamicActualIterMax[type] = result.dynamicActualIterPerCore[idx];
                }
            }
        }

        return result;
    }

    // ========== 8. 周期数转时间（微秒） ==========
    inline static float CycleToTime(int64_t cycles)
    {
        return static_cast<float>(cycles) / kCycleToTimeBase;
    }

    // ========== 9. 写入原始时间戳数据（CSV格式，包含start/end cycle值）==========
    inline static bool WriteTimeResultsRaw(const std::string &filename, const TimerResult &timerResult)
    {
        std::ofstream outfile(filename);
        if (!outfile.is_open()) return false;

        const auto &timingNames = GetTimingNames(timerResult.dynamicActualIterMax);
        const int totalItems = timingNames.size();
        const auto &cycleResult = timerResult.baseCycleData;
        const auto &dynamicMaxIter = timerResult.dynamicActualIterMax;

        outfile << "group_id,core_type,sub_group_id";
        for (const auto &name : timingNames) {
            outfile << "," << name << "_start_cycle," << name << "_end_cycle," << name << "_duration_us";
        }
        for (int t = 0; t < CocTimer::DYNAMIC_TYPE_COUNT; ++t) {
            auto type = static_cast<CocTimer::DynamicTimingType>(t);
            if (dynamicNameMap.count(type)) outfile << "," << dynamicNameMap.at(type) << "_IterCount";
        }
        outfile << std::endl;

        for (int i = 0; i < CocTimer::N_CORE_COUNT; i++) {
            int groupId = i / kCoresPerGroup;
            int subGroupId = i % kCoresPerGroup;
            std::string coreType = (subGroupId == kSubGroupIdForAIC) ? "AIC" : "AIV";
            int subId = (coreType == "AIC") ? 0 : (subGroupId - 1);

            int baseIndex = groupId * CocTimer::N_TIMING_COUNTER_PER_CORE_ALIGN * kCoresPerGroup;
            if (coreType == "AIV") baseIndex += CocTimer::N_TIMING_COUNTER_PER_CORE_ALIGN * subGroupId;

            outfile << groupId << "," << coreType << "," << subId;

            for (int nameIdx = 0; nameIdx < totalItems; ++nameIdx) {
                const std::string &currName = timingNames[nameIdx];
                int elementIdx = -1;

                if (nameIdx < CocTimer::FIXED_TIMING_COUNT) {
                    elementIdx = baseIndex + nameIdx * 2; // 正确：只乘事件索引
                } else {
                    CocTimer::DynamicTimingType dynType;
                    int iter;
                    if (ParseDynamicName(currName, dynType, iter)) {
                        int dynMemIdx = GetDynamicTimingIdx(dynType, iter);
                        elementIdx = baseIndex + dynMemIdx * 2;
                    }
                }

                // 越界检查：判断绝对索引
                if (elementIdx < 0 || elementIdx + 1 >= CocTimer::N_TIMING_COUNTER) {
                    outfile << ",0,0,0";
                } else {
                    int64_t start_cycles = cycleResult[elementIdx];
                    int64_t end_cycles = cycleResult[elementIdx + 1];
                    float duration_us = (end_cycles > start_cycles) ? CycleToTime(end_cycles - start_cycles) : 0.0f;
                    // Raw 输出三个值
                    outfile << "," << start_cycles << "," << end_cycles << "," << duration_us;
                }
            }

            for (int t = 0; t < CocTimer::DYNAMIC_TYPE_COUNT; ++t) {
                int idxIter = i * CocTimer::DYNAMIC_ITER_PER_CORE_ALIGN + t;
                outfile << "," << (t < dynamicMaxIter.size() ? timerResult.dynamicActualIterPerCore[idxIter] : 0);
            }
            outfile << std::endl;
        }
        outfile.close();
        return true;
    }

    // ========== 10. 写入Duration数据 ==========
    inline static bool WriteTimeResultsDuration(const std::string &filename, const TimerResult &timerResult)
    {
        std::ofstream outfile(filename);
        if (!outfile.is_open()) return false;

        const auto &timingNames = GetTimingNames(timerResult.dynamicActualIterMax);
        const int totalItems = timingNames.size();
        const auto &cycleResult = timerResult.baseCycleData;
        const auto &dynamicMaxIter = timerResult.dynamicActualIterMax;

        outfile << "group_id,core_type,sub_group_id";
        for (const auto &name : timingNames) {
            outfile << "," << name << "_duration(us)";
        }
        for (int t = 0; t < CocTimer::DYNAMIC_TYPE_COUNT; ++t) {
            auto type = static_cast<CocTimer::DynamicTimingType>(t);
            if (dynamicNameMap.count(type)) outfile << "," << dynamicNameMap.at(type) << "_IterCount";
        }
        outfile << std::endl;

        for (int i = 0; i < CocTimer::N_CORE_COUNT; i++) {
            int groupId = i / kCoresPerGroup;
            int subGroupId = i % kCoresPerGroup;
            std::string coreType = (subGroupId == kSubGroupIdForAIC) ? "AIC" : "AIV";
            int subId = (coreType == "AIC") ? 0 : (subGroupId - 1);

            int baseIndex = groupId * CocTimer::N_TIMING_COUNTER_PER_CORE_ALIGN * kCoresPerGroup;
            if (coreType == "AIV") baseIndex += CocTimer::N_TIMING_COUNTER_PER_CORE_ALIGN * subGroupId;

            outfile << groupId << "," << coreType << "," << subId;

            for (int nameIdx = 0; nameIdx < totalItems; ++nameIdx) {
                const std::string &currName = timingNames[nameIdx];
                int elementIdx = -1;

                if (nameIdx < CocTimer::FIXED_TIMING_COUNT) {
                    elementIdx = baseIndex + nameIdx * 2;
                } else {
                    CocTimer::DynamicTimingType dynType;
                    int iter;
                    if (ParseDynamicName(currName, dynType, iter)) {
                        int dynMemIdx = GetDynamicTimingIdx(dynType, iter);
                        elementIdx = baseIndex + dynMemIdx * 2;
                    }
                }

                if (elementIdx < 0 || elementIdx + 1 >= CocTimer::N_TIMING_COUNTER) {
                    // Duration 只补一个 0
                    outfile << ",0";
                } else {
                    int64_t start_cycles = cycleResult[elementIdx];
                    int64_t end_cycles = cycleResult[elementIdx + 1];
                    float duration_us = (end_cycles > start_cycles) ? CycleToTime(end_cycles - start_cycles) : 0.0f;
                    // Duration 只输出一个 us 数据
                    outfile << "," << duration_us;
                }
            }

            for (int t = 0; t < CocTimer::DYNAMIC_TYPE_COUNT; ++t) {
                int idxIter = i * CocTimer::DYNAMIC_ITER_PER_CORE_ALIGN + t;
                outfile << "," << (t < dynamicMaxIter.size() ? timerResult.dynamicActualIterPerCore[idxIter] : 0);
            }
            outfile << std::endl;
        }
        outfile.close();
        return true;
    }

    inline static bool WriteAllTimerResults(const std::string &rawFilename, const std::string &durationFilename, const TimerResult &timerResult)
    {
        bool rawOk = WriteTimeResultsRaw(rawFilename, timerResult);
        bool durOk = WriteTimeResultsDuration(durationFilename, timerResult);
        return rawOk && durOk;
    }
};

#endif // ASCEND_TIMER_HOST_H