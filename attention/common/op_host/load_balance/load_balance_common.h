/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file stream_k.h
 * \brief 
 */

#ifndef LOAD_BALANCE_COMMON_H
#define LOAD_BALANCE_COMMON_H

#include <cstdint>
#include <array>
#include <unordered_map>

namespace load_balance {

/******************************** ERROR CODE ********************************/
#define LOAD_BALANCE_SUCCESS                0
#define LOAD_BALANCE_ERROR_INVALID_INPUT    1

/******************************** CLASS DEF  ********************************/
struct DeviceInfo {
    uint32_t aicCoreNum { 0U };
    uint32_t aivCoreNum { 0U };
    std::string socVersion {};
};

struct GeneralSplitParam {
    uint32_t mBaseSize { 1U };
    uint32_t s2BaseSize { 1U };
    int64_t faToleranceRatio { 2 };
    int64_t fdTolerance { 0U };                         // no fd + tolerance < fd, then choose no fd
};

enum class SparseMode : uint8_t {
    DEFAULT_MASK = 0,
    ALL_MASK,
    LEFT_UP_CAUSAL,
    RIGHT_DOWN_CAUSAL,
    BAND,
    BUTT,
};

enum class Layout : uint8_t {
    BSND = 0,
    BNSD,
    BSH,
    NBSD,
    TND,
    NTD,
    PA_NZ,
    BUTT
};

enum class DataType : uint8_t {
    FP32,
    FP16,
    INT4,
    INT8,
    INT32,
    BUTT
};

template<class T>
using Range = std::pair<T, T>;

template<class T, size_t N>
using SquareTable = std::array<std::array<T, N>, N>;

/******************************** UTIL FUNC ********************************/
template<class T>
inline T CeilDiv(T a, T b)
{
    return a / b + (a % b != 0);        // avoid overflow
}

template<class T>
inline T FloorDiv(T a, T b)
{
    return a / b;        // avoid overflow
}

template<typename T>
inline T AddOne(T val)
{
    static_assert(std::is_integral_v<T>, "must be integer type");
    return val + 1;
}

template<typename T>
inline T MinusOne(T val)
{
    static_assert(std::is_integral_v<T>, "must be integer type");
    return val - 1;
}

template<typename T>
inline T ToOpenInterval(T val)
{
    return AddOne(val);
}

template<typename T>
inline T ToClosedInterval(T val)
{
    return MinusOne(val);
}

template<typename T>
inline T IndexToNum(T val)
{
    return AddOne(val);
}

template<typename T>
inline T NumToIndex(T val)
{
    return MinusOne(val);
}

template<typename T>
T Clip(T value, T minValue, T maxValue)
{
    if (value < minValue) {
        return minValue;
    }
    if (value > maxValue) {
        return maxValue;
    }
    return value;
}

template<typename T>
inline bool IsWithinTolerance(T limit, T tolerance, T value)
{
    return limit + tolerance >= value;
}

static Layout ConvertToLayout(const std::string &layoutStr)
{
    static std::unordered_map<std::string, Layout> layoutTable {
        { "BSND",  Layout::BSND },
        { "BNSD",  Layout::BNSD },
        { "BSH",   Layout::BSH },
        { "NBSD",  Layout::NBSD },
        { "TND",   Layout::TND },
        { "NTD",   Layout::NTD },
        { "PA_NZ", Layout::PA_NZ }
    };
    if (layoutTable.find(layoutStr) != layoutTable.end()) {
        return layoutTable[layoutStr];
    }
    return Layout::BUTT;
}

static uint32_t GetDataTypeByteSize(DataType type)
{
    switch (type) {
        case (DataType::FP32):
        case (DataType::INT32):
            return 4U;
        case (DataType::FP16):
            return 2U;
        case (DataType::INT8):
            return 1U;
        default:
            return 2U;
    }
}

}
#endif