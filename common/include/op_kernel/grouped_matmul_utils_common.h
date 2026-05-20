/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file grouped_matmul_utils_common.h
 * \brief Common constants, templates and structures for grouped matmul utilities.
 */
#ifndef GROUPED_MATMUL_UTILS_COMMON_H
#define GROUPED_MATMUL_UTILS_COMMON_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"

namespace CommonGMMUtils {
using namespace AscendC;

constexpr uint32_t INT8_BITS = 8;                                    // a int8 number has 8 bits
constexpr int32_t MKN_LIST_LEN = 128;                                // 128: predefined array legnth
constexpr uint32_t UB_BLOCK_UNIT_SIZE = 32;                          // 32: a block has 32 bytes data
constexpr uint32_t UB_BLOCK_DOUBLE_UNIT_SIZE = 64;                   // 64: a block has 64 bytes data
constexpr uint32_t HALF_UB_BLOCK_UNIT_SIZE = UB_BLOCK_UNIT_SIZE / 2; // 2: a float16 data has two bytes

constexpr uint64_t SYNC_AIV_AIC_FLAG = 3;
constexpr uint64_t SYNC_AIC_AIV_FLAG = 5;
constexpr uint64_t SYNC_MODE2 = 2;
// used for static tiling template
constexpr uint32_t BASIC_BLOCK_SIZE_128 = 128;
constexpr uint32_t BASIC_BLOCK_SIZE_256 = 256;
constexpr int32_t STATIC_TILING_DEPTH_A1_B1 = 8;
constexpr int32_t STATIC_TILING_STEP_KA_KB = 4;
constexpr uint64_t DOUBLE_BUFFER_L0A_L0B = 2;
constexpr uint32_t STATIC_TILING_MAX_K = 8192;
constexpr uint32_t STATIC_TILING_MAX_SINGLE_N = 1024;

template <class AT_, class BT_, class CT_, class BiasT_, const auto &MM_CFG = CFG_MDL> struct MMType {
    using AT = AT_;
    using BT = BT_;
    using CT = CT_;
    using BiasT = BiasT_;
    using MT = matmul::Matmul<AT, BT, CT, BiasT, MM_CFG>;
};

template <class AT_, class BT_, class CT_, class BiasT_, const auto &MM_CFG = CFG_MDL> struct MMImplType {
    using AT = AT_;
    using BT = BT_;
    using CT = CT_;
    using BiasT = BiasT_;
    using MT = matmul::MatmulImpl<AT, BT, CT, BiasT, MM_CFG>;
};

enum class ActiveType : std::uint8_t { INVALID_TYPE = 0, RELU, GELU_TANH, GELU_ERR_FUNC, FASTGELU, SILU };

template <typename T> __aicore__ inline T GreatestCommonDivisor(T a, T b)
{
    T c = a;
    if (a < b) {
        a = b;
        b = c;
    }
    while (b != 0) {
        c = a;
        a = b;
        b = c % b;
    }
    return a;
}

template <typename T> __aicore__ inline T LeastCommonMultiple(T a, T b)
{
    return a * b / GreatestCommonDivisor(a, b);
}

template <typename T> __aicore__ inline T Max(T a, T b)
{
    return a > b ? a : b;
}

template <typename T> __aicore__ inline T Min(T a, T b)
{
    return a > b ? b : a;
}

template <uint32_t base, typename T = uint32_t> __aicore__ inline T AlignUp(T a)
{
    return (a + base - 1) / base * base;
}

template <typename T> __aicore__ inline T AlignUp(T a, T base)
{
    return (a + base - 1) / base * base;
}

template <typename T> __aicore__ inline T AlignDown(T a, T base)
{
    if (unlikely(base == 0)) {
        return a;
    }
    return a / base * base;
}

template <> __aicore__ inline uint32_t AlignUp<4, uint32_t>(uint32_t a)
{
    // to be Multiple of 4, result should be in a format of b(xxxx,x100).
    // This means last two bits should be zero, requiring that
    // result = num & b(1111,1100) = num & (~3).
    // &(~3) operator may reduces num into the range [num, num - 3].
    // As the result should be no less than a (result >= a), it means num - 3 >= a in the worst case.
    // In this case, num >= a+3. On the other hand, num should also be less then a+4, otherwise,
    // the result will not be least multiple of 4 for 3. In other cases like [num, num - 2],
    // num = a + 3 also satisfies the goal condition.
    return (a + 3) & ~3; // & ~3: set last two bits of (a+3) to be zero
}

template <> __aicore__ inline uint32_t AlignUp<8, uint32_t>(uint32_t a)
{
    // In general, if we want to get the least multiple of b (b is the power of 2) for a,
    // it comes to a conclusion from the above comment: result = (a + (b - 1)) & (~b)
    return (a + 7) & ~7; // & ~7: set last four bits of (a+7) to be zero
}

template <> __aicore__ inline uint32_t AlignUp<16, uint32_t>(uint32_t a)
{
    // In general, if we want to get the least multiple of b (b is the power of 2) for a,
    // it comes to a conclusion from the above comment: result = (a + (b - 1)) & (~b)
    return (a + 15) & ~15; // & ~15: set last four bits of (a+15) to be zero
}

template <> __aicore__ inline uint32_t AlignUp<32, uint32_t>(uint32_t a)
{
    // refer to the above comments.
    return (a + 31) & ~31; // & ~31: set last five bits of (a+31) to be zero}
}

template <typename T>
__aicore__ inline __gm__ T* GetTensorAddr(uint16_t index, GM_ADDR tensorPtr) {
    __gm__ uint64_t* dataAddr = reinterpret_cast<__gm__ uint64_t*>(tensorPtr);
    uint64_t tensorPtrOffset = *dataAddr;  // The offset of the data address from the first address.
    // Moving 3 bits to the right means dividing by sizeof(uint64 t).
    __gm__ uint64_t* retPtr = dataAddr + (tensorPtrOffset >> 3);
    return reinterpret_cast<__gm__ T*>(*(retPtr + index));
}

template <typename T> __aicore__ inline constexpr uint32_t GetTypeBits()
{
    if constexpr (IsSameType<T, int4b_t>::value) {
        return 4; // 4: int4 bits number
    }
    return sizeof(T) * INT8_BITS;
}

__aicore__ static constexpr MatmulConfig GenGmmConf(bool isND2NZ)
{
    return {
        .doNorm = false,
        .doBasicBlock = false,
        .doMultiDataLoad = true,
        .basicM = BASIC_BLOCK_SIZE_128,
        .basicN = BASIC_BLOCK_SIZE_256,
        .basicK = BASIC_BLOCK_SIZE_128,
        .intrinsicsCheck = false,
        .isNBatch = false,
        .enVecND2NZ = isND2NZ,
        .doSpecialBasicBlock = false,
        .doMTE2Preload = 0,
        .singleCoreM = BASIC_BLOCK_SIZE_128,
        .singleCoreN = STATIC_TILING_MAX_SINGLE_N,
        .singleCoreK = STATIC_TILING_MAX_K,
        .stepM = 0,
        .stepN = 0,
        .baseMN = 0,
        .singleCoreMN = 0,
        .enUnitFlag = true,
        .isPerTensor = false,
        .hasAntiQuantOffset = false,
        .doIBShareNorm = false,
        .doSpecialMDL = false,
        .enableInit = false,
        .batchMode = BatchMode::NONE,
        .enableEnd = true,
        .enableGetTensorC = true,
        .enableSetOrgShape = true,
        .enableSetBias = false,
        .enableSetTail = true,
        .enableQuantVector = false,
        .enableSetDefineData = false,
        .iterateMode = IterateMode::ITERATE_MODE_DEFAULT,
        .enableReuse = true,
        .enableUBReuse = true,
        .enableL1CacheUB = false,
        .intraBlockPartSum = false,
        .iterateOrder = IterateOrder::UNDEF,
        .scheduleType = ScheduleType::INNER_PRODUCT,
        .enableDoubleCache = false,
        .isBiasBatch = true,
        .enableStaticPadZeros = false,
        .isA2B2Shared = false,
        .enableKdimReorderLoad = false,
        .isCO1Shared = false,
    };
}

} // namespace CommonGMMUtils

#endif // GROUPED_MATMUL_UTILS_COMMON_H
