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
 * \file mc2_adump_engine.h
 * \brief 实现 MC2 算子的 adump 记录引擎
 */
#ifndef MC2_ADUMP_ENGINE_H
#define MC2_ADUMP_ENGINE_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

namespace MC2Adump {
constexpr uint64_t TOTAL_SIZE_PER_CORE = 1 * 1024 * 1024;
constexpr uint32_t RESERVE_SIZE = 20;

enum OpType : uint32_t {
    OP_TYPE_MEGA_MOE = 0x8000,
};

// 算子记录头信息
struct AdumpHeader {
    uint64_t magic;            // 用于确认是否被踩踏
    uint64_t execTimes;        // 执行次数，从0开始累加
    uint32_t version;          // 算子记录版本号
    uint32_t opType;           // 算子类型
    uint32_t stageNum;         // 算子记录总阶段数
    uint64_t timeOutCycles;    // 超时时间（单位：cycle数）
    uint64_t reserve[RESERVE_SIZE];
};

// 算子记录配置信息
struct AdumpConfig {
    uint32_t opType;          // 算子类型
    uint32_t aivId;           // 核Id
    uint32_t sharedFieldSize; // 所有核共用字段大小，后续每个核的记录会从sharedFieldSize开始
    uint32_t stageNum;        // 算子记录总阶段数
    uint64_t timeOutCycles;   // 超时时间（单位：cycle数）
};

class MC2AdumpEngine {
public:
    __aicore__ inline MC2AdumpEngine() = default;
    __aicore__ inline ~MC2AdumpEngine() = default;

    __aicore__ inline void Init(GM_ADDR dumpAddr, AdumpConfig& config) 
    {
        uint32_t aivId = config.aivId;
        __gm__ uint8_t* base = (__gm__ uint8_t*)(dumpAddr);

        // 内存布局：
        // 1. AdumpHeader 算子记录头信息
        // 2. SharedField 各算子自定义，大小config.sharedFieldSize，例如存放关键的tiling字段
        // 3. CustomField 每个核的自定义字段，大小TOTAL_SIZE_PER_CORE，例如存放算子记录的中间结果
        adumpHeader_ = (__gm__ AdumpHeader*)(base);
        adumpSharedFieldPtr_ = (__gm__ uint8_t*)(base + sizeof(AdumpHeader));
        adumpCoreCustomFieldPtr_ = (__gm__ AdumpCoreHeader*)(base + sizeof(AdumpHeader) + config.sharedFieldSize + aivId * TOTAL_SIZE_PER_CORE);

        if (aivId == 0) {
            adumpHeader_->magic = 0x1234567890abcdef;
            adumpHeader_->execTimes = adumpHeader_->execTimes + 1;
            adumpHeader_->version = 910;
            adumpHeader_->opType = config.opType;
            adumpHeader_->stageNum = config.stageNum;
            adumpHeader_->timeOutCycles = config.timeOutCycles;
        }
    }

    __aicore__ inline __gm__ uint8_t* GetSharedFieldPtr()
    {
        return adumpSharedFieldPtr_;
    }

    template<typename T>
    __aicore__ inline __gm__ T* GetCoreCustomFieldPtr()
    {
        static_assert(sizeof(T) <= TOTAL_SIZE_PER_CORE, "Custom data size exceeds limit");
        return (__gm__ T*)(adumpCoreCustomFieldPtr_);
    }

private:
    __gm__ AdumpHeader *adumpHeader_;
    __gm__ uint8_t *adumpSharedFieldPtr_;
    __gm__ uint8_t *adumpCoreCustomFieldPtr_;
};
}

#endif // MC2_ADUMP_ENGINE_H