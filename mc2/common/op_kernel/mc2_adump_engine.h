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

struct AdumpConfig {
    uint32_t opType;
    uint32_t aivId;
    uint32_t sharedFieldSize;
    uint32_t stageNum;
    uint64_t timeOutCycles;
    uint32_t perCoreSize;
    bool enableCacheFlush;
};

class MC2AdumpEngine {
public:
    __aicore__ inline MC2AdumpEngine() = default;
    __aicore__ inline ~MC2AdumpEngine() = default;

    __aicore__ inline void Init(GM_ADDR dumpAddr, AdumpConfig& config)
    {
        uint32_t aivId = config.aivId;
        perCoreSize_ = config.perCoreSize;
        enableCacheFlush_ = config.enableCacheFlush;
        __gm__ uint8_t* base = (__gm__ uint8_t*)(dumpAddr);

        adumpHeader_ = (__gm__ AdumpHeader*)(base);
        adumpSharedFieldPtr_ = (__gm__ uint8_t*)(base + sizeof(AdumpHeader));
        adumpCoreCustomFieldPtr_ = (__gm__ uint8_t*)(base + sizeof(AdumpHeader) + config.sharedFieldSize + aivId * perCoreSize_);

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

    __aicore__ inline void FlushCache()
    {
        if (enableCacheFlush_) {
            gm_dcci(adumpCoreCustomFieldPtr_);
        }
    }

    template<typename T>
    __aicore__ inline __gm__ T* GetCoreCustomFieldPtr()
    {
        static_assert(sizeof(T) <= perCoreSize_, "Custom data size exceeds limit");
        return (__gm__ T*)(adumpCoreCustomFieldPtr_);
    }

private:
    __gm__ AdumpHeader *adumpHeader_;
    __gm__ uint8_t *adumpSharedFieldPtr_;
    __gm__ uint8_t *adumpCoreCustomFieldPtr_;
    uint32_t perCoreSize_ = 0;
    bool enableCacheFlush_ = false;
};
}

#endif // MC2_ADUMP_ENGINE_H