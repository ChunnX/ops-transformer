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

using namespace AscendC;

constexpr uint32_t ADUMP_GM_ALIGN_BYTES = 512;
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
    uint32_t ubStartAddr;     // UB 起始地址
    uint32_t ubSizeBytes;     // UB 大小（单位：字节）
    uint32_t sharedFieldSize; // 所有核共用字段大小，后续每个核的记录会从sharedFieldSize开始
    uint32_t customFieldSize; // 每个核的自定义字段大小
    uint32_t stageNum;        // 算子记录总阶段数
    uint64_t timeOutCycles;   // 超时时间（单位：cycle数）
};

class MC2AdumpEngine {
public:
    __aicore__ inline MC2AdumpEngine() = default;
    __aicore__ inline ~MC2AdumpEngine() = default;

    __aicore__ inline void Init(GM_ADDR adumpAddrBase, AdumpConfig& config) 
    {
        uint32_t aivId = config.aivId;

        // GM内存布局：
        // 1. AdumpHeader 算子记录头信息
        // 2. SharedField 各算子自定义，大小config.sharedFieldSize，例如存放关键的tiling字段
        // 3. CustomField 每个核的自定义字段，大小config.customFieldSize，例如存放算子记录的中间结果
        uint32_t adumpHeaderSize = RoundUp(sizeof(AdumpHeader), ADUMP_GM_ALIGN_BYTES);
        uint32_t sharedFieldSize = RoundUp(config.sharedFieldSize, ADUMP_GM_ALIGN_BYTES);
        uint32_t customFieldSize = RoundUp(config.customFieldSize, ADUMP_GM_ALIGN_BYTES);
        adumpHeader_ = (__gm__ AdumpHeader*)(adumpAddrBase);
        adumpSharedFieldAddr_ = (GM_ADDR)(adumpAddrBase + adumpHeaderSize);
        adumpCoreCustomFieldAddr_ = (GM_ADDR)(adumpAddrBase + adumpHeaderSize + sharedFieldSize + aivId * customFieldSize);

        adumpU8LocalTensor_ = LocalTensor<uint8_t>{TPosition::LCM, config.ubStartAddr, config.ubSizeBytes};

        if (aivId == 0) {
            adumpHeader_->magic = 0x1234567890abcdef;
            adumpHeader_->execTimes = adumpHeader_->execTimes + 1;
            adumpHeader_->version = 910;
            adumpHeader_->opType = config.opType;
            adumpHeader_->stageNum = config.stageNum;
            adumpHeader_->timeOutCycles = config.timeOutCycles;
        }
    }

    __aicore__ inline GM_ADDR GetSharedFieldAddr()
    {
        return adumpSharedFieldAddr_;
    }

    __aicore__ inline GM_ADDR GetCoreCustomFieldAddr()
    {
        return adumpCoreCustomFieldAddr_;
    }

    __aicore__ inline LocalTensor<uint8_t> GetAdumpU8LocalTensor()
    {
        return adumpU8LocalTensor_;
    }

    __aicore__ inline uint32_t GetAdumpUbSizeBytes()
    {
        return adumpU8LocalTensor_.GetSize();
    }

private:
    __gm__ AdumpHeader *adumpHeader_;
    GM_ADDR adumpSharedFieldAddr_;
    GM_ADDR adumpCoreCustomFieldAddr_;
    LocalTensor<uint8_t> adumpU8LocalTensor_;
};
} // namespace MC2Adump

#endif // MC2_ADUMP_ENGINE_H