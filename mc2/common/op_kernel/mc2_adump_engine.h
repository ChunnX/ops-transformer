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
 * \brief 实现 MC2 算子的 adump 记录引擎，用于提高MC2算子的定位能力。与各算子的Adapter配合使用。
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

constexpr uint32_t ADUMP_GM_ALIGN_BYTES = 512; // 512B对齐提高GM访问效率
constexpr uint32_t RESERVE_SIZE = 59; // AdumpHeader大小为512B，保留59个uint64_t字段

// 算子OpType信息
enum OpType : uint32_t {
    OP_TYPE_MEGA_MOE = 1000, // mega moe算子OpType值
};

// 算子记录头信息
struct AdumpHeader {
    uint64_t magic;            // 用于确认是否被踩踏，固定值0x0000000055AA55AA(十进制：1,437,226,410)
    uint64_t execTimes;        // 执行次数，HCCL通信域创建时初始化后值为0，后续每次调用加1，即首次调用值为1
    uint32_t version;          // 算子记录版本号，用于确定版本，当前版本910(表示CANN 9.1.0)，后续版本升级时注意更新
    uint32_t opType;           // 算子类型，参考OpType枚举值
    uint32_t stageNum;         // 算子记录总阶段数
    uint32_t reserved1;        // 保留字段1, 避免编译器自动对齐导致数据读取错误
    uint64_t timeOutCycles;    // 超时时间（单位：cycle数）
    uint64_t reserved2[RESERVE_SIZE]; // 保留字段2, 当有新增字段时，优先使用该字段，修改RESERVE_SIZE，当reserver2不够继续扩充时，注意同步修改解析脚本。
};

// 算子记录配置信息
struct AdumpConfig {
    uint32_t version;         // 算子记录版本号，用于确定算子版本
    uint32_t opType;          // 算子类型
    uint32_t aivId;           // 核Id
    uint32_t ubStartAddr;     // UB 起始地址
    uint32_t ubSizeBytes;     // UB 大小（单位：字节）
    uint32_t sharedFieldSize; // 所有核共用字段大小，后续每个核的记录会从sharedFieldSize开始
    uint32_t customFieldSize; // 每个核的自定义字段大小
    uint32_t stageNum;        // 算子记录总阶段数
    uint64_t timeOutCycles;   // 超时时间（单位：cycle数）
};

__aicore__ inline uint32_t GetHigh32Bit(uint64_t val)
{
    return static_cast<uint32_t>(val >> 32);
}

__aicore__ inline uint32_t GetLow32Bit(uint64_t val)
{
    return static_cast<uint32_t>(val & 0xFFFFFFFF);
}

class MC2AdumpEngine {
public:
    __aicore__ inline MC2AdumpEngine() = default;
    __aicore__ inline MC2AdumpEngine(GM_ADDR adumpAddrBase, AdumpConfig& config)
    {
        static_assert(sizeof(AdumpHeader) == 512,
            "AdumpHeader size should be 512B, please check RESERVE_SIZE, and update parser script if necessary.");

        uint32_t aivId = config.aivId;

        // GM内存布局：
        // 1. AdumpHeader 算子记录头信息
        // 2. SharedField 各算子自定义，大小config.sharedFieldSize，例如存放关键的tiling字段
        // 3. CustomField 每个核的自定义字段，大小config.customFieldSize，例如存放算子记录的中间结果
        uint32_t adumpHeaderSize = sizeof(AdumpHeader);
        uint32_t sharedFieldSize = config.sharedFieldSize;
        uint32_t customFieldSize = config.customFieldSize;
        adumpHeader_ = (__gm__ AdumpHeader*)(adumpAddrBase);
        adumpSharedFieldAddr_ = (GM_ADDR)(adumpAddrBase + adumpHeaderSize);
        adumpCoreCustomFieldAddr_ = (GM_ADDR)(adumpAddrBase + adumpHeaderSize + sharedFieldSize + aivId * customFieldSize);

        adumpU8LocalTensor_ = LocalTensor<uint8_t>{TPosition::LCM, config.ubStartAddr, config.ubSizeBytes};

        if (aivId == 0) { // 仅用0核初始化Header，避免多核写入导致数据错乱
            adumpHeader_->magic = 0x0000000055AA55AA;
            adumpHeader_->execTimes = adumpHeader_->execTimes + 1;
            adumpHeader_->version = config.version;
            adumpHeader_->opType = config.opType;
            adumpHeader_->stageNum = config.stageNum;
            adumpHeader_->timeOutCycles = config.timeOutCycles;
        }
    }
    __aicore__ inline ~MC2AdumpEngine() = default;

    // 多核公用字段地址
    __aicore__ inline GM_ADDR GetSharedFieldAddr()
    {
        return adumpSharedFieldAddr_;
    }

    // 每个核的自定义字段地址
    __aicore__ inline GM_ADDR GetCoreCustomFieldAddr()
    {
        return adumpCoreCustomFieldAddr_;
    }

    // 获取UB地址，用于adump记录，使用时按照需要类型自定义转换
    __aicore__ inline LocalTensor<uint8_t> GetAdumpU8LocalTensor()
    {
        return adumpU8LocalTensor_;
    }

    // 获取UB大小（单位：字节）
    __aicore__ inline uint32_t GetAdumpUbSizeBytes()
    {
        return adumpU8LocalTensor_.GetSize();
    }

private:
    __gm__ AdumpHeader *adumpHeader_{nullptr};
    GM_ADDR adumpSharedFieldAddr_{0};
    GM_ADDR adumpCoreCustomFieldAddr_{0};
    LocalTensor<uint8_t> adumpU8LocalTensor_;
};
} // namespace MC2Adump
#endif // MC2_ADUMP_ENGINE_H