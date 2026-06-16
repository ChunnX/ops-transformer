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
 * \file mega_moe_adump_adapter_a3.h
 * \brief 实现 MegaMoe 算子的 adump 记录引擎，适配 A3 架构
 */
#ifndef MEGA_MOE_ADUMP_ADAPTER_A3
#define MEGA_MOE_ADUMP_ADAPTER_A3

#include "../../../common/op_kernel/mc2_adump_engine.h"

namespace MC2MegaMoeAdump {
constexpr uint32_t CANN_VERSION = 910; // CANN 9.1.0版本，后续版本升级时注意更新
constexpr uint32_t SHARED_FIELD_RESERVE_SIZE = 124; // AdumpSharedField大小为512B，保留124个uint32_t字段
constexpr uint32_t CUSTOM_FIELD_RESERVE_SIZE = 124; // AdumpCoreCustomInfoA3大小为512B，保留124个uint32_t字段
constexpr uint64_t DEFAULT_TIMEOUT_CYCLES = 10 * 50 * 1000 * 1000; // 默认超时时间10s
constexpr uint32_t ADUMP_UB_SIZE_BYTES = 1024;

struct AdumpSharedField {
    uint32_t cclBufferSize;  // hccl buffer大小
    uint32_t worldSize;      // 通信域大小
    uint32_t rankId;         // rankId
    uint32_t aivNum;         // aiv核数
    uint32_t reserved[SHARED_FIELD_RESERVE_SIZE]; // 保留字段，当有新增字段时，优先使用该字段，修改SHARED_FIELD_RESERVE_SIZE
};

enum AdumpSharedFieldPos : uint32_t {
    CCL_BUFFER_SIZE_POS = 0,
    WORLD_SIZE_POS = 1,
    RANK_ID_POS = 2,
    AIV_NUM_POS = 3,
    SHARED_FIELD_POS_END,
};

struct AdumpCoreCustomInfoA3 {
    uint32_t aivId;
    uint32_t currentStageId;
    uint32_t startTimeStampHigh32Bit;
    uint32_t startTimeStampLow32Bit;
    uint32_t reserved[CUSTOM_FIELD_RESERVE_SIZE]; // 保留字段，当有新增字段时，优先使用该字段，修改CUSTOM_FIELD_RESERVE_SIZE
};

enum AdumpCoreCustomInfoPos : uint32_t {
    AIVID_POS = 0,
    CURRENT_STAGEID_POS = 1,
    START_TIMESTAMP_LOW_32BIT_POS = 2,
    START_TIMESTAMP_HIGH_32BIT_POS = 3,
    CUSTOM_FIELD_POS_END,
};

enum StageA3 : uint32_t {
    STAGE_INIT = 0,
    STAGE_END
};

class AdumpAdapterA3 {
public:
    __aicore__ inline void Init(GM_ADDR dumpBase)
    {
        // 解析脚本中，sharedField、customField的大小按照512B，需要与代码保持一致，这里作为提示。
        static_assert(sizeof(AdumpSharedField) == 512,
            "AdumpSharedField size should be 512B, please check SHARED_FIELD_RESERVE_SIZE, and update parser script if necessary.");
        static_assert(SHARED_FIELD_POS_END * sizeof(uint32_t) <= sizeof(AdumpSharedField),
            "SHARED_FIELD_POS_END * sizeof(uint32_t) should be less than AdumpSharedField size.");
        static_assert(sizeof(AdumpCoreCustomInfoA3) == 512,
            "AdumpCoreCustomInfoA3 size should be 512B, please check CUSTOM_FIELD_RESERVE_SIZE, and update parser script if necessary.");
        static_assert(CUSTOM_FIELD_POS_END * sizeof(uint32_t) <= sizeof(AdumpCoreCustomInfoA3),
            "CUSTOM_FIELD_POS_END * sizeof(uint32_t) should be less than AdumpCoreCustomInfoA3 size.");

        aivId_ = AscendC::GetBlockIdx();

        MC2Adump::AdumpConfig config = {
            .version = CANN_VERSION,
            .opType = MC2Adump::OP_TYPE_MEGA_MOE,
            .aivId = aivId_,
            .ubStartAddr = 0,
            .ubSizeBytes = ADUMP_UB_SIZE_BYTES,
            .sharedFieldSize = sizeof(AdumpSharedField),
            .customFieldSize = sizeof(AdumpCoreCustomInfoA3),
            .stageNum = StageA3::STAGE_END,
            .timeOutCycles = DEFAULT_TIMEOUT_CYCLES,
        };
        adumpEngine_ = MC2Adump::MC2AdumpEngine(dumpBase, config);
        InitCoreBasicInfo();
    }

    __aicore__ inline void FillSharedField(AdumpSharedField& sharedField)
    {
        // 仅0核完成共享区域初始化
        uint32_t aivId = AscendC::GetBlockIdx();
        if (aivId == 0) {
            InitAdumpSharedField(sharedField);
        }
    }

    __aicore__ inline uint32_t GetAdumpTotalGmSizeBytes()
    {
        uint32_t headerSize = sizeof(MC2Adump::AdumpHeader);
        uint32_t sharedFieldSize = sizeof(AdumpSharedField);
        uint32_t customFieldSize = sizeof(AdumpCoreCustomInfoA3) * AscendC::GetBlockNum() * AscendC::GetSubBlockNum();
        return headerSize + sharedFieldSize + customFieldSize;
    }

    __aicore__ inline uint32_t GetAdumpTotalUbSizeBytes()
    {
        return ADUMP_UB_SIZE_BYTES;
    }

    __aicore__ inline void UpdateAdumpCurrentStateId(uint32_t stageId)
    {
        LocalTensor<uint32_t> adumpU32LocalTensor = GetAdumpLocalTensor<uint32_t>();
        adumpU32LocalTensor.SetValue(0, stageId);

        GM_ADDR adumpCoreCustomFieldAddr = GetAdumpCoreCustomFieldAddr();
        GlobalTensor<uint32_t> adumpU32GlobalTensor;
        adumpU32GlobalTensor.SetGlobalBuffer((__gm__ uint32_t*)adumpCoreCustomFieldAddr);

        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        DataCopyPad(adumpU32GlobalTensor[AdumpCoreCustomInfoPos::CURRENT_STAGEID_POS], adumpU32LocalTensor, {1, sizeof(uint32_t), 0, 0, 0});
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
    }

private:
    __aicore__ inline void InitAdumpSharedField(AdumpSharedField& sharedField)
    {
        LocalTensor<uint32_t> adumpU32LocalTensor = GetAdumpLocalTensor<uint32_t>();
        adumpU32LocalTensor.SetValue(AdumpSharedFieldPos::CCL_BUFFER_SIZE_POS, sharedField.cclBufferSize);
        adumpU32LocalTensor.SetValue(AdumpSharedFieldPos::WORLD_SIZE_POS, sharedField.worldSize);
        adumpU32LocalTensor.SetValue(AdumpSharedFieldPos::RANK_ID_POS, sharedField.rankId);
        adumpU32LocalTensor.SetValue(AdumpSharedFieldPos::AIV_NUM_POS, sharedField.aivNum);

        GM_ADDR adumpSharedFieldAddr = GetAdumpSharedFieldAddr();
        GlobalTensor<uint32_t> adumpU32GlobalTensor;
        adumpU32GlobalTensor.SetGlobalBuffer((__gm__ uint32_t*)adumpSharedFieldAddr);

        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        DataCopyPad(adumpU32GlobalTensor, adumpU32LocalTensor, {1, AdumpSharedFieldPos::SHARED_FIELD_POS_END * sizeof(uint32_t), 0, 0, 0});
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
    }

    __aicore__ inline void InitCoreBasicInfo()
    {
        uint32_t currentStageId = 0;
        uint64_t startTimeStamp = AscendC::GetSystemCycle();

        LocalTensor<uint32_t> adumpU32LocalTensor = GetAdumpLocalTensor<uint32_t>();
        adumpU32LocalTensor.SetValue(AdumpCoreCustomInfoPos::AIVID_POS, aivId_);
        adumpU32LocalTensor.SetValue(AdumpCoreCustomInfoPos::CURRENT_STAGEID_POS, currentStageId);
        adumpU32LocalTensor.SetValue(AdumpCoreCustomInfoPos::START_TIMESTAMP_LOW_32BIT_POS, MC2Adump::GetLow32Bit(startTimeStamp));
        adumpU32LocalTensor.SetValue(AdumpCoreCustomInfoPos::START_TIMESTAMP_HIGH_32BIT_POS, MC2Adump::GetHigh32Bit(startTimeStamp));

        GM_ADDR adumpCoreCustomFieldAddr = GetAdumpCoreCustomFieldAddr();
        GlobalTensor<uint32_t> adumpU32GlobalTensor;
        adumpU32GlobalTensor.SetGlobalBuffer((__gm__ uint32_t*)adumpCoreCustomFieldAddr);

        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        DataCopyPad(adumpU32GlobalTensor, adumpU32LocalTensor, {1, AdumpCoreCustomInfoPos::CUSTOM_FIELD_POS_END * sizeof(uint32_t), 0, 0, 0});
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
    }

    __aicore__ inline GM_ADDR GetAdumpSharedFieldAddr()
    {
        return adumpEngine_.GetSharedFieldAddr();
    }

    __aicore__ inline GM_ADDR GetAdumpCoreCustomFieldAddr()
    {
        return adumpEngine_.GetCoreCustomFieldAddr();
    }

    template<typename T>
    __aicore__ inline LocalTensor<T> GetAdumpLocalTensor()
    {
        return adumpEngine_.GetAdumpU8LocalTensor().template ReinterpretCast<T>();
    }

private:
    uint32_t aivId_;
    MC2Adump::MC2AdumpEngine adumpEngine_;
};
} // namespace MC2MegaMoeAdump
#endif  // MEGA_MOE_ADUMP_ADAPTER_A3