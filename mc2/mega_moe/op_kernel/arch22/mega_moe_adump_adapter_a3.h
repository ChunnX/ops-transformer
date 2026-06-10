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
#ifdef MEGA_MOE_ADUMP_ADAPTER_A3

#include "../../../common/op_kernel/mc2_adump_engine.h"

struct MegaMoeAdumpSharedField {
    uint32_t worldSize;       // 通信域大小
    uint32_t rankId;
    uint32_t aivNum;
    uint32_t moeExpertNum;
};

struct MegaMoeAdumpStage1Info {
    uint32_t moeExpertNum;
};

struct MegaMoeAdumpStage2Info {
    uint32_t moeExpertNum;
};

union MegaMoeAdumpCustomInfoA3 {
    struct MegaMoeAdumpStage1Info stage1Info;
    struct MegaMoeAdumpStage2Info stage2Info;
};

class MegaMoeAdumpAdapterA3 {
public:
    enum Stage : uint32_t {
        STAGE_INIT = 0,
        STAGE_1,
        STAGE_2,
        STAGE_END
    };

    __aicore__ inline void Init(GM_ADDR dumpBase, MegaMoeAdumpSharedField& sharedField)
    {
        MC2Adump::AdumpConfig config = {
            .opType = MC2Adump::OP_TYPE_MEGA_MOE,
            .aivId = AscendC::GetBlockIdx(),
            .sharedFieldSize = sizeof(MegaMoeAdumpSharedField),
            .stageNum = Stage::STAGE_END,
            .timeOutCycles = 10 * 50 * 1000 * 1000,
        };

        adumpEngine_.Init(dumpBase, config);

        GetAdumpSharedField()->worldSize = sharedField.worldSize;
        GetAdumpSharedField()->rankId = sharedField.rankId;
        GetAdumpSharedField()->aivNum = AscendC::GetBlockNum();
        GetAdumpSharedField()->moeExpertNum = sharedField.moeExpertNum;
    }

    __aicore__ inline uint32_t GetAdumpTotalSize()
    {
        return sizeof(MC2Adump::AdumpHeader) + sizeof(MegaMoeAdumpSharedField) + MC2Adump::TOTAL_SIZE_PER_CORE * AscendC::GetBlockNum();
    }

    __aicore__ inline void UpdateAdumpCurrentStateInfo(uint32_t stageId)
    {
        return adumpEngine_.UpdateCurrentStageId(stageId);
    }

    __aicore__ inline bool IsAdumpTimeOut()
    {
        return adumpEngine_.IsTimeOut();
    }

    __aicore__ inline __gm__ MegaMoeAdumpStage1Info* GetAdumpStage1Info()
    {
        return &GetAdumpCoreCustomField()->stage1Info;
    }

    __aicore__ inline __gm__ MegaMoeAdumpStage2Info* GetAdumpStage2Info()
    {
        return &GetAdumpCoreCustomField()->stage2Info;
    }
    
private:
    __aicore__ inline __gm__ MegaMoeAdumpSharedField* GetAdumpSharedField()
    {
        return (__gm__ MegaMoeAdumpSharedField*)adumpEngine_.GetSharedFieldPtr();
    }
    
    __aicore__ inline __gm__ MegaMoeAdumpCustomInfoA3* GetAdumpCoreCustomField()
    {
        return adumpEngine_.GetCoreCustomFieldPtr<MegaMoeAdumpCustomInfoA3>();
    }

private:
    MC2Adump::MC2AdumpEngine adumpEngine_;
};

#endif  // MEGA_MOE_ADUMP_ADAPTER_A3