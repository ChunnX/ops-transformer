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

struct MegaMoeAdumpSharedField {
    uint32_t cclBufferSize;
    uint32_t worldSize;       // 通信域大小
    uint32_t rankId;
    uint32_t aivNum;

    uint32_t m;
    uint32_t k;
    uint32_t n;
    uint32_t isTransposeW1;
    uint32_t isTransposeW2;
    uint32_t hasBias1;
    uint32_t hasBias2;

    uint32_t expertNumPerRank;
    uint32_t topK;
    uint32_t maxRecvTokenNum;
    uint32_t dispatchQuantMode;
    uint32_t dispatchQuantOutDtype;
    uint32_t combineQuantMode;
    uint32_t commAlgCode;
    uint32_t numMaxTokensPerRank;
    uint32_t activationCode;
    float activationClamp;
};

struct MegaMoeAdumpBasicInfo {
    uint32_t aivId;
    uint32_t currentStageId;
    uint64_t startTimeStamp;
};

struct MegaMoeAdumpStageCrossRankSyncInfo {
    uint32_t targetRankId;  // 等待rankId的Flag
};

struct MegaMoeAdumpStagePullTokensInfo {
    uint32_t localExpertId;           // 当前处理的本地专家id
    uint32_t targetRankId;            // Pull的目标卡
    uint64_t targetRankTokenStartIdx; // 目标卡上的token起始idx
    uint64_t targetRankMoveTokenNum;  // 目标卡上要移动的token数量
};

struct MegaMoeAdumpCoreCustomInfoA3 {
    MegaMoeAdumpBasicInfo basicInfo;
    union MegaMoeAdumpStageInfo {
        MegaMoeAdumpStageCrossRankSyncInfo crossRankSyncInfo;
        MegaMoeAdumpStagePullTokensInfo pullTokensInfo;
    };
};

enum StageA3 : uint32_t {
    STAGE_INIT = 0,
    STAGE_APPLY_XACTIVATE_MASK,
    STAGE_INIT_ROUTING,
    STAGE_CROSS_RANK_SYNC,
    STAGE_GET_CUMSUM,
    STAGE_PULL_TOKENS,
    STAGE_EPILOGUE1,
    STAGE_COMBINE,
    STAGE_RESET,
    STAGE_CROSS_RANK_SYNC2,
    STAGE_UNPERMUTE,
    STAGE_END
};

class MegaMoeAdumpAdapterA3 {
public:
    __aicore__ inline void Init(GM_ADDR dumpBase, MegaMoeAdumpSharedField& sharedField)
    {
        uint32_t aivId = AscendC::GetBlockIdx();

        MC2Adump::AdumpConfig config = {
            .opType = MC2Adump::OP_TYPE_MEGA_MOE,
            .aivId = aivId,
            .sharedFieldSize = sizeof(MegaMoeAdumpSharedField),
            .stageNum = StageA3::STAGE_END,
            .timeOutCycles = 10 * 50 * 1000 * 1000,
        };
        adumpEngine_.Init(dumpBase, config);

        if (aivId == 0) {
            InitAdumpSharedField(sharedField);
        }
    }

    __aicore__ inline uint32_t GetAdumpTotalSize()
    {
        return sizeof(MC2Adump::AdumpHeader) + sizeof(MegaMoeAdumpSharedField) + MC2Adump::TOTAL_SIZE_PER_CORE * AscendC::GetBlockNum();
    }

    __aicore__ inline void UpdateAdumpCurrentStateId(uint32_t stageId)
    {
        return GetAdumpCoreCustomField()->basicInfo.currentStageId = stageId;
    }

    __aicore__ inline void UpdateAdumpCrossRankSyncInfo(MegaMoeAdumpStageCrossRankSyncInfo& crossRankSyncInfo)
    {
        return GetAdumpCoreCustomField()->crossRankSyncInfo = crossRankSyncInfo;
    }

    __aicore__ inline void UpdateAdumpPullTokensInfo(MegaMoeAdumpStagePullTokensInfo& pullTokensInfo)
    {
        return GetAdumpCoreCustomField()->pullTokensInfo = pullTokensInfo;
    }
        
private:
    __aicore__ inline void InitAdumpSharedField(MegaMoeAdumpSharedField& sharedField)
    {
        __gm__ MegaMoeAdumpSharedField* adumpSharedField = adumpEngine_.GetSharedFieldPtr();
        adumpSharedField->cclBufferSize = sharedField.cclBufferSize;
        adumpSharedField->worldSize = sharedField.worldSize;
        adumpSharedField->rankId = sharedField.rankId;
        adumpSharedField->aivNum = sharedField.aivNum;
        adumpSharedField->m = sharedField.m;
        adumpSharedField->k = sharedField.k;
        adumpSharedField->n = sharedField.n;
        adumpSharedField->isTransposeW1 = sharedField.isTransposeW1;
        adumpSharedField->isTransposeW2 = sharedField.isTransposeW2;
        adumpSharedField->hasBias1 = sharedField.hasBias1;
        adumpSharedField->expertNumPerRank = sharedField.expertNumPerRank;
        adumpSharedField->topK = sharedField.topK;
        adumpSharedField->maxRecvTokenNum = sharedField.maxRecvTokenNum;
        adumpSharedField->dispatchQuantMode = sharedField.dispatchQuantMode;
        adumpSharedField->dispatchQuantOutDtype = sharedField.dispatchQuantOutDtype;
        adumpSharedField->combineQuantMode = sharedField.combineQuantMode;
        adumpSharedField->commAlgCode = sharedField.commAlgCode;
        adumpSharedField->numMaxTokensPerRank = sharedField.numMaxTokensPerRank;
        adumpSharedField->activationCode = sharedField.activationCode;
        adumpSharedField->activationClamp = sharedField.activationClamp;
    }

    __aicore__ inline __gm__ MegaMoeAdumpSharedField* GetAdumpSharedField()
    {
        return (__gm__ MegaMoeAdumpSharedField*)adumpEngine_.GetSharedFieldPtr();
    }
    
    __aicore__ inline __gm__ MegaMoeAdumpCoreCustomInfoA3* GetAdumpCoreCustomField()
    {
        return adumpEngine_.GetCoreCustomFieldPtr<MegaMoeAdumpCoreCustomInfoA3>();
    }

private:
    MC2Adump::MC2AdumpEngine adumpEngine_;
};

#endif  // MEGA_MOE_ADUMP_ADAPTER_A3